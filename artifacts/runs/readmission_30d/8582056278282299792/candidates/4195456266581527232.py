# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Initial baseline seed program for clinical Health Clinical Risk & Care Allocation.

Contains the two-stage strategy evolved by AlphaEvolve:
1. `fit_and_score_risk`: Refines patient event probabilities over legacy baseline API scores + EHR/claims features.
2. `allocate_interventions`: Assigns care-management interventions under slot, budget, and clinical eligibility rules.
"""

from __future__ import annotations

from typing import Dict, Sequence

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from clinical_models import (
    CareAllocationPlan,
    CohortSpec,
    InterventionTier,
    PatientRecord,
)
from clinical_toolkit import (
    CareAllocationState,
    ClinicalFeatureBuilder,
    safe_logit,
    safe_sigmoid,
)


# =====================================================================
# EVOLVE-BLOCK-START
# Strategy only. Domain mechanics and feasibility checks live in `clinical_toolkit` (`ClinicalFeatureBuilder`,
# `RiskBlender`, `CareAllocationState`, `safe_logit`, `safe_sigmoid`). Pre-imported libraries available in sandbox:
# `numpy`, `scipy`, `sklearn`, `lightgbm`, `xgboost`.


def _extract_meta_features(records: Sequence[PatientRecord]) -> np.ndarray:
    """Extract clinical submodel, discrepancy, and acuity interaction features."""
    import numpy as np

    meta = []
    for p in records:
        b_prob = float(getattr(p, "baseline_api_prob", 0.5))
        b_prob = max(1e-5, min(1.0 - 1e-5, b_prob))
        b_logit = safe_logit(b_prob)

        sub = getattr(p, "baseline_submodel_scores", None) or {}
        cl = float(sub.get("claims_linear", b_prob))
        cl = max(1e-5, min(1.0 - 1e-5, cl))
        cl_logit = safe_logit(cl)

        ev = float(sub.get("ehr_vitals", b_prob))
        ev = max(1e-5, min(1.0 - 1e-5, ev))
        ev_logit = safe_logit(ev)

        ur = float(sub.get("utilization_rules", b_prob))
        ur = max(1e-5, min(1.0 - 1e-5, ur))
        ur_logit = safe_logit(ur)

        acuity = float(getattr(p, "acuity_score", 0.5))
        contra = getattr(p, "contraindications", ()) or ()
        n_contra = float(len(contra))

        # Acute vs historical claims discrepancy captures sudden clinical decompensation
        vitals_vs_claims = ev_logit - cl_logit
        vitals_vs_util = ev_logit - ur_logit
        claims_vs_util = cl_logit - ur_logit

        # Acuity interaction terms
        acuity_x_bprob = acuity * b_prob
        acuity_x_ev = acuity * ev
        acuity_x_cl = acuity * cl

        # Consensus and spread across legacy views
        sub_mean = (cl + ev + ur) / 3.0
        sub_std = float(np.std([cl, ev, ur]))

        meta.append([
            b_prob,
            b_logit,
            cl,
            cl_logit,
            ev,
            ev_logit,
            ur,
            ur_logit,
            acuity,
            n_contra,
            vitals_vs_claims,
            vitals_vs_util,
            claims_vs_util,
            acuity_x_bprob,
            acuity_x_ev,
            acuity_x_cl,
            sub_mean,
            sub_std,
        ])
    return np.array(meta, dtype=np.float32)


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Clinical feature builder with syndrome interactions, CV multi-model ensemble, and RiskBlender."""
    import numpy as np
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    # 1. Domain feature extraction with clinical syndrome interactions
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)

    x_train_clean = np.nan_to_num(x_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    x_eval_clean = np.nan_to_num(x_eval_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Extract legacy submodel, discrepancy, and acuity meta-features
    meta_train = _extract_meta_features(train_records)
    meta_eval = _extract_meta_features(eval_records)

    x_train_comb = np.hstack([x_train_clean, meta_train])
    x_eval_comb = np.hstack([x_eval_clean, meta_eval])

    # 3. Fit toolkit RiskBlender as anchor model
    rb_eval_probs = None
    try:
        blender = RiskBlender(
            linear_weight=0.35,
            lgbm_weight=0.40,
            xgb_weight=0.25,
            baseline_prior_weight=0.08,
            random_state=42,
        )
        rb_preds = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train_raw,
            y_train=y_train,
            x_eval=x_eval_raw,
        )
        if rb_preds and len(rb_preds) == len(eval_records):
            rb_eval_probs = np.array([float(rb_preds[p.patient_id]) for p in eval_records])
    except Exception:
        rb_eval_probs = None

    # 4. Out-of-fold Stratified 5-Fold multi-model ensemble
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb = np.zeros(len(train_records), dtype=np.float64)
    oof_xgb = np.zeros(len(train_records), dtype=np.float64)
    oof_lr = np.zeros(len(train_records), dtype=np.float64)
    oof_hgb = np.zeros(len(train_records), dtype=np.float64)

    eval_lgb = np.zeros(len(eval_records), dtype=np.float64)
    eval_xgb = np.zeros(len(eval_records), dtype=np.float64)
    eval_lr = np.zeros(len(eval_records), dtype=np.float64)
    eval_hgb = np.zeros(len(eval_records), dtype=np.float64)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(x_train_comb, y_train)):
        X_tr, y_tr = x_train_comb[trn_idx], y_train[trn_idx]
        X_val = x_train_comb[val_idx]

        # Model A: LightGBM
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=1,
        )
        clf_lgb.fit(X_tr, y_tr)
        oof_lgb[val_idx] = clf_lgb.predict_proba(X_val)[:, 1]
        eval_lgb += clf_lgb.predict_proba(x_eval_comb)[:, 1] / n_splits

        # Model B: XGBoost
        clf_xgb = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42 + fold,
            n_jobs=1,
        )
        clf_xgb.fit(X_tr, y_tr)
        oof_xgb[val_idx] = clf_xgb.predict_proba(X_val)[:, 1]
        eval_xgb += clf_xgb.predict_proba(x_eval_comb)[:, 1] / n_splits

        # Model C: Regularized Logistic Regression on Standardized Features
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)
        X_ev_sc = scaler.transform(x_eval_comb)

        clf_lr = LogisticRegression(C=0.5, max_iter=300, random_state=42 + fold)
        clf_lr.fit(X_tr_sc, y_tr)
        oof_lr[val_idx] = clf_lr.predict_proba(X_val_sc)[:, 1]
        eval_lr += clf_lr.predict_proba(X_ev_sc)[:, 1] / n_splits

        # Model D: HistGradientBoosting
        clf_hgb = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=15,
            random_state=42 + fold,
        )
        clf_hgb.fit(X_tr, y_tr)
        oof_hgb[val_idx] = clf_hgb.predict_proba(X_val)[:, 1]
        eval_hgb += clf_hgb.predict_proba(x_eval_comb)[:, 1] / n_splits

    # 5. Logit-space blending with legacy baseline prior
    base_train = np.array([float(getattr(p, "baseline_api_prob", 0.5)) for p in train_records])
    base_eval = np.array([float(getattr(p, "baseline_api_prob", 0.5)) for p in eval_records])

    def to_logit(p_arr: np.ndarray) -> np.ndarray:
        p_clamped = np.clip(p_arr, 1e-5, 1.0 - 1e-5)
        return np.log(p_clamped / (1.0 - p_clamped))

    w_lgb, w_xgb, w_lr, w_hgb, w_base = 0.35, 0.30, 0.15, 0.12, 0.08
    oof_blend_logit = (
        w_lgb * to_logit(oof_lgb)
        + w_xgb * to_logit(oof_xgb)
        + w_lr * to_logit(oof_lr)
        + w_hgb * to_logit(oof_hgb)
        + w_base * to_logit(base_train)
    )

    eval_blend_logit = (
        w_lgb * to_logit(eval_lgb)
        + w_xgb * to_logit(eval_xgb)
        + w_lr * to_logit(eval_lr)
        + w_hgb * to_logit(eval_hgb)
        + w_base * to_logit(base_eval)
    )

    # 6. Platt scaling calibration to optimize Brier score and calibrate to empirical cohort base rate
    calibrator = LogisticRegression(C=10.0, max_iter=300, random_state=42)
    calibrator.fit(oof_blend_logit.reshape(-1, 1), y_train)

    if calibrator.coef_[0][0] > 0.0:
        cv_eval_probs = calibrator.predict_proba(eval_blend_logit.reshape(-1, 1))[:, 1]
    else:
        y_mean = max(1e-4, min(1.0 - 1e-4, float(np.mean(y_train))))
        shift = safe_logit(y_mean) - float(np.mean(oof_blend_logit))
        cv_eval_probs = 1.0 / (1.0 + np.exp(-(eval_blend_logit + shift)))

    # 7. Final ensemble combination with RiskBlender
    if rb_eval_probs is not None:
        final_probs = 0.65 * cv_eval_probs + 0.35 * rb_eval_probs
    else:
        final_probs = cv_eval_probs

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        prob = float(final_probs[idx])
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, prob)), 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Allocation maximizing net benefit under capacity and budget constraints."""
    state = CareAllocationState(cohort_spec, eval_records)
    ranked_patients = sorted(
        eval_records,
        key=lambda p: -float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5))),
    )
    candidate_tiers = [t for t in cohort_spec.interventions.keys() if t != InterventionTier.NONE]

    for p in ranked_patients:
        risk = float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5)))
        best_tier = None
        best_benefit = 0.0
        for tier in candidate_tiers:
            if state.fits(p.patient_id, tier):
                benefit = state.expected_net_benefit(p.patient_id, risk, tier)
                if benefit > best_benefit:
                    best_benefit = benefit
                    best_tier = tier
        if best_tier is not None:
            state.assign(p.patient_id, best_tier)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
