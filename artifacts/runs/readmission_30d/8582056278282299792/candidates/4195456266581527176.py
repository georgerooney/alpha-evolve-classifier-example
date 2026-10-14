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


def _extract_patient_features(records: Sequence[PatientRecord]) -> np.ndarray:
    """Extract clinical discordance, acuity interactions, and elective-correction features."""
    feats = []
    for p in records:
        base_p = float(getattr(p, "baseline_api_prob", 0.5))
        base_logit = safe_logit(max(1e-6, min(1.0 - 1e-6, base_p)))
        sub = getattr(p, "baseline_submodel_scores", {}) or {}
        claims = float(sub.get("claims_linear", base_p))
        ehr = float(sub.get("ehr_vitals", base_p))
        util = float(sub.get("utilization_rules", base_p))
        acuity = float(getattr(p, "acuity_score", 0.5))
        contra_cnt = float(len(getattr(p, "contraindications", [])))

        claims_logit = safe_logit(max(1e-6, min(1.0 - 1e-6, claims)))
        ehr_logit = safe_logit(max(1e-6, min(1.0 - 1e-6, ehr)))
        util_logit = safe_logit(max(1e-6, min(1.0 - 1e-6, util)))

        # Acute vs. chronic discordance (high vitals vs low claims = acute destabilization)
        acute_vs_claims = ehr - claims
        util_vs_claims = util - claims
        acute_vs_util = ehr - util
        spread = max(claims, ehr, util) - min(claims, ehr, util)
        mean_sub = (claims + ehr + util) / 3.0

        # Acuity interactions and elderly low-acuity elective over-prediction correction
        acuity_base = acuity * base_logit
        acuity_ehr = acuity * ehr_logit
        low_acuity_claims = (1.0 - acuity) * claims
        low_acuity_base = (1.0 - acuity) * base_p

        row = [
            base_p, base_logit,
            claims, ehr, util,
            claims_logit, ehr_logit, util_logit,
            acuity, contra_cnt,
            acute_vs_claims, util_vs_claims, acute_vs_util,
            spread, mean_sub,
            acuity_base, acuity_ehr,
            low_acuity_claims, low_acuity_base,
        ]
        feats.append(row)
    return np.asarray(feats, dtype=np.float32)


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Multi-model 5-fold cross-validated stacking ensemble with calibrated baseline blend."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    # 1. Base clinical syndrome interaction features
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_tr_raw, y_tr = builder.fit_transform(cohort_spec, train_records)
    x_ev_raw = builder.transform(cohort_spec, eval_records)

    # 2. Augmented discordance and elective correction features
    extra_tr = _extract_patient_features(train_records)
    extra_ev = _extract_patient_features(eval_records)
    X_tr = np.nan_to_num(np.hstack([x_tr_raw, extra_tr]), nan=0.0, posinf=1.0, neginf=-1.0)
    X_ev = np.nan_to_num(np.hstack([x_ev_raw, extra_ev]), nan=0.0, posinf=1.0, neginf=-1.0)
    y_train = np.asarray(y_tr, dtype=int)

    n_samples = len(train_records)
    n_eval = len(eval_records)
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    n_splits = max(2, min(5, n_pos, n_neg))

    # Baseline logit arrays for prior scaling
    base_tr_logits = np.array(
        [[safe_logit(max(1e-6, min(1.0 - 1e-6, float(p.baseline_api_prob))))] for p in train_records],
        dtype=np.float32,
    )
    base_ev_logits = np.array(
        [[safe_logit(max(1e-6, min(1.0 - 1e-6, float(p.baseline_api_prob))))] for p in eval_records],
        dtype=np.float32,
    )

    # Containers for out-of-fold and evaluation predictions across models
    oof_lr = np.zeros(n_samples, dtype=np.float32)
    oof_lgb = np.zeros(n_samples, dtype=np.float32)
    oof_xgb = np.zeros(n_samples, dtype=np.float32)
    oof_hgb = np.zeros(n_samples, dtype=np.float32)
    oof_base = np.zeros(n_samples, dtype=np.float32)

    eval_lr = np.zeros(n_eval, dtype=np.float32)
    eval_lgb = np.zeros(n_eval, dtype=np.float32)
    eval_xgb = np.zeros(n_eval, dtype=np.float32)
    eval_hgb = np.zeros(n_eval, dtype=np.float32)
    eval_base = np.zeros(n_eval, dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for tr_idx, val_idx in skf.split(X_tr, y_train):
        X_fold_tr, y_fold_tr = X_tr[tr_idx], y_train[tr_idx]
        X_fold_val = X_tr[val_idx]

        # Scaled data for linear models
        scaler = StandardScaler()
        X_fold_tr_s = scaler.fit_transform(X_fold_tr)
        X_fold_val_s = scaler.transform(X_fold_val)
        X_ev_s = scaler.transform(X_ev)

        # 1. Logistic Regression
        lr = LogisticRegression(C=0.25, max_iter=500, penalty="l2", random_state=42)
        lr.fit(X_fold_tr_s, y_fold_tr)
        oof_lr[val_idx] = lr.predict_proba(X_fold_val_s)[:, 1]
        eval_lr += lr.predict_proba(X_ev_s)[:, 1] / float(n_splits)

        # 2. LightGBM
        lgb = LGBMClassifier(
            n_estimators=120,
            learning_rate=0.035,
            max_depth=3,
            num_leaves=7,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        )
        lgb.fit(X_fold_tr, y_fold_tr)
        oof_lgb[val_idx] = lgb.predict_proba(X_fold_val)[:, 1]
        eval_lgb += lgb.predict_proba(X_ev)[:, 1] / float(n_splits)

        # 3. XGBoost
        xgb = XGBClassifier(
            n_estimators=120,
            learning_rate=0.035,
            max_depth=3,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,
        )
        xgb.fit(X_fold_tr, y_fold_tr)
        oof_xgb[val_idx] = xgb.predict_proba(X_fold_val)[:, 1]
        eval_xgb += xgb.predict_proba(X_ev)[:, 1] / float(n_splits)

        # 4. HistGradientBoosting
        hgb = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.035,
            max_depth=3,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=42,
        )
        hgb.fit(X_fold_tr, y_fold_tr)
        oof_hgb[val_idx] = hgb.predict_proba(X_fold_val)[:, 1]
        eval_hgb += hgb.predict_proba(X_ev)[:, 1] / float(n_splits)

        # 5. Baseline API Platt Scaler
        b_lr = LogisticRegression(C=1.0, random_state=42)
        b_lr.fit(base_tr_logits[tr_idx], y_fold_tr)
        oof_base[val_idx] = b_lr.predict_proba(base_tr_logits[val_idx])[:, 1]
        eval_base += b_lr.predict_proba(base_ev_logits)[:, 1] / float(n_splits)

    # Convert OOF predictions to logit meta-features
    def _to_meta_features(p_lr, p_lgb, p_xgb, p_hgb, p_base):
        return np.column_stack([
            [safe_logit(max(1e-6, min(1.0 - 1e-6, float(p)))) for p in p_lr],
            [safe_logit(max(1e-6, min(1.0 - 1e-6, float(p)))) for p in p_lgb],
            [safe_logit(max(1e-6, min(1.0 - 1e-6, float(p)))) for p in p_xgb],
            [safe_logit(max(1e-6, min(1.0 - 1e-6, float(p)))) for p in p_hgb],
            [safe_logit(max(1e-6, min(1.0 - 1e-6, float(p)))) for p in p_base],
        ])

    meta_X_tr = _to_meta_features(oof_lr, oof_lgb, oof_xgb, oof_hgb, oof_base)
    meta_X_ev = _to_meta_features(eval_lr, eval_lgb, eval_xgb, eval_hgb, eval_base)

    # Stacking Super Learner with L2 regularized calibration
    meta_clf = LogisticRegression(C=0.5, penalty="l2", max_iter=500, random_state=42)
    meta_clf.fit(meta_X_tr, y_train)
    meta_probs = meta_clf.predict_proba(meta_X_ev)[:, 1]

    # Incorporate RiskBlender as an additional calibrated anchor if available
    try:
        from clinical_toolkit import RiskBlender
        blender = RiskBlender(random_state=42)
        blender_dict = blender.fit_predict(
            cohort_spec, train_records, eval_records,
            x_train=x_tr_raw, y_train=y_tr, x_eval=x_ev_raw,
        )
        blender_probs = np.array(
            [float(blender_dict.get(p.patient_id, p.baseline_api_prob)) for p in eval_records],
            dtype=np.float32,
        )
    except Exception:
        blender_probs = None

    if blender_probs is not None:
        blended_logits = 0.75 * np.array([safe_logit(p) for p in meta_probs]) + 0.25 * np.array([safe_logit(p) for p in blender_probs])
        final_probs = np.array([safe_sigmoid(l) for l in blended_logits])
    else:
        final_probs = meta_probs

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
    """Stage 2: Greedy care allocation maximizing expected net benefit within budget and constraints."""
    state = CareAllocationState(cohort_spec, eval_records)
    candidate_tiers = [
        t for t in cohort_spec.interventions.keys()
        if t != InterventionTier.NONE
    ]

    ranked_patients = sorted(
        eval_records,
        key=lambda p: -float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5))),
    )

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
