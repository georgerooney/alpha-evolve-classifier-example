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


def _extract_engineered_features(records: Sequence[PatientRecord], cohort_spec: CohortSpec) -> np.ndarray:
    """Extract clinical submodel discordance, acuity margin, and elective correction features."""
    import numpy as np

    min_acuity = float(getattr(cohort_spec, "min_acuity_score", 0.0) or 0.0)
    feats = []
    for p in records:
        b_prob = float(getattr(p, "baseline_api_prob", 0.5) or 0.5)
        b_prob = max(1e-5, min(1.0 - 1e-5, b_prob))
        b_logit = safe_logit(b_prob)

        sub = getattr(p, "baseline_submodel_scores", None) or {}
        c_score = float(sub.get("claims_linear", 0.0) or 0.0)
        e_score = float(sub.get("ehr_vitals", 0.0) or 0.0)
        u_score = float(sub.get("utilization_rules", 0.0) or 0.0)

        acuity = float(getattr(p, "acuity_score", 0.0) or 0.0)
        contra = getattr(p, "contraindications", None) or []
        contra_count = float(len(contra))

        sub_mean = (c_score + e_score + u_score) / 3.0
        sub_std = float(np.std([c_score, e_score, u_score]))
        acute_chronic_diff = e_score - c_score
        acute_util_diff = e_score - u_score
        acuity_diff_threshold = acuity - min_acuity

        # Correction for legacy elective low-acuity over-prediction
        low_acuity_overpred = b_logit / (1.0 + max(0.0, acuity))
        elective_mismatch = 1.0 if (acuity < min_acuity + 0.1 and b_prob > 0.4) else 0.0

        feats.append([
            b_prob,
            b_logit,
            c_score,
            e_score,
            u_score,
            sub_mean,
            sub_std,
            acute_chronic_diff,
            acute_util_diff,
            acuity,
            acuity_diff_threshold,
            contra_count,
            b_logit * acuity,
            c_score * acuity,
            e_score * acuity,
            u_score * acuity,
            low_acuity_overpred,
            elective_mismatch,
        ])
    return np.array(feats, dtype=np.float32)


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Multi-model cross-validated Super Learner ensemble with clinical calibration."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import HistGradientBoostingClassifier
    import lightgbm as lgb
    import xgboost as xgb

    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_tr_raw, y_train_raw = builder.fit_transform(cohort_spec, train_records)
    x_ev_raw = builder.transform(cohort_spec, eval_records)

    y_train = np.array(y_train_raw, dtype=np.int32)
    n_train = len(train_records)
    n_eval = len(eval_records)

    # Augmented feature matrices
    tr_eng = _extract_engineered_features(train_records, cohort_spec)
    ev_eng = _extract_engineered_features(eval_records, cohort_spec)

    x_train_aug = np.nan_to_num(np.hstack([x_tr_raw, tr_eng]), nan=0.0, posinf=1e4, neginf=-1e4)
    x_eval_aug = np.nan_to_num(np.hstack([x_ev_raw, ev_eng]), nan=0.0, posinf=1e4, neginf=-1e4)

    # Stratified 5-fold cross-validation for out-of-fold stacking
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros((n_train, 4), dtype=np.float32)
    eval_fold_preds = np.zeros((n_eval, 4), dtype=np.float32)

    for tr_idx, val_idx in skf.split(x_train_aug, y_train):
        x_tr_fold, y_tr_fold = x_train_aug[tr_idx], y_train[tr_idx]
        x_val_fold = x_train_aug[val_idx]

        # Model 0: LightGBM
        m_lgb = lgb.LGBMClassifier(
            n_estimators=130,
            learning_rate=0.04,
            num_leaves=16,
            max_depth=5,
            min_child_samples=15,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        m_lgb.fit(x_tr_fold, y_tr_fold)
        oof_preds[val_idx, 0] = m_lgb.predict_proba(x_val_fold)[:, 1]
        eval_fold_preds[:, 0] += m_lgb.predict_proba(x_eval_aug)[:, 1] / n_splits

        # Model 1: XGBoost
        m_xgb = xgb.XGBClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        m_xgb.fit(x_tr_fold, y_tr_fold)
        oof_preds[val_idx, 1] = m_xgb.predict_proba(x_val_fold)[:, 1]
        eval_fold_preds[:, 1] += m_xgb.predict_proba(x_eval_aug)[:, 1] / n_splits

        # Model 2: HistGradientBoosting
        m_hgb = HistGradientBoostingClassifier(
            max_iter=110,
            learning_rate=0.04,
            max_leaf_nodes=16,
            min_samples_leaf=15,
            l2_regularization=0.5,
            random_state=42,
        )
        m_hgb.fit(x_tr_fold, y_tr_fold)
        oof_preds[val_idx, 2] = m_hgb.predict_proba(x_val_fold)[:, 1]
        eval_fold_preds[:, 2] += m_hgb.predict_proba(x_eval_aug)[:, 1] / n_splits

        # Model 3: Regularized Scaled Logistic Regression
        scaler = StandardScaler()
        x_tr_scaled = scaler.fit_transform(x_tr_fold)
        x_val_scaled = scaler.transform(x_val_fold)
        x_ev_scaled = scaler.transform(x_eval_aug)

        m_lr = LogisticRegression(C=0.3, max_iter=300, random_state=42, solver="lbfgs")
        m_lr.fit(x_tr_scaled, y_tr_fold)
        oof_preds[val_idx, 3] = m_lr.predict_proba(x_val_scaled)[:, 1]
        eval_fold_preds[:, 3] += m_lr.predict_proba(x_ev_scaled)[:, 1] / n_splits

    # Baseline API prior logits shifted to hospital base rate
    y_mean = max(1e-4, min(1.0 - 1e-4, float(np.mean(y_train))))
    base_train_logits = np.array([safe_logit(float(p.baseline_api_prob)) for p in train_records])
    base_eval_logits = np.array([safe_logit(float(p.baseline_api_prob)) for p in eval_records])
    shift = safe_logit(y_mean) - float(np.mean(base_train_logits))
    train_base_prob = safe_sigmoid(base_train_logits + shift)
    eval_base_prob = safe_sigmoid(base_eval_logits + shift)

    # Convert probabilities to logits for linear meta-combination
    def _to_logits(arr: np.ndarray) -> np.ndarray:
        clipped = np.clip(arr, 1e-5, 1.0 - 1e-5)
        return np.log(clipped / (1.0 - clipped))

    z_train = _to_logits(np.column_stack([oof_preds, train_base_prob]))
    z_eval = _to_logits(np.column_stack([eval_fold_preds, eval_base_prob]))

    # Meta-learner on out-of-fold predictions
    meta_clf = LogisticRegression(C=0.5, solver="lbfgs", max_iter=200, random_state=42)
    meta_clf.fit(z_train, y_train)
    meta_eval_probs = meta_clf.predict_proba(z_eval)[:, 1]

    # Robust ensemble blend (weights: LGBM 0.35, XGB 0.30, HGB 0.15, LR 0.12, Base 0.08)
    fixed_weights = np.array([0.35, 0.30, 0.15, 0.12, 0.08], dtype=np.float32)
    fixed_eval_logits = z_eval @ fixed_weights
    fixed_eval_probs = safe_sigmoid(fixed_eval_logits)

    combined_probs = 0.60 * meta_eval_probs + 0.40 * fixed_eval_probs

    # Incorporate native toolkit RiskBlender if available
    try:
        rb = RiskBlender(
            linear_weight=0.35,
            lgbm_weight=0.40,
            xgb_weight=0.25,
            baseline_prior_weight=0.06,
            random_state=42,
        )
        rb_dict = rb.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_tr_raw,
            y_train=y_train,
            x_eval=x_ev_raw,
        )
        rb_eval = np.array([float(rb_dict[p.patient_id]) for p in eval_records])
        combined_probs = 0.82 * combined_probs + 0.18 * rb_eval
    except Exception:
        pass

    # Strictly monotonic logit recalibration to match cohort base rate (optimizes Brier score)
    comb_logits = _to_logits(combined_probs)
    current_mean_prob = float(np.mean(combined_probs))
    calib_shift = safe_logit(y_mean) - safe_logit(max(1e-4, min(1.0 - 1e-4, current_mean_prob)))
    final_probs = safe_sigmoid(comb_logits + 0.50 * calib_shift)

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
    """Stage 2: Optimal expected net benefit allocation under strict constraints."""
    state = CareAllocationState(cohort_spec, eval_records)

    candidate_tiers = [
        t
        for t, s in sorted(
            cohort_spec.interventions.items(),
            key=lambda kv: -kv[1].relative_risk_reduction,
        )
        if t != InterventionTier.NONE
    ]

    # Rank patients by predicted risk descending
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
