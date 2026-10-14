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


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Multi-model stacking ensemble with clinical discrepancy features and calibrated blending."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    import lightgbm as lgb
    import xgboost as xgb

    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)

    x_train_clean = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_eval_clean = np.nan_to_num(x_eval, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.asarray(y_train, dtype=int)

    # Base rate calibration shift to counteract legacy elective over-prediction
    y_mean = float(np.mean(y_train)) if len(y_train) > 0 else 0.2
    y_mean = max(1e-4, min(1.0 - 1e-4, y_mean))
    base_train_logits = [safe_logit(float(p.baseline_api_prob)) for p in train_records]
    shift = safe_logit(y_mean) - float(np.mean(base_train_logits))

    # Helper: extract domain-specific submodel discordance and interaction features
    def extract_extra_features(records, shift_val):
        rows = []
        min_acuity = float(getattr(cohort_spec, "min_acuity_score", 0.0))
        for p in records:
            base_p = float(getattr(p, "baseline_api_prob", 0.5))
            base_logit = safe_logit(base_p)
            shifted_base_logit = base_logit + shift_val
            shifted_base_p = safe_sigmoid(shifted_base_logit)

            submodels = getattr(p, "baseline_submodel_scores", {}) or {}
            claims_p = float(submodels.get("claims_linear", base_p))
            vitals_p = float(submodels.get("ehr_vitals", base_p))
            util_p = float(submodels.get("utilization_rules", base_p))

            acuity = float(getattr(p, "acuity_score", 0.0))
            contra = getattr(p, "contraindications", ()) or ()
            contra_count = float(len(contra))

            # Acute discordance and uncertainty signals
            v_c_diff = vitals_p - claims_p
            v_u_diff = vitals_p - util_p
            c_u_diff = claims_p - util_p
            b_c_diff = base_p - claims_p
            b_v_diff = base_p - vitals_p

            sub_vals = [claims_p, vitals_p, util_p]
            sub_mean = float(np.mean(sub_vals))
            sub_std = float(np.std(sub_vals))
            sub_spread = max(sub_vals) - min(sub_vals)

            acuity_rel = acuity - min_acuity
            acuity_x_base = acuity * base_p
            acuity_x_vitals = acuity * vitals_p
            elective_corr = base_p / (acuity + 0.1)

            row = [
                base_p,
                base_logit,
                shifted_base_p,
                shifted_base_logit,
                claims_p,
                safe_logit(claims_p),
                vitals_p,
                safe_logit(vitals_p),
                util_p,
                safe_logit(util_p),
                acuity,
                acuity_rel,
                contra_count,
                v_c_diff,
                v_u_diff,
                c_u_diff,
                b_c_diff,
                b_v_diff,
                sub_mean,
                sub_std,
                sub_spread,
                acuity_x_base,
                acuity_x_vitals,
                elective_corr,
            ]
            rows.append(row)
        arr = np.array(rows, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    extra_train = extract_extra_features(train_records, shift)
    extra_eval = extract_extra_features(eval_records, shift)

    X_train_aug = np.hstack([x_train_clean, extra_train])
    X_eval_aug = np.hstack([x_eval_clean, extra_eval])

    # Run toolkit reference RiskBlender
    rb_probs = None
    try:
        blender = RiskBlender(
            linear_weight=0.35,
            lgbm_weight=0.40,
            xgb_weight=0.25,
            baseline_prior_weight=0.08,
            random_state=42,
        )
        rb_dict = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train_clean,
            y_train=y_train,
            x_eval=x_eval_clean,
        )
        rb_probs = np.array(
            [float(rb_dict.get(p.patient_id, p.baseline_api_prob)) for p in eval_records],
            dtype=float,
        )
        rb_probs = np.clip(rb_probs, 1e-4, 1.0 - 1e-4)
    except Exception:
        rb_probs = None

    # Multi-model cross-validated stacking
    n_train = len(train_records)
    n_eval = len(eval_records)
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    n_splits = min(4, max(2, pos_count, neg_count))

    oof_lgb = np.zeros(n_train, dtype=float)
    oof_xgb = np.zeros(n_train, dtype=float)
    oof_hgb = np.zeros(n_train, dtype=float)
    oof_lr = np.zeros(n_train, dtype=float)

    eval_lgb = np.zeros(n_eval, dtype=float)
    eval_xgb = np.zeros(n_eval, dtype=float)
    eval_hgb = np.zeros(n_eval, dtype=float)
    eval_lr = np.zeros(n_eval, dtype=float)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for trn_idx, val_idx in skf.split(X_train_aug, y_train):
        x_tr, y_tr = X_train_aug[trn_idx], y_train[trn_idx]
        x_val = X_train_aug[val_idx]

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=160,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=4,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        )
        clf_lgb.fit(x_tr, y_tr)
        oof_lgb[val_idx] = clf_lgb.predict_proba(x_val)[:, 1]
        eval_lgb += clf_lgb.predict_proba(X_eval_aug)[:, 1] / n_splits

        clf_xgb = xgb.XGBClassifier(
            n_estimators=140,
            learning_rate=0.035,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
        clf_xgb.fit(x_tr, y_tr)
        oof_xgb[val_idx] = clf_xgb.predict_proba(x_val)[:, 1]
        eval_xgb += clf_xgb.predict_proba(X_eval_aug)[:, 1] / n_splits

        clf_hgb = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.035,
            max_depth=4,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=42,
        )
        clf_hgb.fit(x_tr, y_tr)
        oof_hgb[val_idx] = clf_hgb.predict_proba(x_val)[:, 1]
        eval_hgb += clf_hgb.predict_proba(X_eval_aug)[:, 1] / n_splits

        clf_lr = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=0.3, max_iter=300, random_state=42)),
        ])
        clf_lr.fit(x_tr, y_tr)
        oof_lr[val_idx] = clf_lr.predict_proba(x_val)[:, 1]
        eval_lr += clf_lr.predict_proba(X_eval_aug)[:, 1] / n_splits

    base_tr_p = extra_train[:, 2]
    base_ev_p = extra_eval[:, 2]

    def to_logit(arr):
        clipped = np.clip(arr, 1e-4, 1.0 - 1e-4)
        return np.array([safe_logit(float(v)) for v in clipped], dtype=float)

    Z_oof = np.column_stack([
        to_logit(oof_lgb),
        to_logit(oof_xgb),
        to_logit(oof_hgb),
        to_logit(oof_lr),
        to_logit(base_tr_p),
    ])

    Z_ev = np.column_stack([
        to_logit(eval_lgb),
        to_logit(eval_xgb),
        to_logit(eval_hgb),
        to_logit(eval_lr),
        to_logit(base_ev_p),
    ])

    meta_lr = LogisticRegression(C=0.5, max_iter=200, random_state=42)
    meta_lr.fit(Z_oof, y_train)
    stacked_eval = meta_lr.predict_proba(Z_ev)[:, 1]

    simple_eval = (
        0.30 * eval_lgb
        + 0.30 * eval_xgb
        + 0.20 * eval_hgb
        + 0.10 * eval_lr
        + 0.10 * base_ev_p
    )
    custom_probs = 0.70 * stacked_eval + 0.30 * simple_eval

    if rb_probs is not None:
        z_custom = to_logit(custom_probs)
        z_rb = to_logit(rb_probs)
        blended_logit = 0.65 * z_custom + 0.35 * z_rb
        final_probs = np.array([safe_sigmoid(float(v)) for v in blended_logit], dtype=float)
    else:
        final_probs = custom_probs

    final_probs = np.clip(final_probs, 1e-4, 1.0 - 1e-4)

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        out[p.patient_id] = round(float(final_probs[idx]), 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Value-maximizing allocation prioritizing net clinical ROI under constraints."""
    state = CareAllocationState(cohort_spec, eval_records)
    ranked_patients = sorted(
        eval_records,
        key=lambda p: -float(predicted_risks.get(p.patient_id, p.baseline_api_prob)),
    )

    candidate_tiers = [
        t for t in cohort_spec.interventions.keys()
        if t != InterventionTier.NONE
    ]

    for p in ranked_patients:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))

        valid_tiers = []
        for tier in candidate_tiers:
            if state.fits(p.patient_id, tier):
                benefit = state.expected_net_benefit(p.patient_id, risk, tier)
                if benefit > 0.0:
                    valid_tiers.append((tier, benefit))

        if not valid_tiers:
            continue

        # Prioritize the tier yielding highest net benefit for this patient
        valid_tiers.sort(key=lambda x: -x[1])
        for tier, _ in valid_tiers:
            if state.fits(p.patient_id, tier):
                if state.assign(p.patient_id, tier):
                    break

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
