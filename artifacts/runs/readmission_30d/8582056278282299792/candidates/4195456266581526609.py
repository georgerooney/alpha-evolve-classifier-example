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
    """Stage 1: Multi-model stacking ensemble with clinical submodel extraction and calibration."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    import xgboost as xgb

    # 1. Base syndrome features from frozen toolkit
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)
    y_train = np.array(y_train, dtype=np.int32)
    y_mean = max(1e-4, min(1.0 - 1e-4, float(np.mean(y_train))))

    # 2. Extract legacy submodel scores, acuity, and interaction terms
    def extract_extra_features(records):
        feats = []
        base_logit_ref = float(safe_logit(y_mean))
        for p in records:
            b_prob = float(getattr(p, 'baseline_api_prob', 0.5) or 0.5)
            b_prob = max(1e-4, min(1.0 - 1e-4, b_prob))
            b_logit = float(safe_logit(b_prob))

            sub = getattr(p, 'baseline_submodel_scores', None) or {}
            c_lin = float(sub.get('claims_linear', 0.0) or 0.0)
            e_vit = float(sub.get('ehr_vitals', 0.0) or 0.0)
            u_rul = float(sub.get('utilization_rules', 0.0) or 0.0)

            acuity = float(getattr(p, 'acuity_score', 0.0) or 0.0)
            contras = getattr(p, 'contraindications', ()) or ()
            n_contra = float(len(contras))

            # Non-linear syndrome & legacy interactions
            centered_b_logit = b_logit - base_logit_ref
            acuity_x_claims = acuity * c_lin
            acuity_x_vitals = acuity * e_vit
            claims_vitals_diff = c_lin - e_vit
            acuity_x_prob = acuity * b_prob

            feats.append([
                b_prob, b_logit, centered_b_logit,
                c_lin, e_vit, u_rul,
                acuity, n_contra,
                acuity_x_claims, acuity_x_vitals,
                claims_vitals_diff, acuity_x_prob,
            ])
        return np.array(feats, dtype=np.float32)

    extra_tr = extract_extra_features(train_records)
    extra_ev = extract_extra_features(eval_records)

    x_tr_full = np.nan_to_num(np.hstack([x_train_raw, extra_tr]), nan=0.0, posinf=1e4, neginf=-1e4)
    x_ev_full = np.nan_to_num(np.hstack([x_eval_raw, extra_ev]), nan=0.0, posinf=1e4, neginf=-1e4)

    # 3. 5-Fold Stratified Cross-Validation for Stacking
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    n_train = len(train_records)
    n_eval = len(eval_records)

    oof_lgb = np.zeros(n_train, dtype=np.float32)
    oof_xgb = np.zeros(n_train, dtype=np.float32)
    oof_lr = np.zeros(n_train, dtype=np.float32)

    test_lgb = np.zeros(n_eval, dtype=np.float32)
    test_xgb = np.zeros(n_eval, dtype=np.float32)
    test_lr = np.zeros(n_eval, dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_tr_full, y_train)):
        x_tr_fold, y_tr_fold = x_tr_full[tr_idx], y_train[tr_idx]
        x_val_fold = x_tr_full[val_idx]

        # Model 1: LightGBM
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=110,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=1,
        )
        clf_lgb.fit(x_tr_fold, y_tr_fold)
        oof_lgb[val_idx] = clf_lgb.predict_proba(x_val_fold)[:, 1]
        test_lgb += clf_lgb.predict_proba(x_ev_full)[:, 1] / n_splits

        # Model 2: XGBoost
        clf_xgb = xgb.XGBClassifier(
            n_estimators=90,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42 + fold,
            verbosity=0,
            n_jobs=1,
        )
        clf_xgb.fit(x_tr_fold, y_tr_fold)
        oof_xgb[val_idx] = clf_xgb.predict_proba(x_val_fold)[:, 1]
        test_xgb += clf_xgb.predict_proba(x_ev_full)[:, 1] / n_splits

        # Model 3: Standard Scaler + Regularized Logistic Regression
        scaler = StandardScaler()
        x_tr_sc = scaler.fit_transform(x_tr_fold)
        x_val_sc = scaler.transform(x_val_fold)
        x_ev_sc = scaler.transform(x_ev_full)

        clf_lr = LogisticRegression(C=0.5, max_iter=300, random_state=42 + fold)
        clf_lr.fit(x_tr_sc, y_tr_fold)
        oof_lr[val_idx] = clf_lr.predict_proba(x_val_sc)[:, 1]
        test_lr += clf_lr.predict_proba(x_ev_sc)[:, 1] / n_splits

    # 4. Out-of-fold Meta-Learner Logistic Regression
    def clip_prob(arr):
        return np.clip(arr, 1e-5, 1.0 - 1e-5)

    base_tr_probs = np.array([float(getattr(p, 'baseline_api_prob', 0.5) or 0.5) for p in train_records], dtype=np.float32)
    base_ev_probs = np.array([float(getattr(p, 'baseline_api_prob', 0.5) or 0.5) for p in eval_records], dtype=np.float32)

    meta_x_train = np.column_stack([
        [float(safe_logit(p)) for p in clip_prob(oof_lgb)],
        [float(safe_logit(p)) for p in clip_prob(oof_xgb)],
        [float(safe_logit(p)) for p in clip_prob(oof_lr)],
        [float(safe_logit(p)) for p in clip_prob(base_tr_probs)],
    ])

    meta_x_eval = np.column_stack([
        [float(safe_logit(p)) for p in clip_prob(test_lgb)],
        [float(safe_logit(p)) for p in clip_prob(test_xgb)],
        [float(safe_logit(p)) for p in clip_prob(test_lr)],
        [float(safe_logit(p)) for p in clip_prob(base_ev_probs)],
    ])

    meta_learner = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    meta_learner.fit(meta_x_train, y_train)
    stack_probs = meta_learner.predict_proba(meta_x_eval)[:, 1]

    # 5. Blend with Frozen Toolkit RiskBlender
    rb_probs = None
    try:
        blender = RiskBlender(random_state=42)
        try:
            rb_dict = blender.fit_predict(cohort_spec, train_records, eval_records, x_train=x_train_raw, y_train=y_train, x_eval=x_eval_raw)
        except TypeError:
            rb_dict = blender.fit_predict(cohort_spec, train_records, eval_records)
        if rb_dict and len(rb_dict) == n_eval:
            rb_probs = np.array([float(rb_dict.get(p.patient_id, 0.5)) for p in eval_records], dtype=np.float32)
    except Exception:
        rb_probs = None

    if rb_probs is not None:
        combined_logits = np.array([
            0.75 * float(safe_logit(clip_prob(stack_probs)[i])) + 0.25 * float(safe_logit(clip_prob(rb_probs)[i]))
            for i in range(n_eval)
        ])
        final_probs = np.array([float(safe_sigmoid(val)) for val in combined_logits])
    else:
        final_probs = stack_probs

    # 6. Center probabilities to cohort prevalence to optimize Brier score while preserving ranks
    ev_mean = max(1e-4, min(1.0 - 1e-4, float(np.mean(final_probs))))
    if abs(ev_mean - y_mean) > 0.01:
        calib_shift = float(safe_logit(y_mean)) - float(safe_logit(ev_mean))
        final_probs = np.array([
            float(safe_sigmoid(float(safe_logit(clip_prob(p))) + 0.5 * calib_shift))
            for p in final_probs
        ])

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
    """Stage 2: Value-maximizing greedy allocation respecting budget and clinical constraints."""
    state = CareAllocationState(cohort_spec, eval_records)

    ranked_patients = sorted(
        eval_records,
        key=lambda p: -float(predicted_risks.get(p.patient_id, getattr(p, 'baseline_api_prob', 0.5))),
    )

    candidate_tiers = [t for t in cohort_spec.interventions.keys() if t != InterventionTier.NONE]

    for p in ranked_patients:
        risk = float(predicted_risks.get(p.patient_id, getattr(p, 'baseline_api_prob', 0.5)))
        feasible_tiers = []
        for tier in candidate_tiers:
            if state.fits(p.patient_id, tier):
                nb = state.expected_net_benefit(p.patient_id, risk, tier)
                if nb > 0.0:
                    feasible_tiers.append((tier, nb))

        if feasible_tiers:
            best_tier, _ = max(feasible_tiers, key=lambda item: item[1])
            state.assign(p.patient_id, best_tier)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
