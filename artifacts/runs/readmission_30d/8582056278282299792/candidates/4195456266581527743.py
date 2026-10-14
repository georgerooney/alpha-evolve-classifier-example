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


def extract_clinical_dissonance_features(records: Sequence[PatientRecord]) -> np.ndarray:
    """Extract domain-specific dissonance and syndrome features between legacy submodels and acuity."""
    import numpy as np

    feats = []
    for p in records:
        base_p = float(getattr(p, "baseline_api_prob", 0.2))
        sub = getattr(p, "baseline_submodel_scores", None) or {}
        c_lin = float(sub.get("claims_linear", base_p))
        e_vit = float(sub.get("ehr_vitals", base_p))
        u_rul = float(sub.get("utilization_rules", base_p))
        acuity = float(getattr(p, "acuity_score", 1.0) or 1.0)
        contra_cnt = float(len(getattr(p, "contraindications", ()) or ()))

        # Submodel conflict and discrepancy dynamics
        c_minus_v = c_lin - e_vit  # High in elderly low-acuity elective patients
        u_minus_v = u_rul - e_vit  # High in SDoH / frequent utilization bounce-backs
        u_minus_c = u_rul - c_lin
        sub_max = max(c_lin, e_vit, u_rul)
        sub_min = min(c_lin, e_vit, u_rul)
        sub_spread = sub_max - sub_min
        sub_mean = (c_lin + e_vit + u_rul) / 3.0
        base_minus_sub = base_p - sub_mean

        # Submodel logit representations
        c_logit = safe_logit(float(np.clip(c_lin, 1e-4, 1.0 - 1e-4)))
        v_logit = safe_logit(float(np.clip(e_vit, 1e-4, 1.0 - 1e-4)))
        u_logit = safe_logit(float(np.clip(u_rul, 1e-4, 1.0 - 1e-4)))
        base_logit = safe_logit(float(np.clip(base_p, 1e-4, 1.0 - 1e-4)))

        # Acuity-moderated syndrome interactions
        acuity_vital = acuity * e_vit
        acuity_claims = acuity * c_lin
        debiased_base = base_p * (acuity / (acuity + 1.5))

        # Explicit heuristic flags for legacy failure modes
        low_acuity_elective = 1.0 if (c_lin > 0.4 and acuity <= 2.0 and e_vit < 0.3) else 0.0
        sdoh_bounce = 1.0 if (u_rul > 0.4 and u_minus_v > 0.15) else 0.0

        feats.append([
            base_p,
            base_logit,
            c_lin,
            e_vit,
            u_rul,
            c_logit,
            v_logit,
            u_logit,
            acuity,
            contra_cnt,
            c_minus_v,
            u_minus_v,
            u_minus_c,
            sub_spread,
            sub_mean,
            base_minus_sub,
            acuity_vital,
            acuity_claims,
            debiased_base,
            low_acuity_elective,
            sdoh_bounce,
        ])
    return np.array(feats, dtype=np.float32)


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Multi-architecture out-of-fold stacking ensemble with dissonance de-biasing."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    import xgboost as xgb

    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)

    extra_train = extract_clinical_dissonance_features(train_records)
    extra_eval = extract_clinical_dissonance_features(eval_records)

    x_train = np.nan_to_num(np.hstack([x_train_raw, extra_train]), nan=0.0, posinf=5.0, neginf=-5.0)
    x_eval = np.nan_to_num(np.hstack([x_eval_raw, extra_eval]), nan=0.0, posinf=5.0, neginf=-5.0)

    pos_count = int(np.sum(y_train == 1))
    if pos_count < 2 or len(np.unique(y_train)) < 2:
        return {p.patient_id: float(p.baseline_api_prob) for p in eval_records}

    n_splits = min(5, max(2, pos_count))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    n_tr = len(train_records)
    n_ev = len(eval_records)

    oof_lgb = np.zeros(n_tr, dtype=np.float32)
    oof_xgb = np.zeros(n_tr, dtype=np.float32)
    oof_hgb = np.zeros(n_tr, dtype=np.float32)
    oof_lr = np.zeros(n_tr, dtype=np.float32)

    eval_lgb = np.zeros(n_ev, dtype=np.float32)
    eval_xgb = np.zeros(n_ev, dtype=np.float32)
    eval_hgb = np.zeros(n_ev, dtype=np.float32)
    eval_lr = np.zeros(n_ev, dtype=np.float32)

    for tr_idx, va_idx in skf.split(x_train, y_train):
        xt, yt = x_train[tr_idx], y_train[tr_idx]
        xv, yv = x_train[va_idx], y_train[va_idx]

        # 1. LightGBM
        m_lgb = lgb.LGBMClassifier(
            n_estimators=140,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=4,
            min_child_samples=12,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        m_lgb.fit(xt, yt)
        oof_lgb[va_idx] = m_lgb.predict_proba(xv)[:, 1]
        eval_lgb += m_lgb.predict_proba(x_eval)[:, 1] / float(n_splits)

        # 2. XGBoost
        m_xgb = xgb.XGBClassifier(
            n_estimators=140,
            learning_rate=0.04,
            max_depth=3,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=1,
        )
        m_xgb.fit(xt, yt)
        oof_xgb[va_idx] = m_xgb.predict_proba(xv)[:, 1]
        eval_xgb += m_xgb.predict_proba(x_eval)[:, 1] / float(n_splits)

        # 3. HistGradientBoosting
        m_hgb = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.04,
            max_leaf_nodes=15,
            max_depth=4,
            min_samples_leaf=12,
            l2_regularization=1.0,
            random_state=42,
        )
        m_hgb.fit(xt, yt)
        oof_hgb[va_idx] = m_hgb.predict_proba(xv)[:, 1]
        eval_hgb += m_hgb.predict_proba(x_eval)[:, 1] / float(n_splits)

        # 4. Standard Scaled Logistic Regression
        scaler = StandardScaler()
        xt_s = scaler.fit_transform(xt)
        xv_s = scaler.transform(xv)
        xev_s = scaler.transform(x_eval)

        m_lr = LogisticRegression(C=0.2, max_iter=250, solver="lbfgs", random_state=42)
        m_lr.fit(xt_s, yt)
        oof_lr[va_idx] = m_lr.predict_proba(xv_s)[:, 1]
        eval_lr += m_lr.predict_proba(xev_s)[:, 1] / float(n_splits)

    # Build meta-learner feature matrices from OOF predictions and submodel dissonance
    def build_meta_matrix(lgb_p, xgb_p, hgb_p, lr_p, records):
        cols = []
        for i, p in enumerate(records):
            base_p = float(getattr(p, "baseline_api_prob", 0.2))
            sub = getattr(p, "baseline_submodel_scores", None) or {}
            c_lin = float(sub.get("claims_linear", base_p))
            e_vit = float(sub.get("ehr_vitals", base_p))
            u_rul = float(sub.get("utilization_rules", base_p))
            acuity = float(getattr(p, "acuity_score", 1.0) or 1.0)

            c_minus_v = c_lin - e_vit
            u_minus_v = u_rul - e_vit
            debiased_base = base_p * (acuity / (acuity + 1.5))

            cols.append([
                safe_logit(float(np.clip(lgb_p[i], 1e-4, 1.0 - 1e-4))),
                safe_logit(float(np.clip(xgb_p[i], 1e-4, 1.0 - 1e-4))),
                safe_logit(float(np.clip(hgb_p[i], 1e-4, 1.0 - 1e-4))),
                safe_logit(float(np.clip(lr_p[i], 1e-4, 1.0 - 1e-4))),
                safe_logit(float(np.clip(base_p, 1e-4, 1.0 - 1e-4))),
                c_minus_v,
                u_minus_v,
                debiased_base,
                acuity,
            ])
        return np.array(cols, dtype=np.float32)

    m_train = build_meta_matrix(oof_lgb, oof_xgb, oof_hgb, oof_lr, train_records)
    m_eval = build_meta_matrix(eval_lgb, eval_xgb, eval_hgb, eval_lr, eval_records)

    meta_learner = LogisticRegression(C=0.5, penalty="l2", solver="lbfgs", max_iter=250, random_state=42)
    meta_learner.fit(m_train, y_train)
    meta_probs = meta_learner.predict_proba(m_eval)[:, 1]

    # Combine meta-classifier log-loss with direct multi-model ensemble average
    base_ensemble = 0.35 * eval_lgb + 0.35 * eval_xgb + 0.15 * eval_hgb + 0.15 * eval_lr
    combined_probs = 0.75 * meta_probs + 0.25 * base_ensemble

    # Calibration alignment to cohort baseline prevalence
    y_mean = max(1e-4, min(1.0 - 1e-4, float(np.mean(y_train))))
    pred_mean = max(1e-4, min(1.0 - 1e-4, float(np.mean(combined_probs))))
    calib_shift = safe_logit(y_mean) - safe_logit(pred_mean)
    calibrated_probs = safe_sigmoid(safe_logit(combined_probs) + 0.40 * calib_shift)

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        prob = float(np.clip(calibrated_probs[idx], 1e-4, 1.0 - 1e-4))
        out[p.patient_id] = round(prob, 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Care allocation maximizing expected net benefit per patient under all constraints."""
    state = CareAllocationState(cohort_spec, eval_records)

    ranked_patients = sorted(
        eval_records,
        key=lambda p: -float(predicted_risks.get(p.patient_id, p.baseline_api_prob)),
    )

    available_tiers = [
        t
        for t in cohort_spec.interventions.keys()
        if t != InterventionTier.NONE
    ]

    for p in ranked_patients:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        best_tier = InterventionTier.NONE
        best_net_benefit = 0.0

        for tier in available_tiers:
            if state.fits(p.patient_id, tier):
                benefit = state.expected_net_benefit(p.patient_id, risk, tier)
                if benefit > best_net_benefit:
                    best_net_benefit = benefit
                    best_tier = tier

        if best_tier != InterventionTier.NONE:
            state.assign(p.patient_id, best_tier)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
