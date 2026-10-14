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
    """Stage 1: Multi-Manifold Super Learner Ensemble with Actuarial Discordance & TTE Urgency Weighting."""
    import numpy as np
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    # 1. Base syndrome features
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train_raw = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)
    y_train = np.array(y_train_raw, dtype=np.float64)

    # 2. Extract rich domain & actuarial discordance features
    def extract_extra_features(records: Sequence[PatientRecord]) -> np.ndarray:
        feats = []
        for p in records:
            base_p = float(getattr(p, "baseline_api_prob", 0.5))
            sub = getattr(p, "baseline_submodel_scores", {})
            claims = float(sub.get("claims_linear", base_p))
            ehr = float(sub.get("ehr_vitals", base_p))
            util = float(sub.get("utilization_rules", base_p))
            acuity = float(getattr(p, "acuity_score", 0.5))
            contra_count = float(len(getattr(p, "contraindications", [])))

            feats.append([
                base_p,
                safe_logit(base_p),
                claims,
                ehr,
                util,
                ehr - claims,                      # Acute vital decompensation vs claims history
                util - claims,                     # Acute utilization velocity vs baseline claims
                max(ehr, util),                    # Peak acute clinical risk
                max(claims, ehr, util) - min(claims, ehr, util),  # Submodel discordance/uncertainty
                acuity,
                acuity * base_p,                   # High acuity + high baseline interaction
                acuity * ehr,                      # Severe acute vitals interaction
                acuity * util,                     # SDoH adherence / utilization spiral
                max(0.0, base_p - acuity),         # Elderly low-acuity elective over-prediction correction
                1.0 if acuity < 0.35 else 0.0,     # Low acuity elective indicator
                contra_count,                      # Multi-morbidity frailty burden
            ])
        return np.array(feats, dtype=np.float64)

    extra_train = extract_extra_features(train_records)
    extra_eval = extract_extra_features(eval_records)

    x_train_aug = np.nan_to_num(np.hstack([x_train_raw, extra_train]), nan=0.0, posinf=1e4, neginf=-1e4)
    x_eval_aug = np.nan_to_num(np.hstack([x_eval_raw, extra_eval]), nan=0.0, posinf=1e4, neginf=-1e4)

    # 3. Time-to-Event Urgency Sample Weighting (prioritizes rapid readmissions for top-tier sensitivity)
    sample_weights = np.ones(len(train_records), dtype=np.float64)
    for i, p in enumerate(train_records):
        if y_train[i] > 0.5:
            tte = getattr(p, "time_to_event_days", None)
            if tte is not None:
                try:
                    tte_f = float(tte)
                    if 0.0 <= tte_f <= 30.0:
                        sample_weights[i] = 1.0 + 0.35 * (30.0 - tte_f) / 30.0
                except (ValueError, TypeError):
                    pass
    sample_weights = sample_weights / max(1e-4, float(np.mean(sample_weights)))

    # 4. Platt calibration for legacy baseline odds
    train_base_logits = np.array([safe_logit(float(p.baseline_api_prob)) for p in train_records]).reshape(-1, 1)
    eval_base_logits = np.array([safe_logit(float(p.baseline_api_prob)) for p in eval_records]).reshape(-1, 1)

    base_platt = LogisticRegression(C=1.0, max_iter=200, random_state=42)
    base_platt.fit(train_base_logits, y_train)
    oof_base_cal = base_platt.predict_proba(train_base_logits)[:, 1]
    eval_base_cal = base_platt.predict_proba(eval_base_logits)[:, 1]

    # 5. Out-of-fold cross-validated Super Learner Stacking
    min_class = min(int(np.sum(y_train == 0)), int(np.sum(y_train == 1)))
    n_splits = max(2, min(5, min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb = np.zeros(len(train_records))
    oof_xgb = np.zeros(len(train_records))
    oof_lr = np.zeros(len(train_records))
    oof_et = np.zeros(len(train_records))

    eval_lgb_folds = np.zeros(len(eval_records))
    eval_xgb_folds = np.zeros(len(eval_records))
    eval_lr_folds = np.zeros(len(eval_records))
    eval_et_folds = np.zeros(len(eval_records))

    for tr_idx, val_idx in skf.split(x_train_aug, y_train):
        xt_tr, yt_tr = x_train_aug[tr_idx], y_train[tr_idx]
        xt_val, _ = x_train_aug[val_idx], y_train[val_idx]
        sw_tr = sample_weights[tr_idx]

        # Model A: LightGBM
        m_lgb = lgb.LGBMClassifier(
            n_estimators=160,
            learning_rate=0.035,
            num_leaves=16,
            max_depth=4,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=1.5,
            random_state=42,
            verbose=-1,
        )
        m_lgb.fit(xt_tr, yt_tr, sample_weight=sw_tr)
        oof_lgb[val_idx] = m_lgb.predict_proba(xt_val)[:, 1]
        eval_lgb_folds += m_lgb.predict_proba(x_eval_aug)[:, 1] / n_splits

        # Model B: XGBoost
        m_xgb = xgb.XGBClassifier(
            n_estimators=130,
            learning_rate=0.035,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=1.5,
            eval_metric="logloss",
            random_state=42,
        )
        m_xgb.fit(xt_tr, yt_tr, sample_weight=sw_tr)
        oof_xgb[val_idx] = m_xgb.predict_proba(xt_val)[:, 1]
        eval_xgb_folds += m_xgb.predict_proba(x_eval_aug)[:, 1] / n_splits

        # Model C: Regularized Scaled Logistic Regression
        scaler = StandardScaler()
        xt_tr_sc = scaler.fit_transform(xt_tr)
        xt_val_sc = scaler.transform(xt_val)
        x_ev_sc = scaler.transform(x_eval_aug)

        m_lr = LogisticRegression(C=0.5, penalty="l2", solver="lbfgs", max_iter=300, random_state=42)
        m_lr.fit(xt_tr_sc, yt_tr, sample_weight=sw_tr)
        oof_lr[val_idx] = m_lr.predict_proba(xt_val_sc)[:, 1]
        eval_lr_folds += m_lr.predict_proba(x_ev_sc)[:, 1] / n_splits

        # Model D: ExtraTrees Classifier
        m_et = ExtraTreesClassifier(
            n_estimators=120,
            max_depth=6,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=42,
        )
        m_et.fit(xt_tr, yt_tr, sample_weight=sw_tr)
        oof_et[val_idx] = m_et.predict_proba(xt_val)[:, 1]
        eval_et_folds += m_et.predict_proba(x_eval_aug)[:, 1] / n_splits

    # 6. Fit Meta-Learner on Out-Of-Fold Logit Matrix
    def to_logits(prob_arr: np.ndarray) -> np.ndarray:
        return np.array([safe_logit(float(p)) for p in prob_arr]).reshape(-1, 1)

    z_oof = np.hstack([
        to_logits(oof_lgb),
        to_logits(oof_xgb),
        to_logits(oof_lr),
        to_logits(oof_et),
        to_logits(oof_base_cal),
    ])

    z_eval = np.hstack([
        to_logits(eval_lgb_folds),
        to_logits(eval_xgb_folds),
        to_logits(eval_lr_folds),
        to_logits(eval_et_folds),
        to_logits(eval_base_cal),
    ])

    meta_stacker = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=200, random_state=42)
    meta_stacker.fit(z_oof, y_train)
    super_eval_probs = meta_stacker.predict_proba(z_eval)[:, 1]

    # 7. Blend with RiskBlender native ensemble if operational
    final_probs = super_eval_probs
    try:
        rb = RiskBlender(
            linear_weight=0.35,
            lgbm_weight=0.35,
            xgb_weight=0.25,
            baseline_prior_weight=0.05,
            random_state=42,
        )
        rb_dict = rb.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train_raw,
            y_train=y_train,
            x_eval=x_eval_raw,
        )
        rb_eval_probs = np.array([rb_dict.get(p.patient_id, p.baseline_api_prob) for p in eval_records])
        final_probs = 0.80 * super_eval_probs + 0.20 * rb_eval_probs
    except Exception:
        pass

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        prob = float(np.clip(final_probs[idx], 1e-4, 1.0 - 1e-4))
        out[p.patient_id] = round(prob, 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Maximal Net-Benefit Knapsack Care Allocation with Strict Feasibility Guarantees."""
    state = CareAllocationState(cohort_spec, eval_records)

    candidate_tiers = [
        t for t in cohort_spec.interventions.keys() if t != InterventionTier.NONE
    ]

    # Evaluate each patient's candidate tiers sorted by positive expected net benefit
    patient_dockets = []
    for p in eval_records:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        tier_options = []
        for tier in candidate_tiers:
            if state.fits(p.patient_id, tier):
                enb = state.expected_net_benefit(p.patient_id, risk, tier)
                if enb > 0.0:
                    tier_options.append((enb, tier))
        if tier_options:
            tier_options.sort(key=lambda item: -item[0])  # Best ROI tier first
            max_benefit = tier_options[0][0]
            patient_dockets.append((max_benefit, p, tier_options))
        else:
            patient_dockets.append((0.0, p, []))

    # Prioritize assignment for patients with highest potential net clinical & financial gain
    patient_dockets.sort(key=lambda item: -item[0])

    for _, p, tier_options in patient_dockets:
        for enb, tier in tier_options:
            if state.fits(p.patient_id, tier):
                state.assign(p.patient_id, tier)
                break

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
