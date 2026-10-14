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


def _extract_clinical_meta_features(records: Sequence[PatientRecord]) -> np.ndarray:
    """Extract clinical divergence and legacy model meta-features."""
    rows = []
    for p in records:
        b_prob = float(getattr(p, "baseline_api_prob", 0.5))
        b_prob = max(1e-4, min(1.0 - 1e-4, b_prob))
        b_logit = float(safe_logit(b_prob))

        sub = getattr(p, "baseline_submodel_scores", None) or {}
        claims = float(sub.get("claims_linear", b_prob))
        ehr = float(sub.get("ehr_vitals", b_prob))
        util = float(sub.get("utilization_rules", b_prob))

        claims = max(1e-4, min(1.0 - 1e-4, claims))
        ehr = max(1e-4, min(1.0 - 1e-4, ehr))
        util = max(1e-4, min(1.0 - 1e-4, util))

        claims_logit = float(safe_logit(claims))
        ehr_logit = float(safe_logit(ehr))
        util_logit = float(safe_logit(util))

        acuity = float(getattr(p, "acuity_score", 0.0))

        # Divergence: acute vitals vs chronic claims baseline
        acute_vs_claims = ehr - claims
        acute_vs_claims_logit = ehr_logit - claims_logit

        # Frequent utilization vs chronic claims baseline
        util_vs_claims = util - claims
        util_vs_claims_logit = util_logit - claims_logit

        # Acute vitals vs utilization
        ehr_vs_util = ehr - util

        # Acuity interactions
        acuity_x_bprob = acuity * b_prob
        acuity_x_ehr = acuity * ehr
        acuity_x_util = acuity * util

        # Cross-submodel dispersion
        sub_max = max(claims, ehr, util)
        sub_min = min(claims, ehr, util)
        sub_spread = sub_max - sub_min
        sub_mean = (claims + ehr + util) / 3.0

        contra = getattr(p, "contraindications", ())
        n_contra = float(len(contra)) if contra else 0.0

        rows.append([
            b_prob, b_logit,
            claims, claims_logit,
            ehr, ehr_logit,
            util, util_logit,
            acuity,
            acute_vs_claims, acute_vs_claims_logit,
            util_vs_claims, util_vs_claims_logit,
            ehr_vs_util,
            acuity_x_bprob, acuity_x_ehr, acuity_x_util,
            sub_max, sub_min, sub_spread, sub_mean,
            n_contra,
        ])
    return np.array(rows, dtype=np.float32)


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Multi-model K-fold ensemble with syndrome interactions and monotonic Platt calibration."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    # Extract non-linear syndrome interaction features from clinical feature builder
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)

    # Extract domain meta-features and divergence indicators
    meta_train = _extract_clinical_meta_features(train_records)
    meta_eval = _extract_clinical_meta_features(eval_records)

    x_train = np.nan_to_num(np.hstack([x_train_raw, meta_train]), nan=0.0, posinf=1.0, neginf=-1.0)
    x_eval = np.nan_to_num(np.hstack([x_eval_raw, meta_eval]), nan=0.0, posinf=1.0, neginf=-1.0)
    y_train = np.array(y_train, dtype=np.float32)

    n_pos = int(np.sum(y_train == 1))
    n_splits = min(5, max(2, n_pos))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb = np.zeros(len(y_train))
    oof_xgb = np.zeros(len(y_train))
    oof_hgb = np.zeros(len(y_train))
    oof_lr = np.zeros(len(y_train))

    eval_lgb = np.zeros(len(eval_records))
    eval_xgb = np.zeros(len(eval_records))
    eval_hgb = np.zeros(len(eval_records))
    eval_lr = np.zeros(len(eval_records))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_tr_f, y_tr_f = x_train[tr_idx], y_train[tr_idx]
        x_va_f = x_train[val_idx]

        # 1. LightGBM
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=160,
            learning_rate=0.035,
            num_leaves=22,
            max_depth=5,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=1,
        )
        clf_lgb.fit(x_tr_f, y_tr_f)
        oof_lgb[val_idx] = clf_lgb.predict_proba(x_va_f)[:, 1]
        eval_lgb += clf_lgb.predict_proba(x_eval)[:, 1] / float(n_splits)

        # 2. XGBoost
        clf_xgb = xgb.XGBClassifier(
            n_estimators=140,
            learning_rate=0.035,
            max_depth=4,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42 + fold,
            n_jobs=1,
        )
        clf_xgb.fit(x_tr_f, y_tr_f)
        oof_xgb[val_idx] = clf_xgb.predict_proba(x_va_f)[:, 1]
        eval_xgb += clf_xgb.predict_proba(x_eval)[:, 1] / float(n_splits)

        # 3. HistGradientBoostingClassifier
        clf_hgb = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.035,
            max_leaf_nodes=18,
            min_samples_leaf=15,
            l2_regularization=0.5,
            random_state=42 + fold,
        )
        clf_hgb.fit(x_tr_f, y_tr_f)
        oof_hgb[val_idx] = clf_hgb.predict_proba(x_va_f)[:, 1]
        eval_hgb += clf_hgb.predict_proba(x_eval)[:, 1] / float(n_splits)

        # 4. Standardized Logistic Regression
        scaler = StandardScaler()
        x_tr_sc = scaler.fit_transform(x_tr_f)
        x_va_sc = scaler.transform(x_va_f)
        x_ev_sc = scaler.transform(x_eval)

        clf_lr = LogisticRegression(
            C=0.25,
            max_iter=300,
            penalty="l2",
            solver="lbfgs",
            random_state=42 + fold,
        )
        clf_lr.fit(x_tr_sc, y_tr_f)
        oof_lr[val_idx] = clf_lr.predict_proba(x_va_sc)[:, 1]
        eval_lr += clf_lr.predict_proba(x_ev_sc)[:, 1] / float(n_splits)

    # Baseline prior probabilities
    base_tr = np.array([float(getattr(p, "baseline_api_prob", 0.5)) for p in train_records], dtype=np.float32)
    base_ev = np.array([float(getattr(p, "baseline_api_prob", 0.5)) for p in eval_records], dtype=np.float32)

    def _arr_to_logits(arr: np.ndarray) -> np.ndarray:
        c = np.clip(arr, 1e-5, 1.0 - 1e-5)
        return np.log(c / (1.0 - c))

    # Logit-space blending
    z_oof = (
        0.30 * _arr_to_logits(oof_lgb)
        + 0.25 * _arr_to_logits(oof_xgb)
        + 0.15 * _arr_to_logits(oof_hgb)
        + 0.20 * _arr_to_logits(oof_lr)
        + 0.10 * _arr_to_logits(base_tr)
    )

    z_eval = (
        0.30 * _arr_to_logits(eval_lgb)
        + 0.25 * _arr_to_logits(eval_xgb)
        + 0.15 * _arr_to_logits(eval_hgb)
        + 0.20 * _arr_to_logits(eval_lr)
        + 0.10 * _arr_to_logits(base_ev)
    )

    # Blend with toolkit RiskBlender if available
    try:
        blender = RiskBlender(
            linear_weight=0.35,
            lgbm_weight=0.35,
            xgb_weight=0.30,
            baseline_prior_weight=0.08,
            random_state=42,
        )
        rb_dict = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train_raw,
            y_train=y_train,
            x_eval=x_eval_raw,
        )
        if rb_dict and len(rb_dict) == len(eval_records):
            rb_ev_arr = np.array([float(rb_dict.get(p.patient_id, 0.5)) for p in eval_records], dtype=np.float32)
            z_eval = 0.80 * z_eval + 0.20 * _arr_to_logits(rb_ev_arr)
    except Exception:
        pass

    # Strictly monotonic Platt calibration on OOF predictions
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(z_oof.reshape(-1, 1), y_train)

    # If slope is positive, Platt scaling preserves exact rank ordering while optimizing Brier
    if calibrator.coef_[0][0] > 0.01:
        eval_probs = calibrator.predict_proba(z_eval.reshape(-1, 1))[:, 1]
    else:
        y_mean = float(np.mean(y_train))
        delta = np.log(max(1e-4, y_mean) / max(1e-4, 1.0 - y_mean)) - float(np.mean(z_oof))
        eval_probs = 1.0 / (1.0 + np.exp(-(z_eval + delta)))

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        prob = float(eval_probs[idx])
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, prob)), 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Maximize expected net clinical benefit under budget and slot constraints."""
    state = CareAllocationState(cohort_spec, eval_records)
    active_tiers = [t for t in cohort_spec.interventions.keys() if t != InterventionTier.NONE]

    # Prioritize patients by maximum potential net benefit
    prioritized_patients = []
    for p in eval_records:
        risk = float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5)))
        best_potential_nb = 0.0
        for tier in active_tiers:
            if state.fits(p.patient_id, tier):
                nb = state.expected_net_benefit(p.patient_id, risk, tier)
                if nb > best_potential_nb:
                    best_potential_nb = nb
        prioritized_patients.append((best_potential_nb, risk, p))

    # Higher potential savings served first, breaking ties with raw risk
    prioritized_patients.sort(key=lambda item: (-item[0], -item[1]))

    for _, risk, p in prioritized_patients:
        candidates = []
        for tier in active_tiers:
            if state.fits(p.patient_id, tier):
                nb = state.expected_net_benefit(p.patient_id, risk, tier)
                if nb > 0.0:
                    candidates.append((nb, tier))

        if not candidates:
            continue

        # Sort tiers by expected net benefit descending to choose the highest-value tier
        candidates.sort(key=lambda item: -item[0])

        for _, tier in candidates:
            if state.fits(p.patient_id, tier):
                state.assign(p.patient_id, tier)
                break

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
