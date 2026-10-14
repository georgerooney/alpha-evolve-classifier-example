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
import numpy as np
from typing import Dict, Sequence, List, Tuple, Optional, Any

from clinical_toolkit import (
    ClinicalFeatureBuilder,
    RiskBlender,
    CareAllocationState,
)


def _safe_logit(p: float, eps: float = 1e-4) -> float:
    """Numerically stable logit transformation."""
    p_clamped = max(eps, min(1.0 - eps, float(p)))
    return float(np.log(p_clamped / (1.0 - p_clamped)))


def _extract_clinical_meta_features(records: Sequence[Any]) -> np.ndarray:
    """Extract clinical divergence, cross-submodel dispersion, and syndrome indicators."""
    rows = []
    for p in records:
        b_prob = float(getattr(p, "baseline_api_prob", 0.5))
        b_prob = max(1e-4, min(1.0 - 1e-4, b_prob))
        b_logit = _safe_logit(b_prob)

        sub = getattr(p, "baseline_submodel_scores", None) or {}
        claims = float(sub.get("claims_linear", b_prob))
        ehr = float(sub.get("ehr_vitals", b_prob))
        util = float(sub.get("utilization_rules", b_prob))

        claims = max(1e-4, min(1.0 - 1e-4, claims))
        ehr = max(1e-4, min(1.0 - 1e-4, ehr))
        util = max(1e-4, min(1.0 - 1e-4, util))

        claims_logit = _safe_logit(claims)
        ehr_logit = _safe_logit(ehr)
        util_logit = _safe_logit(util)

        acuity = float(getattr(p, "acuity_score", 0.0))

        # Divergence: acute vitals vs chronic claims baseline
        acute_vs_claims = ehr - claims
        acute_vs_claims_logit = ehr_logit - claims_logit

        # Frequent utilization vs chronic claims baseline
        util_vs_claims = util - claims
        util_vs_claims_logit = util_logit - claims_logit

        # Acute vitals vs utilization
        ehr_vs_util = ehr - util
        ehr_vs_util_logit = ehr_logit - util_logit

        # Acuity interactions
        acuity_x_bprob = acuity * b_prob
        acuity_x_ehr = acuity * ehr
        acuity_x_util = acuity * util
        acuity_x_diff = acuity * acute_vs_claims

        # Cross-submodel dispersion
        sub_vals = [claims, ehr, util]
        sub_max = max(sub_vals)
        sub_min = min(sub_vals)
        sub_spread = sub_max - sub_min
        sub_mean = float(np.mean(sub_vals))
        sub_std = float(np.std(sub_vals))

        contra = getattr(p, "contraindications", ())
        n_contra = float(len(contra)) if contra else 0.0

        # Elderly elective overprediction correction indicator:
        # High claims baseline but low acute vitals and low acuity
        elderly_elective_proxy = max(0.0, claims - ehr) / (acuity + 0.5)
        
        # Additional interactions
        acuity_sq = acuity * acuity
        contra_acuity = n_contra * acuity
        util_x_ehr = util * ehr

        rows.append([
            b_prob, b_logit,
            claims, claims_logit,
            ehr, ehr_logit,
            util, util_logit,
            acuity, acuity_sq,
            acute_vs_claims, acute_vs_claims_logit,
            util_vs_claims, util_vs_claims_logit,
            ehr_vs_util, ehr_vs_util_logit,
            acuity_x_bprob, acuity_x_ehr, acuity_x_util, acuity_x_diff,
            sub_max, sub_min, sub_spread, sub_mean, sub_std,
            n_contra, contra_acuity, elderly_elective_proxy,
            util_x_ehr
        ])
    return np.array(rows, dtype=np.float32)


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Multi-model K-fold ensemble with Platt calibration."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.preprocessing import StandardScaler

    # Extract non-linear syndrome interaction features from clinical feature builder
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train_raw = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)

    # Extract domain meta-features and divergence indicators
    meta_train = _extract_clinical_meta_features(train_records)
    meta_eval = _extract_clinical_meta_features(eval_records)

    x_train = np.nan_to_num(np.hstack([x_train_raw, meta_train]), nan=0.0, posinf=1.0, neginf=-1.0)
    x_eval = np.nan_to_num(np.hstack([x_eval_raw, meta_eval]), nan=0.0, posinf=1.0, neginf=-1.0)
    y_train = np.array(y_train_raw, dtype=np.float32)

    sample_weights = np.ones(len(train_records), dtype=np.float32)
    for i, p in enumerate(train_records):
        y_val = float(getattr(p, "outcome_label", 0.0) or 0.0)
        t_val = getattr(p, "time_to_event_days", None)
        if y_val > 0.5:
            if t_val is not None:
                try:
                    t_f = float(t_val)
                    t_f = max(1.0, min(30.0, t_f))
                except Exception:
                    t_f = 15.0
            else:
                t_f = 15.0
            # Higher weight for early readmissions
            sample_weights[i] = 1.0 + 0.5 * (30.0 - t_f) / 30.0

    n_pos = int(np.sum(y_train == 1))
    min_class = min(n_pos, len(y_train) - n_pos)
    if min_class >= 2:
        n_splits = min(3, min_class)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds = list(skf.split(x_train, y_train))
    else:
        n_splits = 2
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        folds = list(kf.split(x_train))

    oof_lgb = np.zeros(len(y_train))
    oof_xgb = np.zeros(len(y_train))
    oof_lr = np.zeros(len(y_train))

    eval_lgb = np.zeros(len(eval_records))
    eval_xgb = np.zeros(len(eval_records))
    eval_lr = np.zeros(len(eval_records))

    for fold, (tr_idx, val_idx) in enumerate(folds):
        x_tr_f, y_tr_f = x_train[tr_idx], y_train[tr_idx]
        x_va_f = x_train[val_idx]
        sw_tr_f = sample_weights[tr_idx]

        # 1. LightGBM Classifier
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=24,
            max_depth=5,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=1,
        )
        clf_lgb.fit(x_tr_f, y_tr_f, sample_weight=sw_tr_f)
        oof_lgb[val_idx] = clf_lgb.predict_proba(x_va_f)[:, 1]
        eval_lgb += clf_lgb.predict_proba(x_eval)[:, 1] / float(n_splits)

        # 2. XGBoost Classifier
        clf_xgb = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.5,
            eval_metric="logloss",
            random_state=42 + fold,
            n_jobs=1,
        )
        clf_xgb.fit(x_tr_f, y_tr_f, sample_weight=sw_tr_f)
        oof_xgb[val_idx] = clf_xgb.predict_proba(x_va_f)[:, 1]
        eval_xgb += clf_xgb.predict_proba(x_eval)[:, 1] / float(n_splits)

        # 3. Regularized Logistic Regression
        scaler = StandardScaler()
        x_tr_sc = scaler.fit_transform(x_tr_f)
        x_va_sc = scaler.transform(x_va_f)
        x_ev_sc = scaler.transform(x_eval)

        clf_lr = LogisticRegression(
            C=0.1,
            max_iter=300,
            penalty="l2",
            solver="lbfgs",
            random_state=42 + fold,
        )
        clf_lr.fit(x_tr_sc, y_tr_f)
        oof_lr[val_idx] = clf_lr.predict_proba(x_va_sc)[:, 1]
        eval_lr += clf_lr.predict_proba(x_ev_sc)[:, 1] / float(n_splits)

    # Baseline legacy predictions
    base_tr = np.array([float(getattr(p, "baseline_api_prob", 0.5)) for p in train_records], dtype=np.float32)
    base_ev = np.array([float(getattr(p, "baseline_api_prob", 0.5)) for p in eval_records], dtype=np.float32)

    def _arr_to_logits(arr: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        c = np.clip(arr, eps, 1.0 - eps)
        return np.log(c / (1.0 - c))

    # Logit-space multi-model blending
    z_oof = (
        0.40 * _arr_to_logits(oof_lgb)
        + 0.30 * _arr_to_logits(oof_xgb)
        + 0.20 * _arr_to_logits(oof_lr)
        + 0.10 * _arr_to_logits(base_tr)
    )

    z_eval = (
        0.40 * _arr_to_logits(eval_lgb)
        + 0.30 * _arr_to_logits(eval_xgb)
        + 0.20 * _arr_to_logits(eval_lr)
        + 0.10 * _arr_to_logits(base_ev)
    )

    # Extract predictions from the toolkit RiskBlender
    try:
        blender = RiskBlender(
            linear_weight=0.30,
            lgbm_weight=0.35,
            xgb_weight=0.25,
            baseline_prior_weight=0.10,
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
            # Blend our custom ensemble with the toolkit's RiskBlender output
            z_eval = 0.70 * z_eval + 0.30 * _arr_to_logits(rb_ev_arr)
    except Exception:
        pass

    # Strictly monotonic Platt calibration on OOF predictions
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(z_oof.reshape(-1, 1), y_train)

    # Monotonic scaling preserves exact ROC-AUC and Sens@Spec0.80 while optimizing Brier
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
    active_tiers = [
        t for t in cohort_spec.interventions.keys()
        if getattr(t, "name", str(t)).upper() != "NONE" and str(t) != "NONE"
    ]

    # Identify promising intervention-patient pairs
    # We create a list of all feasible (patient, tier, nb, cost) options
    options = []
    for p in eval_records:
        risk = float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5)))
        for tier in active_tiers:
            if state.fits(p.patient_id, tier):
                nb = state.expected_net_benefit(p.patient_id, risk, tier)
                if nb > 0.0:
                    cost = float(getattr(cohort_spec.interventions[tier], "cost_usd", 1000.0))
                    options.append({
                        "patient_id": p.patient_id,
                        "tier": tier,
                        "nb": nb,
                        "cost": cost,
                        "eff": nb / (cost + 1e-3),
                        "risk": risk
                    })

    # Sort options globally by a composite score of efficiency and absolute NB
    # This prioritizes high ROI options but also ensures high absolute value is captured.
    options.sort(key=lambda x: (-(x["nb"] * (1.0 + 0.5 * x["eff"])), -x["risk"]))

    assigned_patients = set()
    for opt in options:
        pid = opt["patient_id"]
        if pid in assigned_patients:
            continue
        tier = opt["tier"]
        
        # Check if it still fits within budget and slots
        if state.fits(pid, tier):
            if state.assign(pid, tier):
                assigned_patients.add(pid)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
