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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb

try:
    from clinical_toolkit import (
        ClinicalFeatureBuilder,
        CareAllocationState,
        RiskBlender,
        safe_logit,
        safe_sigmoid,
    )
except ImportError:
    try:
        import clinical_toolkit
        RiskBlender = getattr(clinical_toolkit, "RiskBlender", None)
        ClinicalFeatureBuilder = getattr(clinical_toolkit, "ClinicalFeatureBuilder", None)
        CareAllocationState = getattr(clinical_toolkit, "CareAllocationState", None)
        safe_logit = getattr(clinical_toolkit, "safe_logit", None)
        safe_sigmoid = getattr(clinical_toolkit, "safe_sigmoid", None)
    except Exception:
        RiskBlender = None


def _safe_logit_fn(p, eps=1e-5):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid_fn(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _align_prevalence(probs: np.ndarray, target_prev: float) -> np.ndarray:
    """Monotonically shifts logits so mean probability matches target prevalence.
    Preserves exact patient ranking (ROC-AUC, PR-AUC, Sens@Spec80) while optimizing Brier score.
    """
    target = np.clip(target_prev, 0.01, 0.99)
    current_mean = float(np.mean(probs))
    if abs(current_mean - target) < 1e-4:
        return np.clip(probs, 0.001, 0.999)

    logits = _safe_logit_fn(probs)
    low, high = -6.0, 6.0
    for _ in range(25):
        mid = (low + high) / 2.0
        m = float(np.mean(_sigmoid_fn(logits + mid)))
        if m < target:
            low = mid
        else:
            high = mid
    delta = (low + high) / 2.0
    return np.clip(_sigmoid_fn(logits + delta), 0.001, 0.999)


def _extract_patient_meta(p):
    b_prob = float(getattr(p, "baseline_api_prob", 0.5))
    b_clamped = np.clip(b_prob, 1e-4, 1.0 - 1e-4)
    l_bp = float(np.log(b_clamped / (1.0 - b_clamped)))
    acuity = float(getattr(p, "acuity_score", 0.0))
    sub = getattr(p, "baseline_submodel_scores", {}) or {}
    cl = float(sub.get("claims_linear", b_prob))
    ev = float(sub.get("ehr_vitals", b_prob))
    ur = float(sub.get("utilization_rules", b_prob))
    contra_count = float(len(getattr(p, "contraindications", []) or []))

    # Log-odds of legacy linear submodels
    l_cl = _safe_logit_fn(cl)
    l_ev = _safe_logit_fn(ev)
    l_ur = _safe_logit_fn(ur)

    # Discrepancy & discordance features (acute vs chronic vs utilization mismatch)
    cl_ev_diff = cl - ev
    cl_ur_diff = cl - ur
    ev_ur_diff = ev - ur
    abs_cl_ev = abs(cl - ev)
    abs_cl_ur = abs(cl - ur)
    sub_std = float(np.std([cl, ev, ur]))
    sub_max = max(cl, ev, ur)
    sub_min = min(cl, ev, ur)

    # Acuity interactions
    acuity_claims = acuity * cl
    acuity_vitals = acuity * ev
    acuity_ur = acuity * ur
    acuity_base = acuity * b_prob
    acuity_diff_b = acuity - b_prob
    acuity_diff_ev = acuity - ev

    # Elderly low-acuity elective over-prediction syndrome
    age = float(getattr(p, "age", 65.0) or 65.0)
    elderly_low_acuity = 1.0 if (age >= 65.0 and acuity < 0.35) else 0.0
    age_acuity = (age / 100.0) * acuity

    # Additional clinical signals if present on record
    extra = []
    for attr in [
        "charlson_comorbidity_index",
        "prior_admissions_count",
        "ed_visits_count",
        "length_of_stay",
        "sdoh_risk_score",
    ]:
        val = getattr(p, attr, None)
        if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
            extra.append(float(val))
        else:
            extra.append(0.0)

    return [
        b_prob,
        l_bp,
        acuity,
        cl,
        ev,
        ur,
        l_cl,
        l_ev,
        l_ur,
        contra_count,
        cl_ev_diff,
        cl_ur_diff,
        ev_ur_diff,
        abs_cl_ev,
        abs_cl_ur,
        sub_std,
        sub_max,
        sub_min,
        acuity_claims,
        acuity_vitals,
        acuity_ur,
        acuity_base,
        acuity_diff_b,
        acuity_diff_ev,
        elderly_low_acuity,
        age_acuity,
    ] + extra


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: High-discrimination, calibrated multi-model ensemble for 30-day readmission."""
    if not eval_records:
        return {}

    eval_ids = [p.patient_id for p in eval_records]
    base_eval_probs = np.array(
        [float(getattr(p, "baseline_api_prob", 0.5)) for p in eval_records],
        dtype=np.float64,
    )

    if not train_records:
        return {pid: float(p) for pid, p in zip(eval_ids, base_eval_probs)}

    # Build non-linear interaction features via toolkit
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)

    y_train = np.asarray(y_train, dtype=np.int32)
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        return {pid: float(p) for pid, p in zip(eval_ids, base_eval_probs)}

    # Optional RiskBlender execution if available in toolkit
    rb_preds_arr = None
    if RiskBlender is not None:
        try:
            blender = RiskBlender(
                linear_weight=0.30,
                lgbm_weight=0.40,
                xgb_weight=0.25,
                baseline_prior_weight=0.05,
                random_state=42,
            )
            rb_dict = blender.fit_predict(
                cohort_spec,
                train_records,
                eval_records,
                x_train=x_train,
                y_train=y_train,
                x_eval=x_eval,
            )
            rb_preds_arr = np.array(
                [float(rb_dict.get(pid, base_eval_probs[i])) for i, pid in enumerate(eval_ids)],
                dtype=np.float64,
            )
        except Exception:
            rb_preds_arr = None

    # Augment features with legacy model scores and cross-organ interactions
    meta_train = np.array([_extract_patient_meta(p) for p in train_records], dtype=np.float64)
    meta_eval = np.array([_extract_patient_meta(p) for p in eval_records], dtype=np.float64)

    X_tr = np.nan_to_num(np.hstack([x_train, meta_train]), nan=0.0, posinf=0.0, neginf=0.0)
    X_ev = np.nan_to_num(np.hstack([x_eval, meta_eval]), nan=0.0, posinf=0.0, neginf=0.0)

    n_splits = min(5, len(y_train) // 5)
    if n_splits < 2:
        n_splits = 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y_train), dtype=np.float64)
    fold_eval_preds = np.zeros(len(eval_records), dtype=np.float64)

    for tr_idx, val_idx in skf.split(X_tr, y_train):
        X_fold_tr, y_fold_tr = X_tr[tr_idx], y_train[tr_idx]
        X_fold_val = X_tr[val_idx]

        # 1. LightGBM Classifier (with subsample_freq=1 to activate row bagging)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.035,
            num_leaves=15,
            max_depth=4,
            min_child_samples=12,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.80,
            reg_alpha=0.20,
            reg_lambda=1.5,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        lgb_model.fit(X_fold_tr, y_fold_tr)

        # 2. XGBoost Classifier
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.035,
            max_depth=4,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.20,
            reg_lambda=1.5,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
            n_jobs=1,
        )
        xgb_model.fit(X_fold_tr, y_fold_tr)

        # 3. ExtraTrees Classifier (randomized split bagging for variance reduction)
        et_model = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=1,
        )
        et_model.fit(X_fold_tr, y_fold_tr)

        # 4. Standardized Logistic Regression
        scaler = StandardScaler()
        X_fold_tr_sc = scaler.fit_transform(X_fold_tr)
        X_fold_val_sc = scaler.transform(X_fold_val)
        X_ev_sc = scaler.transform(X_ev)

        lr_model = LogisticRegression(C=0.20, max_iter=1000, random_state=42)
        lr_model.fit(X_fold_tr_sc, y_fold_tr)

        val_lgb = lgb_model.predict_proba(X_fold_val)[:, 1]
        val_xgb = xgb_model.predict_proba(X_fold_val)[:, 1]
        val_et = et_model.predict_proba(X_fold_val)[:, 1]
        val_lr = lr_model.predict_proba(X_fold_val_sc)[:, 1]

        fold_val_blend = 0.35 * val_lgb + 0.30 * val_xgb + 0.20 * val_et + 0.15 * val_lr
        oof_preds[val_idx] = fold_val_blend

        ev_lgb = lgb_model.predict_proba(X_ev)[:, 1]
        ev_xgb = xgb_model.predict_proba(X_ev)[:, 1]
        ev_et = et_model.predict_proba(X_ev)[:, 1]
        ev_lr = lr_model.predict_proba(X_ev_sc)[:, 1]

        fold_ev_blend = 0.35 * ev_lgb + 0.30 * ev_xgb + 0.20 * ev_et + 0.15 * ev_lr
        fold_eval_preds += fold_ev_blend / n_splits

    # Probability calibration (Platt scaling on out-of-fold predictions)
    oof_logits = _safe_logit_fn(oof_preds).reshape(-1, 1)
    ev_logits = _safe_logit_fn(fold_eval_preds).reshape(-1, 1)

    calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    calibrator.fit(oof_logits, y_train)
    calibrated_preds = calibrator.predict_proba(ev_logits)[:, 1]

    # Combine with RiskBlender if available (giving majority weight to calibrated 4-model ensemble)
    if rb_preds_arr is not None:
        custom_preds = 0.65 * calibrated_preds + 0.35 * rb_preds_arr
    else:
        custom_preds = calibrated_preds

    # Monotonic prevalence alignment to target base rate (optimizes Brier without altering AUC/Sens)
    target_prev = float(np.mean(y_train))
    final_probs = _align_prevalence(custom_preds, target_prev)
    return {pid: float(p) for pid, p in zip(eval_ids, final_probs)}


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Maximize expected net benefit and net ROI via multi-heuristic allocation."""
    state_ref = CareAllocationState(cohort_spec, eval_records)

    raw_candidates = []
    for p in eval_records:
        risk = float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5)))
        for tier, spec in cohort_spec.interventions.items():
            if tier != InterventionTier.NONE and state_ref.fits(p.patient_id, tier):
                cost = float(getattr(spec, "cost_usd", 0.0))
                enb = state_ref.expected_net_benefit(p.patient_id, risk, tier)
                if enb > 0.0:
                    raw_candidates.append((p.patient_id, tier, enb, cost))

    if not raw_candidates:
        return state_ref.to_plan()

    # Define candidate ranking heuristics:
    # 1. Pure Expected Net Benefit
    # 2. Budget efficiency (ENB / cost)
    # 3. Hybrid efficiency (ENB / sqrt(cost))
    # 4. Super-linear benefit efficiency ((ENB^1.2) / cost)
    h1 = sorted(raw_candidates, key=lambda c: c[2], reverse=True)
    h2 = sorted(raw_candidates, key=lambda c: c[2] / max(c[3], 15.0), reverse=True)
    h3 = sorted(raw_candidates, key=lambda c: c[2] / (max(c[3], 15.0) ** 0.5), reverse=True)
    h4 = sorted(raw_candidates, key=lambda c: (c[2] ** 1.2) / max(c[3], 15.0), reverse=True)

    best_state = state_ref
    best_benefit = -1e9

    for candidate_order in [h1, h2, h3, h4]:
        candidate_state = CareAllocationState(cohort_spec, eval_records)
        assigned_pids = set()
        accum_benefit = 0.0

        for pid, tier, enb, _ in candidate_order:
            if pid not in assigned_pids and candidate_state.fits(pid, tier):
                if candidate_state.assign(pid, tier):
                    assigned_pids.add(pid)
                    accum_benefit += enb

        if accum_benefit > best_benefit:
            best_benefit = accum_benefit
            best_state = candidate_state

    return best_state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
