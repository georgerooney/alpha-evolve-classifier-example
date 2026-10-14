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

    # Cross-features addressing legacy blindspots
    acuity_claims = acuity * cl
    acuity_vitals = acuity * ev
    claims_vitals_diff = cl - ev
    claims_ur_diff = cl - ur

    return [
        b_prob,
        l_bp,
        acuity,
        cl,
        ev,
        ur,
        contra_count,
        acuity_claims,
        acuity_vitals,
        claims_vitals_diff,
        claims_ur_diff,
    ]


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

    # Augment features with legacy model scores and cross-organ interactions
    meta_train = np.array([_extract_patient_meta(p) for p in train_records], dtype=np.float64)
    meta_eval = np.array([_extract_patient_meta(p) for p in eval_records], dtype=np.float64)

    X_tr = np.nan_to_num(np.hstack([x_train, meta_train]), nan=0.0, posinf=0.0, neginf=0.0)
    X_ev = np.nan_to_num(np.hstack([x_eval, meta_eval]), nan=0.0, posinf=0.0, neginf=0.0)

    n_splits = min(5, len(y_train) // 5)
    if n_splits < 2:
        n_splits = 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    final_probs = np.zeros(len(eval_records), dtype=np.float64)

    for tr_idx, _ in skf.split(X_tr, y_train):
        X_fold_tr, y_fold_tr = X_tr[tr_idx], y_train[tr_idx]

        # LightGBM Classifier
        lgb_model = lgb.LGBMClassifier(
            n_estimators=110,
            learning_rate=0.04,
            num_leaves=16,
            max_depth=4,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=1.2,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        lgb_model.fit(X_fold_tr, y_fold_tr)

        # XGBoost Classifier
        xgb_model = xgb.XGBClassifier(
            n_estimators=90,
            learning_rate=0.04,
            max_depth=3,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=1.2,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
            n_jobs=1,
        )
        xgb_model.fit(X_fold_tr, y_fold_tr)

        # Standardized Logistic Regression
        scaler = StandardScaler()
        X_fold_tr_sc = scaler.fit_transform(X_fold_tr)
        X_ev_sc = scaler.transform(X_ev)

        lr_model = LogisticRegression(C=0.25, max_iter=1000, random_state=42)
        lr_model.fit(X_fold_tr_sc, y_fold_tr)

        ev_lgb = lgb_model.predict_proba(X_ev)[:, 1]
        ev_xgb = xgb_model.predict_proba(X_ev)[:, 1]
        ev_lr = lr_model.predict_proba(X_ev_sc)[:, 1]
        final_probs += (0.45 * ev_lgb + 0.35 * ev_xgb + 0.20 * ev_lr) / n_splits

    final_probs = np.clip(final_probs, 0.001, 0.999)
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
    h1 = sorted(raw_candidates, key=lambda c: c[2], reverse=True)
    h2 = sorted(raw_candidates, key=lambda c: c[2] / max(c[3], 15.0), reverse=True)
    h3 = sorted(raw_candidates, key=lambda c: c[2] / (max(c[3], 15.0) ** 0.5), reverse=True)

    best_state = state_ref
    best_benefit = -1e9

    for candidate_order in [h1, h2, h3]:
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
