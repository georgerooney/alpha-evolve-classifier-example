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
from typing import Dict, Sequence, Any, Optional
import numpy as np

# Import domain components from clinical_toolkit with resilient fallbacks
try:
    from clinical_toolkit import (
        ClinicalFeatureBuilder,
        RiskBlender,
        CareAllocationState,
        InterventionTier,
    )
except (ImportError, AttributeError):
    import sys
    _etk = sys.modules.get("clinical_toolkit", None)
    ClinicalFeatureBuilder = getattr(_etk, "ClinicalFeatureBuilder", globals().get("ClinicalFeatureBuilder"))
    RiskBlender = getattr(_etk, "RiskBlender", globals().get("RiskBlender"))
    CareAllocationState = getattr(_etk, "CareAllocationState", globals().get("CareAllocationState"))
    InterventionTier = getattr(_etk, "InterventionTier", globals().get("InterventionTier"))


def _fit_fallback_ensemble(
    cohort_spec: Any,
    train_records: Sequence[Any],
    eval_records: Sequence[Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
) -> Dict[str, float]:
    """Resilient fallback ensemble using LightGBM, XGBoost, and regularized Logistic Regression."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n_train = len(train_records)
    x_train_clean = np.nan_to_num(x_train, nan=0.0, posinf=1e5, neginf=-1e5)
    x_eval_clean = np.nan_to_num(x_eval, nan=0.0, posinf=1e5, neginf=-1e5)

    # 1. Standardized Logistic Regression for calibrated linear baseline
    try:
        scaler = StandardScaler()
        x_tr_scaled = scaler.fit_transform(x_train_clean)
        x_ev_scaled = scaler.transform(x_eval_clean)
        lr = LogisticRegression(C=0.1, max_iter=500, random_state=42)
        lr.fit(x_tr_scaled, y_train)
        p_lr = lr.predict_proba(x_ev_scaled)[:, 1]
    except Exception:
        p_lr = np.array([float(getattr(p, "baseline_api_prob", 0.2)) for p in eval_records])

    # 2. LightGBM for non-linear interactions
    try:
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            min_child_samples=max(5, int(n_train * 0.01)),
        )
        clf_lgb.fit(x_train_clean, y_train)
        p_lgb = clf_lgb.predict_proba(x_eval_clean)[:, 1]
    except Exception:
        p_lgb = p_lr

    # 3. XGBoost for cross-syndrome regularization
    try:
        clf_xgb = xgb.XGBClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        clf_xgb.fit(x_train_clean, y_train)
        p_xgb = clf_xgb.predict_proba(x_eval_clean)[:, 1]
    except Exception:
        p_xgb = p_lgb

    p_ensemble = 0.20 * p_lr + 0.45 * p_lgb + 0.35 * p_xgb

    scores = {}
    for i, p in enumerate(eval_records):
        base_p = float(getattr(p, "baseline_api_prob", 0.2))
        final_p = 0.92 * float(p_ensemble[i]) + 0.08 * base_p
        scores[p.patient_id] = float(np.clip(final_p, 0.0001, 0.9999))
    return scores


def fit_and_score_risk(
    cohort_spec: Any,
    train_records: Sequence[Any],
    eval_records: Sequence[Any],
) -> Dict[str, float]:
    """Stage 1: Feature generation with non-linear syndromes and blended risk prediction."""
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)

    raw_preds = None
    if RiskBlender is not None:
        try:
            blender = RiskBlender(
                linear_weight=0.20,
                lgbm_weight=0.45,
                xgb_weight=0.35,
                baseline_prior_weight=0.08,
                random_state=42,
            )
            raw_preds = blender.fit_predict(
                cohort_spec,
                train_records,
                eval_records,
                x_train=x_train,
                y_train=y_train,
                x_eval=x_eval,
            )
        except TypeError:
            try:
                blender = RiskBlender()
                raw_preds = blender.fit_predict(
                    cohort_spec,
                    train_records,
                    eval_records,
                    x_train=x_train,
                    y_train=y_train,
                    x_eval=x_eval,
                )
            except Exception:
                raw_preds = None
        except Exception:
            raw_preds = None

    if raw_preds is None:
        raw_preds = _fit_fallback_ensemble(
            cohort_spec, train_records, eval_records, x_train, y_train, x_eval
        )

    # Validate output contract: finite probability in [0.0001, 0.9999] for every patient
    scores = {}
    for p in eval_records:
        pid = p.patient_id
        val = raw_preds.get(pid, None) if isinstance(raw_preds, dict) else None
        if val is None or not np.isfinite(val):
            val = float(getattr(p, "baseline_api_prob", 0.2))
        scores[pid] = float(np.clip(val, 0.0001, 0.9999))
    return scores


def allocate_interventions(
    cohort_spec: Any,
    eval_records: Sequence[Any],
    predicted_risks: Dict[str, float],
) -> Any:
    """Stage 2: Multi-frontier knapsack allocation maximizing expected net benefit."""
    state_probe = CareAllocationState(cohort_spec, eval_records)

    candidate_pool = []
    for p in eval_records:
        pid = p.patient_id
        risk = float(predicted_risks.get(pid, getattr(p, "baseline_api_prob", 0.2)))
        for tier, intervention in cohort_spec.interventions.items():
            tier_name = getattr(tier, "name", str(tier))
            if "NONE" in tier_name.upper():
                continue
            if state_probe.fits(pid, tier):
                enb = state_probe.expected_net_benefit(pid, risk, tier)
                if enb > 0.0:
                    cost = getattr(intervention, "unit_cost", None)
                    if cost is None:
                        cost = getattr(intervention, "cost", None)
                    if cost is None:
                        rrr = getattr(intervention, "relative_risk_reduction", 0.0)
                        cost = max(1.0, risk * rrr * 16800.0 - enb)
                    cost = max(1.0, float(cost))
                    candidate_pool.append((enb, cost, pid, tier))

    # Evaluate multiple trade-off powers from pure ROI (1.0) to pure Net Benefit (0.0)
    best_state = None
    best_benefit = -1.0
    power_factors = [1.0, 0.8, 0.5, 0.2, 0.0]

    for p_factor in power_factors:
        candidate_state = CareAllocationState(cohort_spec, eval_records)
        ranked = sorted(
            candidate_pool,
            key=lambda c: (c[0] / ((c[1] + 10.0) ** p_factor), c[0]),
            reverse=True,
        )

        assigned = set()
        total_benefit = 0.0
        for enb, cost, pid, tier in ranked:
            if pid not in assigned and candidate_state.fits(pid, tier):
                candidate_state.assign(pid, tier)
                assigned.add(pid)
                total_benefit += enb

        if total_benefit > best_benefit or best_state is None:
            best_benefit = total_benefit
            best_state = candidate_state

    return best_state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
