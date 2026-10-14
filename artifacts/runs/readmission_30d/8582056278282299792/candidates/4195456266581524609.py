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


import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Multi-model ensemble using the built-in RiskBlender with non-linear clinical features."""
    # Build clinical features with clinically validated non-linear interactions
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_tr_raw, y_train = builder.fit_transform(cohort_spec, train_records)
    x_ev_raw = builder.transform(cohort_spec, eval_records)

    # Impute potential nulls and scale features
    x_tr_clean = np.nan_to_num(x_tr_raw, nan=0.0, posinf=0.0, neginf=0.0)
    x_ev_clean = np.nan_to_num(x_ev_raw, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    x_tr_scaled = np.clip(scaler.fit_transform(x_tr_clean), -10.0, 10.0)
    x_ev_scaled = np.clip(scaler.transform(x_ev_clean), -10.0, 10.0)

    # Blend using the pre-built RiskBlender which handles the core ensemble tuning natively
    blender = RiskBlender(
        linear_weight=0.40, 
        lgbm_weight=0.35, 
        xgb_weight=0.25, 
        baseline_prior_weight=0.08, 
        random_state=42
    )
    
    return blender.fit_predict(
        cohort_spec, 
        train_records, 
        eval_records, 
        x_train=x_tr_scaled, 
        y_train=y_train, 
        x_eval=x_ev_scaled
    )


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Optimal care allocation maximizing expected net benefit subject to budget and clinical constraints."""
    state = CareAllocationState(cohort_spec, eval_records)

    active_tiers = [
        t
        for t, s in cohort_spec.interventions.items()
        if t != InterventionTier.NONE and s.max_slots > 0 and s.relative_risk_reduction > 0.0
    ]
    if not active_tiers:
        return state.to_plan()

    # Patient priority: maximum positive net benefit achievable across all feasible active tiers
    def patient_priority(p: PatientRecord) -> float:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        max_b = 0.0
        for tier in active_tiers:
            if state.fits(p.patient_id, tier):
                b = state.expected_net_benefit(p.patient_id, risk, tier)
                if b > max_b:
                    max_b = b
        return max_b

    ranked_patients = sorted(eval_records, key=lambda p: -patient_priority(p))

    for p in ranked_patients:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        candidate_tiers = []
        for tier in active_tiers:
            if state.fits(p.patient_id, tier):
                b = state.expected_net_benefit(p.patient_id, risk, tier)
                if b > 0.0:
                    candidate_tiers.append((tier, b))

        # Assign tier with maximum expected net benefit
        candidate_tiers.sort(key=lambda x: -x[1])
        for tier, _ in candidate_tiers:
            if state.fits(p.patient_id, tier):
                if state.assign(p.patient_id, tier):
                    break

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
