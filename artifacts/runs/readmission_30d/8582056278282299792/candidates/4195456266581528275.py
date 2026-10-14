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
    """Stage 1: Use ClinicalFeatureBuilder with interactions and RiskBlender."""
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)

    blender = RiskBlender(
        linear_weight=0.20,
        lgbm_weight=0.40,
        xgb_weight=0.30,
        baseline_prior_weight=0.10,
        random_state=42
    )
    return blender.fit_predict(
        cohort_spec, train_records, eval_records,
        x_train=x_train, y_train=y_train, x_eval=x_eval
    )


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Allocate interventions by maximizing expected net benefit efficiency (ROI)."""
    state = CareAllocationState(cohort_spec, eval_records)
    
    candidates = []
    for p in eval_records:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        for tier, intervention in cohort_spec.interventions.items():
            if tier != InterventionTier.NONE and state.fits(p.patient_id, tier):
                enb = state.expected_net_benefit(p.patient_id, risk, tier)
                if enb > 0.0:
                    try:
                        cost = float(intervention.unit_cost)
                    except AttributeError:
                        # Approximate cost if unit_cost is not directly accessible
                        cost = risk * intervention.relative_risk_reduction * 16800.0 - enb
                    if cost <= 0.0:
                        cost = 1.0
                    efficiency = enb / cost
                    candidates.append((efficiency, enb, p.patient_id, tier))

    # Sort by efficiency (ROI) descending, then by absolute net benefit descending
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    assigned = set()
    for efficiency, enb, pid, tier in candidates:
        if pid not in assigned:
            if state.fits(pid, tier):
                state.assign(pid, tier)
                assigned.add(pid)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
