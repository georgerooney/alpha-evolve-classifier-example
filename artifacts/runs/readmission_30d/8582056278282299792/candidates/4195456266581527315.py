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
    """Stage 1: Clinical feature builder with syndrome interactions and RiskBlender."""
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)

    blender = RiskBlender(
        linear_weight=0.40,
        lgbm_weight=0.35,
        xgb_weight=0.25,
        baseline_prior_weight=0.08,
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
    """Stage 2: Advanced greedy allocation maximizing simulated expected net benefit."""
    state = CareAllocationState(cohort_spec, eval_records)
    
    valid_opts = []
    for p in eval_records:
        pid = p.patient_id
        risk = float(predicted_risks.get(pid, p.baseline_api_prob))
        for tier in cohort_spec.interventions:
            if tier == InterventionTier.NONE:
                continue
            if state.fits(pid, tier):
                enb = state.expected_net_benefit(pid, risk, tier)
                if enb > 0.0:
                    cost = -state.expected_net_benefit(pid, 0.0, tier)
                    valid_opts.append((pid, tier, enb, cost))
                    
    best_plan = []
    max_total_enb = -1.0
    
    for w in [0.0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]:
        current_budget = state.remaining_budget()
        current_slots = {tier: state.remaining_slots(tier) for tier in cohort_spec.interventions}
        
        scored_opts = []
        for pid, tier, enb, cost in valid_opts:
            score = enb / (cost ** w) if cost > 0 else float('inf')
            scored_opts.append((score, enb, cost, pid, tier))
            
        scored_opts.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        plan = []
        assigned_pids = set()
        total_enb = 0.0
        
        for score, enb, cost, pid, tier in scored_opts:
            if pid not in assigned_pids:
                if current_budget >= cost and current_slots.get(tier, 0) > 0:
                    plan.append((pid, tier))
                    assigned_pids.add(pid)
                    current_budget -= cost
                    current_slots[tier] -= 1
                    total_enb += enb
                    
        if total_enb > max_total_enb:
            max_total_enb = total_enb
            best_plan = plan
            
    for pid, tier in best_plan:
        state.assign(pid, tier)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
