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
    """Stage 1: Enhanced feature engineering with interactions and blended ensemble."""
    import numpy as np
    
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)

    def get_extra_features(records):
        extra = []
        for p in records:
            b_api = float(p.baseline_api_prob)
            b_api = max(1e-6, min(1.0 - 1e-6, b_api))
            extra.append([
                b_api,
                safe_logit(b_api),
                float(p.baseline_submodel_scores.get("claims_linear", 0.0)),
                float(p.baseline_submodel_scores.get("ehr_vitals", 0.0)),
                float(p.baseline_submodel_scores.get("utilization_rules", 0.0)),
                float(p.acuity_score),
                float(len(p.contraindications))
            ])
        return np.array(extra)

    x_tr_ext = np.hstack([x_train, get_extra_features(train_records)])
    x_ev_ext = np.hstack([x_eval, get_extra_features(eval_records)])

    blender = RiskBlender(
        linear_weight=0.25,
        lgbm_weight=0.40,
        xgb_weight=0.30,
        baseline_prior_weight=0.05,
        random_state=42
    )
    return blender.fit_predict(cohort_spec, train_records, eval_records, x_train=x_tr_ext, y_train=y_train, x_eval=x_ev_ext)


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Optimal Marginal Greedy (MCKP) allocation via convex hull and max-heap."""
    import heapq
    state = CareAllocationState(cohort_spec, eval_records)
    
    budget = cohort_spec.total_budget_usd
    slots = {t: inter.max_slots for t, inter in cohort_spec.interventions.items() if t != InterventionTier.NONE}
    
    patient_fronts = {}
    for p in eval_records:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        opts = [(InterventionTier.NONE, 0.0, 0.0)]
        for tier, inter in cohort_spec.interventions.items():
            if tier == InterventionTier.NONE:
                continue
            if p.acuity_score >= cohort_spec.min_acuity_score and not (set(p.contraindications) & set(cohort_spec.contraindicated_flags)):
                enb = state.expected_net_benefit(p.patient_id, risk, tier)
                if enb > 0:
                    opts.append((tier, enb, inter.unit_cost))
        
        # Build Upper Convex Hull of efficiency
        opts.sort(key=lambda x: (x[2], -x[1]))
        hull = []
        for opt in opts:
            if hull and opt[2] == hull[-1][2]:
                continue
            if not hull:
                hull.append(opt)
                continue
            if opt[1] <= hull[-1][1]:
                continue
            while len(hull) >= 2:
                prev, prev2 = hull[-1], hull[-2]
                dy_opt, dx_opt = opt[1] - prev2[1], opt[2] - prev2[2]
                dy_prev, dx_prev = prev[1] - prev2[1], prev[2] - prev2[2]
                if dy_opt * dx_prev > dy_prev * dx_opt:
                    hull.pop()
                else:
                    break
            hull.append(opt)
        patient_fronts[p.patient_id] = hull

    assignments = {p.patient_id: 0 for p in eval_records}
    heap = []
    
    def push_next_upgrade(pid, start_idx=None):
        idx = assignments[pid]
        front = patient_fronts[pid]
        curr_tier, curr_enb, curr_cost = front[idx]
        
        start_idx = start_idx if start_idx is not None else idx + 1
        for nx in range(start_idx, len(front)):
            t, e, c = front[nx]
            m_enb = e - curr_enb
            m_cost = c - curr_cost
            if m_enb > 0 and m_cost > 0:
                if m_cost <= budget + 1e-5:
                    m_roi = m_enb / m_cost
                    heapq.heappush(heap, (-m_roi, -m_enb, pid, nx, m_cost, t, curr_tier))
                    break

    for p in eval_records:
        push_next_upgrade(p.patient_id)
        
    while heap:
        neg_roi, neg_m_enb, pid, nx, m_cost, t, expected_curr_tier = heapq.heappop(heap)
        
        front = patient_fronts[pid]
        if front[assignments[pid]][0] != expected_curr_tier:
            continue
            
        if budget >= m_cost - 1e-5 and slots.get(t, 0) > 0:
            if expected_curr_tier != InterventionTier.NONE:
                slots[expected_curr_tier] += 1
            slots[t] -= 1
            budget -= m_cost
            assignments[pid] = nx
            push_next_upgrade(pid)
        else:
            push_next_upgrade(pid, start_idx=nx + 1)
            
    final_plan = []
    for pid, idx in assignments.items():
        t = patient_fronts[pid][idx][0]
        if t != InterventionTier.NONE:
            final_plan.append((pid, t))
            
    if final_plan:
        if not state.assign_all(final_plan):
            # Fallback for transient precision tolerance differences
            for pid, t in final_plan:
                if state.fits(pid, t):
                    state.assign(pid, t)
                    
    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
