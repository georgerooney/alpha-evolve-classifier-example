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
    """Stage 1: Clinical feature builder and advanced ensemble blending via RiskBlender."""
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
        cohort_spec,
        train_records,
        eval_records,
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval
    )


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Optimal multiple-choice knapsack allocation via marginal ROI greedy."""
    state = CareAllocationState(cohort_spec, eval_records)
    
    valid_options = {}
    for p in eval_records:
        pid = p.patient_id
        risk = float(predicted_risks.get(pid, p.baseline_api_prob))
        opts = []
        for tier, t_info in cohort_spec.interventions.items():
            if tier == InterventionTier.NONE:
                continue
            if state.fits(pid, tier):
                enb = state.expected_net_benefit(pid, risk, tier)
                if enb > 0.0:
                    opts.append((tier, int(round(t_info.unit_cost * 100)), enb))
        
        if not opts:
            continue
            
        opts.sort(key=lambda x: (x[1], x[2]))
        
        hull = [(InterventionTier.NONE, 0, 0.0)]
        for opt in opts:
            if opt[1] <= 0:
                opt = (opt[0], 1, opt[2])
                
            while len(hull) > 1 and hull[-1][1] >= opt[1]:
                hull.pop()
                
            if hull[-1][1] >= opt[1]:
                continue
                
            while len(hull) >= 2:
                prev = hull[-1]
                pprev = hull[-2]
                roi1 = (prev[2] - pprev[2]) / float(prev[1] - pprev[1])
                roi2 = (opt[2] - prev[2]) / float(opt[1] - prev[1])
                if roi2 >= roi1:
                    hull.pop()
                else:
                    break
            hull.append(opt)
            
        if len(hull) > 1:
            valid_options[pid] = hull

    upgrades = []
    for pid, hull in valid_options.items():
        for k in range(len(hull) - 1):
            old_tier, old_cost, old_enb = hull[k]
            new_tier, new_cost, new_enb = hull[k+1]
            m_roi = (new_enb - old_enb) / float(new_cost - old_cost)
            upgrades.append({
                'm_roi': m_roi,
                'pid': pid,
                'old_tier': old_tier,
                'new_tier': new_tier,
                'cost_diff': new_cost - old_cost,
                'k': k
            })
            
    upgrades.sort(key=lambda x: x['m_roi'], reverse=True)
    
    budget_cents = int(round(cohort_spec.total_budget_usd * 100))
    slots = {t: cohort_spec.interventions[t].max_slots for t in cohort_spec.interventions if t != InterventionTier.NONE}
    
    final_assigns = {}
    current_k = {pid: 0 for pid in valid_options}
    
    for up in upgrades:
        pid = up['pid']
        if current_k[pid] != up['k']:
            continue
            
        new_tier = up['new_tier']
        old_tier = up['old_tier']
        cost_diff = up['cost_diff']
        
        if budget_cents >= cost_diff and slots[new_tier] > 0:
            budget_cents -= cost_diff
            slots[new_tier] -= 1
            if old_tier != InterventionTier.NONE:
                slots[old_tier] += 1
                
            final_assigns[pid] = new_tier
            current_k[pid] += 1

    if final_assigns:
        state.assign_all(list(final_assigns.items()))
        
    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
