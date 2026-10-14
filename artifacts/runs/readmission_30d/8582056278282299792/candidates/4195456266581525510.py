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
from typing import Dict, Sequence, Any, List, Tuple
import numpy as np

try:
    from clinical_toolkit import (
        ClinicalFeatureBuilder,
        CareAllocationState,
        InterventionTier,
        RiskBlender,
    )
except ImportError:
    pass


def fit_and_score_risk(
    cohort_spec: Any,
    train_records: Sequence[Any],
    eval_records: Sequence[Any],
) -> Dict[str, float]:
    """Stage 1: Predict readmission risk using ClinicalFeatureBuilder and RiskBlender."""
    try:
        builder = ClinicalFeatureBuilder(include_interactions=True)
        x_train, y_train = builder.fit_transform(cohort_spec, train_records)
        x_eval = builder.transform(cohort_spec, eval_records)

        x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
        x_eval = np.nan_to_num(x_eval, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.asarray(y_train, dtype=np.int32)

        if len(np.unique(y_train)) < 2:
            return {p.patient_id: float(getattr(p, "baseline_api_prob", 0.5)) for p in eval_records}

        blender = RiskBlender(
            linear_weight=0.40,
            lgbm_weight=0.35,
            xgb_weight=0.25,
            baseline_prior_weight=0.08,
            random_state=42,
        )
        preds = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
        )

        result: Dict[str, float] = {}
        for p in eval_records:
            val = preds.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5))
            if val is None or np.isnan(val) or np.isinf(val):
                val = getattr(p, "baseline_api_prob", 0.5)
            result[p.patient_id] = float(np.clip(val, 0.001, 0.999))
        return result
    except Exception:
        return {
            p.patient_id: float(np.clip(getattr(p, "baseline_api_prob", 0.5), 0.001, 0.999))
            for p in eval_records
        }


def allocate_interventions(
    cohort_spec: Any,
    eval_records: Sequence[Any],
    predicted_risks: Dict[str, float],
) -> Any:
    """Stage 2: Optimal knapsack allocation with alpha-sweep and upgrade hill-climbing."""
    ref_state = CareAllocationState(cohort_spec, eval_records)

    # 1. Deduce tier costs and slot capacities robustly
    candidate_tiers = [
        tier for tier in cohort_spec.interventions.keys()
        if tier != InterventionTier.NONE
    ]
    tier_costs: Dict[Any, float] = {}
    tier_slots: Dict[Any, int] = {}

    for tier in candidate_tiers:
        spec = cohort_spec.interventions[tier]
        tier_slots[tier] = int(spec.max_slots)
        cost = getattr(spec, "cost_usd", getattr(spec, "cost", None))
        if cost is None:
            for p in eval_records:
                if ref_state.fits(p.patient_id, tier):
                    cost = -ref_state.expected_net_benefit(p.patient_id, 0.0, tier)
                    break
        tier_costs[tier] = max(1.0, float(cost if cost is not None else 0.0))

    total_budget = float(cohort_spec.total_budget_usd)

    # 2. Build candidate patient-tier options with clinical eligibility verified
    patient_options: Dict[str, Dict[Any, Tuple[float, float]]] = {}
    all_pairs = []

    for p in eval_records:
        pid = p.patient_id
        risk = float(predicted_risks.get(pid, getattr(p, "baseline_api_prob", 0.5)))
        patient_options[pid] = {}

        for tier in candidate_tiers:
            if ref_state.fits(pid, tier):
                enb = ref_state.expected_net_benefit(pid, risk, tier)
                if enb > 0.0:
                    cost = tier_costs[tier]
                    patient_options[pid][tier] = (enb, cost)
                    all_pairs.append((pid, tier, enb, cost))

    # 3. Multi-pass optimization over efficiency-yield tradeoff exponents alpha in [0.0, 1.0]
    best_assignment: Dict[str, Any] = {}
    best_total_benefit = -1.0

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for alpha in alphas:
        # Rank by marginal benefit weighted by cost^alpha
        ranked_pairs = sorted(
            all_pairs,
            key=lambda x: (x[2] / (x[3] ** alpha + 1e-6), x[2]),
            reverse=True,
        )

        curr_assigned: Dict[str, Any] = {}
        slots_used = {t: 0 for t in candidate_tiers}
        budget_used = 0.0

        for pid, tier, enb, cost in ranked_pairs:
            if pid not in curr_assigned:
                if (slots_used[tier] + 1 <= tier_slots[tier]) and (budget_used + cost <= total_budget):
                    curr_assigned[pid] = tier
                    slots_used[tier] += 1
                    budget_used += cost

        # Upgrade Hill-Climbing: upgrade assigned patients to higher-benefit tiers if budget and slots permit
        for pid in list(curr_assigned.keys()):
            curr_tier = curr_assigned[pid]
            curr_enb, curr_cost = patient_options[pid][curr_tier]

            upgrade_candidates = [
                (t, patient_options[pid][t][0], patient_options[pid][t][1])
                for t in candidate_tiers
                if t in patient_options[pid] and t != curr_tier and patient_options[pid][t][0] > curr_enb
            ]
            upgrade_candidates.sort(key=lambda x: (x[1] - curr_enb) / max(1.0, x[2] - curr_cost), reverse=True)

            for cand_tier, cand_enb, cand_cost in upgrade_candidates:
                delta_cost = cand_cost - curr_cost
                if (slots_used[cand_tier] + 1 <= tier_slots[cand_tier]) and (budget_used + delta_cost <= total_budget):
                    slots_used[curr_tier] -= 1
                    slots_used[cand_tier] += 1
                    budget_used += delta_cost
                    curr_assigned[pid] = cand_tier
                    curr_tier = cand_tier
                    curr_enb, curr_cost = cand_enb, cand_cost
                    break

        total_enb = sum(patient_options[pid][tier][0] for pid, tier in curr_assigned.items())
        if total_enb > best_total_benefit:
            best_total_benefit = total_enb
            best_assignment = curr_assigned

    # 4. Commit optimal plan into CareAllocationState to guarantee 0 violations
    final_state = CareAllocationState(cohort_spec, eval_records)
    sorted_assignments = sorted(
        best_assignment.items(),
        key=lambda item: patient_options[item[0]][item[1]][0],
        reverse=True,
    )

    for pid, tier in sorted_assignments:
        if final_state.fits(pid, tier):
            final_state.assign(pid, tier)

    plan = final_state.to_plan()
    for p in eval_records:
        if p.patient_id not in plan.assignments:
            plan.assignments[p.patient_id] = InterventionTier.NONE

    return plan


# EVOLVE-BLOCK-END
# =====================================================================
