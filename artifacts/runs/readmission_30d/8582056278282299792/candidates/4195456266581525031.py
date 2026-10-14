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
    """Stage 1: Simplification using built-in RiskBlender."""
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)
    
    blender = RiskBlender(linear_weight=0.40, lgbm_weight=0.35, xgb_weight=0.25, baseline_prior_weight=0.08, random_state=42)
    return blender.fit_predict(cohort_spec, train_records, eval_records, x_train=x_train, y_train=y_train, x_eval=x_eval)


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Simplified greedy allocation globally optimizing expected net benefit."""
    state = CareAllocationState(cohort_spec, eval_records)
    
    options = []
    for p in eval_records:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        for tier in cohort_spec.interventions:
            if tier != InterventionTier.NONE:
                ben = state.expected_net_benefit(p.patient_id, risk, tier)
                if ben > 0:
                    options.append((ben, p.patient_id, tier))
                    
    assigned = set()
    for ben, pid, tier in sorted(options, key=lambda x: x[0], reverse=True):
        if pid not in assigned and state.fits(pid, tier):
            state.assign(pid, tier)
            assigned.add(pid)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
