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
    """Stage 1: Use RiskBlender with baseline scores and acuity augmented to features."""
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)

    def get_extra_features(records):
        feats = []
        for p in records:
            feats.append([
                float(p.baseline_api_prob),
                float(p.baseline_submodel_scores.get("claims_linear", 0.0)),
                float(p.baseline_submodel_scores.get("ehr_vitals", 0.0)),
                float(p.baseline_submodel_scores.get("utilization_rules", 0.0)),
                float(getattr(p, 'acuity_score', 0.0))
            ])
        return np.array(feats, dtype=float)

    x_train_aug = np.hstack([x_train, get_extra_features(train_records)])
    x_eval_aug = np.hstack([x_eval, get_extra_features(eval_records)])

    scaler = StandardScaler()
    x_train_aug = scaler.fit_transform(x_train_aug)
    x_eval_aug = scaler.transform(x_eval_aug)

    blender = RiskBlender(
        linear_weight=0.30,
        lgbm_weight=0.35,
        xgb_weight=0.30,
        baseline_prior_weight=0.05,
        random_state=42
    )
    
    return blender.fit_predict(
        cohort_spec,
        train_records,
        eval_records,
        x_train=x_train_aug,
        y_train=y_train,
        x_eval=x_eval_aug
    )


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Optimize assignments using expected net benefit and ROI knapsack heuristics."""
    state = CareAllocationState(cohort_spec, eval_records)
    
    options = []
    for p in eval_records:
        pid = p.patient_id
        risk = float(predicted_risks.get(pid, p.baseline_api_prob))
        for tier, spec_tier in cohort_spec.interventions.items():
            if tier == InterventionTier.NONE:
                continue
            if state.fits(pid, tier):
                benefit = state.expected_net_benefit(pid, risk, tier)
                if benefit > 0.0:
                    cost = spec_tier.unit_cost
                    # Sort primarily by return on investment, then absolute benefit
                    roi = benefit / cost if cost > 0 else float('inf')
                    options.append((roi, benefit, pid, tier))
                    
    # Sort descending by ROI, then by Benefit
    options.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    assigned = set()
    for roi, benefit, pid, tier in options:
        if pid in assigned:
            continue
        if state.fits(pid, tier):
            state.assign(pid, tier)
            assigned.add(pid)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
