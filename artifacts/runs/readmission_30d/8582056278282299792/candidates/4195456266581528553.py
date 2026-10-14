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
    """Stage 1: Clinical feature builder with syndrome interactions and calibrated logit blend."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)

    # Crazy idea: Let the frozen RiskBlender API do the heavy lifting of ensembling LGBM, XGBoost, and Linear models,
    # but hyper-charge it by injecting the baseline API probabilities directly into the feature matrix!
    # This forces the tree models to learn the non-linear residuals and SDoH interactions against the legacy model.
    def extract_legacy(records):
        return np.array([
            [float(p.baseline_api_prob),
             float(p.acuity_score),
             float(p.baseline_submodel_scores.get('claims_linear', 0.0)),
             float(p.baseline_submodel_scores.get('ehr_vitals', 0.0)),
             float(p.baseline_submodel_scores.get('utilization_rules', 0.0))]
            for p in records
        ])
        
    x_tr_aug = np.hstack([x_train, extract_legacy(train_records)])
    x_ev_aug = np.hstack([x_eval, extract_legacy(eval_records)])
    
    scaler = StandardScaler()
    x_tr_aug = scaler.fit_transform(x_tr_aug)
    x_ev_aug = scaler.transform(x_ev_aug)
    
    # Use frozen RiskBlender to train XGBoost, LightGBM, and Linear Models simultaneously
    blender = RiskBlender(linear_weight=0.15, lgbm_weight=0.40, xgb_weight=0.40, baseline_prior_weight=0.05, random_state=42)
    return blender.fit_predict(cohort_spec, train_records, eval_records, x_train=x_tr_aug, y_train=y_train, x_eval=x_ev_aug)


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Greedy allocation by risk ranking and intervention tier efficacy."""
    # Crazy Idea: Multi-Choice Knapsack Greedy using Marginal ROI (Density)!
    # We deduce the hidden cost by checking the net benefit at 0% risk (which is just -cost).
    state = CareAllocationState(cohort_spec, eval_records)
    
    options = []
    for p in eval_records:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        for tier in cohort_spec.interventions:
            if tier != InterventionTier.NONE and state.fits(p.patient_id, tier):
                enb = state.expected_net_benefit(p.patient_id, risk, tier)
                if enb > 0:
                    # Deduce cost by probing expected net benefit at zero risk
                    cost = -state.expected_net_benefit(p.patient_id, 0.0, tier)
                    density = enb / cost if cost > 0 else float('inf')
                    options.append((density, enb, p.patient_id, tier))
                    
    # Sort globally by highest ROI (density), breaking ties with absolute net benefit
    options.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    assigned_patients = set()
    for density, enb, pid, tier in options:
        if pid not in assigned_patients and state.fits(pid, tier):
            if state.assign(pid, tier):
                assigned_patients.add(pid)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
