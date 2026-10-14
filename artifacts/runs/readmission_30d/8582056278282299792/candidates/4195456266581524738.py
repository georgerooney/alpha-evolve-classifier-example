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
    """Stage 1: Unified logit-prior stacking combining legacy submodels and non-linear clinical syndromes."""
    import numpy as np

    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_tr, y_train = builder.fit_transform(cohort_spec, train_records)
    x_ev = builder.transform(cohort_spec, eval_records)

    def extract_prior_logits(records: Sequence[PatientRecord]) -> np.ndarray:
        priors = []
        for p in records:
            base_p = float(getattr(p, "baseline_api_prob", 0.5))
            sub = getattr(p, "baseline_submodel_scores", {}) or {}
            priors.append([
                safe_logit(base_p),
                safe_logit(float(sub.get("claims_linear", base_p))),
                safe_logit(float(sub.get("ehr_vitals", base_p))),
                safe_logit(float(sub.get("utilization_rules", base_p))),
            ])
        return np.nan_to_num(np.array(priors, dtype=float), nan=0.0, posinf=6.0, neginf=-6.0)

    priors_tr = extract_prior_logits(train_records)
    priors_ev = extract_prior_logits(eval_records)

    x_train_stacked = np.nan_to_num(np.hstack([priors_tr, x_tr]), nan=0.0, posinf=1e4, neginf=-1e4)
    x_eval_stacked = np.nan_to_num(np.hstack([priors_ev, x_ev]), nan=0.0, posinf=1e4, neginf=-1e4)

    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_train_stacked)
    x_ev_scaled = scaler.transform(x_eval_stacked)

    clf = LogisticRegression(C=0.5, max_iter=400, random_state=42)
    clf.fit(x_tr_scaled, y_train)
    raw_probs = clf.predict_proba(x_ev_scaled)[:, 1]

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        base_p = float(getattr(p, "baseline_api_prob", 0.5))
        # 90% weight to optimal unified stacking model with 10% Bayesian anchor to production baseline
        blended = 0.90 * safe_logit(float(raw_probs[idx])) + 0.10 * safe_logit(base_p)
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, safe_sigmoid(blended))), 6)

    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: First-fit highest-risk patients into the highest-efficacy feasible intervention tier."""
    state = CareAllocationState(cohort_spec, eval_records)
    ranked_patients = sorted(
        eval_records,
        key=lambda p: -float(predicted_risks.get(p.patient_id, p.baseline_api_prob)),
    )
    tiers_by_rrr = [
        t
        for t, s in sorted(
            cohort_spec.interventions.items(),
            key=lambda kv: -kv[1].relative_risk_reduction,
        )
        if t != InterventionTier.NONE
    ]

    for p in ranked_patients:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        for tier in tiers_by_rrr:
            if state.fits(p.patient_id, tier) and state.expected_net_benefit(p.patient_id, risk, tier) > 0.0:
                state.assign(p.patient_id, tier)
                break

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
