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


def _extract_extra_features(records: Sequence[PatientRecord]) -> "np.ndarray":
    import numpy as np

    feats = []
    for p in records:
        base_p = float(getattr(p, "baseline_api_prob", 0.5))
        base_p_clamped = max(1e-4, min(1.0 - 1e-4, base_p))
        b_logit = safe_logit(base_p_clamped)

        sub = getattr(p, "baseline_submodel_scores", {}) or {}
        c_lin = float(sub.get("claims_linear", base_p))
        ehr_v = float(sub.get("ehr_vitals", base_p))
        util = float(sub.get("utilization_rules", base_p))
        acuity = float(getattr(p, "acuity_score", 0.0))
        n_contra = float(len(getattr(p, "contraindications", []) or []))

        # Clinical syndrome divergence features
        acute_minus_chronic = ehr_v - c_lin
        util_minus_claims = util - c_lin
        vitals_minus_base = ehr_v - base_p
        claims_logit = safe_logit(max(1e-4, min(1.0 - 1e-4, c_lin)))
        vitals_logit = safe_logit(max(1e-4, min(1.0 - 1e-4, ehr_v)))

        feats.append([
            base_p,
            b_logit,
            c_lin,
            ehr_v,
            util,
            acuity,
            n_contra,
            acute_minus_chronic,
            util_minus_claims,
            vitals_minus_base,
            claims_logit,
            vitals_logit,
        ])
    arr = np.array(feats, dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=-10.0)


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Multi-model ensemble via RiskBlender augmented with cross-organ submodel interactions."""
    import numpy as np

    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)

    extra_tr = _extract_extra_features(train_records)
    extra_ev = _extract_extra_features(eval_records)
    x_aug_train = np.hstack([x_train, extra_tr])
    x_aug_eval = np.hstack([x_eval, extra_ev])

    blender = RiskBlender(
        linear_weight=0.35,
        lgbm_weight=0.40,
        xgb_weight=0.25,
        baseline_prior_weight=0.06,
        random_state=42,
    )
    try:
        preds = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_aug_train,
            y_train=y_train,
            x_eval=x_aug_eval,
        )
    except Exception:
        preds = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
        )

    # Strictly monotonic prevalence calibration to optimize Brier score while preserving rank discrimination
    y_mean = max(1e-4, min(1.0 - 1e-4, float(np.mean(y_train))))
    pred_probs = [float(preds.get(p.patient_id, p.baseline_api_prob)) for p in eval_records]
    pred_mean = max(1e-4, min(1.0 - 1e-4, float(np.mean(pred_probs))))
    shift = safe_logit(y_mean) - safe_logit(pred_mean)

    out: Dict[str, float] = {}
    for p in eval_records:
        raw_p = float(preds.get(p.patient_id, p.baseline_api_prob))
        raw_p_clamped = max(1e-4, min(1.0 - 1e-4, raw_p))
        calibrated_logit = safe_logit(raw_p_clamped) + 0.5 * shift
        calibrated_p = safe_sigmoid(calibrated_logit)
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, float(calibrated_p))), 6)

    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Optimal net-benefit allocation respecting all clinical safety and capacity constraints."""
    state = CareAllocationState(cohort_spec, eval_records)
    active_tiers = [
        t for t in cohort_spec.interventions.keys() if t != InterventionTier.NONE
    ]

    # Prioritize allocation starting with highest predicted readmission risk
    ranked_patients = sorted(
        eval_records,
        key=lambda p: -float(predicted_risks.get(p.patient_id, p.baseline_api_prob)),
    )

    for p in ranked_patients:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        best_tier = None
        best_benefit = 0.0
        for tier in active_tiers:
            if state.fits(p.patient_id, tier):
                enb = state.expected_net_benefit(p.patient_id, risk, tier)
                if enb > best_benefit:
                    best_benefit = enb
                    best_tier = tier
        if best_tier is not None:
            state.assign(p.patient_id, best_tier)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
