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

"""Reference evolved multi-cohort candidate for the AlphaEvolve Clinical Portfolio.

Demonstrates:
1. Stage 1 (`fit_and_score_risk`): Domain-specific clinical trajectory synthesis (cardiorenal shock ratio,
   hemodynamic lactate clearance index, SDoH adherence velocity, and legacy submodel disagreement dispersion),
   time-to-event urgency sample weighting (`compute_urgency_sample_weights`), 4-fold Stratified Out-Of-Fold (OOF)
   stacking across L2 Logistic + LightGBM + XGBoost + HistGradientBoosting (`RiskBlender`), and empirical Bayes
   prevalence calibration.
2. Stage 2 (`allocate_interventions`): Marginal value-density (`expected_net_benefit / cost_usd**0.65`) allocation
   followed by residual budget upgrade refinement in `CareAllocationState`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np

from clinical_models import (
    CareAllocationPlan,
    CohortSpec,
    InterventionTier,
    PatientRecord,
)
from clinical_toolkit import (
    CareAllocationState,
    ClinicalFeatureBuilder,
    RiskBlender,
    compute_urgency_sample_weights,
    safe_logit,
    safe_sigmoid,
)


# =====================================================================
# EVOLVE-BLOCK-START
def _augment_clinical_trajectory_features(
    records: Sequence[PatientRecord],
    base_matrix: np.ndarray,
) -> np.ndarray:
    """Synthesizes higher-order longitudinal and cross-organ risk features."""
    extra = np.zeros((len(records), 4), dtype=np.float64)
    for i, p in enumerate(records):
        f = p.features
        egfr = max(10.0, float(f.get("egfr", 70.0)))
        bnp = max(20.0, float(f.get("nt_probnp", 400.0)))
        shock = float(f.get("shock_index", 0.70))
        lac_clear = float(f.get("lactate_clearance_6h", 0.35))
        ed_v = float(p.prior_ed_visits_6m + 2 * p.prior_ip_admissions_12m)
        boarding = float(f.get("ed_boarding_hours", 2.0))
        adherence = float(f.get("med_adherence_pdc", 0.80))

        # 1. Cardiorenal log-ratio pressure
        extra[i, 0] = np.log1p(bnp) / np.sqrt(egfr)
        # 2. Hemodynamic clearance deficit
        extra[i, 1] = shock * max(0.0, 0.50 - lac_clear)
        # 3. Acute utilization x boarding velocity
        extra[i, 2] = np.log1p(ed_v) * np.log1p(boarding)
        # 4. SDoH x non-adherence vulnerability
        extra[i, 3] = p.sdoh_deprivation_index * (1.0 - adherence) * (1.0 + 0.2 * p.charlson_index)
    return np.hstack([base_matrix, extra])


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Fit 4-fold OOF stacked GBDT + linear ensemble with urgency weighting and calibrate log-odds."""
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)

    x_train = _augment_clinical_trajectory_features(train_records, x_train_raw)
    x_eval = _augment_clinical_trajectory_features(eval_records, x_eval_raw)
    sample_weights = compute_urgency_sample_weights(cohort_spec, train_records, early_event_boost=0.32)

    pid = cohort_spec.problem_id.lower()
    if "sepsis" in pid:
        blender = RiskBlender(
            linear_weight=0.38,
            lgbm_weight=0.30,
            xgb_weight=0.22,
            hgb_weight=0.10,
            baseline_prior_weight=0.03,
            use_oof_meta_learner=True,
        )
    elif "chf" in pid:
        blender = RiskBlender(
            linear_weight=0.35,
            lgbm_weight=0.32,
            xgb_weight=0.23,
            hgb_weight=0.10,
            baseline_prior_weight=0.04,
            use_oof_meta_learner=True,
        )
    else:
        blender = RiskBlender(
            linear_weight=0.36,
            lgbm_weight=0.30,
            xgb_weight=0.24,
            hgb_weight=0.10,
            baseline_prior_weight=0.05,
            use_oof_meta_learner=True,
        )

    raw_probs = blender.fit_predict(
        cohort_spec,
        train_records,
        eval_records,
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        sample_weight=sample_weights,
    )

    # Empirical prevalence calibration anchoring so mean predicted risk matches training cohort incidence
    train_prev = max(
        1e-3,
        min(
            1.0 - 1e-3,
            sum(int(p.outcome_label or 0) for p in train_records) / max(1, len(train_records)),
        ),
    )
    eval_mean = max(1e-3, min(1.0 - 1e-3, sum(raw_probs.values()) / max(1, len(raw_probs))))
    shift_logit = 0.35 * (safe_logit(train_prev) - safe_logit(eval_mean))

    calibrated: Dict[str, float] = {}
    for p in eval_records:
        base_l = safe_logit(raw_probs[p.patient_id])
        calibrated[p.patient_id] = round(
            max(1e-4, min(1.0 - 1e-4, safe_sigmoid(base_l + shift_logit))), 6
        )
    return calibrated


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Maximize expected net dollar savings via value-density greedy + residual budget upgrade pass."""
    state = CareAllocationState(cohort_spec, eval_records)
    active_tiers: List[InterventionTier] = [
        t for t in cohort_spec.interventions if t != InterventionTier.NONE
    ]

    candidates = []
    for p in eval_records:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        for tier in active_tiers:
            if not state.is_clinically_eligible(p.patient_id, tier):
                continue
            net = state.expected_net_benefit(p.patient_id, risk, tier)
            if net > 0.0:
                cost = max(1.0, cohort_spec.interventions[tier].cost_usd)
                efficiency = (net / (cost ** 0.65)) * (1.0 + 0.25 * risk)
                candidates.append((efficiency, net, p.patient_id, tier))

    candidates.sort(key=lambda x: (-x[0], -x[1]))
    for _, net, pid, tier in candidates:
        cur = state.assignments[pid]
        if cur == InterventionTier.NONE:
            state.assign(pid, tier)
        else:
            cur_net = state.expected_net_benefit(pid, float(predicted_risks.get(pid, 0.0)), cur)
            if net > cur_net and state.fits(pid, tier):
                state.assign(pid, tier)

    # Second pass: upgrade already-assigned high-risk members if residual budget and higher-tier slots remain
    for _, net, pid, tier in sorted(candidates, key=lambda x: -x[1]):
        cur = state.assignments[pid]
        cur_net = state.expected_net_benefit(pid, float(predicted_risks.get(pid, 0.0)), cur)
        if net > cur_net and state.fits(pid, tier):
            state.assign(pid, tier)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
