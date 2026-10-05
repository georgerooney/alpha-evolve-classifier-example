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

"""Behavioral unit tests for clinical_metrics.py and clinical_models.py."""

from __future__ import annotations

import math
import pytest

from clinical_models import (
    CareAllocationPlan,
    CohortSpec,
    InterventionSpec,
    InterventionTier,
    PatientRecord,
)
from clinical_metrics import (
    compute_brier_score,
    compute_pr_auc,
    compute_roc_auc,
    compute_sensitivity_at_specificity,
    evaluate_cohort_predictions_and_allocation,
)


def _sample_cohort() -> tuple[CohortSpec, list[PatientRecord]]:
    interventions = {
        InterventionTier.NONE: InterventionSpec(
            tier=InterventionTier.NONE,
            cost_usd=0.0,
            relative_risk_reduction=0.0,
            max_slots=100,
        ),
        InterventionTier.PHARMACY_MED_RECON: InterventionSpec(
            tier=InterventionTier.PHARMACY_MED_RECON,
            cost_usd=250.0,
            relative_risk_reduction=0.15,
            max_slots=2,
            min_acuity_score=2.0,
        ),
        InterventionTier.INTENSIVE_RN_CARE_MGMT: InterventionSpec(
            tier=InterventionTier.INTENSIVE_RN_CARE_MGMT,
            cost_usd=1200.0,
            relative_risk_reduction=0.35,
            max_slots=1,
            min_acuity_score=5.0,
            contraindicated_flags=("PALLIATIVE_HOSPICE",),
        ),
    }
    spec = CohortSpec(
        problem_id="readmission_30d",
        scenario_id="READM-TEST-01",
        scenario_name="Unit Test Readmission Scenario",
        horizon_days=30,
        target_auc=0.70,
        target_specificity=0.75,
        min_sensitivity_floor=0.50,
        event_cost_usd=15000.0,
        hrrp_penalty_multiplier=1.2,
        roi_per_1pct_auc_usd=400000.0,
        total_budget_usd=1600.0,
        interventions=interventions,
        feature_names=("egfr", "hba1c"),
    )
    patients = [
        PatientRecord(
            patient_id="P1",
            age=74.0,
            sex=1,
            charlson_index=5.0,
            prior_ed_visits_6m=3,
            prior_ip_admissions_12m=2,
            length_of_stay_days=6.0,
            sdoh_deprivation_index=0.8,
            acuity_score=7.5,
            features={"egfr": 32.0, "hba1c": 9.1},
            baseline_api_prob=0.55,
            baseline_submodel_scores={"claims_linear": 0.50, "ehr_vitals": 0.60},
            contraindications=(),
            outcome_label=1,
            time_to_event_days=9.0,
        ),
        PatientRecord(
            patient_id="P2",
            age=68.0,
            sex=0,
            charlson_index=4.0,
            prior_ed_visits_6m=2,
            prior_ip_admissions_12m=1,
            length_of_stay_days=4.0,
            sdoh_deprivation_index=0.6,
            acuity_score=4.5,
            features={"egfr": 48.0, "hba1c": 7.8},
            baseline_api_prob=0.60,  # baseline misranks P2 above P1
            baseline_submodel_scores={"claims_linear": 0.62, "ehr_vitals": 0.58},
            contraindications=("PALLIATIVE_HOSPICE",),
            outcome_label=1,
            time_to_event_days=18.0,
        ),
        PatientRecord(
            patient_id="P3",
            age=52.0,
            sex=1,
            charlson_index=1.0,
            prior_ed_visits_6m=0,
            prior_ip_admissions_12m=0,
            length_of_stay_days=2.0,
            sdoh_deprivation_index=0.2,
            acuity_score=2.5,
            features={"egfr": 85.0, "hba1c": 5.6},
            baseline_api_prob=0.65,  # false positive in baseline
            baseline_submodel_scores={"claims_linear": 0.65, "ehr_vitals": 0.65},
            contraindications=(),
            outcome_label=0,
            time_to_event_days=None,
        ),
        PatientRecord(
            patient_id="P4",
            age=45.0,
            sex=0,
            charlson_index=0.0,
            prior_ed_visits_6m=0,
            prior_ip_admissions_12m=0,
            length_of_stay_days=1.0,
            sdoh_deprivation_index=0.1,
            acuity_score=1.0,
            features={"egfr": 95.0, "hba1c": 5.2},
            baseline_api_prob=0.15,
            baseline_submodel_scores={"claims_linear": 0.15, "ehr_vitals": 0.15},
            contraindications=(),
            outcome_label=0,
            time_to_event_days=None,
        ),
    ]
    return spec, patients


def test_roc_and_pr_auc_exact_values() -> None:
    y_true = [1, 1, 0, 0]
    y_perfect = [0.95, 0.80, 0.20, 0.05]
    assert compute_roc_auc(y_true, y_perfect) == pytest.approx(1.0)
    assert compute_pr_auc(y_true, y_perfect) == pytest.approx(1.0)
    assert compute_sensitivity_at_specificity(y_true, y_perfect, 0.75) == pytest.approx(1.0)

    y_inverted = [0.10, 0.20, 0.80, 0.90]
    assert compute_roc_auc(y_true, y_inverted) == pytest.approx(0.0)
    assert compute_brier_score(y_true, y_perfect) < compute_brier_score(y_true, y_inverted)


def test_evaluate_cohort_rewards_predictive_uplift_and_valid_allocation() -> None:
    spec, patients = _sample_cohort()
    good_risks = {"P1": 0.92, "P2": 0.81, "P3": 0.12, "P4": 0.04}
    plan = CareAllocationPlan(
        scenario_id=spec.scenario_id,
        assignments={
            "P1": InterventionTier.INTENSIVE_RN_CARE_MGMT,
            "P2": InterventionTier.PHARMACY_MED_RECON,
            "P3": InterventionTier.NONE,
            "P4": InterventionTier.NONE,
        },
    )
    res = evaluate_cohort_predictions_and_allocation(spec, patients, good_risks, plan)
    assert res["is_feasible"] is True
    assert res["hard_violation_count"] == 0
    assert res["roc_auc"] == pytest.approx(1.0)
    assert res["baseline_roc_auc"] <= 0.5
    assert res["auc_uplift"] >= 0.5
    assert res["net_savings_usd"] > 0.0
    assert res["total_score"] > 9000.0


def test_evaluate_cohort_penalizes_budget_slot_and_contraindication_violations() -> None:
    spec, patients = _sample_cohort()
    good_risks = {"P1": 0.90, "P2": 0.80, "P3": 0.15, "P4": 0.05}
    # Violates slot cap on INTENSIVE_RN (2 > 1), budget ($2400 > $1600), and contraindication on P2
    bad_plan = CareAllocationPlan(
        scenario_id=spec.scenario_id,
        assignments={
            "P1": InterventionTier.INTENSIVE_RN_CARE_MGMT,
            "P2": InterventionTier.INTENSIVE_RN_CARE_MGMT,
            "P3": InterventionTier.NONE,
            "P4": InterventionTier.PHARMACY_MED_RECON,  # P4 acuity 1.0 < min_acuity 2.0
        },
    )
    res = evaluate_cohort_predictions_and_allocation(spec, patients, good_risks, bad_plan)
    assert res["is_feasible"] is False
    assert res["hard_violation_count"] >= 3
    assert res["total_score"] < -1e6


def test_evaluate_cohort_penalizes_invalid_probabilities() -> None:
    spec, patients = _sample_cohort()
    bad_risks = {"P1": 1.3, "P2": -0.1, "P3": 0.2}  # missing P4 + out-of-range P1/P2
    plan = CareAllocationPlan(
        scenario_id=spec.scenario_id,
        assignments={p.patient_id: InterventionTier.NONE for p in patients},
    )
    res = evaluate_cohort_predictions_and_allocation(spec, patients, bad_risks, plan)
    assert res["is_feasible"] is False
    assert res["hard_violation_count"] >= 2
    assert math.isfinite(res["total_score"])
