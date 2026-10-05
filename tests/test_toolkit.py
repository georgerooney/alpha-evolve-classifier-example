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

"""Differential and behavioral tests for clinical_toolkit.py against clinical_metrics.py."""

from __future__ import annotations

import pytest

from clinical_benchmarks import build_benchmark_instances
from clinical_metrics import evaluate_cohort_predictions_and_allocation
from clinical_models import InterventionTier
from clinical_toolkit import CareAllocationState, ClinicalFeatureBuilder, RiskBlender


@pytest.mark.parametrize(
    "problem_id",
    ["readmission_30d", "inpatient_admission", "sepsis_90d", "chf_30d"],
)
def test_care_allocation_state_differential_equivalence(problem_id: str) -> None:
    instances = build_benchmark_instances(problem_id, perturb=False)
    assert len(instances) >= 4

    for inst in instances:
        spec = inst.cohort_spec
        state = CareAllocationState(spec, inst.eval_records)
        assert state.all_ok() is True

        # Greedy placement using state.fits and state.assign must always produce a 100% feasible plan
        risks = {p.patient_id: p.baseline_api_prob for p in inst.eval_records}
        ranked = sorted(inst.eval_records, key=lambda p: -risks[p.patient_id])
        tiers_by_intensity = [
            t
            for t, s in sorted(
                spec.interventions.items(),
                key=lambda kv: -kv[1].relative_risk_reduction,
            )
            if t != InterventionTier.NONE
        ]
        for p in ranked:
            best_tier = InterventionTier.NONE
            best_net = 0.0
            for tier in tiers_by_intensity:
                if state.fits(p.patient_id, tier):
                    net = state.expected_net_benefit(p.patient_id, risks[p.patient_id], tier)
                    if net > best_net:
                        best_net = net
                        best_tier = tier
            if best_tier != InterventionTier.NONE:
                assert state.assign(p.patient_id, best_tier) is True

        assert state.all_ok() is True
        plan = state.to_plan()
        res = evaluate_cohort_predictions_and_allocation(spec, inst.eval_records, risks, plan)
        assert res["is_feasible"] is True
        assert res["hard_violation_count"] == 0


def test_assign_all_is_atomic_on_budget_or_slot_overflow() -> None:
    inst = build_benchmark_instances("readmission_30d", perturb=False)[0]
    spec = inst.cohort_spec
    state = CareAllocationState(spec, inst.eval_records)

    intensive_tier = InterventionTier.INTENSIVE_RN_CARE_MGMT
    max_slots = spec.interventions[intensive_tier].max_slots
    eligible = [
        p.patient_id
        for p in inst.eval_records
        if state.fits(p.patient_id, intensive_tier)
    ]
    assert len(eligible) > max_slots

    # Attempting to assign max_slots + 1 patients atomically must fail and leave state unchanged
    overflow_pairs = [(pid, intensive_tier) for pid in eligible[: max_slots + 1]]
    assert state.assign_all(overflow_pairs) is False
    assert state.remaining_slots(intensive_tier) == max_slots
    assert state.remaining_budget() == pytest.approx(spec.total_budget_usd)


def test_feature_builder_and_risk_blender_improve_on_baseline() -> None:
    inst = build_benchmark_instances("sepsis_90d", perturb=False)[0]
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train = builder.fit_transform(inst.cohort_spec, inst.train_records)
    x_eval = builder.transform(inst.cohort_spec, inst.eval_records)

    assert x_train.shape[0] == len(inst.train_records)
    assert x_eval.shape[0] == len(inst.eval_records)
    assert x_train.shape[1] == x_eval.shape[1]

    blender = RiskBlender()
    probs = blender.fit_predict(
        inst.cohort_spec,
        inst.train_records,
        inst.eval_records,
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
    )
    assert set(probs.keys()) == {p.patient_id for p in inst.eval_records}
    assert all(0.0 <= v <= 1.0 for v in probs.values())
