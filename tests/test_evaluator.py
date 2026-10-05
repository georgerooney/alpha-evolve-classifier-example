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

"""Behavioral tests for evaluator.py sandbox, label isolation, crash isolation, and CLI contract."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import evaluator


def test_evaluator_blocks_filesystem_and_network_in_candidate() -> None:
    malicious_code = """
from clinical_models import CareAllocationPlan, InterventionTier

def fit_and_score_risk(cohort_spec, train_records, eval_records):
    with open("/tmp/pwned_sandbox.txt", "w") as f:
        f.write("compromised")
    return {p.patient_id: p.baseline_api_prob for p in eval_records}

def allocate_interventions(cohort_spec, eval_records, predicted_risks):
    return CareAllocationPlan(
        scenario_id=cohort_spec.scenario_id,
        assignments={p.patient_id: InterventionTier.NONE for p in eval_records},
    )
"""
    res = evaluator.evaluate_program(malicious_code, problem_id="readmission_30d")
    assert res["score"] is None
    assert any("Sandbox" in s or "PermissionError" in s for s in res["insights"])


def test_evaluator_strips_eval_labels_and_redacts_benchmarks_module() -> None:
    probe_code = """
import clinical_benchmarks
from clinical_models import CareAllocationPlan, InterventionTier

def fit_and_score_risk(cohort_spec, train_records, eval_records):
    # Verify eval_records have no ground truth labels in child process
    for p in eval_records:
        if p.outcome_label is not None or p.time_to_event_days is not None:
            raise RuntimeError("LEAKED EVAL LABEL")
    # Verify clinical_benchmarks is stubbed/redacted
    insts = clinical_benchmarks.build_benchmark_instances("readmission_30d")
    if any(i.scenario_id != "REDACTED" for i in insts):
        raise RuntimeError("UNREDACTED BENCHMARKS")
    return {p.patient_id: p.baseline_api_prob for p in eval_records}

def allocate_interventions(cohort_spec, eval_records, predicted_risks):
    return CareAllocationPlan(
        scenario_id=cohort_spec.scenario_id,
        assignments={p.patient_id: InterventionTier.NONE for p in eval_records},
    )
"""
    res = evaluator.evaluate_program(probe_code, problem_id="readmission_30d")
    assert res["score"] is not None
    assert res["score"] > 5000.0


def test_evaluator_per_call_crash_isolation_grades_remaining_scenarios() -> None:
    partial_crash_code = """
from clinical_models import CareAllocationPlan, InterventionTier

_CALLS = 0

def fit_and_score_risk(cohort_spec, train_records, eval_records):
    global _CALLS
    _CALLS += 1
    if _CALLS == 1:
        raise ValueError("Simulated single-scenario numerical failure")
    return {p.patient_id: p.baseline_api_prob for p in eval_records}

def allocate_interventions(cohort_spec, eval_records, predicted_risks):
    return CareAllocationPlan(
        scenario_id=cohort_spec.scenario_id,
        assignments={p.patient_id: InterventionTier.NONE for p in eval_records},
    )
"""
    res = evaluator.evaluate_program(partial_crash_code, problem_id="readmission_30d")
    assert res["score"] is not None
    # Penalized by 1 hard violation (-1e6 / 4 scenarios = -250,000 offset), not None or -1e12
    assert -500000.0 < res["score"] < 0.0
    assert any("RUNTIME ERRORS" in s or "ValueError" in s for s in res["insights"])


def test_evaluator_cli_contract(tmp_path: pathlib.Path) -> None:
    out_file = tmp_path / "eval_out.json"
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "src" / "evaluator.py"),
            "--program-dir",
            str(repo_root / "problems" / "readmission_30d"),
            "--output-file",
            str(out_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert isinstance(payload["score"], float)
    assert payload["score"] > 6000.0
    assert len(payload["insights"]) >= 2
