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

"""Behavioral tests for all 4 clinical problem templates, best_evolved_program.py, and new_problem.py."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import pytest

import evaluator

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROBLEMS = ["readmission_30d", "inpatient_admission", "sepsis_90d", "chf_30d"]


@pytest.mark.parametrize("problem_id", PROBLEMS)
def test_problem_seed_and_best_evolved_lift(problem_id: str) -> None:
    seed_path = REPO_ROOT / "problems" / problem_id / "initial_program.py"
    evolved_path = REPO_ROOT / "artifacts" / "best_evolved_program.py"
    desc_path = REPO_ROOT / "problems" / problem_id / "problem_description.md"
    cfg_path = REPO_ROOT / "problems" / problem_id / "problem_config.json"

    assert seed_path.exists()
    assert evolved_path.exists()
    assert desc_path.exists()
    assert cfg_path.exists()

    seed_code = seed_path.read_text(encoding="utf-8")
    evolved_code = evolved_path.read_text(encoding="utf-8")
    assert "EVOLVE-BLOCK-START" in seed_code
    assert "EVOLVE-BLOCK-END" in seed_code

    seed_res = evaluator.evaluate_in_process(seed_code, problem_id=problem_id, perturb=False)
    evolved_res = evaluator.evaluate_in_process(evolved_code, problem_id=problem_id, perturb=False)

    assert seed_res["score"] is not None and seed_res["score"] > 6500.0
    assert evolved_res["score"] is not None
    # Evolved candidate must strictly beat the seed program on every portfolio problem
    assert evolved_res["score"] > seed_res["score"] + 100.0, (
        f"{problem_id}: evolved={evolved_res['score']} vs seed={seed_res['score']}"
    )


def test_new_problem_scaffolding_cli(tmp_path: pathlib.Path) -> None:
    target_dir = tmp_path / "problems" / " oncology_30d".strip()
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "new_problem.py"),
            "oncology_30d",
            "--title",
            "Oncology 30-Day Acute Toxicity Admission",
            "--target-auc",
            "0.75",
            "--output-dir",
            str(target_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (target_dir / "problem_config.json").exists()
    assert (target_dir / "initial_program.py").exists()
    assert (target_dir / "problem_description.md").exists()


def test_repository_is_generic_and_apache_licensed() -> None:
    import clinical_toolkit  # noqa: F401
    from harness_config import EvaluationWeights

    assert EvaluationWeights().predictive_scale == 10000.0
    assert (REPO_ROOT / "src" / "clinical_toolkit.py").is_file()
    assert (REPO_ROOT / "LICENSE").is_file()
    assert not (REPO_ROOT / "infra").exists()

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for required_doc_term in (
        "diabetes130_readmit30",
        "fetch_diabetes130.sh",
        "tabular_report.py",
        "leakage_check.py",
    ):
        assert required_doc_term in readme

    for rel_file in ("README.md", "example.env", "scripts/provision_gcp.sh", "setup_alphaevolve.py"):
        content = (REPO_ROOT / rel_file).read_text(encoding="utf-8").lower()
        assert "impersonate-service-account" not in content, f"{rel_file} still references SA impersonation"
        assert "client_service_account" not in content, f"{rel_file} still references CLIENT_SERVICE_ACCOUNT"
        assert "cloud_run" not in content and "cloud run" not in content, f"{rel_file} still references Cloud Run"

    lock_path = REPO_ROOT / "uv.lock"
    if lock_path.exists():
        for line in lock_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("source = { registry ="):
                assert "https://pypi.org/simple" in line, f"Non-public PyPI registry in uv.lock: {line}"

    skip_dirs = {".git", ".venv", "__pycache__", "data"}
    unlicensed_py: list[str] = []
    for path in REPO_ROOT.rglob("*.py"):
        if any(part in skip_dirs for part in path.parts) or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "Apache License, Version 2.0" not in text:
            unlicensed_py.append(str(path.relative_to(REPO_ROOT)))

    assert not unlicensed_py, f"Missing Apache-2.0 license header in: {unlicensed_py}"



