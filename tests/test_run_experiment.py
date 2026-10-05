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

"""Behavioral tests for run_experiment.py and setup_alphaevolve.py."""

from __future__ import annotations

import pathlib
import pytest

import run_experiment
import setup_alphaevolve


def test_require_private_seed_rejects_missing_or_public_defaults() -> None:
    with pytest.raises(SystemExit):
        run_experiment.require_private_seed({})

    with pytest.raises(SystemExit):
        run_experiment.require_private_seed(
            {"AE_PERTURB_SEED": "2027", "AE_HELDOUT_SEED": "999999"}
        )

    # Valid private seeds succeed
    run_experiment.require_private_seed(
        {"AE_PERTURB_SEED": "847291", "AE_HELDOUT_SEED": "391847"}
    )


def test_resolve_problem_dir_validates_existence(tmp_path: pathlib.Path) -> None:
    pdir = run_experiment.resolve_problem_dir("readmission_30d")
    assert (pdir / "initial_program.py").exists()
    assert (pdir / "problem_description.md").exists()

    with pytest.raises(SystemExit):
        run_experiment.resolve_problem_dir("nonexistent_model_xyz")


def test_setup_alphaevolve_config_from_env() -> None:
    cfg = setup_alphaevolve.Config.from_env(
        {
            "PROJECT_ID": "alphaevolve-poc-123",
            "GE_APP_ID": "alphaevolve-classifier",
        }
    )
    assert cfg.project_id == "alphaevolve-poc-123"
    assert cfg.engine_id == "alphaevolve-classifier"
    assert cfg.location == "us"
    assert cfg.base_url == "us-discoveryengine.googleapis.com"
    assert "locations/us/" in cfg.engine
    assert not hasattr(cfg, "impersonate")

    # Should not double-prefix if BASE_URL already starts with us-
    cfg_explicit = setup_alphaevolve.Config.from_env(
        {
            "PROJECT_ID": "alphaevolve-poc-123",
            "GE_APP_ID": "alphaevolve-classifier",
            "LOCATION": "us",
            "BASE_URL": "us-discoveryengine.googleapis.com",
        }
    )
    assert cfg_explicit.base_url == "us-discoveryengine.googleapis.com"




def test_candidate_diff_and_manifest_generation(tmp_path: pathlib.Path) -> None:
    seed_code = (
        "# EVOLVE-BLOCK-START\n"
        "def fit_and_score_risk(cohort_spec, train_records, eval_records):\n"
        "    return {p.patient_id: p.baseline_api_prob for p in eval_records}\n"
        "# EVOLVE-BLOCK-END\n"
    )
    cand_code = (
        "# EVOLVE-BLOCK-START\n"
        "def fit_and_score_risk(cohort_spec, train_records, eval_records):\n"
        "    # Added LightGBM + clinical interaction blend\n"
        "    from clinical_toolkit import RiskBlender\n"
        "    return RiskBlender().fit_predict(cohort_spec, train_records, eval_records)\n"
        "# EVOLVE-BLOCK-END\n"
    )
    diff_info = run_experiment.summarize_candidate_diff(seed_code, cand_code)
    assert diff_info["added_lines"] >= 2
    assert "RiskBlender" in diff_info["what_changed"] or "LightGBM" in diff_info["what_changed"]
    assert "unified_diff" in diff_info

    fake_programs = [
        {
            "name": "projects/p/locations/global/experiments/exp1/alphaEvolvePrograms/cand_001",
            "content": {"files": [{"content": cand_code}]},
            "evaluation": {
                "scores": {"scores": [{"metric": "clinical_fitness", "score": 8512.4}]},
                "insights": {
                    "insights": [
                        {
                            "label": "summary",
                            "text": "[readmission_30d] Score=8,512.40 | Feasible=4/4 (Violations=0) | ROC-AUC=0.8120 vs Base=0.6306 (+0.1814, Target>0.70:PASS)",
                        },
                        {
                            "label": "detail_1",
                            "text": "Clinical Metrics: PR-AUC=0.8310 | Sens@Spec80=0.6900 | Brier=0.1410 | TargetGate=4/4 scenarios | EvalTime=510ms",
                        },
                        {
                            "label": "detail_2",
                            "text": "Financial Value Impact: Actuarial Lift Value=$7,618,800 | Stage 2 Care Net ROI=$940,000 (45.2 events prevented)",
                        },
                    ]
                },
            },
        }
    ]
    manifest = run_experiment.save_candidate_catalog(
        fake_programs, seed_code=seed_code, out_dir=tmp_path, problem_id="readmission_30d"
    )
    assert len(manifest) == 1
    assert manifest[0]["candidate_id"] == "cand_001"
    assert manifest[0]["roc_auc"] == pytest.approx(0.8120)
    assert (tmp_path / "candidates" / "cand_001.py").is_file()
    assert (tmp_path / "candidates.json").is_file()


def test_build_client_and_experiment_uses_valid_alpha_evolve_sdk_signatures() -> None:
    env = {
        "PROJECT_ID": "alphaevolve-classifier-example",
        "LOCATION": "us",
        "COLLECTION": "default_collection",
        "GE_APP_ID": "alphaevolve-classifier",
        "ASSISTANT": "default_assistant",
        "BASE_URL": "us-discoveryengine.googleapis.com",
        "MAX_PROGRAMS_EVALUATED": "40",
        "PARALLEL_EVALUATION": "False",
    }
    client, experiment = run_experiment.build_client_and_experiment(
        env, problem_id="readmission_30d", inject_creds=False
    )
    assert client.project_id == "alphaevolve-classifier-example"
    assert client.location == "us"
    assert client.base_url == "https://us-discoveryengine.googleapis.com"
    assert client.engine == "alphaevolve-classifier"
    assert client.assistant == "default_assistant"
    assert experiment.max_programs_evaluated == 40



