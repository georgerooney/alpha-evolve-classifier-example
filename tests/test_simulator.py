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

"""Behavioral tests for simulator/portfolio_simulator.py and simulator_ui.html."""

from __future__ import annotations

import json
import pathlib

import pytest

from portfolio_simulator import PortfolioSimulationSession, build_portfolio_snapshot, list_problems, resolve_problem_id


def test_portfolio_simulation_session_evaluates_all_models_and_whatif_budget() -> None:
    session = PortfolioSimulationSession()
    snap = session.evaluate_portfolio()

    assert len(snap["problems"]) == 4
    assert snap["totals"]["total_actuarial_auc_roi_usd"] > 20_000_000.0
    assert snap["totals"]["total_stage2_net_savings_usd"] > 2_000_000.0
    assert snap["totals"]["mean_evolved_auc"] > snap["totals"]["mean_seed_auc"] > snap["totals"]["mean_baseline_auc"]

    # What-if budget scaling must recompute Stage 2 net savings and intervention counts deterministically
    scaled = session.simulate_budget_and_roi_whatif(
        problem_id="sepsis_90d",
        budget_multiplier=1.5,
        roi_per_1pct_multiplier=1.2,
    )
    assert scaled["problem_id"] == "sepsis_90d"
    assert scaled["evolved_roc_auc"] > 0.80
    assert scaled["stage2_net_savings_usd"] > 0.0


def test_simulator_ui_html_exists_and_uses_safe_dom_apis() -> None:
    ui_path = pathlib.Path(__file__).resolve().parent.parent / "simulator" / "simulator_ui.html"
    assert ui_path.is_file()
    html = ui_path.read_text(encoding="utf-8")
    assert "Content-Security-Policy" in html
    assert "innerHTML" not in html
    assert "outerHTML" not in html
    assert "document.write" not in html
    assert "insertAdjacentHTML" not in html


def test_candidate_run_diagnostics_and_source_links() -> None:
    session = PortfolioSimulationSession()
    run_log = session.get_candidate_run_diagnostics("readmission_30d")

    assert len(run_log["candidates"]) >= 45
    top = run_log["candidates"][0]
    assert "candidate_id" in top
    assert "what_changed" in top
    assert "roc_auc" in top and "pr_auc" in top and "brier" in top
    assert "stage2_net_savings_usd" in top
    assert "source_url" in top

    # Verify we have graded, violation, and runtime error buckets represented for diagnostic triage
    statuses = {c["status"] for c in run_log["candidates"]}
    assert "GRADED" in statuses
    assert "HARD_VIOLATION" in statuses or "RUNTIME_ERROR" in statuses

    # Verify fetching source code and diff for a candidate works
    src_payload = session.get_candidate_source("readmission_30d", top["candidate_id"])
    assert "fit_and_score_risk" in src_payload["code"]
    assert "allocate_interventions" in src_payload["code"]
    assert "--- initial_program.py" in src_payload["unified_diff"]


def test_split_cloud_aws_snowflake_docs_and_svg_exist() -> None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    svg_path = repo_root / "docs" / "aws_snowflake_split_cloud_architecture.svg"
    md_path = repo_root / "docs" / "AWS_SNOWFLAKE_MIGRATION.md"
    assert svg_path.is_file()
    assert md_path.is_file()
    assert "<svg" in svg_path.read_text(encoding="utf-8")
    md_text = md_path.read_text(encoding="utf-8")
    assert "Snowflake" in md_text and "Workload Identity Federation" in md_text


# --- Real-data (tabular) track -------------------------------------------------------------------------------------
REAL = "diabetes130_readmit30"
EXP = "projects/p/locations/global/collections/c/engines/e/sessions/s/alphaEvolveExperiments/123"


def _prog(name: str, code: str, score: float | None, insights: list[str], t: str) -> dict:
    prog = {"name": f"{EXP}/alphaEvolvePrograms/{name}", "createTime": t, "content": {"files": [{"content": code}]}}
    if score is not None:
        prog["evaluation"] = {
            "scores": {"scores": [{"score": score}]},
            "insights": {"insights": [{"text": s} for s in insights]},
        }
    return prog


def _live_session(tmp_path: pathlib.Path, calls: list) -> PortfolioSimulationSession:
    seed = (pathlib.Path(__file__).resolve().parent.parent / "problems" / REAL / "initial_program.py").read_text()
    run_dir = tmp_path / "runs" / REAL / "123"
    run_dir.mkdir(parents=True)
    (run_dir / "experiment.json").write_text(json.dumps({"experiment_name": EXP, "max_programs": 80}))
    programs = [
        _prog("p0", seed, 62.9, [f"[{REAL}] Score=62.9 | ROC-AUC=0.6290 ± 0.0040 over 3 rounds",
                                 "Metrics: PR-AUC=0.2100 | Brier=0.0950 | EvalTime=13000ms",
                                 "Subgroup AUC [race]: worst=a:0.610 best=b:0.650 gap=0.040",
                                 "Subgroup AUC [age]: worst=c:0.600 best=d:0.660 gap=0.060"], "2026-10-02T11:54:00Z"),
        _prog("p2", seed + "\n# crash\n", -1e12, ["RUNTIME ERRORS: boom"], "2026-10-02T11:58:00Z"),
        _prog("p1", seed + "\n# tweak\n", 64.1,
              [f"[{REAL}] Score=64.1 | ROC-AUC=0.6410 ± 0.0030 over 3 rounds",
               "Metrics: PR-AUC=0.2300 | Brier=0.0940 | EvalTime=15000ms"], "2026-10-02T11:56:00Z"),
        _prog("p3", seed, None, [], "2026-10-02T11:59:00Z"),  # generated, not yet evaluated
    ]

    def lister(name: str) -> list:
        calls.append(name)
        return programs

    return PortfolioSimulationSession(artifacts_root=tmp_path, program_lister=lister)


def test_problem_ids_cover_both_tracks_and_reject_traversal() -> None:
    ids = {p["problem_id"]: p["track"] for p in list_problems()}
    assert ids[REAL] == "real_data" and ids["readmission_30d"] == "synthetic"
    assert "_template" not in ids
    for bad in ("../etc", "nope", "", "_template"):
        with pytest.raises(ValueError):
            resolve_problem_id(bad)


def test_real_data_problem_never_gets_fabricated_trace(tmp_path: pathlib.Path) -> None:
    diag = PortfolioSimulationSession(artifacts_root=tmp_path).get_candidate_run_diagnostics(REAL)
    assert diag["source"] == "none"
    assert diag["candidates"] == []


def test_live_run_polls_api_and_summarises_progress(tmp_path: pathlib.Path) -> None:
    calls: list = []
    session = _live_session(tmp_path, calls)
    live = session.get_live_run(REAL)

    assert live["experiment_name"] == EXP
    assert (live["programs_listed"], live["programs_evaluated"], live["max_programs"]) == (4, 3, 80)
    assert live["seed_roc_auc"] == 0.629 and live["best_roc_auc"] == 0.641
    # best-so-far trace is in creation order and monotone; crashes do not reset it
    assert [p["best_roc_auc"] for p in live["trace"]] == [0.629, 0.641, 0.641]

    diag = session.get_candidate_run_diagnostics(REAL)
    assert diag["source"] == "live_alphaevolve_api"
    by_id = {c["candidate_id"]: c for c in diag["candidates"]}
    assert by_id["p2"]["status"] == "RUNTIME_ERROR"
    assert by_id["p3"]["status"] == "PENDING"
    assert by_id["p0"]["subgroup_gap"] == 0.06
    assert diag["candidates"][0]["candidate_id"] == "p1"
    assert len(calls) == 1, "API results must be cached between live/diagnostics calls"

    src = session.get_candidate_source(REAL, "p1")
    assert "fit_and_score_risk" in src["code"] and "+# tweak" in src["unified_diff"]


def test_real_data_scorecard_reads_locked_test_report(tmp_path: pathlib.Path) -> None:
    rep_dir = tmp_path / "reports"
    rep_dir.mkdir()
    arms = [
        {"arm": "tuned_lightgbm", "test_auc": 0.679, "vs_tuned_delta": 0.0, "vs_tuned_ci_low": 0.0, "vs_tuned_ci_high": 0.0},
        {"arm": "seed", "test_auc": 0.652, "vs_tuned_delta": -0.028, "vs_tuned_ci_low": -0.034, "vs_tuned_ci_high": -0.02},
    ]
    (rep_dir / f"{REAL}_20261001_000000.json").write_text(json.dumps({"arms": [], "locked_test_n": 1}))
    (rep_dir / f"{REAL}_20261002_000000.json").write_text(json.dumps({"arms": arms, "locked_test_n": 20000}))

    card = _live_session(tmp_path, []).real_data_scorecard(REAL)
    assert card["locked_test"]["locked_test_n"] == 20000  # latest report wins
    assert card["tuned_test_auc"] == 0.679
    assert card["live"]["best_roc_auc"] == 0.641


def test_ui_has_no_offline_fabrication_and_polls_live_run() -> None:
    html = (pathlib.Path(__file__).resolve().parent.parent / "simulator" / "simulator_ui.html").read_text()
    assert "buildFallbackCandidates" not in html
    assert "/api/live_run" in html and "/api/problems" in html

