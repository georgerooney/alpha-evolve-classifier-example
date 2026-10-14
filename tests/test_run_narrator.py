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

"""Behavioral tests for `scripts/run_narrator.py` (Gemini 3.8 Flash analysis and candidate summaries).

The LLM is injected as a plain `generate(model, prompt, schema) -> str | dict` callable, so these tests exercise the
real file handling, caching, prompt grounding, failure clustering, and multi-run history without network access.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from scripts import run_narrator

PROBLEM_ID = "diabetes130_readmit30"
SEED = (
    "from clinical_toolkit import build_diabetes130_frame\n"
    "# EVOLVE-BLOCK-START\n"
    "def fit_and_score_risk(task, train, y_train, eval_cols):\n"
    "    return [0.2] * task.n_eval\n"
    "# EVOLVE-BLOCK-END\n"
)


def _cand(cid: str, gen: int, score: float | None, status: str, insights: list[str]) -> dict:
    return {
        "candidate_id": cid,
        "generation": gen,
        "score": score,
        "status": status,
        "rank": gen + 1,
        "what_changed": f"change {cid}",
        "roc_auc": round(score / 10000.0, 4) if score and score > 0 else 0.5,
        "pr_auc": 0.24,
        "brier": 0.096,
        "source_file": f"candidates/{cid}.py",
        "insights": insights,
    }


def _runs(tmp_path: pathlib.Path) -> pathlib.Path:
    """Builds `artifacts/runs/diabetes130_readmit30/555` (seed + 2 graded + 1 crash) and an archived run."""
    runs_dir = tmp_path / "runs" / PROBLEM_ID
    run = runs_dir / "555"
    (run / "candidates").mkdir(parents=True)
    rows = [
        _cand("s0", 0, 6645.99, "GRADED", ["[diabetes130_readmit30] Score=6,645.99 | ROC-AUC=0.6646 ± 0.0030 over 3 rounds"]),
        _cand("c1", 1, 6769.69, "GRADED", ["[diabetes130_readmit30] Score=6,769.69 | ROC-AUC=0.6770 ± 0.0028 over 3 rounds"]),
        _cand("c2", 2, 6610.12, "GRADED", ["[diabetes130_readmit30] Score=6,610.12 | ROC-AUC=0.6610 ± 0.0035 over 3 rounds"]),
        _cand(
            "c3",
            3,
            -1e12,
            "RUNTIME_ERROR",
            [
                "RUNTIME ERRORS (1 crashed call). Runtime error in fit_and_score_risk (ValueError: "
                "Categorical feature at index 9 is expected to have a cardinality <= 255) | "
                '  File "<candidate_program>", line 4, in fit_and_score_risk'
            ],
        ),
    ]
    for i, r in enumerate(rows):
        (run / r["source_file"]).write_text(SEED.replace("0.2", f"0.{i + 3}") if i else SEED, encoding="utf-8")
    (run / "candidates.json").write_text(json.dumps(rows), encoding="utf-8")
    (run / "rank1.py").write_text(SEED.replace("0.2", "0.4"), encoding="utf-8")
    (run / "report.json").write_text(
        json.dumps(
            {
                "problem_id": PROBLEM_ID,
                "git_commit": "17f70b1",
                "baseline": {"fitness": 6645.99, "heldout": 6808.36, "canonical": 6655.67},
                "top": [
                    {
                        "rank": 1,
                        "program_name": "experiments/555/alphaEvolvePrograms/c1",
                        "scores": {"fitness": 6769.69, "heldout": 6903.37, "canonical": 6743.68},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    old = runs_dir / "_archive" / "pre_tuned_seed" / "444"
    old.mkdir(parents=True)
    (old / "candidates.json").write_text(json.dumps([_cand("o0", 0, 6334.81, "GRADED", [])]), encoding="utf-8")
    (old / "report.json").write_text(
        json.dumps(
            {
                "problem_id": PROBLEM_ID,
                "git_commit": "afe9fa3",
                "baseline": {"fitness": 6334.81, "heldout": 6518.18},
                "top": [{"rank": 1, "scores": {"fitness": 6772.37, "heldout": -995000.0}}],
            }
        ),
        encoding="utf-8",
    )
    return run


def test_candidate_summaries_are_written_cached_and_fail_soft(tmp_path: pathlib.Path) -> None:
    run = _runs(tmp_path)
    calls: list[tuple[str, str]] = []

    def fake(model: str, prompt: str, schema=None) -> str:
        calls.append((model, prompt))
        if "0.5" in prompt:  # c2: simulate an API failure
            raise RuntimeError("quota")
        return "  Blends out-of-fold patient interaction features into a regularized GBDT ensemble.  "

    n = run_narrator.summarize_candidates(run, PROBLEM_ID, SEED, generate=fake)
    cands = {c["candidate_id"]: c for c in json.loads((run / "candidates.json").read_text())}
    assert n == 2  # c1 (graded) and c3 (crashed); s0 is seed and c2's call failed
    assert cands["c1"]["summary"] == "Blends out-of-fold patient interaction features into a regularized GBDT ensemble."
    assert cands["c1"]["summary_model"] == run_narrator.SUMMARY_MODEL == "gemini-3.5-flash-lite"
    assert "summary" not in cands["c2"]  # failure leaves no fabricated text
    assert cands["s0"]["summary"] == run_narrator.SEED_SUMMARY
    assert all(m == "gemini-3.5-flash-lite" for m, _ in calls)
    assert any("cardinality <= 255" in p for _, p in calls)  # crashed candidates are explained from their error lines

    retried: list[str] = []
    run_narrator.summarize_candidates(
        run, PROBLEM_ID, SEED, generate=lambda m, p, s=None: retried.append(p) or "retried summary"
    )
    assert len(retried) == 1  # only the previously failed candidate is retried
    cands = {c["candidate_id"]: c for c in json.loads((run / "candidates.json").read_text())}
    assert cands["c1"]["summary"] == "Blends out-of-fold patient interaction features into a regularized GBDT ensemble."
    assert cands["c2"]["summary"] == "retried summary"


def test_run_analysis_is_structured_grounded_and_covers_run_history(tmp_path: pathlib.Path, monkeypatch) -> None:
    run = _runs(tmp_path)
    monkeypatch.setenv("AE_PERTURB_SEED", "918273")
    monkeypatch.setenv("AE_HELDOUT_SEED", "564738")
    seen: dict = {}

    def fake(model: str, prompt: str, schema=None) -> dict:
        seen.update(model=model, prompt=prompt, schema=schema)
        return {
            "headline": "h",
            "what_changed": "w",
            "why_it_works": "y",
            "what_didnt_work": "d",
            "risks": "r",
            "next_steps": ["a", "b", "c"],
        }

    out = run_narrator.explain_run(run, PROBLEM_ID, SEED, generate=fake)
    saved = json.loads((run / "analysis.json").read_text())
    assert saved == out and out["next_steps"] == ["a", "b", "c"]
    assert out["model"] == seen["model"] == run_narrator.ANALYSIS_MODEL == "gemini-3.8-flash"
    assert {"headline", "what_changed", "why_it_works", "what_didnt_work", "risks", "next_steps"} <= set(
        seen["schema"]["properties"]
    )
    p = seen["prompt"]
    assert "0.4" in p  # diff from the run's seed to rank1.py
    assert "6645.99" in p and "6903.37" in p  # seed baseline and finalist held-out
    assert "cardinality <= 255" in p  # failure modes
    assert "444" in p and "pre_tuned_seed" in p  # archived run history included
    assert "918273" not in p and "564738" not in p  # private evaluator seeds never reach the LLM


def test_narrator_prompts_are_grounded_against_known_hallucinations(tmp_path: pathlib.Path) -> None:
    run = _runs(tmp_path)
    long_seed = SEED.replace("return [0.2]", "x = 1\n" * 4000 + "    return [0.2]")
    (run / "candidates" / "s0.py").write_text(long_seed, encoding="utf-8")
    prompts: dict[str, str] = {}

    def fake(model: str, prompt: str, schema=None):
        if schema:
            prompts["analysis"] = prompt
            return {k: ["a", "b", "c"] if k == "next_steps" else "x" for k in run_narrator._SECTIONS}
        cid = next(c for c in ("c1", "c2", "c3") if f"0.{int(c[1]) + 3}" in prompt)
        prompts[cid] = prompt
        return "ok"

    run_narrator.summarize_candidates(run, PROBLEM_ID, SEED, generate=fake)
    run_narrator.explain_run(run, PROBLEM_ID, SEED, generate=fake)
    assert "truncated" not in prompts["c1"].lower()
    assert "ran successfully" in prompts["c1"] and "better than the seed" in prompts["c1"]
    assert "ran successfully" in prompts["c2"] and "worse than the seed" in prompts["c2"]
    assert "do not describe it as failed" in prompts["c2"].lower()
    assert "Offending source line 4: return [0.6] * task.n_eval" in prompts["c3"]
    a = prompts["analysis"]
    assert "Best fitness among active runs: 6769.69 (run 555 rank 1)" in a
    assert "6772.37 (run 444" not in a  # archived runs are excluded from active-run best claims


def test_failure_modes_are_clustered_with_counts_and_normalized_numbers(tmp_path: pathlib.Path) -> None:
    run = _runs(tmp_path)
    cat = json.loads((run / "candidates.json").read_text())
    for i in range(3):
        cat.append(
            _cand(
                f"t{i}",
                10 + i,
                -1e12,
                "RUNTIME_ERROR",
                ["TimeoutError: Timed out after 60 seconds during benchmark evaluation."],
            )
        )
    for i, brier in enumerate(("0.294", "0.312")):
        cat.append(
            _cand(
                f"v{i}",
                20 + i,
                -995000.0,
                "HARD_VIOLATION",
                [f"HARD CONSTRAINT VIOLATION: Calibration Brier score {brier} exceeds hard ceiling 0.250."],
            )
        )
    (run / "candidates.json").write_text(json.dumps(cat), encoding="utf-8")
    seen: dict[str, str] = {}

    def fake(model: str, prompt: str, schema=None):
        seen["p"] = prompt
        return {k: ["a", "b", "c"] if k == "next_steps" else "x" for k in run_narrator._SECTIONS}

    run_narrator.explain_run(run, PROBLEM_ID, SEED, generate=fake)
    p = seen["p"]
    assert "3 × TimeoutError" in p
    assert "1 × ValueError" in p
    assert "2 × Calibration Brier score # exceeds hard ceiling #" in p


def test_refresh_resummarises_previous_gemini_summaries(tmp_path: pathlib.Path) -> None:
    run = _runs(tmp_path)
    run_narrator.summarize_candidates(run, PROBLEM_ID, SEED, generate=lambda m, p, s=None: "first")
    n = run_narrator.summarize_candidates(run, PROBLEM_ID, SEED, generate=lambda m, p, s=None: "second", refresh=True)
    cands = {c["candidate_id"]: c for c in json.loads((run / "candidates.json").read_text())}
    assert n == 3 and cands["c1"]["summary"] == "second"
    assert cands["s0"]["summary"] == run_narrator.SEED_SUMMARY


def test_run_analysis_rejects_malformed_llm_output(tmp_path: pathlib.Path) -> None:
    run = _runs(tmp_path)
    with pytest.raises(ValueError):
        run_narrator.explain_run(run, PROBLEM_ID, SEED, generate=lambda m, p, s=None: {"headline": "x"})
    assert not (run / "analysis.json").exists()


def test_narrate_is_best_effort(tmp_path: pathlib.Path) -> None:
    run = _runs(tmp_path)

    def boom(model: str, prompt: str, schema=None):
        raise RuntimeError("API down")

    run_narrator.narrate(run, PROBLEM_ID, SEED, generate=boom)
    assert not (run / "analysis.json").exists()
