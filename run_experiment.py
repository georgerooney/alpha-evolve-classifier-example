#!/usr/bin/env python3
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

"""Run a clinical portfolio evolution on AlphaEvolve (Gemini Enterprise).

Supports modular multi-problem execution (`--problem <problem_id>`) as well as local portfolio evaluation
(`--portfolio-report`) comparing the baseline seed programs and `artifacts/best_evolved_program.py` across all
clinical problems. Saves all 40-200 evaluated candidate programs (`candidates/<id>.py` + `candidates.json`)
with unified diffs and parsed clinical/financial output measures for the diagnostic UI.

    uv run python run_experiment.py --problem readmission_30d
    uv run python run_experiment.py --problem sepsis_90d --resume=<experiment>
    uv run python run_experiment.py --portfolio-report
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import os
import pathlib
import re
import sys
from typing import Any, Mapping, Sequence

HERE = pathlib.Path(__file__).resolve().parent
PROBLEMS_DIR = HERE / "problems"
sys.path.insert(0, str(HERE / "src"))

import evaluator  # noqa: E402
from clinical_benchmarks import DEFAULT_PROBLEM_CATALOG, load_problem_config  # noqa: E402
from setup_alphaevolve import load_dotenv  # noqa: E402

METRIC = "clinical_fitness"
FAILURE_SCORE = -1e12
PUBLIC_SEEDS = {"AE_PERTURB_SEED": "2027", "AE_HELDOUT_SEED": "2026"}


def require_private_seed(env: Mapping[str, str]) -> None:
    """Both evolution and held-out seeds must be private and non-default."""
    for name, public in PUBLIC_SEEDS.items():
        seed = env.get(name)
        if not seed:
            raise SystemExit(f"{name} is not set; copy it from Secret Manager into .env (see README).")
        if seed == public:
            raise SystemExit(f"{name} is the public default; candidates could fingerprint scenarios.")


def resolve_problem_dir(problem_id: str) -> pathlib.Path:
    """Validates and resolves `problems/<problem_id>`."""
    clean_id = pathlib.Path(problem_id).name
    pdir = (PROBLEMS_DIR / clean_id).resolve()
    if not pdir.is_dir() or not (pdir / "initial_program.py").is_file():
        raise SystemExit(f"Problem directory '{pdir}' not found or missing initial_program.py.")
    return pdir


def _extract_evolve_block(code: str) -> str:
    m = re.search(r"# EVOLVE-BLOCK-START(.*?)# EVOLVE-BLOCK-END", code, flags=re.DOTALL)
    return m.group(1).strip() if m else code.strip()


def summarize_candidate_diff(seed_code: str, candidate_code: str) -> dict[str, Any]:
    """Computes unified diff, line delta stats, and a human-readable summary of what changed vs. the seed."""
    seed_lines = seed_code.splitlines()
    cand_lines = candidate_code.splitlines()
    udiff = "\n".join(
        difflib.unified_diff(
            seed_lines,
            cand_lines,
            fromfile="initial_program.py",
            tofile="candidate.py",
            lineterm="",
        )
    )
    added = [l[1:].strip() for l in udiff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:].strip() for l in udiff.splitlines() if l.startswith("-") and not l.startswith("---")]
    eblock_lines = len(_extract_evolve_block(candidate_code).splitlines())

    # Extract meaningful highlights from added comments and key algorithmic primitives
    highlights: list[str] = []
    for line in added:
        if line.startswith("#") and "EVOLVE-BLOCK" not in line and len(line) > 4:
            highlights.append(line.lstrip("# ").strip())
        elif any(
            kw in line
            for kw in (
                "RiskBlender",
                "LGBMClassifier",
                "XGBClassifier",
                "HistGradientBoosting",
                "StratifiedKFold",
                "compute_urgency_sample_weights",
                "include_interactions=True",
                "expected_net_benefit",
                "assign_all",
            )
        ):
            highlights.append(line[:72])

    if not highlights:
        if not added and not removed:
            summary = "Unmodified baseline seed program"
        else:
            summary = f"Modified EVOLVE-BLOCK (+{len(added)} / -{len(removed)} lines)"
    else:
        summary = "; ".join(dict.fromkeys(highlights[:2]))

    return {
        "added_lines": len(added),
        "removed_lines": len(removed),
        "evolve_block_lines": eblock_lines,
        "what_changed": summary,
        "unified_diff": udiff or "--- initial_program.py\n+++ candidate.py\n(no changes)",
    }


def parse_insights_metrics(insights_texts: Sequence[str], score: float | None) -> dict[str, Any]:
    """Extracts structured clinical and financial measures from evaluator insights strings."""
    joined = "\n".join(insights_texts)
    roc_m = re.search(r"ROC-AUC=([0-9.]+)", joined)
    base_m = re.search(r"vs Base=([0-9.]+)", joined)
    pr_m = re.search(r"PR-AUC=([0-9.]+)", joined)
    sens_m = re.search(r"Sens@Spec80=([0-9.]+)", joined)
    brier_m = re.search(r"Brier=([0-9.]+)", joined)
    act_m = re.search(r"Actuarial Lift Value=\$([0-9,]+)", joined)
    roi_m = re.search(r"Stage 2 Care Net ROI=\$([-0-9,]+)", joined)
    viol_m = re.search(r"Violations=([0-9]+)", joined)
    ms_m = re.search(r"EvalTime=([0-9.]+)ms", joined)
    # Tabular track: worst subgroup AUC gap across the configured fairness columns.
    gaps = [float(g) for g in re.findall(r"Subgroup AUC \[[^\]]*\]:.*?gap=([0-9.]+)", joined)]

    violations = int(viol_m.group(1)) if viol_m else 0
    has_runtime_err = "RUNTIME ERRORS" in joined or "Candidate initialization failed" in joined or "Sandbox violation" in joined
    if score is None or score <= FAILURE_SCORE / 10 or has_runtime_err:
        status = "RUNTIME_ERROR" if has_runtime_err else "HARD_VIOLATION"
    elif violations > 0 or score < 0:
        status = "HARD_VIOLATION"
    else:
        status = "GRADED"

    return {
        "status": status,
        "violations": violations,
        "roc_auc": float(roc_m.group(1)) if roc_m else 0.0,
        "baseline_roc_auc": float(base_m.group(1)) if base_m else 0.0,
        "pr_auc": float(pr_m.group(1)) if pr_m else 0.0,
        "sens_at_spec80": float(sens_m.group(1)) if sens_m else 0.0,
        "brier": float(brier_m.group(1)) if brier_m else 1.0,
        "actuarial_auc_roi_usd": float(act_m.group(1).replace(",", "")) if act_m else 0.0,
        "stage2_net_savings_usd": float(roi_m.group(1).replace(",", "")) if roi_m else 0.0,
        "eval_ms": float(ms_m.group(1)) if ms_m else 0.0,
        "subgroup_gap": max(gaps) if gaps else None,
    }


def candidate_entry(
    prog: Mapping[str, Any], idx: int, *, seed_code: str, problem_id: str
) -> tuple[dict[str, Any], str]:
    """Parses one AlphaEvolve program resource into a catalog row plus its source code.

    Shared by `save_candidate_catalog` (end of run) and the simulator's live mode (mid-run) so both views agree.
    """
    raw_name = str(prog.get("name") or f"cand_{idx:03d}")
    cand_id = re.sub(r"[^A-Za-z0-9_.-]", "_", raw_name.rsplit("/", 1)[-1])
    files = prog.get("content", {}).get("files", [])
    code = str(files[0].get("content", "")) if files else ""

    scores_list = prog.get("evaluation", {}).get("scores", {}).get("scores", [])
    score = float(scores_list[0].get("score", FAILURE_SCORE)) if scores_list else None
    raw_insights = prog.get("evaluation", {}).get("insights", {}).get("insights", [])
    insight_lines = [str(item.get("text", "")) for item in raw_insights if item.get("text")]

    diff_meta = summarize_candidate_diff(seed_code, code)
    parsed = parse_insights_metrics(insight_lines, score)
    if "evaluation" not in prog:  # generated but not yet evaluated (live view mid-run)
        parsed["status"] = "PENDING"
    entry = {
        "candidate_id": cand_id,
        "problem_id": problem_id,
        "generation": idx,
        "score": score,
        **parsed,
        "added_lines": diff_meta["added_lines"],
        "removed_lines": diff_meta["removed_lines"],
        "evolve_block_lines": diff_meta["evolve_block_lines"],
        "what_changed": diff_meta["what_changed"],
        "source_file": f"candidates/{cand_id}.py",
        "source_url": f"/api/candidate_source?problem_id={problem_id}&candidate_id={cand_id}",
        "insights": insight_lines,
    }
    return entry, code


def rank_catalog(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    catalog.sort(key=lambda c: float(c["score"] if c["score"] is not None else FAILURE_SCORE), reverse=True)
    for rank, item in enumerate(catalog, start=1):
        item["rank"] = rank
    return catalog


def save_candidate_catalog(
    programs: Sequence[Mapping[str, Any]],
    *,
    seed_code: str,
    out_dir: pathlib.Path,
    problem_id: str,
) -> list[dict[str, Any]]:
    """Writes every candidate's `.py` file and `candidates.json` manifest for diagnostic inspection."""
    cand_dir = out_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    catalog: list[dict[str, Any]] = []
    for idx, prog in enumerate(programs):
        entry, code = candidate_entry(prog, idx, seed_code=seed_code, problem_id=problem_id)
        (out_dir / entry["source_file"]).write_text(code, encoding="utf-8")
        catalog.append(entry)

    rank_catalog(catalog)
    (out_dir / "candidates.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return catalog


def _payload(score: float | None, insights: list[str]) -> dict[str, Any]:
    ok = score is not None and math.isfinite(score)
    return {
        "scores": {"scores": [{"metric": METRIC, "score": float(score) if ok else FAILURE_SCORE}]},
        "insights": {
            "insights": [
                {"label": "summary" if i == 0 else f"detail_{i}", "text": t}
                for i, t in enumerate(insights)
                if t
            ]
            or [{"label": "summary", "text": "no diagnostics returned"}]
        },
    }


def make_evaluation_callback(problem_id: str):
    """Builds the problem-specific AlphaEvolve evaluation callback."""

    def _eval(program_candidate: Mapping[str, Any]) -> dict[str, Any]:
        code = program_candidate["content"]["files"][0]["content"]
        try:
            result = evaluator.evaluate_program(code, problem_id=problem_id, split="all", perturb=True)
        except Exception as e:
            logging.exception("evaluator crashed")
            return _payload(None, [f"Evaluator error: {type(e).__name__}: {e}"])
        return _payload(result.get("score"), [str(s) for s in result.get("insights", [])])

    return _eval


def _models(env: Mapping[str, str]) -> list[dict[str, Any]]:
    weights: dict[str, float] = {}
    for i in (1, 2):
        if name := env.get(f"MODEL_{i}"):
            weights[name] = weights.get(name, 0.0) + float(env.get(f"MODEL_{i}_WEIGHT", "1"))
    if not weights and env.get("MODEL"):
        weights[env["MODEL"]] = 1.0
    return [{"name": n, "weight": round(w, 2)} for n, w in weights.items()]


def _rescore(code: str, problem_id: str, heldout_seed: int | None) -> dict[str, float | None]:
    return {
        "fitness": evaluator.evaluate_program(code, problem_id=problem_id, split="all", perturb=True).get("score"),
        "heldout": evaluator.evaluate_program(
            code, problem_id=problem_id, split="test", heldout_seed=heldout_seed, perturb=True
        ).get("score"),
        "canonical": evaluator.evaluate_program(code, problem_id=problem_id, split="all", perturb=False).get("score"),
    }


def _inject_scoped_credentials(client: Any, project_id: str) -> None:
    import google.auth
    import google.auth.transport.requests

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    if hasattr(creds, "with_quota_project"):
        creds = creds.with_quota_project(project_id)
    client._credentials, client._auth_request = creds, google.auth.transport.requests.Request()


def stop_when_server_done(experiment: Any, client: Any, poll_s: float = 30.0) -> None:
    import time

    local_done = experiment.stopping_criteria_met
    last = [float("-inf")]

    def criteria() -> bool:
        if local_done():
            return True
        if time.monotonic() - last[0] < poll_s:
            return False
        last[0] = time.monotonic()
        try:
            state = (client.get_alpha_evolve_experiment(experiment.experiment_name) or {}).get("state")
        except Exception:
            logging.warning("experiment state poll failed", exc_info=True)
            return False
        if state == "COMPLETED":
            logging.info("server experiment COMPLETED (%s evaluated locally)", experiment.stats["num_programs_evaluated"])
            return True
        return False

    experiment.stopping_criteria_met = criteria


def list_all_programs(experiment: Any, page_size: int = 100) -> list[dict[str, Any]]:
    programs, token = [], None
    while True:
        page = experiment.list_programs({"pageSize": page_size, **({"pageToken": token} if token else {})}) or {}
        programs += page.get("alphaEvolvePrograms", [])
        token = page.get("nextPageToken")
        if not token:
            return programs


def git_head(cwd: pathlib.Path = HERE) -> str | None:
    import subprocess

    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else None


def run_portfolio_report() -> dict[str, Any]:
    """Evaluates seed vs. reference evolved program across all portfolio problems."""
    evolved_code = (HERE / "artifacts" / "best_evolved_program.py").read_text(encoding="utf-8")
    summary: dict[str, Any] = {"git_commit": git_head(), "problems": {}}
    for pid in DEFAULT_PROBLEM_CATALOG:
        pdir = resolve_problem_dir(pid)
        seed_code = (pdir / "initial_program.py").read_text(encoding="utf-8")
        seed_eval = evaluator.evaluate_in_process(seed_code, problem_id=pid, perturb=False)
        evol_eval = evaluator.evaluate_in_process(evolved_code, problem_id=pid, perturb=False)
        summary["problems"][pid] = {
            "seed_score": seed_eval["score"],
            "evolved_score": evol_eval["score"],
            "score_uplift": float(evol_eval["score"] or 0.0) - float(seed_eval["score"] or 0.0),
            "evolved_insights": evol_eval["insights"][:3],
        }
    print(json.dumps(summary, indent=2))
    return summary


def build_client_and_experiment(
    env: Mapping[str, str],
    problem_id: str,
    *,
    inject_creds: bool = True,
) -> tuple[Any, Any]:
    """Constructs the `AlphaEvolveClient` and `AlphaEvolveExperiment` using the upstream SDK signatures."""
    from alpha_evolve.client import AlphaEvolveClient
    from alpha_evolve.experiment import AlphaEvolveExperiment

    project_id = env["PROJECT_ID"]
    parallel = env.get("PARALLEL_EVALUATION", "False").lower() == "true"
    location = (env.get("LOCATION") or "us").lower()
    raw_base = (env.get("BASE_URL") or "discoveryengine.googleapis.com").removeprefix("https://")
    for prefix in ("us-", "eu-"):
        raw_base = raw_base.removeprefix(prefix)
    client = AlphaEvolveClient(
        project_id=project_id,
        location=location,
        collection=env.get("COLLECTION", "default_collection"),
        engine=env["GE_APP_ID"],
        assistant=env.get("ASSISTANT", "default_assistant"),
        base_url=raw_base,
    )
    if inject_creds:
        _inject_scoped_credentials(client, project_id)
    experiment = AlphaEvolveExperiment(
        client,
        make_evaluation_callback(problem_id),
        int(env.get("MAX_PROGRAMS_EVALUATED", "40")),
        parallel_evaluation=parallel,
    )
    return client, experiment


def main() -> None:
    import asyncio
    import nest_asyncio
    from alpha_evolve.controller import run_controller_loop

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--problem", default="readmission_30d", help="Problem ID in problems/<problem_id>")
    p.add_argument("--resume", default=None, help="Existing experiment name or ID to reattach to")
    p.add_argument("--portfolio-report", action="store_true", help="Run local cross-portfolio benchmark scorecard")
    p.add_argument("--env-file", default=HERE / ".env", type=pathlib.Path)
    args = p.parse_args()

    if args.portfolio_report:
        run_portfolio_report()
        return

    if args.env_file.exists():
        for k, v in load_dotenv(args.env_file).items():
            os.environ.setdefault(k, v)
    env = os.environ
    require_private_seed(env)
    heldout_seed = int(env["AE_HELDOUT_SEED"])

    problem_dir = resolve_problem_dir(args.problem)
    seed_code = (problem_dir / "initial_program.py").read_text(encoding="utf-8")
    eval_cb = make_evaluation_callback(args.problem)

    client, experiment = build_client_and_experiment(env, args.problem, inject_creds=True)

    if args.resume:
        experiment.experiment_name = args.resume
        experiment.session_name = args.resume.split("/alphaEvolveExperiments/")[0]
        if (client.get_alpha_evolve_experiment(args.resume) or {}).get("state") != "RUNNING":
            experiment.resume_experiment()
    else:
        cfg_title = load_problem_config(args.problem).get("title", args.problem)
        experiment.create_experiment(
            {
                "title": f"AlphaEvolve — {cfg_title}",
                "problem_description": (problem_dir / "problem_description.md").read_text(encoding="utf-8"),
                "program_language": "python",
                "run_settings": {
                    "max_programs": int(env.get("MAX_PROGRAMS_GENERATED", "40")),
                    "concurrency": int(env.get("CONCURRENCY", "6")),
                },
                "generation_settings": {"models": _models(env)},
            }
        )
        seed_candidate = {"content": {"files": [{"path": "program.py", "content": seed_code}]}}
        experiment.create_initial_program({**seed_candidate, "evaluation": eval_cb(seed_candidate)})
        experiment.start_experiment()
        logging.info(
            "to reattach later: .venv/bin/python run_experiment.py --problem %s --resume=%s",
            args.problem,
            experiment.experiment_name,
        )

    nest_asyncio.apply()
    # Pointer for the simulator's live mode (`/api/live_run`); candidates.json only exists after the run ends.
    exp_id = str(experiment.experiment_name).rsplit("/", 1)[-1]
    out_dir = HERE / env.get("SAVE_PROGRAM_DIR", "artifacts/runs") / args.problem / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_name": experiment.experiment_name,
                "max_programs": int(env.get("MAX_PROGRAMS_EVALUATED", "40")),
                "git_commit": git_head(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    parallel = env.get("PARALLEL_EVALUATION", "False").lower() == "true"
    workers = int(env.get("WORKER_CONCURRENCY", "1")) if parallel else 1
    idle = int(env.get("IDLE_TIMEOUT_S", "900"))
    stop_when_server_done(experiment, client)
    asyncio.run(run_controller_loop(experiment, num_evaluators=workers, idle_timeout_s=idle))

    programs = list_all_programs(experiment)
    scored = [
        prog
        for prog in programs
        if prog.get("evaluation", {}).get("scores", {}).get("scores")
    ]
    scored.sort(
        key=lambda prog: float(prog["evaluation"]["scores"]["scores"][0].get("score", FAILURE_SCORE)),
        reverse=True,
    )

    save_candidate_catalog(programs, seed_code=seed_code, out_dir=out_dir, problem_id=args.problem)

    top_reports = []
    for rank, prog in enumerate(scored[:3], start=1):
        code = prog["content"]["files"][0]["content"]
        (out_dir / f"rank{rank}.py").write_text(code, encoding="utf-8")
        top_reports.append(
            {
                "rank": rank,
                "program_name": prog.get("name"),
                "scores": _rescore(code, args.problem, heldout_seed),
            }
        )

    report = {
        "problem_id": args.problem,
        "experiment_name": experiment.experiment_name,
        "git_commit": git_head(),
        "total_programs_listed": len(programs),
        "baseline": _rescore(seed_code, args.problem, heldout_seed),
        "top": top_reports,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logging.info("Wrote run report and candidate catalog to %s", out_dir)


if __name__ == "__main__":
    main()

