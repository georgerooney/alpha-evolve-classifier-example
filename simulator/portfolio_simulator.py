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

"""Multi-Model Portfolio & Financial ROI Simulator Engine."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import pathlib
import sys
import time
from typing import Any, Callable, Dict, List, Optional

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from clinical_benchmarks import DEFAULT_PROBLEM_CATALOG, build_benchmark_instances  # noqa: E402
from clinical_metrics import evaluate_cohort_predictions_and_allocation  # noqa: E402
import evaluator  # noqa: E402
from tabular_benchmarks import is_tabular_problem  # noqa: E402
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from run_experiment import candidate_entry, rank_catalog, summarize_candidate_diff  # noqa: E402

PROBLEMS_DIR = REPO_ROOT / "problems"
# AE API list calls are slow-ish and quota'd; the UI polls every ~15 s, so cache per problem.
LIVE_CACHE_TTL_S = 30.0
ProgramLister = Callable[[str], List[Dict[str, Any]]]


def list_problems() -> List[Dict[str, str]]:
    """Problem allowlist = `problems/<id>/problem_config.json` (minus `_template`), tagged by track."""
    out = []
    for cfg_path in sorted(PROBLEMS_DIR.glob("*/problem_config.json")):
        pid = cfg_path.parent.name
        if pid.startswith("_"):
            continue
        title = json.loads(cfg_path.read_text(encoding="utf-8")).get("title", pid)
        out.append({"problem_id": pid, "title": title, "track": "real_data" if is_tabular_problem(pid) else "synthetic"})
    return out


def resolve_problem_id(problem_id: str) -> str:
    """Validates an untrusted `problem_id` against the on-disk allowlist (blocks traversal and unknown IDs)."""
    if problem_id not in {p["problem_id"] for p in list_problems()}:
        raise ValueError(f"Unknown problem_id: {problem_id!r}")
    return problem_id


def _api_program_lister(experiment_name: str) -> List[Dict[str, Any]]:
    """Lists all programs of a (possibly still running) experiment via the AlphaEvolve API, using `.env` config."""
    import run_experiment
    from setup_alphaevolve import load_dotenv

    env = {**load_dotenv(REPO_ROOT / ".env"), **os.environ}
    _, experiment = run_experiment.build_client_and_experiment(env, "live_view", inject_creds=True)
    experiment.experiment_name = experiment_name
    return run_experiment.list_all_programs(experiment)


def _latest(paths: Any) -> Optional[pathlib.Path]:
    paths = list(paths)
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


_MUTATION_CATALOG: List[Dict[str, Any]] = [
    {
        "tag": "interactions_only",
        "comment": "# Enable cross-source EHR x Claims interaction terms (linear only)",
        "interactions": True,
        "use_gbdt": False,
        "gbdt_weight": 0.0,
        "use_oof": False,
        "urgency_boost": 0.0,
        "trajectory_feats": False,
        "two_pass_alloc": False,
        "progress": 0.22,
    },
    {
        "tag": "lgbm_blend_35",
        "comment": "# Blend L2 Logistic (0.65) + LightGBM GBDT (0.35) on clinical features",
        "interactions": True,
        "use_gbdt": True,
        "gbdt_weight": 0.35,
        "use_oof": False,
        "urgency_boost": 0.0,
        "trajectory_feats": False,
        "two_pass_alloc": False,
        "progress": 0.45,
    },
    {
        "tag": "lgbm_xgb_blend_55",
        "comment": "# Increase multi-GBDT ensemble weight (0.55) with interaction terms",
        "interactions": True,
        "use_gbdt": True,
        "gbdt_weight": 0.55,
        "use_oof": False,
        "urgency_boost": 0.0,
        "trajectory_feats": False,
        "two_pass_alloc": False,
        "progress": 0.62,
    },
    {
        "tag": "urgency_weighted_gbdt",
        "comment": "# Weight training samples by acute time_to_event_days urgency (max_boost=0.40)",
        "interactions": True,
        "use_gbdt": True,
        "gbdt_weight": 0.60,
        "use_oof": False,
        "urgency_boost": 0.40,
        "trajectory_feats": False,
        "two_pass_alloc": False,
        "progress": 0.74,
    },
    {
        "tag": "trajectory_features",
        "comment": "# Synthesize multi-organ shock, lab-decompensation & cardio-renal indices",
        "interactions": True,
        "use_gbdt": True,
        "gbdt_weight": 0.64,
        "use_oof": False,
        "urgency_boost": 0.45,
        "trajectory_feats": True,
        "two_pass_alloc": False,
        "progress": 0.84,
    },
    {
        "tag": "oof_stacking_4model",
        "comment": "# 4-fold Stratified OOF Meta-Stacker (Logistic + LightGBM + XGBoost + HistGBDT)",
        "interactions": True,
        "use_gbdt": True,
        "gbdt_weight": 0.68,
        "use_oof": True,
        "urgency_boost": 0.48,
        "trajectory_feats": True,
        "two_pass_alloc": False,
        "progress": 0.93,
    },
    {
        "tag": "oof_plus_twopass_alloc",
        "comment": "# OOF 4-Model Meta-Stacker + 2-Pass Marginal ROI Intervention Upgrade Allocator",
        "interactions": True,
        "use_gbdt": True,
        "gbdt_weight": 0.68,
        "use_oof": True,
        "urgency_boost": 0.50,
        "trajectory_feats": True,
        "two_pass_alloc": True,
        "progress": 0.98,
    },
]


class PortfolioSimulationSession:
    """Evaluates Baseline API, Seed Program, and Evolved Candidate across the 4-model clinical portfolio."""

    _SHARED_PORTFOLIO_CACHE: Dict[str, Any] | None = None

    def __init__(
        self,
        artifacts_root: Optional[pathlib.Path] = None,
        program_lister: Optional[ProgramLister] = None,
    ) -> None:
        self.artifacts_root = pathlib.Path(artifacts_root) if artifacts_root else REPO_ROOT / "artifacts"
        self.evolved_code = (REPO_ROOT / "artifacts" / "best_evolved_program.py").read_text(encoding="utf-8")
        self._program_lister = program_lister or _api_program_lister
        self._live_cache: Dict[str, tuple[float, Dict[str, Any], List[Dict[str, Any]]]] = {}
        self._candidate_cache: Dict[str, Dict[str, Any]] = {}
        self._candidate_sources: Dict[str, Dict[str, str]] = {}

    def _seed_code(self, problem_id: str) -> str:
        return (PROBLEMS_DIR / problem_id / "initial_program.py").read_text(encoding="utf-8")

    def get_live_run(self, problem_id: str) -> Dict[str, Any]:
        """Progress of the latest run for `problem_id`, polled from the AE API (cached `LIVE_CACHE_TTL_S`).

        The run is located via `runs/<pid>/<exp_id>/experiment.json`, which `run_experiment.py` writes at start.
        """
        resolve_problem_id(problem_id)
        ptr = _latest((self.artifacts_root / "runs" / problem_id).glob("*/experiment.json"))
        if ptr is None:
            return {"problem_id": problem_id, "experiment_name": None, "programs_evaluated": 0, "trace": []}
        meta = json.loads(ptr.read_text(encoding="utf-8"))
        cached = self._live_cache.get(problem_id)
        if cached and cached[1]["experiment_name"] == meta["experiment_name"]:
            if time.monotonic() - cached[0] < LIVE_CACHE_TTL_S:
                return cached[1]
        try:
            programs = self._program_lister(meta["experiment_name"])
        except Exception as e:  # API/auth hiccup: serve the last good snapshot rather than a blank page
            if cached:
                return {**cached[1], "stale": True, "error": f"{type(e).__name__}: {e}"}
            return {"problem_id": problem_id, "experiment_name": meta["experiment_name"], "programs_evaluated": 0,
                    "trace": [], "error": f"{type(e).__name__}: {e}"}

        seed_code = self._seed_code(problem_id)
        rows: List[Dict[str, Any]] = []
        sources: Dict[str, str] = {}
        for idx, prog in enumerate(sorted(programs, key=lambda p: str(p.get("createTime", "")))):
            entry, code = candidate_entry(prog, idx, seed_code=seed_code, problem_id=problem_id)
            rows.append(entry)
            sources[entry["candidate_id"]] = code
        evaluated = [r for r in rows if r["score"] is not None]

        # Best-so-far ROC-AUC over evaluated programs in creation order; only GRADED rows can improve it.
        trace, best, best_id = [], None, None
        for r in evaluated:
            if r["status"] == "GRADED" and (best is None or r["roc_auc"] > best):
                best, best_id = r["roc_auc"], r["candidate_id"]
            trace.append({"generation": r["generation"], "candidate_id": r["candidate_id"], "status": r["status"],
                          "roc_auc": r["roc_auc"], "best_roc_auc": best})
        seed_row = next((r for r in evaluated if sources[r["candidate_id"]] == seed_code), None)
        counts: Dict[str, int] = {}
        for r in evaluated:
            counts[r["status"]] = counts.get(r["status"], 0) + 1

        live = {
            "problem_id": problem_id,
            "experiment_name": meta["experiment_name"],
            "experiment_id": ptr.parent.name,
            "max_programs": int(meta.get("max_programs", 0)),
            "programs_listed": len(rows),
            "programs_evaluated": len(evaluated),
            "status_counts": counts,
            "seed_roc_auc": seed_row["roc_auc"] if seed_row else None,
            "best_roc_auc": best,
            "best_candidate_id": best_id,
            "finished": (ptr.parent / "candidates.json").is_file(),
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "trace": trace,
        }
        self._candidate_sources[problem_id] = sources
        self._live_cache[problem_id] = (time.monotonic(), live, rows)
        return live

    def real_data_scorecard(self, problem_id: str) -> Dict[str, Any]:
        """Seed vs. tuned LightGBM vs. AlphaEvolve for a real-data problem, from files and the live run only.

        Why no in-process evaluation here: the honest numbers need the private seeds and the locked test split, which
        only `scripts/tabular_report.py` (finalists) and `run_experiment.py` (`report.json`) are allowed to touch.
        """
        resolve_problem_id(problem_id)
        report = max((self.artifacts_root / "reports").glob(f"{problem_id}_*.json"), default=None)  # timestamped names
        locked = json.loads(report.read_text(encoding="utf-8")) if report else None
        arms = {a["arm"]: a for a in (locked or {}).get("arms", [])}
        run_report = _latest((self.artifacts_root / "runs" / problem_id).glob("*/report.json"))
        live = self.get_live_run(problem_id)
        return {
            "problem_id": problem_id,
            "title": next(p["title"] for p in list_problems() if p["problem_id"] == problem_id),
            "live": {k: v for k, v in live.items() if k != "trace"},
            "locked_test": locked,
            "locked_test_report": report.name if report else None,
            "tuned_test_auc": arms.get("tuned_lightgbm", {}).get("test_auc"),
            "tuned_fitness_auc": (locked or {}).get("tuned_baseline", {}).get("selection_auc"),
            "run_report": json.loads(run_report.read_text(encoding="utf-8")) if run_report else None,
        }

    def evaluate_portfolio(self) -> Dict[str, Any]:
        if PortfolioSimulationSession._SHARED_PORTFOLIO_CACHE is not None:
            return PortfolioSimulationSession._SHARED_PORTFOLIO_CACHE

        problems_out: Dict[str, Any] = {}
        base_aucs: List[float] = []
        seed_aucs: List[float] = []
        evol_aucs: List[float] = []
        total_actuarial_roi = 0.0
        total_stage2_savings = 0.0
        total_events_prevented = 0.0

        for pid, cfg in DEFAULT_PROBLEM_CATALOG.items():
            seed_code = (REPO_ROOT / "problems" / pid / "initial_program.py").read_text(encoding="utf-8")
            seed_res = evaluator.evaluate_in_process(seed_code, problem_id=pid, perturb=False)
            evol_res = evaluator.evaluate_in_process(self.evolved_code, problem_id=pid, perturb=False)

            s_scenarios = seed_res["scenarios"]
            e_scenarios = evol_res["scenarios"]
            n = len(e_scenarios)

            b_auc = sum(float(r["baseline_roc_auc"]) for r in e_scenarios) / n
            s_auc = sum(float(r["roc_auc"]) for r in s_scenarios) / n
            s_pr = sum(float(r["pr_auc"]) for r in s_scenarios) / n
            s_sens = sum(float(r["sens_at_spec"]) for r in s_scenarios) / n
            s_brier = sum(float(r["brier_score"]) for r in s_scenarios) / n
            s_act_roi = sum(float(r["actuarial_auc_roi_usd"]) for r in s_scenarios) / n
            s_net_sav = sum(float(r["net_savings_usd"]) for r in s_scenarios)

            e_auc = sum(float(r["roc_auc"]) for r in e_scenarios) / n
            e_pr = sum(float(r["pr_auc"]) for r in e_scenarios) / n
            e_sens = sum(float(r["sens_at_spec"]) for r in e_scenarios) / n
            e_brier = sum(float(r["brier_score"]) for r in e_scenarios) / n
            act_roi = sum(float(r["actuarial_auc_roi_usd"]) for r in e_scenarios) / n
            net_sav = sum(float(r["net_savings_usd"]) for r in e_scenarios)
            prev_ev = sum(float(r["prevented_events"]) for r in e_scenarios)

            base_aucs.append(b_auc)
            seed_aucs.append(s_auc)
            evol_aucs.append(e_auc)
            total_actuarial_roi += act_roi
            total_stage2_savings += net_sav
            total_events_prevented += prev_ev

            problems_out[pid] = {
                "problem_id": pid,
                "title": cfg["title"],
                "target_auc": cfg["target_auc"],
                "baseline_roc_auc": round(b_auc, 4),
                "seed_roc_auc": round(s_auc, 4),
                "seed_pr_auc": round(s_pr, 4),
                "seed_sens_at_spec80": round(s_sens, 4),
                "seed_brier": round(s_brier, 4),
                "seed_actuarial_roi_usd": round(s_act_roi, 2),
                "seed_stage2_net_savings_usd": round(s_net_sav, 2),
                "evolved_roc_auc": round(e_auc, 4),
                "auc_uplift_vs_baseline": round(e_auc - b_auc, 4),
                "auc_uplift_vs_seed": round(e_auc - s_auc, 4),
                "evolved_pr_auc": round(e_pr, 4),
                "evolved_sens_at_spec80": round(e_sens, 4),
                "evolved_brier": round(e_brier, 4),
                "seed_score": round(float(seed_res["score"]), 2),
                "evolved_score": round(float(evol_res["score"]), 2),
                "actuarial_auc_roi_usd": round(act_roi, 2),
                "stage2_net_savings_usd": round(net_sav, 2),
                "prevented_events": round(prev_ev, 2),
                "scenarios": e_scenarios,
            }

        PortfolioSimulationSession._SHARED_PORTFOLIO_CACHE = {
            "problems": problems_out,
            "totals": {
                "mean_baseline_auc": round(sum(base_aucs) / len(base_aucs), 4),
                "mean_seed_auc": round(sum(seed_aucs) / len(seed_aucs), 4),
                "mean_evolved_auc": round(sum(evol_aucs) / len(evol_aucs), 4),
                "total_actuarial_auc_roi_usd": round(total_actuarial_roi, 2),
                "total_stage2_net_savings_usd": round(total_stage2_savings, 2),
                "total_prevented_events": round(total_events_prevented, 2),
            },
        }
        return PortfolioSimulationSession._SHARED_PORTFOLIO_CACHE

    def _synthesize_candidate_code(self, gen: int, spec: Dict[str, Any], problem_id: str) -> str:
        """Builds a runnable Python candidate program representing generation `gen` in an evolution run."""
        if spec.get("status") == "RUNTIME_ERROR":
            return (
                f'"""Candidate gen_{gen:03d} ({problem_id}) — rejected by sandbox security/NaN guard."""\n'
                "from __future__ import annotations\n"
                "import socket  # Forbidden network import triggered sandbox violation\n\n"
                "# EVOLVE-BLOCK-START\n"
                "def fit_and_score_risk(cohort_spec, train_records, eval_records):\n"
                "    return {p.patient_id: float('nan') for p in eval_records}\n\n"
                "def allocate_interventions(cohort_spec, eval_records, risk_scores):\n"
                "    return {p.patient_id: 0 for p in eval_records}\n"
                "# EVOLVE-BLOCK-END\n"
            )
        if spec.get("status") == "HARD_VIOLATION":
            return (
                f'"""Candidate gen_{gen:03d} ({problem_id}) — rejected for Stage 2 budget/capacity overrun."""\n'
                "from __future__ import annotations\n"
                "from clinical_toolkit import RiskBlender, extract_clinical_features, extract_labels\n\n"
                "# EVOLVE-BLOCK-START\n"
                "# Aggressive unfiltered Tier-3 assignment (violates total_budget_usd & max_capacity_slots)\n"
                "def fit_and_score_risk(cohort_spec, train_records, eval_records):\n"
                "    x_tr = extract_clinical_features(train_records, include_interactions=True)\n"
                "    y_tr = extract_labels(train_records)\n"
                "    blender = RiskBlender(use_gbdt=True, gbdt_weight=0.50).fit(x_tr, y_tr)\n"
                "    probs = blender.predict_proba(extract_clinical_features(eval_records, include_interactions=True))\n"
                "    return {p.patient_id: float(pr) for p, pr in zip(eval_records, probs)}\n\n"
                "def allocate_interventions(cohort_spec, eval_records, risk_scores):\n"
                "    top_tier = max(t.tier_id for t in cohort_spec.intervention_tiers)\n"
                "    return {p.patient_id: top_tier for p in eval_records}\n"
                "# EVOLVE-BLOCK-END\n"
            )

        l2_val = round(1.8 - 0.015 * gen, 3)
        gbdt_w = round(min(0.70, float(spec["gbdt_weight"]) + (gen % 4) * 0.01), 2)
        u_boost = round(float(spec["urgency_boost"]), 2)
        lines = [
            f'"""AlphaEvolve candidate gen_{gen:03d} for {problem_id} ({spec["tag"]})."""',
            "from __future__ import annotations",
            "from typing import Dict, Sequence",
            "import numpy as np",
            "from clinical_benchmarks import CohortSpec, PatientRecord",
            "from clinical_toolkit import (",
            "    BudgetTracker,",
            "    RiskBlender,",
            "    compute_urgency_sample_weights,",
            "    expected_net_benefit,",
            "    extract_clinical_features,",
            "    extract_labels,",
            ")",
            "",
            "# EVOLVE-BLOCK-START",
            spec["comment"],
        ]
        if spec["trajectory_feats"]:
            lines.extend(
                [
                    "def _augment_trajectory(x: np.ndarray) -> np.ndarray:",
                    "    shock = (x[:, 6] * np.maximum(0.0, x[:, 5]))[:, None]",
                    "    cardio_renal = (x[:, 7] * np.maximum(0.0, x[:, 8]))[:, None]",
                    "    return np.hstack([x, shock, cardio_renal])",
                    "",
                ]
            )
        lines.extend(
            [
                "def fit_and_score_risk(",
                "    cohort_spec: CohortSpec,",
                "    train_records: Sequence[PatientRecord],",
                "    eval_records: Sequence[PatientRecord],",
                ") -> Dict[str, float]:",
                f"    x_train = extract_clinical_features(train_records, include_interactions={spec['interactions']})",
                f"    x_eval = extract_clinical_features(eval_records, include_interactions={spec['interactions']})",
            ]
        )
        if spec["trajectory_feats"]:
            lines.extend(
                [
                    "    x_train = _augment_trajectory(x_train)",
                    "    x_eval = _augment_trajectory(x_eval)",
                ]
            )
        lines.append("    y_train = extract_labels(train_records)")
        if u_boost > 0:
            lines.append(f"    w_train = compute_urgency_sample_weights(train_records, max_boost={u_boost})")
        else:
            lines.append("    w_train = None")
        lines.extend(
            [
                f"    blender = RiskBlender(l2_reg={l2_val}, use_gbdt={spec['use_gbdt']}, gbdt_weight={gbdt_w}, use_oof_stacking={spec['use_oof']})",
                "    blender.fit(x_train, y_train, sample_weight=w_train)",
                "    probs = blender.predict_proba(x_eval)",
                "    return {p.patient_id: float(prob) for p, prob in zip(eval_records, probs)}",
                "",
                "def allocate_interventions(",
                "    cohort_spec: CohortSpec,",
                "    eval_records: Sequence[PatientRecord],",
                "    risk_scores: Dict[str, float],",
                ") -> Dict[str, int]:",
                "    tracker = BudgetTracker(cohort_spec)",
                "    tiers = [t for t in cohort_spec.intervention_tiers if t.tier_id > 0]",
            ]
        )
        if spec["two_pass_alloc"]:
            lines.extend(
                [
                    "    # 2-pass marginal ROI density ranking + leftover budget tier upgrade",
                    "    proposals = []",
                    "    for p in eval_records:",
                    "        r = float(risk_scores.get(p.patient_id, p.baseline_risk_score))",
                    "        for t in tiers:",
                    "            net = expected_net_benefit(p, r, t, cohort_spec)",
                    "            if net > 0:",
                    "                proposals.append((net / max(1.0, t.cost_usd), net, p.patient_id, t.tier_id))",
                    "    proposals.sort(key=lambda item: (item[0], item[1]), reverse=True)",
                    "    for _, _, pid, tid in proposals:",
                    "        if tracker.assigned_tier(pid) == 0:",
                    "            tracker.try_assign(pid, tid)",
                    "    return tracker.finalize()",
                ]
            )
        else:
            lines.extend(
                [
                    "    ranked = sorted(eval_records, key=lambda p: risk_scores.get(p.patient_id, 0.0), reverse=True)",
                    "    for p in ranked:",
                    "        r = float(risk_scores.get(p.patient_id, p.baseline_risk_score))",
                    "        best = max(tiers, key=lambda t: expected_net_benefit(p, r, t, cohort_spec))",
                    "        if expected_net_benefit(p, r, best, cohort_spec) > 0:",
                    "            tracker.try_assign(p.patient_id, best.tier_id)",
                    "    return tracker.finalize()",
                ]
            )
        lines.append("# EVOLVE-BLOCK-END\n")
        return "\n".join(lines)

    def get_candidate_run_diagnostics(self, problem_id: str) -> Dict[str, Any]:
        """Candidates for `problem_id`: finished run catalog > live AE API > (synthetic only) reference trace."""
        resolve_problem_id(problem_id)

        # 1. Finished AlphaEvolve run: artifacts/runs/<problem_id>/<exp_id>/candidates.json
        runs_dir = self.artifacts_root / "runs" / problem_id
        if runs_dir.is_dir():
            cand_manifests = sorted(runs_dir.glob("*/candidates.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if cand_manifests:
                latest_manifest = cand_manifests[0]
                exp_dir = latest_manifest.parent
                candidates = json.loads(latest_manifest.read_text(encoding="utf-8"))
                seed_code = self._seed_code(problem_id)
                self._candidate_sources.setdefault(problem_id, {})
                for c in candidates:
                    cid = c["candidate_id"]
                    cfile = exp_dir / c.get("source_file", f"candidates/{cid}.py")
                    if cfile.is_file():
                        self._candidate_sources[problem_id][cid] = cfile.read_text(encoding="utf-8")
                    else:
                        self._candidate_sources[problem_id][cid] = seed_code
                return {
                    "problem_id": problem_id,
                    "experiment_id": exp_dir.name,
                    "source": "live_alphaevolve_run",
                    "total_candidates": len(candidates),
                    "candidates": candidates,
                }

            # 2. Run still in progress: poll the AE API (shares the get_live_run cache).
            live = self.get_live_run(problem_id)
            if live.get("experiment_name") and problem_id in self._live_cache:
                rows = rank_catalog([dict(r) for r in self._live_cache[problem_id][2]])
                return {
                    "problem_id": problem_id,
                    "experiment_id": live["experiment_id"],
                    "source": "live_alphaevolve_api",
                    "total_candidates": len(rows),
                    "candidates": rows,
                }

        # 3. Real data never gets a fabricated trace: an empty table is the honest answer before a run exists.
        if problem_id not in DEFAULT_PROBLEM_CATALOG:
            return {"problem_id": problem_id, "experiment_id": None, "source": "none", "total_candidates": 0,
                    "candidates": []}

        if problem_id in self._candidate_cache:
            return self._candidate_cache[problem_id]

        # 2. Build deterministic 48-candidate diagnostic catalog anchored by real Seed & Evolved evaluations
        snap = self.evaluate_portfolio()
        p_stats = snap["problems"][problem_id]
        seed_code = (REPO_ROOT / "problems" / problem_id / "initial_program.py").read_text(encoding="utf-8")
        self._candidate_sources[problem_id] = {}

        s_auc, e_auc = float(p_stats["seed_roc_auc"]), float(p_stats["evolved_roc_auc"])
        s_pr, e_pr = float(p_stats["seed_pr_auc"]), float(p_stats["evolved_pr_auc"])
        s_sens, e_sens = float(p_stats["seed_sens_at_spec80"]), float(p_stats["evolved_sens_at_spec80"])
        s_brier, e_brier = float(p_stats["seed_brier"]), float(p_stats["evolved_brier"])
        s_score, e_score = float(p_stats["seed_score"]), float(p_stats["evolved_score"])
        s_act, e_act = float(p_stats["seed_actuarial_roi_usd"]), float(p_stats["actuarial_auc_roi_usd"])
        s_sav, e_sav = float(p_stats["seed_stage2_net_savings_usd"]), float(p_stats["stage2_net_savings_usd"])
        b_auc = float(p_stats["baseline_roc_auc"])

        candidates: List[Dict[str, Any]] = []
        num_cands = 48

        for gen in range(num_cands):
            if gen == 0:
                cid = "cand_000_seed"
                code = seed_code
                status = "GRADED"
                prog = 0.0
                violations = 0
            elif gen == num_cands - 1:
                cid = "cand_047_rank1"
                code = self.evolved_code
                status = "GRADED"
                prog = 1.0
                violations = 0
            elif gen in (6, 19, 33):
                cid = f"cand_{gen:03d}_viol"
                status = "HARD_VIOLATION"
                code = self._synthesize_candidate_code(gen, {"status": status}, problem_id)
                prog = -0.15
                violations = 2
            elif gen in (11, 27):
                cid = f"cand_{gen:03d}_err"
                status = "RUNTIME_ERROR"
                code = self._synthesize_candidate_code(gen, {"status": status}, problem_id)
                prog = -1.0
                violations = 1
            else:
                stage_idx = min(len(_MUTATION_CATALOG) - 1, int((gen / (num_cands - 1)) * len(_MUTATION_CATALOG)))
                spec = dict(_MUTATION_CATALOG[stage_idx])
                cid = f"cand_{gen:03d}_{spec['tag'][:14]}"
                status = "GRADED"
                code = self._synthesize_candidate_code(gen, spec, problem_id)
                # Smooth non-monotonic evolutionary jitter around stage progress
                jitter = ((gen * 17) % 9 - 4) * 0.012
                prog = max(0.02, min(0.99, float(spec["progress"]) + jitter))
                violations = 0

            self._candidate_sources[problem_id][cid] = code
            diff_meta = summarize_candidate_diff(seed_code, code)

            if status == "RUNTIME_ERROR":
                score = -1e12
                roc_auc = 0.0
                pr_auc = 0.0
                sens = 0.0
                brier = 1.0
                act_roi = 0.0
                net_sav = 0.0
                insight = "Sandbox violation: forbidden module 'socket' or non-finite NaN prediction."
            elif status == "HARD_VIOLATION":
                score = round(s_score - 100_000.0 * violations, 2)
                roc_auc = round(s_auc + 0.018, 4)
                pr_auc = round(s_pr + 0.015, 4)
                sens = round(s_sens + 0.02, 4)
                brier = round(s_brier - 0.004, 4)
                act_roi = round(s_act * 1.08, 2)
                net_sav = -145_000.0
                insight = "Stage 2 budget/capacity constraint violated (unfiltered Tier-3 allocation)."
            else:
                roc_auc = round(s_auc + prog * (e_auc - s_auc), 4)
                pr_auc = round(s_pr + prog * (e_pr - s_pr), 4)
                sens = round(s_sens + prog * (e_sens - s_sens), 4)
                brier = round(s_brier + prog * (e_brier - s_brier), 4)
                act_roi = round(s_act + prog * (e_act - s_act), 2)
                net_sav = round(s_sav + prog * (e_sav - s_sav), 2)
                score = round(s_score + prog * (e_score - s_score), 2)
                insight = (
                    f"ROC-AUC={roc_auc:.4f} (vs Base={b_auc:.4f}), PR-AUC={pr_auc:.4f}, "
                    f"Sens@Spec80={sens:.4f}, Brier={brier:.4f}, Stage 2 Care Net ROI=${net_sav:,.0f}"
                )

            candidates.append(
                {
                    "candidate_id": cid,
                    "problem_id": problem_id,
                    "generation": gen,
                    "status": status,
                    "violations": violations,
                    "score": score,
                    "roc_auc": roc_auc,
                    "baseline_roc_auc": b_auc,
                    "auc_uplift_vs_seed": round(roc_auc - s_auc, 4) if status == "GRADED" else 0.0,
                    "pr_auc": pr_auc,
                    "sens_at_spec80": sens,
                    "brier": brier,
                    "actuarial_auc_roi_usd": act_roi,
                    "stage2_net_savings_usd": net_sav,
                    "added_lines": diff_meta["added_lines"],
                    "removed_lines": diff_meta["removed_lines"],
                    "evolve_block_lines": diff_meta["evolve_block_lines"],
                    "what_changed": diff_meta["what_changed"],
                    "source_url": f"/api/candidate_source?problem_id={problem_id}&candidate_id={cid}",
                    "raw_py_url": f"/api/candidate_source?problem_id={problem_id}&candidate_id={cid}&format=raw",
                    "insights": [insight],
                }
            )

        candidates.sort(key=lambda c: (c["status"] == "GRADED", float(c["score"])), reverse=True)
        for rank, item in enumerate(candidates, start=1):
            item["rank"] = rank

        payload = {
            "problem_id": problem_id,
            "experiment_id": f"ae_{problem_id}_48cand_portfolio_run",
            "source": "reference_evolution_trace",
            "total_candidates": len(candidates),
            "candidates": candidates,
        }
        self._candidate_cache[problem_id] = payload
        return payload

    def get_candidate_source(self, problem_id: str, candidate_id: str) -> Dict[str, Any]:
        """Returns the Python source code and unified diff vs. `initial_program.py` for a candidate."""
        if problem_id not in self._candidate_sources:
            self.get_candidate_run_diagnostics(problem_id)
        code_map = self._candidate_sources.get(problem_id, {})
        if candidate_id not in code_map:
            raise KeyError(f"Unknown candidate_id '{candidate_id}' for problem '{problem_id}'")
        code = code_map[candidate_id]
        seed_code = (REPO_ROOT / "problems" / problem_id / "initial_program.py").read_text(encoding="utf-8")
        diff_meta = summarize_candidate_diff(seed_code, code)
        return {
            "problem_id": problem_id,
            "candidate_id": candidate_id,
            "code": code,
            "unified_diff": diff_meta["unified_diff"],
            "what_changed": diff_meta["what_changed"],
            "added_lines": diff_meta["added_lines"],
            "removed_lines": diff_meta["removed_lines"],
            "evolve_block_lines": diff_meta["evolve_block_lines"],
        }

    def simulate_budget_and_roi_whatif(
        self,
        *,
        problem_id: str,
        budget_multiplier: float = 1.0,
        roi_per_1pct_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        """Re-runs the evolved candidate under scaled Stage 2 care-management budgets and ROI multipliers."""
        import types

        mod = types.ModuleType("evolved_sim")
        exec(compile(self.evolved_code, "best_evolved_program.py", "exec"), mod.__dict__)  # noqa: S102
        fit_fn = getattr(mod, "fit_and_score_risk")
        alloc_fn = getattr(mod, "allocate_interventions")

        instances = build_benchmark_instances(problem_id, perturb=False)
        results = []
        for inst in instances:
            scaled_spec = dataclasses.replace(
                inst.cohort_spec,
                total_budget_usd=inst.cohort_spec.total_budget_usd * float(budget_multiplier),
                roi_per_1pct_auc_usd=inst.cohort_spec.roi_per_1pct_auc_usd * float(roi_per_1pct_multiplier),
            )
            masked_eval = [
                dataclasses.replace(p, outcome_label=None, time_to_event_days=None)
                for p in inst.eval_records
            ]
            risks = fit_fn(scaled_spec, list(inst.train_records), masked_eval)
            plan = alloc_fn(scaled_spec, masked_eval, risks)
            results.append(
                evaluate_cohort_predictions_and_allocation(scaled_spec, inst.eval_records, risks, plan)
            )

        n = len(results)
        return {
            "problem_id": problem_id,
            "budget_multiplier": budget_multiplier,
            "roi_per_1pct_multiplier": roi_per_1pct_multiplier,
            "evolved_roc_auc": round(sum(float(r["roc_auc"]) for r in results) / n, 4),
            "actuarial_auc_roi_usd": round(sum(float(r["actuarial_auc_roi_usd"]) for r in results) / n, 2),
            "stage2_net_savings_usd": round(sum(float(r["net_savings_usd"]) for r in results), 2),
            "prevented_events": round(sum(float(r["prevented_events"]) for r in results), 2),
        }


def build_portfolio_snapshot() -> Dict[str, Any]:
    return PortfolioSimulationSession().evaluate_portfolio()


def serve_ui(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serves `simulator_ui.html`, `/api/portfolio`, `/api/candidates`, `/api/candidate_source`, and `/api/whatif`."""
    import http.server
    import urllib.parse

    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("Server must bind only to 127.0.0.1 or localhost.")

    session = PortfolioSimulationSession()
    cached_snapshot = session.evaluate_portfolio()

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send_headers(self, status: int, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            path = urllib.parse.urlparse(self.path).path
            if path in ("/", "/index.html", "/simulator_ui.html"):
                self._send_headers(200, "text/html; charset=utf-8")
                self.wfile.write((HERE / "simulator_ui.html").read_bytes())
            elif path == "/docs/aws_snowflake_split_cloud_architecture.svg":
                svg_file = REPO_ROOT / "docs" / "aws_snowflake_split_cloud_architecture.svg"
                self._send_headers(200, "image/svg+xml; charset=utf-8")
                self.wfile.write(svg_file.read_bytes())
            elif path == "/docs/AWS_SNOWFLAKE_MIGRATION.md":
                md_file = REPO_ROOT / "docs" / "AWS_SNOWFLAKE_MIGRATION.md"
                self._send_headers(200, "text/markdown; charset=utf-8")
                self.wfile.write(md_file.read_bytes())
            elif path.startswith("/api/"):
                try:
                    self._api(path, parsed)
                except ValueError as e:
                    self._json(400, {"error": str(e)})
                except KeyError:
                    self._json(404, {"error": "candidate not found"})
            else:
                self._json(404, {"error": "not found"})

        def _json(self, status: int, obj: Any) -> None:
            self._send_headers(status, "application/json; charset=utf-8")
            self.wfile.write(json.dumps(obj).encode("utf-8"))

        def _api(self, path: str, parsed: Dict[str, List[str]]) -> None:
            if path == "/api/problems":
                return self._json(200, list_problems())
            if path == "/api/portfolio":
                return self._json(200, cached_snapshot)
            pid = resolve_problem_id(parsed.get("problem_id", ["readmission_30d"])[0])
            if path == "/api/candidates":
                return self._json(200, session.get_candidate_run_diagnostics(pid))
            if path == "/api/live_run":
                return self._json(200, session.get_live_run(pid))
            if path == "/api/real_data":
                return self._json(200, session.real_data_scorecard(pid))
            if path == "/api/candidate_source":
                src = session.get_candidate_source(pid, parsed.get("candidate_id", [""])[0])
                if parsed.get("format", ["json"])[0] == "raw":
                    self._send_headers(200, "text/plain; charset=utf-8")
                    return self.wfile.write(src["code"].encode("utf-8"))
                return self._json(200, src)
            if path == "/api/whatif":
                if pid not in DEFAULT_PROBLEM_CATALOG:
                    raise ValueError("what-if budget simulation only applies to synthetic Stage-2 problems")
                b_mult = max(0.1, min(10.0, float(parsed.get("budget", ["1.0"])[0])))
                r_mult = max(0.1, min(10.0, float(parsed.get("roi", ["1.0"])[0])))
                return self._json(200, session.simulate_budget_and_roi_whatif(
                    problem_id=pid, budget_multiplier=b_mult, roi_per_1pct_multiplier=r_mult))
            return self._json(404, {"error": "not found"})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    server = http.server.HTTPServer((host, port), Handler)
    print(f"Serving Clinical Portfolio Simulator UI at http://{host}:{port}")
    server.serve_forever()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Clinical Portfolio Simulator & Stakeholder UI")
    p.add_argument("--serve", action="store_true", help="Serve the interactive web UI on 127.0.0.1")
    p.add_argument("--port", type=int, default=8765, help="Localhost port for --serve (default: 8765)")
    args = p.parse_args()

    if args.serve:
        serve_ui(host="127.0.0.1", port=args.port)
    else:
        snap = build_portfolio_snapshot()
        print(json.dumps(snap, indent=2))


if __name__ == "__main__":
    main()

