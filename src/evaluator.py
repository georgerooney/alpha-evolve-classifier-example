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

"""GCP AlphaEvolve Evaluator Harness for the Multi-Model Clinical Portfolio.

Conforms to the official AlphaEvolve CLI and library contracts:
- Exports `evaluate_program(code: str, timeout_seconds: int = 45, problem_id: str = "readmission_30d", ...)`
  returning `{"score": float | None, "insights": list[str]}`.
- Accepts `--program-dir <path>`, `--output-file <path>`, `--problem <id>`, `--split <all|test>`, and `--no-perturb`.
- Isolates candidate execution in a `python -P -s` child process (`PYTHONHASHSEED=0`), pre-imports the GBDT/ML
  stack, installs a C-level `sys.addaudithook` blocking filesystem/network/subprocess/import escape, strips ground-truth
  labels from `eval_records` before they cross the pipe, and enforces per-call crash isolation.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import pathlib
import re
import resource
import select
import subprocess
import sys
import time
import traceback
import types
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from clinical_benchmarks import (  # noqa: E402
    build_benchmark_instances,
    build_heldout_benchmark_instances,
    load_problem_config,
)
from clinical_metrics import (  # noqa: E402
    compute_brier_score,
    compute_pr_auc,
    compute_roc_auc,
    evaluate_cohort_predictions_and_allocation,
)
from clinical_models import (  # noqa: E402
    CareAllocationPlan,
    CohortSpec,
    InterventionSpec,
    InterventionTier,
    PatientRecord,
    TabularTask,
)
from harness_config import DEFAULT_WEIGHTS  # noqa: E402
from tabular_benchmarks import (  # noqa: E402
    TabularRound,
    build_tabular_rounds,
    is_tabular_problem,
    subgroup_columns,
)


def _reject_nonfinite_json(val: str) -> Any:
    """Rejects NaN, Infinity, and -Infinity during JSON parsing."""
    raise ValueError(f"Non-finite JSON constant not allowed: {val}")


def _sanitize_line(text: str, max_len: int = 160) -> str:
    """Strips absolute file paths and caps line length for safe LLM prompt feedback."""
    cleaned = re.sub(r"(/[\w.\-]+)+/([\w.\-]+\.py)", r"\2", str(text))
    cleaned = " ".join(cleaned.strip().split())
    return cleaned[: max_len - 3] + "..." if len(cleaned) > max_len else cleaned


def _serialize_patient(p: PatientRecord, include_labels: bool) -> Dict[str, Any]:
    return {
        "patient_id": str(p.patient_id),
        "age": float(p.age),
        "sex": int(p.sex),
        "charlson_index": float(p.charlson_index),
        "prior_ed_visits_6m": int(p.prior_ed_visits_6m),
        "prior_ip_admissions_12m": int(p.prior_ip_admissions_12m),
        "length_of_stay_days": float(p.length_of_stay_days),
        "sdoh_deprivation_index": float(p.sdoh_deprivation_index),
        "acuity_score": float(p.acuity_score),
        "features": {str(k): float(v) for k, v in p.features.items()},
        "baseline_api_prob": float(p.baseline_api_prob),
        "baseline_submodel_scores": {
            str(k): float(v) for k, v in p.baseline_submodel_scores.items()
        },
        "contraindications": [str(x) for x in p.contraindications],
        # CRITICAL anti-leakage invariant: eval_records never carry outcome_label or time_to_event_days
        "outcome_label": (
            int(p.outcome_label)
            if include_labels and p.outcome_label is not None
            else None
        ),
        "time_to_event_days": (
            float(p.time_to_event_days)
            if include_labels and p.time_to_event_days is not None
            else None
        ),
    }


def _deserialize_patient(d: Mapping[str, Any]) -> PatientRecord:
    return PatientRecord(
        patient_id=str(d["patient_id"]),
        age=float(d["age"]),
        sex=int(d["sex"]),
        charlson_index=float(d["charlson_index"]),
        prior_ed_visits_6m=int(d["prior_ed_visits_6m"]),
        prior_ip_admissions_12m=int(d["prior_ip_admissions_12m"]),
        length_of_stay_days=float(d["length_of_stay_days"]),
        sdoh_deprivation_index=float(d["sdoh_deprivation_index"]),
        acuity_score=float(d["acuity_score"]),
        features={str(k): float(v) for k, v in d.get("features", {}).items()},
        baseline_api_prob=float(d["baseline_api_prob"]),
        baseline_submodel_scores={
            str(k): float(v) for k, v in d.get("baseline_submodel_scores", {}).items()
        },
        contraindications=tuple(str(x) for x in d.get("contraindications", ())),
        outcome_label=(
            int(d["outcome_label"]) if d.get("outcome_label") is not None else None
        ),
        time_to_event_days=(
            float(d["time_to_event_days"])
            if d.get("time_to_event_days") is not None
            else None
        ),
    )


def _serialize_cohort_spec(spec: CohortSpec) -> Dict[str, Any]:
    return {
        "problem_id": str(spec.problem_id),
        "scenario_id": str(spec.scenario_id),
        "scenario_name": str(spec.scenario_name),
        "horizon_days": int(spec.horizon_days),
        "target_auc": float(spec.target_auc),
        "target_specificity": float(spec.target_specificity),
        "min_sensitivity_floor": float(spec.min_sensitivity_floor),
        "event_cost_usd": float(spec.event_cost_usd),
        "hrrp_penalty_multiplier": float(spec.hrrp_penalty_multiplier),
        "roi_per_1pct_auc_usd": float(spec.roi_per_1pct_auc_usd),
        "total_budget_usd": float(spec.total_budget_usd),
        "interventions": {
            t.value: {
                "tier": t.value,
                "cost_usd": float(s.cost_usd),
                "relative_risk_reduction": float(s.relative_risk_reduction),
                "max_slots": int(s.max_slots),
                "min_acuity_score": float(s.min_acuity_score),
                "contraindicated_flags": list(s.contraindicated_flags),
            }
            for t, s in spec.interventions.items()
        },
        "feature_names": list(spec.feature_names),
    }


def _deserialize_cohort_spec(d: Mapping[str, Any]) -> CohortSpec:
    interventions: Dict[InterventionTier, InterventionSpec] = {}
    for k, s in d.get("interventions", {}).items():
        tier = InterventionTier(str(k))
        interventions[tier] = InterventionSpec(
            tier=tier,
            cost_usd=float(s["cost_usd"]),
            relative_risk_reduction=float(s["relative_risk_reduction"]),
            max_slots=int(s["max_slots"]),
            min_acuity_score=float(s.get("min_acuity_score", 0.0)),
            contraindicated_flags=tuple(str(x) for x in s.get("contraindicated_flags", ())),
        )
    return CohortSpec(
        problem_id=str(d["problem_id"]),
        scenario_id=str(d["scenario_id"]),
        scenario_name=str(d["scenario_name"]),
        horizon_days=int(d["horizon_days"]),
        target_auc=float(d["target_auc"]),
        target_specificity=float(d["target_specificity"]),
        min_sensitivity_floor=float(d["min_sensitivity_floor"]),
        event_cost_usd=float(d["event_cost_usd"]),
        hrrp_penalty_multiplier=float(d["hrrp_penalty_multiplier"]),
        roi_per_1pct_auc_usd=float(d["roi_per_1pct_auc_usd"]),
        total_budget_usd=float(d["total_budget_usd"]),
        interventions=interventions,
        feature_names=tuple(str(x) for x in d.get("feature_names", ())),
    )


def _write_json_line(fd: int, payload: Mapping[str, Any]) -> None:
    raw = (json.dumps(payload, allow_nan=False) + "\n").encode("utf-8")
    view = memoryview(raw)
    sent_total = 0
    while sent_total < len(raw):
        n = os.write(fd, view[sent_total:])
        if n <= 0:
            raise EOFError("Pipe closed while writing JSON payload")
        sent_total += n


def _read_json_line(fd: int, timeout_s: float) -> Dict[str, Any]:
    buf = bytearray()
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Worker timed out after {timeout_s:.1f}s")
        rlist, _, _ = select.select([fd], [], [], min(0.25, remaining))
        if not rlist:
            continue
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            raise EOFError("Worker closed pipe unexpectedly")
        # Only scan the new chunk: tabular payloads are tens of MB, and rescanning `buf` each read is quadratic.
        nl_in_chunk = chunk.find(b"\n")
        if nl_in_chunk != -1:
            buf.extend(chunk[:nl_in_chunk])
            line = bytes(buf).decode("utf-8", errors="replace")
            return json.loads(line, parse_constant=_reject_nonfinite_json)
        buf.extend(chunk)


def _columns_to_payload(cols: Mapping[str, np.ndarray], task: TabularTask) -> Dict[str, list]:
    """Columnar JSON: numeric NaN -> null (strict JSON forbids NaN); categoricals stay str/None."""
    out: Dict[str, list] = {}
    for name in task.numeric_columns:
        out[name] = [None if v != v else v for v in np.asarray(cols[name], dtype=np.float64).tolist()]
    for name in task.categorical_columns:
        out[name] = [None if v is None else str(v) for v in np.asarray(cols[name], dtype=object).tolist()]
    return out


def _payload_to_columns(payload: Mapping[str, list], task: TabularTask) -> Dict[str, np.ndarray]:
    cols: Dict[str, np.ndarray] = {}
    for name in task.numeric_columns:
        cols[name] = np.array(payload[name], dtype=np.float64)  # None -> NaN
    for name in task.categorical_columns:
        cols[name] = np.array(payload[name], dtype=object)
    return cols


def _task_from_payload(d: Mapping[str, Any]) -> TabularTask:
    return TabularTask(
        problem_id=str(d["problem_id"]),
        round_id=str(d["round_id"]),
        numeric_columns=tuple(str(x) for x in d["numeric_columns"]),
        categorical_columns=tuple(str(x) for x in d["categorical_columns"]),
        n_train=int(d["n_train"]),
        n_eval=int(d["n_eval"]),
        label_description=str(d.get("label_description", "")),
    )


def _install_redacted_benchmarks_stub() -> None:
    """Replaces `clinical_benchmarks` and `tabular_benchmarks` in child `sys.modules` with redacted stubs."""
    stub = types.ModuleType("clinical_benchmarks")

    @dataclasses.dataclass(frozen=True)
    class RedactedInstance:
        problem_id: str = "REDACTED"
        scenario_id: str = "REDACTED"
        scenario_name: str = "Redacted Benchmark Instance"
        description: str = "Redacted"
        cohort_spec: Any = None
        train_records: tuple = ()
        eval_records: tuple = ()

    def _redacted_list(*args: Any, **kwargs: Any) -> list:
        return [RedactedInstance() for _ in range(4)]

    stub.ClinicalBenchmarkInstance = RedactedInstance  # type: ignore[attr-defined]
    stub.build_benchmark_instances = _redacted_list  # type: ignore[attr-defined]
    stub.build_heldout_benchmark_instances = _redacted_list  # type: ignore[attr-defined]
    tab_stub = types.ModuleType("tabular_benchmarks")
    for mod in (stub, tab_stub):
        mod.__file__ = None
        mod.__spec__ = None
        mod.__loader__ = None
    sys.modules["clinical_benchmarks"] = stub
    sys.modules["tabular_benchmarks"] = tab_stub


def _worker_main(
    cmd_read_fd: int, resp_write_fd: int, timeout_seconds: int, cpu_limit_seconds: Optional[int] = None
) -> None:
    """Isolated child worker executing untrusted candidate code under a C-level audit hook."""
    try:
        try:
            # RLIMIT_CPU is cumulative over the worker's life, so multi-round tabular evals pass an explicit budget.
            cpu_lim = max(5, int(cpu_limit_seconds or (int(timeout_seconds) + 5)))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_lim, cpu_lim))
        except (ValueError, OSError):
            pass

        # Strip private evaluation seeds in case any survived environment scrubbing
        os.environ.pop("AE_PERTURB_SEED", None)
        os.environ.pop("AE_HELDOUT_SEED", None)

        # Pre-import domain & numerical/GBDT libraries BEFORE installing the audit hook
        import bisect  # noqa: F401
        import collections  # noqa: F401
        import copy  # noqa: F401
        import functools  # noqa: F401
        import heapq  # noqa: F401
        import itertools  # noqa: F401
        import statistics  # noqa: F401
        import numpy as np  # noqa: F401
        import scipy  # noqa: F401
        import scipy.optimize  # noqa: F401
        import scipy.special  # noqa: F401
        import scipy.stats  # noqa: F401
        import sklearn  # noqa: F401
        import sklearn.calibration  # noqa: F401
        import sklearn.ensemble  # noqa: F401
        import sklearn.impute  # noqa: F401
        import sklearn.linear_model  # noqa: F401
        import sklearn.metrics  # noqa: F401
        import sklearn.model_selection  # noqa: F401
        import sklearn.naive_bayes  # noqa: F401
        import sklearn.neighbors  # noqa: F401
        import sklearn.pipeline  # noqa: F401
        import sklearn.preprocessing  # noqa: F401
        import sklearn.tree  # noqa: F401
        import lightgbm  # noqa: F401
        import xgboost  # noqa: F401
        import clinical_models  # noqa: F401
        import clinical_toolkit  # noqa: F401

        _install_redacted_benchmarks_stub()
        sys.modules.pop("evaluator", None)
        sys.modules.pop("__main__", None)

        init_msg = _read_json_line(cmd_read_fd, float(timeout_seconds))
        code_str = str(init_msg.get("code", ""))

        candidate_mod = types.ModuleType("candidate_program")
        sys.modules["candidate_program"] = candidate_mod

        # Install audit hook blocking file I/O, subprocesses, sockets, and frame introspection
        _blocked_prefixes = (
            "subprocess.",
            "os.system",
            "os.spawn",
            "os.exec",
            "os.fork",
            "os.popen",
            "os.kill",
            "os.remove",
            "os.unlink",
            "os.rmdir",
            "os.mkdir",
            "os.rename",
            "os.chmod",
            "os.chown",
            "os.putenv",
            "os.unsetenv",
            "socket.",
            "urllib.",
            "http.",
            "ftplib.",
            "smtplib.",
            " telnetlib.",
            "webbrowser.",
            "shutil.",
            "tempfile.",
            "multiprocessing.",
            " sys._getframe",
            "sys._current_frames",
        )

        # Read-only opens of import files under the interpreter's stdlib/site-packages are allowed. Why: run 1 lost 7/85
        # candidates to lazy submodule imports (sklearn.decomposition, scipy.sparse, ...) that we hadn't pre-imported.
        # The roots are site-packages/stdlib only, never the repo, so `src/`, `data/` and `.env` stay unreadable.
        import sysconfig

        _lib_roots = tuple(
            {os.path.realpath(sysconfig.get_paths()[k]) + os.sep for k in ("stdlib", "platstdlib", "purelib", "platlib")}
        )
        _import_suffixes = (".py", ".pyc", ".so")

        def _audit_hook(event: str, args: tuple) -> None:
            if event == "open":
                path_arg = str(args[0]) if args else ""
                mode_arg = str(args[1]) if len(args) > 1 else "r"
                # Block all writes or opens outside basic /proc or libgomp runtime reads
                if any(m in mode_arg for m in ("w", "a", "+", "x")):
                    raise PermissionError(f"Sandbox blocked file write: {os.path.basename(path_arg)}")
                real = os.path.realpath(path_arg)  # defeats `site-packages/../../repo/src/x.py` and symlinks
                is_lib_import = real.startswith(_lib_roots) and real.endswith(_import_suffixes)
                if not is_lib_import and not path_arg.startswith(("/proc/", "/sys/devices/", "/etc/ld.so")):
                    raise PermissionError(f"Sandbox blocked file open: {os.path.basename(path_arg)}")
            elif event in ("sys._getframe", "sys._current_frames"):
                raise PermissionError(f"Sandbox blocked frame introspection: {event}")
            elif any(event.startswith(pfx.strip()) for pfx in _blocked_prefixes):
                raise PermissionError(f"Sandbox blocked restricted system event: {event}")

        sys.addaudithook(_audit_hook)

        try:
            compiled = compile(code_str, "candidate_program.py", "exec")
            exec(compiled, candidate_mod.__dict__)  # noqa: S102
        except PermissionError as pe:
            _write_json_line(resp_write_fd, {"status": "sandbox_error", "error": _sanitize_line(str(pe))})
            return
        except Exception as e:
            tb = _sanitize_line(traceback.format_exc().splitlines()[-1])
            _write_json_line(resp_write_fd, {"status": "compile_error", "error": f"{type(e).__name__}: {e} ({tb})"})
            return

        mode = str(init_msg.get("mode", "clinical"))
        fit_fn = getattr(candidate_mod, "fit_and_score_risk", None)
        alloc_fn = getattr(candidate_mod, "allocate_interventions", None)
        if mode == "tabular":
            missing_fn = None if callable(fit_fn) else "Candidate must define callable `fit_and_score_risk`"
        else:
            missing_fn = (
                None
                if callable(fit_fn) and callable(alloc_fn)
                else "Candidate must define callable `fit_and_score_risk` and `allocate_interventions`"
            )
        if missing_fn:
            _write_json_line(resp_write_fd, {"status": "compile_error", "error": missing_fn})
            return

        _write_json_line(resp_write_fd, {"status": "ready"})

        while True:
            msg = _read_json_line(cmd_read_fd, float(timeout_seconds))
            cmd = msg.get("cmd")
            if cmd == "stop":
                break
            elif cmd == "tabular_round":
                task = _task_from_payload(msg["task"])
                train_cols = _payload_to_columns(msg["train"], task)
                y_train = np.asarray(msg["y_train"], dtype=np.int64)
                eval_cols = _payload_to_columns(msg["eval"], task)
                try:
                    raw = fit_fn(task, train_cols, y_train, eval_cols)
                    probs = np.asarray(raw, dtype=np.float64).ravel()
                    if not np.all(np.isfinite(probs)):
                        raise ValueError("fit_and_score_risk returned non-finite probabilities")
                    _write_json_line(resp_write_fd, {"status": "ok", "probs": probs.tolist()})
                except PermissionError as pe:
                    _write_json_line(resp_write_fd, {"status": "sandbox_error", "error": _sanitize_line(str(pe))})
                    return
                except Exception as e:
                    _write_json_line(
                        resp_write_fd,
                        {"status": "runtime_error", "error": _sanitize_line(f"{type(e).__name__}: {e}")},
                    )
            elif cmd == "stage1":
                spec = _deserialize_cohort_spec(msg["cohort_spec"])
                train_recs = [_deserialize_patient(r) for r in msg["train_records"]]
                eval_recs = [_deserialize_patient(r) for r in msg["eval_records"]]
                try:
                    raw_risks = fit_fn(spec, train_recs, eval_recs)
                    if not isinstance(raw_risks, Mapping):
                        raise TypeError(f"fit_and_score_risk must return dict, got {type(raw_risks).__name__}")
                    clean_risks: Dict[str, float] = {}
                    for k, v in raw_risks.items():
                        fv = float(v)
                        if not math.isfinite(fv):
                            raise ValueError(f"Non-finite risk score for {k}")
                        clean_risks[str(k)] = fv
                    _write_json_line(resp_write_fd, {"status": "ok", "risks": clean_risks})
                except PermissionError as pe:
                    _write_json_line(resp_write_fd, {"status": "sandbox_error", "error": _sanitize_line(str(pe))})
                    return
                except Exception as e:
                    _write_json_line(
                        resp_write_fd,
                        {
                            "status": "runtime_error",
                            "error": _sanitize_line(f"{type(e).__name__}: {e}"),
                        },
                    )
            elif cmd == "stage2":
                spec = _deserialize_cohort_spec(msg["cohort_spec"])
                eval_recs = [_deserialize_patient(r) for r in msg["eval_records"]]
                risks = {str(k): float(v) for k, v in msg["predicted_risks"].items()}
                try:
                    raw_plan = alloc_fn(spec, eval_recs, risks)
                    if isinstance(raw_plan, CareAllocationPlan):
                        assign_map = {
                            str(k): (v.value if hasattr(v, "value") else str(v))
                            for k, v in raw_plan.assignments.items()
                        }
                    elif isinstance(raw_plan, Mapping):
                        assign_map = {
                            str(k): (v.value if hasattr(v, "value") else str(v))
                            for k, v in raw_plan.items()
                        }
                    else:
                        raise TypeError(
                            f"allocate_interventions must return CareAllocationPlan, got {type(raw_plan).__name__}"
                        )
                    _write_json_line(resp_write_fd, {"status": "ok", "assignments": assign_map})
                except PermissionError as pe:
                    _write_json_line(resp_write_fd, {"status": "sandbox_error", "error": _sanitize_line(str(pe))})
                    return
                except Exception as e:
                    _write_json_line(
                        resp_write_fd,
                        {
                            "status": "runtime_error",
                            "error": _sanitize_line(f"{type(e).__name__}: {e}"),
                        },
                    )
    except PermissionError as pe:
        try:
            _write_json_line(resp_write_fd, {"status": "sandbox_error", "error": _sanitize_line(str(pe))})
        except Exception:
            pass
    except Exception as e:
        try:
            _write_json_line(
                resp_write_fd,
                {"status": "fatal_error", "error": _sanitize_line(f"{type(e).__name__}: {e}")},
            )
        except Exception:
            pass


def _build_insights(
    problem_id: str,
    scenario_results: Sequence[Dict[str, Any]],
    runtime_errors: Sequence[str],
    elapsed_ms: float,
) -> Tuple[float, List[str]]:
    """Aggregates scenario scores and formats structured LLM + executive insights."""
    n = len(scenario_results)
    if n == 0:
        return -1e6, ["No benchmark scenarios evaluated."]

    mean_score = sum(float(r["total_score"]) for r in scenario_results) / n
    mean_auc = sum(float(r["roc_auc"]) for r in scenario_results) / n
    mean_base_auc = sum(float(r["baseline_roc_auc"]) for r in scenario_results) / n
    mean_uplift = mean_auc - mean_base_auc
    mean_pr_auc = sum(float(r["pr_auc"]) for r in scenario_results) / n
    mean_sens = sum(float(r["sens_at_spec"]) for r in scenario_results) / n
    mean_brier = sum(float(r["brier_score"]) for r in scenario_results) / n
    target_auc = float(scenario_results[0]["target_auc"])
    total_net_savings = sum(float(r["net_savings_usd"]) for r in scenario_results)
    mean_actuarial_roi = sum(float(r["actuarial_auc_roi_usd"]) for r in scenario_results) / n
    total_prevented = sum(float(r["prevented_events"]) for r in scenario_results)
    total_violations = sum(int(r["hard_violation_count"]) for r in scenario_results)
    feasible_count = sum(1 for r in scenario_results if r["is_feasible"])
    target_pass_count = sum(1 for r in scenario_results if r["meets_target_auc"])

    gate_str = "PASS" if mean_auc >= target_auc else "BELOW_TARGET"
    insights: List[str] = [
        _sanitize_line(
            f"[{problem_id}] Score={mean_score:,.2f} | Feasible={feasible_count}/{n} (Violations={total_violations}) | "
            f"ROC-AUC={mean_auc:.4f} vs Base={mean_base_auc:.4f} ({mean_uplift:+.4f}, Target>{target_auc:.2f}:{gate_str})"
        ),
        _sanitize_line(
            f"Clinical Metrics: PR-AUC={mean_pr_auc:.4f} | Sens@Spec80={mean_sens:.4f} | Brier={mean_brier:.4f} | "
            f"TargetGate={target_pass_count}/{n} scenarios | EvalTime={elapsed_ms:.0f}ms"
        ),
        _sanitize_line(
            f"Financial Value Impact: Actuarial Lift Value=${mean_actuarial_roi:,.0f} | "
            f"Stage 2 Care Net ROI=${total_net_savings:,.0f} ({total_prevented:.1f} events prevented)"
        ),
    ]

    for r in scenario_results:
        status_tag = "OK" if r["is_feasible"] else f"VIOL({r['hard_violation_count']})"
        insights.append(
            _sanitize_line(
                f"  {r['scenario_id']} ({status_tag}): Score={r['total_score']:,.1f} | "
                f"AUC={r['roc_auc']:.4f} ({r['auc_uplift']:+.4f} vs base) | "
                f"PR-AUC={r['pr_auc']:.4f} | Sens={r['sens_at_spec']:.3f} | "
                f"NetROI=${r['net_savings_usd']:,.0f}"
            )
        )
        for v in r.get("violations", [])[:2]:
            insights.append(_sanitize_line(f"    ! Violation [{r['scenario_id']}]: {v}"))

    if runtime_errors:
        insights.append(_sanitize_line(f"RUNTIME ERRORS ({len(runtime_errors)} isolated call failures):"))
        for err in runtime_errors[:4]:
            insights.append(_sanitize_line(f"  * {err}"))

    return float(mean_score), insights


def _resolve_instances(
    problem_id: str,
    split: str,
    perturb: bool,
    perturb_seed: Optional[int],
    heldout_seed: Optional[int],
) -> List[Any]:
    if split == "test":
        return build_heldout_benchmark_instances(problem_id, heldout_seed=heldout_seed)
    return build_benchmark_instances(problem_id, perturb=perturb, perturb_seed=perturb_seed)


class _CandidateRejected(Exception):
    """Candidate failed to initialise or breached the sandbox; the whole program scores `None`."""

    def __init__(self, insight: str) -> None:
        super().__init__(insight)
        self.insight = insight


@contextlib.contextmanager
def _sandbox_worker(
    code: str,
    timeout_seconds: int,
    *,
    mode: str = "clinical",
    cpu_limit_seconds: Optional[int] = None,
) -> Iterator[Callable[[Mapping[str, Any]], Dict[str, Any]]]:
    """Spawns the isolated `python -P -s` worker, loads `code`, and yields `request(payload) -> response`.

    Shared by the synthetic two-stage track and the real-data tabular track so the sandbox lifecycle (clean env,
    fd hygiene, kill-on-exit) lives in one place. Raises `_CandidateRejected` on init failure or any sandbox
    PermissionError; lets `TimeoutError`/`EOFError` propagate. Timing stays on the parent clock.
    """
    p2c_read, p2c_write = os.pipe()
    c2p_read, c2p_write = os.pipe()
    os.set_inheritable(p2c_read, True)
    os.set_inheritable(c2p_write, True)

    # Clean environment: strip PYTHON* and private seeds, enforce PYTHONHASHSEED=0
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("PYTHON") and k not in ("AE_PERTURB_SEED", "AE_HELDOUT_SEED")
    }
    clean_env["PYTHONHASHSEED"] = "0"
    clean_env["OMP_NUM_THREADS"] = "1"
    clean_env["OPENBLAS_NUM_THREADS"] = "1"
    clean_env["MKL_NUM_THREADS"] = "1"

    worker_boot = (
        f"import sys; sys.path.insert(0, {repr(str(HERE))}); "
        f"import evaluator; evaluator._worker_main({p2c_read}, {c2p_write}, {int(timeout_seconds)}, "
        f"{repr(int(cpu_limit_seconds) if cpu_limit_seconds else None)})"
    )

    proc: Optional[subprocess.Popen[bytes]] = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-P", "-s", "-c", worker_boot],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            # DEVNULL, not PIPE: stderr was never drained, so a chatty candidate (e.g. LightGBM without verbose=-1)
            # could fill the 64KB pipe buffer, block the child, and surface as a spurious timeout.
            stderr=subprocess.DEVNULL,
            pass_fds=(p2c_read, c2p_write),
            env=clean_env,
            cwd=str(HERE),
        )
        os.close(p2c_read)
        p2c_read = -1
        os.close(c2p_write)
        c2p_write = -1

        _write_json_line(p2c_write, {"code": code, "mode": mode})
        ready_msg = _read_json_line(c2p_read, float(timeout_seconds))
        status = ready_msg.get("status")
        if status == "sandbox_error":
            raise _CandidateRejected(f"Sandbox violation (PermissionError): {ready_msg.get('error')}")
        if status != "ready":
            raise _CandidateRejected(f"Candidate initialization failed: {ready_msg.get('error', status)}")

        def request(payload: Mapping[str, Any]) -> Dict[str, Any]:
            _write_json_line(p2c_write, payload)
            resp = _read_json_line(c2p_read, float(timeout_seconds))
            if resp.get("status") == "sandbox_error":
                raise _CandidateRejected(f"Sandbox violation (PermissionError): {resp.get('error')}")
            return resp

        yield request

        try:
            _write_json_line(p2c_write, {"cmd": "stop"})
        except Exception:
            pass
    finally:
        for fd in (p2c_read, p2c_write, c2p_read, c2p_write):
            if fd != -1:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=2)
            except Exception:
                pass


def _early_termination(e: BaseException) -> Dict[str, Any]:
    return {
        "score": None,
        "insights": [_sanitize_line(f"Worker evaluation terminated early: {type(e).__name__}: {e}")],
    }


def evaluate_in_process(
    code: str,
    *,
    problem_id: str = "readmission_30d",
    split: str = "all",
    perturb: bool = False,
    perturb_seed: Optional[int] = None,
    heldout_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Fast in-process reference scorer for trusted local tests and simulator visualization."""
    if is_tabular_problem(problem_id):
        return evaluate_tabular_in_process(
            code, problem_id=problem_id, split=split, perturb=perturb, perturb_seed=perturb_seed,
            heldout_seed=heldout_seed,
        )
    t0 = time.monotonic()
    mod = types.ModuleType("trusted_candidate")
    exec(compile(code, "trusted_candidate.py", "exec"), mod.__dict__)  # noqa: S102
    fit_fn = getattr(mod, "fit_and_score_risk")
    alloc_fn = getattr(mod, "allocate_interventions")

    instances = _resolve_instances(problem_id, split, perturb, perturb_seed, heldout_seed)
    scenario_results: List[Dict[str, Any]] = []
    runtime_errors: List[str] = []

    for inst in instances:
        masked_eval = [
            dataclasses.replace(p, outcome_label=None, time_to_event_days=None)
            for p in inst.eval_records
        ]
        stage_violations: List[str] = []
        try:
            risks = fit_fn(inst.cohort_spec, list(inst.train_records), masked_eval)
        except Exception as e:
            runtime_errors.append(f"{inst.scenario_id} Stage1 {type(e).__name__}: {e}")
            stage_violations.append(f"Stage 1 runtime exception: {type(e).__name__}: {e}")
            risks = {p.patient_id: p.baseline_api_prob for p in masked_eval}

        try:
            plan = alloc_fn(inst.cohort_spec, masked_eval, dict(risks))
        except Exception as e:
            runtime_errors.append(f"{inst.scenario_id} Stage2 {type(e).__name__}: {e}")
            stage_violations.append(f"Stage 2 runtime exception: {type(e).__name__}: {e}")
            plan = CareAllocationPlan(
                scenario_id=inst.scenario_id,
                assignments={p.patient_id: InterventionTier.NONE for p in masked_eval},
            )

        res = evaluate_cohort_predictions_and_allocation(
            inst.cohort_spec, inst.eval_records, risks, plan, DEFAULT_WEIGHTS
        )
        if stage_violations:
            res["violations"] = stage_violations + list(res["violations"])
            res["hard_violation_count"] += len(stage_violations)
            res["is_feasible"] = False
            res["total_score"] -= len(stage_violations) * DEFAULT_WEIGHTS.hard_violation_penalty
        scenario_results.append(res)

    elapsed_ms = (time.monotonic() - t0) * 1000.0
    score, insights = _build_insights(problem_id, scenario_results, runtime_errors, elapsed_ms)
    return {
        "score": score,
        "insights": insights,
        "scenarios": scenario_results,
    }


def evaluate_program(
    code: str,
    timeout_seconds: int = 45,
    *,
    problem_id: str = "readmission_30d",
    split: str = "all",
    perturb: bool = True,
    perturb_seed: Optional[int] = None,
    heldout_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Sandboxed evaluation of untrusted candidate `code` in an isolated subprocess.

    Dispatches to the real-data tabular track when the problem config declares `task_type: tabular_classification`;
    otherwise runs the synthetic two-phase (risk, then allocation) protocol.
    """
    if is_tabular_problem(problem_id):
        return evaluate_tabular_program(
            code, timeout_seconds, problem_id=problem_id, split=split, perturb=perturb,
            perturb_seed=perturb_seed, heldout_seed=heldout_seed,
        )
    t0 = time.monotonic()
    instances = _resolve_instances(problem_id, split, perturb, perturb_seed, heldout_seed)

    try:
        with _sandbox_worker(code, timeout_seconds) as request:
            scenario_results: List[Dict[str, Any]] = []
            runtime_errors: List[str] = []

            for inst in instances:
                spec_payload = _serialize_cohort_spec(inst.cohort_spec)
                train_payload = [_serialize_patient(p, include_labels=True) for p in inst.train_records]
                eval_payload = [_serialize_patient(p, include_labels=False) for p in inst.eval_records]

                stage_violations: List[str] = []

                # --- Phase 1: Fit on train_records, predict probabilities on label-stripped eval_records ---
                s1 = request(
                    {
                        "cmd": "stage1",
                        "cohort_spec": spec_payload,
                        "train_records": train_payload,
                        "eval_records": eval_payload,
                    }
                )
                if s1.get("status") == "ok":
                    risks = {str(k): float(v) for k, v in s1.get("risks", {}).items()}
                else:
                    err_txt = str(s1.get("error", "Stage 1 failed"))
                    runtime_errors.append(f"[{inst.scenario_id} Stage 1] {err_txt}")
                    stage_violations.append(f"Stage 1 runtime error: {err_txt}")
                    risks = {p.patient_id: float(p.baseline_api_prob) for p in inst.eval_records}

                # --- Phase 2: Send committed risks, receive CareAllocationPlan ---
                s2 = request(
                    {
                        "cmd": "stage2",
                        "cohort_spec": spec_payload,
                        "eval_records": eval_payload,
                        "predicted_risks": risks,
                    }
                )
                if s2.get("status") == "ok":
                    raw_assign = s2.get("assignments", {})
                    parsed_assign: Dict[str, InterventionTier] = {}
                    for pid, t_str in raw_assign.items():
                        try:
                            parsed_assign[str(pid)] = InterventionTier(str(t_str))
                        except ValueError:
                            stage_violations.append(f"Invalid InterventionTier '{t_str}' for {pid}")
                            parsed_assign[str(pid)] = InterventionTier.NONE
                    plan = CareAllocationPlan(scenario_id=inst.scenario_id, assignments=parsed_assign)
                else:
                    err_txt = str(s2.get("error", "Stage 2 failed"))
                    runtime_errors.append(f"[{inst.scenario_id} Stage 2] {err_txt}")
                    stage_violations.append(f"Stage 2 runtime error: {err_txt}")
                    plan = CareAllocationPlan(
                        scenario_id=inst.scenario_id,
                        assignments={p.patient_id: InterventionTier.NONE for p in inst.eval_records},
                    )

                res = evaluate_cohort_predictions_and_allocation(
                    inst.cohort_spec, inst.eval_records, risks, plan, DEFAULT_WEIGHTS
                )
                if stage_violations:
                    res["violations"] = stage_violations + list(res["violations"])
                    res["hard_violation_count"] += len(stage_violations)
                    res["is_feasible"] = False
                    res["total_score"] -= len(stage_violations) * DEFAULT_WEIGHTS.hard_violation_penalty
                scenario_results.append(res)

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        score, insights = _build_insights(problem_id, scenario_results, runtime_errors, elapsed_ms)
        return {"score": score, "insights": insights}
    except _CandidateRejected as e:
        return {"score": None, "insights": [e.insight]}
    except (TimeoutError, EOFError) as e:
        return _early_termination(e)


# ---------------------------------------------------------------------------------------------------------------------
# Real-data tabular track
# ---------------------------------------------------------------------------------------------------------------------


def _round_payload(rd: TabularRound) -> Dict[str, Any]:
    """Only the schema, feature columns and *training* labels cross the pipe; `y_eval` and group IDs never do."""
    return {
        "cmd": "tabular_round",
        "task": dataclasses.asdict(rd.task),
        "train": _columns_to_payload(rd.train, rd.task),
        "y_train": rd.y_train.tolist(),
        "eval": _columns_to_payload(rd.eval, rd.task),
    }


def _check_round_probs(rd: TabularRound, raw: Optional[Sequence[float]]) -> Tuple[np.ndarray, List[str]]:
    """Trusted-side validation. Invalid output costs one violation and falls back to a constant (AUC 0.5) score."""
    fallback = np.full(rd.task.n_eval, float(rd.y_train.mean()) if rd.y_train.size else 0.5)
    if raw is None:
        return fallback, []
    probs = np.asarray(raw, dtype=np.float64).ravel()
    if probs.shape[0] != rd.task.n_eval:
        return fallback, [f"{rd.task.round_id}: returned {probs.shape[0]} predictions for {rd.task.n_eval} eval rows"]
    if not np.all(np.isfinite(probs)) or probs.min() < 0.0 or probs.max() > 1.0:
        return np.clip(np.nan_to_num(probs, nan=0.5), 0.0, 1.0), [
            f"{rd.task.round_id}: probabilities outside [0, 1] or non-finite"
        ]
    return probs, []


def _subgroup_insights(problem_id: str, rounds: Sequence[TabularRound], probs: Sequence[np.ndarray]) -> List[str]:
    """Worst/best subgroup AUC pooled over rounds (groups need >=30 positives and negatives to be reported)."""
    y = np.concatenate([rd.y_eval for rd in rounds])
    p = np.concatenate(list(probs))
    lines: List[str] = []
    for col in subgroup_columns(problem_id):
        if col not in rounds[0].eval:
            continue
        vals = np.concatenate([np.asarray(rd.eval[col], dtype=object) for rd in rounds])
        aucs: Dict[str, float] = {}
        for v in sorted({x for x in vals.tolist() if x is not None}, key=str):
            m = vals == v
            n_pos = int(y[m].sum())
            if n_pos >= 30 and int(m.sum()) - n_pos >= 30:
                aucs[str(v)] = compute_roc_auc(y[m], p[m])
        if len(aucs) >= 2:
            worst = min(aucs, key=aucs.get)
            best = max(aucs, key=aucs.get)
            lines.append(
                _sanitize_line(
                    f"Subgroup AUC [{col}]: worst={worst}:{aucs[worst]:.3f} best={best}:{aucs[best]:.3f} "
                    f"gap={aucs[best] - aucs[worst]:.3f}"
                )
            )
    return lines


def _score_tabular(
    problem_id: str,
    rounds: Sequence[TabularRound],
    outputs: Sequence[Optional[Sequence[float]]],
    errors: Sequence[Optional[str]],
    elapsed_ms: float,
    keep_predictions: bool = False,
) -> Dict[str, Any]:
    """Fitness = predictive_scale x mean round ROC-AUC - hard_violation_penalty x violations.

    Why a single primary metric: the synthetic track's 4-metric blend is prevalence-sensitive and hard to defend to
    stakeholders. AUC is the agreed headline. PR-AUC, Brier, calibration-in-the-large and subgroup gaps are reported
    in insights so the LLM can still see them.
    """
    w = DEFAULT_WEIGHTS
    results: List[Dict[str, Any]] = []
    all_probs: List[np.ndarray] = []
    for rd, raw, err in zip(rounds, outputs, errors):
        violations = [f"{rd.task.round_id}: runtime error: {err}"] if err else []
        probs, invalid = _check_round_probs(rd, raw)
        violations += invalid
        all_probs.append(probs)
        res: Dict[str, Any] = {
            "round_id": rd.task.round_id,
            "n_train": rd.task.n_train,
            "n_eval": rd.task.n_eval,
            "roc_auc": compute_roc_auc(rd.y_eval, probs),
            "pr_auc": compute_pr_auc(rd.y_eval.tolist(), probs.tolist()),
            "brier": compute_brier_score(rd.y_eval.tolist(), probs.tolist()),
            "mean_pred": float(probs.mean()) if probs.size else 0.0,
            "prevalence": float(rd.y_eval.mean()) if rd.y_eval.size else 0.0,
            "violations": violations,
        }
        if keep_predictions:
            res["probs"] = probs
            res["y_eval"] = rd.y_eval
        results.append(res)

    n = len(results)
    aucs = np.array([r["roc_auc"] for r in results], dtype=np.float64)
    mean_auc = float(aucs.mean())
    sd_auc = float(aucs.std(ddof=1)) if n > 1 else 0.0
    n_viol = sum(len(r["violations"]) for r in results)
    n_ok = sum(1 for r in results if not r["violations"])
    score = w.predictive_scale * mean_auc - n_viol * w.hard_violation_penalty
    mean = lambda key: float(np.mean([r[key] for r in results]))  # noqa: E731

    insights = [
        _sanitize_line(
            f"[{problem_id}] Score={score:,.2f} | Feasible={n_ok}/{n} (Violations={n_viol}) | "
            f"ROC-AUC={mean_auc:.4f} ± {sd_auc:.4f} over {n} rounds (n_eval={sum(r['n_eval'] for r in results):,})"
        ),
        _sanitize_line(
            f"Metrics: PR-AUC={mean('pr_auc'):.4f} | Brier={mean('brier'):.4f} | "
            f"MeanPred={mean('mean_pred'):.3f} vs Prevalence={mean('prevalence'):.3f} | EvalTime={elapsed_ms:.0f}ms"
        ),
    ]
    for r in results:
        insights.append(
            _sanitize_line(
                f"  {r['round_id']}: AUC={r['roc_auc']:.4f} | PR-AUC={r['pr_auc']:.4f} | Brier={r['brier']:.4f} | "
                f"n_train={r['n_train']:,} n_eval={r['n_eval']:,}"
            )
        )
        for v in r["violations"][:2]:
            insights.append(_sanitize_line(f"    ! Violation: {v}"))
    insights.extend(_subgroup_insights(problem_id, rounds, all_probs))
    errs = [f"[{rd.task.round_id}] {e}" for rd, e in zip(rounds, errors) if e]
    if errs:
        insights.append(_sanitize_line(f"RUNTIME ERRORS ({len(errs)} isolated round failures):"))
        insights.extend(_sanitize_line(f"  * {e}") for e in errs[:4])
    return {"score": float(score), "insights": insights, "rounds": results}


def evaluate_tabular_program(
    code: str,
    timeout_seconds: int = 45,
    *,
    problem_id: str,
    split: str = "all",
    perturb: bool = True,
    perturb_seed: Optional[int] = None,
    heldout_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Sandboxed real-data evaluation: one `fit_and_score_risk` call per round inside the isolated worker."""
    t0 = time.monotonic()
    timeout = int(load_problem_config(problem_id).get("timeout_seconds", timeout_seconds))
    rounds = build_tabular_rounds(
        problem_id, split=split, perturb=perturb, perturb_seed=perturb_seed, heldout_seed=heldout_seed
    )
    outputs: List[Optional[Sequence[float]]] = []
    errors: List[Optional[str]] = []
    try:
        with _sandbox_worker(code, timeout, mode="tabular", cpu_limit_seconds=timeout * (len(rounds) + 1)) as request:
            for rd in rounds:
                resp = request(_round_payload(rd))
                ok = resp.get("status") == "ok"
                outputs.append(resp.get("probs") if ok else None)
                errors.append(None if ok else str(resp.get("error", "round failed")))
    except _CandidateRejected as e:
        return {"score": None, "insights": [e.insight]}
    except (TimeoutError, EOFError) as e:
        return _early_termination(e)
    res = _score_tabular(problem_id, rounds, outputs, errors, (time.monotonic() - t0) * 1000.0)
    return {"score": res["score"], "insights": res["insights"]}


def evaluate_tabular_in_process(
    program: Union[str, Callable[..., Sequence[float]]],
    *,
    problem_id: str,
    split: str = "all",
    perturb: bool = False,
    perturb_seed: Optional[int] = None,
    heldout_seed: Optional[int] = None,
    keep_predictions: bool = False,
) -> Dict[str, Any]:
    """Trusted in-process scorer (reports, baseline tuning, tests). Accepts source code or a `fit_and_score_risk`
    callable. Never use it for unreviewed LLM output: there is no sandbox here."""
    t0 = time.monotonic()
    if callable(program):
        fit_fn = program
    else:
        mod = types.ModuleType("trusted_candidate")
        exec(compile(program, "trusted_candidate.py", "exec"), mod.__dict__)  # noqa: S102
        fit_fn = getattr(mod, "fit_and_score_risk")
    rounds = build_tabular_rounds(
        problem_id, split=split, perturb=perturb, perturb_seed=perturb_seed, heldout_seed=heldout_seed
    )
    outputs: List[Optional[Sequence[float]]] = []
    errors: List[Optional[str]] = []
    for rd in rounds:
        try:
            outputs.append(fit_fn(rd.task, rd.train, rd.y_train, rd.eval))
            errors.append(None)
        except Exception as e:
            outputs.append(None)
            errors.append(f"{type(e).__name__}: {e}")
    return _score_tabular(problem_id, rounds, outputs, errors, (time.monotonic() - t0) * 1000.0, keep_predictions)


def _infer_problem_id_from_dir(program_dir: pathlib.Path) -> str:
    cfg_file = program_dir / "problem_config.json"
    if cfg_file.is_file():
        try:
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
            if data.get("problem_id"):
                return str(data["problem_id"])
        except Exception:
            pass
    return program_dir.name if program_dir.name != "problem" else "readmission_30d"


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaEvolve Clinical Portfolio Evaluator Harness")
    parser.add_argument("program_path", nargs="?", help="Optional direct path to candidate .py file")
    parser.add_argument("--program-dir", type=pathlib.Path, help="Problem directory containing initial_program.py")
    parser.add_argument("--problem", default=None, help="Problem ID (e.g. readmission_30d, sepsis_90d, chf_30d, inpatient_admission)")
    parser.add_argument("--output-file", type=pathlib.Path, help="Path to write JSON evaluation output")
    parser.add_argument("--split", choices=["all", "test"], default="all", help="Benchmark split to evaluate")
    parser.add_argument("--no-perturb", action="store_true", help="Evaluate canonical unperturbed scenarios")
    parser.add_argument("--timeout", type=int, default=45, help="Worker timeout in seconds")
    args = parser.parse_args()

    if args.program_path:
        prog_file = pathlib.Path(args.program_path).resolve()
        problem_id = args.problem or (
            _infer_problem_id_from_dir(prog_file.parent)
            if (prog_file.parent / "problem_config.json").is_file()
            else "readmission_30d"
        )
    elif args.program_dir:
        pdir = args.program_dir.resolve()
        prog_file = pdir / "initial_program.py"
        problem_id = args.problem or _infer_problem_id_from_dir(pdir)
    else:
        problem_id = args.problem or "readmission_30d"
        prog_file = (HERE.parent / "problems" / problem_id / "initial_program.py").resolve()

    code = prog_file.read_text(encoding="utf-8")
    result = evaluate_program(
        code,
        timeout_seconds=args.timeout,
        problem_id=problem_id,
        split=args.split,
        perturb=not args.no_perturb,
    )
    output_json = json.dumps(result, indent=2, allow_nan=False)
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(output_json + "\n", encoding="utf-8")
    print(output_json)


if __name__ == "__main__":
    main()
