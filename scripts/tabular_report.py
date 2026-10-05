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

"""Baseline arms + locked-test report for real-data tabular problems (e.g. `diabetes130_readmit30`).

Why this exists: "evolved beats seed" is not a credible headline when the seed is weak. This script:

1. **Tunes LightGBM and XGBoost baselines** by random search on exactly the same fitness rounds AlphaEvolve optimises
   (`split="all"`, perturbed with `AE_PERTURB_SEED` from the environment or `.env`). Trial 0 of the LightGBM
   search is the seed's own hyper-parameters, so the tuned arm is never worse than the seed on selection.
2. **Builds a human stack** (`StackArm`): domain features (comorbidities, medication changes, utilisation, out-of-fold
   target encoding) + the two tuned GBDTs stacked with a logistic meta-learner. AE run 1 rediscovered roughly this, so
   it is the fair bar for "did AE find something a careful data scientist wouldn't?".
3. **Scores every arm and the candidate programs once on the locked test set** (`split="test"`: trained on all
   training patients, scored on patients never used for fitness), and reports Δ AUC vs tuned LightGBM and vs the
   human stack with paired-bootstrap 95% CIs, plus the mean/SD of Δ across candidates (one finalist per run).

`--selection-only` stops after step 2's selection scores, so the bar is known before spending on AE.
Candidates run in-process (no sandbox), so pass only reviewed programs. The output JSON records the git commit but
never the private seeds.

Usage:
    .venv/bin/python scripts/tabular_report.py --problem diabetes130_readmit30 --jobs 12 \\
        --candidates artifacts/runs/diabetes130_readmit30/<exp>/rank1.py ...
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import os
import pathlib
import random
import subprocess
import sys
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
# Same as the sandbox: candidates with n_jobs=-1 otherwise grab every core. Must precede numpy/lightgbm imports.
os.environ.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})

import numpy as np  # noqa: E402

import evaluator  # noqa: E402
from clinical_metrics import paired_bootstrap_auc_diff  # noqa: E402
from clinical_toolkit import TabularEncoder, icd9_comorbidities, icd9_group, oof_target_encode  # noqa: E402
from setup_alphaevolve import load_dotenv  # noqa: E402

# Trial 0 of each search. LightGBM's entry mirrors the seed program's hyper-parameters, so the tuned arm is never
# worse than the seed on selection.
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "lightgbm": {  # Best of the 48-trial search (selection AUC 0.6646); the seed program uses exactly these.
        "n_estimators": 150, "learning_rate": 0.0158, "num_leaves": 31, "min_child_samples": 200, "subsample": 0.69,
        "colsample_bytree": 0.63, "reg_lambda": 0.022, "min_count": 100, "cat_smooth": 69.23,
    },
    "xgboost": {
        "n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 1.0, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_lambda": 1.0, "min_count": 20,
    },
}
SEED_PARAMS = DEFAULT_PARAMS["lightgbm"]


def _lgbm(params: Dict[str, Any]):
    import lightgbm as lgb

    return lgb.LGBMClassifier(subsample_freq=1, random_state=42, n_jobs=1, verbose=-1, **params)


def _xgb(params: Dict[str, Any]):
    import xgboost as xgb

    return xgb.XGBClassifier(tree_method="hist", random_state=42, n_jobs=1, verbosity=0, **params)


class LgbmArm:
    """Picklable `fit_and_score_risk` for one LightGBM hyper-parameter setting (same encoder as the seed)."""

    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = dict(params)

    def __call__(self, task, train, y_train, eval_cols) -> np.ndarray:
        p = dict(self.params)
        enc = TabularEncoder(min_count=int(p.pop("min_count")))
        x_tr = enc.fit_transform(task, train)
        x_ev = enc.transform(eval_cols)
        model = _lgbm(p)
        model.fit(x_tr, y_train, categorical_feature=enc.categorical_indices)
        return model.predict_proba(x_ev)[:, 1]


class XgbArm(LgbmArm):
    """Same encoder; XGBoost treats the category codes as ordinal numbers (NaN routed natively)."""

    def __call__(self, task, train, y_train, eval_cols) -> np.ndarray:
        p = dict(self.params)
        enc = TabularEncoder(min_count=int(p.pop("min_count")))
        model = _xgb(p).fit(enc.fit_transform(task, train), y_train)
        return model.predict_proba(enc.transform(eval_cols))[:, 1]


_MED_LEVELS = {"No", "Steady", "Up", "Down"}
_TE_COLUMNS = ("diag_1", "diag_2", "diag_3", "medical_specialty", "discharge_disposition_id",
               "admission_source_id", "payer_code")


def stack_features(task, train, y_train, eval_cols) -> tuple:
    """Domain features a careful human would add. Every statistic is fitted on `train`/`y_train` only.

    Comorbidity flags, ICD-9 chapters, medication Up/Down/Steady counts (medication columns are detected by their
    level set, not hard-coded), total prior utilisation, and out-of-fold target encodings of high-cardinality codes.
    Returns `(train_cols, eval_cols, extra_numeric, extra_categorical)`.
    """
    tr, ev = dict(train), dict(eval_cols)
    cats, nums = set(task.categorical_columns), set(task.numeric_columns)
    extra_num: List[str] = []
    extra_cat: List[str] = []

    def add(name: str, a: np.ndarray, b: np.ndarray, numeric: bool = True) -> None:
        tr[name], ev[name] = a, b
        (extra_num if numeric else extra_cat).append(name)

    diags = [c for c in ("diag_1", "diag_2", "diag_3") if c in cats]
    if diags:
        cm_tr, cm_ev = icd9_comorbidities(*[train[c] for c in diags]), icd9_comorbidities(*[eval_cols[c] for c in diags])
        for k in cm_tr:
            add(k, cm_tr[k], cm_ev[k])
        for c in diags:
            grp = lambda col: np.array([icd9_group(v) for v in col], dtype=object)  # noqa: E731
            add(f"grp_{c}", grp(train[c]), grp(eval_cols[c]), numeric=False)
    meds = [c for c in task.categorical_columns
            if {v for v in train[c].tolist() if v is not None} <= _MED_LEVELS and len(set(train[c].tolist())) > 1]
    for level in ("Up", "Down", "Steady"):
        count = lambda cols: np.sum([cols[m] == level for m in meds], axis=0).astype(float) if meds else (  # noqa: E731
            np.zeros(len(next(iter(cols.values())))))
        add(f"med_{level.lower()}", count(train), count(eval_cols))
    util = [c for c in ("number_outpatient", "number_emergency", "number_inpatient") if c in nums]
    if util:
        total = lambda cols: np.nansum([cols[c] for c in util], axis=0)  # noqa: E731
        add("util_total", total(train), total(eval_cols))
    for c in (c for c in _TE_COLUMNS if c in cats):
        a, b = oof_target_encode(train[c], y_train, eval_cols[c], n_folds=5, seed=0)
        add(f"te_{c}", a, b)
    return tr, ev, extra_num, extra_cat


class StackArm:
    """Human stack baseline: `stack_features` + LightGBM and XGBoost (tuned params), 5-fold out-of-fold predictions,
    logistic-regression meta-learner on their logits. Roughly what AE run 1 discovered, written by hand."""

    def __init__(self, lgb_params: Dict[str, Any], xgb_params: Dict[str, Any], n_folds: int = 5) -> None:
        self.lgb_params, self.xgb_params, self.n_folds = dict(lgb_params), dict(xgb_params), n_folds

    def __call__(self, task, train, y_train, eval_cols) -> np.ndarray:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        tr, ev, extra_num, extra_cat = stack_features(task, train, y_train, eval_cols)
        lp, xp = dict(self.lgb_params), dict(self.xgb_params)
        enc = TabularEncoder(min_count=int(lp.pop("min_count")))
        xp.pop("min_count")
        x_tr = enc.fit_transform(task, tr, extra_numeric=extra_num, extra_categorical=extra_cat)
        x_ev = enc.transform(ev)
        y = np.asarray(y_train)
        fits = (
            lambda xa, ya: _lgbm(lp).fit(xa, ya, categorical_feature=enc.categorical_indices),
            lambda xa, ya: _xgb(xp).fit(xa, ya),
        )
        oof, ev_pred = np.zeros((len(y), len(fits))), np.zeros((len(x_ev), len(fits)))
        for fit_idx, held_idx in StratifiedKFold(self.n_folds, shuffle=True, random_state=0).split(x_tr, y):
            for m, fit in enumerate(fits):
                oof[held_idx, m] = fit(x_tr[fit_idx], y[fit_idx]).predict_proba(x_tr[held_idx])[:, 1]
        for m, fit in enumerate(fits):
            ev_pred[:, m] = fit(x_tr, y).predict_proba(x_ev)[:, 1]
        logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
        meta = LogisticRegression(C=1.0, max_iter=1000).fit(logit(oof), y)
        return meta.predict_proba(logit(ev_pred))[:, 1]


def sample_params(rng: random.Random, kind: str = "lightgbm") -> Dict[str, Any]:
    log_u = lambda lo, hi: math.exp(rng.uniform(math.log(lo), math.log(hi)))  # noqa: E731
    shared = {
        "n_estimators": rng.choice([150, 250, 400, 600, 900]),
        "learning_rate": round(log_u(0.01, 0.1), 4),
        "subsample": round(rng.uniform(0.6, 1.0), 2),
        "colsample_bytree": round(rng.uniform(0.4, 1.0), 2),
        "reg_lambda": round(log_u(0.01, 30.0), 3),
        "min_count": rng.choice([5, 10, 20, 50, 100]),
    }
    if kind == "xgboost":
        return {**shared, "max_depth": rng.choice([3, 4, 5, 6, 8]), "min_child_weight": round(log_u(0.5, 50.0), 2)}
    return {**shared, "num_leaves": rng.choice([7, 15, 31, 63]),
            "min_child_samples": rng.choice([20, 50, 100, 200, 400]), "cat_smooth": round(log_u(1.0, 100.0), 2)}


ARM_TYPES = {"lightgbm": LgbmArm, "xgboost": XgbArm}


def _selection_auc(args: tuple) -> tuple:
    problem_id, arm = args
    res = evaluator.evaluate_tabular_in_process(arm, problem_id=problem_id, split="all", perturb=True)
    return float(np.mean([r["roc_auc"] for r in res["rounds"]])), res["score"]


def tune_baseline(problem_id: str, trials: int, jobs: int, kind: str = "lightgbm") -> Dict[str, Any]:
    rng = random.Random(0)
    grid = [DEFAULT_PARAMS[kind]] + [sample_params(rng, kind) for _ in range(max(0, trials - 1))]
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        scored = list(pool.map(_selection_auc, [(problem_id, ARM_TYPES[kind](g)) for g in grid]))
    results = sorted(((auc, g, score) for (auc, score), g in zip(scored, grid)), key=lambda r: -r[0])
    for auc, params, _ in results[:3]:
        print(f"  {kind} selection AUC={auc:.4f}  {params}")
    best_auc, best_params, best_score = results[0]
    return {"params": best_params, "selection_auc": best_auc, "selection_score": best_score, "trials": len(grid)}


def _locked(problem_id: str, program: Any) -> Dict[str, Any]:
    res = evaluator.evaluate_tabular_in_process(program, problem_id=problem_id, split="test", keep_predictions=True)
    (rd,) = res["rounds"]
    return {"auc": rd["roc_auc"], "pr_auc": rd["pr_auc"], "brier": rd["brier"], "probs": rd["probs"], "y": rd["y_eval"],
            "violations": rd["violations"], "insights": res["insights"]}


def arm_labels(paths: List[pathlib.Path]) -> List[str]:
    """File stems, prefixed with the parent dir when stems clash (e.g. `rank2.py` from two runs) so no arm is dropped."""
    stems = [p.stem for p in paths]
    return [f"{p.parent.name}/{p.stem}" if stems.count(p.stem) > 1 else p.stem for p in paths]


def _git_head() -> Optional[str]:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.stdout.strip() or None


REFS = {"tuned": "tuned_lightgbm", "stack": "human_stack"}  # JSON key prefix -> reference arm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problem", default="diabetes130_readmit30")
    ap.add_argument("--trials", type=int, default=48, help="Random-search trials per tuned GBDT baseline")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--candidates", nargs="*", default=[], type=pathlib.Path)
    ap.add_argument("--selection-only", action="store_true",
                    help="Tune and score baselines on the fitness rounds only; never touches the locked test")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    for k, v in load_dotenv(ROOT / ".env").items():
        if k in ("AE_PERTURB_SEED", "AE_HELDOUT_SEED") and v:
            os.environ.setdefault(k, v)

    print(f"[1/3] Tuning LightGBM and XGBoost on fitness rounds ({args.trials} trials each, {args.jobs} jobs)...")
    tuned = {kind: tune_baseline(args.problem, args.trials, args.jobs, kind) for kind in ARM_TYPES}
    stack = StackArm(tuned["lightgbm"]["params"], tuned["xgboost"]["params"])
    stack_auc, stack_score = _selection_auc((args.problem, stack))
    print(f"  human_stack selection AUC={stack_auc:.4f}")
    if args.selection_only:
        return

    print("[2/3] Scoring arms once on the locked test set...")
    seed_code = (ROOT / "problems" / args.problem / "initial_program.py").read_text(encoding="utf-8")
    arms: Dict[str, Dict[str, Any]] = {
        "tuned_lightgbm": _locked(args.problem, LgbmArm(tuned["lightgbm"]["params"])),
        "tuned_xgboost": _locked(args.problem, XgbArm(tuned["xgboost"]["params"])),
        "human_stack": _locked(args.problem, stack),
        "seed": _locked(args.problem, seed_code),
    }
    labels = arm_labels(args.candidates)
    for label, path in zip(labels, args.candidates):
        arms[label] = _locked(args.problem, path.read_text(encoding="utf-8"))

    print("[3/3] Paired bootstrap vs tuned LightGBM and the human stack...")
    rows: List[Dict[str, Any]] = []
    for name, arm in arms.items():
        row = {"arm": name, "test_auc": arm["auc"], "test_pr_auc": arm["pr_auc"], "test_brier": arm["brier"],
               "violations": arm["violations"]}
        for prefix, ref_name in REFS.items():
            ref = arms[ref_name]
            assert np.array_equal(arm["y"], ref["y"]), "locked-test rows must be identical across arms"
            cmp = (
                paired_bootstrap_auc_diff(arm["y"], arm["probs"], ref["probs"], n_boot=args.n_boot, seed=0)
                if name != ref_name
                else {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value_one_sided": 1.0}
            )
            row.update({f"vs_{prefix}_{k}": v for k, v in cmp.items()})
        rows.append(row)

    ref_y = arms["tuned_lightgbm"]["y"]
    n_test = int(ref_y.shape[0])
    print(f"\nLocked test: n={n_test:,} encounters, prevalence={float(ref_y.mean()):.3f}\n")
    print("| Arm | Test ROC-AUC | Δ vs tuned LightGBM [95% CI] | Δ vs human stack [95% CI] | PR-AUC | Brier |")
    print("|---|---|---|---|---|---|")
    ci = lambda r, p: f"{r[f'vs_{p}_delta']:+.4f} [{r[f'vs_{p}_ci_low']:+.4f}, {r[f'vs_{p}_ci_high']:+.4f}]"  # noqa: E731
    for r in rows:
        print(f"| {r['arm']} | {r['test_auc']:.4f} | {ci(r, 'tuned')} | {ci(r, 'stack')} | "
              f"{r['test_pr_auc']:.4f} | {r['test_brier']:.4f} |")

    # Across-run headline: with one finalist per run, the mean (and spread) of Δ is what generalises, not the max.
    cand_rows = [r for r in rows if r["arm"] in labels]
    summary = {
        prefix: {"mean_delta": float(np.mean(d)), "sd_delta": float(np.std(d, ddof=1)) if len(d) > 1 else 0.0,
                 "n": len(d), "n_ci_above_zero": int(sum(r[f"vs_{prefix}_ci_low"] > 0 for r in cand_rows))}
        for prefix in REFS
        for d in [[r[f"vs_{prefix}_delta"] for r in cand_rows]]
        if d
    }
    for prefix, s in summary.items():
        print(f"\nCandidates vs {REFS[prefix]}: mean Δ={s['mean_delta']:+.4f} (SD {s['sd_delta']:.4f}, n={s['n']}), "
              f"CI above 0 for {s['n_ci_above_zero']}/{s['n']}")

    report = {
        "problem_id": args.problem,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "private_seeds_from_env": bool(os.environ.get("AE_PERTURB_SEED")),
        "tuned_baseline": tuned["lightgbm"],
        "tuned_xgboost": tuned["xgboost"],
        "human_stack": {"selection_auc": stack_auc, "selection_score": stack_score},
        "locked_test_n": n_test,
        "arms": rows,
        "candidate_summary": summary,
    }
    out = args.out or ROOT / "artifacts" / "reports" / f"{args.problem}_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
