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

"""Behavioral tests for the real-data tabular classification track (Diabetes 130-US readmission).

Why: the synthetic portfolio can't measure AlphaEvolve's lift because the toolkit encodes the generator's latent risk
formula. These tests pin the guarantees the real-data track relies on: group-disjoint splits, a locked test set that
fitness never touches, labels never crossing into the sandbox, NaN-preserving encoding, and per-round crash isolation.
"""

from __future__ import annotations

import csv
import json
import pathlib
import random

import numpy as np
import pytest

import clinical_benchmarks
import evaluator
import tabular_benchmarks
from clinical_metrics import paired_bootstrap_auc_diff
from clinical_toolkit import TabularEncoder, icd9_group

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL_PROBLEM = "diabetes130_readmit30"
FIXTURE_PROBLEM = "tabular_fixture"
DIAGS = ["428", "250.83", "414", "V57", "786", "?", "996", "585"]


def _write_fixture(root: pathlib.Path, n_patients: int = 2500) -> dict:
    """Writes a small Diabetes-130-shaped CSV with real signal plus a problem config under `root/problems`."""
    rng = random.Random(7)
    rows = []
    enc = 0
    for pid in range(n_patients):
        for _ in range(rng.choice([1, 2, 3])):
            enc += 1
            n_inp = min(6, int(rng.expovariate(0.9)))
            diag = rng.choice(DIAGS)
            logit = -2.4 + 0.55 * n_inp + (0.8 if diag == "428" else 0.0)
            label = "<30" if rng.random() < 1 / (1 + np.exp(-logit)) else rng.choice([">30", "NO"])
            rows.append(
                {
                    "encounter_id": str(enc),
                    "patient_nbr": str(10_000 + pid),
                    "race": rng.choice(["Caucasian", "AfricanAmerican", "?"]),
                    "gender": rng.choice(["Female", "Male"]),
                    "age": rng.choice(["[50-60)", "[60-70)", "[70-80)"]),
                    "discharge_disposition_id": rng.choice(["1", "1", "1", "3", "11"]),
                    "time_in_hospital": str(rng.randint(1, 14)),
                    "number_inpatient": str(n_inp),
                    "diag_1": diag,
                    "readmitted": label,
                }
            )
    data_path = root / "fixture.csv"
    with data_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cfg = {
        "problem_id": FIXTURE_PROBLEM,
        "task_type": "tabular_classification",
        "title": "Fixture",
        "data_path": "fixture.csv",
        "label_column": "readmitted",
        "positive_values": ["<30"],
        "group_column": "patient_nbr",
        "id_columns": ["encounter_id"],
        "missing_values": ["?"],
        "exclude_rows": {"discharge_disposition_id": ["11"]},
        "numeric_columns": ["time_in_hospital", "number_inpatient"],
        "categorical_columns": ["race", "gender", "age", "discharge_disposition_id", "diag_1"],
        "subgroup_columns": ["gender"],
        "partition_salt": "fixture-v1",
        "test_fraction": 0.2,
        "selection_fraction": 0.3,
        "n_rounds": 3,
        "train_rows": 2000,  # tuned seed uses min_child_samples=200; 400 rows left it unable to split
        "timeout_seconds": 60,
    }
    pdir = root / "problems" / FIXTURE_PROBLEM
    pdir.mkdir(parents=True)
    (pdir / "problem_config.json").write_text(json.dumps(cfg))
    return cfg


@pytest.fixture()
def fixture_cfg(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    cfg = _write_fixture(tmp_path)
    monkeypatch.setattr(clinical_benchmarks, "PROBLEMS_ROOT", tmp_path / "problems")
    return cfg


def _seed_code() -> str:
    return (REPO_ROOT / "problems" / REAL_PROBLEM / "initial_program.py").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- data loading & splits


def test_load_problem_config_accepts_tabular_config_without_scenarios(fixture_cfg: dict) -> None:
    assert clinical_benchmarks.load_problem_config(FIXTURE_PROBLEM)["task_type"] == "tabular_classification"
    assert tabular_benchmarks.is_tabular_problem(FIXTURE_PROBLEM)
    assert not tabular_benchmarks.is_tabular_problem("readmission_30d")


def test_loader_maps_missing_to_none_excludes_rows_and_binarizes_label(fixture_cfg: dict) -> None:
    ds = tabular_benchmarks.load_tabular_dataset(fixture_cfg)
    assert "discharge_disposition_id" in ds.columns
    assert "11" not in set(ds.columns["discharge_disposition_id"].tolist())
    assert "?" not in set(ds.columns["race"].tolist()) and None in set(ds.columns["race"].tolist())
    assert ds.columns["number_inpatient"].dtype == np.float64
    # Label, group and ID columns never become features
    for leaked in ("readmitted", "patient_nbr", "encounter_id"):
        assert leaked not in ds.columns
    assert set(np.unique(ds.y).tolist()) == {0, 1}


def test_partitions_are_patient_disjoint_and_deterministic(fixture_cfg: dict) -> None:
    ds = tabular_benchmarks.load_tabular_dataset(fixture_cfg)
    by_group: dict = {}
    for g, part in zip(ds.groups.tolist(), ds.partition.tolist()):
        by_group.setdefault(g, set()).add(part)
    assert all(len(parts) == 1 for parts in by_group.values())
    assert set(ds.partition.tolist()) == {"train", "selection", "test"}
    again = tabular_benchmarks.load_tabular_dataset(fixture_cfg)
    assert again.partition.tolist() == ds.partition.tolist()


def test_fitness_rounds_never_touch_locked_test_and_carry_no_labels(fixture_cfg: dict) -> None:
    ds = tabular_benchmarks.load_tabular_dataset(fixture_cfg)
    part_of = dict(zip(ds.groups.tolist(), ds.partition.tolist()))
    rounds = tabular_benchmarks.build_tabular_rounds(FIXTURE_PROBLEM, split="all", perturb=True, perturb_seed=11)
    assert len(rounds) == 3
    shards = [set(r.eval_groups.tolist()) for r in rounds]
    for r in rounds:
        assert {part_of[g] for g in r.train_groups.tolist()} == {"train"}
        assert {part_of[g] for g in r.eval_groups.tolist()} == {"selection"}
        assert len(r.y_train) == r.task.n_train <= 2000
        assert set(r.eval) == set(fixture_cfg["numeric_columns"]) | set(fixture_cfg["categorical_columns"])
    assert not (shards[0] & shards[1]) and not (shards[1] & shards[2])

    (locked,) = tabular_benchmarks.build_tabular_rounds(FIXTURE_PROBLEM, split="test", heldout_seed=5)
    assert {part_of[g] for g in locked.eval_groups.tolist()} == {"test"}


def test_perturb_seed_changes_training_subsample(fixture_cfg: dict) -> None:
    a = tabular_benchmarks.build_tabular_rounds(FIXTURE_PROBLEM, perturb=True, perturb_seed=1)
    b = tabular_benchmarks.build_tabular_rounds(FIXTURE_PROBLEM, perturb=True, perturb_seed=2)
    a2 = tabular_benchmarks.build_tabular_rounds(FIXTURE_PROBLEM, perturb=True, perturb_seed=1)
    assert a[0].train_groups.tolist() != b[0].train_groups.tolist()
    assert a[0].train_groups.tolist() == a2[0].train_groups.tolist()


# ----------------------------------------------------------------------------- toolkit


def test_encoder_preserves_missing_and_maps_unseen_and_rare_to_other() -> None:
    task = tabular_benchmarks.TabularTask(
        problem_id="t", round_id="R1", numeric_columns=("x",), categorical_columns=("c",), n_train=6, n_eval=3
    )
    train = {"x": np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0]), "c": np.array(["a", "a", "a", "b", None, "z"], dtype=object)}
    ev = {"x": np.array([np.nan, 2.0, 0.0]), "c": np.array(["never_seen", None, "a"], dtype=object)}
    enc = TabularEncoder(min_count=2)
    x_tr = enc.fit_transform(task, train)
    x_ev = enc.transform(ev)
    assert enc.feature_names == ["x", "c"] and enc.categorical_indices == [1]
    assert np.isnan(x_tr[1, 0]) and np.isnan(x_ev[0, 0])
    assert np.isnan(x_tr[4, 1]) and np.isnan(x_ev[1, 1])
    other = x_tr[3, 1]  # "b" is rare (count 1 < 2) -> OTHER
    assert x_tr[5, 1] == other and x_ev[0, 1] == other
    assert x_ev[2, 1] == x_tr[0, 1] != other

    one_hot = TabularEncoder(min_count=2, one_hot=True)
    oh = one_hot.fit_transform(task, train)
    assert one_hot.categorical_indices == [] and oh.shape[1] == 1 + 2  # x + {a, OTHER}
    assert np.nansum(oh[4, 1:]) == 0.0  # missing -> all-zero indicators


@pytest.mark.parametrize(
    "code,group",
    [
        ("428", "circulatory"), ("785", "circulatory"), ("486", "respiratory"), ("786", "respiratory"),
        ("250.83", "diabetes"), ("250", "diabetes"), ("996", "injury"), ("715", "musculoskeletal"),
        ("599", "genitourinary"), ("788", "genitourinary"), ("174", "neoplasms"), ("530", "digestive"),
        ("V57", "other"), ("E888", "other"), (None, "missing"), ("?", "missing"),
    ],
)
def test_icd9_group_follows_strack_2014_chapters(code, group) -> None:
    assert icd9_group(code) == group


# ----------------------------------------------------------------------------- metrics


def test_paired_bootstrap_auc_diff_detects_real_lift_and_null() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    weak = y * 0.3 + rng.normal(0, 1, 3000)
    strong = y * 1.0 + rng.normal(0, 1, 3000)
    lift = paired_bootstrap_auc_diff(y, strong, weak, n_boot=300, seed=1)
    assert lift["delta"] > 0.1 and lift["ci_low"] > 0.0
    null = paired_bootstrap_auc_diff(y, weak, weak, n_boot=100, seed=1)
    assert null["delta"] == 0.0 and null["ci_low"] == 0.0 == null["ci_high"]


# ----------------------------------------------------------------------------- sandboxed evaluator


def test_sandboxed_seed_program_is_graded_on_auc(fixture_cfg: dict) -> None:
    res = evaluator.evaluate_program(_seed_code(), problem_id=FIXTURE_PROBLEM, perturb=True, perturb_seed=3)
    assert res["score"] is not None, res["insights"]
    joined = "\n".join(res["insights"])
    assert "Violations=0" in joined and "ROC-AUC=" in joined and "RUNTIME ERRORS" not in joined
    assert 6000.0 < res["score"] < 10000.0


def test_sandbox_round_crash_costs_one_violation_and_other_rounds_still_graded(fixture_cfg: dict) -> None:
    code = _seed_code().replace(
        "def fit_and_score_risk(",
        "_CALLS = [0]\n\n\ndef fit_and_score_risk(",
    ).replace(
        '    """Fit',
        '    _CALLS[0] += 1\n    if _CALLS[0] == 1:\n        raise ValueError("boom")\n    """Fit',
    )
    assert "boom" in code
    res = evaluator.evaluate_program(code, problem_id=FIXTURE_PROBLEM, perturb=True, perturb_seed=3)
    assert res["score"] is not None
    assert -1e6 + 4000 < res["score"] < -1e6 + 10000
    assert any("RUNTIME ERRORS" in s for s in res["insights"])


def test_sandbox_rejects_wrong_length_predictions_as_violation(fixture_cfg: dict) -> None:
    code = """
import numpy as np

def fit_and_score_risk(task, train, y_train, eval_cols):
    return np.full(task.n_eval - 1, 0.1)
"""
    res = evaluator.evaluate_program(code, problem_id=FIXTURE_PROBLEM, perturb=True, perturb_seed=3)
    assert res["score"] is not None and res["score"] < -2e6
    assert "Violations=3" in res["insights"][0]


def test_sandbox_blocks_reading_the_dataset_and_benchmark_module_is_redacted(fixture_cfg: dict, tmp_path) -> None:
    code = f"""
import tabular_benchmarks

def fit_and_score_risk(task, train, y_train, eval_cols):
    assert not hasattr(tabular_benchmarks, "load_tabular_dataset")
    return open({str(tmp_path / 'fixture.csv')!r}).read()
"""
    res = evaluator.evaluate_program(code, problem_id=FIXTURE_PROBLEM, perturb=True, perturb_seed=3)
    assert res["score"] is None
    assert any("Sandbox" in s for s in res["insights"])


def test_sandbox_allows_lazy_imports_of_installed_library_submodules(fixture_cfg: dict) -> None:
    # Run 1: 7/85 candidates died on "Sandbox blocked file open: __init__.py" for exactly these imports.
    code = """
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

def fit_and_score_risk(task, train, y_train, eval_cols):
    import scipy.sparse as sp
    from sklearn.cluster import KMeans
    assert sp.csr_matrix(np.eye(2)).nnz == 2 and TruncatedSVD and CountVectorizer and KMeans
    return np.full(task.n_eval, float(np.mean(y_train)))
"""
    res = evaluator.evaluate_program(code, problem_id=FIXTURE_PROBLEM, perturb=True, perturb_seed=3)
    joined = "\n".join(res["insights"])
    assert res["score"] is not None and "Sandbox" not in joined, joined
    assert "Violations=0" in joined


_SITE = pathlib.Path(np.__file__).resolve().parent.parent  # an allowed library root
_TRAVERSAL = str(_SITE) + "/.." * len(_SITE.parts) + str(REPO_ROOT / "src" / "evaluator.py")


@pytest.mark.parametrize(
    "target",
    [REPO_ROOT / "src" / "evaluator.py", REPO_ROOT / ".env", pathlib.Path("/etc/passwd"), _TRAVERSAL],
)
def test_sandbox_still_blocks_reading_repo_sources_secrets_and_system_files(fixture_cfg: dict, target) -> None:
    code = f"""
def fit_and_score_risk(task, train, y_train, eval_cols):
    return open({str(target)!r}).read()
"""
    res = evaluator.evaluate_program(code, problem_id=FIXTURE_PROBLEM, perturb=True, perturb_seed=3)
    assert any("Sandbox blocked file open" in s for s in res["insights"]), res["insights"]


def test_label_shuffle_check_collapses_real_signal_to_chance(fixture_cfg: dict) -> None:
    from scripts.leakage_check import label_shuffle_check

    res = label_shuffle_check(_seed_code(), problem_id=FIXTURE_PROBLEM, n_shuffles=6)
    assert res["true_auc"] > 0.62  # the fixture has real signal
    assert res["verdict"] == "PASS", res


def test_label_shuffle_check_flags_signal_that_does_not_come_from_y_train(fixture_cfg: dict) -> None:
    from scripts.leakage_check import label_shuffle_check

    # Ignores y_train entirely: a stand-in for any leak (eval-time labels, IDs, row order). Shuffling can't hurt it.
    code = """
import numpy as np

def fit_and_score_risk(task, train, y_train, eval_cols):
    return np.asarray(eval_cols["number_inpatient"], dtype=float) / 10.0
"""
    res = label_shuffle_check(code, problem_id=FIXTURE_PROBLEM, n_shuffles=6)
    assert res["verdict"] == "FAIL", res


# ----------------------------------------------------------------------------- real problem wiring


def test_real_problem_files_are_strategy_only() -> None:
    pdir = REPO_ROOT / "problems" / REAL_PROBLEM
    cfg = json.loads((pdir / "problem_config.json").read_text())
    assert cfg["task_type"] == "tabular_classification"
    code = _seed_code()
    assert "EVOLVE-BLOCK-START" in code and "EVOLVE-BLOCK-END" in code
    assert "def fit_and_score_risk(" in code and "tabular_benchmarks" not in code
    desc = (pdir / "problem_description.md").read_text()
    for api in ("TabularEncoder", "icd9_group", "icd9_comorbidities", "oof_target_encode", "fit_and_score_risk"):
        assert api in desc


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "diabetes130" / "diabetic_data.csv").is_file(),
    reason="Run scripts/fetch_diabetes130.sh to download the UCI dataset",
)
def test_real_dataset_matches_published_cohort_and_seed_auc_is_plausible() -> None:
    cfg = clinical_benchmarks.load_problem_config(REAL_PROBLEM)
    ds = tabular_benchmarks.load_tabular_dataset(cfg)
    assert len(ds.y) == 99_343  # 101,766 encounters minus expired/hospice dispositions
    assert 0.11 < float(ds.y.mean()) < 0.12
    res = evaluator.evaluate_in_process(_seed_code(), problem_id=REAL_PROBLEM, perturb=False)
    mean_auc = float(np.mean([r["roc_auc"] for r in res["rounds"]]))
    assert 0.60 < mean_auc < 0.75, res["insights"]


def test_report_arm_labels_never_collide() -> None:
    """Why: arms were keyed by file stem, so `run1/rank2.py` silently overwrote `run2/rank2.py` in the locked report."""
    from scripts.tabular_report import arm_labels

    paths = [pathlib.Path("runs/A/rank1.py"), pathlib.Path("runs/A/rank2.py"), pathlib.Path("runs/B/rank2.py")]
    labels = arm_labels(paths)
    assert len(set(labels)) == len(paths)
    assert labels[0] == "rank1"  # unique stems stay short
    assert labels[1:] == ["A/rank2", "B/rank2"]


# ----------------------------------------------------------------------------- feature helpers & baseline arms


def test_icd9_comorbidities_flags_conditions_across_diag_columns() -> None:
    from clinical_toolkit import icd9_comorbidities

    d1 = np.array(["428", "250.83", "V57", None, "585"], dtype=object)
    d2 = np.array(["585", "?", "196", "491", None], dtype=object)
    cm = icd9_comorbidities(d1, d2)
    assert cm["cm_chf"].tolist() == [1, 0, 0, 0, 0]
    assert cm["cm_renal"].tolist() == [1, 0, 0, 0, 1]
    assert cm["cm_diabetes_complicated"].tolist() == [0, 1, 0, 0, 0]
    assert cm["cm_metastatic"].tolist() == [0, 0, 1, 0, 0]
    assert cm["cm_pulmonary"].tolist() == [0, 0, 0, 1, 0]
    assert cm["cm_count"].tolist() == [2, 1, 1, 1, 1]  # a condition coded twice counts once
    assert all(v.dtype == np.float64 and len(v) == 5 for v in cm.values())


def test_oof_target_encode_never_uses_a_rows_own_label() -> None:
    from clinical_toolkit import oof_target_encode

    rng = np.random.default_rng(0)
    col = np.array(rng.choice(["a", "b", "c", None], size=400), dtype=object)
    y = (rng.random(400) < np.where(col == "a", 0.5, 0.1)).astype(int)
    ev = np.array(["a", "b", "zzz_unseen", None], dtype=object)
    tr_enc, ev_enc = oof_target_encode(col, y, ev, n_folds=5, seed=0)
    assert tr_enc.shape == (400,) and ev_enc.shape == (4,)
    assert ev_enc[0] > ev_enc[1]  # learnt signal
    assert ev_enc[2] == pytest.approx(y.mean())  # unseen -> prior
    flipped = y.copy()
    flipped[7] = 1 - flipped[7]
    tr2, _ = oof_target_encode(col, flipped, ev, n_folds=5, seed=0)
    assert tr2[7] == tr_enc[7]  # row 7's encoding is independent of its own label
    assert not np.array_equal(tr2, tr_enc)  # ...but other folds do see it


@pytest.mark.parametrize("arm_name", ["xgboost", "stack"])
def test_report_baseline_arms_produce_valid_predictions_with_signal(fixture_cfg: dict, arm_name: str) -> None:
    from scripts.tabular_report import DEFAULT_PARAMS, StackArm, XgbArm

    arm = XgbArm(DEFAULT_PARAMS["xgboost"]) if arm_name == "xgboost" else StackArm(
        DEFAULT_PARAMS["lightgbm"], DEFAULT_PARAMS["xgboost"])
    res = evaluator.evaluate_tabular_in_process(arm, problem_id=FIXTURE_PROBLEM, perturb=True, perturb_seed=3)
    assert "Violations=0" in res["insights"][0], res["insights"]
    assert float(np.mean([r["roc_auc"] for r in res["rounds"]])) > 0.62


def test_tuning_trial_zero_reproduces_the_seed_program(fixture_cfg: dict) -> None:
    """Why: the tuned arm is only guaranteed >= seed if trial 0 *is* the seed. Pins it when the seed changes."""
    from scripts.tabular_report import DEFAULT_PARAMS, LgbmArm

    kw = dict(problem_id=FIXTURE_PROBLEM, perturb=True, perturb_seed=3, keep_predictions=True)
    seed = evaluator.evaluate_tabular_in_process(_seed_code(), **kw)
    arm = evaluator.evaluate_tabular_in_process(LgbmArm(DEFAULT_PARAMS["lightgbm"]), **kw)
    for a, b in zip(seed["rounds"], arm["rounds"]):
        np.testing.assert_allclose(a["probs"], b["probs"])


@pytest.mark.parametrize("script", ["tabular_report", "leakage_check"])
def test_trusted_scoring_scripts_pin_native_threads_like_the_sandbox(script: str) -> None:
    """Why: the sandbox sets OMP/BLAS threads to 1, the in-process scripts didn't. Candidates using `n_jobs=-1` then
    grabbed every core (load 50 on 24 cores), causing wall-clock timeouts in live runs, and thread count can change
    GBDT results versus how the candidate was graded."""
    import subprocess
    import sys

    probe = (f"import scripts.{script}, lightgbm, threadpoolctl; "
             "print(max(p['num_threads'] for p in threadpoolctl.threadpool_info()))")
    out = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
                         env={k: v for k, v in __import__("os").environ.items() if not k.endswith("_NUM_THREADS")})
    assert out.stdout.strip().splitlines()[-1] == "1", out.stdout + out.stderr
