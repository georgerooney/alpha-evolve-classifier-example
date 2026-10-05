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

"""Real-data tabular benchmark builder (parent process only): loading, patient-grouped splits, and evaluation rounds.

Why this exists alongside `clinical_benchmarks`: the synthetic generator can't measure AlphaEvolve's real lift (the
toolkit encodes its latent formula, prevalence is 35-69%, and n_eval≈170 gives ±0.08 AUC CIs). This module turns any
CSV classification dataset described by `problems/<id>/problem_config.json` (`task_type: tabular_classification`)
into rounds with three guarantees:

1. **Group-disjoint partitions.** Each `group_column` value (patient) is hashed with `partition_salt` into exactly
   one of `train` / `selection` / `test`, so repeat encounters never straddle a split.
2. **Fitness never touches the locked test set.** `split="all"` (evolution fitness) trains on a seeded subsample of
   `train` and scores disjoint patient shards of `selection`. `split="test"` (finalists only) trains on the full
   `train` partition and scores `test`.
3. **No labels or identifiers reach the candidate.** `TabularRound.eval` holds only the declared feature columns;
   labels, group IDs and ID columns stay in the parent.

The candidate worker never sees this module (the evaluator swaps in a redacted stub), and the sandbox blocks file
reads, so the dataset can't be opened from candidate code.
"""

from __future__ import annotations

import csv
import dataclasses
import functools
import hashlib
import json
import os
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np

import clinical_benchmarks
from clinical_benchmarks import DEFAULT_HELDOUT_SEED, DEFAULT_PERTURB_SEED, load_problem_config
from clinical_models import TabularTask

TASK_TYPE = "tabular_classification"


@dataclasses.dataclass(frozen=True)
class TabularDataset:
    """Cleaned dataset: feature columns, binary label, group IDs and the per-row partition name."""

    columns: Dict[str, np.ndarray]
    y: np.ndarray
    groups: np.ndarray
    partition: np.ndarray


@dataclasses.dataclass(frozen=True)
class TabularRound:
    """One fit/score round. `train`/`eval` go to the candidate; `y_eval` and `*_groups` stay in the parent."""

    task: TabularTask
    train: Dict[str, np.ndarray]
    y_train: np.ndarray
    eval: Dict[str, np.ndarray]
    y_eval: np.ndarray
    train_groups: np.ndarray
    eval_groups: np.ndarray


def is_tabular_problem(problem_id: str) -> bool:
    try:
        return load_problem_config(problem_id).get("task_type") == TASK_TYPE
    except ValueError:
        return False


def resolve_data_path(cfg: Dict) -> pathlib.Path:
    """`data_path` is relative to the repo root (the parent of `problems/`) unless absolute."""
    path = pathlib.Path(cfg["data_path"])
    return path if path.is_absolute() else (clinical_benchmarks.PROBLEMS_ROOT.parent / path).resolve()


def partition_groups(
    groups: np.ndarray, salt: str, test_fraction: float, selection_fraction: float
) -> np.ndarray:
    """Deterministically assigns every group to `test` / `selection` / `train` by salted SHA-256 hash."""
    out = np.empty(groups.shape[0], dtype=object)
    cache: Dict[str, str] = {}
    for i, g in enumerate(groups.tolist()):
        part = cache.get(g)
        if part is None:
            u = int(hashlib.sha256(f"{salt}:{g}".encode()).hexdigest()[:12], 16) / float(16**12)
            part = "test" if u < test_fraction else ("selection" if u < test_fraction + selection_fraction else "train")
            cache[g] = part
        out[i] = part
    return out


@functools.lru_cache(maxsize=4)
def _load_cached(cfg_json: str, data_path: str, mtime_ns: int) -> TabularDataset:
    cfg = json.loads(cfg_json)
    missing = set(cfg.get("missing_values", ["?"])) | {""}
    exclude = {k: set(v) for k, v in cfg.get("exclude_rows", {}).items()}
    numeric = list(cfg["numeric_columns"])
    categorical = list(cfg["categorical_columns"])
    positives = set(cfg["positive_values"])
    label_col, group_col = cfg["label_column"], cfg["group_column"]
    leaked = {label_col, group_col, *cfg.get("id_columns", [])} & set(numeric + categorical)
    if leaked:
        raise ValueError(f"Label/group/ID columns must not be features: {sorted(leaked)}")

    raw: Dict[str, List] = {c: [] for c in numeric + categorical}
    labels: List[int] = []
    groups: List[str] = []
    with open(data_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if any(row.get(col) in vals for col, vals in exclude.items()):
                continue
            for c in numeric:
                v = row[c]
                raw[c].append(np.nan if v in missing else float(v))
            for c in categorical:
                v = row[c]
                raw[c].append(None if v in missing else v)
            labels.append(1 if row[label_col] in positives else 0)
            groups.append(str(row[group_col]))

    columns = {c: np.asarray(raw[c], dtype=np.float64) for c in numeric}
    columns.update({c: np.asarray(raw[c], dtype=object) for c in categorical})
    groups_arr = np.asarray(groups, dtype=object)
    partition = partition_groups(
        groups_arr,
        str(cfg.get("partition_salt", cfg["problem_id"])),
        float(cfg.get("test_fraction", 0.2)),
        float(cfg.get("selection_fraction", 0.2)),
    )
    return TabularDataset(columns=columns, y=np.asarray(labels, dtype=np.int8), groups=groups_arr, partition=partition)


def load_tabular_dataset(cfg: Dict) -> TabularDataset:
    """Loads and partitions the CSV declared in `cfg` (cached on config + file mtime)."""
    path = resolve_data_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}; see problems/{cfg['problem_id']}/problem_description.md")
    return _load_cached(json.dumps(cfg, sort_keys=True), str(path), path.stat().st_mtime_ns)


def _take(ds: TabularDataset, idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {name: col[idx] for name, col in ds.columns.items()}


def _make_round(cfg: Dict, ds: TabularDataset, round_id: str, tr: np.ndarray, ev: np.ndarray) -> TabularRound:
    task = TabularTask(
        problem_id=str(cfg["problem_id"]),
        round_id=round_id,
        numeric_columns=tuple(cfg["numeric_columns"]),
        categorical_columns=tuple(cfg["categorical_columns"]),
        n_train=int(tr.shape[0]),
        n_eval=int(ev.shape[0]),
        label_description=str(cfg.get("label_description", "")),
    )
    return TabularRound(
        task=task,
        train=_take(ds, tr),
        y_train=ds.y[tr].astype(np.int64),
        eval=_take(ds, ev),
        y_eval=ds.y[ev].astype(np.int64),
        train_groups=ds.groups[tr],
        eval_groups=ds.groups[ev],
    )


def build_tabular_rounds(
    problem_id: str,
    *,
    split: str = "all",
    perturb: bool = True,
    perturb_seed: Optional[int] = None,
    heldout_seed: Optional[int] = None,
) -> List[TabularRound]:
    """Builds fitness rounds (`split="all"`) or the single locked-test round (`split="test"`).

    `perturb=False` gives the canonical public rounds (seed 0). `perturb=True` uses `perturb_seed`, else the private
    `AE_PERTURB_SEED`, to choose the training subsample and the selection shards, so candidates can't overfit
    one fixed draw.
    """
    cfg = load_problem_config(problem_id)
    ds = load_tabular_dataset(cfg)

    if split == "test":
        seed = heldout_seed if heldout_seed is not None else int(os.environ.get("AE_HELDOUT_SEED") or DEFAULT_HELDOUT_SEED)
        rng = np.random.default_rng(seed)
        tr = rng.permutation(np.flatnonzero(ds.partition == "train"))
        cap = cfg.get("test_train_rows")
        if cap:
            tr = tr[: int(cap)]
        ev = rng.permutation(np.flatnonzero(ds.partition == "test"))
        return [_make_round(cfg, ds, "LOCKED-TEST", tr, ev)]

    if perturb:
        seed = perturb_seed if perturb_seed is not None else int(os.environ.get("AE_PERTURB_SEED") or DEFAULT_PERTURB_SEED)
    else:
        seed = 0
    rng = np.random.default_rng(seed)
    n_rounds = int(cfg.get("n_rounds", 3))
    train_pool = np.flatnonzero(ds.partition == "train")
    sel_idx = np.flatnonzero(ds.partition == "selection")

    # Shard the selection partition by patient so each round scores a disjoint set of patients.
    sel_groups = np.unique(ds.groups[sel_idx].astype(str))
    shard_of = {g: i % n_rounds for i, g in enumerate(rng.permutation(sel_groups).tolist())}
    shard = np.fromiter((shard_of[g] for g in ds.groups[sel_idx].tolist()), dtype=np.int64, count=sel_idx.shape[0])

    n_train = min(int(cfg.get("train_rows", 20000)), train_pool.shape[0])
    rounds: List[TabularRound] = []
    for r in range(n_rounds):
        tr = rng.choice(train_pool, size=n_train, replace=False)
        ev = rng.permutation(sel_idx[shard == r])
        rounds.append(_make_round(cfg, ds, f"R{r + 1}", tr, ev))
    return rounds


def subgroup_columns(problem_id: str) -> Tuple[str, ...]:
    return tuple(load_problem_config(problem_id).get("subgroup_columns", ()))
