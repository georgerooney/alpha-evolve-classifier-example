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

"""Seed program for the Diabetes 130-US Hospitals 30-day readmission problem (real data).

Strategy only: an untuned single LightGBM on the raw columns. Encoding mechanics live in `clinical_toolkit`.
"""

from __future__ import annotations

from typing import Dict

import lightgbm as lgb
import numpy as np

from clinical_models import TabularTask
from clinical_toolkit import TabularEncoder, icd9_group  # noqa: F401  (icd9_group available for feature ideas)


# =====================================================================
# EVOLVE-BLOCK-START
# Strategy only. Pre-imported in the sandbox: numpy, scipy, sklearn, lightgbm, xgboost. No file/network access.


import numpy as np
from typing import Any, Dict, List, Sequence, Tuple
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


def _get_icd9_grouper():
    """Returns the icd9_group function from toolkit or a robust fallback."""
    fn = globals().get("icd9_group")
    if fn is not None:
        return fn
    try:
        from clinical_toolkit import icd9_group
        return icd9_group
    except Exception:
        def _fallback(val: Any) -> str:
            if val is None or val == "" or val == "?" or str(val) == "nan":
                return "missing"
            s = str(val).strip()
            if not s or s == "?":
                return "missing"
            if s.startswith(("V", "v", "E", "e")):
                return "other"
            try:
                code = float(s.split(".")[0])
            except ValueError:
                return "other"
            if 390 <= code <= 459 or code == 785:
                return "circulatory"
            elif 460 <= code <= 519 or code == 786:
                return "respiratory"
            elif 520 <= code <= 579 or code == 787:
                return "digestive"
            elif int(code) == 250:
                return "diabetes"
            elif 800 <= code <= 999:
                return "injury"
            elif 710 <= code <= 739:
                return "musculoskeletal"
            elif 580 <= code <= 629 or code == 788:
                return "genitourinary"
            elif 140 <= code <= 239:
                return "neoplasms"
            return "other"
        return _fallback


def _parse_age(val: Any) -> float:
    if val is None:
        return 65.0
    s = str(val).strip()
    if s.startswith("[") and "-" in s:
        try:
            p = s[1:].rstrip(")").split("-")
            return (float(p[0]) + float(p[1])) / 2.0
        except Exception:
            return 65.0
    return 65.0


def _extract_icd9_3digit(val: Any) -> str:
    if val is None:
        return "missing"
    s = str(val).strip()
    if not s or s in ("?", "nan", "None"):
        return "missing"
    return s.split(".")[0].strip()


def _has_prefix(d1: np.ndarray, d2: np.ndarray, d3: np.ndarray, prefixes: Tuple[str, ...]) -> np.ndarray:
    res = np.zeros(len(d1), dtype=bool)
    for p in prefixes:
        res |= np.char.startswith(d1, p) | np.char.startswith(d2, p) | np.char.startswith(d3, p)
    return res.astype(np.float64)


def _target_encode(
    train_vals: np.ndarray,
    eval_vals: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    m_smoothing: float = 18.0,
) -> Tuple[np.ndarray, np.ndarray]:
    n_train = len(y)

    def clean_val(v):
        if v is None or str(v) in ("nan", "None", "?"):
            return "missing"
        return str(v).strip()

    train_clean = np.array([clean_val(v) for v in train_vals], dtype=object)
    eval_clean = np.array([clean_val(v) for v in eval_vals], dtype=object)

    global_mean = float(np.mean(y))
    train_encoded = np.zeros(n_train, dtype=np.float64)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(train_clean, y):
        y_f = y[train_idx]
        cats_f = train_clean[train_idx]

        unique_cats, inverse, counts = np.unique(cats_f, return_inverse=True, return_counts=True)
        sums = np.zeros_like(counts, dtype=np.float64)
        np.add.at(sums, inverse, y_f)

        encoded_vals = (sums + m_smoothing * global_mean) / (counts + m_smoothing)
        mapping = dict(zip(unique_cats, encoded_vals))

        val_cats = train_clean[val_idx]
        train_encoded[val_idx] = np.array([mapping.get(c, global_mean) for c in val_cats], dtype=np.float64)

    # Fit on full train
    unique_cats, inverse, counts = np.unique(train_clean, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=np.float64)
    np.add.at(sums, inverse, y)
    encoded_vals = (sums + m_smoothing * global_mean) / (counts + m_smoothing)
    global_mapping = dict(zip(unique_cats, encoded_vals))

    eval_encoded = np.array([global_mapping.get(c, global_mean) for c in eval_clean], dtype=np.float64)
    return train_encoded, eval_encoded


def _pooled_diagnosis_target_encode(
    train_d1: np.ndarray,
    train_d2: np.ndarray,
    train_d3: np.ndarray,
    eval_d1: np.ndarray,
    eval_d2: np.ndarray,
    eval_d3: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    m_smoothing: float = 20.0,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Empirical Bayes target encoding pooled across all 3 diagnosis positions."""
    def clean_val(v):
        if v is None or str(v) in ("nan", "None", "?"):
            return "missing"
        return str(v).strip()

    tr_d1 = np.array([clean_val(v) for v in train_d1], dtype=object)
    tr_d2 = np.array([clean_val(v) for v in train_d2], dtype=object)
    tr_d3 = np.array([clean_val(v) for v in train_d3], dtype=object)

    ev_d1 = np.array([clean_val(v) for v in eval_d1], dtype=object)
    ev_d2 = np.array([clean_val(v) for v in eval_d2], dtype=object)
    ev_d3 = np.array([clean_val(v) for v in eval_d3], dtype=object)

    n_train = len(y)
    global_mean = float(np.mean(y))

    tr_te1 = np.zeros(n_train, dtype=np.float64)
    tr_te2 = np.zeros(n_train, dtype=np.float64)
    tr_te3 = np.zeros(n_train, dtype=np.float64)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(tr_d1, y):
        # Pool all 3 diagnosis positions for the training fold
        pool_cats = np.concatenate([tr_d1[train_idx], tr_d2[train_idx], tr_d3[train_idx]])
        pool_y = np.concatenate([y[train_idx], y[train_idx], y[train_idx]])

        unique_cats, inverse, counts = np.unique(pool_cats, return_inverse=True, return_counts=True)
        sums = np.zeros_like(counts, dtype=np.float64)
        np.add.at(sums, inverse, pool_y)

        fold_mean = float(np.mean(y[train_idx]))
        encoded_vals = (sums + m_smoothing * fold_mean) / (counts + m_smoothing)
        mapping = dict(zip(unique_cats, encoded_vals))

        tr_te1[val_idx] = np.array([mapping.get(c, fold_mean) for c in tr_d1[val_idx]], dtype=np.float64)
        tr_te2[val_idx] = np.array([mapping.get(c, fold_mean) for c in tr_d2[val_idx]], dtype=np.float64)
        tr_te3[val_idx] = np.array([mapping.get(c, fold_mean) for c in tr_d3[val_idx]], dtype=np.float64)

    # Full train pooled fit
    pool_cats_full = np.concatenate([tr_d1, tr_d2, tr_d3])
    pool_y_full = np.concatenate([y, y, y])

    unique_cats, inverse, counts = np.unique(pool_cats_full, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=np.float64)
    np.add.at(sums, inverse, pool_y_full)

    encoded_vals = (sums + m_smoothing * global_mean) / (counts + m_smoothing)
    global_mapping = dict(zip(unique_cats, encoded_vals))

    ev_te1 = np.array([global_mapping.get(c, global_mean) for c in ev_d1], dtype=np.float64)
    ev_te2 = np.array([global_mapping.get(c, global_mean) for c in ev_d2], dtype=np.float64)
    ev_te3 = np.array([global_mapping.get(c, global_mean) for c in ev_d3], dtype=np.float64)

    return (tr_te1, tr_te2, tr_te3), (ev_te1, ev_te2, ev_te3)


def _hierarchical_diagnosis_te(
    tr_codes: np.ndarray,
    tr_3d: np.ndarray,
    tr_grp: np.ndarray,
    ev_codes: np.ndarray,
    ev_3d: np.ndarray,
    ev_grp: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    m_grp: float = 30.0,
    m_3d: float = 15.0,
    m_code: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hierarchical Empirical Bayes shrinkage encoding (Full Code -> 3-Digit Category -> Group -> Global Mean)."""
    def clean(v):
        if v is None or str(v) in ("nan", "None", "?"):
            return "missing"
        return str(v).strip()

    tc = [clean(v) for v in tr_codes]
    t3 = [clean(v) for v in tr_3d]
    tg = [clean(v) for v in tr_grp]

    ec = [clean(v) for v in ev_codes]
    e3 = [clean(v) for v in ev_3d]
    eg = [clean(v) for v in ev_grp]

    n_train = len(y)
    global_mean = float(np.mean(y))
    tr_te = np.zeros(n_train, dtype=np.float64)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    def _fit_hier(codes, c3ds, grps, labels, gm):
        g_counts: Dict[str, float] = {}
        g_sums: Dict[str, float] = {}
        for g, val in zip(grps, labels):
            g_counts[g] = g_counts.get(g, 0.0) + 1.0
            g_sums[g] = g_sums.get(g, 0.0) + float(val)
        g_te = {g: (g_sums[g] + m_grp * gm) / (g_counts[g] + m_grp) for g in g_counts}

        d3_counts: Dict[str, float] = {}
        d3_sums: Dict[str, float] = {}
        d3_to_g: Dict[str, str] = {}
        for d, g, val in zip(c3ds, grps, labels):
            d3_counts[d] = d3_counts.get(d, 0.0) + 1.0
            d3_sums[d] = d3_sums.get(d, 0.0) + float(val)
            d3_to_g[d] = g
        d3_te = {
            d: (d3_sums[d] + m_3d * g_te.get(d3_to_g[d], gm)) / (d3_counts[d] + m_3d)
            for d in d3_counts
        }

        f_counts: Dict[str, float] = {}
        f_sums: Dict[str, float] = {}
        f_to_d3: Dict[str, str] = {}
        f_to_g: Dict[str, str] = {}
        for f, d, g, val in zip(codes, c3ds, grps, labels):
            f_counts[f] = f_counts.get(f, 0.0) + 1.0
            f_sums[f] = f_sums.get(f, 0.0) + float(val)
            f_to_d3[f] = d
            f_to_g[f] = g
        f_te = {
            f: (f_sums[f] + m_code * d3_te.get(f_to_d3[f], g_te.get(f_to_g[f], gm))) / (f_counts[f] + m_code)
            for f in f_counts
        }
        return f_te, d3_te, g_te

    for train_idx, val_idx in skf.split(tc, y):
        f_te, d3_te, g_te = _fit_hier(
            [tc[i] for i in train_idx],
            [t3[i] for i in train_idx],
            [tg[i] for i in train_idx],
            y[train_idx],
            float(np.mean(y[train_idx])),
        )
        fold_gm = float(np.mean(y[train_idx]))
        for i in val_idx:
            c, d, g = tc[i], t3[i], tg[i]
            tr_te[i] = f_te.get(c, d3_te.get(d, g_te.get(g, fold_gm)))

    # Fit on all training encounters
    f_te_all, d3_te_all, g_te_all = _fit_hier(tc, t3, tg, y, global_mean)
    ev_te = np.zeros(len(ec), dtype=np.float64)
    for i in range(len(ec)):
        c, d, g = ec[i], e3[i], eg[i]
        ev_te[i] = f_te_all.get(c, d3_te_all.get(d, g_te_all.get(g, global_mean)))

    return tr_te, ev_te


def _compute_diagnosis_logodds(
    train_diags: Sequence[np.ndarray],
    eval_diags: Sequence[np.ndarray],
    y: np.ndarray,
    n_folds: int = 5,
    smoothing: float = 15.0,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Empirical Bayes smoothed log-odds diagnosis risk features out-of-fold."""
    def clean(v):
        if v is None or str(v) in ("nan", "None", "?", ""):
            return "missing"
        return str(v).strip()

    tr_clean = [[clean(v) for v in d] for d in train_diags]
    ev_clean = [[clean(v) for v in d] for d in eval_diags]
    n_train = len(y)
    n_eval = len(ev_clean[0])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    def fit_logodds(indices, labels):
        pos_n = float(np.sum(labels))
        neg_n = float(len(labels) - pos_n)
        prior_pos = pos_n / float(len(labels))
        prior_neg = neg_n / float(len(labels))
        prior_logodds = np.log((pos_n + 1e-6) / (neg_n + 1e-6))
        code_pos: Dict[str, float] = {}
        code_neg: Dict[str, float] = {}
        for idx, yi in zip(indices, labels):
            codes = set(tr_clean[k][idx] for k in range(len(train_diags)))
            codes.discard("missing")
            for c in codes:
                if yi == 1:
                    code_pos[c] = code_pos.get(c, 0.0) + 1.0
                else:
                    code_neg[c] = code_neg.get(c, 0.0) + 1.0
        all_codes = set(code_pos.keys()) | set(code_neg.keys())
        weights: Dict[str, float] = {}
        for c in all_codes:
            cp = code_pos.get(c, 0.0)
            cn = code_neg.get(c, 0.0)
            weights[c] = float(np.log((cp + smoothing * prior_pos) / (cn + smoothing * prior_neg)) - prior_logodds)
        return weights

    tr_sum = np.zeros(n_train, dtype=np.float64)
    tr_max = np.zeros(n_train, dtype=np.float64)
    tr_prim = np.zeros(n_train, dtype=np.float64)
    for train_idx, val_idx in skf.split(np.zeros(n_train), y):
        w = fit_logodds(train_idx, y[train_idx])
        for idx in val_idx:
            w1 = w.get(tr_clean[0][idx], 0.0)
            w2 = w.get(tr_clean[1][idx], 0.0)
            w3 = w.get(tr_clean[2][idx], 0.0)
            tr_sum[idx] = w1 + w2 + w3
            tr_max[idx] = max(w1, w2, w3)
            tr_prim[idx] = w1

    w_full = fit_logodds(np.arange(n_train), y)
    ev_sum = np.zeros(n_eval, dtype=np.float64)
    ev_max = np.zeros(n_eval, dtype=np.float64)
    ev_prim = np.zeros(n_eval, dtype=np.float64)
    for idx in range(n_eval):
        w1 = w_full.get(ev_clean[0][idx], 0.0)
        w2 = w_full.get(ev_clean[1][idx], 0.0)
        w3 = w_full.get(ev_clean[2][idx], 0.0)
        ev_sum[idx] = w1 + w2 + w3
        ev_max[idx] = max(w1, w2, w3)
        ev_prim[idx] = w1

    return {"sum": tr_sum, "max": tr_max, "prim": tr_prim}, {"sum": ev_sum, "max": ev_max, "prim": ev_prim}


def _compute_charlson(d1: np.ndarray, d2: np.ndarray, d3: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """Computes the validated Charlson Comorbidity Index with age adjustment."""
    d1_s = d1.astype(str)
    d2_s = d2.astype(str)
    d3_s = d3.astype(str)

    # 1 point conditions
    mi = _has_prefix(d1_s, d2_s, d3_s, ("410", "412"))
    chf = _has_prefix(d1_s, d2_s, d3_s, ("428",))
    pvd = _has_prefix(d1_s, d2_s, d3_s, ("440", "441", "442", "443", "447"))
    cvd = _has_prefix(d1_s, d2_s, d3_s, ("430", "431", "432", "433", "434", "435", "436", "437", "438"))
    dementia = _has_prefix(d1_s, d2_s, d3_s, ("290",))
    copd = _has_prefix(d1_s, d2_s, d3_s, ("490", "491", "492", "493", "494", "495", "496", "500", "501", "502", "503", "504", "505"))
    connective = _has_prefix(d1_s, d2_s, d3_s, ("710", "714", "725"))
    pud = _has_prefix(d1_s, d2_s, d3_s, ("531", "532", "533", "534"))
    mild_liver = _has_prefix(d1_s, d2_s, d3_s, ("571.2", "571.4", "571.5", "571.6"))
    diab_uncomp = _has_prefix(d1_s, d2_s, d3_s, ("250.0", "250.1", "250.2", "250.3"))

    # 2 point conditions
    hemi = _has_prefix(d1_s, d2_s, d3_s, ("342", "344"))
    renal_mod = _has_prefix(d1_s, d2_s, d3_s, ("582", "583", "585", "586"))
    diab_comp = _has_prefix(d1_s, d2_s, d3_s, ("250.4", "250.5", "250.6", "250.7", "250.8", "250.9"))
    malig_codes = tuple(str(x) for x in range(140, 196)) + tuple(str(x) for x in range(200, 209))
    malig = _has_prefix(d1_s, d2_s, d3_s, malig_codes)

    # 3 point conditions
    mod_liver = _has_prefix(d1_s, d2_s, d3_s, ("572.2", "572.3", "572.4", "572.8"))

    # 6 point conditions
    mets = _has_prefix(d1_s, d2_s, d3_s, ("196", "197", "198", "199"))
    hiv = _has_prefix(d1_s, d2_s, d3_s, ("042", "043", "044"))

    age_pts = np.where(ages >= 80.0, 4.0,
              np.where(ages >= 70.0, 3.0,
              np.where(ages >= 60.0, 2.0,
              np.where(ages >= 50.0, 1.0, 0.0))))

    cci = (
        mi + chf + pvd + cvd + dementia + copd + connective + pud + mild_liver + diab_uncomp +
        hemi * 2.0 + renal_mod * 2.0 + diab_comp * 2.0 + malig * 2.0 +
        mod_liver * 3.0 +
        mets * 6.0 + hiv * 6.0 +
        age_pts
    )
    return cci


def _frequency_encode(
    train_vals: np.ndarray,
    eval_vals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    def clean_val(v):
        if v is None or str(v) in ("nan", "None", "?"):
            return "missing"
        return str(v).strip()

    tr_clean = [clean_val(v) for v in train_vals]
    ev_clean = [clean_val(v) for v in eval_vals]
    n = float(len(tr_clean))
    counts: Dict[str, float] = {}
    for x in tr_clean:
        counts[x] = counts.get(x, 0.0) + 1.0

    tr_freq = np.array([counts.get(x, 0.0) / n for x in tr_clean], dtype=np.float64)
    ev_freq = np.array([counts.get(x, 0.0) / n for x in ev_clean], dtype=np.float64)
    return tr_freq, ev_freq


def _map_a1c(v) -> float:
    if v == "Norm": return 0.0
    if v == ">7": return 1.0
    if v == ">8": return 2.0
    return -1.0


def _map_glu(v) -> float:
    if v == "Norm": return 0.0
    if v == ">200": return 1.0
    if v == ">300": return 2.0
    return -1.0




def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit an ensemble of gradient-boosted trees on `train`/`y_train` and return readmission probabilities."""
    icd9 = _get_icd9_grouper()

    train_data = dict(train)
    eval_data = dict(eval_cols)
    extra_numeric: List[str] = []
    extra_categorical: List[str] = []

    # 1. Diagnosis ICD-9 grouping & 3-digit category extraction
    for diag_col in ("diag_1", "diag_2", "diag_3"):
        grp_name = f"{diag_col}_group"
        train_data[grp_name] = np.array([icd9(c) for c in train[diag_col]], dtype=object)
        eval_data[grp_name] = np.array([icd9(c) for c in eval_cols[diag_col]], dtype=object)
        extra_categorical.append(grp_name)

        d3_name = f"{diag_col}_3digit"
        train_data[d3_name] = np.array([_extract_icd9_3digit(c) for c in train[diag_col]], dtype=object)
        eval_data[d3_name] = np.array([_extract_icd9_3digit(c) for c in eval_cols[diag_col]], dtype=object)
        extra_categorical.append(d3_name)

    # 2. Granular comorbidity flags & primary admission drivers
    for data in (train_data, eval_data):
        d1_g = data["diag_1_group"]
        d2_g = data["diag_2_group"]
        d3_g = data["diag_3_group"]

        has_circ = ((d1_g == "circulatory") | (d2_g == "circulatory") | (d3_g == "circulatory")).astype(np.float64)
        has_resp = ((d1_g == "respiratory") | (d2_g == "respiratory") | (d3_g == "respiratory")).astype(np.float64)
        has_diab = ((d1_g == "diabetes") | (d2_g == "diabetes") | (d3_g == "diabetes")).astype(np.float64)
        has_renal = ((d1_g == "genitourinary") | (d2_g == "genitourinary") | (d3_g == "genitourinary")).astype(np.float64)
        has_neoplasm = ((d1_g == "neoplasms") | (d2_g == "neoplasms") | (d3_g == "neoplasms")).astype(np.float64)

        d1_s = data["diag_1"].astype(str)
        d2_s = data["diag_2"].astype(str)
        d3_s = data["diag_3"].astype(str)

        has_chf = _has_prefix(d1_s, d2_s, d3_s, ("428",))
        has_hypertension = _has_prefix(d1_s, d2_s, d3_s, ("401", "402", "403", "404", "405"))
        has_ckd = _has_prefix(d1_s, d2_s, d3_s, ("585",))
        has_renal_failure = _has_prefix(d1_s, d2_s, d3_s, ("584", "585", "586"))
        has_copd = _has_prefix(d1_s, d2_s, d3_s, ("491", "492", "496"))
        has_pneumonia = _has_prefix(d1_s, d2_s, d3_s, ("480", "481", "482", "483", "484", "485", "486"))
        has_sepsis = _has_prefix(d1_s, d2_s, d3_s, ("038", "995.91", "995.92"))
        has_fluid_elec = _has_prefix(d1_s, d2_s, d3_s, ("276",))
        has_ketoacidosis = _has_prefix(d1_s, d2_s, d3_s, ("250.1",))
        has_cad = _has_prefix(d1_s, d2_s, d3_s, ("410", "411", "412", "413", "414"))
        has_arrhythmia = _has_prefix(d1_s, d2_s, d3_s, ("427",))
        has_pvd = _has_prefix(d1_s, d2_s, d3_s, ("440", "441", "442", "443", "447"))
        has_stroke = _has_prefix(d1_s, d2_s, d3_s, ("430", "431", "432", "433", "434", "435", "436", "437", "438"))
        has_anemia = _has_prefix(d1_s, d2_s, d3_s, ("280", "281", "285"))
        has_uti = _has_prefix(d1_s, d2_s, d3_s, ("599.0", "599", "590"))
        has_skin_ulcer = _has_prefix(d1_s, d2_s, d3_s, ("681", "682", "707"))

        is_diab1 = np.char.startswith(d1_s, "250")
        is_simple1 = (d1_s == "250") | np.char.startswith(d1_s, "250.0")
        is_diab2 = np.char.startswith(d2_s, "250")
        is_simple2 = (d2_s == "250") | np.char.startswith(d2_s, "250.0")
        is_diab3 = np.char.startswith(d3_s, "250")
        is_simple3 = (d3_s == "250") | np.char.startswith(d3_s, "250.0")
        has_diab_comp = ((is_diab1 & ~is_simple1) | (is_diab2 & ~is_simple2) | (is_diab3 & ~is_simple3)).astype(np.float64)

        # Multi-organ syndromic interactions
        has_ckm_syndrome = (has_diab * has_circ * has_renal).astype(np.float64)
        has_chf_and_ckd = (has_chf * has_ckd).astype(np.float64)
        has_cardiopulmonary = (has_circ * has_resp).astype(np.float64)
        has_diab_htn_ckd = (has_diab * has_hypertension * has_ckd).astype(np.float64)

        # Count of distinct organ system chapters affected
        n_rows = len(d1_s)
        distinct_groups = np.zeros(n_rows, dtype=np.float64)
        for i in range(n_rows):
            g_set = set()
            for g in (d1_g[i], d2_g[i], d3_g[i]):
                if g != "missing":
                    g_set.add(g)
            distinct_groups[i] = float(len(g_set))

        data["has_diabetes_diag"] = has_diab
        data["has_chf_diag"] = has_chf
        data["has_circ_diag"] = has_circ
        data["has_resp_diag"] = has_resp
        data["has_renal_diag"] = has_renal
        data["has_neoplasm_diag"] = has_neoplasm
        data["has_hypertension"] = has_hypertension
        data["has_ckd"] = has_ckd
        data["has_renal_failure"] = has_renal_failure
        data["has_copd"] = has_copd
        data["has_pneumonia"] = has_pneumonia
        data["has_sepsis"] = has_sepsis
        data["has_fluid_elec"] = has_fluid_elec
        data["has_ketoacidosis"] = has_ketoacidosis
        data["has_diabetes_with_complications"] = has_diab_comp
        data["has_cad"] = has_cad
        data["has_arrhythmia"] = has_arrhythmia
        data["has_pvd"] = has_pvd
        data["has_stroke"] = has_stroke
        data["has_anemia"] = has_anemia
        data["has_uti"] = has_uti
        data["has_skin_ulcer"] = has_skin_ulcer
        data["has_ckm_syndrome"] = has_ckm_syndrome
        data["has_chf_and_ckd"] = has_chf_and_ckd
        data["has_cardiopulmonary"] = has_cardiopulmonary
        data["has_diab_htn_ckd"] = has_diab_htn_ckd
        data["n_distinct_diag_groups"] = distinct_groups

        data["comorbidity_score"] = (
            has_diab + has_chf * 2.0 + has_circ + has_resp + has_renal * 1.5 +
            has_neoplasm * 2.0 + has_hypertension + has_copd + has_sepsis * 2.0 +
            has_ckd * 1.5 + has_cad + has_arrhythmia + has_stroke + has_skin_ulcer + has_anemia
        )

        # Primary diagnosis driver flags
        data["diag1_is_chf"] = np.char.startswith(d1_s, "428").astype(np.float64)
        data["diag1_is_circ"] = (d1_g == "circulatory").astype(np.float64)
        data["diag1_is_resp"] = (d1_g == "respiratory").astype(np.float64)
        data["diag1_is_diab"] = (d1_g == "diabetes").astype(np.float64)

        # CMS HRRP Target Admission flags (high national readmission penalty conditions)
        is_hrrp_chf = np.char.startswith(d1_s, "428").astype(np.float64)
        is_hrrp_ami = np.char.startswith(d1_s, "410").astype(np.float64)
        is_hrrp_pna = _has_prefix(d1_s, d1_s, d1_s, ("480", "481", "482", "483", "484", "485", "486"))
        is_hrrp_copd = _has_prefix(d1_s, d1_s, d1_s, ("491", "492", "496"))
        is_hrrp_stroke = _has_prefix(d1_s, d1_s, d1_s, ("430", "431", "432", "433", "434", "435", "436", "437", "438"))
        data["is_hrrp_chf"] = is_hrrp_chf
        data["is_hrrp_ami"] = is_hrrp_ami
        data["is_hrrp_pna"] = is_hrrp_pna
        data["is_hrrp_copd"] = is_hrrp_copd
        data["is_hrrp_stroke"] = is_hrrp_stroke
        data["is_hrrp_any"] = np.maximum.reduce([is_hrrp_chf, is_hrrp_ami, is_hrrp_pna, is_hrrp_copd, is_hrrp_stroke])

        # Diagnostic synergies
        data["ckm_stage_4"] = (has_diab * has_chf * has_renal_failure).astype(np.float64)
        data["cardiorenal_syndrome"] = (has_chf * has_renal).astype(np.float64)
        data["respiratory_decompensation"] = (has_resp * has_circ).astype(np.float64)
        data["sepsis_with_renal"] = (has_sepsis * has_renal).astype(np.float64)
        data["diabetic_foot_ulcer"] = (has_diab * has_skin_ulcer).astype(np.float64)
        data["diab_with_anemia"] = (has_diab * has_anemia).astype(np.float64)

    extra_numeric.extend([
        "has_diabetes_diag",
        "has_chf_diag",
        "has_circ_diag",
        "has_resp_diag",
        "has_renal_diag",
        "has_neoplasm_diag",
        "has_hypertension",
        "has_ckd",
        "has_renal_failure",
        "has_copd",
        "has_pneumonia",
        "has_sepsis",
        "has_fluid_elec",
        "has_ketoacidosis",
        "has_diabetes_with_complications",
        "has_cad",
        "has_arrhythmia",
        "has_pvd",
        "has_stroke",
        "has_anemia",
        "has_uti",
        "has_skin_ulcer",
        "has_ckm_syndrome",
        "has_chf_and_ckd",
        "has_cardiopulmonary",
        "has_diab_htn_ckd",
        "n_distinct_diag_groups",
        "comorbidity_score",
        "diag1_is_chf",
        "diag1_is_circ",
        "diag1_is_resp",
        "diag1_is_diab",
    ])

    # 3. Utilization, clinical intensity, and stay complexity features
    for data in (train_data, eval_data):
        inpat = np.nan_to_num(data["number_inpatient"].astype(np.float64), nan=0.0)
        emerg = np.nan_to_num(data["number_emergency"].astype(np.float64), nan=0.0)
        outpat = np.nan_to_num(data["number_outpatient"].astype(np.float64), nan=0.0)
        hosp_days = np.maximum(np.nan_to_num(data["time_in_hospital"].astype(np.float64), nan=1.0), 1.0)
        num_meds = np.nan_to_num(data["num_medications"].astype(np.float64), nan=0.0)
        num_labs = np.nan_to_num(data["num_lab_procedures"].astype(np.float64), nan=0.0)
        num_procs = np.nan_to_num(data["num_procedures"].astype(np.float64), nan=0.0)
        num_diags = np.maximum(np.nan_to_num(data["number_diagnoses"].astype(np.float64), nan=1.0), 1.0)

        total_visits = inpat + emerg + outpat
        data["total_prior_visits"] = total_visits
        data["has_prior_inpatient"] = (inpat > 0.0).astype(np.float64)
        data["inpatient_multi"] = (inpat >= 2.0).astype(np.float64)
        data["acute_prior_visits"] = inpat + emerg
        data["inpatient_ratio"] = (inpat + 0.1) / (total_visits + 1.0)
        data["emergency_ratio"] = (emerg + 0.1) / (total_visits + 1.0)
        data["acute_ratio"] = (inpat + emerg + 0.1) / (total_visits + 1.0)

        data["log_inpatient"] = np.log1p(inpat)
        data["log_emergency"] = np.log1p(emerg)
        data["log_outpatient"] = np.log1p(outpat)
        data["log_total_visits"] = np.log1p(total_visits)

        data["med_per_day"] = num_meds / hosp_days
        data["labs_per_day"] = num_labs / hosp_days
        data["procs_per_day"] = num_procs / hosp_days
        data["meds_per_diag"] = num_meds / num_diags
        data["labs_per_med"] = num_labs / (num_meds + 1.0)
        data["clinical_intensity"] = np.log1p(num_labs) + np.log1p(num_meds) + np.log1p(num_procs * 2.0)

        data["ratio_inpatient_to_stay"] = inpat / (hosp_days + 1.0)
        data["ratio_visits_to_stay"] = total_visits / hosp_days
        data["ratio_meds_to_procedures"] = num_meds / (num_procs + 1.0)
        data["ratio_labs_to_procedures"] = num_labs / (num_procs + 1.0)
        data["frequent_flyer"] = ((inpat >= 2.0) | (emerg >= 3.0)).astype(np.float64)

        # Non-linear prior utilization steps & revolving door dynamics
        data["inpat_sq"] = inpat ** 2
        data["inpat_is_0"] = (inpat == 0.0).astype(np.float64)
        data["inpat_is_1"] = (inpat == 1.0).astype(np.float64)
        data["inpat_is_2plus"] = (inpat >= 2.0).astype(np.float64)
        data["inpat_is_3plus"] = (inpat >= 3.0).astype(np.float64)
        data["emerg_is_0"] = (emerg == 0.0).astype(np.float64)
        data["emerg_is_1plus"] = (emerg >= 1.0).astype(np.float64)
        data["outpat_is_0"] = (outpat == 0.0).astype(np.float64)
        data["revolving_door_ratio"] = inpat / (inpat + outpat + 1.0)
        data["revolving_door_index"] = inpat * 3.0 + emerg * 1.5 + (1.0 - (outpat > 0.0).astype(np.float64)) * inpat
        data["acute_care_density"] = (inpat * 2.0 + emerg) / (hosp_days + 1.0)
        data["high_utilizer"] = ((inpat >= 2.0) | (emerg >= 2.0) | (total_visits >= 5.0)).astype(np.float64)
        data["zero_prior_visits"] = (total_visits == 0.0).astype(np.float64)

        # Age parsing & subgroup protection
        age_num = np.array([_parse_age(a) for a in data["age"]], dtype=np.float64)
        data["age_num"] = age_num
        data["is_elderly"] = (age_num >= 75.0).astype(np.float64)
        data["is_very_elderly"] = (age_num >= 85.0).astype(np.float64)
        data["is_young"] = (age_num < 50.0).astype(np.float64)

        # Charlson Comorbidity Index
        data["charlson_score"] = _compute_charlson(data["diag_1"], data["diag_2"], data["diag_3"], age_num)

        # Discharge details
        disp = data["discharge_disposition_id"]
        discharged_home = (disp == "1").astype(np.float64)
        discharged_facility = np.isin(disp, ["3", "4", "5", "6", "22", "23", "24", "25"]).astype(np.float64)
        discharged_to_acute_hospital = np.isin(disp, ["2", "22"]).astype(np.float64)
        discharged_home_health = (disp == "6").astype(np.float64)
        discharged_snf = (disp == "3").astype(np.float64)
        discharged_icf = (disp == "4").astype(np.float64)
        discharged_ama = (disp == "7").astype(np.float64)
        is_high_risk_discharge = np.isin(disp, ["2", "3", "4", "5", "7", "22", "23", "24", "28"]).astype(np.float64)

        data["discharged_home"] = discharged_home
        data["discharged_facility"] = discharged_facility
        data["discharged_to_acute_hospital"] = discharged_to_acute_hospital
        data["discharged_home_health"] = discharged_home_health
        data["discharged_snf"] = discharged_snf
        data["discharged_icf"] = discharged_icf
        data["discharged_snf_icf"] = np.isin(disp, ["3", "4", "24"]).astype(np.float64)
        data["discharged_ama"] = discharged_ama
        data["discharge_not_home"] = (disp != "1").astype(np.float64)
        data["is_high_risk_discharge"] = is_high_risk_discharge
        data["high_risk_disp_with_prior_inpat"] = (is_high_risk_discharge * (inpat > 0.0)).astype(np.float64)
        data["high_risk_disp_with_elderly"] = (is_high_risk_discharge * data["is_elderly"]).astype(np.float64)

        # Very elderly specific subgroup protective interactions
        data["very_elderly_disp_facility"] = data["is_very_elderly"] * discharged_facility
        data["very_elderly_disp_home"] = data["is_very_elderly"] * discharged_home
        data["very_elderly_prior_inpat"] = data["is_very_elderly"] * (inpat > 0.0).astype(np.float64)
        data["very_elderly_chf"] = data["is_very_elderly"] * data["has_chf_diag"]
        data["very_elderly_renal"] = data["is_very_elderly"] * data["has_renal_diag"]
        data["very_elderly_med_burden"] = data["is_very_elderly"] * (num_meds >= 15.0).astype(np.float64)

        # Admission details
        adm_type = data["admission_type_id"]
        adm_emergency = np.isin(adm_type, ["1", "2", "7"]).astype(np.float64)
        data["admission_emergency"] = adm_emergency
        data["admission_elective"] = (adm_type == "3").astype(np.float64)
        data["adm_acute_to_facility"] = (adm_emergency * discharged_facility).astype(np.float64)

        adm_src = data["admission_source_id"]
        data["admission_source_er"] = (adm_src == "7").astype(np.float64)
        data["admission_source_transfer"] = np.isin(adm_src, ["4", "5", "6", "10", "18", "22", "25"]).astype(np.float64)
        data["admission_source_snf_transfer"] = (adm_src == "5").astype(np.float64)
        data["admission_source_referral"] = np.isin(adm_src, ["1", "2", "3"]).astype(np.float64)

        # Payer code clinical indicators
        payer = data["payer_code"]
        data["payer_is_medicare"] = (payer == "MC").astype(np.float64)
        data["payer_is_medicaid"] = (payer == "MD").astype(np.float64)
        data["payer_is_selfpay"] = np.isin(payer, ["SP", "UN"]).astype(np.float64)
        data["payer_is_missing"] = np.isin(payer, [None, "nan", "None", "?", ""]).astype(np.float64)

        # LACE Readmission Risk Index surrogate
        l_pts = np.where(hosp_days <= 1.0, 1.0,
                np.where(hosp_days == 2.0, 2.0,
                np.where(hosp_days == 3.0, 3.0,
                np.where(hosp_days <= 6.0, 4.0,
                np.where(hosp_days <= 13.0, 5.0, 7.0)))))
        a_pts = np.where(adm_emergency > 0.0, 3.0, 0.0)
        c_pts = (
            np.minimum(num_diags, 6.0) * 0.5 +
            data["has_chf_diag"] * 2.0 +
            data["has_renal_diag"] * 1.5 +
            data["has_diabetes_with_complications"] * 1.0 +
            data["has_ckd"] * 1.5 +
            data["has_copd"] * 1.0 +
            data["has_cad"] * 1.0 +
            data["has_stroke"] * 1.0
        )
        e_pts = np.minimum(inpat + emerg, 4.0)
        data["lace_surrogate"] = l_pts + a_pts + c_pts + e_pts

        # Interactions
        data["age_times_diagnoses"] = age_num * num_diags
        data["age_times_comorbidity"] = age_num * data["comorbidity_score"]
        data["inpatient_times_comorbidity"] = inpat * data["comorbidity_score"]
        data["inpatient_times_lace"] = inpat * data["lace_surrogate"]
        data["age_times_inpatient"] = age_num * inpat

        # Lab indicators & clinical inertia
        a1c_tested = np.isin(data["A1Cresult"], ["Norm", ">7", ">8"]).astype(np.float64)
        a1c_high = np.isin(data["A1Cresult"], [">7", ">8"]).astype(np.float64)
        a1c_very_high = (data["A1Cresult"] == ">8").astype(np.float64)
        a1c_normal = (data["A1Cresult"] == "Norm").astype(np.float64)
        med_changed = (data["change"] == "Ch").astype(np.float64)
        no_med_change = (data["change"] == "No").astype(np.float64)

        data["a1c_tested"] = a1c_tested
        data["a1c_high"] = a1c_high
        data["a1c_very_high"] = a1c_very_high
        data["a1c_normal"] = a1c_normal
        data["a1c_high_no_change"] = a1c_high * no_med_change
        data["a1c_high_with_change"] = a1c_high * med_changed
        data["a1c_not_tested_with_change"] = (1.0 - a1c_tested) * med_changed

        glu_tested = np.isin(data["max_glu_serum"], ["Norm", ">200", ">300"]).astype(np.float64)
        glu_high = np.isin(data["max_glu_serum"], [">200", ">300"]).astype(np.float64)
        glu_very_high = (data["max_glu_serum"] == ">300").astype(np.float64)
        data["glu_tested"] = glu_tested
        data["glu_high"] = glu_high
        data["glu_very_high"] = glu_very_high
        data["glu_high_no_change"] = glu_high * no_med_change
        data["glu_high_with_change"] = glu_high * med_changed

        data["a1c_score"] = np.array([_map_a1c(v) for v in data["A1Cresult"]], dtype=np.float64)
        data["glu_score"] = np.array([_map_glu(v) for v in data["max_glu_serum"]], dtype=np.float64)

        # Medication summary across the 23 medication slice
        med_cols = [
            "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
            "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
            "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
            "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
        ]
        n_rows = len(disp)
        active_cnt = np.zeros(n_rows, dtype=np.float64)
        changes_cnt = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        down_cnt = np.zeros(n_rows, dtype=np.float64)

        for m_col in med_cols:
            if m_col in data:
                vals = data[m_col]
                active_cnt += np.isin(vals, ["Steady", "Up", "Down"]).astype(np.float64)
                changes_cnt += np.isin(vals, ["Up", "Down"]).astype(np.float64)
                up_cnt += (vals == "Up").astype(np.float64)
                down_cnt += (vals == "Down").astype(np.float64)

        data["num_active_meds"] = active_cnt
        data["med_changes_sum"] = changes_cnt
        data["num_meds_up"] = up_cnt
        data["num_meds_down"] = down_cnt
        data["change_flag"] = med_changed
        data["diabetes_med_flag"] = (data["diabetesMed"] == "Yes").astype(np.float64)
        insulin_act = (data["insulin"] != "No").astype(np.float64)
        data["insulin_active"] = insulin_act
        data["insulin_is_up"] = (data["insulin"] == "Up").astype(np.float64)
        data["insulin_is_down"] = (data["insulin"] == "Down").astype(np.float64)
        data["insulin_titrated"] = np.isin(data["insulin"], ["Up", "Down"]).astype(np.float64)

        metformin_active = (data["metformin"] != "No").astype(np.float64) if "metformin" in data else np.zeros(n_rows, dtype=np.float64)
        data["insulin_and_metformin"] = (insulin_act * metformin_active).astype(np.float64)

        # Regimen classifications
        sulfo_cols = ["glipizide", "glyburide", "glimepiride", "chlorpropamide", "tolbutamide", "acetohexamide", "tolazamide"]
        has_sulfo = np.zeros(n_rows, dtype=np.float64)
        for sc in sulfo_cols:
            if sc in data:
                has_sulfo = np.maximum(has_sulfo, (data[sc] != "No").astype(np.float64))
        data["sulfonylurea_active"] = has_sulfo

        tzd_cols = ["pioglitazone", "rosiglitazone", "troglitazone"]
        has_tzd = np.zeros(n_rows, dtype=np.float64)
        for tc in tzd_cols:
            if tc in data:
                has_tzd = np.maximum(has_tzd, (data[tc] != "No").astype(np.float64))
        data["tzd_active"] = has_tzd
        data["tzd_with_chf"] = has_tzd * data["has_chf_diag"]

        # High-risk medication regimen combinations
        data["insulin_and_sulfo"] = insulin_act * has_sulfo
        data["insulin_and_tzd"] = insulin_act * has_tzd
        data["metformin_and_renal"] = metformin_active * data["has_renal_diag"]
        data["insulin_up_with_inpat"] = data["insulin_is_up"] * (inpat > 0.0).astype(np.float64)
        data["elderly_hypoglycemia_risk"] = data["is_elderly"] * (insulin_act + has_sulfo)
        data["med_changes_density"] = changes_cnt / hosp_days

        num_oral = np.maximum(0.0, active_cnt - insulin_act)
        data["num_oral_meds"] = num_oral
        data["oral_meds_count"] = num_oral
        data["insulin_only"] = ((insulin_act > 0.0) & (num_oral == 0.0)).astype(np.float64)
        data["oral_only"] = ((insulin_act == 0.0) & (num_oral > 0.0)).astype(np.float64)
        data["insulin_and_oral"] = ((insulin_act > 0.0) & (num_oral > 0.0)).astype(np.float64)
        data["no_diabetes_meds"] = (active_cnt == 0.0).astype(np.float64)

        data["polytherapy_3plus"] = (active_cnt >= 3.0).astype(np.float64)
        data["high_diagnoses"] = (num_diags > 8.0).astype(np.float64)

        weight_val = data["weight"]
        data["weight_not_missing"] = np.array([1.0 if (w is not None and str(w).strip() != "" and str(w).strip() != "?") else 0.0 for w in weight_val], dtype=np.float64)

    extra_numeric.extend([
        "total_prior_visits",
        "has_prior_inpatient",
        "inpatient_multi",
        "acute_prior_visits",
        "inpatient_ratio",
        "emergency_ratio",
        "acute_ratio",
        "log_inpatient",
        "log_emergency",
        "log_outpatient",
        "log_total_visits",
        "med_per_day",
        "labs_per_day",
        "procs_per_day",
        "meds_per_diag",
        "labs_per_med",
        "clinical_intensity",
        "ratio_inpatient_to_stay",
        "ratio_visits_to_stay",
        "ratio_meds_to_procedures",
        "ratio_labs_to_procedures",
        "frequent_flyer",
        "inpat_sq",
        "inpat_is_0",
        "inpat_is_1",
        "inpat_is_2plus",
        "inpat_is_3plus",
        "emerg_is_0",
        "emerg_is_1plus",
        "outpat_is_0",
        "revolving_door_ratio",
        "revolving_door_index",
        "acute_care_density",
        "high_utilizer",
        "zero_prior_visits",
        "age_num",
        "is_elderly",
        "is_very_elderly",
        "is_young",
        "charlson_score",
        "discharged_home",
        "discharged_facility",
        "discharged_to_acute_hospital",
        "discharged_home_health",
        "discharged_snf",
        "discharged_icf",
        "discharged_snf_icf",
        "discharged_ama",
        "discharge_not_home",
        "is_high_risk_discharge",
        "high_risk_disp_with_prior_inpat",
        "high_risk_disp_with_elderly",
        "admission_emergency",
        "admission_elective",
        "adm_acute_to_facility",
        "admission_source_er",
        "admission_source_transfer",
        "admission_source_snf_transfer",
        "admission_source_referral",
        "payer_is_medicare",
        "payer_is_medicaid",
        "payer_is_selfpay",
        "payer_is_missing",
        "lace_surrogate",
        "a1c_tested",
        "a1c_high",
        "a1c_very_high",
        "a1c_normal",
        "a1c_high_no_change",
        "a1c_high_with_change",
        "a1c_not_tested_with_change",
        "glu_tested",
        "glu_high",
        "glu_very_high",
        "glu_high_no_change",
        "glu_high_with_change",
        "a1c_score",
        "glu_score",
        "num_active_meds",
        "med_changes_sum",
        "num_meds_up",
        "num_meds_down",
        "change_flag",
        "diabetes_med_flag",
        "insulin_active",
        "insulin_is_up",
        "insulin_is_down",
        "insulin_titrated",
        "insulin_and_metformin",
        "sulfonylurea_active",
        "tzd_active",
        "tzd_with_chf",
        "insulin_and_sulfo",
        "insulin_and_tzd",
        "metformin_and_renal",
        "insulin_up_with_inpat",
        "elderly_hypoglycemia_risk",
        "med_changes_density",
        "oral_meds_count",
        "num_oral_meds",
        "insulin_only",
        "oral_only",
        "insulin_and_oral",
        "no_diabetes_meds",
        "polytherapy_3plus",
        "high_diagnoses",
        "weight_not_missing",
    ])

    # 3.5 Stratified Target Encoding & Frequency Encoding of High Cardinality Features
    te_cols = [
        "diag_1",
        "diag_2",
        "diag_3",
        "diag_1_3digit",
        "diag_2_3digit",
        "diag_3_3digit",
        "medical_specialty",
        "discharge_disposition_id",
        "admission_source_id",
        "payer_code",
        "admission_type_id",
    ]
    for col in te_cols:
        te_train, te_eval = _target_encode(
            train_data[col],
            eval_data[col],
            y_train,
            n_folds=5,
            m_smoothing=18.0,
        )
        te_name = f"{col}_te"
        train_data[te_name] = te_train
        eval_data[te_name] = te_eval
        extra_numeric.append(te_name)

    # Position-invariant diagnosis risk summaries
    train_data["diag_te_max"] = np.maximum(train_data["diag_1_te"], np.maximum(train_data["diag_2_te"], train_data["diag_3_te"]))
    eval_data["diag_te_max"] = np.maximum(eval_data["diag_1_te"], np.maximum(eval_data["diag_2_te"], eval_data["diag_3_te"]))

    train_data["diag_te_mean"] = (train_data["diag_1_te"] + train_data["diag_2_te"] + train_data["diag_3_te"]) / 3.0
    eval_data["diag_te_mean"] = (eval_data["diag_1_te"] + eval_data["diag_2_te"] + eval_data["diag_3_te"]) / 3.0

    train_data["diag3d_te_max"] = np.maximum(train_data["diag_1_3digit_te"], np.maximum(train_data["diag_2_3digit_te"], train_data["diag_3_3digit_te"]))
    eval_data["diag3d_te_max"] = np.maximum(eval_data["diag_1_3digit_te"], np.maximum(eval_data["diag_2_3digit_te"], eval_data["diag_3_3digit_te"]))

    train_data["diag3d_te_mean"] = (train_data["diag_1_3digit_te"] + train_data["diag_2_3digit_te"] + train_data["diag_3_3digit_te"]) / 3.0
    eval_data["diag3d_te_mean"] = (eval_data["diag_1_3digit_te"] + eval_data["diag_2_3digit_te"] + eval_data["diag_3_3digit_te"]) / 3.0

    # Pooled Cross-Diagnosis Target Encoding (combining all diagnosis positions)
    (tr_p1, tr_p2, tr_p3), (ev_p1, ev_p2, ev_p3) = _pooled_diagnosis_target_encode(
        train_data["diag_1"], train_data["diag_2"], train_data["diag_3"],
        eval_data["diag_1"], eval_data["diag_2"], eval_data["diag_3"],
        y_train, n_folds=5, m_smoothing=20.0,
    )
    train_data["pooled_diag1_te"] = tr_p1
    train_data["pooled_diag2_te"] = tr_p2
    train_data["pooled_diag3_te"] = tr_p3
    eval_data["pooled_diag1_te"] = ev_p1
    eval_data["pooled_diag2_te"] = ev_p2
    eval_data["pooled_diag3_te"] = ev_p3

    train_data["pooled_diag_max"] = np.maximum(tr_p1, np.maximum(tr_p2, tr_p3))
    eval_data["pooled_diag_max"] = np.maximum(ev_p1, np.maximum(ev_p2, ev_p3))
    train_data["pooled_diag_mean"] = (tr_p1 + tr_p2 + tr_p3) / 3.0
    eval_data["pooled_diag_mean"] = (ev_p1 + ev_p2 + ev_p3) / 3.0
    train_data["pooled_diag_sum"] = tr_p1 + tr_p2 + tr_p3
    eval_data["pooled_diag_sum"] = ev_p1 + ev_p2 + ev_p3

    (tr_p3d1, tr_p3d2, tr_p3d3), (ev_p3d1, ev_p3d2, ev_p3d3) = _pooled_diagnosis_target_encode(
        train_data["diag_1_3digit"], train_data["diag_2_3digit"], train_data["diag_3_3digit"],
        eval_data["diag_1_3digit"], eval_data["diag_2_3digit"], eval_data["diag_3_3digit"],
        y_train, n_folds=5, m_smoothing=20.0,
    )
    train_data["pooled_d3d_1_te"] = tr_p3d1
    train_data["pooled_d3d_2_te"] = tr_p3d2
    train_data["pooled_d3d_3_te"] = tr_p3d3
    eval_data["pooled_d3d_1_te"] = ev_p3d1
    eval_data["pooled_d3d_2_te"] = ev_p3d2
    eval_data["pooled_d3d_3_te"] = ev_p3d3

    train_data["pooled_d3d_max"] = np.maximum(tr_p3d1, np.maximum(tr_p3d2, tr_p3d3))
    eval_data["pooled_d3d_max"] = np.maximum(ev_p3d1, np.maximum(ev_p3d2, ev_p3d3))
    train_data["pooled_d3d_mean"] = (tr_p3d1 + tr_p3d2 + tr_p3d3) / 3.0
    eval_data["pooled_d3d_mean"] = (ev_p3d1 + ev_p3d2 + ev_p3d3) / 3.0
    train_data["pooled_d3d_sum"] = tr_p3d1 + tr_p3d2 + tr_p3d3
    eval_data["pooled_d3d_sum"] = ev_p3d1 + ev_p3d2 + ev_p3d3

    extra_numeric.extend([
        "diag_te_max", "diag_te_mean", "diag3d_te_max", "diag3d_te_mean",
        "pooled_diag1_te", "pooled_diag2_te", "pooled_diag3_te",
        "pooled_diag_max", "pooled_diag_mean", "pooled_diag_sum",
        "pooled_d3d_1_te", "pooled_d3d_2_te", "pooled_d3d_3_te",
        "pooled_d3d_max", "pooled_d3d_mean", "pooled_d3d_sum",
    ])

    # Frequency encoding for common clinical categoricals
    for col in ("diag_1_3digit", "medical_specialty", "discharge_disposition_id"):
        f_tr, f_ev = _frequency_encode(train_data[col], eval_data[col])
        f_name = f"{col}_freq"
        train_data[f_name] = f_tr
        eval_data[f_name] = f_ev
        extra_numeric.append(f_name)

    # Hierarchical Empirical Bayes diagnosis shrinkage encoding
    for pos in ("1", "2", "3"):
        h_tr, h_ev = _hierarchical_diagnosis_te(
            train_data[f"diag_{pos}"],
            train_data[f"diag_{pos}_3digit"],
            train_data[f"diag_{pos}_group"],
            eval_data[f"diag_{pos}"],
            eval_data[f"diag_{pos}_3digit"],
            eval_data[f"diag_{pos}_group"],
            y_train,
            n_folds=5,
            m_grp=30.0,
            m_3d=15.0,
            m_code=8.0,
        )
        h_name = f"hier_diag{pos}_te"
        train_data[h_name] = h_tr
        eval_data[h_name] = h_ev
        extra_numeric.append(h_name)

    train_data["hier_diag_max"] = np.maximum(train_data["hier_diag1_te"], np.maximum(train_data["hier_diag2_te"], train_data["hier_diag3_te"]))
    eval_data["hier_diag_max"] = np.maximum(eval_data["hier_diag1_te"], np.maximum(eval_data["hier_diag2_te"], eval_data["hier_diag3_te"]))
    train_data["hier_diag_mean"] = (train_data["hier_diag1_te"] + train_data["hier_diag2_te"] + train_data["hier_diag3_te"]) / 3.0
    eval_data["hier_diag_mean"] = (eval_data["hier_diag1_te"] + eval_data["hier_diag2_te"] + eval_data["hier_diag3_te"]) / 3.0
    extra_numeric.extend(["hier_diag_max", "hier_diag_mean"])

    # High-impact joint clinical interaction target encodings
    cross_specs = [
        ("cross_diag1_disp", "diag_1_3digit", "discharge_disposition_id"),
        ("cross_disp_source", "discharge_disposition_id", "admission_source_id"),
        ("cross_diag1grp_disp", "diag_1_group", "discharge_disposition_id"),
        ("cross_disp_admtype", "discharge_disposition_id", "admission_type_id"),
    ]
    for c_name, col_a, col_b in cross_specs:
        c_tr_vals = np.array([f"{a}_{b}" for a, b in zip(train_data[col_a], train_data[col_b])], dtype=object)
        c_ev_vals = np.array([f"{a}_{b}" for a, b in zip(eval_data[col_a], eval_data[col_b])], dtype=object)
        c_tr_te, c_ev_te = _target_encode(c_tr_vals, c_ev_vals, y_train, n_folds=5, m_smoothing=15.0)
        c_full_name = f"{c_name}_te"
        train_data[c_full_name] = c_tr_te
        eval_data[c_full_name] = c_ev_te
        extra_numeric.append(c_full_name)

    # Add Bayesian smoothed log-odds diagnosis risk stream
    tr_dlog, ev_dlog = _compute_diagnosis_logodds(
        [train_data["diag_1"], train_data["diag_2"], train_data["diag_3"]],
        [eval_data["diag_1"], eval_data["diag_2"], eval_data["diag_3"]],
        y_train, n_folds=5, smoothing=15.0,
    )
    tr_3log, ev_3log = _compute_diagnosis_logodds(
        [train_data["diag_1_3digit"], train_data["diag_2_3digit"], train_data["diag_3_3digit"]],
        [eval_data["diag_1_3digit"], eval_data["diag_2_3digit"], eval_data["diag_3_3digit"]],
        y_train, n_folds=5, smoothing=15.0,
    )
    for k in ("sum", "max", "prim"):
        train_data[f"diag_logodds_{k}"] = tr_dlog[k]
        eval_data[f"diag_logodds_{k}"] = ev_dlog[k]
        extra_numeric.append(f"diag_logodds_{k}")

        train_data[f"diag3d_logodds_{k}"] = tr_3log[k]
        eval_data[f"diag3d_logodds_{k}"] = ev_3log[k]
        extra_numeric.append(f"diag3d_logodds_{k}")

    # 4. Fit TabularEncoder on all task + derived features
    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(
        task,
        train_data,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_eval = encoder.transform(eval_data)

    # 5. Multi-architecture ensemble with structural split diversity
    models = [
        # LGBM 1: Balanced capacity tree (depth 6, 31 leaves)
        lgb.LGBMClassifier(
            n_estimators=450,
            learning_rate=0.024,
            num_leaves=31,
            max_depth=6,
            min_child_samples=35,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.3,
            reg_lambda=4.0,
            min_data_per_group=20,
            cat_smooth=15.0,
            cat_l2=8.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        # LGBM 2: Shallow regularized tree (depth 4, 15 leaves) - high stability on rare elderly subgroups
        lgb.LGBMClassifier(
            n_estimators=480,
            learning_rate=0.020,
            num_leaves=15,
            max_depth=4,
            min_child_samples=55,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_alpha=1.0,
            reg_lambda=8.0,
            min_data_per_group=30,
            cat_smooth=20.0,
            cat_l2=12.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        # LGBM 3: Wide feature fraction, moderate depth (depth 5, 24 leaves)
        lgb.LGBMClassifier(
            n_estimators=440,
            learning_rate=0.025,
            num_leaves=24,
            max_depth=5,
            min_child_samples=40,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=3.0,
            min_data_per_group=15,
            cat_smooth=12.0,
            cat_l2=5.0,
            random_state=456,
            n_jobs=1,
            verbose=-1,
        ),
        # LGBM 4: Extremely Randomized Trees (extra_trees=True) - orthogonal split diversity
        lgb.LGBMClassifier(
            n_estimators=450,
            learning_rate=0.024,
            num_leaves=31,
            max_depth=6,
            min_child_samples=30,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.5,
            reg_lambda=5.0,
            extra_trees=True,
            min_data_per_group=20,
            cat_smooth=15.0,
            cat_l2=8.0,
            random_state=789,
            n_jobs=1,
            verbose=-1,
        ),
        # LGBM 5: Balanced medium-depth with large min_child_samples (depth 5, 28 leaves)
        lgb.LGBMClassifier(
            n_estimators=440,
            learning_rate=0.022,
            num_leaves=28,
            max_depth=5,
            min_child_samples=45,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_alpha=0.6,
            reg_lambda=6.0,
            min_data_per_group=25,
            cat_smooth=18.0,
            cat_l2=10.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    cat_indices = encoder.categorical_indices
    # Train streamlined diverse ensemble
    m1 = lgb.LGBMClassifier(
        n_estimators=480, learning_rate=0.024, num_leaves=31, max_depth=6,
        min_child_samples=35, subsample=0.8, subsample_freq=1, colsample_bytree=0.65,
        reg_alpha=0.5, reg_lambda=5.0, min_data_per_group=20, cat_smooth=15.0,
        cat_l2=8.0, random_state=42, n_jobs=1, verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=cat_indices)

    m2 = lgb.LGBMClassifier(
        n_estimators=480, learning_rate=0.024, num_leaves=31, max_depth=6,
        min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.65,
        reg_alpha=0.5, reg_lambda=5.0, extra_trees=True, min_data_per_group=20,
        cat_smooth=15.0, cat_l2=8.0, random_state=789, n_jobs=1, verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=cat_indices)

    m3 = lgb.LGBMClassifier(
        n_estimators=480, learning_rate=0.020, num_leaves=15, max_depth=4,
        min_child_samples=55, subsample=0.75, subsample_freq=1, colsample_bytree=0.55,
        reg_alpha=1.0, reg_lambda=8.0, min_data_per_group=30, cat_smooth=20.0,
        cat_l2=12.0, random_state=123, n_jobs=1, verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=cat_indices)

    preds = 0.40 * m1.predict_proba(x_eval)[:, 1] + 0.35 * m2.predict_proba(x_eval)[:, 1] + 0.25 * m3.predict_proba(x_eval)[:, 1]

    if XGB_AVAILABLE:
        try:
            m_xgb = xgb.XGBClassifier(
                n_estimators=380, learning_rate=0.025, max_depth=5, min_child_weight=30,
                subsample=0.8, colsample_bytree=0.65, reg_alpha=0.5, reg_lambda=5.0,
                tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=1,
            )
            m_xgb.fit(x_train, y_train)
            preds = 0.82 * preds + 0.18 * m_xgb.predict_proba(x_eval)[:, 1]
        except Exception:
            pass

    return preds

    # Legacy loop disabled
    for model in []:
        model.fit(x_train, y_train, categorical_feature=cat_indices)
        preds += 0.175 * model.predict_proba(x_eval)[:, 1]

    # Model 6: XGBoost histogram classifier for cross-algorithm gradient boosting diversity
    xgb_used = False
    if XGB_AVAILABLE:
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=380,
                learning_rate=0.026,
                max_depth=5,
                min_child_weight=30,
                subsample=0.80,
                colsample_bytree=0.60,
                reg_alpha=0.5,
                reg_lambda=5.0,
                tree_method="hist",
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
            )
            xgb_model.fit(x_train, y_train)
            preds += 0.125 * xgb_model.predict_proba(x_eval)[:, 1]
            xgb_used = True
        except Exception:
            xgb_used = False

    if not xgb_used:
        preds = preds / 0.875

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
