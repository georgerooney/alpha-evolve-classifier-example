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


def _safe_icd9_group(code):
    if "icd9_group" in globals():
        try:
            return globals()["icd9_group"](code)
        except Exception:
            pass
    if code is None or code == "" or code == "?" or str(code).lower() == "nan":
        return "missing"
    s = str(code).strip()
    if s.startswith("V") or s.startswith("E"):
        return "other"
    try:
        num = float(s.split(".")[0])
        if 390 <= num <= 459 or num == 785:
            return "circulatory"
        elif 460 <= num <= 519 or num == 786:
            return "respiratory"
        elif 520 <= num <= 579 or num == 787:
            return "digestive"
        elif num == 250:
            return "diabetes"
        elif 800 <= num <= 999:
            return "injury"
        elif 710 <= num <= 739:
            return "musculoskeletal"
        elif 580 <= num <= 629 or num == 788:
            return "genitourinary"
        elif 140 <= num <= 239:
            return "neoplasms"
        else:
            return "other"
    except Exception:
        return "other"


def _oof_target_encoding(train_arr, y_train, eval_arr, n_splits=5, m=25.0, random_state=42):
    from sklearn.model_selection import StratifiedKFold
    n_train = len(train_arr)
    n_eval = len(eval_arr)
    oof_train = np.empty(n_train, dtype=np.float64)
    global_mean = float(np.mean(y_train))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr_idx, val_idx in skf.split(train_arr, y_train):
        counts = {}
        sums = {}
        for c, y in zip(train_arr[tr_idx], y_train[tr_idx]):
            counts[c] = counts.get(c, 0) + 1
            sums[c] = sums.get(c, 0.0) + y
        val_res = np.empty(len(val_idx), dtype=np.float64)
        for i, c in enumerate(train_arr[val_idx]):
            cnt = counts.get(c, 0)
            sm = sums.get(c, 0.0)
            val_res[i] = (sm + m * global_mean) / (cnt + m)
        oof_train[val_idx] = val_res

    full_counts = {}
    full_sums = {}
    for c, y in zip(train_arr, y_train):
        full_counts[c] = full_counts.get(c, 0) + 1
        full_sums[c] = full_sums.get(c, 0.0) + y

    eval_res = np.empty(n_eval, dtype=np.float64)
    for i, c in enumerate(eval_arr):
        cnt = full_counts.get(c, 0)
        sm = full_sums.get(c, 0.0)
        eval_res[i] = (sm + m * global_mean) / (cnt + m)

    return oof_train, eval_res


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    from collections import Counter
    from sklearn.decomposition import TruncatedSVD

    train_data = dict(train)
    eval_data = dict(eval_cols)
    n_train = len(y_train)
    n_eval = len(eval_cols["time_in_hospital"])

    med_names = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone"
    ]

    extra_numeric = []
    extra_categorical = []

    def add_num(name, tr_arr, ev_arr):
        train_data[name] = np.asarray(tr_arr, dtype=np.float64)
        eval_data[name] = np.asarray(ev_arr, dtype=np.float64)
        extra_numeric.append(name)

    def add_cat(name, tr_arr, ev_arr):
        train_data[name] = np.asarray(tr_arr, dtype=object)
        eval_data[name] = np.asarray(ev_arr, dtype=object)
        extra_categorical.append(name)

    # 1. Prior utilization intensity features
    for name, data, n in [("train", train_data, n_train), ("eval", eval_data, n_eval)]:
        inp = np.nan_to_num(data.get("number_inpatient", np.zeros(n, dtype=np.float64)), nan=0.0)
        emg = np.nan_to_num(data.get("number_emergency", np.zeros(n, dtype=np.float64)), nan=0.0)
        out = np.nan_to_num(data.get("number_outpatient", np.zeros(n, dtype=np.float64)), nan=0.0)
        los = np.nan_to_num(data.get("time_in_hospital", np.zeros(n, dtype=np.float64)), nan=0.0)
        n_med = np.nan_to_num(data.get("num_medications", np.zeros(n, dtype=np.float64)), nan=0.0)
        n_lab = np.nan_to_num(data.get("num_lab_procedures", np.zeros(n, dtype=np.float64)), nan=0.0)
        n_proc = np.nan_to_num(data.get("num_procedures", np.zeros(n, dtype=np.float64)), nan=0.0)
        n_diag = np.nan_to_num(data.get("number_diagnoses", np.zeros(n, dtype=np.float64)), nan=0.0)

        tot_v = inp + emg + out
        data["_tot_v"] = tot_v
        data["_inp_ratio"] = inp / (tot_v + 1.0)
        data["_emg_ratio"] = emg / (tot_v + 1.0)
        data["_has_inp"] = (inp > 0).astype(np.float64)
        data["_has_emg"] = (emg > 0).astype(np.float64)
        data["_high_inp"] = (inp >= 2).astype(np.float64)
        data["_util_severity"] = inp * 3.0 + emg * 2.0 + out * 0.5

        data["_lab_per_day"] = n_lab / (los + 1.0)
        data["_med_per_day"] = n_med / (los + 1.0)
        data["_proc_per_day"] = n_proc / (los + 1.0)
        data["_diag_per_day"] = n_diag / (los + 1.0)
        data["_med_per_diag"] = n_med / (n_diag + 1.0)
        data["_complexity"] = np.log1p(np.maximum(0, los)) * np.log1p(np.maximum(0, n_med)) * np.log1p(np.maximum(0, n_diag))

        # Medication aggregations
        med_act = np.zeros(n, dtype=np.float64)
        med_up = np.zeros(n, dtype=np.float64)
        med_down = np.zeros(n, dtype=np.float64)
        for m_col in med_names:
            if m_col in data:
                vals = data[m_col]
                med_act += ((vals != "No") & (vals != None)).astype(np.float64)
                med_up += (vals == "Up").astype(np.float64)
                med_down += (vals == "Down").astype(np.float64)
        data["_med_active"] = med_act
        data["_med_up"] = med_up
        data["_med_down"] = med_down
        data["_med_chg"] = med_up + med_down

        ins = data.get("insulin")
        if ins is not None:
            data["_ins_act"] = ((ins != "No") & (ins != None)).astype(np.float64)
            data["_ins_chg"] = np.isin(ins, ["Up", "Down"]).astype(np.float64)
            data["_ins_up"] = (ins == "Up").astype(np.float64)
            data["_ins_down"] = (ins == "Down").astype(np.float64)
        else:
            data["_ins_act"] = np.zeros(n, dtype=np.float64)
            data["_ins_chg"] = np.zeros(n, dtype=np.float64)
            data["_ins_up"] = np.zeros(n, dtype=np.float64)
            data["_ins_down"] = np.zeros(n, dtype=np.float64)

        # Discharge / admission indicators
        disp = data.get("discharge_disposition_id")
        data["_disp_home"] = (disp == "1").astype(np.float64)
        data["_disp_snf"] = np.isin(disp, ["3", "4", "22", "23", "24"]).astype(np.float64)
        data["_disp_hha"] = (disp == "6").astype(np.float64)

        adm = data.get("admission_type_id")
        data["_adm_emg"] = (adm == "1").astype(np.float64)
        data["_adm_urg"] = (adm == "2").astype(np.float64)
        data["_adm_elec"] = (adm == "3").astype(np.float64)

        src = data.get("admission_source_id")
        data["_src_emg"] = (src == "7").astype(np.float64)

        a1c = data.get("A1Cresult")
        data["_a1c_tested"] = ((a1c != "None") & (a1c != None)).astype(np.float64)
        data["_a1c_high"] = (a1c == ">8").astype(np.float64)

        glu = data.get("max_glu_serum")
        data["_glu_tested"] = ((glu != "None") & (glu != None)).astype(np.float64)
        data["_glu_high"] = np.isin(glu, [">200", ">300"]).astype(np.float64)

        # Continuous age
        age = data.get("age")
        age_map = {
            "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
            "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
            "[80-90)": 85.0, "[90-100)": 95.0
        }
        num_age = np.array([age_map.get(a, 65.0) for a in age], dtype=np.float64)
        data["_num_age"] = num_age
        data["_is_elderly"] = (num_age >= 75.0).astype(np.float64)

    num_keys = [
        "_tot_v", "_inp_ratio", "_emg_ratio", "_has_inp", "_has_emg", "_high_inp",
        "_util_severity", "_lab_per_day", "_med_per_day", "_proc_per_day",
        "_diag_per_day", "_med_per_diag", "_complexity", "_med_active", "_med_up",
        "_med_down", "_med_chg", "_ins_act", "_ins_chg", "_ins_up", "_ins_down",
        "_disp_home", "_disp_snf", "_disp_hha", "_adm_emg", "_adm_urg", "_adm_elec",
        "_src_emg", "_a1c_tested", "_a1c_high", "_glu_tested", "_glu_high",
        "_num_age", "_is_elderly"
    ]
    for k in num_keys:
        add_num(k, train_data[k], eval_data[k])

    # 2. Diagnosis chapters and 3-digit prefixes
    for data, n in [(train_data, n_train), (eval_data, n_eval)]:
        d1 = data.get("diag_1")
        d2 = data.get("diag_2")
        d3 = data.get("diag_3")

        g1 = np.array([_safe_icd9_group(d1[i] if d1 is not None else None) for i in range(n)], dtype=object)
        g2 = np.array([_safe_icd9_group(d2[i] if d2 is not None else None) for i in range(n)], dtype=object)
        g3 = np.array([_safe_icd9_group(d3[i] if d3 is not None else None) for i in range(n)], dtype=object)

        p1 = np.array([str(d1[i])[:3] if d1 is not None and str(d1[i]) not in ("?", "None", "nan") else "missing" for i in range(n)], dtype=object)
        p2 = np.array([str(d2[i])[:3] if d2 is not None and str(d2[i]) not in ("?", "None", "nan") else "missing" for i in range(n)], dtype=object)
        p3 = np.array([str(d3[i])[:3] if d3 is not None and str(d3[i]) not in ("?", "None", "nan") else "missing" for i in range(n)], dtype=object)

        data["_grp_1"] = g1
        data["_grp_2"] = g2
        data["_grp_3"] = g3
        data["_pref_1"] = p1
        data["_pref_2"] = p2
        data["_pref_3"] = p3

        has_circ = ((g1 == "circulatory") | (g2 == "circulatory") | (g3 == "circulatory")).astype(np.float64)
        has_resp = ((g1 == "respiratory") | (g2 == "respiratory") | (g3 == "respiratory")).astype(np.float64)
        has_diab = ((g1 == "diabetes") | (g2 == "diabetes") | (g3 == "diabetes")).astype(np.float64)
        has_genit = ((g1 == "genitourinary") | (g2 == "genitourinary") | (g3 == "genitourinary")).astype(np.float64)

        data["_has_circ"] = has_circ
        data["_has_resp"] = has_resp
        data["_has_diab"] = has_diab
        data["_has_genit"] = has_genit
        data["_circ_diab"] = has_circ * has_diab
        data["_genit_diab"] = has_genit * has_diab

        disp = data.get("discharge_disposition_id")
        has_inp_str = (data["_has_inp"] > 0).astype(str)
        data["_disp_inp_cross"] = np.array([f"{d}_{h}" for d, h in zip(disp, has_inp_str)], dtype=object)

    add_cat("_grp_1", train_data["_grp_1"], eval_data["_grp_1"])
    add_cat("_grp_2", train_data["_grp_2"], eval_data["_grp_2"])
    add_cat("_grp_3", train_data["_grp_3"], eval_data["_grp_3"])
    add_cat("_pref_1", train_data["_pref_1"], eval_data["_pref_1"])
    add_cat("_pref_2", train_data["_pref_2"], eval_data["_pref_2"])
    add_cat("_pref_3", train_data["_pref_3"], eval_data["_pref_3"])
    add_cat("_disp_inp_cross", train_data["_disp_inp_cross"], eval_data["_disp_inp_cross"])

    for k in ["_has_circ", "_has_resp", "_has_diab", "_has_genit", "_circ_diab", "_genit_diab"]:
        add_num(k, train_data[k], eval_data[k])

    # 3. Clinical Multimodal Tensor Decomposition via Randomized Truncated SVD
    prefix_counter = Counter()
    for p in train_data["_pref_1"]:
        if p != "missing":
            prefix_counter[p] += 2
    for p in train_data["_pref_2"]:
        if p != "missing":
            prefix_counter[p] += 1
    for p in train_data["_pref_3"]:
        if p != "missing":
            prefix_counter[p] += 1

    top_prefixes = [p for p, _ in prefix_counter.most_common(35)]

    def build_clin_tensor(data, n):
        feats = []
        p1 = data["_pref_1"]
        p2 = data["_pref_2"]
        p3 = data["_pref_3"]
        for p in top_prefixes:
            col = ((p1 == p) | (p2 == p) | (p3 == p)).astype(np.float64)
            feats.append(col)
        for g_name in ["circulatory", "respiratory", "diabetes", "digestive", "genitourinary", "injury"]:
            g1 = data["_grp_1"]
            g2 = data["_grp_2"]
            g3 = data["_grp_3"]
            col = ((g1 == g_name) | (g2 == g_name) | (g3 == g_name)).astype(np.float64)
            feats.append(col)
        for m_col in med_names:
            if m_col in data:
                vals = data[m_col]
                feats.append(((vals != "No") & (vals != None)).astype(np.float64))
                feats.append(np.isin(vals, ["Up", "Down"]).astype(np.float64))
        return np.column_stack(feats)

    clin_tr = build_clin_tensor(train_data, n_train)
    clin_ev = build_clin_tensor(eval_data, n_eval)

    svd = TruncatedSVD(n_components=8, algorithm="randomized", random_state=42)
    svd_tr = svd.fit_transform(clin_tr)
    svd_ev = svd.transform(clin_ev)

    for comp_idx in range(8):
        add_num(f"_svd_{comp_idx}", svd_tr[:, comp_idx], svd_ev[:, comp_idx])

    # 4. Out-of-fold smoothed empirical Bayes target encoding
    target_encode_cols = [
        ("discharge_disposition_id", "_te_disp"),
        ("admission_source_id", "_te_src"),
        ("medical_specialty", "_te_spec"),
        ("_pref_1", "_te_pref1"),
        ("_grp_1", "_te_grp1"),
    ]
    for col_name, te_name in target_encode_cols:
        tr_c = np.array([str(x) for x in train_data[col_name]], dtype=object)
        ev_c = np.array([str(x) for x in eval_data[col_name]], dtype=object)
        te_tr, te_ev = _oof_target_encoding(tr_c, y_train, ev_c, n_splits=5, m=25.0)
        add_num(te_name, te_tr, te_ev)

    # 5. Fit TabularEncoder on augmented data
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_data, extra_numeric=tuple(extra_numeric), extra_categorical=tuple(extra_categorical))
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)

    # 6. Diverse Tree Ensemble Optimization (LightGBM + XGBoost)
    cat_idx = encoder.categorical_indices

    # Model A: Leaf-wise LightGBM with moderate depth and regularization
    lgb_a = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    lgb_a.fit(x_train, y_train, categorical_feature=cat_idx)
    pred_a = lgb_a.predict_proba(x_eval)[:, 1]

    # Model B: Depth-constrained LightGBM
    lgb_b = lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.03,
        num_leaves=24,
        max_depth=6,
        min_child_samples=30,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=4.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    lgb_b.fit(x_train, y_train, categorical_feature=cat_idx)
    pred_b = lgb_b.predict_proba(x_eval)[:, 1]

    # Model C: Conservative LightGBM with higher shrinkage
    lgb_c = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.025,
        num_leaves=18,
        max_depth=5,
        min_child_samples=50,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=1.0,
        reg_lambda=8.0,
        random_state=456,
        n_jobs=1,
        verbose=-1,
    )
    lgb_c.fit(x_train, y_train, categorical_feature=cat_idx)
    pred_c = lgb_c.predict_proba(x_eval)[:, 1]

    # Model D: Histogram XGBoost (adds second-order split diversity)
    pred_d = None
    try:
        import xgboost as xgb
        xgb_m = xgb.XGBClassifier(
            n_estimators=280,
            learning_rate=0.035,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=3.0,
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
        )
        xgb_m.fit(x_train, y_train)
        pred_d = xgb_m.predict_proba(x_eval)[:, 1]
    except Exception:
        pass

    if pred_d is not None:
        preds = 0.35 * pred_a + 0.30 * pred_b + 0.20 * pred_c + 0.15 * pred_d
    else:
        preds = 0.40 * pred_a + 0.35 * pred_b + 0.25 * pred_c

    preds = np.clip(preds, 0.0, 1.0)
    if np.isnan(preds).any():
        preds = np.nan_to_num(preds, nan=0.115)

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
