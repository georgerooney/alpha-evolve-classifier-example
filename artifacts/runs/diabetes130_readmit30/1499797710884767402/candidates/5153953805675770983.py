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




def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    import lightgbm as lgb

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

    # 1. Clean clinical utilization & intensity features (no redundant dummy threshold columns)
    for data, n in [(train_data, n_train), (eval_data, n_eval)]:
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

        # Glycemic testing presence
        a1c = data.get("A1Cresult")
        data["_a1c_tested"] = ((a1c != "None") & (a1c != None)).astype(np.float64) if a1c is not None else np.zeros(n, dtype=np.float64)

        glu = data.get("max_glu_serum")
        data["_glu_tested"] = ((glu != "None") & (glu != None)).astype(np.float64) if glu is not None else np.zeros(n, dtype=np.float64)

        # Continuous age midpoint
        age = data.get("age")
        age_map = {
            "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
            "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
            "[80-90)": 85.0, "[90-100)": 95.0
        }
        data["_num_age"] = np.array([age_map.get(a, 65.0) for a in age], dtype=np.float64) if age is not None else np.full(n, 65.0, dtype=np.float64)

    num_keys = [
        "_tot_v", "_inp_ratio", "_emg_ratio", "_util_severity",
        "_lab_per_day", "_med_per_day", "_proc_per_day", "_diag_per_day",
        "_med_per_diag", "_complexity", "_med_active", "_med_up", "_med_down",
        "_med_chg", "_ins_act", "_ins_chg", "_ins_up", "_ins_down",
        "_a1c_tested", "_glu_tested", "_num_age"
    ]
    for k in num_keys:
        add_num(k, train_data[k], eval_data[k])

    # 2. Diagnosis chapters, 3-digit prefixes, and comorbidity interaction
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
        inp = np.nan_to_num(data.get("number_inpatient", np.zeros(n, dtype=np.float64)), nan=0.0)
        has_inp_str = (inp > 0).astype(int).astype(str)
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

    # 3. Fit TabularEncoder on the clean augmented tabular data
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_data, extra_numeric=tuple(extra_numeric), extra_categorical=tuple(extra_categorical))
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)
    cat_idx = encoder.categorical_indices

    # 4. Streamlined LightGBM Ensemble with native categorical regularization
    # Model 1: Deeper leaf-wise model with cat_smooth regularization
    m1 = lgb.LGBMClassifier(
        n_estimators=380,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=35,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.3,
        reg_lambda=2.5,
        cat_smooth=15.0,
        cat_l2=10.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=cat_idx)
    pred_1 = m1.predict_proba(x_eval)[:, 1]

    # Model 2: Conservative shallower model with higher regularisation
    m2 = lgb.LGBMClassifier(
        n_estimators=420,
        learning_rate=0.025,
        num_leaves=20,
        max_depth=5,
        min_child_samples=50,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=1.0,
        reg_lambda=5.0,
        cat_smooth=25.0,
        cat_l2=15.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=cat_idx)
    pred_2 = m2.predict_proba(x_eval)[:, 1]

    # Model 3: Unconstrained-depth leaf-wise tree with moderate leaves
    m3 = lgb.LGBMClassifier(
        n_estimators=360,
        learning_rate=0.03,
        num_leaves=25,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=0.5,
        reg_lambda=3.0,
        cat_smooth=20.0,
        cat_l2=10.0,
        random_state=456,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=cat_idx)
    pred_3 = m3.predict_proba(x_eval)[:, 1]

    # Clean unweighted ensemble of structural variants
    preds = (pred_1 + pred_2 + pred_3) / 3.0
    preds = np.clip(preds, 0.0, 1.0)
    if np.isnan(preds).any():
        preds = np.nan_to_num(preds, nan=0.115)

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
