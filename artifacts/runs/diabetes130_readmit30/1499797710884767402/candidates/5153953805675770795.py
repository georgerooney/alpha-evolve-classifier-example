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




def _parse_icd9_clean(code):
    """Safely extracts (main_code, sub_digit) for ICD-9 clinical classification."""
    if code is None or code == "" or code == "?" or str(code).lower() == "nan":
        return None, 0
    s = str(code).strip()
    if s.startswith("V") or s.startswith("E"):
        return None, 0
    try:
        parts = s.split(".")
        main_num = int(parts[0])
        sub_digit = int(parts[1][0]) if len(parts) > 1 and len(parts[1]) > 0 and parts[1][0].isdigit() else 0
        return main_num, sub_digit
    except Exception:
        return None, 0


def _compute_oof_target_encoding(train_arr, y_arr, eval_arr, n_splits=5, smoothing=20.0, random_state=42):
    """Computes leak-free StratifiedKFold OOF target encoding for train and full-sample encoding for eval."""
    from sklearn.model_selection import StratifiedKFold

    n_tr = len(y_arr)
    n_ev = len(eval_arr)
    global_mean = float(np.mean(y_arr))

    tr_clean = [str(x) if (x is not None and str(x) not in ("nan", "None", "?")) else "__MISSING__" for x in train_arr]
    ev_clean = [str(x) if (x is not None and str(x) not in ("nan", "None", "?")) else "__MISSING__" for x in eval_arr]

    oof_train = np.empty(n_tr, dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for tr_idx, val_idx in skf.split(tr_clean, y_arr):
        counts = {}
        sums = {}
        sub_y = y_arr[tr_idx]
        fold_mean = float(np.mean(sub_y))
        for i in tr_idx:
            c = tr_clean[i]
            val = y_arr[i]
            counts[c] = counts.get(c, 0) + 1
            sums[c] = sums.get(c, 0.0) + val

        val_res = np.empty(len(val_idx), dtype=np.float64)
        for j, idx in enumerate(val_idx):
            c = tr_clean[idx]
            if c in counts:
                val_res[j] = (sums[c] + smoothing * fold_mean) / (counts[c] + smoothing)
            else:
                val_res[j] = fold_mean
        oof_train[val_idx] = val_res

    full_counts = {}
    full_sums = {}
    for c, val in zip(tr_clean, y_arr):
        full_counts[c] = full_counts.get(c, 0) + 1
        full_sums[c] = full_sums.get(c, 0.0) + val

    oof_eval = np.empty(n_ev, dtype=np.float64)
    for j, c in enumerate(ev_clean):
        if c in full_counts:
            oof_eval[j] = (full_sums[c] + smoothing * global_mean) / (full_counts[c] + smoothing)
        else:
            oof_eval[j] = global_mean

    return oof_train, oof_eval


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

    # 1. Clean clinical utilization & intensity features
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
        data["_has_prior_inp"] = (inp > 0).astype(np.float64)
        data["_has_prior_emg"] = (emg > 0).astype(np.float64)
        data["_inp_ratio"] = inp / (tot_v + 1.0)
        data["_emg_ratio"] = emg / (tot_v + 1.0)
        data["_util_severity"] = inp * 3.0 + emg * 2.0 + out * 0.5
        data["_frequent_utilizer"] = (tot_v >= 3.0).astype(np.float64)

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
        data["_polypharmacy"] = (n_med >= 15.0).astype(np.float64)

        ins = data.get("insulin")
        if ins is not None:
            ins_act = ((ins != "No") & (ins != None)).astype(np.float64)
            data["_ins_act"] = ins_act
            data["_ins_chg"] = np.isin(ins, ["Up", "Down"]).astype(np.float64)
            data["_ins_up"] = (ins == "Up").astype(np.float64)
            data["_ins_down"] = (ins == "Down").astype(np.float64)
            data["_ins_steady"] = (ins == "Steady").astype(np.float64)
            data["_ins_and_oral"] = ((ins_act > 0) & (med_act > 1)).astype(np.float64)
        else:
            data["_ins_act"] = np.zeros(n, dtype=np.float64)
            data["_ins_chg"] = np.zeros(n, dtype=np.float64)
            data["_ins_up"] = np.zeros(n, dtype=np.float64)
            data["_ins_down"] = np.zeros(n, dtype=np.float64)
            data["_ins_steady"] = np.zeros(n, dtype=np.float64)
            data["_ins_and_oral"] = np.zeros(n, dtype=np.float64)

        # Glycemic testing and clinical inertia
        chg = data.get("change")
        chg_flag = (chg == "Ch").astype(np.float64) if chg is not None else np.zeros(n, dtype=np.float64)
        data["_med_changed"] = chg_flag

        a1c = data.get("A1Cresult")
        if a1c is not None:
            data["_a1c_tested"] = ((a1c != "None") & (a1c != None)).astype(np.float64)
            a1c_high = np.isin(a1c, [">7", ">8"]).astype(np.float64)
            a1c_vhigh = (a1c == ">8").astype(np.float64)
            data["_a1c_high"] = a1c_high
            data["_a1c_vhigh"] = a1c_vhigh
            data["_a1c_inertia"] = a1c_vhigh * (1.0 - chg_flag)
        else:
            data["_a1c_tested"] = np.zeros(n, dtype=np.float64)
            data["_a1c_high"] = np.zeros(n, dtype=np.float64)
            data["_a1c_vhigh"] = np.zeros(n, dtype=np.float64)
            data["_a1c_inertia"] = np.zeros(n, dtype=np.float64)

        glu = data.get("max_glu_serum")
        if glu is not None:
            data["_glu_tested"] = ((glu != "None") & (glu != None)).astype(np.float64)
            glu_high = np.isin(glu, [">200", ">300"]).astype(np.float64)
            data["_glu_high"] = glu_high
            data["_glu_inertia"] = glu_high * (1.0 - chg_flag)
        else:
            data["_glu_tested"] = np.zeros(n, dtype=np.float64)
            data["_glu_high"] = np.zeros(n, dtype=np.float64)
            data["_glu_inertia"] = np.zeros(n, dtype=np.float64)

        # Discharge and admission indicators
        disp = data.get("discharge_disposition_id")
        if disp is not None:
            data["_disp_home"] = (disp == "1").astype(np.float64)
            data["_disp_snf"] = (disp == "3").astype(np.float64)
            data["_disp_home_health"] = (disp == "6").astype(np.float64)
            data["_disp_ama"] = (disp == "7").astype(np.float64)
            data["_disp_facility"] = np.isin(disp, ["2", "3", "4", "5", "6", "22", "23", "24"]).astype(np.float64)
        else:
            data["_disp_home"] = np.zeros(n, dtype=np.float64)
            data["_disp_snf"] = np.zeros(n, dtype=np.float64)
            data["_disp_home_health"] = np.zeros(n, dtype=np.float64)
            data["_disp_ama"] = np.zeros(n, dtype=np.float64)
            data["_disp_facility"] = np.zeros(n, dtype=np.float64)

        adm_src = data.get("admission_source_id")
        if adm_src is not None:
            data["_adm_emergency"] = (adm_src == "7").astype(np.float64)
            data["_adm_referral"] = np.isin(adm_src, ["1", "2"]).astype(np.float64)
        else:
            data["_adm_emergency"] = np.zeros(n, dtype=np.float64)
            data["_adm_referral"] = np.zeros(n, dtype=np.float64)

        # Missingness patterns
        payer = data.get("payer_code")
        spec = data.get("medical_specialty")
        data["_payer_missing"] = np.array([(p is None or str(p) in ("?", "None", "nan")) for p in (payer if payer is not None else [None]*n)], dtype=np.float64)
        data["_spec_missing"] = np.array([(s is None or str(s) in ("?", "None", "nan")) for s in (spec if spec is not None else [None]*n)], dtype=np.float64)

        # Continuous age midpoint
        age = data.get("age")
        age_map = {
            "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
            "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
            "[80-90)": 85.0, "[90-100)": 95.0
        }
        data["_num_age"] = np.array([age_map.get(a, 65.0) for a in age], dtype=np.float64) if age is not None else np.full(n, 65.0, dtype=np.float64)

    num_keys = [
        "_tot_v", "_has_prior_inp", "_has_prior_emg", "_inp_ratio", "_emg_ratio",
        "_util_severity", "_frequent_utilizer", "_lab_per_day", "_med_per_day",
        "_proc_per_day", "_diag_per_day", "_med_per_diag", "_complexity",
        "_med_active", "_med_up", "_med_down", "_med_chg", "_polypharmacy",
        "_ins_act", "_ins_chg", "_ins_up", "_ins_down", "_ins_steady", "_ins_and_oral",
        "_med_changed", "_a1c_tested", "_a1c_high", "_a1c_vhigh", "_a1c_inertia",
        "_glu_tested", "_glu_high", "_glu_inertia",
        "_disp_home", "_disp_snf", "_disp_home_health", "_disp_ama", "_disp_facility",
        "_adm_emergency", "_adm_referral", "_payer_missing", "_spec_missing", "_num_age"
    ]
    for k in num_keys:
        add_num(k, train_data[k], eval_data[k])

    # 2. Diagnosis chapters, 3-digit prefixes, comorbidities & interactions
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

        # Parse ICD-9 comorbidity phenotypes
        parsed_d1 = [_parse_icd9_clean(d1[i] if d1 is not None else None) for i in range(n)]
        parsed_d2 = [_parse_icd9_clean(d2[i] if d2 is not None else None) for i in range(n)]
        parsed_d3 = [_parse_icd9_clean(d3[i] if d3 is not None else None) for i in range(n)]

        comorb_chf = np.zeros(n, dtype=np.float64)
        comorb_cad = np.zeros(n, dtype=np.float64)
        comorb_htn = np.zeros(n, dtype=np.float64)
        comorb_ckd = np.zeros(n, dtype=np.float64)
        comorb_copd = np.zeros(n, dtype=np.float64)
        comorb_diab_comp = np.zeros(n, dtype=np.float64)
        comorb_fluid = np.zeros(n, dtype=np.float64)
        comorb_stroke = np.zeros(n, dtype=np.float64)

        for i in range(n):
            codes = (parsed_d1[i], parsed_d2[i], parsed_d3[i])
            m_vals = [c[0] for c in codes if c[0] is not None]
            if any(m == 428 for m in m_vals):
                comorb_chf[i] = 1.0
            if any(410 <= m <= 414 for m in m_vals):
                comorb_cad[i] = 1.0
            if any(401 <= m <= 405 for m in m_vals):
                comorb_htn[i] = 1.0
            if any(582 <= m <= 588 for m in m_vals):
                comorb_ckd[i] = 1.0
            if any(490 <= m <= 496 for m in m_vals):
                comorb_copd[i] = 1.0
            if any(c[0] == 250 and c[1] >= 4 for c in codes):
                comorb_diab_comp[i] = 1.0
            if any(m == 276 for m in m_vals):
                comorb_fluid[i] = 1.0
            if any(430 <= m <= 438 for m in m_vals):
                comorb_stroke[i] = 1.0

        data["_comorb_chf"] = comorb_chf
        data["_comorb_cad"] = comorb_cad
        data["_comorb_htn"] = comorb_htn
        data["_comorb_ckd"] = comorb_ckd
        data["_comorb_copd"] = comorb_copd
        data["_comorb_diab_comp"] = comorb_diab_comp
        data["_comorb_fluid"] = comorb_fluid
        data["_comorb_stroke"] = comorb_stroke
        data["_comorb_count"] = comorb_chf + comorb_cad + comorb_htn + comorb_ckd + comorb_copd + comorb_diab_comp + comorb_fluid + comorb_stroke

        cardiorenal = ((comorb_chf > 0) | (comorb_ckd > 0)).astype(np.float64)
        data["_cardiorenal"] = cardiorenal

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
        has_inp_flag = (inp > 0).astype(np.float64)
        data["_cardiorenal_prior_inp"] = cardiorenal * has_inp_flag
        data["_comp_diab_prior_inp"] = comorb_diab_comp * has_inp_flag

        has_inp_str = (inp > 0).astype(int).astype(str)
        data["_disp_inp_cross"] = np.array([f"{d}_{h}" for d, h in zip(disp, has_inp_str)], dtype=object)

    add_cat("_grp_1", train_data["_grp_1"], eval_data["_grp_1"])
    add_cat("_grp_2", train_data["_grp_2"], eval_data["_grp_2"])
    add_cat("_grp_3", train_data["_grp_3"], eval_data["_grp_3"])
    add_cat("_pref_1", train_data["_pref_1"], eval_data["_pref_1"])
    add_cat("_pref_2", train_data["_pref_2"], eval_data["_pref_2"])
    add_cat("_pref_3", train_data["_pref_3"], eval_data["_pref_3"])
    add_cat("_disp_inp_cross", train_data["_disp_inp_cross"], eval_data["_disp_inp_cross"])

    diag_num_keys = [
        "_comorb_chf", "_comorb_cad", "_comorb_htn", "_comorb_ckd",
        "_comorb_copd", "_comorb_diab_comp", "_comorb_fluid", "_comorb_stroke",
        "_comorb_count", "_cardiorenal", "_cardiorenal_prior_inp", "_comp_diab_prior_inp",
        "_has_circ", "_has_resp", "_has_diab", "_has_genit", "_circ_diab", "_genit_diab"
    ]
    for k in diag_num_keys:
        add_num(k, train_data[k], eval_data[k])

    # 3. Out-Of-Fold Empirical Bayes Target Encoding for High-Cardinality Categoricals
    te_configs = [
        ("diag_1", "_te_diag_1", 20.0),
        ("diag_2", "_te_diag_2", 20.0),
        ("diag_3", "_te_diag_3", 20.0),
        ("_pref_1", "_te_pref_1", 15.0),
        ("discharge_disposition_id", "_te_discharge", 10.0),
        ("admission_source_id", "_te_admission_src", 10.0),
        ("medical_specialty", "_te_specialty", 15.0),
        ("_disp_inp_cross", "_te_disp_inp", 15.0),
    ]

    for src_col, target_col, smoothing in te_configs:
        tr_src = train_data.get(src_col)
        ev_src = eval_data.get(src_col)
        oof_tr, oof_ev = _compute_oof_target_encoding(
            tr_src if tr_src is not None else [None] * n_train,
            y_train,
            ev_src if ev_src is not None else [None] * n_eval,
            n_splits=5,
            smoothing=smoothing,
            random_state=42,
        )
        add_num(target_col, oof_tr, oof_ev)

    # Derived aggregate diagnosis risk summaries
    max_diag_tr = np.maximum(np.maximum(train_data["_te_diag_1"], train_data["_te_diag_2"]), train_data["_te_diag_3"])
    max_diag_ev = np.maximum(np.maximum(eval_data["_te_diag_1"], eval_data["_te_diag_2"]), eval_data["_te_diag_3"])
    mean_diag_tr = (train_data["_te_diag_1"] + train_data["_te_diag_2"] + train_data["_te_diag_3"]) / 3.0
    mean_diag_ev = (eval_data["_te_diag_1"] + eval_data["_te_diag_2"] + eval_data["_te_diag_3"]) / 3.0
    add_num("_te_max_diag", max_diag_tr, max_diag_ev)
    add_num("_te_mean_diag", mean_diag_tr, mean_diag_ev)

    # 4. Fit TabularEncoder on the clean augmented tabular data
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_data, extra_numeric=tuple(extra_numeric), extra_categorical=tuple(extra_categorical))
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)
    cat_idx = encoder.categorical_indices

    # 5. Diverse 4-Model LightGBM Ensemble
    # Model 1: Balanced leaf-wise model with cat_smooth regularization
    m1 = lgb.LGBMClassifier(
        n_estimators=420,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=6,
        min_child_samples=35,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.4,
        reg_lambda=3.0,
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
        n_estimators=460,
        learning_rate=0.022,
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
        n_estimators=380,
        learning_rate=0.028,
        num_leaves=28,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=0.5,
        reg_lambda=3.5,
        cat_smooth=20.0,
        cat_l2=10.0,
        random_state=456,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=cat_idx)
    pred_3 = m3.predict_proba(x_eval)[:, 1]

    # Model 4: ExtraTrees variant in LightGBM for split-level structural diversity
    m4 = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.025,
        num_leaves=32,
        max_depth=6,
        min_child_samples=30,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.70,
        extra_trees=True,
        reg_alpha=0.5,
        reg_lambda=4.0,
        cat_smooth=15.0,
        cat_l2=10.0,
        random_state=789,
        n_jobs=1,
        verbose=-1,
    )
    m4.fit(x_train, y_train, categorical_feature=cat_idx)
    pred_4 = m4.predict_proba(x_eval)[:, 1]

    # Equal-weight ensemble across structural variants
    preds = 0.25 * pred_1 + 0.25 * pred_2 + 0.25 * pred_3 + 0.25 * pred_4
    preds = np.clip(preds, 0.0, 1.0)
    if np.isnan(preds).any():
        preds = np.nan_to_num(preds, nan=0.115)

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
