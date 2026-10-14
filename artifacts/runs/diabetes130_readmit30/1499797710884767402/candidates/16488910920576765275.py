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
from typing import Dict, Sequence
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

try:
    from clinical_toolkit import TabularEncoder, icd9_group
except ImportError:
    pass


def _clean_code(c) -> str:
    if c is None:
        return "MISSING"
    s = str(c).strip()
    if not s or s in ("?", "None"):
        return "MISSING"
    return s.split(".")[0].strip()


def _extract_comorbidities(p1_arr: np.ndarray, p2_arr: np.ndarray, p3_arr: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(p1_arr)
    has_diabetes = np.zeros(n, dtype=np.float64)
    is_primary_diabetes = np.zeros(n, dtype=np.float64)
    has_diabetes_comp = np.zeros(n, dtype=np.float64)
    has_chf = np.zeros(n, dtype=np.float64)
    has_cad = np.zeros(n, dtype=np.float64)
    has_htn = np.zeros(n, dtype=np.float64)
    has_ckd = np.zeros(n, dtype=np.float64)
    has_copd = np.zeros(n, dtype=np.float64)
    has_fluid = np.zeros(n, dtype=np.float64)
    has_stroke = np.zeros(n, dtype=np.float64)
    has_pvd = np.zeros(n, dtype=np.float64)
    has_neoplasm = np.zeros(n, dtype=np.float64)

    cad_p = ("410", "411", "412", "413", "414")
    htn_p = ("401", "402", "403", "404", "405")
    ckd_p = ("585", "586", "582", "583", "584")
    copd_p = ("490", "491", "492", "493", "494", "495", "496", "480", "481", "482", "483", "484", "485", "486")
    stroke_p = ("430", "431", "432", "433", "434", "435", "436", "437", "438")
    pvd_p = ("440", "441", "442", "443", "444", "445", "446", "447")

    for i in range(n):
        d1, d2, d3 = p1_arr[i], p2_arr[i], p3_arr[i]
        if d1.startswith("250"):
            is_primary_diabetes[i] = 1.0
            has_diabetes[i] = 1.0
        elif d2.startswith("250") or d3.startswith("250"):
            has_diabetes[i] = 1.0

        for d in (d1, d2, d3):
            if d.startswith("250.") and not d.startswith("250.0"):
                has_diabetes_comp[i] = 1.0
            if d.startswith("428"):
                has_chf[i] = 1.0
            if d.startswith(cad_p):
                has_cad[i] = 1.0
            if d.startswith(htn_p):
                has_htn[i] = 1.0
            if d.startswith(ckd_p):
                has_ckd[i] = 1.0
            if d.startswith(copd_p):
                has_copd[i] = 1.0
            if d.startswith("276"):
                has_fluid[i] = 1.0
            if d.startswith(stroke_p):
                has_stroke[i] = 1.0
            if d.startswith(pvd_p):
                has_pvd[i] = 1.0
            if d.isdigit() and len(d) >= 3 and 140 <= int(d[:3]) <= 239:
                has_neoplasm[i] = 1.0

    comorb_cnt = (
        has_diabetes_comp + has_chf + has_cad + has_htn +
        has_ckd + has_copd + has_fluid + has_stroke + has_pvd + has_neoplasm
    )

    return {
        "has_diabetes": has_diabetes,
        "is_primary_diabetes": is_primary_diabetes,
        "has_diabetes_comp": has_diabetes_comp,
        "has_chf": has_chf,
        "has_cad": has_cad,
        "has_htn": has_htn,
        "has_ckd": has_ckd,
        "has_copd": has_copd,
        "has_fluid": has_fluid,
        "has_stroke": has_stroke,
        "has_pvd": has_pvd,
        "has_neoplasm": has_neoplasm,
        "comorbidity_count": comorb_cnt,
        "cardiorenal": has_chf * has_ckd,
        "cardiodiabetes": has_chf * has_diabetes_comp,
    }


def _extract_medications(cols: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    n = len(cols["time_in_hospital"])
    num_active = np.zeros(n, dtype=np.float64)
    num_up = np.zeros(n, dtype=np.float64)
    num_down = np.zeros(n, dtype=np.float64)

    med_names = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide",
        "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide",
        "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
        "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone"
    ]

    for med in med_names:
        if med in cols:
            arr = cols[med]
            for i in range(n):
                v = arr[i]
                if v == "Up":
                    num_up[i] += 1.0
                    num_active[i] += 1.0
                elif v == "Down":
                    num_down[i] += 1.0
                    num_active[i] += 1.0
                elif v == "Steady":
                    num_active[i] += 1.0

    insulin_treatment = np.zeros(n, dtype=np.float64)
    insulin_up = np.zeros(n, dtype=np.float64)
    insulin_down = np.zeros(n, dtype=np.float64)
    insulin_steady = np.zeros(n, dtype=np.float64)
    if "insulin" in cols:
        ins = cols["insulin"]
        for i in range(n):
            v = ins[i]
            if v == "Up":
                insulin_treatment[i] = 1.0
                insulin_up[i] = 1.0
            elif v == "Down":
                insulin_treatment[i] = 1.0
                insulin_down[i] = 1.0
            elif v == "Steady":
                insulin_treatment[i] = 1.0
                insulin_steady[i] = 1.0

    metformin_treatment = np.zeros(n, dtype=np.float64)
    if "metformin" in cols:
        met = cols["metformin"]
        for i in range(n):
            if met[i] in ("Up", "Down", "Steady"):
                metformin_treatment[i] = 1.0

    sulfonylurea = np.zeros(n, dtype=np.float64)
    for s_name in ("glimepiride", "glipizide", "glyburide"):
        if s_name in cols:
            arr = cols[s_name]
            for i in range(n):
                if arr[i] in ("Up", "Down", "Steady"):
                    sulfonylurea[i] = 1.0

    ch_arr = cols.get("change")
    dm_arr = cols.get("diabetesMed")
    med_change = np.array([1.0 if (ch_arr is not None and ch_arr[i] == "Ch") else 0.0 for i in range(n)], dtype=np.float64)
    dm_flag = np.array([1.0 if (dm_arr is not None and dm_arr[i] == "Yes") else 0.0 for i in range(n)], dtype=np.float64)

    return {
        "num_active_meds": num_active,
        "num_meds_up": num_up,
        "num_meds_down": num_down,
        "num_meds_changed": num_up + num_down,
        "insulin_treatment": insulin_treatment,
        "insulin_up": insulin_up,
        "insulin_down": insulin_down,
        "insulin_steady": insulin_steady,
        "metformin_treatment": metformin_treatment,
        "sulfonylurea_treatment": sulfonylurea,
        "insulin_and_oral": ((insulin_treatment > 0) & (num_active > 1)).astype(np.float64),
        "med_change_flag": med_change,
        "diabetes_med_flag": dm_flag,
        "change_and_insulin": med_change * insulin_treatment,
    }


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    n_tr = len(y_train)
    n_ev = len(eval_cols["time_in_hospital"])
    global_prior = float(np.mean(y_train)) if n_tr > 0 else 0.114

    aug_train = dict(train)
    aug_eval = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    def add_num(name: str, tr_arr, ev_arr):
        aug_train[name] = np.asarray(tr_arr, dtype=np.float64)
        aug_eval[name] = np.asarray(ev_arr, dtype=np.float64)
        extra_numeric.append(name)

    def add_cat(name: str, tr_arr, ev_arr):
        aug_train[name] = np.asarray(tr_arr, dtype=object)
        aug_eval[name] = np.asarray(ev_arr, dtype=object)
        extra_categorical.append(name)

    # 1. Cleaned ICD-9 Prefixes & Chapters
    p1_tr = np.array([_clean_code(c) for c in train["diag_1"]], dtype=object)
    p2_tr = np.array([_clean_code(c) for c in train["diag_2"]], dtype=object)
    p3_tr = np.array([_clean_code(c) for c in train["diag_3"]], dtype=object)
    p1_ev = np.array([_clean_code(c) for c in eval_cols["diag_1"]], dtype=object)
    p2_ev = np.array([_clean_code(c) for c in eval_cols["diag_2"]], dtype=object)
    p3_ev = np.array([_clean_code(c) for c in eval_cols["diag_3"]], dtype=object)

    add_cat("diag_1_prefix", p1_tr, p1_ev)
    add_cat("diag_2_prefix", p2_tr, p2_ev)
    add_cat("diag_3_prefix", p3_tr, p3_ev)

    _icd9 = globals().get("icd9_group", lambda c: "other")
    ch1_tr = np.array([_icd9(c) for c in train["diag_1"]], dtype=object)
    ch2_tr = np.array([_icd9(c) for c in train["diag_2"]], dtype=object)
    ch3_tr = np.array([_icd9(c) for c in train["diag_3"]], dtype=object)
    ch1_ev = np.array([_icd9(c) for c in eval_cols["diag_1"]], dtype=object)
    ch2_ev = np.array([_icd9(c) for c in eval_cols["diag_2"]], dtype=object)
    ch3_ev = np.array([_icd9(c) for c in eval_cols["diag_3"]], dtype=object)

    add_cat("diag_1_chapter", ch1_tr, ch1_ev)
    add_cat("diag_2_chapter", ch2_tr, ch2_ev)
    add_cat("diag_3_chapter", ch3_tr, ch3_ev)

    add_num("diag_1_is_circ", (ch1_tr == "circulatory").astype(float), (ch1_ev == "circulatory").astype(float))
    add_num("diag_1_is_resp", (ch1_tr == "respiratory").astype(float), (ch1_ev == "respiratory").astype(float))
    add_num("diag_1_is_diab", (ch1_tr == "diabetes").astype(float), (ch1_ev == "diabetes").astype(float))

    # 2. Comorbidities
    comorb_tr = _extract_comorbidities(p1_tr, p2_tr, p3_tr)
    comorb_ev = _extract_comorbidities(p1_ev, p2_ev, p3_ev)
    for k in comorb_tr:
        add_num(k, comorb_tr[k], comorb_ev[k])

    # 3. Medications
    med_feats_tr = _extract_medications(train)
    med_feats_ev = _extract_medications(eval_cols)
    for k in med_feats_tr:
        add_num(k, med_feats_tr[k], med_feats_ev[k])

    # 4. Out-of-fold Bayesian Target Encodings
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 4a. Primary diagnosis TE & Deduplicated Pooled diagnosis TE
    te_diag_p1_tr = np.full(n_tr, global_prior, dtype=np.float64)
    te_diag_max_tr = np.full(n_tr, global_prior, dtype=np.float64)
    te_diag_mean_tr = np.full(n_tr, global_prior, dtype=np.float64)
    te_diag_sum_tr = np.full(n_tr, 3.0 * global_prior, dtype=np.float64)

    smoothing_p1 = 20.0
    smoothing_pool = 25.0

    for trn_idx, val_idx in skf.split(np.zeros(n_tr), y_train):
        f_prior = float(np.mean(y_train[trn_idx]))
        p1_cnt, p1_sum = {}, {}
        pool_cnt, pool_sum = {}, {}

        for i in trn_idx:
            y_i = float(y_train[i])
            c1 = p1_tr[i]
            if c1 != "MISSING":
                p1_cnt[c1] = p1_cnt.get(c1, 0) + 1
                p1_sum[c1] = p1_sum.get(c1, 0.0) + y_i

            codes = set(p for p in (p1_tr[i], p2_tr[i], p3_tr[i]) if p != "MISSING")
            for c in codes:
                pool_cnt[c] = pool_cnt.get(c, 0) + 1
                pool_sum[c] = pool_sum.get(c, 0.0) + y_i

        for i in val_idx:
            c1 = p1_tr[i]
            if c1 == "MISSING":
                te_diag_p1_tr[i] = f_prior
            else:
                cnt = p1_cnt.get(c1, 0)
                te_diag_p1_tr[i] = (p1_sum.get(c1, 0.0) + smoothing_p1 * f_prior) / (cnt + smoothing_p1)

            def _calc_p(code):
                if code == "MISSING":
                    return f_prior
                cnt = pool_cnt.get(code, 0)
                return (pool_sum.get(code, 0.0) + smoothing_pool * f_prior) / (cnt + smoothing_pool)

            r1, r2, r3 = _calc_p(p1_tr[i]), _calc_p(p2_tr[i]), _calc_p(p3_tr[i])
            te_diag_max_tr[i] = max(r1, r2, r3)
            te_diag_mean_tr[i] = (r1 + r2 + r3) / 3.0
            te_diag_sum_tr[i] = r1 + r2 + r3

    full_p1_cnt, full_p1_sum = {}, {}
    full_pool_cnt, full_pool_sum = {}, {}
    for i in range(n_tr):
        y_i = float(y_train[i])
        c1 = p1_tr[i]
        if c1 != "MISSING":
            full_p1_cnt[c1] = full_p1_cnt.get(c1, 0) + 1
            full_p1_sum[c1] = full_p1_sum.get(c1, 0.0) + y_i

        codes = set(p for p in (p1_tr[i], p2_tr[i], p3_tr[i]) if p != "MISSING")
        for c in codes:
            full_pool_cnt[c] = full_pool_cnt.get(c, 0) + 1
            full_pool_sum[c] = full_pool_sum.get(c, 0.0) + y_i

    te_diag_p1_ev = np.empty(n_ev, dtype=np.float64)
    te_diag_max_ev = np.empty(n_ev, dtype=np.float64)
    te_diag_mean_ev = np.empty(n_ev, dtype=np.float64)
    te_diag_sum_ev = np.empty(n_ev, dtype=np.float64)

    for j in range(n_ev):
        c1 = p1_ev[j]
        if c1 == "MISSING":
            te_diag_p1_ev[j] = global_prior
        else:
            cnt = full_p1_cnt.get(c1, 0)
            te_diag_p1_ev[j] = (full_p1_sum.get(c1, 0.0) + smoothing_p1 * global_prior) / (cnt + smoothing_p1)

        def _calc_p_full(code):
            if code == "MISSING":
                return global_prior
            cnt = full_pool_cnt.get(code, 0)
            return (full_pool_sum.get(code, 0.0) + smoothing_pool * global_prior) / (cnt + smoothing_pool)

        r1, r2, r3 = _calc_p_full(p1_ev[j]), _calc_p_full(p2_ev[j]), _calc_p_full(p3_ev[j])
        te_diag_max_ev[j] = max(r1, r2, r3)
        te_diag_mean_ev[j] = (r1 + r2 + r3) / 3.0
        te_diag_sum_ev[j] = r1 + r2 + r3

    add_num("te_diag_primary", te_diag_p1_tr, te_diag_p1_ev)
    add_num("te_diag_max", te_diag_max_tr, te_diag_max_ev)
    add_num("te_diag_mean", te_diag_mean_tr, te_diag_mean_ev)
    add_num("te_diag_sum", te_diag_sum_tr, te_diag_sum_ev)

    # 4b. Administrative variables TE
    def _compute_single_te(col_tr, col_ev, m=20.0):
        t_tr = np.full(n_tr, global_prior, dtype=np.float64)
        t_ev = np.full(n_ev, global_prior, dtype=np.float64)
        c_tr = np.array([str(x).strip() if (x is not None and str(x).strip() not in ("", "?", "None")) else "MISSING" for x in col_tr], dtype=object)
        c_ev = np.array([str(x).strip() if (x is not None and str(x).strip() not in ("", "?", "None")) else "MISSING" for x in col_ev], dtype=object)

        for trn_idx, val_idx in skf.split(np.zeros(n_tr), y_train):
            f_prior = float(np.mean(y_train[trn_idx]))
            counts, sums = {}, {}
            for i in trn_idx:
                k = c_tr[i]
                counts[k] = counts.get(k, 0) + 1
                sums[k] = sums.get(k, 0.0) + float(y_train[i])
            for i in val_idx:
                k = c_tr[i]
                cnt = counts.get(k, 0)
                t_tr[i] = (sums.get(k, 0.0) + m * f_prior) / (cnt + m)

        tot_counts, tot_sums = {}, {}
        for i in range(n_tr):
            k = c_tr[i]
            tot_counts[k] = tot_counts.get(k, 0) + 1
            tot_sums[k] = tot_sums.get(k, 0.0) + float(y_train[i])
        for j in range(n_ev):
            k = c_ev[j]
            cnt = tot_counts.get(k, 0)
            t_ev[j] = (tot_sums.get(k, 0.0) + m * global_prior) / (cnt + m)
        return t_tr, t_ev

    t_disp_tr, t_disp_ev = _compute_single_te(train["discharge_disposition_id"], eval_cols["discharge_disposition_id"], m=20.0)
    t_src_tr, t_src_ev = _compute_single_te(train["admission_source_id"], eval_cols["admission_source_id"], m=20.0)
    t_adm_tr, t_adm_ev = _compute_single_te(train["admission_type_id"], eval_cols["admission_type_id"], m=20.0)
    t_spec_tr, t_spec_ev = _compute_single_te(train["medical_specialty"], eval_cols["medical_specialty"], m=25.0)
    t_pay_tr, t_pay_ev = _compute_single_te(train["payer_code"], eval_cols["payer_code"], m=25.0)
    t_age_tr, t_age_ev = _compute_single_te(train["age"], eval_cols["age"], m=30.0)

    add_num("te_discharge", t_disp_tr, t_disp_ev)
    add_num("te_admission_source", t_src_tr, t_src_ev)
    add_num("te_admission_type", t_adm_tr, t_adm_ev)
    add_num("te_specialty", t_spec_tr, t_spec_ev)
    add_num("te_payer_code", t_pay_tr, t_pay_ev)
    add_num("te_age", t_age_tr, t_age_ev)

    # 5. Utilization, Acuity & Interactions
    def _safe_arr(d, k, def_val=0.0):
        return np.nan_to_num(np.array(d.get(k, def_val), dtype=np.float64), nan=def_val)

    inp_tr, inp_ev = _safe_arr(train, "number_inpatient"), _safe_arr(eval_cols, "number_inpatient")
    emg_tr, emg_ev = _safe_arr(train, "number_emergency"), _safe_arr(eval_cols, "number_emergency")
    out_tr, out_ev = _safe_arr(train, "number_outpatient"), _safe_arr(eval_cols, "number_outpatient")
    los_tr = np.maximum(1.0, _safe_arr(train, "time_in_hospital", 1.0))
    los_ev = np.maximum(1.0, _safe_arr(eval_cols, "time_in_hospital", 1.0))
    med_tr, med_ev = _safe_arr(train, "num_medications"), _safe_arr(eval_cols, "num_medications")
    lab_tr, lab_ev = _safe_arr(train, "num_lab_procedures"), _safe_arr(eval_cols, "num_lab_procedures")
    proc_tr, proc_ev = _safe_arr(train, "num_procedures"), _safe_arr(eval_cols, "num_procedures")
    diag_tr = np.maximum(1.0, _safe_arr(train, "number_diagnoses", 1.0))
    diag_ev = np.maximum(1.0, _safe_arr(eval_cols, "number_diagnoses", 1.0))

    tot_tr = inp_tr + emg_tr + out_tr
    tot_ev = inp_ev + emg_ev + out_ev

    add_num("log_inpatient", np.log1p(np.maximum(0.0, inp_tr)), np.log1p(np.maximum(0.0, inp_ev)))
    add_num("log_emergency", np.log1p(np.maximum(0.0, emg_tr)), np.log1p(np.maximum(0.0, emg_ev)))
    add_num("log_outpatient", np.log1p(np.maximum(0.0, out_tr)), np.log1p(np.maximum(0.0, out_ev)))
    add_num("log_total_visits", np.log1p(np.maximum(0.0, tot_tr)), np.log1p(np.maximum(0.0, tot_ev)))
    add_num("total_prior_visits", tot_tr, tot_ev)
    add_num("has_prior_inpatient", (inp_tr > 0).astype(float), (inp_ev > 0).astype(float))
    add_num("has_prior_emergency", (emg_tr > 0).astype(float), (emg_ev > 0).astype(float))
    add_num("frequent_inpatient", (inp_tr >= 2).astype(float), (inp_ev >= 2).astype(float))
    add_num("high_utilizer", (tot_tr >= 3).astype(float), (tot_ev >= 3).astype(float))
    add_num("prior_acute_load", np.log1p(2.0 * inp_tr + emg_tr), np.log1p(2.0 * inp_ev + emg_ev))
    add_num("weighted_prior_util", 2.0 * inp_tr + 1.5 * emg_tr + 0.5 * out_tr, 2.0 * inp_ev + 1.5 * emg_ev + 0.5 * out_ev)
    add_num("inpatient_ratio", inp_tr / (tot_tr + 1.0), inp_ev / (tot_ev + 1.0))
    add_num("emergency_ratio", emg_tr / (tot_tr + 1.0), emg_ev / (tot_ev + 1.0))
    add_num("outpatient_ratio", out_tr / (tot_tr + 1.0), out_ev / (tot_ev + 1.0))

    add_num("stay_intensity", (med_tr + lab_tr) / np.sqrt(los_tr), (med_ev + lab_ev) / np.sqrt(los_ev))
    add_num("meds_per_day", med_tr / los_tr, med_ev / los_ev)
    add_num("labs_per_day", lab_tr / los_tr, lab_ev / los_ev)
    add_num("procs_per_day", proc_tr / los_tr, proc_ev / los_ev)
    add_num("total_procedures", lab_tr + proc_tr, lab_ev + proc_ev)
    add_num("meds_per_diag", med_tr / diag_tr, med_ev / diag_ev)
    add_num("diagnoses_per_day", diag_tr / los_tr, diag_ev / los_ev)
    add_num("labs_per_med", lab_tr / (med_tr + 1.0), lab_ev / (med_ev + 1.0))

    add_num("inpatient_x_los", inp_tr * los_tr, inp_ev * los_ev)
    add_num("inpatient_x_diags", inp_tr * diag_tr, inp_ev * diag_ev)
    add_num("inpatient_x_emergency", inp_tr * emg_tr, inp_ev * emg_ev)
    add_num("inpatient_per_day", inp_tr / los_tr, inp_ev / los_ev)

    # 6. Age & Demographics
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0
    }
    age_tr = np.array([age_map.get(x, 65.0) for x in train["age"]], dtype=np.float64)
    age_ev = np.array([age_map.get(x, 65.0) for x in eval_cols["age"]], dtype=np.float64)
    add_num("age_numeric", age_tr, age_ev)
    add_num("is_elderly", (age_tr >= 75.0).astype(float), (age_ev >= 75.0).astype(float))
    add_num("is_young", (age_tr <= 30.0).astype(float), (age_ev <= 30.0).astype(float))
    add_num("elderly_with_inpatient", (age_tr >= 75.0).astype(float) * (inp_tr > 0).astype(float), (age_ev >= 75.0).astype(float) * (inp_ev > 0).astype(float))

    # 7. Lab Results
    a1c_tr = np.array([1.0 if (x is not None and str(x) != "None") else 0.0 for x in train["A1Cresult"]], dtype=np.float64)
    a1c_ev = np.array([1.0 if (x is not None and str(x) != "None") else 0.0 for x in eval_cols["A1Cresult"]], dtype=np.float64)
    a1c_h_tr = np.array([1.0 if (x in (">7", ">8")) else 0.0 for x in train["A1Cresult"]], dtype=np.float64)
    a1c_h_ev = np.array([1.0 if (x in (">7", ">8")) else 0.0 for x in eval_cols["A1Cresult"]], dtype=np.float64)
    add_num("a1c_tested", a1c_tr, a1c_ev)
    add_num("a1c_high", a1c_h_tr, a1c_h_ev)
    add_num("a1c_high_no_change", a1c_h_tr * (1.0 - med_feats_tr["med_change_flag"]), a1c_h_ev * (1.0 - med_feats_ev["med_change_flag"]))

    glu_tr = np.array([1.0 if (x is not None and str(x) != "None") else 0.0 for x in train["max_glu_serum"]], dtype=np.float64)
    glu_ev = np.array([1.0 if (x is not None and str(x) != "None") else 0.0 for x in eval_cols["max_glu_serum"]], dtype=np.float64)
    glu_h_tr = np.array([1.0 if (x in (">200", ">300")) else 0.0 for x in train["max_glu_serum"]], dtype=np.float64)
    glu_h_ev = np.array([1.0 if (x in (">200", ">300")) else 0.0 for x in eval_cols["max_glu_serum"]], dtype=np.float64)
    add_num("glu_tested", glu_tr, glu_ev)
    add_num("glu_high", glu_h_tr, glu_h_ev)

    # 8. Admission & Discharge Flags
    d_disp_tr = np.array([str(x) if x is not None else "" for x in train["discharge_disposition_id"]], dtype=object)
    d_disp_ev = np.array([str(x) if x is not None else "" for x in eval_cols["discharge_disposition_id"]], dtype=object)
    add_num("disch_home", (d_disp_tr == "1").astype(float), (d_disp_ev == "1").astype(float))
    add_num("disch_snf", (d_disp_tr == "3").astype(float), (d_disp_ev == "3").astype(float))
    add_num("disch_home_health", (d_disp_tr == "6").astype(float), (d_disp_ev == "6").astype(float))
    add_num("disch_facility", np.isin(d_disp_tr, ["2", "3", "4", "5", "6", "22"]).astype(float), np.isin(d_disp_ev, ["2", "3", "4", "5", "6", "22"]).astype(float))
    add_num("disch_ama", (d_disp_tr == "7").astype(float), (d_disp_ev == "7").astype(float))

    a_src_tr = np.array([str(x) if x is not None else "" for x in train["admission_source_id"]], dtype=object)
    a_src_ev = np.array([str(x) if x is not None else "" for x in eval_cols["admission_source_id"]], dtype=object)
    add_num("admit_from_er", (a_src_tr == "7").astype(float), (a_src_ev == "7").astype(float))
    add_num("admit_transfer", np.isin(a_src_tr, ["4", "5", "6"]).astype(float), np.isin(a_src_ev, ["4", "5", "6"]).astype(float))
    add_num("admit_physician", (a_src_tr == "1").astype(float), (a_src_ev == "1").astype(float))

    a_typ_tr = np.array([str(x) if x is not None else "" for x in train["admission_type_id"]], dtype=object)
    a_typ_ev = np.array([str(x) if x is not None else "" for x in eval_cols["admission_type_id"]], dtype=object)
    add_num("admit_emergency", (a_typ_tr == "1").astype(float), (a_typ_ev == "1").astype(float))
    add_num("admit_urgent", (a_typ_tr == "2").astype(float), (a_typ_ev == "2").astype(float))
    add_num("admit_elective", (a_typ_tr == "3").astype(float), (a_typ_ev == "3").astype(float))

    # 9. Informative Missingness
    add_num("has_weight", np.array([1.0 if (w is not None and str(w).strip() not in ("", "?")) else 0.0 for w in train["weight"]], dtype=float),
            np.array([1.0 if (w is not None and str(w).strip() not in ("", "?")) else 0.0 for w in eval_cols["weight"]], dtype=float))
    add_num("has_payer_code", np.array([1.0 if (p is not None and str(p).strip() not in ("", "?")) else 0.0 for p in train["payer_code"]], dtype=float),
            np.array([1.0 if (p is not None and str(p).strip() not in ("", "?")) else 0.0 for p in eval_cols["payer_code"]], dtype=float))
    add_num("has_medical_specialty", np.array([1.0 if (s is not None and str(s).strip() not in ("", "?")) else 0.0 for s in train["medical_specialty"]], dtype=float),
            np.array([1.0 if (s is not None and str(s).strip() not in ("", "?")) else 0.0 for s in eval_cols["medical_specialty"]], dtype=float))

    # 10. Fit TabularEncoder
    encoder = TabularEncoder(min_count=15)
    encoder.fit(task, aug_train, extra_numeric=tuple(extra_numeric), extra_categorical=tuple(extra_categorical))
    x_train = encoder.transform(aug_train)
    x_eval = encoder.transform(aug_eval)
    cat_idx = encoder.categorical_indices

    # 11. Dual-Forest 5-Fold Bagging + Full-Data Model Blend
    cv_preds = np.zeros(n_ev, dtype=np.float64)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        xt, yt = x_train[trn_idx], y_train[trn_idx]

        m1 = lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.032,
            num_leaves=22,
            max_depth=6,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.3,
            reg_lambda=5.0,
            min_data_per_group=15,
            cat_smooth=20.0,
            cat_l2=10.0,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m1.fit(xt, yt, categorical_feature=cat_idx)
        p1 = m1.predict_proba(x_eval)[:, 1]

        m2 = lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.035,
            num_leaves=28,
            max_depth=7,
            min_child_samples=45,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.70,
            extra_trees=True,
            reg_alpha=0.2,
            reg_lambda=4.0,
            min_data_per_group=15,
            cat_smooth=20.0,
            cat_l2=10.0,
            random_state=101 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m2.fit(xt, yt, categorical_feature=cat_idx)
        p2 = m2.predict_proba(x_eval)[:, 1]

        m3 = lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.036,
            num_leaves=34,
            max_depth=8,
            min_child_samples=35,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.60,
            min_split_gain=0.01,
            reg_alpha=0.5,
            reg_lambda=6.0,
            min_data_per_group=15,
            cat_smooth=25.0,
            cat_l2=10.0,
            random_state=202 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m3.fit(xt, yt, categorical_feature=cat_idx)
        p3 = m3.predict_proba(x_eval)[:, 1]

        cv_preds += (0.35 * p1 + 0.35 * p2 + 0.30 * p3) / 5.0

    # Train two diverse models on 100% of x_train to capture full-sample patterns
    full_m1 = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.03,
        num_leaves=24,
        max_depth=6,
        min_child_samples=55,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.68,
        reg_alpha=0.3,
        reg_lambda=5.0,
        min_data_per_group=15,
        cat_smooth=20.0,
        cat_l2=10.0,
        random_state=777,
        n_jobs=1,
        verbose=-1,
    )
    full_m1.fit(x_train, y_train, categorical_feature=cat_idx)
    fp1 = full_m1.predict_proba(x_eval)[:, 1]

    full_m2 = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.03,
        num_leaves=30,
        max_depth=7,
        min_child_samples=45,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.72,
        extra_trees=True,
        reg_alpha=0.2,
        reg_lambda=4.0,
        min_data_per_group=15,
        cat_smooth=20.0,
        cat_l2=10.0,
        random_state=888,
        n_jobs=1,
        verbose=-1,
    )
    full_m2.fit(x_train, y_train, categorical_feature=cat_idx)
    fp2 = full_m2.predict_proba(x_eval)[:, 1]

    full_preds = 0.50 * fp1 + 0.50 * fp2

    # Blend 5-fold bagging (variance reduction) with full-data models (full sample power)
    final_preds = 0.70 * cv_preds + 0.30 * full_preds
    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
