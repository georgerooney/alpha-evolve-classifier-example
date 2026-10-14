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
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold


def _clean_code(c) -> str:
    if c is None:
        return "MISSING"
    s = str(c).strip()
    if not s or s == "?" or s == "None":
        return "MISSING"
    return s.split(".")[0].strip()
    n = len(code_arr)
    res = np.empty(n, dtype=object)
    for i in range(n):
        c = code_arr[i]
        if c is None or not isinstance(c, str) or c == "" or c == "?":
            res[i] = "MISSING"
        else:
            res[i] = c.split(".")[0].strip()
    return res


def _extract_icd9_groups(code_arr: np.ndarray) -> np.ndarray:
    n = len(code_arr)
    res = np.empty(n, dtype=object)
    for i in range(n):
        res[i] = icd9_group(code_arr[i])
    return res


def _compute_comorbidities(diag1: np.ndarray, diag2: np.ndarray, diag3: np.ndarray):
    n = len(diag1)
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

    htn_prefixes = ("401", "402", "403", "404", "405")
    ckd_prefixes = ("585", "586", "582", "583", "584")
    copd_prefixes = (
        "490", "491", "492", "493", "494", "495", "496",
        "480", "481", "482", "483", "484", "485", "486",
    )
    cad_prefixes = ("410", "411", "412", "413", "414")
    stroke_prefixes = ("430", "431", "432", "433", "434", "435", "436", "437", "438")
    pvd_prefixes = ("440", "441", "442", "443", "444", "445", "446", "447")

    for i in range(n):
        d1 = str(diag1[i]) if diag1[i] is not None else ""
        d2 = str(diag2[i]) if diag2[i] is not None else ""
        d3 = str(diag3[i]) if diag3[i] is not None else ""

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
            if any(d.startswith(p) for p in cad_prefixes):
                has_cad[i] = 1.0
            if any(d.startswith(p) for p in htn_prefixes):
                has_htn[i] = 1.0
            if any(d.startswith(p) for p in ckd_prefixes):
                has_ckd[i] = 1.0
            if any(d.startswith(p) for p in copd_prefixes):
                has_copd[i] = 1.0
            if d.startswith("276"):
                has_fluid[i] = 1.0
            if any(d.startswith(p) for p in stroke_prefixes):
                has_stroke[i] = 1.0
            if any(d.startswith(p) for p in pvd_prefixes):
                has_pvd[i] = 1.0

    comorbidity_count = (
        has_diabetes + has_diabetes_comp + has_chf + has_cad +
        has_htn + has_ckd + has_copd + has_fluid + has_stroke + has_pvd
    )
    return (
        has_diabetes,
        is_primary_diabetes,
        has_diabetes_comp,
        has_chf,
        has_cad,
        has_htn,
        has_ckd,
        has_copd,
        has_fluid,
        has_stroke,
        has_pvd,
        comorbidity_count,
    )


def _compute_med_features(cols: Dict[str, np.ndarray]):
    n = len(cols["time_in_hospital"])
    num_active = np.zeros(n, dtype=np.float64)
    num_up = np.zeros(n, dtype=np.float64)
    num_down = np.zeros(n, dtype=np.float64)
    num_change = np.zeros(n, dtype=np.float64)

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
                    num_change[i] += 1.0
                    num_active[i] += 1.0
                elif v == "Down":
                    num_down[i] += 1.0
                    num_change[i] += 1.0
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

    diabetes_med_flag = np.zeros(n, dtype=np.float64)
    if "diabetesMed" in cols:
        dm = cols["diabetesMed"]
        for i in range(n):
            if dm[i] == "Yes":
                diabetes_med_flag[i] = 1.0

    med_change_flag = np.zeros(n, dtype=np.float64)
    if "change" in cols:
        ch = cols["change"]
        for i in range(n):
            if ch[i] == "Ch":
                med_change_flag[i] = 1.0

    insulin_and_oral = ((insulin_treatment > 0) & (num_active > 1)).astype(np.float64)
    multiple_meds = (num_active >= 2).astype(np.float64)

    return (
        num_active, num_up, num_down, num_change,
        insulin_treatment, insulin_up, insulin_down, insulin_steady,
        metformin_treatment, diabetes_med_flag, med_change_flag,
        insulin_and_oral, multiple_meds
    )


def _augment_columns(raw_cols: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    cols = dict(raw_cols)
    n = len(cols["time_in_hospital"])

    # 1. Diagnoses grouping & prefixes
    cols["diag_1_group"] = _extract_icd9_groups(cols["diag_1"])
    cols["diag_2_group"] = _extract_icd9_groups(cols["diag_2"])
    cols["diag_3_group"] = _extract_icd9_groups(cols["diag_3"])
    cols["diag_1_prefix"] = _extract_prefix(cols["diag_1"])
    cols["diag_2_prefix"] = _extract_prefix(cols["diag_2"])
    cols["diag_3_prefix"] = _extract_prefix(cols["diag_3"])

    # 2. Comorbidity flags
    (
        cols["has_diabetes_comorb"], cols["is_primary_diabetes"],
        cols["has_diabetes_complications"], cols["has_chf"], cols["has_cad"],
        cols["has_htn"], cols["has_ckd"], cols["has_copd_resp"],
        cols["has_fluid_elect"], cols["has_stroke"], cols["has_pvd"],
        cols["comorbidity_count"]
    ) = _compute_comorbidities(cols["diag_1"], cols["diag_2"], cols["diag_3"])

    # 3. Medications
    (
        cols["num_active_meds"], cols["num_meds_up"], cols["num_meds_down"],
        cols["num_meds_changed"], cols["insulin_treatment"], cols["insulin_up"],
        cols["insulin_down"], cols["insulin_steady"], cols["metformin_treatment"],
        cols["diabetes_med_prescribed"], cols["med_change_flag"],
        cols["insulin_and_oral"], cols["multiple_diabetes_meds"]
    ) = _compute_med_features(cols)

    # 4. Numeric Age mapping
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0
    }
    age_arr = cols["age"]
    age_num = np.empty(n, dtype=np.float64)
    for i in range(n):
        age_num[i] = age_map.get(age_arr[i], 65.0)
    cols["age_numeric"] = age_num

    # 5. Lab test indicators
    a1c_tested = np.zeros(n, dtype=np.float64)
    a1c_high = np.zeros(n, dtype=np.float64)
    a1c_normal = np.zeros(n, dtype=np.float64)
    if "A1Cresult" in cols:
        a1c = cols["A1Cresult"]
        for i in range(n):
            v = a1c[i]
            if v is not None and v != "None":
                a1c_tested[i] = 1.0
                if v == ">8" or v == ">7":
                    a1c_high[i] = 1.0
                elif v == "Norm":
                    a1c_normal[i] = 1.0
    cols["a1c_tested"] = a1c_tested
    cols["a1c_high"] = a1c_high
    cols["a1c_normal"] = a1c_normal
    cols["a1c_high_no_change"] = a1c_high * (1.0 - cols["med_change_flag"])

    glu_tested = np.zeros(n, dtype=np.float64)
    glu_high = np.zeros(n, dtype=np.float64)
    if "max_glu_serum" in cols:
        glu = cols["max_glu_serum"]
        for i in range(n):
            v = glu[i]
            if v is not None and v != "None":
                glu_tested[i] = 1.0
                if v == ">200" or v == ">300":
                    glu_high[i] = 1.0
    cols["glu_tested"] = glu_tested
    cols["glu_high"] = glu_high

    # 6. Admission and discharge
    disch_home = np.zeros(n, dtype=np.float64)
    disch_snf = np.zeros(n, dtype=np.float64)
    disch_ama = np.zeros(n, dtype=np.float64)
    if "discharge_disposition_id" in cols:
        dd = cols["discharge_disposition_id"]
        for i in range(n):
            v = str(dd[i]) if dd[i] is not None else ""
            if v == "1":
                disch_home[i] = 1.0
            elif v in ("3", "4", "6", "22"):
                disch_snf[i] = 1.0
            elif v == "7":
                disch_ama[i] = 1.0
    cols["discharge_home"] = disch_home
    cols["discharge_snf_care"] = disch_snf
    cols["discharge_ama"] = disch_ama

    admit_emerg = np.zeros(n, dtype=np.float64)
    admit_urg = np.zeros(n, dtype=np.float64)
    admit_elec = np.zeros(n, dtype=np.float64)
    if "admission_type_id" in cols:
        at = cols["admission_type_id"]
        for i in range(n):
            v = str(at[i]) if at[i] is not None else ""
            if v == "1":
                admit_emerg[i] = 1.0
            elif v == "2":
                admit_urg[i] = 1.0
            elif v == "3":
                admit_elec[i] = 1.0
    cols["admission_emergency"] = admit_emerg
    cols["admission_urgent"] = admit_urg
    cols["admission_elective"] = admit_elec

    admit_from_er = np.zeros(n, dtype=np.float64)
    admit_transfer = np.zeros(n, dtype=np.float64)
    if "admission_source_id" in cols:
        src = cols["admission_source_id"]
        for i in range(n):
            v = str(src[i]) if src[i] is not None else ""
            if v == "7":
                admit_from_er[i] = 1.0
            elif v in ("4", "5", "6"):
                admit_transfer[i] = 1.0
    cols["admission_from_er"] = admit_from_er
    cols["admission_transfer"] = admit_transfer

    # 7. Missingness
    cols["has_weight"] = np.array(
        [1.0 if (w is not None and w != "" and w != "?") else 0.0 for w in cols["weight"]],
        dtype=np.float64,
    )
    cols["has_payer_code"] = np.array(
        [1.0 if (p is not None and p != "" and p != "?") else 0.0 for p in cols["payer_code"]],
        dtype=np.float64,
    )
    cols["has_medical_specialty"] = np.array(
        [1.0 if (s is not None and s != "" and s != "?") else 0.0 for s in cols["medical_specialty"]],
        dtype=np.float64,
    )

    # 8. Utilization and intensity ratios
    inpatient = np.nan_to_num(cols.get("number_inpatient", 0.0))
    emergency = np.nan_to_num(cols.get("number_emergency", 0.0))
    outpatient = np.nan_to_num(cols.get("number_outpatient", 0.0))
    time_hosp = np.maximum(1.0, np.nan_to_num(cols.get("time_in_hospital", 1.0)))
    num_labs = np.nan_to_num(cols.get("num_lab_procedures", 0.0))
    num_procs = np.nan_to_num(cols.get("num_procedures", 0.0))
    num_meds = np.nan_to_num(cols.get("num_medications", 0.0))
    num_diags = np.maximum(1.0, np.nan_to_num(cols.get("number_diagnoses", 1.0)))

    total_prior = inpatient + emergency + outpatient
    cols["total_prior_visits"] = total_prior
    cols["has_prior_inpatient"] = (inpatient > 0).astype(np.float64)
    cols["has_prior_emergency"] = (emergency > 0).astype(np.float64)
    cols["has_prior_outpatient"] = (outpatient > 0).astype(np.float64)
    cols["has_prior_visits"] = (total_prior > 0).astype(np.float64)
    cols["inpatient_frequent"] = (inpatient >= 2.0).astype(np.float64)
    cols["inpatient_ratio"] = inpatient / (total_prior + 1.0)
    cols["emergency_ratio"] = emergency / (total_prior + 1.0)
    cols["log_number_inpatient"] = np.log1p(np.maximum(0.0, inpatient))
    cols["log_number_emergency"] = np.log1p(np.maximum(0.0, emergency))
    cols["log_number_outpatient"] = np.log1p(np.maximum(0.0, outpatient))
    cols["log_total_visits"] = np.log1p(np.maximum(0.0, total_prior))
    cols["weighted_prior_utilization"] = inpatient * 2.0 + emergency * 1.5 + outpatient * 0.5

    cols["lab_procedures_per_day"] = num_labs / time_hosp
    cols["medications_per_day"] = num_meds / time_hosp
    cols["procedures_per_day"] = num_procs / time_hosp
    cols["total_procedures"] = num_labs + num_procs
    cols["diagnoses_per_day"] = num_diags / time_hosp
    cols["medications_per_diagnosis"] = num_meds / num_diags
    cols["labs_per_medication"] = num_labs / (num_meds + 1.0)

    # 9. Key clinical interaction features
    cols["inpatient_x_diagnoses"] = inpatient * num_diags
    cols["inpatient_x_emergency"] = inpatient * emergency
    cols["is_elderly"] = (cols["age_numeric"] >= 75.0).astype(np.float64)
    cols["elderly_with_inpatient"] = cols["is_elderly"] * cols["has_prior_inpatient"]
    cols["high_utilizer"] = (total_prior >= 3.0).astype(np.float64)
    cols["change_and_insulin"] = cols["med_change_flag"] * cols["insulin_treatment"]
    cols["inpatient_per_day"] = inpatient / time_hosp

    return cols


def _add_target_encodings(
    train_dict: Dict[str, np.ndarray],
    eval_dict: Dict[str, np.ndarray],
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
):
    n_tr = len(y)
    n_ev = len(eval_dict["time_in_hospital"])
    prior = float(np.mean(y)) if n_tr > 0 else 0.114

    # Target encode diagnosis 3-digit prefixes using fold statistics
    d1_tr = train_dict["diag_1_prefix"]
    d2_tr = train_dict["diag_2_prefix"]
    d3_tr = train_dict["diag_3_prefix"]
    d1_ev = eval_dict["diag_1_prefix"]
    d2_ev = eval_dict["diag_2_prefix"]
    d3_ev = eval_dict["diag_3_prefix"]

    d1_te_tr = np.full(n_tr, prior, dtype=np.float64)
    d2_te_tr = np.full(n_tr, prior, dtype=np.float64)
    d3_te_tr = np.full(n_tr, prior, dtype=np.float64)

    rng = np.random.RandomState(seed)
    indices = np.arange(n_tr)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)
    smoothing_diag = 20.0

    for fold in folds:
        val_idx = fold
        tr_idx = np.setdiff1d(indices, val_idx)

        f_prior = float(np.mean(y[tr_idx])) if len(tr_idx) > 0 else prior
        sums = {}
        counts = {}
        for c, val in zip(d1_tr[tr_idx], y[tr_idx]):
            k = str(c) if c is not None else "__MISSING__"
            sums[k] = sums.get(k, 0.0) + val
            counts[k] = counts.get(k, 0.0) + 1.0

        for i in val_idx:
            k1 = str(d1_tr[i]) if d1_tr[i] is not None else "__MISSING__"
            d1_te_tr[i] = (sums.get(k1, 0.0) + smoothing_diag * f_prior) / (counts.get(k1, 0.0) + smoothing_diag)

            k2 = str(d2_tr[i]) if d2_tr[i] is not None else "__MISSING__"
            d2_te_tr[i] = (sums.get(k2, 0.0) + smoothing_diag * f_prior) / (counts.get(k2, 0.0) + smoothing_diag)

            k3 = str(d3_tr[i]) if d3_tr[i] is not None else "__MISSING__"
            d3_te_tr[i] = (sums.get(k3, 0.0) + smoothing_diag * f_prior) / (counts.get(k3, 0.0) + smoothing_diag)

    full_sums = {}
    full_counts = {}
    for c, val in zip(d1_tr, y):
        k = str(c) if c is not None else "__MISSING__"
        full_sums[k] = full_sums.get(k, 0.0) + val
        full_counts[k] = full_counts.get(k, 0.0) + 1.0

    d1_te_ev = np.full(n_ev, prior, dtype=np.float64)
    d2_te_ev = np.full(n_ev, prior, dtype=np.float64)
    d3_te_ev = np.full(n_ev, prior, dtype=np.float64)

    for j in range(n_ev):
        k1 = str(d1_ev[j]) if d1_ev[j] is not None else "__MISSING__"
        d1_te_ev[j] = (full_sums.get(k1, 0.0) + smoothing_diag * prior) / (full_counts.get(k1, 0.0) + smoothing_diag)

        k2 = str(d2_ev[j]) if d2_ev[j] is not None else "__MISSING__"
        d2_te_ev[j] = (full_sums.get(k2, 0.0) + smoothing_diag * prior) / (full_counts.get(k2, 0.0) + smoothing_diag)

        k3 = str(d3_ev[j]) if d3_ev[j] is not None else "__MISSING__"
        d3_te_ev[j] = (full_sums.get(k3, 0.0) + smoothing_diag * prior) / (full_counts.get(k3, 0.0) + smoothing_diag)

    train_dict["diag_1_prefix_te"] = d1_te_tr
    train_dict["diag_2_prefix_te"] = d2_te_tr
    train_dict["diag_3_prefix_te"] = d3_te_tr
    train_dict["diag_risk_max"] = np.maximum(d1_te_tr, np.maximum(d2_te_tr, d3_te_tr))
    train_dict["diag_risk_mean"] = (d1_te_tr + d2_te_tr + d3_te_tr) / 3.0

    eval_dict["diag_1_prefix_te"] = d1_te_ev
    eval_dict["diag_2_prefix_te"] = d2_te_ev
    eval_dict["diag_3_prefix_te"] = d3_te_ev
    eval_dict["diag_risk_max"] = np.maximum(d1_te_ev, np.maximum(d2_te_ev, d3_te_ev))
    eval_dict["diag_risk_mean"] = (d1_te_ev + d2_te_ev + d3_te_ev) / 3.0

    # Categorical columns with strong clinical signals
    single_te_cols = [
        ("discharge_disposition_id", "discharge_disp_te", 25.0),
        ("admission_source_id", "admission_src_te", 25.0),
        ("admission_type_id", "admission_type_te", 30.0),
        ("medical_specialty", "medical_specialty_te", 20.0),
        ("payer_code", "payer_code_te", 30.0),
    ]

    for raw_name, te_name, smooth in single_te_cols:
        tr_c = train_dict.get(raw_name)
        ev_c = eval_dict.get(raw_name)
        if tr_c is None or ev_c is None:
            continue

        oof_tr = np.full(n_tr, prior, dtype=np.float64)
        for fold in folds:
            val_idx = fold
            tr_idx = np.setdiff1d(indices, val_idx)
            f_prior = float(np.mean(y[tr_idx])) if len(tr_idx) > 0 else prior

            sums = {}
            counts = {}
            for c, val in zip(tr_c[tr_idx], y[tr_idx]):
                k = str(c) if c is not None else "__MISSING__"
                sums[k] = sums.get(k, 0.0) + val
                counts[k] = counts.get(k, 0.0) + 1.0

            for i in val_idx:
                k = str(tr_c[i]) if tr_c[i] is not None else "__MISSING__"
                oof_tr[i] = (sums.get(k, 0.0) + smooth * f_prior) / (counts.get(k, 0.0) + smooth)

        full_sums = {}
        full_counts = {}
        for c, val in zip(tr_c, y):
            k = str(c) if c is not None else "__MISSING__"
            full_sums[k] = full_sums.get(k, 0.0) + val
            full_counts[k] = full_counts.get(k, 0.0) + 1.0

        eval_te = np.full(n_ev, prior, dtype=np.float64)
        for j in range(n_ev):
            k = str(ev_c[j]) if ev_c[j] is not None else "__MISSING__"
            eval_te[j] = (full_sums.get(k, 0.0) + smooth * prior) / (full_counts.get(k, 0.0) + smooth)

        train_dict[te_name] = oof_tr
        eval_dict[te_name] = eval_te


from sklearn.model_selection import StratifiedKFold


def _clean_code(c) -> str:
    if c is None:
        return "MISSING"
    s = str(c).strip()
    if not s or s == "?" or s == "None":
        return "MISSING"
    return s.split(".")[0].strip()


def _compute_med_vectors(cols: Dict[str, np.ndarray]):
    n = len(cols["time_in_hospital"])
    num_active = np.zeros(n, dtype=np.float64)
    num_up = np.zeros(n, dtype=np.float64)
    num_down = np.zeros(n, dtype=np.float64)
    num_change = np.zeros(n, dtype=np.float64)

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
                    num_change[i] += 1.0
                    num_active[i] += 1.0
                elif v == "Down":
                    num_down[i] += 1.0
                    num_change[i] += 1.0
                    num_active[i] += 1.0
                elif v == "Steady":
                    num_active[i] += 1.0

    insulin_treatment = np.zeros(n, dtype=np.float64)
    insulin_changed = np.zeros(n, dtype=np.float64)
    if "insulin" in cols:
        ins = cols["insulin"]
        for i in range(n):
            v = ins[i]
            if v in ("Up", "Down", "Steady"):
                insulin_treatment[i] = 1.0
            if v in ("Up", "Down"):
                insulin_changed[i] = 1.0

    med_changed = np.zeros(n, dtype=np.float64)
    if "change" in cols:
        ch = cols["change"]
        for i in range(n):
            if ch[i] == "Ch":
                med_changed[i] = 1.0

    dm_med = np.zeros(n, dtype=np.float64)
    if "diabetesMed" in cols:
        dm = cols["diabetesMed"]
        for i in range(n):
            if dm[i] == "Yes":
                dm_med[i] = 1.0

    insulin_and_oral = ((insulin_treatment > 0) & (num_active > 1)).astype(np.float64)
    multiple_meds = (num_active >= 2).astype(np.float64)

    return (
        num_active, num_up, num_down, num_change,
        insulin_treatment, insulin_changed, med_changed, dm_med,
        insulin_and_oral, multiple_meds
    )


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit a tensor-factorized, hierarchical Bayesian ensemble for readmission risk scoring."""
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

    # 1. Diagnosis Parsing and Chapter Extraction
    p1_tr = np.array([_clean_code(c) for c in train["diag_1"]], dtype=object)
    p2_tr = np.array([_clean_code(c) for c in train["diag_2"]], dtype=object)
    p3_tr = np.array([_clean_code(c) for c in train["diag_3"]], dtype=object)
    p1_ev = np.array([_clean_code(c) for c in eval_cols["diag_1"]], dtype=object)
    p2_ev = np.array([_clean_code(c) for c in eval_cols["diag_2"]], dtype=object)
    p3_ev = np.array([_clean_code(c) for c in eval_cols["diag_3"]], dtype=object)

    add_cat("diag_1_prefix", p1_tr, p1_ev)
    add_cat("diag_2_prefix", p2_tr, p2_ev)
    add_cat("diag_3_prefix", p3_tr, p3_ev)

    _icd9 = globals().get("icd9_group")
    if _icd9 is None:
        try:
            from clinical_toolkit import icd9_group as _icd9
        except Exception:
            _icd9 = lambda c: "other"

    ch1_tr = np.array([_icd9(c) for c in train["diag_1"]], dtype=object)
    ch2_tr = np.array([_icd9(c) for c in train["diag_2"]], dtype=object)
    ch3_tr = np.array([_icd9(c) for c in train["diag_3"]], dtype=object)
    ch1_ev = np.array([_icd9(c) for c in eval_cols["diag_1"]], dtype=object)
    ch2_ev = np.array([_icd9(c) for c in eval_cols["diag_2"]], dtype=object)
    ch3_ev = np.array([_icd9(c) for c in eval_cols["diag_3"]], dtype=object)

    add_cat("diag_1_chapter", ch1_tr, ch1_ev)
    add_cat("diag_2_chapter", ch2_tr, ch2_ev)
    add_cat("diag_3_chapter", ch3_tr, ch3_ev)

    distinct_ch_tr = np.array([len({ch1_tr[i], ch2_tr[i], ch3_tr[i]} - {"missing"}) for i in range(n_tr)], dtype=np.float64)
    distinct_ch_ev = np.array([len({ch1_ev[i], ch2_ev[i], ch3_ev[i]} - {"missing"}) for i in range(n_ev)], dtype=np.float64)
    add_num("distinct_diag_chapters", distinct_ch_tr, distinct_ch_ev)

    # 2. Hierarchical Empirical Bayes Diagnosis Shrinkage
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    m_chap = 40.0
    m_code = 25.0

    te_diag_max_tr = np.full(n_tr, global_prior, dtype=np.float64)
    te_diag_mean_tr = np.full(n_tr, global_prior, dtype=np.float64)
    te_diag_sum_tr = np.full(n_tr, 3.0 * global_prior, dtype=np.float64)
    te_diag_p1_tr = np.full(n_tr, global_prior, dtype=np.float64)

    for trn_idx, val_idx in skf.split(np.zeros(n_tr), y_train):
        f_prior = float(np.mean(y_train[trn_idx]))

        # Chapter level
        chap_counts = {}
        chap_sums = {}
        for i in trn_idx:
            yi = float(y_train[i])
            for ch in (ch1_tr[i], ch2_tr[i], ch3_tr[i]):
                chap_counts[ch] = chap_counts.get(ch, 0) + 1
                chap_sums[ch] = chap_sums.get(ch, 0.0) + yi

        chap_prior = {}
        for ch, cnt in chap_counts.items():
            chap_prior[ch] = (chap_sums[ch] + m_chap * f_prior) / (cnt + m_chap)

        # Code level with chapter prior shrinkage
        code_counts = {}
        code_sums = {}
        code_to_chap = {}
        for i in trn_idx:
            yi = float(y_train[i])
            for p, ch in ((p1_tr[i], ch1_tr[i]), (p2_tr[i], ch2_tr[i]), (p3_tr[i], ch3_tr[i])):
                if p != "MISSING":
                    code_counts[p] = code_counts.get(p, 0) + 1
                    code_sums[p] = code_sums.get(p, 0.0) + yi
                    code_to_chap[p] = ch

        def _calc_oof(p, ch):
            if p == "MISSING":
                return f_prior
            cp = chap_prior.get(ch, f_prior)
            cnt = code_counts.get(p, 0)
            return (code_sums.get(p, 0.0) + m_code * cp) / (cnt + m_code)

        for i in val_idx:
            r1 = _calc_oof(p1_tr[i], ch1_tr[i])
            r2 = _calc_oof(p2_tr[i], ch2_tr[i])
            r3 = _calc_oof(p3_tr[i], ch3_tr[i])
            te_diag_max_tr[i] = max(r1, r2, r3)
            te_diag_mean_tr[i] = (r1 + r2 + r3) / 3.0
            te_diag_sum_tr[i] = r1 + r2 + r3
            te_diag_p1_tr[i] = r1

    # Full train statistics for eval
    full_chap_counts = {}
    full_chap_sums = {}
    for i in range(n_tr):
        yi = float(y_train[i])
        for ch in (ch1_tr[i], ch2_tr[i], ch3_tr[i]):
            full_chap_counts[ch] = full_chap_counts.get(ch, 0) + 1
            full_chap_sums[ch] = full_chap_sums.get(ch, 0.0) + yi

    full_chap_prior = {}
    for ch, cnt in full_chap_counts.items():
        full_chap_prior[ch] = (full_chap_sums[ch] + m_chap * global_prior) / (cnt + m_chap)

    full_code_counts = {}
    full_code_sums = {}
    for i in range(n_tr):
        yi = float(y_train[i])
        for p in (p1_tr[i], p2_tr[i], p3_tr[i]):
            if p != "MISSING":
                full_code_counts[p] = full_code_counts.get(p, 0) + 1
                full_code_sums[p] = full_code_sums.get(p, 0.0) + yi

    def _calc_full(p, ch):
        if p == "MISSING":
            return global_prior
        cp = full_chap_prior.get(ch, global_prior)
        cnt = full_code_counts.get(p, 0)
        return (full_code_sums.get(p, 0.0) + m_code * cp) / (cnt + m_code)

    te_diag_max_ev = np.empty(n_ev, dtype=np.float64)
    te_diag_mean_ev = np.empty(n_ev, dtype=np.float64)
    te_diag_sum_ev = np.empty(n_ev, dtype=np.float64)
    te_diag_p1_ev = np.empty(n_ev, dtype=np.float64)

    for j in range(n_ev):
        r1 = _calc_full(p1_ev[j], ch1_ev[j])
        r2 = _calc_full(p2_ev[j], ch2_ev[j])
        r3 = _calc_full(p3_ev[j], ch3_ev[j])
        te_diag_max_ev[j] = max(r1, r2, r3)
        te_diag_mean_ev[j] = (r1 + r2 + r3) / 3.0
        te_diag_sum_ev[j] = r1 + r2 + r3
        te_diag_p1_ev[j] = r1

    add_num("te_diag_max", te_diag_max_tr, te_diag_max_ev)
    add_num("te_diag_mean", te_diag_mean_tr, te_diag_mean_ev)
    add_num("te_diag_sum", te_diag_sum_tr, te_diag_sum_ev)
    add_num("te_diag_primary", te_diag_p1_tr, te_diag_p1_ev)

    # 3. Administrative Bayesian Target Encodings
    def compute_single_te(col_tr, col_ev, m=20.0):
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

    t_disp_tr, t_disp_ev = compute_single_te(train["discharge_disposition_id"], eval_cols["discharge_disposition_id"], m=25.0)
    t_src_tr, t_src_ev = compute_single_te(train["admission_source_id"], eval_cols["admission_source_id"], m=25.0)
    t_spec_tr, t_spec_ev = compute_single_te(train["medical_specialty"], eval_cols["medical_specialty"], m=30.0)
    t_adm_tr, t_adm_ev = compute_single_te(train["admission_type_id"], eval_cols["admission_type_id"], m=25.0)
    t_pyr_tr, t_pyr_ev = compute_single_te(train["payer_code"], eval_cols["payer_code"], m=30.0)

    add_num("te_discharge", t_disp_tr, t_disp_ev)
    add_num("te_admission_source", t_src_tr, t_src_ev)
    add_num("te_specialty", t_spec_tr, t_spec_ev)
    add_num("te_admission_type", t_adm_tr, t_adm_ev)
    add_num("te_payer_code", t_pyr_tr, t_pyr_ev)

    # Cross interaction TE: admission source x discharge disposition
    src_disp_tr = [f"{train['admission_source_id'][i]}_{train['discharge_disposition_id'][i]}" for i in range(n_tr)]
    src_disp_ev = [f"{eval_cols['admission_source_id'][j]}_{eval_cols['discharge_disposition_id'][j]}" for j in range(n_ev)]
    t_sd_tr, t_sd_ev = compute_single_te(src_disp_tr, src_disp_ev, m=35.0)
    add_num("te_src_x_disp", t_sd_tr, t_sd_ev)

    # 4. Multi-system Clinical Comorbidity Indicators
    def _detect_conditions(diag1, diag2, diag3):
        n = len(diag1)
        d_prim = np.zeros(n, dtype=np.float64)
        d_comp = np.zeros(n, dtype=np.float64)
        chf = np.zeros(n, dtype=np.float64)
        cad = np.zeros(n, dtype=np.float64)
        htn = np.zeros(n, dtype=np.float64)
        ckd = np.zeros(n, dtype=np.float64)
        copd = np.zeros(n, dtype=np.float64)
        fluid = np.zeros(n, dtype=np.float64)
        stroke = np.zeros(n, dtype=np.float64)
        pvd = np.zeros(n, dtype=np.float64)
        neo = np.zeros(n, dtype=np.float64)

        htn_p = ("401", "402", "403", "404", "405")
        ckd_p = ("585", "586", "582", "583", "584")
        copd_p = ("490", "491", "492", "493", "494", "495", "496", "480", "481", "482", "483", "484", "485", "486")
        cad_p = ("410", "411", "412", "413", "414")
        strk_p = ("430", "431", "432", "433", "434", "435", "436", "437", "438")
        pvd_p = ("440", "441", "442", "443", "444", "445", "446", "447")

        for i in range(n):
            d1 = str(diag1[i]) if diag1[i] is not None else ""
            d2 = str(diag2[i]) if diag2[i] is not None else ""
            d3 = str(diag3[i]) if diag3[i] is not None else ""

            if d1.startswith("250"):
                d_prim[i] = 1.0

            for d in (d1, d2, d3):
                if d.startswith("250.") and not d.startswith("250.0"):
                    d_comp[i] = 1.0
                if d.startswith("428"):
                    chf[i] = 1.0
                if any(d.startswith(p) for p in cad_p):
                    cad[i] = 1.0
                if any(d.startswith(p) for p in htn_p):
                    htn[i] = 1.0
                if any(d.startswith(p) for p in ckd_p):
                    ckd[i] = 1.0
                if any(d.startswith(p) for p in copd_p):
                    copd[i] = 1.0
                if d.startswith("276"):
                    fluid[i] = 1.0
                if any(d.startswith(p) for p in strk_p):
                    stroke[i] = 1.0
                if any(d.startswith(p) for p in pvd_p):
                    pvd[i] = 1.0
                if d and d[0].isdigit() and 140 <= int(d.split(".")[0][:3] if d[:3].isdigit() else 0) <= 239:
                    neo[i] = 1.0

        return d_prim, d_comp, chf, cad, htn, ckd, copd, fluid, stroke, pvd, neo

    (
        dp_tr, dc_tr, chf_tr, cad_tr, htn_tr, ckd_tr, copd_tr, fl_tr, strk_tr, pvd_tr, neo_tr
    ) = _detect_conditions(train["diag_1"], train["diag_2"], train["diag_3"])
    (
        dp_ev, dc_ev, chf_ev, cad_ev, htn_ev, ckd_ev, copd_ev, fl_ev, strk_ev, pvd_ev, neo_ev
    ) = _detect_conditions(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])

    comorb_cnt_tr = dp_tr + dc_tr + chf_tr + cad_tr + htn_tr + ckd_tr + copd_tr + fl_tr + strk_tr + pvd_tr + neo_tr
    comorb_cnt_ev = dp_ev + dc_ev + chf_ev + cad_ev + htn_ev + ckd_ev + copd_ev + fl_ev + strk_ev + pvd_ev + neo_ev

    add_num("has_primary_diabetes", dp_tr, dp_ev)
    add_num("has_diabetes_comp", dc_tr, dc_ev)
    add_num("has_chf", chf_tr, chf_ev)
    add_num("has_cad", cad_tr, cad_ev)
    add_num("has_htn", htn_tr, htn_ev)
    add_num("has_ckd", ckd_tr, ckd_ev)
    add_num("has_copd", copd_tr, copd_ev)
    add_num("has_fluid_elect", fl_tr, fl_ev)
    add_num("has_stroke", strk_tr, strk_ev)
    add_num("has_pvd", pvd_tr, pvd_ev)
    add_num("has_neoplasm", neo_tr, neo_ev)
    add_num("comorbidity_count", comorb_cnt_tr, comorb_cnt_ev)

    # 5. Low-Rank Comorbidity Latent Phenotyping via Truncated SVD
    mat_tr = np.column_stack([
        dp_tr, dc_tr, chf_tr, cad_tr, htn_tr, ckd_tr, copd_tr, fl_tr, strk_tr, pvd_tr, neo_tr,
        (ch1_tr == "circulatory").astype(float), (ch1_tr == "respiratory").astype(float),
        (ch1_tr == "digestive").astype(float), (ch1_tr == "genitourinary").astype(float),
    ])
    mat_ev = np.column_stack([
        dp_ev, dc_ev, chf_ev, cad_ev, htn_ev, ckd_ev, copd_ev, fl_ev, strk_ev, pvd_ev, neo_ev,
        (ch1_ev == "circulatory").astype(float), (ch1_ev == "respiratory").astype(float),
        (ch1_ev == "digestive").astype(float), (ch1_ev == "genitourinary").astype(float),
    ])
    svd = TruncatedSVD(n_components=3, random_state=42)
    pheno_tr = svd.fit_transform(mat_tr)
    pheno_ev = svd.transform(mat_ev)

    add_num("phenotype_svd_1", pheno_tr[:, 0], pheno_ev[:, 0])
    add_num("phenotype_svd_2", pheno_tr[:, 1], pheno_ev[:, 1])
    add_num("phenotype_svd_3", pheno_tr[:, 2], pheno_ev[:, 2])

    # 6. Longitudinal Utilization, Intensity & Velocity
    def safe_f(arr, def_val=0.0):
        return np.nan_to_num(np.array(arr, dtype=np.float64), nan=def_val)

    inp_tr, inp_ev = safe_f(train["number_inpatient"]), safe_f(eval_cols["number_inpatient"])
    emg_tr, emg_ev = safe_f(train["number_emergency"]), safe_f(eval_cols["number_emergency"])
    out_tr, out_ev = safe_f(train["number_outpatient"]), safe_f(eval_cols["number_outpatient"])
    los_tr = np.maximum(1.0, safe_f(train["time_in_hospital"], 1.0))
    los_ev = np.maximum(1.0, safe_f(eval_cols["time_in_hospital"], 1.0))
    med_tr, med_ev = safe_f(train["num_medications"]), safe_f(eval_cols["num_medications"])
    lab_tr, lab_ev = safe_f(train["num_lab_procedures"]), safe_f(eval_cols["num_lab_procedures"])
    proc_tr, proc_ev = safe_f(train["num_procedures"]), safe_f(eval_cols["num_procedures"])
    diag_tr = np.maximum(1.0, safe_f(train["number_diagnoses"], 1.0))
    diag_ev = np.maximum(1.0, safe_f(eval_cols["number_diagnoses"], 1.0))

    tot_tr = inp_tr + emg_tr + out_tr
    tot_ev = inp_ev + emg_ev + out_ev

    add_num("log_number_inpatient", np.log1p(inp_tr), np.log1p(inp_ev))
    add_num("log_number_emergency", np.log1p(emg_tr), np.log1p(emg_ev))
    add_num("log_total_visits", np.log1p(tot_tr), np.log1p(tot_ev))
    add_num("prior_acute_load", np.log1p(3.0 * inp_tr + 1.5 * emg_tr + 0.5 * out_tr), np.log1p(3.0 * inp_ev + 1.5 * emg_ev + 0.5 * out_ev))
    add_num("has_prior_inpatient", (inp_tr > 0).astype(float), (inp_ev > 0).astype(float))
    add_num("frequent_inpatient", (inp_tr >= 2).astype(float), (inp_ev >= 2).astype(float))
    add_num("inpatient_ratio", inp_tr / (tot_tr + 1.0), inp_ev / (tot_ev + 1.0))

    add_num("stay_intensity", (med_tr + lab_tr) / np.sqrt(los_tr), (med_ev + lab_ev) / np.sqrt(los_ev))
    add_num("meds_per_day", med_tr / los_tr, med_ev / los_ev)
    add_num("labs_per_day", lab_tr / los_tr, lab_ev / los_ev)
    add_num("procs_per_day", proc_tr / los_tr, proc_ev / los_ev)
    add_num("meds_per_diag", med_tr / diag_tr, med_ev / diag_ev)
    add_num("labs_per_med", lab_tr / (med_tr + 1.0), lab_ev / (med_ev + 1.0))

    # 7. Demographics and Care Pathway Indicators
    age_map = {"[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0, "[40-50)": 45.0,
               "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0, "[80-90)": 85.0, "[90-100)": 95.0}
    age_num_tr = np.array([age_map.get(x, 65.0) for x in train["age"]], dtype=np.float64)
    age_num_ev = np.array([age_map.get(x, 65.0) for x in eval_cols["age"]], dtype=np.float64)
    add_num("age_numeric", age_num_tr, age_num_ev)

    is_elderly_tr = (age_num_tr >= 75.0).astype(np.float64)
    is_elderly_ev = (age_num_ev >= 75.0).astype(np.float64)
    add_num("is_elderly", is_elderly_tr, is_elderly_ev)

    disch_home_tr = (train["discharge_disposition_id"] == "1").astype(np.float64)
    disch_home_ev = (eval_cols["discharge_disposition_id"] == "1").astype(np.float64)
    disch_snf_tr = (train["discharge_disposition_id"] == "3").astype(np.float64)
    disch_snf_ev = (eval_cols["discharge_disposition_id"] == "3").astype(np.float64)
    disch_fac_tr = np.isin(train["discharge_disposition_id"], ["2", "3", "4", "5", "6", "22"]).astype(np.float64)
    disch_fac_ev = np.isin(eval_cols["discharge_disposition_id"], ["2", "3", "4", "5", "6", "22"]).astype(np.float64)
    disch_ama_tr = (train["discharge_disposition_id"] == "7").astype(np.float64)
    disch_ama_ev = (eval_cols["discharge_disposition_id"] == "7").astype(np.float64)

    add_num("disch_home", disch_home_tr, disch_home_ev)
    add_num("disch_snf", disch_snf_tr, disch_snf_ev)
    add_num("disch_facility", disch_fac_tr, disch_fac_ev)
    add_num("disch_ama", disch_ama_tr, disch_ama_ev)

    adm_er_tr = (train["admission_source_id"] == "7").astype(np.float64)
    adm_er_ev = (eval_cols["admission_source_id"] == "7").astype(np.float64)
    adm_emg_tr = (train["admission_type_id"] == "1").astype(np.float64)
    adm_emg_ev = (eval_cols["admission_type_id"] == "1").astype(np.float64)

    add_num("admission_from_er", adm_er_tr, adm_er_ev)
    add_num("admission_emergency", adm_emg_tr, adm_emg_ev)

    add_num("has_weight", np.array([1.0 if (w is not None and w != "" and w != "?") else 0.0 for w in train["weight"]]),
            np.array([1.0 if (w is not None and w != "" and w != "?") else 0.0 for w in eval_cols["weight"]]))
    add_num("has_payer_code", np.array([1.0 if (p is not None and p != "" and p != "?") else 0.0 for p in train["payer_code"]]),
            np.array([1.0 if (p is not None and p != "" and p != "?") else 0.0 for p in eval_cols["payer_code"]]))
    add_num("has_specialty", np.array([1.0 if (s is not None and s != "" and s != "?") else 0.0 for s in train["medical_specialty"]]),
            np.array([1.0 if (s is not None and s != "" and s != "?") else 0.0 for s in eval_cols["medical_specialty"]]))

    # 8. Medications and Glycemic Trajectory
    (
        n_act_tr, n_up_tr, n_dn_tr, n_chg_tr, ins_tx_tr, ins_chg_tr,
        med_chg_tr, dm_med_tr, ins_oral_tr, mult_med_tr
    ) = _compute_med_vectors(train)
    (
        n_act_ev, n_up_ev, n_dn_ev, n_chg_ev, ins_tx_ev, ins_chg_ev,
        med_chg_ev, dm_med_ev, ins_oral_ev, mult_med_ev
    ) = _compute_med_vectors(eval_cols)

    add_num("num_active_meds", n_act_tr, n_act_ev)
    add_num("num_meds_up", n_up_tr, n_up_ev)
    add_num("num_meds_down", n_dn_tr, n_dn_ev)
    add_num("num_meds_changed", n_chg_tr, n_chg_ev)
    add_num("insulin_treatment", ins_tx_tr, ins_tx_ev)
    add_num("insulin_changed", ins_chg_tr, ins_chg_ev)
    add_num("med_changed", med_chg_tr, med_chg_ev)
    add_num("diabetes_med_flag", dm_med_tr, dm_med_ev)
    add_num("insulin_and_oral", ins_oral_tr, ins_oral_ev)
    add_num("multiple_diabetes_meds", mult_med_tr, mult_med_ev)

    a1c_tested_tr = (train["A1Cresult"] != "None").astype(np.float64)
    a1c_tested_ev = (eval_cols["A1Cresult"] != "None").astype(np.float64)
    a1c_high_tr = np.isin(train["A1Cresult"], [">7", ">8"]).astype(np.float64)
    a1c_high_ev = np.isin(eval_cols["A1Cresult"], [">7", ">8"]).astype(np.float64)
    a1c_norm_tr = (train["A1Cresult"] == "Norm").astype(np.float64)
    a1c_norm_ev = (eval_cols["A1Cresult"] == "Norm").astype(np.float64)
    glu_high_tr = np.isin(train["max_glu_serum"], [">200", ">300"]).astype(np.float64)
    glu_high_ev = np.isin(eval_cols["max_glu_serum"], [">200", ">300"]).astype(np.float64)

    add_num("a1c_tested", a1c_tested_tr, a1c_tested_ev)
    add_num("a1c_high", a1c_high_tr, a1c_high_ev)
    add_num("a1c_normal", a1c_norm_tr, a1c_norm_ev)
    add_num("glu_high", glu_high_tr, glu_high_ev)
    add_num("a1c_high_no_change", a1c_high_tr * (1.0 - med_chg_tr), a1c_high_ev * (1.0 - med_chg_ev))

    # 9. Multilinear Tensor Contraction Features (Low-Rank Interaction Factorization)
    add_num("tensor_destab",
            np.log1p(inp_tr) * (1.0 + med_chg_tr + ins_chg_tr) * (med_tr / los_tr),
            np.log1p(inp_ev) * (1.0 + med_chg_ev + ins_chg_ev) * (med_ev / los_ev))
    add_num("tensor_frailty",
            (is_elderly_tr + disch_fac_tr) * np.log1p(inp_tr) * (1.0 + comorb_cnt_tr),
            (is_elderly_ev + disch_fac_ev) * np.log1p(inp_ev) * (1.0 + comorb_cnt_ev))
    add_num("tensor_cardiorenal",
            (chf_tr + fl_tr) * (ckd_tr + htn_tr),
            (chf_ev + fl_ev) * (ckd_ev + htn_ev))
    add_num("tensor_metabolic_vasc",
            dc_tr * (cad_tr + pvd_tr + strk_tr),
            dc_ev * (cad_ev + pvd_ev + strk_ev))
    add_num("tensor_inertia",
            a1c_high_tr * (1.0 - med_chg_tr) * np.log1p(inp_tr),
            a1c_high_ev * (1.0 - med_chg_ev) * np.log1p(inp_ev))
    add_num("tensor_velocity",
            (np.log1p(inp_tr) / np.sqrt(los_tr)) * (diag_tr / (med_tr + 1.0)),
            (np.log1p(inp_ev) / np.sqrt(los_ev)) * (diag_ev / (med_ev + 1.0)))
    add_num("tensor_acute_pathway",
            adm_er_tr * disch_fac_tr * np.log1p(inp_tr),
            adm_er_ev * disch_fac_ev * np.log1p(inp_ev))

    # 10. Fit Tabular Encoder
    encoder = TabularEncoder(min_count=15)
    encoder.fit(task, aug_train, extra_numeric=tuple(extra_numeric), extra_categorical=tuple(extra_categorical))
    x_train = encoder.transform(aug_train)
    x_eval = encoder.transform(aug_eval)
    cat_idx = encoder.categorical_indices

    # 11. Cross-Validated Diverse GBDT Ensemble
    p1_eval_total = np.zeros(n_ev, dtype=np.float64)
    p2_eval_total = np.zeros(n_ev, dtype=np.float64)
    p3_eval_total = np.zeros(n_ev, dtype=np.float64)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        xt, yt = x_train[trn_idx], y_train[trn_idx]

        # Model 1: Balanced Main GBDT
        m1 = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.033,
            num_leaves=24,
            max_depth=6,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=5.0,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m1.fit(xt, yt, categorical_feature=cat_idx)
        p1_eval_total += m1.predict_proba(x_eval)[:, 1]

        # Model 2: Extremely Randomized Forest GBDT (Orthogonal Splits)
        m2 = lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.035,
            num_leaves=28,
            max_depth=6,
            min_child_samples=45,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.65,
            extra_trees=True,
            reg_alpha=0.1,
            reg_lambda=4.0,
            random_state=101 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m2.fit(xt, yt, categorical_feature=cat_idx)
        p2_eval_total += m2.predict_proba(x_eval)[:, 1]

        # Model 3: Deep Interaction GBDT with Feature Fractionation
        m3 = lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.035,
            num_leaves=32,
            max_depth=7,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            min_split_gain=0.01,
            reg_alpha=0.3,
            reg_lambda=6.0,
            random_state=202 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m3.fit(xt, yt, categorical_feature=cat_idx)
        p3_eval_total += m3.predict_proba(x_eval)[:, 1]

    # Average raw probability predictions per model family
    p1_avg = p1_eval_total / 5.0
    p2_avg = p2_eval_total / 5.0
    p3_avg = p3_eval_total / 5.0

    # 12. Quantile Rank-Averaged Ensemble Optimization
    r1 = (rankdata(p1_avg) - 0.5) / n_ev
    r2 = (rankdata(p2_avg) - 0.5) / n_ev
    r3 = (rankdata(p3_avg) - 0.5) / n_ev

    blend_rank = 0.38 * r1 + 0.34 * r2 + 0.28 * r3
    raw_blend = 0.38 * p1_avg + 0.34 * p2_avg + 0.28 * p3_avg

    # Monotonic quantile recalibration to exact empirical risk distribution
    calibrated_preds = np.quantile(raw_blend, np.clip(blend_rank, 0.0, 1.0))
    return np.clip(calibrated_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
