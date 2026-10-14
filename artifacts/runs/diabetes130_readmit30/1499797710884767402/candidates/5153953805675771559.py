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


def _extract_prefix(code_arr: np.ndarray) -> np.ndarray:
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


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit an ensemble on augmented training data and return readmission probabilities."""
    aug_train = _augment_columns(train)
    aug_eval = _augment_columns(eval_cols)
    _add_target_encodings(aug_train, aug_eval, y_train)

    extra_numeric = [
        "has_diabetes_comorb", "is_primary_diabetes", "has_diabetes_complications",
        "has_chf", "has_cad", "has_htn", "has_ckd", "has_copd_resp", "has_fluid_elect",
        "has_stroke", "has_pvd", "comorbidity_count",
        "num_active_meds", "num_meds_up", "num_meds_down", "num_meds_changed",
        "insulin_treatment", "insulin_up", "insulin_down", "insulin_steady",
        "metformin_treatment", "diabetes_med_prescribed", "med_change_flag",
        "insulin_and_oral", "multiple_diabetes_meds",
        "age_numeric",
        "a1c_tested", "a1c_high", "a1c_normal", "a1c_high_no_change",
        "glu_tested", "glu_high",
        "discharge_home", "discharge_snf_care", "discharge_ama",
        "admission_emergency", "admission_urgent", "admission_elective",
        "admission_from_er", "admission_transfer",
        "has_weight", "has_payer_code", "has_medical_specialty",
        "total_prior_visits", "has_prior_inpatient", "has_prior_emergency",
        "has_prior_outpatient", "has_prior_visits", "inpatient_frequent",
        "inpatient_ratio", "emergency_ratio",
        "log_number_inpatient", "log_number_emergency", "log_number_outpatient",
        "log_total_visits", "weighted_prior_utilization",
        "lab_procedures_per_day", "medications_per_day", "procedures_per_day",
        "total_procedures", "diagnoses_per_day", "medications_per_diagnosis",
        "labs_per_medication",
        "inpatient_x_diagnoses", "inpatient_x_emergency", "is_elderly",
        "elderly_with_inpatient", "high_utilizer", "change_and_insulin",
        "inpatient_per_day",
        "diag_1_prefix_te", "diag_2_prefix_te", "diag_3_prefix_te",
        "diag_risk_max", "diag_risk_mean",
        "discharge_disp_te", "admission_src_te", "admission_type_te",
        "medical_specialty_te", "payer_code_te",
    ]

    extra_categorical = [
        "diag_1_group", "diag_2_group", "diag_3_group",
        "diag_1_prefix", "diag_2_prefix", "diag_3_prefix",
    ]

    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, aug_train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(aug_train)
    x_eval = encoder.transform(aug_eval)

    # Heterogeneous LightGBM models with complementary capacities and regularization
    models = [
        lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.035,
            num_leaves=18,
            max_depth=5,
            min_child_samples=60,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=6.0,
            min_data_per_group=20,
            cat_smooth=20.0,
            cat_l2=10.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=24,
            max_depth=6,
            min_child_samples=45,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=1.0,
            reg_lambda=10.0,
            min_data_per_group=20,
            cat_smooth=20.0,
            cat_l2=10.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.04,
            num_leaves=20,
            max_depth=5,
            min_child_samples=75,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=4.0,
            min_data_per_group=20,
            cat_smooth=20.0,
            cat_l2=10.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.035,
            num_leaves=14,
            max_depth=4,
            min_child_samples=50,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=1.5,
            reg_lambda=8.0,
            min_data_per_group=20,
            cat_smooth=20.0,
            cat_l2=10.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = np.zeros(x_eval.shape[0], dtype=np.float64)
    for model in models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1]

    preds /= len(models)
    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
