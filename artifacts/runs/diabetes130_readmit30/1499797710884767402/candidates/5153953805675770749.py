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

        # Age parsing & subgroup protection
        age_num = np.array([_parse_age(a) for a in data["age"]], dtype=np.float64)
        data["age_num"] = age_num
        data["is_elderly"] = (age_num >= 75.0).astype(np.float64)
        data["is_very_elderly"] = (age_num >= 85.0).astype(np.float64)
        data["is_young"] = (age_num < 50.0).astype(np.float64)

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

        num_oral = np.maximum(0.0, active_cnt - insulin_act)
        data["num_oral_meds"] = num_oral
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
        "age_num",
        "is_elderly",
        "is_very_elderly",
        "is_young",
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
        "age_times_diagnoses",
        "age_times_comorbidity",
        "inpatient_times_comorbidity",
        "inpatient_times_lace",
        "age_times_inpatient",
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

    # 4. Fit TabularEncoder on all task + derived features
    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(
        task,
        train_data,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_eval = encoder.transform(eval_data)

    # 5. Compact, low-rank regularized ensemble using native Fisher categorical optimization
    models = [
        # Model 1: Balanced capacity tree (depth 6, 31 leaves)
        lgb.LGBMClassifier(
            n_estimators=460,
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
        # Model 2: Shallow regularized tree (depth 4, 15 leaves) - high stability on rare subgroups
        lgb.LGBMClassifier(
            n_estimators=500,
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
        # Model 3: Wide feature fraction, moderate depth (depth 5, 24 leaves)
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
        # Model 4: Extremely Randomized Trees (extra_trees=True) - orthogonal split diversity
        lgb.LGBMClassifier(
            n_estimators=460,
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
        # Model 5: Balanced medium-depth with large min_child_samples (depth 5, 28 leaves)
        lgb.LGBMClassifier(
            n_estimators=450,
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

    preds = np.zeros(x_eval.shape[0], dtype=np.float64)
    cat_indices = encoder.categorical_indices
    for model in models:
        model.fit(x_train, y_train, categorical_feature=cat_indices)
        preds += model.predict_proba(x_eval)[:, 1]

    preds /= len(models)
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
