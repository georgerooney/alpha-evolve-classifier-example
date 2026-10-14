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


def _target_encode(
    train_vals: np.ndarray,
    eval_vals: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    m_smoothing: float = 15.0
) -> Tuple[np.ndarray, np.ndarray]:
    n_train = len(y)
    
    def clean_val(v):
        if v is None or str(v) == 'nan' or str(v) == 'None' or str(v) == '?':
            return 'missing'
        return str(v).strip()
    
    train_clean = np.array([clean_val(v) for v in train_vals], dtype=object)
    eval_clean = np.array([clean_val(v) for v in eval_vals], dtype=object)
    
    global_mean = float(np.mean(y))
    train_encoded = np.zeros(n_train, dtype=np.float64)
    
    indices = np.arange(n_train)
    fold_ids = indices % n_folds
    
    for f in range(n_folds):
        train_idx = fold_ids != f
        val_idx = fold_ids == f
        
        y_f = y[train_idx]
        cats_f = train_clean[train_idx]
        
        unique_cats, inverse, counts = np.unique(cats_f, return_inverse=True, return_counts=True)
        sums = np.zeros_like(counts, dtype=np.float64)
        np.add.at(sums, inverse, y_f)
        
        encoded_vals = (sums + m_smoothing * global_mean) / (counts + m_smoothing)
        mapping = dict(zip(unique_cats, encoded_vals))
        
        val_cats = train_clean[val_idx]
        train_encoded[val_idx] = np.array([mapping.get(c, global_mean) for c in val_cats], dtype=np.float64)
        
    # Fit on all train
    unique_cats, inverse, counts = np.unique(train_clean, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=np.float64)
    np.add.at(sums, inverse, y)
    encoded_vals = (sums + m_smoothing * global_mean) / (counts + m_smoothing)
    global_mapping = dict(zip(unique_cats, encoded_vals))
    
    eval_encoded = np.array([global_mapping.get(c, global_mean) for c in eval_clean], dtype=np.float64)
    
    return train_encoded, eval_encoded


def _pooled_diag_encode(
    train_diags: Tuple[np.ndarray, np.ndarray, np.ndarray],
    eval_diags: Tuple[np.ndarray, np.ndarray, np.ndarray],
    y: np.ndarray,
    n_folds: int = 5,
    m_smoothing: float = 25.0,
):
    n_train = len(y)
    def clean(v):
        if v is None or str(v) in ('nan', 'None', '?'):
            return 'missing'
        return str(v).strip()

    t1 = np.array([clean(v) for v in train_diags[0]], dtype=object)
    t2 = np.array([clean(v) for v in train_diags[1]], dtype=object)
    t3 = np.array([clean(v) for v in train_diags[2]], dtype=object)
    e1 = np.array([clean(v) for v in eval_diags[0]], dtype=object)
    e2 = np.array([clean(v) for v in eval_diags[1]], dtype=object)
    e3 = np.array([clean(v) for v in eval_diags[2]], dtype=object)

    global_mean = float(np.mean(y))
    tr1 = np.zeros(n_train, dtype=np.float64)
    tr2 = np.zeros(n_train, dtype=np.float64)
    tr3 = np.zeros(n_train, dtype=np.float64)
    fold_ids = np.arange(n_train) % n_folds

    for f in range(n_folds):
        tr = fold_ids != f
        val = fold_ids == f
        cats = np.concatenate([t1[tr], t2[tr], t3[tr]])
        targets = np.concatenate([y[tr], y[tr], y[tr]])
        unq, inv, cnt = np.unique(cats, return_inverse=True, return_counts=True)
        sums = np.zeros_like(cnt, dtype=np.float64)
        np.add.at(sums, inv, targets)
        mapping = dict(zip(unq, (sums + m_smoothing * global_mean) / (cnt + m_smoothing)))

        tr1[val] = np.array([mapping.get(c, global_mean) for c in t1[val]], dtype=np.float64)
        tr2[val] = np.array([mapping.get(c, global_mean) for c in t2[val]], dtype=np.float64)
        tr3[val] = np.array([mapping.get(c, global_mean) for c in t3[val]], dtype=np.float64)

    cats_all = np.concatenate([t1, t2, t3])
    targets_all = np.concatenate([y, y, y])
    unq, inv, cnt = np.unique(cats_all, return_inverse=True, return_counts=True)
    sums = np.zeros_like(cnt, dtype=np.float64)
    np.add.at(sums, inv, targets_all)
    all_map = dict(zip(unq, (sums + m_smoothing * global_mean) / (cnt + m_smoothing)))

    ev1 = np.array([all_map.get(c, global_mean) for c in e1], dtype=np.float64)
    ev2 = np.array([all_map.get(c, global_mean) for c in e2], dtype=np.float64)
    ev3 = np.array([all_map.get(c, global_mean) for c in e3], dtype=np.float64)

    return (tr1, tr2, tr3), (ev1, ev2, ev3)


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

    # 1. Diagnosis ICD-9 grouping
    for diag_col in ("diag_1", "diag_2", "diag_3"):
        grp_name = f"{diag_col}_group"
        train_data[grp_name] = np.array([icd9(c) for c in train[diag_col]], dtype=object)
        eval_data[grp_name] = np.array([icd9(c) for c in eval_cols[diag_col]], dtype=object)
        extra_categorical.append(grp_name)

    # 2. Vectorized comorbidity flags decomposed from ICD-9 chapters
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
        has_chf = (np.char.startswith(d1_s, "428") | np.char.startswith(d2_s, "428") | np.char.startswith(d3_s, "428")).astype(np.float64)

        has_hypertension = (np.char.startswith(d1_s, "401") | np.char.startswith(d1_s, "402") | np.char.startswith(d1_s, "403") | np.char.startswith(d1_s, "404") | np.char.startswith(d1_s, "405") |
                            np.char.startswith(d2_s, "401") | np.char.startswith(d2_s, "402") | np.char.startswith(d2_s, "403") | np.char.startswith(d2_s, "404") | np.char.startswith(d2_s, "405") |
                            np.char.startswith(d3_s, "401") | np.char.startswith(d3_s, "402") | np.char.startswith(d3_s, "403") | np.char.startswith(d3_s, "404") | np.char.startswith(d3_s, "405")).astype(np.float64)
        
        has_ckd = (np.char.startswith(d1_s, "585") | np.char.startswith(d2_s, "585") | np.char.startswith(d3_s, "585")).astype(np.float64)
        has_ketoacidosis = (np.char.startswith(d1_s, "250.1") | np.char.startswith(d2_s, "250.1") | np.char.startswith(d3_s, "250.1")).astype(np.float64)
        
        is_diab1 = np.char.startswith(d1_s, "250")
        is_simple1 = (d1_s == "250") | np.char.startswith(d1_s, "250.0")
        is_diab2 = np.char.startswith(d2_s, "250")
        is_simple2 = (d2_s == "250") | np.char.startswith(d2_s, "250.0")
        is_diab3 = np.char.startswith(d3_s, "250")
        is_simple3 = (d3_s == "250") | np.char.startswith(d3_s, "250.0")
        
        has_diab_comp = ((is_diab1 & ~is_simple1) | (is_diab2 & ~is_simple2) | (is_diab3 & ~is_simple3)).astype(np.float64)

        data["has_diabetes_diag"] = has_diab
        data["has_chf_diag"] = has_chf
        data["has_circ_diag"] = has_circ
        data["has_resp_diag"] = has_resp
        data["has_renal_diag"] = has_renal
        data["has_neoplasm_diag"] = has_neoplasm
        data["comorbidity_score"] = has_diab + has_chf + has_circ + has_resp + has_renal + has_neoplasm
        data["has_hypertension"] = has_hypertension
        data["has_ckd"] = has_ckd
        data["has_ketoacidosis"] = has_ketoacidosis
        data["has_diabetes_with_complications"] = has_diab_comp

    extra_numeric.extend([
        "has_diabetes_diag",
        "has_chf_diag",
        "has_circ_diag",
        "has_resp_diag",
        "has_renal_diag",
        "has_neoplasm_diag",
        "comorbidity_score",
        "has_hypertension",
        "has_ckd",
        "has_ketoacidosis",
        "has_diabetes_with_complications",
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
        data["inpatient_ratio"] = (inpat + 0.1) / (total_visits + 1.0)
        data["emergency_ratio"] = (emerg + 0.1) / (total_visits + 1.0)

        data["med_per_day"] = num_meds / hosp_days
        data["labs_per_day"] = num_labs / hosp_days
        data["procs_per_day"] = num_procs / hosp_days
        data["meds_per_diag"] = num_meds / num_diags
        data["labs_per_med"] = num_labs / (num_meds + 1.0)

        # Age parsing
        data["age_num"] = np.array([_parse_age(a) for a in data["age"]], dtype=np.float64)

        # Interactions
        data["age_times_diagnoses"] = data["age_num"] * num_diags
        data["age_times_inpatient"] = data["age_num"] * inpat
        data["age_times_comorbidity"] = data["age_num"] * data["comorbidity_score"]
        data["inpatient_times_comorbidity"] = inpat * data["comorbidity_score"]

        # Discharge & admission indicators
        disp = data["discharge_disposition_id"]
        data["discharged_home"] = (disp == "1").astype(np.float64)
        data["discharged_facility"] = np.isin(disp, ["3", "4", "5", "6", "22", "23", "24"]).astype(np.float64)
        data["discharged_home_health"] = (disp == "6").astype(np.float64)
        data["discharged_ama"] = (disp == "7").astype(np.float64)
        data["discharged_snf_icf"] = np.isin(disp, ["3", "4", "24"]).astype(np.float64)
        data["discharged_other_hospital_rehab"] = np.isin(disp, ["2", "5", "22", "23", "25"]).astype(np.float64)

        # Admission details
        adm_type = data["admission_type_id"]
        data["admission_emergency"] = np.isin(adm_type, ["1", "2", "7"]).astype(np.float64)
        data["admission_elective"] = (adm_type == "3").astype(np.float64)

        adm_src = data["admission_source_id"]
        data["admission_source_er"] = (adm_src == "7").astype(np.float64)
        data["admission_source_transfer"] = np.isin(adm_src, ["4", "5", "6"]).astype(np.float64)
        data["admission_source_referral"] = np.isin(adm_src, ["1", "2", "3"]).astype(np.float64)

        # Lab indicators
        data["a1c_tested"] = np.isin(data["A1Cresult"], ["Norm", ">7", ">8"]).astype(np.float64)
        data["a1c_high"] = np.isin(data["A1Cresult"], [">7", ">8"]).astype(np.float64)
        data["glu_high"] = np.isin(data["max_glu_serum"], [">200", ">300"]).astype(np.float64)
        data["a1c_score"] = np.array([_map_a1c(v) for v in data["A1Cresult"]], dtype=np.float64)
        data["glu_score"] = np.array([_map_glu(v) for v in data["max_glu_serum"]], dtype=np.float64)

        # Medication summary (rank-1 projection across the 23 medication slice)
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
        data["change_flag"] = (data["change"] == "Ch").astype(np.float64)
        data["diabetes_med_flag"] = (data["diabetesMed"] == "Yes").astype(np.float64)
        data["insulin_active"] = (data["insulin"] != "No").astype(np.float64)
        data["insulin_is_up"] = (data["insulin"] == "Up").astype(np.float64)
        data["insulin_is_down"] = (data["insulin"] == "Down").astype(np.float64)

        metformin_active = (data["metformin"] != "No").astype(np.float64) if "metformin" in data else np.zeros(n_rows, dtype=np.float64)
        data["insulin_and_metformin"] = (data["insulin_active"] * metformin_active).astype(np.float64)
        data["high_diagnoses"] = (num_diags > 8.0).astype(np.float64)

        weight_val = data["weight"]
        data["weight_not_missing"] = np.array([1.0 if (w is not None and str(w).strip() != "" and str(w).strip() != "?") else 0.0 for w in weight_val], dtype=np.float64)

    extra_numeric.extend([
        "total_prior_visits",
        "has_prior_inpatient",
        "inpatient_ratio",
        "emergency_ratio",
        "med_per_day",
        "labs_per_day",
        "procs_per_day",
        "meds_per_diag",
        "labs_per_med",
        "age_num",
        "age_times_diagnoses",
        "age_times_inpatient",
        "age_times_comorbidity",
        "inpatient_times_comorbidity",
        "discharged_home",
        "discharged_facility",
        "discharged_home_health",
        "discharged_ama",
        "discharged_snf_icf",
        "discharged_other_hospital_rehab",
        "admission_emergency",
        "admission_elective",
        "admission_source_er",
        "admission_source_transfer",
        "admission_source_referral",
        "a1c_tested",
        "a1c_high",
        "glu_high",
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
        "insulin_and_metformin",
        "high_diagnoses",
        "weight_not_missing",
    ])

    # 3.5 Target Encoding of high cardinality features
    # Pooled rank-1 diagnosis tensor factorization
    (d1_tr, d2_tr, d3_tr), (d1_ev, d2_ev, d3_ev) = _pooled_diag_encode(
        (train["diag_1"], train["diag_2"], train["diag_3"]),
        (eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"]),
        y_train,
        n_folds=5,
        m_smoothing=25.0,
    )
    for name, tr_arr, ev_arr in [
        ("diag_1_te", d1_tr, d1_ev),
        ("diag_2_te", d2_tr, d2_ev),
        ("diag_3_te", d3_tr, d3_ev),
        ("diag_te_max", np.maximum(d1_tr, np.maximum(d2_tr, d3_tr)), np.maximum(d1_ev, np.maximum(d2_ev, d3_ev))),
        ("diag_te_mean", (d1_tr + d2_tr + d3_tr) / 3.0, (d1_ev + d2_ev + d3_ev) / 3.0),
    ]:
        train_data[name] = tr_arr
        eval_data[name] = ev_arr
        extra_numeric.append(name)

    train_data["inpatient_times_diag_risk"] = np.nan_to_num(train_data["number_inpatient"].astype(np.float64), nan=0.0) * train_data["diag_te_max"]
    eval_data["inpatient_times_diag_risk"] = np.nan_to_num(eval_data["number_inpatient"].astype(np.float64), nan=0.0) * eval_data["diag_te_max"]
    extra_numeric.append("inpatient_times_diag_risk")

    te_cols = [
        "medical_specialty",
        "discharge_disposition_id",
        "admission_source_id",
        "payer_code"
    ]
    for col in te_cols:
        te_train, te_eval = _target_encode(
            train[col],
            eval_cols[col],
            y_train,
            n_folds=5,
            m_smoothing=15.0
        )
        te_name = f"{col}_te"
        train_data[te_name] = te_train
        eval_data[te_name] = te_eval
        extra_numeric.append(te_name)

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
        # Model 1: Deeper, higher capacity tree
        lgb.LGBMClassifier(
            n_estimators=450,
            learning_rate=0.025,
            num_leaves=31,
            max_depth=6,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0.3,
            reg_lambda=4.0,
            min_data_per_group=20,
            cat_smooth=15.0,
            cat_l2=8.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 2: Shallow, highly regularized tree for stability
        lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.02,
            num_leaves=16,
            max_depth=4,
            min_child_samples=60,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_alpha=0.8,
            reg_lambda=8.0,
            min_data_per_group=30,
            cat_smooth=20.0,
            cat_l2=12.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 3: Moderate depth with higher feature fraction
        lgb.LGBMClassifier(
            n_estimators=450,
            learning_rate=0.025,
            num_leaves=24,
            max_depth=5,
            min_child_samples=45,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=2.0,
            min_data_per_group=15,
            cat_smooth=10.0,
            cat_l2=4.0,
            random_state=456,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 4: Deep tree with high regularization
        lgb.LGBMClassifier(
            n_estimators=380,
            learning_rate=0.03,
            num_leaves=28,
            max_depth=5,
            min_child_samples=45,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=1.0,
            reg_lambda=10.0,
            min_data_per_group=25,
            cat_smooth=15.0,
            cat_l2=10.0,
            random_state=789,
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
