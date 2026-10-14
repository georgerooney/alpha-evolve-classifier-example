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


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    import lightgbm as lgb
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    train_copy = dict(train)
    eval_copy = dict(eval_cols)

    extra_numeric = []
    extra_categorical = []

    def add_num(name: str, trn_arr: np.ndarray, val_arr: np.ndarray) -> None:
        train_copy[name] = np.asarray(trn_arr, dtype=np.float64)
        eval_copy[name] = np.asarray(val_arr, dtype=np.float64)
        extra_numeric.append(name)

    def add_cat(name: str, trn_arr: np.ndarray, val_arr: np.ndarray) -> None:
        train_copy[name] = np.asarray(trn_arr, dtype=object)
        eval_copy[name] = np.asarray(val_arr, dtype=object)
        extra_categorical.append(name)

    def safe_num(arr: np.ndarray, default: float = 0.0) -> np.ndarray:
        res = np.array(arr, dtype=np.float64)
        return np.nan_to_num(res, nan=default)

    # 1. Age Mapping to Numeric
    age_map = {
        "[0-10)": 5.0,
        "[10-20)": 15.0,
        "[20-30)": 25.0,
        "[30-40)": 35.0,
        "[40-50)": 45.0,
        "[50-60)": 55.0,
        "[60-70)": 65.0,
        "[70-80)": 75.0,
        "[80-90)": 85.0,
        "[90-100)": 95.0,
    }
    age_trn = np.array([age_map.get(x, 65.0) for x in train["age"]], dtype=np.float64)
    age_eval = np.array([age_map.get(x, 65.0) for x in eval_cols["age"]], dtype=np.float64)
    add_num("age_num", age_trn, age_eval)
    add_num("is_elderly", (age_trn >= 70.0).astype(float), (age_eval >= 70.0).astype(float))
    add_num("is_very_elderly", (age_trn >= 80.0).astype(float), (age_eval >= 80.0).astype(float))

    # 2. Prior Utilization Signals
    inp_trn = safe_num(train["number_inpatient"])
    inp_eval = safe_num(eval_cols["number_inpatient"])
    emg_trn = safe_num(train["number_emergency"])
    emg_eval = safe_num(eval_cols["number_emergency"])
    out_trn = safe_num(train["number_outpatient"])
    out_eval = safe_num(eval_cols["number_outpatient"])

    tot_trn = inp_trn + emg_trn + out_trn
    tot_eval = inp_eval + emg_eval + out_eval
    add_num("total_prior_visits", tot_trn, tot_eval)
    add_num("has_prior_inpatient", (inp_trn > 0).astype(float), (inp_eval > 0).astype(float))
    add_num("has_multi_inpatient", (inp_trn > 1).astype(float), (inp_eval > 1).astype(float))
    add_num("has_prior_emergency", (emg_trn > 0).astype(float), (emg_eval > 0).astype(float))
    add_num("has_prior_outpatient", (out_trn > 0).astype(float), (out_eval > 0).astype(float))
    add_num("inpatient_ratio", inp_trn / (tot_trn + 1.0), inp_eval / (tot_eval + 1.0))
    add_num("emergency_ratio", emg_trn / (tot_trn + 1.0), emg_eval / (tot_eval + 1.0))
    add_num("log_inpatient", np.log1p(inp_trn), np.log1p(inp_eval))
    add_num("log_total_visits", np.log1p(tot_trn), np.log1p(tot_eval))

    # 3. Hospital Stay & Care Intensity Metrics
    los_trn = safe_num(train["time_in_hospital"], default=1.0)
    los_eval = safe_num(eval_cols["time_in_hospital"], default=1.0)
    los_trn_clip = np.maximum(los_trn, 1.0)
    los_eval_clip = np.maximum(los_eval, 1.0)

    lab_trn = safe_num(train["num_lab_procedures"])
    lab_eval = safe_num(eval_cols["num_lab_procedures"])
    prc_trn = safe_num(train["num_procedures"])
    prc_eval = safe_num(eval_cols["num_procedures"])
    med_trn = safe_num(train["num_medications"])
    med_eval = safe_num(eval_cols["num_medications"])
    diag_trn = safe_num(train["number_diagnoses"], default=1.0)
    diag_eval = safe_num(eval_cols["number_diagnoses"], default=1.0)

    tot_care_trn = lab_trn + prc_trn + med_trn
    tot_care_eval = lab_eval + prc_eval + med_eval
    add_num("total_care_load", tot_care_trn, tot_care_eval)
    add_num("med_per_day", med_trn / los_trn_clip, med_eval / los_eval_clip)
    add_num("labs_per_day", lab_trn / los_trn_clip, lab_eval / los_eval_clip)
    add_num("procs_per_day", prc_trn / los_trn_clip, prc_eval / los_eval_clip)
    add_num("care_per_day", tot_care_trn / los_trn_clip, tot_care_eval / los_eval_clip)
    add_num("diag_per_day", diag_trn / los_trn_clip, diag_eval / los_eval_clip)
    add_num("med_per_diag", med_trn / np.maximum(diag_trn, 1.0), med_eval / np.maximum(diag_eval, 1.0))

    # Stay interactions
    add_num("inpatient_x_los", inp_trn * los_trn, inp_eval * los_eval)
    add_num("inpatient_x_diag", inp_trn * diag_trn, inp_eval * diag_eval)
    add_num("med_x_diag", med_trn * diag_trn, med_eval * diag_eval)

    # 4. ICD-9 Diagnosis Grouping & Comorbidities
    _icd9_fn = globals().get("icd9_group", None)
    if _icd9_fn is None:
        try:
            from clinical_toolkit import icd9_group as _icd9_fn
        except Exception:
            pass

    def safe_icd9(code):
        if _icd9_fn is not None:
            try:
                return _icd9_fn(code)
            except Exception:
                pass
        if code is None or code == "?" or str(code).strip() == "" or code != code:
            return "missing"
        s = str(code).strip()
        if s.startswith(("V", "v", "E", "e")):
            return "other"
        try:
            val = float(s)
        except ValueError:
            return "other"
        if (390 <= val <= 459) or val == 785:
            return "circulatory"
        if (460 <= val <= 519) or val == 786:
            return "respiratory"
        if (520 <= val <= 579) or val == 787:
            return "digestive"
        if 250 <= val < 251:
            return "diabetes"
        if 800 <= val <= 999:
            return "injury"
        if 710 <= val <= 739:
            return "musculoskeletal"
        if (580 <= val <= 629) or val == 788:
            return "genitourinary"
        if 140 <= val <= 239:
            return "neoplasms"
        return "other"

    def get_3digit(code):
        if code is None or code == "?" or str(code).strip() == "" or code != code:
            return "missing"
        s = str(code).strip().split(".")[0]
        return s if s else "missing"

    for d in ["diag_1", "diag_2", "diag_3"]:
        add_cat(f"{d}_group", [safe_icd9(c) for c in train[d]], [safe_icd9(c) for c in eval_cols[d]])
        add_cat(f"{d}_3digit", [get_3digit(c) for c in train[d]], [get_3digit(c) for c in eval_cols[d]])

    def extract_comorbidities(cols_dict):
        n = len(cols_dict["time_in_hospital"])
        has_chf = np.zeros(n, dtype=np.float64)
        has_ckd = np.zeros(n, dtype=np.float64)
        has_copd = np.zeros(n, dtype=np.float64)
        has_htn = np.zeros(n, dtype=np.float64)
        has_diab = np.zeros(n, dtype=np.float64)
        has_comp_diab = np.zeros(n, dtype=np.float64)
        circ_cnt = np.zeros(n, dtype=np.float64)
        resp_cnt = np.zeros(n, dtype=np.float64)
        diab_cnt = np.zeros(n, dtype=np.float64)

        copd_pref = ("491", "492", "493", "494", "496", "480", "481", "482", "486")
        htn_pref = ("401", "402", "403", "404", "405")
        ckd_pref = ("585", "586", "582", "583", "584")

        for d in ["diag_1", "diag_2", "diag_3"]:
            for i, c in enumerate(cols_dict[d]):
                if c is None or c == "?" or c != c:
                    continue
                s = str(c).strip()
                grp = safe_icd9(s)
                if grp == "circulatory":
                    circ_cnt[i] += 1.0
                elif grp == "respiratory":
                    resp_cnt[i] += 1.0
                elif grp == "diabetes":
                    diab_cnt[i] += 1.0

                if s.startswith("428"):
                    has_chf[i] = 1.0
                if any(s.startswith(p) for p in ckd_pref):
                    has_ckd[i] = 1.0
                if any(s.startswith(p) for p in copd_pref):
                    has_copd[i] = 1.0
                if any(s.startswith(p) for p in htn_pref):
                    has_htn[i] = 1.0
                if s.startswith("250"):
                    has_diab[i] = 1.0
                    if "." in s and not s.split(".")[1].startswith("0"):
                        has_comp_diab[i] = 1.0

        chronic_cnt = has_chf + has_ckd + has_copd + has_htn + has_diab
        return {
            "has_chf": has_chf,
            "has_ckd": has_ckd,
            "has_copd": has_copd,
            "has_htn": has_htn,
            "has_diabetes_diag": has_diab,
            "has_comp_diabetes": has_comp_diab,
            "chronic_count": chronic_cnt,
            "circulatory_count": circ_cnt,
            "respiratory_count": resp_cnt,
            "diabetes_group_count": diab_cnt,
        }

    comorb_trn = extract_comorbidities(train)
    comorb_eval = extract_comorbidities(eval_cols)
    for k in comorb_trn:
        add_num(k, comorb_trn[k], comorb_eval[k])

    # 5. Medication Features
    diabetes_meds = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    ]

    def extract_med_stats(cols_dict):
        n = len(cols_dict["time_in_hospital"])
        active = np.zeros(n, dtype=np.float64)
        up = np.zeros(n, dtype=np.float64)
        down = np.zeros(n, dtype=np.float64)
        for med in diabetes_meds:
            if med in cols_dict:
                arr = cols_dict[med]
                u = (arr == "Up")
                d = (arr == "Down")
                s = (arr == "Steady")
                up += u
                down += d
                active += (u | d | s)
        return active, up, down, up + down

    act_t, up_t, dn_t, chg_t = extract_med_stats(train)
    act_e, up_e, dn_e, chg_e = extract_med_stats(eval_cols)
    add_num("num_active_meds", act_t, act_e)
    add_num("num_meds_up", up_t, up_e)
    add_num("num_meds_down", dn_t, dn_e)
    add_num("num_meds_changed", chg_t, chg_e)

    for key in ["insulin", "metformin", "glipizide", "glyburide"]:
        if key in train and key in eval_cols:
            add_num(f"{key}_prescribed", (train[key] != "No").astype(float), (eval_cols[key] != "No").astype(float))
            add_num(f"{key}_changed", np.isin(train[key], ["Up", "Down"]).astype(float), np.isin(eval_cols[key], ["Up", "Down"]).astype(float))

    add_num("change_flag", (train["change"] == "Ch").astype(float), (eval_cols["change"] == "Ch").astype(float))
    add_num("diabetesMed_flag", (train["diabetesMed"] == "Yes").astype(float), (eval_cols["diabetesMed"] == "Yes").astype(float))

    # 6. Admission / Discharge Dispositions & Lab Test Flags
    add_num("discharged_home", (train["discharge_disposition_id"] == "1").astype(float), (eval_cols["discharge_disposition_id"] == "1").astype(float))
    add_num("discharged_snf", (train["discharge_disposition_id"] == "3").astype(float), (eval_cols["discharge_disposition_id"] == "3").astype(float))
    add_num("discharged_home_health", (train["discharge_disposition_id"] == "6").astype(float), (eval_cols["discharge_disposition_id"] == "6").astype(float))
    add_num("discharged_transfer", np.isin(train["discharge_disposition_id"], ["2", "4", "5", "22", "23", "24"]).astype(float),
            np.isin(eval_cols["discharge_disposition_id"], ["2", "4", "5", "22", "23", "24"]).astype(float))

    add_num("admitted_er", (train["admission_source_id"] == "7").astype(float), (eval_cols["admission_source_id"] == "7").astype(float))
    add_num("admission_emergency", (train["admission_type_id"] == "1").astype(float), (eval_cols["admission_type_id"] == "1").astype(float))
    add_num("admission_elective", (train["admission_type_id"] == "3").astype(float), (eval_cols["admission_type_id"] == "3").astype(float))

    add_num("a1c_tested", (train["A1Cresult"] != "None").astype(float), (eval_cols["A1Cresult"] != "None").astype(float))
    add_num("a1c_high", np.isin(train["A1Cresult"], [">7", ">8"]).astype(float), np.isin(eval_cols["A1Cresult"], [">7", ">8"]).astype(float))
    add_num("glu_tested", (train["max_glu_serum"] != "None").astype(float), (eval_cols["max_glu_serum"] != "None").astype(float))
    add_num("glu_high", np.isin(train["max_glu_serum"], [">200", ">300"]).astype(float), np.isin(eval_cols["max_glu_serum"], [">200", ">300"]).astype(float))

    for col in ["payer_code", "medical_specialty"]:
        trn_m = ((train[col] == None) | (train[col] == "?")).astype(float)
        eval_m = ((eval_cols[col] == None) | (eval_cols[col] == "?")).astype(float)
        add_num(f"{col}_missing", trn_m, eval_m)

    # 7. Fit & Transform via TabularEncoder
    encoder = TabularEncoder(min_count=15)
    encoder.fit(
        task,
        train_copy,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_copy)
    x_eval = encoder.transform(eval_copy)

    # 8. Cross-Validated Bagging Ensemble with Complementary Architectures
    n_eval = len(los_eval)
    eval_preds = np.zeros(n_eval, dtype=np.float64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cat_idx = encoder.categorical_indices

    for fold, (trn_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        xt, yt = x_train[trn_idx], y_train[trn_idx]

        # Model 1: Regularized, shallow depth-constrained tree
        m1 = lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.035,
            num_leaves=18,
            max_depth=5,
            min_child_samples=60,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=5.0,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m1.fit(xt, yt, categorical_feature=cat_idx)
        p1 = m1.predict_proba(x_eval)[:, 1]

        # Model 2: Moderately deeper tree capturing feature interactions
        m2 = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.04,
            num_leaves=28,
            max_depth=7,
            min_child_samples=45,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=3.0,
            random_state=100 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m2.fit(xt, yt, categorical_feature=cat_idx)
        p2 = m2.predict_proba(x_eval)[:, 1]

        eval_preds += (0.5 * p1 + 0.5 * p2) / 5.0

    return np.clip(eval_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
