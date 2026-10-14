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


from typing import Any, Dict, List, Sequence, Tuple
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


def _parse_icd9_prefix(val: Any) -> Tuple[bool, float]:
    """Returns (is_valid_num, num_val) for range checks."""
    if val is None:
        return False, -1.0
    s = str(val).strip()
    if not s or s == "?" or s.startswith(("V", "v", "E", "e")):
        return False, -1.0
    try:
        return True, float(s.split(".")[0])
    except ValueError:
        return False, -1.0


def _compute_target_encoding(
    train_vals: np.ndarray,
    eval_vals: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    m: float = 15.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes out-of-fold target encoding for train and full-dataset target encoding for eval."""
    global_mean = float(np.mean(y))
    n_train = len(y)
    oof_train = np.full(n_train, global_mean, dtype=np.float64)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(train_vals, y):
        cat_tr = train_vals[tr_idx]
        y_tr = y[tr_idx]
        prior = float(np.mean(y_tr))

        counts: Dict[Any, int] = {}
        sums: Dict[Any, float] = {}
        for c, label in zip(cat_tr, y_tr):
            if c is not None and c != "" and c != "?":
                counts[c] = counts.get(c, 0) + 1
                sums[c] = sums.get(c, 0.0) + label

        val_enc = np.empty(len(val_idx), dtype=np.float64)
        for i, c in enumerate(train_vals[val_idx]):
            if c is not None and c != "" and c != "?" and c in counts:
                val_enc[i] = (sums[c] + m * prior) / (counts[c] + m)
            else:
                val_enc[i] = prior
        oof_train[val_idx] = val_enc

    # Full encoding for eval
    counts_full: Dict[Any, int] = {}
    sums_full: Dict[Any, float] = {}
    for c, label in zip(train_vals, y):
        if c is not None and c != "" and c != "?":
            counts_full[c] = counts_full.get(c, 0) + 1
            sums_full[c] = sums_full.get(c, 0.0) + label

    eval_enc = np.empty(len(eval_vals), dtype=np.float64)
    for i, c in enumerate(eval_vals):
        if c is not None and c != "" and c != "?" and c in counts_full:
            eval_enc[i] = (sums_full[c] + m * global_mean) / (counts_full[c] + m)
        else:
            eval_enc[i] = global_mean

    return oof_train, eval_enc


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

    # 2. Comorbidity flags across diag 1..3
    for prefix, data in (("train", train_data), ("eval", eval_data)):
        n_rows = len(data["diag_1"])
        has_diabetes = np.zeros(n_rows, dtype=np.float64)
        has_chf = np.zeros(n_rows, dtype=np.float64)
        has_circ = np.zeros(n_rows, dtype=np.float64)
        has_resp = np.zeros(n_rows, dtype=np.float64)
        has_renal = np.zeros(n_rows, dtype=np.float64)
        has_neoplasm = np.zeros(n_rows, dtype=np.float64)

        d1 = data["diag_1"]
        d2 = data["diag_2"]
        d3 = data["diag_3"]

        for i in range(n_rows):
            for code in (d1[i], d2[i], d3[i]):
                if code is None:
                    continue
                s = str(code).strip()
                if s.startswith("250"):
                    has_diabetes[i] = 1.0
                if s.startswith("428"):
                    has_chf[i] = 1.0
                is_num, num = _parse_icd9_prefix(code)
                if is_num:
                    if 390 <= num <= 459 or num == 785:
                        has_circ[i] = 1.0
                    elif 460 <= num <= 519 or num == 786:
                        has_resp[i] = 1.0
                    elif 580 <= num <= 589:
                        has_renal[i] = 1.0
                    elif 140 <= num <= 239:
                        has_neoplasm[i] = 1.0

        comorbidity_score = has_diabetes + has_chf + has_circ + has_resp + has_renal + has_neoplasm

        data["has_diabetes_diag"] = has_diabetes
        data["has_chf_diag"] = has_chf
        data["has_circ_diag"] = has_circ
        data["has_resp_diag"] = has_resp
        data["has_renal_diag"] = has_renal
        data["has_neoplasm_diag"] = has_neoplasm
        data["comorbidity_score"] = comorbidity_score

    extra_numeric.extend([
        "has_diabetes_diag",
        "has_chf_diag",
        "has_circ_diag",
        "has_resp_diag",
        "has_renal_diag",
        "has_neoplasm_diag",
        "comorbidity_score",
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
        data["high_prior_inpatient"] = (inpat >= 2.0).astype(np.float64)
        data["inpatient_ratio"] = (inpat + 0.1) / (total_visits + 1.0)
        data["emergency_ratio"] = (emerg + 0.1) / (total_visits + 1.0)
        data["inpatient_per_day"] = inpat / hosp_days

        data["med_per_day"] = num_meds / hosp_days
        data["labs_per_day"] = num_labs / hosp_days
        data["procs_per_day"] = num_procs / hosp_days
        data["total_procedures"] = num_labs + num_procs
        data["clinical_activity"] = num_meds + num_labs + num_procs
        data["meds_per_diag"] = num_meds / num_diags
        data["labs_per_proc"] = (num_labs + 1.0) / (num_procs + 1.0)

        # Age parsing
        age_num = np.array([_parse_age(a) for a in data["age"]], dtype=np.float64)
        data["age_num"] = age_num
        data["is_elderly"] = (age_num >= 70.0).astype(np.float64)

        # Discharge & admission indicators
        disp = data["discharge_disposition_id"]
        data["discharged_home"] = (disp == "1").astype(np.float64)
        data["discharged_facility"] = np.isin(disp, ["3", "4", "5", "6", "22", "23", "24"]).astype(np.float64)
        data["discharged_snf"] = (disp == "3").astype(np.float64)
        data["discharged_home_health"] = (disp == "6").astype(np.float64)
        adm_type = data["admission_type_id"]
        data["admission_emergency"] = (adm_type == "1").astype(np.float64)
        data["admission_elective"] = (adm_type == "3").astype(np.float64)

        # Lab indicators
        a1c = data["A1Cresult"]
        data["a1c_tested"] = np.isin(a1c, ["Norm", ">7", ">8"]).astype(np.float64)
        data["a1c_high"] = np.isin(a1c, [">7", ">8"]).astype(np.float64)
        glu = data["max_glu_serum"]
        data["glu_tested"] = np.isin(glu, ["Norm", ">200", ">300"]).astype(np.float64)
        data["glu_high"] = np.isin(glu, [">200", ">300"]).astype(np.float64)

        # Medication summary
        med_cols = [
            "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
            "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
            "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
            "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
        ]
        n_rows = len(disp)
        active_cnt = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        down_cnt = np.zeros(n_rows, dtype=np.float64)
        for m_col in med_cols:
            if m_col in data:
                vals = data[m_col]
                active_cnt += np.isin(vals, ["Steady", "Up", "Down"]).astype(np.float64)
                up_cnt += (vals == "Up").astype(np.float64)
                down_cnt += (vals == "Down").astype(np.float64)

        data["num_active_meds"] = active_cnt
        data["num_meds_up"] = up_cnt
        data["num_meds_down"] = down_cnt
        data["med_changes_sum"] = up_cnt + down_cnt
        data["change_flag"] = (data["change"] == "Ch").astype(np.float64)
        data["diabetes_med_flag"] = (data["diabetesMed"] == "Yes").astype(np.float64)
        data["insulin_is_up"] = (data["insulin"] == "Up").astype(np.float64)
        data["insulin_is_down"] = (data["insulin"] == "Down").astype(np.float64)
        data["insulin_active"] = (data["insulin"] != "No").astype(np.float64)

    extra_numeric.extend([
        "total_prior_visits",
        "has_prior_inpatient",
        "high_prior_inpatient",
        "inpatient_ratio",
        "emergency_ratio",
        "inpatient_per_day",
        "med_per_day",
        "labs_per_day",
        "procs_per_day",
        "total_procedures",
        "clinical_activity",
        "meds_per_diag",
        "labs_per_proc",
        "age_num",
        "is_elderly",
        "discharged_home",
        "discharged_facility",
        "discharged_snf",
        "discharged_home_health",
        "admission_emergency",
        "admission_elective",
        "a1c_tested",
        "a1c_high",
        "glu_tested",
        "glu_high",
        "num_active_meds",
        "num_meds_up",
        "num_meds_down",
        "med_changes_sum",
        "change_flag",
        "diabetes_med_flag",
        "insulin_is_up",
        "insulin_is_down",
        "insulin_active",
    ])

    # 4. Out-of-fold target encoding for high-cardinality features
    te_target_cols = [
        "discharge_disposition_id",
        "admission_source_id",
        "admission_type_id",
        "medical_specialty",
        "diag_1_group",
        "diag_1",
    ]
    for col in te_target_cols:
        col_te_name = f"{col}_te"
        oof_tr, te_ev = _compute_target_encoding(
            train_data[col],
            eval_data[col],
            y_train,
            n_splits=5,
            m=15.0,
            seed=42,
        )
        train_data[col_te_name] = oof_tr
        eval_data[col_te_name] = te_ev
        extra_numeric.append(col_te_name)

    # 5. Fit TabularEncoder on all task + derived features
    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(
        task,
        train_data,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_eval = encoder.transform(eval_data)

    # 6. Diverse ensemble of LightGBM models
    models = [
        # Model 1: Balanced tree depth and capacity
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 2: Shallower, regularized tree for noisy tabular EHR robustness
        lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.025,
            num_leaves=18,
            max_depth=5,
            min_child_samples=60,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=5.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 3: Interaction-rich tree with diverse feature fraction
        lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.035,
            num_leaves=42,
            max_depth=7,
            min_child_samples=35,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.05,
            reg_lambda=1.5,
            random_state=456,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 4: Intermediate depth with alternative seed and regularization
        lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=24,
            max_depth=5,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.2,
            reg_lambda=3.0,
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
