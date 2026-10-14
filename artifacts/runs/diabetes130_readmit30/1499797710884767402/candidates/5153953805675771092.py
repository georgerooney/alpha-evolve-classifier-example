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

        data["has_diabetes_diag"] = has_diab
        data["has_chf_diag"] = has_chf
        data["has_circ_diag"] = has_circ
        data["has_resp_diag"] = has_resp
        data["has_renal_diag"] = has_renal
        data["has_neoplasm_diag"] = has_neoplasm
        data["comorbidity_score"] = has_diab + has_chf + has_circ + has_resp + has_renal + has_neoplasm

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
        data["inpatient_ratio"] = (inpat + 0.1) / (total_visits + 1.0)
        data["emergency_ratio"] = (emerg + 0.1) / (total_visits + 1.0)

        data["med_per_day"] = num_meds / hosp_days
        data["labs_per_day"] = num_labs / hosp_days
        data["procs_per_day"] = num_procs / hosp_days
        data["meds_per_diag"] = num_meds / num_diags

        # Age parsing
        data["age_num"] = np.array([_parse_age(a) for a in data["age"]], dtype=np.float64)

        # Discharge & admission indicators
        disp = data["discharge_disposition_id"]
        data["discharged_home"] = (disp == "1").astype(np.float64)
        data["discharged_facility"] = np.isin(disp, ["3", "4", "5", "6", "22", "23", "24"]).astype(np.float64)
        data["admission_emergency"] = (data["admission_type_id"] == "1").astype(np.float64)

        # Lab indicators
        data["a1c_tested"] = np.isin(data["A1Cresult"], ["Norm", ">7", ">8"]).astype(np.float64)
        data["a1c_high"] = np.isin(data["A1Cresult"], [">7", ">8"]).astype(np.float64)
        data["glu_high"] = np.isin(data["max_glu_serum"], [">200", ">300"]).astype(np.float64)

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
        for m_col in med_cols:
            if m_col in data:
                vals = data[m_col]
                active_cnt += np.isin(vals, ["Steady", "Up", "Down"]).astype(np.float64)
                changes_cnt += np.isin(vals, ["Up", "Down"]).astype(np.float64)

        data["num_active_meds"] = active_cnt
        data["med_changes_sum"] = changes_cnt
        data["change_flag"] = (data["change"] == "Ch").astype(np.float64)
        data["diabetes_med_flag"] = (data["diabetesMed"] == "Yes").astype(np.float64)
        data["insulin_active"] = (data["insulin"] != "No").astype(np.float64)
        data["insulin_is_up"] = (data["insulin"] == "Up").astype(np.float64)
        data["insulin_is_down"] = (data["insulin"] == "Down").astype(np.float64)

    extra_numeric.extend([
        "total_prior_visits",
        "has_prior_inpatient",
        "inpatient_ratio",
        "emergency_ratio",
        "med_per_day",
        "labs_per_day",
        "procs_per_day",
        "meds_per_diag",
        "age_num",
        "discharged_home",
        "discharged_facility",
        "admission_emergency",
        "a1c_tested",
        "a1c_high",
        "glu_high",
        "num_active_meds",
        "med_changes_sum",
        "change_flag",
        "diabetes_med_flag",
        "insulin_active",
        "insulin_is_up",
        "insulin_is_down",
    ])

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
        # Model 1: Balanced low-rank tree with moderate depth
        lgb.LGBMClassifier(
            n_estimators=360,
            learning_rate=0.03,
            num_leaves=24,
            max_depth=5,
            min_child_samples=45,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.2,
            reg_lambda=3.0,
            min_data_per_group=20,
            cat_smooth=12.0,
            cat_l2=5.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 2: Conservative shallow tree for noise reduction
        lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.025,
            num_leaves=18,
            max_depth=5,
            min_child_samples=55,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=5.0,
            min_data_per_group=20,
            cat_smooth=12.0,
            cat_l2=5.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 3: Higher subspace coverage with depth 6
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.035,
            num_leaves=28,
            max_depth=6,
            min_child_samples=40,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=2.0,
            min_data_per_group=20,
            cat_smooth=12.0,
            cat_l2=5.0,
            random_state=456,
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
