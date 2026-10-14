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

Strategy only: a single LightGBM on the raw columns with hyper-parameters already tuned by random search on the
fitness rounds (`scripts/tabular_report.py`), so re-tuning alone cannot beat it. Encoding mechanics live in
`clinical_toolkit`.
"""

from __future__ import annotations

from typing import Dict

import lightgbm as lgb
import numpy as np

from clinical_models import TabularTask
from clinical_toolkit import TabularEncoder, icd9_comorbidities, icd9_group, oof_target_encode  # noqa: F401


# =====================================================================
# EVOLVE-BLOCK-START
# Strategy only. Pre-imported in the sandbox: numpy, scipy, sklearn, lightgbm, xgboost. No file/network access.


import numpy as np
import lightgbm as lgb


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return readmission probabilities for `eval_cols`."""
    train_data = dict(train)
    eval_data = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    n_tr = len(y_train)
    n_ev = len(eval_data["time_in_hospital"])

    # 1. Comorbidities across ICD-9 diagnoses
    cm_train = icd9_comorbidities(
        train_data["diag_1"], train_data["diag_2"], train_data["diag_3"]
    )
    cm_eval = icd9_comorbidities(
        eval_data["diag_1"], eval_data["diag_2"], eval_data["diag_3"]
    )
    for k, arr in cm_train.items():
        train_data[k] = arr
        eval_data[k] = cm_eval[k]
        extra_numeric.append(k)

    # Weighted Charlson Comorbidity Index
    charlson_weights = {
        "cm_mi": 1.0,
        "cm_chf": 1.0,
        "cm_pvd": 1.0,
        "cm_cerebrovascular": 1.0,
        "cm_dementia": 1.0,
        "cm_pulmonary": 1.0,
        "cm_rheumatic": 1.0,
        "cm_peptic_ulcer": 1.0,
        "cm_liver": 1.0,
        "cm_diabetes_complicated": 2.0,
        "cm_hemiplegia": 2.0,
        "cm_renal": 2.0,
        "cm_cancer": 2.0,
        "cm_metastatic": 6.0,
        "cm_hiv": 6.0,
    }
    cci_tr = np.zeros(n_tr, dtype=np.float64)
    cci_ev = np.zeros(n_ev, dtype=np.float64)
    for cond, wt in charlson_weights.items():
        if cond in cm_train:
            cci_tr += wt * cm_train[cond]
            cci_ev += wt * cm_eval[cond]
    train_data["charlson_index"] = cci_tr
    eval_data["charlson_index"] = cci_ev
    extra_numeric.append("charlson_index")

    # 2. ICD-9 Chapter groups
    for d_col in ["diag_1", "diag_2", "diag_3"]:
        grp_tr = np.array([icd9_group(c) for c in train_data[d_col]], dtype=object)
        grp_ev = np.array([icd9_group(c) for c in eval_data[d_col]], dtype=object)
        grp_name = f"{d_col}_group"
        train_data[grp_name] = grp_tr
        eval_data[grp_name] = grp_ev
        extra_categorical.append(grp_name)

    # High-risk diagnosis group indicators across any diag position
    for grp in ["circulatory", "respiratory", "diabetes"]:
        fn = f"diag_{grp}_any"
        train_data[fn] = (
            (train_data["diag_1_group"] == grp)
            | (train_data["diag_2_group"] == grp)
            | (train_data["diag_3_group"] == grp)
        ).astype(np.float64)
        eval_data[fn] = (
            (eval_data["diag_1_group"] == grp)
            | (eval_data["diag_2_group"] == grp)
            | (eval_data["diag_3_group"] == grp)
        ).astype(np.float64)
        extra_numeric.append(fn)

    # 3. Target encodings for high-cardinality categorical fields
    te_target_cols = [
        "diag_1",
        "diag_2",
        "diag_3",
        "discharge_disposition_id",
        "admission_source_id",
        "medical_specialty",
        "payer_code",
    ]
    for col in te_target_cols:
        if col in train_data and col in eval_data:
            te_tr, te_ev = oof_target_encode(
                train_data[col],
                y_train,
                eval_data[col],
                n_folds=5,
                smoothing=20.0,
                seed=42,
            )
            fn = f"{col}_te"
            train_data[fn] = te_tr
            eval_data[fn] = te_ev
            extra_numeric.append(fn)

    # 4. Clean domain, utilization, and medication feature engineering
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
    all_med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]
    present_med_cols = [m for m in all_med_cols if m in train_data]

    clinical_num_cols = [
        "total_prior_visits", "inpatient_ratio", "labs_per_day", "meds_per_day",
        "procs_per_day", "age_num", "num_active_diabetes_meds", "num_meds_changed",
    ]
    extra_numeric.extend(clinical_num_cols)

    for d, n_rows in [(train_data, n_tr), (eval_data, n_ev)]:
        inp = d["number_inpatient"].astype(np.float64)
        emg = d["number_emergency"].astype(np.float64)
        out = d["number_outpatient"].astype(np.float64)
        time_hosp = d["time_in_hospital"].astype(np.float64)
        n_meds = d["num_medications"].astype(np.float64)
        n_labs = d["num_lab_procedures"].astype(np.float64)
        n_procs = d["num_procedures"].astype(np.float64)

        tot_prior = inp + emg + out
        d["total_prior_visits"] = tot_prior
        d["inpatient_ratio"] = inp / (tot_prior + 1.0)

        safe_hosp = np.maximum(time_hosp, 1.0)
        d["labs_per_day"] = n_labs / safe_hosp
        d["meds_per_day"] = n_meds / safe_hosp
        d["procs_per_day"] = n_procs / safe_hosp

        age_col = d["age"]
        d["age_num"] = np.array([age_map.get(str(a), np.nan) for a in age_col], dtype=np.float64)

        active_cnt = np.zeros(n_rows, dtype=np.float64)
        changed_cnt = np.zeros(n_rows, dtype=np.float64)
        for m in present_med_cols:
            m_val = d[m]
            active_cnt += (m_val != "No").astype(np.float64)
            changed_cnt += ((m_val == "Up") | (m_val == "Down")).astype(np.float64)

        d["num_active_diabetes_meds"] = active_cnt
        d["num_meds_changed"] = changed_cnt

    # 5. Encoding
    encoder = TabularEncoder(min_count=25)
    encoder.fit(
        task,
        train_data,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)
    cat_idx = encoder.categorical_indices

    # 6. Diverse, native-categorical LightGBM ensemble
    # Model 1: Balanced leaf-wise
    lgb1 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=1.0,
        cat_smooth=40.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    lgb1.fit(x_train, y_train, categorical_feature=cat_idx)
    p_lgb1 = lgb1.predict_proba(x_eval)[:, 1]

    # Model 2: Depth-constrained trees
    lgb2 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.02,
        max_depth=5,
        num_leaves=24,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=2.0,
        cat_smooth=50.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    lgb2.fit(x_train, y_train, categorical_feature=cat_idx)
    p_lgb2 = lgb2.predict_proba(x_eval)[:, 1]

    # Model 3: Regularized leaf-wise
    lgb3 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.02,
        num_leaves=40,
        min_child_samples=120,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=5.0,
        cat_smooth=30.0,
        random_state=999,
        n_jobs=1,
        verbose=-1,
    )
    lgb3.fit(x_train, y_train, categorical_feature=cat_idx)
    p_lgb3 = lgb3.predict_proba(x_eval)[:, 1]

    # Equal-weight average blend
    return (p_lgb1 + p_lgb2 + p_lgb3) / 3.0


# EVOLVE-BLOCK-END
# =====================================================================
