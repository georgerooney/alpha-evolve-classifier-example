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


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    import numpy as np
    import lightgbm as lgb
    import xgboost as xgb
    
    # 1. Feature Engineering
    extra_numeric = []
    extra_categorical = []
    
    # Comorbidities
    diag_cols_present = [col for col in ["diag_1", "diag_2", "diag_3"] if col in train]
    if diag_cols_present:
        comorbs_train = icd9_comorbidities(*(train[col] for col in diag_cols_present))
        comorbs_eval = icd9_comorbidities(*(eval_cols[col] for col in diag_cols_present))
        for k in comorbs_train.keys():
            train[k] = comorbs_train[k]
            eval_cols[k] = comorbs_eval[k]
            extra_numeric.append(k)
            
    # Grouped ICD9 chapters
    def safe_icd9_group(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "missing"
        return icd9_group(str(x))
        
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in train:
            grp_train = np.array([safe_icd9_group(x) for x in train[col]], dtype=object)
            grp_eval = np.array([safe_icd9_group(x) for x in eval_cols[col]], dtype=object)
            
            grp_name = f"{col}_group"
            train[grp_name] = grp_train
            eval_cols[grp_name] = grp_eval
            extra_categorical.append(grp_name)
            
    # Medication Features
    med_cols = [
        col for col in task.categorical_columns 
        if col in train and any(val in ["Steady", "Up", "Down"] for val in np.unique(train[col].astype(str)))
    ]
    if med_cols:
        for d in [train, eval_cols]:
            active = np.zeros(len(d["gender"]), dtype=np.float64)
            up = np.zeros(len(d["gender"]), dtype=np.float64)
            down = np.zeros(len(d["gender"]), dtype=np.float64)
            steady = np.zeros(len(d["gender"]), dtype=np.float64)
            for col in med_cols:
                val = d[col]
                active += (val != "No") & (val != None)
                up += (val == "Up")
                down += (val == "Down")
                steady += (val == "Steady")
            d["num_active_meds"] = active
            d["num_up_meds"] = up
            d["num_down_meds"] = down
            d["num_steady_meds"] = steady
        extra_numeric.extend(["num_active_meds", "num_up_meds", "num_down_meds", "num_steady_meds"])
        
    # Interaction & Utilization Features
    num_cols = ["number_outpatient", "number_emergency", "number_inpatient", "num_lab_procedures", "num_procedures", "num_medications", "time_in_hospital"]
    if all(col in train for col in num_cols):
        for d in [train, eval_cols]:
            d["total_visits"] = d["number_outpatient"] + d["number_emergency"] + d["number_inpatient"]
            d["has_inpatient"] = (d["number_inpatient"] > 0).astype(float)
            d["has_emergency"] = (d["number_emergency"] > 0).astype(float)
            d["has_outpatient"] = (d["number_outpatient"] > 0).astype(float)
            d["severity_idx"] = d["num_lab_procedures"] + d["num_procedures"] + d["num_medications"]
            d["log_num_lab_procedures"] = np.log1p(d["num_lab_procedures"])
            d["log_num_medications"] = np.log1p(d["num_medications"])
            d["log_total_visits"] = np.log1p(d["total_visits"])
            
            time_h = d["time_in_hospital"] + 1e-5
            d["procedures_per_day"] = d["num_procedures"] / time_h
            d["meds_per_day"] = d["num_medications"] / time_h
            d["labs_per_day"] = d["num_lab_procedures"] / time_h

        extra_numeric.extend([
            "total_visits", "has_inpatient", "has_emergency", "has_outpatient", 
            "severity_idx", "log_num_lab_procedures", "log_num_medications", 
            "log_total_visits", "procedures_per_day", "meds_per_day", "labs_per_day"
        ])
        
    # Target Encoding of High Cardinality columns
    te_cols = [
        "diag_1", "diag_2", "diag_3", 
        "medical_specialty", "discharge_disposition_id", 
        "admission_source_id", "admission_type_id", 
        "payer_code", "age"
    ]
    for col in te_cols:
        if col in train:
            train_enc, eval_enc = oof_target_encode(
                train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42
            )
            te_name = f"{col}_te"
            train[te_name] = train_enc
            eval_cols[te_name] = eval_enc
            extra_numeric.append(te_name)
            
    # 2. Encode all columns with TabularEncoder
    encoder = TabularEncoder(min_count=50)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_eval = encoder.transform(eval_cols)
    
    # 3. Fit Model Ensemble
    model1 = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.015,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=0.1,
        reg_alpha=0.1,
        cat_smooth=50.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    
    model2 = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.01,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        reg_alpha=0.5,
        cat_smooth=100.0,
        random_state=100,
        n_jobs=1,
        verbose=-1,
    )
    
    model3 = lgb.LGBMClassifier(
        n_estimators=120,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=250,
        subsample=0.6,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=0.01,
        cat_smooth=20.0,
        random_state=2023,
        n_jobs=1,
        verbose=-1,
    )
    
    model4 = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.5,
        reg_alpha=0.2,
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    
    # Fit models
    cat_feats = encoder.categorical_indices
    model1.fit(x_train, y_train, categorical_feature=cat_feats)
    model2.fit(x_train, y_train, categorical_feature=cat_feats)
    model3.fit(x_train, y_train, categorical_feature=cat_feats)
    model4.fit(x_train, y_train)
    
    # Predict and average
    p1 = model1.predict_proba(x_eval)[:, 1]
    p2 = model2.predict_proba(x_eval)[:, 1]
    p3 = model3.predict_proba(x_eval)[:, 1]
    p4 = model4.predict_proba(x_eval)[:, 1]
    
    return 0.35 * p1 + 0.25 * p2 + 0.20 * p3 + 0.20 * p4


# EVOLVE-BLOCK-END
# =====================================================================
