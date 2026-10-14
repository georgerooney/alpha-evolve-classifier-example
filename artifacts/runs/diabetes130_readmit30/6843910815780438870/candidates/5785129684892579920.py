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
    
    n_train = len(y_train)
    n_eval = len(next(iter(eval_cols.values())))
    
    # Comorbidities & Charlson Comorbidity Index
    diag_cols_present = [col for col in ["diag_1", "diag_2", "diag_3"] if col in train]
    if diag_cols_present:
        comorbs_train = icd9_comorbidities(*(train[col] for col in diag_cols_present))
        comorbs_eval = icd9_comorbidities(*(eval_cols[col] for col in diag_cols_present))
        for k in comorbs_train.keys():
            train[k] = comorbs_train[k]
            eval_cols[k] = comorbs_eval[k]
            extra_numeric.append(k)
            
        cci_weights = {
            "cm_mi": 1.0, "cm_chf": 1.0, "cm_pvd": 1.0, "cm_cerebrovascular": 1.0,
            "cm_dementia": 1.0, "cm_pulmonary": 1.0, "cm_rheumatic": 1.0, "cm_peptic_ulcer": 1.0,
            "cm_liver": 1.0, "cm_hemiplegia": 2.0, "cm_renal": 2.0, "cm_diabetes_complicated": 2.0,
            "cm_cancer": 2.0, "cm_metastatic": 6.0, "cm_hiv": 6.0
        }
        cci_train = np.zeros(n_train)
        cci_eval = np.zeros(n_eval)
        for k, w in cci_weights.items():
            if k in train:
                cci_train += train[k] * w
                cci_eval += eval_cols[k] * w
        train["charlson_index"] = cci_train
        eval_cols["charlson_index"] = cci_eval
        extra_numeric.append("charlson_index")
            
    # Grouped ICD9 chapters
    def safe_icd9_group(x):
        if x is None or x == "" or x == "?" or (isinstance(x, float) and np.isnan(x)):
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
        active_train = np.zeros(n_train, dtype=np.float64)
        up_train = np.zeros(n_train, dtype=np.float64)
        down_train = np.zeros(n_train, dtype=np.float64)
        steady_train = np.zeros(n_train, dtype=np.float64)
        
        active_eval = np.zeros(n_eval, dtype=np.float64)
        up_eval = np.zeros(n_eval, dtype=np.float64)
        down_eval = np.zeros(n_eval, dtype=np.float64)
        steady_eval = np.zeros(n_eval, dtype=np.float64)
        
        for col in med_cols:
            vt = train[col]
            ve = eval_cols[col]
            active_train += (vt != "No") & (vt != None)
            up_train += (vt == "Up")
            down_train += (vt == "Down")
            steady_train += (vt == "Steady")
            
            active_eval += (ve != "No") & (ve != None)
            up_eval += (ve == "Up")
            down_eval += (ve == "Down")
            steady_eval += (ve == "Steady")
            
        train["num_active_meds"] = active_train
        train["num_up_meds"] = up_train
        train["num_down_meds"] = down_train
        train["num_steady_meds"] = steady_train
        
        eval_cols["num_active_meds"] = active_eval
        eval_cols["num_up_meds"] = up_eval
        eval_cols["num_down_meds"] = down_eval
        eval_cols["num_steady_meds"] = steady_eval
        
        extra_numeric.extend(["num_active_meds", "num_up_meds", "num_down_meds", "num_steady_meds"])

        # Low-Rank Medication Matrix Factorization (PCA)
        from sklearn.decomposition import PCA
        
        # 1. Intensity Matrix: mapping No->0, Down->1, Steady->2, Up->3
        med_matrix_intensity_train = np.zeros((n_train, len(med_cols)))
        med_matrix_intensity_eval = np.zeros((n_eval, len(med_cols)))
        mapping_intensity = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}
        for i, col in enumerate(med_cols):
            val_train = train[col]
            val_eval = eval_cols[col]
            med_matrix_intensity_train[:, i] = np.array([mapping_intensity.get(x, 0) for x in val_train])
            med_matrix_intensity_eval[:, i] = np.array([mapping_intensity.get(x, 0) for x in val_eval])

        pca_int = PCA(n_components=3, random_state=42)
        pca_int_train = pca_int.fit_transform(med_matrix_intensity_train)
        pca_int_eval = pca_int.transform(med_matrix_intensity_eval)

        for j in range(3):
            col_name = f"med_int_pca_{j}"
            train[col_name] = pca_int_train[:, j]
            eval_cols[col_name] = pca_int_eval[:, j]
            extra_numeric.append(col_name)

        # 2. Change Matrix: mapping No->0, Steady->0, Down->-1, Up->1
        med_matrix_change_train = np.zeros((n_train, len(med_cols)))
        med_matrix_change_eval = np.zeros((n_eval, len(med_cols)))
        mapping_change = {"No": 0, "Steady": 0, "Down": -1, "Up": 1}
        for i, col in enumerate(med_cols):
            val_train = train[col]
            val_eval = eval_cols[col]
            med_matrix_change_train[:, i] = np.array([mapping_change.get(x, 0) for x in val_train])
            med_matrix_change_eval[:, i] = np.array([mapping_change.get(x, 0) for x in val_eval])

        pca_chg = PCA(n_components=2, random_state=43)
        pca_chg_train = pca_chg.fit_transform(med_matrix_change_train)
        pca_chg_eval = pca_chg.transform(med_matrix_change_eval)

        for j in range(2):
            col_name = f"med_chg_pca_{j}"
            train[col_name] = pca_chg_train[:, j]
            eval_cols[col_name] = pca_chg_eval[:, j]
            extra_numeric.append(col_name)

    # Demographics and Continuous Age Parsing
    def parse_age(age_str):
        if age_str is None or not isinstance(age_str, str):
            return np.nan
        try:
            cleaned = age_str.replace("[", "").replace(")", "").replace("]", "")
            parts = cleaned.split("-")
            if len(parts) == 2:
                return (float(parts[0]) + float(parts[1])) / 2.0
        except Exception:
            pass
        return np.nan

    if "age" in train:
        train["age_numeric"] = np.array([parse_age(x) for x in train["age"]])
        eval_cols["age_numeric"] = np.array([parse_age(x) for x in eval_cols["age"]])
        extra_numeric.append("age_numeric")

    # Informative Missingness indicators
    if "weight" in train:
        train["has_weight"] = np.array([float(x is not None and x != "?") for x in train["weight"]])
        eval_cols["has_weight"] = np.array([float(x is not None and x != "?") for x in eval_cols["weight"]])
        extra_numeric.append("has_weight")
        
    for col in ["medical_specialty", "payer_code"]:
        if col in train:
            col_name = f"has_{col}"
            train[col_name] = np.array([float(x is not None and x != "?") for x in train[col]])
            eval_cols[col_name] = np.array([float(x is not None and x != "?") for x in eval_cols[col]])
            extra_numeric.append(col_name)

    # Key discharge destination indicators
    if "discharge_disposition_id" in train:
        for dest in ["1", "3", "6", "18"]:
            col_name = f"discharged_to_{dest}"
            train[col_name] = (train["discharge_disposition_id"] == dest).astype(float)
            eval_cols[col_name] = (eval_cols["discharge_disposition_id"] == dest).astype(float)
            extra_numeric.append(col_name)
        
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
            
            d["labs_per_med"] = d["num_lab_procedures"] / (d["num_medications"] + 1e-5)
            d["inpatient_ratio"] = d["number_inpatient"] / (d["total_visits"] + 1e-5)
            d["emergency_ratio"] = d["number_emergency"] / (d["total_visits"] + 1e-5)

        extra_numeric.extend([
            "total_visits", "has_inpatient", "has_emergency", "has_outpatient", 
            "severity_idx", "log_num_lab_procedures", "log_num_medications", 
            "log_total_visits", "procedures_per_day", "meds_per_day", "labs_per_day",
            "labs_per_med", "inpatient_ratio", "emergency_ratio"
        ])
        
    # Cartesian Interaction Pairs for High-Cardinality Encodings
    pairs = [
        ("discharge_disposition_id", "admission_source_id"),
        ("admission_type_id", "discharge_disposition_id"),
        ("discharge_disposition_id", "diag_1_group"),
        ("medical_specialty", "discharge_disposition_id"),
        ("age", "discharge_disposition_id"),
        ("admission_source_id", "medical_specialty")
    ]
    for col1, col2 in pairs:
        if col1 in train and col2 in train:
            comb_train = np.array([f"{str(x1)}||{str(x2)}" for x1, x2 in zip(train[col1], train[col2])], dtype=object)
            comb_eval = np.array([f"{str(x1)}||{str(x2)}" for x1, x2 in zip(eval_cols[col1], eval_cols[col2])], dtype=object)
            comb_name = f"{col1}_{col2}_comb"
            train[comb_name] = comb_train
            eval_cols[comb_name] = comb_eval
            extra_categorical.append(comb_name)
            
            train_enc, eval_enc = oof_target_encode(
                comb_train, y_train, comb_eval, n_folds=5, smoothing=30.0, seed=42
            )
            te_name = f"{comb_name}_te"
            train[te_name] = train_enc
            eval_cols[te_name] = eval_enc
            extra_numeric.append(te_name)
            
    # Target Encoding of standard High/Low Cardinality columns
    te_cols = [
        "diag_1", "diag_2", "diag_3", 
        "medical_specialty", "discharge_disposition_id", 
        "admission_source_id", "admission_type_id", 
        "payer_code", "age", "race", "gender",
        "max_glu_serum", "A1Cresult", "change", "diabetesMed"
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

    # 1.5. Latent Risk Factor Decomposition (PCA on all Target-Encoded columns)
    te_names_list = []
    for col1, col2 in pairs:
        if col1 in train and col2 in train:
            te_names_list.append(f"{col1}_{col2}_comb_te")
    for col in te_cols:
        if col in train:
            te_names_list.append(f"{col}_te")

    if te_names_list:
        prior_mean = float(y_train.mean())
        te_train_mat = np.column_stack([np.nan_to_num(train[name], nan=prior_mean) for name in te_names_list])
        te_eval_mat = np.column_stack([np.nan_to_num(eval_cols[name], nan=prior_mean) for name in te_names_list])
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        scaler = StandardScaler()
        te_train_scaled = scaler.fit_transform(te_train_mat)
        te_eval_scaled = scaler.transform(te_eval_mat)
        
        pca_te = PCA(n_components=3, random_state=123)
        pca_te_train = pca_te.fit_transform(te_train_scaled)
        pca_te_eval = pca_te.transform(te_eval_scaled)
        
        for j in range(3):
            col_name = f"risk_pca_{j}"
            train[col_name] = pca_te_train[:, j]
            eval_cols[col_name] = pca_te_eval[:, j]
            extra_numeric.append(col_name)

    # Latent Comorbidity Profile Decomposition
    if diag_cols_present:
        comorb_flags = [k for k in comorbs_train.keys() if k != "cm_count"]
        if comorb_flags:
            comorb_mat_train = np.column_stack([comorbs_train[k] for k in comorb_flags])
            comorb_mat_eval = np.column_stack([comorbs_eval[k] for k in comorb_flags])
            comorb_mat_train = np.nan_to_num(comorb_mat_train)
            comorb_mat_eval = np.nan_to_num(comorb_mat_eval)
            
            from sklearn.decomposition import PCA
            pca_cm = PCA(n_components=3, random_state=4242)
            pca_cm_train = pca_cm.fit_transform(comorb_mat_train)
            pca_cm_eval = pca_cm.transform(comorb_mat_eval)
            
            for j in range(3):
                col_name = f"comorb_pca_{j}"
                train[col_name] = pca_cm_train[:, j]
                eval_cols[col_name] = pca_cm_eval[:, j]
                extra_numeric.append(col_name)
            
    # 2. Encode all columns with TabularEncoder
    encoder = TabularEncoder(min_count=25)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_eval = encoder.transform(eval_cols)
    
    # 3. Fit Model Ensemble
    model1 = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.015,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=1.0,
        reg_alpha=0.2,
        cat_smooth=40.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    
    model2 = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.01,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=2.0,
        reg_alpha=1.0,
        cat_smooth=80.0,
        random_state=100,
        n_jobs=1,
        verbose=-1,
    )
    
    model3 = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=200,
        subsample=0.6,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=0.1,
        cat_smooth=15.0,
        random_state=2023,
        n_jobs=1,
        verbose=-1,
    )
    
    model4 = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_lambda=2.0,
        reg_alpha=0.5,
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )

    model5 = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.015,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        reg_alpha=1.0,
        tree_method="hist",
        random_state=2024,
        n_jobs=1,
        verbosity=0,
    )
    
    # Fit models
    cat_feats = encoder.categorical_indices
    model1.fit(x_train, y_train, categorical_feature=cat_feats)
    model2.fit(x_train, y_train, categorical_feature=cat_feats)
    model3.fit(x_train, y_train, categorical_feature=cat_feats)
    model4.fit(x_train, y_train)
    model5.fit(x_train, y_train)
    
    # Predict and average
    p1 = model1.predict_proba(x_eval)[:, 1]
    p2 = model2.predict_proba(x_eval)[:, 1]
    p3 = model3.predict_proba(x_eval)[:, 1]
    p4 = model4.predict_proba(x_eval)[:, 1]
    p5 = model5.predict_proba(x_eval)[:, 1]
    
    return 0.25 * p1 + 0.25 * p2 + 0.10 * p3 + 0.20 * p4 + 0.20 * p5


# EVOLVE-BLOCK-END
# =====================================================================
