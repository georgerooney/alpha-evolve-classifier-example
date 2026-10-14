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
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    
    # 1. Initialize lists for new features
    extra_numeric = []
    extra_categorical = []
    
    # Make shallow copies to avoid modifying caller's dict keys
    train = dict(train)
    eval_cols = dict(eval_cols)
    
    # 2. Extract comorbidities using icd9_comorbidities
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    train_diags = [train[col] for col in diag_cols if col in train]
    eval_diags = [eval_cols[col] for col in diag_cols if col in eval_cols]
    
    train_cm = icd9_comorbidities(*train_diags)
    eval_cm = icd9_comorbidities(*eval_diags)
    
    for k in train_cm:
        train[k] = train_cm[k]
        eval_cols[k] = eval_cm[k]
        extra_numeric.append(k)
        
    # 3. Grouped ICD-9 chapters
    for col in diag_cols:
        grouped_col = col + "_grp"
        train[grouped_col] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_cols[grouped_col] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(grouped_col)
        
    # 4. Target encoding of high-cardinality columns
    te_cols = ['diag_1', 'diag_2', 'diag_3', 'medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'payer_code']
    for col in te_cols:
        if col in train and col in eval_cols:
            tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42)
            te_col = col + "_te"
            train[te_col] = tr_enc
            eval_cols[te_col] = ev_enc
            extra_numeric.append(te_col)
            
    # 5. Utilisation intensity features
    train_total_visits = train['number_outpatient'] + train['number_emergency'] + train['number_inpatient']
    eval_total_visits = eval_cols['number_outpatient'] + eval_cols['number_emergency'] + eval_cols['number_inpatient']
    train['total_visits'] = train_total_visits
    eval_cols['total_visits'] = eval_total_visits
    extra_numeric.append('total_visits')
    
    train['inpatient_ratio'] = train['number_inpatient'] / (train_total_visits + 1.0)
    eval_cols['inpatient_ratio'] = eval_cols['number_inpatient'] / (eval_total_visits + 1.0)
    extra_numeric.append('inpatient_ratio')
    
    # 6. Medication active and change counts
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    train_active_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_active_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_change_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_change_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    
    for col in med_cols:
        if col in train:
            train_active_meds += (train[col] != "No") & (train[col] != None)
            eval_active_meds += (eval_cols[col] != "No") & (eval_cols[col] != None)
            train_change_meds += (train[col] == "Up") | (train[col] == "Down")
            eval_change_meds += (eval_cols[col] == "Up") | (eval_cols[col] == "Down")
            
    train['active_meds_count'] = train_active_meds
    eval_cols['active_meds_count'] = eval_active_meds
    extra_numeric.append('active_meds_count')
    
    train['change_meds_count'] = train_change_meds
    eval_cols['change_meds_count'] = eval_change_meds
    extra_numeric.append('change_meds_count')
    
    # 7. Insulin change interaction
    train['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(train['insulin'], train['change'])], dtype=object)
    eval_cols['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(eval_cols['insulin'], eval_cols['change'])], dtype=object)
    extra_categorical.append('insulin_change')
    
    # 8. Numeric age representation
    def age_to_numeric(age_str):
        if age_str is None:
            return np.nan
        try:
            cleaned = age_str.replace('[', '').replace(')', '').replace(']', '')
            parts = cleaned.split('-')
            return (float(parts[0]) + float(parts[1])) / 2.0
        except Exception:
            return np.nan

    train['age_num'] = np.array([age_to_numeric(x) for x in train['age']], dtype=np.float64)
    eval_cols['age_num'] = np.array([age_to_numeric(x) for x in eval_cols['age']], dtype=np.float64)
    extra_numeric.append('age_num')

    # 9. Hospitalization Velocity & Physiological Fragility Indices
    time_hosp_train = np.maximum(train['time_in_hospital'], 1.0)
    time_hosp_eval = np.maximum(eval_cols['time_in_hospital'], 1.0)
    
    train['lab_velocity'] = train['num_lab_procedures'] / time_hosp_train
    eval_cols['lab_velocity'] = eval_cols['num_lab_procedures'] / time_hosp_eval
    extra_numeric.append('lab_velocity')
    
    train['med_velocity'] = train['num_medications'] / time_hosp_train
    eval_cols['med_velocity'] = eval_cols['num_medications'] / time_hosp_eval
    extra_numeric.append('med_velocity')
    
    train['proc_velocity'] = train['num_procedures'] / time_hosp_train
    eval_cols['proc_velocity'] = eval_cols['num_procedures'] / time_hosp_eval
    extra_numeric.append('proc_velocity')
    
    train['diag_velocity'] = train['number_diagnoses'] / time_hosp_train
    eval_cols['diag_velocity'] = eval_cols['number_diagnoses'] / time_hosp_eval
    extra_numeric.append('diag_velocity')
    
    train['prior_acuity'] = 3.0 * train['number_inpatient'] + 1.5 * train['number_emergency'] + 0.5 * train['number_outpatient']
    eval_cols['prior_acuity'] = 3.0 * eval_cols['number_inpatient'] + 1.5 * eval_cols['number_emergency'] + 0.5 * eval_cols['number_outpatient']
    extra_numeric.append('prior_acuity')
    
    train['unstable_diabetic'] = ((train_change_meds > 0) & (train['number_inpatient'] > 0)).astype(np.float64)
    eval_cols['unstable_diabetic'] = ((eval_change_meds > 0) & (eval_cols['number_inpatient'] > 0)).astype(np.float64)
    extra_numeric.append('unstable_diabetic')
    
    train['cardiorenal'] = ((train_cm.get('cm_chf', 0) > 0) | (train_cm.get('cm_renal', 0) > 0)).astype(np.float64)
    eval_cols['cardiorenal'] = ((eval_cm.get('cm_chf', 0) > 0) | (eval_cm.get('cm_renal', 0) > 0)).astype(np.float64)
    extra_numeric.append('cardiorenal')

    # 10. High-Order Trajectory Interactions & Target Encoding
    train['disch_admit'] = np.array([f"{d}_{a}" for d, a in zip(train['discharge_disposition_id'], train['admission_source_id'])], dtype=object)
    eval_cols['disch_admit'] = np.array([f"{d}_{a}" for d, a in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])], dtype=object)
    extra_categorical.append('disch_admit')
    
    train['diag1_diag2'] = np.array([f"{d1}_{d2}" for d1, d2 in zip(train['diag_1_grp'], train['diag_2_grp'])], dtype=object)
    eval_cols['diag1_diag2'] = np.array([f"{d1}_{d2}" for d1, d2 in zip(eval_cols['diag_1_grp'], eval_cols['diag_2_grp'])], dtype=object)
    extra_categorical.append('diag1_diag2')
    
    for interaction_col in ['disch_admit', 'diag1_diag2']:
        tr_enc, ev_enc = oof_target_encode(train[interaction_col], y_train, eval_cols[interaction_col], n_folds=5, smoothing=20.0, seed=42)
        te_name = interaction_col + "_te"
        train[te_name] = tr_enc
        eval_cols[te_name] = ev_enc
        extra_numeric.append(te_name)

    # 11. Bipartite Clinical Phenotype Manifold (LSA / TruncatedSVD)
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD
    import xgboost as xgb
    
    n_train_rows = len(y_train)
    n_eval_rows = len(eval_cols['insulin'])
    n_total = n_train_rows + n_eval_rows
    
    token_map = {}
    rows_list = []
    cols_list = []
    
    def add_token(row_idx, token_str):
        if token_str not in token_map:
            token_map[token_str] = len(token_map)
        rows_list.append(row_idx)
        cols_list.append(token_map[token_str])
        
    for i in range(n_total):
        is_tr = i < n_train_rows
        idx = i if is_tr else i - n_train_rows
        src = train if is_tr else eval_cols
        
        for d_c in diag_cols:
            val = src[d_c][idx]
            if val is not None and isinstance(val, str) and len(val) > 0:
                prefix = val.split('.')[0]
                add_token(i, f"dx3_{prefix}")
                add_token(i, f"grp_{icd9_group(val)}")
                
        for cm_k in train_cm:
            if src[cm_k][idx] > 0:
                add_token(i, f"cm_{cm_k}")
                
        for m_c in med_cols:
            m_v = src[m_c][idx]
            if m_v is not None and m_v != "No":
                add_token(i, f"med_{m_c}_{m_v}")
                
        d_val = src['discharge_disposition_id'][idx]
        if d_val is not None:
            add_token(i, f"disch_{d_val}")
            
        a_val = src['admission_source_id'][idx]
        if a_val is not None:
            add_token(i, f"asrc_{a_val}")
            
    sparse_data = np.ones(len(rows_list), dtype=np.float32)
    X_sparse = csr_matrix((sparse_data, (rows_list, cols_list)), shape=(n_total, len(token_map)), dtype=np.float32)
    
    n_components = min(6, len(token_map) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd_factors = svd.fit_transform(X_sparse)
    
    for comp in range(n_components):
        col_name = f"svd_pheno_{comp}"
        train[col_name] = svd_factors[:n_train_rows, comp].astype(np.float64)
        eval_cols[col_name] = svd_factors[n_train_rows:, comp].astype(np.float64)
        extra_numeric.append(col_name)

    # 12. Encode datasets
    encoder = TabularEncoder(min_count=35)
    encoder.fit(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)
    
    # 13. Fit Dual-Topology Ensemble (LightGBM + XGBoost) across 5 folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lgb_preds = np.zeros(x_eval.shape[0], dtype=np.float64)
    xgb_preds = np.zeros(x_eval.shape[0], dtype=np.float64)
    
    for train_idx, val_idx in skf.split(x_train, y_train):
        fold_x_train, fold_y_train = x_train[train_idx], y_train[train_idx]
        
        # Leaf-wise LightGBM with categorical feature regularization
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.018,
            num_leaves=31,
            min_child_samples=130,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=0.5,
            cat_smooth=40.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        lgb_model.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        lgb_preds += lgb_model.predict_proba(x_eval)[:, 1] / 5.0
        
        # Depth-wise XGBoost with strict L2 regularization
        xgb_model = xgb.XGBClassifier(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.025,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_lambda=2.0,
            tree_method='hist',
            random_state=42,
            n_jobs=1,
            eval_metric='logloss',
        )
        xgb_model.fit(fold_x_train, fold_y_train)
        xgb_preds += xgb_model.predict_proba(x_eval)[:, 1] / 5.0
        
    return 0.60 * lgb_preds + 0.40 * xgb_preds


# EVOLVE-BLOCK-END
# =====================================================================
