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
        
    # 3. Grouped ICD-9 chapters and interactive category pairs
    for col in diag_cols:
        grouped_col = col + "_grp"
        train[grouped_col] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_cols[grouped_col] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(grouped_col)
        
    # Discharge & Admission pair
    train['disch_adm_pair'] = np.array([f"{d}_{a}" for d, a in zip(train['discharge_disposition_id'], train['admission_source_id'])], dtype=object)
    eval_cols['disch_adm_pair'] = np.array([f"{d}_{a}" for d, a in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])], dtype=object)
    extra_categorical.append('disch_adm_pair')
    
    # Primary Diagnosis & Discharge pair
    train['diag1_disch_pair'] = np.array([f"{dg}_{d}" for dg, d in zip(train['diag_1_grp'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['diag1_disch_pair'] = np.array([f"{dg}_{d}" for dg, d in zip(eval_cols['diag_1_grp'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('diag1_disch_pair')
        
    # 4. Target encoding of high-cardinality columns
    te_cols = [
        'diag_1', 'diag_2', 'diag_3', 'medical_specialty', 
        'discharge_disposition_id', 'admission_source_id', 'payer_code',
        'disch_adm_pair', 'diag1_disch_pair'
    ]
    for col in te_cols:
        if col in train and col in eval_cols:
            tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42)
            te_col = col + "_te"
            train[te_col] = tr_enc
            eval_cols[te_col] = ev_enc
            extra_numeric.append(te_col)
            
    # 5. Utilisation intensity features and indicators
    train_total_visits = train['number_outpatient'] + train['number_emergency'] + train['number_inpatient']
    eval_total_visits = eval_cols['number_outpatient'] + eval_cols['number_emergency'] + eval_cols['number_inpatient']
    train['total_visits'] = train_total_visits
    eval_cols['total_visits'] = eval_total_visits
    extra_numeric.append('total_visits')
    
    train['inpatient_ratio'] = train['number_inpatient'] / (train_total_visits + 1.0)
    eval_cols['inpatient_ratio'] = eval_cols['number_inpatient'] / (eval_total_visits + 1.0)
    extra_numeric.append('inpatient_ratio')

    # Binary flags for prior utilization
    for prefix, data in [('train', train), ('eval', eval_cols)]:
        data['has_outpatient'] = (data['number_outpatient'] > 0).astype(np.float64)
        data['has_emergency'] = (data['number_emergency'] > 0).astype(np.float64)
        data['has_inpatient'] = (data['number_inpatient'] > 0).astype(np.float64)
    extra_numeric.extend(['has_outpatient', 'has_emergency', 'has_inpatient'])
    
    # 6. Medication active, change counts, and indicators
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
    
    # 7. Insulin change interaction and dosage adjustments
    train['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(train['insulin'], train['change'])], dtype=object)
    eval_cols['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(eval_cols['insulin'], eval_cols['change'])], dtype=object)
    extra_categorical.append('insulin_change')

    for prefix, data in [('train', train), ('eval', eval_cols)]:
        data['insulin_up'] = (data['insulin'] == 'Up').astype(np.float64)
        data['insulin_down'] = (data['insulin'] == 'Down').astype(np.float64)
        data['insulin_steady'] = (data['insulin'] == 'Steady').astype(np.float64)
    extra_numeric.extend(['insulin_up', 'insulin_down', 'insulin_steady'])
    
    # 8. Numeric age representation and clinical stay features
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

    # Hospital care intensity density ratios
    for prefix, data in [('train', train), ('eval', eval_cols)]:
        h_days = data['time_in_hospital'] + 1.0
        data['meds_per_hospital_day'] = data['num_medications'] / h_days
        data['labs_per_hospital_day'] = data['num_lab_procedures'] / h_days
        data['procedures_per_hospital_day'] = data['num_procedures'] / h_days
    extra_numeric.extend(['meds_per_hospital_day', 'labs_per_hospital_day', 'procedures_per_hospital_day'])

    # Laboratory results flags
    for prefix, data in [('train', train), ('eval', eval_cols)]:
        data['A1C_elevated'] = np.isin(data['A1Cresult'], ['>7', '>8']).astype(np.float64)
        data['glu_elevated'] = np.isin(data['max_glu_serum'], ['>200', '>300']).astype(np.float64)
        data['A1C_measured'] = (data['A1Cresult'] != 'None').astype(np.float64)
        data['glu_measured'] = (data['max_glu_serum'] != 'None').astype(np.float64)
    extra_numeric.extend(['A1C_elevated', 'glu_elevated', 'A1C_measured', 'glu_measured'])

    # Missingness indicators
    for prefix, data in [('train', train), ('eval', eval_cols)]:
        data['payer_code_missing'] = (data['payer_code'] == None).astype(np.float64)
        data['medical_specialty_missing'] = (data['medical_specialty'] == None).astype(np.float64)
        data['weight_missing'] = (data['weight'] == None).astype(np.float64)
    extra_numeric.extend(['payer_code_missing', 'medical_specialty_missing', 'weight_missing'])

    # Medical complexity index
    for prefix, data in [('train', train), ('eval', eval_cols)]:
        data['complexity_index'] = (
            data['num_lab_procedures']
            + data['num_procedures'] * 5.0
            + data['num_medications'] * 2.0
            + data['number_diagnoses'] * 3.0
        )
    extra_numeric.append('complexity_index')

    # 9. Encode datasets
    encoder = TabularEncoder(min_count=30)
    encoder.fit(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)
    
    # 10. Fit Stratified 5-Fold Diverse LGBM Blend
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_preds = np.zeros(x_eval.shape[0])
    
    for train_idx, val_idx in skf.split(x_train, y_train):
        fold_x_train, fold_y_train = x_train[train_idx], y_train[train_idx]
        
        # Model A: Balanced Tuned Model
        model_a = lgb.LGBMClassifier(
            n_estimators=180,
            learning_rate=0.015,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=0.1,
            cat_smooth=50.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        model_a.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        pred_a = model_a.predict_proba(x_eval)[:, 1]
        
        # Model B: Shallow, conservative Model (robust to variance)
        model_b = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.015,
            num_leaves=15,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=1.0,
            cat_smooth=30.0,
            random_state=100,
            n_jobs=1,
            verbose=-1,
        )
        model_b.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        pred_b = model_b.predict_proba(x_eval)[:, 1]
        
        # Model C: Deeper, highly-regularized Model (captures high-order interactions)
        model_c = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.012,
            num_leaves=63,
            min_child_samples=200,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=5.0,
            reg_alpha=0.1,
            cat_smooth=70.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        )
        model_c.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        pred_c = model_c.predict_proba(x_eval)[:, 1]
        
        # Weighted Ensemble average prediction
        fold_pred = 0.4 * pred_a + 0.3 * pred_b + 0.3 * pred_c
        eval_preds += fold_pred / 5.0
        
    return eval_preds


# EVOLVE-BLOCK-END
# =====================================================================
