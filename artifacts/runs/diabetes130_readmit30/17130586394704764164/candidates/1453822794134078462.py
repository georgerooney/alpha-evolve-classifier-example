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
        
    # Compute Charlson Comorbidity Index
    train_cci = (
        train_cm.get('cm_mi', 0.0) * 1.0 +
        train_cm.get('cm_chf', 0.0) * 1.0 +
        train_cm.get('cm_pvd', 0.0) * 1.0 +
        train_cm.get('cm_cerebrovascular', 0.0) * 1.0 +
        train_cm.get('cm_dementia', 0.0) * 1.0 +
        train_cm.get('cm_pulmonary', 0.0) * 1.0 +
        train_cm.get('cm_rheumatic', 0.0) * 1.0 +
        train_cm.get('cm_peptic_ulcer', 0.0) * 1.0 +
        train_cm.get('cm_liver', 0.0) * 1.0 +
        train_cm.get('cm_hemiplegia', 0.0) * 2.0 +
        train_cm.get('cm_renal', 0.0) * 2.0 +
        train_cm.get('cm_cancer', 0.0) * 2.0 +
        train_cm.get('cm_metastatic', 0.0) * 6.0 +
        train_cm.get('cm_hiv', 0.0) * 6.0 +
        train_cm.get('cm_diabetes_complicated', 0.0) * 2.0
    )
    eval_cci = (
        eval_cm.get('cm_mi', 0.0) * 1.0 +
        eval_cm.get('cm_chf', 0.0) * 1.0 +
        eval_cm.get('cm_pvd', 0.0) * 1.0 +
        eval_cm.get('cm_cerebrovascular', 0.0) * 1.0 +
        eval_cm.get('cm_dementia', 0.0) * 1.0 +
        eval_cm.get('cm_pulmonary', 0.0) * 1.0 +
        eval_cm.get('cm_rheumatic', 0.0) * 1.0 +
        eval_cm.get('cm_peptic_ulcer', 0.0) * 1.0 +
        eval_cm.get('cm_liver', 0.0) * 1.0 +
        eval_cm.get('cm_hemiplegia', 0.0) * 2.0 +
        eval_cm.get('cm_renal', 0.0) * 2.0 +
        eval_cm.get('cm_cancer', 0.0) * 2.0 +
        eval_cm.get('cm_metastatic', 0.0) * 6.0 +
        eval_cm.get('cm_hiv', 0.0) * 6.0 +
        eval_cm.get('cm_diabetes_complicated', 0.0) * 2.0
    )
    train['charlson_index'] = train_cci
    eval_cols['charlson_index'] = eval_cci
    extra_numeric.append('charlson_index')
        
    # 3. Grouped ICD-9 chapters
    for col in diag_cols:
        grouped_col = col + "_grp"
        train[grouped_col] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_cols[grouped_col] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(grouped_col)
        
    # Add diabetes and circulatory indicators
    train['primary_diabetes'] = np.array([1.0 if x == 'diabetes' else 0.0 for x in train['diag_1_grp']], dtype=np.float64)
    eval_cols['primary_diabetes'] = np.array([1.0 if x == 'diabetes' else 0.0 for x in eval_cols['diag_1_grp']], dtype=np.float64)
    extra_numeric.append('primary_diabetes')
    
    train['any_diabetes_diag'] = np.array([
        1.0 if 'diabetes' in (d1, d2, d3) else 0.0 
        for d1, d2, d3 in zip(train['diag_1_grp'], train['diag_2_grp'], train['diag_3_grp'])
    ], dtype=np.float64)
    eval_cols['any_diabetes_diag'] = np.array([
        1.0 if 'diabetes' in (d1, d2, d3) else 0.0 
        for d1, d2, d3 in zip(eval_cols['diag_1_grp'], eval_cols['diag_2_grp'], eval_cols['diag_3_grp'])
    ], dtype=np.float64)
    extra_numeric.append('any_diabetes_diag')

    train['any_circulatory_diag'] = np.array([
        1.0 if 'circulatory' in (d1, d2, d3) else 0.0 
        for d1, d2, d3 in zip(train['diag_1_grp'], train['diag_2_grp'], train['diag_3_grp'])
    ], dtype=np.float64)
    eval_cols['any_circulatory_diag'] = np.array([
        1.0 if 'circulatory' in (d1, d2, d3) else 0.0 
        for d1, d2, d3 in zip(eval_cols['diag_1_grp'], eval_cols['diag_2_grp'], eval_cols['diag_3_grp'])
    ], dtype=np.float64)
    extra_numeric.append('any_circulatory_diag')

    # 4. Target encoding of high-cardinality columns
    te_cols = ['diag_1', 'diag_2', 'diag_3', 'medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'payer_code']
    for col in te_cols:
        if col in train and col in eval_cols:
            tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42)
            te_col = col + "_te"
            train[te_col] = tr_enc
            eval_cols[te_col] = ev_enc
            extra_numeric.append(te_col)
            
    # Target encode high-cardinality interactions
    train['spec_discharge'] = np.array([f"{spec}_{disp}" for spec, disp in zip(train['medical_specialty'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['spec_discharge'] = np.array([f"{spec}_{disp}" for spec, disp in zip(eval_cols['medical_specialty'], eval_cols['discharge_disposition_id'])], dtype=object)
    tr_enc, ev_enc = oof_target_encode(train['spec_discharge'], y_train, eval_cols['spec_discharge'], n_folds=5, smoothing=20.0, seed=42)
    train['spec_discharge_te'] = tr_enc
    eval_cols['spec_discharge_te'] = ev_enc
    extra_numeric.append('spec_discharge_te')

    train['diag1_discharge'] = np.array([f"{diag}_{disp}" for diag, disp in zip(train['diag_1'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['diag1_discharge'] = np.array([f"{diag}_{disp}" for diag, disp in zip(eval_cols['diag_1'], eval_cols['discharge_disposition_id'])], dtype=object)
    tr_enc, ev_enc = oof_target_encode(train['diag1_discharge'], y_train, eval_cols['diag1_discharge'], n_folds=5, smoothing=20.0, seed=42)
    train['diag1_discharge_te'] = tr_enc
    eval_cols['diag1_discharge_te'] = ev_enc
    extra_numeric.append('diag1_discharge_te')
            
    # 5. Utilisation intensity features
    train_total_visits = train['number_outpatient'] + train['number_emergency'] + train['number_inpatient']
    eval_total_visits = eval_cols['number_outpatient'] + eval_cols['number_emergency'] + eval_cols['number_inpatient']
    train['total_visits'] = train_total_visits
    eval_cols['total_visits'] = eval_total_visits
    extra_numeric.append('total_visits')
    
    train['inpatient_ratio'] = train['number_inpatient'] / (train_total_visits + 1.0)
    eval_cols['inpatient_ratio'] = eval_cols['number_inpatient'] / (eval_total_visits + 1.0)
    extra_numeric.append('inpatient_ratio')
    
    train['emergency_ratio'] = train['number_emergency'] / (train_total_visits + 1.0)
    eval_cols['emergency_ratio'] = eval_cols['number_emergency'] / (eval_total_visits + 1.0)
    extra_numeric.append('emergency_ratio')
    
    train['outpatient_ratio'] = train['number_outpatient'] / (train_total_visits + 1.0)
    eval_cols['outpatient_ratio'] = eval_cols['number_outpatient'] / (eval_total_visits + 1.0)
    extra_numeric.append('outpatient_ratio')
    
    train['lab_proc_ratio'] = train['num_lab_procedures'] / (train['num_procedures'] + 1.0)
    eval_cols['lab_proc_ratio'] = eval_cols['num_lab_procedures'] / (eval_cols['num_procedures'] + 1.0)
    extra_numeric.append('lab_proc_ratio')
    
    train['prior_util_score'] = train['number_inpatient'] * 3.0 + train['number_emergency'] * 1.5 + train['number_outpatient'] * 0.5
    eval_cols['prior_util_score'] = eval_cols['number_inpatient'] * 3.0 + eval_cols['number_emergency'] * 1.5 + eval_cols['number_outpatient'] * 0.5
    extra_numeric.append('prior_util_score')
    
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
    
    train_meds_up = np.zeros(len(y_train), dtype=np.float64)
    eval_meds_up = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_meds_down = np.zeros(len(y_train), dtype=np.float64)
    eval_meds_down = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_meds_steady = np.zeros(len(y_train), dtype=np.float64)
    eval_meds_steady = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    
    for col in med_cols:
        if col in train:
            train_active_meds += (train[col] != "No") & (train[col] != None)
            eval_active_meds += (eval_cols[col] != "No") & (eval_cols[col] != None)
            train_change_meds += (train[col] == "Up") | (train[col] == "Down")
            eval_change_meds += (eval_cols[col] == "Up") | (eval_cols[col] == "Down")
            
            train_meds_up += (train[col] == "Up")
            eval_meds_up += (eval_cols[col] == "Up")
            train_meds_down += (train[col] == "Down")
            eval_meds_down += (eval_cols[col] == "Down")
            train_meds_steady += (train[col] == "Steady")
            eval_meds_steady += (eval_cols[col] == "Steady")
            
    train['active_meds_count'] = train_active_meds
    eval_cols['active_meds_count'] = eval_active_meds
    extra_numeric.append('active_meds_count')
    
    train['change_meds_count'] = train_change_meds
    eval_cols['change_meds_count'] = eval_change_meds
    extra_numeric.append('change_meds_count')
    
    train['meds_up_count'] = train_meds_up
    eval_cols['meds_up_count'] = eval_meds_up
    extra_numeric.append('meds_up_count')
    
    train['meds_down_count'] = train_meds_down
    eval_cols['meds_down_count'] = eval_meds_down
    extra_numeric.append('meds_down_count')
    
    train['meds_steady_count'] = train_meds_steady
    eval_cols['meds_steady_count'] = eval_meds_steady
    extra_numeric.append('meds_steady_count')
    
    # 7. Insulin change interaction
    train['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(train['insulin'], train['change'])], dtype=object)
    eval_cols['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(eval_cols['insulin'], eval_cols['change'])], dtype=object)
    extra_categorical.append('insulin_change')
    
    # Lab results and insulin interaction
    train['A1C_insulin'] = np.array([f"{a1c}_{ins}" for a1c, ins in zip(train['A1Cresult'], train['insulin'])], dtype=object)
    eval_cols['A1C_insulin'] = np.array([f"{a1c}_{ins}" for a1c, ins in zip(eval_cols['A1Cresult'], eval_cols['insulin'])], dtype=object)
    extra_categorical.append('A1C_insulin')
    
    train['glu_insulin'] = np.array([f"{glu}_{ins}" for glu, ins in zip(train['max_glu_serum'], train['insulin'])], dtype=object)
    eval_cols['glu_insulin'] = np.array([f"{glu}_{ins}" for glu, ins in zip(eval_cols['max_glu_serum'], eval_cols['insulin'])], dtype=object)
    extra_categorical.append('glu_insulin')
    
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

    # Age interaction features
    train['age_diagnoses'] = train['age_num'] * train['number_diagnoses']
    eval_cols['age_diagnoses'] = eval_cols['age_num'] * eval_cols['number_diagnoses']
    extra_numeric.append('age_diagnoses')
    
    train['age_medications'] = train['age_num'] * train['num_medications']
    eval_cols['age_medications'] = eval_cols['age_num'] * eval_cols['num_medications']
    extra_numeric.append('age_medications')
    
    train['meds_per_diagnosis'] = train['num_medications'] / (train['number_diagnoses'] + 1.0)
    eval_cols['meds_per_diagnosis'] = eval_cols['num_medications'] / (eval_cols['number_diagnoses'] + 1.0)
    extra_numeric.append('meds_per_diagnosis')

    # Admission & discharge specific indicator features
    def is_discharged_home(disp_id):
        if disp_id is None:
            return 0.0
        s = str(disp_id).strip()
        return 1.0 if s in ('1', '6', '8') else 0.0

    train['discharged_home'] = np.array([is_discharged_home(x) for x in train['discharge_disposition_id']], dtype=np.float64)
    eval_cols['discharged_home'] = np.array([is_discharged_home(x) for x in eval_cols['discharge_disposition_id']], dtype=np.float64)
    extra_numeric.append('discharged_home')

    def is_discharged_facility(disp_id):
        if disp_id is None:
            return 0.0
        s = str(disp_id).strip()
        return 1.0 if s in ('3', '4', '23', '24') else 0.0

    train['discharged_facility'] = np.array([is_discharged_facility(x) for x in train['discharge_disposition_id']], dtype=np.float64)
    eval_cols['discharged_facility'] = np.array([is_discharged_facility(x) for x in eval_cols['discharge_disposition_id']], dtype=np.float64)
    extra_numeric.append('discharged_facility')

    def is_admission_emergency(adm_id):
        if adm_id is None:
            return 0.0
        s = str(adm_id).strip()
        return 1.0 if s == '7' else 0.0

    train['admission_emergency'] = np.array([is_admission_emergency(x) for x in train['admission_source_id']], dtype=np.float64)
    eval_cols['admission_emergency'] = np.array([is_admission_emergency(x) for x in eval_cols['admission_source_id']], dtype=np.float64)
    extra_numeric.append('admission_emergency')

    def is_admission_transfer(adm_id):
        if adm_id is None:
            return 0.0
        s = str(adm_id).strip()
        return 1.0 if s in ('4', '5', '6', '20', '22', '25') else 0.0

    train['admission_transfer'] = np.array([is_admission_transfer(x) for x in train['admission_source_id']], dtype=np.float64)
    eval_cols['admission_transfer'] = np.array([is_admission_transfer(x) for x in eval_cols['admission_source_id']], dtype=np.float64)
    extra_numeric.append('admission_transfer')

    def is_type_emergency(adm_type):
        if adm_type is None:
            return 0.0
        s = str(adm_type).strip()
        return 1.0 if s in ('1', '2', '7', '8') else 0.0

    train['type_emergency'] = np.array([is_type_emergency(x) for x in train['admission_type_id']], dtype=np.float64)
    eval_cols['type_emergency'] = np.array([is_type_emergency(x) for x in eval_cols['admission_type_id']], dtype=np.float64)
    extra_numeric.append('type_emergency')

    # Missingness/presence indicators
    train['has_weight'] = np.array([0.0 if x is None else 1.0 for x in train['weight']], dtype=np.float64)
    eval_cols['has_weight'] = np.array([0.0 if x is None else 1.0 for x in eval_cols['weight']], dtype=np.float64)
    extra_numeric.append('has_weight')

    train['has_specialty'] = np.array([0.0 if x is None else 1.0 for x in train['medical_specialty']], dtype=np.float64)
    eval_cols['has_specialty'] = np.array([0.0 if x is None else 1.0 for x in eval_cols['medical_specialty']], dtype=np.float64)
    extra_numeric.append('has_specialty')

    # 9. Encode datasets
    encoder = TabularEncoder(min_count=50)
    encoder.fit(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)
    
    # 10. Fit Stratified 5-Fold LGBM Heterogeneous Ensemble
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_preds = np.zeros(x_eval.shape[0])
    
    model_params = [
        {
            'n_estimators': 200,
            'learning_rate': 0.015,
            'num_leaves': 31,
            'min_child_samples': 150,
            'subsample': 0.7,
            'subsample_freq': 1,
            'colsample_bytree': 0.6,
            'reg_lambda': 0.1,
            'cat_smooth': 50.0,
            'random_state': 42,
            'n_jobs': 1,
            'verbose': -1,
        },
        {
            'n_estimators': 250,
            'learning_rate': 0.012,
            'num_leaves': 45,
            'min_child_samples': 100,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.75,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'cat_smooth': 30.0,
            'random_state': 101,
            'n_jobs': 1,
            'verbose': -1,
        },
        {
            'n_estimators': 180,
            'learning_rate': 0.018,
            'num_leaves': 20,
            'min_child_samples': 200,
            'subsample': 0.6,
            'subsample_freq': 1,
            'colsample_bytree': 0.5,
            'reg_lambda': 10.0,
            'cat_smooth': 100.0,
            'random_state': 2023,
            'n_jobs': 1,
            'verbose': -1,
        }
    ]
    
    for train_idx, val_idx in skf.split(x_train, y_train):
        fold_x_train, fold_y_train = x_train[train_idx], y_train[train_idx]
        
        for params in model_params:
            model = lgb.LGBMClassifier(**params)
            model.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
            eval_preds += model.predict_proba(x_eval)[:, 1] / (5.0 * len(model_params))
        
    return eval_preds


# EVOLVE-BLOCK-END
# =====================================================================
