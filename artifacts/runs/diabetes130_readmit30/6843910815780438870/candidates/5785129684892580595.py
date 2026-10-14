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


def add_features(task, train, eval_cols, y_train):
    # Shallow copy to avoid mutating inputs directly
    train_feat = dict(train)
    eval_feat = dict(eval_cols)
    
    extra_numeric = []
    extra_categorical = []
    
    def parse_range(val):
        if not isinstance(val, str) or val in ('?', 'None', ''):
            return np.nan
        val = val.strip('[]() ')
        if '-' in val:
            parts = val.split('-')
            if len(parts) == 2:
                try:
                    return (float(parts[0]) + float(parts[1])) / 2.0
                except ValueError:
                    pass
        try:
            return float(val)
        except ValueError:
            pass
        return np.nan

    def parse_a1c(val):
        if not isinstance(val, str) or val == 'None':
            return np.nan, 0.0, 0.0
        perf = 1.0
        high = 1.0 if ('>' in val) else 0.0
        if val == '>8':
            num = 8.5
        elif val == '>7':
            num = 7.5
        elif val == 'Normal':
            num = 5.5
        else:
            num = np.nan
        return num, perf, high

    def parse_glu(val):
        if not isinstance(val, str) or val == 'None':
            return np.nan, 0.0, 0.0
        perf = 1.0
        high = 1.0 if ('>' in val) else 0.0
        if val == '>300':
            num = 350.0
        elif val == '>200':
            num = 250.0
        elif val == 'Normal':
            num = 100.0
        else:
            num = np.nan
        return num, perf, high

    # 1. Comorbidities
    train_cm = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    eval_cm = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    
    for k in train_cm.keys():
        train_feat[f'cm_{k}'] = train_cm[k]
        eval_feat[f'cm_{k}'] = eval_cm[k]
        extra_numeric.append(f'cm_{k}')
        
    # Specific clinical interactions
    train_feat['cm_chf_renal'] = train_cm['cm_chf'] * train_cm['cm_renal']
    eval_feat['cm_chf_renal'] = eval_cm['cm_chf'] * eval_cm['cm_renal']
    extra_numeric.append('cm_chf_renal')
    
    train_feat['cm_diab_renal'] = train_cm['cm_diabetes_complicated'] * train_cm['cm_renal']
    eval_feat['cm_diab_renal'] = eval_cm['cm_diabetes_complicated'] * eval_cm['cm_renal']
    extra_numeric.append('cm_diab_renal')
    
    train_feat['cm_chf_pulmonary'] = train_cm['cm_chf'] * train_cm['cm_pulmonary']
    eval_feat['cm_chf_pulmonary'] = eval_cm['cm_chf'] * eval_cm['cm_pulmonary']
    extra_numeric.append('cm_chf_pulmonary')
    
    train_feat['cm_cancer_metastatic'] = train_cm['cm_cancer'] * train_cm['cm_metastatic']
    eval_feat['cm_cancer_metastatic'] = eval_cm['cm_cancer'] * eval_cm['cm_metastatic']
    extra_numeric.append('cm_cancer_metastatic')
    
    train_feat['cm_vascular_count'] = train_cm['cm_chf'] + train_cm['cm_mi'] + train_cm['cm_pvd'] + train_cm['cm_cerebrovascular']
    eval_feat['cm_vascular_count'] = eval_cm['cm_chf'] + eval_cm['cm_mi'] + eval_cm['cm_pvd'] + eval_cm['cm_cerebrovascular']
    extra_numeric.append('cm_vascular_count')
        
    # 2. ICD9 groups (categorical)
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    for col in diag_cols:
        train_feat[f'{col}_grp'] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_feat[f'{col}_grp'] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(f'{col}_grp')
        
    # 3. Admission/Discharge interaction
    train_feat['dis_adm'] = np.array([
        f"{d}_{a}" for d, a in zip(train['discharge_disposition_id'], train['admission_source_id'])
    ], dtype=object)
    eval_feat['dis_adm'] = np.array([
        f"{d}_{a}" for d, a in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])
    ], dtype=object)
    extra_categorical.append('dis_adm')
    
    train_feat['dis_diag_grp'] = np.array([
        f"{d}_{g}" for d, g in zip(train['discharge_disposition_id'], train_feat['diag_1_grp'])
    ], dtype=object)
    eval_feat['dis_diag_grp'] = np.array([
        f"{d}_{g}" for d, g in zip(eval_cols['discharge_disposition_id'], eval_feat['diag_1_grp'])
    ], dtype=object)
    extra_categorical.append('dis_diag_grp')
        
    # 4. Target encoding of high-cardinality categorical variables
    high_card_cols = [
        'diag_1', 'diag_2', 'diag_3', 'medical_specialty', 
        'discharge_disposition_id', 'admission_source_id', 'payer_code', 'dis_adm', 'dis_diag_grp'
    ]
    for col in high_card_cols:
        tr_enc, ev_enc = oof_target_encode(train_feat[col], y_train, eval_feat[col], n_folds=5, smoothing=20.0, seed=42)
        train_feat[f'{col}_te'] = tr_enc
        eval_feat[f'{col}_te'] = ev_enc
        extra_numeric.append(f'{col}_te')
        
    # 5. Medication features
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    existing_med_cols = [c for c in med_cols if c in train]
    
    def get_med_features(data_dict):
        n_rows = len(data_dict['time_in_hospital'])
        num_up = np.zeros(n_rows, dtype=np.float64)
        num_down = np.zeros(n_rows, dtype=np.float64)
        num_steady = np.zeros(n_rows, dtype=np.float64)
        for col in existing_med_cols:
            arr = data_dict[col]
            num_up += (arr == 'Up').astype(np.float64)
            num_down += (arr == 'Down').astype(np.float64)
            num_steady += (arr == 'Steady').astype(np.float64)
            
        num_active = num_up + num_down + num_steady
        num_change = num_up + num_down
        
        # Insulin specific
        insulin_arr = data_dict.get('insulin', np.array(['No'] * n_rows))
        insulin_up = (insulin_arr == 'Up').astype(np.float64)
        insulin_down = (insulin_arr == 'Down').astype(np.float64)
        insulin_steady = (insulin_arr == 'Steady').astype(np.float64)
        insulin_active = (insulin_arr != 'No').astype(np.float64)
        
        meds_count = data_dict.get('num_medications', np.zeros(n_rows, dtype=np.float64))
        diabetic_meds_ratio = num_active / (meds_count + 1.0)
        
        return {
            'med_num_up': num_up,
            'med_num_down': num_down,
            'med_num_steady': num_steady,
            'med_num_active': num_active,
            'med_num_change': num_change,
            'insulin_up': insulin_up,
            'insulin_down': insulin_down,
            'insulin_steady': insulin_steady,
            'insulin_active': insulin_active,
            'diabetic_meds_ratio': diabetic_meds_ratio
        }
        
    train_med_feats = get_med_features(train)
    eval_med_feats = get_med_features(eval_cols)
    for k in train_med_feats.keys():
        train_feat[k] = train_med_feats[k]
        eval_feat[k] = eval_med_feats[k]
        extra_numeric.append(k)
        
    # 6. Utilisation and clinical severity interaction indices
    def add_custom_numeric(data_dict, target_dict):
        outpatient = data_dict.get('number_outpatient', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        emergency = data_dict.get('number_emergency', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        inpatient = data_dict.get('number_inpatient', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        
        util_index = outpatient + 2.0 * emergency + 3.0 * inpatient
        util_any = ((outpatient > 0) | (emergency > 0) | (inpatient > 0)).astype(np.float64)
        
        total_visits = outpatient + emergency + inpatient
        inpatient_ratio = inpatient / (total_visits + 1.0)
        emergency_ratio = emergency / (total_visits + 1.0)
        outpatient_ratio = outpatient / (total_visits + 1.0)
        
        lab_proc = data_dict.get('num_lab_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        proc = data_dict.get('num_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        meds = data_dict.get('num_medications', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        diags = data_dict.get('number_diagnoses', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        los = data_dict.get('time_in_hospital', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        
        severity_index = lab_proc + 2.0 * proc + meds + 5.0 * diags
        los_med_ratio = los / (meds + 1.0)
        
        lab_per_day = lab_proc / (los + 1.0)
        meds_per_day = meds / (los + 1.0)
        procedures_per_day = proc / (los + 1.0)
        
        target_dict['util_index'] = util_index
        target_dict['util_any'] = util_any
        target_dict['total_visits'] = total_visits
        target_dict['inpatient_ratio'] = inpatient_ratio
        target_dict['emergency_ratio'] = emergency_ratio
        target_dict['outpatient_ratio'] = outpatient_ratio
        target_dict['severity_index'] = severity_index
        target_dict['los_med_ratio'] = los_med_ratio
        target_dict['lab_per_day'] = lab_per_day
        target_dict['meds_per_day'] = meds_per_day
        target_dict['procedures_per_day'] = procedures_per_day
        
    add_custom_numeric(train, train_feat)
    add_custom_numeric(eval_cols, eval_feat)
    extra_numeric.extend([
        'util_index', 'util_any', 'total_visits', 'inpatient_ratio', 'emergency_ratio', 'outpatient_ratio',
        'severity_index', 'los_med_ratio', 'lab_per_day', 'meds_per_day', 'procedures_per_day'
    ])
    
    # 7. A1C and Glucose Parsing
    a1c_tr = [parse_a1c(x) for x in train['A1Cresult']]
    a1c_ev = [parse_a1c(x) for x in eval_cols['A1Cresult']]
    train_feat['a1c_numeric'] = np.array([x[0] for x in a1c_tr], dtype=np.float64)
    train_feat['a1c_performed'] = np.array([x[1] for x in a1c_tr], dtype=np.float64)
    train_feat['a1c_high'] = np.array([x[2] for x in a1c_tr], dtype=np.float64)
    eval_feat['a1c_numeric'] = np.array([x[0] for x in a1c_ev], dtype=np.float64)
    eval_feat['a1c_performed'] = np.array([x[1] for x in a1c_ev], dtype=np.float64)
    eval_feat['a1c_high'] = np.array([x[2] for x in a1c_ev], dtype=np.float64)
    extra_numeric.extend(['a1c_numeric', 'a1c_performed', 'a1c_high'])
    
    glu_tr = [parse_glu(x) for x in train['max_glu_serum']]
    glu_ev = [parse_glu(x) for x in eval_cols['max_glu_serum']]
    train_feat['glu_numeric'] = np.array([x[0] for x in glu_tr], dtype=np.float64)
    train_feat['glu_performed'] = np.array([x[1] for x in glu_tr], dtype=np.float64)
    train_feat['glu_high'] = np.array([x[2] for x in glu_tr], dtype=np.float64)
    eval_feat['glu_numeric'] = np.array([x[0] for x in glu_ev], dtype=np.float64)
    eval_feat['glu_performed'] = np.array([x[1] for x in glu_ev], dtype=np.float64)
    eval_feat['glu_high'] = np.array([x[2] for x in glu_ev], dtype=np.float64)
    extra_numeric.extend(['glu_numeric', 'glu_performed', 'glu_high'])

    # 8. Age and Weight Parsing
    train_feat['age_numeric'] = np.array([parse_range(x) for x in train['age']], dtype=np.float64)
    eval_feat['age_numeric'] = np.array([parse_range(x) for x in eval_cols['age']], dtype=np.float64)
    extra_numeric.append('age_numeric')
    
    train_feat['weight_numeric'] = np.array([parse_range(x) for x in train['weight']], dtype=np.float64)
    eval_feat['weight_numeric'] = np.array([parse_range(x) for x in eval_cols['weight']], dtype=np.float64)
    extra_numeric.append('weight_numeric')

    train_feat['weight_recorded'] = np.array([0.0 if x in (None, '?', 'None') else 1.0 for x in train['weight']], dtype=np.float64)
    eval_feat['weight_recorded'] = np.array([0.0 if x in (None, '?', 'None') else 1.0 for x in eval_cols['weight']], dtype=np.float64)
    extra_numeric.append('weight_recorded')
    
    return train_feat, eval_feat, extra_numeric, extra_categorical


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    train_feat, eval_feat, extra_numeric, extra_categorical = add_features(task, train, eval_cols, y_train)
    
    encoder = TabularEncoder(min_count=25)
    encoder.fit(task, train_feat, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)

    # Diverse Multi-Model LightGBM Ensemble
    lgb_models = [
        lgb.LGBMClassifier(
            n_estimators=180,
            learning_rate=0.015,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=0.05,
            cat_smooth=50.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.012,
            num_leaves=24,
            min_child_samples=250,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=0.1,
            cat_smooth=80.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=160,
            learning_rate=0.018,
            num_leaves=40,
            min_child_samples=100,
            subsample=0.6,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=0.01,
            cat_smooth=30.0,
            random_state=2023,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.014,
            num_leaves=18,
            min_child_samples=200,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=0.2,
            cat_smooth=60.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        )
    ]

    lgb_preds = np.zeros(len(x_eval))
    for model in lgb_models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        lgb_preds += model.predict_proba(x_eval)[:, 1]
    lgb_preds /= len(lgb_models)
    
    # Diverse XGBoost model to capture different structure and ordinal patterns
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(
        n_estimators=160,
        learning_rate=0.015,
        max_depth=5,
        min_child_weight=10,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        eval_metric='logloss'
    )
    xgb_model.fit(x_train, y_train)
    xgb_preds = xgb_model.predict_proba(x_eval)[:, 1]
    
    # Blend LightGBM (with native categorical logic) and XGBoost (treated ordinally)
    return 0.8 * lgb_preds + 0.2 * xgb_preds


# EVOLVE-BLOCK-END
# =====================================================================
