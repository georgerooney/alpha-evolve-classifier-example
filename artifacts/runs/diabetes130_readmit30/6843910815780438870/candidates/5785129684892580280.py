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
    
    train_feat['cm_chf_diab'] = train_cm['cm_chf'] * train_cm['cm_diabetes_complicated']
    eval_feat['cm_chf_diab'] = eval_cm['cm_chf'] * eval_cm['cm_diabetes_complicated']
    extra_numeric.append('cm_chf_diab')
    
    train_feat['cm_pulmonary_chf'] = train_cm['cm_pulmonary'] * train_cm['cm_chf']
    eval_feat['cm_pulmonary_chf'] = eval_cm['cm_pulmonary'] * eval_cm['cm_chf']
    extra_numeric.append('cm_pulmonary_chf')
        
    # 2. ICD9 groups (categorical)
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    for col in diag_cols:
        train_feat[f'{col}_grp'] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_feat[f'{col}_grp'] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(f'{col}_grp')
        
    # Group counts across diagnoses
    groups_list = ['circulatory', 'respiratory', 'digestive', 'diabetes', 'injury', 'musculoskeletal', 'genitourinary', 'neoplasms', 'other']
    for grp in groups_list:
        train_feat[f'grp_cnt_{grp}'] = (
            (train_feat['diag_1_grp'] == grp).astype(float) +
            (train_feat['diag_2_grp'] == grp).astype(float) +
            (train_feat['diag_3_grp'] == grp).astype(float)
        )
        eval_feat[f'grp_cnt_{grp}'] = (
            (eval_feat['diag_1_grp'] == grp).astype(float) +
            (eval_feat['diag_2_grp'] == grp).astype(float) +
            (eval_feat['diag_3_grp'] == grp).astype(float)
        )
        extra_numeric.append(f'grp_cnt_{grp}')
        
    # 3. Admission/Discharge interaction
    train_feat['dis_adm'] = np.array([
        f"{d}_{a}" for d, a in zip(train['discharge_disposition_id'], train['admission_source_id'])
    ], dtype=object)
    eval_feat['dis_adm'] = np.array([
        f"{d}_{a}" for d, a in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])
    ], dtype=object)
    extra_categorical.append('dis_adm')
    
    # 3b. Medication combo interaction
    train_feat['med_combo'] = np.array([
        f"{m}_{i}_{g}" for m, i, g in zip(train['metformin'], train['insulin'], train['glipizide'])
    ], dtype=object)
    eval_feat['med_combo'] = np.array([
        f"{m}_{i}_{g}" for m, i, g in zip(eval_cols['metformin'], eval_cols['insulin'], eval_cols['glipizide'])
    ], dtype=object)
    extra_categorical.append('med_combo')
    
    # 3c. Change & diabetesMed interaction
    train_feat['change_med'] = np.array([
        f"{c}_{d}" for c, d in zip(train['change'], train['diabetesMed'])
    ], dtype=object)
    eval_feat['change_med'] = np.array([
        f"{c}_{d}" for c, d in zip(eval_cols['change'], eval_cols['diabetesMed'])
    ], dtype=object)
    extra_categorical.append('change_med')
        
    # 4. Target encoding of high-cardinality categorical variables
    high_card_cols = [
        'diag_1', 'diag_2', 'diag_3', 'medical_specialty', 
        'discharge_disposition_id', 'admission_source_id', 'payer_code', 'dis_adm', 'med_combo', 'change_med'
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
        
        return {
            'med_num_up': num_up,
            'med_num_down': num_down,
            'med_num_steady': num_steady,
            'med_num_active': num_active,
            'med_num_change': num_change,
            'insulin_up': insulin_up,
            'insulin_down': insulin_down,
            'insulin_steady': insulin_steady,
            'insulin_active': insulin_active
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
        
        lab_proc = data_dict.get('num_lab_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        proc = data_dict.get('num_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        meds = data_dict.get('num_medications', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        diags = data_dict.get('number_diagnoses', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        los = data_dict.get('time_in_hospital', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        
        severity_index = lab_proc + 2.0 * proc + meds + 5.0 * diags
        los_med_ratio = los / (meds + 1.0)
        
        target_dict['util_index'] = util_index
        target_dict['util_any'] = util_any
        target_dict['severity_index'] = severity_index
        target_dict['los_med_ratio'] = los_med_ratio
        
        target_dict['has_prior_inpatient'] = (inpatient > 0).astype(float)
        target_dict['has_prior_emergency'] = (emergency > 0).astype(float)
        target_dict['has_prior_outpatient'] = (outpatient > 0).astype(float)
        target_dict['frequent_inpatient'] = (inpatient >= 2).astype(float)
        target_dict['frequent_emergency'] = (emergency >= 2).astype(float)
        
    add_custom_numeric(train, train_feat)
    add_custom_numeric(eval_cols, eval_feat)
    extra_numeric.extend([
        'util_index', 'util_any', 'severity_index', 'los_med_ratio',
        'has_prior_inpatient', 'has_prior_emergency', 'has_prior_outpatient', 'frequent_inpatient', 'frequent_emergency'
    ])
    
    # 7. Patient demography, lab indicators, ratios, and frailty
    def parse_age(age_col):
        mapping = {
            "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
            "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
            "[80-90)": 85.0, "[90-100)": 95.0
        }
        return np.array([mapping.get(x, 55.0) for x in age_col], dtype=float)
        
    train_feat['age_num'] = parse_age(train.get('age', [None]*len(train['time_in_hospital'])))
    eval_feat['age_num'] = parse_age(eval_cols.get('age', [None]*len(eval_cols['time_in_hospital'])))
    extra_numeric.append('age_num')
    
    def add_frailty(data_dict, target_dict, age_array):
        meds = data_dict.get('num_medications', np.zeros_like(age_array, dtype=float))
        diags = data_dict.get('number_diagnoses', np.zeros_like(age_array, dtype=float))
        target_dict['frailty_index'] = (age_array / 100.0) * (meds + 2.0 * diags)

    add_frailty(train, train_feat, train_feat['age_num'])
    add_frailty(eval_cols, eval_feat, eval_feat['age_num'])
    extra_numeric.append('frailty_index')
    
    def parse_weight(weight_col):
        mapping = {
            "[0-25)": 12.5, "[25-50)": 37.5, "[50-75)": 62.5, "[75-100)": 87.5,
            "[100-125)": 112.5, "[125-150)": 137.5, "[150-175)": 162.5,
            "[175-200)": 187.5, ">200": 225.0
        }
        return np.array([mapping.get(x, np.nan) for x in weight_col], dtype=float)

    train_feat['weight_num'] = parse_weight(train.get('weight', [None]*len(train['time_in_hospital'])))
    eval_feat['weight_num'] = parse_weight(eval_cols.get('weight', [None]*len(eval_cols['time_in_hospital'])))
    extra_numeric.append('weight_num')

    train_feat['weight_missing'] = np.isnan(train_feat['weight_num']).astype(float)
    eval_feat['weight_missing'] = np.isnan(eval_feat['weight_num']).astype(float)
    extra_numeric.append('weight_missing')
    
    train_feat['gender_female'] = (train.get('gender', np.array(['Unknown']*len(train['time_in_hospital']))) == 'Female').astype(float)
    eval_feat['gender_female'] = (eval_cols.get('gender', np.array(['Unknown']*len(eval_cols['time_in_hospital']))) == 'Female').astype(float)
    extra_numeric.append('gender_female')
    
    def add_lab_features(data_dict, target_dict):
        glu = data_dict.get('max_glu_serum', np.array(['None'] * len(data_dict['time_in_hospital'])))
        a1c = data_dict.get('A1Cresult', np.array(['None'] * len(data_dict['time_in_hospital'])))
        
        target_dict['glu_high'] = np.isin(glu, ['>200', '>300']).astype(float)
        target_dict['glu_tested'] = (glu != 'None').astype(float)
        target_dict['a1c_high'] = np.isin(a1c, ['>7', '>8']).astype(float)
        target_dict['a1c_tested'] = (a1c != 'None').astype(float)

    add_lab_features(train, train_feat)
    add_lab_features(eval_cols, eval_feat)
    extra_numeric.extend(['glu_high', 'glu_tested', 'a1c_high', 'a1c_tested'])
    
    def add_ratios(data_dict, target_dict):
        los = data_dict.get('time_in_hospital', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        lab = data_dict.get('num_lab_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        proc = data_dict.get('num_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        meds = data_dict.get('num_medications', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        
        target_dict['lab_per_day'] = lab / (los + 1.0)
        target_dict['proc_per_day'] = proc / (los + 1.0)
        target_dict['meds_per_day'] = meds / (los + 1.0)

    add_ratios(train, train_feat)
    add_ratios(eval_cols, eval_feat)
    extra_numeric.extend(['lab_per_day', 'proc_per_day', 'meds_per_day'])
    
    train_feat['max_diagnoses'] = (train.get('number_diagnoses', np.zeros_like(train_feat['age_num'])) >= 9).astype(float)
    eval_feat['max_diagnoses'] = (eval_cols.get('number_diagnoses', np.zeros_like(eval_feat['age_num'])) >= 9).astype(float)
    extra_numeric.append('max_diagnoses')

    # 8. Clinical admission, discharge, and high-risk diagnosis flags
    high_risk_disch = {'3', '6', '7', '22', '28'}
    train_feat['high_risk_discharge'] = np.isin(train['discharge_disposition_id'], list(high_risk_disch)).astype(float)
    eval_feat['high_risk_discharge'] = np.isin(eval_cols['discharge_disposition_id'], list(high_risk_disch)).astype(float)
    extra_numeric.append('high_risk_discharge')

    train_feat['emergency_admission'] = np.isin(train.get('admission_type_id', []), ['1', '2']).astype(float)
    eval_feat['emergency_admission'] = np.isin(eval_cols.get('admission_type_id', []), ['1', '2']).astype(float)
    extra_numeric.append('emergency_admission')

    train_feat['er_admission_source'] = (train.get('admission_source_id', []) == '7').astype(float)
    eval_feat['er_admission_source'] = (eval_cols.get('admission_source_id', []) == '7').astype(float)
    extra_numeric.append('er_admission_source')

    # Primary diagnosis key risk categories (CHF: 428, Renal: 584-586, Resp: 480-486, Diabetes: 250)
    def check_diag_prefix(diag_col, prefixes):
        return np.array([any(str(x).startswith(p) for p in prefixes) if x is not None else False for x in diag_col], dtype=float)

    train_feat['primary_chf'] = check_diag_prefix(train['diag_1'], ['428'])
    eval_feat['primary_chf'] = check_diag_prefix(eval_cols['diag_1'], ['428'])
    extra_numeric.append('primary_chf')

    train_feat['primary_renal'] = check_diag_prefix(train['diag_1'], ['584', '585', '586'])
    eval_feat['primary_renal'] = check_diag_prefix(eval_cols['diag_1'], ['584', '585', '586'])
    extra_numeric.append('primary_renal')

    train_feat['primary_resp_inf'] = check_diag_prefix(train['diag_1'], ['480', '481', '482', '486'])
    eval_feat['primary_resp_inf'] = check_diag_prefix(eval_cols['diag_1'], ['480', '481', '482', '486'])
    extra_numeric.append('primary_resp_inf')

    train_feat['primary_diabetes'] = check_diag_prefix(train['diag_1'], ['250'])
    eval_feat['primary_diabetes'] = check_diag_prefix(eval_cols['diag_1'], ['250'])
    extra_numeric.append('primary_diabetes')

    # Combination therapy: insulin + oral agent
    train_feat['insulin_and_oral'] = ((train_med_feats['insulin_active'] > 0) & 
                                      ((train_med_feats['med_num_active'] - train_med_feats['insulin_active']) > 0)).astype(float)
    eval_feat['insulin_and_oral'] = ((eval_med_feats['insulin_active'] > 0) & 
                                     ((eval_med_feats['med_num_active'] - eval_med_feats['insulin_active']) > 0)).astype(float)
    extra_numeric.append('insulin_and_oral')

    # Nonlinear prior utilization dynamics
    inpatient_tr = train.get('number_inpatient', np.zeros_like(train['time_in_hospital'], dtype=float))
    emergency_tr = train.get('number_emergency', np.zeros_like(train['time_in_hospital'], dtype=float))
    outpatient_tr = train.get('number_outpatient', np.zeros_like(train['time_in_hospital'], dtype=float))

    inpatient_ev = eval_cols.get('number_inpatient', np.zeros_like(eval_cols['time_in_hospital'], dtype=float))
    emergency_ev = eval_cols.get('number_emergency', np.zeros_like(eval_cols['time_in_hospital'], dtype=float))
    outpatient_ev = eval_cols.get('number_outpatient', np.zeros_like(eval_cols['time_in_hospital'], dtype=float))

    total_prior_tr = inpatient_tr + emergency_tr + outpatient_tr
    total_prior_ev = inpatient_ev + emergency_ev + outpatient_ev

    train_feat['log_inpatient'] = np.log1p(inpatient_tr)
    eval_feat['log_inpatient'] = np.log1p(inpatient_ev)
    train_feat['log_emergency'] = np.log1p(emergency_tr)
    eval_feat['log_emergency'] = np.log1p(emergency_ev)
    train_feat['total_prior_visits'] = total_prior_tr
    eval_feat['total_prior_visits'] = total_prior_ev
    train_feat['inpatient_ratio'] = inpatient_tr / (total_prior_tr + 1.0)
    eval_feat['inpatient_ratio'] = inpatient_ev / (total_prior_ev + 1.0)
    extra_numeric.extend(['log_inpatient', 'log_emergency', 'total_prior_visits', 'inpatient_ratio'])

    # Return the clean processed feature maps
    return train_feat, eval_feat, extra_numeric, extra_categorical


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    import lightgbm as lgb
    from scipy.stats import rankdata

    train_feat, eval_feat, extra_numeric, extra_categorical = add_features(task, train, eval_cols, y_train)
    
    encoder = TabularEncoder(min_count=30)
    encoder.fit(task, train_feat, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)

    # Diverse Multi-Model LightGBM Ensemble spanning GBDT, GOSS, Extra Trees, and Regression MSE
    models = [
        lgb.LGBMClassifier(
            objective='binary',
            n_estimators=240,
            learning_rate=0.015,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=1.0,
            cat_smooth=40.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            objective='binary',
            n_estimators=260,
            learning_rate=0.012,
            num_leaves=45,
            min_child_samples=200,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=5.0,
            cat_smooth=80.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMRegressor(
            objective='regression',
            n_estimators=220,
            learning_rate=0.015,
            num_leaves=24,
            min_child_samples=120,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=0.5,
            cat_smooth=30.0,
            random_state=2023,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            objective='binary',
            extra_trees=True,
            n_estimators=200,
            learning_rate=0.018,
            num_leaves=18,
            min_child_samples=100,
            subsample=0.6,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=0.1,
            cat_smooth=20.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            objective='binary',
            boosting_type='goss',
            n_estimators=230,
            learning_rate=0.013,
            num_leaves=28,
            min_child_samples=150,
            colsample_bytree=0.65,
            reg_lambda=2.0,
            cat_smooth=50.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        )
    ]

    preds = np.zeros(len(x_eval))
    for model in models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(x_eval)[:, 1]
        else:
            p = model.predict(x_eval)
        preds += rankdata(p) / len(p)
    preds /= len(models)

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
