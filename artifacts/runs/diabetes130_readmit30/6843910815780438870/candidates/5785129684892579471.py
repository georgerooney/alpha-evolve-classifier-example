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
    
    n_train = len(train['time_in_hospital'])
    n_eval = len(eval_cols['time_in_hospital'])
    
    # 0. Numeric demographics: Age and Frailty early for clinical indices
    def parse_age(age_col):
        res = []
        for x in age_col:
            if x is None:
                res.append(55.0)
            else:
                try:
                    parts = str(x).replace('[', '').replace(')', '').replace(']', '').split('-')
                    val = 0.5 * (float(parts[0]) + float(parts[1]))
                    res.append(val)
                except Exception:
                    res.append(55.0)
        return np.array(res, dtype=float)
        
    train_feat['age_num'] = parse_age(train.get('age', [None] * n_train))
    eval_feat['age_num'] = parse_age(eval_cols.get('age', [None] * n_eval))
    extra_numeric.append('age_num')
    
    # 1. Comorbidities & Evidence-based Charlson Comorbidity Index (CCI)
    train_cm = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    eval_cm = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    
    for k in train_cm.keys():
        train_feat[f'cm_{k}'] = train_cm[k]
        eval_feat[f'cm_{k}'] = eval_cm[k]
        extra_numeric.append(f'cm_{k}')
        
    # Charlson Comorbidity Index (standard clinical weights)
    cci_tr = (
        train_cm['cm_mi'] + train_cm['cm_chf'] + train_cm['cm_pvd'] +
        train_cm['cm_cerebrovascular'] + train_cm['cm_dementia'] +
        train_cm['cm_pulmonary'] + train_cm['cm_rheumatic'] + train_cm['cm_peptic_ulcer'] +
        2.0 * (train_cm['cm_diabetes_complicated'] + train_cm['cm_hemiplegia'] +
               train_cm['cm_renal'] + train_cm['cm_cancer']) +
        3.0 * train_cm['cm_liver'] +
        6.0 * (train_cm['cm_metastatic'] + train_cm['cm_hiv'])
    )
    cci_ev = (
        eval_cm['cm_mi'] + eval_cm['cm_chf'] + eval_cm['cm_pvd'] +
        eval_cm['cm_cerebrovascular'] + eval_cm['cm_dementia'] +
        eval_cm['cm_pulmonary'] + eval_cm['cm_rheumatic'] + eval_cm['cm_peptic_ulcer'] +
        2.0 * (eval_cm['cm_diabetes_complicated'] + eval_cm['cm_hemiplegia'] +
               eval_cm['cm_renal'] + eval_cm['cm_cancer']) +
        3.0 * eval_cm['cm_liver'] +
        6.0 * (eval_cm['cm_metastatic'] + eval_cm['cm_hiv'])
    )
    train_feat['charlson_score'] = cci_tr
    eval_feat['charlson_score'] = cci_ev
    extra_numeric.append('charlson_score')

    # Age-Adjusted Charlson Comorbidity Index (AACCI): +1 point per decade > 40
    train_feat['aacci_score'] = cci_tr + np.maximum(0.0, np.floor((train_feat['age_num'] - 40.0) / 10.0))
    eval_feat['aacci_score'] = cci_ev + np.maximum(0.0, np.floor((eval_feat['age_num'] - 40.0) / 10.0))
    extra_numeric.append('aacci_score')

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
        
    # 2. ICD9 groups and 3-digit categories
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    for col in diag_cols:
        train_feat[f'{col}_grp'] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_feat[f'{col}_grp'] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(f'{col}_grp')

    def get_icd9_3digit(diag_col):
        res = []
        for x in diag_col:
            if x is None:
                res.append('missing')
            else:
                s = str(x).strip()
                res.append(s[:3] if len(s) >= 3 else s)
        return np.array(res, dtype=object)

    for col in diag_cols:
        col_3d = f'{col}_3digit'
        train_feat[col_3d] = get_icd9_3digit(train[col])
        eval_feat[col_3d] = get_icd9_3digit(eval_cols[col])
        extra_categorical.append(col_3d)

    # Primary diagnosis key condition flags
    def get_diag1_flags(data_dict, target_dict):
        d1 = data_dict['diag_1']
        target_dict['diag1_hf'] = np.array([x is not None and str(x).startswith('428') for x in d1], dtype=float)
        target_dict['diag1_dm'] = np.array([x is not None and str(x).startswith('250') for x in d1], dtype=float)
        target_dict['diag1_ami'] = np.array([x is not None and str(x).startswith('410') for x in d1], dtype=float)
        target_dict['diag1_ckd'] = np.array([x is not None and str(x).startswith('585') for x in d1], dtype=float)
        
    get_diag1_flags(train, train_feat)
    get_diag1_flags(eval_cols, eval_feat)
    extra_numeric.extend(['diag1_hf', 'diag1_dm', 'diag1_ami', 'diag1_ckd'])
        
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
        
    # 3. Admission/Discharge interactions & phenotypes
    train_feat['dis_adm'] = np.array([
        f"{d}_{a}" for d, a in zip(train['discharge_disposition_id'], train['admission_source_id'])
    ], dtype=object)
    eval_feat['dis_adm'] = np.array([
        f"{d}_{a}" for d, a in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])
    ], dtype=object)
    extra_categorical.append('dis_adm')

    train_feat['adm_type_dis'] = np.array([
        f"{a}_{d}" for a, d in zip(train['admission_type_id'], train['discharge_disposition_id'])
    ], dtype=object)
    eval_feat['adm_type_dis'] = np.array([
        f"{a}_{d}" for a, d in zip(eval_cols['admission_type_id'], eval_cols['discharge_disposition_id'])
    ], dtype=object)
    extra_categorical.append('adm_type_dis')

    # Discharge destination phenotypes
    train_feat['discharged_home'] = (train['discharge_disposition_id'] == '1').astype(float)
    eval_feat['discharged_home'] = (eval_cols['discharge_disposition_id'] == '1').astype(float)
    train_feat['discharged_snf_hha'] = np.isin(train['discharge_disposition_id'], ['3', '6']).astype(float)
    eval_feat['discharged_snf_hha'] = np.isin(eval_cols['discharge_disposition_id'], ['3', '6']).astype(float)
    train_feat['discharged_ama'] = (train['discharge_disposition_id'] == '7').astype(float)
    eval_feat['discharged_ama'] = (eval_cols['discharge_disposition_id'] == '7').astype(float)
    train_feat['is_elective_admission'] = (train['admission_type_id'] == '3').astype(float)
    eval_feat['is_elective_admission'] = (eval_cols['admission_type_id'] == '3').astype(float)
    train_feat['is_emergency_admission'] = (train['admission_type_id'] == '1').astype(float)
    eval_feat['is_emergency_admission'] = (eval_cols['admission_type_id'] == '1').astype(float)
    extra_numeric.extend([
        'discharged_home', 'discharged_snf_hha', 'discharged_ama',
        'is_elective_admission', 'is_emergency_admission'
    ])
    
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
        
    # 4. Target encoding of high-cardinality categorical variables & diagnosis aggregates
    high_card_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_3digit', 'diag_2_3digit', 'diag_3_3digit',
        'medical_specialty', 'admission_type_id',
        'discharge_disposition_id', 'admission_source_id',
        'payer_code', 'dis_adm', 'adm_type_dis', 'med_combo', 'change_med'
    ]
    for col in high_card_cols:
        tr_enc, ev_enc = oof_target_encode(train_feat[col], y_train, eval_feat[col], n_folds=5, smoothing=20.0, seed=42)
        train_feat[f'{col}_te'] = tr_enc
        eval_feat[f'{col}_te'] = ev_enc
        extra_numeric.append(f'{col}_te')

    # Aggregated diagnosis target risk summaries
    train_feat['max_diag_te'] = np.maximum(train_feat['diag_1_te'], np.maximum(train_feat['diag_2_te'], train_feat['diag_3_te']))
    eval_feat['max_diag_te'] = np.maximum(eval_feat['diag_1_te'], np.maximum(eval_feat['diag_2_te'], eval_feat['diag_3_te']))
    extra_numeric.append('max_diag_te')

    train_feat['mean_diag_te'] = (train_feat['diag_1_te'] + train_feat['diag_2_te'] + train_feat['diag_3_te']) / 3.0
    eval_feat['mean_diag_te'] = (eval_feat['diag_1_te'] + eval_feat['diag_2_te'] + eval_feat['diag_3_te']) / 3.0
    extra_numeric.append('mean_diag_te')

    train_feat['max_diag3_te'] = np.maximum(train_feat['diag_1_3digit_te'], np.maximum(train_feat['diag_2_3digit_te'], train_feat['diag_3_3digit_te']))
    eval_feat['max_diag3_te'] = np.maximum(eval_feat['diag_1_3digit_te'], np.maximum(eval_feat['diag_2_3digit_te'], eval_feat['diag_3_3digit_te']))
    extra_numeric.append('max_diag3_te')

    train_feat['mean_diag3_te'] = (train_feat['diag_1_3digit_te'] + train_feat['diag_2_3digit_te'] + train_feat['diag_3_3digit_te']) / 3.0
    eval_feat['mean_diag3_te'] = (eval_feat['diag_1_3digit_te'] + eval_feat['diag_2_3digit_te'] + eval_feat['diag_3_3digit_te']) / 3.0
    extra_numeric.append('mean_diag3_te')
        
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

    # Hypoglycemia and medication escalation interactions
    train_feat['insulin_with_change'] = ((train['insulin'] != 'No') & (train['change'] == 'Ch')).astype(float)
    eval_feat['insulin_with_change'] = ((eval_cols['insulin'] != 'No') & (eval_cols['change'] == 'Ch')).astype(float)
    
    on_su_tr = (
        (train.get('glipizide', np.array(['No'] * n_train)) != 'No') |
        (train.get('glyburide', np.array(['No'] * n_train)) != 'No') |
        (train.get('glimepiride', np.array(['No'] * n_train)) != 'No')
    ).astype(float)
    on_su_ev = (
        (eval_cols.get('glipizide', np.array(['No'] * n_eval)) != 'No') |
        (eval_cols.get('glyburide', np.array(['No'] * n_eval)) != 'No') |
        (eval_cols.get('glimepiride', np.array(['No'] * n_eval)) != 'No')
    ).astype(float)
    
    train_feat['insulin_and_su'] = train_med_feats['insulin_active'] * on_su_tr
    eval_feat['insulin_and_su'] = eval_med_feats['insulin_active'] * on_su_ev
    extra_numeric.extend(['insulin_with_change', 'insulin_and_su'])
        
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

    # Utilisation proportions and disease interaction features
    inpatient_tr = train.get('number_inpatient', np.zeros(n_train, dtype=float))
    inpatient_ev = eval_cols.get('number_inpatient', np.zeros(n_eval, dtype=float))
    emergency_tr = train.get('number_emergency', np.zeros(n_train, dtype=float))
    emergency_ev = eval_cols.get('number_emergency', np.zeros(n_eval, dtype=float))
    outpatient_tr = train.get('number_outpatient', np.zeros(n_train, dtype=float))
    outpatient_ev = eval_cols.get('number_outpatient', np.zeros(n_eval, dtype=float))

    total_prior_tr = inpatient_tr + emergency_tr + outpatient_tr
    total_prior_ev = inpatient_ev + emergency_ev + outpatient_ev

    train_feat['total_prior_visits'] = total_prior_tr
    eval_feat['total_prior_visits'] = total_prior_ev
    train_feat['inpatient_visit_ratio'] = inpatient_tr / (total_prior_tr + 1.0)
    eval_feat['inpatient_visit_ratio'] = inpatient_ev / (total_prior_ev + 1.0)
    train_feat['emergency_visit_ratio'] = emergency_tr / (total_prior_tr + 1.0)
    eval_feat['emergency_visit_ratio'] = emergency_ev / (total_prior_ev + 1.0)

    train_feat['inpatient_x_los'] = inpatient_tr * train['time_in_hospital']
    eval_feat['inpatient_x_los'] = inpatient_ev * eval_cols['time_in_hospital']
    train_feat['inpatient_x_meds'] = inpatient_tr * train['num_medications']
    eval_feat['inpatient_x_meds'] = inpatient_ev * eval_cols['num_medications']
    train_feat['inpatient_x_diags'] = inpatient_tr * train['number_diagnoses']
    eval_feat['inpatient_x_diags'] = inpatient_ev * eval_cols['number_diagnoses']
    train_feat['inpatient_x_cci'] = inpatient_tr * cci_tr
    eval_feat['inpatient_x_cci'] = inpatient_ev * cci_ev
    extra_numeric.extend([
        'total_prior_visits', 'inpatient_visit_ratio', 'emergency_visit_ratio',
        'inpatient_x_los', 'inpatient_x_meds', 'inpatient_x_diags', 'inpatient_x_cci'
    ])
    
    # 7. Patient demography, lab indicators, ratios, and frailty
    def add_frailty(data_dict, target_dict, age_array):
        meds = data_dict.get('num_medications', np.zeros_like(age_array, dtype=float))
        diags = data_dict.get('number_diagnoses', np.zeros_like(age_array, dtype=float))
        target_dict['frailty_index'] = (age_array / 100.0) * (meds + 2.0 * diags)

    add_frailty(train, train_feat, train_feat['age_num'])
    add_frailty(eval_cols, eval_feat, eval_feat['age_num'])
    extra_numeric.append('frailty_index')
    
    def parse_weight(weight_col):
        res = []
        for x in weight_col:
            if x is None:
                res.append(np.nan)
            else:
                try:
                    parts = str(x).replace('[', '').replace(')', '').replace(']', '').split('-')
                    val = 0.5 * (float(parts[0]) + float(parts[1]))
                    res.append(val)
                except Exception:
                    res.append(np.nan)
        return np.array(res, dtype=float)

    train_feat['weight_num'] = parse_weight(train.get('weight', [None] * n_train))
    eval_feat['weight_num'] = parse_weight(eval_cols.get('weight', [None] * n_eval))
    extra_numeric.append('weight_num')

    train_feat['weight_missing'] = np.isnan(train_feat['weight_num']).astype(float)
    eval_feat['weight_missing'] = np.isnan(eval_feat['weight_num']).astype(float)
    extra_numeric.append('weight_missing')
    
    train_feat['gender_female'] = (train.get('gender', np.array(['Unknown'] * n_train)) == 'Female').astype(float)
    eval_feat['gender_female'] = (eval_cols.get('gender', np.array(['Unknown'] * n_eval)) == 'Female').astype(float)
    extra_numeric.append('gender_female')
    
    def add_lab_features(data_dict, target_dict):
        n_r = len(data_dict['time_in_hospital'])
        glu = data_dict.get('max_glu_serum', np.array(['None'] * n_r))
        a1c = data_dict.get('A1Cresult', np.array(['None'] * n_r))
        
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
    
    return train_feat, eval_feat, extra_numeric, extra_categorical


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import HistGradientBoostingClassifier
    from scipy.stats import rankdata

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
        ),
        lgb.LGBMClassifier(
            n_estimators=170,
            learning_rate=0.016,
            num_leaves=48,
            min_child_samples=130,
            subsample=0.65,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=0.5,
            cat_smooth=40.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        )
    ]

    # Diverse XGBoost Ensemble
    xgb_models = [
        xgb.XGBClassifier(
            n_estimators=170,
            max_depth=5,
            learning_rate=0.016,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_lambda=1.5,
            random_state=42,
            tree_method='hist',
            n_jobs=1,
            eval_metric='logloss'
        ),
        xgb.XGBClassifier(
            n_estimators=210,
            max_depth=4,
            learning_rate=0.012,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_lambda=2.5,
            random_state=101,
            tree_method='hist',
            n_jobs=1,
            eval_metric='logloss'
        ),
        xgb.XGBClassifier(
            n_estimators=190,
            max_depth=3,
            learning_rate=0.018,
            subsample=0.75,
            colsample_bytree=0.65,
            reg_lambda=1.0,
            random_state=2024,
            tree_method='hist',
            n_jobs=1,
            eval_metric='logloss'
        )
    ]

    # Diverse HistGradientBoosting Ensemble
    hgb_models = [
        HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.015,
            max_leaf_nodes=31,
            min_samples_leaf=150,
            l2_regularization=0.1,
            random_state=42
        ),
        HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.012,
            max_leaf_nodes=24,
            min_samples_leaf=200,
            l2_regularization=0.5,
            random_state=101
        ),
        HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.016,
            max_leaf_nodes=18,
            min_samples_leaf=250,
            l2_regularization=1.0,
            random_state=777
        )
    ]

    # Fit & predict LightGBM models
    lgb_preds = np.zeros(len(x_eval))
    for model in lgb_models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        lgb_preds += model.predict_proba(x_eval)[:, 1]
    lgb_preds /= len(lgb_models)
    
    # Fit & predict XGBoost models
    xgb_preds = np.zeros(len(x_eval))
    for model in xgb_models:
        model.fit(x_train, y_train)
        xgb_preds += model.predict_proba(x_eval)[:, 1]
    xgb_preds /= len(xgb_models)
    
    # Fit & predict HistGradientBoosting models
    hgb_preds = np.zeros(len(x_eval))
    for model in hgb_models:
        model.fit(x_train, y_train)
        hgb_preds += model.predict_proba(x_eval)[:, 1]
    hgb_preds /= len(hgb_models)
    
    # Rank Averaging across Heterogeneous GBDT Families
    n_eval_rows = len(x_eval)
    lgb_ranks = rankdata(lgb_preds) / n_eval_rows
    xgb_ranks = rankdata(xgb_preds) / n_eval_rows
    hgb_ranks = rankdata(hgb_preds) / n_eval_rows
    
    preds = 0.50 * lgb_ranks + 0.30 * xgb_ranks + 0.20 * hgb_ranks
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
