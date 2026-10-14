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
        
    # 2. ICD9 groups and 3-digit categories (categorical)
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

    # 3-digit ICD9 category grouping (standard CMS/epidemiology risk adjustment abstraction)
    def get_icd9_prefix(arr):
        res = []
        for val in arr:
            if val is None or not isinstance(val, str):
                res.append("missing")
            else:
                val = val.strip()
                if not val or val == "?":
                    res.append("missing")
                elif val[0] in ("V", "E"):
                    res.append(val[:3])
                else:
                    res.append(val.split(".")[0][:3])
        return np.array(res, dtype=object)

    train_feat['diag_1_3d'] = get_icd9_prefix(train['diag_1'])
    eval_feat['diag_1_3d'] = get_icd9_prefix(eval_cols['diag_1'])
    extra_categorical.append('diag_1_3d')

    train_feat['diag_2_3d'] = get_icd9_prefix(train['diag_2'])
    eval_feat['diag_2_3d'] = get_icd9_prefix(eval_cols['diag_2'])
    extra_categorical.append('diag_2_3d')

    train_feat['diag_pair_3d'] = np.array([
        f"{d1}_{d2}" for d1, d2 in zip(train_feat['diag_1_3d'], train_feat['diag_2_3d'])
    ], dtype=object)
    eval_feat['diag_pair_3d'] = np.array([
        f"{d1}_{d2}" for d1, d2 in zip(eval_feat['diag_1_3d'], eval_feat['diag_2_3d'])
    ], dtype=object)
    extra_categorical.append('diag_pair_3d')
        
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
        'diag_1', 'diag_2', 'diag_3', 'diag_1_3d', 'diag_2_3d', 'diag_pair_3d',
        'medical_specialty', 'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'payer_code', 'dis_adm', 'med_combo', 'change_med'
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

        insulin_dosage_change = insulin_up - insulin_down
        total_dosage_change = num_up - num_down

        metformin_arr = data_dict.get('metformin', np.array(['No'] * n_rows))
        on_metformin = (metformin_arr != 'No').astype(np.float64)
        metformin_and_insulin = on_metformin * insulin_active
        no_diabetes_med = (data_dict.get('diabetesMed', np.array(['Yes'] * n_rows)) == 'No').astype(np.float64)
        
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
            'insulin_dosage_change': insulin_dosage_change,
            'total_dosage_change': total_dosage_change,
            'on_metformin': on_metformin,
            'metformin_and_insulin': metformin_and_insulin,
            'no_diabetes_med': no_diabetes_med
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

        target_dict['diag_per_day'] = diags / (los + 1.0)
        target_dict['med_per_diag'] = meds / (diags + 1.0)
        target_dict['lab_per_proc'] = lab_proc / (proc + 1.0)
        target_dict['total_clinical_actions'] = lab_proc + proc + meds
        
    add_custom_numeric(train, train_feat)
    add_custom_numeric(eval_cols, eval_feat)
    extra_numeric.extend([
        'util_index', 'util_any', 'severity_index', 'los_med_ratio',
        'has_prior_inpatient', 'has_prior_emergency', 'has_prior_outpatient', 'frequent_inpatient', 'frequent_emergency',
        'diag_per_day', 'med_per_diag', 'lab_per_proc', 'total_clinical_actions'
    ])
    
    # 7. Patient demography, lab indicators, ratios, and frailty
    def parse_age(age_col):
        res = []
        for x in age_col:
            if x is None:
                res.append(55.0)
            else:
                try:
                    parts = x.replace('[', '').replace(')', '').replace(']', '').split('-')
                    val = 0.5 * (float(parts[0]) + float(parts[1]))
                    res.append(val)
                except:
                    res.append(55.0)
        return np.array(res, dtype=float)
        
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
        res = []
        for x in weight_col:
            if x is None:
                res.append(np.nan)
            else:
                try:
                    parts = x.replace('[', '').replace(')', '').replace(']', '').split('-')
                    val = 0.5 * (float(parts[0]) + float(parts[1]))
                    res.append(val)
                except:
                    res.append(np.nan)
        return np.array(res, dtype=float)

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
        chg = data_dict.get('change', np.array(['No'] * len(data_dict['time_in_hospital'])))
        
        target_dict['glu_high'] = np.isin(glu, ['>200', '>300']).astype(float)
        target_dict['glu_tested'] = (glu != 'None').astype(float)
        target_dict['a1c_high'] = np.isin(a1c, ['>7', '>8']).astype(float)
        target_dict['a1c_tested'] = (a1c != 'None').astype(float)
        # Clinical therapeutic inertia: poorly controlled HbA1c without medication change
        target_dict['a1c_inertia'] = ((target_dict['a1c_high'] > 0) & (chg == 'No')).astype(float)

    add_lab_features(train, train_feat)
    add_lab_features(eval_cols, eval_feat)
    extra_numeric.extend(['glu_high', 'glu_tested', 'a1c_high', 'a1c_tested', 'a1c_inertia'])
    
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

    train_feat['discharge_home'] = (train['discharge_disposition_id'] == '1').astype(float)
    eval_feat['discharge_home'] = (eval_cols['discharge_disposition_id'] == '1').astype(float)
    extra_numeric.append('discharge_home')

    train_feat['emergency_admission'] = np.isin(train.get('admission_type_id', []), ['1', '2']).astype(float)
    eval_feat['emergency_admission'] = np.isin(eval_cols.get('admission_type_id', []), ['1', '2']).astype(float)
    extra_numeric.append('emergency_admission')

    train_feat['er_admission_source'] = (train.get('admission_source_id', []) == '7').astype(float)
    eval_feat['er_admission_source'] = (eval_cols.get('admission_source_id', []) == '7').astype(float)
    extra_numeric.append('er_admission_source')

    # Primary diagnosis key risk categories (CHF: 428, Renal: 584-586, Resp: 480-486, Diabetes: 250, COPD: 491-496, Fluid/Electrolyte: 276)
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

    train_feat['primary_copd'] = check_diag_prefix(train['diag_1'], ['491', '492', '496'])
    eval_feat['primary_copd'] = check_diag_prefix(eval_cols['diag_1'], ['491', '492', '496'])
    extra_numeric.append('primary_copd')

    train_feat['primary_fluid'] = check_diag_prefix(train['diag_1'], ['276'])
    eval_feat['primary_fluid'] = check_diag_prefix(eval_cols['diag_1'], ['276'])
    extra_numeric.append('primary_fluid')

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
    train_feat['emergent_util_ratio'] = (inpatient_tr + emergency_tr) / (total_prior_tr + 1.0)
    eval_feat['emergent_util_ratio'] = (inpatient_ev + emergency_ev) / (total_prior_ev + 1.0)
    extra_numeric.extend(['log_inpatient', 'log_emergency', 'total_prior_visits', 'inpatient_ratio', 'emergent_util_ratio'])

    # 9. Decoupled Multi-Modal Clinical Tensor Decomposition (Tucker / CP Factorization)
    from sklearn.decomposition import TruncatedSVD

    n_train_rows = len(train['time_in_hospital'])
    n_eval_rows = len(eval_cols['time_in_hospital'])

    # Mode M: Morbidity & Disease Subspace (Comorbidities, chapters, primary clinical categories)
    morb_cols = [
        'cm_mi', 'cm_chf', 'cm_pvd', 'cm_cerebrovascular', 'cm_dementia', 
        'cm_pulmonary', 'cm_rheumatic', 'cm_peptic_ulcer', 'cm_liver', 
        'cm_hemiplegia', 'cm_renal', 'cm_cancer', 'cm_metastatic', 'cm_hiv', 
        'cm_diabetes_complicated'
    ]
    morb_tr_list = [train_feat[f'cm_{k}'] for k in morb_cols]
    morb_ev_list = [eval_feat[f'cm_{k}'] for k in morb_cols]
    for grp in groups_list:
        morb_tr_list.append(train_feat[f'grp_cnt_{grp}'])
        morb_ev_list.append(eval_feat[f'grp_cnt_{grp}'])
    for p_diag in ['primary_chf', 'primary_renal', 'primary_resp_inf', 'primary_diabetes', 'primary_copd', 'primary_fluid']:
        morb_tr_list.append(train_feat[p_diag])
        morb_ev_list.append(eval_feat[p_diag])

    mat_m_tr = np.nan_to_num(np.column_stack(morb_tr_list), nan=0.0)
    mat_m_ev = np.nan_to_num(np.column_stack(morb_ev_list), nan=0.0)
    mean_m = np.mean(mat_m_tr, axis=0)
    std_m = np.std(mat_m_tr, axis=0) + 1e-6
    mat_m_tr_norm = (mat_m_tr - mean_m) / std_m
    mat_m_ev_norm = (mat_m_ev - mean_m) / std_m

    svd_m = TruncatedSVD(n_components=4, random_state=42)
    u_tr = svd_m.fit_transform(mat_m_tr_norm)
    u_ev = svd_m.transform(mat_m_ev_norm)

    for i in range(4):
        train_feat[f'latent_morb_{i}'] = u_tr[:, i]
        eval_feat[f'latent_morb_{i}'] = u_ev[:, i]
        extra_numeric.append(f'latent_morb_{i}')

    # Mode T: Pharmacotherapy Subspace (Medication intensities, dosage directions, combinations)
    med_intensity_map = {'No': 0.0, 'Steady': 1.0, 'Down': 1.5, 'Up': 2.0}
    ther_tr_list = []
    ther_ev_list = []
    for m in existing_med_cols:
        arr_tr = train.get(m, np.array(['No'] * n_train_rows))
        arr_ev = eval_cols.get(m, np.array(['No'] * n_eval_rows))
        ther_tr_list.append(np.array([med_intensity_map.get(x, 0.0) for x in arr_tr], dtype=np.float64))
        ther_ev_list.append(np.array([med_intensity_map.get(x, 0.0) for x in arr_ev], dtype=np.float64))
    for t_col in ['med_num_active', 'med_num_change', 'insulin_active', 'insulin_up', 'insulin_down', 'insulin_and_oral']:
        ther_tr_list.append(train_feat[t_col])
        ther_ev_list.append(eval_feat[t_col])

    mat_t_tr = np.nan_to_num(np.column_stack(ther_tr_list), nan=0.0)
    mat_t_ev = np.nan_to_num(np.column_stack(ther_ev_list), nan=0.0)
    mean_t = np.mean(mat_t_tr, axis=0)
    std_t = np.std(mat_t_tr, axis=0) + 1e-6
    mat_t_tr_norm = (mat_t_tr - mean_t) / std_t
    mat_t_ev_norm = (mat_t_ev - mean_t) / std_t

    svd_t = TruncatedSVD(n_components=3, random_state=42)
    v_tr = svd_t.fit_transform(mat_t_tr_norm)
    v_ev = svd_t.transform(mat_t_ev_norm)

    for j in range(3):
        train_feat[f'latent_ther_{j}'] = v_tr[:, j]
        eval_feat[f'latent_ther_{j}'] = v_ev[:, j]
        extra_numeric.append(f'latent_ther_{j}')

    # Mode U: Utilization & Acuity Subspace
    util_cols = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_diagnoses',
        'number_inpatient', 'number_emergency', 'number_outpatient', 'total_prior_visits',
        'log_inpatient', 'log_emergency', 'inpatient_ratio', 'emergent_util_ratio', 'age_num', 'frailty_index'
    ]
    mat_u_tr = np.nan_to_num(np.column_stack([train_feat[c] for c in util_cols]), nan=0.0)
    mat_u_ev = np.nan_to_num(np.column_stack([eval_feat[c] for c in util_cols]), nan=0.0)
    mean_u = np.mean(mat_u_tr, axis=0)
    std_u = np.std(mat_u_tr, axis=0) + 1e-6
    mat_u_tr_norm = (mat_u_tr - mean_u) / std_u
    mat_u_ev_norm = (mat_u_ev - mean_u) / std_u

    svd_u = TruncatedSVD(n_components=3, random_state=42)
    w_tr = svd_u.fit_transform(mat_u_tr_norm)
    w_ev = svd_u.transform(mat_u_ev_norm)

    for k in range(3):
        train_feat[f'latent_util_{k}'] = w_tr[:, k]
        eval_feat[f'latent_util_{k}'] = w_ev[:, k]
        extra_numeric.append(f'latent_util_{k}')

    # Multilinear CP / Tucker Tensor Cross-Factor Interactions
    # Trilinear CP component: Morbidity x Therapy x Utilization
    train_feat['cp_tensor_000'] = u_tr[:, 0] * v_tr[:, 0] * w_tr[:, 0]
    eval_feat['cp_tensor_000'] = u_ev[:, 0] * v_ev[:, 0] * w_ev[:, 0]
    extra_numeric.append('cp_tensor_000')

    train_feat['cp_tensor_100'] = u_tr[:, 1] * v_tr[:, 0] * w_tr[:, 0]
    eval_feat['cp_tensor_100'] = u_ev[:, 1] * v_ev[:, 0] * w_ev[:, 0]
    extra_numeric.append('cp_tensor_100')

    # Bilinear Mode M x Mode T: Phenotype-Treatment Alignment
    train_feat['cp_m0_t0'] = u_tr[:, 0] * v_tr[:, 0]
    eval_feat['cp_m0_t0'] = u_ev[:, 0] * v_ev[:, 0]
    extra_numeric.append('cp_m0_t0')

    train_feat['cp_m1_t0'] = u_tr[:, 1] * v_tr[:, 0]
    eval_feat['cp_m1_t0'] = u_ev[:, 1] * v_ev[:, 0]
    extra_numeric.append('cp_m1_t0')

    train_feat['cp_m1_t1'] = u_tr[:, 1] * v_tr[:, 1]
    eval_feat['cp_m1_t1'] = u_ev[:, 1] * v_ev[:, 1]
    extra_numeric.append('cp_m1_t1')

    # Bilinear Mode M x Mode U: Phenotype-Utilization Coupling
    train_feat['cp_m0_u0'] = u_tr[:, 0] * w_tr[:, 0]
    eval_feat['cp_m0_u0'] = u_ev[:, 0] * w_ev[:, 0]
    extra_numeric.append('cp_m0_u0')

    train_feat['cp_m0_u1'] = u_tr[:, 0] * w_tr[:, 1]
    eval_feat['cp_m0_u1'] = u_ev[:, 0] * w_ev[:, 1]
    extra_numeric.append('cp_m0_u1')

    train_feat['cp_m1_u0'] = u_tr[:, 1] * w_tr[:, 0]
    eval_feat['cp_m1_u0'] = u_ev[:, 1] * w_ev[:, 0]
    extra_numeric.append('cp_m1_u0')

    # Bilinear Mode T x Mode U: Therapy-Acuity Coupling
    train_feat['cp_t0_u0'] = v_tr[:, 0] * w_tr[:, 0]
    eval_feat['cp_t0_u0'] = v_ev[:, 0] * w_ev[:, 0]
    extra_numeric.append('cp_t0_u0')

    train_feat['cp_t1_u0'] = v_tr[:, 1] * w_tr[:, 0]
    eval_feat['cp_t1_u0'] = v_ev[:, 1] * w_ev[:, 0]
    extra_numeric.append('cp_t1_u0')

    train_feat['cp_t2_u1'] = v_tr[:, 2] * w_tr[:, 1]
    eval_feat['cp_t2_u1'] = v_ev[:, 2] * w_ev[:, 1]
    extra_numeric.append('cp_t2_u1')
    
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

    train_feat, eval_feat, extra_numeric, extra_categorical = add_features(task, train, eval_cols, y_train)
    
    encoder = TabularEncoder(min_count=30)
    encoder.fit(task, train_feat, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)

    # Diverse Multi-Model LightGBM Ensemble (Weight: 0.50)
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

    # XGBoost Ensemble (Weight: 0.30)
    xgb_models = [
        xgb.XGBClassifier(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.018,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_lambda=1.0,
            min_child_weight=25.0,
            random_state=42,
            tree_method='hist',
            n_jobs=1,
            eval_metric='logloss'
        ),
        xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.012,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_lambda=2.0,
            min_child_weight=30.0,
            random_state=101,
            tree_method='hist',
            n_jobs=1,
            eval_metric='logloss'
        ),
        xgb.XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.015,
            subsample=0.75,
            colsample_bytree=0.55,
            reg_lambda=1.5,
            min_child_weight=20.0,
            random_state=777,
            tree_method='hist',
            n_jobs=1,
            eval_metric='logloss'
        )
    ]

    # HistGradientBoosting Ensemble (Weight: 0.20)
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
    
    # Rank-Preserving Quantile Consensus Blending (Optimizing ROC-AUC)
    from scipy.stats import rankdata

    lgb_rank = rankdata(lgb_preds) / len(lgb_preds)
    xgb_rank = rankdata(xgb_preds) / len(xgb_preds)
    hgb_rank = rankdata(hgb_preds) / len(hgb_preds)

    consensus_rank = 0.50 * lgb_rank + 0.30 * xgb_rank + 0.20 * hgb_rank
    raw_preds = 0.50 * lgb_preds + 0.30 * xgb_preds + 0.20 * hgb_preds

    # Project consensus ranking onto sorted empirical probability distribution
    order = np.argsort(consensus_rank)
    sorted_raw = np.sort(raw_preds)
    preds = np.empty(len(x_eval), dtype=np.float64)
    preds[order] = sorted_raw

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
