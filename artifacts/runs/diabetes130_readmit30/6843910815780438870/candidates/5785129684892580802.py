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
    n_train_rows = len(train['time_in_hospital'])
    n_eval_rows = len(eval_cols['time_in_hospital'])
    
    # 1. Comorbidities & Weighted Charlson Comorbidity Index
    train_cm = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    eval_cm = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    
    for k in train_cm.keys():
        train_feat[f'cm_{k}'] = train_cm[k]
        eval_feat[f'cm_{k}'] = eval_cm[k]
        extra_numeric.append(f'cm_{k}')
        
    def calc_charlson(cm):
        return (
            cm['cm_mi'] + cm['cm_chf'] + cm['cm_pvd'] + cm['cm_cerebrovascular'] +
            cm['cm_dementia'] + cm['cm_pulmonary'] + cm['cm_rheumatic'] +
            cm['cm_peptic_ulcer'] + 2.0 * cm['cm_renal'] + 2.0 * cm['cm_cancer'] +
            2.0 * cm['cm_diabetes_complicated'] + 2.0 * cm['cm_hemiplegia'] +
            3.0 * cm['cm_liver'] + 6.0 * cm['cm_metastatic'] + 6.0 * cm['cm_hiv']
        )
    train_feat['charlson_index'] = calc_charlson(train_cm)
    eval_feat['charlson_index'] = calc_charlson(eval_cm)
    extra_numeric.append('charlson_index')
        
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
        
        # 3-digit ICD9 code grouping (e.g., '250', '428', '585')
        def get_diag_3d(arr):
            res = []
            for x in arr:
                if x is None:
                    res.append(None)
                else:
                    s = str(x)
                    res.append(s.split('.')[0] if '.' in s else s)
            return np.array(res, dtype=object)
        
        col_3d = f'{col}_3d'
        train_feat[col_3d] = get_diag_3d(train[col])
        eval_feat[col_3d] = get_diag_3d(eval_cols[col])
        extra_categorical.append(col_3d)
        
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
        
    # High-impact disease indicator flags
    def has_prefix_any_diag(d1, d2, d3, prefixes):
        res = np.zeros(len(d1), dtype=np.float64)
        pref_tuple = tuple(prefixes)
        for col in (d1, d2, d3):
            for idx, val in enumerate(col):
                if val is not None and str(val).startswith(pref_tuple):
                    res[idx] = 1.0
        return res

    disease_flags = {
        'diag_has_hf': ('428',),
        'diag_has_ckd': ('585', '586'),
        'diag_has_htn': ('401', '402', '403', '404', '405'),
        'diag_has_pneumonia': ('480', '481', '482', '486'),
        'diag_has_fluid_elect': ('276',)
    }
    for flag_name, prefixes in disease_flags.items():
        train_feat[flag_name] = has_prefix_any_diag(train['diag_1'], train['diag_2'], train['diag_3'], prefixes)
        eval_feat[flag_name] = has_prefix_any_diag(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'], prefixes)
        extra_numeric.append(flag_name)

    train_feat['diag_primary_diabetes'] = (train_feat['diag_1_grp'] == 'diabetes').astype(float)
    eval_feat['diag_primary_diabetes'] = (eval_feat['diag_1_grp'] == 'diabetes').astype(float)
    extra_numeric.append('diag_primary_diabetes')

    train_feat['diag_primary_circulatory'] = (train_feat['diag_1_grp'] == 'circulatory').astype(float)
    eval_feat['diag_primary_circulatory'] = (eval_feat['diag_1_grp'] == 'circulatory').astype(float)
    extra_numeric.append('diag_primary_circulatory')
        
    # 3. Admission/Discharge interaction and flags
    train_feat['dis_adm'] = np.array([
        f"{d}_{a}" for d, a in zip(train['discharge_disposition_id'], train['admission_source_id'])
    ], dtype=object)
    eval_feat['dis_adm'] = np.array([
        f"{d}_{a}" for d, a in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])
    ], dtype=object)
    extra_categorical.append('dis_adm')

    train_feat['adm_type_source'] = np.array([
        f"{t}_{s}" for t, s in zip(train['admission_type_id'], train['admission_source_id'])
    ], dtype=object)
    eval_feat['adm_type_source'] = np.array([
        f"{t}_{s}" for t, s in zip(eval_cols['admission_type_id'], eval_cols['admission_source_id'])
    ], dtype=object)
    extra_categorical.append('adm_type_source')
    
    # Discharge & Admission indicators
    def add_adm_disch_indicators(data_dict, target_dict):
        disch = data_dict.get('discharge_disposition_id', np.array([None]*len(data_dict['time_in_hospital'])))
        adm_src = data_dict.get('admission_source_id', np.array([None]*len(data_dict['time_in_hospital'])))
        adm_type = data_dict.get('admission_type_id', np.array([None]*len(data_dict['time_in_hospital'])))
        
        target_dict['disch_home'] = (disch == '1').astype(float)
        target_dict['disch_snf'] = (disch == '3').astype(float)
        target_dict['disch_home_health'] = (disch == '6').astype(float)
        target_dict['disch_ama'] = (disch == '7').astype(float)
        target_dict['disch_transfer'] = np.isin(disch, ['2', '4', '5', '18', '22', '23', '24', '27', '28']).astype(float)
        
        target_dict['adm_is_emergency'] = ((adm_type == '1') | (adm_src == '7')).astype(float)
        target_dict['adm_is_elective'] = (adm_type == '3').astype(float)
        target_dict['adm_is_urgent'] = (adm_type == '2').astype(float)
        target_dict['adm_is_referral'] = np.isin(adm_src, ['1', '2', '3']).astype(float)

    add_adm_disch_indicators(train, train_feat)
    add_adm_disch_indicators(eval_cols, eval_feat)
    extra_numeric.extend([
        'disch_home', 'disch_snf', 'disch_home_health', 'disch_ama', 'disch_transfer',
        'adm_is_emergency', 'adm_is_elective', 'adm_is_urgent', 'adm_is_referral'
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
        
    # 4. Target encoding of high-cardinality categorical variables
    high_card_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_3d', 'diag_2_3d', 'diag_3_3d',
        'diag_1_grp', 'medical_specialty', 
        'discharge_disposition_id', 'admission_source_id', 'admission_type_id',
        'payer_code', 'dis_adm', 'adm_type_source', 'med_combo', 'change_med'
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
    oral_med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide'
    ]
    existing_med_cols = [c for c in med_cols if c in train]
    existing_oral_cols = [c for c in oral_med_cols if c in train]
    
    def get_med_features(data_dict):
        n_rows = len(data_dict['time_in_hospital'])
        num_up = np.zeros(n_rows, dtype=np.float64)
        num_down = np.zeros(n_rows, dtype=np.float64)
        num_steady = np.zeros(n_rows, dtype=np.float64)
        num_oral = np.zeros(n_rows, dtype=np.float64)
        for col in existing_med_cols:
            arr = data_dict[col]
            num_up += (arr == 'Up').astype(np.float64)
            num_down += (arr == 'Down').astype(np.float64)
            num_steady += (arr == 'Steady').astype(np.float64)

        for col in existing_oral_cols:
            num_oral += (data_dict[col] != 'No').astype(np.float64)
            
        num_active = num_up + num_down + num_steady
        num_change = num_up + num_down
        
        # Insulin specific
        insulin_arr = data_dict.get('insulin', np.array(['No'] * n_rows))
        insulin_up = (insulin_arr == 'Up').astype(np.float64)
        insulin_down = (insulin_arr == 'Down').astype(np.float64)
        insulin_steady = (insulin_arr == 'Steady').astype(np.float64)
        insulin_active = (insulin_arr != 'No').astype(np.float64)
        insulin_change = ((insulin_arr == 'Up') | (insulin_arr == 'Down')).astype(np.float64)

        metformin_arr = data_dict.get('metformin', np.array(['No'] * n_rows))
        on_metformin = (metformin_arr != 'No').astype(np.float64)
        insulin_only = (insulin_active > 0) & (num_oral == 0)
        insulin_plus_oral = (insulin_active > 0) & (num_oral > 0)
        metformin_and_insulin = (on_metformin > 0) & (insulin_active > 0)
        
        return {
            'med_num_up': num_up,
            'med_num_down': num_down,
            'med_num_steady': num_steady,
            'med_num_active': num_active,
            'med_num_change': num_change,
            'med_num_oral': num_oral,
            'med_dose_decrease': (num_down > 0).astype(np.float64),
            'insulin_up': insulin_up,
            'insulin_down': insulin_down,
            'insulin_steady': insulin_steady,
            'insulin_active': insulin_active,
            'insulin_change': insulin_change,
            'on_metformin': on_metformin,
            'insulin_only': insulin_only.astype(np.float64),
            'insulin_plus_oral': insulin_plus_oral.astype(np.float64),
            'metformin_and_insulin': metformin_and_insulin.astype(np.float64)
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
        total_prior_visits = outpatient + emergency + inpatient
        prior_acute_visits = emergency + inpatient
        
        inpatient_ratio = inpatient / (total_prior_visits + 1.0)
        acute_ratio = prior_acute_visits / (total_prior_visits + 1.0)
        log_inpatient = np.log1p(inpatient)
        log_emergency = np.log1p(emergency)
        log_outpatient = np.log1p(outpatient)
        log_total_visits = np.log1p(total_prior_visits)
        
        lab_proc = data_dict.get('num_lab_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        proc = data_dict.get('num_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        meds = data_dict.get('num_medications', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        diags = data_dict.get('number_diagnoses', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        los = data_dict.get('time_in_hospital', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        
        severity_index = lab_proc + 2.0 * proc + meds + 5.0 * diags
        los_med_ratio = los / (meds + 1.0)
        los_per_diag = los / (diags + 1.0)
        
        target_dict['util_index'] = util_index
        target_dict['util_any'] = util_any
        target_dict['total_prior_visits'] = total_prior_visits
        target_dict['prior_acute_visits'] = prior_acute_visits
        target_dict['inpatient_ratio'] = inpatient_ratio
        target_dict['acute_ratio'] = acute_ratio
        target_dict['log_inpatient'] = log_inpatient
        target_dict['log_emergency'] = log_emergency
        target_dict['log_outpatient'] = log_outpatient
        target_dict['log_total_visits'] = log_total_visits
        
        target_dict['severity_index'] = severity_index
        target_dict['los_med_ratio'] = los_med_ratio
        target_dict['los_per_diag'] = los_per_diag
        
        target_dict['has_prior_inpatient'] = (inpatient > 0).astype(float)
        target_dict['has_prior_emergency'] = (emergency > 0).astype(float)
        target_dict['has_prior_outpatient'] = (outpatient > 0).astype(float)
        target_dict['frequent_inpatient'] = (inpatient >= 2).astype(float)
        target_dict['high_inpatient'] = (inpatient >= 3).astype(float)
        target_dict['frequent_emergency'] = (emergency >= 2).astype(float)
        target_dict['polypharmacy'] = (meds >= 15).astype(float)
        target_dict['severe_polypharmacy'] = (meds >= 20).astype(float)
        
    add_custom_numeric(train, train_feat)
    add_custom_numeric(eval_cols, eval_feat)
    extra_numeric.extend([
        'util_index', 'util_any', 'total_prior_visits', 'prior_acute_visits',
        'inpatient_ratio', 'acute_ratio', 'log_inpatient', 'log_emergency',
        'log_outpatient', 'log_total_visits', 'severity_index', 'los_med_ratio',
        'los_per_diag', 'has_prior_inpatient', 'has_prior_emergency',
        'has_prior_outpatient', 'frequent_inpatient', 'high_inpatient',
        'frequent_emergency', 'polypharmacy', 'severe_polypharmacy'
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
        
    train_feat['age_num'] = parse_age(train.get('age', [None]*n_train_rows))
    eval_feat['age_num'] = parse_age(eval_cols.get('age', [None]*n_eval_rows))
    extra_numeric.append('age_num')
    
    def add_frailty(data_dict, target_dict, age_array):
        meds = data_dict.get('num_medications', np.zeros_like(age_array, dtype=float))
        diags = data_dict.get('number_diagnoses', np.zeros_like(age_array, dtype=float))
        inpatient = data_dict.get('number_inpatient', np.zeros_like(age_array, dtype=float))
        target_dict['frailty_index'] = (age_array / 100.0) * (meds + 2.0 * diags)
        target_dict['age_times_inpatient'] = age_array * inpatient
        target_dict['elderly'] = (age_array >= 70.0).astype(float)

    add_frailty(train, train_feat, train_feat['age_num'])
    add_frailty(eval_cols, eval_feat, eval_feat['age_num'])
    extra_numeric.extend(['frailty_index', 'age_times_inpatient', 'elderly'])
    
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

    train_feat['weight_num'] = parse_weight(train.get('weight', [None]*n_train_rows))
    eval_feat['weight_num'] = parse_weight(eval_cols.get('weight', [None]*n_eval_rows))
    extra_numeric.append('weight_num')

    train_feat['weight_missing'] = np.isnan(train_feat['weight_num']).astype(float)
    eval_feat['weight_missing'] = np.isnan(eval_feat['weight_num']).astype(float)
    extra_numeric.append('weight_missing')
    
    train_feat['gender_female'] = (train.get('gender', np.array(['Unknown']*n_train_rows)) == 'Female').astype(float)
    eval_feat['gender_female'] = (eval_cols.get('gender', np.array(['Unknown']*n_eval_rows)) == 'Female').astype(float)
    extra_numeric.append('gender_female')
    
    # 8. Strack et al. (2014) Findings: Lab results & medication change dynamics
    def add_lab_features(data_dict, target_dict):
        n_rows = len(data_dict['time_in_hospital'])
        glu = data_dict.get('max_glu_serum', np.array(['None'] * n_rows))
        a1c = data_dict.get('A1Cresult', np.array(['None'] * n_rows))
        ch = data_dict.get('change', np.array(['No'] * n_rows))
        ins = data_dict.get('insulin', np.array(['No'] * n_rows))
        inpat = data_dict.get('number_inpatient', np.zeros(n_rows, dtype=float))
        
        target_dict['glu_high'] = np.isin(glu, ['>200', '>300']).astype(float)
        target_dict['glu_tested'] = (glu != 'None').astype(float)
        target_dict['a1c_high'] = np.isin(a1c, ['>7', '>8']).astype(float)
        target_dict['a1c_tested'] = (a1c != 'None').astype(float)
        
        target_dict['a1c_high_no_change'] = ((a1c == '>8') & (ch == 'No')).astype(float)
        target_dict['a1c_high_med_change'] = ((a1c == '>8') & (ch == 'Ch')).astype(float)
        target_dict['insulin_and_high_a1c'] = ((ins != 'No') & np.isin(a1c, ['>7', '>8'])).astype(float)
        target_dict['insulin_and_prior_inpatient'] = ((ins != 'No') & (inpat > 0)).astype(float)

    add_lab_features(train, train_feat)
    add_lab_features(eval_cols, eval_feat)
    extra_numeric.extend([
        'glu_high', 'glu_tested', 'a1c_high', 'a1c_tested',
        'a1c_high_no_change', 'a1c_high_med_change',
        'insulin_and_high_a1c', 'insulin_and_prior_inpatient'
    ])
    
    def add_ratios(data_dict, target_dict):
        los = data_dict.get('time_in_hospital', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        lab = data_dict.get('num_lab_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        proc = data_dict.get('num_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        meds = data_dict.get('num_medications', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        
        target_dict['lab_per_day'] = lab / (los + 1.0)
        target_dict['proc_per_day'] = proc / (los + 1.0)
        target_dict['meds_per_day'] = meds / (los + 1.0)
        target_dict['high_los'] = (los >= 7).astype(float)
        target_dict['short_los'] = (los <= 2).astype(float)
        target_dict['complex_stay'] = ((los >= 5) & (proc >= 2) & (lab >= 40)).astype(float)

    add_ratios(train, train_feat)
    add_ratios(eval_cols, eval_feat)
    extra_numeric.extend(['lab_per_day', 'proc_per_day', 'meds_per_day', 'high_los', 'short_los', 'complex_stay'])
    
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

    train_feat, eval_feat, extra_numeric, extra_categorical = add_features(task, train, eval_cols, y_train)
    
    encoder = TabularEncoder(min_count=25)
    encoder.fit(task, train_feat, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)

    # Diverse Multi-Model LightGBM Ensemble (Weight: 0.50)
    lgb_models = [
        lgb.LGBMClassifier(
            n_estimators=200,
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
            n_estimators=230,
            learning_rate=0.012,
            num_leaves=24,
            min_child_samples=220,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=0.2,
            cat_smooth=80.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=180,
            learning_rate=0.018,
            num_leaves=38,
            min_child_samples=100,
            subsample=0.65,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=0.01,
            cat_smooth=30.0,
            random_state=2023,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.014,
            num_leaves=18,
            min_child_samples=200,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=0.5,
            cat_smooth=60.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=210,
            learning_rate=0.018,
            num_leaves=31,
            min_child_samples=120,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=0.1,
            extra_trees=True,
            cat_smooth=40.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        )
    ]

    # XGBoost Ensemble (Weight: 0.30)
    xgb_models = [
        xgb.XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.015,
            subsample=0.7,
            colsample_bytree=0.6,
            min_child_weight=10,
            reg_lambda=1.5,
            random_state=42,
            tree_method='hist',
            n_jobs=1,
            eval_metric='logloss'
        ),
        xgb.XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.012,
            subsample=0.8,
            colsample_bytree=0.5,
            min_child_weight=15,
            reg_lambda=3.0,
            random_state=101,
            tree_method='hist',
            n_jobs=1,
            eval_metric='logloss'
        )
    ]

    # HistGradientBoosting Ensemble (Weight: 0.20)
    hgb_models = [
        HistGradientBoostingClassifier(
            max_iter=170,
            learning_rate=0.015,
            max_leaf_nodes=31,
            min_samples_leaf=150,
            l2_regularization=0.2,
            random_state=42
        ),
        HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.012,
            max_leaf_nodes=24,
            min_samples_leaf=200,
            l2_regularization=1.0,
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
    
    # Weighted Blend
    preds = 0.50 * lgb_preds + 0.30 * xgb_preds + 0.20 * hgb_preds
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
