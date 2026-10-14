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
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler

    # Shallow copy to avoid mutating inputs directly
    train_feat = dict(train)
    eval_feat = dict(eval_cols)
    
    extra_numeric = []
    extra_categorical = []
    
    # 1. Comorbidities
    train_cm = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    eval_cm = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    
    for k in train_cm.keys():
        train_feat[k] = train_cm[k]
        eval_feat[k] = eval_cm[k]
        extra_numeric.append(k)
        
    # Standard Charlson Comorbidity Index
    train_feat['charlson_score'] = (
        train_cm['cm_mi'] + train_cm['cm_chf'] + train_cm['cm_pvd'] +
        train_cm['cm_cerebrovascular'] + train_cm['cm_dementia'] +
        train_cm['cm_pulmonary'] + train_cm['cm_rheumatic'] +
        train_cm['cm_peptic_ulcer'] + 2.0 * train_cm['cm_renal'] +
        2.0 * train_cm['cm_diabetes_complicated'] +
        2.0 * train_cm['cm_hemiplegia'] + 2.0 * train_cm['cm_cancer'] +
        3.0 * train_cm['cm_liver'] + 6.0 * train_cm['cm_metastatic'] +
        6.0 * train_cm['cm_hiv']
    )
    eval_feat['charlson_score'] = (
        eval_cm['cm_mi'] + eval_cm['cm_chf'] + eval_cm['cm_pvd'] +
        eval_cm['cm_cerebrovascular'] + eval_cm['cm_dementia'] +
        eval_cm['cm_pulmonary'] + eval_cm['cm_rheumatic'] +
        eval_cm['cm_peptic_ulcer'] + 2.0 * eval_cm['cm_renal'] +
        2.0 * eval_cm['cm_diabetes_complicated'] +
        2.0 * eval_cm['cm_hemiplegia'] + 2.0 * eval_cm['cm_cancer'] +
        3.0 * eval_cm['cm_liver'] + 6.0 * eval_cm['cm_metastatic'] +
        6.0 * eval_cm['cm_hiv']
    )
    extra_numeric.append('charlson_score')
        
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
        
    # Multi-system multimorbidity: count of unique non-missing ICD9 groups across 3 diagnoses
    train_feat['n_unique_diag_grps'] = np.array([
        len({g1, g2, g3} - {'missing'})
        for g1, g2, g3 in zip(train_feat['diag_1_grp'], train_feat['diag_2_grp'], train_feat['diag_3_grp'])
    ], dtype=float)
    eval_feat['n_unique_diag_grps'] = np.array([
        len({g1, g2, g3} - {'missing'})
        for g1, g2, g3 in zip(eval_feat['diag_1_grp'], eval_feat['diag_2_grp'], eval_feat['diag_3_grp'])
    ], dtype=float)
    extra_numeric.append('n_unique_diag_grps')

    # 3. Demographics and age parsing early for downstream interactions
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
        target_dict['elderly'] = (age_array >= 75.0).astype(float)

    add_frailty(train, train_feat, train_feat['age_num'])
    add_frailty(eval_cols, eval_feat, eval_feat['age_num'])
    extra_numeric.extend(['frailty_index', 'elderly'])
        
    # 4. Multi-categorical clinical interactions
    train_feat['dis_adm'] = np.array([
        f"{d}_{a}" for d, a in zip(train['discharge_disposition_id'], train['admission_source_id'])
    ], dtype=object)
    eval_feat['dis_adm'] = np.array([
        f"{d}_{a}" for d, a in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])
    ], dtype=object)
    extra_categorical.append('dis_adm')
    
    train_feat['med_combo'] = np.array([
        f"{m}_{i}_{g}" for m, i, g in zip(train['metformin'], train['insulin'], train['glipizide'])
    ], dtype=object)
    eval_feat['med_combo'] = np.array([
        f"{m}_{i}_{g}" for m, i, g in zip(eval_cols['metformin'], eval_cols['insulin'], eval_cols['glipizide'])
    ], dtype=object)
    extra_categorical.append('med_combo')
    
    train_feat['change_med'] = np.array([
        f"{c}_{d}" for c, d in zip(train['change'], train['diabetesMed'])
    ], dtype=object)
    eval_feat['change_med'] = np.array([
        f"{c}_{d}" for c, d in zip(eval_cols['change'], eval_cols['diabetesMed'])
    ], dtype=object)
    extra_categorical.append('change_med')
    
    train_feat['dis_diag1_grp'] = np.array([
        f"{d}_{g}" for d, g in zip(train['discharge_disposition_id'], train_feat['diag_1_grp'])
    ], dtype=object)
    eval_feat['dis_diag1_grp'] = np.array([
        f"{d}_{g}" for d, g in zip(eval_cols['discharge_disposition_id'], eval_feat['diag_1_grp'])
    ], dtype=object)
    extra_categorical.append('dis_diag1_grp')
    
    train_feat['adm_dis'] = np.array([
        f"{a}_{d}" for a, d in zip(train['admission_type_id'], train['discharge_disposition_id'])
    ], dtype=object)
    eval_feat['adm_dis'] = np.array([
        f"{a}_{d}" for a, d in zip(eval_cols['admission_type_id'], eval_cols['discharge_disposition_id'])
    ], dtype=object)
    extra_categorical.append('adm_dis')

    # Enhanced Clinical Interactions
    train_feat['dis_insulin'] = np.array([
        f"{d}_{i}" for d, i in zip(train['discharge_disposition_id'], train['insulin'])
    ], dtype=object)
    eval_feat['dis_insulin'] = np.array([
        f"{d}_{i}" for d, i in zip(eval_cols['discharge_disposition_id'], eval_cols['insulin'])
    ], dtype=object)
    extra_categorical.append('dis_insulin')

    train_feat['adm_diag1'] = np.array([
        f"{a}_{g}" for a, g in zip(train['admission_type_id'], train_feat['diag_1_grp'])
    ], dtype=object)
    eval_feat['adm_diag1'] = np.array([
        f"{a}_{g}" for a, g in zip(eval_cols['admission_type_id'], eval_feat['diag_1_grp'])
    ], dtype=object)
    extra_categorical.append('adm_diag1')

    train_feat['diag1_diag2'] = np.array([
        f"{g1}_{g2}" for g1, g2 in zip(train_feat['diag_1_grp'], train_feat['diag_2_grp'])
    ], dtype=object)
    eval_feat['diag1_diag2'] = np.array([
        f"{g1}_{g2}" for g1, g2 in zip(eval_feat['diag_1_grp'], eval_feat['diag_2_grp'])
    ], dtype=object)
    extra_categorical.append('diag1_diag2')

    age_train = train.get('age', np.array(['Unknown']*len(train['time_in_hospital'])))
    age_eval = eval_cols.get('age', np.array(['Unknown']*len(eval_cols['time_in_hospital'])))
    train_feat['age_dis'] = np.array([
        f"{ag}_{d}" for ag, d in zip(age_train, train['discharge_disposition_id'])
    ], dtype=object)
    eval_feat['age_dis'] = np.array([
        f"{ag}_{d}" for ag, d in zip(age_eval, eval_cols['discharge_disposition_id'])
    ], dtype=object)
    extra_categorical.append('age_dis')
        
    # 5. Target encoding of high-cardinality categorical variables
    high_card_cols = [
        'diag_1', 'diag_2', 'diag_3', 'medical_specialty', 
        'discharge_disposition_id', 'admission_source_id', 'admission_type_id',
        'payer_code', 'dis_adm', 'med_combo', 'change_med',
        'dis_diag1_grp', 'adm_dis', 'diag_1_grp',
        'dis_insulin', 'adm_diag1', 'diag1_diag2', 'age_dis'
    ]
    for col in high_card_cols:
        tr_enc, ev_enc = oof_target_encode(train_feat[col], y_train, eval_feat[col], n_folds=5, smoothing=20.0, seed=42)
        train_feat[f'{col}_te'] = tr_enc
        eval_feat[f'{col}_te'] = ev_enc
        extra_numeric.append(f'{col}_te')
        
    # 6. Medication features
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
        insulin_change = ((insulin_up + insulin_down) > 0).astype(np.float64)
        
        # Metformin & Sulfonylurea flags
        metformin_arr = data_dict.get('metformin', np.array(['No'] * n_rows))
        metformin_active = (metformin_arr != 'No').astype(np.float64)
        
        glip_arr = data_dict.get('glipizide', np.array(['No'] * n_rows))
        glyb_arr = data_dict.get('glyburide', np.array(['No'] * n_rows))
        glim_arr = data_dict.get('glimepiride', np.array(['No'] * n_rows))
        sulf_active = ((glip_arr != 'No') | (glyb_arr != 'No') | (glim_arr != 'No')).astype(np.float64)
        
        # Insulin plus oral therapy combination
        insulin_plus_oral = (insulin_active * (num_active > 1).astype(np.float64))
        
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
            'insulin_change': insulin_change,
            'metformin_active': metformin_active,
            'sulf_active': sulf_active,
            'insulin_plus_oral': insulin_plus_oral
        }
        
    train_med_feats = get_med_features(train)
    eval_med_feats = get_med_features(eval_cols)
    for k in train_med_feats.keys():
        train_feat[k] = train_med_feats[k]
        eval_feat[k] = eval_med_feats[k]
        extra_numeric.append(k)
        
    # 7. Utilisation and clinical severity interaction indices
    def add_custom_numeric(data_dict, target_dict):
        outpatient = data_dict.get('number_outpatient', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        emergency = data_dict.get('number_emergency', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        inpatient = data_dict.get('number_inpatient', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        
        total_visits = outpatient + emergency + inpatient
        acute_visits = emergency + inpatient
        util_index = outpatient + 2.0 * emergency + 3.0 * inpatient
        util_any = ((outpatient > 0) | (emergency > 0) | (inpatient > 0)).astype(np.float64)
        
        inpatient_ratio = inpatient / (total_visits + 1.0)
        emergency_ratio = emergency / (total_visits + 1.0)
        
        lab_proc = data_dict.get('num_lab_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        proc = data_dict.get('num_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        meds = data_dict.get('num_medications', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        diags = data_dict.get('number_diagnoses', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        los = data_dict.get('time_in_hospital', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        
        severity_index = lab_proc + 2.0 * proc + meds + 5.0 * diags
        los_med_ratio = los / (meds + 1.0)
        inpatient_los = inpatient * los
        
        target_dict['total_visits'] = total_visits
        target_dict['acute_visits'] = acute_visits
        target_dict['inpatient_ratio'] = inpatient_ratio
        target_dict['emergency_ratio'] = emergency_ratio
        target_dict['util_index'] = util_index
        target_dict['util_any'] = util_any
        target_dict['severity_index'] = severity_index
        target_dict['los_med_ratio'] = los_med_ratio
        target_dict['inpatient_los'] = inpatient_los
        
        # Log transforms and non-linear utilization signals
        target_dict['log_inpatient'] = np.log1p(inpatient)
        target_dict['log_emergency'] = np.log1p(emergency)
        target_dict['log_total_visits'] = np.log1p(total_visits)
        target_dict['inpatient_sq'] = inpatient ** 2
        
        target_dict['has_prior_inpatient'] = (inpatient > 0).astype(float)
        target_dict['has_prior_emergency'] = (emergency > 0).astype(float)
        target_dict['has_prior_outpatient'] = (outpatient > 0).astype(float)
        target_dict['frequent_inpatient'] = (inpatient >= 2).astype(float)
        target_dict['very_frequent_inpatient'] = (inpatient >= 3).astype(float)
        target_dict['has_many_inpatient'] = (inpatient >= 4).astype(float)
        target_dict['frequent_emergency'] = (emergency >= 2).astype(float)
        
        # High complexity & polypharmacy markers
        target_dict['polypharmacy'] = (meds >= 16).astype(float)
        target_dict['extreme_polypharmacy'] = (meds >= 25).astype(float)
        target_dict['high_diagnoses'] = (diags >= 8).astype(float)
        target_dict['complex_patient'] = ((meds >= 15) & (diags >= 7)).astype(float)
        
        # Age-utilization interactions
        target_dict['inpatient_age_interaction'] = inpatient * (target_dict['age_num'] / 50.0)
        target_dict['emergency_age_interaction'] = emergency * (target_dict['age_num'] / 50.0)
        
        # Intensity ratios per day of hospital stay
        target_dict['lab_per_day'] = lab_proc / (los + 1.0)
        target_dict['proc_per_day'] = proc / (los + 1.0)
        target_dict['meds_per_day'] = meds / (los + 1.0)
        target_dict['diags_per_day'] = diags / (los + 1.0)
        
        # Clinical admission and discharge status flags
        disch = data_dict.get('discharge_disposition_id', np.array(['1'] * len(los)))
        admit_src = data_dict.get('admission_source_id', np.array(['1'] * len(los)))
        admit_type = data_dict.get('admission_type_id', np.array(['1'] * len(los)))
        
        target_dict['disch_home'] = (disch == '1').astype(float)
        target_dict['disch_snf'] = (disch == '3').astype(float)
        target_dict['disch_homehealth'] = (disch == '6').astype(float)
        target_dict['disch_ama'] = (disch == '7').astype(float)
        target_dict['disch_transfer'] = np.isin(disch, ['2', '4', '5', '22']).astype(float)
        target_dict['disch_rehab'] = (disch == '22').astype(float)
        target_dict['high_risk_discharge'] = np.isin(disch, ['3', '6', '7', '22']).astype(float)
        
        target_dict['admit_emergency'] = (admit_src == '7').astype(float)
        target_dict['admit_transfer'] = np.isin(admit_src, ['4', '5', '6']).astype(float)
        target_dict['adm_type_emergency'] = (admit_type == '1').astype(float)
        target_dict['adm_type_urgent'] = (admit_type == '2').astype(float)
        target_dict['adm_type_elective'] = (admit_type == '3').astype(float)
        
    add_custom_numeric(train, train_feat)
    add_custom_numeric(eval_cols, eval_feat)
    extra_numeric.extend([
        'total_visits', 'acute_visits', 'inpatient_ratio', 'emergency_ratio',
        'util_index', 'util_any', 'severity_index', 'los_med_ratio', 'inpatient_los',
        'log_inpatient', 'log_emergency', 'log_total_visits', 'inpatient_sq',
        'has_prior_inpatient', 'has_prior_emergency', 'has_prior_outpatient',
        'frequent_inpatient', 'very_frequent_inpatient', 'has_many_inpatient', 'frequent_emergency',
        'polypharmacy', 'extreme_polypharmacy', 'high_diagnoses', 'complex_patient',
        'inpatient_age_interaction', 'emergency_age_interaction',
        'lab_per_day', 'proc_per_day', 'meds_per_day', 'diags_per_day',
        'disch_home', 'disch_snf', 'disch_homehealth', 'disch_ama', 'disch_transfer',
        'disch_rehab', 'high_risk_discharge',
        'admit_emergency', 'admit_transfer', 'adm_type_emergency', 'adm_type_urgent', 'adm_type_elective'
    ])
    
    # 8. Weight, Gender, and Lab Indicators
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
        
        target_dict['glu_high'] = np.isin(glu, ['>200', '>300']).astype(float)
        target_dict['glu_tested'] = (glu != 'None').astype(float)
        target_dict['a1c_high'] = np.isin(a1c, ['>7', '>8']).astype(float)
        target_dict['a1c_tested'] = (a1c != 'None').astype(float)
        
        glu_map = {'None': 0.0, 'Norm': 1.0, '>200': 2.0, '>300': 3.0}
        a1c_map = {'None': 0.0, 'Norm': 1.0, '>7': 2.0, '>8': 3.0}
        target_dict['glu_ord'] = np.array([glu_map.get(x, 0.0) for x in glu], dtype=float)
        target_dict['a1c_ord'] = np.array([a1c_map.get(x, 0.0) for x in a1c], dtype=float)

    add_lab_features(train, train_feat)
    add_lab_features(eval_cols, eval_feat)
    extra_numeric.extend(['glu_high', 'glu_tested', 'a1c_high', 'a1c_tested', 'glu_ord', 'a1c_ord'])
    
    train_feat['max_diagnoses'] = (train.get('number_diagnoses', np.zeros_like(train_feat['age_num'])) >= 9).astype(float)
    eval_feat['max_diagnoses'] = (eval_cols.get('number_diagnoses', np.zeros_like(eval_feat['age_num'])) >= 9).astype(float)
    extra_numeric.append('max_diagnoses')
    
    # 9. Multi-Modal Tensor Decomposition: Condition x Treatment x Utilization Subspace
    cm_tensor_keys = [
        'cm_chf', 'cm_renal', 'cm_diabetes_complicated', 'cm_pulmonary',
        'cm_cerebrovascular', 'cm_mi', 'cm_liver', 'cm_cancer', 'cm_pvd', 'cm_peptic_ulcer'
    ]
    med_tensor_keys = [
        'insulin_active', 'insulin_up', 'insulin_down', 'metformin_active',
        'sulf_active', 'med_num_change', 'insulin_plus_oral'
    ]
    util_tensor_keys = [
        'has_prior_inpatient', 'frequent_inpatient', 'disch_snf',
        'disch_homehealth', 'disch_transfer', 'admit_emergency', 'acute_visits'
    ]
    
    def build_interaction_tensor(feat_dict):
        c_mat = np.column_stack([feat_dict[k] for k in cm_tensor_keys])
        m_mat = np.column_stack([feat_dict[k] for k in med_tensor_keys])
        u_mat = np.column_stack([feat_dict[k] for k in util_tensor_keys])
        
        # Outer product tensors: C x M (shape N x 70) and C x U (shape N x 70)
        cm_outer = (c_mat[:, :, None] * m_mat[:, None, :]).reshape(len(c_mat), -1)
        cu_outer = (c_mat[:, :, None] * u_mat[:, None, :]).reshape(len(c_mat), -1)
        return np.hstack([cm_outer, cu_outer])
        
    t_train = build_interaction_tensor(train_feat)
    t_eval = build_interaction_tensor(eval_feat)
    
    n_tensor_comps = 8
    tensor_svd = TruncatedSVD(n_components=n_tensor_comps, algorithm='randomized', n_iter=5, random_state=42)
    train_tensor_svd = tensor_svd.fit_transform(t_train)
    eval_tensor_svd = tensor_svd.transform(t_eval)
    
    for i in range(n_tensor_comps):
        col_name = f'tensor_svd_{i}'
        train_feat[col_name] = train_tensor_svd[:, i]
        eval_feat[col_name] = eval_tensor_svd[:, i]
        extra_numeric.append(col_name)
        
    # Tensor reconstruction anomaly score
    t_train_rec = tensor_svd.inverse_transform(train_tensor_svd)
    t_eval_rec = tensor_svd.inverse_transform(eval_tensor_svd)
    train_feat['tensor_anomaly'] = np.linalg.norm(t_train - t_train_rec, axis=1)
    eval_feat['tensor_anomaly'] = np.linalg.norm(t_eval - t_eval_rec, axis=1)
    extra_numeric.append('tensor_anomaly')
    
    # 10. Low-Rank Latent Clinical Phenotype Decomposition
    pheno_cols = [
        'cm_mi', 'cm_chf', 'cm_pvd', 'cm_cerebrovascular', 'cm_dementia',
        'cm_pulmonary', 'cm_rheumatic', 'cm_peptic_ulcer', 'cm_renal',
        'cm_diabetes_complicated', 'cm_hemiplegia', 'cm_cancer', 'cm_liver',
        'cm_metastatic', 'cm_hiv', 'charlson_score',
        'grp_cnt_circulatory', 'grp_cnt_respiratory', 'grp_cnt_digestive',
        'grp_cnt_diabetes', 'grp_cnt_genitourinary',
        'time_in_hospital', 'num_medications', 'num_procedures',
        'num_lab_procedures', 'number_diagnoses', 'total_visits', 'frailty_index'
    ]
    
    def extract_pheno_matrix(feat_dict, base_dict):
        cols = []
        for c in pheno_cols:
            if c in feat_dict:
                cols.append(feat_dict[c])
            elif c in base_dict:
                cols.append(base_dict[c])
            else:
                cols.append(np.zeros(len(feat_dict['time_in_hospital']), dtype=float))
        return np.column_stack(cols).astype(float)
        
    p_train = extract_pheno_matrix(train_feat, train)
    p_eval = extract_pheno_matrix(eval_feat, eval_cols)
    
    pheno_scaler = StandardScaler()
    p_train_std = pheno_scaler.fit_transform(np.nan_to_num(p_train, nan=0.0))
    p_eval_std = pheno_scaler.transform(np.nan_to_num(p_eval, nan=0.0))
    
    n_pheno_comps = 5
    pheno_svd = TruncatedSVD(n_components=n_pheno_comps, algorithm='randomized', n_iter=5, random_state=101)
    train_pheno_svd = pheno_svd.fit_transform(p_train_std)
    eval_pheno_svd = pheno_svd.transform(p_eval_std)
    
    for i in range(n_pheno_comps):
        col_name = f'pheno_svd_{i}'
        train_feat[col_name] = train_pheno_svd[:, i]
        eval_feat[col_name] = eval_pheno_svd[:, i]
        extra_numeric.append(col_name)
        
    p_train_rec = pheno_svd.inverse_transform(train_pheno_svd)
    p_eval_rec = pheno_svd.inverse_transform(eval_pheno_svd)
    train_feat['pheno_anomaly'] = np.linalg.norm(p_train_std - p_train_rec, axis=1)
    eval_feat['pheno_anomaly'] = np.linalg.norm(p_eval_std - p_eval_rec, axis=1)
    extra_numeric.append('pheno_anomaly')
    
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from scipy.stats import rankdata

    train_feat, eval_feat, extra_numeric, extra_categorical = add_features(task, train, eval_cols, y_train)

    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_feat, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)

    # Diverse Multi-Model LightGBM Ensemble (Weight: 0.45)
    # Includes an Extremely Randomized Trees booster (extra_trees=True) for split stochasticity
    lgb_models = [
        lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=120,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0.1,
            reg_lambda=1.0,
            cat_smooth=40.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.018,
            num_leaves=24,
            min_child_samples=180,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_alpha=0.05,
            reg_lambda=2.0,
            cat_smooth=60.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.022,
            num_leaves=38,
            min_child_samples=90,
            subsample=0.65,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=0.5,
            cat_smooth=30.0,
            random_state=2023,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.02,
            num_leaves=28,
            min_child_samples=130,
            extra_trees=True,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_alpha=0.1,
            reg_lambda=1.5,
            cat_smooth=50.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        )
    ]

    # XGBoost Ensemble (Weight: 0.30)
    xgb_models = [
        xgb.XGBClassifier(
            n_estimators=240,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.75,
            colsample_bytree=0.6,
            colsample_bylevel=0.8,
            reg_lambda=2.0,
            reg_alpha=0.5,
            random_state=42,
            tree_method='hist',
            n_jobs=1,
            eval_metric='logloss'
        ),
        xgb.XGBClassifier(
            n_estimators=280,
            max_depth=4,
            learning_rate=0.018,
            subsample=0.8,
            colsample_bytree=0.5,
            colsample_bylevel=0.8,
            reg_lambda=3.0,
            reg_alpha=1.0,
            random_state=101,
            tree_method='hist',
            n_jobs=1,
            eval_metric='logloss'
        ),
        xgb.XGBClassifier(
            n_estimators=220,
            max_depth=6,
            learning_rate=0.022,
            subsample=0.7,
            colsample_bytree=0.65,
            colsample_bylevel=0.8,
            reg_lambda=1.5,
            reg_alpha=0.2,
            random_state=2024,
            tree_method='hist',
            n_jobs=1,
            eval_metric='logloss'
        )
    ]

    # HistGradientBoosting Ensemble (Weight: 0.17)
    hgb_models = [
        HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.02,
            max_leaf_nodes=31,
            min_samples_leaf=120,
            l2_regularization=1.0,
            random_state=42
        ),
        HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.018,
            max_leaf_nodes=24,
            min_samples_leaf=180,
            l2_regularization=2.0,
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

    # Fit & predict Regularized Linear Model via L-BFGS (Weight: 0.08)
    lr_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        LogisticRegression(C=0.02, max_iter=250, solver='lbfgs', random_state=42)
    )
    lr_pipeline.fit(x_train, y_train)
    lr_preds = lr_pipeline.predict_proba(x_eval)[:, 1]
    
    # Rank normalization of family predictions to align marginal distributions for ROC-AUC
    n_eval = float(len(x_eval))
    lgb_rank = rankdata(lgb_preds) / n_eval
    xgb_rank = rankdata(xgb_preds) / n_eval
    hgb_rank = rankdata(hgb_preds) / n_eval
    lr_rank = rankdata(lr_preds) / n_eval

    # Heterogeneous Multi-Family Rank Blend
    preds = 0.45 * lgb_rank + 0.30 * xgb_rank + 0.17 * hgb_rank + 0.08 * lr_rank
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
