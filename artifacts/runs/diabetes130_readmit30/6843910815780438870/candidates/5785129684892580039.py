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
    
    # 3d. Discharge & Primary Diagnosis Group Interaction
    train_feat['dis_diag1_grp'] = np.array([
        f"{d}_{g}" for d, g in zip(train['discharge_disposition_id'], train_feat['diag_1_grp'])
    ], dtype=object)
    eval_feat['dis_diag1_grp'] = np.array([
        f"{d}_{g}" for d, g in zip(eval_cols['discharge_disposition_id'], eval_feat['diag_1_grp'])
    ], dtype=object)
    extra_categorical.append('dis_diag1_grp')
    
    # 3e. Admission Type & Discharge Disposition Interaction
    train_feat['adm_dis'] = np.array([
        f"{a}_{d}" for a, d in zip(train['admission_type_id'], train['discharge_disposition_id'])
    ], dtype=object)
    eval_feat['adm_dis'] = np.array([
        f"{a}_{d}" for a, d in zip(eval_cols['admission_type_id'], eval_cols['discharge_disposition_id'])
    ], dtype=object)
    extra_categorical.append('adm_dis')
        
    # Helpers for macro routing groups
    def get_discharge_group(disch_col):
        res = []
        for x in disch_col:
            if x == '1':
                res.append('home')
            elif x in ['3', '4']:
                res.append('nursing')
            elif x in ['2', '5', '22', '23', '24']:
                res.append('hospital_rehab')
            elif x == '6':
                res.append('home_health')
            elif x == '7':
                res.append('ama')
            else:
                res.append('other')
        return np.array(res, dtype=object)

    def get_admission_source_group(src_col):
        res = []
        for x in src_col:
            if x == '7':
                res.append('emergency')
            elif x in ['1', '2', '3']:
                res.append('referral')
            elif x in ['4', '5', '6']:
                res.append('transfer')
            else:
                res.append('other')
        return np.array(res, dtype=object)

    def get_admission_type_group(type_col):
        res = []
        for x in type_col:
            if x == '1':
                res.append('emergency')
            elif x == '2':
                res.append('urgent')
            elif x == '3':
                res.append('elective')
            elif x in ['4', '7']:
                res.append('special')
            else:
                res.append('other')
        return np.array(res, dtype=object)

    # 3f. Macro-routing and medication profile features
    train_feat['discharge_group'] = get_discharge_group(train.get('discharge_disposition_id', []))
    eval_feat['discharge_group'] = get_discharge_group(eval_cols.get('discharge_disposition_id', []))
    extra_categorical.append('discharge_group')
    
    train_feat['admission_source_group'] = get_admission_source_group(train.get('admission_source_id', []))
    eval_feat['admission_source_group'] = get_admission_source_group(eval_cols.get('admission_source_id', []))
    extra_categorical.append('admission_source_group')
    
    train_feat['admission_type_group'] = get_admission_type_group(train.get('admission_type_id', []))
    eval_feat['admission_type_group'] = get_admission_type_group(eval_cols.get('admission_type_id', []))
    extra_categorical.append('admission_type_group')
    
    train_feat['routing_macro'] = np.array([f"{dg}_{asg}" for dg, asg in zip(train_feat['discharge_group'], train_feat['admission_source_group'])], dtype=object)
    eval_feat['routing_macro'] = np.array([f"{dg}_{asg}" for dg, asg in zip(eval_feat['discharge_group'], eval_feat['admission_source_group'])], dtype=object)
    extra_categorical.append('routing_macro')

    train_feat['med_profile'] = np.array([
        f"{met}_{gly}_{glip}_{ins}"
        for met, gly, glip, ins in zip(train.get('metformin', ['No']*len(train['time_in_hospital'])),
                                        train.get('glyburide', ['No']*len(train['time_in_hospital'])),
                                        train.get('glipizide', ['No']*len(train['time_in_hospital'])),
                                        train.get('insulin', ['No']*len(train['time_in_hospital'])))
    ], dtype=object)
    eval_feat['med_profile'] = np.array([
        f"{met}_{gly}_{glip}_{ins}"
        for met, gly, glip, ins in zip(eval_cols.get('metformin', ['No']*len(eval_cols['time_in_hospital'])),
                                        eval_cols.get('glyburide', ['No']*len(eval_cols['time_in_hospital'])),
                                        eval_cols.get('glipizide', ['No']*len(eval_cols['time_in_hospital'])),
                                        eval_cols.get('insulin', ['No']*len(eval_cols['time_in_hospital'])))
    ], dtype=object)
    extra_categorical.append('med_profile')

    train_feat['diag1_disch'] = np.array([f"{d1}_{dis}" for d1, dis in zip(train['diag_1'], train['discharge_disposition_id'])], dtype=object)
    eval_feat['diag1_disch'] = np.array([f"{d1}_{dis}" for d1, dis in zip(eval_cols['diag_1'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('diag1_disch')

    train_feat['diag1grp_disch'] = np.array([f"{g1}_{dis}" for g1, dis in zip(train_feat['diag_1_grp'], train['discharge_disposition_id'])], dtype=object)
    eval_feat['diag1grp_disch'] = np.array([f"{g1}_{dis}" for g1, dis in zip(eval_feat['diag_1_grp'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('diag1grp_disch')

    train_feat['age_disch'] = np.array([f"{a}_{dis}" for a, dis in zip(train.get('age', ['Unknown']*len(train['time_in_hospital'])), train['discharge_disposition_id'])], dtype=object)
    eval_feat['age_disch'] = np.array([f"{a}_{dis}" for a, dis in zip(eval_cols.get('age', ['Unknown']*len(eval_cols['time_in_hospital'])), eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('age_disch')

    train_feat['gender_disch'] = np.array([f"{g}_{dis}" for g, dis in zip(train.get('gender', ['Unknown']*len(train['time_in_hospital'])), train['discharge_disposition_id'])], dtype=object)
    eval_feat['gender_disch'] = np.array([f"{g}_{dis}" for g, dis in zip(eval_cols.get('gender', ['Unknown']*len(eval_cols['time_in_hospital'])), eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('gender_disch')

    train_feat['adm_dis_src'] = np.array([f"{at}_{dis}_{as}" for at, dis, as in zip(train['admission_type_id'], train['discharge_disposition_id'], train['admission_source_id'])], dtype=object)
    eval_feat['adm_dis_src'] = np.array([f"{at}_{dis}_{as}" for at, dis, as in zip(eval_cols['admission_type_id'], eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])], dtype=object)
    extra_categorical.append('adm_dis_src')

    # 4. Target encoding of high-cardinality categorical variables
    high_card_cols = [
        'diag_1', 'diag_2', 'diag_3', 'medical_specialty', 
        'discharge_disposition_id', 'admission_source_id', 'admission_type_id',
        'payer_code', 'dis_adm', 'med_combo', 'change_med',
        'dis_diag1_grp', 'adm_dis', 'diag_1_grp',
        'diag1_disch', 'diag1grp_disch', 'age_disch', 'gender_disch', 'adm_dis_src',
        'discharge_group', 'admission_source_group', 'admission_type_group', 'routing_macro', 'med_profile'
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
        
        # Hypoglycemic risk medications active and change counts
        hypo_med_cols = ['insulin', 'glipizide', 'glyburide', 'glimepiride', 'chlorpropamide', 'tolbutamide', 'acetohexamide', 'tolazamide']
        existing_hypo = [c for c in hypo_med_cols if c in data_dict]
        num_hypo_active = np.zeros(n_rows, dtype=np.float64)
        num_hypo_change = np.zeros(n_rows, dtype=np.float64)
        for col in existing_hypo:
            arr = data_dict[col]
            num_hypo_active += (arr != 'No').astype(np.float64)
            num_hypo_change += np.isin(arr, ['Up', 'Down']).astype(np.float64)
        
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
            'insulin_plus_oral': insulin_plus_oral,
            'num_hypo_active': num_hypo_active,
            'num_hypo_change': num_hypo_change
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
        
        target_dict['has_prior_inpatient'] = (inpatient > 0).astype(float)
        target_dict['has_prior_emergency'] = (emergency > 0).astype(float)
        target_dict['has_prior_outpatient'] = (outpatient > 0).astype(float)
        target_dict['frequent_inpatient'] = (inpatient >= 2).astype(float)
        target_dict['very_frequent_inpatient'] = (inpatient >= 3).astype(float)
        target_dict['frequent_emergency'] = (emergency >= 2).astype(float)
        
        # Clinical admission and discharge status flags
        disch = data_dict.get('discharge_disposition_id', np.array(['1'] * len(los)))
        admit_src = data_dict.get('admission_source_id', np.array(['1'] * len(los)))
        admit_type = data_dict.get('admission_type_id', np.array(['1'] * len(los)))
        
        target_dict['disch_home'] = (disch == '1').astype(float)
        target_dict['disch_snf'] = (disch == '3').astype(float)
        target_dict['disch_homehealth'] = (disch == '6').astype(float)
        target_dict['disch_ama'] = (disch == '7').astype(float)
        target_dict['disch_transfer'] = np.isin(disch, ['2', '4', '5', '22']).astype(float)
        target_dict['disch_rehab'] = np.isin(disch, ['22', '23', '24']).astype(float)
        target_dict['disch_high_risk'] = np.isin(disch, ['3', '5', '9', '15', '17', '22']).astype(float)

        target_dict['log_outpatient'] = np.log1p(outpatient)
        target_dict['log_emergency'] = np.log1p(emergency)
        target_dict['log_inpatient'] = np.log1p(inpatient)
        target_dict['log_total_visits'] = np.log1p(total_visits)
        target_dict['log_util_index'] = np.log1p(util_index)
        target_dict['inpatient_sq'] = inpatient ** 2
        target_dict['emergency_sq'] = emergency ** 2
        target_dict['util_index_sq'] = util_index ** 2
        
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
        'has_prior_inpatient', 'has_prior_emergency', 'has_prior_outpatient',
        'frequent_inpatient', 'very_frequent_inpatient', 'frequent_emergency',
        'disch_home', 'disch_snf', 'disch_homehealth', 'disch_ama', 'disch_transfer',
        'admit_emergency', 'admit_transfer', 'adm_type_emergency', 'adm_type_urgent', 'adm_type_elective',
        'disch_rehab', 'disch_high_risk',
        'log_outpatient', 'log_emergency', 'log_inpatient', 'log_total_visits', 'log_util_index',
        'inpatient_sq', 'emergency_sq', 'util_index_sq'
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
        target_dict['elderly'] = (age_array >= 75.0).astype(float)
        target_dict['age_times_charlson'] = age_array * target_dict['charlson_score']
        target_dict['age_times_num_diagnoses'] = age_array * diags
        target_dict['charlson_per_year_of_age'] = target_dict['charlson_score'] / (age_array + 1.0)

    add_frailty(train, train_feat, train_feat['age_num'])
    add_frailty(eval_cols, eval_feat, eval_feat['age_num'])
    extra_numeric.extend(['frailty_index', 'elderly', 'age_times_charlson', 'age_times_num_diagnoses', 'charlson_per_year_of_age'])
    
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
    
    def add_ratios(data_dict, target_dict):
        los = data_dict.get('time_in_hospital', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        lab = data_dict.get('num_lab_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        proc = data_dict.get('num_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        meds = data_dict.get('num_medications', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        diags = data_dict.get('number_diagnoses', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        
        target_dict['lab_per_day'] = lab / (los + 1.0)
        target_dict['proc_per_day'] = proc / (los + 1.0)
        target_dict['meds_per_day'] = meds / (los + 1.0)
        target_dict['procedures_per_lab'] = proc / (lab + 1.0)
        target_dict['meds_per_procedure'] = meds / (proc + 1.0)
        target_dict['diagnoses_per_day'] = diags / (los + 1.0)

    add_ratios(train, train_feat)
    add_ratios(eval_cols, eval_feat)
    extra_numeric.extend(['lab_per_day', 'proc_per_day', 'meds_per_day', 'procedures_per_lab', 'meds_per_procedure', 'diagnoses_per_day'])
    
    train_feat['max_diagnoses'] = (train.get('number_diagnoses', np.zeros_like(train_feat['age_num'])) >= 9).astype(float)
    eval_feat['max_diagnoses'] = (eval_cols.get('number_diagnoses', np.zeros_like(eval_feat['age_num'])) >= 9).astype(float)
    extra_numeric.append('max_diagnoses')
    
    # 8. Mismatch, complexity subscores and missingness indicators
    def add_advanced_features(feat_dict, data_dict):
        # Missingness flags
        med_spec = data_dict.get('medical_specialty', np.array([None] * len(feat_dict['time_in_hospital'])))
        pay_code = data_dict.get('payer_code', np.array([None] * len(feat_dict['time_in_hospital'])))
        feat_dict['specialty_missing'] = (med_spec == None).astype(float)
        feat_dict['payer_code_missing'] = (pay_code == None).astype(float)

        # Mismatch features
        a1c_high = feat_dict['a1c_high']
        ins_act = feat_dict['insulin_active']
        ins_std = feat_dict['insulin_steady']
        ins_chg = feat_dict['insulin_change']
        met_act = feat_dict['metformin_active']
        sulf_act = feat_dict['sulf_active']
        
        feat_dict['uncontrolled_no_insulin'] = ((a1c_high == 1.0) & (ins_act == 0.0)).astype(float)
        feat_dict['uncontrolled_steady_insulin'] = ((a1c_high == 1.0) & (ins_std == 1.0)).astype(float)
        feat_dict['uncontrolled_changing_insulin'] = ((a1c_high == 1.0) & (ins_chg == 1.0)).astype(float)
        feat_dict['controlled_on_insulin'] = ((a1c_high == 0.0) & (ins_act == 1.0)).astype(float)
        feat_dict['uncontrolled_no_oral'] = ((a1c_high == 1.0) & (met_act == 0.0) & (sulf_act == 0.0)).astype(float)
        feat_dict['severe_uncontrolled'] = ((a1c_high == 1.0) & (feat_dict['a1c_ord'] == 3.0)).astype(float)

        # Complexity subscores
        feat_dict['cardio_complexity'] = feat_dict['cm_mi'] + feat_dict['cm_chf'] + feat_dict['cm_pvd'] + feat_dict['cm_cerebrovascular']
        feat_dict['renal_metabolic_complexity'] = feat_dict['cm_renal'] + feat_dict['cm_diabetes_complicated']
        feat_dict['cancer_severity'] = feat_dict['cm_cancer'] + feat_dict['cm_metastatic']
        feat_dict['pulmonary_infectious_severity'] = feat_dict['cm_pulmonary'] + feat_dict['cm_hiv']
        
        # Overlap flags
        diag1_g = feat_dict['diag_1_grp']
        diag2_g = feat_dict['diag_2_grp']
        diag3_g = feat_dict['diag_3_grp']
        
        has_cardio = ((diag1_g == 'circulatory') | (diag2_g == 'circulatory') | (diag3_g == 'circulatory')).astype(float)
        has_resp = ((diag1_g == 'respiratory') | (diag2_g == 'respiratory') | (diag3_g == 'respiratory')).astype(float)
        has_renal = ((diag1_g == 'genitourinary') | (diag2_g == 'genitourinary') | (diag3_g == 'genitourinary')).astype(float)
        has_diab = ((diag1_g == 'diabetes') | (diag2_g == 'diabetes') | (diag3_g == 'diabetes')).astype(float)
        
        feat_dict['cardio_resp_overlap'] = has_cardio * has_resp
        feat_dict['cardio_renal_overlap'] = has_cardio * has_renal
        feat_dict['diab_renal_overlap'] = has_diab * has_renal
        feat_dict['diab_cardio_overlap'] = has_diab * has_cardio

    add_advanced_features(train_feat, train)
    add_advanced_features(eval_feat, eval_cols)
    extra_numeric.extend([
        'specialty_missing', 'payer_code_missing',
        'uncontrolled_no_insulin', 'uncontrolled_steady_insulin', 'uncontrolled_changing_insulin',
        'controlled_on_insulin', 'uncontrolled_no_oral', 'severe_uncontrolled',
        'cardio_complexity', 'renal_metabolic_complexity', 'cancer_severity', 'pulmonary_infectious_severity',
        'cardio_resp_overlap', 'cardio_renal_overlap', 'diab_renal_overlap', 'diab_cardio_overlap'
    ])
    
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
    
    from scipy.stats import rankdata

    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_feat, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)

    # Diverse Multi-Model LightGBM Ensemble (Weight: 0.50)
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
            n_estimators=220,
            learning_rate=0.025,
            num_leaves=40,
            min_child_samples=80,
            subsample=0.65,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=0.5,
            cat_smooth=30.0,
            extra_trees=True,
            random_state=2023,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.02,
            num_leaves=18,
            min_child_samples=150,
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
            reg_lambda=1.5,
            reg_alpha=0.2,
            random_state=2024,
            tree_method='hist',
            n_jobs=1,
            eval_metric='logloss'
        )
    ]

    # HistGradientBoosting Ensemble (Weight: 0.20)
    hgb_models = [
        HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.02,
            max_leaf_nodes=31,
            min_samples_leaf=120,
            l2_regularization=1.0,
            random_state=42,
            categorical_features=encoder.categorical_indices
        ),
        HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.018,
            max_leaf_nodes=24,
            min_samples_leaf=180,
            l2_regularization=2.0,
            random_state=101,
            categorical_features=encoder.categorical_indices
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
    
    # Rank normalization of family predictions to align marginal distributions for ROC-AUC
    n_eval = float(len(x_eval))
    lgb_rank = rankdata(lgb_preds) / n_eval
    xgb_rank = rankdata(xgb_preds) / n_eval
    hgb_rank = rankdata(hgb_preds) / n_eval

    # Weighted Rank Average Blend
    preds = 0.50 * lgb_rank + 0.30 * xgb_rank + 0.20 * hgb_rank
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
