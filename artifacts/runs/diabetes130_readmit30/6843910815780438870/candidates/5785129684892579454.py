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
        
    # Validated Charlson Comorbidity Index (CCI) scoring
    train_cci = (
        train_cm['cm_mi'] + train_cm['cm_chf'] + train_cm['cm_pvd'] +
        train_cm['cm_cerebrovascular'] + train_cm['cm_dementia'] +
        train_cm['cm_pulmonary'] + train_cm['cm_rheumatic'] +
        train_cm['cm_peptic_ulcer'] + train_cm['cm_liver'] +
        2.0 * train_cm['cm_hemiplegia'] + 2.0 * train_cm['cm_renal'] +
        2.0 * train_cm['cm_cancer'] + 2.0 * train_cm['cm_diabetes_complicated'] +
        6.0 * train_cm['cm_metastatic'] + 6.0 * train_cm['cm_hiv']
    )
    eval_cci = (
        eval_cm['cm_mi'] + eval_cm['cm_chf'] + eval_cm['cm_pvd'] +
        eval_cm['cm_cerebrovascular'] + eval_cm['cm_dementia'] +
        eval_cm['cm_pulmonary'] + eval_cm['cm_rheumatic'] +
        eval_cm['cm_peptic_ulcer'] + eval_cm['cm_liver'] +
        2.0 * eval_cm['cm_hemiplegia'] + 2.0 * eval_cm['cm_renal'] +
        2.0 * eval_cm['cm_cancer'] + 2.0 * eval_cm['cm_diabetes_complicated'] +
        6.0 * eval_cm['cm_metastatic'] + 6.0 * eval_cm['cm_hiv']
    )
    train_feat['cci'] = train_cci
    eval_feat['cci'] = eval_cci
    extra_numeric.append('cci')

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

    # High-risk chronic comorbidity interactions with prior inpatient history
    train_inp = train.get('number_inpatient', np.zeros(len(y_train), dtype=float))
    eval_inp = eval_cols.get('number_inpatient', np.zeros(len(eval_cols['time_in_hospital']), dtype=float))
    train_feat['cm_chf_prior_inpatient'] = train_cm['cm_chf'] * (train_inp > 0).astype(float)
    eval_feat['cm_chf_prior_inpatient'] = eval_cm['cm_chf'] * (eval_inp > 0).astype(float)
    extra_numeric.append('cm_chf_prior_inpatient')

    train_feat['cm_renal_prior_inpatient'] = train_cm['cm_renal'] * (train_inp > 0).astype(float)
    eval_feat['cm_renal_prior_inpatient'] = eval_cm['cm_renal'] * (eval_inp > 0).astype(float)
    extra_numeric.append('cm_renal_prior_inpatient')

    train_feat['cm_diab_prior_inpatient'] = train_cm['cm_diabetes_complicated'] * (train_inp > 0).astype(float)
    eval_feat['cm_diab_prior_inpatient'] = eval_cm['cm_diabetes_complicated'] * (eval_inp > 0).astype(float)
    extra_numeric.append('cm_diab_prior_inpatient')
        
    # 2. ICD9 groups and 3-digit root disease codes
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

    # 3d. Additional compound clinical interactions
    train_feat['diag1_disposition'] = np.array([
        f"{d}_{s}" for d, s in zip(train_feat['diag_1_root'], train['discharge_disposition_id'])
    ], dtype=object)
    eval_feat['diag1_disposition'] = np.array([
        f"{d}_{s}" for d, s in zip(eval_feat['diag_1_root'], eval_cols['discharge_disposition_id'])
    ], dtype=object)
    extra_categorical.append('diag1_disposition')

    train_feat['admtype_disposition'] = np.array([
        f"{a}_{d}" for a, d in zip(train['admission_type_id'], train['discharge_disposition_id'])
    ], dtype=object)
    eval_feat['admtype_disposition'] = np.array([
        f"{a}_{d}" for a, d in zip(eval_cols['admission_type_id'], eval_cols['discharge_disposition_id'])
    ], dtype=object)
    extra_categorical.append('admtype_disposition')

    train_feat['diag_grp_pair'] = np.array([
        f"{g1}_{g2}" for g1, g2 in zip(train_feat['diag_1_grp'], train_feat['diag_2_grp'])
    ], dtype=object)
    eval_feat['diag_grp_pair'] = np.array([
        f"{g1}_{g2}" for g1, g2 in zip(eval_feat['diag_1_grp'], eval_feat['diag_2_grp'])
    ], dtype=object)
    extra_categorical.append('diag_grp_pair')

    train_feat['age_prior_inpatient'] = np.array([
        f"{a}_{inp > 0}" for a, inp in zip(train.get('age', ['?']*len(train_inp)), train_inp)
    ], dtype=object)
    eval_feat['age_prior_inpatient'] = np.array([
        f"{a}_{inp > 0}" for a, inp in zip(eval_cols.get('age', ['?']*len(eval_inp)), eval_inp)
    ], dtype=object)
    extra_categorical.append('age_prior_inpatient')
        
    # 4. Target encoding of high-cardinality categorical variables
    high_card_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_root', 'diag_2_root', 'diag_3_root',
        'medical_specialty', 
        'discharge_disposition_id', 'admission_source_id', 'admission_type_id', 'payer_code', 
        'dis_adm', 'diag1_disposition', 'admtype_disposition', 'diag_grp_pair',
        'med_combo', 'change_med', 'age_prior_inpatient'
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

        # Non-linear utilization transformations and ratios
        total_visits = outpatient + emergency + inpatient
        acute_visits = emergency + inpatient
        target_dict['log_inpatient'] = np.log1p(inpatient)
        target_dict['log_emergency'] = np.log1p(emergency)
        target_dict['log_total_visits'] = np.log1p(total_visits)
        target_dict['acute_visits'] = acute_visits
        target_dict['acute_visit_ratio'] = acute_visits / (total_visits + 1.0)
        target_dict['inpatient_ratio'] = inpatient / (total_visits + 1.0)
        target_dict['inpatient_per_day_stay'] = inpatient / (los + 1.0)
        
    add_custom_numeric(train, train_feat)
    add_custom_numeric(eval_cols, eval_feat)
    extra_numeric.extend([
        'util_index', 'util_any', 'severity_index', 'los_med_ratio',
        'has_prior_inpatient', 'has_prior_emergency', 'has_prior_outpatient', 'frequent_inpatient', 'frequent_emergency',
        'log_inpatient', 'log_emergency', 'log_total_visits', 'acute_visits', 'acute_visit_ratio', 'inpatient_ratio', 'inpatient_per_day_stay'
    ])

    # LACE Index (Length of stay, Acuity, Comorbidity, Emergency visits)
    def calc_lace(los, adm_typ, cci_arr, emerg):
        lace_l = np.zeros_like(los, dtype=float)
        lace_l[los == 1] = 1.0
        lace_l[los == 2] = 2.0
        lace_l[los == 3] = 3.0
        lace_l[(los >= 4) & (los <= 6)] = 4.0
        lace_l[(los >= 7) & (los <= 13)] = 5.0
        lace_l[los >= 14] = 7.0

        lace_a = np.zeros_like(los, dtype=float)
        lace_a[adm_typ == '1'] = 3.0
        lace_a[adm_typ == '2'] = 2.0

        lace_c = np.clip(cci_arr, 0.0, 5.0)
        lace_e = np.clip(emerg, 0.0, 4.0)
        return lace_l + lace_a + lace_c + lace_e

    train_los = train.get('time_in_hospital', np.zeros(len(y_train), dtype=float))
    eval_los = eval_cols.get('time_in_hospital', np.zeros(len(eval_inp), dtype=float))
    train_adm_typ = train.get('admission_type_id', np.array(['1'] * len(train_los)))
    eval_adm_typ = eval_cols.get('admission_type_id', np.array(['1'] * len(eval_los)))
    train_emerg = train.get('number_emergency', np.zeros_like(train_los, dtype=float))
    eval_emerg = eval_cols.get('number_emergency', np.zeros_like(eval_los, dtype=float))

    train_feat['lace_index'] = calc_lace(train_los, train_adm_typ, train_cci, train_emerg)
    eval_feat['lace_index'] = calc_lace(eval_los, eval_adm_typ, eval_cci, eval_emerg)
    extra_numeric.append('lace_index')

    # Discharge disposition and admission pathway indicators
    def add_disposition_flags(data_dict, target_dict):
        disp = data_dict.get('discharge_disposition_id', np.array(['1'] * len(data_dict['time_in_hospital'])))
        adm_src = data_dict.get('admission_source_id', np.array(['1'] * len(data_dict['time_in_hospital'])))
        adm_typ = data_dict.get('admission_type_id', np.array(['1'] * len(data_dict['time_in_hospital'])))

        target_dict['discharged_home'] = (disp == '1').astype(float)
        target_dict['discharged_ama'] = (disp == '7').astype(float)
        target_dict['discharged_snf_facility'] = np.isin(disp, ['2', '3', '4', '5', '22', '23', '24']).astype(float)
        target_dict['discharged_home_health'] = (disp == '6').astype(float)

        target_dict['adm_emergency_type'] = (adm_typ == '1').astype(float)
        target_dict['adm_urgent_type'] = (adm_typ == '2').astype(float)
        target_dict['adm_elective_type'] = (adm_typ == '3').astype(float)

        target_dict['adm_emergency_src'] = (adm_src == '7').astype(float)
        target_dict['adm_transfer_src'] = np.isin(adm_src, ['1', '2', '3', '4', '5', '6']).astype(float)

    add_disposition_flags(train, train_feat)
    add_disposition_flags(eval_cols, eval_feat)
    extra_numeric.extend([
        'discharged_home', 'discharged_ama', 'discharged_snf_facility', 'discharged_home_health',
        'adm_emergency_type', 'adm_urgent_type', 'adm_elective_type', 'adm_emergency_src', 'adm_transfer_src'
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
        
        target_dict['glu_high'] = np.isin(glu, ['>200', '>300']).astype(float)
        target_dict['glu_tested'] = (glu != 'None').astype(float)
        target_dict['a1c_high'] = np.isin(a1c, ['>7', '>8']).astype(float)
        target_dict['a1c_tested'] = (a1c != 'None').astype(float)

    add_lab_features(train, train_feat)
    add_lab_features(eval_cols, eval_feat)
    extra_numeric.extend(['glu_high', 'glu_tested', 'a1c_high', 'a1c_tested'])

    # Clinical inertia: high A1C or glucose with no medication change (Strack et al. 2014)
    train_change = train.get('change', np.array(['No'] * len(train['time_in_hospital'])))
    eval_change = eval_cols.get('change', np.array(['No'] * len(eval_cols['time_in_hospital'])))
    train_feat['a1c_high_no_change'] = ((train_feat['a1c_high'] == 1.0) & (train_change == 'No')).astype(float)
    eval_feat['a1c_high_no_change'] = ((eval_feat['a1c_high'] == 1.0) & (eval_change == 'No')).astype(float)
    train_feat['glu_high_no_change'] = ((train_feat['glu_high'] == 1.0) & (train_change == 'No')).astype(float)
    eval_feat['glu_high_no_change'] = ((eval_feat['glu_high'] == 1.0) & (eval_change == 'No')).astype(float)
    extra_numeric.extend(['a1c_high_no_change', 'glu_high_no_change'])

    # Diabetes drug classes and polypharmacy
    def add_med_classes(data_dict, target_dict):
        n_rows = len(data_dict['time_in_hospital'])
        sulf_cols = ['glimepiride', 'glipizide', 'glyburide', 'chlorpropamide', 'tolbutamide', 'acetohexamide', 'tolazamide']
        has_sulf = np.zeros(n_rows, dtype=float)
        for c in sulf_cols:
            if c in data_dict:
                has_sulf = np.maximum(has_sulf, (data_dict[c] != 'No').astype(float))

        met_cols = ['metformin', 'glyburide-metformin', 'glipizide-metformin']
        has_met = np.zeros(n_rows, dtype=float)
        for c in met_cols:
            if c in data_dict:
                has_met = np.maximum(has_met, (data_dict[c] != 'No').astype(float))

        tzd_cols = ['pioglitazone', 'rosiglitazone', 'troglitazone']
        has_tzd = np.zeros(n_rows, dtype=float)
        for c in tzd_cols:
            if c in data_dict:
                has_tzd = np.maximum(has_tzd, (data_dict[c] != 'No').astype(float))

        has_insulin = (data_dict.get('insulin', np.array(['No']*n_rows)) != 'No').astype(float)
        target_dict['diab_class_count'] = has_sulf + has_met + has_tzd + has_insulin
        target_dict['insulin_plus_oral'] = has_insulin * ((has_sulf + has_met + has_tzd) > 0).astype(float)

    add_med_classes(train, train_feat)
    add_med_classes(eval_cols, eval_feat)
    extra_numeric.extend(['diab_class_count', 'insulin_plus_oral'])
    
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

    train_feat, eval_feat, extra_numeric, extra_categorical = add_features(task, train, eval_cols, y_train)
    
    from scipy.stats import rankdata

    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_feat, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)

    # Diverse Multi-Model LightGBM Ensemble (Weight: 0.52)
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
            n_estimators=200,
            learning_rate=0.015,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            extra_trees=True,
            reg_lambda=0.1,
            cat_smooth=40.0,
            random_state=777,
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
            random_state=101,
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
    
    # Probability blend
    prob_blend = 0.52 * lgb_preds + 0.30 * xgb_preds + 0.18 * hgb_preds

    # Rank-percentile blend (immunizes against cross-model probability calibration skew)
    n_eval = len(x_eval)
    lgb_rank = (rankdata(lgb_preds) - 0.5) / n_eval
    xgb_rank = (rankdata(xgb_preds) - 0.5) / n_eval
    hgb_rank = (rankdata(hgb_preds) - 0.5) / n_eval
    rank_blend = 0.52 * lgb_rank + 0.30 * xgb_rank + 0.18 * hgb_rank

    # Hybrid blend: combines probability tail discrimination with rank regularization
    preds = 0.50 * prob_blend + 0.50 * rank_blend
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
