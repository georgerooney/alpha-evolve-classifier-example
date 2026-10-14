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

    # Cardiorenal-metabolic triad
    train_feat['cm_cardiorenal_diab'] = train_cm['cm_chf'] * train_cm['cm_renal'] * train_cm['cm_diabetes_complicated']
    eval_feat['cm_cardiorenal_diab'] = eval_cm['cm_chf'] * eval_cm['cm_renal'] * eval_cm['cm_diabetes_complicated']
    extra_numeric.append('cm_cardiorenal_diab')
        
    # 2. ICD9 groups and 3-Digit Category Prefixes
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    for col in diag_cols:
        train_feat[f'{col}_grp'] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_feat[f'{col}_grp'] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(f'{col}_grp')
        
        train_feat[f'{col}_pfx'] = np.array([
            (str(x)[:3] if x is not None and len(str(x)) >= 3 else (str(x) if x is not None else 'missing'))
            for x in train[col]
        ], dtype=object)
        eval_feat[f'{col}_pfx'] = np.array([
            (str(x)[:3] if x is not None and len(str(x)) >= 3 else (str(x) if x is not None else 'missing'))
            for x in eval_cols[col]
        ], dtype=object)
        extra_categorical.append(f'{col}_pfx')
        
    # Primary Diagnosis HRRP & High-Risk Clinical Phenotypes
    def get_diag_phenotypes(diag_1_arr):
        n = len(diag_1_arr)
        d1_chf = np.zeros(n, dtype=float)
        d1_ami = np.zeros(n, dtype=float)
        d1_copd = np.zeros(n, dtype=float)
        d1_pneumonia = np.zeros(n, dtype=float)
        d1_renal = np.zeros(n, dtype=float)
        d1_dka = np.zeros(n, dtype=float)
        d1_diab = np.zeros(n, dtype=float)
        d1_sepsis = np.zeros(n, dtype=float)
        
        for i, code in enumerate(diag_1_arr):
            if code is None:
                continue
            c = str(code)
            if c.startswith('428'):
                d1_chf[i] = 1.0
            elif c.startswith('410'):
                d1_ami[i] = 1.0
            elif c.startswith(('491', '492', '496')):
                d1_copd[i] = 1.0
            elif c.startswith(('480', '481', '482', '485', '486')):
                d1_pneumonia[i] = 1.0
            elif c.startswith(('584', '585', '586')):
                d1_renal[i] = 1.0
            elif c.startswith('250.1'):
                d1_dka[i] = 1.0
            elif c.startswith('250'):
                d1_diab[i] = 1.0
            elif c.startswith(('038', '995.91', '599.0')):
                d1_sepsis[i] = 1.0
                
        return {
            'd1_chf': d1_chf,
            'd1_ami': d1_ami,
            'd1_copd': d1_copd,
            'd1_pneumonia': d1_pneumonia,
            'd1_renal': d1_renal,
            'd1_dka': d1_dka,
            'd1_diab': d1_diab,
            'd1_sepsis': d1_sepsis,
        }

    tr_d1_pheno = get_diag_phenotypes(train['diag_1'])
    ev_d1_pheno = get_diag_phenotypes(eval_cols['diag_1'])
    for pk in tr_d1_pheno.keys():
        train_feat[pk] = tr_d1_pheno[pk]
        eval_feat[pk] = ev_d1_pheno[pk]
        extra_numeric.append(pk)
        
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
        
    # 3. Categorical Interactions
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

    train_feat['age_dis'] = np.array([
        f"{ag}_{d}" for ag, d in zip(train['age'], train['discharge_disposition_id'])
    ], dtype=object)
    eval_feat['age_dis'] = np.array([
        f"{ag}_{d}" for ag, d in zip(eval_cols['age'], eval_cols['discharge_disposition_id'])
    ], dtype=object)
    extra_categorical.append('age_dis')

    train_feat['diag1_age'] = np.array([
        f"{g}_{ag}" for g, ag in zip(train_feat['diag_1_grp'], train['age'])
    ], dtype=object)
    eval_feat['diag1_age'] = np.array([
        f"{g}_{ag}" for g, ag in zip(eval_feat['diag_1_grp'], eval_cols['age'])
    ], dtype=object)
    extra_categorical.append('diag1_age')
        
    # 4. Target encoding of high-cardinality categorical variables
    high_card_cols = [
        'diag_1', 'diag_2', 'diag_3', 'medical_specialty', 
        'discharge_disposition_id', 'admission_source_id', 'admission_type_id',
        'payer_code', 'dis_adm', 'med_combo', 'change_med',
        'dis_diag1_grp', 'adm_dis', 'diag_1_grp', 'diag_1_pfx', 'diag_2_pfx',
        'diag_3_pfx', 'age_dis', 'diag1_age'
    ]
    for col in high_card_cols:
        tr_enc, ev_enc = oof_target_encode(train_feat[col], y_train, eval_feat[col], n_folds=5, smoothing=20.0, seed=42)
        train_feat[f'{col}_te'] = tr_enc
        eval_feat[f'{col}_te'] = ev_enc
        extra_numeric.append(f'{col}_te')

    # Frequency encoding of key high-cardinality features
    def frequency_encode(train_col, eval_col):
        vals, counts = np.unique(train_col, return_counts=True)
        count_dict = dict(zip(vals, counts.astype(float) / float(len(train_col))))
        train_freq = np.array([count_dict.get(x, 0.0) for x in train_col], dtype=float)
        eval_freq = np.array([count_dict.get(x, 0.0) for x in eval_col], dtype=float)
        return train_freq, eval_freq

    for freq_col in ['diag_1_pfx', 'diag_1', 'medical_specialty', 'discharge_disposition_id']:
        tr_fq, ev_fq = frequency_encode(train_feat[freq_col], eval_feat[freq_col])
        train_feat[f'{freq_col}_freq'] = tr_fq
        eval_feat[f'{freq_col}_freq'] = ev_fq
        extra_numeric.append(f'{freq_col}_freq')
        
    # 5. Medication and Pharmacotherapy Complexity features
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    existing_med_cols = [c for c in med_cols if c in train]
    
    def get_med_features(data_dict, cm_dict):
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
        
        # Metformin & Sulfonylurea & TZD flags
        metformin_arr = data_dict.get('metformin', np.array(['No'] * n_rows))
        metformin_active = (metformin_arr != 'No').astype(np.float64)
        
        glip_arr = data_dict.get('glipizide', np.array(['No'] * n_rows))
        glyb_arr = data_dict.get('glyburide', np.array(['No'] * n_rows))
        glim_arr = data_dict.get('glimepiride', np.array(['No'] * n_rows))
        sulf_active = ((glip_arr != 'No') | (glyb_arr != 'No') | (glim_arr != 'No')).astype(np.float64)

        pio_arr = data_dict.get('pioglitazone', np.array(['No'] * n_rows))
        rosi_arr = data_dict.get('rosiglitazone', np.array(['No'] * n_rows))
        tro_arr = data_dict.get('troglitazone', np.array(['No'] * n_rows))
        tzd_active = ((pio_arr != 'No') | (rosi_arr != 'No') | (tro_arr != 'No')).astype(np.float64)
        
        # Regimen complexity and discordance
        oral_count = num_active - insulin_active
        oral_only = ((insulin_active == 0.0) & (oral_count > 0.0)).astype(np.float64)
        insulin_only = ((insulin_active == 1.0) & (oral_count == 0.0)).astype(np.float64)
        insulin_plus_oral = (insulin_active * (num_active > 1).astype(np.float64))
        no_diabetes_meds = (num_active == 0.0).astype(np.float64)
        
        med_increased_only = ((num_up > 0) & (num_down == 0)).astype(np.float64)
        med_decreased_only = ((num_down > 0) & (num_up == 0)).astype(np.float64)
        med_mixed_change = ((num_up > 0) & (num_down > 0)).astype(np.float64)

        # High-risk clinical contraindication interactions
        metformin_renal_risk = metformin_active * cm_dict['cm_renal']
        tzd_chf_risk = tzd_active * cm_dict['cm_chf']
        
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
            'tzd_active': tzd_active,
            'oral_med_count': oral_count,
            'oral_only': oral_only,
            'insulin_only': insulin_only,
            'insulin_plus_oral': insulin_plus_oral,
            'no_diabetes_meds': no_diabetes_meds,
            'med_increased_only': med_increased_only,
            'med_decreased_only': med_decreased_only,
            'med_mixed_change': med_mixed_change,
            'metformin_renal_risk': metformin_renal_risk,
            'tzd_chf_risk': tzd_chf_risk
        }
        
    train_med_feats = get_med_features(train, train_cm)
    eval_med_feats = get_med_features(eval_cols, eval_cm)
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
        acute_ratio = (acute_visits + 1.0) / (outpatient + 1.0)
        
        lab_proc = data_dict.get('num_lab_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        proc = data_dict.get('num_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        meds = data_dict.get('num_medications', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        diags = data_dict.get('number_diagnoses', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        los = data_dict.get('time_in_hospital', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        
        severity_index = lab_proc + 2.0 * proc + meds + 5.0 * diags
        los_med_ratio = los / (meds + 1.0)
        inpatient_los = inpatient * los
        inpatient_per_day = inpatient / (los + 1.0)
        
        target_dict['total_visits'] = total_visits
        target_dict['acute_visits'] = acute_visits
        target_dict['inpatient_ratio'] = inpatient_ratio
        target_dict['emergency_ratio'] = emergency_ratio
        target_dict['acute_ratio'] = acute_ratio
        target_dict['util_index'] = util_index
        target_dict['util_any'] = util_any
        target_dict['severity_index'] = severity_index
        target_dict['los_med_ratio'] = los_med_ratio
        target_dict['inpatient_los'] = inpatient_los
        target_dict['inpatient_per_day'] = inpatient_per_day
        
        # Log transforms for highly skewed visit counts
        target_dict['log_inpatient'] = np.log1p(inpatient)
        target_dict['log_emergency'] = np.log1p(emergency)
        target_dict['log_outpatient'] = np.log1p(outpatient)
        target_dict['log_total_visits'] = np.log1p(total_visits)
        
        target_dict['has_prior_inpatient'] = (inpatient > 0).astype(float)
        target_dict['has_prior_emergency'] = (emergency > 0).astype(float)
        target_dict['has_prior_outpatient'] = (outpatient > 0).astype(float)
        target_dict['frequent_inpatient'] = (inpatient >= 2).astype(float)
        target_dict['very_frequent_inpatient'] = (inpatient >= 3).astype(float)
        target_dict['frequent_emergency'] = (emergency >= 2).astype(float)
        target_dict['frequent_flyer'] = ((inpatient >= 2) & (emergency >= 1)).astype(float)
        
        # Clinical admission and discharge status flags
        disch = data_dict.get('discharge_disposition_id', np.array(['1'] * len(los)))
        admit_src = data_dict.get('admission_source_id', np.array(['1'] * len(los)))
        admit_type = data_dict.get('admission_type_id', np.array(['1'] * len(los)))
        
        target_dict['disch_home'] = (disch == '1').astype(float)
        target_dict['disch_snf'] = (disch == '3').astype(float)
        target_dict['disch_homehealth'] = (disch == '6').astype(float)
        target_dict['disch_ama'] = (disch == '7').astype(float)
        target_dict['disch_transfer'] = np.isin(disch, ['2', '4', '5', '22']).astype(float)
        
        target_dict['admit_emergency'] = (admit_src == '7').astype(float)
        target_dict['admit_transfer'] = np.isin(admit_src, ['4', '5', '6']).astype(float)
        target_dict['adm_type_emergency'] = (admit_type == '1').astype(float)
        target_dict['adm_type_urgent'] = (admit_type == '2').astype(float)
        target_dict['adm_type_elective'] = (admit_type == '3').astype(float)

        target_dict['short_emergency_stay'] = ((los <= 2.0) & (target_dict['admit_emergency'] == 1.0)).astype(float)
        target_dict['long_stay'] = (los >= 7.0).astype(float)
        target_dict['medical_management'] = (proc == 0.0).astype(float)
        target_dict['med_per_diag'] = meds / (diags + 1.0)
        target_dict['lab_per_diag'] = lab_proc / (diags + 1.0)
        
    add_custom_numeric(train, train_feat)
    add_custom_numeric(eval_cols, eval_feat)
    extra_numeric.extend([
        'total_visits', 'acute_visits', 'inpatient_ratio', 'emergency_ratio', 'acute_ratio',
        'util_index', 'util_any', 'severity_index', 'los_med_ratio', 'inpatient_los', 'inpatient_per_day',
        'log_inpatient', 'log_emergency', 'log_outpatient', 'log_total_visits',
        'has_prior_inpatient', 'has_prior_emergency', 'has_prior_outpatient',
        'frequent_inpatient', 'very_frequent_inpatient', 'frequent_emergency', 'frequent_flyer',
        'disch_home', 'disch_snf', 'disch_homehealth', 'disch_ama', 'disch_transfer',
        'admit_emergency', 'admit_transfer', 'adm_type_emergency', 'adm_type_urgent', 'adm_type_elective',
        'short_emergency_stay', 'long_stay', 'medical_management', 'med_per_diag', 'lab_per_diag'
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
        target_dict['elderly_sulf'] = (target_dict['elderly'] * target_dict['sulf_active']).astype(float)
        target_dict['elderly_snf'] = (target_dict['elderly'] * target_dict['disch_snf']).astype(float)
        target_dict['elderly_homehealth'] = (target_dict['elderly'] * target_dict['disch_homehealth']).astype(float)

    add_frailty(train, train_feat, train_feat['age_num'])
    add_frailty(eval_cols, eval_feat, eval_feat['age_num'])
    extra_numeric.extend(['frailty_index', 'elderly', 'elderly_sulf', 'elderly_snf', 'elderly_homehealth'])

    # 8. Clinical LACE and LACE+ Readmission Indices
    def compute_lace(los, admit_type, charlson, emergency, inpatient, age_num):
        n = len(los)
        lace = np.zeros(n, dtype=float)
        lace_plus = np.zeros(n, dtype=float)
        for i in range(n):
            l_val = los[i]
            if l_val <= 1:
                l_s = 1.0
            elif l_val == 2:
                l_s = 2.0
            elif l_val == 3:
                l_s = 3.0
            elif 4 <= l_val <= 6:
                l_s = 4.0
            elif 7 <= l_val <= 13:
                l_s = 5.0
            else:
                l_s = 7.0
                
            at = str(admit_type[i])
            a_s = 3.0 if at in ('1', '2') else 0.0
            
            c_val = charlson[i]
            if c_val <= 0:
                c_s = 0.0
            elif c_val == 1:
                c_s = 1.0
            elif c_val == 2:
                c_s = 2.0
            elif c_val == 3:
                c_s = 3.0
            else:
                c_s = 5.0
                
            e_val = emergency[i]
            if e_val <= 0:
                e_s = 0.0
            elif e_val == 1:
                e_s = 1.0
            elif e_val == 2:
                e_s = 2.0
            elif e_val == 3:
                e_s = 3.0
            else:
                e_s = 4.0
                
            score = l_s + a_s + c_s + e_s
            lace[i] = score
            
            inp_val = min(float(inpatient[i]), 3.0)
            age_v = age_num[i]
            age_s = 3.0 if age_v >= 75 else (2.0 if age_v >= 65 else (1.0 if age_v >= 45 else 0.0))
            lace_plus[i] = score + inp_val + age_s
            
        return lace, lace_plus

    tr_lace, tr_lace_plus = compute_lace(
        train['time_in_hospital'], train['admission_type_id'], train_feat['charlson_score'],
        train['number_emergency'], train['number_inpatient'], train_feat['age_num']
    )
    ev_lace, ev_lace_plus = compute_lace(
        eval_cols['time_in_hospital'], eval_cols['admission_type_id'], eval_feat['charlson_score'],
        eval_cols['number_emergency'], eval_cols['number_inpatient'], eval_feat['age_num']
    )
    train_feat['lace_score'] = tr_lace
    eval_feat['lace_score'] = ev_lace
    train_feat['lace_plus_score'] = tr_lace_plus
    eval_feat['lace_plus_score'] = ev_lace_plus
    train_feat['lace_high_risk'] = (tr_lace >= 10.0).astype(float)
    eval_feat['lace_high_risk'] = (ev_lace >= 10.0).astype(float)
    train_feat['lace_extreme_risk'] = (tr_lace >= 13.0).astype(float)
    eval_feat['lace_extreme_risk'] = (ev_lace >= 13.0).astype(float)
    extra_numeric.extend(['lace_score', 'lace_plus_score', 'lace_high_risk', 'lace_extreme_risk'])
    
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
    
    # 9. Glycemic Control & Testing Cohorts (Strack et al. 2014)
    def add_lab_features(data_dict, target_dict):
        glu = data_dict.get('max_glu_serum', np.array(['None'] * len(data_dict['time_in_hospital'])))
        a1c = data_dict.get('A1Cresult', np.array(['None'] * len(data_dict['time_in_hospital'])))
        chg = data_dict.get('change', np.array(['No'] * len(data_dict['time_in_hospital'])))
        
        target_dict['glu_high'] = np.isin(glu, ['>200', '>300']).astype(float)
        target_dict['glu_tested'] = (glu != 'None').astype(float)
        target_dict['glu_untested'] = (glu == 'None').astype(float)
        target_dict['glu_extreme'] = (glu == '>300').astype(float)

        target_dict['a1c_high'] = np.isin(a1c, ['>7', '>8']).astype(float)
        target_dict['a1c_tested'] = (a1c != 'None').astype(float)
        target_dict['a1c_untested'] = (a1c == 'None').astype(float)
        target_dict['a1c_norm'] = (a1c == 'Norm').astype(float)

        # Clinical therapeutic inertia: HbA1c > 8 and NO medication change
        target_dict['a1c_high_no_change'] = ((a1c == '>8') & (chg == 'No')).astype(float)
        target_dict['a1c_high_with_change'] = ((a1c == '>8') & (chg == 'Ch')).astype(float)
        
        glu_map = {'None': 0.0, 'Norm': 1.0, '>200': 2.0, '>300': 3.0}
        a1c_map = {'None': 0.0, 'Norm': 1.0, '>7': 2.0, '>8': 3.0}
        target_dict['glu_ord'] = np.array([glu_map.get(x, 0.0) for x in glu], dtype=float)
        target_dict['a1c_ord'] = np.array([a1c_map.get(x, 0.0) for x in a1c], dtype=float)

    add_lab_features(train, train_feat)
    add_lab_features(eval_cols, eval_feat)
    extra_numeric.extend([
        'glu_high', 'glu_tested', 'glu_untested', 'glu_extreme',
        'a1c_high', 'a1c_tested', 'a1c_untested', 'a1c_norm',
        'a1c_high_no_change', 'a1c_high_with_change', 'glu_ord', 'a1c_ord'
    ])
    
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

    # Diverse Multi-Model LightGBM Ensemble (Weight: 0.50)
    # Including balanced models, cost-sensitive models, deep, and conservative trees
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
        # Cost-sensitive LightGBM: prioritizes splitting on positive cluster separation
        lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.02,
            num_leaves=28,
            min_child_samples=140,
            scale_pos_weight=1.5,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.2,
            reg_lambda=1.5,
            cat_smooth=35.0,
            random_state=777,
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
        # Cost-sensitive XGBoost
        xgb.XGBClassifier(
            n_estimators=260,
            max_depth=4,
            learning_rate=0.018,
            scale_pos_weight=1.4,
            subsample=0.8,
            colsample_bytree=0.55,
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

    n_eval = float(len(x_eval))

    # Fit & rank-predict LightGBM models
    lgb_ranks = np.zeros(len(x_eval))
    for model in lgb_models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        p = model.predict_proba(x_eval)[:, 1]
        lgb_ranks += rankdata(p) / n_eval
    lgb_rank = lgb_ranks / len(lgb_models)
    
    # Fit & rank-predict XGBoost models
    xgb_ranks = np.zeros(len(x_eval))
    for model in xgb_models:
        model.fit(x_train, y_train)
        p = model.predict_proba(x_eval)[:, 1]
        xgb_ranks += rankdata(p) / n_eval
    xgb_rank = xgb_ranks / len(xgb_models)
    
    # Fit & rank-predict HistGradientBoosting models
    hgb_ranks = np.zeros(len(x_eval))
    for model in hgb_models:
        model.fit(x_train, y_train)
        p = model.predict_proba(x_eval)[:, 1]
        hgb_ranks += rankdata(p) / n_eval
    hgb_rank = hgb_ranks / len(hgb_models)

    # Weighted Rank Average Blend across families
    preds = 0.50 * lgb_rank + 0.30 * xgb_rank + 0.20 * hgb_rank
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
