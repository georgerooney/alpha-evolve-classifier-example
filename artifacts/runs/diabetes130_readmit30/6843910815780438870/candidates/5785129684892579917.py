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
    
    # Parse age early for age-dependent scoring and clinical indices
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

    # Age-adjusted Charlson Comorbidity Index (Charlson-Deyo standard)
    def charlson_age_points(age_arr):
        pts = np.zeros_like(age_arr, dtype=float)
        pts[(age_arr >= 50) & (age_arr < 60)] = 1.0
        pts[(age_arr >= 60) & (age_arr < 70)] = 2.0
        pts[(age_arr >= 70) & (age_arr < 80)] = 3.0
        pts[age_arr >= 80] = 4.0
        return pts
        
    train_feat['charlson_age_pts'] = charlson_age_points(train_feat['age_num'])
    eval_feat['charlson_age_pts'] = charlson_age_points(eval_feat['age_num'])
    train_feat['charlson_age_adj'] = train_feat['charlson_score'] + train_feat['charlson_age_pts']
    eval_feat['charlson_age_adj'] = eval_feat['charlson_score'] + eval_feat['charlson_age_pts']
    extra_numeric.extend(['charlson_age_pts', 'charlson_age_adj'])

    # Multimorbidity count thresholds
    train_feat['multimorbidity'] = (train_cm['cm_count'] >= 2.0).astype(float)
    eval_feat['multimorbidity'] = (eval_cm['cm_count'] >= 2.0).astype(float)
    train_feat['high_multimorbidity'] = (train_cm['cm_count'] >= 3.0).astype(float)
    eval_feat['high_multimorbidity'] = (eval_cm['cm_count'] >= 3.0).astype(float)
    extra_numeric.extend(['multimorbidity', 'high_multimorbidity'])
        
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
        
    def parse_icd9_3digit(code):
        if code is None or not isinstance(code, str) or code == '?':
            return 'missing'
        code = code.strip()
        if not code:
            return 'missing'
        if code.startswith('E'):
            return code.split('.')[0][:4]
        return code.split('.')[0][:3]

    for col in diag_cols:
        train_feat[f'{col}_3d'] = np.array([parse_icd9_3digit(x) for x in train[col]], dtype=object)
        eval_feat[f'{col}_3d'] = np.array([parse_icd9_3digit(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(f'{col}_3d')
        
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
    
    # Key clinical primary diagnosis indicators (CMS Hospital Readmission Reduction Program - HRRP targets)
    def add_diag_flags(data_dict, target_dict):
        d1 = data_dict['diag_1']
        d2 = data_dict['diag_2']
        d3 = data_dict['diag_3']
        n_rows = len(d1)
        
        target_dict['primary_is_hf'] = np.array([1.0 if isinstance(x, str) and x.startswith('428') else 0.0 for x in d1], dtype=float)
        target_dict['primary_is_diabetes'] = np.array([1.0 if isinstance(x, str) and x.startswith('250') else 0.0 for x in d1], dtype=float)
        target_dict['primary_is_cad'] = np.array([1.0 if isinstance(x, str) and x.startswith(('410', '411', '412', '413', '414')) else 0.0 for x in d1], dtype=float)
        target_dict['primary_is_resp'] = np.array([1.0 if isinstance(x, str) and x.startswith(('480', '481', '482', '483', '484', '485', '486', '491', '492', '493', '496')) else 0.0 for x in d1], dtype=float)
        target_dict['primary_is_renal'] = np.array([1.0 if isinstance(x, str) and x.startswith(('584', '585', '586')) else 0.0 for x in d1], dtype=float)
        target_dict['primary_is_sepsis'] = np.array([1.0 if isinstance(x, str) and (x.startswith('038') or x in ('995.91', '995.92', '785.52')) else 0.0 for x in d1], dtype=float)
        
        # Acute diabetic complications (ketoacidosis / hyperosmolarity)
        target_dict['acute_diab_comp'] = np.array([
            1.0 if any(isinstance(c, str) and c.startswith(('250.1', '250.2', '250.3')) for c in (a, b, c_code)) else 0.0
            for a, b, c_code in zip(d1, d2, d3)
        ], dtype=float)
        
        # Combined CMS HRRP condition indicator
        target_dict['primary_is_hrrp'] = np.array([
            1.0 if isinstance(x, str) and x.startswith(('428', '410', '491', '492', '496', '480', '481', '482', '483', '484', '485', '486')) else 0.0
            for x in d1
        ], dtype=float)

    add_diag_flags(train, train_feat)
    add_diag_flags(eval_cols, eval_feat)
    extra_numeric.extend([
        'primary_is_hf', 'primary_is_diabetes', 'primary_is_cad', 'primary_is_resp',
        'primary_is_renal', 'primary_is_sepsis', 'acute_diab_comp', 'primary_is_hrrp'
    ])
        
    # 3. Admission/Discharge and Clinical Cross Interactions
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

    # Disposition and Age band interaction (captures high-risk elderly SNF/home-health transitions)
    train_feat['dis_age'] = np.array([
        f"{d}_{a}" for d, a in zip(train['discharge_disposition_id'], train['age'])
    ], dtype=object)
    eval_feat['dis_age'] = np.array([
        f"{d}_{a}" for d, a in zip(eval_cols['discharge_disposition_id'], eval_cols['age'])
    ], dtype=object)
    extra_categorical.append('dis_age')

    train_feat['adm_age'] = np.array([
        f"{ad}_{a}" for ad, a in zip(train['admission_type_id'], train['age'])
    ], dtype=object)
    eval_feat['adm_age'] = np.array([
        f"{ad}_{a}" for ad, a in zip(eval_cols['admission_type_id'], eval_cols['age'])
    ], dtype=object)
    extra_categorical.append('adm_age')

    train_feat['diag1_age'] = np.array([
        f"{dg}_{a}" for dg, a in zip(train_feat['diag_1_grp'], train['age'])
    ], dtype=object)
    eval_feat['diag1_age'] = np.array([
        f"{dg}_{a}" for dg, a in zip(eval_feat['diag_1_grp'], eval_cols['age'])
    ], dtype=object)
    extra_categorical.append('diag1_age')
        
    # 4. Target encoding of high-cardinality categorical variables
    high_card_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_3d', 'diag_2_3d', 'diag_3_3d',
        'medical_specialty', 
        'discharge_disposition_id', 'admission_source_id', 'admission_type_id',
        'payer_code', 'dis_adm', 'med_combo', 'change_med',
        'dis_diag1_grp', 'adm_dis', 'diag_1_grp',
        'dis_age', 'adm_age', 'diag1_age'
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
        
        # Regimen complexity and combinations
        insulin_plus_oral = (insulin_active * (num_active > 1).astype(np.float64))
        on_metformin_only = ((metformin_active == 1.0) & (num_active == 1.0)).astype(np.float64)
        on_insulin_only = ((insulin_active == 1.0) & (num_active == 1.0)).astype(np.float64)
        insulin_and_metformin = (insulin_active * metformin_active).astype(np.float64)
        insulin_and_sulf = (insulin_active * sulf_active).astype(np.float64)
        
        med_complexity_score = num_active + 2.0 * insulin_active + 1.5 * num_change
        med_up_ratio = num_up / (num_active + 1.0)
        med_down_ratio = num_down / (num_active + 1.0)
        has_med_change = (num_change > 0).astype(np.float64)
        multiple_med_changes = (num_change >= 2).astype(np.float64)
        
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
            'on_metformin_only': on_metformin_only,
            'on_insulin_only': on_insulin_only,
            'insulin_and_metformin': insulin_and_metformin,
            'insulin_and_sulf': insulin_and_sulf,
            'med_complexity_score': med_complexity_score,
            'med_up_ratio': med_up_ratio,
            'med_down_ratio': med_down_ratio,
            'has_med_change': has_med_change,
            'multiple_med_changes': multiple_med_changes
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
        
        # Non-linear log transforms
        target_dict['log_inpatient'] = np.log1p(inpatient)
        target_dict['log_emergency'] = np.log1p(emergency)
        target_dict['log_outpatient'] = np.log1p(outpatient)
        target_dict['log_total_visits'] = np.log1p(total_visits)
        
        lab_proc = data_dict.get('num_lab_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        proc = data_dict.get('num_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        meds = data_dict.get('num_medications', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        diags = data_dict.get('number_diagnoses', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        los = data_dict.get('time_in_hospital', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        
        severity_index = lab_proc + 2.0 * proc + meds + 5.0 * diags
        los_med_ratio = los / (meds + 1.0)
        inpatient_los = inpatient * los
        util_intensity = (inpatient * 3.0 + emergency * 1.5 + outpatient * 0.5) / (los + 1.0)
        
        target_dict['total_visits'] = total_visits
        target_dict['acute_visits'] = acute_visits
        target_dict['inpatient_ratio'] = inpatient_ratio
        target_dict['emergency_ratio'] = emergency_ratio
        target_dict['util_index'] = util_index
        target_dict['util_any'] = util_any
        target_dict['severity_index'] = severity_index
        target_dict['los_med_ratio'] = los_med_ratio
        target_dict['inpatient_los'] = inpatient_los
        target_dict['util_intensity'] = util_intensity
        
        target_dict['has_prior_inpatient'] = (inpatient > 0).astype(float)
        target_dict['has_prior_emergency'] = (emergency > 0).astype(float)
        target_dict['has_prior_outpatient'] = (outpatient > 0).astype(float)
        target_dict['frequent_inpatient'] = (inpatient >= 2).astype(float)
        target_dict['very_frequent_inpatient'] = (inpatient >= 3).astype(float)
        target_dict['frequent_emergency'] = (emergency >= 2).astype(float)
        
        # Length of stay patterns and ratios
        target_dict['stay_long'] = (los >= 7).astype(float)
        target_dict['stay_very_long'] = (los >= 10).astype(float)
        target_dict['stay_short'] = (los <= 2).astype(float)
        target_dict['los_diag_ratio'] = los / (diags + 1.0)
        target_dict['lab_per_med'] = lab_proc / (meds + 1.0)
        target_dict['meds_per_diag'] = meds / (diags + 1.0)
        target_dict['polypharmacy'] = (meds >= 15).astype(float)
        target_dict['heavy_polypharmacy'] = (meds >= 20).astype(float)
        
        # Clinical admission and discharge status flags
        disch = data_dict.get('discharge_disposition_id', np.array(['1'] * len(los)))
        admit_src = data_dict.get('admission_source_id', np.array(['1'] * len(los)))
        admit_type = data_dict.get('admission_type_id', np.array(['1'] * len(los)))
        
        target_dict['disch_home'] = (disch == '1').astype(float)
        target_dict['disch_not_home'] = (disch != '1').astype(float)
        target_dict['disch_snf'] = (disch == '3').astype(float)
        target_dict['disch_homehealth'] = (disch == '6').astype(float)
        target_dict['disch_ama'] = (disch == '7').astype(float)
        target_dict['disch_transfer'] = np.isin(disch, ['2', '4', '5', '22']).astype(float)
        target_dict['disch_post_acute'] = np.isin(disch, ['3', '4', '5', '6', '22', '23', '24']).astype(float)
        
        target_dict['admit_emergency'] = (admit_src == '7').astype(float)
        target_dict['admit_transfer'] = np.isin(admit_src, ['4', '5', '6']).astype(float)
        target_dict['adm_type_emergency'] = (admit_type == '1').astype(float)
        target_dict['adm_type_urgent'] = (admit_type == '2').astype(float)
        target_dict['adm_type_elective'] = (admit_type == '3').astype(float)
        target_dict['emerg_adm_post_acute_disch'] = (
            ((admit_type == '1') | (admit_src == '7')).astype(float) * target_dict['disch_post_acute']
        )
        
    add_custom_numeric(train, train_feat)
    add_custom_numeric(eval_cols, eval_feat)
    extra_numeric.extend([
        'log_inpatient', 'log_emergency', 'log_outpatient', 'log_total_visits',
        'total_visits', 'acute_visits', 'inpatient_ratio', 'emergency_ratio',
        'util_index', 'util_any', 'severity_index', 'los_med_ratio', 'inpatient_los', 'util_intensity',
        'has_prior_inpatient', 'has_prior_emergency', 'has_prior_outpatient',
        'frequent_inpatient', 'very_frequent_inpatient', 'frequent_emergency',
        'stay_long', 'stay_very_long', 'stay_short', 'los_diag_ratio', 'lab_per_med', 'meds_per_diag',
        'polypharmacy', 'heavy_polypharmacy',
        'disch_home', 'disch_not_home', 'disch_snf', 'disch_homehealth', 'disch_ama', 'disch_transfer', 'disch_post_acute',
        'admit_emergency', 'admit_transfer', 'adm_type_emergency', 'adm_type_urgent', 'adm_type_elective',
        'emerg_adm_post_acute_disch'
    ])
    
    # Inpatient interactions with age, Charlson score, medications, and SNF discharge
    def add_util_interactions(target_dict):
        inp = target_dict['has_prior_inpatient']
        target_dict['inpatient_x_charlson'] = target_dict['log_inpatient'] * target_dict['charlson_score']
        target_dict['inpatient_x_age'] = target_dict['log_inpatient'] * (target_dict['age_num'] / 50.0)
        target_dict['inpatient_x_disch_snf'] = inp * target_dict['disch_snf']
        target_dict['disch_snf_elderly'] = target_dict['disch_snf'] * (target_dict['age_num'] >= 65.0).astype(float)

    add_util_interactions(train_feat)
    add_util_interactions(eval_feat)
    extra_numeric.extend(['inpatient_x_charlson', 'inpatient_x_age', 'inpatient_x_disch_snf', 'disch_snf_elderly'])

    # 7. Demography, lab indicators, frailty, and ratios
    def add_frailty_features(data_dict, target_dict):
        age_arr = target_dict['age_num']
        meds = data_dict.get('num_medications', np.zeros_like(age_arr, dtype=float))
        diags = data_dict.get('number_diagnoses', np.zeros_like(age_arr, dtype=float))
        inp = data_dict.get('number_inpatient', np.zeros_like(age_arr, dtype=float))
        
        target_dict['frailty_index'] = (age_arr / 100.0) * (meds + 2.0 * diags)
        target_dict['elderly'] = (age_arr >= 75.0).astype(float)
        target_dict['age_gte_80'] = (age_arr >= 80.0).astype(float)
        target_dict['age_lt_50'] = (age_arr < 50.0).astype(float)
        target_dict['frailty_score'] = (age_arr / 100.0) * (target_dict['cm_count'] + 1.0) * (inp + 1.0)

    add_frailty_features(train, train_feat)
    add_frailty_features(eval_cols, eval_feat)
    extra_numeric.extend(['frailty_index', 'elderly', 'age_gte_80', 'age_lt_50', 'frailty_score'])
    
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
        
        glu_map = {'None': 0.0, 'Norm': 1.0, '>200': 2.0, '>300': 3.0}
        a1c_map = {'None': 0.0, 'Norm': 1.0, '>7': 2.0, '>8': 3.0}
        target_dict['glu_ord'] = np.array([glu_map.get(x, 0.0) for x in glu], dtype=float)
        target_dict['a1c_ord'] = np.array([a1c_map.get(x, 0.0) for x in a1c], dtype=float)
        
        # Clinical guideline interactions: abnormal HbA1c/glucose with vs without medication titration
        target_dict['a1c_abnormal_med_changed'] = target_dict['a1c_high'] * (chg == 'Ch').astype(float)
        target_dict['a1c_abnormal_med_not_changed'] = target_dict['a1c_high'] * (chg != 'Ch').astype(float)
        target_dict['glu_abnormal_med_not_changed'] = target_dict['glu_high'] * (chg != 'Ch').astype(float)
        target_dict['no_labs_tested'] = ((glu == 'None') & (a1c == 'None')).astype(float)

    add_lab_features(train, train_feat)
    add_lab_features(eval_cols, eval_feat)
    extra_numeric.extend([
        'glu_high', 'glu_tested', 'a1c_high', 'a1c_tested', 'glu_ord', 'a1c_ord',
        'a1c_abnormal_med_changed', 'a1c_abnormal_med_not_changed', 'glu_abnormal_med_not_changed', 'no_labs_tested'
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

    # Diverse Multi-Model LightGBM Ensemble (Weight: 0.52)
    lgb_models = [
        lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.02,
            num_leaves=31,
            max_depth=6,
            min_child_samples=100,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.1,
            reg_lambda=1.5,
            cat_smooth=40.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=340,
            learning_rate=0.018,
            num_leaves=22,
            max_depth=5,
            min_child_samples=160,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_alpha=0.05,
            reg_lambda=3.0,
            cat_smooth=60.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.02,
            num_leaves=28,
            max_depth=6,
            min_child_samples=120,
            scale_pos_weight=1.4,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0.2,
            reg_lambda=1.0,
            cat_smooth=35.0,
            random_state=2023,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.022,
            num_leaves=40,
            max_depth=7,
            min_child_samples=80,
            subsample=0.65,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=0.8,
            cat_smooth=30.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        )
    ]

    # XGBoost Ensemble (Weight: 0.30)
    xgb_models = [
        xgb.XGBClassifier(
            n_estimators=280,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.75,
            colsample_bytree=0.6,
            min_child_weight=8.0,
            reg_lambda=3.0,
            reg_alpha=0.5,
            random_state=42,
            tree_method='hist',
            n_jobs=1,
            eval_metric='logloss'
        ),
        xgb.XGBClassifier(
            n_estimators=320,
            max_depth=4,
            learning_rate=0.018,
            subsample=0.8,
            colsample_bytree=0.5,
            min_child_weight=12.0,
            reg_lambda=4.0,
            reg_alpha=1.0,
            random_state=101,
            tree_method='hist',
            n_jobs=1,
            eval_metric='logloss'
        ),
        xgb.XGBClassifier(
            n_estimators=260,
            max_depth=6,
            learning_rate=0.022,
            subsample=0.7,
            colsample_bytree=0.65,
            min_child_weight=6.0,
            scale_pos_weight=1.3,
            reg_lambda=2.0,
            reg_alpha=0.3,
            random_state=2024,
            tree_method='hist',
            n_jobs=1,
            eval_metric='logloss'
        )
    ]

    # HistGradientBoosting Ensemble (Weight: 0.18)
    hgb_models = [
        HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.02,
            max_leaf_nodes=28,
            min_samples_leaf=120,
            l2_regularization=2.0,
            random_state=42
        ),
        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.018,
            max_leaf_nodes=20,
            min_samples_leaf=160,
            l2_regularization=3.5,
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
    
    # Rank normalization of family predictions to align marginal distributions for ROC-AUC
    n_eval = float(len(x_eval))
    lgb_rank = rankdata(lgb_preds) / n_eval
    xgb_rank = rankdata(xgb_preds) / n_eval
    hgb_rank = rankdata(hgb_preds) / n_eval

    # Weighted Rank Average Blend
    preds = 0.52 * lgb_rank + 0.30 * xgb_rank + 0.18 * hgb_rank
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
