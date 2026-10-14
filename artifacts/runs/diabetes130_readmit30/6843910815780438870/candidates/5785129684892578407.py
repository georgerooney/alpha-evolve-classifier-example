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


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    import lightgbm as lgb
    import xgboost as xgb
    
    n_train = len(y_train)
    first_col = next(iter(eval_cols.values()))
    n_eval = len(first_col)

    train_feat = dict(train)
    eval_feat = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Comorbidities via icd9_comorbidities & Weighted Charlson Score
    train_cm = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    eval_cm = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k in train_cm:
        train_feat[k] = train_cm[k]
        eval_feat[k] = eval_cm[k]
        extra_numeric.append(k)

    for cm_dict, feat_dict in [(train_cm, train_feat), (eval_cm, eval_feat)]:
        # Charlson Comorbidity Index weighting
        feat_dict['charlson_score'] = (
            1.0 * (cm_dict['cm_mi'] + cm_dict['cm_chf'] + cm_dict['cm_pvd'] + cm_dict['cm_cerebrovascular'] +
                   cm_dict['cm_dementia'] + cm_dict['cm_pulmonary'] + cm_dict['cm_rheumatic'] +
                   cm_dict['cm_peptic_ulcer'] + cm_dict['cm_liver']) +
            2.0 * (cm_dict['cm_hemiplegia'] + cm_dict['cm_renal'] + cm_dict['cm_diabetes_complicated'] + cm_dict['cm_cancer']) +
            6.0 * (cm_dict['cm_metastatic'] + cm_dict['cm_hiv'])
        )
        feat_dict['cm_cardiorenal'] = cm_dict['cm_chf'] * cm_dict['cm_renal']
        feat_dict['cm_cardiopulmonary'] = cm_dict['cm_chf'] * cm_dict['cm_pulmonary']
        feat_dict['cm_renometabolic'] = cm_dict['cm_renal'] * cm_dict['cm_diabetes_complicated']
        feat_dict['cm_high_burden'] = (cm_dict['cm_count'] >= 3.0).astype(np.float64)

    extra_numeric.extend(['charlson_score', 'cm_cardiorenal', 'cm_cardiopulmonary', 'cm_renometabolic', 'cm_high_burden'])

    # 2. ICD-9 Chapter groups, multi-organ count and 3-digit prefix categories
    def clean_prefix(val: Any) -> str:
        if val is None or not isinstance(val, str) or val in ('?', '', 'None'):
            return 'missing'
        val = val.strip()
        if '.' in val:
            return val.split('.')[0]
        return val[:3] if len(val) >= 3 else val

    def match_prefix_any(diag_lists: List[np.ndarray], prefixes: Tuple[str, ...]) -> np.ndarray:
        n = len(diag_lists[0])
        res = np.zeros(n, dtype=np.float64)
        for d_arr in diag_lists:
            for i in range(n):
                c = d_arr[i]
                if c is not None and isinstance(c, str):
                    s = c.strip()
                    if s.startswith(prefixes):
                        res[i] = 1.0
        return res

    for d_col in ['diag_1', 'diag_2', 'diag_3']:
        grp_col = f"{d_col}_group"
        train_feat[grp_col] = np.array([icd9_group(c) for c in train[d_col]], dtype=object)
        eval_feat[grp_col] = np.array([icd9_group(c) for c in eval_cols[d_col]], dtype=object)
        extra_categorical.append(grp_col)

        # Target encode chapter group
        grp_te_name = f"{grp_col}_te"
        tr_grp_te, ev_grp_te = oof_target_encode(
            train_feat[grp_col], y_train, eval_feat[grp_col], n_folds=5, smoothing=20.0, seed=42
        )
        train_feat[grp_te_name] = tr_grp_te.astype(np.float64)
        eval_feat[grp_te_name] = ev_grp_te.astype(np.float64)
        extra_numeric.append(grp_te_name)

        pfx_col = f"{d_col}_pfx"
        train_feat[pfx_col] = np.array([clean_prefix(c) for c in train[d_col]], dtype=object)
        eval_feat[pfx_col] = np.array([clean_prefix(c) for c in eval_cols[d_col]], dtype=object)
        extra_categorical.append(pfx_col)

        # Target encode the 3-digit prefixes
        te_pfx_name = f"{pfx_col}_te"
        tr_pfx_te, ev_pfx_te = oof_target_encode(
            train_feat[pfx_col], y_train, eval_feat[pfx_col], n_folds=5, smoothing=20.0, seed=42
        )
        train_feat[te_pfx_name] = tr_pfx_te.astype(np.float64)
        eval_feat[te_pfx_name] = ev_pfx_te.astype(np.float64)
        extra_numeric.append(te_pfx_name)

    # Diagnosis group indicators
    for grp in ['circulatory', 'respiratory', 'digestive', 'diabetes', 'injury', 'musculoskeletal', 'genitourinary', 'neoplasms']:
        grp_name = f'has_grp_{grp}'
        train_feat[grp_name] = (
            (train_feat['diag_1_group'] == grp) |
            (train_feat['diag_2_group'] == grp) |
            (train_feat['diag_3_group'] == grp)
        ).astype(np.float64)
        eval_feat[grp_name] = (
            (eval_feat['diag_1_group'] == grp) |
            (eval_feat['diag_2_group'] == grp) |
            (eval_feat['diag_3_group'] == grp)
        ).astype(np.float64)
        extra_numeric.append(grp_name)

    # Distinct organ systems involved across diagnoses
    for d_src, d_dst, n_rows in [(train, train_feat, n_train), (eval_cols, eval_feat, n_eval)]:
        dist_organs = np.zeros(n_rows, dtype=np.float64)
        for i in range(n_rows):
            g_set = {d_dst['diag_1_group'][i], d_dst['diag_2_group'][i], d_dst['diag_3_group'][i]} - {'missing', 'other'}
            dist_organs[i] = len(g_set)
        d_dst['num_distinct_organ_systems'] = dist_organs

    extra_numeric.append('num_distinct_organ_systems')

    # Prefix summary across diagnoses
    train_feat['mean_diag_pfx_te'] = (train_feat['diag_1_pfx_te'] + train_feat['diag_2_pfx_te'] + train_feat['diag_3_pfx_te']) / 3.0
    eval_feat['mean_diag_pfx_te'] = (eval_feat['diag_1_pfx_te'] + eval_feat['diag_2_pfx_te'] + eval_feat['diag_3_pfx_te']) / 3.0
    train_feat['max_diag_pfx_te'] = np.maximum(np.maximum(train_feat['diag_1_pfx_te'], train_feat['diag_2_pfx_te']), train_feat['diag_3_pfx_te'])
    eval_feat['max_diag_pfx_te'] = np.maximum(np.maximum(eval_feat['diag_1_pfx_te'], eval_feat['diag_2_pfx_te']), eval_feat['diag_3_pfx_te'])
    extra_numeric.extend(['mean_diag_pfx_te', 'max_diag_pfx_te'])

    # Specific Chronic Disease Signatures
    tr_diags = [train['diag_1'], train['diag_2'], train['diag_3']]
    ev_diags = [eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3']]

    clin_pfx_map = {
        'has_chf_code': ('428',),
        'has_ckd_code': ('585', '586'),
        'has_arf_code': ('584',),
        'has_copd_code': ('491', '492', '493', '494', '496'),
        'has_diabetes_code': ('250',),
        'has_dka_hhs_code': ('250.1', '250.2', '250.3'),
        'has_cad_code': ('410', '411', '412', '413', '414'),
        'has_dysrhythmia_code': ('427',),
        'has_htn_code': ('401', '402', '403', '404', '405'),
        'has_sepsis_code': ('038', '995.9'),
        'has_pneumonia_code': ('480', '481', '482', '483', '484', '485', '486'),
        'has_ulcer_code': ('707', '681', '682'),
    }

    for col_n, pfx_tup in clin_pfx_map.items():
        train_feat[col_n] = match_prefix_any(tr_diags, pfx_tup)
        eval_feat[col_n] = match_prefix_any(ev_diags, pfx_tup)
        extra_numeric.append(col_n)

    # Primary diagnosis specifics
    for d_src, d_dst in [(train, train_feat), (eval_cols, eval_feat)]:
        d1 = d_src['diag_1']
        d_dst['diag1_is_chf'] = np.array([1.0 if (c is not None and str(c).strip().startswith('428')) else 0.0 for c in d1], dtype=np.float64)
        d_dst['diag1_is_diabetes'] = np.array([1.0 if (c is not None and str(c).strip().startswith('250')) else 0.0 for c in d1], dtype=np.float64)
        d_dst['diag1_is_cad'] = np.array([1.0 if (c is not None and str(c).strip().startswith(('410', '411', '412', '413', '414'))) else 0.0 for c in d1], dtype=np.float64)
        d_dst['diag1_is_copd'] = np.array([1.0 if (c is not None and str(c).strip().startswith(('491', '492', '493', '494', '496'))) else 0.0 for c in d1], dtype=np.float64)

        d_dst['cardiorenal_clinical'] = d_dst['has_chf_code'] * np.maximum(d_dst['has_ckd_code'], d_dst['has_arf_code'])
        d_dst['triad_cardiorenal_diabetes'] = d_dst['cardiorenal_clinical'] * d_dst['has_diabetes_code']

    extra_numeric.extend([
        'diag1_is_chf', 'diag1_is_diabetes', 'diag1_is_cad', 'diag1_is_copd',
        'cardiorenal_clinical', 'triad_cardiorenal_diabetes'
    ])

    # Diagnosis prefix target encoding scaled by prior inpatient visits
    train_feat['mean_diag_te_inpatient'] = train_feat['mean_diag_pfx_te'] * train['number_inpatient'].astype(np.float64)
    eval_feat['mean_diag_te_inpatient'] = eval_feat['mean_diag_pfx_te'] * eval_cols['number_inpatient'].astype(np.float64)
    extra_numeric.append('mean_diag_te_inpatient')

    # 3. Admission/discharge key clinical indicators (Denoised, no combinatorial string bloat)
    for d_src, d_dst in [(train, train_feat), (eval_cols, eval_feat)]:
        disp = d_src['discharge_disposition_id']
        adm_t = d_src['admission_type_id']
        adm_s = d_src['admission_source_id']

        d_dst['disch_home'] = (disp == '1').astype(np.float64)
        d_dst['disch_snf'] = ((disp == '3') | (disp == '4') | (disp == '5')).astype(np.float64)
        d_dst['disch_home_health'] = (disp == '6').astype(np.float64)
        d_dst['disch_rehab'] = (disp == '22').astype(np.float64)
        d_dst['disch_ama'] = (disp == '7').astype(np.float64)
        d_dst['disch_transfer'] = ((disp == '2') | (disp == '3') | (disp == '4') | (disp == '5') | (disp == '22') | (disp == '28')).astype(np.float64)

        d_dst['adm_emergency'] = (adm_t == '1').astype(np.float64)
        d_dst['adm_urgent'] = (adm_t == '2').astype(np.float64)
        d_dst['adm_elective'] = (adm_t == '3').astype(np.float64)
        d_dst['source_er'] = (adm_s == '7').astype(np.float64)
        d_dst['source_snf'] = (adm_s == '5').astype(np.float64)
        d_dst['source_transfer'] = ((adm_s == '4') | (adm_s == '5') | (adm_s == '6')).astype(np.float64)

    clin_trans_cols = [
        'disch_home', 'disch_snf', 'disch_home_health', 'disch_rehab', 'disch_ama', 'disch_transfer',
        'adm_emergency', 'adm_urgent', 'adm_elective', 'source_er', 'source_snf', 'source_transfer'
    ]
    extra_numeric.extend(clin_trans_cols)

    # 4. Out-of-fold target encoding for primary categoricals only
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'discharge_disposition_id', 'admission_source_id', 'admission_type_id',
        'medical_specialty', 'payer_code', 'race'
    ]
    for col in te_cols:
        if col in train and col in eval_cols:
            te_name = f"{col}_te"
            tr_te, ev_te = oof_target_encode(
                train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42
            )
            train_feat[te_name] = tr_te.astype(np.float64)
            eval_feat[te_name] = ev_te.astype(np.float64)
            extra_numeric.append(te_name)

    train_feat['max_diag_te'] = np.maximum(np.maximum(train_feat['diag_1_te'], train_feat['diag_2_te']), train_feat['diag_3_te'])
    eval_feat['max_diag_te'] = np.maximum(np.maximum(eval_feat['diag_1_te'], eval_feat['diag_2_te']), eval_feat['diag_3_te'])
    train_feat['mean_diag_te'] = (train_feat['diag_1_te'] + train_feat['diag_2_te'] + train_feat['diag_3_te']) / 3.0
    eval_feat['mean_diag_te'] = (eval_feat['diag_1_te'] + eval_feat['diag_2_te'] + eval_feat['diag_3_te']) / 3.0
    extra_numeric.extend(['max_diag_te', 'mean_diag_te'])

    # 4.5. Out-of-fold target encoding for key categorical interactions
    # Interaction: discharge and admission source
    train_feat['disp_source_comb'] = train['discharge_disposition_id'].astype(str) + "_" + train['admission_source_id'].astype(str)
    eval_feat['disp_source_comb'] = eval_cols['discharge_disposition_id'].astype(str) + "_" + eval_cols['admission_source_id'].astype(str)
    extra_categorical.append('disp_source_comb')
    
    tr_comb_te, ev_comb_te = oof_target_encode(
        train_feat['disp_source_comb'], y_train, eval_feat['disp_source_comb'], n_folds=5, smoothing=20.0, seed=42
    )
    train_feat['disp_source_comb_te'] = tr_comb_te.astype(np.float64)
    eval_feat['disp_source_comb_te'] = ev_comb_te.astype(np.float64)
    extra_numeric.append('disp_source_comb_te')

    # Interaction: age and gender
    train_feat['age_gender_comb'] = train['age'].astype(str) + "_" + train['gender'].astype(str)
    eval_feat['age_gender_comb'] = eval_cols['age'].astype(str) + "_" + eval_cols['gender'].astype(str)
    extra_categorical.append('age_gender_comb')
    
    tr_ag_te, ev_ag_te = oof_target_encode(
        train_feat['age_gender_comb'], y_train, eval_feat['age_gender_comb'], n_folds=5, smoothing=20.0, seed=42
    )
    train_feat['age_gender_comb_te'] = tr_ag_te.astype(np.float64)
    eval_feat['age_gender_comb_te'] = ev_ag_te.astype(np.float64)
    extra_numeric.append('age_gender_comb_te')

    # Interaction: diag1_group and discharge_disposition
    train_feat['diag1_disp_comb'] = train_feat['diag_1_group'].astype(str) + "_" + train['discharge_disposition_id'].astype(str)
    eval_feat['diag1_disp_comb'] = eval_feat['diag_1_group'].astype(str) + "_" + eval_cols['discharge_disposition_id'].astype(str)
    extra_categorical.append('diag1_disp_comb')
    
    tr_d1d_te, ev_d1d_te = oof_target_encode(
        train_feat['diag1_disp_comb'], y_train, eval_feat['diag1_disp_comb'], n_folds=5, smoothing=20.0, seed=42
    )
    train_feat['diag1_disp_comb_te'] = tr_d1d_te.astype(np.float64)
    eval_feat['diag1_disp_comb_te'] = ev_d1d_te.astype(np.float64)
    extra_numeric.append('diag1_disp_comb_te')

    # Interaction: specialty and admission_type
    train_feat['spec_adm_comb'] = train['medical_specialty'].astype(str) + "_" + train['admission_type_id'].astype(str)
    eval_feat['spec_adm_comb'] = eval_cols['medical_specialty'].astype(str) + "_" + eval_cols['admission_type_id'].astype(str)
    extra_categorical.append('spec_adm_comb')
    
    tr_sa_te, ev_sa_te = oof_target_encode(
        train_feat['spec_adm_comb'], y_train, eval_feat['spec_adm_comb'], n_folds=5, smoothing=20.0, seed=42
    )
    train_feat['spec_adm_comb_te'] = tr_sa_te.astype(np.float64)
    eval_feat['spec_adm_comb_te'] = ev_sa_te.astype(np.float64)
    extra_numeric.append('spec_adm_comb_te')

    # Interaction: admission_type and admission_source
    train_feat['adm_type_src_comb'] = train['admission_type_id'].astype(str) + "_" + train['admission_source_id'].astype(str)
    eval_feat['adm_type_src_comb'] = eval_cols['admission_type_id'].astype(str) + "_" + eval_cols['admission_source_id'].astype(str)
    extra_categorical.append('adm_type_src_comb')
    tr_ats_te, ev_ats_te = oof_target_encode(
        train_feat['adm_type_src_comb'], y_train, eval_feat['adm_type_src_comb'], n_folds=5, smoothing=20.0, seed=42
    )
    train_feat['adm_type_src_comb_te'] = tr_ats_te.astype(np.float64)
    eval_feat['adm_type_src_comb_te'] = ev_ats_te.astype(np.float64)
    extra_numeric.append('adm_type_src_comb_te')

    # Interaction: discharge_disposition and payer_code
    train_feat['disp_payer_comb'] = train['discharge_disposition_id'].astype(str) + "_" + train['payer_code'].astype(str)
    eval_feat['disp_payer_comb'] = eval_cols['discharge_disposition_id'].astype(str) + "_" + eval_cols['payer_code'].astype(str)
    extra_categorical.append('disp_payer_comb')
    tr_dp_te, ev_dp_te = oof_target_encode(
        train_feat['disp_payer_comb'], y_train, eval_feat['disp_payer_comb'], n_folds=5, smoothing=20.0, seed=42
    )
    train_feat['disp_payer_comb_te'] = tr_dp_te.astype(np.float64)
    eval_feat['disp_payer_comb_te'] = ev_dp_te.astype(np.float64)
    extra_numeric.append('disp_payer_comb_te')

    # Interaction: diag1_group and prior inpatient status
    tr_inp_cat = np.where(train['number_inpatient'] == 0, '0', np.where(train['number_inpatient'] == 1, '1', '2+'))
    ev_inp_cat = np.where(eval_cols['number_inpatient'] == 0, '0', np.where(eval_cols['number_inpatient'] == 1, '1', '2+'))
    train_feat['diag1_inp_comb'] = train_feat['diag_1_group'].astype(str) + "_" + tr_inp_cat
    eval_feat['diag1_inp_comb'] = eval_feat['diag_1_group'].astype(str) + "_" + ev_inp_cat
    extra_categorical.append('diag1_inp_comb')
    tr_d1i_te, ev_d1i_te = oof_target_encode(
        train_feat['diag1_inp_comb'], y_train, eval_feat['diag1_inp_comb'], n_folds=5, smoothing=20.0, seed=42
    )
    train_feat['diag1_inp_comb_te'] = tr_d1i_te.astype(np.float64)
    eval_feat['diag1_inp_comb_te'] = ev_d1i_te.astype(np.float64)
    extra_numeric.append('diag1_inp_comb_te')

    # Interaction: inpatient visits and discharge disposition target encoding
    train_feat['inpatient_disp_te'] = train['number_inpatient'].astype(np.float64) * train_feat['discharge_disposition_id_te']
    eval_feat['inpatient_disp_te'] = eval_cols['number_inpatient'].astype(np.float64) * eval_feat['discharge_disposition_id_te']
    extra_numeric.append('inpatient_disp_te')

    # 5. Frequency encoding for high-cardinality features
    for f_col in ['medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'diag_1_pfx']:
        vals, counts = np.unique(train_feat[f_col].astype(str), return_counts=True)
        f_map = dict(zip(vals, counts.astype(np.float64) / n_train))
        fq_name = f"{f_col}_freq"
        train_feat[fq_name] = np.array([f_map.get(str(x), 0.0) for x in train_feat[f_col]], dtype=np.float64)
        eval_feat[fq_name] = np.array([f_map.get(str(x), 0.0) for x in eval_feat[f_col]], dtype=np.float64)
        extra_numeric.append(fq_name)

    # 6. Demographics, Age mapping & missingness signals
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0,
    }
    train_age = np.array([age_map.get(str(x), np.nan) for x in train['age']], dtype=np.float64)
    eval_age = np.array([age_map.get(str(x), np.nan) for x in eval_cols['age']], dtype=np.float64)
    train_feat['age_num'] = train_age
    eval_feat['age_num'] = eval_age

    for d_src, d_dst, age_arr in [(train, train_feat, train_age), (eval_cols, eval_feat, eval_age)]:
        d_dst['has_weight'] = np.array([1.0 if (w is not None and str(w) not in ('?', '', 'None')) else 0.0 for w in d_src['weight']], dtype=np.float64)
        d_dst['has_specialty'] = np.array([1.0 if (s is not None and str(s) not in ('?', '', 'None')) else 0.0 for s in d_src['medical_specialty']], dtype=np.float64)
        d_dst['has_payer'] = np.array([1.0 if (p is not None and str(p) not in ('?', '', 'None')) else 0.0 for p in d_src['payer_code']], dtype=np.float64)
        inp = d_src['number_inpatient'].astype(np.float64)
        th = d_src['time_in_hospital'].astype(np.float64)
        d_dst['elderly_prior_inpatient'] = (age_arr >= 70.0).astype(np.float64) * inp
        d_dst['age_time_in_hospital'] = age_arr * th
        d_dst['gender_is_female'] = (d_src['gender'] == 'Female').astype(np.float64)
        d_dst['is_elderly'] = (age_arr >= 75.0).astype(np.float64)
        d_dst['is_very_elderly'] = (age_arr >= 85.0).astype(np.float64)
        # Age-Adjusted Charlson Comorbidity Index (ACCI)
        age_pts = np.where(age_arr >= 80.0, 4.0,
                  np.where(age_arr >= 70.0, 3.0,
                  np.where(age_arr >= 60.0, 2.0,
                  np.where(age_arr >= 50.0, 1.0, 0.0))))
        d_dst['acci_score'] = d_dst['charlson_score'] + age_pts

    extra_numeric.extend(['age_num', 'has_weight', 'has_specialty', 'has_payer', 'elderly_prior_inpatient', 'age_time_in_hospital', 'gender_is_female', 'is_elderly', 'is_very_elderly', 'acci_score'])

    # 7. Healthcare utilization & hospital care intensity
    for d_src, d_dst in [(train, train_feat), (eval_cols, eval_feat)]:
        inp = d_src['number_inpatient'].astype(np.float64)
        emg = d_src['number_emergency'].astype(np.float64)
        outp = d_src['number_outpatient'].astype(np.float64)
        time_hosp = np.maximum(d_src['time_in_hospital'].astype(np.float64), 1.0)
        num_meds = d_src['num_medications'].astype(np.float64)
        num_labs = d_src['num_lab_procedures'].astype(np.float64)
        num_procs = d_src['num_procedures'].astype(np.float64)
        num_diags = d_src['number_diagnoses'].astype(np.float64)

        tot_prior = inp + emg + outp
        d_dst['total_prior_visits'] = tot_prior
        d_dst['has_prior_inpatient'] = (inp > 0).astype(np.float64)
        d_dst['frequent_inpatient'] = (inp >= 2).astype(np.float64)
        d_dst['has_prior_emergency'] = (emg > 0).astype(np.float64)
        d_dst['has_any_prior_visit'] = (tot_prior > 0).astype(np.float64)
        d_dst['heavy_utilizer'] = (tot_prior >= 4).astype(np.float64)
        d_dst['inpatient_ratio'] = np.where(tot_prior > 0, inp / tot_prior, 0.0)
        d_dst['acute_prior_visits'] = inp + emg
        d_dst['weighted_prior_acuity'] = inp * 3.0 + emg * 1.5 + outp * 0.5

        d_dst['log_inpatient'] = np.log1p(inp)
        d_dst['log_emergency'] = np.log1p(emg)
        d_dst['log_outpatient'] = np.log1p(outp)
        d_dst['log_total_prior'] = np.log1p(tot_prior)

        d_dst['lab_per_day'] = num_labs / time_hosp
        d_dst['meds_per_day'] = num_meds / time_hosp
        d_dst['procs_per_day'] = num_procs / time_hosp
        d_dst['diags_per_day'] = num_diags / time_hosp
        d_dst['total_procedures'] = num_labs + num_procs
        d_dst['med_proc_ratio'] = num_meds / np.maximum(num_procs, 1.0)
        d_dst['severity_index'] = num_meds * time_hosp
        d_dst['inpatient_stay_interaction'] = inp * time_hosp

        d_dst['meds_to_diags'] = num_meds / np.maximum(num_diags, 1.0)
        d_dst['labs_to_diags'] = num_labs / np.maximum(num_diags, 1.0)
        d_dst['procs_to_diags'] = num_procs / np.maximum(num_diags, 1.0)

        d_dst['inp_sq'] = inp ** 2
        d_dst['inp_to_time_hosp'] = inp / time_hosp
        d_dst['inp_to_meds'] = inp / np.maximum(num_meds, 1.0)
        d_dst['emerg_to_time_hosp'] = emg / time_hosp
        d_dst['outp_to_time_hosp'] = outp / time_hosp
        d_dst['utilization_risk_score'] = (
            (inp >= 1.0).astype(np.float64) * 2.0 +
            (inp >= 2.0).astype(np.float64) * 2.0 +
            (emg >= 1.0).astype(np.float64) * 1.5 +
            (time_hosp >= 4.0).astype(np.float64) * 1.0
        )

        d_dst['inpatient_with_chf'] = (inp > 0).astype(np.float64) * d_dst['has_chf_code']
        d_dst['inpatient_with_renal'] = (inp > 0).astype(np.float64) * np.maximum(d_dst['has_ckd_code'], d_dst['has_arf_code'])
        d_dst['inpatient_with_disch_snf'] = (inp > 0).astype(np.float64) * d_dst['disch_snf']
        d_dst['inpatient_with_emergent'] = (inp > 0).astype(np.float64) * d_dst['adm_emergency']

        # Validated Clinical Indices: LACE Index & HOSPITAL Score
        l_pts = np.where(time_hosp < 1.0, 0.0,
                np.where(time_hosp == 1.0, 1.0,
                np.where(time_hosp == 2.0, 2.0,
                np.where(time_hosp == 3.0, 3.0,
                np.where(time_hosp <= 6.0, 4.0,
                np.where(time_hosp <= 13.0, 5.0, 7.0))))))
        a_pts = 3.0 * d_dst['adm_emergency']
        ch = d_dst['charlson_score']
        c_pts = np.where(ch <= 0.0, 0.0,
                np.where(ch == 1.0, 1.0,
                np.where(ch == 2.0, 2.0,
                np.where(ch == 3.0, 3.0, 5.0))))
        e_pts = np.minimum(emg, 4.0)

        lace = l_pts + a_pts + c_pts + e_pts
        d_dst['lace_index'] = lace
        d_dst['lace_high_risk'] = (lace >= 10.0).astype(np.float64)

        cm_d = train_cm if d_src is train else eval_cm
        hosp_score = (
            2.0 * ((cm_d['cm_cancer'] > 0) | (cm_d['cm_metastatic'] > 0)).astype(np.float64) +
            1.0 * (num_procs > 0).astype(np.float64) +
            1.0 * d_dst['adm_emergency'] +
            np.where(inp >= 5.0, 5.0, np.where(inp >= 1.0, 2.0, 0.0)) +
            2.0 * (time_hosp >= 5.0).astype(np.float64)
        )
        d_dst['hospital_score'] = hosp_score
        d_dst['hospital_high_risk'] = (hosp_score >= 7.0).astype(np.float64)
        d_dst['clinical_fragility_ratio'] = (ch + 1.0) * (inp + emg + 1.0) / (time_hosp + 1.0)

    util_cols = [
        'total_prior_visits', 'has_prior_inpatient', 'frequent_inpatient', 'has_prior_emergency',
        'has_any_prior_visit', 'heavy_utilizer', 'inpatient_ratio', 'acute_prior_visits',
        'weighted_prior_acuity', 'log_inpatient', 'log_emergency', 'log_outpatient', 'log_total_prior',
        'lab_per_day', 'meds_per_day', 'procs_per_day', 'diags_per_day', 'total_procedures',
        'med_proc_ratio', 'severity_index', 'inpatient_stay_interaction',
        'meds_to_diags', 'labs_to_diags', 'procs_to_diags',
        'inp_sq', 'inp_to_time_hosp', 'inp_to_meds', 'emerg_to_time_hosp', 'outp_to_time_hosp',
        'utilization_risk_score', 'inpatient_with_chf', 'inpatient_with_renal', 'inpatient_with_disch_snf', 'inpatient_with_emergent',
        'lace_index', 'lace_high_risk', 'hospital_score', 'hospital_high_risk', 'clinical_fragility_ratio'
    ]
    extra_numeric.extend(util_cols)

    # 8. Medication patterns, specific agents & combination regimens
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    avail_meds = [m for m in med_names if m in train and m in eval_cols]
    sulf_cols = ['glimepiride', 'glipizide', 'glyburide', 'tolbutamide', 'chlorpropamide', 'acetohexamide', 'tolazamide']

    med_ord_map = {'No': 0.0, 'Down': 1.0, 'Steady': 2.0, 'Up': 3.0}

    for d_src, d_dst, n_rows in [(train, train_feat, n_train), (eval_cols, eval_feat, n_eval)]:
        n_act = np.zeros(n_rows, dtype=np.float64)
        n_up = np.zeros(n_rows, dtype=np.float64)
        n_dn = np.zeros(n_rows, dtype=np.float64)
        oral_chg = np.zeros(n_rows, dtype=np.float64)

        for m in avail_meds:
            arr = d_src[m]
            for val in ['Steady', 'Up', 'Down']:
                n_act += (arr == val).astype(np.float64)
            n_up += (arr == 'Up').astype(np.float64)
            n_dn += (arr == 'Down').astype(np.float64)
            if m != 'insulin':
                oral_chg += ((arr == 'Up') | (arr == 'Down')).astype(np.float64)

        d_dst['num_meds_active'] = n_act
        d_dst['num_meds_up'] = n_up
        d_dst['num_meds_down'] = n_dn
        d_dst['num_meds_changed'] = n_up + n_dn
        d_dst['oral_med_changed'] = (oral_chg > 0).astype(np.float64)
        num_prescribed = d_src['num_medications'].astype(np.float64)
        d_dst['non_diabetic_meds'] = np.maximum(num_prescribed - n_act, 0.0)
        d_dst['diabetic_med_share'] = n_act / np.maximum(num_prescribed, 1.0)

        if 'insulin' in d_src:
            ins = d_src['insulin']
            d_dst['insulin_active'] = ((ins == 'Steady') | (ins == 'Up') | (ins == 'Down')).astype(np.float64)
            d_dst['insulin_up'] = (ins == 'Up').astype(np.float64)
            d_dst['insulin_down'] = (ins == 'Down').astype(np.float64)
            d_dst['insulin_steady'] = (ins == 'Steady').astype(np.float64)
            d_dst['insulin_ord'] = np.array([med_ord_map.get(str(x), 0.0) for x in ins], dtype=np.float64)
        else:
            d_dst['insulin_active'] = np.zeros(n_rows, dtype=np.float64)
            d_dst['insulin_up'] = np.zeros(n_rows, dtype=np.float64)
            d_dst['insulin_down'] = np.zeros(n_rows, dtype=np.float64)
            d_dst['insulin_steady'] = np.zeros(n_rows, dtype=np.float64)
            d_dst['insulin_ord'] = np.zeros(n_rows, dtype=np.float64)

        d_dst['insulin_changed'] = d_dst['insulin_up'] + d_dst['insulin_down']
        d_dst['both_ins_and_oral_changed'] = ((d_dst['insulin_changed'] > 0) & (d_dst['oral_med_changed'] > 0)).astype(np.float64)

        # Key medication ordinals
        for m_k in ['metformin', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone']:
            if m_k in d_src:
                d_dst[f"{m_k}_ord"] = np.array([med_ord_map.get(str(x), 0.0) for x in d_src[m_k]], dtype=np.float64)
            else:
                d_dst[f"{m_k}_ord"] = np.zeros(n_rows, dtype=np.float64)

        if 'metformin' in d_src:
            met = d_src['metformin']
            d_dst['metformin_active'] = ((met == 'Steady') | (met == 'Up') | (met == 'Down')).astype(np.float64)
        else:
            d_dst['metformin_active'] = np.zeros(n_rows, dtype=np.float64)

        has_sulf = np.zeros(n_rows, dtype=np.float64)
        for s_c in sulf_cols:
            if s_c in d_src:
                s_arr = d_src[s_c]
                has_sulf = np.maximum(has_sulf, ((s_arr == 'Steady') | (s_arr == 'Up') | (s_arr == 'Down')).astype(np.float64))
        d_dst['sulfonylurea_active'] = has_sulf

        meg_cols = ['repaglinide', 'nateglinide']
        tzd_cols = ['pioglitazone', 'rosiglitazone', 'troglitazone']

        has_meg = np.zeros(n_rows, dtype=np.float64)
        for m_c in meg_cols:
            if m_c in d_src:
                m_arr = d_src[m_c]
                has_meg = np.maximum(has_meg, ((m_arr == 'Steady') | (m_arr == 'Up') | (m_arr == 'Down')).astype(np.float64))
        d_dst['meglitinide_active'] = has_meg

        has_tzd = np.zeros(n_rows, dtype=np.float64)
        for t_c in tzd_cols:
            if t_c in d_src:
                t_arr = d_src[t_c]
                has_tzd = np.maximum(has_tzd, ((t_arr == 'Steady') | (t_arr == 'Up') | (t_arr == 'Down')).astype(np.float64))
        d_dst['tzd_active'] = has_tzd

        d_dst['insulin_and_metformin'] = d_dst['insulin_active'] * d_dst['metformin_active']
        d_dst['insulin_and_sulf'] = d_dst['insulin_active'] * has_sulf

        d_dst['insulin_only'] = ((d_dst['insulin_active'] == 1.0) & (d_dst['num_meds_active'] == 1.0)).astype(np.float64)
        d_dst['no_diabetes_meds'] = (d_dst['num_meds_active'] == 0.0).astype(np.float64)
        d_dst['regimen_oral_only'] = ((d_dst['num_meds_active'] > 0) & (d_dst['insulin_active'] == 0)).astype(np.float64)
        d_dst['regimen_combination'] = ((d_dst['insulin_active'] == 1) & (d_dst['num_meds_active'] > 1)).astype(np.float64)

        if 'change' in d_src:
            d_dst['change_flag'] = (d_src['change'] == 'Ch').astype(np.float64)
        else:
            d_dst['change_flag'] = np.zeros(n_rows, dtype=np.float64)

        d_dst['med_change_with_prior_inp'] = d_dst['change_flag'] * d_dst['has_prior_inpatient']
        d_dst['medication_turbulence'] = d_dst['num_meds_changed'] * (d_dst['insulin_changed'] + 1.0)
        d_dst['inpatient_with_insulin_up'] = d_dst['has_prior_inpatient'] * d_dst['insulin_up']
        d_dst['inpatient_with_insulin_down'] = d_dst['has_prior_inpatient'] * d_dst['insulin_down']

        d_dst['many_diagnoses'] = (d_src['number_diagnoses'].astype(np.float64) >= 9.0).astype(np.float64)

        valid_count = np.zeros(n_rows, dtype=np.float64)
        for d_col in ['diag_1', 'diag_2', 'diag_3']:
            arr = d_src[d_col]
            valid_count += np.array([1.0 if (c is not None and str(c) not in ('?', '', 'None')) else 0.0 for c in arr], dtype=np.float64)
        d_dst['num_valid_diags'] = valid_count

        if 'diabetesMed' in d_src:
            d_dst['diabetes_med_flag'] = (d_src['diabetesMed'] == 'Yes').astype(np.float64)
        else:
            d_dst['diabetes_med_flag'] = np.zeros(n_rows, dtype=np.float64)

        d_dst['no_diabetes_med_and_prior_inpatient'] = (1.0 - d_dst['diabetes_med_flag']) * d_dst['has_prior_inpatient']

        if 'A1Cresult' in d_src:
            a1c = d_src['A1Cresult']
            d_dst['a1c_tested'] = ((a1c == 'Norm') | (a1c == '>7') | (a1c == '>8')).astype(np.float64)
            d_dst['a1c_high'] = ((a1c == '>7') | (a1c == '>8')).astype(np.float64)
        else:
            d_dst['a1c_tested'] = np.zeros(n_rows, dtype=np.float64)
            d_dst['a1c_high'] = np.zeros(n_rows, dtype=np.float64)

        if 'max_glu_serum' in d_src:
            glu = d_src['max_glu_serum']
            d_dst['glu_tested'] = ((glu == 'Norm') | (glu == '>200') | (glu == '>300')).astype(np.float64)
            d_dst['glu_high'] = ((glu == '>200') | (glu == '>300')).astype(np.float64)
        else:
            d_dst['glu_tested'] = np.zeros(n_rows, dtype=np.float64)
            d_dst['glu_high'] = np.zeros(n_rows, dtype=np.float64)

        d_dst['a1c_tested_and_med_change'] = d_dst['a1c_tested'] * d_dst['change_flag']
        d_dst['a1c_high_and_med_change'] = d_dst['a1c_high'] * d_dst['change_flag']
        d_dst['a1c_high_no_med_change'] = d_dst['a1c_high'] * (1.0 - d_dst['change_flag'])
        d_dst['glu_tested_and_med_change'] = d_dst['glu_tested'] * d_dst['change_flag']
        d_dst['glu_high_and_med_change'] = d_dst['glu_high'] * d_dst['change_flag']
        d_dst['glu_high_no_med_change'] = d_dst['glu_high'] * (1.0 - d_dst['change_flag'])
        d_dst['no_glycemic_test'] = (1.0 - d_dst['a1c_tested']) * (1.0 - d_dst['glu_tested'])

    med_feature_names = [
        'num_meds_active', 'num_meds_up', 'num_meds_down', 'num_meds_changed', 'oral_med_changed',
        'non_diabetic_meds', 'diabetic_med_share',
        'insulin_active', 'insulin_up', 'insulin_down', 'insulin_steady', 'insulin_changed', 'insulin_ord',
        'both_ins_and_oral_changed',
        'metformin_ord', 'glimepiride_ord', 'glipizide_ord', 'glyburide_ord', 'pioglitazone_ord', 'rosiglitazone_ord',
        'metformin_active', 'sulfonylurea_active', 'meglitinide_active', 'tzd_active',
        'insulin_and_metformin', 'insulin_and_sulf', 'insulin_only', 'no_diabetes_meds',
        'regimen_oral_only', 'regimen_combination',
        'change_flag', 'med_change_with_prior_inp', 'diabetes_med_flag', 'no_diabetes_med_and_prior_inpatient', 'medication_turbulence',
        'inpatient_with_insulin_up', 'inpatient_with_insulin_down',
        'a1c_tested', 'a1c_high', 'glu_tested', 'glu_high',
        'a1c_tested_and_med_change', 'a1c_high_and_med_change', 'a1c_high_no_med_change',
        'glu_tested_and_med_change', 'glu_high_and_med_change', 'glu_high_no_med_change', 'no_glycemic_test',
        'many_diagnoses', 'num_valid_diags'
    ]
    extra_numeric.extend(med_feature_names)

    # 9. Fit TabularEncoder on clean clinical feature set
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_feat, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)

    # 10. Diverse LightGBM Ensemble with Native Categorical Splits
    lgb_models = [
        lgb.LGBMClassifier(
            n_estimators=380,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=65,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=2.5,
            reg_alpha=0.3,
            cat_smooth=35.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=420,
            learning_rate=0.025,
            num_leaves=22,
            max_depth=5,
            min_child_samples=50,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=4.0,
            reg_alpha=0.5,
            cat_smooth=25.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=360,
            learning_rate=0.03,
            num_leaves=38,
            max_depth=7,
            min_child_samples=80,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=2.0,
            reg_alpha=0.2,
            extra_trees=True,
            cat_smooth=45.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=380,
            learning_rate=0.028,
            num_leaves=26,
            min_child_samples=70,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=5.0,
            reg_alpha=1.0,
            path_smooth=1.0,
            cat_smooth=30.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    lgb_preds = np.zeros(n_eval, dtype=np.float64)
    for model in lgb_models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        lgb_preds += model.predict_proba(x_eval)[:, 1]
    lgb_preds /= len(lgb_models)

    # 11. XGBoost with Clean Numeric/Encoded Feature Matrix & Fast Histogram Splits
    # Drop raw categorical integer codes for XGBoost to avoid arbitrary ordinal splits
    cat_set = set(encoder.categorical_indices)
    xgb_feat_indices = [i for i in range(x_train.shape[1]) if i not in cat_set]
    x_train_xgb = x_train[:, xgb_feat_indices]
    x_eval_xgb = x_eval[:, xgb_feat_indices]

    xgb_models = [
        xgb.XGBClassifier(
            n_estimators=360,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=4,
            subsample=0.80,
            colsample_bytree=0.60,
            reg_lambda=3.0,
            reg_alpha=0.5,
            tree_method='hist',
            random_state=42,
            n_jobs=1,
            eval_metric='logloss',
            use_label_encoder=False
        ),
        xgb.XGBClassifier(
            n_estimators=380,
            learning_rate=0.025,
            max_depth=4,
            min_child_weight=6,
            subsample=0.85,
            colsample_bytree=0.50,
            reg_lambda=5.0,
            reg_alpha=1.0,
            tree_method='hist',
            random_state=2024,
            n_jobs=1,
            eval_metric='logloss',
            use_label_encoder=False
        ),
    ]

    xgb_preds = np.zeros(n_eval, dtype=np.float64)
    for model in xgb_models:
        model.fit(x_train_xgb, y_train)
        xgb_preds += model.predict_proba(x_eval_xgb)[:, 1]
    xgb_preds /= len(xgb_models)

    # 12. Percentile Rank Ensembling (Strictly superior to probability averaging for ROC-AUC)
    from scipy.stats import rankdata
    lgb_rank = rankdata(lgb_preds) / float(n_eval)
    xgb_rank = rankdata(xgb_preds) / float(n_eval)

    final_preds = 0.78 * lgb_rank + 0.22 * xgb_rank

    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
