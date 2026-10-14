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
    train_cols = dict(train)
    eval_cols_dict = dict(eval_cols)

    extra_numeric = []
    extra_categorical = []

    # Helper to extract 3-digit ICD-9 disease group
    def _to_3digit(code: object) -> str:
        if code is None:
            return 'missing'
        s = str(code).strip()
        if not s or s == '?':
            return 'missing'
        return s.split('.')[0]

    # 1. Comorbidity features & Weighted Charlson Comorbidity Index (CCI)
    cm_train = icd9_comorbidities(train_cols['diag_1'], train_cols['diag_2'], train_cols['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols_dict['diag_1'], eval_cols_dict['diag_2'], eval_cols_dict['diag_3'])
    for k in cm_train:
        train_cols[k] = cm_train[k]
        eval_cols_dict[k] = cm_eval[k]
        extra_numeric.append(k)

    # Standard clinical Charlson Comorbidity weights
    charlson_tr = (
        cm_train['cm_mi'] + cm_train['cm_chf'] + cm_train['cm_pvd'] +
        cm_train['cm_cerebrovascular'] + cm_train['cm_dementia'] +
        cm_train['cm_pulmonary'] + cm_train['cm_rheumatic'] +
        cm_train['cm_peptic_ulcer'] + cm_train['cm_liver'] +
        2.0 * cm_train['cm_hemiplegia'] + 2.0 * cm_train['cm_renal'] +
        2.0 * cm_train['cm_cancer'] + 2.0 * cm_train['cm_diabetes_complicated'] +
        4.0 * cm_train['cm_metastatic'] + 4.0 * cm_train['cm_hiv']
    )
    charlson_ev = (
        cm_eval['cm_mi'] + cm_eval['cm_chf'] + cm_eval['cm_pvd'] +
        cm_eval['cm_cerebrovascular'] + cm_eval['cm_dementia'] +
        cm_eval['cm_pulmonary'] + cm_eval['cm_rheumatic'] +
        cm_eval['cm_peptic_ulcer'] + cm_eval['cm_liver'] +
        2.0 * cm_eval['cm_hemiplegia'] + 2.0 * cm_eval['cm_renal'] +
        2.0 * cm_eval['cm_cancer'] + 2.0 * cm_eval['cm_diabetes_complicated'] +
        4.0 * cm_eval['cm_metastatic'] + 4.0 * cm_eval['cm_hiv']
    )
    train_cols['charlson_score'] = charlson_tr.astype(np.float64)
    eval_cols_dict['charlson_score'] = charlson_ev.astype(np.float64)
    extra_numeric.append('charlson_score')

    # Synergistic high-risk comorbidity interactions (Cardiorenal & Cardiopulmonary)
    train_cols['cm_chf_and_renal'] = (cm_train['cm_chf'] * cm_train['cm_renal']).astype(np.float64)
    eval_cols_dict['cm_chf_and_renal'] = (cm_eval['cm_chf'] * cm_eval['cm_renal']).astype(np.float64)
    extra_numeric.append('cm_chf_and_renal')

    train_cols['cm_pulm_and_chf'] = (cm_train['cm_pulmonary'] * cm_train['cm_chf']).astype(np.float64)
    eval_cols_dict['cm_pulm_and_chf'] = (cm_eval['cm_pulmonary'] * cm_eval['cm_chf']).astype(np.float64)
    extra_numeric.append('cm_pulm_and_chf')

    train_cols['cm_diab_comp_and_renal'] = (cm_train['cm_diabetes_complicated'] * cm_train['cm_renal']).astype(np.float64)
    eval_cols_dict['cm_diab_comp_and_renal'] = (cm_eval['cm_diabetes_complicated'] * cm_eval['cm_renal']).astype(np.float64)
    extra_numeric.append('cm_diab_comp_and_renal')

    # Fast pure-numpy low-rank factor decomposition via empirical covariance eigendecomposition
    def _low_rank_factors(x_tr: np.ndarray, x_ev: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        mu = np.mean(x_tr, axis=0, keepdims=True)
        xc_tr = x_tr - mu
        xc_ev = x_ev - mu
        cov = (xc_tr.T @ xc_tr) / max(len(x_tr), 1)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1][:k]
        w = evecs[:, idx]
        return xc_tr @ w, xc_ev @ w

    # Latent Comorbidity Phenotyping (Top 3 low-rank syndromic axes)
    cm_keys = [
        'cm_mi', 'cm_chf', 'cm_pvd', 'cm_cerebrovascular', 'cm_dementia',
        'cm_pulmonary', 'cm_rheumatic', 'cm_peptic_ulcer', 'cm_liver',
        'cm_hemiplegia', 'cm_renal', 'cm_cancer', 'cm_metastatic', 'cm_hiv',
        'cm_diabetes_complicated'
    ]
    cm_tr_mat = np.column_stack([cm_train[k] for k in cm_keys])
    cm_ev_mat = np.column_stack([cm_eval[k] for k in cm_keys])
    cm_p_tr, cm_p_ev = _low_rank_factors(cm_tr_mat, cm_ev_mat, k=3)
    for i in range(3):
        fname = f'cm_latent_{i}'
        train_cols[fname] = cm_p_tr[:, i].astype(np.float64)
        eval_cols_dict[fname] = cm_p_ev[:, i].astype(np.float64)
        extra_numeric.append(fname)

    # 2. ICD-9 Chapter Groups, 3-Digit Category Hierarchy & Multi-System Burden
    for col in ('diag_1', 'diag_2', 'diag_3'):
        g_name = f'{col}_group'
        train_cols[g_name] = np.array([icd9_group(x) for x in train_cols[col]], dtype=object)
        eval_cols_dict[g_name] = np.array([icd9_group(x) for x in eval_cols_dict[col]], dtype=object)
        extra_categorical.append(g_name)

        d3_name = f'{col}_3digit'
        train_cols[d3_name] = np.array([_to_3digit(x) for x in train_cols[col]], dtype=object)
        eval_cols_dict[d3_name] = np.array([_to_3digit(x) for x in eval_cols_dict[col]], dtype=object)
        extra_categorical.append(d3_name)

    # Comprehensive multi-system chapter counts across all 3 diagnoses
    all_chapters = ['circulatory', 'respiratory', 'digestive', 'diabetes', 'injury', 'musculoskeletal', 'genitourinary', 'neoplasms', 'other']
    diag_tr_mat = np.zeros((len(y_train), len(all_chapters)), dtype=np.float64)
    diag_ev_mat = np.zeros((len(eval_cols_dict['age']), len(all_chapters)), dtype=np.float64)
    for ch_idx, ch in enumerate(all_chapters):
        c_tr = (train_cols['diag_1_group'] == ch).astype(np.float64) + (train_cols['diag_2_group'] == ch).astype(np.float64) + (train_cols['diag_3_group'] == ch).astype(np.float64)
        c_ev = (eval_cols_dict['diag_1_group'] == ch).astype(np.float64) + (eval_cols_dict['diag_2_group'] == ch).astype(np.float64) + (eval_cols_dict['diag_3_group'] == ch).astype(np.float64)
        diag_tr_mat[:, ch_idx] = c_tr
        diag_ev_mat[:, ch_idx] = c_ev
        fname = f'diag_count_{ch}'
        train_cols[fname] = c_tr
        eval_cols_dict[fname] = c_ev
        extra_numeric.append(fname)

    # Low-rank multi-system diagnostic phenotype factors
    diag_p_tr, diag_p_ev = _low_rank_factors(diag_tr_mat, diag_ev_mat, k=2)
    for i in range(2):
        fname = f'diag_latent_{i}'
        train_cols[fname] = diag_p_tr[:, i].astype(np.float64)
        eval_cols_dict[fname] = diag_p_ev[:, i].astype(np.float64)
        extra_numeric.append(fname)

    # Primary diagnosis chapter indicator flags
    for grp in ('circulatory', 'respiratory', 'diabetes', 'digestive', 'genitourinary'):
        f_name = f'diag1_is_{grp}'
        train_cols[f_name] = (train_cols['diag_1_group'] == grp).astype(np.float64)
        eval_cols_dict[f_name] = (eval_cols_dict['diag_1_group'] == grp).astype(np.float64)
        extra_numeric.append(f_name)

    # 3. High-cardinality Target Encodings (Individual & Interaction Tensor Slices)
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_3digit', 'diag_2_3digit', 'diag_3_3digit',
        'medical_specialty', 'discharge_disposition_id',
        'admission_source_id', 'payer_code'
    ]
    for col in te_cols:
        if col in train_cols and col in eval_cols_dict:
            te_name = f'{col}_te'
            tr_te, ev_te = oof_target_encode(
                train_cols[col], y_train, eval_cols_dict[col],
                n_folds=5, smoothing=20.0, seed=42
            )
            train_cols[te_name] = tr_te.astype(np.float64)
            eval_cols_dict[te_name] = ev_te.astype(np.float64)
            extra_numeric.append(te_name)

    # Interaction Tensor Slices: Empirical Bayes encodings of joint categorical manifolds
    tensor_interactions = [
        ('disp_x_diag1',
         train_cols['discharge_disposition_id'].astype(str) + '_' + train_cols['diag_1_group'].astype(str),
         eval_cols_dict['discharge_disposition_id'].astype(str) + '_' + eval_cols_dict['diag_1_group'].astype(str)),
        ('adm_x_disp',
         train_cols['admission_type_id'].astype(str) + '_' + train_cols['discharge_disposition_id'].astype(str),
         eval_cols_dict['admission_type_id'].astype(str) + '_' + eval_cols_dict['discharge_disposition_id'].astype(str)),
        ('diag1_x_adm_source',
         train_cols['diag_1_group'].astype(str) + '_' + train_cols['admission_source_id'].astype(str),
         eval_cols_dict['diag_1_group'].astype(str) + '_' + eval_cols_dict['admission_source_id'].astype(str)),
    ]
    for iname, itr, iev in tensor_interactions:
        te_name = f'{iname}_te'
        tr_te, ev_te = oof_target_encode(
            itr, y_train, iev,
            n_folds=5, smoothing=20.0, seed=42
        )
        train_cols[te_name] = tr_te.astype(np.float64)
        eval_cols_dict[te_name] = ev_te.astype(np.float64)
        extra_numeric.append(te_name)

    # 4. Utilization & Intensity features
    total_vis_tr = train_cols['number_outpatient'] + train_cols['number_emergency'] + train_cols['number_inpatient']
    total_vis_ev = eval_cols_dict['number_outpatient'] + eval_cols_dict['number_emergency'] + eval_cols_dict['number_inpatient']

    acute_vis_tr = train_cols['number_emergency'] + train_cols['number_inpatient']
    acute_vis_ev = eval_cols_dict['number_emergency'] + eval_cols_dict['number_inpatient']

    train_cols['total_prior_visits'] = total_vis_tr.astype(np.float64)
    eval_cols_dict['total_prior_visits'] = total_vis_ev.astype(np.float64)
    extra_numeric.append('total_prior_visits')

    train_cols['acute_prior_visits'] = acute_vis_tr.astype(np.float64)
    eval_cols_dict['acute_prior_visits'] = acute_vis_ev.astype(np.float64)
    extra_numeric.append('acute_prior_visits')

    train_cols['inpatient_ratio'] = (train_cols['number_inpatient'] / (total_vis_tr + 1.0)).astype(np.float64)
    eval_cols_dict['inpatient_ratio'] = (eval_cols_dict['number_inpatient'] / (total_vis_ev + 1.0)).astype(np.float64)
    extra_numeric.append('inpatient_ratio')

    train_cols['acute_ratio'] = (acute_vis_tr / (total_vis_tr + 1.0)).astype(np.float64)
    eval_cols_dict['acute_ratio'] = (acute_vis_ev / (total_vis_ev + 1.0)).astype(np.float64)
    extra_numeric.append('acute_ratio')

    train_cols['has_inpatient'] = (train_cols['number_inpatient'] > 0).astype(np.float64)
    eval_cols_dict['has_inpatient'] = (eval_cols_dict['number_inpatient'] > 0).astype(np.float64)
    extra_numeric.append('has_inpatient')

    train_cols['has_emergency'] = (train_cols['number_emergency'] > 0).astype(np.float64)
    eval_cols_dict['has_emergency'] = (eval_cols_dict['number_emergency'] > 0).astype(np.float64)
    extra_numeric.append('has_emergency')

    train_cols['frequent_inpatient'] = (train_cols['number_inpatient'] >= 2).astype(np.float64)
    eval_cols_dict['frequent_inpatient'] = (eval_cols_dict['number_inpatient'] >= 2).astype(np.float64)
    extra_numeric.append('frequent_inpatient')

    train_cols['inpatient_per_los'] = (train_cols['number_inpatient'] / (train_cols['time_in_hospital'] + 1.0)).astype(np.float64)
    eval_cols_dict['inpatient_per_los'] = (eval_cols_dict['number_inpatient'] / (eval_cols_dict['time_in_hospital'] + 1.0)).astype(np.float64)
    extra_numeric.append('inpatient_per_los')

    train_cols['lab_per_day'] = (train_cols['num_lab_procedures'] / (train_cols['time_in_hospital'] + 0.5)).astype(np.float64)
    eval_cols_dict['lab_per_day'] = (eval_cols_dict['num_lab_procedures'] / (eval_cols_dict['time_in_hospital'] + 0.5)).astype(np.float64)
    extra_numeric.append('lab_per_day')

    train_cols['med_per_day'] = (train_cols['num_medications'] / (train_cols['time_in_hospital'] + 0.5)).astype(np.float64)
    eval_cols_dict['med_per_day'] = (eval_cols_dict['num_medications'] / (eval_cols_dict['time_in_hospital'] + 0.5)).astype(np.float64)
    extra_numeric.append('med_per_day')

    train_cols['diag_per_day'] = (train_cols['number_diagnoses'] / (train_cols['time_in_hospital'] + 0.5)).astype(np.float64)
    eval_cols_dict['diag_per_day'] = (eval_cols_dict['number_diagnoses'] / (eval_cols_dict['time_in_hospital'] + 0.5)).astype(np.float64)
    extra_numeric.append('diag_per_day')

    train_cols['lab_med_intensity'] = ((train_cols['num_lab_procedures'] + train_cols['num_medications']) / (train_cols['time_in_hospital'] + 0.5)).astype(np.float64)
    eval_cols_dict['lab_med_intensity'] = ((eval_cols_dict['num_lab_procedures'] + eval_cols_dict['num_medications']) / (eval_cols_dict['time_in_hospital'] + 0.5)).astype(np.float64)
    extra_numeric.append('lab_med_intensity')

    train_cols['inpatient_times_los'] = (train_cols['number_inpatient'] * train_cols['time_in_hospital']).astype(np.float64)
    eval_cols_dict['inpatient_times_los'] = (eval_cols_dict['number_inpatient'] * eval_cols_dict['time_in_hospital']).astype(np.float64)
    extra_numeric.append('inpatient_times_los')

    train_cols['revolving_door_risk'] = (train_cols['number_inpatient'] * acute_vis_tr).astype(np.float64)
    eval_cols_dict['revolving_door_risk'] = (eval_cols_dict['number_inpatient'] * acute_vis_ev).astype(np.float64)
    extra_numeric.append('revolving_door_risk')

    train_cols['emergency_ratio'] = (train_cols['number_emergency'] / (total_vis_tr + 1.0)).astype(np.float64)
    eval_cols_dict['emergency_ratio'] = (eval_cols_dict['number_emergency'] / (total_vis_ev + 1.0)).astype(np.float64)
    extra_numeric.append('emergency_ratio')

    train_cols['polypharmacy'] = (train_cols['num_medications'] >= 15).astype(np.float64)
    eval_cols_dict['polypharmacy'] = (eval_cols_dict['num_medications'] >= 15).astype(np.float64)
    extra_numeric.append('polypharmacy')

    train_cols['high_charlson'] = (charlson_tr >= 3.0).astype(np.float64)
    eval_cols_dict['high_charlson'] = (charlson_ev >= 3.0).astype(np.float64)
    extra_numeric.append('high_charlson')

    train_cols['complex_readmit_risk'] = ((train_cols['number_inpatient'] > 0) & (charlson_tr >= 2.0)).astype(np.float64)
    eval_cols_dict['complex_readmit_risk'] = ((eval_cols_dict['number_inpatient'] > 0) & (charlson_ev >= 2.0)).astype(np.float64)
    extra_numeric.append('complex_readmit_risk')

    train_cols['diagnoses_capped'] = (train_cols['number_diagnoses'] >= 9).astype(np.float64)
    eval_cols_dict['diagnoses_capped'] = (eval_cols_dict['number_diagnoses'] >= 9).astype(np.float64)
    extra_numeric.append('diagnoses_capped')

    # Discharge disposition & Admission patterns
    train_cols['disch_home'] = (train_cols['discharge_disposition_id'] == '1').astype(np.float64)
    eval_cols_dict['disch_home'] = (eval_cols_dict['discharge_disposition_id'] == '1').astype(np.float64)
    extra_numeric.append('disch_home')

    facility_ids = ['2', '3', '4', '5', '22', '23', '24']
    train_cols['disch_facility'] = np.isin(train_cols['discharge_disposition_id'], facility_ids).astype(np.float64)
    eval_cols_dict['disch_facility'] = np.isin(eval_cols_dict['discharge_disposition_id'], facility_ids).astype(np.float64)
    extra_numeric.append('disch_facility')

    train_cols['disch_home_health'] = (train_cols['discharge_disposition_id'] == '6').astype(np.float64)
    eval_cols_dict['disch_home_health'] = (eval_cols_dict['discharge_disposition_id'] == '6').astype(np.float64)
    extra_numeric.append('disch_home_health')

    train_cols['disch_ama'] = (train_cols['discharge_disposition_id'] == '7').astype(np.float64)
    eval_cols_dict['disch_ama'] = (eval_cols_dict['discharge_disposition_id'] == '7').astype(np.float64)
    extra_numeric.append('disch_ama')

    train_cols['adm_emergency'] = (train_cols['admission_type_id'] == '1').astype(np.float64)
    eval_cols_dict['adm_emergency'] = (eval_cols_dict['admission_type_id'] == '1').astype(np.float64)
    extra_numeric.append('adm_emergency')

    train_cols['adm_source_er'] = (train_cols['admission_source_id'] == '7').astype(np.float64)
    eval_cols_dict['adm_source_er'] = (eval_cols_dict['admission_source_id'] == '7').astype(np.float64)
    extra_numeric.append('adm_source_er')

    # Lab testing and result indicators
    train_cols['a1c_tested'] = (train_cols['A1Cresult'] != 'None').astype(np.float64)
    eval_cols_dict['a1c_tested'] = (eval_cols_dict['A1Cresult'] != 'None').astype(np.float64)
    extra_numeric.append('a1c_tested')

    train_cols['a1c_high'] = np.isin(train_cols['A1Cresult'], ['>7', '>8']).astype(np.float64)
    eval_cols_dict['a1c_high'] = np.isin(eval_cols_dict['A1Cresult'], ['>7', '>8']).astype(np.float64)
    extra_numeric.append('a1c_high')

    train_cols['glu_tested'] = (train_cols['max_glu_serum'] != 'None').astype(np.float64)
    eval_cols_dict['glu_tested'] = (eval_cols_dict['max_glu_serum'] != 'None').astype(np.float64)
    extra_numeric.append('glu_tested')

    train_cols['glu_high'] = np.isin(train_cols['max_glu_serum'], ['>200', '>300']).astype(np.float64)
    eval_cols_dict['glu_high'] = np.isin(eval_cols_dict['max_glu_serum'], ['>200', '>300']).astype(np.float64)
    extra_numeric.append('glu_high')

    # Informative missingness flags
    train_cols['specialty_missing'] = (train_cols['medical_specialty'] == None).astype(np.float64)
    eval_cols_dict['specialty_missing'] = (eval_cols_dict['medical_specialty'] == None).astype(np.float64)
    extra_numeric.append('specialty_missing')

    train_cols['payer_missing'] = (train_cols['payer_code'] == None).astype(np.float64)
    eval_cols_dict['payer_missing'] = (eval_cols_dict['payer_code'] == None).astype(np.float64)
    extra_numeric.append('payer_missing')

    # 5. Age numeric midpoint
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    train_cols['age_num'] = np.array([age_map.get(a, 65.0) for a in train_cols['age']], dtype=np.float64)
    eval_cols_dict['age_num'] = np.array([age_map.get(a, 65.0) for a in eval_cols_dict['age']], dtype=np.float64)
    extra_numeric.append('age_num')

    # 6. Concise medication counts & Latent Pharmacotherapy Phenotypes
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    present_meds = [m for m in med_names if m in train_cols and m in eval_cols_dict]

    act_tr = np.zeros(len(y_train), dtype=np.float64)
    act_ev = np.zeros(len(eval_cols_dict['age']), dtype=np.float64)
    chg_tr = np.zeros(len(y_train), dtype=np.float64)
    chg_ev = np.zeros(len(eval_cols_dict['age']), dtype=np.float64)

    # Encode medication matrix for low-rank pharmacotherapy factorization
    med_tr_mat = np.zeros((len(y_train), len(present_meds)), dtype=np.float64)
    med_ev_mat = np.zeros((len(eval_cols_dict['age']), len(present_meds)), dtype=np.float64)
    med_val_map = {'No': 0.0, 'Steady': 1.0, 'Up': 2.0, 'Down': -1.0}

    for j, m in enumerate(present_meds):
        tr_m = train_cols[m]
        ev_m = eval_cols_dict[m]
        act_tr += (tr_m != 'No').astype(np.float64)
        act_ev += (ev_m != 'No').astype(np.float64)
        chg_tr += ((tr_m == 'Up') | (tr_m == 'Down')).astype(np.float64)
        chg_ev += ((ev_m == 'Up') | (ev_m == 'Down')).astype(np.float64)
        med_tr_mat[:, j] = [med_val_map.get(v, 0.0) for v in tr_m]
        med_ev_mat[:, j] = [med_val_map.get(v, 0.0) for v in ev_m]

    train_cols['num_active_meds'] = act_tr
    eval_cols_dict['num_active_meds'] = act_ev
    extra_numeric.append('num_active_meds')

    train_cols['num_med_changes'] = chg_tr
    eval_cols_dict['num_med_changes'] = chg_ev
    extra_numeric.append('num_med_changes')

    # Decompose pharmacotherapy regimen matrix into 3 latent medication phenotypes
    med_p_tr, med_p_ev = _low_rank_factors(med_tr_mat, med_ev_mat, k=3)
    for i in range(3):
        fname = f'med_latent_{i}'
        train_cols[fname] = med_p_tr[:, i].astype(np.float64)
        eval_cols_dict[fname] = med_p_ev[:, i].astype(np.float64)
        extra_numeric.append(fname)

    train_cols['metformin_prescribed'] = (train_cols['metformin'] != 'No').astype(np.float64)
    eval_cols_dict['metformin_prescribed'] = (eval_cols_dict['metformin'] != 'No').astype(np.float64)
    extra_numeric.append('metformin_prescribed')

    # Specific insulin dosing trajectory and medication change patterns
    train_cols['insulin_prescribed'] = (train_cols['insulin'] != 'No').astype(np.float64)
    eval_cols_dict['insulin_prescribed'] = (eval_cols_dict['insulin'] != 'No').astype(np.float64)
    extra_numeric.append('insulin_prescribed')

    train_cols['insulin_up'] = (train_cols['insulin'] == 'Up').astype(np.float64)
    eval_cols_dict['insulin_up'] = (eval_cols_dict['insulin'] == 'Up').astype(np.float64)
    extra_numeric.append('insulin_up')

    train_cols['insulin_down'] = (train_cols['insulin'] == 'Down').astype(np.float64)
    eval_cols_dict['insulin_down'] = (eval_cols_dict['insulin'] == 'Down').astype(np.float64)
    extra_numeric.append('insulin_down')

    train_cols['insulin_changed'] = np.isin(train_cols['insulin'], ['Up', 'Down']).astype(np.float64)
    eval_cols_dict['insulin_changed'] = np.isin(eval_cols_dict['insulin'], ['Up', 'Down']).astype(np.float64)
    extra_numeric.append('insulin_changed')

    train_cols['med_change_flag'] = (train_cols['change'] == 'Ch').astype(np.float64)
    eval_cols_dict['med_change_flag'] = (eval_cols_dict['change'] == 'Ch').astype(np.float64)
    extra_numeric.append('med_change_flag')

    train_cols['on_diabetes_med'] = (train_cols['diabetesMed'] == 'Yes').astype(np.float64)
    eval_cols_dict['on_diabetes_med'] = (eval_cols_dict['diabetesMed'] == 'Yes').astype(np.float64)
    extra_numeric.append('on_diabetes_med')

    train_cols['insulin_and_oral'] = ((train_cols['insulin'] != 'No') & (act_tr > 1)).astype(np.float64)
    eval_cols_dict['insulin_and_oral'] = ((eval_cols_dict['insulin'] != 'No') & (act_ev > 1)).astype(np.float64)
    extra_numeric.append('insulin_and_oral')

    # Fit encoder with standard min_count to retain specialized transfer/care locations
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task, train_cols,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical)
    )
    x_train = encoder.transform(train_cols)
    x_eval = encoder.transform(eval_cols_dict)

    # Subspace-diverse ensemble: deep interactions + feature decorrelation + shallow regularized
    model_configs = [
        # Model 1: Deep Interaction Forest
        {
            'n_estimators': 290, 'learning_rate': 0.022, 'num_leaves': 35, 'max_depth': 6,
            'min_child_samples': 90, 'subsample': 0.80, 'subsample_freq': 1,
            'colsample_bytree': 0.60, 'reg_lambda': 3.0, 'reg_alpha': 0.4,
            'cat_smooth': 30.0, 'random_state': 42
        },
        # Model 2: Balanced Feature-Subspace Forest (low colsample forces discovery of alternate splits)
        {
            'n_estimators': 280, 'learning_rate': 0.024, 'num_leaves': 24, 'max_depth': 5,
            'min_child_samples': 75, 'subsample': 0.85, 'subsample_freq': 1,
            'colsample_bytree': 0.50, 'reg_lambda': 2.5, 'reg_alpha': 0.2,
            'cat_smooth': 40.0, 'random_state': 123
        },
        # Model 3: Shallow Regularized Main-Effects Forest
        {
            'n_estimators': 270, 'learning_rate': 0.025, 'num_leaves': 16, 'max_depth': 4,
            'min_child_samples': 60, 'subsample': 0.90, 'subsample_freq': 1,
            'colsample_bytree': 0.75, 'reg_lambda': 5.0, 'reg_alpha': 0.8,
            'cat_smooth': 50.0, 'random_state': 456
        },
        # Model 4: Wide Capacity Conservative Leaf Forest
        {
            'n_estimators': 310, 'learning_rate': 0.020, 'num_leaves': 28, 'max_depth': -1,
            'min_child_samples': 110, 'subsample': 0.75, 'subsample_freq': 1,
            'colsample_bytree': 0.65, 'reg_lambda': 4.0, 'reg_alpha': 0.5,
            'cat_smooth': 35.0, 'random_state': 789
        },
    ]

    preds = np.zeros(len(x_eval), dtype=np.float64)
    for cfg in model_configs:
        model = lgb.LGBMClassifier(
            **cfg,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1]
    preds /= len(model_configs)

    preds = np.clip(preds, 0.0, 1.0)
    if np.isnan(preds).any():
        preds = np.nan_to_num(preds, nan=0.114)

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
