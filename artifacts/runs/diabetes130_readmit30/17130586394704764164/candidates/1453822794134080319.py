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
    from sklearn.model_selection import StratifiedKFold
    
    # 1. Initialize lists for new features
    extra_numeric = []
    extra_categorical = []
    
    # Make shallow copies to avoid modifying caller's dict keys
    train = dict(train)
    eval_cols = dict(eval_cols)
    
    # 2. Numeric age representation (moved early)
    def age_to_numeric(age_str):
        if age_str is None:
            return np.nan
        try:
            cleaned = age_str.replace('[', '').replace(')', '').replace(']', '')
            parts = cleaned.split('-')
            return (float(parts[0]) + float(parts[1])) / 2.0
        except Exception:
            return np.nan

    train['age_num'] = np.array([age_to_numeric(x) for x in train['age']], dtype=np.float64)
    eval_cols['age_num'] = np.array([age_to_numeric(x) for x in eval_cols['age']], dtype=np.float64)
    extra_numeric.append('age_num')
    
    # 3. Extract comorbidities using icd9_comorbidities
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    train_diags = [train[col] for col in diag_cols if col in train]
    eval_diags = [eval_cols[col] for col in diag_cols if col in eval_cols]
    
    train_cm = icd9_comorbidities(*train_diags)
    eval_cm = icd9_comorbidities(*eval_diags)
    
    for k in train_cm:
        train[k] = train_cm[k]
        eval_cols[k] = eval_cm[k]
        extra_numeric.append(k)
        
    # 4. Grouped ICD-9 chapters, 3-digit category prefixes & combinatorial groupings
    def icd9_prefix(code):
        if code is None:
            return "missing"
        s = str(code).strip()
        if not s or s == "?":
            return "missing"
        return s.split(".")[0]

    for col in diag_cols:
        prefix_col = col + "_cat"
        train[prefix_col] = np.array([icd9_prefix(x) for x in train[col]], dtype=object)
        eval_cols[prefix_col] = np.array([icd9_prefix(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(prefix_col)

    for col in diag_cols:
        grouped_col = col + "_grp"
        train[grouped_col] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_cols[grouped_col] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(grouped_col)

    train['diag_12_grp'] = np.array([f"{g1}_{g2}" for g1, g2 in zip(train['diag_1_grp'], train['diag_2_grp'])], dtype=object)
    eval_cols['diag_12_grp'] = np.array([f"{g1}_{g2}" for g1, g2 in zip(eval_cols['diag_1_grp'], eval_cols['diag_2_grp'])], dtype=object)
    extra_categorical.append('diag_12_grp')

    train['diag_13_grp'] = np.array([f"{g1}_{g3}" for g1, g3 in zip(train['diag_1_grp'], train['diag_3_grp'])], dtype=object)
    eval_cols['diag_13_grp'] = np.array([f"{g1}_{g3}" for g1, g3 in zip(eval_cols['diag_1_grp'], eval_cols['diag_3_grp'])], dtype=object)
    extra_categorical.append('diag_13_grp')

    # 5. High-order categorical interactions
    train['diag1_disch'] = np.array([f"{g}_{disp}" for g, disp in zip(train['diag_1_grp'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['diag1_disch'] = np.array([f"{g}_{disp}" for g, disp in zip(eval_cols['diag_1_grp'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('diag1_disch')

    train['admsrc_disch'] = np.array([f"{s}_{disp}" for s, disp in zip(train['admission_source_id'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['admsrc_disch'] = np.array([f"{s}_{disp}" for s, disp in zip(eval_cols['admission_source_id'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('admsrc_disch')

    train['admtype_disch'] = np.array([f"{t}_{disp}" for t, disp in zip(train['admission_type_id'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['admtype_disch'] = np.array([f"{t}_{disp}" for t, disp in zip(eval_cols['admission_type_id'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('admtype_disch')

    train['disch_spec'] = np.array([f"{disp}_{spec}" for disp, spec in zip(train['discharge_disposition_id'], train['medical_specialty'])], dtype=object)
    eval_cols['disch_spec'] = np.array([f"{disp}_{spec}" for disp, spec in zip(eval_cols['discharge_disposition_id'], eval_cols['medical_specialty'])], dtype=object)
    extra_categorical.append('disch_spec')

    train['adm_spec'] = np.array([f"{adm}_{spec}" for adm, spec in zip(train['admission_source_id'], train['medical_specialty'])], dtype=object)
    eval_cols['adm_spec'] = np.array([f"{adm}_{spec}" for adm, spec in zip(eval_cols['admission_source_id'], eval_cols['medical_specialty'])], dtype=object)
    extra_categorical.append('adm_spec')

    train['disch_age'] = np.array([f"{disp}_{age}" for disp, age in zip(train['discharge_disposition_id'], train['age'])], dtype=object)
    eval_cols['disch_age'] = np.array([f"{disp}_{age}" for disp, age in zip(eval_cols['discharge_disposition_id'], eval_cols['age'])], dtype=object)
    extra_categorical.append('disch_age')

    train['diag1_admsrc'] = np.array([f"{g}_{src}" for g, src in zip(train['diag_1_grp'], train['admission_source_id'])], dtype=object)
    eval_cols['diag1_admsrc'] = np.array([f"{g}_{src}" for g, src in zip(eval_cols['diag_1_grp'], eval_cols['admission_source_id'])], dtype=object)
    extra_categorical.append('diag1_admsrc')

    # 6. Target encoding with double seed-averaging
    def double_oof_target_encode(train_col, y_train, eval_col, n_folds=5, smoothing=20.0):
        tr1, ev1 = oof_target_encode(train_col, y_train, eval_col, n_folds=n_folds, smoothing=smoothing, seed=42)
        tr2, ev2 = oof_target_encode(train_col, y_train, eval_col, n_folds=n_folds, smoothing=smoothing, seed=2023)
        return 0.5 * (tr1 + tr2), 0.5 * (ev1 + ev2)

    te_cols = [
        'diag_1', 'diag_2', 'diag_3', 'diag_1_cat', 'diag_2_cat', 'medical_specialty',
        'discharge_disposition_id', 'admission_source_id', 'payer_code', 'admission_type_id',
        'diag_1_grp', 'diag1_disch', 'admsrc_disch', 'diag_12_grp', 'diag_13_grp',
        'admtype_disch', 'disch_spec', 'adm_spec', 'disch_age', 'diag1_admsrc'
    ]
    for col in te_cols:
        if col in train and col in eval_cols:
            tr_enc, ev_enc = double_oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0)
            te_col = col + "_te"
            train[te_col] = tr_enc
            eval_cols[te_col] = ev_enc
            extra_numeric.append(te_col)
            
    # 7. Utilisation intensity features
    train_total_visits = train['number_outpatient'] + train['number_emergency'] + train['number_inpatient']
    eval_total_visits = eval_cols['number_outpatient'] + eval_cols['number_emergency'] + eval_cols['number_inpatient']
    train['total_visits'] = train_total_visits
    eval_cols['total_visits'] = eval_total_visits
    extra_numeric.append('total_visits')
    
    train['inpatient_ratio'] = train['number_inpatient'] / (train_total_visits + 1.0)
    eval_cols['inpatient_ratio'] = eval_cols['number_inpatient'] / (eval_total_visits + 1.0)
    extra_numeric.append('inpatient_ratio')
    
    # 8. Medication active, change, up, down, and stable counts
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    train_active_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_active_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_change_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_change_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_up_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_up_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_down_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_down_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_steady_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_steady_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    
    for col in med_cols:
        if col in train:
            train_active_meds += (train[col] != "No") & (train[col] != None)
            eval_active_meds += (eval_cols[col] != "No") & (eval_cols[col] != None)
            train_change_meds += (train[col] == "Up") | (train[col] == "Down")
            eval_change_meds += (eval_cols[col] == "Up") | (eval_cols[col] == "Down")
            train_up_meds += (train[col] == "Up")
            eval_up_meds += (eval_cols[col] == "Up")
            train_down_meds += (train[col] == "Down")
            eval_down_meds += (eval_cols[col] == "Down")
            train_steady_meds += (train[col] == "Steady")
            eval_steady_meds += (eval_cols[col] == "Steady")
            
    train['active_meds_count'] = train_active_meds
    eval_cols['active_meds_count'] = eval_active_meds
    extra_numeric.append('active_meds_count')
    
    train['change_meds_count'] = train_change_meds
    eval_cols['change_meds_count'] = eval_change_meds
    extra_numeric.append('change_meds_count')

    train['num_up_meds'] = train_up_meds
    eval_cols['num_up_meds'] = eval_up_meds
    extra_numeric.append('num_up_meds')

    train['num_down_meds'] = train_down_meds
    eval_cols['num_down_meds'] = eval_down_meds
    extra_numeric.append('num_down_meds')

    train['num_steady_meds'] = train_steady_meds
    eval_cols['num_steady_meds'] = eval_steady_meds
    extra_numeric.append('num_steady_meds')

    # 8b. Low-Rank Tensor/Matrix Decomposition of Medication Regimen
    from sklearn.decomposition import TruncatedSVD
    status_map = {'No': 0.0, 'Down': 1.0, 'Steady': 2.0, 'Up': 3.0}
    change_map = {'No': 0.0, 'Steady': 0.0, 'Down': -1.0, 'Up': 1.0}
    n_tr = len(y_train)
    n_ev = len(eval_cols['insulin'])
    med_mat_tr = np.zeros((n_tr, len(med_cols) * 2), dtype=np.float64)
    med_mat_ev = np.zeros((n_ev, len(med_cols) * 2), dtype=np.float64)
    for i, col in enumerate(med_cols):
        tr_v = train.get(col, np.array(['No'] * n_tr))
        ev_v = eval_cols.get(col, np.array(['No'] * n_ev))
        med_mat_tr[:, 2 * i] = [status_map.get(v, 0.0) for v in tr_v]
        med_mat_ev[:, 2 * i] = [status_map.get(v, 0.0) for v in ev_v]
        med_mat_tr[:, 2 * i + 1] = [change_map.get(v, 0.0) for v in tr_v]
        med_mat_ev[:, 2 * i + 1] = [change_map.get(v, 0.0) for v in ev_v]

    svd_med = TruncatedSVD(n_components=4, random_state=42)
    med_svd_tr = svd_med.fit_transform(med_mat_tr)
    med_svd_ev = svd_med.transform(med_mat_ev)
    for k in range(4):
        col_name = f'med_svd_{k}'
        train[col_name] = med_svd_tr[:, k]
        eval_cols[col_name] = med_svd_ev[:, k]
        extra_numeric.append(col_name)
    
    # 9. Insulin change interaction
    train['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(train['insulin'], train['change'])], dtype=object)
    eval_cols['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(eval_cols['insulin'], eval_cols['change'])], dtype=object)
    extra_categorical.append('insulin_change')

    # 10. Comorbidity weighted Charlson proxy and syndromic interactions
    train_charlson = (
        train_cm['cm_mi'] + train_cm['cm_chf'] + train_cm['cm_pvd'] +
        train_cm['cm_cerebrovascular'] + train_cm['cm_dementia'] +
        train_cm['cm_pulmonary'] + train_cm['cm_rheumatic'] +
        train_cm['cm_peptic_ulcer'] + train_cm['cm_liver'] +
        2.0 * (train_cm['cm_hemiplegia'] + train_cm['cm_renal'] +
               train_cm['cm_diabetes_complicated'] + train_cm['cm_cancer']) +
        4.0 * (train_cm['cm_metastatic'] + train_cm['cm_hiv'])
    )
    eval_charlson = (
        eval_cm['cm_mi'] + eval_cm['cm_chf'] + eval_cm['cm_pvd'] +
        eval_cm['cm_cerebrovascular'] + eval_cm['cm_dementia'] +
        eval_cm['cm_pulmonary'] + eval_cm['cm_rheumatic'] +
        eval_cm['cm_peptic_ulcer'] + eval_cm['cm_liver'] +
        2.0 * (eval_cm['cm_hemiplegia'] + eval_cm['cm_renal'] +
               eval_cm['cm_diabetes_complicated'] + eval_cm['cm_cancer']) +
        4.0 * (eval_cm['cm_metastatic'] + eval_cm['cm_hiv'])
    )
    train['charlson_proxy'] = train_charlson
    eval_cols['charlson_proxy'] = eval_charlson
    extra_numeric.append('charlson_proxy')

    for d, cm in [(train, train_cm), (eval_cols, eval_cm)]:
        d['cm_chf_renal'] = cm['cm_chf'] * cm['cm_renal']
        d['cm_chf_dm'] = cm['cm_chf'] * cm['cm_diabetes_complicated']
        d['cm_renal_dm'] = cm['cm_renal'] * cm['cm_diabetes_complicated']
        d['cm_pulmonary_chf'] = cm['cm_pulmonary'] * cm['cm_chf']
    extra_numeric.extend(['cm_chf_renal', 'cm_chf_dm', 'cm_renal_dm', 'cm_pulmonary_chf'])

    # 10b. Comorbidity & Diagnostic Chapter Low-Rank Phenotyping + Bilinear Tensor Cross-Products
    cm_keys = [
        'cm_mi', 'cm_chf', 'cm_pvd', 'cm_cerebrovascular', 'cm_dementia',
        'cm_pulmonary', 'cm_rheumatic', 'cm_peptic_ulcer', 'cm_liver',
        'cm_hemiplegia', 'cm_renal', 'cm_cancer', 'cm_metastatic', 'cm_hiv',
        'cm_diabetes_complicated'
    ]
    cm_mat_tr = np.column_stack([train_cm[k] for k in cm_keys])
    cm_mat_ev = np.column_stack([eval_cm[k] for k in cm_keys])
    svd_cm = TruncatedSVD(n_components=3, random_state=42)
    cm_svd_tr = svd_cm.fit_transform(cm_mat_tr)
    cm_svd_ev = svd_cm.transform(cm_mat_ev)
    for k in range(3):
        col_name = f'cm_svd_{k}'
        train[col_name] = cm_svd_tr[:, k]
        eval_cols[col_name] = cm_svd_ev[:, k]
        extra_numeric.append(col_name)

    # Bilinear Tensor cross-products: Comorbidity Phenotype x Medication Regimen
    for d, c_svd, m_svd in [(train, cm_svd_tr, med_svd_tr), (eval_cols, cm_svd_ev, med_svd_ev)]:
        d['cm_med_svd_00'] = c_svd[:, 0] * m_svd[:, 0]
        d['cm_med_svd_01'] = c_svd[:, 0] * m_svd[:, 1]
        d['cm_med_svd_10'] = c_svd[:, 1] * m_svd[:, 0]
        d['cm_med_svd_11'] = c_svd[:, 1] * m_svd[:, 1]
    extra_numeric.extend(['cm_med_svd_00', 'cm_med_svd_01', 'cm_med_svd_10', 'cm_med_svd_11'])

    # Bag-of-chapters Diagnostic Co-occurrence Factorization
    chapter_list = ['circulatory', 'respiratory', 'digestive', 'diabetes', 'injury', 'musculoskeletal', 'genitourinary', 'neoplasms', 'other', 'missing']
    diag_mat_tr = np.zeros((n_tr, len(chapter_list)), dtype=np.float64)
    diag_mat_ev = np.zeros((n_ev, len(chapter_list)), dtype=np.float64)
    for ch_idx, ch_name in enumerate(chapter_list):
        diag_mat_tr[:, ch_idx] = ((train['diag_1_grp'] == ch_name).astype(np.float64) +
                                  (train['diag_2_grp'] == ch_name).astype(np.float64) +
                                  (train['diag_3_grp'] == ch_name).astype(np.float64))
        diag_mat_ev[:, ch_idx] = ((eval_cols['diag_1_grp'] == ch_name).astype(np.float64) +
                                  (eval_cols['diag_2_grp'] == ch_name).astype(np.float64) +
                                  (eval_cols['diag_3_grp'] == ch_name).astype(np.float64))
    svd_diag = TruncatedSVD(n_components=3, random_state=42)
    diag_svd_tr = svd_diag.fit_transform(diag_mat_tr)
    diag_svd_ev = svd_diag.transform(diag_mat_ev)
    for k in range(3):
        col_name = f'diag_svd_{k}'
        train[col_name] = diag_svd_tr[:, k]
        eval_cols[col_name] = diag_svd_ev[:, k]
        extra_numeric.append(col_name)

    train['diag_cm_svd_00'] = diag_svd_tr[:, 0] * cm_svd_tr[:, 0]
    eval_cols['diag_cm_svd_00'] = diag_svd_ev[:, 0] * cm_svd_ev[:, 0]
    extra_numeric.append('diag_cm_svd_00')

    # 11. Prior utilization details & acute care ratios
    for d in (train, eval_cols):
        tot = d['number_outpatient'] + d['number_emergency'] + d['number_inpatient']
        d['has_inpatient'] = (d['number_inpatient'] > 0).astype(np.float64)
        d['has_emergency'] = (d['number_emergency'] > 0).astype(np.float64)
        d['has_outpatient'] = (d['number_outpatient'] > 0).astype(np.float64)
        d['frequent_inpatient'] = (d['number_inpatient'] >= 2).astype(np.float64)
        d['emergency_ratio'] = d['number_emergency'] / (tot + 1.0)
        d['acute_ratio'] = (d['number_inpatient'] + d['number_emergency']) / (tot + 1.0)
        d['weighted_prior_visits'] = 3.0 * d['number_inpatient'] + 2.0 * d['number_emergency'] + 1.0 * d['number_outpatient']
        d['log_inpatient'] = np.log1p(np.maximum(0.0, d['number_inpatient']))
        d['log_emergency'] = np.log1p(np.maximum(0.0, d['number_emergency']))
        d['log_total_visits'] = np.log1p(np.maximum(0.0, tot))
        
        # Risk ratio and age-comorbidity interactions
        d['emergency_to_inpatient_ratio'] = d['number_emergency'] / (d['number_inpatient'] + 1.0)
        d['outpatient_to_inpatient_ratio'] = d['number_outpatient'] / (d['number_inpatient'] + 1.0)
        d['age_comorbidity_interaction'] = d['age_num'] * d['charlson_proxy']
        d['age_visits_interaction'] = d['age_num'] * d['log_total_visits']

    extra_numeric.extend([
        'has_inpatient', 'has_emergency', 'has_outpatient', 'frequent_inpatient',
        'emergency_ratio', 'acute_ratio', 'weighted_prior_visits',
        'log_inpatient', 'log_emergency', 'log_total_visits',
        'emergency_to_inpatient_ratio', 'outpatient_to_inpatient_ratio',
        'age_comorbidity_interaction', 'age_visits_interaction'
    ])

    # 12. Hospital stay intensity & procedures ratios
    for d in (train, eval_cols):
        t_hosp = np.maximum(1.0, d['time_in_hospital'])
        d['labs_per_day'] = d['num_lab_procedures'] / t_hosp
        d['meds_per_day'] = d['num_medications'] / t_hosp
        d['procs_per_day'] = d['num_procedures'] / t_hosp
        d['diagnoses_per_day'] = d['number_diagnoses'] / t_hosp
        d['total_procedures'] = d['num_lab_procedures'] + d['num_procedures']
        d['meds_per_diag'] = d['num_medications'] / (d['number_diagnoses'] + 1.0)
        d['procs_per_med'] = d['total_procedures'] / (d['num_medications'] + 1.0)
        d['long_stay'] = (d['time_in_hospital'] >= 7).astype(np.float64)
    extra_numeric.extend([
        'labs_per_day', 'meds_per_day', 'procs_per_day', 'diagnoses_per_day',
        'total_procedures', 'meds_per_diag', 'procs_per_med', 'long_stay'
    ])

    # 13. Discharge and admission indicators
    for d in (train, eval_cols):
        disp = d['discharge_disposition_id']
        d['disch_home'] = (disp == '1').astype(np.float64)
        d['disch_snf'] = (disp == '3').astype(np.float64)
        d['disch_home_health'] = (disp == '6').astype(np.float64)
        d['disch_ama'] = (disp == '7').astype(np.float64)
        d['disch_transfer'] = np.isin(disp, ['2', '3', '4', '5', '22']).astype(np.float64)

        adm_typ = d['admission_type_id']
        adm_src = d['admission_source_id']
        d['adm_is_emergency'] = np.isin(adm_typ, ['1', '2']).astype(np.float64)
        d['adm_from_er'] = (adm_src == '7').astype(np.float64)
        d['adm_high_acuity'] = (d['adm_is_emergency'] * d['adm_from_er']).astype(np.float64)
    extra_numeric.extend([
        'disch_home', 'disch_snf', 'disch_home_health', 'disch_ama', 'disch_transfer',
        'adm_is_emergency', 'adm_from_er', 'adm_high_acuity'
    ])

    # 14. LACE Index bedside score proxy & clinical severity score
    def calc_lace(los, is_er, charlson, ed_visits):
        l_score = np.zeros_like(los, dtype=np.float64)
        l_score[los == 2] = 1.0
        l_score[los == 3] = 2.0
        l_score[(los >= 4) & (los <= 6)] = 3.0
        l_score[(los >= 7) & (los <= 13)] = 4.0
        l_score[los >= 14] = 5.0
        a_score = 3.0 * is_er
        c_score = np.clip(charlson, 0.0, 5.0)
        e_score = np.clip(ed_visits, 0.0, 4.0)
        return l_score + a_score + c_score + e_score

    train['lace_score'] = calc_lace(train['time_in_hospital'], train['adm_is_emergency'], train_charlson, train['number_emergency'])
    eval_cols['lace_score'] = calc_lace(eval_cols['time_in_hospital'], eval_cols['adm_is_emergency'], eval_charlson, eval_cols['number_emergency'])
    extra_numeric.append('lace_score')

    for d in (train, eval_cols):
        d['clinical_severity_score'] = (
            d['lace_score'] * 1.5 +
            d['inpatient_ratio'] * 10.0 +
            d['charlson_proxy'] * 2.0 +
            d['log_total_visits'] * 3.0 +
            (d['time_in_hospital'] * d['num_medications'] / 100.0)
        )
    extra_numeric.append('clinical_severity_score')

    # 15. Lab test indicators & Glycemic control clinical inertia
    for d in (train, eval_cols):
        a1c = d['A1Cresult']
        glu = d['max_glu_serum']
        chg = d['change']

        a1c_tested = (a1c != 'None').astype(np.float64)
        a1c_high = ((a1c == '>7') | (a1c == '>8')).astype(np.float64)
        glu_tested = (glu != 'None').astype(np.float64)
        glu_high = ((glu == '>200') | (glu == '>300')).astype(np.float64)
        
        a1c_normal = (a1c == 'Norm').astype(np.float64)
        glu_normal = (glu == 'Norm').astype(np.float64)

        d['a1c_tested'] = a1c_tested
        d['a1c_high'] = a1c_high
        d['glu_tested'] = glu_tested
        d['glu_high'] = glu_high
        d['a1c_normal'] = a1c_normal
        d['glu_normal'] = glu_normal
        d['good_glycemic_control'] = ((a1c_normal == 1.0) | (glu_normal == 1.0)).astype(np.float64)
        d['diabetes_uncontrolled'] = ((a1c_high == 1.0) | (glu_high == 1.0)).astype(np.float64)
        d['high_a1c_no_med_change'] = ((a1c_high == 1.0) & (chg == 'No')).astype(np.float64)
        d['high_a1c_med_change'] = ((a1c_high == 1.0) & (chg == 'Ch')).astype(np.float64)
    extra_numeric.extend([
        'a1c_tested', 'a1c_high', 'glu_tested', 'glu_high', 'a1c_normal', 'glu_normal',
        'good_glycemic_control', 'diabetes_uncontrolled', 'high_a1c_no_med_change', 'high_a1c_med_change'
    ])

    # 16. Medication directionality & polypharmacy
    for d in (train, eval_cols):
        ins = d['insulin']
        d['on_insulin'] = (ins != 'No').astype(np.float64)
        d['insulin_up'] = (ins == 'Up').astype(np.float64)
        d['insulin_down'] = (ins == 'Down').astype(np.float64)
        d['insulin_steady'] = (ins == 'Steady').astype(np.float64)
        d['insulin_changed'] = ((ins == 'Up') | (ins == 'Down')).astype(np.float64)
        d['on_metformin'] = (d['metformin'] != 'No').astype(np.float64)
        d['polypharmacy'] = (d['num_medications'] >= 16).astype(np.float64)
        d['diabetes_med_flag'] = (d['diabetesMed'] == 'Yes').astype(np.float64)
        d['med_change_flag'] = (d['change'] == 'Ch').astype(np.float64)
        
        # Oral meds
        d['on_glipizide'] = (d['glipizide'] != 'No').astype(np.float64)
        d['on_glyburide'] = (d['glyburide'] != 'No').astype(np.float64)
        d['on_pioglitazone'] = (d['pioglitazone'] != 'No').astype(np.float64)
        d['on_rosiglitazone'] = (d['rosiglitazone'] != 'No').astype(np.float64)
        
        # Elderly polypharmacy risk
        d['elderly_polypharmacy_change'] = ((d['age_num'] >= 70.0) & (d['polypharmacy'] == 1.0) & (d['med_change_flag'] == 1.0)).astype(np.float64)

    extra_numeric.extend([
        'on_insulin', 'insulin_up', 'insulin_down', 'insulin_steady', 'insulin_changed',
        'on_metformin', 'polypharmacy', 'diabetes_med_flag', 'med_change_flag',
        'on_glipizide', 'on_glyburide', 'on_pioglitazone', 'on_rosiglitazone',
        'elderly_polypharmacy_change'
    ])

    # 17. Missingness flags
    for d in (train, eval_cols):
        d['missing_specialty'] = (d['medical_specialty'] == None).astype(np.float64)
        d['missing_payer'] = (d['payer_code'] == None).astype(np.float64)
    extra_numeric.extend(['missing_specialty', 'missing_payer'])

    # 18. Tabular encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)

    # 19. Heterogeneous Stratified 5-Fold Ensemble: 2 LightGBM + 2 XGBoost Models
    import xgboost as xgb
    from scipy.stats import rankdata
    from scipy.optimize import minimize
    from sklearn.metrics import roc_auc_score

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_models = 4
    eval_preds_list = [np.zeros(x_eval.shape[0]) for _ in range(n_models)]
    oof_preds_list = [np.zeros(len(y_train)) for _ in range(n_models)]

    for train_idx, val_idx in skf.split(x_train, y_train):
        fold_x_tr, fold_y_tr = x_train[train_idx], y_train[train_idx]
        fold_x_val, fold_y_val = x_train[val_idx], y_train[val_idx]

        # Model 0: LightGBM (leaf-wise 31, seed 42)
        m0 = lgb.LGBMClassifier(
            n_estimators=220, learning_rate=0.02, num_leaves=31, min_child_samples=120,
            subsample=0.75, subsample_freq=1, colsample_bytree=0.6, reg_alpha=0.1,
            reg_lambda=1.0, cat_smooth=35.0, random_state=42, n_jobs=1, verbose=-1,
        )
        m0.fit(fold_x_tr, fold_y_tr, categorical_feature=encoder.categorical_indices)
        oof_preds_list[0][val_idx] = m0.predict_proba(fold_x_val)[:, 1]
        eval_preds_list[0] += m0.predict_proba(x_eval)[:, 1] / 5.0

        # Model 1: LightGBM (deeper num_leaves 45, seed 2023)
        m1 = lgb.LGBMClassifier(
            n_estimators=260, learning_rate=0.015, num_leaves=45, min_child_samples=80,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.5,
            reg_lambda=2.0, cat_smooth=25.0, random_state=2023, n_jobs=1, verbose=-1,
        )
        m1.fit(fold_x_tr, fold_y_tr, categorical_feature=encoder.categorical_indices)
        oof_preds_list[1][val_idx] = m1.predict_proba(fold_x_val)[:, 1]
        eval_preds_list[1] += m1.predict_proba(x_eval)[:, 1] / 5.0

        # Model 2: XGBoost (depth 5, colsample 0.6, seed 42)
        m2 = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.025, max_depth=5, subsample=0.75,
            colsample_bytree=0.6, reg_alpha=0.2, reg_lambda=1.5, tree_method='hist',
            random_state=42, n_jobs=1, eval_metric='logloss',
        )
        m2.fit(fold_x_tr, fold_y_tr)
        oof_preds_list[2][val_idx] = m2.predict_proba(fold_x_val)[:, 1]
        eval_preds_list[2] += m2.predict_proba(x_eval)[:, 1] / 5.0

        # Model 3: XGBoost (shallower depth 4, colsample 0.75, seed 2023)
        m3 = xgb.XGBClassifier(
            n_estimators=210, learning_rate=0.02, max_depth=4, subsample=0.85,
            colsample_bytree=0.75, reg_alpha=0.8, reg_lambda=2.5, tree_method='hist',
            random_state=2023, n_jobs=1, eval_metric='logloss',
        )
        m3.fit(fold_x_tr, fold_y_tr)
        oof_preds_list[3][val_idx] = m3.predict_proba(fold_x_val)[:, 1]
        eval_preds_list[3] += m3.predict_proba(x_eval)[:, 1] / 5.0

    # 20. Advanced Continuous Optimization across Probability, Logit & Rank Ensembles
    def to_logit(p, eps=1e-5):
        clipped = np.clip(p, eps, 1.0 - eps)
        return np.log(clipped / (1.0 - clipped))

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))

    oof_prob_mat = np.column_stack(oof_preds_list)
    oof_rank_mat = np.column_stack([rankdata(p) / float(len(p)) for p in oof_preds_list])
    oof_logit_mat = np.column_stack([to_logit(p) for p in oof_preds_list])

    # Generate grid on 4-simplex with resolution 0.1 (286 candidate weight vectors)
    grid_w = []
    for i in range(11):
        for j in range(11 - i):
            for k in range(11 - i - j):
                l = 10 - i - j - k
                grid_w.append((i / 10.0, j / 10.0, k / 10.0, l / 10.0))
    grid_w = np.array(grid_w)

    best_score = -1.0
    best_weights = np.array([0.35, 0.25, 0.25, 0.15])
    best_mode = 'logit'

    # Fast vectorized scoring of candidate weights across all three representations
    for mode_name, mat in [('logit', oof_logit_mat), ('prob', oof_prob_mat), ('rank', oof_rank_mat)]:
        for w in grid_w:
            if mode_name == 'logit':
                blend = sigmoid(mat @ w)
            else:
                blend = mat @ w
            score = roc_auc_score(y_train, blend)
            if score > best_score:
                best_score = score
                best_weights = w.copy()
                best_mode = mode_name

    # Fine-tune the continuous weights using Nelder-Mead optimization
    def objective(w):
        w_pos = np.maximum(0.0, w)
        s = np.sum(w_pos)
        if s <= 0:
            return 0.0
        w_norm = w_pos / s
        if best_mode == 'logit':
            pred = sigmoid(oof_logit_mat @ w_norm)
        elif best_mode == 'prob':
            pred = oof_prob_mat @ w_norm
        else:
            pred = oof_rank_mat @ w_norm
        return -roc_auc_score(y_train, pred)

    opt_res = minimize(objective, best_weights, method='Nelder-Mead', options={'maxiter': 35, 'disp': False})
    if opt_res.success and -opt_res.fun > best_score:
        w_final = np.maximum(0.0, opt_res.x)
        final_w = w_final / np.sum(w_final)
    else:
        final_w = best_weights

    # Apply optimal weights and chosen transformation to test predictions
    if best_mode == 'logit':
        eval_logit_mat = np.column_stack([to_logit(p) for p in eval_preds_list])
        eval_preds = sigmoid(eval_logit_mat @ final_w)
    elif best_mode == 'prob':
        eval_prob_mat = np.column_stack(eval_preds_list)
        eval_preds = eval_prob_mat @ final_w
    else:
        eval_rank_mat = np.column_stack([rankdata(p) / float(len(p)) for p in eval_preds_list])
        eval_preds = eval_rank_mat @ final_w

    return np.clip(eval_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
