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
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from scipy.stats import rankdata
    import lightgbm as lgb
    import xgboost as xgb
    from clinical_toolkit import (
        icd9_comorbidities,
        icd9_group,
        oof_target_encode,
        TabularEncoder
    )

    # Copy the dictionaries to avoid modifying input data in-place
    train_copy = dict(train)
    eval_cols_copy = dict(eval_cols)

    new_numeric_cols = []
    new_categorical_cols = []

    # 1. Comorbidities (Charlson clinical risk indicators)
    train_cm = icd9_comorbidities(train_copy['diag_1'], train_copy['diag_2'], train_copy['diag_3'])
    eval_cm = icd9_comorbidities(eval_cols_copy['diag_1'], eval_cols_copy['diag_2'], eval_cols_copy['diag_3'])
    for k in train_cm.keys():
        train_copy[k] = train_cm[k]
        eval_cols_copy[k] = eval_cm[k]
        new_numeric_cols.append(k)

    # 1b. Weighted Charlson Comorbidity Index
    train_weighted_cm = (
        train_cm['cm_mi'] + train_cm['cm_chf'] + train_cm['cm_pvd'] +
        train_cm['cm_cerebrovascular'] + train_cm['cm_dementia'] +
        train_cm['cm_pulmonary'] + train_cm['cm_rheumatic'] +
        train_cm['cm_peptic_ulcer'] + train_cm['cm_liver'] * 2.0 +
        train_cm['cm_hemiplegia'] * 2.0 + train_cm['cm_renal'] * 2.0 +
        train_cm['cm_cancer'] * 2.0 + train_cm['cm_metastatic'] * 4.0 +
        train_cm['cm_hiv'] * 4.0 + train_cm['cm_diabetes_complicated'] * 2.0
    ).astype(float)
    eval_weighted_cm = (
        eval_cm['cm_mi'] + eval_cm['cm_chf'] + eval_cm['cm_pvd'] +
        eval_cm['cm_cerebrovascular'] + eval_cm['cm_dementia'] +
        eval_cm['cm_pulmonary'] + eval_cm['cm_rheumatic'] +
        eval_cm['cm_peptic_ulcer'] + eval_cm['cm_liver'] * 2.0 +
        eval_cm['cm_hemiplegia'] * 2.0 + eval_cm['cm_renal'] * 2.0 +
        eval_cm['cm_cancer'] * 2.0 + eval_cm['cm_metastatic'] * 4.0 +
        eval_cm['cm_hiv'] * 4.0 + eval_cm['cm_diabetes_complicated'] * 2.0
    ).astype(float)
    train_copy['weighted_charlson'] = train_weighted_cm
    eval_cols_copy['weighted_charlson'] = eval_weighted_cm
    new_numeric_cols.append('weighted_charlson')

    # 1c. Cardiorenal & Cardiopulmonary Clinical Syndromes
    train_copy['cm_cardiorenal'] = (train_cm['cm_chf'] * train_cm['cm_renal']).astype(float)
    eval_cols_copy['cm_cardiorenal'] = (eval_cm['cm_chf'] * eval_cm['cm_renal']).astype(float)
    new_numeric_cols.append('cm_cardiorenal')

    train_copy['cm_cardiopulmonary'] = (train_cm['cm_chf'] * train_cm['cm_pulmonary']).astype(float)
    eval_cols_copy['cm_cardiopulmonary'] = (eval_cm['cm_chf'] * eval_cm['cm_pulmonary']).astype(float)
    new_numeric_cols.append('cm_cardiopulmonary')

    train_copy['cm_vascular_burden'] = (train_cm['cm_pvd'] + train_cm['cm_cerebrovascular'] + train_cm['cm_mi']).astype(float)
    eval_cols_copy['cm_vascular_burden'] = (eval_cm['cm_pvd'] + eval_cm['cm_cerebrovascular'] + eval_cm['cm_mi']).astype(float)
    new_numeric_cols.append('cm_vascular_burden')

    train_copy['cm_high_risk_count'] = (
        train_cm['cm_chf'] + train_cm['cm_renal'] + train_cm['cm_pulmonary'] +
        train_cm['cm_cancer'] + train_cm['cm_liver']
    ).astype(float)
    eval_cols_copy['cm_high_risk_count'] = (
        eval_cm['cm_chf'] + eval_cm['cm_renal'] + eval_cm['cm_pulmonary'] +
        eval_cm['cm_cancer'] + eval_cm['cm_liver']
    ).astype(float)
    new_numeric_cols.append('cm_high_risk_count')

    # 2. ICD-9 Chapter Groups
    for col in ['diag_1', 'diag_2', 'diag_3']:
        group_col_name = f"{col}_group"
        train_copy[group_col_name] = np.array([icd9_group(x) for x in train_copy[col]], dtype=object)
        eval_cols_copy[group_col_name] = np.array([icd9_group(x) for x in eval_cols_copy[col]], dtype=object)
        new_categorical_cols.append(group_col_name)

    # Primary Diagnosis Admitting Chapter Flags
    for grp_val in ['circulatory', 'respiratory', 'diabetes', 'digestive', 'genitourinary']:
        f_name = f"diag1_is_{grp_val}"
        train_copy[f_name] = (train_copy['diag_1_group'] == grp_val).astype(float)
        eval_cols_copy[f_name] = (eval_cols_copy['diag_1_group'] == grp_val).astype(float)
        new_numeric_cols.append(f_name)

    # 3. Multi-Scale ICD-9 3-Digit Roots & Clinical Flag Conditions
    def extract_icd9_3d(arr):
        out = []
        for x in arr:
            if x is None or x == '?' or x == 'None' or str(x).strip() == '':
                out.append('missing')
            else:
                s = str(x).strip().split('.')[0]
                out.append(s if s else 'missing')
        return np.array(out, dtype=object)

    for col in ['diag_1', 'diag_2', 'diag_3']:
        d3_col = f"{col}_3d"
        train_copy[d3_col] = extract_icd9_3d(train_copy[col])
        eval_cols_copy[d3_col] = extract_icd9_3d(eval_cols_copy[col])
        new_categorical_cols.append(d3_col)

    def make_diag_flag(d1, d2, d3, codes):
        code_set = set(codes)
        return (
            np.isin(d1, list(code_set)) |
            np.isin(d2, list(code_set)) |
            np.isin(d3, list(code_set))
        ).astype(float)

    clinical_flags = {
        'flag_chf': ['428'],
        'flag_diabetes_diag': ['250'],
        'flag_copd_pneu': ['486', '491', '492', '493', '494', '496'],
        'flag_renal': ['584', '585', '586'],
        'flag_cad_mi': ['410', '411', '412', '413', '414'],
        'flag_hypertension': ['401', '402', '403', '404', '405'],
        'flag_electrolyte': ['276'],
        'flag_dysrhythmia': ['427'],
        'flag_cerebrovasc': ['430', '431', '432', '433', '434', '435', '436', '437', '438'],
        'flag_sepsis': ['038', '995'],
        'flag_uti': ['599'],
        'flag_skin_ulcer': ['681', '682', '707'],
        'flag_hypoglycemia': ['251'],
        'flag_esrd': ['585'],
    }
    for flag_name, codes in clinical_flags.items():
        train_copy[flag_name] = make_diag_flag(
            train_copy['diag_1_3d'], train_copy['diag_2_3d'], train_copy['diag_3_3d'], codes
        )
        eval_cols_copy[flag_name] = make_diag_flag(
            eval_cols_copy['diag_1_3d'], eval_cols_copy['diag_2_3d'], eval_cols_copy['diag_3_3d'], codes
        )
        new_numeric_cols.append(flag_name)

    # 3b. Diabetic Complications & Subphenotype Specific Flags (ICD-9 250.xx digits)
    def extract_diab_subtypes(d1, d2, d3):
        n = len(d1)
        uncontrolled = np.zeros(n, dtype=float)
        type1_diab = np.zeros(n, dtype=float)
        ketoacidosis = np.zeros(n, dtype=float)
        renal_diab = np.zeros(n, dtype=float)
        neuro_diab = np.zeros(n, dtype=float)

        for i in range(n):
            for d in (d1[i], d2[i], d3[i]):
                if d is None or d == '?' or d == 'None':
                    continue
                s = str(d).strip()
                if s.startswith('250'):
                    if s.endswith('2') or s.endswith('3'):
                        uncontrolled[i] = 1.0
                    if s.endswith('1') or s.endswith('3'):
                        type1_diab[i] = 1.0
                    if s.startswith('250.1'):
                        ketoacidosis[i] = 1.0
                    if s.startswith('250.4'):
                        renal_diab[i] = 1.0
                    if s.startswith('250.6'):
                        neuro_diab[i] = 1.0
        return uncontrolled, type1_diab, ketoacidosis, renal_diab, neuro_diab

    tr_u, tr_t1, tr_k, tr_rn, tr_nu = extract_diab_subtypes(train_copy['diag_1'], train_copy['diag_2'], train_copy['diag_3'])
    ev_u, ev_t1, ev_k, ev_rn, ev_nu = extract_diab_subtypes(eval_cols_copy['diag_1'], eval_cols_copy['diag_2'], eval_cols_copy['diag_3'])

    diab_sub_feats = [
        ('diab_uncontrolled', tr_u, ev_u),
        ('diab_type1', tr_t1, ev_t1),
        ('diab_ketoacidosis', tr_k, ev_k),
        ('diab_renal_complication', tr_rn, ev_rn),
        ('diab_neuro_complication', tr_nu, ev_nu),
    ]
    for feat_name, tr_f, ev_f in diab_sub_feats:
        train_copy[feat_name] = tr_f
        eval_cols_copy[feat_name] = ev_f
        new_numeric_cols.append(feat_name)

    # 4. Out-of-Fold Target Encodings
    cols_to_target_encode = [
        'diag_1', 'diag_2', 'diag_3', 'medical_specialty',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'payer_code'
    ]
    for col in cols_to_target_encode:
        te_col_name = f"{col}_te"
        train_enc, eval_enc = oof_target_encode(
            train_copy[col], y_train, eval_cols_copy[col], smoothing=20.0, seed=42
        )
        train_copy[te_col_name] = train_enc.astype(float)
        eval_cols_copy[te_col_name] = eval_enc.astype(float)
        new_numeric_cols.append(te_col_name)

    for col in ['diag_1_3d', 'diag_2_3d', 'diag_3_3d']:
        te_col_name = f"{col}_te"
        tr_enc, ev_enc = oof_target_encode(
            train_copy[col], y_train, eval_cols_copy[col], smoothing=15.0, seed=42
        )
        train_copy[te_col_name] = tr_enc.astype(float)
        eval_cols_copy[te_col_name] = ev_enc.astype(float)
        new_numeric_cols.append(te_col_name)

    # 5. Clinical Interactions & Target Encodings
    def make_interaction(arr1, arr2):
        return np.array([f"{a}_{b}" for a, b in zip(arr1, arr2)], dtype=object)

    interactions = [
        ('disch_x_admsrc', 'discharge_disposition_id', 'admission_source_id'),
        ('diag1_x_disch', 'diag_1_group', 'discharge_disposition_id'),
        ('admtype_x_disch', 'admission_type_id', 'discharge_disposition_id'),
        ('disch_x_specialty', 'discharge_disposition_id', 'medical_specialty'),
        ('admsrc_x_specialty', 'admission_source_id', 'medical_specialty'),
        ('age_x_disch', 'age', 'discharge_disposition_id'),
        ('diag1_3d_x_disch', 'diag_1_3d', 'discharge_disposition_id'),
        ('diag1_x_age', 'diag_1_group', 'age'),
        ('payer_x_disch', 'payer_code', 'discharge_disposition_id'),
    ]
    for inter_name, c1, c2 in interactions:
        train_copy[inter_name] = make_interaction(train_copy[c1], train_copy[c2])
        eval_cols_copy[inter_name] = make_interaction(eval_cols_copy[c1], eval_cols_copy[c2])
        new_categorical_cols.append(inter_name)
        tr_enc, ev_enc = oof_target_encode(
            train_copy[inter_name], y_train, eval_cols_copy[inter_name], smoothing=25.0, seed=42
        )
        te_name = f"{inter_name}_te"
        train_copy[te_name] = tr_enc.astype(float)
        eval_cols_copy[te_name] = ev_enc.astype(float)
        new_numeric_cols.append(te_name)

    # 6. Frequency Encodings for Categorical Cardinality/Prevalence
    def calc_freq(tr_arr, ev_arr):
        vals, counts = np.unique([str(x) for x in tr_arr], return_counts=True)
        count_dict = dict(zip(vals, counts.astype(float) / len(tr_arr)))
        tr_freq = np.array([count_dict.get(str(x), 0.0) for x in tr_arr], dtype=float)
        ev_freq = np.array([count_dict.get(str(x), 0.0) for x in ev_arr], dtype=float)
        return tr_freq, ev_freq

    for col in ['diag_1_3d', 'discharge_disposition_id', 'admission_source_id', 'medical_specialty']:
        f_name = f"{col}_freq"
        tr_f, ev_f = calc_freq(train_copy[col], eval_cols_copy[col])
        train_copy[f_name] = tr_f
        eval_cols_copy[f_name] = ev_f
        new_numeric_cols.append(f_name)

    # 7. Utilization Intensity & Longitudinal Risk Dynamics
    train_copy['total_prior_visits'] = (
        train_copy['number_outpatient'] + train_copy['number_emergency'] + train_copy['number_inpatient']
    )
    eval_cols_copy['total_prior_visits'] = (
        eval_cols_copy['number_outpatient'] + eval_cols_copy['number_emergency'] + eval_cols_copy['number_inpatient']
    )
    new_numeric_cols.append('total_prior_visits')

    train_copy['has_prior_inpatient'] = (train_copy['number_inpatient'] > 0).astype(float)
    eval_cols_copy['has_prior_inpatient'] = (eval_cols_copy['number_inpatient'] > 0).astype(float)
    new_numeric_cols.append('has_prior_inpatient')

    train_copy['is_frequent_inpatient'] = (train_copy['number_inpatient'] >= 2).astype(float)
    eval_cols_copy['is_frequent_inpatient'] = (eval_cols_copy['number_inpatient'] >= 2).astype(float)
    new_numeric_cols.append('is_frequent_inpatient')

    train_copy['has_prior_emergency'] = (train_copy['number_emergency'] > 0).astype(float)
    eval_cols_copy['has_prior_emergency'] = (eval_cols_copy['number_emergency'] > 0).astype(float)
    new_numeric_cols.append('has_prior_emergency')

    train_copy['has_prior_outpatient'] = (train_copy['number_outpatient'] > 0).astype(float)
    eval_cols_copy['has_prior_outpatient'] = (eval_cols_copy['number_outpatient'] > 0).astype(float)
    new_numeric_cols.append('has_prior_outpatient')

    train_copy['utilization_severity_score'] = (
        train_copy['number_inpatient'] * 4.0 + train_copy['number_emergency'] * 2.0 + train_copy['number_outpatient'] * 0.5
    ).astype(float)
    eval_cols_copy['utilization_severity_score'] = (
        eval_cols_copy['number_inpatient'] * 4.0 + eval_cols_copy['number_emergency'] * 2.0 + eval_cols_copy['number_outpatient'] * 0.5
    ).astype(float)
    new_numeric_cols.append('utilization_severity_score')

    train_copy['inpatient_ratio'] = (
        train_copy['number_inpatient'] / (train_copy['total_prior_visits'] + 1.0)
    ).astype(float)
    eval_cols_copy['inpatient_ratio'] = (
        eval_cols_copy['number_inpatient'] / (eval_cols_copy['total_prior_visits'] + 1.0)
    ).astype(float)
    new_numeric_cols.append('inpatient_ratio')

    train_copy['emergency_ratio'] = (
        train_copy['number_emergency'] / (train_copy['total_prior_visits'] + 1.0)
    ).astype(float)
    eval_cols_copy['emergency_ratio'] = (
        eval_cols_copy['number_emergency'] / (eval_cols_copy['total_prior_visits'] + 1.0)
    ).astype(float)
    new_numeric_cols.append('emergency_ratio')

    train_copy['inpatient_x_los'] = (train_copy['number_inpatient'] * train_copy['time_in_hospital']).astype(float)
    eval_cols_copy['inpatient_x_los'] = (eval_cols_copy['number_inpatient'] * eval_cols_copy['time_in_hospital']).astype(float)
    new_numeric_cols.append('inpatient_x_los')

    train_copy['inpatient_and_emergency'] = ((train_copy['number_inpatient'] > 0) & (train_copy['number_emergency'] > 0)).astype(float)
    eval_cols_copy['inpatient_and_emergency'] = ((eval_cols_copy['number_inpatient'] > 0) & (eval_cols_copy['number_emergency'] > 0)).astype(float)
    new_numeric_cols.append('inpatient_and_emergency')

    train_copy['outpatient_only'] = (
        (train_copy['number_outpatient'] > 0) & (train_copy['number_inpatient'] == 0) & (train_copy['number_emergency'] == 0)
    ).astype(float)
    eval_cols_copy['outpatient_only'] = (
        (eval_cols_copy['number_outpatient'] > 0) & (eval_cols_copy['number_inpatient'] == 0) & (eval_cols_copy['number_emergency'] == 0)
    ).astype(float)
    new_numeric_cols.append('outpatient_only')

    train_copy['days_per_diagnosis'] = (train_copy['time_in_hospital'] / (train_copy['number_diagnoses'] + 1.0)).astype(float)
    eval_cols_copy['days_per_diagnosis'] = (eval_cols_copy['time_in_hospital'] / (eval_cols_copy['number_diagnoses'] + 1.0)).astype(float)
    new_numeric_cols.append('days_per_diagnosis')

    train_copy['procedures_per_diagnosis'] = (train_copy['num_procedures'] / (train_copy['number_diagnoses'] + 1.0)).astype(float)
    eval_cols_copy['procedures_per_diagnosis'] = (eval_cols_copy['num_procedures'] / (eval_cols_copy['number_diagnoses'] + 1.0)).astype(float)
    new_numeric_cols.append('procedures_per_diagnosis')

    train_copy['labs_per_medication'] = (train_copy['num_lab_procedures'] / (train_copy['num_medications'] + 1.0)).astype(float)
    eval_cols_copy['labs_per_medication'] = (eval_cols_copy['num_lab_procedures'] / (eval_cols_copy['num_medications'] + 1.0)).astype(float)
    new_numeric_cols.append('labs_per_medication')

    # Ratios per hospital stay days
    train_copy['med_by_hospital_days'] = train_copy['num_medications'] / (train_copy['time_in_hospital'] + 1.0)
    eval_cols_copy['med_by_hospital_days'] = eval_cols_copy['num_medications'] / (eval_cols_copy['time_in_hospital'] + 1.0)
    new_numeric_cols.append('med_by_hospital_days')

    train_copy['lab_by_hospital_days'] = train_copy['num_lab_procedures'] / (train_copy['time_in_hospital'] + 1.0)
    eval_cols_copy['lab_by_hospital_days'] = eval_cols_copy['num_lab_procedures'] / (eval_cols_copy['time_in_hospital'] + 1.0)
    new_numeric_cols.append('lab_by_hospital_days')

    train_copy['procedures_by_hospital_days'] = train_copy['num_procedures'] / (train_copy['time_in_hospital'] + 1.0)
    eval_cols_copy['procedures_by_hospital_days'] = eval_cols_copy['num_procedures'] / (eval_cols_copy['time_in_hospital'] + 1.0)
    new_numeric_cols.append('procedures_by_hospital_days')

    train_copy['med_per_diagnosis'] = train_copy['num_medications'] / (train_copy['number_diagnoses'] + 1.0)
    eval_cols_copy['med_per_diagnosis'] = eval_cols_copy['num_medications'] / (eval_cols_copy['number_diagnoses'] + 1.0)
    new_numeric_cols.append('med_per_diagnosis')

    train_copy['overall_clinical_intensity'] = (
        (train_copy['num_lab_procedures'] + train_copy['num_medications'] * 2.0 + train_copy['num_procedures'] * 3.0) /
        (train_copy['time_in_hospital'] + 1.0)
    ).astype(float)
    eval_cols_copy['overall_clinical_intensity'] = (
        (eval_cols_copy['num_lab_procedures'] + eval_cols_copy['num_medications'] * 2.0 + eval_cols_copy['num_procedures'] * 3.0) /
        (eval_cols_copy['time_in_hospital'] + 1.0)
    ).astype(float)
    new_numeric_cols.append('overall_clinical_intensity')

    # 8. Medication Regimen and Directional Changes
    med_cols = []
    for col in task.categorical_columns:
        if col in ['change', 'diabetesMed']:
            continue
        unique_vals = set(train_copy[col])
        if any(v in unique_vals for v in ['Steady', 'Up', 'Down']):
            med_cols.append(col)

    train_num_active = np.zeros(len(y_train), dtype=float)
    eval_num_active = np.zeros(len(eval_cols_copy['time_in_hospital']), dtype=float)
    train_num_up = np.zeros(len(y_train), dtype=float)
    eval_num_up = np.zeros(len(eval_cols_copy['time_in_hospital']), dtype=float)
    train_num_down = np.zeros(len(y_train), dtype=float)
    eval_num_down = np.zeros(len(eval_cols_copy['time_in_hospital']), dtype=float)
    train_num_changes = np.zeros(len(y_train), dtype=float)
    eval_num_changes = np.zeros(len(eval_cols_copy['time_in_hospital']), dtype=float)

    for col in med_cols:
        train_num_active += ((train_copy[col] != 'No') & (train_copy[col] != None)).astype(float)
        eval_num_active += ((eval_cols_copy[col] != 'No') & (eval_cols_copy[col] != None)).astype(float)

        train_num_up += (train_copy[col] == 'Up').astype(float)
        eval_num_up += (eval_cols_copy[col] == 'Up').astype(float)

        train_num_down += (train_copy[col] == 'Down').astype(float)
        eval_num_down += (eval_cols_copy[col] == 'Down').astype(float)

        train_num_changes += ((train_copy[col] == 'Up') | (train_copy[col] == 'Down')).astype(float)
        eval_num_changes += ((eval_cols_copy[col] == 'Up') | (eval_cols_copy[col] == 'Down')).astype(float)

    train_copy['num_active_meds'] = train_num_active
    eval_cols_copy['num_active_meds'] = eval_num_active
    new_numeric_cols.append('num_active_meds')

    train_copy['num_up_meds'] = train_num_up
    eval_cols_copy['num_up_meds'] = eval_num_up
    new_numeric_cols.append('num_up_meds')

    train_copy['num_down_meds'] = train_num_down
    eval_cols_copy['num_down_meds'] = eval_num_down
    new_numeric_cols.append('num_down_meds')

    train_copy['num_changes'] = train_num_changes
    eval_cols_copy['num_changes'] = eval_num_changes
    new_numeric_cols.append('num_changes')

    # Insulin specific features
    train_copy['on_insulin'] = ((train_copy['insulin'] != 'No') & (train_copy['insulin'] != None)).astype(float)
    eval_cols_copy['on_insulin'] = ((eval_cols_copy['insulin'] != 'No') & (eval_cols_copy['insulin'] != None)).astype(float)
    new_numeric_cols.append('on_insulin')

    train_copy['insulin_up'] = (train_copy['insulin'] == 'Up').astype(float)
    eval_cols_copy['insulin_up'] = (eval_cols_copy['insulin'] == 'Up').astype(float)
    new_numeric_cols.append('insulin_up')

    train_copy['insulin_down'] = (train_copy['insulin'] == 'Down').astype(float)
    eval_cols_copy['insulin_down'] = (eval_cols_copy['insulin'] == 'Down').astype(float)
    new_numeric_cols.append('insulin_down')

    train_copy['insulin_changed'] = ((train_copy['insulin'] == 'Up') | (train_copy['insulin'] == 'Down')).astype(float)
    eval_cols_copy['insulin_changed'] = ((eval_cols_copy['insulin'] == 'Up') | (eval_cols_copy['insulin'] == 'Down')).astype(float)
    new_numeric_cols.append('insulin_changed')

    train_copy['insulin_and_oral'] = (train_copy['on_insulin'] * (train_copy['num_active_meds'] > 1)).astype(float)
    eval_cols_copy['insulin_and_oral'] = (eval_cols_copy['on_insulin'] * (eval_cols_copy['num_active_meds'] > 1)).astype(float)
    new_numeric_cols.append('insulin_and_oral')

    train_copy['on_metformin'] = ((train_copy['metformin'] != 'No') & (train_copy['metformin'] != None)).astype(float)
    eval_cols_copy['on_metformin'] = ((eval_cols_copy['metformin'] != 'No') & (eval_cols_copy['metformin'] != None)).astype(float)
    new_numeric_cols.append('on_metformin')

    train_copy['regimen_changed'] = (train_copy['change'] == 'Ch').astype(float)
    eval_cols_copy['regimen_changed'] = (eval_cols_copy['change'] == 'Ch').astype(float)
    new_numeric_cols.append('regimen_changed')

    train_copy['diabetes_med_prescribed'] = (train_copy['diabetesMed'] == 'Yes').astype(float)
    eval_cols_copy['diabetes_med_prescribed'] = (eval_cols_copy['diabetesMed'] == 'Yes').astype(float)
    new_numeric_cols.append('diabetes_med_prescribed')

    train_copy['change_and_insulin'] = (train_copy['regimen_changed'] * train_copy['on_insulin']).astype(float)
    eval_cols_copy['change_and_insulin'] = (eval_cols_copy['regimen_changed'] * eval_cols_copy['on_insulin']).astype(float)
    new_numeric_cols.append('change_and_insulin')

    # Specific Pharmacological Classes: Sulfonylureas, TZDs, and Drug Risk Interactions
    su_drugs = ['glipizide', 'glyburide', 'glimepiride', 'chlorpropamide', 'tolbutamide', 'tolazamide', 'acetohexamide']
    tr_on_su = np.zeros(len(y_train), dtype=float)
    ev_on_su = np.zeros(len(eval_cols_copy['time_in_hospital']), dtype=float)
    for med in su_drugs:
        if med in train_copy:
            tr_on_su = np.maximum(tr_on_su, ((train_copy[med] != 'No') & (train_copy[med] != None)).astype(float))
            ev_on_su = np.maximum(ev_on_su, ((eval_cols_copy[med] != 'No') & (eval_cols_copy[med] != None)).astype(float))
    train_copy['on_sulfonylurea'] = tr_on_su
    eval_cols_copy['on_sulfonylurea'] = ev_on_su
    new_numeric_cols.append('on_sulfonylurea')

    tzd_drugs = ['pioglitazone', 'rosiglitazone', 'troglitazone']
    tr_on_tzd = np.zeros(len(y_train), dtype=float)
    ev_on_tzd = np.zeros(len(eval_cols_copy['time_in_hospital']), dtype=float)
    for med in tzd_drugs:
        if med in train_copy:
            tr_on_tzd = np.maximum(tr_on_tzd, ((train_copy[med] != 'No') & (train_copy[med] != None)).astype(float))
            ev_on_tzd = np.maximum(ev_on_tzd, ((eval_cols_copy[med] != 'No') & (eval_cols_copy[med] != None)).astype(float))
    train_copy['on_tzd'] = tr_on_tzd
    eval_cols_copy['on_tzd'] = ev_on_tzd
    new_numeric_cols.append('on_tzd')

    train_copy['num_oral_meds'] = np.maximum(0.0, train_copy['num_active_meds'] - train_copy['on_insulin']).astype(float)
    eval_cols_copy['num_oral_meds'] = np.maximum(0.0, eval_cols_copy['num_active_meds'] - eval_cols_copy['on_insulin']).astype(float)
    new_numeric_cols.append('num_oral_meds')

    train_copy['insulin_and_su'] = (train_copy['on_insulin'] * tr_on_su).astype(float)
    eval_cols_copy['insulin_and_su'] = (eval_cols_copy['on_insulin'] * ev_on_su).astype(float)
    new_numeric_cols.append('insulin_and_su')

    train_copy['metformin_and_renal'] = (train_copy['on_metformin'] * train_cm['cm_renal']).astype(float)
    eval_cols_copy['metformin_and_renal'] = (eval_cols_copy['on_metformin'] * eval_cm['cm_renal']).astype(float)
    new_numeric_cols.append('metformin_and_renal')

    train_copy['tzd_and_chf'] = (tr_on_tzd * train_cm['cm_chf']).astype(float)
    eval_cols_copy['tzd_and_chf'] = (ev_on_tzd * eval_cm['cm_chf']).astype(float)
    new_numeric_cols.append('tzd_and_chf')

    # 9. Age continuous mapping & Demographics
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0, '[40-50)': 45.0,
        '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0, '[80-90)': 85.0, '[90-100)': 95.0
    }
    def map_age(age_arr):
        return np.array([age_map.get(x, 65.0) for x in age_arr], dtype=float)

    train_copy['age_numeric'] = map_age(train_copy['age'])
    eval_cols_copy['age_numeric'] = map_age(eval_cols_copy['age'])
    new_numeric_cols.append('age_numeric')

    train_copy['is_elderly'] = (train_copy['age_numeric'] >= 75.0).astype(float)
    eval_cols_copy['is_elderly'] = (eval_cols_copy['age_numeric'] >= 75.0).astype(float)
    new_numeric_cols.append('is_elderly')

    # 10. Validated Clinical Indices: LACE & LACE+ Extension
    def calc_lace(los, adm_type, cm_count, emerg):
        l_score = np.where(los < 1, 0,
                  np.where(los == 1, 1,
                  np.where(los == 2, 2,
                  np.where(los == 3, 3,
                  np.where(los <= 6, 4,
                  np.where(los <= 13, 5, 7))))))
        a_score = np.where(adm_type == '1', 3.0, 0.0)
        c_score = np.clip(cm_count, 0, 5)
        e_score = np.clip(emerg, 0, 4)
        return (l_score + a_score + c_score + e_score).astype(float)

    train_copy['lace_score'] = calc_lace(
        train_copy['time_in_hospital'], train_copy['admission_type_id'],
        train_copy['cm_count'], train_copy['number_emergency']
    )
    eval_cols_copy['lace_score'] = calc_lace(
        eval_cols_copy['time_in_hospital'], eval_cols_copy['admission_type_id'],
        eval_cols_copy['cm_count'], eval_cols_copy['number_emergency']
    )
    new_numeric_cols.append('lace_score')

    train_copy['lace_plus'] = (
        train_copy['lace_score'] +
        np.clip(train_copy['number_inpatient'] * 2.0, 0, 6) +
        np.clip(train_copy['weighted_charlson'], 0, 5) +
        np.clip((train_copy['age_numeric'] - 40.0) / 10.0, 0, 5)
    ).astype(float)
    eval_cols_copy['lace_plus'] = (
        eval_cols_copy['lace_score'] +
        np.clip(eval_cols_copy['number_inpatient'] * 2.0, 0, 6) +
        np.clip(eval_cols_copy['weighted_charlson'], 0, 5) +
        np.clip((eval_cols_copy['age_numeric'] - 40.0) / 10.0, 0, 5)
    ).astype(float)
    new_numeric_cols.append('lace_plus')

    # 11. Discharge and Admission Disposition Indicators
    train_copy['disch_home'] = (train_copy['discharge_disposition_id'] == '1').astype(float)
    eval_cols_copy['disch_home'] = (eval_cols_copy['discharge_disposition_id'] == '1').astype(float)
    new_numeric_cols.append('disch_home')

    train_copy['disch_snf'] = (train_copy['discharge_disposition_id'] == '3').astype(float)
    eval_cols_copy['disch_snf'] = (eval_cols_copy['discharge_disposition_id'] == '3').astype(float)
    new_numeric_cols.append('disch_snf')

    train_copy['disch_home_health'] = (train_copy['discharge_disposition_id'] == '6').astype(float)
    eval_cols_copy['disch_home_health'] = (eval_cols_copy['discharge_disposition_id'] == '6').astype(float)
    new_numeric_cols.append('disch_home_health')

    train_copy['disch_ama'] = (train_copy['discharge_disposition_id'] == '7').astype(float)
    eval_cols_copy['disch_ama'] = (eval_cols_copy['discharge_disposition_id'] == '7').astype(float)
    new_numeric_cols.append('disch_ama')

    train_copy['disch_not_home'] = (train_copy['discharge_disposition_id'] != '1').astype(float)
    eval_cols_copy['disch_not_home'] = (eval_cols_copy['discharge_disposition_id'] != '1').astype(float)
    new_numeric_cols.append('disch_not_home')

    train_copy['admsrc_er'] = (train_copy['admission_source_id'] == '7').astype(float)
    eval_cols_copy['admsrc_er'] = (eval_cols_copy['admission_source_id'] == '7').astype(float)
    new_numeric_cols.append('admsrc_er')

    train_copy['admtype_emergency'] = (train_copy['admission_type_id'] == '1').astype(float)
    eval_cols_copy['admtype_emergency'] = (eval_cols_copy['admission_type_id'] == '1').astype(float)
    new_numeric_cols.append('admtype_emergency')

    train_copy['disch_transfer_hospital'] = np.isin(train_copy['discharge_disposition_id'], ['2', '4', '5']).astype(float)
    eval_cols_copy['disch_transfer_hospital'] = np.isin(eval_cols_copy['discharge_disposition_id'], ['2', '4', '5']).astype(float)
    new_numeric_cols.append('disch_transfer_hospital')

    train_copy['disch_rehab'] = (train_copy['discharge_disposition_id'] == '22').astype(float)
    eval_cols_copy['disch_rehab'] = (eval_cols_copy['discharge_disposition_id'] == '22').astype(float)
    new_numeric_cols.append('disch_rehab')

    train_copy['admsrc_transfer'] = np.isin(train_copy['admission_source_id'], ['4', '5', '6']).astype(float)
    eval_cols_copy['admsrc_transfer'] = np.isin(eval_cols_copy['admission_source_id'], ['4', '5', '6']).astype(float)
    new_numeric_cols.append('admsrc_transfer')

    train_copy['admsrc_physician'] = (train_copy['admission_source_id'] == '1').astype(float)
    eval_cols_copy['admsrc_physician'] = (eval_cols_copy['admission_source_id'] == '1').astype(float)
    new_numeric_cols.append('admsrc_physician')

    train_copy['admsrc_clinic'] = (train_copy['admission_source_id'] == '2').astype(float)
    eval_cols_copy['admsrc_clinic'] = (eval_cols_copy['admission_source_id'] == '2').astype(float)
    new_numeric_cols.append('admsrc_clinic')

    # Lab and Informative Missingness Flags
    train_copy['a1c_tested'] = ((train_copy['A1Cresult'] != 'None') & (train_copy['A1Cresult'] != None)).astype(float)
    eval_cols_copy['a1c_tested'] = ((eval_cols_copy['A1Cresult'] != 'None') & (eval_cols_copy['A1Cresult'] != None)).astype(float)
    new_numeric_cols.append('a1c_tested')

    train_copy['a1c_high'] = (train_copy['A1Cresult'] == '>8').astype(float)
    eval_cols_copy['a1c_high'] = (eval_cols_copy['A1Cresult'] == '>8').astype(float)
    new_numeric_cols.append('a1c_high')

    train_copy['glu_high'] = (
        (train_copy['max_glu_serum'] == '>300') | (train_copy['max_glu_serum'] == '>200')
    ).astype(float)
    eval_cols_copy['glu_high'] = (
        (eval_cols_copy['max_glu_serum'] == '>300') | (eval_cols_copy['max_glu_serum'] == '>200')
    ).astype(float)
    new_numeric_cols.append('glu_high')

    train_copy['missing_specialty'] = (
        (train_copy['medical_specialty'] == None) | (train_copy['medical_specialty'] == '?')
    ).astype(float)
    eval_cols_copy['missing_specialty'] = (
        (eval_cols_copy['medical_specialty'] == None) | (eval_cols_copy['medical_specialty'] == '?')
    ).astype(float)
    new_numeric_cols.append('missing_specialty')

    train_copy['missing_payer'] = (
        (train_copy['payer_code'] == None) | (train_copy['payer_code'] == '?')
    ).astype(float)
    eval_cols_copy['missing_payer'] = (
        (eval_cols_copy['payer_code'] == None) | (eval_cols_copy['payer_code'] == '?')
    ).astype(float)
    new_numeric_cols.append('missing_payer')

    # Clinical Inertia: High A1C without Therapy Intensification
    train_copy['clinical_inertia'] = ((train_copy['a1c_high'] == 1.0) & (train_copy['regimen_changed'] == 0.0)).astype(float)
    eval_cols_copy['clinical_inertia'] = ((eval_cols_copy['a1c_high'] == 1.0) & (eval_cols_copy['regimen_changed'] == 0.0)).astype(float)
    new_numeric_cols.append('clinical_inertia')

    train_copy['intensified_high_a1c'] = ((train_copy['a1c_high'] == 1.0) & (train_copy['regimen_changed'] == 1.0)).astype(float)
    eval_cols_copy['intensified_high_a1c'] = ((eval_cols_copy['a1c_high'] == 1.0) & (eval_cols_copy['regimen_changed'] == 1.0)).astype(float)
    new_numeric_cols.append('intensified_high_a1c')

    # High-Risk Discharge Transition & Frailty Interactions
    train_copy['disch_snf_x_inpatient'] = (train_copy['disch_snf'] * train_copy['number_inpatient']).astype(float)
    eval_cols_copy['disch_snf_x_inpatient'] = (eval_cols_copy['disch_snf'] * eval_cols_copy['number_inpatient']).astype(float)
    new_numeric_cols.append('disch_snf_x_inpatient')

    train_copy['disch_home_health_x_inpatient'] = (train_copy['disch_home_health'] * train_copy['number_inpatient']).astype(float)
    eval_cols_copy['disch_home_health_x_inpatient'] = (eval_cols_copy['disch_home_health'] * eval_cols_copy['number_inpatient']).astype(float)
    new_numeric_cols.append('disch_home_health_x_inpatient')

    train_copy['polypharmacy'] = (train_copy['num_medications'] >= 15.0).astype(float)
    eval_cols_copy['polypharmacy'] = (eval_cols_copy['num_medications'] >= 15.0).astype(float)
    new_numeric_cols.append('polypharmacy')

    train_copy['frequent_inpatient_user'] = (train_copy['number_inpatient'] >= 3.0).astype(float)
    eval_cols_copy['frequent_inpatient_user'] = (eval_cols_copy['number_inpatient'] >= 3.0).astype(float)
    new_numeric_cols.append('frequent_inpatient_user')

    train_copy['a1c_unmeasured_emergency'] = ((train_copy['a1c_tested'] == 0.0) & (train_copy['admtype_emergency'] == 1.0)).astype(float)
    eval_cols_copy['a1c_unmeasured_emergency'] = ((eval_cols_copy['a1c_tested'] == 0.0) & (eval_cols_copy['admtype_emergency'] == 1.0)).astype(float)
    new_numeric_cols.append('a1c_unmeasured_emergency')

    train_copy['high_risk_transition'] = (train_copy['disch_not_home'] * (train_copy['weighted_charlson'] >= 2.0)).astype(float)
    eval_cols_copy['high_risk_transition'] = (eval_cols_copy['disch_not_home'] * (eval_cols_copy['weighted_charlson'] >= 2.0)).astype(float)
    new_numeric_cols.append('high_risk_transition')

    # 12. Latent Clinical Phenotyping via Sublinear TF-IDF + TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    def make_clinical_docs(data_dict, n_rows):
        d1 = data_dict['diag_1_3d']
        d2 = data_dict['diag_2_3d']
        d3 = data_dict['diag_3_3d']
        g1 = data_dict['diag_1_group']
        g2 = data_dict['diag_2_group']
        g3 = data_dict['diag_3_group']
        admsrc = data_dict['admission_source_id']
        disch = data_dict['discharge_disposition_id']
        admtype = data_dict['admission_type_id']
        spec = data_dict['medical_specialty']
        insulin = data_dict['insulin']
        a1c = data_dict['A1Cresult']
        glu = data_dict['max_glu_serum']

        docs = []
        for i in range(n_rows):
            tokens = [
                f"d1_{d1[i]}", f"d2_{d2[i]}", f"d3_{d3[i]}",
                f"g1_{g1[i]}", f"g2_{g2[i]}", f"g3_{g3[i]}",
                f"as_{admsrc[i]}", f"dd_{disch[i]}", f"at_{admtype[i]}",
            ]
            if spec[i] is not None and spec[i] != '?':
                tokens.append(f"sp_{spec[i]}")
            if insulin[i] not in (None, 'No'):
                tokens.append(f"ins_{insulin[i]}")
            if a1c[i] not in (None, 'None'):
                tokens.append(f"a1c_{a1c[i]}")
            if glu[i] not in (None, 'None'):
                tokens.append(f"glu_{glu[i]}")
            if data_dict['number_inpatient'][i] >= 2.0:
                tokens.append("inp_high")
            elif data_dict['number_inpatient'][i] == 1.0:
                tokens.append("inp_mid")
            if data_dict['weighted_charlson'][i] >= 3.0:
                tokens.append("cm_severe")
            docs.append(" ".join(tokens))
        return docs

    train_n = len(y_train)
    eval_n = len(eval_cols_copy['time_in_hospital'])
    train_docs = make_clinical_docs(train_copy, train_n)
    eval_docs = make_clinical_docs(eval_cols_copy, eval_n)

    tfidf = TfidfVectorizer(min_df=5, token_pattern=r'\S+', sublinear_tf=True)
    train_doc_mat = tfidf.fit_transform(train_docs)
    eval_doc_mat = tfidf.transform(eval_docs)

    n_svd = 12
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    train_svd = svd.fit_transform(train_doc_mat)
    eval_svd = svd.transform(eval_doc_mat)

    for j in range(n_svd):
        svd_name = f"clinical_svd_{j}"
        train_copy[svd_name] = train_svd[:, j].astype(float)
        eval_cols_copy[svd_name] = eval_svd[:, j].astype(float)
        new_numeric_cols.append(svd_name)

    # Fisher Linear Discriminant Direction in Latent SVD Space
    skf_svd = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_svd_risk = np.zeros(train_n, dtype=float)
    for tr_i, val_i in skf_svd.split(train_svd, y_train):
        mu_p = np.mean(train_svd[tr_i][y_train[tr_i] == 1], axis=0)
        mu_n = np.mean(train_svd[tr_i][y_train[tr_i] == 0], axis=0)
        diff = mu_p - mu_n
        diff = diff / (np.linalg.norm(diff) + 1e-9)
        train_svd_risk[val_i] = np.dot(train_svd[val_i], diff)

    mu_p_full = np.mean(train_svd[y_train == 1], axis=0)
    mu_n_full = np.mean(train_svd[y_train == 0], axis=0)
    diff_full = mu_p_full - mu_n_full
    diff_full = diff_full / (np.linalg.norm(diff_full) + 1e-9)
    eval_svd_risk = np.dot(eval_svd, diff_full)

    train_copy['svd_risk_direction'] = train_svd_risk
    eval_cols_copy['svd_risk_direction'] = eval_svd_risk
    new_numeric_cols.append('svd_risk_direction')

    # 12b. Higher-Order Singular Value Decomposition (HOSVD) on Multi-Way Clinical Tensors
    def hosvd_3d(T, r1=6, r2=4, r3=4):
        n1, n2, n3 = T.shape
        r1, r2, r3 = min(r1, n1), min(r2, n2), min(r3, n3)
        # Mode-1 unfolding
        T1 = T.reshape(n1, n2 * n3)
        U1, _, _ = np.linalg.svd(T1, full_matrices=False)
        U1 = U1[:, :r1]
        # Mode-2 unfolding
        T2 = np.transpose(T, (1, 0, 2)).reshape(n2, n1 * n3)
        U2, _, _ = np.linalg.svd(T2, full_matrices=False)
        U2 = U2[:, :r2]
        # Mode-3 unfolding
        T3 = np.transpose(T, (2, 0, 1)).reshape(n3, n1 * n2)
        U3, _, _ = np.linalg.svd(T3, full_matrices=False)
        U3 = U3[:, :r3]
        # Core tensor G = T x1 U1^T x2 U2^T x3 U3^T
        G = np.einsum('ijk,ir,js,kt->rst', T, U1, U2, U3)
        # Low-rank multilinear tensor reconstruction
        T_hat = np.einsum('rst,ir,js,kt->ijk', G, U1, U2, U3)
        return U1, U2, U3, G, T_hat

    # Tensor 1 Categories: Care Transition (Diagnosis x Discharge x Acuity)
    d_vals, d_counts = np.unique([str(x) for x in train_copy['diag_1_3d']], return_counts=True)
    top_d_codes = set(d_vals[np.argsort(-d_counts)[:35]])
    train_d_keys = [str(c) if str(c) in top_d_codes else f"grp_{g}" for c, g in zip(train_copy['diag_1_3d'], train_copy['diag_1_group'])]
    unique_d_keys = sorted(list(set(train_d_keys)))
    d_map = {k: idx for idx, k in enumerate(unique_d_keys)}
    eval_d_keys = [str(c) if str(c) in top_d_codes else f"grp_{g}" for c, g in zip(eval_cols_copy['diag_1_3d'], eval_cols_copy['diag_1_group'])]
    d_train_idx = np.array([d_map.get(k, 0) for k in train_d_keys], dtype=int)
    d_eval_idx = np.array([d_map.get(k, 0) for k in eval_d_keys], dtype=int)
    N_D = len(unique_d_keys)

    s_vals, s_counts = np.unique([str(x) for x in train_copy['discharge_disposition_id']], return_counts=True)
    top_s_codes = set(s_vals[np.argsort(-s_counts)[:14]])
    train_s_keys = [str(x) if str(x) in top_s_codes else 'other' for x in train_copy['discharge_disposition_id']]
    unique_s_keys = sorted(list(set(train_s_keys)))
    s_map = {k: idx for idx, k in enumerate(unique_s_keys)}
    eval_s_keys = [str(x) if str(x) in top_s_codes else 'other' for x in eval_cols_copy['discharge_disposition_id']]
    s_train_idx = np.array([s_map.get(k, 0) for k in train_s_keys], dtype=int)
    s_eval_idx = np.array([s_map.get(k, 0) for k in eval_s_keys], dtype=int)
    N_S = len(unique_s_keys)

    def get_acuity_idx(inpatient_arr, admsrc_arr):
        inp_b = np.where(inpatient_arr == 0, 0, np.where(inpatient_arr == 1, 1, 2))
        src_b = np.where(admsrc_arr == '7', 0, np.where(admsrc_arr == '1', 1, np.where(np.isin(admsrc_arr, ['4', '5', '6']), 2, 3)))
        return (inp_b * 4 + src_b).astype(int)

    a_train_idx = get_acuity_idx(train_copy['number_inpatient'], train_copy['admission_source_id'])
    a_eval_idx = get_acuity_idx(eval_cols_copy['number_inpatient'], eval_cols_copy['admission_source_id'])
    N_A = 12

    # Tensor 2 Categories: Multimorbidity & Regimen (Diag1_Group x Diag2_Group x Regimen_Status)
    grp_names = ['circulatory', 'respiratory', 'digestive', 'diabetes', 'injury', 'musculoskeletal', 'genitourinary', 'neoplasms', 'other', 'missing']
    grp_map = {g: idx for idx, g in enumerate(grp_names)}
    g1_tr = np.array([grp_map.get(str(x), 8) for x in train_copy['diag_1_group']], dtype=int)
    g2_tr = np.array([grp_map.get(str(x), 8) for x in train_copy['diag_2_group']], dtype=int)
    g1_ev = np.array([grp_map.get(str(x), 8) for x in eval_cols_copy['diag_1_group']], dtype=int)
    g2_ev = np.array([grp_map.get(str(x), 8) for x in eval_cols_copy['diag_2_group']], dtype=int)
    N_G = len(grp_names)

    m_tr = (train_copy['on_insulin'].astype(int) * 4 + train_copy['regimen_changed'].astype(int) * 2 + (train_copy['number_emergency'] > 0).astype(int)).astype(int)
    m_ev = (eval_cols_copy['on_insulin'].astype(int) * 4 + eval_cols_copy['regimen_changed'].astype(int) * 2 + (eval_cols_copy['number_emergency'] > 0).astype(int)).astype(int)
    N_M = 8

    # Out-of-fold computation for Training rows
    tr_care_recon = np.zeros(train_n, dtype=float)
    tr_care_u1_0 = np.zeros(train_n, dtype=float)
    tr_care_u1_1 = np.zeros(train_n, dtype=float)
    tr_care_u2_0 = np.zeros(train_n, dtype=float)
    tr_care_u3_0 = np.zeros(train_n, dtype=float)
    tr_care_inter = np.zeros(train_n, dtype=float)
    tr_reg_recon = np.zeros(train_n, dtype=float)
    tr_reg_concord = np.zeros(train_n, dtype=float)

    for tr_idx, val_idx in skf_svd.split(train_copy['diag_1'], y_train):
        # Tensor 1: Care Transition
        cnt1 = np.zeros((N_D, N_S, N_A), dtype=float)
        pos1 = np.zeros((N_D, N_S, N_A), dtype=float)
        np.add.at(cnt1, (d_train_idx[tr_idx], s_train_idx[tr_idx], a_train_idx[tr_idx]), 1.0)
        np.add.at(pos1, (d_train_idx[tr_idx], s_train_idx[tr_idx], a_train_idx[tr_idx]), y_train[tr_idx])
        p1 = np.mean(y_train[tr_idx])
        rate1 = (pos1 + 15.0 * p1) / (cnt1 + 15.0)
        T1 = np.log(rate1 / (1.0 - rate1)) - np.log(p1 / (1.0 - p1))
        u1, u2, u3, _, That1 = hosvd_3d(T1, r1=6, r2=4, r3=4)

        d_v, s_v, a_v = d_train_idx[val_idx], s_train_idx[val_idx], a_train_idx[val_idx]
        tr_care_recon[val_idx] = That1[d_v, s_v, a_v]
        tr_care_u1_0[val_idx] = u1[d_v, 0]
        tr_care_u1_1[val_idx] = u1[d_v, 1]
        tr_care_u2_0[val_idx] = u2[s_v, 0]
        tr_care_u3_0[val_idx] = u3[a_v, 0]
        tr_care_inter[val_idx] = u1[d_v, 0] * u2[s_v, 0] + u1[d_v, 1] * u2[s_v, 1]

        # Tensor 2: Multimorbidity & Regimen
        cnt2 = np.zeros((N_G, N_G, N_M), dtype=float)
        pos2 = np.zeros((N_G, N_G, N_M), dtype=float)
        np.add.at(cnt2, (g1_tr[tr_idx], g2_tr[tr_idx], m_tr[tr_idx]), 1.0)
        np.add.at(pos2, (g1_tr[tr_idx], g2_tr[tr_idx], m_tr[tr_idx]), y_train[tr_idx])
        rate2 = (pos2 + 15.0 * p1) / (cnt2 + 15.0)
        T2 = np.log(rate2 / (1.0 - rate2)) - np.log(p1 / (1.0 - p1))
        ug1, ug2, _, _, That2 = hosvd_3d(T2, r1=4, r2=4, r3=3)

        g1_v, g2_v, mv_v = g1_tr[val_idx], g2_tr[val_idx], m_tr[val_idx]
        tr_reg_recon[val_idx] = That2[g1_v, g2_v, mv_v]
        tr_reg_concord[val_idx] = ug1[g1_v, 0] * ug2[g2_v, 0] + ug1[g1_v, 1] * ug2[g2_v, 1]

    # Full Train Tensor Decomposition for Evaluation rows
    cnt1_full = np.zeros((N_D, N_S, N_A), dtype=float)
    pos1_full = np.zeros((N_D, N_S, N_A), dtype=float)
    np.add.at(cnt1_full, (d_train_idx, s_train_idx, a_train_idx), 1.0)
    np.add.at(pos1_full, (d_train_idx, s_train_idx, a_train_idx), y_train)
    p_full = np.mean(y_train)
    rate1_full = (pos1_full + 15.0 * p_full) / (cnt1_full + 15.0)
    T1_full = np.log(rate1_full / (1.0 - rate1_full)) - np.log(p_full / (1.0 - p_full))
    u1_f, u2_f, u3_f, _, That1_f = hosvd_3d(T1_full, r1=6, r2=4, r3=4)

    ev_care_recon = That1_f[d_eval_idx, s_eval_idx, a_eval_idx]
    ev_care_u1_0 = u1_f[d_eval_idx, 0]
    ev_care_u1_1 = u1_f[d_eval_idx, 1]
    ev_care_u2_0 = u2_f[s_eval_idx, 0]
    ev_care_u3_0 = u3_f[a_eval_idx, 0]
    ev_care_inter = u1_f[d_eval_idx, 0] * u2_f[s_eval_idx, 0] + u1_f[d_eval_idx, 1] * u2_f[s_eval_idx, 1]

    cnt2_full = np.zeros((N_G, N_G, N_M), dtype=float)
    pos2_full = np.zeros((N_G, N_G, N_M), dtype=float)
    np.add.at(cnt2_full, (g1_tr, g2_tr, m_tr), 1.0)
    np.add.at(pos2_full, (g1_tr, g2_tr, m_tr), y_train)
    rate2_full = (pos2_full + 15.0 * p_full) / (cnt2_full + 15.0)
    T2_full = np.log(rate2_full / (1.0 - rate2_full)) - np.log(p_full / (1.0 - p_full))
    ug1_f, ug2_f, _, _, That2_f = hosvd_3d(T2_full, r1=4, r2=4, r3=3)

    ev_reg_recon = That2_f[g1_ev, g2_ev, m_ev]
    ev_reg_concord = ug1_f[g1_ev, 0] * ug2_f[g2_ev, 0] + ug1_f[g1_ev, 1] * ug2_f[g2_ev, 1]

    tensor_features = [
        ('tensor_care_recon', tr_care_recon, ev_care_recon),
        ('tensor_care_u1_0', tr_care_u1_0, ev_care_u1_0),
        ('tensor_care_u1_1', tr_care_u1_1, ev_care_u1_1),
        ('tensor_care_u2_0', tr_care_u2_0, ev_care_u2_0),
        ('tensor_care_u3_0', tr_care_u3_0, ev_care_u3_0),
        ('tensor_care_inter', tr_care_inter, ev_care_inter),
        ('tensor_reg_recon', tr_reg_recon, ev_reg_recon),
        ('tensor_reg_concord', tr_reg_concord, ev_reg_concord),
    ]
    for tf_name, tr_arr, ev_arr in tensor_features:
        train_copy[tf_name] = tr_arr
        eval_cols_copy[tf_name] = ev_arr
        new_numeric_cols.append(tf_name)

    # 13. Encode features using TabularEncoder
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        train_copy,
        extra_numeric=tuple(new_numeric_cols),
        extra_categorical=tuple(new_categorical_cols)
    )
    x_train = encoder.transform(train_copy)
    x_eval = encoder.transform(eval_cols_copy)

    # 14. 5-Fold Stratified Tri-Engine Heterogeneous Ensemble
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds_1 = np.zeros(len(x_train), dtype=float)
    oof_preds_2 = np.zeros(len(x_train), dtype=float)
    oof_preds_3 = np.zeros(len(x_train), dtype=float)

    eval_preds_1 = np.zeros(len(x_eval), dtype=float)
    eval_preds_2 = np.zeros(len(x_eval), dtype=float)
    eval_preds_3 = np.zeros(len(x_eval), dtype=float)

    for train_idx, val_idx in skf.split(x_train, y_train):
        x_tr, y_tr = x_train[train_idx], y_train[train_idx]
        x_val = x_train[val_idx]

        # Engine 1: LightGBM Standard (Leaf-wise balanced gradient booster)
        lgb_1 = lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.020,
            num_leaves=35,
            min_child_samples=110,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.62,
            reg_lambda=2.5,
            reg_alpha=0.4,
            cat_smooth=35.0,
            path_smooth=1.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        lgb_1.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_preds_1[val_idx] = lgb_1.predict_proba(x_val)[:, 1]
        eval_preds_1 += lgb_1.predict_proba(x_eval)[:, 1] / 5.0

        # Engine 2: LightGBM Deep/Interactive (Higher-capacity interaction booster)
        lgb_2 = lgb.LGBMClassifier(
            n_estimators=290,
            learning_rate=0.018,
            num_leaves=55,
            max_depth=7,
            min_child_samples=85,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.58,
            reg_lambda=3.0,
            reg_alpha=0.5,
            cat_smooth=30.0,
            path_smooth=1.5,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        )
        lgb_2.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_preds_2[val_idx] = lgb_2.predict_proba(x_val)[:, 1]
        eval_preds_2 += lgb_2.predict_proba(x_eval)[:, 1] / 5.0

        # Engine 3: XGBoost (Depth-wise histogram gradient booster)
        xgb_model = xgb.XGBClassifier(
            n_estimators=240,
            learning_rate=0.024,
            max_depth=5,
            min_child_weight=14,
            subsample=0.8,
            colsample_bytree=0.58,
            reg_lambda=2.5,
            reg_alpha=0.3,
            gamma=0.1,
            tree_method='hist',
            eval_metric='logloss',
            random_state=42,
            n_jobs=1,
        )
        xgb_model.fit(x_tr, y_tr)
        oof_preds_3[val_idx] = xgb_model.predict_proba(x_val)[:, 1]
        eval_preds_3 += xgb_model.predict_proba(x_eval)[:, 1] / 5.0

    # 15. Multi-Space Simplex Optimization (Logit / Rank / Prob) & Platt Calibration
    from sklearn.linear_model import LogisticRegression

    n_samples = len(y_train)
    r1 = rankdata(oof_preds_1) / n_samples
    r2 = rankdata(oof_preds_2) / n_samples
    r3 = rankdata(oof_preds_3) / n_samples

    def standardize_logits(p):
        p_c = np.clip(p, 1e-6, 1.0 - 1e-6)
        z = np.log(p_c / (1.0 - p_c))
        mu = np.mean(z)
        sig = np.std(z) + 1e-9
        return (z - mu) / sig, mu, sig

    z1, mu1, sig1 = standardize_logits(oof_preds_1)
    z2, mu2, sig2 = standardize_logits(oof_preds_2)
    z3, mu3, sig3 = standardize_logits(oof_preds_3)

    ev_z1 = (np.log(np.clip(eval_preds_1, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(eval_preds_1, 1e-6, 1.0 - 1e-6))) - mu1) / sig1
    ev_z2 = (np.log(np.clip(eval_preds_2, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(eval_preds_2, 1e-6, 1.0 - 1e-6))) - mu2) / sig2
    ev_z3 = (np.log(np.clip(eval_preds_3, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(eval_preds_3, 1e-6, 1.0 - 1e-6))) - mu3) / sig3

    best_auc = -1.0
    best_weights = (0.40, 0.35, 0.25)
    best_mode = 'logit'

    # Simplex Grid Search across Logit, Rank, and Probability spaces
    steps = np.arange(0.05, 0.85, 0.04)
    for w1 in steps:
        for w2 in steps:
            w3 = 1.0 - w1 - w2
            if w3 < 0.04:
                continue

            # 1. Log-Odds Consensus Pooling
            blend_z = w1 * z1 + w2 * z2 + w3 * z3
            score_z = roc_auc_score(y_train, blend_z)
            if score_z > best_auc:
                best_auc = score_z
                best_weights = (w1, w2, w3)
                best_mode = 'logit'

            # 2. Non-Parametric Rank Consensus
            blend_r = w1 * r1 + w2 * r2 + w3 * r3
            score_r = roc_auc_score(y_train, blend_r)
            if score_r > best_auc:
                best_auc = score_r
                best_weights = (w1, w2, w3)
                best_mode = 'rank'

            # 3. Direct Probability Mixture
            blend_p = w1 * oof_preds_1 + w2 * oof_preds_2 + w3 * oof_preds_3
            score_p = roc_auc_score(y_train, blend_p)
            if score_p > best_auc:
                best_auc = score_p
                best_weights = (w1, w2, w3)
                best_mode = 'prob'

    # Local Simplex Coordinate Refinement
    w1_opt, w2_opt, w3_opt = best_weights
    delta_grid = [-0.02, -0.01, 0.0, 0.01, 0.02]
    for d1 in delta_grid:
        for d2 in delta_grid:
            w1_c = np.clip(w1_opt + d1, 0.05, 0.90)
            w2_c = np.clip(w2_opt + d2, 0.05, 0.90)
            w3_c = 1.0 - w1_c - w2_c
            if w3_c < 0.04:
                continue
            if best_mode == 'logit':
                sc = roc_auc_score(y_train, w1_c * z1 + w2_c * z2 + w3_c * z3)
            elif best_mode == 'rank':
                sc = roc_auc_score(y_train, w1_c * r1 + w2_c * r2 + w3_c * r3)
            else:
                sc = roc_auc_score(y_train, w1_c * oof_preds_1 + w2_c * oof_preds_2 + w3_c * oof_preds_3)
            if sc > best_auc:
                best_auc = sc
                best_weights = (w1_c, w2_c, w3_c)

    # Monotonic Platt Calibration on Best Consensus
    bw1, bw2, bw3 = best_weights
    if best_mode == 'logit':
        oof_best = bw1 * z1 + bw2 * z2 + bw3 * z3
        ev_best = bw1 * ev_z1 + bw2 * ev_z2 + bw3 * ev_z3
    elif best_mode == 'rank':
        n_eval = len(eval_preds_1)
        ev_r1 = rankdata(eval_preds_1) / n_eval
        ev_r2 = rankdata(eval_preds_2) / n_eval
        ev_r3 = rankdata(eval_preds_3) / n_eval
        oof_best = bw1 * r1 + bw2 * r2 + bw3 * r3
        ev_best = bw1 * ev_r1 + bw2 * ev_r2 + bw3 * ev_r3
    else:
        oof_best = bw1 * oof_preds_1 + bw2 * oof_preds_2 + bw3 * oof_preds_3
        ev_best = bw1 * eval_preds_1 + bw2 * eval_preds_2 + bw3 * eval_preds_3

    calibrator = LogisticRegression(C=1.0, solver='lbfgs', max_iter=200)
    calibrator.fit(oof_best.reshape(-1, 1), y_train)
    final_eval_preds = calibrator.predict_proba(ev_best.reshape(-1, 1))[:, 1]

    return np.clip(np.nan_to_num(final_eval_preds, nan=0.115), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
