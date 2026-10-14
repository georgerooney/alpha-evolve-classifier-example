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

    # 11b. Advanced Clinical Pathophysiology: Deadly Triad, Discordance & Drug Classes
    train_n = len(y_train)
    eval_n = len(eval_cols_copy['time_in_hospital'])

    # Diabetic Cardiorenal Triad: Diabetes + CHF + Renal Failure
    train_copy['deadly_triad'] = (
        train_cm['cm_chf'] * train_cm['cm_renal'] *
        ((train_copy['flag_diabetes_diag'] > 0) | (train_copy['diabetes_med_prescribed'] > 0)).astype(float)
    ).astype(float)
    eval_cols_copy['deadly_triad'] = (
        eval_cm['cm_chf'] * eval_cm['cm_renal'] *
        ((eval_cols_copy['flag_diabetes_diag'] > 0) | (eval_cols_copy['diabetes_med_prescribed'] > 0)).astype(float)
    ).astype(float)
    new_numeric_cols.append('deadly_triad')

    # Multi-system organ involvement: number of distinct ICD-9 chapters involved
    def count_distinct_chapters(g1, g2, g3):
        n = len(g1)
        out = np.zeros(n, dtype=float)
        for i in range(n):
            s = set()
            for g in (g1[i], g2[i], g3[i]):
                if g not in ('missing', 'other', None):
                    s.add(g)
            out[i] = float(len(s)) if len(s) > 0 else 1.0
        return out

    train_copy['distinct_diag_groups'] = count_distinct_chapters(train_copy['diag_1_group'], train_copy['diag_2_group'], train_copy['diag_3_group'])
    eval_cols_copy['distinct_diag_groups'] = count_distinct_chapters(eval_cols_copy['diag_1_group'], eval_cols_copy['diag_2_group'], eval_cols_copy['diag_3_group'])
    new_numeric_cols.append('distinct_diag_groups')

    # Multi-class Diabetic Regimen Complexity
    diab_classes = [
        ['metformin', 'glyburide-metformin', 'glipizide-metformin', 'metformin-rosiglitazone', 'metformin-pioglitazone'],
        ['glipizide', 'glyburide', 'glimepiride', 'chlorpropamide', 'tolbutamide', 'tolazamide', 'acetohexamide', 'glimepiride-pioglitazone'],
        ['pioglitazone', 'rosiglitazone', 'troglitazone', 'metformin-pioglitazone', 'metformin-rosiglitazone', 'glimepiride-pioglitazone'],
        ['repaglinide', 'nateglinide'],
        ['acarbose', 'miglitol'],
        ['insulin']
    ]
    def count_classes(d_dict, n_rows):
        counts = np.zeros(n_rows, dtype=float)
        for cls_drugs in diab_classes:
            has_cls = np.zeros(n_rows, dtype=bool)
            for drug in cls_drugs:
                if drug in d_dict:
                    has_cls = has_cls | ((d_dict[drug] != 'No') & (d_dict[drug] != None))
            counts += has_cls.astype(float)
        return counts

    train_copy['num_diab_drug_classes'] = count_classes(train_copy, train_n)
    eval_cols_copy['num_diab_drug_classes'] = count_classes(eval_cols_copy, eval_n)
    new_numeric_cols.append('num_diab_drug_classes')

    # High-Risk Discharge Synergy: SNF x Elderly, AMA x Prior Emergency
    train_copy['disch_high_risk'] = np.isin(train_copy['discharge_disposition_id'], ['3', '7', '22']).astype(float)
    eval_cols_copy['disch_high_risk'] = np.isin(eval_cols_copy['discharge_disposition_id'], ['3', '7', '22']).astype(float)
    new_numeric_cols.append('disch_high_risk')

    train_copy['snf_and_elderly'] = (train_copy['disch_snf'] * train_copy['is_elderly']).astype(float)
    eval_cols_copy['snf_and_elderly'] = (eval_cols_copy['disch_snf'] * eval_cols_copy['is_elderly']).astype(float)
    new_numeric_cols.append('snf_and_elderly')

    train_copy['ama_and_prior_emerg'] = (train_copy['disch_ama'] * (train_copy['number_emergency'] > 0)).astype(float)
    eval_cols_copy['ama_and_prior_emerg'] = (eval_cols_copy['disch_ama'] * (eval_cols_copy['number_emergency'] > 0)).astype(float)
    new_numeric_cols.append('ama_and_prior_emerg')

    # === CLINICAL PHENOME GRAPH DIFFUSION & TOPOLOGICAL SHOCK ENGINE ===
    from collections import Counter
    import scipy.sparse as sp

    def get_encounter_tokens(data_dict, n_rows):
        d1 = data_dict['diag_1_3d']
        d2 = data_dict['diag_2_3d']
        d3 = data_dict['diag_3_3d']
        g1 = data_dict['diag_1_group']
        g2 = data_dict['diag_2_group']
        g3 = data_dict['diag_3_group']
        disch = data_dict['discharge_disposition_id']
        admsrc = data_dict['admission_source_id']
        admtype = data_dict['admission_type_id']
        spec = data_dict['medical_specialty']
        ins = data_dict['insulin']
        chg = data_dict['change']

        row_tokens = []
        for i in range(n_rows):
            toks = [
                f"d1_{d1[i]}", f"d2_{d2[i]}", f"d3_{d3[i]}",
                f"g1_{g1[i]}", f"g2_{g2[i]}", f"g3_{g3[i]}",
                f"dd_{disch[i]}", f"as_{admsrc[i]}", f"at_{admtype[i]}",
            ]
            if spec[i] is not None and spec[i] != '?':
                toks.append(f"sp_{spec[i]}")
            if ins[i] not in (None, 'No'):
                toks.append(f"ins_{ins[i]}")
            if chg[i] == 'Ch':
                toks.append("chg_Ch")
            row_tokens.append(toks)
        return row_tokens

    train_tokens = get_encounter_tokens(train_copy, train_n)
    eval_tokens = get_encounter_tokens(eval_cols_copy, eval_n)

    token_counts = Counter()
    for toks in train_tokens:
        token_counts.update(toks)

    # Filter frequent graph nodes (min_freq=15)
    vocab = [tok for tok, cnt in token_counts.items() if cnt >= 15]
    vocab_map = {tok: idx for idx, tok in enumerate(vocab)}
    n_nodes = len(vocab)

    def build_indicator_matrix(row_toks, n_rows, vocab_dict, num_nodes):
        rows, cols = [], []
        for i, toks in enumerate(row_toks):
            for t in toks:
                if t in vocab_dict:
                    rows.append(i)
                    cols.append(vocab_dict[t])
        data = np.ones(len(rows), dtype=np.float32)
        return sp.csr_matrix((data, (rows, cols)), shape=(n_rows, num_nodes))

    B_tr = build_indicator_matrix(train_tokens, train_n, vocab_map, n_nodes)
    B_ev = build_indicator_matrix(eval_tokens, eval_n, vocab_map, n_nodes)

    # Co-occurrence Affinity & Row-Stochastic Diffusion Kernel
    C_mat = (B_tr.T @ B_tr).toarray().astype(float)
    diag_c = np.diag(C_mat)
    denom = np.sqrt(np.outer(diag_c, diag_c)) + 1e-9
    A_mat = C_mat / denom
    np.fill_diagonal(A_mat, 1.0)

    deg = np.sum(A_mat, axis=1, keepdims=True)
    P_mat = A_mat / np.maximum(deg, 1e-9)

    # Closed-form Random Walk with Restart (RWR) kernel: K = alpha * (I - (1 - alpha) * P)^-1
    alpha_rwr = 0.35
    M_mat = np.eye(n_nodes) - (1.0 - alpha_rwr) * P_mat
    K_mat = alpha_rwr * np.linalg.inv(M_mat)

    # Diffuse patient encounter states through the clinical phenome graph
    row_sums_tr = np.array(B_tr.sum(axis=1)).flatten()
    row_sums_ev = np.array(B_ev.sum(axis=1)).flatten()
    inv_s_tr = np.where(row_sums_tr > 0, 1.0 / row_sums_tr, 0.0)
    inv_s_ev = np.where(row_sums_ev > 0, 1.0 / row_sums_ev, 0.0)

    S_tr = B_tr.multiply(inv_s_tr[:, None])
    S_ev = B_ev.multiply(inv_s_ev[:, None])

    Phi_tr = S_tr.dot(K_mat)
    Phi_ev = S_ev.dot(K_mat)

    # Signal 1: Out-of-Fold Diffused Readmission Risk
    tr_diff_risk = np.zeros(train_n, dtype=float)
    smooth_node = 25.0
    global_prior = float(np.mean(y_train))
    node_sums_tr = np.array(B_tr.T.dot(y_train)).flatten()
    node_counts_tr = np.array(B_tr.sum(axis=0)).flatten()
    global_node_risk = (node_sums_tr + smooth_node * global_prior) / (node_counts_tr + smooth_node)
    ev_diff_risk = Phi_ev.dot(global_node_risk)

    kf_diff = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_tr_idx, fold_val_idx in kf_diff.split(Phi_tr, y_train):
        b_sub = B_tr[fold_tr_idx]
        y_sub = y_train[fold_tr_idx]
        prior_sub = float(np.mean(y_sub))
        n_sums = np.array(b_sub.T.dot(y_sub)).flatten()
        n_counts = np.array(b_sub.sum(axis=0)).flatten()
        fold_risk = (n_sums + smooth_node * prior_sub) / (n_counts + smooth_node)
        tr_diff_risk[fold_val_idx] = Phi_tr[fold_val_idx].dot(fold_risk)

    train_copy['graph_diffused_risk'] = tr_diff_risk
    eval_cols_copy['graph_diffused_risk'] = ev_diff_risk
    new_numeric_cols.append('graph_diffused_risk')

    # Signal 2: Phenotypic Graph Entropy (Systemic vs. Focal Disease Instability)
    tr_entropy = -np.sum(Phi_tr * np.log(Phi_tr + 1e-12), axis=1)
    ev_entropy = -np.sum(Phi_ev * np.log(Phi_ev + 1e-12), axis=1)
    train_copy['graph_phenotype_entropy'] = tr_entropy
    eval_cols_copy['graph_phenotype_entropy'] = ev_entropy
    new_numeric_cols.append('graph_phenotype_entropy')

    # Signal 3: Acute Decompensation Cluster Mass
    acute_tokens = [
        'd1_428', 'd2_428', 'd3_428',
        'd1_584', 'd1_585', 'd2_585', 'd3_585',
        'd1_410', 'd1_411',
        'd1_038', 'd2_038',
        'd1_486', 'd1_491', 'd1_496',
        'dd_3', 'dd_7'
    ]
    acute_indices = [vocab_map[tok] for tok in acute_tokens if tok in vocab_map]
    if len(acute_indices) > 0:
        train_copy['graph_acute_mass'] = np.sum(Phi_tr[:, acute_indices], axis=1)
        eval_cols_copy['graph_acute_mass'] = np.sum(Phi_ev[:, acute_indices], axis=1)
        new_numeric_cols.append('graph_acute_mass')

    # Signal 4: Continuous Spectral Diffusion Manifold Coordinates
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    svd_diff = TruncatedSVD(n_components=5, random_state=42)
    diff_coords_tr = svd_diff.fit_transform(Phi_tr)
    diff_coords_ev = svd_diff.transform(Phi_ev)
    for j in range(5):
        coord_name = f"graph_diffusion_dim_{j}"
        train_copy[coord_name] = diff_coords_tr[:, j].astype(float)
        eval_cols_copy[coord_name] = diff_coords_ev[:, j].astype(float)
        new_numeric_cols.append(coord_name)

    # 12. Latent Clinical Phenotyping via Sublinear TF-IDF + TruncatedSVD

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
            docs.append(" ".join(tokens))
        return docs

    train_docs = make_clinical_docs(train_copy, train_n)
    eval_docs = make_clinical_docs(eval_cols_copy, eval_n)

    tfidf = TfidfVectorizer(min_df=5, token_pattern=r'\S+', sublinear_tf=True)
    train_doc_mat = tfidf.fit_transform(train_docs)
    eval_doc_mat = tfidf.transform(eval_docs)

    n_svd = 10
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    train_svd = svd.fit_transform(train_doc_mat)
    eval_svd = svd.transform(eval_doc_mat)

    for j in range(n_svd):
        svd_name = f"clinical_svd_{j}"
        train_copy[svd_name] = train_svd[:, j].astype(float)
        eval_cols_copy[svd_name] = eval_svd[:, j].astype(float)
        new_numeric_cols.append(svd_name)

    # === MULTI-WAY CLINICAL & UTILIZATION TENSOR FACTORIZATION (CP-ALS) ===
    def get_clinical_phenotype(flag_chf, flag_cad, flag_copd, flag_renal, diab_uncontrolled,
                               flag_sepsis, flag_cerebrovasc, flag_electrolyte, flag_uti, g1):
        n = len(g1)
        pheno = np.zeros(n, dtype=int)
        for i in range(n):
            if flag_chf[i] > 0:
                pheno[i] = 0
            elif flag_cad[i] > 0:
                pheno[i] = 1
            elif flag_copd[i] > 0:
                pheno[i] = 2
            elif flag_renal[i] > 0:
                pheno[i] = 3
            elif diab_uncontrolled[i] > 0:
                pheno[i] = 4
            elif flag_sepsis[i] > 0:
                pheno[i] = 5
            elif flag_cerebrovasc[i] > 0:
                pheno[i] = 6
            elif flag_electrolyte[i] > 0:
                pheno[i] = 7
            elif flag_uti[i] > 0:
                pheno[i] = 8
            else:
                grp = str(g1[i])
                if grp == 'circulatory':
                    pheno[i] = 9
                elif grp == 'respiratory':
                    pheno[i] = 10
                elif grp == 'digestive':
                    pheno[i] = 11
                elif grp == 'diabetes':
                    pheno[i] = 12
                elif grp == 'genitourinary':
                    pheno[i] = 13
                elif grp == 'neoplasms':
                    pheno[i] = 14
                elif grp == 'musculoskeletal':
                    pheno[i] = 15
                elif grp == 'injury':
                    pheno[i] = 16
                else:
                    pheno[i] = 17
        return pheno

    def get_discharge_trajectory(disch_arr):
        n = len(disch_arr)
        traj = np.zeros(n, dtype=int)
        mapping = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '18': 7, '22': 8}
        for i in range(n):
            v = str(disch_arr[i])
            traj[i] = mapping.get(v, 9)
        return traj

    def get_treatment_regimen(insulin_arr, change_arr, num_active_arr, on_su_arr, on_met_arr):
        n = len(insulin_arr)
        reg = np.zeros(n, dtype=int)
        for i in range(n):
            ins = str(insulin_arr[i])
            chg = str(change_arr[i])
            n_act = num_active_arr[i]
            su = on_su_arr[i]
            met = on_met_arr[i]
            if ins == 'Up':
                reg[i] = 0
            elif ins == 'Down':
                reg[i] = 1
            elif ins == 'Steady':
                if n_act > 1:
                    reg[i] = 2
                else:
                    reg[i] = 3
            else:
                if chg == 'Ch':
                    reg[i] = 4
                elif su > 0:
                    reg[i] = 5
                elif met > 0:
                    reg[i] = 6
                else:
                    reg[i] = 7
        return reg

    def get_util_tier(inp, emerg, outp):
        n = len(inp)
        tier = np.zeros(n, dtype=int)
        for i in range(n):
            ni = inp[i]
            ne = emerg[i]
            no = outp[i]
            if ni >= 2:
                tier[i] = 0
            elif ni == 1:
                tier[i] = 1
            elif ne >= 2:
                tier[i] = 2
            elif ne == 1:
                tier[i] = 3
            elif no > 0:
                tier[i] = 4
            else:
                tier[i] = 5
        return tier

    def get_stay_acuity(los, adm_type):
        n = len(los)
        acuity = np.zeros(n, dtype=int)
        for i in range(n):
            l = los[i]
            is_emerg = (str(adm_type[i]) == '1')
            if l <= 2:
                acuity[i] = 0 if is_emerg else 1
            elif l <= 5:
                acuity[i] = 2 if is_emerg else 3
            else:
                acuity[i] = 4
        return acuity

    def get_age_bracket(age_num):
        n = len(age_num)
        b = np.zeros(n, dtype=int)
        for i in range(n):
            a = age_num[i]
            if a < 50:
                b[i] = 0
            elif a < 60:
                b[i] = 1
            elif a < 70:
                b[i] = 2
            elif a < 80:
                b[i] = 3
            else:
                b[i] = 4
        return b

    def cp_als(tensor, rank=4, n_iter=10, reg=1e-4, seed=42):
        I, J, K = tensor.shape
        R = rank

        T1 = tensor.reshape(I, -1)
        u1, _, _ = np.linalg.svd(T1, full_matrices=False)
        A = u1[:, :R].copy()
        if A.shape[1] < R:
            pad = np.random.RandomState(seed).randn(I, R - A.shape[1])
            A = np.hstack([A, pad])

        T2 = np.transpose(tensor, (1, 0, 2)).reshape(J, -1)
        u2, _, _ = np.linalg.svd(T2, full_matrices=False)
        B = u2[:, :R].copy()
        if B.shape[1] < R:
            pad = np.random.RandomState(seed).randn(J, R - B.shape[1])
            B = np.hstack([B, pad])

        T3 = np.transpose(tensor, (2, 0, 1)).reshape(K, -1)
        u3, _, _ = np.linalg.svd(T3, full_matrices=False)
        C = u3[:, :R].copy()
        if C.shape[1] < R:
            pad = np.random.RandomState(seed).randn(K, R - C.shape[1])
            C = np.hstack([C, pad])

        A /= (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
        B /= (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
        C /= (np.linalg.norm(C, axis=0, keepdims=True) + 1e-12)

        lambdas = np.ones(R, dtype=float)
        eye_R = np.eye(R) * reg

        for _ in range(n_iter):
            BC = np.einsum('jr,kr->jkr', B, C).reshape(J * K, R)
            V_A = (B.T @ B) * (C.T @ C) + eye_R
            A = (T1 @ BC) @ np.linalg.inv(V_A)
            norm_A = np.linalg.norm(A, axis=0) + 1e-12
            A /= norm_A

            AC = np.einsum('ir,kr->ikr', A, C).reshape(I * K, R)
            V_B = (A.T @ A) * (C.T @ C) + eye_R
            B = (T2 @ AC) @ np.linalg.inv(V_B)
            norm_B = np.linalg.norm(B, axis=0) + 1e-12
            B /= norm_B

            AB = np.einsum('ir,jr->ijr', A, B).reshape(I * J, R)
            V_C = (A.T @ A) * (B.T @ B) + eye_R
            C = (T3 @ AB) @ np.linalg.inv(V_C)
            lambdas = np.linalg.norm(C, axis=0) + 1e-12
            C /= lambdas

        T_recon = np.einsum('r,ir,jr,kr->ijk', lambdas, A, B, C)
        return A, B, C, lambdas, T_recon

    def build_smoothed_risk_tensor(i_arr, j_arr, k_arr, y_arr, I, J, K, alpha=15.0):
        counts = np.zeros((I, J, K), dtype=float)
        positives = np.zeros((I, J, K), dtype=float)
        np.add.at(counts, (i_arr, j_arr, k_arr), 1.0)
        np.add.at(positives, (i_arr, j_arr, k_arr), y_arr.astype(float))
        global_mean = float(np.mean(y_arr))
        risk_tensor = (positives + alpha * global_mean) / (counts + alpha)
        return risk_tensor

    # Mode encodings for Tensor 1: Phenotype (18) x Trajectory (10) x Regimen (8)
    tr_pheno = get_clinical_phenotype(
        train_copy['flag_chf'], train_copy['flag_cad_mi'], train_copy['flag_copd_pneu'],
        train_copy['flag_renal'], train_copy['diab_uncontrolled'], train_copy['flag_sepsis'],
        train_copy['flag_cerebrovasc'], train_copy['flag_electrolyte'], train_copy['flag_uti'],
        train_copy['diag_1_group']
    )
    ev_pheno = get_clinical_phenotype(
        eval_cols_copy['flag_chf'], eval_cols_copy['flag_cad_mi'], eval_cols_copy['flag_copd_pneu'],
        eval_cols_copy['flag_renal'], eval_cols_copy['diab_uncontrolled'], eval_cols_copy['flag_sepsis'],
        eval_cols_copy['flag_cerebrovasc'], eval_cols_copy['flag_electrolyte'], eval_cols_copy['flag_uti'],
        eval_cols_copy['diag_1_group']
    )

    tr_traj = get_discharge_trajectory(train_copy['discharge_disposition_id'])
    ev_traj = get_discharge_trajectory(eval_cols_copy['discharge_disposition_id'])

    tr_reg = get_treatment_regimen(
        train_copy['insulin'], train_copy['change'], train_copy['num_active_meds'],
        train_copy['on_sulfonylurea'], train_copy['on_metformin']
    )
    ev_reg = get_treatment_regimen(
        eval_cols_copy['insulin'], eval_cols_copy['change'], eval_cols_copy['num_active_meds'],
        eval_cols_copy['on_sulfonylurea'], eval_cols_copy['on_metformin']
    )

    # 5-fold Out-Of-Fold Tensor Decomposition for Train
    tr_t1_risk = np.zeros(train_n, dtype=float)
    tr_t1_resid = np.zeros(train_n, dtype=float)
    tr_t1_syn_pt = np.zeros(train_n, dtype=float)
    tr_t1_syn_pr = np.zeros(train_n, dtype=float)
    tr_t1_syn_tr = np.zeros(train_n, dtype=float)
    tr_t1_contrib = np.zeros((train_n, 4), dtype=float)
    tr_t1_latent_p = np.zeros((train_n, 4), dtype=float)
    tr_t1_latent_t = np.zeros((train_n, 4), dtype=float)
    tr_t1_latent_r = np.zeros((train_n, 4), dtype=float)

    kf_t = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fit_idx, val_idx in kf_t.split(train_copy['time_in_hospital'], y_train):
        T_fit = build_smoothed_risk_tensor(
            tr_pheno[fit_idx], tr_traj[fit_idx], tr_reg[fit_idx], y_train[fit_idx], 18, 10, 8
        )
        A_f, B_f, C_f, lam_f, T_rec_f = cp_als(T_fit, rank=4, n_iter=10)

        vp = tr_pheno[val_idx]
        vt = tr_traj[val_idx]
        vr = tr_reg[val_idx]

        tr_t1_risk[val_idx] = T_rec_f[vp, vt, vr]
        tr_t1_resid[val_idx] = T_fit[vp, vt, vr] - T_rec_f[vp, vt, vr]
        tr_t1_syn_pt[val_idx] = np.sum(A_f[vp] * B_f[vt], axis=1)
        tr_t1_syn_pr[val_idx] = np.sum(A_f[vp] * C_f[vr], axis=1)
        tr_t1_syn_tr[val_idx] = np.sum(B_f[vt] * C_f[vr], axis=1)
        for r_idx in range(4):
            tr_t1_contrib[val_idx, r_idx] = lam_f[r_idx] * A_f[vp, r_idx] * B_f[vt, r_idx] * C_f[vr, r_idx]
            tr_t1_latent_p[val_idx, r_idx] = A_f[vp, r_idx]
            tr_t1_latent_t[val_idx, r_idx] = B_f[vt, r_idx]
            tr_t1_latent_r[val_idx, r_idx] = C_f[vr, r_idx]

    # Full Train Fit for Eval
    T_full = build_smoothed_risk_tensor(tr_pheno, tr_traj, tr_reg, y_train, 18, 10, 8)
    A_all, B_all, C_all, lam_all, T_rec_all = cp_als(T_full, rank=4, n_iter=10)

    ev_p = ev_pheno
    ev_t = ev_traj
    ev_r = ev_reg

    ev_t1_risk = T_rec_all[ev_p, ev_t, ev_r]
    ev_t1_resid = T_full[ev_p, ev_t, ev_r] - T_rec_all[ev_p, ev_t, ev_r]
    ev_t1_syn_pt = np.sum(A_all[ev_p] * B_all[ev_t], axis=1)
    ev_t1_syn_pr = np.sum(A_all[ev_p] * C_all[ev_r], axis=1)
    ev_t1_syn_tr = np.sum(B_all[ev_t] * C_all[ev_r], axis=1)
    ev_t1_contrib = np.zeros((eval_n, 4), dtype=float)
    ev_t1_latent_p = np.zeros((eval_n, 4), dtype=float)
    ev_t1_latent_t = np.zeros((eval_n, 4), dtype=float)
    ev_t1_latent_r = np.zeros((eval_n, 4), dtype=float)
    for r_idx in range(4):
        ev_t1_contrib[:, r_idx] = lam_all[r_idx] * A_all[ev_p, r_idx] * B_all[ev_t, r_idx] * C_all[ev_r, r_idx]
        ev_t1_latent_p[:, r_idx] = A_all[ev_p, r_idx]
        ev_t1_latent_t[:, r_idx] = B_all[ev_t, r_idx]
        ev_t1_latent_r[:, r_idx] = C_all[ev_r, r_idx]

    train_copy['tensor_clinical_risk'] = tr_t1_risk
    eval_cols_copy['tensor_clinical_risk'] = ev_t1_risk
    new_numeric_cols.append('tensor_clinical_risk')

    train_copy['tensor_clinical_resid'] = tr_t1_resid
    eval_cols_copy['tensor_clinical_resid'] = ev_t1_resid
    new_numeric_cols.append('tensor_clinical_resid')

    train_copy['tensor_syn_pheno_traj'] = tr_t1_syn_pt
    eval_cols_copy['tensor_syn_pheno_traj'] = ev_t1_syn_pt
    new_numeric_cols.append('tensor_syn_pheno_traj')

    train_copy['tensor_syn_pheno_reg'] = tr_t1_syn_pr
    eval_cols_copy['tensor_syn_pheno_reg'] = ev_t1_syn_pr
    new_numeric_cols.append('tensor_syn_pheno_reg')

    train_copy['tensor_syn_traj_reg'] = tr_t1_syn_tr
    eval_cols_copy['tensor_syn_traj_reg'] = ev_t1_syn_tr
    new_numeric_cols.append('tensor_syn_traj_reg')

    for r_idx in range(4):
        c_name = f"tensor_contrib_{r_idx}"
        train_copy[c_name] = tr_t1_contrib[:, r_idx]
        eval_cols_copy[c_name] = ev_t1_contrib[:, r_idx]
        new_numeric_cols.append(c_name)

        lp_name = f"tensor_latent_pheno_{r_idx}"
        train_copy[lp_name] = tr_t1_latent_p[:, r_idx]
        eval_cols_copy[lp_name] = ev_t1_latent_p[:, r_idx]
        new_numeric_cols.append(lp_name)

        lt_name = f"tensor_latent_traj_{r_idx}"
        train_copy[lt_name] = tr_t1_latent_t[:, r_idx]
        eval_cols_copy[lt_name] = ev_t1_latent_t[:, r_idx]
        new_numeric_cols.append(lt_name)

        lr_name = f"tensor_latent_reg_{r_idx}"
        train_copy[lr_name] = tr_t1_latent_r[:, r_idx]
        eval_cols_copy[lr_name] = ev_t1_latent_r[:, r_idx]
        new_numeric_cols.append(lr_name)

    # Tensor 2: Prior Utilization (6) x Hospital Stay Acuity (5) x Age Bracket (5)
    tr_u = get_util_tier(train_copy['number_inpatient'], train_copy['number_emergency'], train_copy['number_outpatient'])
    ev_u = get_util_tier(eval_cols_copy['number_inpatient'], eval_cols_copy['number_emergency'], eval_cols_copy['number_outpatient'])

    tr_l = get_stay_acuity(train_copy['time_in_hospital'], train_copy['admission_type_id'])
    ev_l = get_stay_acuity(eval_cols_copy['time_in_hospital'], eval_cols_copy['admission_type_id'])

    tr_g = get_age_bracket(train_copy['age_numeric'])
    ev_g = get_age_bracket(eval_cols_copy['age_numeric'])

    tr_t2_risk = np.zeros(train_n, dtype=float)
    tr_t2_syn = np.zeros(train_n, dtype=float)
    tr_t2_contrib = np.zeros((train_n, 3), dtype=float)

    for fit_idx, val_idx in kf_t.split(train_copy['time_in_hospital'], y_train):
        T2_fit = build_smoothed_risk_tensor(tr_u[fit_idx], tr_l[fit_idx], tr_g[fit_idx], y_train[fit_idx], 6, 5, 5)
        A2_f, B2_f, C2_f, lam2_f, T2_rec_f = cp_als(T2_fit, rank=3, n_iter=10)

        vu = tr_u[val_idx]
        vl = tr_l[val_idx]
        vg = tr_g[val_idx]

        tr_t2_risk[val_idx] = T2_rec_f[vu, vl, vg]
        tr_t2_syn[val_idx] = np.sum(A2_f[vu] * B2_f[vl], axis=1)
        for r_idx in range(3):
            tr_t2_contrib[val_idx, r_idx] = lam2_f[r_idx] * A2_f[vu, r_idx] * B2_f[vl, r_idx] * C2_f[vg, r_idx]

    T2_full = build_smoothed_risk_tensor(tr_u, tr_l, tr_g, y_train, 6, 5, 5)
    A2_all, B2_all, C2_all, lam2_all, T2_rec_all = cp_als(T2_full, rank=3, n_iter=10)

    ev_t2_risk = T2_rec_all[ev_u, ev_l, ev_g]
    ev_t2_syn = np.sum(A2_all[ev_u] * B2_all[ev_l], axis=1)
    ev_t2_contrib = np.zeros((eval_n, 3), dtype=float)
    for r_idx in range(3):
        ev_t2_contrib[:, r_idx] = lam2_all[r_idx] * A2_all[ev_u, r_idx] * B2_all[ev_l, r_idx] * C2_all[ev_g, r_idx]

    train_copy['tensor_util_risk'] = tr_t2_risk
    eval_cols_copy['tensor_util_risk'] = ev_t2_risk
    new_numeric_cols.append('tensor_util_risk')

    train_copy['tensor_util_synergy'] = tr_t2_syn
    eval_cols_copy['tensor_util_synergy'] = ev_t2_syn
    new_numeric_cols.append('tensor_util_synergy')

    for r_idx in range(3):
        c_name = f"tensor_util_contrib_{r_idx}"
        train_copy[c_name] = tr_t2_contrib[:, r_idx]
        eval_cols_copy[c_name] = ev_t2_contrib[:, r_idx]
        new_numeric_cols.append(c_name)

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

        # Engine 1: LightGBM Standard (Balanced leaf-wise gradient booster)
        lgb_1 = lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.018,
            num_leaves=35,
            min_child_samples=110,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=2.0,
            reg_alpha=0.3,
            cat_smooth=35.0,
            path_smooth=1.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        lgb_1.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_preds_1[val_idx] = lgb_1.predict_proba(x_val)[:, 1]
        eval_preds_1 += lgb_1.predict_proba(x_eval)[:, 1] / 5.0

        # Engine 2: LightGBM Asymmetric / Deep (Focuses on minority positive readmissions)
        lgb_2 = lgb.LGBMClassifier(
            n_estimators=270,
            learning_rate=0.018,
            num_leaves=48,
            max_depth=6,
            min_child_samples=75,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.50,
            reg_lambda=3.5,
            reg_alpha=0.5,
            scale_pos_weight=1.25,
            cat_smooth=25.0,
            path_smooth=1.2,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        )
        lgb_2.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_preds_2[val_idx] = lgb_2.predict_proba(x_val)[:, 1]
        eval_preds_2 += lgb_2.predict_proba(x_eval)[:, 1] / 5.0

        # Engine 3: XGBoost (Depth-wise histogram gradient booster)
        xgb_model = xgb.XGBClassifier(
            n_estimators=210,
            learning_rate=0.026,
            max_depth=5,
            min_child_weight=16,
            subsample=0.80,
            colsample_bytree=0.55,
            reg_lambda=2.5,
            reg_alpha=0.3,
            gamma=0.15,
            tree_method='hist',
            eval_metric='logloss',
            random_state=42,
            n_jobs=1,
        )
        xgb_model.fit(x_tr, y_tr)
        oof_preds_3[val_idx] = xgb_model.predict_proba(x_val)[:, 1]
        eval_preds_3 += xgb_model.predict_proba(x_eval)[:, 1] / 5.0

    # 15. Fine-Grained Simplex Rank & Prob Blending Solver
    n_samples = len(y_train)
    r1 = rankdata(oof_preds_1) / n_samples
    r2 = rankdata(oof_preds_2) / n_samples
    r3 = rankdata(oof_preds_3) / n_samples

    best_auc = -1.0
    best_weights = (0.40, 0.30, 0.30)
    use_rank = True

    # Coarse global simplex search
    for w1 in np.linspace(0.10, 0.70, 31):
        for w2 in np.linspace(0.10, 0.70, 31):
            w3 = 1.0 - w1 - w2
            if w3 < 0.04:
                continue

            blend_rank = w1 * r1 + w2 * r2 + w3 * r3
            score_rank = roc_auc_score(y_train, blend_rank)
            if score_rank > best_auc:
                best_auc = score_rank
                best_weights = (w1, w2, w3)
                use_rank = True

            blend_prob = w1 * oof_preds_1 + w2 * oof_preds_2 + w3 * oof_preds_3
            score_prob = roc_auc_score(y_train, blend_prob)
            if score_prob > best_auc:
                best_auc = score_prob
                best_weights = (w1, w2, w3)
                use_rank = False

    # Fine local simplex refinement
    bw1, bw2, _ = best_weights
    for dw1 in np.linspace(-0.04, 0.04, 17):
        for dw2 in np.linspace(-0.04, 0.04, 17):
            w1 = bw1 + dw1
            w2 = bw2 + dw2
            w3 = 1.0 - w1 - w2
            if w1 < 0.02 or w2 < 0.02 or w3 < 0.02:
                continue
            if use_rank:
                blend_rank = w1 * r1 + w2 * r2 + w3 * r3
                score_rank = roc_auc_score(y_train, blend_rank)
                if score_rank > best_auc:
                    best_auc = score_rank
                    best_weights = (w1, w2, w3)
            else:
                blend_prob = w1 * oof_preds_1 + w2 * oof_preds_2 + w3 * oof_preds_3
                score_prob = roc_auc_score(y_train, blend_prob)
                if score_prob > best_auc:
                    best_auc = score_prob
                    best_weights = (w1, w2, w3)

    if use_rank:
        n_eval = len(eval_preds_1)
        ev_r1 = rankdata(eval_preds_1) / n_eval
        ev_r2 = rankdata(eval_preds_2) / n_eval
        ev_r3 = rankdata(eval_preds_3) / n_eval
        final_eval_preds = (
            best_weights[0] * ev_r1 +
            best_weights[1] * ev_r2 +
            best_weights[2] * ev_r3
        )
    else:
        final_eval_preds = (
            best_weights[0] * eval_preds_1 +
            best_weights[1] * eval_preds_2 +
            best_weights[2] * eval_preds_3
        )

    return np.clip(final_eval_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
