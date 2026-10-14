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

    # Diagnostic Concordance & Chapter Complexity
    train_copy['diag_1_2_concordant'] = (train_copy['diag_1_group'] == train_copy['diag_2_group']).astype(float)
    eval_cols_copy['diag_1_2_concordant'] = (eval_cols_copy['diag_1_group'] == eval_cols_copy['diag_2_group']).astype(float)
    new_numeric_cols.append('diag_1_2_concordant')

    train_copy['diag_1_3_concordant'] = (train_copy['diag_1_group'] == train_copy['diag_3_group']).astype(float)
    eval_cols_copy['diag_1_3_concordant'] = (eval_cols_copy['diag_1_group'] == eval_cols_copy['diag_3_group']).astype(float)
    new_numeric_cols.append('diag_1_3_concordant')

    def count_unique_chapters(g1, g2, g3):
        return np.array([len(set([a, b, c]) - {'missing'}) for a, b, c in zip(g1, g2, g3)], dtype=float)

    train_copy['num_unique_diag_chapters'] = count_unique_chapters(train_copy['diag_1_group'], train_copy['diag_2_group'], train_copy['diag_3_group'])
    eval_cols_copy['num_unique_diag_chapters'] = count_unique_chapters(eval_cols_copy['diag_1_group'], eval_cols_copy['diag_2_group'], eval_cols_copy['diag_3_group'])
    new_numeric_cols.append('num_unique_diag_chapters')

    # 3. Additional Clinical Conditions & Multi-Organ Vulnerability
    clinical_flags['flag_anemia'] = ['280', '281', '282', '283', '284', '285']
    clinical_flags['flag_fluid_overload'] = ['276', '782']
    clinical_flags['flag_pneumonia'] = ['480', '481', '482', '483', '485', '486']
    clinical_flags['flag_mental_health'] = ['290', '294', '296', '300', '311', '331']

    for flag_name in ['flag_anemia', 'flag_fluid_overload', 'flag_pneumonia', 'flag_mental_health']:
        train_copy[flag_name] = make_diag_flag(
            train_copy['diag_1_3d'], train_copy['diag_2_3d'], train_copy['diag_3_3d'], clinical_flags[flag_name]
        )
        eval_cols_copy[flag_name] = make_diag_flag(
            eval_cols_copy['diag_1_3d'], eval_cols_copy['diag_2_3d'], eval_cols_copy['diag_3_3d'], clinical_flags[flag_name]
        )
        new_numeric_cols.append(flag_name)

    cardiac_tr = np.maximum.reduce([train_copy['flag_chf'], train_copy['flag_cad_mi'], train_copy['flag_dysrhythmia']])
    renal_tr = np.maximum(train_copy['flag_renal'], train_cm['cm_renal'])
    infect_tr = np.maximum(train_copy['flag_sepsis'], train_copy['flag_uti'])
    vasc_tr = np.maximum(train_cm['cm_pvd'], train_copy['flag_cerebrovasc'])
    train_copy['organ_systems_count'] = (
        cardiac_tr + renal_tr + train_copy['flag_copd_pneu'] + train_copy['flag_anemia'] + infect_tr + vasc_tr
    ).astype(float)
    train_copy['multi_organ_risk'] = (train_copy['organ_systems_count'] >= 2.0).astype(float)

    cardiac_ev = np.maximum.reduce([eval_cols_copy['flag_chf'], eval_cols_copy['flag_cad_mi'], eval_cols_copy['flag_dysrhythmia']])
    renal_ev = np.maximum(eval_cols_copy['flag_renal'], eval_cm['cm_renal'])
    infect_ev = np.maximum(eval_cols_copy['flag_sepsis'], eval_cols_copy['flag_uti'])
    vasc_ev = np.maximum(eval_cm['cm_pvd'], eval_cols_copy['flag_cerebrovasc'])
    eval_cols_copy['organ_systems_count'] = (
        cardiac_ev + renal_ev + eval_cols_copy['flag_copd_pneu'] + eval_cols_copy['flag_anemia'] + infect_ev + vasc_ev
    ).astype(float)
    eval_cols_copy['multi_organ_risk'] = (eval_cols_copy['organ_systems_count'] >= 2.0).astype(float)

    new_numeric_cols.append('organ_systems_count')
    new_numeric_cols.append('multi_organ_risk')

    # 4. Utilization Intensity & Longitudinal Risk Dynamics
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

    train_copy['prior_inpatient_ge3'] = (train_copy['number_inpatient'] >= 3).astype(float)
    eval_cols_copy['prior_inpatient_ge3'] = (eval_cols_copy['number_inpatient'] >= 3).astype(float)
    new_numeric_cols.append('prior_inpatient_ge3')

    train_copy['has_prior_emergency'] = (train_copy['number_emergency'] > 0).astype(float)
    eval_cols_copy['has_prior_emergency'] = (eval_cols_copy['number_emergency'] > 0).astype(float)
    new_numeric_cols.append('has_prior_emergency')

    train_copy['prior_emergency_ge2'] = (train_copy['number_emergency'] >= 2).astype(float)
    eval_cols_copy['prior_emergency_ge2'] = (eval_cols_copy['number_emergency'] >= 2).astype(float)
    new_numeric_cols.append('prior_emergency_ge2')

    train_copy['acute_prior_visits'] = (train_copy['number_inpatient'] + train_copy['number_emergency']).astype(float)
    eval_cols_copy['acute_prior_visits'] = (eval_cols_copy['number_inpatient'] + eval_cols_copy['number_emergency']).astype(float)
    new_numeric_cols.append('acute_prior_visits')

    train_copy['acute_visit_ratio'] = (train_copy['acute_prior_visits'] / (train_copy['total_prior_visits'] + 1.0)).astype(float)
    eval_cols_copy['acute_visit_ratio'] = (eval_cols_copy['acute_prior_visits'] / (eval_cols_copy['total_prior_visits'] + 1.0)).astype(float)
    new_numeric_cols.append('acute_visit_ratio')

    # Log1p transforms on skewed clinical and utilization metrics
    for col in ['number_inpatient', 'number_emergency', 'number_outpatient', 'total_prior_visits', 'time_in_hospital', 'num_medications', 'num_lab_procedures']:
        log_name = f"{col}_log1p"
        train_copy[log_name] = np.log1p(np.maximum(0.0, train_copy[col].astype(float)))
        eval_cols_copy[log_name] = np.log1p(np.maximum(0.0, eval_cols_copy[col].astype(float)))
        new_numeric_cols.append(log_name)

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

    # 5. Out-of-Fold Target Encodings
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

    # 6. Clinical Interactions & Target Encodings
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
        ('disch_x_has_inpat', 'discharge_disposition_id', 'has_prior_inpatient'),
        ('diag1_grp_x_has_inpat', 'diag_1_group', 'has_prior_inpatient'),
        ('admsrc_x_has_inpat', 'admission_source_id', 'has_prior_inpatient'),
        ('age_x_has_inpat', 'age', 'has_prior_inpatient'),
        ('diag1_grp_x_diag2_grp', 'diag_1_group', 'diag_2_group'),
    ]
    for inter_name, c1, c2 in interactions:
        inter_tr = make_interaction(train_copy[c1], train_copy[c2])
        inter_ev = make_interaction(eval_cols_copy[c1], eval_cols_copy[c2])
        tr_enc, ev_enc = oof_target_encode(
            inter_tr, y_train, inter_ev, smoothing=25.0, seed=42
        )
        te_name = f"{inter_name}_te"
        train_copy[te_name] = tr_enc.astype(float)
        eval_cols_copy[te_name] = ev_enc.astype(float)
        new_numeric_cols.append(te_name)

    # 7. Frequency Encodings for Categorical Cardinality/Prevalence
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

    train_copy['insulin_monotherapy'] = ((train_copy['on_insulin'] == 1.0) & (train_copy['num_active_meds'] == 1.0)).astype(float)
    eval_cols_copy['insulin_monotherapy'] = ((eval_cols_copy['on_insulin'] == 1.0) & (eval_cols_copy['num_active_meds'] == 1.0)).astype(float)
    new_numeric_cols.append('insulin_monotherapy')

    train_copy['glycemic_instability'] = (train_copy['num_changes'] > 0).astype(float)
    eval_cols_copy['glycemic_instability'] = (eval_cols_copy['num_changes'] > 0).astype(float)
    new_numeric_cols.append('glycemic_instability')

    train_copy['polypharmacy'] = (train_copy['num_medications'] >= 15).astype(float)
    eval_cols_copy['polypharmacy'] = (eval_cols_copy['num_medications'] >= 15).astype(float)
    new_numeric_cols.append('polypharmacy')

    train_copy['su_and_renal'] = (train_copy['on_sulfonylurea'] * train_cm['cm_renal']).astype(float)
    eval_cols_copy['su_and_renal'] = (eval_cols_copy['on_sulfonylurea'] * eval_cm['cm_renal']).astype(float)
    new_numeric_cols.append('su_and_renal')

    train_copy['insulin_and_renal'] = (train_copy['on_insulin'] * train_cm['cm_renal']).astype(float)
    eval_cols_copy['insulin_and_renal'] = (eval_cols_copy['on_insulin'] * eval_cm['cm_renal']).astype(float)
    new_numeric_cols.append('insulin_and_renal')

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

    # Transitions of Care & Discharge Risk Interactions
    train_copy['high_risk_discharge'] = np.isin(train_copy['discharge_disposition_id'], ['2', '3', '4', '5', '6', '7', '22']).astype(float)
    eval_cols_copy['high_risk_discharge'] = np.isin(eval_cols_copy['discharge_disposition_id'], ['2', '3', '4', '5', '6', '7', '22']).astype(float)
    new_numeric_cols.append('high_risk_discharge')

    train_copy['disch_snf_x_prior_inpatient'] = (train_copy['disch_snf'] * train_copy['has_prior_inpatient']).astype(float)
    eval_cols_copy['disch_snf_x_prior_inpatient'] = (eval_cols_copy['disch_snf'] * eval_cols_copy['has_prior_inpatient']).astype(float)
    new_numeric_cols.append('disch_snf_x_prior_inpatient')

    train_copy['disch_homehealth_x_prior_inpatient'] = (train_copy['disch_home_health'] * train_copy['has_prior_inpatient']).astype(float)
    eval_cols_copy['disch_homehealth_x_prior_inpatient'] = (eval_cols_copy['disch_home_health'] * eval_cols_copy['has_prior_inpatient']).astype(float)
    new_numeric_cols.append('disch_homehealth_x_prior_inpatient')

    train_copy['disch_not_home_x_prior_inpatient'] = (train_copy['disch_not_home'] * train_copy['has_prior_inpatient']).astype(float)
    eval_cols_copy['disch_not_home_x_prior_inpatient'] = (eval_cols_copy['disch_not_home'] * eval_cols_copy['has_prior_inpatient']).astype(float)
    new_numeric_cols.append('disch_not_home_x_prior_inpatient')

    train_copy['emergency_adm_x_prior_inpatient'] = (train_copy['admsrc_er'] * train_copy['has_prior_inpatient']).astype(float)
    eval_cols_copy['emergency_adm_x_prior_inpatient'] = (eval_cols_copy['admsrc_er'] * eval_cols_copy['has_prior_inpatient']).astype(float)
    new_numeric_cols.append('emergency_adm_x_prior_inpatient')

    train_copy['payer_medicare'] = (train_copy['payer_code'] == 'MC').astype(float)
    eval_cols_copy['payer_medicare'] = (eval_cols_copy['payer_code'] == 'MC').astype(float)
    new_numeric_cols.append('payer_medicare')

    train_copy['payer_medicaid'] = (train_copy['payer_code'] == 'MD').astype(float)
    eval_cols_copy['payer_medicaid'] = (eval_cols_copy['payer_code'] == 'MD').astype(float)
    new_numeric_cols.append('payer_medicaid')

    train_copy['medicaid_x_prior_inpatient'] = (train_copy['payer_medicaid'] * train_copy['has_prior_inpatient']).astype(float)
    eval_cols_copy['medicaid_x_prior_inpatient'] = (eval_cols_copy['payer_medicaid'] * eval_cols_copy['has_prior_inpatient']).astype(float)
    new_numeric_cols.append('medicaid_x_prior_inpatient')

    # Clinical Inertia: High A1C without Therapy Intensification
    train_copy['clinical_inertia'] = ((train_copy['a1c_high'] == 1.0) & (train_copy['regimen_changed'] == 0.0)).astype(float)
    eval_cols_copy['clinical_inertia'] = ((eval_cols_copy['a1c_high'] == 1.0) & (eval_cols_copy['regimen_changed'] == 0.0)).astype(float)
    new_numeric_cols.append('clinical_inertia')

    train_copy['intensified_high_a1c'] = ((train_copy['a1c_high'] == 1.0) & (train_copy['regimen_changed'] == 1.0)).astype(float)
    eval_cols_copy['intensified_high_a1c'] = ((eval_cols_copy['a1c_high'] == 1.0) & (eval_cols_copy['regimen_changed'] == 1.0)).astype(float)
    new_numeric_cols.append('intensified_high_a1c')

    # 12. Encode features using TabularEncoder
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        train_copy,
        extra_numeric=tuple(new_numeric_cols),
        extra_categorical=tuple(new_categorical_cols)
    )
    x_train = encoder.transform(train_copy)
    x_eval = encoder.transform(eval_cols_copy)

    # 13. Prepare Distinct Feature Views for Model Specialization
    cat_set = set(encoder.categorical_indices)
    numeric_indices = np.array([i for i in range(x_train.shape[1]) if i not in cat_set], dtype=int)

    # XGBoost receives only continuous features, ratios, and target/frequency encodings
    x_train_xgb = x_train[:, numeric_indices]
    x_eval_xgb = x_eval[:, numeric_indices]

    # 14. 5-Fold Stratified Tri-Engine Heterogeneous Ensemble
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds_lgb1 = np.zeros(len(x_train), dtype=float)
    oof_preds_lgb2 = np.zeros(len(x_train), dtype=float)
    oof_preds_xgb = np.zeros(len(x_train), dtype=float)

    eval_preds_lgb1 = np.zeros(len(x_eval), dtype=float)
    eval_preds_lgb2 = np.zeros(len(x_eval), dtype=float)
    eval_preds_xgb = np.zeros(len(x_eval), dtype=float)

    for train_idx, val_idx in skf.split(x_train, y_train):
        x_tr, y_tr = x_train[train_idx], y_train[train_idx]
        x_val = x_train[val_idx]

        x_tr_xgb = x_train_xgb[train_idx]
        x_val_xgb = x_train_xgb[val_idx]

        # Engine 1: LightGBM Deep (Leaf-wise booster exploring high-order categorical partitions)
        lgb1 = lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.022,
            num_leaves=40,
            max_depth=7,
            min_child_samples=90,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=3.0,
            reg_alpha=0.5,
            cat_smooth=35.0,
            path_smooth=1.2,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        lgb1.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_preds_lgb1[val_idx] = lgb1.predict_proba(x_val)[:, 1]
        eval_preds_lgb1 += lgb1.predict_proba(x_eval)[:, 1] / 5.0

        # Engine 2: LightGBM Shallow (Conservative booster regularized for low-order main effects)
        lgb2 = lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.028,
            num_leaves=18,
            max_depth=4,
            min_child_samples=130,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=4.0,
            reg_alpha=1.0,
            cat_smooth=45.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        )
        lgb2.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_preds_lgb2[val_idx] = lgb2.predict_proba(x_val)[:, 1]
        eval_preds_lgb2 += lgb2.predict_proba(x_eval)[:, 1] / 5.0

        # Engine 3: XGBoost Hist (Depth-wise booster trained strictly on numeric/target-encoded features)
        xgb_model = xgb.XGBClassifier(
            n_estimators=260,
            learning_rate=0.025,
            max_depth=5,
            min_child_weight=14,
            subsample=0.8,
            colsample_bytree=0.65,
            reg_lambda=3.0,
            reg_alpha=0.4,
            gamma=0.15,
            tree_method='hist',
            eval_metric='logloss',
            random_state=42,
            n_jobs=1,
        )
        xgb_model.fit(x_tr_xgb, y_tr)
        oof_preds_xgb[val_idx] = xgb_model.predict_proba(x_val_xgb)[:, 1]
        eval_preds_xgb += xgb_model.predict_proba(x_eval_xgb)[:, 1] / 5.0

    # 15. Tri-Engine Simplex Calibration & Optimal Weight Solver
    n_samples = len(y_train)
    r_lgb1 = rankdata(oof_preds_lgb1) / n_samples
    r_lgb2 = rankdata(oof_preds_lgb2) / n_samples
    r_xgb = rankdata(oof_preds_xgb) / n_samples

    best_auc = -1.0
    best_weights = (0.45, 0.25, 0.30)
    use_rank = True

    w_grid = []
    for w1 in np.linspace(0.2, 0.6, 9):
        for w2 in np.linspace(0.1, 0.5, 9):
            w3 = 1.0 - w1 - w2
            if w3 >= 0.1:
                w_grid.append((w1, w2, w3))

    for w1, w2, w3 in w_grid:
        # Rank blend OOF score
        score_rank = roc_auc_score(y_train, w1 * r_lgb1 + w2 * r_lgb2 + w3 * r_xgb)
        if score_rank > best_auc:
            best_auc = score_rank
            best_weights = (w1, w2, w3)
            use_rank = True

        # Probability blend OOF score
        score_prob = roc_auc_score(y_train, w1 * oof_preds_lgb1 + w2 * oof_preds_lgb2 + w3 * oof_preds_xgb)
        if score_prob > best_auc:
            best_auc = score_prob
            best_weights = (w1, w2, w3)
            use_rank = False

    w1, w2, w3 = best_weights
    if use_rank:
        n_eval = len(eval_preds_lgb1)
        ev_r_lgb1 = rankdata(eval_preds_lgb1) / n_eval
        ev_r_lgb2 = rankdata(eval_preds_lgb2) / n_eval
        ev_r_xgb = rankdata(eval_preds_xgb) / n_eval
        final_eval_preds = w1 * ev_r_lgb1 + w2 * ev_r_lgb2 + w3 * ev_r_xgb
    else:
        final_eval_preds = w1 * eval_preds_lgb1 + w2 * eval_preds_lgb2 + w3 * eval_preds_xgb

    return np.clip(final_eval_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
