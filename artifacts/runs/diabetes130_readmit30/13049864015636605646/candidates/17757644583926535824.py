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
    import lightgbm as lgb
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

    # 2. ICD-9 Chapter Groups
    for col in ['diag_1', 'diag_2', 'diag_3']:
        group_col_name = f"{col}_group"
        train_copy[group_col_name] = np.array([icd9_group(x) for x in train_copy[col]], dtype=object)
        eval_cols_copy[group_col_name] = np.array([icd9_group(x) for x in eval_cols_copy[col]], dtype=object)
        new_categorical_cols.append(group_col_name)

    # 3. High-cardinality and important categorical Out-of-Fold Target Encodings
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
        train_copy[te_col_name] = train_enc
        eval_cols_copy[te_col_name] = eval_enc
        new_numeric_cols.append(te_col_name)

    # 4. Utilization Intensity & Hospitalization Metrics
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

    train_copy['med_by_hospital_days'] = train_copy['num_medications'] / (train_copy['time_in_hospital'] + 1.0)
    eval_cols_copy['med_by_hospital_days'] = eval_cols_copy['num_medications'] / (eval_cols_copy['time_in_hospital'] + 1.0)
    new_numeric_cols.append('med_by_hospital_days')

    train_copy['lab_by_hospital_days'] = train_copy['num_lab_procedures'] / (train_copy['time_in_hospital'] + 1.0)
    eval_cols_copy['lab_by_hospital_days'] = eval_cols_copy['num_lab_procedures'] / (eval_cols_copy['time_in_hospital'] + 1.0)
    new_numeric_cols.append('lab_by_hospital_days')

    train_copy['procedures_by_hospital_days'] = train_copy['num_procedures'] / (train_copy['time_in_hospital'] + 1.0)
    eval_cols_copy['procedures_by_hospital_days'] = eval_cols_copy['num_procedures'] / (eval_cols_copy['time_in_hospital'] + 1.0)
    new_numeric_cols.append('procedures_by_hospital_days')

    # 5. Medication and Dosage Change Counts
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

    # 6. Continuous representation of age groups
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0, '[40-50)': 45.0,
        '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0, '[80-90)': 85.0, '[90-100)': 95.0
    }
    def map_age(age_arr):
        return np.array([age_map.get(x, np.nan) for x in age_arr], dtype=float)

    train_copy['age_numeric'] = map_age(train_copy['age'])
    eval_cols_copy['age_numeric'] = map_age(eval_cols_copy['age'])
    new_numeric_cols.append('age_numeric')

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

    # 1c. Multi-Scale ICD-9 3-Digit Roots & HRRP Target Condition Indicators
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
    }
    for flag_name, codes in clinical_flags.items():
        train_copy[flag_name] = make_diag_flag(
            train_copy['diag_1_3d'], train_copy['diag_2_3d'], train_copy['diag_3_3d'], codes
        )
        eval_cols_copy[flag_name] = make_diag_flag(
            eval_cols_copy['diag_1_3d'], eval_cols_copy['diag_2_3d'], eval_cols_copy['diag_3_3d'], codes
        )
        new_numeric_cols.append(flag_name)

    # 1d. Clinical Interaction Categories & Target Encodings
    def make_interaction(arr1, arr2):
        return np.array([f"{a}_{b}" for a, b in zip(arr1, arr2)], dtype=object)

    interactions = [
        ('disch_x_admsrc', 'discharge_disposition_id', 'admission_source_id'),
        ('diag1_x_disch', 'diag_1_group', 'discharge_disposition_id'),
        ('admtype_x_disch', 'admission_type_id', 'discharge_disposition_id'),
    ]
    for inter_name, c1, c2 in interactions:
        train_copy[inter_name] = make_interaction(train_copy[c1], train_copy[c2])
        eval_cols_copy[inter_name] = make_interaction(eval_cols_copy[c1], eval_cols_copy[c2])
        new_categorical_cols.append(inter_name)
        # Out-of-fold target encode the interaction
        tr_enc, ev_enc = oof_target_encode(
            train_copy[inter_name], y_train, eval_cols_copy[inter_name], smoothing=25.0, seed=42
        )
        te_name = f"{inter_name}_te"
        train_copy[te_name] = tr_enc.astype(float)
        eval_cols_copy[te_name] = ev_enc.astype(float)
        new_numeric_cols.append(te_name)

    # Target encode 3-digit ICD-9 roots as well
    for col in ['diag_1_3d', 'diag_2_3d', 'diag_3_3d']:
        te_col_name = f"{col}_te"
        tr_enc, ev_enc = oof_target_encode(
            train_copy[col], y_train, eval_cols_copy[col], smoothing=15.0, seed=42
        )
        train_copy[te_col_name] = tr_enc.astype(float)
        eval_cols_copy[te_col_name] = ev_enc.astype(float)
        new_numeric_cols.append(te_col_name)

    # 1e. Latent Clinical Phenotyping via TruncatedSVD Encounter Embeddings
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import TruncatedSVD

    def make_clinical_docs(data_dict, n_rows):
        d1 = data_dict['diag_1_3d']
        d2 = data_dict['diag_2_3d']
        d3 = data_dict['diag_3_3d']
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

    train_n = len(y_train)
    eval_n = len(eval_cols_copy['time_in_hospital'])
    train_docs = make_clinical_docs(train_copy, train_n)
    eval_docs = make_clinical_docs(eval_cols_copy, eval_n)

    cv = CountVectorizer(min_df=10, token_pattern=r'\S+')
    train_doc_mat = cv.fit_transform(train_docs)
    eval_doc_mat = cv.transform(eval_docs)

    n_svd = 8
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    train_svd = svd.fit_transform(train_doc_mat)
    eval_svd = svd.transform(eval_doc_mat)

    for j in range(n_svd):
        svd_name = f"clinical_svd_{j}"
        train_copy[svd_name] = train_svd[:, j].astype(float)
        eval_cols_copy[svd_name] = eval_svd[:, j].astype(float)
        new_numeric_cols.append(svd_name)

    # 1f. Additional High-Impact Utilization & Clinical Risk Ratios
    train_copy['is_frequent_inpatient'] = (train_copy['number_inpatient'] >= 2).astype(float)
    eval_cols_copy['is_frequent_inpatient'] = (eval_cols_copy['number_inpatient'] >= 2).astype(float)
    new_numeric_cols.append('is_frequent_inpatient')

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

    train_copy['med_per_diagnosis'] = (
        train_copy['num_medications'] / (train_copy['number_diagnoses'] + 1.0)
    ).astype(float)
    eval_cols_copy['med_per_diagnosis'] = (
        eval_cols_copy['num_medications'] / (eval_cols_copy['number_diagnoses'] + 1.0)
    ).astype(float)
    new_numeric_cols.append('med_per_diagnosis')

    train_copy['lab_per_diagnosis'] = (
        train_copy['num_lab_procedures'] / (train_copy['number_diagnoses'] + 1.0)
    ).astype(float)
    eval_cols_copy['lab_per_diagnosis'] = (
        eval_cols_copy['num_lab_procedures'] / (eval_cols_copy['number_diagnoses'] + 1.0)
    ).astype(float)
    new_numeric_cols.append('lab_per_diagnosis')

    # LACE Index approximation
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

    # Key discharge disposition flags
    train_copy['disch_home'] = (train_copy['discharge_disposition_id'] == '1').astype(float)
    eval_cols_copy['disch_home'] = (eval_cols_copy['discharge_disposition_id'] == '1').astype(float)
    new_numeric_cols.append('disch_home')

    train_copy['disch_snf'] = (train_copy['discharge_disposition_id'] == '3').astype(float)
    eval_cols_copy['disch_snf'] = (eval_cols_copy['discharge_disposition_id'] == '3').astype(float)
    new_numeric_cols.append('disch_snf')

    train_copy['disch_ama'] = (train_copy['discharge_disposition_id'] == '7').astype(float)
    eval_cols_copy['disch_ama'] = (eval_cols_copy['discharge_disposition_id'] == '7').astype(float)
    new_numeric_cols.append('disch_ama')

    # Insulin & oral regimen combinations
    train_copy['on_insulin'] = ((train_copy['insulin'] != 'No') & (train_copy['insulin'] != None)).astype(float)
    eval_cols_copy['on_insulin'] = ((eval_cols_copy['insulin'] != 'No') & (eval_cols_copy['insulin'] != None)).astype(float)
    new_numeric_cols.append('on_insulin')

    train_copy['insulin_up'] = (train_copy['insulin'] == 'Up').astype(float)
    eval_cols_copy['insulin_up'] = (eval_cols_copy['insulin'] == 'Up').astype(float)
    new_numeric_cols.append('insulin_up')

    train_copy['insulin_down'] = (train_copy['insulin'] == 'Down').astype(float)
    eval_cols_copy['insulin_down'] = (eval_cols_copy['insulin'] == 'Down').astype(float)
    new_numeric_cols.append('insulin_down')

    train_copy['on_metformin'] = ((train_copy['metformin'] != 'No') & (train_copy['metformin'] != None)).astype(float)
    eval_cols_copy['on_metformin'] = ((eval_cols_copy['metformin'] != 'No') & (eval_cols_copy['metformin'] != None)).astype(float)
    new_numeric_cols.append('on_metformin')

    train_copy['insulin_and_oral'] = (train_copy['on_insulin'] * (train_copy['num_active_meds'] > 1)).astype(float)
    eval_cols_copy['insulin_and_oral'] = (eval_cols_copy['on_insulin'] * (eval_cols_copy['num_active_meds'] > 1)).astype(float)
    new_numeric_cols.append('insulin_and_oral')

    # Lab and specialty flags
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

    train_copy['is_elderly'] = (train_copy['age_numeric'] >= 75.0).astype(float)
    eval_cols_copy['is_elderly'] = (eval_cols_copy['age_numeric'] >= 75.0).astype(float)
    new_numeric_cols.append('is_elderly')

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

    # 1g. Clinical Counterfactual Ghost Encounters (Residual Anomaly Profiling)
    # Fast expectation engine conditioned on baseline status to isolate clinical surprises
    from sklearn.linear_model import Ridge
    baseline_feats = [
        'age_numeric', 'number_diagnoses', 'number_inpatient', 'number_emergency', 'number_outpatient',
        'weighted_charlson', 'cm_mi', 'cm_chf', 'cm_pvd', 'cm_cerebrovascular', 'cm_dementia',
        'cm_pulmonary', 'cm_rheumatic', 'cm_peptic_ulcer', 'cm_liver', 'cm_hemiplegia',
        'cm_renal', 'cm_cancer', 'cm_metastatic', 'cm_hiv', 'cm_diabetes_complicated'
    ]
    X_base_tr = np.column_stack([np.nan_to_num(train_copy[f].astype(float), nan=0.0) for f in baseline_feats])
    X_base_ev = np.column_stack([np.nan_to_num(eval_cols_copy[f].astype(float), nan=0.0) for f in baseline_feats])

    # Expected Length of Stay -> Premature discharge & protracted stay flags
    ridge_los = Ridge(alpha=100.0).fit(X_base_tr, train_copy['time_in_hospital'])
    pred_los_tr = ridge_los.predict(X_base_tr)
    pred_los_ev = ridge_los.predict(X_base_ev)

    train_copy['delta_los'] = (train_copy['time_in_hospital'] - pred_los_tr).astype(float)
    eval_cols_copy['delta_los'] = (eval_cols_copy['time_in_hospital'] - pred_los_ev).astype(float)
    new_numeric_cols.append('delta_los')

    train_copy['ratio_los'] = (train_copy['time_in_hospital'] / np.clip(pred_los_tr, 0.5, None)).astype(float)
    eval_cols_copy['ratio_los'] = (eval_cols_copy['time_in_hospital'] / np.clip(pred_los_ev, 0.5, None)).astype(float)
    new_numeric_cols.append('ratio_los')

    train_copy['premature_discharge'] = (train_copy['delta_los'] < -1.5).astype(float)
    eval_cols_copy['premature_discharge'] = (eval_cols_copy['delta_los'] < -1.5).astype(float)
    new_numeric_cols.append('premature_discharge')

    # Expected Medications -> Polypharmacy surge
    ridge_meds = Ridge(alpha=100.0).fit(X_base_tr, train_copy['num_medications'])
    pred_meds_tr = ridge_meds.predict(X_base_tr)
    pred_meds_ev = ridge_meds.predict(X_base_ev)

    train_copy['delta_meds'] = (train_copy['num_medications'] - pred_meds_tr).astype(float)
    eval_cols_copy['delta_meds'] = (eval_cols_copy['num_medications'] - pred_meds_ev).astype(float)
    new_numeric_cols.append('delta_meds')

    train_copy['ratio_meds'] = (train_copy['num_medications'] / np.clip(pred_meds_tr, 1.0, None)).astype(float)
    eval_cols_copy['ratio_meds'] = (eval_cols_copy['num_medications'] / np.clip(pred_meds_ev, 1.0, None)).astype(float)
    new_numeric_cols.append('ratio_meds')

    # Expected Lab Procedures -> Diagnostic instability index
    ridge_labs = Ridge(alpha=100.0).fit(X_base_tr, train_copy['num_lab_procedures'])
    pred_labs_tr = ridge_labs.predict(X_base_tr)
    pred_labs_ev = ridge_labs.predict(X_base_ev)

    train_copy['delta_labs'] = (train_copy['num_lab_procedures'] - pred_labs_tr).astype(float)
    eval_cols_copy['delta_labs'] = (eval_cols_copy['num_lab_procedures'] - pred_labs_ev).astype(float)
    new_numeric_cols.append('delta_labs')

    train_copy['ratio_labs'] = (train_copy['num_lab_procedures'] / np.clip(pred_labs_tr, 1.0, None)).astype(float)
    eval_cols_copy['ratio_labs'] = (eval_cols_copy['num_lab_procedures'] / np.clip(pred_labs_ev, 1.0, None)).astype(float)
    new_numeric_cols.append('ratio_labs')

    # 1h. Multimorbidity Synergy Network (Unordered Diagnosis Dyads)
    def make_unordered_pair(a1, a2):
        out = []
        for x, y in zip(a1, a2):
            sx, sy = str(x), str(y)
            out.append(f"{sx}_{sy}" if sx < sy else f"{sy}_{sx}")
        return np.array(out, dtype=object)

    diag_pair_12_tr = make_unordered_pair(train_copy['diag_1_3d'], train_copy['diag_2_3d'])
    diag_pair_12_ev = make_unordered_pair(eval_cols_copy['diag_1_3d'], eval_cols_copy['diag_2_3d'])
    train_copy['diag_pair_12'] = diag_pair_12_tr
    eval_cols_copy['diag_pair_12'] = diag_pair_12_ev
    new_categorical_cols.append('diag_pair_12')

    tr_enc_12, ev_enc_12 = oof_target_encode(diag_pair_12_tr, y_train, diag_pair_12_ev, smoothing=20.0, seed=42)
    train_copy['diag_pair_12_te'] = tr_enc_12.astype(float)
    eval_cols_copy['diag_pair_12_te'] = ev_enc_12.astype(float)
    new_numeric_cols.append('diag_pair_12_te')

    diag_pair_13_tr = make_unordered_pair(train_copy['diag_1_3d'], train_copy['diag_3_3d'])
    diag_pair_13_ev = make_unordered_pair(eval_cols_copy['diag_1_3d'], eval_cols_copy['diag_3_3d'])
    train_copy['diag_pair_13'] = diag_pair_13_tr
    eval_cols_copy['diag_pair_13'] = diag_pair_13_ev
    new_categorical_cols.append('diag_pair_13')

    tr_enc_13, ev_enc_13 = oof_target_encode(diag_pair_13_tr, y_train, diag_pair_13_ev, smoothing=20.0, seed=42)
    train_copy['diag_pair_13_te'] = tr_enc_13.astype(float)
    eval_cols_copy['diag_pair_13_te'] = ev_enc_13.astype(float)
    new_numeric_cols.append('diag_pair_13_te')

    # 1i. Nonparametric Clinical Twin Phenotype Density (k-NN Manifold Search)
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    twin_feat_names = [
        'clinical_svd_0', 'clinical_svd_1', 'clinical_svd_2', 'clinical_svd_3', 'clinical_svd_4',
        'age_numeric', 'time_in_hospital', 'number_inpatient', 'total_prior_visits',
        'weighted_charlson', 'lace_score'
    ]
    X_twin_tr = np.column_stack([np.nan_to_num(train_copy[f].astype(float), nan=0.0) for f in twin_feat_names])
    X_twin_ev = np.column_stack([np.nan_to_num(eval_cols_copy[f].astype(float), nan=0.0) for f in twin_feat_names])

    twin_scaler = StandardScaler()
    X_twin_tr_s = twin_scaler.fit_transform(X_twin_tr)
    X_twin_ev_s = twin_scaler.transform(X_twin_ev)

    nn = NearestNeighbors(n_neighbors=31, algorithm='kd_tree', n_jobs=1)
    nn.fit(X_twin_tr_s)

    # Train OOF: exclude neighbor 0 (the sample itself) so label is strictly out-of-fold
    dist_tr, idx_tr = nn.kneighbors(X_twin_tr_s)
    train_copy['twin_risk_30'] = np.mean(y_train[idx_tr[:, 1:31]], axis=1).astype(float)
    train_copy['twin_risk_10'] = np.mean(y_train[idx_tr[:, 1:11]], axis=1).astype(float)
    train_copy['twin_isolation'] = np.mean(dist_tr[:, 1:31], axis=1).astype(float)

    # Eval: query 30 nearest clinical neighbors from train cohort
    dist_ev, idx_ev = nn.kneighbors(X_twin_ev_s, n_neighbors=30)
    eval_cols_copy['twin_risk_30'] = np.mean(y_train[idx_ev[:, :30]], axis=1).astype(float)
    eval_cols_copy['twin_risk_10'] = np.mean(y_train[idx_ev[:, :10]], axis=1).astype(float)
    eval_cols_copy['twin_isolation'] = np.mean(dist_ev[:, :30], axis=1).astype(float)

    new_numeric_cols.extend(['twin_risk_30', 'twin_risk_10', 'twin_isolation'])

    # 1j. Pharmacological Vulnerability & Velocity Interactions
    sulfonylurea_meds = [
        'glimepiride', 'glipizide', 'glyburide', 'chlorpropamide',
        'tolbutamide', 'acetohexamide', 'tolazamide'
    ]
    on_su_tr = np.zeros(train_n, dtype=float)
    on_su_ev = np.zeros(eval_n, dtype=float)
    for m in sulfonylurea_meds:
        if m in train_copy:
            on_su_tr += ((train_copy[m] != 'No') & (train_copy[m] != None)).astype(float)
            on_su_ev += ((eval_cols_copy[m] != 'No') & (eval_cols_copy[m] != None)).astype(float)

    train_copy['on_sulfonylurea'] = (on_su_tr > 0).astype(float)
    eval_cols_copy['on_sulfonylurea'] = (on_su_ev > 0).astype(float)
    new_numeric_cols.append('on_sulfonylurea')

    train_copy['su_and_elderly'] = (train_copy['on_sulfonylurea'] * train_copy['is_elderly']).astype(float)
    eval_cols_copy['su_and_elderly'] = (eval_cols_copy['on_sulfonylurea'] * eval_cols_copy['is_elderly']).astype(float)
    new_numeric_cols.append('su_and_elderly')

    train_copy['su_and_renal'] = (train_copy['on_sulfonylurea'] * train_copy['flag_renal']).astype(float)
    eval_cols_copy['su_and_renal'] = (eval_cols_copy['on_sulfonylurea'] * eval_cols_copy['flag_renal']).astype(float)
    new_numeric_cols.append('su_and_renal')

    train_copy['utilization_severity'] = (
        3.0 * train_copy['number_inpatient'] + 1.5 * train_copy['number_emergency'] + 0.5 * train_copy['number_outpatient']
    ).astype(float)
    eval_cols_copy['utilization_severity'] = (
        3.0 * eval_cols_copy['number_inpatient'] + 1.5 * eval_cols_copy['number_emergency'] + 0.5 * eval_cols_copy['number_outpatient']
    ).astype(float)
    new_numeric_cols.append('utilization_severity')

    train_copy['med_instability'] = (
        train_copy['num_changes'] / (train_copy['num_active_meds'] + 1.0)
    ).astype(float)
    eval_cols_copy['med_instability'] = (
        eval_cols_copy['num_changes'] / (eval_cols_copy['num_active_meds'] + 1.0)
    ).astype(float)
    new_numeric_cols.append('med_instability')

    # Encode features with original + newly engineered variables
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        train_copy,
        extra_numeric=tuple(new_numeric_cols),
        extra_categorical=tuple(new_categorical_cols)
    )
    x_train = encoder.transform(train_copy)
    x_eval = encoder.transform(eval_cols_copy)

    # 5-fold Stratified Dual-Engine Ensemble (LightGBM + XGBoost)
    import xgboost as xgb
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(x_eval), dtype=float)

    for train_idx, val_idx in skf.split(x_train, y_train):
        x_tr, y_tr = x_train[train_idx], y_train[train_idx]

        # Engine 1: LightGBM (Leaf-wise histogram gradient booster)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.018,
            num_leaves=31,
            min_child_samples=130,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=1.5,
            reg_alpha=0.2,
            cat_smooth=45.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        lgb_model.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        p_lgb = lgb_model.predict_proba(x_eval)[:, 1]

        # Engine 2: XGBoost (Depth-wise histogram gradient booster)
        xgb_model = xgb.XGBClassifier(
            n_estimators=240,
            learning_rate=0.025,
            max_depth=5,
            min_child_weight=12,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_lambda=2.0,
            reg_alpha=0.2,
            tree_method='hist',
            eval_metric='logloss',
            random_state=42,
            n_jobs=1,
        )
        xgb_model.fit(x_tr, y_tr)
        p_xgb = xgb_model.predict_proba(x_eval)[:, 1]

        # Percentile rank fusion per fold (eliminates cross-model calibration distortion for ROC-AUC)
        from scipy.stats import rankdata
        rank_lgb = rankdata(p_lgb) / len(p_lgb)
        rank_xgb = rankdata(p_xgb) / len(p_xgb)
        preds += (0.55 * rank_lgb + 0.45 * rank_xgb) / 5.0

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
