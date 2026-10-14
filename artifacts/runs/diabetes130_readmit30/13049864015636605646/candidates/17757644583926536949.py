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

    import numpy as np

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

    # 3. High-cardinality and important categorical Out-of-Fold Target Encodings (including Chapter Groups)
    cols_to_target_encode = [
        'diag_1', 'diag_2', 'diag_3', 'medical_specialty',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'payer_code',
        'diag_1_group', 'diag_2_group', 'diag_3_group'
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

    # 4b. Ratios & Interactions
    train_copy['emergency_ratio'] = train_copy['number_emergency'] / (train_copy['total_prior_visits'] + 1e-5)
    eval_cols_copy['emergency_ratio'] = eval_cols_copy['number_emergency'] / (eval_cols_copy['total_prior_visits'] + 1e-5)
    new_numeric_cols.append('emergency_ratio')

    train_copy['inpatient_ratio'] = train_copy['number_inpatient'] / (train_copy['total_prior_visits'] + 1e-5)
    eval_cols_copy['inpatient_ratio'] = eval_cols_copy['number_inpatient'] / (eval_cols_copy['total_prior_visits'] + 1e-5)
    new_numeric_cols.append('inpatient_ratio')

    train_copy['inpatient_times_hospital_days'] = train_copy['number_inpatient'] * train_copy['time_in_hospital']
    eval_cols_copy['inpatient_times_hospital_days'] = eval_cols_copy['number_inpatient'] * eval_cols_copy['time_in_hospital']
    new_numeric_cols.append('inpatient_times_hospital_days')

    train_copy['diagnoses_times_medications'] = train_copy['number_diagnoses'] * train_copy['num_medications']
    eval_cols_copy['diagnoses_times_medications'] = eval_cols_copy['number_diagnoses'] * eval_cols_copy['num_medications']
    new_numeric_cols.append('diagnoses_times_medications')

    # 4c. Specific Admission and Discharge Indicators
    train_copy['discharged_to_home'] = (train_copy['discharge_disposition_id'] == '1').astype(float)
    eval_cols_copy['discharged_to_home'] = (eval_cols_copy['discharge_disposition_id'] == '1').astype(float)
    new_numeric_cols.append('discharged_to_home')

    train_copy['discharged_to_snf'] = (train_copy['discharge_disposition_id'] == '3').astype(float)
    eval_cols_copy['discharged_to_snf'] = (eval_cols_copy['discharge_disposition_id'] == '3').astype(float)
    new_numeric_cols.append('discharged_to_snf')

    train_copy['discharged_to_hha'] = (train_copy['discharge_disposition_id'] == '6').astype(float)
    eval_cols_copy['discharged_to_hha'] = (eval_cols_copy['discharge_disposition_id'] == '6').astype(float)
    new_numeric_cols.append('discharged_to_hha')

    train_copy['admission_from_er'] = (train_copy['admission_source_id'] == '7').astype(float)
    eval_cols_copy['admission_from_er'] = (eval_cols_copy['admission_source_id'] == '7').astype(float)
    new_numeric_cols.append('admission_from_er')

    train_copy['admission_from_referral'] = (train_copy['admission_source_id'] == '1').astype(float)
    eval_cols_copy['admission_from_referral'] = (eval_cols_copy['admission_source_id'] == '1').astype(float)
    new_numeric_cols.append('admission_from_referral')

    train_copy['admission_type_emergency'] = (train_copy['admission_type_id'] == '1').astype(float)
    eval_cols_copy['admission_type_emergency'] = (eval_cols_copy['admission_type_id'] == '1').astype(float)
    new_numeric_cols.append('admission_type_emergency')

    train_copy['admission_type_elective'] = (train_copy['admission_type_id'] == '3').astype(float)
    eval_cols_copy['admission_type_elective'] = (eval_cols_copy['admission_type_id'] == '3').astype(float)
    new_numeric_cols.append('admission_type_elective')

    # 4d. Lab results indicators
    train_copy['has_a1c_test'] = ((train_copy['A1Cresult'] != 'None') & (train_copy['A1Cresult'] != None)).astype(float)
    eval_cols_copy['has_a1c_test'] = ((eval_cols_copy['A1Cresult'] != 'None') & (eval_cols_copy['A1Cresult'] != None)).astype(float)
    new_numeric_cols.append('has_a1c_test')

    train_copy['has_glu_test'] = ((train_copy['max_glu_serum'] != 'None') & (train_copy['max_glu_serum'] != None)).astype(float)
    eval_cols_copy['has_glu_test'] = ((eval_cols_copy['max_glu_serum'] != 'None') & (eval_cols_copy['max_glu_serum'] != None)).astype(float)
    new_numeric_cols.append('has_glu_test')

    train_copy['a1c_high'] = ((train_copy['A1Cresult'] == '>7') | (train_copy['A1Cresult'] == '>8')).astype(float)
    eval_cols_copy['a1c_high'] = ((eval_cols_copy['A1Cresult'] == '>7') | (eval_cols_copy['A1Cresult'] == '>8')).astype(float)
    new_numeric_cols.append('a1c_high')

    train_copy['glu_high'] = ((train_copy['max_glu_serum'] == '>200') | (train_copy['max_glu_serum'] == '>300')).astype(float)
    eval_cols_copy['glu_high'] = ((eval_cols_copy['max_glu_serum'] == '>200') | (eval_cols_copy['max_glu_serum'] == '>300')).astype(float)
    new_numeric_cols.append('glu_high')

    # 4e. Missingness indicators
    train_copy['has_weight'] = ((train_copy['weight'] != None) & (train_copy['weight'] != 'None') & (train_copy['weight'] != '?')).astype(float)
    eval_cols_copy['has_weight'] = ((eval_cols_copy['weight'] != None) & (eval_cols_copy['weight'] != 'None') & (eval_cols_copy['weight'] != '?')).astype(float)
    new_numeric_cols.append('has_weight')

    train_copy['has_medical_specialty'] = ((train_copy['medical_specialty'] != None) & (train_copy['medical_specialty'] != 'None') & (train_copy['medical_specialty'] != '?')).astype(float)
    eval_cols_copy['has_medical_specialty'] = ((eval_cols_copy['medical_specialty'] != None) & (eval_cols_copy['medical_specialty'] != 'None') & (eval_cols_copy['medical_specialty'] != '?')).astype(float)
    new_numeric_cols.append('has_medical_specialty')

    train_copy['has_payer_code'] = ((train_copy['payer_code'] != None) & (train_copy['payer_code'] != 'None') & (train_copy['payer_code'] != '?')).astype(float)
    eval_cols_copy['has_payer_code'] = ((eval_cols_copy['payer_code'] != None) & (eval_cols_copy['payer_code'] != 'None') & (eval_cols_copy['payer_code'] != '?')).astype(float)
    new_numeric_cols.append('has_payer_code')

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

    # 5b. Continuous Medication Features
    med_map = {'No': 0.0, 'Down': 1.0, 'Steady': 2.0, 'Up': 3.0}
    for col in med_cols:
        med_num_col = f"{col}_numeric"
        train_copy[med_num_col] = np.array([med_map.get(x, 0.0) for x in train_copy[col]], dtype=float)
        eval_cols_copy[med_num_col] = np.array([med_map.get(x, 0.0) for x in eval_cols_copy[col]], dtype=float)
        new_numeric_cols.append(med_num_col)

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

    # Age interactions
    train_copy['age_times_inpatient_ratio'] = train_copy['age_numeric'] * train_copy['inpatient_ratio']
    eval_cols_copy['age_times_inpatient_ratio'] = eval_cols_copy['age_numeric'] * eval_cols_copy['inpatient_ratio']
    new_numeric_cols.append('age_times_inpatient_ratio')

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

    # 5-fold Cross-Validated Model Ensemble
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(x_eval))

    for train_idx, val_idx in skf.split(x_train, y_train):
        x_tr, y_tr = x_train[train_idx], y_train[train_idx]
        
        # Model 1: baseline optimized leaves
        model1 = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.015,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=1.0,
            reg_alpha=0.1,
            cat_smooth=50.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        model1.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        preds += 0.5 * model1.predict_proba(x_eval)[:, 1] / 5.0

        # Model 2: deeper leaves, different learning rate and regularization
        model2 = lgb.LGBMClassifier(
            n_estimators=380,
            learning_rate=0.012,
            num_leaves=45,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=5.0,
            reg_alpha=0.5,
            cat_smooth=30.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        )
        model2.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        preds += 0.5 * model2.predict_proba(x_eval)[:, 1] / 5.0

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
