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
    from sklearn.decomposition import TruncatedSVD
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

    # 3b. Diagnosis Chapter Presence across any diagnosis slot & chapter diversity
    diag_chapters = [
        'circulatory', 'respiratory', 'digestive', 'diabetes',
        'injury', 'musculoskeletal', 'genitourinary', 'neoplasms'
    ]
    for ch in diag_chapters:
        ch_col = f'has_diag_{ch}'
        train_copy[ch_col] = (
            (train_copy['diag_1_group'] == ch) |
            (train_copy['diag_2_group'] == ch) |
            (train_copy['diag_3_group'] == ch)
        ).astype(float)
        eval_cols_copy[ch_col] = (
            (eval_cols_copy['diag_1_group'] == ch) |
            (eval_cols_copy['diag_2_group'] == ch) |
            (eval_cols_copy['diag_3_group'] == ch)
        ).astype(float)
        new_numeric_cols.append(ch_col)

    def count_distinct_chapters(g1, g2, g3):
        return np.array([
            len(set([a, b, c]) - {'missing', 'other'})
            for a, b, c in zip(g1, g2, g3)
        ], dtype=float)

    train_copy['num_distinct_chapters'] = count_distinct_chapters(
        train_copy['diag_1_group'], train_copy['diag_2_group'], train_copy['diag_3_group']
    )
    eval_cols_copy['num_distinct_chapters'] = count_distinct_chapters(
        eval_cols_copy['diag_1_group'], eval_cols_copy['diag_2_group'], eval_cols_copy['diag_3_group']
    )
    new_numeric_cols.append('num_distinct_chapters')

    # 3c. Multi-way Tensor Interaction Encodings
    # Primary diagnosis chapter interacted with discharge disposition
    tr_d1_disch = np.array([f"{g}__{d}" for g, d in zip(train_copy['diag_1_group'], train_copy['discharge_disposition_id'])], dtype=object)
    ev_d1_disch = np.array([f"{g}__{d}" for g, d in zip(eval_cols_copy['diag_1_group'], eval_cols_copy['discharge_disposition_id'])], dtype=object)
    tr_enc, ev_enc = oof_target_encode(tr_d1_disch, y_train, ev_d1_disch, smoothing=25.0, seed=42)
    train_copy['diag1_discharge_te'] = tr_enc
    eval_cols_copy['diag1_discharge_te'] = ev_enc
    new_numeric_cols.append('diag1_discharge_te')

    # Primary diagnosis chapter interacted with secondary diagnosis chapter
    tr_d1_d2 = np.array([f"{g1}__{g2}" for g1, g2 in zip(train_copy['diag_1_group'], train_copy['diag_2_group'])], dtype=object)
    ev_d1_d2 = np.array([f"{g1}__{g2}" for g1, g2 in zip(eval_cols_copy['diag_1_group'], eval_cols_copy['diag_2_group'])], dtype=object)
    tr_enc, ev_enc = oof_target_encode(tr_d1_d2, y_train, ev_d1_d2, smoothing=25.0, seed=42)
    train_copy['diag1_diag2_te'] = tr_enc
    eval_cols_copy['diag1_diag2_te'] = ev_enc
    new_numeric_cols.append('diag1_diag2_te')

    # Admission source interacted with admission type
    tr_adm = np.array([f"{s}__{t}" for s, t in zip(train_copy['admission_source_id'], train_copy['admission_type_id'])], dtype=object)
    ev_adm = np.array([f"{s}__{t}" for s, t in zip(eval_cols_copy['admission_source_id'], eval_cols_copy['admission_type_id'])], dtype=object)
    tr_enc, ev_enc = oof_target_encode(tr_adm, y_train, ev_adm, smoothing=25.0, seed=42)
    train_copy['adm_source_type_te'] = tr_enc
    eval_cols_copy['adm_source_type_te'] = ev_enc
    new_numeric_cols.append('adm_source_type_te')

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

    train_copy['diagnoses_by_hospital_days'] = train_copy['number_diagnoses'] / (train_copy['time_in_hospital'] + 1.0)
    eval_cols_copy['diagnoses_by_hospital_days'] = eval_cols_copy['number_diagnoses'] / (eval_cols_copy['time_in_hospital'] + 1.0)
    new_numeric_cols.append('diagnoses_by_hospital_days')

    train_copy['inpatient_ratio'] = train_copy['number_inpatient'] / (train_copy['total_prior_visits'] + 1.0)
    eval_cols_copy['inpatient_ratio'] = eval_cols_copy['number_inpatient'] / (eval_cols_copy['total_prior_visits'] + 1.0)
    new_numeric_cols.append('inpatient_ratio')

    train_copy['emergency_ratio'] = train_copy['number_emergency'] / (train_copy['total_prior_visits'] + 1.0)
    eval_cols_copy['emergency_ratio'] = eval_cols_copy['number_emergency'] / (eval_cols_copy['total_prior_visits'] + 1.0)
    new_numeric_cols.append('emergency_ratio')

    # 5. Vectorized Medication Processing & Joint Clinical Tensor Decomposition
    med_cols = [
        col for col in task.categorical_columns
        if col not in ['change', 'diabetesMed'] and any(v in set(train_copy[col]) for v in ['Steady', 'Up', 'Down'])
    ]

    med_val_map = {'No': 0.0, 'Steady': 1.0, 'Up': 2.0, 'Down': -1.0}
    train_med_mat = np.column_stack([
        np.array([med_val_map.get(x, 0.0) for x in train_copy[col]], dtype=np.float32)
        for col in med_cols
    ])
    eval_med_mat = np.column_stack([
        np.array([med_val_map.get(x, 0.0) for x in eval_cols_copy[col]], dtype=np.float32)
        for col in med_cols
    ])

    train_copy['num_active_meds'] = np.sum(train_med_mat != 0.0, axis=1).astype(float)
    eval_cols_copy['num_active_meds'] = np.sum(eval_med_mat != 0.0, axis=1).astype(float)
    new_numeric_cols.append('num_active_meds')

    train_copy['num_up_meds'] = np.sum(train_med_mat == 2.0, axis=1).astype(float)
    eval_cols_copy['num_up_meds'] = np.sum(eval_med_mat == 2.0, axis=1).astype(float)
    new_numeric_cols.append('num_up_meds')

    train_copy['num_down_meds'] = np.sum(train_med_mat == -1.0, axis=1).astype(float)
    eval_cols_copy['num_down_meds'] = np.sum(eval_med_mat == -1.0, axis=1).astype(float)
    new_numeric_cols.append('num_down_meds')

    train_copy['num_changes'] = train_copy['num_up_meds'] + train_copy['num_down_meds']
    eval_cols_copy['num_changes'] = eval_cols_copy['num_up_meds'] + eval_cols_copy['num_down_meds']
    new_numeric_cols.append('num_changes')

    # Unified Low-Rank Clinical Tensor Decomposition (Coupled Comorbidity & Medication Phenotype)
    cm_keys = [
        'cm_mi', 'cm_chf', 'cm_pvd', 'cm_cerebrovascular', 'cm_dementia',
        'cm_pulmonary', 'cm_rheumatic', 'cm_peptic_ulcer', 'cm_liver',
        'cm_hemiplegia', 'cm_renal', 'cm_cancer', 'cm_metastatic', 'cm_hiv',
        'cm_diabetes_complicated'
    ]
    train_cm_mat = np.column_stack([train_copy[k] for k in cm_keys])
    eval_cm_mat = np.column_stack([eval_cols_copy[k] for k in cm_keys])

    train_clinical_mat = np.column_stack([train_cm_mat, train_med_mat])
    eval_clinical_mat = np.column_stack([eval_cm_mat, eval_med_mat])

    svd_clinical = TruncatedSVD(n_components=6, random_state=42)
    tr_clin_svd = svd_clinical.fit_transform(train_clinical_mat)
    ev_clin_svd = svd_clinical.transform(eval_clinical_mat)
    for i in range(6):
        col_name = f'clinical_svd_{i}'
        train_copy[col_name] = tr_clin_svd[:, i]
        eval_cols_copy[col_name] = ev_clin_svd[:, i]
        new_numeric_cols.append(col_name)

    # 5c. Lab Result Severity and Clinical Inertia
    a1c_map = {'None': 0.0, 'Norm': 5.0, '>7': 7.5, '>8': 9.0}
    train_copy['a1c_numeric'] = np.array([a1c_map.get(x, 0.0) for x in train_copy['A1Cresult']], dtype=float)
    eval_cols_copy['a1c_numeric'] = np.array([a1c_map.get(x, 0.0) for x in eval_cols_copy['A1Cresult']], dtype=float)
    new_numeric_cols.append('a1c_numeric')

    glu_map = {'None': 0.0, 'Norm': 100.0, '>200': 200.0, '>300': 300.0}
    train_copy['glu_numeric'] = np.array([glu_map.get(x, 0.0) for x in train_copy['max_glu_serum']], dtype=float)
    eval_cols_copy['glu_numeric'] = np.array([glu_map.get(x, 0.0) for x in eval_cols_copy['max_glu_serum']], dtype=float)
    new_numeric_cols.append('glu_numeric')

    train_copy['high_a1c_no_med_change'] = (
        np.isin(train_copy['A1Cresult'], ['>7', '>8']) & (train_copy['change'] == 'No')
    ).astype(float)
    eval_cols_copy['high_a1c_no_med_change'] = (
        np.isin(eval_cols_copy['A1Cresult'], ['>7', '>8']) & (eval_cols_copy['change'] == 'No')
    ).astype(float)
    new_numeric_cols.append('high_a1c_no_med_change')

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

    # 5-fold Cross-Validated Gradient Boosting Ensemble
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(x_eval))

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_tr, y_tr = x_train[train_idx], y_train[train_idx]

        model = lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.018,
            num_leaves=24,
            min_child_samples=160,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=3.0,
            reg_alpha=0.3,
            cat_smooth=45.0,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1] / 5.0

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
