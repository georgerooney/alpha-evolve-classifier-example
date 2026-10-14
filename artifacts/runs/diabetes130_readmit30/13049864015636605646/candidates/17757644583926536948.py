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

    # 7. ICD-9 3-Digit Disease Prefixes
    def get_icd9_prefix(arr):
        res = []
        for x in arr:
            if x is None or str(x).lower() in ('none', 'nan', ''):
                res.append('missing')
            else:
                res.append(str(x).split('.')[0])
        return np.array(res, dtype=object)

    for col in ['diag_1', 'diag_2', 'diag_3']:
        prefix_col = f"{col}_prefix"
        train_copy[prefix_col] = get_icd9_prefix(train_copy[col])
        eval_cols_copy[prefix_col] = get_icd9_prefix(eval_cols_copy[col])
        new_categorical_cols.append(prefix_col)

        te_name = f"{prefix_col}_te"
        tr_enc, ev_enc = oof_target_encode(
            train_copy[prefix_col], y_train, eval_cols_copy[prefix_col], smoothing=15.0, seed=42
        )
        train_copy[te_name] = tr_enc
        eval_cols_copy[te_name] = ev_enc
        new_numeric_cols.append(te_name)

    # 8. Admission & Discharge Source Interactions
    adm_disch_tr = np.array([f"{a}_{d}" for a, d in zip(train_copy['admission_source_id'], train_copy['discharge_disposition_id'])], dtype=object)
    adm_disch_ev = np.array([f"{a}_{d}" for a, d in zip(eval_cols_copy['admission_source_id'], eval_cols_copy['discharge_disposition_id'])], dtype=object)
    train_copy['adm_disch_interaction'] = adm_disch_tr
    eval_cols_copy['adm_disch_interaction'] = adm_disch_ev
    new_categorical_cols.append('adm_disch_interaction')

    ad_tr_enc, ad_ev_enc = oof_target_encode(adm_disch_tr, y_train, adm_disch_ev, smoothing=20.0, seed=42)
    train_copy['adm_disch_te'] = ad_tr_enc
    eval_cols_copy['adm_disch_te'] = ad_ev_enc
    new_numeric_cols.append('adm_disch_te')

    # Target encode age
    age_te_tr, age_te_ev = oof_target_encode(train_copy['age'], y_train, eval_cols_copy['age'], smoothing=50.0, seed=42)
    train_copy['age_te'] = age_te_tr
    eval_cols_copy['age_te'] = age_te_ev
    new_numeric_cols.append('age_te')

    # 9. Clinical Acuity and Utilization Indicators
    for df in [train_copy, eval_cols_copy]:
        df['is_discharged_home'] = (df['discharge_disposition_id'] == '1').astype(float)
        df['discharged_to_facility'] = np.isin(df['discharge_disposition_id'], ['3', '4', '5', '6', '22']).astype(float)
        df['is_emergency_admission'] = (df['admission_type_id'] == '1').astype(float)
        df['is_emergency_source'] = (df['admission_source_id'] == '7').astype(float)

        df['has_multiple_inpatient'] = (df['number_inpatient'] >= 2.0).astype(float)
        df['inpatient_ratio'] = df['number_inpatient'] / (df['total_prior_visits'] + 1.0)
        df['emergency_ratio'] = df['number_emergency'] / (df['total_prior_visits'] + 1.0)
        df['high_utilizer'] = ((df['number_inpatient'] >= 2.0) | (df['total_prior_visits'] >= 5.0)).astype(float)

        df['log_prior_inpatient'] = np.log1p(np.maximum(0.0, df['number_inpatient']))
        df['log_total_prior'] = np.log1p(np.maximum(0.0, df['total_prior_visits']))
        df['log_hospital_time'] = np.log1p(np.maximum(0.0, df['time_in_hospital']))
        df['log_num_medications'] = np.log1p(np.maximum(0.0, df['num_medications']))
        df['med_per_diagnosis'] = df['num_medications'] / (df['number_diagnoses'] + 1.0)
        df['total_acuity'] = df['time_in_hospital'] * (df['number_diagnoses'] + df['cm_count'])

        # Primary diagnosis groups
        df['primary_circulatory'] = (df['diag_1_group'] == 'circulatory').astype(float)
        df['primary_respiratory'] = (df['diag_1_group'] == 'respiratory').astype(float)
        df['primary_diabetes'] = (df['diag_1_group'] == 'diabetes').astype(float)

        # Diabetes lab and medication markers
        df['a1c_tested'] = (df['A1Cresult'] != 'None').astype(float)
        df['a1c_high'] = np.isin(df['A1Cresult'], ['>7', '>8']).astype(float)
        df['max_glu_tested'] = (df['max_glu_serum'] != 'None').astype(float)
        df['max_glu_high'] = np.isin(df['max_glu_serum'], ['>200', '>300']).astype(float)
        df['change_on_high_a1c'] = ((df['a1c_high'] == 1.0) & (df['change'] == 'Ch')).astype(float)

        df['on_insulin'] = (df['insulin'] != 'No').astype(float)
        df['insulin_up'] = (df['insulin'] == 'Up').astype(float)
        df['insulin_down'] = (df['insulin'] == 'Down').astype(float)
        insulin_map = {'No': 0.0, 'Steady': 1.0, 'Down': 2.0, 'Up': 3.0}
        df['insulin_level'] = np.array([insulin_map.get(x, 0.0) for x in df['insulin']], dtype=float)
        df['on_metformin'] = (df['metformin'] != 'No').astype(float)
        df['metformin_and_insulin'] = ((df['metformin'] != 'No') & (df['insulin'] != 'No')).astype(float)
        df['insulin_without_metformin'] = ((df['insulin'] != 'No') & (df['metformin'] == 'No')).astype(float)
        df['change_flag'] = (df['change'] == 'Ch').astype(float)
        df['diabetes_med_flag'] = (df['diabetesMed'] == 'Yes').astype(float)

        # Demographics & Missingness
        df['age_x_diagnoses'] = df['age_numeric'] * df['number_diagnoses']
        df['age_x_inpatient'] = df['age_numeric'] * df['number_inpatient']
        df['specialty_missing'] = np.array([1.0 if x is None or str(x).lower() in ('?', 'none') else 0.0 for x in df['medical_specialty']], dtype=float)
        df['payer_missing'] = np.array([1.0 if x is None or str(x).lower() in ('?', 'none') else 0.0 for x in df['payer_code']], dtype=float)
        df['has_weight'] = np.array([1.0 if x is not None and str(x).lower() not in ('?', 'none') else 0.0 for x in df['weight']], dtype=float)

    added_numeric = [
        'is_discharged_home', 'discharged_to_facility', 'is_emergency_admission', 'is_emergency_source',
        'has_multiple_inpatient', 'inpatient_ratio', 'emergency_ratio', 'high_utilizer',
        'log_prior_inpatient', 'log_total_prior', 'log_hospital_time', 'log_num_medications',
        'med_per_diagnosis', 'total_acuity', 'primary_circulatory', 'primary_respiratory',
        'primary_diabetes', 'a1c_tested', 'a1c_high', 'max_glu_tested', 'max_glu_high',
        'change_on_high_a1c', 'on_insulin', 'insulin_up', 'insulin_down', 'insulin_level',
        'on_metformin', 'metformin_and_insulin', 'insulin_without_metformin', 'change_flag',
        'diabetes_med_flag', 'age_x_diagnoses', 'age_x_inpatient', 'specialty_missing',
        'payer_missing', 'has_weight'
    ]
    new_numeric_cols.extend(added_numeric)

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

    import xgboost as xgb

    # 5-fold Cross-Validated LightGBM + XGBoost Ensemble
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(x_eval))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_tr, y_tr = x_train[train_idx], y_train[train_idx]

        # Model 1: LightGBM (leaf-wise tree growth)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.018,
            num_leaves=31,
            min_child_samples=120,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=2.0,
            reg_alpha=0.2,
            cat_smooth=40.0,
            random_state=42 + fold_idx,
            n_jobs=1,
            verbose=-1,
        )
        lgb_model.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        lgb_preds = lgb_model.predict_proba(x_eval)[:, 1]

        # Model 2: XGBoost (depth-wise tree growth with histogram binning)
        xgb_model = xgb.XGBClassifier(
            n_estimators=240,
            learning_rate=0.025,
            max_depth=4,
            min_child_weight=6,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_lambda=3.0,
            reg_alpha=0.5,
            tree_method='hist',
            random_state=42 + fold_idx,
            n_jobs=1,
            verbosity=0,
        )
        xgb_model.fit(x_tr, y_tr)
        xgb_preds = xgb_model.predict_proba(x_eval)[:, 1]

        # Blend predictions across models
        fold_blend = 0.65 * lgb_preds + 0.35 * xgb_preds
        preds += fold_blend / 5.0

    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
