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

    # 2. ICD-9 Chapter Groups and 3-digit Clinical Prefixes
    def get_icd9_prefix(val):
        if val is None or val == '?' or val == '' or (isinstance(val, float) and np.isnan(val)):
            return 'missing'
        return str(val).strip().split('.')[0]

    for col in ['diag_1', 'diag_2', 'diag_3']:
        group_col_name = f"{col}_group"
        train_copy[group_col_name] = np.array([icd9_group(x) for x in train_copy[col]], dtype=object)
        eval_cols_copy[group_col_name] = np.array([icd9_group(x) for x in eval_cols_copy[col]], dtype=object)
        new_categorical_cols.append(group_col_name)

        prefix_col_name = f"{col}_prefix"
        train_copy[prefix_col_name] = np.array([get_icd9_prefix(x) for x in train_copy[col]], dtype=object)
        eval_cols_copy[prefix_col_name] = np.array([get_icd9_prefix(x) for x in eval_cols_copy[col]], dtype=object)
        new_categorical_cols.append(prefix_col_name)

    # Admission-Discharge trajectory combination
    train_adm_disch = np.array([
        f"{str(a)}_{str(d)}"
        for a, d in zip(train_copy['admission_type_id'], train_copy['discharge_disposition_id'])
    ], dtype=object)
    eval_adm_disch = np.array([
        f"{str(a)}_{str(d)}"
        for a, d in zip(eval_cols_copy['admission_type_id'], eval_cols_copy['discharge_disposition_id'])
    ], dtype=object)
    train_copy['adm_disch_combo'] = train_adm_disch
    eval_cols_copy['adm_disch_combo'] = eval_adm_disch
    new_categorical_cols.append('adm_disch_combo')

    # 3. High-cardinality and important categorical Out-of-Fold Target Encodings
    cols_to_target_encode = [
        'diag_1', 'diag_2', 'diag_3', 'diag_1_prefix', 'diag_2_prefix',
        'medical_specialty', 'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'payer_code', 'adm_disch_combo', 'age'
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

    train_copy['multi_prior_inpatient'] = (train_copy['number_inpatient'] > 1).astype(float)
    eval_cols_copy['multi_prior_inpatient'] = (eval_cols_copy['number_inpatient'] > 1).astype(float)
    new_numeric_cols.append('multi_prior_inpatient')

    train_copy['log_inpatient'] = np.log1p(train_copy['number_inpatient'])
    eval_cols_copy['log_inpatient'] = np.log1p(eval_cols_copy['number_inpatient'])
    new_numeric_cols.append('log_inpatient')

    train_copy['inpatient_ratio'] = train_copy['number_inpatient'] / (train_copy['total_prior_visits'] + 1.0)
    eval_cols_copy['inpatient_ratio'] = eval_cols_copy['number_inpatient'] / (eval_cols_copy['total_prior_visits'] + 1.0)
    new_numeric_cols.append('inpatient_ratio')

    train_copy['utilization_risk_score'] = (
        3.0 * train_copy['number_inpatient'] + 1.5 * train_copy['number_emergency'] + 0.5 * train_copy['number_outpatient']
    )
    eval_cols_copy['utilization_risk_score'] = (
        3.0 * eval_cols_copy['number_inpatient'] + 1.5 * eval_cols_copy['number_emergency'] + 0.5 * eval_cols_copy['number_outpatient']
    )
    new_numeric_cols.append('utilization_risk_score')

    train_copy['inp_times_hospital'] = train_copy['number_inpatient'] * train_copy['time_in_hospital']
    eval_cols_copy['inp_times_hospital'] = eval_cols_copy['number_inpatient'] * eval_cols_copy['time_in_hospital']
    new_numeric_cols.append('inp_times_hospital')

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

    train_copy['age_x_inpatient'] = train_copy['age_numeric'] * train_copy['number_inpatient']
    eval_cols_copy['age_x_inpatient'] = eval_cols_copy['age_numeric'] * eval_cols_copy['number_inpatient']
    new_numeric_cols.append('age_x_inpatient')

    # 7. Clinical Indicators (Insulin, Lab tests, High-risk diagnoses, Inertia)
    for df in [train_copy, eval_cols_copy]:
        ins = df['insulin']
        df['insulin_on'] = ((ins != 'No') & (ins != None)).astype(float)
        df['insulin_up'] = (ins == 'Up').astype(float)
        df['insulin_down'] = (ins == 'Down').astype(float)
        df['insulin_change'] = ((ins == 'Up') | (ins == 'Down')).astype(float)

        a1c = df['A1Cresult']
        glu = df['max_glu_serum']
        df['a1c_tested'] = ((a1c != 'None') & (a1c != None)).astype(float)
        df['a1c_high'] = ((a1c == '>7') | (a1c == '>8')).astype(float)
        df['glu_tested'] = ((glu != 'None') & (glu != None)).astype(float)
        df['glu_high'] = ((glu == '>200') | (glu == '>300')).astype(float)

        has_change = (df['change'] == 'Ch').astype(float)
        df['clinical_inertia'] = (df['a1c_high'] * (1.0 - has_change)).astype(float)

        d1 = df['diag_1_prefix']
        d2 = df['diag_2_prefix']
        d3 = df['diag_3_prefix']
        df['has_heart_failure'] = ((d1 == '428') | (d2 == '428') | (d3 == '428') | (df['cm_chf'] == 1.0)).astype(float)
        df['has_kidney_failure'] = ((d1 == '584') | (d2 == '584') | (d3 == '584') | (d1 == '585') | (d2 == '585') | (d3 == '585') | (df['cm_renal'] == 1.0)).astype(float)
        df['cardiorenal'] = (df['has_heart_failure'] * df['has_kidney_failure']).astype(float)

        disch = df['discharge_disposition_id']
        df['disch_home'] = (disch == '1').astype(float)
        df['disch_snf'] = ((disch == '3') | (disch == '4') | (disch == '5') | (disch == '22')).astype(float)
        df['disch_home_health'] = (disch == '6').astype(float)
        df['disch_ama'] = (disch == '7').astype(float)

        df['num_diag_groups'] = np.array([
            float(len(set(g for g in [g1, g2, g3] if g and g != 'missing')))
            for g1, g2, g3 in zip(df['diag_1_group'], df['diag_2_group'], df['diag_3_group'])
        ], dtype=float)

    clinical_added = [
        'insulin_on', 'insulin_up', 'insulin_down', 'insulin_change',
        'a1c_tested', 'a1c_high', 'glu_tested', 'glu_high', 'clinical_inertia',
        'has_heart_failure', 'has_kidney_failure', 'cardiorenal',
        'disch_home', 'disch_snf', 'disch_home_health', 'disch_ama', 'num_diag_groups'
    ]
    new_numeric_cols.extend(clinical_added)

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

    # 5-fold Cross-Validated Heterogeneous Ensemble (LightGBM + XGBoost)
    import xgboost as xgb
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(x_eval))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_tr, y_tr = x_train[train_idx], y_train[train_idx]

        # 1. LightGBM with native categorical handling
        lgb_model = lgb.LGBMClassifier(
            n_estimators=360,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=120,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=2.0,
            reg_alpha=0.3,
            cat_smooth=30.0,
            random_state=42 + fold_idx,
            n_jobs=1,
            verbose=-1,
        )
        lgb_model.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        lgb_preds = lgb_model.predict_proba(x_eval)[:, 1]

        # 2. XGBoost with histogram method
        xgb_model = xgb.XGBClassifier(
            n_estimators=320,
            learning_rate=0.025,
            max_depth=5,
            min_child_weight=15,
            subsample=0.75,
            colsample_bytree=0.6,
            reg_alpha=0.3,
            reg_lambda=2.0,
            tree_method='hist',
            random_state=42 + fold_idx,
            n_jobs=1,
            eval_metric='logloss',
        )
        xgb_model.fit(x_tr, y_tr)
        xgb_preds = xgb_model.predict_proba(x_eval)[:, 1]

        # Blend predictions across the fold
        fold_blend = 0.60 * lgb_preds + 0.40 * xgb_preds
        preds += fold_blend / 5.0

    preds = np.clip(preds, 0.0, 1.0)
    if np.isnan(preds).any():
        preds = np.nan_to_num(preds, nan=float(np.mean(y_train)))

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
