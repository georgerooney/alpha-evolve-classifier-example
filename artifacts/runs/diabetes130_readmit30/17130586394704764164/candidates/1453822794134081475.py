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
    train_dict = dict(train)
    eval_dict = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []
    n_train = len(y_train)
    n_eval = len(eval_cols['time_in_hospital'])

    # 1. Comorbidities
    cm_train = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k, v in cm_train.items():
        train_dict[k] = v.astype(np.float64)
        eval_dict[k] = cm_eval[k].astype(np.float64)
        extra_numeric.append(k)

    # 2. ICD-9 Chapter Groups
    for diag in ('diag_1', 'diag_2', 'diag_3'):
        grp_col = f'{diag}_grp'
        train_dict[grp_col] = np.array([icd9_group(c) for c in train[diag]], dtype=object)
        eval_dict[grp_col] = np.array([icd9_group(c) for c in eval_cols[diag]], dtype=object)
        extra_categorical.append(grp_col)

    for grp_val in ('circulatory', 'respiratory', 'diabetes', 'digestive'):
        col_name = f'has_grp_{grp_val}'
        train_dict[col_name] = ((train_dict['diag_1_grp'] == grp_val) |
                                (train_dict['diag_2_grp'] == grp_val) |
                                (train_dict['diag_3_grp'] == grp_val)).astype(np.float64)
        eval_dict[col_name] = ((eval_dict['diag_1_grp'] == grp_val) |
                               (eval_dict['diag_2_grp'] == grp_val) |
                               (eval_dict['diag_3_grp'] == grp_val)).astype(np.float64)
        extra_numeric.append(col_name)

    train_dict['diag1_is_circulatory'] = (train_dict['diag_1_grp'] == 'circulatory').astype(np.float64)
    eval_dict['diag1_is_circulatory'] = (eval_dict['diag_1_grp'] == 'circulatory').astype(np.float64)
    extra_numeric.append('diag1_is_circulatory')

    train_dict['diag1_is_diabetes'] = (train_dict['diag_1_grp'] == 'diabetes').astype(np.float64)
    eval_dict['diag1_is_diabetes'] = (eval_dict['diag_1_grp'] == 'diabetes').astype(np.float64)
    extra_numeric.append('diag1_is_diabetes')

    # 3. Out-of-Fold Target Encoding
    te_candidates = [
        'diag_1', 'diag_2', 'diag_3',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'medical_specialty'
    ]
    for col in te_candidates:
        if col in train and col in eval_cols:
            tr_te, ev_te = oof_target_encode(
                train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42
            )
            col_name = f'te_{col}'
            train_dict[col_name] = tr_te.astype(np.float64)
            eval_dict[col_name] = ev_te.astype(np.float64)
            extra_numeric.append(col_name)

    if 'te_diag_1' in train_dict and 'te_diag_2' in train_dict and 'te_diag_3' in train_dict:
        train_dict['te_diag_max'] = np.maximum(
            np.maximum(train_dict['te_diag_1'], train_dict['te_diag_2']), train_dict['te_diag_3']
        )
        eval_dict['te_diag_max'] = np.maximum(
            np.maximum(eval_dict['te_diag_1'], eval_dict['te_diag_2']), eval_dict['te_diag_3']
        )
        extra_numeric.append('te_diag_max')

        train_dict['te_diag_mean'] = (
            train_dict['te_diag_1'] + train_dict['te_diag_2'] + train_dict['te_diag_3']
        ) / 3.0
        eval_dict['te_diag_mean'] = (
            eval_dict['te_diag_1'] + eval_dict['te_diag_2'] + eval_dict['te_diag_3']
        ) / 3.0
        extra_numeric.append('te_diag_mean')

    # 4. Utilization & Encounter Intensity
    tr_inp = np.nan_to_num(train['number_inpatient'], nan=0.0)
    ev_inp = np.nan_to_num(eval_cols['number_inpatient'], nan=0.0)
    tr_emg = np.nan_to_num(train['number_emergency'], nan=0.0)
    ev_emg = np.nan_to_num(eval_cols['number_emergency'], nan=0.0)
    tr_out = np.nan_to_num(train['number_outpatient'], nan=0.0)
    ev_out = np.nan_to_num(eval_cols['number_outpatient'], nan=0.0)
    tr_los = np.maximum(np.nan_to_num(train['time_in_hospital'], nan=1.0), 1.0)
    ev_los = np.maximum(np.nan_to_num(eval_cols['time_in_hospital'], nan=1.0), 1.0)
    tr_med = np.nan_to_num(train['num_medications'], nan=0.0)
    ev_med = np.nan_to_num(eval_cols['num_medications'], nan=0.0)
    tr_lab = np.nan_to_num(train['num_lab_procedures'], nan=0.0)
    ev_lab = np.nan_to_num(eval_cols['num_lab_procedures'], nan=0.0)
    tr_prc = np.nan_to_num(train['num_procedures'], nan=0.0)
    ev_prc = np.nan_to_num(eval_cols['num_procedures'], nan=0.0)
    tr_dx = np.nan_to_num(train['number_diagnoses'], nan=0.0)
    ev_dx = np.nan_to_num(eval_cols['number_diagnoses'], nan=0.0)

    tr_tot_prior = tr_inp + tr_emg + tr_out
    ev_tot_prior = ev_inp + ev_emg + ev_out
    train_dict['total_prior_visits'] = tr_tot_prior
    eval_dict['total_prior_visits'] = ev_tot_prior
    extra_numeric.append('total_prior_visits')

    train_dict['inpatient_ratio'] = tr_inp / (tr_tot_prior + 1.0)
    eval_dict['inpatient_ratio'] = ev_inp / (ev_tot_prior + 1.0)
    extra_numeric.append('inpatient_ratio')

    train_dict['emergency_ratio'] = tr_emg / (tr_tot_prior + 1.0)
    eval_dict['emergency_ratio'] = ev_emg / (ev_tot_prior + 1.0)
    extra_numeric.append('emergency_ratio')

    train_dict['has_inpatient'] = (tr_inp > 0).astype(np.float64)
    eval_dict['has_inpatient'] = (ev_inp > 0).astype(np.float64)
    extra_numeric.append('has_inpatient')

    train_dict['has_emergency'] = (tr_emg > 0).astype(np.float64)
    eval_dict['has_emergency'] = (ev_emg > 0).astype(np.float64)
    extra_numeric.append('has_emergency')

    train_dict['has_prior_visit'] = (tr_tot_prior > 0).astype(np.float64)
    eval_dict['has_prior_visit'] = (ev_tot_prior > 0).astype(np.float64)
    extra_numeric.append('has_prior_visit')

    train_dict['frequent_inpatient'] = (tr_inp >= 2).astype(np.float64)
    eval_dict['frequent_inpatient'] = (ev_inp >= 2).astype(np.float64)
    extra_numeric.append('frequent_inpatient')

    train_dict['meds_per_day'] = tr_med / tr_los
    eval_dict['meds_per_day'] = ev_med / ev_los
    extra_numeric.append('meds_per_day')

    train_dict['labs_per_day'] = tr_lab / tr_los
    eval_dict['labs_per_day'] = ev_lab / ev_los
    extra_numeric.append('labs_per_day')

    train_dict['procs_per_day'] = tr_prc / tr_los
    eval_dict['procs_per_day'] = ev_prc / ev_los
    extra_numeric.append('procs_per_day')

    train_dict['dx_per_day'] = tr_dx / tr_los
    eval_dict['dx_per_day'] = ev_dx / ev_los
    extra_numeric.append('dx_per_day')

    train_dict['total_services'] = tr_med + tr_lab + tr_prc
    eval_dict['total_services'] = ev_med + ev_lab + ev_prc
    extra_numeric.append('total_services')

    train_dict['services_per_day'] = train_dict['total_services'] / tr_los
    eval_dict['services_per_day'] = eval_dict['total_services'] / ev_los
    extra_numeric.append('services_per_day')

    # 5. Age Numeric & Interactions
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    tr_age_num = np.array([age_map.get(x, 65.0) for x in train['age']], dtype=np.float64)
    ev_age_num = np.array([age_map.get(x, 65.0) for x in eval_cols['age']], dtype=np.float64)
    train_dict['age_num'] = tr_age_num
    eval_dict['age_num'] = ev_age_num
    extra_numeric.append('age_num')

    train_dict['is_elderly'] = (tr_age_num >= 70).astype(np.float64)
    eval_dict['is_elderly'] = (ev_age_num >= 70).astype(np.float64)
    extra_numeric.append('is_elderly')

    train_dict['age_x_inpatient'] = tr_age_num * tr_inp
    eval_dict['age_x_inpatient'] = ev_age_num * ev_inp
    extra_numeric.append('age_x_inpatient')

    # 6. Medication Dynamics
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    active_med_cols = [c for c in med_names if c in task.categorical_columns]

    tr_active = np.zeros(n_train, dtype=np.float64)
    ev_active = np.zeros(n_eval, dtype=np.float64)
    tr_up = np.zeros(n_train, dtype=np.float64)
    ev_up = np.zeros(n_eval, dtype=np.float64)
    tr_down = np.zeros(n_train, dtype=np.float64)
    ev_down = np.zeros(n_eval, dtype=np.float64)

    for m in active_med_cols:
        tr_m = train[m]
        ev_m = eval_cols[m]
        tr_active += (tr_m != 'No') & (tr_m is not None)
        ev_active += (ev_m != 'No') & (ev_m is not None)
        tr_up += (tr_m == 'Up')
        ev_up += (ev_m == 'Up')
        tr_down += (tr_m == 'Down')
        ev_down += (ev_m == 'Down')

    train_dict['num_active_diabetes_meds'] = tr_active
    eval_dict['num_active_diabetes_meds'] = ev_active
    extra_numeric.append('num_active_diabetes_meds')

    train_dict['num_meds_up'] = tr_up
    eval_dict['num_meds_up'] = ev_up
    extra_numeric.append('num_meds_up')

    train_dict['num_meds_down'] = tr_down
    eval_dict['num_meds_down'] = ev_down
    extra_numeric.append('num_meds_down')

    train_dict['num_meds_changed'] = tr_up + tr_down
    eval_dict['num_meds_changed'] = ev_up + ev_down
    extra_numeric.append('num_meds_changed')

    if 'insulin' in train:
        tr_ins = train['insulin']
        ev_ins = eval_cols['insulin']
        train_dict['insulin_active'] = ((tr_ins != 'No') & (tr_ins is not None)).astype(np.float64)
        eval_dict['insulin_active'] = ((ev_ins != 'No') & (ev_ins is not None)).astype(np.float64)
        extra_numeric.append('insulin_active')

        train_dict['insulin_up'] = (tr_ins == 'Up').astype(np.float64)
        eval_dict['insulin_up'] = (ev_ins == 'Up').astype(np.float64)
        extra_numeric.append('insulin_up')

        train_dict['insulin_down'] = (tr_ins == 'Down').astype(np.float64)
        eval_dict['insulin_down'] = (ev_ins == 'Down').astype(np.float64)
        extra_numeric.append('insulin_down')

        train_dict['insulin_changed'] = ((tr_ins == 'Up') | (tr_ins == 'Down')).astype(np.float64)
        eval_dict['insulin_changed'] = ((ev_ins == 'Up') | (ev_ins == 'Down')).astype(np.float64)
        extra_numeric.append('insulin_changed')

    if 'change' in train:
        train_dict['change_numeric'] = (train['change'] == 'Ch').astype(np.float64)
        eval_dict['change_numeric'] = (eval_cols['change'] == 'Ch').astype(np.float64)
        extra_numeric.append('change_numeric')

    if 'diabetesMed' in train:
        train_dict['diabetesMed_numeric'] = (train['diabetesMed'] == 'Yes').astype(np.float64)
        eval_dict['diabetesMed_numeric'] = (eval_cols['diabetesMed'] == 'Yes').astype(np.float64)
        extra_numeric.append('diabetesMed_numeric')

    train_dict['polypharmacy'] = (tr_med >= 16).astype(np.float64)
    eval_dict['polypharmacy'] = (ev_med >= 16).astype(np.float64)
    extra_numeric.append('polypharmacy')

    # 7. Lab Indicators
    if 'A1Cresult' in train:
        tr_a1c = train['A1Cresult']
        ev_a1c = eval_cols['A1Cresult']
        train_dict['a1c_tested'] = ((tr_a1c != 'None') & (tr_a1c is not None)).astype(np.float64)
        eval_dict['a1c_tested'] = ((ev_a1c != 'None') & (ev_a1c is not None)).astype(np.float64)
        extra_numeric.append('a1c_tested')

        train_dict['a1c_high'] = ((tr_a1c == '>8') | (tr_a1c == '>7')).astype(np.float64)
        eval_dict['a1c_high'] = ((ev_a1c == '>8') | (ev_a1c == '>7')).astype(np.float64)
        extra_numeric.append('a1c_high')

    if 'max_glu_serum' in train:
        tr_glu = train['max_glu_serum']
        ev_glu = eval_cols['max_glu_serum']
        train_dict['glu_tested'] = ((tr_glu != 'None') & (tr_glu is not None)).astype(np.float64)
        eval_dict['glu_tested'] = ((ev_glu != 'None') & (ev_glu is not None)).astype(np.float64)
        extra_numeric.append('glu_tested')

        train_dict['glu_high'] = ((tr_glu == '>200') | (tr_glu == '>300')).astype(np.float64)
        eval_dict['glu_high'] = ((ev_glu == '>200') | (ev_glu == '>300')).astype(np.float64)
        extra_numeric.append('glu_high')

    # 8. Admission & Discharge Indicators
    if 'discharge_disposition_id' in train:
        tr_disp = train['discharge_disposition_id']
        ev_disp = eval_cols['discharge_disposition_id']
        train_dict['disp_home'] = (tr_disp == '1').astype(np.float64)
        eval_dict['disp_home'] = (ev_disp == '1').astype(np.float64)
        extra_numeric.append('disp_home')

        train_dict['disp_snf'] = (tr_disp == '3').astype(np.float64)
        eval_dict['disp_snf'] = (ev_disp == '3').astype(np.float64)
        extra_numeric.append('disp_snf')

        train_dict['disp_home_health'] = (tr_disp == '6').astype(np.float64)
        eval_dict['disp_home_health'] = (ev_disp == '6').astype(np.float64)
        extra_numeric.append('disp_home_health')

        train_dict['disp_ama'] = ((tr_disp == '7') | (tr_disp == '18')).astype(np.float64)
        eval_dict['disp_ama'] = ((ev_disp == '7') | (ev_disp == '18')).astype(np.float64)
        extra_numeric.append('disp_ama')

    if 'admission_source_id' in train:
        train_dict['adm_from_er'] = (train['admission_source_id'] == '7').astype(np.float64)
        eval_dict['adm_from_er'] = (eval_cols['admission_source_id'] == '7').astype(np.float64)
        extra_numeric.append('adm_from_er')

    if 'admission_type_id' in train:
        train_dict['adm_type_emergency'] = (train['admission_type_id'] == '1').astype(np.float64)
        eval_dict['adm_type_emergency'] = (eval_cols['admission_type_id'] == '1').astype(np.float64)
        extra_numeric.append('adm_type_emergency')

    # 9. Missingness
    for miss_col in ('weight', 'medical_specialty', 'payer_code'):
        if miss_col in train:
            train_dict[f'{miss_col}_missing'] = np.array([(x is None) for x in train[miss_col]], dtype=np.float64)
            eval_dict[f'{miss_col}_missing'] = np.array([(x is None) for x in eval_cols[miss_col]], dtype=np.float64)
            extra_numeric.append(f'{miss_col}_missing')

    encoder = TabularEncoder(min_count=30)
    x_train = encoder.fit_transform(
        task, train_dict, extra_numeric=tuple(extra_numeric), extra_categorical=tuple(extra_categorical)
    )
    x_eval = encoder.transform(eval_dict)

    # Ensemble of 3 diverse LightGBM models
    model1 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=180,
        subsample=0.72,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=0.1,
        cat_smooth=50.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p1 = model1.predict_proba(x_eval)[:, 1]

    model2 = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.025,
        num_leaves=18,
        min_child_samples=120,
        subsample=0.68,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=1.0,
        cat_smooth=75.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    model2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p2 = model2.predict_proba(x_eval)[:, 1]

    model3 = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.028,
        num_leaves=45,
        min_child_samples=220,
        subsample=0.78,
        subsample_freq=1,
        colsample_bytree=0.72,
        reg_lambda=0.4,
        cat_smooth=35.0,
        random_state=789,
        n_jobs=1,
        verbose=-1,
    )
    model3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p3 = model3.predict_proba(x_eval)[:, 1]

    preds = (p1 + p2 + p3) / 3.0
    return np.clip(preds, 0.0, 1.0)



# EVOLVE-BLOCK-END
# =====================================================================
