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

    # 1. Charlson comorbidity flags and counts from ICD-9 diagnosis codes
    cm_train = icd9_comorbidities(train_dict['diag_1'], train_dict['diag_2'], train_dict['diag_3'])
    cm_eval = icd9_comorbidities(eval_dict['diag_1'], eval_dict['diag_2'], eval_dict['diag_3'])
    for k in cm_train:
        train_dict[k] = cm_train[k]
        eval_dict[k] = cm_eval[k]
        extra_numeric.append(k)

    # Key clinical interaction risks (cardiorenal and multi-organ burden)
    train_dict['cm_cardiorenal'] = cm_train['cm_chf'] * cm_train['cm_renal']
    eval_dict['cm_cardiorenal'] = cm_eval['cm_chf'] * cm_eval['cm_renal']
    extra_numeric.append('cm_cardiorenal')

    train_dict['cm_high_risk'] = cm_train['cm_chf'] + cm_train['cm_renal'] + cm_train['cm_pulmonary']
    eval_dict['cm_high_risk'] = cm_eval['cm_chf'] + cm_eval['cm_renal'] + cm_eval['cm_pulmonary']
    extra_numeric.append('cm_high_risk')

    # 2. ICD-9 Chapter Groups
    for diag in ('diag_1', 'diag_2', 'diag_3'):
        grp_col = f'{diag}_group'
        train_dict[grp_col] = np.array([icd9_group(x) for x in train_dict[diag]], dtype=object)
        eval_dict[grp_col] = np.array([icd9_group(x) for x in eval_dict[diag]], dtype=object)
        extra_categorical.append(grp_col)

    train_dict['diag_1_2_group'] = np.array(
        [f'{a}_{b}' for a, b in zip(train_dict['diag_1_group'], train_dict['diag_2_group'])],
        dtype=object,
    )
    eval_dict['diag_1_2_group'] = np.array(
        [f'{a}_{b}' for a, b in zip(eval_dict['diag_1_group'], eval_dict['diag_2_group'])],
        dtype=object,
    )
    extra_categorical.append('diag_1_2_group')

    # 3. Out-Of-Fold Target Encoding of High-Cardinality Categoricals
    te_cols = [
        'diag_1',
        'diag_2',
        'diag_3',
        'discharge_disposition_id',
        'admission_source_id',
        'admission_type_id',
        'medical_specialty',
        'payer_code',
        'diag_1_group',
    ]
    for col in te_cols:
        if col in train_dict:
            tr_te, ev_te = oof_target_encode(
                train_dict[col], y_train, eval_dict[col], n_folds=5, smoothing=20.0, seed=42
            )
            col_te = f'{col}_te'
            train_dict[col_te] = tr_te
            eval_dict[col_te] = ev_te
            extra_numeric.append(col_te)

    # Diagnosis risk composites
    train_dict['diag_te_sum'] = train_dict['diag_1_te'] + train_dict['diag_2_te'] + train_dict['diag_3_te']
    eval_dict['diag_te_sum'] = eval_dict['diag_1_te'] + eval_dict['diag_2_te'] + eval_dict['diag_3_te']
    extra_numeric.append('diag_te_sum')

    train_dict['diag_te_max'] = np.maximum(
        np.maximum(train_dict['diag_1_te'], train_dict['diag_2_te']), train_dict['diag_3_te']
    )
    eval_dict['diag_te_max'] = np.maximum(
        np.maximum(eval_dict['diag_1_te'], eval_dict['diag_2_te']), eval_dict['diag_3_te']
    )
    extra_numeric.append('diag_te_max')

    # 4. Numeric Age & Demographics
    age_map = {
        '[0-10)': 5.0,
        '[10-20)': 15.0,
        '[20-30)': 25.0,
        '[30-40)': 35.0,
        '[40-50)': 45.0,
        '[50-60)': 55.0,
        '[60-70)': 65.0,
        '[70-80)': 75.0,
        '[80-90)': 85.0,
        '[90-100)': 95.0,
    }
    for d in (train_dict, eval_dict):
        d['age_num'] = np.array([age_map.get(x, np.nan) for x in d['age']], dtype=np.float64)
        d['is_elderly'] = (d['age_num'] >= 65.0).astype(np.float64)
        d['weight_known'] = np.array([0.0 if x is None else 1.0 for x in d['weight']], dtype=np.float64)
        d['specialty_known'] = np.array([0.0 if x is None else 1.0 for x in d['medical_specialty']], dtype=np.float64)
    extra_numeric.extend(['age_num', 'is_elderly', 'weight_known', 'specialty_known'])

    # 5. Prior Utilisation Intensity
    for d in (train_dict, eval_dict):
        inp = np.nan_to_num(d['number_inpatient'], nan=0.0)
        emg = np.nan_to_num(d['number_emergency'], nan=0.0)
        out = np.nan_to_num(d['number_outpatient'], nan=0.0)
        tot = inp + emg + out

        d['total_prior_visits'] = tot
        d['has_prior_inpatient'] = (inp > 0).astype(np.float64)
        d['has_prior_emergency'] = (emg > 0).astype(np.float64)
        d['has_prior_outpatient'] = (out > 0).astype(np.float64)
        d['has_any_prior'] = (tot > 0).astype(np.float64)
        d['inpatient_ratio'] = inp / (tot + 1.0)
        d['emergency_ratio'] = emg / (tot + 1.0)
        d['inpatient_x_stay'] = inp * d['time_in_hospital']

    extra_numeric.extend([
        'total_prior_visits',
        'has_prior_inpatient',
        'has_prior_emergency',
        'has_prior_outpatient',
        'has_any_prior',
        'inpatient_ratio',
        'emergency_ratio',
        'inpatient_x_stay',
    ])

    # 6. Encounter Intensity Rates
    for d in (train_dict, eval_dict):
        stay = np.maximum(np.nan_to_num(d['time_in_hospital'], nan=1.0), 1.0)
        d['labs_per_day'] = d['num_lab_procedures'] / stay
        d['procs_per_day'] = d['num_procedures'] / stay
        d['meds_per_day'] = d['num_medications'] / stay
        d['diag_per_day'] = d['number_diagnoses'] / stay
        d['total_procedures'] = d['num_procedures'] + d['num_lab_procedures']
        d['total_clinical_events'] = d['num_procedures'] + d['num_lab_procedures'] + d['num_medications']
        d['meds_x_diags'] = d['num_medications'] * d['number_diagnoses']

    extra_numeric.extend([
        'labs_per_day',
        'procs_per_day',
        'meds_per_day',
        'diag_per_day',
        'total_procedures',
        'total_clinical_events',
        'meds_x_diags',
    ])

    # 7. Medication Changes and Insulin Patterns
    all_meds = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone',
    ]
    med_cols = [c for c in all_meds if c in train_dict]

    for d in (train_dict, eval_dict):
        med_mat = np.column_stack([d[c] for c in med_cols])
        d['med_active_count'] = np.isin(med_mat, ['Steady', 'Up', 'Down']).sum(axis=1).astype(np.float64)
        d['med_up_count'] = (med_mat == 'Up').sum(axis=1).astype(np.float64)
        d['med_down_count'] = (med_mat == 'Down').sum(axis=1).astype(np.float64)
        d['med_change_count'] = d['med_up_count'] + d['med_down_count']

        ins = d['insulin']
        d['insulin_treated'] = np.isin(ins, ['Steady', 'Up', 'Down']).astype(np.float64)
        d['insulin_up'] = (ins == 'Up').astype(np.float64)
        d['insulin_down'] = (ins == 'Down').astype(np.float64)
        d['insulin_steady'] = (ins == 'Steady').astype(np.float64)
        d['insulin_change'] = np.isin(ins, ['Up', 'Down']).astype(np.float64)

        d['has_change_flag'] = (d['change'] == 'Ch').astype(np.float64)
        d['on_diabetes_med'] = (d['diabetesMed'] == 'Yes').astype(np.float64)

    extra_numeric.extend([
        'med_active_count',
        'med_up_count',
        'med_down_count',
        'med_change_count',
        'insulin_treated',
        'insulin_up',
        'insulin_down',
        'insulin_steady',
        'insulin_change',
        'has_change_flag',
        'on_diabetes_med',
    ])

    # 8. Lab and Discharge Status Indicators
    for d in (train_dict, eval_dict):
        a1c = d['A1Cresult']
        glu = d['max_glu_serum']
        d['a1c_tested'] = (a1c != 'None').astype(np.float64)
        d['a1c_high'] = np.isin(a1c, ['>7', '>8']).astype(np.float64)
        d['glu_tested'] = (glu != 'None').astype(np.float64)
        d['glu_high'] = np.isin(glu, ['>200', '>300']).astype(np.float64)

        disp = d['discharge_disposition_id']
        adm_src = d['admission_source_id']
        adm_typ = d['admission_type_id']
        d['discharged_home'] = (disp == '1').astype(np.float64)
        d['adm_emergency'] = ((adm_typ == '1') | (adm_src == '7')).astype(np.float64)

    extra_numeric.extend([
        'a1c_tested',
        'a1c_high',
        'glu_tested',
        'glu_high',
        'discharged_home',
        'adm_emergency',
    ])

    # Fit TabularEncoder with tuned frequency threshold
    encoder = TabularEncoder(min_count=30)
    x_train = encoder.fit_transform(
        task, train_dict, extra_numeric=extra_numeric, extra_categorical=extra_categorical
    )
    x_eval = encoder.transform(eval_dict)
    cat_idx = encoder.categorical_indices

    # 9. Diverse LightGBM Models Ensemble
    m1 = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.020,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=1.0,
        reg_alpha=0.1,
        cat_smooth=40.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m2 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.022,
        num_leaves=20,
        min_child_samples=100,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_lambda=3.0,
        reg_alpha=0.5,
        cat_smooth=50.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m3 = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.018,
        num_leaves=45,
        min_child_samples=200,
        subsample=0.65,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_lambda=0.5,
        reg_alpha=0.05,
        cat_smooth=60.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )

    m1.fit(x_train, y_train, categorical_feature=cat_idx)
    m2.fit(x_train, y_train, categorical_feature=cat_idx)
    m3.fit(x_train, y_train, categorical_feature=cat_idx)

    p1 = m1.predict_proba(x_eval)[:, 1]
    p2 = m2.predict_proba(x_eval)[:, 1]
    p3 = m3.predict_proba(x_eval)[:, 1]

    return (p1 + p2 + p3) / 3.0


# EVOLVE-BLOCK-END
# =====================================================================
