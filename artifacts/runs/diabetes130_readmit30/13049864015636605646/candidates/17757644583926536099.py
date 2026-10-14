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
    train_cols = dict(train)
    eval_cols_dict = dict(eval_cols)

    extra_numeric = []
    extra_categorical = []

    # 1. Comorbidity features
    cm_train = icd9_comorbidities(train_cols['diag_1'], train_cols['diag_2'], train_cols['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols_dict['diag_1'], eval_cols_dict['diag_2'], eval_cols_dict['diag_3'])
    for k in cm_train:
        train_cols[k] = cm_train[k]
        eval_cols_dict[k] = cm_eval[k]
        extra_numeric.append(k)

    # 2. ICD-9 Chapter Groups
    for col in ('diag_1', 'diag_2', 'diag_3'):
        g_name = f'{col}_group'
        train_cols[g_name] = np.array([icd9_group(x) for x in train_cols[col]], dtype=object)
        eval_cols_dict[g_name] = np.array([icd9_group(x) for x in eval_cols_dict[col]], dtype=object)
        extra_categorical.append(g_name)

    # 3. High-cardinality Target Encodings
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'medical_specialty', 'diag_1_group'
    ]
    for col in te_cols:
        if col in train_cols and col in eval_cols_dict:
            te_name = f'{col}_te'
            tr_te, ev_te = oof_target_encode(
                train_cols[col], y_train, eval_cols_dict[col],
                n_folds=5, smoothing=20.0, seed=42
            )
            train_cols[te_name] = tr_te.astype(np.float64)
            eval_cols_dict[te_name] = ev_te.astype(np.float64)
            extra_numeric.append(te_name)

    # Clinical interaction target encoding: Primary disease group x Discharge disposition
    inter_tr = np.array([f"{g}_{d}" for g, d in zip(train_cols['diag_1_group'], train_cols['discharge_disposition_id'])], dtype=object)
    inter_ev = np.array([f"{g}_{d}" for g, d in zip(eval_cols_dict['diag_1_group'], eval_cols_dict['discharge_disposition_id'])], dtype=object)
    tr_ite, ev_ite = oof_target_encode(inter_tr, y_train, inter_ev, n_folds=5, smoothing=25.0, seed=42)
    train_cols['diag_disposition_te'] = tr_ite.astype(np.float64)
    eval_cols_dict['diag_disposition_te'] = ev_ite.astype(np.float64)
    extra_numeric.append('diag_disposition_te')

    # 4. Utilization & Intensity features
    total_vis_tr = train_cols['number_outpatient'] + train_cols['number_emergency'] + train_cols['number_inpatient']
    total_vis_ev = eval_cols_dict['number_outpatient'] + eval_cols_dict['number_emergency'] + eval_cols_dict['number_inpatient']

    train_cols['total_prior_visits'] = total_vis_tr.astype(np.float64)
    eval_cols_dict['total_prior_visits'] = total_vis_ev.astype(np.float64)
    extra_numeric.append('total_prior_visits')

    train_cols['inpatient_ratio'] = (train_cols['number_inpatient'] / (total_vis_tr + 1.0)).astype(np.float64)
    eval_cols_dict['inpatient_ratio'] = (eval_cols_dict['number_inpatient'] / (total_vis_ev + 1.0)).astype(np.float64)
    extra_numeric.append('inpatient_ratio')

    train_cols['has_inpatient'] = (train_cols['number_inpatient'] > 0).astype(np.float64)
    eval_cols_dict['has_inpatient'] = (eval_cols_dict['number_inpatient'] > 0).astype(np.float64)
    extra_numeric.append('has_inpatient')

    train_cols['has_emergency'] = (train_cols['number_emergency'] > 0).astype(np.float64)
    eval_cols_dict['has_emergency'] = (eval_cols_dict['number_emergency'] > 0).astype(np.float64)
    extra_numeric.append('has_emergency')

    los_tr = np.maximum(train_cols['time_in_hospital'], 1.0)
    los_ev = np.maximum(eval_cols_dict['time_in_hospital'], 1.0)

    train_cols['med_per_day'] = (train_cols['num_medications'] / los_tr).astype(np.float64)
    eval_cols_dict['med_per_day'] = (eval_cols_dict['num_medications'] / los_ev).astype(np.float64)
    extra_numeric.append('med_per_day')

    train_cols['lab_per_day'] = (train_cols['num_lab_procedures'] / los_tr).astype(np.float64)
    eval_cols_dict['lab_per_day'] = (eval_cols_dict['num_lab_procedures'] / los_ev).astype(np.float64)
    extra_numeric.append('lab_per_day')

    train_cols['proc_per_day'] = (train_cols['num_procedures'] / los_tr).astype(np.float64)
    eval_cols_dict['proc_per_day'] = (eval_cols_dict['num_procedures'] / los_ev).astype(np.float64)
    extra_numeric.append('proc_per_day')

    train_cols['diag_per_day'] = (train_cols['number_diagnoses'] / los_tr).astype(np.float64)
    eval_cols_dict['diag_per_day'] = (eval_cols_dict['number_diagnoses'] / los_ev).astype(np.float64)
    extra_numeric.append('diag_per_day')

    train_cols['total_procedures'] = (train_cols['num_lab_procedures'] + train_cols['num_procedures']).astype(np.float64)
    eval_cols_dict['total_procedures'] = (eval_cols_dict['num_lab_procedures'] + eval_cols_dict['num_procedures']).astype(np.float64)
    extra_numeric.append('total_procedures')

    # 5. Age numeric midpoint
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    train_cols['age_num'] = np.array([age_map.get(a, 65.0) for a in train_cols['age']], dtype=np.float64)
    eval_cols_dict['age_num'] = np.array([age_map.get(a, 65.0) for a in eval_cols_dict['age']], dtype=np.float64)
    extra_numeric.append('age_num')

    # 6. Medication patterns & changes
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    present_meds = [m for m in med_names if m in train_cols and m in eval_cols_dict]

    n_tr = len(y_train)
    n_ev = len(eval_cols_dict['age'])
    act_tr, act_ev = np.zeros(n_tr, dtype=np.float64), np.zeros(n_ev, dtype=np.float64)
    chg_tr, chg_ev = np.zeros(n_tr, dtype=np.float64), np.zeros(n_ev, dtype=np.float64)
    up_tr, up_ev = np.zeros(n_tr, dtype=np.float64), np.zeros(n_ev, dtype=np.float64)
    dn_tr, dn_ev = np.zeros(n_tr, dtype=np.float64), np.zeros(n_ev, dtype=np.float64)

    for m in present_meds:
        c_tr = train_cols[m]
        c_ev = eval_cols_dict[m]
        act_tr += ((c_tr == 'Steady') | (c_tr == 'Up') | (c_tr == 'Down')).astype(np.float64)
        act_ev += ((c_ev == 'Steady') | (c_ev == 'Up') | (c_ev == 'Down')).astype(np.float64)
        chg_tr += ((c_tr == 'Up') | (c_tr == 'Down')).astype(np.float64)
        chg_ev += ((c_ev == 'Up') | (c_ev == 'Down')).astype(np.float64)
        up_tr += (c_tr == 'Up').astype(np.float64)
        up_ev += (c_ev == 'Up').astype(np.float64)
        dn_tr += (c_tr == 'Down').astype(np.float64)
        dn_ev += (c_ev == 'Down').astype(np.float64)

    train_cols['med_count_active'] = act_tr
    eval_cols_dict['med_count_active'] = act_ev
    extra_numeric.append('med_count_active')

    train_cols['med_count_changes'] = chg_tr
    eval_cols_dict['med_count_changes'] = chg_ev
    extra_numeric.append('med_count_changes')

    train_cols['med_count_up'] = up_tr
    eval_cols_dict['med_count_up'] = up_ev
    extra_numeric.append('med_count_up')

    train_cols['med_count_down'] = dn_tr
    eval_cols_dict['med_count_down'] = dn_ev
    extra_numeric.append('med_count_down')

    if 'insulin' in train_cols:
        ins_tr = train_cols['insulin']
        ins_ev = eval_cols_dict['insulin']
        train_cols['insulin_up'] = (ins_tr == 'Up').astype(np.float64)
        eval_cols_dict['insulin_up'] = (ins_ev == 'Up').astype(np.float64)
        extra_numeric.append('insulin_up')

        train_cols['insulin_down'] = (ins_tr == 'Down').astype(np.float64)
        eval_cols_dict['insulin_down'] = (ins_ev == 'Down').astype(np.float64)
        extra_numeric.append('insulin_down')

        train_cols['insulin_steady'] = (ins_tr == 'Steady').astype(np.float64)
        eval_cols_dict['insulin_steady'] = (ins_ev == 'Steady').astype(np.float64)
        extra_numeric.append('insulin_steady')

    if 'change' in train_cols:
        train_cols['had_change'] = (train_cols['change'] == 'Ch').astype(np.float64)
        eval_cols_dict['had_change'] = (eval_cols_dict['change'] == 'Ch').astype(np.float64)
        extra_numeric.append('had_change')

    if 'diabetesMed' in train_cols:
        train_cols['on_diabetes_med'] = (train_cols['diabetesMed'] == 'Yes').astype(np.float64)
        eval_cols_dict['on_diabetes_med'] = (eval_cols_dict['diabetesMed'] == 'Yes').astype(np.float64)
        extra_numeric.append('on_diabetes_med')

    # 7. Lab testing indicators
    if 'A1Cresult' in train_cols:
        a1c_tr = train_cols['A1Cresult']
        a1c_ev = eval_cols_dict['A1Cresult']
        train_cols['a1c_tested'] = (a1c_tr != 'None').astype(np.float64)
        eval_cols_dict['a1c_tested'] = (a1c_ev != 'None').astype(np.float64)
        extra_numeric.append('a1c_tested')

        train_cols['a1c_high'] = ((a1c_tr == '>8') | (a1c_tr == '>7')).astype(np.float64)
        eval_cols_dict['a1c_high'] = ((a1c_ev == '>8') | (a1c_ev == '>7')).astype(np.float64)
        extra_numeric.append('a1c_high')

    if 'max_glu_serum' in train_cols:
        glu_tr = train_cols['max_glu_serum']
        glu_ev = eval_cols_dict['max_glu_serum']
        train_cols['glu_tested'] = (glu_tr != 'None').astype(np.float64)
        eval_cols_dict['glu_tested'] = (glu_ev != 'None').astype(np.float64)
        extra_numeric.append('glu_tested')

    # Fit encoder
    encoder = TabularEncoder(min_count=50)
    encoder.fit(
        task, train_cols,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical)
    )
    x_train = encoder.transform(train_cols)
    x_eval = encoder.transform(eval_cols_dict)

    # Diverse LightGBM ensemble
    models = [
        lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.025,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=1.0,
            reg_alpha=0.1,
            cat_smooth=50.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.020,
            num_leaves=20,
            min_child_samples=100,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=3.0,
            reg_alpha=0.3,
            cat_smooth=35.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.028,
            num_leaves=45,
            min_child_samples=200,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=0.5,
            reg_alpha=0.05,
            cat_smooth=65.0,
            random_state=456,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.025,
            num_leaves=26,
            min_child_samples=120,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=2.0,
            reg_alpha=0.2,
            cat_smooth=45.0,
            random_state=789,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = np.zeros(len(x_eval), dtype=np.float64)
    for model in models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1]
    preds /= len(models)

    preds = np.clip(preds, 0.0, 1.0)
    if np.isnan(preds).any():
        preds = np.nan_to_num(preds, nan=0.114)

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
