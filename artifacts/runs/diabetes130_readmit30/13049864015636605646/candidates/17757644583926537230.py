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
    import numpy as np
    import lightgbm as lgb
    from clinical_toolkit import TabularEncoder, icd9_group, icd9_comorbidities, oof_target_encode

    # Make local copies of dicts to add engineered features
    tr = dict(train)
    ev = dict(eval_cols)

    extra_numeric_cols = []
    extra_categorical_cols = []

    # 1. Comorbidities
    cm_train = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for key in cm_train.keys():
        tr[key] = cm_train[key]
        ev[key] = cm_eval[key]
        extra_numeric_cols.append(key)

    # 2. ICD-9 Chapter Groups
    tr['diag_1_group'] = np.array([icd9_group(c) for c in train['diag_1']])
    ev['diag_1_group'] = np.array([icd9_group(c) for c in eval_cols['diag_1']])
    tr['diag_2_group'] = np.array([icd9_group(c) for c in train['diag_2']])
    ev['diag_2_group'] = np.array([icd9_group(c) for c in eval_cols['diag_2']])
    tr['diag_3_group'] = np.array([icd9_group(c) for c in train['diag_3']])
    ev['diag_3_group'] = np.array([icd9_group(c) for c in eval_cols['diag_3']])
    extra_categorical_cols.extend(['diag_1_group', 'diag_2_group', 'diag_3_group'])

    # 3. Utilization Dynamics
    tr['service_utilization'] = train['number_outpatient'] + train['number_emergency'] + train['number_inpatient']
    ev['service_utilization'] = eval_cols['number_outpatient'] + eval_cols['number_emergency'] + eval_cols['number_inpatient']
    extra_numeric_cols.append('service_utilization')

    tr['has_prior_inpatient'] = (train['number_inpatient'] > 0).astype(np.float64)
    ev['has_prior_inpatient'] = (eval_cols['number_inpatient'] > 0).astype(np.float64)
    extra_numeric_cols.append('has_prior_inpatient')

    # 4. Medication Patterns
    med_cols = []
    for col in task.categorical_columns:
        vals = set(train[col])
        vals.discard(None)
        if vals.issubset({'No', 'Steady', 'Up', 'Down'}):
            med_cols.append(col)

    num_active_meds_train = np.zeros(len(y_train), dtype=np.float64)
    num_active_meds_eval = np.zeros(len(eval_cols['diag_1']), dtype=np.float64)
    num_med_changes_train = np.zeros(len(y_train), dtype=np.float64)
    num_med_changes_eval = np.zeros(len(eval_cols['diag_1']), dtype=np.float64)

    for col in med_cols:
        tr_str = np.array([str(x) if x is not None else 'None' for x in train[col]])
        ev_str = np.array([str(x) if x is not None else 'None' for x in eval_cols[col]])

        num_active_meds_train += np.isin(tr_str, ['Steady', 'Up', 'Down']).astype(np.float64)
        num_active_meds_eval += np.isin(ev_str, ['Steady', 'Up', 'Down']).astype(np.float64)

        num_med_changes_train += np.isin(tr_str, ['Up', 'Down']).astype(np.float64)
        num_med_changes_eval += np.isin(ev_str, ['Up', 'Down']).astype(np.float64)

    tr['num_active_meds'] = num_active_meds_train
    ev['num_active_meds'] = num_active_meds_eval
    extra_numeric_cols.append('num_active_meds')

    tr['num_med_changes'] = num_med_changes_train
    ev['num_med_changes'] = num_med_changes_eval
    extra_numeric_cols.append('num_med_changes')

    # 5. Lab Tests & Thresholds
    def is_tested(val):
        return np.array([0.0 if (x is None or str(x) == 'None') else 1.0 for x in val])

    def is_high(val, indicators):
        return np.array([1.0 if (x is not None and str(x) in indicators) else 0.0 for x in val])

    tr['a1c_tested'] = is_tested(train['A1Cresult'])
    ev['a1c_tested'] = is_tested(eval_cols['A1Cresult'])
    extra_numeric_cols.append('a1c_tested')

    tr['glucose_tested'] = is_tested(train['max_glu_serum'])
    ev['glucose_tested'] = is_tested(eval_cols['max_glu_serum'])
    extra_numeric_cols.append('glucose_tested')

    tr['a1c_high'] = is_high(train['A1Cresult'], ['>7', '>8'])
    ev['a1c_high'] = is_high(eval_cols['A1Cresult'], ['>7', '>8'])
    extra_numeric_cols.append('a1c_high')

    tr['glucose_high'] = is_high(train['max_glu_serum'], ['>200', '>300'])
    ev['glucose_high'] = is_high(eval_cols['max_glu_serum'], ['>200', '>300'])
    extra_numeric_cols.append('glucose_high')

    # 6. Target Encodings (Out-of-Fold)
    te_cols = ['diag_1', 'diag_2', 'diag_3', 'medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'admission_type_id', 'payer_code']
    for col in te_cols:
        train_enc, eval_enc = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=30.0, seed=42)
        col_name = f'{col}_te'
        tr[col_name] = train_enc
        ev[col_name] = eval_enc
        extra_numeric_cols.append(col_name)

    # Transform all inputs
    encoder = TabularEncoder(min_count=30)
    x_train = encoder.fit_transform(
        task,
        tr,
        extra_numeric=tuple(extra_numeric_cols),
        extra_categorical=tuple(extra_categorical_cols)
    )
    x_eval = encoder.transform(ev)

    # 7. Model Ensembling
    models = [
        lgb.LGBMClassifier(
            n_estimators=160,
            learning_rate=0.015,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=0.05,
            cat_smooth=50.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=180,
            learning_rate=0.012,
            num_leaves=24,
            min_child_samples=200,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=0.5,
            reg_alpha=0.1,
            cat_smooth=100.0,
            random_state=100,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=140,
            learning_rate=0.02,
            num_leaves=40,
            min_child_samples=100,
            subsample=0.65,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=0.01,
            cat_smooth=30.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        )
    ]

    preds = []
    for model in models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds.append(model.predict_proba(x_eval)[:, 1])

    return np.mean(preds, axis=0)


# EVOLVE-BLOCK-END
# =====================================================================
