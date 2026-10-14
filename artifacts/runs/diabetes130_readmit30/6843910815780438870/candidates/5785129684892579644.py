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
    from sklearn.model_selection import StratifiedKFold

    train_new = train.copy()
    eval_new = eval_cols.copy()

    # 1. Comorbidities
    cm_train = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    extra_numeric = list(cm_train.keys())
    for k in cm_train.keys():
        train_new[k] = cm_train[k]
        eval_new[k] = cm_eval[k]

    # 2. ICD9 Groups
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    extra_categorical = []
    for col in diag_cols:
        group_col = col + "_group"
        train_new[group_col] = np.array([icd9_group(c) for c in train[col]], dtype=object)
        eval_new[group_col] = np.array([icd9_group(c) for c in eval_cols[col]], dtype=object)
        extra_categorical.append(group_col)

    # 3. Target Encodings
    te_cols = ['diag_1', 'diag_2', 'diag_3', 'medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'payer_code']
    for col in te_cols:
        te_name = col + "_te"
        train_te, eval_te = oof_target_encode(
            train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42
        )
        train_new[te_name] = train_te
        eval_new[te_name] = eval_te
        extra_numeric.append(te_name)

    # 4. Medication Features
    med_cols = []
    for col in task.categorical_columns:
        unique_set = set(train[col])
        if 'Steady' in unique_set or 'Up' in unique_set or 'Down' in unique_set:
            med_cols.append(col)

    for d in [train_new, eval_new]:
        col_length = len(d['age'])
        active_count = np.zeros(col_length, dtype=np.float64)
        up_count = np.zeros(col_length, dtype=np.float64)
        down_count = np.zeros(col_length, dtype=np.float64)
        for col in med_cols:
            col_vals = d[col]
            is_steady = (col_vals == 'Steady')
            is_up = (col_vals == 'Up')
            is_down = (col_vals == 'Down')
            active_count += (is_steady | is_up | is_down)
            up_count += is_up
            down_count += is_down
        d['num_meds_active'] = active_count
        d['num_meds_up'] = up_count
        d['num_meds_down'] = down_count
        d['num_meds_changed'] = up_count + down_count

    med_features = ['num_meds_active', 'num_meds_up', 'num_meds_down', 'num_meds_changed']
    extra_numeric.extend(med_features)

    # 5. Additional interactions / utilization features
    for d in [train_new, eval_new]:
        hosp_days = d['time_in_hospital'].astype(np.float64) + 0.1
        d['lab_per_day'] = d['num_lab_procedures'].astype(np.float64) / hosp_days
        d['med_per_day'] = d['num_medications'].astype(np.float64) / hosp_days
        d['proc_per_day'] = d['num_procedures'].astype(np.float64) / hosp_days
        d['total_visits'] = d['number_outpatient'].astype(np.float64) + d['number_emergency'].astype(np.float64) + d['number_inpatient'].astype(np.float64)
        d['has_inpatient'] = (d['number_inpatient'].astype(np.float64) > 0).astype(np.float64)
        d['severity_index'] = d['num_lab_procedures'].astype(np.float64) + d['num_procedures'].astype(np.float64) + d['num_medications'].astype(np.float64)

    util_features = ['lab_per_day', 'med_per_day', 'proc_per_day', 'total_visits', 'has_inpatient', 'severity_index']
    extra_numeric.extend(util_features)

    # Encode tabular data
    encoder = TabularEncoder(min_count=50)
    x_train = encoder.fit_transform(task, train_new, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_eval = encoder.transform(eval_new)

    # Stratified K-Fold CV Ensemble
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_preds = np.zeros(len(eval_cols['age']))

    params1 = {
        'n_estimators': 150,
        'learning_rate': 0.015,
        'num_leaves': 31,
        'min_child_samples': 200,
        'subsample': 0.7,
        'subsample_freq': 1,
        'colsample_bytree': 0.6,
        'reg_lambda': 0.1,
        'reg_alpha': 0.1,
        'cat_smooth': 50.0,
        'random_state': 42,
        'n_jobs': 1,
        'verbose': -1,
    }

    params2 = {
        'n_estimators': 200,
        'learning_rate': 0.012,
        'num_leaves': 20,
        'min_child_samples': 150,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.7,
        'reg_lambda': 1.0,
        'reg_alpha': 0.5,
        'cat_smooth': 100.0,
        'random_state': 1337,
        'n_jobs': 1,
        'verbose': -1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_tr, y_tr = x_train[train_idx], y_train[train_idx]
        
        # Fit Model 1
        model1 = lgb.LGBMClassifier(**params1)
        model1.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        eval_preds += model1.predict_proba(x_eval)[:, 1] / 10.0
        
        # Fit Model 2
        model2 = lgb.LGBMClassifier(**params2)
        model2.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        eval_preds += model2.predict_proba(x_eval)[:, 1] / 10.0

    return eval_preds


# EVOLVE-BLOCK-END
# =====================================================================
