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
    import xgboost as xgb

    train_feats = dict(train)
    eval_feats = dict(eval_cols)
    extra_numeric_cols = []
    extra_categorical_cols = []

    # 1. Comorbidities from clinical mapping
    train_cm = icd9_comorbidities(train_feats['diag_1'], train_feats['diag_2'], train_feats['diag_3'])
    eval_cm = icd9_comorbidities(eval_feats['diag_1'], eval_feats['diag_2'], eval_feats['diag_3'])
    for k in train_cm.keys():
        train_feats[k] = train_cm[k]
        eval_feats[k] = eval_cm[k]
        extra_numeric_cols.append(k)

    # 2. ICD-9 Chapter groups
    for col in ['diag_1', 'diag_2', 'diag_3']:
        new_col = f"{col}_group"
        train_feats[new_col] = np.array([icd9_group(x) for x in train_feats[col]], dtype=object)
        eval_feats[new_col] = np.array([icd9_group(x) for x in eval_feats[col]], dtype=object)
        extra_categorical_cols.append(new_col)

    # 3. Out-of-fold target encoding for primary high-cardinality categoricals
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'medical_specialty', 'discharge_disposition_id',
        'admission_source_id', 'admission_type_id', 'payer_code',
    ]
    for col in te_cols:
        new_col = f"{col}_te"
        tr_te, ev_te = oof_target_encode(train_feats[col], y_train, eval_feats[col], n_folds=5, smoothing=20.0)
        train_feats[new_col] = tr_te
        eval_feats[new_col] = ev_te
        extra_numeric_cols.append(new_col)

    # 4. Continuous age representation
    def parse_age(age_str):
        if not isinstance(age_str, str):
            return np.nan
        s = age_str.replace('[', '').replace(')', '').replace(']', '')
        parts = s.split('-')
        if len(parts) == 2:
            try:
                return (float(parts[0]) + float(parts[1])) / 2.0
            except ValueError:
                pass
        return np.nan

    train_feats['age_numeric'] = np.array([parse_age(x) for x in train_feats['age']], dtype=float)
    eval_feats['age_numeric'] = np.array([parse_age(x) for x in eval_feats['age']], dtype=float)
    extra_numeric_cols.append('age_numeric')

    # 5. Core utilization and medication patterns
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
    ]

    for df in [train_feats, eval_feats]:
        outpatient = np.nan_to_num(df['number_outpatient'])
        emergency = np.nan_to_num(df['number_emergency'])
        inpatient = np.nan_to_num(df['number_inpatient'])

        df['total_utilization'] = outpatient + emergency + inpatient
        df['utilization_weighted'] = outpatient + 2.0 * emergency + 3.0 * inpatient
        df['has_prior_inpatient'] = (inpatient > 0).astype(float)
        df['has_prior_emergency'] = (emergency > 0).astype(float)

        active_meds = np.zeros(len(outpatient))
        for m in med_cols:
            if m in df:
                is_active = np.isin(df[m], ['Steady', 'Up', 'Down'])
                active_meds += is_active.astype(float)
        df['num_active_meds'] = active_meds

        df['has_change'] = (df['change'] == 'Ch').astype(float)
        df['has_diabetes_med'] = (df['diabetesMed'] == 'Yes').astype(float)
        df['insulin_up'] = (df['insulin'] == 'Up').astype(float)
        df['insulin_down'] = (df['insulin'] == 'Down').astype(float)
        df['insulin_steady'] = (df['insulin'] == 'Steady').astype(float)

    extra_numeric_cols.extend([
        'total_utilization', 'utilization_weighted', 'has_prior_inpatient', 'has_prior_emergency',
        'num_active_meds', 'has_change', 'has_diabetes_med', 'insulin_up', 'insulin_down', 'insulin_steady',
    ])

    # 6. Fit TabularEncoder
    extra_numeric_cols = list(dict.fromkeys(extra_numeric_cols))
    extra_categorical_cols = list(dict.fromkeys(extra_categorical_cols))

    encoder = TabularEncoder(min_count=50, one_hot=False)
    encoder.fit(task, train_feats, extra_numeric=extra_numeric_cols, extra_categorical=extra_categorical_cols)
    x_train = encoder.transform(train_feats)
    x_eval = encoder.transform(eval_feats)

    # 7. Heterogeneous model ensemble
    model1 = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.015,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=0.05,
        reg_alpha=0.01,
        cat_smooth=64.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    model2 = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.012,
        num_leaves=45,
        min_child_samples=250,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=0.1,
        reg_alpha=0.05,
        cat_smooth=80.0,
        random_state=2023,
        n_jobs=1,
        verbose=-1,
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=130,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=15,
        subsample=0.75,
        colsample_bytree=0.6,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )

    # 8. Train models and compute weighted blend
    model1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    preds1 = model1.predict_proba(x_eval)[:, 1]

    model2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    preds2 = model2.predict_proba(x_eval)[:, 1]

    xgb_model.fit(x_train, y_train)
    preds3 = xgb_model.predict_proba(x_eval)[:, 1]

    return 0.45 * preds1 + 0.40 * preds2 + 0.15 * preds3


# EVOLVE-BLOCK-END
# =====================================================================
