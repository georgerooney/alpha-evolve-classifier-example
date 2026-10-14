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

    train_feats = dict(train)
    eval_feats = dict(eval_cols)
    extra_numeric_cols = []
    extra_categorical_cols = []

    # 1. Comorbidities from clinical mapping
    train_cm = icd9_comorbidities(train_feats['diag_1'], train_feats['diag_2'], train_feats['diag_3'])
    eval_cm = icd9_comorbidities(eval_feats['diag_1'], eval_feats['diag_2'], eval_feats['diag_3'])
    for k, v in train_cm.items():
        train_feats[k] = v
        eval_feats[k] = eval_cm[k]
        extra_numeric_cols.append(k)

    # 2. ICD-9 Chapter groups
    for col in ['diag_1', 'diag_2', 'diag_3']:
        grp_col = f"{col}_group"
        train_feats[grp_col] = np.array([icd9_group(x) for x in train_feats[col]], dtype=object)
        eval_feats[grp_col] = np.array([icd9_group(x) for x in eval_feats[col]], dtype=object)
        extra_categorical_cols.append(grp_col)

    # 3. Target encoding for high-cardinality features
    for col in ['diag_1', 'diag_2', 'diag_3', 'medical_specialty', 'discharge_disposition_id']:
        te_col = f"{col}_te"
        tr_te, ev_te = oof_target_encode(train_feats[col], y_train, eval_feats[col], n_folds=5, smoothing=20.0)
        train_feats[te_col] = tr_te
        eval_feats[te_col] = ev_te
        extra_numeric_cols.append(te_col)

    # 4. Total prior utilization and active diabetes medication count
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
    ]
    for df in [train_feats, eval_feats]:
        df['total_visits'] = (
            np.nan_to_num(df['number_outpatient'])
            + np.nan_to_num(df['number_emergency'])
            + np.nan_to_num(df['number_inpatient'])
        )
        active = np.zeros(len(df['diag_1']), dtype=np.float64)
        for m in med_cols:
            if m in df:
                active += np.isin(df[m], ['Steady', 'Up', 'Down'])
        df['num_active_meds'] = active

    extra_numeric_cols.extend(['total_visits', 'num_active_meds'])

    # 5. Fit TabularEncoder
    encoder = TabularEncoder(min_count=20, one_hot=False)
    encoder.fit(task, train_feats, extra_numeric=extra_numeric_cols, extra_categorical=extra_categorical_cols)
    x_train = encoder.transform(train_feats)
    x_eval = encoder.transform(eval_feats)

    # 6. Multi-seed LightGBM bagging for robust variance reduction
    seeds = [42, 100, 2024, 7, 999]
    preds = np.zeros(len(eval_cols['diag_1']), dtype=np.float64)
    for seed in seeds:
        model = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.015,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=0.1,
            reg_alpha=0.02,
            cat_smooth=60.0,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1]

    return preds / len(seeds)


# EVOLVE-BLOCK-END
# =====================================================================
