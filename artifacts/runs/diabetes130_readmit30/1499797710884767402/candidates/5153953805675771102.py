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

Strategy only: an untuned single LightGBM on the raw columns. Encoding mechanics live in `clinical_toolkit`.
"""

from __future__ import annotations

from typing import Dict

import lightgbm as lgb
import numpy as np

from clinical_models import TabularTask
from clinical_toolkit import TabularEncoder, icd9_group  # noqa: F401  (icd9_group available for feature ideas)


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
    from sklearn.ensemble import HistGradientBoostingClassifier

    # Dynamically extract all active medication columns to avoid hardcoding
    non_med_cols = {
        'race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty',
        'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id'
    }
    med_cols = [col for col in task.categorical_columns if col not in non_med_cols]

    extra_num = ['total_visits', 'med_per_day', 'active_meds', 'num_med_changes']
    extra_cat = ['diag_1_grp', 'diag_2_grp', 'diag_3_grp']

    # In-place feature construction to keep the codebase perfectly minimal
    for d in [train, eval_cols]:
        # Prior utilization and medication density
        d['total_visits'] = d['number_outpatient'] + d['number_emergency'] + d['number_inpatient']
        d['med_per_day'] = d['num_medications'] / (d['time_in_hospital'] + 1.0)
        
        # Aggregate medication metrics
        active = np.zeros(len(d['gender']), dtype=float)
        changes = np.zeros(len(d['gender']), dtype=float)
        for col in med_cols:
            if col in d:
                val = d[col]
                active += (val != 'No').astype(float)
                changes += np.isin(val, ['Up', 'Down']).astype(float)
        d['active_meds'] = active
        d['num_med_changes'] = changes

        # Hierarchical grouping of clinical ICD-9 diagnoses
        for col in ['diag_1', 'diag_2', 'diag_3']:
            d[f'{col}_grp'] = np.array([icd9_group(x) for x in d[col]], dtype=object)

    # Encode with slightly lower min_count to retain key low-frequency clinical signals
    encoder = TabularEncoder(min_count=10)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_cols)

    # Model 1: LightGBM Classifier (with tuned leaf-wise parameters)
    model_lgb = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model_lgb.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p_lgb = model_lgb.predict_proba(x_eval)[:, 1]

    # Model 2: HistGradientBoosting Classifier (extremely robust depth-wise tree builder)
    model_hgb = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.04,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=3.0,
        categorical_features=encoder.categorical_indices,
        random_state=42,
    )
    model_hgb.fit(x_train, y_train)
    p_hgb = model_hgb.predict_proba(x_eval)[:, 1]

    # Weighted blend of the dual architectures
    return 0.55 * p_lgb + 0.45 * p_hgb


# EVOLVE-BLOCK-END
# =====================================================================
