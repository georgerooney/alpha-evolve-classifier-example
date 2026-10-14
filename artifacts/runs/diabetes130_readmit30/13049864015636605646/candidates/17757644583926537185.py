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
    train_copy.update(train_cm)
    eval_cols_copy.update(eval_cm)
    new_numeric_cols.extend(train_cm.keys())

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
    for d in (train_copy, eval_cols_copy):
        d['total_prior_visits'] = d['number_outpatient'] + d['number_emergency'] + d['number_inpatient']
        d['has_prior_inpatient'] = (d['number_inpatient'] > 0).astype(float)
        d['med_by_hospital_days'] = d['num_medications'] / (d['time_in_hospital'] + 1.0)
        d['lab_by_hospital_days'] = d['num_lab_procedures'] / (d['time_in_hospital'] + 1.0)
        d['procedures_by_hospital_days'] = d['num_procedures'] / (d['time_in_hospital'] + 1.0)
    new_numeric_cols.extend([
        'total_prior_visits', 'has_prior_inpatient',
        'med_by_hospital_days', 'lab_by_hospital_days', 'procedures_by_hospital_days'
    ])





    # 5. Medication and Dosage Change Counts
    med_cols = [
        c for c in task.categorical_columns
        if c not in ('change', 'diabetesMed') and any(v in ('Steady', 'Up', 'Down') for v in set(train_copy[c]))
    ]
    for metric, vals in [
        ('num_active_meds', ('Steady', 'Up', 'Down')),
        ('num_up_meds', ('Up',)),
        ('num_down_meds', ('Down',)),
        ('num_changes', ('Up', 'Down')),
    ]:
        train_copy[metric] = sum(np.isin(train_copy[c], vals).astype(float) for c in med_cols)
        eval_cols_copy[metric] = sum(np.isin(eval_cols_copy[c], vals).astype(float) for c in med_cols)
        new_numeric_cols.append(metric)










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

    # Multi-seed Ensemble on full training data
    seeds = [42, 123, 456]
    preds = np.zeros(len(x_eval))

    for seed in seeds:
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=1.0,
            reg_alpha=0.1,
            cat_smooth=50.0,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1] / len(seeds)

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
