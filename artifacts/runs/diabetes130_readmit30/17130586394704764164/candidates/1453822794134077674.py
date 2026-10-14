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
    from sklearn.model_selection import StratifiedKFold
    
    # 1. Initialize lists for new features
    extra_numeric = []
    extra_categorical = []
    
    # Make shallow copies to avoid modifying caller's dict keys
    train = dict(train)
    eval_cols = dict(eval_cols)
    
    # 2. Extract comorbidities using icd9_comorbidities
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    train_diags = [train[col] for col in diag_cols if col in train]
    eval_diags = [eval_cols[col] for col in diag_cols if col in eval_cols]
    
    train_cm = icd9_comorbidities(*train_diags)
    eval_cm = icd9_comorbidities(*eval_diags)
    
    for k in train_cm:
        train[k] = train_cm[k]
        eval_cols[k] = eval_cm[k]
        extra_numeric.append(k)
        
    # 3. Clean ICD-9 chapters
    for col in diag_cols:
        grouped_col = col + "_grp"
        train[grouped_col] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_cols[grouped_col] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(grouped_col)
        
    # 4. Target encoding strictly on true high-cardinality columns
    te_cols = ['diag_1', 'diag_2', 'diag_3', 'medical_specialty']
    for col in te_cols:
        if col in train and col in eval_cols:
            tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42)
            te_col = col + "_te"
            train[te_col] = tr_enc
            eval_cols[te_col] = ev_enc
            extra_numeric.append(te_col)
            
    # 5. Core clinical utilization, stay intensity & key medication indicators
    for data in (train, eval_cols):
        tot_visits = data['number_outpatient'] + data['number_emergency'] + data['number_inpatient']
        data['total_visits'] = tot_visits
        data['inpatient_ratio'] = data['number_inpatient'] / (tot_visits + 1.0)
        data['emergency_ratio'] = data['number_emergency'] / (tot_visits + 1.0)
        
        h_days = data['time_in_hospital'] + 1.0
        data['meds_per_day'] = data['num_medications'] / h_days
        data['labs_per_day'] = data['num_lab_procedures'] / h_days
        
        data['insulin_up'] = (data['insulin'] == 'Up').astype(np.float64)
        data['insulin_down'] = (data['insulin'] == 'Down').astype(np.float64)
        data['med_changed'] = (data['change'] == 'Ch').astype(np.float64)
        data['diabetes_med'] = (data['diabetesMed'] == 'Yes').astype(np.float64)
        
    extra_numeric.extend([
        'total_visits', 'inpatient_ratio', 'emergency_ratio',
        'meds_per_day', 'labs_per_day',
        'insulin_up', 'insulin_down', 'med_changed', 'diabetes_med'
    ])
    
    # 6. Ordinal numeric age representation
    def age_to_numeric(age_str):
        if age_str is None:
            return np.nan
        try:
            cleaned = age_str.replace('[', '').replace(')', '').replace(']', '')
            parts = cleaned.split('-')
            return (float(parts[0]) + float(parts[1])) / 2.0
        except Exception:
            return np.nan

    train['age_num'] = np.array([age_to_numeric(x) for x in train['age']], dtype=np.float64)
    eval_cols['age_num'] = np.array([age_to_numeric(x) for x in eval_cols['age']], dtype=np.float64)
    extra_numeric.append('age_num')

    # 7. Encode clean tabular dataset
    encoder = TabularEncoder(min_count=25)
    encoder.fit(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)
    
    # 8. Train diverse models on 100% of data (zero fold data loss, full convergence)
    from scipy.stats import rankdata

    # Model 1: Balanced Canonical LightGBM
    m1 = lgb.LGBMClassifier(
        n_estimators=650,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        cat_smooth=30.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred_1 = m1.predict_proba(x_eval)[:, 1]

    # Model 2: Deeper Interaction LightGBM
    m2 = lgb.LGBMClassifier(
        n_estimators=550,
        learning_rate=0.018,
        num_leaves=50,
        max_depth=7,
        min_child_samples=150,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=5.0,
        reg_alpha=0.2,
        cat_smooth=50.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred_2 = m2.predict_proba(x_eval)[:, 1]

    # Model 3: Conservative Regularized Shallow LightGBM
    m3 = lgb.LGBMClassifier(
        n_estimators=700,
        learning_rate=0.016,
        num_leaves=16,
        max_depth=5,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=0.5,
        cat_smooth=20.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred_3 = m3.predict_proba(x_eval)[:, 1]

    # Model 4: Subspace Focused LightGBM
    m4 = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.02,
        num_leaves=28,
        min_child_samples=120,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=2.0,
        cat_smooth=40.0,
        random_state=777,
        n_jobs=1,
        verbose=-1,
    )
    m4.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred_4 = m4.predict_proba(x_eval)[:, 1]

    # 9. Optimal ROC-AUC Rank-Averaging
    n_eval = len(pred_1)
    rank_1 = rankdata(pred_1) / n_eval
    rank_2 = rankdata(pred_2) / n_eval
    rank_3 = rankdata(pred_3) / n_eval
    rank_4 = rankdata(pred_4) / n_eval

    return 0.35 * rank_1 + 0.25 * rank_2 + 0.25 * rank_3 + 0.15 * rank_4


# EVOLVE-BLOCK-END
# =====================================================================
