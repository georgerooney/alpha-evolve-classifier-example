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

    # 3. High-cardinality Target Encodings (genuine high-cardinality only)
    te_cols = ['diag_1', 'diag_2', 'diag_3', 'medical_specialty', 'discharge_disposition_id']
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

    # 5. Age numeric midpoint
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    train_cols['age_num'] = np.array([age_map.get(a, 65.0) for a in train_cols['age']], dtype=np.float64)
    eval_cols_dict['age_num'] = np.array([age_map.get(a, 65.0) for a in eval_cols_dict['age']], dtype=np.float64)
    extra_numeric.append('age_num')

    # 6. Concise medication counts (avoids duplicating categorical indicators)
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    present_meds = [m for m in med_names if m in train_cols and m in eval_cols_dict]

    act_tr = np.zeros(len(y_train), dtype=np.float64)
    act_ev = np.zeros(len(eval_cols_dict['age']), dtype=np.float64)
    chg_tr = np.zeros(len(y_train), dtype=np.float64)
    chg_ev = np.zeros(len(eval_cols_dict['age']), dtype=np.float64)

    for m in present_meds:
        tr_m = train_cols[m]
        ev_m = eval_cols_dict[m]
        act_tr += (tr_m != 'No').astype(np.float64)
        act_ev += (ev_m != 'No').astype(np.float64)
        chg_tr += ((tr_m == 'Up') | (tr_m == 'Down')).astype(np.float64)
        chg_ev += ((ev_m == 'Up') | (ev_m == 'Down')).astype(np.float64)

    train_cols['num_active_meds'] = act_tr
    eval_cols_dict['num_active_meds'] = act_ev
    extra_numeric.append('num_active_meds')

    train_cols['num_med_changes'] = chg_tr
    eval_cols_dict['num_med_changes'] = chg_ev
    extra_numeric.append('num_med_changes')

    # Fit encoder with standard min_count to retain specialized transfer/care locations
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task, train_cols,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical)
    )
    x_train = encoder.transform(train_cols)
    x_eval = encoder.transform(eval_cols_dict)

    # Clean, seed-averaged LightGBM ensemble
    seeds = [42, 123, 456, 789]
    preds = np.zeros(len(x_eval), dtype=np.float64)
    for seed in seeds:
        model = lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.025,
            num_leaves=28,
            min_child_samples=120,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=2.0,
            reg_alpha=0.2,
            cat_smooth=40.0,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1]
    preds /= len(seeds)

    preds = np.clip(preds, 0.0, 1.0)
    if np.isnan(preds).any():
        preds = np.nan_to_num(preds, nan=0.114)

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
