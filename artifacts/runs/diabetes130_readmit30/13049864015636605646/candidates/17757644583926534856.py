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
    from scipy.stats import rankdata

    train_feats = dict(train)
    eval_feats = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    def add_num(name: str, tr_arr, ev_arr):
        train_feats[name] = np.asarray(tr_arr, dtype=np.float64)
        eval_feats[name] = np.asarray(ev_arr, dtype=np.float64)
        extra_numeric.append(name)

    def add_cat(name: str, tr_arr, ev_arr):
        train_feats[name] = np.asarray(tr_arr, dtype=object)
        eval_feats[name] = np.asarray(ev_arr, dtype=object)
        extra_categorical.append(name)

    # 1. Core Clinical Comorbidities (Charlson indicators & total condition count)
    cm_tr = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_ev = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k in cm_tr:
        add_num(k, cm_tr[k], cm_ev[k])

    # 2. ICD-9 Chapter groups (consolidating 700+ codes into 10 clinical bodies)
    for diag in ('diag_1', 'diag_2', 'diag_3'):
        if diag in train and diag in eval_cols:
            add_cat(f"{diag}_group",
                    [icd9_group(c) for c in train[diag]],
                    [icd9_group(c) for c in eval_cols[diag]])

    # 3. Clean Out-Of-Fold Target Encoding for Primary High-Cardinality Codes
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'medical_specialty',
        'discharge_disposition_id',
        'admission_source_id',
        'admission_type_id',
        'payer_code'
    ]
    for col in te_cols:
        if col in train and col in eval_cols:
            tr_te, ev_te = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=20.0, seed=42)
            add_num(f"{col}_te", tr_te, ev_te)

    # 4. Total Prior Healthcare Utilization (uniting inpatient, emergency, outpatient)
    inpat_tr = np.nan_to_num(train.get('number_inpatient', 0.0), nan=0.0)
    emerg_tr = np.nan_to_num(train.get('number_emergency', 0.0), nan=0.0)
    outpat_tr = np.nan_to_num(train.get('number_outpatient', 0.0), nan=0.0)
    inpat_ev = np.nan_to_num(eval_cols.get('number_inpatient', 0.0), nan=0.0)
    emerg_ev = np.nan_to_num(eval_cols.get('number_emergency', 0.0), nan=0.0)
    outpat_ev = np.nan_to_num(eval_cols.get('number_outpatient', 0.0), nan=0.0)
    add_num('total_prior_visits', inpat_tr + emerg_tr + outpat_tr, inpat_ev + emerg_ev + outpat_ev)

    # 5. Ordinal Midpoint Mapping for Demographics & Daily Intensity
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    if 'age' in train and 'age' in eval_cols:
        add_num('age_numeric',
                [age_map.get(a, 50.0) for a in train['age']],
                [age_map.get(a, 50.0) for a in eval_cols['age']])

    th_tr = np.maximum(np.nan_to_num(train.get('time_in_hospital', 1.0), nan=1.0), 1.0)
    th_ev = np.maximum(np.nan_to_num(eval_cols.get('time_in_hospital', 1.0), nan=1.0), 1.0)
    add_num('labs_per_day', np.nan_to_num(train.get('num_lab_procedures', 0.0)) / th_tr,
                            np.nan_to_num(eval_cols.get('num_lab_procedures', 0.0)) / th_ev)
    add_num('meds_per_day', np.nan_to_num(train.get('num_medications', 0.0)) / th_tr,
                            np.nan_to_num(eval_cols.get('num_medications', 0.0)) / th_ev)

    # 6. Fit TabularEncoder on Clean Signal Matrix
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        train_feats,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_feats)
    x_eval = encoder.transform(eval_feats)

    # 7. Tri-Architecture Complementary LightGBM Ensemble
    lgb_models = [
        # Explorer: deeper tree, agile learning
        lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.024,
            num_leaves=32,
            min_child_samples=100,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.68,
            reg_lambda=2.0,
            reg_alpha=0.1,
            cat_smooth=35.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        # Regularizer: conservative, shallower, high lambda
        lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.018,
            num_leaves=18,
            min_child_samples=180,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=6.0,
            reg_alpha=0.5,
            cat_smooth=50.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        # Generalist: wider leaf capacity, different subsampling trajectory
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.022,
            num_leaves=42,
            min_child_samples=130,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.62,
            reg_lambda=3.5,
            reg_alpha=0.2,
            cat_smooth=40.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = []
    for model in lgb_models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds.append(model.predict_proba(x_eval)[:, 1])

    # 8. Consensus Rank-Averaged Probability Blending:
    # Blends predictions in rank space (optimal for ROC-AUC) while preserving calibrated probability distribution
    ranks = np.mean([rankdata(p) for p in preds], axis=0)
    avg_probs = np.mean(preds, axis=0)
    final_pred = np.sort(avg_probs)[np.argsort(np.argsort(ranks))]
    return np.clip(np.nan_to_num(final_pred, nan=0.114), 1e-6, 1.0 - 1e-6)


# EVOLVE-BLOCK-END
# =====================================================================
