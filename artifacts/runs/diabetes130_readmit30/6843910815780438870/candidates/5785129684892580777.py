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

    train_feats = dict(train)
    eval_feats = dict(eval_cols)

    # 1. Extract clinical comorbidities flags
    comorb_train = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    comorb_eval = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
    train_feats.update(comorb_train)
    eval_feats.update(comorb_eval)

    # 2. Add grouped ICD-9 chapters
    extra_categorical = []
    for col in ["diag_1", "diag_2", "diag_3"]:
        gcol = f"{col}_grouped"
        train_feats[gcol] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_feats[gcol] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(gcol)

    # 3. Add out-of-fold target encodings for high-cardinality features
    te_cols = ["diag_1", "diag_2", "diag_3", "discharge_disposition_id", "medical_specialty"]
    for col in te_cols:
        tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=20.0, seed=42)
        train_feats[f"{col}_te"] = tr_enc
        eval_feats[f"{col}_te"] = ev_enc

    extra_numeric = list(comorb_train.keys()) + [f"{col}_te" for col in te_cols]

    # 4. Tabular encoding
    encoder = TabularEncoder()
    encoder.fit(
        task,
        train_feats,
        extra_numeric=extra_numeric,
        extra_categorical=extra_categorical,
    )
    x_train = encoder.transform(train_feats)
    x_eval = encoder.transform(eval_feats)

    # 5. Fit on full training data with seed-averaging
    seeds = [42, 123, 456]
    preds = np.zeros(len(x_eval))

    for seed in seeds:
        model = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            cat_smooth=30.0,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(
            x_train,
            y_train,
            categorical_feature=encoder.categorical_indices,
        )
        preds += model.predict_proba(x_eval)[:, 1]

    return preds / len(seeds)


# EVOLVE-BLOCK-END
# =====================================================================
