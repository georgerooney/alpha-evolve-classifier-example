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
    encoder = TabularEncoder(min_count=100)
    x_train = encoder.fit_transform(task, train)
    x_eval = encoder.transform(eval_cols)

    model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.0158,
        num_leaves=31,
        min_child_samples=200,
        subsample=0.69,
        subsample_freq=1,
        colsample_bytree=0.63,
        reg_lambda=0.022,
        cat_smooth=69.23,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    return model.predict_proba(x_eval)[:, 1]


# EVOLVE-BLOCK-END
# =====================================================================
