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
    n_train = len(y_train)
    first_col = next(iter(eval_cols.values()))
    n_eval = len(first_col)

    train_feat = dict(train)
    eval_feat = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Comorbidities via icd9_comorbidities
    train_cm = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    eval_cm = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k in train_cm:
        train_feat[k] = train_cm[k]
        eval_feat[k] = eval_cm[k]
        extra_numeric.append(k)

    # Cardiorenal interaction (CHF + Renal disease)
    train_feat['cm_cardiorenal'] = train_cm['cm_chf'] * train_cm['cm_renal']
    eval_feat['cm_cardiorenal'] = eval_cm['cm_chf'] * eval_cm['cm_renal']
    extra_numeric.append('cm_cardiorenal')

    # 2. ICD-9 Chapter groups
    for d_col in ['diag_1', 'diag_2', 'diag_3']:
        grp_col = f"{d_col}_group"
        train_feat[grp_col] = np.array([icd9_group(c) for c in train[d_col]], dtype=object)
        eval_feat[grp_col] = np.array([icd9_group(c) for c in eval_cols[d_col]], dtype=object)
        extra_categorical.append(grp_col)

    # 3. Out-of-fold target encoding for high-cardinality columns
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'discharge_disposition_id', 'admission_source_id', 'medical_specialty'
    ]
    for col in te_cols:
        if col in train and col in eval_cols:
            te_name = f"{col}_te"
            tr_te, ev_te = oof_target_encode(
                train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42
            )
            train_feat[te_name] = tr_te.astype(np.float64)
            eval_feat[te_name] = ev_te.astype(np.float64)
            extra_numeric.append(te_name)

    # 4. Age numeric midpoint mapping
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0,
    }
    train_feat['age_num'] = np.array([age_map.get(str(x), np.nan) for x in train['age']], dtype=np.float64)
    eval_feat['age_num'] = np.array([age_map.get(str(x), np.nan) for x in eval_cols['age']], dtype=np.float64)
    extra_numeric.append('age_num')

    # 5. Prior healthcare utilization & hospital intensity ratios
    for d_src, d_dst in [(train, train_feat), (eval_cols, eval_feat)]:
        inp = d_src['number_inpatient'].astype(np.float64)
        emg = d_src['number_emergency'].astype(np.float64)
        outp = d_src['number_outpatient'].astype(np.float64)
        time_hosp = np.maximum(d_src['time_in_hospital'].astype(np.float64), 1.0)

        tot_prior = inp + emg + outp
        d_dst['total_prior_visits'] = tot_prior
        d_dst['has_prior_inpatient'] = (inp > 0).astype(np.float64)
        d_dst['has_prior_emergency'] = (emg > 0).astype(np.float64)
        d_dst['has_any_prior_visit'] = (tot_prior > 0).astype(np.float64)
        d_dst['inpatient_ratio'] = np.where(tot_prior > 0, inp / tot_prior, 0.0)
        d_dst['acute_prior_visits'] = inp + emg

        num_meds = d_src['num_medications'].astype(np.float64)
        num_labs = d_src['num_lab_procedures'].astype(np.float64)
        num_procs = d_src['num_procedures'].astype(np.float64)
        num_diags = d_src['number_diagnoses'].astype(np.float64)

        d_dst['lab_per_day'] = num_labs / time_hosp
        d_dst['meds_per_day'] = num_meds / time_hosp
        d_dst['procs_per_day'] = num_procs / time_hosp
        d_dst['diags_per_day'] = num_diags / time_hosp
        d_dst['total_procedures'] = num_labs + num_procs

    util_cols = [
        'total_prior_visits', 'has_prior_inpatient', 'has_prior_emergency',
        'has_any_prior_visit', 'inpatient_ratio', 'acute_prior_visits',
        'lab_per_day', 'meds_per_day', 'procs_per_day', 'diags_per_day', 'total_procedures'
    ]
    extra_numeric.extend(util_cols)

    # 6. Medication activity, changes, and insulin patterns
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    avail_meds = [m for m in med_names if m in train and m in eval_cols]

    for d_src, d_dst, n_rows in [(train, train_feat, n_train), (eval_cols, eval_feat, n_eval)]:
        n_act = np.zeros(n_rows, dtype=np.float64)
        n_up = np.zeros(n_rows, dtype=np.float64)
        n_dn = np.zeros(n_rows, dtype=np.float64)
        for m in avail_meds:
            arr = d_src[m]
            for val in ['Steady', 'Up', 'Down']:
                n_act += (arr == val).astype(np.float64)
            n_up += (arr == 'Up').astype(np.float64)
            n_dn += (arr == 'Down').astype(np.float64)

        d_dst['num_meds_active'] = n_act
        d_dst['num_meds_up'] = n_up
        d_dst['num_meds_down'] = n_dn
        d_dst['num_meds_changed'] = n_up + n_dn

        if 'insulin' in d_src:
            ins = d_src['insulin']
            d_dst['insulin_active'] = ((ins == 'Steady') | (ins == 'Up') | (ins == 'Down')).astype(np.float64)
            d_dst['insulin_up'] = (ins == 'Up').astype(np.float64)
            d_dst['insulin_down'] = (ins == 'Down').astype(np.float64)
            d_dst['insulin_steady'] = (ins == 'Steady').astype(np.float64)
        else:
            d_dst['insulin_active'] = np.zeros(n_rows, dtype=np.float64)
            d_dst['insulin_up'] = np.zeros(n_rows, dtype=np.float64)
            d_dst['insulin_down'] = np.zeros(n_rows, dtype=np.float64)
            d_dst['insulin_steady'] = np.zeros(n_rows, dtype=np.float64)

        if 'change' in d_src:
            d_dst['change_flag'] = (d_src['change'] == 'Ch').astype(np.float64)
        else:
            d_dst['change_flag'] = np.zeros(n_rows, dtype=np.float64)

        if 'diabetesMed' in d_src:
            d_dst['diabetes_med_flag'] = (d_src['diabetesMed'] == 'Yes').astype(np.float64)
        else:
            d_dst['diabetes_med_flag'] = np.zeros(n_rows, dtype=np.float64)

        if 'A1Cresult' in d_src:
            a1c = d_src['A1Cresult']
            d_dst['a1c_tested'] = ((a1c == 'Norm') | (a1c == '>7') | (a1c == '>8')).astype(np.float64)
            d_dst['a1c_high'] = ((a1c == '>7') | (a1c == '>8')).astype(np.float64)
        else:
            d_dst['a1c_tested'] = np.zeros(n_rows, dtype=np.float64)
            d_dst['a1c_high'] = np.zeros(n_rows, dtype=np.float64)

        if 'max_glu_serum' in d_src:
            glu = d_src['max_glu_serum']
            d_dst['glu_tested'] = ((glu == 'Norm') | (glu == '>200') | (glu == '>300')).astype(np.float64)
            d_dst['glu_high'] = ((glu == '>200') | (glu == '>300')).astype(np.float64)
        else:
            d_dst['glu_tested'] = np.zeros(n_rows, dtype=np.float64)
            d_dst['glu_high'] = np.zeros(n_rows, dtype=np.float64)

    med_feature_names = [
        'num_meds_active', 'num_meds_up', 'num_meds_down', 'num_meds_changed',
        'insulin_active', 'insulin_up', 'insulin_down', 'insulin_steady',
        'change_flag', 'diabetes_med_flag', 'a1c_tested', 'a1c_high', 'glu_tested', 'glu_high'
    ]
    extra_numeric.extend(med_feature_names)

    # 7. Fit TabularEncoder with tuned threshold
    encoder = TabularEncoder(min_count=25)
    encoder.fit(task, train_feat, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)

    # 8. Diverse multi-architecture LightGBM ensemble
    models = [
        lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=80,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=1.0,
            cat_smooth=50.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.025,
            num_leaves=20,
            min_child_samples=50,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=3.0,
            cat_smooth=30.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=45,
            min_child_samples=120,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=0.5,
            cat_smooth=70.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = np.zeros(n_eval, dtype=np.float64)
    for model in models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds += model.predict_proba(x_eval)[:, 1]

    preds /= len(models)
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
