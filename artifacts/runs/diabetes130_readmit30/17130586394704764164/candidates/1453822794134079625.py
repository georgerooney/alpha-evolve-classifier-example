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
    try:
        import xgboost as xgb
        xgb_available = True
    except Exception:
        xgb_available = False

    # Shallow copy input column dictionaries
    train_feat = dict(train)
    eval_feat = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Charlson Comorbidities from ICD-9 codes
    cm_train = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k, v in cm_train.items():
        train_feat[k] = v.astype(np.float64)
        eval_feat[k] = cm_eval[k].astype(np.float64)
        extra_numeric.append(k)

    # 2. ICD-9 Chapter Category Mapping
    diag_1_grp_tr = np.array([icd9_group(c) for c in train['diag_1']], dtype=object)
    diag_1_grp_ev = np.array([icd9_group(c) for c in eval_cols['diag_1']], dtype=object)
    diag_2_grp_tr = np.array([icd9_group(c) for c in train['diag_2']], dtype=object)
    diag_2_grp_ev = np.array([icd9_group(c) for c in eval_cols['diag_2']], dtype=object)
    diag_3_grp_tr = np.array([icd9_group(c) for c in train['diag_3']], dtype=object)
    diag_3_grp_ev = np.array([icd9_group(c) for c in eval_cols['diag_3']], dtype=object)

    train_feat['diag_1_group'] = diag_1_grp_tr
    eval_feat['diag_1_group'] = diag_1_grp_ev
    train_feat['diag_2_group'] = diag_2_grp_tr
    eval_feat['diag_2_group'] = diag_2_grp_ev
    train_feat['diag_3_group'] = diag_3_grp_tr
    eval_feat['diag_3_group'] = diag_3_grp_ev
    extra_categorical.extend(['diag_1_group', 'diag_2_group', 'diag_3_group'])

    # 3. Out-of-fold Target Encoding for High-Cardinality & Key Categoricals
    te_columns = [
        'discharge_disposition_id',
        'admission_source_id',
        'admission_type_id',
        'medical_specialty',
        'payer_code',
        'diag_1',
        'diag_2',
        'diag_3',
        'diag_1_group',
    ]
    for col in te_columns:
        c_tr = train_feat[col]
        c_ev = eval_feat[col]
        tr_te, ev_te = oof_target_encode(c_tr, y_train, c_ev, n_folds=5, smoothing=20.0, seed=42)
        te_name = f'te_{col}'
        train_feat[te_name] = tr_te.astype(np.float64)
        eval_feat[te_name] = ev_te.astype(np.float64)
        extra_numeric.append(te_name)

    # Unified diagnosis risk summaries
    train_feat['te_diag_mean'] = (train_feat['te_diag_1'] + train_feat['te_diag_2'] + train_feat['te_diag_3']) / 3.0
    eval_feat['te_diag_mean'] = (eval_feat['te_diag_1'] + eval_feat['te_diag_2'] + eval_feat['te_diag_3']) / 3.0
    train_feat['te_diag_max'] = np.maximum(train_feat['te_diag_1'], np.maximum(train_feat['te_diag_2'], train_feat['te_diag_3']))
    eval_feat['te_diag_max'] = np.maximum(eval_feat['te_diag_1'], np.maximum(eval_feat['te_diag_2'], eval_feat['te_diag_3']))
    extra_numeric.extend(['te_diag_mean', 'te_diag_max'])

    # 4. Domain Feature Extraction (Utilization, Intensity, Medication Regimen)
    def compute_domain_features(cols: Dict[str, np.ndarray], cm: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        res = {}
        time = np.nan_to_num(cols['time_in_hospital'].astype(float), nan=1.0)
        safe_time = np.maximum(1.0, time)
        lab = np.nan_to_num(cols['num_lab_procedures'].astype(float), nan=0.0)
        proc = np.nan_to_num(cols['num_procedures'].astype(float), nan=0.0)
        med = np.nan_to_num(cols['num_medications'].astype(float), nan=0.0)
        outp = np.nan_to_num(cols['number_outpatient'].astype(float), nan=0.0)
        emrg = np.nan_to_num(cols['number_emergency'].astype(float), nan=0.0)
        inp = np.nan_to_num(cols['number_inpatient'].astype(float), nan=0.0)
        diag_cnt = np.nan_to_num(cols['number_diagnoses'].astype(float), nan=1.0)

        # Prior utilization metrics
        total_visits = outp + emrg + inp
        res['feat_total_prior_visits'] = total_visits
        res['feat_has_prior_inpatient'] = (inp > 0.0).astype(np.float64)
        res['feat_has_prior_emergency'] = (emrg > 0.0).astype(np.float64)
        res['feat_has_prior_visits'] = (total_visits > 0.0).astype(np.float64)
        res['feat_inpatient_ratio'] = (inp + 0.1) / (total_visits + 1.0)
        res['feat_emergency_ratio'] = (emrg + 0.1) / (total_visits + 1.0)
        res['feat_utilization_risk'] = inp * 4.0 + emrg * 2.0 + outp * 0.5
        res['feat_log_inpatient'] = np.log1p(np.maximum(0.0, inp))
        res['feat_log_emergency'] = np.log1p(np.maximum(0.0, emrg))
        res['feat_log_total_visits'] = np.log1p(np.maximum(0.0, total_visits))

        # Hospital stay & care intensity
        res['feat_lab_per_day'] = lab / safe_time
        res['feat_med_per_day'] = med / safe_time
        res['feat_proc_per_day'] = proc / safe_time
        res['feat_diag_per_day'] = diag_cnt / safe_time
        res['feat_med_x_time'] = med * time
        res['feat_lab_x_med'] = lab * med
        res['feat_long_stay'] = (time >= 7.0).astype(np.float64)

        # Weighted Charlson Comorbidity Index
        charlson = (
            cm['cm_mi'] + cm['cm_chf'] + cm['cm_pvd'] + cm['cm_cerebrovascular'] +
            cm['cm_dementia'] + cm['cm_pulmonary'] + cm['cm_rheumatic'] +
            cm['cm_peptic_ulcer'] + cm['cm_liver'] + cm['cm_diabetes_complicated'] * 2.0 +
            cm['cm_hemiplegia'] * 2.0 + cm['cm_renal'] * 2.0 + cm['cm_cancer'] * 2.0 +
            cm['cm_metastatic'] * 6.0 + cm['cm_hiv'] * 6.0
        )
        res['feat_weighted_charlson'] = charlson
        res['feat_inp_x_chf'] = inp * cm['cm_chf']
        res['feat_inp_x_renal'] = inp * cm['cm_renal']
        res['feat_inp_x_diab_comp'] = inp * cm['cm_diabetes_complicated']
        res['feat_inp_x_time'] = inp * time
        res['feat_charlson_per_diag'] = charlson / np.maximum(1.0, diag_cnt)

        # Medication regimen changes
        known_meds = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
            'miglitol', 'troglitazone', 'tolazamide', 'examide',
            'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        n_rows = len(time)
        med_count = np.zeros(n_rows, dtype=np.float64)
        med_up = np.zeros(n_rows, dtype=np.float64)
        med_down = np.zeros(n_rows, dtype=np.float64)
        med_change = np.zeros(n_rows, dtype=np.float64)

        for mc in known_meds:
            if mc in cols:
                arr = cols[mc]
                med_count += ((arr != 'No') & (arr != None)).astype(np.float64)
                is_up = (arr == 'Up')
                is_down = (arr == 'Down')
                med_up += is_up.astype(np.float64)
                med_down += is_down.astype(np.float64)
                med_change += (is_up | is_down).astype(np.float64)

        res['feat_med_prescribed_cnt'] = med_count
        res['feat_med_up_cnt'] = med_up
        res['feat_med_down_cnt'] = med_down
        res['feat_med_change_cnt'] = med_change
        res['feat_has_med_change'] = (med_change > 0.0).astype(np.float64)

        if 'insulin' in cols:
            ins = cols['insulin']
            res['feat_insulin_active'] = ((ins != 'No') & (ins != None)).astype(np.float64)
            res['feat_insulin_up'] = (ins == 'Up').astype(np.float64)
            res['feat_insulin_down'] = (ins == 'Down').astype(np.float64)
            res['feat_insulin_changed'] = ((ins == 'Up') | (ins == 'Down')).astype(np.float64)

        if 'change' in cols:
            res['feat_change_num'] = (cols['change'] == 'Ch').astype(np.float64)
        if 'diabetesMed' in cols:
            res['feat_diabetesMed_num'] = (cols['diabetesMed'] == 'Yes').astype(np.float64)

        # Lab results
        if 'A1Cresult' in cols:
            a1c = cols['A1Cresult']
            res['feat_a1c_tested'] = ((a1c != 'None') & (a1c != None)).astype(np.float64)
            res['feat_a1c_high'] = ((a1c == '>7') | (a1c == '>8')).astype(np.float64)

        if 'max_glu_serum' in cols:
            glu = cols['max_glu_serum']
            res['feat_glu_tested'] = ((glu != 'None') & (glu != None)).astype(np.float64)
            res['feat_glu_high'] = ((glu == '>200') | (glu == '>300')).astype(np.float64)

        # Demographic age mapping
        if 'age' in cols:
            age_midpoints = {
                '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
                '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
                '[80-90)': 85.0, '[90-100)': 95.0
            }
            res['feat_age_num'] = np.array([age_midpoints.get(a, 65.0) for a in cols['age']], dtype=np.float64)
            res['feat_age_x_inp'] = res['feat_age_num'] * inp

        return res

    domain_train = compute_domain_features(train, cm_train)
    domain_eval = compute_domain_features(eval_cols, cm_eval)
    for k, v in domain_train.items():
        train_feat[k] = v
        eval_feat[k] = domain_eval[k]
        extra_numeric.append(k)

    # 5. Tabular Encoding
    encoder = TabularEncoder(min_count=25)
    encoder.fit(task, train_feat, extra_numeric=tuple(extra_numeric), extra_categorical=tuple(extra_categorical))
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)

    # 6. Diverse GBDT Ensemble
    # LightGBM Model 1: Balanced leaf-wise with moderate depth
    lgb1 = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.018,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.72,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_lambda=1.5,
        reg_alpha=0.1,
        cat_smooth=35.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    lgb1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p_lgb1 = lgb1.predict_proba(x_eval)[:, 1]

    # LightGBM Model 2: Shallow, robust regularized trees
    lgb2 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.020,
        num_leaves=18,
        min_child_samples=250,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.48,
        reg_lambda=5.0,
        reg_alpha=0.5,
        cat_smooth=70.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    lgb2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p_lgb2 = lgb2.predict_proba(x_eval)[:, 1]

    # LightGBM Model 3: Deeper trees capturing granular multi-way interactions
    lgb3 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.016,
        num_leaves=42,
        min_child_samples=90,
        subsample=0.68,
        subsample_freq=1,
        colsample_bytree=0.68,
        reg_lambda=0.8,
        reg_alpha=0.2,
        cat_smooth=25.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    lgb3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p_lgb3 = lgb3.predict_proba(x_eval)[:, 1]

    # XGBoost: Level-wise tree splitting
    p_xgb = None
    if xgb_available:
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=220,
                learning_rate=0.028,
                max_depth=5,
                subsample=0.80,
                colsample_bytree=0.55,
                reg_lambda=3.0,
                reg_alpha=0.5,
                random_state=777,
                n_jobs=1,
                eval_metric='logloss',
            )
            xgb_model.fit(x_train, y_train)
            p_xgb = xgb_model.predict_proba(x_eval)[:, 1]
        except Exception:
            p_xgb = None

    if p_xgb is not None:
        p_final = 0.35 * p_lgb1 + 0.30 * p_lgb2 + 0.20 * p_lgb3 + 0.15 * p_xgb
    else:
        p_final = 0.40 * p_lgb1 + 0.35 * p_lgb2 + 0.25 * p_lgb3

    return np.clip(p_final, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
