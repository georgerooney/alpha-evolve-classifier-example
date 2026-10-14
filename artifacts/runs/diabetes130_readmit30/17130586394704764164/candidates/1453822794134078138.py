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
        
    # 3. Grouped ICD-9 chapters
    for col in diag_cols:
        grouped_col = col + "_grp"
        train[grouped_col] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_cols[grouped_col] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(grouped_col)
        
    # Clinical and administrative indicators
    train['discharged_to_home'] = (train['discharge_disposition_id'] == "1").astype(np.float64)
    eval_cols['discharged_to_home'] = (eval_cols['discharge_disposition_id'] == "1").astype(np.float64)
    extra_numeric.append('discharged_to_home')
    
    train['discharged_to_snf'] = (train['discharge_disposition_id'] == "3").astype(np.float64)
    eval_cols['discharged_to_snf'] = (eval_cols['discharge_disposition_id'] == "3").astype(np.float64)
    extra_numeric.append('discharged_to_snf')

    train['discharged_to_home_health'] = (train['discharge_disposition_id'] == "6").astype(np.float64)
    eval_cols['discharged_to_home_health'] = (eval_cols['discharge_disposition_id'] == "6").astype(np.float64)
    extra_numeric.append('discharged_to_home_health')

    train['admission_from_er'] = (train['admission_source_id'] == "7").astype(np.float64)
    eval_cols['admission_from_er'] = (eval_cols['admission_source_id'] == "7").astype(np.float64)
    extra_numeric.append('admission_from_er')

    # Combined disposition and source interaction
    train['disp_source'] = np.array([f"{d}_{s}" for d, s in zip(train['discharge_disposition_id'], train['admission_source_id'])], dtype=object)
    eval_cols['disp_source'] = np.array([f"{d}_{s}" for d, s in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])], dtype=object)
    extra_categorical.append('disp_source')
        
    # 4. Target encoding of high-cardinality columns
    te_cols = ['diag_1', 'diag_2', 'diag_3', 'medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'payer_code', 'disp_source', 'admission_type_id']
    for col in te_cols:
        if col in train and col in eval_cols:
            tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42)
            te_col = col + "_te"
            train[te_col] = tr_enc
            eval_cols[te_col] = ev_enc
            extra_numeric.append(te_col)
            
    # 5. Utilisation intensity features
    train_total_visits = train['number_outpatient'] + train['number_emergency'] + train['number_inpatient']
    eval_total_visits = eval_cols['number_outpatient'] + eval_cols['number_emergency'] + eval_cols['number_inpatient']
    train['total_visits'] = train_total_visits
    eval_cols['total_visits'] = eval_total_visits
    extra_numeric.append('total_visits')
    
    train['inpatient_ratio'] = train['number_inpatient'] / (train_total_visits + 1.0)
    eval_cols['inpatient_ratio'] = eval_cols['number_inpatient'] / (eval_total_visits + 1.0)
    extra_numeric.append('inpatient_ratio')

    train['has_prior_inpatient'] = (train['number_inpatient'] > 0).astype(np.float64)
    eval_cols['has_prior_inpatient'] = (eval_cols['number_inpatient'] > 0).astype(np.float64)
    extra_numeric.append('has_prior_inpatient')
    
    train['has_prior_emergency'] = (train['number_emergency'] > 0).astype(np.float64)
    eval_cols['has_prior_emergency'] = (eval_cols['number_emergency'] > 0).astype(np.float64)
    extra_numeric.append('has_prior_emergency')

    # Treatment intensity per day
    train['labs_per_day'] = train['num_lab_procedures'] / train['time_in_hospital']
    eval_cols['labs_per_day'] = eval_cols['num_lab_procedures'] / eval_cols['time_in_hospital']
    extra_numeric.append('labs_per_day')

    train['meds_per_day'] = train['num_medications'] / train['time_in_hospital']
    eval_cols['meds_per_day'] = eval_cols['num_medications'] / eval_cols['time_in_hospital']
    extra_numeric.append('meds_per_day')

    # Medical test results indicators
    train['a1c_measured'] = (train['A1Cresult'] != "None").astype(np.float64)
    eval_cols['a1c_measured'] = (eval_cols['A1Cresult'] != "None").astype(np.float64)
    extra_numeric.append('a1c_measured')

    train['a1c_high'] = ((train['A1Cresult'] == ">7") | (train['A1Cresult'] == ">8")).astype(np.float64)
    eval_cols['a1c_high'] = ((eval_cols['A1Cresult'] == ">7") | (eval_cols['A1Cresult'] == ">8")).astype(np.float64)
    extra_numeric.append('a1c_high')

    train['glu_measured'] = (train['max_glu_serum'] != "None").astype(np.float64)
    eval_cols['glu_measured'] = (eval_cols['max_glu_serum'] != "None").astype(np.float64)
    extra_numeric.append('glu_measured')

    train['glu_high'] = ((train['max_glu_serum'] == ">200") | (train['max_glu_serum'] == ">300")).astype(np.float64)
    eval_cols['glu_high'] = ((eval_cols['max_glu_serum'] == ">200") | (eval_cols['max_glu_serum'] == ">300")).astype(np.float64)
    extra_numeric.append('glu_high')
    
    # 6. Medication active and change counts
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    train_active_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_active_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_change_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_change_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_stable_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_stable_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    
    for col in med_cols:
        if col in train:
            train_active_meds += (train[col] != "No") & (train[col] != None)
            eval_active_meds += (eval_cols[col] != "No") & (eval_cols[col] != None)
            train_change_meds += (train[col] == "Up") | (train[col] == "Down")
            eval_change_meds += (eval_cols[col] == "Up") | (eval_cols[col] == "Down")
            train_stable_meds += (train[col] == "Steady")
            eval_stable_meds += (eval_cols[col] == "Steady")
            
    train['active_meds_count'] = train_active_meds
    eval_cols['active_meds_count'] = eval_active_meds
    extra_numeric.append('active_meds_count')
    
    train['change_meds_count'] = train_change_meds
    eval_cols['change_meds_count'] = eval_change_meds
    extra_numeric.append('change_meds_count')

    train['stable_meds_count'] = train_stable_meds
    eval_cols['stable_meds_count'] = eval_stable_meds
    extra_numeric.append('stable_meds_count')
    
    # 7. Insulin change interaction
    train['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(train['insulin'], train['change'])], dtype=object)
    eval_cols['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(eval_cols['insulin'], eval_cols['change'])], dtype=object)
    extra_categorical.append('insulin_change')
    
    # 8. Numeric age representation
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

    train['age_inpatient_interaction'] = train['age_num'] * train['number_inpatient']
    eval_cols['age_inpatient_interaction'] = eval_cols['age_num'] * eval_cols['number_inpatient']
    extra_numeric.append('age_inpatient_interaction')

    train['age_diagnoses_interaction'] = train['age_num'] * train['number_diagnoses']
    eval_cols['age_diagnoses_interaction'] = eval_cols['age_num'] * eval_cols['number_diagnoses']
    extra_numeric.append('age_diagnoses_interaction')

    # 9. Encode datasets
    encoder = TabularEncoder(min_count=30)
    encoder.fit(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)
    
    # 10. Fit Stratified 5-Fold LGBM Ensemble
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_preds = np.zeros(x_eval.shape[0])
    
    for train_idx, val_idx in skf.split(x_train, y_train):
        fold_x_train, fold_y_train = x_train[train_idx], y_train[train_idx]
        
        model1 = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.015,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=0.5,
            cat_smooth=50.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        model2 = lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.012,
            num_leaves=23,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            cat_smooth=30.0,
            random_state=2023,
            n_jobs=1,
            verbose=-1,
        )
        model3 = lgb.LGBMClassifier(
            n_estimators=180,
            learning_rate=0.018,
            num_leaves=45,
            min_child_samples=200,
            subsample=0.6,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=0.1,
            cat_smooth=100.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        )
        
        model1.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        preds1 = model1.predict_proba(x_eval)[:, 1]
        
        model2.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        preds2 = model2.predict_proba(x_eval)[:, 1]
        
        model3.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        preds3 = model3.predict_proba(x_eval)[:, 1]
        
        fold_preds = 0.4 * preds1 + 0.3 * preds2 + 0.3 * preds3
        eval_preds += fold_preds / 5.0
        
    return eval_preds


# EVOLVE-BLOCK-END
# =====================================================================
