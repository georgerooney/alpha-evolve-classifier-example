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
        
    high_risk_cms = ['cm_chf', 'cm_renal', 'cm_metastatic', 'cm_liver', 'cm_dementia']
    train_hr_cm = np.zeros(len(y_train), dtype=np.float64)
    eval_hr_cm = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    for k in high_risk_cms:
        if k in train_cm:
            train_hr_cm += train_cm[k]
            eval_hr_cm += eval_cm[k]
    train['high_risk_cm_count'] = train_hr_cm
    eval_cols['high_risk_cm_count'] = eval_hr_cm
    extra_numeric.append('high_risk_cm_count')

    # Weighted Charlson Comorbidity Index & clinical syndromic interactions
    train_charlson = (
        train_cm.get('cm_mi', 0.0) + train_cm.get('cm_chf', 0.0) + train_cm.get('cm_pvd', 0.0) +
        train_cm.get('cm_cerebrovascular', 0.0) + train_cm.get('cm_dementia', 0.0) +
        train_cm.get('cm_pulmonary', 0.0) + train_cm.get('cm_rheumatic', 0.0) +
        train_cm.get('cm_peptic_ulcer', 0.0) + train_cm.get('cm_liver', 0.0) +
        2.0 * (train_cm.get('cm_hemiplegia', 0.0) + train_cm.get('cm_renal', 0.0) +
               train_cm.get('cm_diabetes_complicated', 0.0) + train_cm.get('cm_cancer', 0.0)) +
        4.0 * (train_cm.get('cm_metastatic', 0.0) + train_cm.get('cm_hiv', 0.0))
    )
    eval_charlson = (
        eval_cm.get('cm_mi', 0.0) + eval_cm.get('cm_chf', 0.0) + eval_cm.get('cm_pvd', 0.0) +
        eval_cm.get('cm_cerebrovascular', 0.0) + eval_cm.get('cm_dementia', 0.0) +
        eval_cm.get('cm_pulmonary', 0.0) + eval_cm.get('cm_rheumatic', 0.0) +
        eval_cm.get('cm_peptic_ulcer', 0.0) + eval_cm.get('cm_liver', 0.0) +
        2.0 * (eval_cm.get('cm_hemiplegia', 0.0) + eval_cm.get('cm_renal', 0.0) +
               eval_cm.get('cm_diabetes_complicated', 0.0) + eval_cm.get('cm_cancer', 0.0)) +
        4.0 * (eval_cm.get('cm_metastatic', 0.0) + eval_cm.get('cm_hiv', 0.0))
    )
    train['charlson_index'] = train_charlson
    eval_cols['charlson_index'] = eval_charlson
    extra_numeric.append('charlson_index')

    train['cardiorenal_syndrome'] = (train_cm.get('cm_chf', 0.0) * train_cm.get('cm_renal', 0.0)).astype(np.float64)
    eval_cols['cardiorenal_syndrome'] = (eval_cm.get('cm_chf', 0.0) * eval_cm.get('cm_renal', 0.0)).astype(np.float64)
    extra_numeric.append('cardiorenal_syndrome')

    train['diabetic_nephropathy'] = (train_cm.get('cm_diabetes_complicated', 0.0) * train_cm.get('cm_renal', 0.0)).astype(np.float64)
    eval_cols['diabetic_nephropathy'] = (eval_cm.get('cm_diabetes_complicated', 0.0) * eval_cm.get('cm_renal', 0.0)).astype(np.float64)
    extra_numeric.append('diabetic_nephropathy')
        
    # 3. Grouped ICD-9 chapters and 3-digit category prefixes
    def icd9_prefix3(code):
        if code is None or code == "None" or not isinstance(code, str):
            return "missing"
        return code.split('.')[0]

    for col in diag_cols:
        grouped_col = col + "_grp"
        train[grouped_col] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_cols[grouped_col] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(grouped_col)

        pfx_col = col + "_pfx"
        train[pfx_col] = np.array([icd9_prefix3(x) for x in train[col]], dtype=object)
        eval_cols[pfx_col] = np.array([icd9_prefix3(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(pfx_col)

    # Joint clinical interactions
    train['disch_adm_joint'] = np.array([f"{d}_{a}" for d, a in zip(train['discharge_disposition_id'], train['admission_source_id'])], dtype=object)
    eval_cols['disch_adm_joint'] = np.array([f"{d}_{a}" for d, a in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])], dtype=object)
    extra_categorical.append('disch_adm_joint')
    
    train['diag_12_joint'] = np.array([f"{d1}_{d2}" for d1, d2 in zip(train['diag_1_grp'], train['diag_2_grp'])], dtype=object)
    eval_cols['diag_12_joint'] = np.array([f"{d1}_{d2}" for d1, d2 in zip(eval_cols['diag_1_grp'], eval_cols['diag_2_grp'])], dtype=object)
    extra_categorical.append('diag_12_joint')
        
    # 4. Target encoding of high-cardinality columns
    te_cols = [
        'diag_1', 'diag_2', 'diag_3', 'diag_1_pfx', 'diag_2_pfx', 'diag_3_pfx',
        'medical_specialty', 'discharge_disposition_id', 'admission_source_id',
        'payer_code', 'admission_type_id', 'age', 'disch_adm_joint', 'diag_12_joint'
    ]
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
    
    train['emergency_ratio'] = train['number_emergency'] / (train_total_visits + 1.0)
    eval_cols['emergency_ratio'] = eval_cols['number_emergency'] / (eval_total_visits + 1.0)
    extra_numeric.append('emergency_ratio')
    
    train['has_prior_visits'] = (train_total_visits > 0).astype(np.float64)
    eval_cols['has_prior_visits'] = (eval_total_visits > 0).astype(np.float64)
    extra_numeric.append('has_prior_visits')
    
    train['hospital_intensity'] = train['num_lab_procedures'] + train['num_procedures'] + train['num_medications']
    eval_cols['hospital_intensity'] = eval_cols['num_lab_procedures'] + eval_cols['num_procedures'] + eval_cols['num_medications']
    extra_numeric.append('hospital_intensity')
    
    train['med_per_day'] = train['num_medications'] / (train['time_in_hospital'] + 0.1)
    eval_cols['med_per_day'] = eval_cols['num_medications'] / (eval_cols['time_in_hospital'] + 0.1)
    extra_numeric.append('med_per_day')
    
    train['lab_per_day'] = train['num_lab_procedures'] / (train['time_in_hospital'] + 0.1)
    eval_cols['lab_per_day'] = eval_cols['num_lab_procedures'] / (eval_cols['time_in_hospital'] + 0.1)
    extra_numeric.append('lab_per_day')
    
    train['proc_per_day'] = train['num_procedures'] / (train['time_in_hospital'] + 0.1)
    eval_cols['proc_per_day'] = eval_cols['num_procedures'] / (eval_cols['time_in_hospital'] + 0.1)
    extra_numeric.append('proc_per_day')
    
    train['diagnoses_per_day'] = train['number_diagnoses'] / (train['time_in_hospital'] + 0.1)
    eval_cols['diagnoses_per_day'] = eval_cols['number_diagnoses'] / (eval_cols['time_in_hospital'] + 0.1)
    extra_numeric.append('diagnoses_per_day')

    # Prior utilization non-linear signals
    train['has_prior_inpatient'] = (train['number_inpatient'] > 0).astype(np.float64)
    eval_cols['has_prior_inpatient'] = (eval_cols['number_inpatient'] > 0).astype(np.float64)
    extra_numeric.append('has_prior_inpatient')

    train['prior_inpatient_gt1'] = (train['number_inpatient'] > 1).astype(np.float64)
    eval_cols['prior_inpatient_gt1'] = (eval_cols['number_inpatient'] > 1).astype(np.float64)
    extra_numeric.append('prior_inpatient_gt1')

    train['has_prior_emergency'] = (train['number_emergency'] > 0).astype(np.float64)
    eval_cols['has_prior_emergency'] = (eval_cols['number_emergency'] > 0).astype(np.float64)
    extra_numeric.append('has_prior_emergency')

    train['log_inpatient'] = np.log1p(np.maximum(0.0, train['number_inpatient']))
    eval_cols['log_inpatient'] = np.log1p(np.maximum(0.0, eval_cols['number_inpatient']))
    extra_numeric.append('log_inpatient')

    train['log_total_visits'] = np.log1p(np.maximum(0.0, train_total_visits))
    eval_cols['log_total_visits'] = np.log1p(np.maximum(0.0, eval_total_visits))
    extra_numeric.append('log_total_visits')

    train['diagnoses_per_med'] = train['number_diagnoses'] / (train['num_medications'] + 1.0)
    eval_cols['diagnoses_per_med'] = eval_cols['number_diagnoses'] / (eval_cols['num_medications'] + 1.0)
    extra_numeric.append('diagnoses_per_med')

    # Discharge disposition & admission acuity indicators
    facility_set = {'2', '3', '4', '5', '6', '18', '22', '28'}
    train['disch_home'] = np.array([1.0 if x == '1' else 0.0 for x in train['discharge_disposition_id']], dtype=np.float64)
    eval_cols['disch_home'] = np.array([1.0 if x == '1' else 0.0 for x in eval_cols['discharge_disposition_id']], dtype=np.float64)
    extra_numeric.append('disch_home')

    train['disch_facility'] = np.array([1.0 if x in facility_set else 0.0 for x in train['discharge_disposition_id']], dtype=np.float64)
    eval_cols['disch_facility'] = np.array([1.0 if x in facility_set else 0.0 for x in eval_cols['discharge_disposition_id']], dtype=np.float64)
    extra_numeric.append('disch_facility')

    train['disch_ama'] = np.array([1.0 if x == '7' else 0.0 for x in train['discharge_disposition_id']], dtype=np.float64)
    eval_cols['disch_ama'] = np.array([1.0 if x == '7' else 0.0 for x in eval_cols['discharge_disposition_id']], dtype=np.float64)
    extra_numeric.append('disch_ama')

    train['adm_emergency'] = np.array([1.0 if s == '7' or t == '1' else 0.0 for s, t in zip(train['admission_source_id'], train['admission_type_id'])], dtype=np.float64)
    eval_cols['adm_emergency'] = np.array([1.0 if s == '7' or t == '1' else 0.0 for s, t in zip(eval_cols['admission_source_id'], eval_cols['admission_type_id'])], dtype=np.float64)
    extra_numeric.append('adm_emergency')

    # Lab result status (HbA1c & Max Glucose)
    def a1c_score(v):
        if v is None or v == 'None':
            return 0.0
        if v == 'Norm':
            return 1.0
        if v == '>7':
            return 2.0
        if v == '>8':
            return 3.0
        return 0.0

    def glu_score(v):
        if v is None or v == 'None':
            return 0.0
        if v == 'Norm':
            return 1.0
        if v == '>200':
            return 2.0
        if v == '>300':
            return 3.0
        return 0.0

    train['a1c_ordinal'] = np.array([a1c_score(x) for x in train['A1Cresult']], dtype=np.float64)
    eval_cols['a1c_ordinal'] = np.array([a1c_score(x) for x in eval_cols['A1Cresult']], dtype=np.float64)
    extra_numeric.append('a1c_ordinal')

    train['glu_ordinal'] = np.array([glu_score(x) for x in train['max_glu_serum']], dtype=np.float64)
    eval_cols['glu_ordinal'] = np.array([glu_score(x) for x in eval_cols['max_glu_serum']], dtype=np.float64)
    extra_numeric.append('glu_ordinal')

    train['a1c_tested'] = (train['a1c_ordinal'] > 0.0).astype(np.float64)
    eval_cols['a1c_tested'] = (eval_cols['a1c_ordinal'] > 0.0).astype(np.float64)
    extra_numeric.append('a1c_tested')

    train['a1c_high_no_change'] = ((train['a1c_ordinal'] >= 2.0) & (train['change'] == 'No')).astype(np.float64)
    eval_cols['a1c_high_no_change'] = ((eval_cols['a1c_ordinal'] >= 2.0) & (eval_cols['change'] == 'No')).astype(np.float64)
    extra_numeric.append('a1c_high_no_change')
    
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
    train_up_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_up_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_down_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_down_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_steady_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_steady_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    
    for col in med_cols:
        if col in train:
            tr_not_no_none = (train[col] != "No") & (train[col] != None)
            ev_not_no_none = (eval_cols[col] != "No") & (eval_cols[col] != None)
            
            train_active_meds += tr_not_no_none
            eval_active_meds += ev_not_no_none
            
            train_change_meds += (train[col] == "Up") | (train[col] == "Down")
            eval_change_meds += (eval_cols[col] == "Up") | (eval_cols[col] == "Down")
            
            train_up_meds += (train[col] == "Up")
            eval_up_meds += (eval_cols[col] == "Up")
            
            train_down_meds += (train[col] == "Down")
            eval_down_meds += (eval_cols[col] == "Down")
            
            train_steady_meds += (train[col] == "Steady")
            eval_steady_meds += (eval_cols[col] == "Steady")
            
    train['active_meds_count'] = train_active_meds
    eval_cols['active_meds_count'] = eval_active_meds
    extra_numeric.append('active_meds_count')
    
    train['change_meds_count'] = train_change_meds
    eval_cols['change_meds_count'] = eval_change_meds
    extra_numeric.append('change_meds_count')
    
    train['up_meds_count'] = train_up_meds
    eval_cols['up_meds_count'] = eval_up_meds
    extra_numeric.append('up_meds_count')
    
    train['down_meds_count'] = train_down_meds
    eval_cols['down_meds_count'] = eval_down_meds
    extra_numeric.append('down_meds_count')
    
    train['steady_meds_count'] = train_steady_meds
    eval_cols['steady_meds_count'] = eval_steady_meds
    extra_numeric.append('steady_meds_count')
    
    # 7. Specific key medications and insulin change interaction
    train['on_insulin'] = ((train['insulin'] != 'No') & (train['insulin'] != None)).astype(np.float64)
    eval_cols['on_insulin'] = ((eval_cols['insulin'] != 'No') & (eval_cols['insulin'] != None)).astype(np.float64)
    extra_numeric.append('on_insulin')

    train['on_metformin'] = ((train['metformin'] != 'No') & (train['metformin'] != None)).astype(np.float64)
    eval_cols['on_metformin'] = ((eval_cols['metformin'] != 'No') & (eval_cols['metformin'] != None)).astype(np.float64)
    extra_numeric.append('on_metformin')

    train['insulin_and_oral'] = ((train['on_insulin'] > 0.0) & (train_active_meds > 1.0)).astype(np.float64)
    eval_cols['insulin_and_oral'] = ((eval_cols['on_insulin'] > 0.0) & (eval_active_meds > 1.0)).astype(np.float64)
    extra_numeric.append('insulin_and_oral')

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

    # 9. Encode datasets
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)
    
    # 10. Fit Stratified 5-Fold LGBM Ensemble with ExtraTrees Diversity
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_preds = np.zeros(x_eval.shape[0])
    
    for train_idx, val_idx in skf.split(x_train, y_train):
        fold_x_train, fold_y_train = x_train[train_idx], y_train[train_idx]
        
        # Model A: Standard GBDT with balanced depth
        model_a = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.025,
            num_leaves=31,
            max_depth=6,
            min_child_samples=120,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=0.5,
            cat_smooth=50.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        
        # Model B: Extremely Randomized Trees GBDT for split decorrelation
        model_b = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.025,
            num_leaves=25,
            max_depth=6,
            extra_trees=True,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=1.5,
            cat_smooth=30.0,
            random_state=100,
            n_jobs=1,
            verbose=-1,
        )
        
        # Model C: Robust shallow GBDT
        model_c = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.025,
            num_leaves=15,
            max_depth=4,
            min_child_samples=180,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=0.1,
            cat_smooth=80.0,
            random_state=2023,
            n_jobs=1,
            verbose=-1,
        )
        
        model_a.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        model_b.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        model_c.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        
        pred_a = model_a.predict_proba(x_eval)[:, 1]
        pred_b = model_b.predict_proba(x_eval)[:, 1]
        pred_c = model_c.predict_proba(x_eval)[:, 1]
        
        fold_preds = 0.35 * pred_a + 0.35 * pred_b + 0.30 * pred_c
        eval_preds += fold_preds / 5.0
        
    return eval_preds


# EVOLVE-BLOCK-END
# =====================================================================
