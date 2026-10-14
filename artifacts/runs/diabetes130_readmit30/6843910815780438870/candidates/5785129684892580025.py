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
import lightgbm as lgb
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def get_icd9_prefix(val):
    if val is None or val == '' or val == 'None' or val == '?':
        return 'missing'
    return str(val).strip().split('.')[0]


def add_features(task, train, eval_cols, y_train):
    # Shallow copy to avoid mutating inputs directly
    train_feat = dict(train)
    eval_feat = dict(eval_cols)
    
    extra_numeric = []
    extra_categorical = []
    
    # 1. Comorbidities
    train_cm = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    eval_cm = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    
    for k in train_cm.keys():
        train_feat[f'cm_{k}'] = train_cm[k]
        eval_feat[f'cm_{k}'] = eval_cm[k]
        extra_numeric.append(f'cm_{k}')
        
    # Specific clinical interactions
    train_feat['cm_chf_renal'] = train_cm['cm_chf'] * train_cm['cm_renal']
    eval_feat['cm_chf_renal'] = eval_cm['cm_chf'] * eval_cm['cm_renal']
    extra_numeric.append('cm_chf_renal')
    
    train_feat['cm_diab_renal'] = train_cm['cm_diabetes_complicated'] * train_cm['cm_renal']
    eval_feat['cm_diab_renal'] = eval_cm['cm_diabetes_complicated'] * eval_cm['cm_renal']
    extra_numeric.append('cm_diab_renal')
        
    # 2. Weighted Charlson Comorbidity Index
    train_charlson = (
        1.0 * (train_cm['cm_mi'] + train_cm['cm_chf'] + train_cm['cm_pvd'] + train_cm['cm_cerebrovascular'] + 
               train_cm['cm_dementia'] + train_cm['cm_pulmonary'] + train_cm['cm_rheumatic'] + 
               train_cm['cm_peptic_ulcer'] + train_cm['cm_diabetes_complicated']) +
        2.0 * (train_cm['cm_hemiplegia'] + train_cm['cm_renal'] + train_cm['cm_cancer']) +
        3.0 * (train_cm['cm_liver']) +
        6.0 * (train_cm['cm_metastatic'] + train_cm['cm_hiv'])
    )
    eval_charlson = (
        1.0 * (eval_cm['cm_mi'] + eval_cm['cm_chf'] + eval_cm['cm_pvd'] + eval_cm['cm_cerebrovascular'] + 
               eval_cm['cm_dementia'] + eval_cm['cm_pulmonary'] + eval_cm['cm_rheumatic'] + 
               eval_cm['cm_peptic_ulcer'] + eval_cm['cm_diabetes_complicated']) +
        2.0 * (eval_cm['cm_hemiplegia'] + eval_cm['cm_renal'] + eval_cm['cm_cancer']) +
        3.0 * (eval_cm['cm_liver']) +
        6.0 * (eval_cm['cm_metastatic'] + eval_cm['cm_hiv'])
    )
    train_feat['charlson_index'] = train_charlson
    eval_feat['charlson_index'] = eval_charlson
    extra_numeric.append('charlson_index')

    train_feat['charlson_inpatient'] = train_charlson * np.log1p(train['number_inpatient'])
    eval_feat['charlson_inpatient'] = eval_charlson * np.log1p(eval_cols['number_inpatient'])
    extra_numeric.append('charlson_inpatient')

    # 3. ICD9 groups and 3-digit prefixes
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    for col in diag_cols:
        train_feat[f'{col}_grp'] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_feat[f'{col}_grp'] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(f'{col}_grp')
        
        train_feat[f'{col}_prefix'] = np.array([get_icd9_prefix(x) for x in train[col]], dtype=object)
        eval_feat[f'{col}_prefix'] = np.array([get_icd9_prefix(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(f'{col}_prefix')
        
    # 4. Clinical Interactions: Discharge Disposition with Primary Diag & Admission Source
    train_feat['dis_adm'] = np.array([
        f"{d}_{a}" for d, a in zip(train['discharge_disposition_id'], train['admission_source_id'])
    ], dtype=object)
    eval_feat['dis_adm'] = np.array([
        f"{d}_{a}" for d, a in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])
    ], dtype=object)
    extra_categorical.append('dis_adm')

    train_feat['dis_diag1'] = np.array([
        f"{d}_{p}" for d, p in zip(train['discharge_disposition_id'], train_feat['diag_1_prefix'])
    ], dtype=object)
    eval_feat['dis_diag1'] = np.array([
        f"{d}_{p}" for d, p in zip(eval_cols['discharge_disposition_id'], eval_feat['diag_1_prefix'])
    ], dtype=object)
    extra_categorical.append('dis_diag1')

    # 5. Latent Clinical Phenotyping via EHR Topic Modeling (TruncatedSVD)
    n_train = len(train['time_in_hospital'])
    n_eval = len(eval_cols['time_in_hospital'])
    train_docs = [
        f"d1_{train_feat['diag_1_prefix'][i]} d2_{train_feat['diag_2_prefix'][i]} d3_{train_feat['diag_3_prefix'][i]} "
        f"dis_{train['discharge_disposition_id'][i]} src_{train['admission_source_id'][i]} "
        f"typ_{train['admission_type_id'][i]} sp_{train['medical_specialty'][i]} ag_{train['age'][i]}"
        for i in range(n_train)
    ]
    eval_docs = [
        f"d1_{eval_feat['diag_1_prefix'][i]} d2_{eval_feat['diag_2_prefix'][i]} d3_{eval_feat['diag_3_prefix'][i]} "
        f"dis_{eval_cols['discharge_disposition_id'][i]} src_{eval_cols['admission_source_id'][i]} "
        f"typ_{eval_cols['admission_type_id'][i]} sp_{eval_cols['medical_specialty'][i]} ag_{eval_cols['age'][i]}"
        for i in range(n_eval)
    ]
    tfidf = TfidfVectorizer(min_df=5, binary=True, norm='l2', token_pattern=r"\S+")
    X_tr_tfidf = tfidf.fit_transform(train_docs)
    X_ev_tfidf = tfidf.transform(eval_docs)
    
    n_pheno_comp = 8
    svd = TruncatedSVD(n_components=n_pheno_comp, random_state=42)
    X_tr_svd = svd.fit_transform(X_tr_tfidf)
    X_ev_svd = svd.transform(X_ev_tfidf)
    for j in range(n_pheno_comp):
        pheno_name = f'pheno_axis_{j}'
        train_feat[pheno_name] = X_tr_svd[:, j].astype(np.float64)
        eval_feat[pheno_name] = X_ev_svd[:, j].astype(np.float64)
        extra_numeric.append(pheno_name)

    # 6. Target encoding of high-cardinality & interaction variables
    high_card_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_prefix', 'diag_2_prefix', 'diag_3_prefix',
        'medical_specialty', 'discharge_disposition_id', 
        'admission_source_id', 'admission_type_id',
        'payer_code', 'dis_adm', 'dis_diag1'
    ]
    for col in high_card_cols:
        tr_enc, ev_enc = oof_target_encode(train_feat[col], y_train, eval_feat[col], n_folds=5, smoothing=20.0, seed=42)
        train_feat[f'{col}_te'] = tr_enc
        eval_feat[f'{col}_te'] = ev_enc
        extra_numeric.append(f'{col}_te')
        
    # 5. Medication features
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    existing_med_cols = [c for c in med_cols if c in train]
    
    def get_med_features(data_dict):
        n_rows = len(data_dict['time_in_hospital'])
        num_up = np.zeros(n_rows, dtype=np.float64)
        num_down = np.zeros(n_rows, dtype=np.float64)
        num_steady = np.zeros(n_rows, dtype=np.float64)
        for col in existing_med_cols:
            arr = data_dict[col]
            num_up += (arr == 'Up').astype(np.float64)
            num_down += (arr == 'Down').astype(np.float64)
            num_steady += (arr == 'Steady').astype(np.float64)
            
        num_active = num_up + num_down + num_steady
        num_change = num_up + num_down
        
        # Insulin specific
        insulin_arr = data_dict.get('insulin', np.array(['No'] * n_rows))
        insulin_up = (insulin_arr == 'Up').astype(np.float64)
        insulin_down = (insulin_arr == 'Down').astype(np.float64)
        insulin_steady = (insulin_arr == 'Steady').astype(np.float64)
        insulin_active = (insulin_arr != 'No').astype(np.float64)
        
        return {
            'med_num_up': num_up,
            'med_num_down': num_down,
            'med_num_steady': num_steady,
            'med_num_active': num_active,
            'med_num_change': num_change,
            'insulin_up': insulin_up,
            'insulin_down': insulin_down,
            'insulin_steady': insulin_steady,
            'insulin_active': insulin_active
        }
        
    train_med_feats = get_med_features(train)
    eval_med_feats = get_med_features(eval_cols)
    for k in train_med_feats.keys():
        train_feat[k] = train_med_feats[k]
        eval_feat[k] = eval_med_feats[k]
        extra_numeric.append(k)
        
    # 7. Age to Continuous Numeric & Acute Utilization Trajectory
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    train_age_num = np.array([age_map.get(str(x), 65.0) for x in train['age']], dtype=np.float64)
    eval_age_num = np.array([age_map.get(str(x), 65.0) for x in eval_cols['age']], dtype=np.float64)
    train_feat['age_num'] = train_age_num
    eval_feat['age_num'] = eval_age_num
    extra_numeric.append('age_num')

    train_feat['age_inpatient'] = train_age_num * train['number_inpatient']
    eval_feat['age_inpatient'] = eval_age_num * eval_cols['number_inpatient']
    extra_numeric.append('age_inpatient')

    # 8. Utilisation and clinical severity interaction indices
    def add_custom_numeric(data_dict, target_dict):
        outpatient = data_dict.get('number_outpatient', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        emergency = data_dict.get('number_emergency', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        inpatient = data_dict.get('number_inpatient', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        
        util_index = outpatient + 2.0 * emergency + 3.0 * inpatient
        util_any = ((outpatient > 0) | (emergency > 0) | (inpatient > 0)).astype(np.float64)
        
        total_visits = outpatient + emergency + inpatient
        inpatient_ratio = inpatient / (total_visits + 1.0)
        has_inpatient = (inpatient > 0).astype(np.float64)
        frequent_inpatient = (inpatient >= 2).astype(np.float64)
        
        log_inpatient = np.log1p(inpatient)
        log_emergency = np.log1p(emergency)
        
        lab_proc = data_dict.get('num_lab_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        proc = data_dict.get('num_procedures', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        meds = data_dict.get('num_medications', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        diags = data_dict.get('number_diagnoses', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        los = data_dict.get('time_in_hospital', np.zeros_like(data_dict['time_in_hospital'], dtype=float))
        
        severity_index = lab_proc + 2.0 * proc + meds + 5.0 * diags
        los_med_ratio = los / (meds + 1.0)
        lab_per_day = lab_proc / (los + 0.1)
        med_per_day = meds / (los + 0.1)
        meds_per_diag = meds / (diags + 1.0)

        # High-risk discharge and admission categories
        disch = data_dict['discharge_disposition_id']
        admsrc = data_dict['admission_source_id']
        is_disch_home = (disch == '1').astype(np.float64)
        is_disch_snf = (disch == '3').astype(np.float64)
        is_disch_homehealth = (disch == '6').astype(np.float64)
        is_disch_ama = (disch == '7').astype(np.float64)
        is_adm_emergency = (admsrc == '7').astype(np.float64)

        # Glycemic and lab flags
        a1c = data_dict['A1Cresult']
        glu = data_dict['max_glu_serum']
        a1c_tested = (a1c != 'None').astype(np.float64)
        a1c_high = ((a1c == '>8') | (a1c == '>7')).astype(np.float64)
        glu_high = ((glu == '>200') | (glu == '>300')).astype(np.float64)

        cols_to_add = {
            'util_index': util_index,
            'util_any': util_any,
            'total_visits': total_visits,
            'inpatient_ratio': inpatient_ratio,
            'has_inpatient': has_inpatient,
            'frequent_inpatient': frequent_inpatient,
            'log_inpatient': log_inpatient,
            'log_emergency': log_emergency,
            'severity_index': severity_index,
            'los_med_ratio': los_med_ratio,
            'lab_per_day': lab_per_day,
            'med_per_day': med_per_day,
            'meds_per_diag': meds_per_diag,
            'is_disch_home': is_disch_home,
            'is_disch_snf': is_disch_snf,
            'is_disch_homehealth': is_disch_homehealth,
            'is_disch_ama': is_disch_ama,
            'is_adm_emergency': is_adm_emergency,
            'a1c_tested': a1c_tested,
            'a1c_high': a1c_high,
            'glu_high': glu_high,
        }
        for k, v in cols_to_add.items():
            target_dict[k] = v
        return list(cols_to_add.keys())
        
    added_num_cols = add_custom_numeric(train, train_feat)
    add_custom_numeric(eval_cols, eval_feat)
    extra_numeric.extend(added_num_cols)
    
    return train_feat, eval_feat, extra_numeric, extra_categorical


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    train_feat, eval_feat, extra_numeric, extra_categorical = add_features(task, train, eval_cols, y_train)
    
    encoder = TabularEncoder(min_count=30)
    encoder.fit(task, train_feat, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)

    # Multi-Engine Ensemble: Leaf-wise LightGBM + Level-wise XGBoost
    lgb_models = [
        lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.015,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=0.05,
            cat_smooth=40.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.012,
            num_leaves=45,
            min_child_samples=80,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_lambda=0.5,
            cat_smooth=60.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=180,
            learning_rate=0.018,
            num_leaves=18,
            min_child_samples=250,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=0.05,
            cat_smooth=30.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        )
    ]

    lgb_preds = np.zeros(len(x_eval), dtype=np.float64)
    for model in lgb_models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        lgb_preds += model.predict_proba(x_eval)[:, 1]
    lgb_preds /= len(lgb_models)

    xgb_model = xgb.XGBClassifier(
        n_estimators=180,
        learning_rate=0.025,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_lambda=1.5,
        min_child_weight=15,
        tree_method='hist',
        eval_metric='logloss',
        random_state=999,
        n_jobs=1,
    )
    xgb_model.fit(x_train, y_train)
    xgb_preds = xgb_model.predict_proba(x_eval)[:, 1]

    return 0.75 * lgb_preds + 0.25 * xgb_preds


# EVOLVE-BLOCK-END
# =====================================================================
