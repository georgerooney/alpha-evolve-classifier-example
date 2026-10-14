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
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.decomposition import TruncatedSVD
    
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
        
    # Clinical Charlson Comorbidity Index (CCI) Score
    train_charlson = (
        train['cm_mi'] + train['cm_chf'] + train['cm_pvd'] + train['cm_cerebrovascular'] +
        train['cm_dementia'] + train['cm_pulmonary'] + train['cm_rheumatic'] +
        train['cm_peptic_ulcer'] + train['cm_liver'] +
        2.0 * (train['cm_hemiplegia'] + train['cm_renal'] + train['cm_diabetes_complicated'] + train['cm_cancer']) +
        6.0 * (train['cm_metastatic'] + train['cm_hiv'])
    )
    eval_charlson = (
        eval_cols['cm_mi'] + eval_cols['cm_chf'] + eval_cols['cm_pvd'] + eval_cols['cm_cerebrovascular'] +
        eval_cols['cm_dementia'] + eval_cols['cm_pulmonary'] + eval_cols['cm_rheumatic'] +
        eval_cols['cm_peptic_ulcer'] + eval_cols['cm_liver'] +
        2.0 * (eval_cols['cm_hemiplegia'] + eval_cols['cm_renal'] + eval_cols['cm_diabetes_complicated'] + eval_cols['cm_cancer']) +
        6.0 * (eval_cols['cm_metastatic'] + eval_cols['cm_hiv'])
    )
    train['charlson_score'] = train_charlson
    eval_cols['charlson_score'] = eval_charlson
    extra_numeric.append('charlson_score')

    # Clinical LACE Readmission Risk Index
    def compute_lace(los, adm_type, cci, ed_visits):
        l = np.zeros_like(los, dtype=np.float64)
        l[los == 1] = 1.0
        l[los == 2] = 2.0
        l[los == 3] = 3.0
        l[(los >= 4) & (los <= 6)] = 4.0
        l[(los >= 7) & (los <= 13)] = 5.0
        l[los >= 14] = 7.0
        a = np.where(np.isin(adm_type, ['1', '2']), 3.0, 0.0)
        c = np.minimum(cci, 5.0)
        e = np.minimum(ed_visits, 4.0)
        return l + a + c + e

    train['lace_score'] = compute_lace(train['time_in_hospital'], train['admission_type_id'], train_charlson, train['number_emergency'])
    eval_cols['lace_score'] = compute_lace(eval_cols['time_in_hospital'], eval_cols['admission_type_id'], eval_charlson, eval_cols['number_emergency'])
    extra_numeric.append('lace_score')

    # 3. Grouped ICD-9 chapters & Syndemic Pair
    for col in diag_cols:
        grouped_col = col + "_grp"
        train[grouped_col] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_cols[grouped_col] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(grouped_col)

    train['diag_pair_12'] = np.array([f"{d1}_{d2}" for d1, d2 in zip(train['diag_1_grp'], train['diag_2_grp'])], dtype=object)
    eval_cols['diag_pair_12'] = np.array([f"{d1}_{d2}" for d1, d2 in zip(eval_cols['diag_1_grp'], eval_cols['diag_2_grp'])], dtype=object)
    extra_categorical.append('diag_pair_12')

    tr_pair_enc, ev_pair_enc = oof_target_encode(train['diag_pair_12'], y_train, eval_cols['diag_pair_12'], n_folds=5, smoothing=20.0, seed=42)
    train['diag_pair_12_te'] = tr_pair_enc
    eval_cols['diag_pair_12_te'] = ev_pair_enc
    extra_numeric.append('diag_pair_12_te')

    # 3-digit ICD9 category target encodings
    def get_icd9_cat3(val):
        if val is None or val == 'None' or val == '?':
            return 'missing'
        s = str(val).split('.')[0].strip()
        return s if s else 'missing'

    for col in diag_cols:
        cat3_col = col + "_cat3"
        train_cat3 = np.array([get_icd9_cat3(x) for x in train[col]], dtype=object)
        eval_cat3 = np.array([get_icd9_cat3(x) for x in eval_cols[col]], dtype=object)
        tr_cat3_enc, ev_cat3_enc = oof_target_encode(train_cat3, y_train, eval_cat3, n_folds=5, smoothing=20.0, seed=42)
        te_cat3_col = cat3_col + "_te"
        train[te_cat3_col] = tr_cat3_enc
        eval_cols[te_cat3_col] = ev_cat3_enc
        extra_numeric.append(te_cat3_col)
        
    # 4. Target encoding of high-cardinality columns
    te_cols = ['diag_1', 'diag_2', 'diag_3', 'medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'payer_code']
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

    train['acute_prior_ratio'] = (train['number_emergency'] + train['number_inpatient']) / (train_total_visits + 1.0)
    eval_cols['acute_prior_ratio'] = (eval_cols['number_emergency'] + eval_cols['number_inpatient']) / (eval_total_visits + 1.0)
    extra_numeric.append('acute_prior_ratio')

    train['has_prior_inpatient'] = (train['number_inpatient'] > 0).astype(np.float64)
    eval_cols['has_prior_inpatient'] = (eval_cols['number_inpatient'] > 0).astype(np.float64)
    extra_numeric.append('has_prior_inpatient')

    train['frequent_inpatient'] = (train['number_inpatient'] >= 2).astype(np.float64)
    eval_cols['frequent_inpatient'] = (eval_cols['number_inpatient'] >= 2).astype(np.float64)
    extra_numeric.append('frequent_inpatient')

    # Clinical resource intensity ratios
    train['med_intensity'] = train['num_medications'] / np.maximum(train['time_in_hospital'], 1.0)
    eval_cols['med_intensity'] = eval_cols['num_medications'] / np.maximum(eval_cols['time_in_hospital'], 1.0)
    extra_numeric.append('med_intensity')

    train['lab_intensity'] = train['num_lab_procedures'] / np.maximum(train['time_in_hospital'], 1.0)
    eval_cols['lab_intensity'] = eval_cols['num_lab_procedures'] / np.maximum(eval_cols['time_in_hospital'], 1.0)
    extra_numeric.append('lab_intensity')

    train['proc_intensity'] = train['num_procedures'] / np.maximum(train['time_in_hospital'], 1.0)
    eval_cols['proc_intensity'] = eval_cols['num_procedures'] / np.maximum(eval_cols['time_in_hospital'], 1.0)
    extra_numeric.append('proc_intensity')

    train['meds_per_diag'] = train['num_medications'] / np.maximum(train['number_diagnoses'], 1.0)
    eval_cols['meds_per_diag'] = eval_cols['num_medications'] / np.maximum(eval_cols['number_diagnoses'], 1.0)
    extra_numeric.append('meds_per_diag')

    # High-risk discharge disposition indicators
    train['discharge_to_facility'] = np.isin(train['discharge_disposition_id'], ['2', '3', '4', '5', '6', '22']).astype(np.float64)
    eval_cols['discharge_to_facility'] = np.isin(eval_cols['discharge_disposition_id'], ['2', '3', '4', '5', '6', '22']).astype(np.float64)
    extra_numeric.append('discharge_to_facility')

    train['discharge_ama'] = (train['discharge_disposition_id'] == '7').astype(np.float64)
    eval_cols['discharge_ama'] = (eval_cols['discharge_disposition_id'] == '7').astype(np.float64)
    extra_numeric.append('discharge_ama')

    # Glycemic crisis & Therapeutic inertia indicators
    glu_abn_tr = np.isin(train['max_glu_serum'], ['>200', '>300'])
    glu_abn_ev = np.isin(eval_cols['max_glu_serum'], ['>200', '>300'])
    a1c_abn_tr = np.isin(train['A1Cresult'], ['>7', '>8'])
    a1c_abn_ev = np.isin(eval_cols['A1Cresult'], ['>7', '>8'])

    train['glycemic_crisis'] = (glu_abn_tr | a1c_abn_tr).astype(np.float64)
    eval_cols['glycemic_crisis'] = (glu_abn_ev | a1c_abn_ev).astype(np.float64)
    extra_numeric.append('glycemic_crisis')

    train['therapeutic_inertia'] = ((train['glycemic_crisis'] == 1.0) & (train['change'] == 'No')).astype(np.float64)
    eval_cols['therapeutic_inertia'] = ((eval_cols['glycemic_crisis'] == 1.0) & (eval_cols['change'] == 'No')).astype(np.float64)
    extra_numeric.append('therapeutic_inertia')

    train['no_labs_done'] = ((train['max_glu_serum'] == 'None') & (train['A1Cresult'] == 'None')).astype(np.float64)
    eval_cols['no_labs_done'] = ((eval_cols['max_glu_serum'] == 'None') & (eval_cols['A1Cresult'] == 'None')).astype(np.float64)
    extra_numeric.append('no_labs_done')
    
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
    
    for col in med_cols:
        if col in train:
            train_active_meds += (train[col] != "No") & (train[col] != None)
            eval_active_meds += (eval_cols[col] != "No") & (eval_cols[col] != None)
            train_change_meds += (train[col] == "Up") | (train[col] == "Down")
            eval_change_meds += (eval_cols[col] == "Up") | (eval_cols[col] == "Down")
            
    train['active_meds_count'] = train_active_meds
    eval_cols['active_meds_count'] = eval_active_meds
    extra_numeric.append('active_meds_count')
    
    train['change_meds_count'] = train_change_meds
    eval_cols['change_meds_count'] = eval_change_meds
    extra_numeric.append('change_meds_count')
    
    # 7. Insulin change interaction
    train['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(train['insulin'], train['change'])], dtype=object)
    eval_cols['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(eval_cols['insulin'], eval_cols['change'])], dtype=object)
    extra_categorical.append('insulin_change')

    train['insulin_increased'] = (train['insulin'] == 'Up').astype(np.float64)
    eval_cols['insulin_increased'] = (eval_cols['insulin'] == 'Up').astype(np.float64)
    extra_numeric.append('insulin_increased')

    train['insulin_decreased'] = (train['insulin'] == 'Down').astype(np.float64)
    eval_cols['insulin_decreased'] = (eval_cols['insulin'] == 'Down').astype(np.float64)
    extra_numeric.append('insulin_decreased')

    train_oral_meds = train_active_meds - (train['insulin'] != 'No')
    eval_oral_meds = eval_active_meds - (eval_cols['insulin'] != 'No')
    train['combination_therapy'] = (((train['insulin'] != 'No')) & (train_oral_meds > 0)).astype(np.float64)
    eval_cols['combination_therapy'] = (((eval_cols['insulin'] != 'No')) & (eval_oral_meds > 0)).astype(np.float64)
    extra_numeric.append('combination_therapy')

    train['polypharmacy'] = (train['num_medications'] >= 15).astype(np.float64)
    eval_cols['polypharmacy'] = (eval_cols['num_medications'] >= 15).astype(np.float64)
    extra_numeric.append('polypharmacy')
    
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

    # 8b. Latent Clinical Phenotype Manifold (Unsupervised Transductive TruncatedSVD)
    svd_feats_tr, svd_feats_ev = [], []
    for k in train_cm:
        svd_feats_tr.append(train[k].reshape(-1, 1))
        svd_feats_ev.append(eval_cols[k].reshape(-1, 1))

    for ch in ['circulatory', 'respiratory', 'digestive', 'diabetes', 'injury', 'musculoskeletal', 'genitourinary', 'neoplasms', 'other']:
        for c in diag_cols:
            svd_feats_tr.append((train[c + '_grp'] == ch).astype(np.float64).reshape(-1, 1))
            svd_feats_ev.append((eval_cols[c + '_grp'] == ch).astype(np.float64).reshape(-1, 1))

    for feat in ['glycemic_crisis', 'therapeutic_inertia', 'no_labs_done', 'combination_therapy']:
        svd_feats_tr.append(train[feat].reshape(-1, 1))
        svd_feats_ev.append(eval_cols[feat].reshape(-1, 1))

    for c in ['insulin', 'metformin', 'glipizide', 'glyburide', 'glimepiride', 'pioglitazone', 'rosiglitazone']:
        if c in train:
            svd_feats_tr.append(((train[c] != 'No') & (train[c] != None)).astype(np.float64).reshape(-1, 1))
            svd_feats_ev.append(((eval_cols[c] != 'No') & (eval_cols[c] != None)).astype(np.float64).reshape(-1, 1))

    mat_tr = np.hstack(svd_feats_tr)
    mat_ev = np.hstack(svd_feats_ev)
    mat_all = np.vstack([mat_tr, mat_ev])

    svd = TruncatedSVD(n_components=5, random_state=42)
    all_svd = svd.fit_transform(mat_all)
    n_train_rows = len(y_train)
    tr_svd, ev_svd = all_svd[:n_train_rows], all_svd[n_train_rows:]

    for i in range(5):
        s_col = f"pheno_svd_{i}"
        train[s_col] = tr_svd[:, i]
        eval_cols[s_col] = ev_svd[:, i]
        extra_numeric.append(s_col)

    # 9. Encode datasets
    encoder = TabularEncoder(min_count=50)
    encoder.fit(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)
    
    # 10. Fit Stratified 5-Fold Dual-Engine Ensemble (LightGBM + XGBoost)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_preds = np.zeros(x_eval.shape[0], dtype=np.float64)
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        fold_x_train, fold_y_train = x_train[train_idx], y_train[train_idx]
        
        # Engine A: Leaf-wise LightGBM with categorical feature support
        lgb_model = lgb.LGBMClassifier(
            n_estimators=190,
            learning_rate=0.018,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_lambda=0.1,
            cat_smooth=50.0,
            random_state=42 + fold_idx,
            n_jobs=1,
            verbose=-1,
        )
        lgb_model.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        lgb_fold_preds = lgb_model.predict_proba(x_eval)[:, 1]

        # Engine B: Depth-wise histogram XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.025,
            max_depth=5,
            min_child_weight=15,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_lambda=1.0,
            tree_method='hist',
            eval_metric='logloss',
            random_state=42 + fold_idx,
            n_jobs=1,
        )
        xgb_model.fit(fold_x_train, fold_y_train)
        xgb_fold_preds = xgb_model.predict_proba(x_eval)[:, 1]

        eval_preds += (0.60 * lgb_fold_preds + 0.40 * xgb_fold_preds) / 5.0
        
    return eval_preds


# EVOLVE-BLOCK-END
# =====================================================================
