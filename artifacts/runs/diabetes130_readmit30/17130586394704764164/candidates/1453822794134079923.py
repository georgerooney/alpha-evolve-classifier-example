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
    train_ext = dict(train)
    eval_ext = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    prior = float(np.mean(y_train))

    # 1. Comorbidities extraction (Charlson mapping)
    cm_train = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k, v in cm_train.items():
        train_ext[k] = v
        eval_ext[k] = cm_eval[k]
        extra_numeric.append(k)

    # Comorbidity interactions
    train_ext['cm_cardiorenal'] = (cm_train['cm_chf'] * cm_train['cm_renal']).astype(np.float64)
    eval_ext['cm_cardiorenal'] = (cm_eval['cm_chf'] * cm_eval['cm_renal']).astype(np.float64)
    extra_numeric.append('cm_cardiorenal')

    train_ext['cm_chf_copd'] = (cm_train['cm_chf'] * cm_train['cm_pulmonary']).astype(np.float64)
    eval_ext['cm_chf_copd'] = (cm_eval['cm_chf'] * cm_eval['cm_pulmonary']).astype(np.float64)
    extra_numeric.append('cm_chf_copd')

    train_ext['cm_renal_diab'] = (cm_train['cm_renal'] * cm_train['cm_diabetes_complicated']).astype(np.float64)
    eval_ext['cm_renal_diab'] = (cm_eval['cm_renal'] * cm_eval['cm_diabetes_complicated']).astype(np.float64)
    extra_numeric.append('cm_renal_diab')

    train_ext['cm_frail_prior'] = ((cm_train['cm_count'] >= 2.0).astype(np.float64) *
                                  (np.nan_to_num(train['number_inpatient'], nan=0.0) > 0).astype(np.float64))
    eval_ext['cm_frail_prior'] = ((cm_eval['cm_count'] >= 2.0).astype(np.float64) *
                                 (np.nan_to_num(eval_cols['number_inpatient'], nan=0.0) > 0).astype(np.float64))
    extra_numeric.append('cm_frail_prior')

    # 2. Healthcare utilization & intensity metrics
    for cols in (train_ext, eval_ext):
        inp = np.nan_to_num(cols['number_inpatient'], nan=0.0)
        emg = np.nan_to_num(cols['number_emergency'], nan=0.0)
        outp = np.nan_to_num(cols['number_outpatient'], nan=0.0)
        los = np.maximum(np.nan_to_num(cols['time_in_hospital'], nan=1.0), 1.0)
        meds = np.nan_to_num(cols['num_medications'], nan=0.0)
        labs = np.nan_to_num(cols['num_lab_procedures'], nan=0.0)
        procs = np.nan_to_num(cols['num_procedures'], nan=0.0)
        diags = np.maximum(np.nan_to_num(cols['number_diagnoses'], nan=1.0), 1.0)

        tot_prior = inp + emg + outp
        cols['total_prior_visits'] = tot_prior
        cols['log_prior_visits'] = np.log1p(tot_prior)
        cols['log_prior_inpatient'] = np.log1p(inp)
        cols['has_prior_inpatient'] = (inp > 0.0).astype(np.float64)
        cols['frequent_inpatient'] = (inp >= 2.0).astype(np.float64)
        cols['has_prior_emergency'] = (emg > 0.0).astype(np.float64)
        cols['inpatient_ratio'] = inp / (tot_prior + 1.0)
        cols['emergency_ratio'] = emg / (tot_prior + 1.0)

        cols['meds_per_day'] = meds / los
        cols['labs_per_day'] = labs / los
        cols['procs_per_day'] = procs / los
        cols['diags_per_day'] = diags / los
        cols['meds_per_diag'] = meds / diags
        cols['clinical_intensity'] = (meds + labs + procs) / los

    for feat in ['total_prior_visits', 'log_prior_visits', 'log_prior_inpatient',
                 'has_prior_inpatient', 'frequent_inpatient', 'has_prior_emergency',
                 'inpatient_ratio', 'emergency_ratio', 'meds_per_day', 'labs_per_day',
                 'procs_per_day', 'diags_per_day', 'meds_per_diag', 'clinical_intensity']:
        extra_numeric.append(feat)

    # 3. Medication dynamics & polypharmacy
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]

    for cols in (train_ext, eval_ext):
        n = len(cols['time_in_hospital'])
        active_cnt = np.zeros(n, dtype=np.float64)
        up_cnt = np.zeros(n, dtype=np.float64)
        down_cnt = np.zeros(n, dtype=np.float64)
        steady_cnt = np.zeros(n, dtype=np.float64)

        for m in med_cols:
            if m in cols:
                arr = cols[m]
                active_cnt += ((arr != 'No') & (arr != None)).astype(np.float64)
                up_cnt += (arr == 'Up').astype(np.float64)
                down_cnt += (arr == 'Down').astype(np.float64)
                steady_cnt += (arr == 'Steady').astype(np.float64)

        cols['num_active_meds'] = active_cnt
        cols['num_meds_up'] = up_cnt
        cols['num_meds_down'] = down_cnt
        cols['num_meds_steady'] = steady_cnt
        cols['meds_any_change'] = up_cnt + down_cnt
        cols['polypharmacy_flag'] = (active_cnt >= 2.0).astype(np.float64)

        ins = cols.get('insulin', np.array([None] * n))
        cols['insulin_active'] = ((ins != 'No') & (ins != None)).astype(np.float64)
        cols['insulin_up'] = (ins == 'Up').astype(np.float64)
        cols['insulin_down'] = (ins == 'Down').astype(np.float64)
        cols['insulin_steady'] = (ins == 'Steady').astype(np.float64)

        met = cols.get('metformin', np.array([None] * n))
        cols['metformin_active'] = ((met != 'No') & (met != None)).astype(np.float64)
        cols['insulin_and_metformin'] = cols['insulin_active'] * cols['metformin_active']

    for feat in ['num_active_meds', 'num_meds_up', 'num_meds_down', 'num_meds_steady',
                 'meds_any_change', 'polypharmacy_flag', 'insulin_active', 'insulin_up',
                 'insulin_down', 'insulin_steady', 'metformin_active', 'insulin_and_metformin']:
        extra_numeric.append(feat)

    # 4. Age and Diagnostic Tests
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    for cols in (train_ext, eval_ext):
        ages = cols['age']
        age_num = np.array([age_map.get(a, 65.0) for a in ages], dtype=np.float64)
        cols['age_numeric'] = age_num
        cols['is_elderly'] = (age_num >= 75.0).astype(np.float64)

        a1c = cols.get('A1Cresult', None)
        if a1c is not None:
            cols['a1c_tested'] = np.array([1.0 if (x is not None and x != 'None') else 0.0 for x in a1c], dtype=np.float64)
            cols['a1c_high'] = np.array([1.0 if x in ('>7', '>8') else 0.0 for x in a1c], dtype=np.float64)
        else:
            cols['a1c_tested'] = np.zeros(len(ages), dtype=np.float64)
            cols['a1c_high'] = np.zeros(len(ages), dtype=np.float64)

        glu = cols.get('max_glu_serum', None)
        if glu is not None:
            cols['glu_tested'] = np.array([1.0 if (x is not None and x != 'None') else 0.0 for x in glu], dtype=np.float64)
            cols['glu_high'] = np.array([1.0 if x in ('>200', '>300') else 0.0 for x in glu], dtype=np.float64)
        else:
            cols['glu_tested'] = np.zeros(len(ages), dtype=np.float64)
            cols['glu_high'] = np.zeros(len(ages), dtype=np.float64)

    for feat in ['age_numeric', 'is_elderly', 'a1c_tested', 'a1c_high', 'glu_tested', 'glu_high']:
        extra_numeric.append(feat)

    # 5. Admission and Discharge specific flags
    disp_tr = train['discharge_disposition_id']
    disp_ev = eval_cols['discharge_disposition_id']
    train_ext['disp_is_home'] = (disp_tr == '1').astype(np.float64)
    eval_ext['disp_is_home'] = (disp_ev == '1').astype(np.float64)
    train_ext['disp_is_snf'] = (disp_tr == '3').astype(np.float64)
    eval_ext['disp_is_snf'] = (disp_ev == '3').astype(np.float64)
    train_ext['disp_is_home_health'] = (disp_tr == '6').astype(np.float64)
    eval_ext['disp_is_home_health'] = (disp_ev == '6').astype(np.float64)

    adm_tr = train['admission_type_id']
    adm_ev = eval_cols['admission_type_id']
    train_ext['adm_is_emergency'] = (adm_tr == '1').astype(np.float64)
    eval_ext['adm_is_emergency'] = (adm_ev == '1').astype(np.float64)
    train_ext['adm_is_urgent'] = (adm_tr == '2').astype(np.float64)
    eval_ext['adm_is_urgent'] = (adm_ev == '2').astype(np.float64)

    for feat in ['disp_is_home', 'disp_is_snf', 'disp_is_home_health', 'adm_is_emergency', 'adm_is_urgent']:
        extra_numeric.append(feat)

    # 6. Diagnosis Chapter Grouping
    diag1_grp_tr = np.array([icd9_group(c) for c in train['diag_1']], dtype=object)
    diag1_grp_ev = np.array([icd9_group(c) for c in eval_cols['diag_1']], dtype=object)
    diag2_grp_tr = np.array([icd9_group(c) for c in train['diag_2']], dtype=object)
    diag2_grp_ev = np.array([icd9_group(c) for c in eval_cols['diag_2']], dtype=object)
    diag3_grp_tr = np.array([icd9_group(c) for c in train['diag_3']], dtype=object)
    diag3_grp_ev = np.array([icd9_group(c) for c in eval_cols['diag_3']], dtype=object)

    train_ext['diag1_group'] = diag1_grp_tr
    eval_ext['diag1_group'] = diag1_grp_ev
    train_ext['diag2_group'] = diag2_grp_tr
    eval_ext['diag2_group'] = diag2_grp_ev
    train_ext['diag3_group'] = diag3_grp_tr
    eval_ext['diag3_group'] = diag3_grp_ev
    extra_categorical.extend(['diag1_group', 'diag2_group', 'diag3_group'])

    disp_diag1_tr = np.array([f"{d}_{g}" for d, g in zip(disp_tr, diag1_grp_tr)], dtype=object)
    disp_diag1_ev = np.array([f"{d}_{g}" for d, g in zip(disp_ev, diag1_grp_ev)], dtype=object)
    train_ext['disp_diag1'] = disp_diag1_tr
    eval_ext['disp_diag1'] = disp_diag1_ev
    extra_categorical.append('disp_diag1')

    # 7. Out-of-fold target encoding of high-cardinality clinical variables
    te_specs = [
        ('diag_1', train['diag_1'], eval_cols['diag_1']),
        ('diag_2', train['diag_2'], eval_cols['diag_2']),
        ('diag_3', train['diag_3'], eval_cols['diag_3']),
        ('discharge_disp', disp_tr, disp_ev),
        ('admission_src', train['admission_source_id'], eval_cols['admission_source_id']),
        ('med_specialty', train['medical_specialty'], eval_cols['medical_specialty']),
        ('disp_diag1', disp_diag1_tr, disp_diag1_ev),
    ]

    for col_name, tr_c, ev_c in te_specs:
        tr_te, ev_te = oof_target_encode(tr_c, y_train, ev_c, n_folds=5, smoothing=20.0, seed=42)
        te_key = f'te_{col_name}'
        train_ext[te_key] = np.nan_to_num(tr_te, nan=prior).astype(np.float64)
        eval_ext[te_key] = np.nan_to_num(ev_te, nan=prior).astype(np.float64)
        extra_numeric.append(te_key)

    # 8. Crazy Idea: Latent Clinical Phenotype (SVD) Embeddings
    svd_cols_tr = []
    svd_cols_ev = []
    for cm_key in ['cm_chf', 'cm_renal', 'cm_diabetes_complicated', 'cm_pulmonary',
                   'cm_mi', 'cm_pvd', 'cm_cerebrovascular', 'cm_cancer', 'cm_liver']:
        svd_cols_tr.append(cm_train[cm_key])
        svd_cols_ev.append(cm_eval[cm_key])

    for f_key in ['insulin_active', 'insulin_up', 'insulin_down', 'metformin_active',
                  'polypharmacy_flag', 'has_prior_inpatient', 'frequent_inpatient',
                  'has_prior_emergency', 'disp_is_home', 'disp_is_snf', 'disp_is_home_health',
                  'adm_is_emergency', 'a1c_tested', 'a1c_high', 'glu_tested']:
        svd_cols_tr.append(train_ext[f_key])
        svd_cols_ev.append(eval_ext[f_key])

    for g_val in ['circulatory', 'respiratory', 'diabetes', 'digestive', 'genitourinary']:
        svd_cols_tr.append((diag1_grp_tr == g_val).astype(np.float64))
        svd_cols_ev.append((diag1_grp_ev == g_val).astype(np.float64))

    mat_tr = np.column_stack(svd_cols_tr)
    mat_ev = np.column_stack(svd_cols_ev)

    try:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=6, random_state=42)
        svd_tr = svd.fit_transform(mat_tr)
        svd_ev = svd.transform(mat_ev)
        for i in range(6):
            k = f'svd_pheno_{i}'
            train_ext[k] = svd_tr[:, i].astype(np.float64)
            eval_ext[k] = svd_ev[:, i].astype(np.float64)
            extra_numeric.append(k)
    except Exception:
        pass

    # 9. Encoding and Feature Transformation
    encoder = TabularEncoder(min_count=50)
    encoder.fit(task, train_ext, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_ext)
    x_eval = encoder.transform(eval_ext)

    # 10. Multi-Perspective Diverse LightGBM Ensemble
    model1 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=120,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=0.1,
        reg_lambda=2.0,
        cat_smooth=35.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p1 = model1.predict_proba(x_eval)[:, 1]

    model2 = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.025,
        num_leaves=18,
        max_depth=5,
        min_child_samples=160,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=0.5,
        reg_lambda=5.0,
        cat_smooth=60.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    model2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p2 = model2.predict_proba(x_eval)[:, 1]

    model3 = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.035,
        num_leaves=42,
        max_depth=7,
        min_child_samples=90,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=1.0,
        cat_smooth=25.0,
        random_state=777,
        n_jobs=1,
        verbose=-1,
    )
    model3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p3 = model3.predict_proba(x_eval)[:, 1]

    preds = 0.40 * p1 + 0.35 * p2 + 0.25 * p3
    preds = np.clip(np.nan_to_num(preds, nan=prior), 0.0, 1.0)
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
