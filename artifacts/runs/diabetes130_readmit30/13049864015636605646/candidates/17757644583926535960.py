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
    train_data = dict(train)
    eval_data = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Charlson Comorbidities & Severity
    cm_tr = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_ev = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k in cm_tr:
        train_data[k] = cm_tr[k]
        eval_data[k] = cm_ev[k]
        extra_numeric.append(k)

    # Weighted Charlson Index calculation
    train_data['charlson_idx'] = (
        cm_tr['cm_mi'] + cm_tr['cm_chf'] + cm_tr['cm_pvd'] + cm_tr['cm_cerebrovascular']
        + cm_tr['cm_dementia'] + cm_tr['cm_pulmonary'] + cm_tr['cm_rheumatic'] + cm_tr['cm_peptic_ulcer']
        + cm_tr['cm_liver'] + 2.0 * cm_tr['cm_renal'] + 2.0 * cm_tr['cm_cancer']
        + 2.0 * cm_tr['cm_diabetes_complicated'] + 2.0 * cm_tr['cm_hemiplegia']
        + 4.0 * cm_tr['cm_metastatic'] + 4.0 * cm_tr['cm_hiv']
    )
    eval_data['charlson_idx'] = (
        cm_ev['cm_mi'] + cm_ev['cm_chf'] + cm_ev['cm_pvd'] + cm_ev['cm_cerebrovascular']
        + cm_ev['cm_dementia'] + cm_ev['cm_pulmonary'] + cm_ev['cm_rheumatic'] + cm_ev['cm_peptic_ulcer']
        + cm_ev['cm_liver'] + 2.0 * cm_ev['cm_renal'] + 2.0 * cm_ev['cm_cancer']
        + 2.0 * cm_ev['cm_diabetes_complicated'] + 2.0 * cm_ev['cm_hemiplegia']
        + 4.0 * cm_ev['cm_metastatic'] + 4.0 * cm_ev['cm_hiv']
    )
    extra_numeric.append('charlson_idx')

    train_data['cardiorenal'] = (cm_tr['cm_chf'] * cm_tr['cm_renal']).astype(np.float64)
    eval_data['cardiorenal'] = (cm_ev['cm_chf'] * cm_ev['cm_renal']).astype(np.float64)
    extra_numeric.append('cardiorenal')

    train_data['severe_multimorbid'] = (cm_tr['cm_count'] >= 3.0).astype(np.float64)
    eval_data['severe_multimorbid'] = (cm_ev['cm_count'] >= 3.0).astype(np.float64)
    extra_numeric.append('severe_multimorbid')

    # 2. ICD-9 Chapter Groupings
    for diag in ['diag_1', 'diag_2', 'diag_3']:
        grp_name = f'{diag}_grp'
        train_data[grp_name] = np.array([icd9_group(c) for c in train[diag]], dtype=object)
        eval_data[grp_name] = np.array([icd9_group(c) for c in eval_cols[diag]], dtype=object)
        extra_categorical.append(grp_name)

    train_data['diag_1_2_grp'] = np.array(
        [f"{g1}_{g2}" for g1, g2 in zip(train_data['diag_1_grp'], train_data['diag_2_grp'])],
        dtype=object,
    )
    eval_data['diag_1_2_grp'] = np.array(
        [f"{g1}_{g2}" for g1, g2 in zip(eval_data['diag_1_grp'], eval_data['diag_2_grp'])],
        dtype=object,
    )
    extra_categorical.append('diag_1_2_grp')

    # 3. Out-of-Fold Target Encodings
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'medical_specialty', 'payer_code',
    ]
    for c in te_cols:
        if c in train and c in eval_cols:
            te_tr, te_ev = oof_target_encode(train[c], y_train, eval_cols[c], n_folds=5, smoothing=20.0, seed=42)
            te_name = f'{c}_te'
            train_data[te_name] = te_tr.astype(np.float64)
            eval_data[te_name] = te_ev.astype(np.float64)
            extra_numeric.append(te_name)

    train_data['diag_te_max'] = np.maximum(
        train_data['diag_1_te'],
        np.maximum(train_data['diag_2_te'], train_data['diag_3_te']),
    )
    eval_data['diag_te_max'] = np.maximum(
        eval_data['diag_1_te'],
        np.maximum(eval_data['diag_2_te'], eval_data['diag_3_te']),
    )
    extra_numeric.append('diag_te_max')

    train_data['diag_te_mean'] = (
        train_data['diag_1_te'] + train_data['diag_2_te'] + train_data['diag_3_te']
    ) / 3.0
    eval_data['diag_te_mean'] = (
        eval_data['diag_1_te'] + eval_data['diag_2_te'] + eval_data['diag_3_te']
    ) / 3.0
    extra_numeric.append('diag_te_mean')

    # 4. Visit History & Prior Utilization
    for df, d in [(train_data, train), (eval_data, eval_cols)]:
        outp = np.nan_to_num(d['number_outpatient'].astype(np.float64), nan=0.0)
        emrg = np.nan_to_num(d['number_emergency'].astype(np.float64), nan=0.0)
        inp = np.nan_to_num(d['number_inpatient'].astype(np.float64), nan=0.0)
        tot_visits = outp + emrg + inp

        df['total_prior_visits'] = tot_visits
        df['has_prior_inpatient'] = (inp > 0.0).astype(np.float64)
        df['has_multiple_inpatient'] = (inp > 1.0).astype(np.float64)
        df['has_prior_emergency'] = (emrg > 0.0).astype(np.float64)
        df['has_any_prior_visit'] = (tot_visits > 0.0).astype(np.float64)

        df['log_inpatient'] = np.log1p(np.maximum(inp, 0.0))
        df['log_emergency'] = np.log1p(np.maximum(emrg, 0.0))
        df['log_outpatient'] = np.log1p(np.maximum(outp, 0.0))
        df['log_total_visits'] = np.log1p(np.maximum(tot_visits, 0.0))

        df['inpatient_ratio'] = inp / np.maximum(tot_visits, 1.0)
        df['emergency_ratio'] = emrg / np.maximum(tot_visits, 1.0)
        df['inp_x_emrg'] = df['log_inpatient'] * df['log_emergency']

    extra_numeric.extend([
        'total_prior_visits', 'has_prior_inpatient', 'has_multiple_inpatient',
        'has_prior_emergency', 'has_any_prior_visit', 'log_inpatient',
        'log_emergency', 'log_outpatient', 'log_total_visits',
        'inpatient_ratio', 'emergency_ratio', 'inp_x_emrg',
    ])

    # 5. Hospital Stay Intensity & Resource Use
    for df, d in [(train_data, train), (eval_data, eval_cols)]:
        los = np.maximum(np.nan_to_num(d['time_in_hospital'].astype(np.float64), nan=1.0), 1.0)
        n_labs = np.nan_to_num(d['num_lab_procedures'].astype(np.float64), nan=0.0)
        n_procs = np.nan_to_num(d['num_procedures'].astype(np.float64), nan=0.0)
        n_meds = np.nan_to_num(d['num_medications'].astype(np.float64), nan=0.0)
        n_diags = np.maximum(np.nan_to_num(d['number_diagnoses'].astype(np.float64), nan=1.0), 1.0)

        df['lab_per_day'] = n_labs / los
        df['meds_per_day'] = n_meds / los
        df['procs_per_day'] = n_procs / los
        df['total_procs'] = n_labs + n_procs
        df['total_intensity_per_day'] = (n_labs + n_procs + n_meds) / los
        df['meds_per_diag'] = n_meds / n_diags
        df['diags_per_day'] = n_diags / los
        df['has_procedures'] = (n_procs > 0.0).astype(np.float64)
        df['inp_x_los'] = df['log_inpatient'] * np.log1p(los)

    extra_numeric.extend([
        'lab_per_day', 'meds_per_day', 'procs_per_day', 'total_procs',
        'total_intensity_per_day', 'meds_per_diag', 'diags_per_day',
        'has_procedures', 'inp_x_los',
    ])

    # 6. Age Numeric & Interactions
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0,
    }
    for df, d in [(train_data, train), (eval_data, eval_cols)]:
        age_arr = np.array([age_map.get(a, 65.0) if a is not None else 65.0 for a in d['age']], dtype=np.float64)
        df['age_num'] = age_arr
        df['age_x_inpatient'] = age_arr * df['log_inpatient']
        df['age_is_elderly'] = (age_arr >= 75.0).astype(np.float64)

    extra_numeric.extend(['age_num', 'age_x_inpatient', 'age_is_elderly'])

    # 7. Medication & Insulin Dynamics
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
    ]

    for df, d in [(train_data, train), (eval_data, eval_cols)]:
        present_meds = [m for m in med_names if m in d]
        n_rows = len(d['time_in_hospital'])
        active_cnt = np.zeros(n_rows, dtype=np.float64)
        up_cnt = np.zeros(n_rows, dtype=np.float64)
        down_cnt = np.zeros(n_rows, dtype=np.float64)
        steady_cnt = np.zeros(n_rows, dtype=np.float64)

        for m in present_meds:
            arr = d[m]
            active_cnt += ((arr != 'No') & (arr != None)).astype(np.float64)
            up_cnt += (arr == 'Up').astype(np.float64)
            down_cnt += (arr == 'Down').astype(np.float64)
            steady_cnt += (arr == 'Steady').astype(np.float64)

        df['num_active_meds'] = active_cnt
        df['num_med_up'] = up_cnt
        df['num_med_down'] = down_cnt
        df['num_med_change'] = up_cnt + down_cnt
        df['num_med_steady'] = steady_cnt

        if 'insulin' in d:
            ins = d['insulin']
            df['insulin_treated'] = ((ins != 'No') & (ins != None)).astype(np.float64)
            df['insulin_up'] = (ins == 'Up').astype(np.float64)
            df['insulin_down'] = (ins == 'Down').astype(np.float64)
            df['insulin_steady'] = (ins == 'Steady').astype(np.float64)
        else:
            df['insulin_treated'] = np.zeros(n_rows, dtype=np.float64)
            df['insulin_up'] = np.zeros(n_rows, dtype=np.float64)
            df['insulin_down'] = np.zeros(n_rows, dtype=np.float64)
            df['insulin_steady'] = np.zeros(n_rows, dtype=np.float64)

        if 'change' in d:
            df['change_flag'] = (d['change'] == 'Ch').astype(np.float64)
        else:
            df['change_flag'] = np.zeros(n_rows, dtype=np.float64)

        if 'diabetesMed' in d:
            df['diabetes_med_flag'] = (d['diabetesMed'] == 'Yes').astype(np.float64)
        else:
            df['diabetes_med_flag'] = np.zeros(n_rows, dtype=np.float64)

    extra_numeric.extend([
        'num_active_meds', 'num_med_up', 'num_med_down', 'num_med_change',
        'num_med_steady', 'insulin_treated', 'insulin_up', 'insulin_down',
        'insulin_steady', 'change_flag', 'diabetes_med_flag',
    ])

    # 8. Clinical Disposition, Admission Type & Lab Results
    for df, d in [(train_data, train), (eval_data, eval_cols)]:
        disp = d.get('discharge_disposition_id')
        if disp is not None:
            df['disp_home'] = (disp == '1').astype(np.float64)
            df['disp_snf'] = (disp == '3').astype(np.float64)
            df['disp_home_health'] = (disp == '6').astype(np.float64)
            df['disp_facility'] = np.isin(disp, ['3', '4', '5', '6', '22', '25']).astype(np.float64)
        else:
            df['disp_home'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['disp_snf'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['disp_home_health'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['disp_facility'] = np.zeros(len(df['age_num']), dtype=np.float64)

        adm_type = d.get('admission_type_id')
        if adm_type is not None:
            df['adm_emergency'] = (adm_type == '1').astype(np.float64)
            df['adm_urgent'] = (adm_type == '2').astype(np.float64)
            df['adm_elective'] = (adm_type == '3').astype(np.float64)
        else:
            df['adm_emergency'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['adm_urgent'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['adm_elective'] = np.zeros(len(df['age_num']), dtype=np.float64)

        adm_src = d.get('admission_source_id')
        if adm_src is not None:
            df['adm_src_er'] = (adm_src == '7').astype(np.float64)
        else:
            df['adm_src_er'] = np.zeros(len(df['age_num']), dtype=np.float64)

        a1c = d.get('A1Cresult')
        if a1c is not None:
            df['a1c_tested'] = (a1c != 'None').astype(np.float64)
            df['a1c_high'] = np.isin(a1c, ['>7', '>8']).astype(np.float64)
        else:
            df['a1c_tested'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['a1c_high'] = np.zeros(len(df['age_num']), dtype=np.float64)

        glu = d.get('max_glu_serum')
        if glu is not None:
            df['glu_tested'] = (glu != 'None').astype(np.float64)
            df['glu_high'] = np.isin(glu, ['>200', '>300']).astype(np.float64)
        else:
            df['glu_tested'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['glu_high'] = np.zeros(len(df['age_num']), dtype=np.float64)

        df['risk_disp_x_inp'] = df['disp_facility'] * df['log_inpatient']
        df['risk_charlson_x_inp'] = df['charlson_idx'] * df['log_inpatient']

    extra_numeric.extend([
        'disp_home', 'disp_snf', 'disp_home_health', 'disp_facility',
        'adm_emergency', 'adm_urgent', 'adm_elective', 'adm_src_er',
        'a1c_tested', 'a1c_high', 'glu_tested', 'glu_high',
        'risk_disp_x_inp', 'risk_charlson_x_inp',
    ])

    # Encode with TabularEncoder
    encoder = TabularEncoder(min_count=25)
    encoder.fit(task, train_data, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)
    cat_indices = encoder.categorical_indices

    # 9. Ensemble of Diverse LightGBM Architectures
    # Model 1: Balanced, moderate capacity
    m1 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.022,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_alpha=0.1,
        reg_lambda=1.0,
        cat_smooth=50.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=cat_indices)
    p1 = m1.predict_proba(x_eval)[:, 1]

    # Model 2: Conservative, depth-constrained to avoid overfitting noisy shards
    m2 = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.025,
        num_leaves=18,
        max_depth=5,
        min_child_samples=250,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_alpha=0.5,
        reg_lambda=3.0,
        cat_smooth=75.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=cat_indices)
    p2 = m2.predict_proba(x_eval)[:, 1]

    # Model 3: Deeper multi-way interactions
    m3 = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.020,
        num_leaves=45,
        max_depth=7,
        min_child_samples=100,
        subsample=0.65,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=0.02,
        reg_lambda=0.5,
        cat_smooth=30.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=cat_indices)
    p3 = m3.predict_proba(x_eval)[:, 1]

    # Soft voting ensemble
    preds = 0.35 * p1 + 0.35 * p2 + 0.30 * p3
    preds = np.clip(preds, 0.0, 1.0)
    if np.isnan(preds).any():
        preds = np.nan_to_num(preds, nan=0.114)
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
