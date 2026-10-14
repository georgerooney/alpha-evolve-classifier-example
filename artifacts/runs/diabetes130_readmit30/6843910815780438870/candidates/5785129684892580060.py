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

    # 1. Comorbidities & Charlson Comorbidity Index
    cm_train = icd9_comorbidities(train_data['diag_1'], train_data['diag_2'], train_data['diag_3'])
    cm_eval = icd9_comorbidities(eval_data['diag_1'], eval_data['diag_2'], eval_data['diag_3'])
    for k in cm_train:
        train_data[k] = cm_train[k].astype(np.float64)
        eval_data[k] = cm_eval[k].astype(np.float64)
        extra_numeric.append(k)

    charlson_tr = (
        cm_train['cm_mi'] + cm_train['cm_chf'] + cm_train['cm_pvd'] +
        cm_train['cm_cerebrovascular'] + cm_train['cm_dementia'] +
        cm_train['cm_pulmonary'] + cm_train['cm_rheumatic'] +
        cm_train['cm_peptic_ulcer'] + cm_train['cm_liver'] +
        2.0 * (cm_train['cm_hemiplegia'] + cm_train['cm_renal'] +
               cm_train['cm_cancer'] + cm_train['cm_diabetes_complicated']) +
        6.0 * (cm_train['cm_metastatic'] + cm_train['cm_hiv'])
    )
    charlson_ev = (
        cm_eval['cm_mi'] + cm_eval['cm_chf'] + cm_eval['cm_pvd'] +
        cm_eval['cm_cerebrovascular'] + cm_eval['cm_dementia'] +
        cm_eval['cm_pulmonary'] + cm_eval['cm_rheumatic'] +
        cm_eval['cm_peptic_ulcer'] + cm_eval['cm_liver'] +
        2.0 * (cm_eval['cm_hemiplegia'] + cm_eval['cm_renal'] +
               cm_eval['cm_cancer'] + cm_eval['cm_diabetes_complicated']) +
        6.0 * (cm_eval['cm_metastatic'] + cm_eval['cm_hiv'])
    )
    train_data['charlson_score'] = charlson_tr.astype(np.float64)
    eval_data['charlson_score'] = charlson_ev.astype(np.float64)
    extra_numeric.append('charlson_score')

    # 2. ICD-9 Chapter groups
    for diag in ['diag_1', 'diag_2', 'diag_3']:
        if diag in train_data:
            grp_name = f'{diag}_group'
            train_data[grp_name] = np.array([icd9_group(c) for c in train_data[diag]], dtype=object)
            eval_data[grp_name] = np.array([icd9_group(c) for c in eval_data[diag]], dtype=object)
            extra_categorical.append(grp_name)

    # 3. Out-of-fold target encoding for high-cardinality categoricals
    te_cols = [
        'diag_1', 'diag_2', 'diag_3', 'medical_specialty',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'payer_code'
    ]
    for c in te_cols:
        if c in train_data:
            tr_te, ev_te = oof_target_encode(
                train_data[c], y_train, eval_data[c],
                n_folds=5, smoothing=25.0, seed=42
            )
            te_name = f'{c}_te'
            train_data[te_name] = tr_te.astype(np.float64)
            eval_data[te_name] = ev_te.astype(np.float64)
            extra_numeric.append(te_name)

    if 'diag_1_te' in train_data and 'diag_2_te' in train_data and 'diag_3_te' in train_data:
        train_data['max_diag_te'] = np.maximum(
            train_data['diag_1_te'],
            np.maximum(train_data['diag_2_te'], train_data['diag_3_te'])
        )
        eval_data['max_diag_te'] = np.maximum(
            eval_data['diag_1_te'],
            np.maximum(eval_data['diag_2_te'], eval_data['diag_3_te'])
        )
        extra_numeric.append('max_diag_te')

    # 4. Age parsing to continuous midpoint
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    if 'age' in train_data:
        train_data['age_num'] = np.array([age_map.get(str(x), 65.0) for x in train_data['age']], dtype=np.float64)
        eval_data['age_num'] = np.array([age_map.get(str(x), 65.0) for x in eval_data['age']], dtype=np.float64)
        extra_numeric.append('age_num')

    # 5. Prior healthcare utilization & stay intensity features
    for d in [train_data, eval_data]:
        inp = d['number_inpatient']
        emg = d['number_emergency']
        outp = d['number_outpatient']
        time_h = d['time_in_hospital']
        n_med = d['num_medications']
        n_lab = d['num_lab_procedures']
        n_proc = d['num_procedures']
        n_diag = d['number_diagnoses']

        tot_prior = inp + emg + outp
        d['tot_prior_visits'] = tot_prior.astype(np.float64)
        d['has_inpatient'] = (inp > 0).astype(np.float64)
        d['has_emergency'] = (emg > 0).astype(np.float64)
        d['inpatient_ratio'] = (inp / (tot_prior + 1.0)).astype(np.float64)
        d['emergency_ratio'] = (emg / (tot_prior + 1.0)).astype(np.float64)
        d['inpatient_per_day'] = (inp / (time_h + 1.0)).astype(np.float64)

        d['med_per_day'] = (n_med / (time_h + 1e-4)).astype(np.float64)
        d['lab_per_day'] = (n_lab / (time_h + 1e-4)).astype(np.float64)
        d['proc_per_day'] = (n_proc / (time_h + 1e-4)).astype(np.float64)
        d['total_intensity'] = (n_med + n_lab + n_proc).astype(np.float64)
        d['intensity_per_day'] = ((n_med + n_lab + n_proc) / (time_h + 1e-4)).astype(np.float64)
        d['diag_per_day'] = (n_diag / (time_h + 1e-4)).astype(np.float64)

        if 'age_num' in d:
            d['inp_x_age'] = (inp * (d['age_num'] / 50.0)).astype(np.float64)
        if 'charlson_score' in d:
            d['inp_x_charlson'] = (inp * d['charlson_score']).astype(np.float64)

    for k in [
        'tot_prior_visits', 'has_inpatient', 'has_emergency',
        'inpatient_ratio', 'emergency_ratio', 'inpatient_per_day',
        'med_per_day', 'lab_per_day', 'proc_per_day',
        'total_intensity', 'intensity_per_day', 'diag_per_day',
        'inp_x_age', 'inp_x_charlson'
    ]:
        if k in train_data:
            extra_numeric.append(k)

    # 6. Medication patterns & insulin titration
    known_meds = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    med_cols_present = [m for m in known_meds if m in train_data]

    for d in [train_data, eval_data]:
        n = len(d['time_in_hospital'])
        act = np.zeros(n, dtype=np.float64)
        up = np.zeros(n, dtype=np.float64)
        down = np.zeros(n, dtype=np.float64)
        steady = np.zeros(n, dtype=np.float64)
        for m in med_cols_present:
            v = d[m]
            act += ((v != 'No') & (v != None)).astype(np.float64)
            up += (v == 'Up').astype(np.float64)
            down += (v == 'Down').astype(np.float64)
            steady += (v == 'Steady').astype(np.float64)
        d['num_active_meds'] = act
        d['num_meds_up'] = up
        d['num_meds_down'] = down
        d['num_meds_steady'] = steady
        d['num_meds_changed'] = up + down

        if 'insulin' in d:
            ins = d['insulin']
            d['insulin_active'] = ((ins != 'No') & (ins != None)).astype(np.float64)
            d['insulin_up'] = (ins == 'Up').astype(np.float64)
            d['insulin_down'] = (ins == 'Down').astype(np.float64)
            d['insulin_steady'] = (ins == 'Steady').astype(np.float64)
            d['insulin_changed'] = ((ins == 'Up') | (ins == 'Down')).astype(np.float64)

    for k in [
        'num_active_meds', 'num_meds_up', 'num_meds_down', 'num_meds_steady',
        'num_meds_changed', 'insulin_active', 'insulin_up', 'insulin_down',
        'insulin_steady', 'insulin_changed'
    ]:
        if k in train_data:
            extra_numeric.append(k)

    # 7. Lab indicators & Admission/Discharge clinical flags
    for d in [train_data, eval_data]:
        if 'A1Cresult' in d:
            a1c = d['A1Cresult']
            d['a1c_tested'] = (a1c != 'None').astype(np.float64)
            d['a1c_high'] = ((a1c == '>7') | (a1c == '>8')).astype(np.float64)
            d['a1c_very_high'] = (a1c == '>8').astype(np.float64)
        if 'max_glu_serum' in d:
            glu = d['max_glu_serum']
            d['glu_tested'] = (glu != 'None').astype(np.float64)
            d['glu_high'] = ((glu == '>200') | (glu == '>300')).astype(np.float64)
        if 'discharge_disposition_id' in d:
            dd = d['discharge_disposition_id']
            d['disch_home'] = (dd == '1').astype(np.float64)
            d['disch_snf'] = (dd == '3').astype(np.float64)
            d['disch_homehealth'] = (dd == '6').astype(np.float64)
        if 'admission_source_id' in d:
            d['adm_emergency'] = (d['admission_source_id'] == '7').astype(np.float64)
        if 'admission_type_id' in d:
            d['adm_type_emergency'] = (d['admission_type_id'] == '1').astype(np.float64)

    for k in [
        'a1c_tested', 'a1c_high', 'a1c_very_high', 'glu_tested', 'glu_high',
        'disch_home', 'disch_snf', 'disch_homehealth', 'adm_emergency', 'adm_type_emergency'
    ]:
        if k in train_data:
            extra_numeric.append(k)

    # Deduplicate extra feature lists
    extra_numeric = list(dict.fromkeys(extra_numeric))
    extra_categorical = list(dict.fromkeys(extra_categorical))

    # 8. Encode features
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_data, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)
    x_train = np.asarray(x_train, dtype=np.float64)
    x_eval = np.asarray(x_eval, dtype=np.float64)
    cat_indices = encoder.categorical_indices

    # 9. Fit LightGBM Models
    lgb_mod = globals().get('lgb') or globals().get('lightgbm')

    model_lgb1 = lgb_mod.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=0.5,
        cat_smooth=50.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model_lgb1.fit(x_train, y_train, categorical_feature=cat_indices)
    p_lgb1 = model_lgb1.predict_proba(x_eval)[:, 1]

    model_lgb2 = lgb_mod.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.02,
        num_leaves=18,
        min_child_samples=80,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=2.0,
        reg_alpha=0.5,
        cat_smooth=75.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    model_lgb2.fit(x_train, y_train, categorical_feature=cat_indices)
    p_lgb2 = model_lgb2.predict_proba(x_eval)[:, 1]

    model_lgb3 = lgb_mod.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.025,
        num_leaves=24,
        min_child_samples=160,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=0.1,
        cat_smooth=60.0,
        random_state=456,
        n_jobs=1,
        verbose=-1,
    )
    model_lgb3.fit(x_train, y_train, categorical_feature=cat_indices)
    p_lgb3 = model_lgb3.predict_proba(x_eval)[:, 1]

    # 10. Fit XGBoost with fail-safe handling
    p_xgb = None
    try:
        xgb_mod = globals().get('xgb') or globals().get('xgboost')
        if xgb_mod is not None:
            xgb_clf = xgb_mod.XGBClassifier(
                n_estimators=220,
                learning_rate=0.025,
                max_depth=5,
                min_child_weight=30,
                subsample=0.75,
                colsample_bytree=0.6,
                reg_lambda=2.0,
                reg_alpha=0.5,
                tree_method='hist',
                random_state=42,
                n_jobs=1,
                eval_metric='logloss',
            )
            xgb_clf.fit(x_train, y_train)
            cand = xgb_clf.predict_proba(x_eval)[:, 1]
            if not np.isnan(cand).any():
                p_xgb = cand
    except Exception:
        p_xgb = None

    if p_xgb is not None:
        preds = 0.35 * p_lgb1 + 0.30 * p_lgb2 + 0.15 * p_lgb3 + 0.20 * p_xgb
    else:
        preds = 0.40 * p_lgb1 + 0.35 * p_lgb2 + 0.25 * p_lgb3

    preds = np.nan_to_num(preds, nan=0.114)
    preds = np.clip(preds, 0.0, 1.0)
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
