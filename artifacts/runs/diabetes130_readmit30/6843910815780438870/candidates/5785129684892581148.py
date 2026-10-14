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

    # 1. Comorbidities via icd9_comorbidities & Weighted Charlson Score
    train_cm = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    eval_cm = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k in train_cm:
        train_feat[k] = train_cm[k]
        eval_feat[k] = eval_cm[k]
        extra_numeric.append(k)

    for cm_dict, feat_dict in [(train_cm, train_feat), (eval_cm, eval_feat)]:
        # Charlson Comorbidity Index weighting
        feat_dict['charlson_score'] = (
            1.0 * (cm_dict['cm_mi'] + cm_dict['cm_chf'] + cm_dict['cm_pvd'] + cm_dict['cm_cerebrovascular'] +
                   cm_dict['cm_dementia'] + cm_dict['cm_pulmonary'] + cm_dict['cm_rheumatic'] +
                   cm_dict['cm_peptic_ulcer'] + cm_dict['cm_liver']) +
            2.0 * (cm_dict['cm_hemiplegia'] + cm_dict['cm_renal'] + cm_dict['cm_diabetes_complicated'] + cm_dict['cm_cancer']) +
            6.0 * (cm_dict['cm_metastatic'] + cm_dict['cm_hiv'])
        )
        feat_dict['cm_cardiorenal'] = cm_dict['cm_chf'] * cm_dict['cm_renal']
        feat_dict['cm_cardiopulmonary'] = cm_dict['cm_chf'] * cm_dict['cm_pulmonary']
        feat_dict['cm_renometabolic'] = cm_dict['cm_renal'] * cm_dict['cm_diabetes_complicated']
        feat_dict['cm_high_burden'] = (cm_dict['cm_count'] >= 3.0).astype(np.float64)

    extra_numeric.extend(['charlson_score', 'cm_cardiorenal', 'cm_cardiopulmonary', 'cm_renometabolic', 'cm_high_burden'])

    # 2. ICD-9 Chapter groups and 3-digit prefix categories
    def clean_prefix(val: Any) -> str:
        if val is None or not isinstance(val, str) or val in ('?', '', 'None'):
            return 'missing'
        val = val.strip()
        if '.' in val:
            return val.split('.')[0]
        return val[:3] if len(val) >= 3 else val

    for d_col in ['diag_1', 'diag_2', 'diag_3']:
        grp_col = f"{d_col}_group"
        train_feat[grp_col] = np.array([icd9_group(c) for c in train[d_col]], dtype=object)
        eval_feat[grp_col] = np.array([icd9_group(c) for c in eval_cols[d_col]], dtype=object)
        extra_categorical.append(grp_col)

        pfx_col = f"{d_col}_pfx"
        train_feat[pfx_col] = np.array([clean_prefix(c) for c in train[d_col]], dtype=object)
        eval_feat[pfx_col] = np.array([clean_prefix(c) for c in eval_cols[d_col]], dtype=object)
        extra_categorical.append(pfx_col)

        # Target encode the 3-digit prefixes
        te_pfx_name = f"{pfx_col}_te"
        tr_pfx_te, ev_pfx_te = oof_target_encode(
            train_feat[pfx_col], y_train, eval_feat[pfx_col], n_folds=5, smoothing=20.0, seed=42
        )
        train_feat[te_pfx_name] = tr_pfx_te.astype(np.float64)
        eval_feat[te_pfx_name] = ev_pfx_te.astype(np.float64)
        extra_numeric.append(te_pfx_name)

    # Prefix summary across diagnoses
    train_feat['mean_diag_pfx_te'] = (train_feat['diag_1_pfx_te'] + train_feat['diag_2_pfx_te'] + train_feat['diag_3_pfx_te']) / 3.0
    eval_feat['mean_diag_pfx_te'] = (eval_feat['diag_1_pfx_te'] + eval_feat['diag_2_pfx_te'] + eval_feat['diag_3_pfx_te']) / 3.0
    train_feat['max_diag_pfx_te'] = np.maximum(np.maximum(train_feat['diag_1_pfx_te'], train_feat['diag_2_pfx_te']), train_feat['diag_3_pfx_te'])
    eval_feat['max_diag_pfx_te'] = np.maximum(np.maximum(eval_feat['diag_1_pfx_te'], eval_feat['diag_2_pfx_te']), eval_feat['diag_3_pfx_te'])
    extra_numeric.extend(['mean_diag_pfx_te', 'max_diag_pfx_te'])

    # 3. Admission/discharge interactions and clinical indicators
    train_feat['source_disp'] = np.array([f"{str(s)}_{str(d)}" for s, d in zip(train['admission_source_id'], train['discharge_disposition_id'])], dtype=object)
    eval_feat['source_disp'] = np.array([f"{str(s)}_{str(d)}" for s, d in zip(eval_cols['admission_source_id'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('source_disp')

    tr_sd_te, ev_sd_te = oof_target_encode(
        train_feat['source_disp'], y_train, eval_feat['source_disp'], n_folds=5, smoothing=20.0, seed=42
    )
    train_feat['source_disp_te'] = tr_sd_te.astype(np.float64)
    eval_feat['source_disp_te'] = ev_sd_te.astype(np.float64)
    extra_numeric.append('source_disp_te')

    for d_src, d_dst in [(train, train_feat), (eval_cols, eval_feat)]:
        disp = d_src['discharge_disposition_id']
        adm_t = d_src['admission_type_id']
        adm_s = d_src['admission_source_id']

        d_dst['disch_home'] = (disp == '1').astype(np.float64)
        d_dst['disch_snf'] = ((disp == '3') | (disp == '4') | (disp == '5')).astype(np.float64)
        d_dst['disch_home_health'] = (disp == '6').astype(np.float64)
        d_dst['disch_ama'] = (disp == '7').astype(np.float64)
        d_dst['disch_transfer'] = ((disp == '2') | (disp == '3') | (disp == '4') | (disp == '5') | (disp == '22') | (disp == '28')).astype(np.float64)

        d_dst['adm_emergency'] = (adm_t == '1').astype(np.float64)
        d_dst['adm_urgent'] = (adm_t == '2').astype(np.float64)
        d_dst['adm_elective'] = (adm_t == '3').astype(np.float64)
        d_dst['source_er'] = (adm_s == '7').astype(np.float64)
        d_dst['source_transfer'] = ((adm_s == '4') | (adm_s == '5') | (adm_s == '6')).astype(np.float64)

    clin_trans_cols = [
        'disch_home', 'disch_snf', 'disch_home_health', 'disch_ama', 'disch_transfer',
        'adm_emergency', 'adm_urgent', 'adm_elective', 'source_er', 'source_transfer'
    ]
    extra_numeric.extend(clin_trans_cols)

    # 4. Out-of-fold target encoding for primary categoricals
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'discharge_disposition_id', 'admission_source_id', 'admission_type_id',
        'medical_specialty', 'payer_code'
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

    train_feat['max_diag_te'] = np.maximum(np.maximum(train_feat['diag_1_te'], train_feat['diag_2_te']), train_feat['diag_3_te'])
    eval_feat['max_diag_te'] = np.maximum(np.maximum(eval_feat['diag_1_te'], eval_feat['diag_2_te']), eval_feat['diag_3_te'])
    train_feat['mean_diag_te'] = (train_feat['diag_1_te'] + train_feat['diag_2_te'] + train_feat['diag_3_te']) / 3.0
    eval_feat['mean_diag_te'] = (eval_feat['diag_1_te'] + eval_feat['diag_2_te'] + eval_feat['diag_3_te']) / 3.0
    extra_numeric.extend(['max_diag_te', 'mean_diag_te'])

    # 5. Frequency encoding for high-cardinality features
    for f_col in ['medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'diag_1_pfx']:
        vals, counts = np.unique(train_feat[f_col].astype(str), return_counts=True)
        f_map = dict(zip(vals, counts.astype(np.float64) / n_train))
        fq_name = f"{f_col}_freq"
        train_feat[fq_name] = np.array([f_map.get(str(x), 0.0) for x in train_feat[f_col]], dtype=np.float64)
        eval_feat[fq_name] = np.array([f_map.get(str(x), 0.0) for x in eval_feat[f_col]], dtype=np.float64)
        extra_numeric.append(fq_name)

    # 6. Demographics, Age mapping & missingness signals
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0,
    }
    train_age = np.array([age_map.get(str(x), np.nan) for x in train['age']], dtype=np.float64)
    eval_age = np.array([age_map.get(str(x), np.nan) for x in eval_cols['age']], dtype=np.float64)
    train_feat['age_num'] = train_age
    eval_feat['age_num'] = eval_age

    for d_src, d_dst, age_arr in [(train, train_feat, train_age), (eval_cols, eval_feat, eval_age)]:
        d_dst['has_weight'] = np.array([1.0 if (w is not None and str(w) not in ('?', '', 'None')) else 0.0 for w in d_src['weight']], dtype=np.float64)
        d_dst['has_specialty'] = np.array([1.0 if (s is not None and str(s) not in ('?', '', 'None')) else 0.0 for s in d_src['medical_specialty']], dtype=np.float64)
        d_dst['has_payer'] = np.array([1.0 if (p is not None and str(p) not in ('?', '', 'None')) else 0.0 for p in d_src['payer_code']], dtype=np.float64)
        inp = d_src['number_inpatient'].astype(np.float64)
        th = d_src['time_in_hospital'].astype(np.float64)
        d_dst['elderly_prior_inpatient'] = (age_arr >= 70.0).astype(np.float64) * inp
        d_dst['age_time_in_hospital'] = age_arr * th

    extra_numeric.extend(['age_num', 'has_weight', 'has_specialty', 'has_payer', 'elderly_prior_inpatient', 'age_time_in_hospital'])

    # 7. Healthcare utilization & hospital care intensity
    for d_src, d_dst in [(train, train_feat), (eval_cols, eval_feat)]:
        inp = d_src['number_inpatient'].astype(np.float64)
        emg = d_src['number_emergency'].astype(np.float64)
        outp = d_src['number_outpatient'].astype(np.float64)
        time_hosp = np.maximum(d_src['time_in_hospital'].astype(np.float64), 1.0)
        num_meds = d_src['num_medications'].astype(np.float64)
        num_labs = d_src['num_lab_procedures'].astype(np.float64)
        num_procs = d_src['num_procedures'].astype(np.float64)
        num_diags = d_src['number_diagnoses'].astype(np.float64)

        tot_prior = inp + emg + outp
        d_dst['total_prior_visits'] = tot_prior
        d_dst['has_prior_inpatient'] = (inp > 0).astype(np.float64)
        d_dst['frequent_inpatient'] = (inp >= 2).astype(np.float64)
        d_dst['has_prior_emergency'] = (emg > 0).astype(np.float64)
        d_dst['has_any_prior_visit'] = (tot_prior > 0).astype(np.float64)
        d_dst['heavy_utilizer'] = (tot_prior >= 4).astype(np.float64)
        d_dst['inpatient_ratio'] = np.where(tot_prior > 0, inp / tot_prior, 0.0)
        d_dst['acute_prior_visits'] = inp + emg
        d_dst['weighted_prior_acuity'] = inp * 3.0 + emg * 1.5 + outp * 0.5

        d_dst['log_inpatient'] = np.log1p(inp)
        d_dst['log_emergency'] = np.log1p(emg)
        d_dst['log_outpatient'] = np.log1p(outp)
        d_dst['log_total_prior'] = np.log1p(tot_prior)

        d_dst['lab_per_day'] = num_labs / time_hosp
        d_dst['meds_per_day'] = num_meds / time_hosp
        d_dst['procs_per_day'] = num_procs / time_hosp
        d_dst['diags_per_day'] = num_diags / time_hosp
        d_dst['total_procedures'] = num_labs + num_procs
        d_dst['med_proc_ratio'] = num_meds / np.maximum(num_procs, 1.0)
        d_dst['severity_index'] = num_meds * time_hosp
        d_dst['inpatient_stay_interaction'] = inp * time_hosp

    util_cols = [
        'total_prior_visits', 'has_prior_inpatient', 'frequent_inpatient', 'has_prior_emergency',
        'has_any_prior_visit', 'heavy_utilizer', 'inpatient_ratio', 'acute_prior_visits',
        'weighted_prior_acuity', 'log_inpatient', 'log_emergency', 'log_outpatient', 'log_total_prior',
        'lab_per_day', 'meds_per_day', 'procs_per_day', 'diags_per_day', 'total_procedures',
        'med_proc_ratio', 'severity_index', 'inpatient_stay_interaction'
    ]
    extra_numeric.extend(util_cols)

    # 8. Medication patterns, specific agents & combination regimens
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    avail_meds = [m for m in med_names if m in train and m in eval_cols]
    sulf_cols = ['glimepiride', 'glipizide', 'glyburide', 'tolbutamide', 'chlorpropamide', 'acetohexamide', 'tolazamide']

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

        d_dst['insulin_changed'] = d_dst['insulin_up'] + d_dst['insulin_down']

        if 'metformin' in d_src:
            met = d_src['metformin']
            d_dst['metformin_active'] = ((met == 'Steady') | (met == 'Up') | (met == 'Down')).astype(np.float64)
        else:
            d_dst['metformin_active'] = np.zeros(n_rows, dtype=np.float64)

        has_sulf = np.zeros(n_rows, dtype=np.float64)
        for s_c in sulf_cols:
            if s_c in d_src:
                s_arr = d_src[s_c]
                has_sulf = np.maximum(has_sulf, ((s_arr == 'Steady') | (s_arr == 'Up') | (s_arr == 'Down')).astype(np.float64))
        d_dst['sulfonylurea_active'] = has_sulf

        d_dst['insulin_and_metformin'] = d_dst['insulin_active'] * d_dst['metformin_active']
        d_dst['insulin_and_sulf'] = d_dst['insulin_active'] * has_sulf

        if 'change' in d_src:
            d_dst['change_flag'] = (d_src['change'] == 'Ch').astype(np.float64)
        else:
            d_dst['change_flag'] = np.zeros(n_rows, dtype=np.float64)

        d_dst['med_change_with_prior_inp'] = d_dst['change_flag'] * d_dst['has_prior_inpatient']

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
        'insulin_active', 'insulin_up', 'insulin_down', 'insulin_steady', 'insulin_changed',
        'metformin_active', 'sulfonylurea_active', 'insulin_and_metformin', 'insulin_and_sulf',
        'change_flag', 'med_change_with_prior_inp', 'diabetes_med_flag',
        'a1c_tested', 'a1c_high', 'glu_tested', 'glu_high'
    ]
    extra_numeric.extend(med_feature_names)

    # 9. Fit TabularEncoder
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_feat, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_feat)
    x_eval = encoder.transform(eval_feat)

    # 10. Diverse Model Ensemble: LightGBM (leaf-wise) + XGBoost (depth-wise)
    lgb_models = [
        lgb.LGBMClassifier(
            n_estimators=360,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=75,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=1.5,
            reg_alpha=0.1,
            cat_smooth=40.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=420,
            learning_rate=0.025,
            num_leaves=20,
            max_depth=5,
            min_child_samples=50,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=3.5,
            reg_alpha=0.5,
            cat_smooth=25.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=48,
            max_depth=7,
            min_child_samples=100,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=0.8,
            reg_alpha=0.05,
            cat_smooth=60.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    lgb_preds = np.zeros(n_eval, dtype=np.float64)
    for model in lgb_models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        lgb_preds += model.predict_proba(x_eval)[:, 1]
    lgb_preds /= len(lgb_models)

    # XGBoost with histogram-based depth-wise splits for architectural diversity
    xgb_mod = globals().get('xgb') or globals().get('xgboost')
    if xgb_mod is not None:
        xgb_model = xgb_mod.XGBClassifier(
            n_estimators=280,
            learning_rate=0.035,
            max_depth=5,
            min_child_weight=30,
            subsample=0.80,
            colsample_bytree=0.65,
            reg_lambda=3.0,
            reg_alpha=0.5,
            tree_method='hist',
            eval_metric='logloss',
            random_state=777,
            verbosity=0,
            n_jobs=1,
        )
        xgb_model.fit(x_train, y_train)
        xgb_preds = xgb_model.predict_proba(x_eval)[:, 1]
        final_preds = 0.78 * lgb_preds + 0.22 * xgb_preds
    else:
        final_preds = lgb_preds

    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
