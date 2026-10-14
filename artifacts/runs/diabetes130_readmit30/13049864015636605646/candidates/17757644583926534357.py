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
    tr = dict(train)
    ev = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # Safe references to toolkit helpers from globals
    cb_fn = globals().get('icd9_comorbidities')
    grp_fn = globals().get('icd9_group')
    te_fn = globals().get('oof_target_encode')

    # 1. Comorbidities extraction
    if cb_fn is not None and 'diag_1' in train and 'diag_2' in train and 'diag_3' in train:
        cm_tr = cb_fn(train['diag_1'], train['diag_2'], train['diag_3'])
        cm_ev = cb_fn(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
        for k in cm_tr:
            tr[k] = cm_tr[k].astype(np.float64)
            ev[k] = cm_ev[k].astype(np.float64)
            extra_numeric.append(k)

        # Weighted Charlson Comorbidity Index & cardiorenal flag
        def _calc_cci(cm):
            return (
                1.0 * (cm.get('cm_mi', 0.0) + cm.get('cm_chf', 0.0) + cm.get('cm_pvd', 0.0) +
                       cm.get('cm_cerebrovascular', 0.0) + cm.get('cm_dementia', 0.0) +
                       cm.get('cm_pulmonary', 0.0) + cm.get('cm_rheumatic', 0.0) +
                       cm.get('cm_peptic_ulcer', 0.0) + cm.get('cm_liver', 0.0)) +
                2.0 * (cm.get('cm_hemiplegia', 0.0) + cm.get('cm_renal', 0.0) +
                       cm.get('cm_diabetes_complicated', 0.0) + cm.get('cm_cancer', 0.0)) +
                4.0 * (cm.get('cm_metastatic', 0.0) + cm.get('cm_hiv', 0.0))
            ).astype(np.float64)

        tr['cm_weighted_cci'] = _calc_cci(cm_tr)
        ev['cm_weighted_cci'] = _calc_cci(cm_ev)
        extra_numeric.append('cm_weighted_cci')

        if 'cm_chf' in cm_tr and 'cm_renal' in cm_tr:
            tr['cm_cardiorenal'] = (cm_tr['cm_chf'] * cm_tr['cm_renal']).astype(np.float64)
            ev['cm_cardiorenal'] = (cm_ev['cm_chf'] * cm_ev['cm_renal']).astype(np.float64)
            extra_numeric.append('cm_cardiorenal')

    # 2. ICD-9 Chapter Groups
    if grp_fn is not None:
        for diag_c in ['diag_1', 'diag_2', 'diag_3']:
            if diag_c in train:
                g_tr = np.array([grp_fn(x) if x is not None else 'missing' for x in train[diag_c]], dtype=object)
                g_ev = np.array([grp_fn(x) if x is not None else 'missing' for x in eval_cols[diag_c]], dtype=object)
                grp_col_name = f'{diag_c}_group'
                tr[grp_col_name] = g_tr
                ev[grp_col_name] = g_ev
                extra_categorical.append(grp_col_name)

        if 'diag_1_group' in tr and 'diag_2_group' in tr and 'diag_3_group' in tr:
            for g_name in ['circulatory', 'respiratory', 'diabetes', 'genitourinary']:
                has_g_tr = ((tr['diag_1_group'] == g_name) | (tr['diag_2_group'] == g_name) | (tr['diag_3_group'] == g_name)).astype(np.float64)
                has_g_ev = ((ev['diag_1_group'] == g_name) | (ev['diag_2_group'] == g_name) | (ev['diag_3_group'] == g_name)).astype(np.float64)
                col_name = f'has_chapter_{g_name}'
                tr[col_name] = has_g_tr
                ev[col_name] = has_g_ev
                extra_numeric.append(col_name)

    # 3. Target Encoding of High-Cardinality Categoricals
    if te_fn is not None:
        te_candidates = [
            'diag_1', 'diag_2', 'diag_3',
            'medical_specialty', 'discharge_disposition_id',
            'admission_source_id', 'admission_type_id', 'payer_code'
        ]
        diag_te_trs = []
        diag_te_evs = []
        for c in te_candidates:
            if c in train and c in eval_cols:
                tr_te, ev_te = te_fn(train[c], y_train, eval_cols[c], n_folds=5, smoothing=20.0, seed=42)
                te_name = f'te_{c}'
                tr[te_name] = tr_te.astype(np.float64)
                ev[te_name] = ev_te.astype(np.float64)
                extra_numeric.append(te_name)
                if c in ['diag_1', 'diag_2', 'diag_3']:
                    diag_te_trs.append(tr[te_name])
                    diag_te_evs.append(ev[te_name])

        if len(diag_te_trs) == 3:
            tr['te_diag_mean'] = (diag_te_trs[0] + diag_te_trs[1] + diag_te_trs[2]) / 3.0
            ev['te_diag_mean'] = (diag_te_evs[0] + diag_te_evs[1] + diag_te_evs[2]) / 3.0
            tr['te_diag_max'] = np.maximum(np.maximum(diag_te_trs[0], diag_te_trs[1]), diag_te_trs[2])
            ev['te_diag_max'] = np.maximum(np.maximum(diag_te_evs[0], diag_te_evs[1]), diag_te_evs[2])
            tr['te_diag_sum'] = diag_te_trs[0] + diag_te_trs[1] + diag_te_trs[2]
            ev['te_diag_sum'] = diag_te_evs[0] + diag_te_evs[1] + diag_te_evs[2]
            extra_numeric.extend(['te_diag_mean', 'te_diag_max', 'te_diag_sum'])

    # 4. Domain & Clinical Feature Engineering Helper
    def _extract_clinical_features(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        res = {}
        n_rows = len(d['time_in_hospital'])

        # Utilisation
        inp = np.nan_to_num(d.get('number_inpatient', np.zeros(n_rows)), nan=0.0)
        emg = np.nan_to_num(d.get('number_emergency', np.zeros(n_rows)), nan=0.0)
        out = np.nan_to_num(d.get('number_outpatient', np.zeros(n_rows)), nan=0.0)
        los = np.maximum(1.0, np.nan_to_num(d.get('time_in_hospital', np.ones(n_rows)), nan=1.0))
        num_med = np.nan_to_num(d.get('num_medications', np.zeros(n_rows)), nan=0.0)
        num_lab = np.nan_to_num(d.get('num_lab_procedures', np.zeros(n_rows)), nan=0.0)
        num_proc = np.nan_to_num(d.get('num_procedures', np.zeros(n_rows)), nan=0.0)
        num_diag = np.nan_to_num(d.get('number_diagnoses', np.zeros(n_rows)), nan=0.0)

        tot_visits = inp + emg + out
        res['total_visits'] = tot_visits
        res['has_visits'] = (tot_visits > 0).astype(np.float64)
        res['has_inp'] = (inp > 0).astype(np.float64)
        res['has_emg'] = (emg > 0).astype(np.float64)
        res['has_out'] = (out > 0).astype(np.float64)
        res['frequent_inp'] = (inp >= 2).astype(np.float64)
        res['acute_visits'] = inp + emg
        res['inpatient_ratio'] = (inp + 0.05) / (tot_visits + 1.0)
        res['emergency_ratio'] = (emg + 0.05) / (tot_visits + 1.0)

        # Intensities per day & interactions
        res['meds_per_day'] = num_med / (los + 0.5)
        res['labs_per_day'] = num_lab / (los + 0.5)
        res['procs_per_day'] = num_proc / (los + 0.5)
        res['inp_per_day'] = inp / (los + 0.5)
        res['meds_per_diag'] = num_med / (num_diag + 0.5)
        res['labs_per_med'] = num_lab / (num_med + 0.5)
        res['total_procedures'] = num_proc + num_lab
        res['activity_burden'] = num_med + num_proc + (num_lab * 0.1)

        # Non-linear utilisation interactions
        res['inp_x_diag'] = inp * num_diag
        res['inp_x_los'] = inp * los

        # Age conversion
        age_map = {
            '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
            '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
            '[80-90)': 85.0, '[90-100)': 95.0
        }
        age_arr = d.get('age')
        if age_arr is not None:
            age_num = np.array([age_map.get(x, 65.0) for x in age_arr], dtype=np.float64)
            res['age_numeric'] = age_num
            res['age_elderly'] = (age_num >= 75.0).astype(np.float64)
            res['age_x_diag'] = age_num * num_diag
            res['age_x_inp'] = age_num * inp

        # Medication features
        med_list = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
            'miglitol', 'troglitazone', 'tolazamide', 'examide',
            'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        n_meds_active = np.zeros(n_rows, dtype=np.float64)
        n_meds_up = np.zeros(n_rows, dtype=np.float64)
        n_meds_down = np.zeros(n_rows, dtype=np.float64)
        for m in med_list:
            if m in d:
                col = d[m]
                n_meds_active += ((col != 'No') & (col != None)).astype(np.float64)
                n_meds_up += (col == 'Up').astype(np.float64)
                n_meds_down += (col == 'Down').astype(np.float64)

        res['num_meds_active'] = n_meds_active
        res['num_meds_up'] = n_meds_up
        res['num_meds_down'] = n_meds_down
        res['num_meds_changed'] = n_meds_up + n_meds_down

        # Specific insulin patterns
        ins = d.get('insulin')
        if ins is not None:
            res['insulin_active'] = ((ins != 'No') & (ins != None)).astype(np.float64)
            res['insulin_up'] = (ins == 'Up').astype(np.float64)
            res['insulin_down'] = (ins == 'Down').astype(np.float64)
            res['insulin_steady'] = (ins == 'Steady').astype(np.float64)
            res['insulin_changed'] = ((ins == 'Up') | (ins == 'Down')).astype(np.float64)

        met = d.get('metformin')
        if met is not None:
            res['metformin_active'] = ((met != 'No') & (met != None)).astype(np.float64)

        ch = d.get('change')
        res['change_numeric'] = (ch == 'Ch').astype(np.float64) if ch is not None else np.zeros(n_rows, dtype=np.float64)

        dm = d.get('diabetesMed')
        res['diabetes_med_numeric'] = (dm == 'Yes').astype(np.float64) if dm is not None else np.zeros(n_rows, dtype=np.float64)

        # Lab tests
        a1c = d.get('A1Cresult')
        if a1c is not None:
            res['a1c_tested'] = (~np.isin(a1c, ['None', None])).astype(np.float64)
            res['a1c_high'] = np.isin(a1c, ['>7', '>8']).astype(np.float64)
            res['a1c_very_high'] = (a1c == '>8').astype(np.float64)
            res['a1c_tested_med_changed'] = res['a1c_tested'] * res['change_numeric']

        glu = d.get('max_glu_serum')
        if glu is not None:
            res['glu_tested'] = (~np.isin(glu, ['None', None])).astype(np.float64)
            res['glu_high'] = np.isin(glu, ['>200', '>300']).astype(np.float64)

        # Disposition & Admission types
        disch = d.get('discharge_disposition_id')
        if disch is not None:
            res['disch_home'] = (disch == '1').astype(np.float64)
            res['disch_snf'] = (disch == '3').astype(np.float64)
            res['disch_hh'] = (disch == '6').astype(np.float64)
            res['disch_ama'] = (disch == '7').astype(np.float64)
            res['disch_transfer'] = np.isin(disch, ['2', '3', '4', '5', '6', '22', '28']).astype(np.float64)

        adm_type = d.get('admission_type_id')
        if adm_type is not None:
            res['adm_emergency'] = (adm_type == '1').astype(np.float64)
            res['adm_urgent'] = (adm_type == '2').astype(np.float64)
            res['adm_elective'] = (adm_type == '3').astype(np.float64)

        adm_src = d.get('admission_source_id')
        if adm_src is not None:
            res['src_er'] = (adm_src == '7').astype(np.float64)
            res['src_transfer'] = np.isin(adm_src, ['4', '5', '6', '10', '22']).astype(np.float64)

        return res

    clin_tr = _extract_clinical_features(tr)
    clin_ev = _extract_clinical_features(ev)
    for feat_name, arr_tr in clin_tr.items():
        tr[feat_name] = arr_tr.astype(np.float64)
        ev[feat_name] = clin_ev[feat_name].astype(np.float64)
        extra_numeric.append(feat_name)

    # 5. Fit TabularEncoder
    encoder = TabularEncoder(min_count=35)
    encoder.fit(task, tr, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(tr)
    x_eval = encoder.transform(ev)

    # 6. Diverse LightGBM Ensemble
    models_config = [
        # Model 1: Balanced baseline configuration, slightly tuned
        {
            'n_estimators': 260,
            'learning_rate': 0.02,
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 180,
            'subsample': 0.75,
            'subsample_freq': 1,
            'colsample_bytree': 0.60,
            'reg_lambda': 2.0,
            'reg_alpha': 0.1,
            'cat_smooth': 45.0,
            'random_state': 42,
        },
        # Model 2: Shallow, highly regularized trees (strong generalizer)
        {
            'n_estimators': 280,
            'learning_rate': 0.022,
            'num_leaves': 18,
            'max_depth': 4,
            'min_child_samples': 120,
            'subsample': 0.80,
            'subsample_freq': 1,
            'colsample_bytree': 0.70,
            'reg_lambda': 5.0,
            'reg_alpha': 0.5,
            'cat_smooth': 55.0,
            'random_state': 101,
        },
        # Model 3: Deeper trees, lower feature subsampling (forces feature diversity)
        {
            'n_estimators': 250,
            'learning_rate': 0.018,
            'num_leaves': 42,
            'max_depth': 7,
            'min_child_samples': 220,
            'subsample': 0.70,
            'subsample_freq': 1,
            'colsample_bytree': 0.50,
            'reg_lambda': 4.0,
            'reg_alpha': 0.2,
            'cat_smooth': 35.0,
            'random_state': 202,
        },
        # Model 4: Robust intermediate depth with higher L2 penalty
        {
            'n_estimators': 270,
            'learning_rate': 0.02,
            'num_leaves': 26,
            'max_depth': 5,
            'min_child_samples': 160,
            'subsample': 0.75,
            'subsample_freq': 1,
            'colsample_bytree': 0.55,
            'reg_lambda': 6.0,
            'reg_alpha': 0.3,
            'cat_smooth': 50.0,
            'random_state': 303,
        },
    ]

    ensemble_preds = np.zeros(x_eval.shape[0], dtype=np.float64)
    for cfg in models_config:
        model = lgb.LGBMClassifier(
            **cfg,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(
            x_train,
            y_train,
            categorical_feature=encoder.categorical_indices,
        )
        preds = model.predict_proba(x_eval)[:, 1]
        ensemble_preds += preds

    ensemble_preds /= len(models_config)
    ensemble_preds = np.clip(ensemble_preds, 0.0, 1.0)
    np.nan_to_num(ensemble_preds, copy=False, nan=0.114)

    return ensemble_preds


# EVOLVE-BLOCK-END
# =====================================================================
