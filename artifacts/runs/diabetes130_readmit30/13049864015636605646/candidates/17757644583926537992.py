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
    train_feats = dict(train)
    eval_feats = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    def add_num(name: str, tr_arr, ev_arr):
        train_feats[name] = np.asarray(tr_arr, dtype=np.float64)
        eval_feats[name] = np.asarray(ev_arr, dtype=np.float64)
        extra_numeric.append(name)

    def add_cat(name: str, tr_arr, ev_arr):
        train_feats[name] = np.asarray(tr_arr, dtype=object)
        eval_feats[name] = np.asarray(ev_arr, dtype=object)
        extra_categorical.append(name)

    # 1. Charlson-style comorbidities
    cm_tr = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_ev = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k in cm_tr:
        add_num(k, cm_tr[k], cm_ev[k])

    cm_cardio_tr = np.maximum.reduce([cm_tr['cm_mi'], cm_tr['cm_chf'], cm_tr['cm_pvd'], cm_tr['cm_cerebrovascular']])
    cm_cardio_ev = np.maximum.reduce([cm_ev['cm_mi'], cm_ev['cm_chf'], cm_ev['cm_pvd'], cm_ev['cm_cerebrovascular']])
    add_num('cm_cardio', cm_cardio_tr, cm_cardio_ev)
    add_num('cm_multi', (cm_tr['cm_count'] >= 2.0).astype(np.float64), (cm_ev['cm_count'] >= 2.0).astype(np.float64))

    # 2. ICD-9 Chapter groups
    for diag in ('diag_1', 'diag_2', 'diag_3'):
        if diag in train and diag in eval_cols:
            grp_name = f"{diag}_group"
            add_cat(grp_name,
                    [icd9_group(c) for c in train[diag]],
                    [icd9_group(c) for c in eval_cols[diag]])

    # 3. Out-of-fold target encoding for high-cardinality features
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'medical_specialty',
        'discharge_disposition_id',
        'admission_source_id',
        'admission_type_id',
        'payer_code'
    ]
    for col in te_cols:
        if col in train and col in eval_cols:
            tr_te, ev_te = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=20.0, seed=42)
            add_num(f"{col}_te", tr_te, ev_te)

    # 4. Utilization intensity features
    for split, data in (('tr', train), ('ev', eval_cols)):
        inpat = np.nan_to_num(data.get('number_inpatient', 0.0), nan=0.0)
        emerg = np.nan_to_num(data.get('number_emergency', 0.0), nan=0.0)
        outpat = np.nan_to_num(data.get('number_outpatient', 0.0), nan=0.0)
        tot_prior = inpat + emerg + outpat
        if split == 'tr':
            inpat_tr, emerg_tr, outpat_tr, tot_prior_tr = inpat, emerg, outpat, tot_prior
        else:
            inpat_ev, emerg_ev, outpat_ev, tot_prior_ev = inpat, emerg, outpat, tot_prior

    add_num('total_prior_visits', tot_prior_tr, tot_prior_ev)
    add_num('has_inpatient', (inpat_tr > 0).astype(np.float64), (inpat_ev > 0).astype(np.float64))
    add_num('has_emergency', (emerg_tr > 0).astype(np.float64), (emerg_ev > 0).astype(np.float64))
    add_num('inpatient_ratio', np.where(tot_prior_tr > 0, inpat_tr / (tot_prior_tr + 1e-5), 0.0),
                               np.where(tot_prior_ev > 0, inpat_ev / (tot_prior_ev + 1e-5), 0.0))
    add_num('emergency_ratio', np.where(tot_prior_tr > 0, emerg_tr / (tot_prior_tr + 1e-5), 0.0),
                               np.where(tot_prior_ev > 0, emerg_ev / (tot_prior_ev + 1e-5), 0.0))
    add_num('inpatient_squared', inpat_tr ** 2, inpat_ev ** 2)

    # 5. Encounter intensity per hospital day
    for split, data in (('tr', train), ('ev', eval_cols)):
        th = np.maximum(np.nan_to_num(data.get('time_in_hospital', 1.0), nan=1.0), 1.0)
        nl = np.nan_to_num(data.get('num_lab_procedures', 0.0), nan=0.0)
        np_proc = np.nan_to_num(data.get('num_procedures', 0.0), nan=0.0)
        nm = np.nan_to_num(data.get('num_medications', 0.0), nan=0.0)
        nd = np.nan_to_num(data.get('number_diagnoses', 0.0), nan=0.0)

        lpd = nl / th
        mpd = nm / th
        ppd = np_proc / th
        dpd = nd / th
        mdr = nm / np.maximum(nd, 1.0)
        comp = nl * 0.1 + np_proc * 2.0 + nm * 0.5 + nd * 1.5

        if split == 'tr':
            lpd_tr, mpd_tr, ppd_tr, dpd_tr, mdr_tr, comp_tr = lpd, mpd, ppd, dpd, mdr, comp
        else:
            lpd_ev, mpd_ev, ppd_ev, dpd_ev, mdr_ev, comp_ev = lpd, mpd, ppd, dpd, mdr, comp

    add_num('labs_per_day', lpd_tr, lpd_ev)
    add_num('meds_per_day', mpd_tr, mpd_ev)
    add_num('procs_per_day', ppd_tr, ppd_ev)
    add_num('diags_per_day', dpd_tr, dpd_ev)
    add_num('med_diag_ratio', mdr_tr, mdr_ev)
    add_num('encounter_complexity', comp_tr, comp_ev)

    # 6. Medication change patterns and insulin direction
    all_meds = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
        'metformin-pioglitazone'
    ]
    active_meds = [m for m in all_meds if m in train and m in eval_cols]

    for split, data in (('tr', train), ('ev', eval_cols)):
        n_r = len(data['time_in_hospital'])
        act = np.zeros(n_r, dtype=np.float64)
        chg = np.zeros(n_r, dtype=np.float64)
        up = np.zeros(n_r, dtype=np.float64)
        dn = np.zeros(n_r, dtype=np.float64)
        for m in active_meds:
            arr = data[m]
            act += np.isin(arr, ['Steady', 'Up', 'Down']).astype(np.float64)
            chg += np.isin(arr, ['Up', 'Down']).astype(np.float64)
            up += (arr == 'Up').astype(np.float64)
            dn += (arr == 'Down').astype(np.float64)
        if split == 'tr':
            act_tr, chg_tr, up_tr, dn_tr = act, chg, up, dn
        else:
            act_ev, chg_ev, up_ev, dn_ev = act, chg, up, dn

    add_num('active_meds_count', act_tr, act_ev)
    add_num('changed_meds_count', chg_tr, chg_ev)
    add_num('up_meds_count', up_tr, up_ev)
    add_num('down_meds_count', dn_tr, dn_ev)

    if 'insulin' in train and 'insulin' in eval_cols:
        add_num('insulin_active',
                np.isin(train['insulin'], ['Steady', 'Up', 'Down']).astype(np.float64),
                np.isin(eval_cols['insulin'], ['Steady', 'Up', 'Down']).astype(np.float64))
        add_num('insulin_changed',
                np.isin(train['insulin'], ['Up', 'Down']).astype(np.float64),
                np.isin(eval_cols['insulin'], ['Up', 'Down']).astype(np.float64))
        add_num('insulin_up',
                (train['insulin'] == 'Up').astype(np.float64),
                (eval_cols['insulin'] == 'Up').astype(np.float64))
        add_num('insulin_down',
                (train['insulin'] == 'Down').astype(np.float64),
                (eval_cols['insulin'] == 'Down').astype(np.float64))

    if 'change' in train and 'change' in eval_cols:
        add_num('med_change_flag',
                (train['change'] == 'Ch').astype(np.float64),
                (eval_cols['change'] == 'Ch').astype(np.float64))

    if 'diabetesMed' in train and 'diabetesMed' in eval_cols:
        add_num('diabetes_med_flag',
                (train['diabetesMed'] == 'Yes').astype(np.float64),
                (eval_cols['diabetesMed'] == 'Yes').astype(np.float64))

    # 7. Demographics, lab results and admission indicators
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    if 'age' in train and 'age' in eval_cols:
        age_tr = np.array([age_map.get(a, np.nan) for a in train['age']], dtype=np.float64)
        age_ev = np.array([age_map.get(a, np.nan) for a in eval_cols['age']], dtype=np.float64)
        add_num('age_numeric', age_tr, age_ev)
        add_num('is_elderly', (age_tr >= 70.0).astype(np.float64), (age_ev >= 70.0).astype(np.float64))

    if 'weight' in train and 'weight' in eval_cols:
        add_num('has_weight',
                np.array([w is not None and w != '?' and w != '' for w in train['weight']], dtype=np.float64),
                np.array([w is not None and w != '?' and w != '' for w in eval_cols['weight']], dtype=np.float64))

    if 'medical_specialty' in train and 'medical_specialty' in eval_cols:
        add_num('has_medical_specialty',
                np.array([m is not None and m != '?' and m != '' for m in train['medical_specialty']], dtype=np.float64),
                np.array([m is not None and m != '?' and m != '' for m in eval_cols['medical_specialty']], dtype=np.float64))

    if 'A1Cresult' in train and 'A1Cresult' in eval_cols:
        add_num('a1c_tested',
                (~np.isin(train['A1Cresult'], ['None', None, ''])).astype(np.float64),
                (~np.isin(eval_cols['A1Cresult'], ['None', None, ''])).astype(np.float64))
        add_num('a1c_high',
                np.isin(train['A1Cresult'], ['>7', '>8']).astype(np.float64),
                np.isin(eval_cols['A1Cresult'], ['>7', '>8']).astype(np.float64))
        add_num('a1c_vhigh',
                (train['A1Cresult'] == '>8').astype(np.float64),
                (eval_cols['A1Cresult'] == '>8').astype(np.float64))

    if 'max_glu_serum' in train and 'max_glu_serum' in eval_cols:
        add_num('glu_tested',
                (~np.isin(train['max_glu_serum'], ['None', None, ''])).astype(np.float64),
                (~np.isin(eval_cols['max_glu_serum'], ['None', None, ''])).astype(np.float64))
        add_num('glu_high',
                np.isin(train['max_glu_serum'], ['>200', '>300']).astype(np.float64),
                np.isin(eval_cols['max_glu_serum'], ['>200', '>300']).astype(np.float64))

    if 'admission_type_id' in train and 'admission_type_id' in eval_cols:
        add_num('adm_emergency',
                (train['admission_type_id'] == '1').astype(np.float64),
                (eval_cols['admission_type_id'] == '1').astype(np.float64))
        add_num('adm_urgent',
                (train['admission_type_id'] == '2').astype(np.float64),
                (eval_cols['admission_type_id'] == '2').astype(np.float64))

    if 'discharge_disposition_id' in train and 'discharge_disposition_id' in eval_cols:
        add_num('disch_home',
                (train['discharge_disposition_id'] == '1').astype(np.float64),
                (eval_cols['discharge_disposition_id'] == '1').astype(np.float64))

    # Fit TabularEncoder with enriched numeric and categorical feature sets
    encoder = TabularEncoder(min_count=30)
    encoder.fit(
        task,
        train_feats,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_feats)
    x_eval = encoder.transform(eval_feats)

    # 8. Diverse multi-capacity LightGBM ensemble
    models = [
        # Model 1: Balanced depth and regularization
        lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.022,
            num_leaves=31,
            min_child_samples=150,
            subsample=0.72,
            subsample_freq=1,
            colsample_bytree=0.62,
            reg_lambda=2.5,
            reg_alpha=0.1,
            cat_smooth=50.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 2: Shallower, more conservative trees with heavier regularization
        lgb.LGBMClassifier(
            n_estimators=340,
            learning_rate=0.018,
            num_leaves=20,
            min_child_samples=200,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.52,
            reg_lambda=5.0,
            reg_alpha=0.4,
            cat_smooth=75.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 3: Deeper trees capturing multi-way feature interactions
        lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.025,
            num_leaves=42,
            min_child_samples=180,
            subsample=0.78,
            subsample_freq=1,
            colsample_bytree=0.68,
            reg_lambda=1.5,
            reg_alpha=0.05,
            cat_smooth=35.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    preds = []
    for model in models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        preds.append(model.predict_proba(x_eval)[:, 1])

    final_pred = np.mean(preds, axis=0)
    return np.clip(np.nan_to_num(final_pred, nan=0.114), 1e-6, 1.0 - 1e-6)


# EVOLVE-BLOCK-END
# =====================================================================
