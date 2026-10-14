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
    n_train = len(y_train)
    n_eval = len(ev['time_in_hospital'])

    extra_numeric = []
    extra_categorical = []

    # 1. Comorbidities via icd9_comorbidities
    cm_tr = icd9_comorbidities(tr['diag_1'], tr['diag_2'], tr['diag_3'])
    cm_ev = icd9_comorbidities(ev['diag_1'], ev['diag_2'], ev['diag_3'])
    for cm_key, arr in cm_tr.items():
        tr[cm_key] = arr
        ev[cm_key] = cm_ev[cm_key]
        extra_numeric.append(cm_key)

    # Comorbidity interactions with prior inpatient visits
    for cm in ['cm_chf', 'cm_renal', 'cm_diabetes_complicated', 'cm_pulmonary']:
        if cm in cm_tr:
            inter_key = f"{cm}_x_inpatient"
            tr[inter_key] = cm_tr[cm] * tr['number_inpatient']
            ev[inter_key] = cm_ev[cm] * ev['number_inpatient']
            extra_numeric.append(inter_key)

    tr['cm_count_x_inpatient'] = cm_tr['cm_count'] * tr['number_inpatient']
    ev['cm_count_x_inpatient'] = cm_ev['cm_count'] * ev['number_inpatient']
    extra_numeric.append('cm_count_x_inpatient')

    # 2. ICD-9 Chapter groups
    for col in ['diag_1', 'diag_2', 'diag_3']:
        grp_col = f"{col}_group"
        tr[grp_col] = np.array([icd9_group(c) for c in tr[col]], dtype=object)
        ev[grp_col] = np.array([icd9_group(c) for c in ev[col]], dtype=object)
        extra_categorical.append(grp_col)

    # 3. Target encodings of high-cardinality columns
    te_cols = [
        'discharge_disposition_id',
        'admission_source_id',
        'admission_type_id',
        'diag_1',
        'diag_2',
        'diag_3',
        'medical_specialty',
        'payer_code',
    ]
    for col in te_cols:
        if col in tr and col in ev:
            tr_te, ev_te = oof_target_encode(
                tr[col], y_train, ev[col], n_folds=5, smoothing=25.0, seed=42
            )
            te_key = f"{col}_te"
            tr[te_key] = tr_te.astype(np.float64)
            ev[te_key] = ev_te.astype(np.float64)
            extra_numeric.append(te_key)

    # 4. Utilization & care intensity metrics
    total_prior_tr = tr['number_inpatient'] + tr['number_emergency'] + tr['number_outpatient']
    total_prior_ev = ev['number_inpatient'] + ev['number_emergency'] + ev['number_outpatient']
    tr['total_prior_visits'] = total_prior_tr
    ev['total_prior_visits'] = total_prior_ev
    extra_numeric.append('total_prior_visits')

    tr['has_prior_inpatient'] = (tr['number_inpatient'] > 0).astype(np.float64)
    ev['has_prior_inpatient'] = (ev['number_inpatient'] > 0).astype(np.float64)
    extra_numeric.append('has_prior_inpatient')

    tr['inpatient_ratio'] = tr['number_inpatient'] / (total_prior_tr + 1.0)
    ev['inpatient_ratio'] = ev['number_inpatient'] / (total_prior_ev + 1.0)
    extra_numeric.append('inpatient_ratio')

    tr['emergency_ratio'] = tr['number_emergency'] / (total_prior_tr + 1.0)
    ev['emergency_ratio'] = ev['number_emergency'] / (total_prior_ev + 1.0)
    extra_numeric.append('emergency_ratio')

    hosp_days_tr = np.maximum(tr['time_in_hospital'], 1.0)
    hosp_days_ev = np.maximum(ev['time_in_hospital'], 1.0)
    tr['lab_per_day'] = tr['num_lab_procedures'] / hosp_days_tr
    ev['lab_per_day'] = ev['num_lab_procedures'] / hosp_days_ev
    tr['meds_per_day'] = tr['num_medications'] / hosp_days_tr
    ev['meds_per_day'] = ev['num_medications'] / hosp_days_ev
    tr['procs_per_day'] = tr['num_procedures'] / hosp_days_tr
    ev['procs_per_day'] = ev['num_procedures'] / hosp_days_ev
    extra_numeric.extend(['lab_per_day', 'meds_per_day', 'procs_per_day'])

    # 5. Medication dynamics (changes and active counts)
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    present_meds = [m for m in med_cols if m in tr and m in ev]
    if present_meds:
        tr_active = np.zeros(n_train, dtype=np.float64)
        ev_active = np.zeros(n_eval, dtype=np.float64)
        tr_changed = np.zeros(n_train, dtype=np.float64)
        ev_changed = np.zeros(n_eval, dtype=np.float64)
        tr_up = np.zeros(n_train, dtype=np.float64)
        ev_up = np.zeros(n_eval, dtype=np.float64)

        for m in present_meds:
            tm, em = tr[m], ev[m]
            tr_active += ((tm != 'No') & (tm != None)).astype(np.float64)
            ev_active += ((em != 'No') & (em != None)).astype(np.float64)
            tr_changed += np.isin(tm, ['Up', 'Down']).astype(np.float64)
            ev_changed += np.isin(em, ['Up', 'Down']).astype(np.float64)
            tr_up += (tm == 'Up').astype(np.float64)
            ev_up += (em == 'Up').astype(np.float64)

        tr['num_active_meds'] = tr_active
        ev['num_active_meds'] = ev_active
        tr['num_changed_meds'] = tr_changed
        ev['num_changed_meds'] = ev_changed
        tr['num_up_meds'] = tr_up
        ev['num_up_meds'] = ev_up
        extra_numeric.extend(['num_active_meds', 'num_changed_meds', 'num_up_meds'])

    if 'insulin' in tr and 'insulin' in ev:
        ins_map = {'No': 0.0, 'Steady': 1.0, 'Down': 2.0, 'Up': 3.0}
        tr['insulin_code'] = np.array([ins_map.get(x, 0.0) for x in tr['insulin']], dtype=np.float64)
        ev['insulin_code'] = np.array([ins_map.get(x, 0.0) for x in ev['insulin']], dtype=np.float64)
        extra_numeric.append('insulin_code')

    # 6. Age parsing to numeric midpoint
    if 'age' in tr and 'age' in ev:
        def _parse_age(arr):
            res = np.empty(len(arr), dtype=np.float64)
            for i, val in enumerate(arr):
                if val is None or not isinstance(val, str):
                    res[i] = np.nan
                else:
                    s = val.strip("[]()")
                    parts = s.split("-")
                    if len(parts) == 2:
                        try:
                            res[i] = (float(parts[0]) + float(parts[1])) / 2.0
                        except ValueError:
                            res[i] = np.nan
                    else:
                        res[i] = np.nan
            return res

        tr['age_numeric'] = _parse_age(tr['age'])
        ev['age_numeric'] = _parse_age(ev['age'])
        extra_numeric.append('age_numeric')

    # 7. Lab test flags
    if 'A1Cresult' in tr and 'A1Cresult' in ev:
        tr['a1c_tested'] = ((tr['A1Cresult'] != 'None') & (tr['A1Cresult'] != None)).astype(np.float64)
        ev['a1c_tested'] = ((ev['A1Cresult'] != 'None') & (ev['A1Cresult'] != None)).astype(np.float64)
        tr['a1c_abnormal'] = np.isin(tr['A1Cresult'], ['>7', '>8']).astype(np.float64)
        ev['a1c_abnormal'] = np.isin(ev['A1Cresult'], ['>7', '>8']).astype(np.float64)
        extra_numeric.extend(['a1c_tested', 'a1c_abnormal'])

    if 'max_glu_serum' in tr and 'max_glu_serum' in ev:
        tr['glu_tested'] = ((tr['max_glu_serum'] != 'None') & (tr['max_glu_serum'] != None)).astype(np.float64)
        ev['glu_tested'] = ((ev['max_glu_serum'] != 'None') & (ev['max_glu_serum'] != None)).astype(np.float64)
        tr['glu_abnormal'] = np.isin(tr['max_glu_serum'], ['>200', '>300']).astype(np.float64)
        ev['glu_abnormal'] = np.isin(ev['max_glu_serum'], ['>200', '>300']).astype(np.float64)
        extra_numeric.extend(['glu_tested', 'glu_abnormal'])

    # Encode tabular data
    encoder = TabularEncoder(min_count=40)
    encoder.fit(task, tr, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(tr)
    x_eval = encoder.transform(ev)
    cat_indices = encoder.categorical_indices

    # Diverse LightGBM ensemble configurations
    model_configs = [
        {
            'n_estimators': 220,
            'learning_rate': 0.02,
            'num_leaves': 31,
            'min_child_samples': 150,
            'subsample': 0.70,
            'colsample_bytree': 0.65,
            'reg_lambda': 0.5,
            'cat_smooth': 50.0,
            'random_state': 42,
        },
        {
            'n_estimators': 250,
            'learning_rate': 0.016,
            'num_leaves': 42,
            'min_child_samples': 90,
            'subsample': 0.75,
            'colsample_bytree': 0.70,
            'reg_lambda': 1.5,
            'cat_smooth': 30.0,
            'random_state': 101,
        },
        {
            'n_estimators': 180,
            'learning_rate': 0.025,
            'num_leaves': 20,
            'min_child_samples': 220,
            'subsample': 0.80,
            'colsample_bytree': 0.55,
            'reg_lambda': 5.0,
            'cat_smooth': 80.0,
            'random_state': 2024,
        },
        {
            'n_estimators': 230,
            'learning_rate': 0.018,
            'num_leaves': 28,
            'min_child_samples': 130,
            'subsample': 0.65,
            'colsample_bytree': 0.60,
            'reg_lambda': 0.1,
            'cat_smooth': 60.0,
            'random_state': 777,
        },
        {
            'n_estimators': 210,
            'learning_rate': 0.022,
            'num_leaves': 36,
            'min_child_samples': 170,
            'subsample': 0.70,
            'colsample_bytree': 0.75,
            'reg_lambda': 2.0,
            'cat_smooth': 40.0,
            'random_state': 9999,
        },
    ]

    preds = np.zeros(n_eval, dtype=np.float64)
    for cfg in model_configs:
        model = lgb.LGBMClassifier(
            **cfg,
            subsample_freq=1,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(x_train, y_train, categorical_feature=cat_indices)
        preds += model.predict_proba(x_eval)[:, 1]

    preds /= len(model_configs)
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
