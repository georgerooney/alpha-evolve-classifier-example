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
    if "icd9_group" not in globals():
        from clinical_toolkit import icd9_group, icd9_comorbidities, oof_target_encode

    tr = dict(train)
    ev = dict(eval_cols)
    extra_num = []
    extra_cat = []

    # 1. Comorbidities & Syndemic Interactions
    cm_train = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k, v in cm_train.items():
        tr[k] = v.astype(np.float64)
        ev[k] = cm_eval[k].astype(np.float64)
        extra_num.append(k)

    tr['cm_cardiorenal'] = (cm_train['cm_renal'] * cm_train['cm_chf']).astype(np.float64)
    ev['cm_cardiorenal'] = (cm_eval['cm_renal'] * cm_eval['cm_chf']).astype(np.float64)
    extra_num.append('cm_cardiorenal')

    tr['cm_severe_score'] = (cm_train['cm_count'] + 2.0 * cm_train['cm_renal'] + 2.0 * cm_train['cm_chf']).astype(np.float64)
    ev['cm_severe_score'] = (cm_eval['cm_count'] + 2.0 * cm_eval['cm_renal'] + 2.0 * cm_eval['cm_chf']).astype(np.float64)
    extra_num.append('cm_severe_score')

    # 2. ICD-9 Chapter Topology & Relational Crossing
    tr['diag_1_group'] = np.array([icd9_group(c) for c in train['diag_1']], dtype=object)
    ev['diag_1_group'] = np.array([icd9_group(c) for c in eval_cols['diag_1']], dtype=object)
    tr['diag_2_group'] = np.array([icd9_group(c) for c in train['diag_2']], dtype=object)
    ev['diag_2_group'] = np.array([icd9_group(c) for c in eval_cols['diag_2']], dtype=object)
    tr['diag_3_group'] = np.array([icd9_group(c) for c in train['diag_3']], dtype=object)
    ev['diag_3_group'] = np.array([icd9_group(c) for c in eval_cols['diag_3']], dtype=object)
    extra_cat.extend(['diag_1_group', 'diag_2_group', 'diag_3_group'])

    tr['disch_x_diag1'] = np.array([f"{d}_{g}" for d, g in zip(train['discharge_disposition_id'], tr['diag_1_group'])], dtype=object)
    ev['disch_x_diag1'] = np.array([f"{d}_{g}" for d, g in zip(eval_cols['discharge_disposition_id'], ev['diag_1_group'])], dtype=object)
    extra_cat.append('disch_x_diag1')

    # 3. Out-Of-Fold Smoothed Target Encoding on High-Cardinality Fields
    te_cols = ['diag_1', 'discharge_disposition_id', 'medical_specialty', 'admission_source_id', 'disch_x_diag1']
    for cname in te_cols:
        t_enc, e_enc = oof_target_encode(tr[cname], y_train, ev[cname], n_folds=5, smoothing=25.0, seed=42)
        te_name = f'{cname}_te'
        tr[te_name] = t_enc.astype(np.float64)
        ev[te_name] = e_enc.astype(np.float64)
        extra_num.append(te_name)

    # 4. Metabolic Polypharmacy & Regimen Volatility
    med_value_set = {'No', 'Steady', 'Up', 'Down'}
    med_columns = [
        c for c in task.categorical_columns
        if c in train and set(x for x in train[c] if x is not None).issubset(med_value_set)
    ]

    for split, src in [(tr, train), (ev, eval_cols)]:
        n_r = len(src['time_in_hospital'])
        act = np.zeros(n_r, dtype=np.float64)
        up = np.zeros(n_r, dtype=np.float64)
        down = np.zeros(n_r, dtype=np.float64)
        for c in med_columns:
            vals = src[c]
            act += ((vals == 'Steady') | (vals == 'Up') | (vals == 'Down')).astype(np.float64)
            up += (vals == 'Up').astype(np.float64)
            down += (vals == 'Down').astype(np.float64)

        split['med_active_count'] = act
        split['med_up_count'] = up
        split['med_down_count'] = down
        split['med_changed_count'] = up + down
        split['med_volatility'] = (up + down) / (act + 1.0)

        ins = src['insulin']
        split['insulin_active'] = ((ins == 'Steady') | (ins == 'Up') | (ins == 'Down')).astype(np.float64)
        split['insulin_up'] = (ins == 'Up').astype(np.float64)
        split['insulin_down'] = (ins == 'Down').astype(np.float64)
        split['insulin_changed'] = ((ins == 'Up') | (ins == 'Down')).astype(np.float64)

    extra_num.extend([
        'med_active_count', 'med_up_count', 'med_down_count', 'med_changed_count', 'med_volatility',
        'insulin_active', 'insulin_up', 'insulin_down', 'insulin_changed'
    ])

    # 5. Clinical Velocity, Care Intensity & Utilization Kinetics
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }

    for split, src in [(tr, train), (ev, eval_cols)]:
        t_hosp = np.maximum(src['time_in_hospital'].astype(np.float64), 0.5)
        split['labs_per_day'] = src['num_lab_procedures'] / t_hosp
        split['meds_per_day'] = src['num_medications'] / t_hosp
        split['procedures_per_day'] = src['num_procedures'] / t_hosp
        split['diagnoses_per_day'] = src['number_diagnoses'] / t_hosp
        split['lab_per_med'] = src['num_lab_procedures'] / (src['num_medications'] + 1.0)
        split['proc_per_med'] = src['num_procedures'] / (src['num_medications'] + 1.0)
        split['care_intensity'] = (src['num_lab_procedures'] + 2.0 * src['num_procedures'] + 1.5 * src['num_medications']) / t_hosp

        inpatient = src['number_inpatient'].astype(np.float64)
        emergency = src['number_emergency'].astype(np.float64)
        outpatient = src['number_outpatient'].astype(np.float64)
        tot_visits = inpatient + emergency + outpatient

        split['prior_visits_total'] = tot_visits
        split['has_prior_inpatient'] = (inpatient > 0).astype(np.float64)
        split['frequent_inpatient'] = (inpatient >= 2).astype(np.float64)
        split['acute_velocity'] = (inpatient * 2.0 + emergency) / (tot_visits + 1.0)
        split['inpatient_intensity'] = inpatient / t_hosp

        age_num = np.array([age_map.get(a, 65.0) for a in src['age']], dtype=np.float64)
        split['age_num'] = age_num
        split['age_x_inpatient'] = age_num * inpatient

        split['discharged_to_facility'] = np.isin(
            src['discharge_disposition_id'],
            ['3', '4', '5', '6', '22', '23', '24', '28']
        ).astype(np.float64)
        split['emergency_admission'] = np.isin(src['admission_type_id'], ['1', '2']).astype(np.float64)

        split['a1c_tested'] = (src['A1Cresult'] != 'None').astype(np.float64)
        split['glu_tested'] = (src['max_glu_serum'] != 'None').astype(np.float64)

    extra_num.extend([
        'labs_per_day', 'meds_per_day', 'procedures_per_day', 'diagnoses_per_day',
        'lab_per_med', 'proc_per_med', 'care_intensity',
        'prior_visits_total', 'has_prior_inpatient', 'frequent_inpatient',
        'acute_velocity', 'inpatient_intensity',
        'age_num', 'age_x_inpatient',
        'discharged_to_facility', 'emergency_admission',
        'a1c_tested', 'glu_tested'
    ])

    encoder = TabularEncoder(min_count=35)
    encoder.fit(task, tr, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_train = encoder.transform(tr)
    x_eval = encoder.transform(ev)

    # 6. Heterogeneous Multi-Scale GBDT Ensemble
    m1 = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.020,
        num_leaves=28,
        min_child_samples=160,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=1.0,
        reg_alpha=0.05,
        cat_smooth=50.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p1 = m1.predict_proba(x_eval)[:, 1]

    m2 = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.016,
        num_leaves=18,
        min_child_samples=100,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=4.0,
        reg_alpha=0.50,
        cat_smooth=80.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p2 = m2.predict_proba(x_eval)[:, 1]

    m3 = lgb.LGBMClassifier(
        n_estimators=160,
        learning_rate=0.022,
        num_leaves=36,
        min_child_samples=220,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=0.1,
        reg_alpha=0.01,
        cat_smooth=65.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p3 = m3.predict_proba(x_eval)[:, 1]

    return 0.40 * p1 + 0.35 * p2 + 0.25 * p3


# EVOLVE-BLOCK-END
# =====================================================================
