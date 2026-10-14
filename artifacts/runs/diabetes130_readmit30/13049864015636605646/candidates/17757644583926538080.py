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
    """Fit an ensemble on `train`/`y_train` with clinical feature engineering and score `eval_cols`."""
    tr = dict(train)
    ev = dict(eval_cols)
    y_int = y_train.astype(int)

    def safe_num(arr, fill=0.0):
        return np.where(np.isnan(arr), fill, arr)

    # 1. Utilisation & Prior Admissions
    inp_tr, inp_ev = safe_num(tr['number_inpatient']), safe_num(ev['number_inpatient'])
    em_tr, em_ev = safe_num(tr['number_emergency']), safe_num(ev['number_emergency'])
    out_tr, out_ev = safe_num(tr['number_outpatient']), safe_num(ev['number_outpatient'])

    tot_tr = inp_tr + em_tr + out_tr
    tot_ev = inp_ev + em_ev + out_ev
    tr['total_prior_visits'] = tot_tr
    ev['total_prior_visits'] = tot_ev

    tr['prior_visit_severity'] = inp_tr * 3.0 + em_tr * 2.0 + out_tr
    ev['prior_visit_severity'] = inp_ev * 3.0 + em_ev * 2.0 + out_ev

    tr['inpatient_ratio'] = inp_tr / (tot_tr + 1.0)
    ev['inpatient_ratio'] = inp_ev / (tot_ev + 1.0)
    tr['emergency_ratio'] = em_tr / (tot_tr + 1.0)
    ev['emergency_ratio'] = em_ev / (tot_ev + 1.0)

    tr['has_inpatient'] = (inp_tr > 0).astype(np.float64)
    ev['has_inpatient'] = (inp_ev > 0).astype(np.float64)
    tr['has_emergency'] = (em_tr > 0).astype(np.float64)
    ev['has_emergency'] = (em_ev > 0).astype(np.float64)
    tr['has_multiple_inpatient'] = (inp_tr >= 2).astype(np.float64)
    ev['has_multiple_inpatient'] = (inp_ev >= 2).astype(np.float64)

    tr['log1p_inpatient'] = np.log1p(np.maximum(inp_tr, 0.0))
    ev['log1p_inpatient'] = np.log1p(np.maximum(inp_ev, 0.0))
    tr['log1p_total_visits'] = np.log1p(np.maximum(tot_tr, 0.0))
    ev['log1p_total_visits'] = np.log1p(np.maximum(tot_ev, 0.0))

    # 2. Encounter Intensity & Complexity
    time_tr = np.maximum(safe_num(tr['time_in_hospital'], 1.0), 1.0)
    time_ev = np.maximum(safe_num(ev['time_in_hospital'], 1.0), 1.0)
    meds_tr, meds_ev = safe_num(tr['num_medications']), safe_num(ev['num_medications'])
    labs_tr, labs_ev = safe_num(tr['num_lab_procedures']), safe_num(ev['num_lab_procedures'])
    procs_tr, procs_ev = safe_num(tr['num_procedures']), safe_num(ev['num_procedures'])
    diags_tr, diags_ev = safe_num(tr['number_diagnoses']), safe_num(ev['number_diagnoses'])

    tr['meds_per_day'] = meds_tr / time_tr
    ev['meds_per_day'] = meds_ev / time_ev
    tr['labs_per_day'] = labs_tr / time_tr
    ev['labs_per_day'] = labs_ev / time_ev
    tr['procs_per_day'] = procs_tr / time_tr
    ev['procs_per_day'] = procs_ev / time_ev
    tr['diag_per_day'] = diags_tr / time_tr
    ev['diag_per_day'] = diags_ev / time_ev

    tr['service_intensity'] = time_tr * (meds_tr + labs_tr + procs_tr)
    ev['service_intensity'] = time_ev * (meds_ev + labs_ev + procs_ev)
    tr['labs_per_med'] = labs_tr / (meds_tr + 1.0)
    ev['labs_per_med'] = labs_ev / (meds_ev + 1.0)
    tr['procs_per_med'] = procs_tr / (meds_tr + 1.0)
    ev['procs_per_med'] = procs_ev / (meds_ev + 1.0)
    tr['diag_per_med'] = diags_tr / (meds_tr + 1.0)
    ev['diag_per_med'] = diags_ev / (meds_ev + 1.0)
    tr['severity_score'] = (meds_tr / 10.0) + (time_tr / 3.0) + (diags_tr / 3.0) + (inp_tr * 2.0)
    ev['severity_score'] = (meds_ev / 10.0) + (time_ev / 3.0) + (diags_ev / 3.0) + (inp_ev * 2.0)

    # 3. Numeric Age & Demographics
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    age_tr = np.array([age_map.get(a, np.nan) for a in tr['age']], dtype=np.float64)
    age_ev = np.array([age_map.get(a, np.nan) for a in ev['age']], dtype=np.float64)
    tr['age_num'] = age_tr
    ev['age_num'] = age_ev
    tr['is_elderly'] = (age_tr >= 75.0).astype(np.float64)
    ev['is_elderly'] = (age_ev >= 75.0).astype(np.float64)
    tr['is_young'] = (age_tr < 40.0).astype(np.float64)
    ev['is_young'] = (age_ev < 40.0).astype(np.float64)

    # 4. Charlson Comorbidities & Risk Indexes
    cm_tr = icd9_comorbidities(tr['diag_1'], tr['diag_2'], tr['diag_3'])
    cm_ev = icd9_comorbidities(ev['diag_1'], ev['diag_2'], ev['diag_3'])
    for k in cm_tr:
        tr[k] = cm_tr[k]
        ev[k] = cm_ev[k]

    def weighted_charlson(c):
        return (
            1.0 * (c['cm_mi'] + c['cm_chf'] + c['cm_pvd'] + c['cm_cerebrovascular'] +
                   c['cm_dementia'] + c['cm_pulmonary'] + c['cm_rheumatic'] + c['cm_peptic_ulcer']) +
            2.0 * (c['cm_renal'] + c['cm_diabetes_complicated'] + c['cm_hemiplegia'] + c['cm_cancer']) +
            3.0 * c['cm_liver'] +
            6.0 * (c['cm_metastatic'] + c['cm_hiv'])
        )

    tr['charlson_weighted'] = weighted_charlson(cm_tr)
    ev['charlson_weighted'] = weighted_charlson(cm_ev)
    tr['cardiorenal'] = ((cm_tr['cm_chf'] > 0) & (cm_tr['cm_renal'] > 0)).astype(np.float64)
    ev['cardiorenal'] = ((cm_ev['cm_chf'] > 0) & (cm_ev['cm_renal'] > 0)).astype(np.float64)
    tr['diabetes_renal'] = ((cm_tr['cm_diabetes_complicated'] > 0) & (cm_tr['cm_renal'] > 0)).astype(np.float64)
    ev['diabetes_renal'] = ((cm_ev['cm_diabetes_complicated'] > 0) & (cm_ev['cm_renal'] > 0)).astype(np.float64)
    tr['has_multimorbidity'] = (cm_tr['cm_count'] >= 2.0).astype(np.float64)
    ev['has_multimorbidity'] = (cm_ev['cm_count'] >= 2.0).astype(np.float64)

    # 5. Diagnosis Chapter Groupings
    extra_categorical = ['diag_1_grp', 'diag_2_grp', 'diag_3_grp']
    d1_grp_tr = np.array([icd9_group(c) for c in tr['diag_1']], dtype=object)
    d1_grp_ev = np.array([icd9_group(c) for c in ev['diag_1']], dtype=object)
    d2_grp_tr = np.array([icd9_group(c) for c in tr['diag_2']], dtype=object)
    d2_grp_ev = np.array([icd9_group(c) for c in ev['diag_2']], dtype=object)
    d3_grp_tr = np.array([icd9_group(c) for c in tr['diag_3']], dtype=object)
    d3_grp_ev = np.array([icd9_group(c) for c in ev['diag_3']], dtype=object)

    tr['diag_1_grp'], ev['diag_1_grp'] = d1_grp_tr, d1_grp_ev
    tr['diag_2_grp'], ev['diag_2_grp'] = d2_grp_tr, d2_grp_ev
    tr['diag_3_grp'], ev['diag_3_grp'] = d3_grp_tr, d3_grp_ev

    tr['diag1_circulatory'] = (d1_grp_tr == 'circulatory').astype(np.float64)
    ev['diag1_circulatory'] = (d1_grp_ev == 'circulatory').astype(np.float64)
    tr['diag1_diabetes'] = (d1_grp_tr == 'diabetes').astype(np.float64)
    ev['diag1_diabetes'] = (d1_grp_ev == 'diabetes').astype(np.float64)
    tr['diag1_respiratory'] = (d1_grp_tr == 'respiratory').astype(np.float64)
    ev['diag1_respiratory'] = (d1_grp_ev == 'respiratory').astype(np.float64)
    tr['diag1_digestive'] = (d1_grp_tr == 'digestive').astype(np.float64)
    ev['diag1_digestive'] = (d1_grp_ev == 'digestive').astype(np.float64)

    # 6. Medication Dynamics & Regimen Adjustments
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    avail_meds = [m for m in med_cols if m in tr and m in ev]
    n_tr, n_ev = len(time_tr), len(time_ev)
    active_tr = np.zeros(n_tr, dtype=np.float64)
    active_ev = np.zeros(n_ev, dtype=np.float64)
    up_tr = np.zeros(n_tr, dtype=np.float64)
    up_ev = np.zeros(n_ev, dtype=np.float64)
    down_tr = np.zeros(n_tr, dtype=np.float64)
    down_ev = np.zeros(n_ev, dtype=np.float64)
    steady_tr = np.zeros(n_tr, dtype=np.float64)
    steady_ev = np.zeros(n_ev, dtype=np.float64)

    for m in avail_meds:
        c_tr, c_ev = tr[m], ev[m]
        active_tr += ((c_tr != 'No') & (c_tr != None)).astype(np.float64)
        active_ev += ((c_ev != 'No') & (c_ev != None)).astype(np.float64)
        up_tr += (c_tr == 'Up').astype(np.float64)
        up_ev += (c_ev == 'Up').astype(np.float64)
        down_tr += (c_tr == 'Down').astype(np.float64)
        down_ev += (c_ev == 'Down').astype(np.float64)
        steady_tr += (c_tr == 'Steady').astype(np.float64)
        steady_ev += (c_ev == 'Steady').astype(np.float64)

    tr['num_active_meds'] = active_tr
    ev['num_active_meds'] = active_ev
    tr['num_meds_up'] = up_tr
    ev['num_meds_up'] = up_ev
    tr['num_meds_down'] = down_tr
    ev['num_meds_down'] = down_ev
    tr['num_meds_steady'] = steady_tr
    ev['num_meds_steady'] = steady_ev
    tr['num_meds_changed'] = up_tr + down_tr
    ev['num_meds_changed'] = up_ev + down_ev

    if 'insulin' in tr:
        ins_tr, ins_ev = tr['insulin'], ev['insulin']
        tr['insulin_prescribed'] = ((ins_tr != 'No') & (ins_tr != None)).astype(np.float64)
        ev['insulin_prescribed'] = ((ins_ev != 'No') & (ins_ev != None)).astype(np.float64)
        tr['insulin_up'] = (ins_tr == 'Up').astype(np.float64)
        ev['insulin_up'] = (ins_ev == 'Up').astype(np.float64)
        tr['insulin_down'] = (ins_tr == 'Down').astype(np.float64)
        ev['insulin_down'] = (ins_ev == 'Down').astype(np.float64)
        tr['insulin_steady'] = (ins_tr == 'Steady').astype(np.float64)
        ev['insulin_steady'] = (ins_ev == 'Steady').astype(np.float64)
        tr['insulin_changed'] = ((ins_tr == 'Up') | (ins_tr == 'Down')).astype(np.float64)
        ev['insulin_changed'] = ((ins_ev == 'Up') | (ins_ev == 'Down')).astype(np.float64)
        tr['oral_and_insulin'] = ((tr['insulin_prescribed'] == 1.0) & (active_tr > 1.0)).astype(np.float64)
        ev['oral_and_insulin'] = ((ev['insulin_prescribed'] == 1.0) & (active_ev > 1.0)).astype(np.float64)

    if 'change' in tr:
        tr['med_changed_flag'] = (tr['change'] == 'Ch').astype(np.float64)
        ev['med_changed_flag'] = (ev['change'] == 'Ch').astype(np.float64)
    if 'diabetesMed' in tr:
        tr['diabetes_med_flag'] = (tr['diabetesMed'] == 'Yes').astype(np.float64)
        ev['diabetes_med_flag'] = (ev['diabetesMed'] == 'Yes').astype(np.float64)

    # 7. Glycemic Testing (A1C & Serum Glucose)
    if 'A1Cresult' in tr:
        a1c_tr, a1c_ev = tr['A1Cresult'], ev['A1Cresult']
        tr['a1c_tested'] = ((a1c_tr != 'None') & (a1c_tr != None)).astype(np.float64)
        ev['a1c_tested'] = ((a1c_ev != 'None') & (a1c_ev != None)).astype(np.float64)
        tr['a1c_high'] = ((a1c_tr == '>8') | (a1c_tr == '>7')).astype(np.float64)
        ev['a1c_high'] = ((a1c_ev == '>8') | (a1c_ev == '>7')).astype(np.float64)
        tr['a1c_normal'] = (a1c_tr == 'Norm').astype(np.float64)
        ev['a1c_normal'] = (a1c_ev == 'Norm').astype(np.float64)
        if 'change' in tr:
            tr['a1c_high_no_change'] = ((tr['a1c_high'] == 1.0) & (tr['change'] == 'No')).astype(np.float64)
            ev['a1c_high_no_change'] = ((ev['a1c_high'] == 1.0) & (ev['change'] == 'No')).astype(np.float64)
            tr['a1c_high_with_change'] = ((tr['a1c_high'] == 1.0) & (tr['change'] == 'Ch')).astype(np.float64)
            ev['a1c_high_with_change'] = ((ev['a1c_high'] == 1.0) & (ev['change'] == 'Ch')).astype(np.float64)

    if 'max_glu_serum' in tr:
        glu_tr, glu_ev = tr['max_glu_serum'], ev['max_glu_serum']
        tr['glu_tested'] = ((glu_tr != 'None') & (glu_tr != None)).astype(np.float64)
        ev['glu_tested'] = ((glu_ev != 'None') & (glu_ev != None)).astype(np.float64)
        tr['glu_high'] = ((glu_tr == '>200') | (glu_tr == '>300')).astype(np.float64)
        ev['glu_high'] = ((glu_ev == '>200') | (glu_ev == '>300')).astype(np.float64)
        tr['glu_normal'] = (glu_tr == 'Norm').astype(np.float64)
        ev['glu_normal'] = (glu_ev == 'Norm').astype(np.float64)

    # 8. Admission / Discharge Dispositions & Documentation Signals
    if 'discharge_disposition_id' in tr:
        tr['discharged_home'] = (tr['discharge_disposition_id'] == '1').astype(np.float64)
        ev['discharged_home'] = (ev['discharge_disposition_id'] == '1').astype(np.float64)
        snf_set = {'2', '3', '4', '5', '6', '22', '23', '24'}
        tr['discharged_facility'] = np.array([1.0 if x in snf_set else 0.0 for x in tr['discharge_disposition_id']], dtype=np.float64)
        ev['discharged_facility'] = np.array([1.0 if x in snf_set else 0.0 for x in ev['discharge_disposition_id']], dtype=np.float64)

    if 'admission_type_id' in tr:
        tr['admission_er'] = (tr['admission_type_id'] == '1').astype(np.float64)
        ev['admission_er'] = (ev['admission_type_id'] == '1').astype(np.float64)
        tr['admission_elective'] = (tr['admission_type_id'] == '3').astype(np.float64)
        ev['admission_elective'] = (ev['admission_type_id'] == '3').astype(np.float64)

    if 'admission_source_id' in tr:
        tr['source_er'] = (tr['admission_source_id'] == '7').astype(np.float64)
        ev['source_er'] = (ev['admission_source_id'] == '7').astype(np.float64)
        tr['source_referral'] = ((tr['admission_source_id'] == '1') | (tr['admission_source_id'] == '2')).astype(np.float64)
        ev['source_referral'] = ((ev['admission_source_id'] == '1') | (ev['admission_source_id'] == '2')).astype(np.float64)

    if 'medical_specialty' in tr:
        tr['specialty_missing'] = np.array([1.0 if x is None else 0.0 for x in tr['medical_specialty']], dtype=np.float64)
        ev['specialty_missing'] = np.array([1.0 if x is None else 0.0 for x in ev['medical_specialty']], dtype=np.float64)

    if 'payer_code' in tr:
        tr['payer_missing'] = np.array([1.0 if x is None else 0.0 for x in tr['payer_code']], dtype=np.float64)
        ev['payer_missing'] = np.array([1.0 if x is None else 0.0 for x in ev['payer_code']], dtype=np.float64)

    if 'weight' in tr:
        tr['weight_recorded'] = np.array([1.0 if x is not None else 0.0 for x in tr['weight']], dtype=np.float64)
        ev['weight_recorded'] = np.array([1.0 if x is not None else 0.0 for x in ev['weight']], dtype=np.float64)

    # 9. Out-of-Fold Target Encodings
    te_columns = [
        'diag_1', 'diag_2', 'diag_3',
        'discharge_disposition_id', 'admission_source_id', 'admission_type_id',
        'medical_specialty', 'payer_code', 'diag_1_grp'
    ]
    for col in te_columns:
        if col in tr and col in ev:
            te_tr, te_ev = oof_target_encode(tr[col], y_int, ev[col], n_folds=5, smoothing=25.0, seed=42)
            tr[f'{col}_te'] = te_tr
            ev[f'{col}_te'] = te_ev

    # Diagnosis x Discharge Interaction Target Encoding
    if 'diag_1' in tr and 'discharge_disposition_id' in tr:
        d1_dd_tr = np.array([f"{d}_{s}" for d, s in zip(tr['diag_1'], tr['discharge_disposition_id'])], dtype=object)
        d1_dd_ev = np.array([f"{d}_{s}" for d, s in zip(ev['diag_1'], ev['discharge_disposition_id'])], dtype=object)
        te_d1_dd_tr, te_d1_dd_ev = oof_target_encode(d1_dd_tr, y_int, d1_dd_ev, n_folds=5, smoothing=30.0, seed=42)
        tr['diag1_discharge_te'] = te_d1_dd_tr
        ev['diag1_discharge_te'] = te_d1_dd_ev

    if 'diag_1_grp' in tr and 'discharge_disposition_id' in tr:
        grp_dd_tr = np.array([f"{g}_{s}" for g, s in zip(tr['diag_1_grp'], tr['discharge_disposition_id'])], dtype=object)
        grp_dd_ev = np.array([f"{g}_{s}" for g, s in zip(ev['diag_1_grp'], ev['discharge_disposition_id'])], dtype=object)
        te_grp_dd_tr, te_grp_dd_ev = oof_target_encode(grp_dd_tr, y_int, grp_dd_ev, n_folds=5, smoothing=20.0, seed=42)
        tr['diag1grp_discharge_te'] = te_grp_dd_tr
        ev['diag1grp_discharge_te'] = te_grp_dd_ev

    # Identify all derived numeric columns
    extra_numeric = [
        k for k in tr
        if k not in task.numeric_columns and k not in task.categorical_columns and k not in extra_categorical
    ]

    # Encode Tabular Matrix
    encoder = TabularEncoder(min_count=35)
    encoder.fit(task, tr, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(tr)
    x_eval = encoder.transform(ev)
    cat_indices = encoder.categorical_indices

    # 10. Diverse Multi-Model Gradient-Boosted Ensemble
    m1 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.018,
        num_leaves=31,
        min_child_samples=180,
        subsample=0.72,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_lambda=1.0,
        reg_alpha=0.1,
        cat_smooth=60.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    m2 = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.015,
        num_leaves=40,
        min_child_samples=150,
        subsample=0.68,
        subsample_freq=1,
        colsample_bytree=0.52,
        reg_lambda=2.0,
        reg_alpha=0.3,
        cat_smooth=45.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )

    m3 = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.022,
        num_leaves=20,
        max_depth=5,
        min_child_samples=220,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.68,
        reg_lambda=0.2,
        reg_alpha=0.01,
        cat_smooth=75.0,
        random_state=999,
        n_jobs=1,
        verbose=-1,
    )

    m4 = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.016,
        num_leaves=25,
        min_child_samples=160,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.58,
        reg_lambda=0.8,
        reg_alpha=0.15,
        cat_smooth=55.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )

    m1.fit(x_train, y_train, categorical_feature=cat_indices)
    m2.fit(x_train, y_train, categorical_feature=cat_indices)
    m3.fit(x_train, y_train, categorical_feature=cat_indices)
    m4.fit(x_train, y_train, categorical_feature=cat_indices)

    p1 = m1.predict_proba(x_eval)[:, 1]
    p2 = m2.predict_proba(x_eval)[:, 1]
    p3 = m3.predict_proba(x_eval)[:, 1]
    p4 = m4.predict_proba(x_eval)[:, 1]

    pred = 0.30 * p1 + 0.30 * p2 + 0.20 * p3 + 0.20 * p4
    pred = np.clip(pred, 0.0, 1.0)
    if np.isnan(pred).any():
        pred = np.nan_to_num(pred, nan=0.114)

    return pred


# EVOLVE-BLOCK-END
# =====================================================================
