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

    # 1. Charlson comorbidity flags from diagnosis codes
    diag_cols = [c for c in ['diag_1', 'diag_2', 'diag_3'] if c in tr and c in ev]
    if diag_cols:
        cm_tr = icd9_comorbidities(*(tr[c] for c in diag_cols))
        cm_ev = icd9_comorbidities(*(ev[c] for c in diag_cols))
        for k, v in cm_tr.items():
            tr[k] = v.astype(np.float64)
            ev[k] = cm_ev[k].astype(np.float64)
            extra_numeric.append(k)

        # ICD-9 chapter groupings
        for col in diag_cols:
            grp_col = f"{col}_group"
            tr[grp_col] = np.array([icd9_group(c) for c in tr[col]], dtype=object)
            ev[grp_col] = np.array([icd9_group(c) for c in ev[col]], dtype=object)
            extra_categorical.append(grp_col)

    # Create categorical interaction columns
    if 'discharge_disposition_id' in tr and 'admission_source_id' in tr:
        tr['disch_adm'] = np.array([f"{d}_{s}" for d, s in zip(tr['discharge_disposition_id'], tr['admission_source_id'])], dtype=object)
        ev['disch_adm'] = np.array([f"{d}_{s}" for d, s in zip(ev['discharge_disposition_id'], ev['admission_source_id'])], dtype=object)
        extra_categorical.append('disch_adm')

    if 'admission_type_id' in tr and 'admission_source_id' in tr:
        tr['type_adm'] = np.array([f"{t}_{s}" for t, s in zip(tr['admission_type_id'], tr['admission_source_id'])], dtype=object)
        ev['type_adm'] = np.array([f"{t}_{s}" for t, s in zip(ev['admission_type_id'], ev['admission_source_id'])], dtype=object)
        extra_categorical.append('type_adm')

    if 'diag_1_group' in tr and 'diag_2_group' in tr:
        tr['diag12_group'] = np.array([f"{g1}_{g2}" for g1, g2 in zip(tr['diag_1_group'], tr['diag_2_group'])], dtype=object)
        ev['diag12_group'] = np.array([f"{g1}_{g2}" for g1, g2 in zip(ev['diag_1_group'], ev['diag_2_group'])], dtype=object)
        extra_categorical.append('diag12_group')

    # 2. Out-of-fold target encoding for high-cardinality categoricals
    te_candidates = [
        'discharge_disposition_id',
        'admission_type_id',
        'admission_source_id',
        'medical_specialty',
        'diag_1',
        'diag_2',
        'diag_3',
        'diag_1_group',
        'diag_2_group',
        'diag_3_group',
        'payer_code',
        'disch_adm',
        'type_adm',
        'diag12_group',
    ]
    prior_mean = float(np.mean(y_train))
    for c in te_candidates:
        if c in tr and c in ev:
            te_name = f"{c}_te"
            tr_te, ev_te = oof_target_encode(tr[c], y_train, ev[c], n_folds=5, smoothing=25.0, seed=42)
            tr[te_name] = np.nan_to_num(tr_te.astype(np.float64), nan=prior_mean)
            ev[te_name] = np.nan_to_num(ev_te.astype(np.float64), nan=prior_mean)
            extra_numeric.append(te_name)

    # 2.5 Clinical indicator flags from IDs
    if 'discharge_disposition_id' in tr and 'discharge_disposition_id' in ev:
        tr_disch = tr['discharge_disposition_id']
        ev_disch = ev['discharge_disposition_id']
        tr['disch_home'] = np.isin(tr_disch, ['1', '6', '8']).astype(np.float64)
        ev['disch_home'] = np.isin(ev_disch, ['1', '6', '8']).astype(np.float64)
        extra_numeric.append('disch_home')
        
        tr['disch_facility'] = np.isin(tr_disch, ['2', '3', '4', '5', '22', '23', '24', '30']).astype(np.float64)
        ev['disch_facility'] = np.isin(ev_disch, ['2', '3', '4', '5', '22', '23', '24', '30']).astype(np.float64)
        extra_numeric.append('disch_facility')

    if 'admission_source_id' in tr and 'admission_source_id' in ev:
        tr_adm = tr['admission_source_id']
        ev_adm = ev['admission_source_id']
        tr['adm_emergency'] = (tr_adm == '7').astype(np.float64)
        ev['adm_emergency'] = (ev_adm == '7').astype(np.float64)
        extra_numeric.append('adm_emergency')
        
        tr['adm_transfer'] = np.isin(tr_adm, ['4', '5', '6']).astype(np.float64)
        ev['adm_transfer'] = np.isin(ev_adm, ['4', '5', '6']).astype(np.float64)
        extra_numeric.append('adm_transfer')

    if 'admission_type_id' in tr and 'admission_type_id' in ev:
        tr_type = tr['admission_type_id']
        ev_type = ev['admission_type_id']
        tr['adm_type_urgent'] = np.isin(tr_type, ['1', '2', '7', '8']).astype(np.float64)
        ev['adm_type_urgent'] = np.isin(ev_type, ['1', '2', '7', '8']).astype(np.float64)
        extra_numeric.append('adm_type_urgent')

    if 'diag_1_group' in tr and 'diag_1_group' in ev:
        for grp in ['circulatory', 'respiratory', 'digestive', 'diabetes', 'injury']:
            col_name = f'primary_{grp}'
            tr[col_name] = (tr['diag_1_group'] == grp).astype(np.float64)
            ev[col_name] = (ev['diag_1_group'] == grp).astype(np.float64)
            extra_numeric.append(col_name)

    # 3. Utilization and intensity interactions
    inp_tr = np.nan_to_num(tr.get('number_inpatient', np.zeros(task.n_train)), nan=0.0)
    emg_tr = np.nan_to_num(tr.get('number_emergency', np.zeros(task.n_train)), nan=0.0)
    out_tr = np.nan_to_num(tr.get('number_outpatient', np.zeros(task.n_train)), nan=0.0)

    inp_ev = np.nan_to_num(ev.get('number_inpatient', np.zeros(task.n_eval)), nan=0.0)
    emg_ev = np.nan_to_num(ev.get('number_emergency', np.zeros(task.n_eval)), nan=0.0)
    out_ev = np.nan_to_num(ev.get('number_outpatient', np.zeros(task.n_eval)), nan=0.0)

    tot_tr = inp_tr + emg_tr + out_tr
    tot_ev = inp_ev + emg_ev + out_ev
    tr['total_visits'] = tot_tr
    ev['total_visits'] = tot_ev
    extra_numeric.append('total_visits')

    tr['inpatient_ratio'] = inp_tr / (tot_tr + 1.0)
    ev['inpatient_ratio'] = inp_ev / (tot_ev + 1.0)
    extra_numeric.append('inpatient_ratio')

    tr['emergency_ratio'] = emg_tr / (tot_tr + 1.0)
    ev['emergency_ratio'] = emg_ev / (tot_ev + 1.0)
    extra_numeric.append('emergency_ratio')

    tr['util_severity'] = inp_tr * 3.0 + emg_tr * 1.5 + out_tr * 0.5
    ev['util_severity'] = inp_ev * 3.0 + emg_ev * 1.5 + out_ev * 0.5
    extra_numeric.append('util_severity')

    tr['has_inpatient'] = (inp_tr > 0).astype(np.float64)
    ev['has_inpatient'] = (inp_ev > 0).astype(np.float64)
    extra_numeric.append('has_inpatient')

    time_tr = np.nan_to_num(tr.get('time_in_hospital', np.ones(task.n_train)), nan=1.0)
    time_ev = np.nan_to_num(ev.get('time_in_hospital', np.ones(task.n_eval)), nan=1.0)
    meds_tr = np.nan_to_num(tr.get('num_medications', np.zeros(task.n_train)), nan=0.0)
    meds_ev = np.nan_to_num(ev.get('num_medications', np.zeros(task.n_eval)), nan=0.0)
    labs_tr = np.nan_to_num(tr.get('num_lab_procedures', np.zeros(task.n_train)), nan=0.0)
    labs_ev = np.nan_to_num(ev.get('num_lab_procedures', np.zeros(task.n_eval)), nan=0.0)
    procs_tr = np.nan_to_num(tr.get('num_procedures', np.zeros(task.n_train)), nan=0.0)
    procs_ev = np.nan_to_num(ev.get('num_procedures', np.zeros(task.n_eval)), nan=0.0)

    tr['med_per_day'] = meds_tr / (time_tr + 0.1)
    ev['med_per_day'] = meds_ev / (time_ev + 0.1)
    extra_numeric.append('med_per_day')

    tr['lab_per_day'] = labs_tr / (time_tr + 0.1)
    ev['lab_per_day'] = labs_ev / (time_ev + 0.1)
    extra_numeric.append('lab_per_day')

    tr['proc_per_day'] = procs_tr / (time_tr + 0.1)
    ev['proc_per_day'] = procs_ev / (time_ev + 0.1)
    extra_numeric.append('proc_per_day')

    tr['time_x_inpatient'] = time_tr * (inp_tr + 1.0)
    ev['time_x_inpatient'] = time_ev * (inp_ev + 1.0)
    extra_numeric.append('time_x_inpatient')

    tr['meds_x_procs'] = meds_tr * procs_tr
    ev['meds_x_procs'] = meds_ev * procs_ev
    extra_numeric.append('meds_x_procs')

    # Continuous age mapping
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0,
    }
    if 'age' in tr and 'age' in ev:
        tr['age_numeric'] = np.array([age_map.get(a, np.nan) for a in tr['age']], dtype=np.float64)
        ev['age_numeric'] = np.array([age_map.get(a, np.nan) for a in ev['age']], dtype=np.float64)
        extra_numeric.append('age_numeric')

    # Age interactions and multimorbidity features
    if 'age_numeric' in tr and 'cm_count' in tr:
        tr['age_x_cm_count'] = tr['age_numeric'] * tr['cm_count']
        ev['age_x_cm_count'] = ev['age_numeric'] * ev['cm_count']
        extra_numeric.append('age_x_cm_count')

    if 'age_numeric' in tr and 'total_visits' in tr:
        tr['age_x_visits'] = tr['age_numeric'] * tr['total_visits']
        ev['age_x_visits'] = ev['age_numeric'] * ev['total_visits']
        extra_numeric.append('age_x_visits')

    if 'number_inpatient' in tr and 'cm_count' in tr:
        tr['inpatient_x_cm_count'] = tr['number_inpatient'] * tr['cm_count']
        ev['inpatient_x_cm_count'] = ev['number_inpatient'] * ev['cm_count']
        extra_numeric.append('inpatient_x_cm_count')

    if 'age_numeric' in tr:
        age_tr_val = tr['age_numeric']
        age_ev_val = ev['age_numeric']
        cm_count_tr = tr.get('cm_count', np.zeros(task.n_train))
        cm_count_ev = ev.get('cm_count', np.zeros(task.n_eval))
        
        tr['elderly_multimorbidity'] = ((age_tr_val >= 70.0) & (cm_count_tr >= 3.0)).astype(np.float64)
        ev['elderly_multimorbidity'] = ((age_ev_val >= 70.0) & (cm_count_ev >= 3.0)).astype(np.float64)
        extra_numeric.append('elderly_multimorbidity')
        
        tr['elderly_polypharmacy'] = ((age_tr_val >= 70.0) & (meds_tr >= 15.0)).astype(np.float64)
        ev['elderly_polypharmacy'] = ((age_ev_val >= 70.0) & (meds_ev >= 15.0)).astype(np.float64)
        extra_numeric.append('elderly_polypharmacy')

    tr['inpatient_x_diagnoses'] = inp_tr * np.nan_to_num(tr.get('number_diagnoses', np.zeros(task.n_train)), nan=0.0)
    ev['inpatient_x_diagnoses'] = inp_ev * np.nan_to_num(ev.get('number_diagnoses', np.zeros(task.n_eval)), nan=0.0)
    extra_numeric.append('inpatient_x_diagnoses')

    # 4. Medication regimen patterns and Low-Rank SVD Phenotype Decomposition
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    present_meds = [m for m in med_names if m in tr and m in ev]

    svd_feats_tr = []
    svd_feats_ev = []

    tr_active = np.zeros(task.n_train, dtype=np.float64)
    ev_active = np.zeros(task.n_eval, dtype=np.float64)
    tr_changed = np.zeros(task.n_train, dtype=np.float64)
    ev_changed = np.zeros(task.n_eval, dtype=np.float64)
    tr_up = np.zeros(task.n_train, dtype=np.float64)
    ev_up = np.zeros(task.n_eval, dtype=np.float64)
    tr_down = np.zeros(task.n_train, dtype=np.float64)
    ev_down = np.zeros(task.n_eval, dtype=np.float64)

    for m in present_meds:
        m_tr = tr[m]
        m_ev = ev[m]
        act_tr = ((m_tr != 'No') & (m_tr != None)).astype(np.float64)
        act_ev = ((m_ev != 'No') & (m_ev != None)).astype(np.float64)
        chg_tr = ((m_tr == 'Up') | (m_tr == 'Down')).astype(np.float64)
        chg_ev = ((m_ev == 'Up') | (m_ev == 'Down')).astype(np.float64)
        dir_tr = np.where(m_tr == 'Up', 1.0, np.where(m_tr == 'Down', -1.0, 0.0))
        dir_ev = np.where(m_ev == 'Up', 1.0, np.where(m_ev == 'Down', -1.0, 0.0))

        tr_active += act_tr
        ev_active += act_ev
        tr_changed += chg_tr
        ev_changed += chg_ev
        tr_up += (m_tr == 'Up').astype(np.float64)
        ev_up += (m_ev == 'Up').astype(np.float64)
        tr_down += (m_tr == 'Down').astype(np.float64)
        ev_down += (m_ev == 'Down').astype(np.float64)

        svd_feats_tr.extend([act_tr, chg_tr, dir_tr])
        svd_feats_ev.extend([act_ev, chg_ev, dir_ev])

    tr['num_active_meds'] = tr_active
    ev['num_active_meds'] = ev_active
    extra_numeric.append('num_active_meds')

    tr['diabetes_med_ratio'] = tr_active / (meds_tr + 1.0)
    ev['diabetes_med_ratio'] = ev_active / (meds_ev + 1.0)
    extra_numeric.append('diabetes_med_ratio')

    if 'metformin' in tr and 'insulin' in tr:
        met_tr = ((tr['metformin'] != 'No') & (tr['metformin'] != None)).astype(np.float64)
        met_ev = ((ev['metformin'] != 'No') & (ev['metformin'] != None)).astype(np.float64)
        ins_tr = ((tr['insulin'] != 'No') & (tr['insulin'] != None)).astype(np.float64)
        ins_ev = ((ev['insulin'] != 'No') & (ev['insulin'] != None)).astype(np.float64)
        
        tr['metformin_and_insulin'] = ((met_tr == 1.0) & (ins_tr == 1.0)).astype(np.float64)
        ev['metformin_and_insulin'] = ((met_ev == 1.0) & (ins_ev == 1.0)).astype(np.float64)
        extra_numeric.append('metformin_and_insulin')

    if 'insulin' in tr and 'insulin' in ev:
        ins_tr = ((tr['insulin'] != 'No') & (tr['insulin'] != None)).astype(np.float64)
        ins_ev = ((ev['insulin'] != 'No') & (ev['insulin'] != None)).astype(np.float64)
        
        tr['insulin_only'] = ((ins_tr == 1.0) & (tr_active == 1.0)).astype(np.float64)
        ev['insulin_only'] = ((ins_ev == 1.0) & (ev_active == 1.0)).astype(np.float64)
        extra_numeric.append('insulin_only')
        
        tr['insulin_and_others'] = ((ins_tr == 1.0) & (tr_active > 1.0)).astype(np.float64)
        ev['insulin_and_others'] = ((ins_ev == 1.0) & (ev_active > 1.0)).astype(np.float64)
        extra_numeric.append('insulin_and_others')

    tr['num_meds_changed'] = tr_changed
    ev['num_meds_changed'] = ev_changed
    extra_numeric.append('num_meds_changed')

    tr['num_meds_up'] = tr_up
    ev['num_meds_up'] = ev_up
    extra_numeric.append('num_meds_up')

    tr['num_meds_down'] = tr_down
    ev['num_meds_down'] = ev_down
    extra_numeric.append('num_meds_down')

    if 'insulin' in tr and 'insulin' in ev:
        tr['insulin_up'] = (tr['insulin'] == 'Up').astype(np.float64)
        ev['insulin_up'] = (ev['insulin'] == 'Up').astype(np.float64)
        extra_numeric.append('insulin_up')

        tr['insulin_steady'] = (tr['insulin'] == 'Steady').astype(np.float64)
        ev['insulin_steady'] = (ev['insulin'] == 'Steady').astype(np.float64)
        extra_numeric.append('insulin_steady')

        tr['insulin_active'] = ((tr['insulin'] != 'No') & (tr['insulin'] != None)).astype(np.float64)
        ev['insulin_active'] = ((ev['insulin'] != 'No') & (ev['insulin'] != None)).astype(np.float64)
        extra_numeric.append('insulin_active')

    if 'change' in tr and 'change' in ev:
        tr['change_binary'] = (tr['change'] == 'Ch').astype(np.float64)
        ev['change_binary'] = (ev['change'] == 'Ch').astype(np.float64)
        extra_numeric.append('change_binary')

    if 'diabetesMed' in tr and 'diabetesMed' in ev:
        tr['diabetesMed_binary'] = (tr['diabetesMed'] == 'Yes').astype(np.float64)
        ev['diabetesMed_binary'] = (ev['diabetesMed'] == 'Yes').astype(np.float64)
        extra_numeric.append('diabetesMed_binary')

    if 'A1Cresult' in tr and 'A1Cresult' in ev:
        tr['a1c_tested'] = (tr['A1Cresult'] != 'None').astype(np.float64)
        ev['a1c_tested'] = (ev['A1Cresult'] != 'None').astype(np.float64)
        extra_numeric.append('a1c_tested')

        tr['a1c_high'] = (tr['A1Cresult'] == '>8').astype(np.float64)
        ev['a1c_high'] = (ev['A1Cresult'] == '>8').astype(np.float64)
        extra_numeric.append('a1c_high')

    # Append Charlson flags to SVD interaction matrix
    if diag_cols:
        for k in cm_tr:
            if k != 'cm_count':
                svd_feats_tr.append(tr[k])
                svd_feats_ev.append(ev[k])

    if svd_feats_tr:
        X_tr_mat = np.column_stack(svd_feats_tr).astype(np.float64)
        X_ev_mat = np.column_stack(svd_feats_ev).astype(np.float64)
        mu = np.mean(X_tr_mat, axis=0, keepdims=True)
        X_tr_c = X_tr_mat - mu
        X_ev_c = X_ev_mat - mu

        # Rank-6 optimal SVD factor decomposition
        n_comps = min(6, X_tr_c.shape[1])
        _, _, vt = np.linalg.svd(X_tr_c, full_matrices=False)
        v_k = vt[:n_comps].T

        z_tr = X_tr_c @ v_k
        z_ev = X_ev_c @ v_k

        for comp_i in range(n_comps):
            fname = f'svd_phenotype_{comp_i}'
            tr[fname] = z_tr[:, comp_i]
            ev[fname] = z_ev[:, comp_i]
            extra_numeric.append(fname)

        # Regimen complexity / anomaly score via reconstruction residual norm
        tr['regimen_residual'] = np.linalg.norm(X_tr_c - z_tr @ v_k.T, axis=1)
        ev['regimen_residual'] = np.linalg.norm(X_ev_c - z_ev @ v_k.T, axis=1)
        extra_numeric.append('regimen_residual')

    # 5. Fit TabularEncoder and encode datasets
    encoder = TabularEncoder(min_count=40)
    encoder.fit(
        task,
        tr,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(tr)
    x_eval = encoder.transform(ev)

    # 6. Diverse multi-capacity LightGBM and XGBoost ensemble
    import xgboost as xgb

    m1 = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=150,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=0.05,
        reg_lambda=0.5,
        cat_smooth=50.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    m2 = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.018,
        num_leaves=20,
        min_child_samples=250,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=0.2,
        reg_lambda=2.0,
        cat_smooth=75.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )

    m3 = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.022,
        num_leaves=40,
        min_child_samples=120,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=0.1,
        reg_lambda=1.0,
        cat_smooth=60.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )

    m4 = xgb.XGBClassifier(
        n_estimators=180,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=6,
        subsample=0.80,
        colsample_bytree=0.65,
        reg_alpha=0.1,
        reg_lambda=1.5,
        tree_method='hist',
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )

    m1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    m2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    m3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    m4.fit(x_train, y_train)

    preds = (
        0.30 * m1.predict_proba(x_eval)[:, 1]
        + 0.25 * m2.predict_proba(x_eval)[:, 1]
        + 0.25 * m3.predict_proba(x_eval)[:, 1]
        + 0.20 * m4.predict_proba(x_eval)[:, 1]
    )

    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
