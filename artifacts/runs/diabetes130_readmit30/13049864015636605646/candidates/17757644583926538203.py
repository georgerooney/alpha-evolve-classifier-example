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

    # 2. ICD-9 Chapter Groupings & Truncated 3-Digit Codes
    def to_icd9_3digit(code):
        if code is None or code == '?':
            return 'missing'
        s = str(code).strip()
        if not s or s == 'None':
            return 'missing'
        if '.' in s:
            return s.split('.')[0]
        return s[:3] if len(s) >= 3 else s

    for diag in ['diag_1', 'diag_2', 'diag_3']:
        grp_name = f'{diag}_grp'
        train_data[grp_name] = np.array([icd9_group(c) for c in train[diag]], dtype=object)
        eval_data[grp_name] = np.array([icd9_group(c) for c in eval_cols[diag]], dtype=object)
        extra_categorical.append(grp_name)

        d3_name = f'{diag}_3d'
        train_data[d3_name] = np.array([to_icd9_3digit(c) for c in train[diag]], dtype=object)
        eval_data[d3_name] = np.array([to_icd9_3digit(c) for c in eval_cols[diag]], dtype=object)
        extra_categorical.append(d3_name)

    train_data['diag_1_2_grp'] = np.array(
        [f"{g1}_{g2}" for g1, g2 in zip(train_data['diag_1_grp'], train_data['diag_2_grp'])],
        dtype=object,
    )
    eval_data['diag_1_2_grp'] = np.array(
        [f"{g1}_{g2}" for g1, g2 in zip(eval_data['diag_1_grp'], eval_data['diag_2_grp'])],
        dtype=object,
    )
    extra_categorical.append('diag_1_2_grp')

    # Care pathway: Admission Source to Discharge Disposition trajectory
    train_adm_disp = np.array(
        [f"{s}_{d}" for s, d in zip(train['admission_source_id'], train['discharge_disposition_id'])],
        dtype=object,
    )
    eval_adm_disp = np.array(
        [f"{s}_{d}" for s, d in zip(eval_cols['admission_source_id'], eval_cols['discharge_disposition_id'])],
        dtype=object,
    )
    train_data['adm_disp_path'] = train_adm_disp
    eval_data['adm_disp_path'] = eval_adm_disp
    extra_categorical.append('adm_disp_path')

    # Specific High-Impact Primary Diagnoses
    d1_3d_tr = train_data['diag_1_3d']
    d1_3d_ev = eval_data['diag_1_3d']
    train_data['diag1_is_chf'] = np.isin(d1_3d_tr, ['428']).astype(np.float64)
    eval_data['diag1_is_chf'] = np.isin(d1_3d_ev, ['428']).astype(np.float64)
    train_data['diag1_is_diabetes'] = np.isin(d1_3d_tr, ['250']).astype(np.float64)
    eval_data['diag1_is_diabetes'] = np.isin(d1_3d_ev, ['250']).astype(np.float64)
    train_data['diag1_is_renal'] = np.isin(d1_3d_tr, ['584', '585', '586']).astype(np.float64)
    eval_data['diag1_is_renal'] = np.isin(d1_3d_ev, ['584', '585', '586']).astype(np.float64)
    train_data['diag1_is_copd_pneu'] = np.isin(d1_3d_tr, ['486', '491', '492', '493', '496']).astype(np.float64)
    eval_data['diag1_is_copd_pneu'] = np.isin(d1_3d_ev, ['486', '491', '492', '493', '496']).astype(np.float64)
    extra_numeric.extend(['diag1_is_chf', 'diag1_is_diabetes', 'diag1_is_renal', 'diag1_is_copd_pneu'])

    # 3. Out-of-Fold Target Encodings
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_3d', 'diag_2_3d', 'diag_3_3d',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'medical_specialty', 'payer_code',
        'diag_1_grp', 'diag_1_2_grp', 'adm_disp_path',
    ]
    for c in te_cols:
        if c in train_data and c in eval_data:
            te_tr, te_ev = oof_target_encode(
                train_data[c], y_train, eval_data[c], n_folds=5, smoothing=25.0, seed=42
            )
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

    train_data['diag_3d_te_max'] = np.maximum(
        train_data['diag_1_3d_te'],
        np.maximum(train_data['diag_2_3d_te'], train_data['diag_3_3d_te']),
    )
    eval_data['diag_3d_te_max'] = np.maximum(
        eval_data['diag_1_3d_te'],
        np.maximum(eval_data['diag_2_3d_te'], eval_data['diag_3_3d_te']),
    )
    extra_numeric.append('diag_3d_te_max')

    train_data['diag_3d_te_mean'] = (
        train_data['diag_1_3d_te'] + train_data['diag_2_3d_te'] + train_data['diag_3_3d_te']
    ) / 3.0
    eval_data['diag_3d_te_mean'] = (
        eval_data['diag_1_3d_te'] + eval_data['diag_2_3d_te'] + eval_data['diag_3_3d_te']
    ) / 3.0
    extra_numeric.append('diag_3d_te_mean')

    # 4. Visit History & Prior Utilization
    for df, d in [(train_data, train), (eval_data, eval_cols)]:
        outp = np.nan_to_num(d['number_outpatient'].astype(np.float64), nan=0.0)
        emrg = np.nan_to_num(d['number_emergency'].astype(np.float64), nan=0.0)
        inp = np.nan_to_num(d['number_inpatient'].astype(np.float64), nan=0.0)
        tot_visits = outp + emrg + inp

        acute_visits = inp + emrg
        df['total_prior_visits'] = tot_visits
        df['has_prior_inpatient'] = (inp > 0.0).astype(np.float64)
        df['has_multiple_inpatient'] = (inp > 1.0).astype(np.float64)
        df['frequent_inpatient'] = (inp >= 2.0).astype(np.float64)
        df['has_prior_emergency'] = (emrg > 0.0).astype(np.float64)
        df['has_any_prior_visit'] = (tot_visits > 0.0).astype(np.float64)

        df['log_inpatient'] = np.log1p(np.maximum(inp, 0.0))
        df['log_emergency'] = np.log1p(np.maximum(emrg, 0.0))
        df['log_outpatient'] = np.log1p(np.maximum(outp, 0.0))
        df['log_total_visits'] = np.log1p(np.maximum(tot_visits, 0.0))

        df['inpatient_ratio'] = inp / np.maximum(tot_visits, 1.0)
        df['emergency_ratio'] = emrg / np.maximum(tot_visits, 1.0)
        df['acute_ratio'] = acute_visits / np.maximum(tot_visits, 1.0)
        df['emrg_no_inp'] = ((emrg > 0.0) & (inp == 0.0)).astype(np.float64)
        df['high_outpatient_stable'] = ((outp >= 3.0) & (inp == 0.0) & (emrg == 0.0)).astype(np.float64)
        df['inp_x_emrg'] = df['log_inpatient'] * df['log_emergency']

    extra_numeric.extend([
        'total_prior_visits', 'has_prior_inpatient', 'has_multiple_inpatient',
        'frequent_inpatient', 'has_prior_emergency', 'has_any_prior_visit',
        'log_inpatient', 'log_emergency', 'log_outpatient', 'log_total_visits',
        'inpatient_ratio', 'emergency_ratio', 'acute_ratio', 'emrg_no_inp',
        'high_outpatient_stable', 'inp_x_emrg',
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
            ins_treated = ((ins != 'No') & (ins != None)).astype(np.float64)
            df['insulin_treated'] = ins_treated
            df['insulin_up'] = (ins == 'Up').astype(np.float64)
            df['insulin_down'] = (ins == 'Down').astype(np.float64)
            df['insulin_steady'] = (ins == 'Steady').astype(np.float64)
            df['insulin_dose_change'] = df['insulin_up'] + df['insulin_down']
            df['insulin_only'] = (ins_treated * (active_cnt == 1.0)).astype(np.float64)
            df['insulin_combo'] = (ins_treated * (active_cnt > 1.0)).astype(np.float64)
            df['oral_only'] = ((ins_treated == 0.0) & (active_cnt > 0.0)).astype(np.float64)
            df['no_diab_meds'] = (active_cnt == 0.0).astype(np.float64)
        else:
            df['insulin_treated'] = np.zeros(n_rows, dtype=np.float64)
            df['insulin_up'] = np.zeros(n_rows, dtype=np.float64)
            df['insulin_down'] = np.zeros(n_rows, dtype=np.float64)
            df['insulin_steady'] = np.zeros(n_rows, dtype=np.float64)
            df['insulin_dose_change'] = np.zeros(n_rows, dtype=np.float64)
            df['insulin_only'] = np.zeros(n_rows, dtype=np.float64)
            df['insulin_combo'] = np.zeros(n_rows, dtype=np.float64)
            df['oral_only'] = np.zeros(n_rows, dtype=np.float64)
            df['no_diab_meds'] = np.zeros(n_rows, dtype=np.float64)

        # Key antidiabetic drug classes
        metf = d.get('metformin')
        df['metformin_active'] = ((metf != 'No') & (metf != None)).astype(np.float64) if metf is not None else np.zeros(n_rows, dtype=np.float64)

        glim = d.get('glimepiride')
        glip = d.get('glipizide')
        glyb = d.get('glyburide')
        df['sulfonylurea_active'] = (
            (((glim != 'No') & (glim != None)) if glim is not None else 0) |
            (((glip != 'No') & (glip != None)) if glip is not None else 0) |
            (((glyb != 'No') & (glyb != None)) if glyb is not None else 0)
        ).astype(np.float64)

        pio = d.get('pioglitazone')
        rosi = d.get('rosiglitazone')
        df['tzd_active'] = (
            (((pio != 'No') & (pio != None)) if pio is not None else 0) |
            (((rosi != 'No') & (rosi != None)) if rosi is not None else 0)
        ).astype(np.float64)

        n_meds_arr = np.nan_to_num(d['num_medications'].astype(np.float64), nan=0.0)
        df['polypharmacy_high'] = (n_meds_arr >= 16.0).astype(np.float64)
        df['polypharmacy_extreme'] = (n_meds_arr >= 22.0).astype(np.float64)

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
        'insulin_steady', 'insulin_dose_change', 'insulin_only', 'insulin_combo',
        'oral_only', 'no_diab_meds', 'metformin_active', 'sulfonylurea_active',
        'tzd_active', 'polypharmacy_high', 'polypharmacy_extreme',
        'change_flag', 'diabetes_med_flag',
    ])

    # 8. Clinical Disposition, Admission Type & Lab Results
    for df, d in [(train_data, train), (eval_data, eval_cols)]:
        disp = d.get('discharge_disposition_id')
        if disp is not None:
            df['disp_home'] = (disp == '1').astype(np.float64)
            df['disp_snf'] = (disp == '3').astype(np.float64)
            df['disp_home_health'] = (disp == '6').astype(np.float64)
            df['disp_facility'] = np.isin(disp, ['3', '4', '5', '6', '22', '25']).astype(np.float64)
            df['disp_ama'] = (disp == '7').astype(np.float64)
            df['disp_transfer'] = np.isin(disp, ['2', '4', '5', '22', '23', '24']).astype(np.float64)
            df['disp_not_home'] = (disp != '1').astype(np.float64)
        else:
            df['disp_home'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['disp_snf'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['disp_home_health'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['disp_facility'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['disp_ama'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['disp_transfer'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['disp_not_home'] = np.zeros(len(df['age_num']), dtype=np.float64)

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
            df['adm_src_snf'] = (adm_src == '5').astype(np.float64)
            df['adm_src_transfer'] = np.isin(adm_src, ['4', '5', '6']).astype(np.float64)
        else:
            df['adm_src_er'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['adm_src_snf'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['adm_src_transfer'] = np.zeros(len(df['age_num']), dtype=np.float64)

        df['adm_er_to_snf'] = df['adm_src_er'] * df['disp_snf']

        a1c = d.get('A1Cresult')
        if a1c is not None:
            a1c_tested = (a1c != 'None').astype(np.float64)
            a1c_high = np.isin(a1c, ['>7', '>8']).astype(np.float64)
            df['a1c_tested'] = a1c_tested
            df['a1c_high'] = a1c_high
            df['a1c_high_no_change'] = (a1c_high * (df['change_flag'] == 0.0)).astype(np.float64)
            df['a1c_tested_and_change'] = (a1c_tested * df['change_flag']).astype(np.float64)
        else:
            df['a1c_tested'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['a1c_high'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['a1c_high_no_change'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['a1c_tested_and_change'] = np.zeros(len(df['age_num']), dtype=np.float64)

        glu = d.get('max_glu_serum')
        if glu is not None:
            df['glu_tested'] = (glu != 'None').astype(np.float64)
            df['glu_high'] = np.isin(glu, ['>200', '>300']).astype(np.float64)
        else:
            df['glu_tested'] = np.zeros(len(df['age_num']), dtype=np.float64)
            df['glu_high'] = np.zeros(len(df['age_num']), dtype=np.float64)

        # Informative Missingness Signals
        payer = d.get('payer_code')
        df['payer_missing'] = np.isin(payer, [None, '?']).astype(np.float64) if payer is not None else np.ones(len(df['age_num']), dtype=np.float64)
        spec = d.get('medical_specialty')
        df['specialty_missing'] = np.isin(spec, [None, '?']).astype(np.float64) if spec is not None else np.ones(len(df['age_num']), dtype=np.float64)
        wt = d.get('weight')
        df['weight_recorded'] = (~np.isin(wt, [None, '?'])).astype(np.float64) if wt is not None else np.zeros(len(df['age_num']), dtype=np.float64)

        # Synergistic Interactions
        df['risk_disp_x_inp'] = df['disp_facility'] * df['log_inpatient']
        df['risk_charlson_x_inp'] = df['charlson_idx'] * df['log_inpatient']
        df['prior_inp_x_disp_fac'] = df['has_prior_inpatient'] * df['disp_facility']
        df['prior_inp_x_meds'] = df['log_inpatient'] * np.log1p(np.maximum(df['num_active_meds'], 0.0))

    extra_numeric.extend([
        'disp_home', 'disp_snf', 'disp_home_health', 'disp_facility',
        'disp_ama', 'disp_transfer', 'disp_not_home',
        'adm_emergency', 'adm_urgent', 'adm_elective',
        'adm_src_er', 'adm_src_snf', 'adm_src_transfer', 'adm_er_to_snf',
        'a1c_tested', 'a1c_high', 'a1c_high_no_change', 'a1c_tested_and_change',
        'glu_tested', 'glu_high',
        'payer_missing', 'specialty_missing', 'weight_recorded',
        'risk_disp_x_inp', 'risk_charlson_x_inp', 'prior_inp_x_disp_fac', 'prior_inp_x_meds',
    ])

    extra_numeric = list(dict.fromkeys(extra_numeric))
    extra_categorical = list(dict.fromkeys(extra_categorical))

    # Encode with TabularEncoder
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_data, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)
    cat_indices = encoder.categorical_indices

    # 9. Ensemble of Strategically Diverse LightGBM Architectures
    # Model 1: Balanced GBDT, moderate capacity
    m1 = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.022,
        num_leaves=31,
        max_depth=6,
        min_child_samples=150,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_alpha=0.2,
        reg_lambda=2.0,
        cat_smooth=45.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=cat_indices)
    p1 = m1.predict_proba(x_eval)[:, 1]

    # Model 2: Conservative, shallow depth regularized (low variance baseline)
    m2 = lgb.LGBMClassifier(
        n_estimators=360,
        learning_rate=0.024,
        num_leaves=16,
        max_depth=4,
        min_child_samples=300,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_alpha=0.8,
        reg_lambda=5.0,
        cat_smooth=80.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=cat_indices)
    p2 = m2.predict_proba(x_eval)[:, 1]

    # Model 3: Deeper multi-way interaction explorer
    m3 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.018,
        num_leaves=48,
        max_depth=7,
        min_child_samples=90,
        subsample=0.68,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=0.05,
        reg_lambda=1.0,
        cat_smooth=30.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=cat_indices)
    p3 = m3.predict_proba(x_eval)[:, 1]

    # Model 4: Feature-subspace constrained GBDT (forces splits on orthogonal feature sets)
    m4 = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.020,
        num_leaves=24,
        max_depth=5,
        min_child_samples=180,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.45,
        reg_alpha=0.4,
        reg_lambda=3.0,
        cat_smooth=60.0,
        random_state=999,
        n_jobs=1,
        verbose=-1,
    )
    m4.fit(x_train, y_train, categorical_feature=cat_indices)
    p4 = m4.predict_proba(x_eval)[:, 1]

    # Soft voting ensemble
    preds = 0.30 * p1 + 0.30 * p2 + 0.20 * p3 + 0.20 * p4
    preds = np.clip(preds, 0.0, 1.0)
    if np.isnan(preds).any():
        preds = np.nan_to_num(preds, nan=0.114)
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
