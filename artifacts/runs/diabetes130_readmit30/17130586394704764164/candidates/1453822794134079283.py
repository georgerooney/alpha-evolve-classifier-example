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
    train_dict = dict(train)
    eval_dict = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []
    n_train = len(y_train)
    n_eval = len(eval_cols['time_in_hospital'])

    # 1. Comorbidities
    cm_train = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k, v in cm_train.items():
        train_dict[k] = v.astype(np.float64)
        eval_dict[k] = cm_eval[k].astype(np.float64)
        extra_numeric.append(k)

    # 2. ICD-9 Chapter Groups and 3-Digit Disease Prefixes
    def _clean_icd9(c):
        if c is None or c == '?' or c == '':
            return 'missing'
        return str(c).split('.')[0].strip()

    for diag in ('diag_1', 'diag_2', 'diag_3'):
        grp_col = f'{diag}_grp'
        train_dict[grp_col] = np.array([icd9_group(c) for c in train[diag]], dtype=object)
        eval_dict[grp_col] = np.array([icd9_group(c) for c in eval_cols[diag]], dtype=object)
        extra_categorical.append(grp_col)

        pfx_col = f'{diag}_prefix'
        train_dict[pfx_col] = np.array([_clean_icd9(c) for c in train[diag]], dtype=object)
        eval_dict[pfx_col] = np.array([_clean_icd9(c) for c in eval_cols[diag]], dtype=object)
        extra_categorical.append(pfx_col)

    for grp_val in ('circulatory', 'respiratory', 'diabetes', 'digestive'):
        col_name = f'has_grp_{grp_val}'
        train_dict[col_name] = ((train_dict['diag_1_grp'] == grp_val) |
                                (train_dict['diag_2_grp'] == grp_val) |
                                (train_dict['diag_3_grp'] == grp_val)).astype(np.float64)
        eval_dict[col_name] = ((eval_dict['diag_1_grp'] == grp_val) |
                               (eval_dict['diag_2_grp'] == grp_val) |
                               (eval_dict['diag_3_grp'] == grp_val)).astype(np.float64)
        extra_numeric.append(col_name)

    train_dict['diag1_is_circulatory'] = (train_dict['diag_1_grp'] == 'circulatory').astype(np.float64)
    eval_dict['diag1_is_circulatory'] = (eval_dict['diag_1_grp'] == 'circulatory').astype(np.float64)
    extra_numeric.append('diag1_is_circulatory')

    train_dict['diag1_is_diabetes'] = (train_dict['diag_1_grp'] == 'diabetes').astype(np.float64)
    eval_dict['diag1_is_diabetes'] = (eval_dict['diag_1_grp'] == 'diabetes').astype(np.float64)
    extra_numeric.append('diag1_is_diabetes')

    # Canonical Multi-Morbidity Triad (order-invariant clinical syndrome)
    tr_triads = ["_".join(sorted([str(g1), str(g2), str(g3)]))
                 for g1, g2, g3 in zip(train_dict['diag_1_grp'], train_dict['diag_2_grp'], train_dict['diag_3_grp'])]
    ev_triads = ["_".join(sorted([str(g1), str(g2), str(g3)]))
                 for g1, g2, g3 in zip(eval_dict['diag_1_grp'], eval_dict['diag_2_grp'], eval_dict['diag_3_grp'])]
    train_dict['comorbidity_triad'] = np.array(tr_triads, dtype=object)
    eval_dict['comorbidity_triad'] = np.array(ev_triads, dtype=object)
    extra_categorical.append('comorbidity_triad')

    tr_triad_te, ev_triad_te = oof_target_encode(
        train_dict['comorbidity_triad'], y_train, eval_dict['comorbidity_triad'], n_folds=5, smoothing=25.0, seed=42
    )
    train_dict['te_comorbidity_triad'] = tr_triad_te.astype(np.float64)
    eval_dict['te_comorbidity_triad'] = ev_triad_te.astype(np.float64)
    extra_numeric.append('te_comorbidity_triad')

    # 3. Out-of-Fold Target Encoding
    te_candidates = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_prefix', 'diag_2_prefix', 'diag_3_prefix',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'medical_specialty', 'payer_code'
    ]
    for col in te_candidates:
        if col in train and col in eval_cols:
            tr_te, ev_te = oof_target_encode(
                train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42
            )
            col_name = f'te_{col}'
            train_dict[col_name] = tr_te.astype(np.float64)
            eval_dict[col_name] = ev_te.astype(np.float64)
            extra_numeric.append(col_name)

    if 'te_diag_1' in train_dict and 'te_diag_2' in train_dict and 'te_diag_3' in train_dict:
        train_dict['te_diag_max'] = np.maximum(
            np.maximum(train_dict['te_diag_1'], train_dict['te_diag_2']), train_dict['te_diag_3']
        )
        eval_dict['te_diag_max'] = np.maximum(
            np.maximum(eval_dict['te_diag_1'], eval_dict['te_diag_2']), eval_dict['te_diag_3']
        )
        extra_numeric.append('te_diag_max')

        train_dict['te_diag_mean'] = (
            train_dict['te_diag_1'] + train_dict['te_diag_2'] + train_dict['te_diag_3']
        ) / 3.0
        eval_dict['te_diag_mean'] = (
            eval_dict['te_diag_1'] + eval_dict['te_diag_2'] + eval_dict['te_diag_3']
        ) / 3.0
        extra_numeric.append('te_diag_mean')

    if 'te_diag_1_prefix' in train_dict and 'te_diag_2_prefix' in train_dict and 'te_diag_3_prefix' in train_dict:
        train_dict['te_prefix_max'] = np.maximum(
            np.maximum(train_dict['te_diag_1_prefix'], train_dict['te_diag_2_prefix']), train_dict['te_diag_3_prefix']
        )
        eval_dict['te_prefix_max'] = np.maximum(
            np.maximum(eval_dict['te_diag_1_prefix'], eval_dict['te_diag_2_prefix']), eval_dict['te_diag_3_prefix']
        )
        extra_numeric.append('te_prefix_max')

        train_dict['te_prefix_mean'] = (
            train_dict['te_diag_1_prefix'] + train_dict['te_diag_2_prefix'] + train_dict['te_diag_3_prefix']
        ) / 3.0
        eval_dict['te_prefix_mean'] = (
            eval_dict['te_diag_1_prefix'] + eval_dict['te_diag_2_prefix'] + eval_dict['te_diag_3_prefix']
        ) / 3.0
        extra_numeric.append('te_prefix_mean')

    if 'discharge_disposition_id' in train and 'admission_source_id' in train:
        disp_src_train = np.array([
            f"{d}_{s}" for d, s in zip(train['discharge_disposition_id'], train['admission_source_id'])
        ], dtype=object)
        disp_src_eval = np.array([
            f"{d}_{s}" for d, s in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])
        ], dtype=object)
        tr_disp_src_te, ev_disp_src_te = oof_target_encode(
            disp_src_train, y_train, disp_src_eval, n_folds=5, smoothing=25.0, seed=42
        )
        train_dict['te_disp_x_source'] = tr_disp_src_te.astype(np.float64)
        eval_dict['te_disp_x_source'] = ev_disp_src_te.astype(np.float64)
        extra_numeric.append('te_disp_x_source')

    if 'discharge_disposition_id' in train and 'number_inpatient' in train:
        tr_inp_bin = np.clip(np.nan_to_num(train['number_inpatient'], nan=0.0), 0, 3).astype(int).astype(str)
        ev_inp_bin = np.clip(np.nan_to_num(eval_cols['number_inpatient'], nan=0.0), 0, 3).astype(int).astype(str)
        disp_inp_tr = np.array([f"{d}_{b}" for d, b in zip(train['discharge_disposition_id'], tr_inp_bin)], dtype=object)
        disp_inp_ev = np.array([f"{d}_{b}" for d, b in zip(eval_cols['discharge_disposition_id'], ev_inp_bin)], dtype=object)
        tr_di_te, ev_di_te = oof_target_encode(disp_inp_tr, y_train, disp_inp_ev, n_folds=5, smoothing=20.0, seed=42)
        train_dict['te_disp_x_inp'] = tr_di_te.astype(np.float64)
        eval_dict['te_disp_x_inp'] = ev_di_te.astype(np.float64)
        extra_numeric.append('te_disp_x_inp')

        diag1_disp_tr = np.array([f"{g}_{d}" for g, d in zip(train_dict['diag_1_grp'], train['discharge_disposition_id'])], dtype=object)
        diag1_disp_ev = np.array([f"{g}_{d}" for g, d in zip(eval_dict['diag_1_grp'], eval_cols['discharge_disposition_id'])], dtype=object)
        tr_dd_te, ev_dd_te = oof_target_encode(diag1_disp_tr, y_train, diag1_disp_ev, n_folds=5, smoothing=20.0, seed=42)
        train_dict['te_diag1_x_disp'] = tr_dd_te.astype(np.float64)
        eval_dict['te_diag1_x_disp'] = ev_dd_te.astype(np.float64)
        extra_numeric.append('te_diag1_x_disp')

    # Multi-attribute Dyads: Primary-Secondary Diagnosis & Care Transition interactions
    diag_dyad_tr = np.array([f"{g1}_{g2}" for g1, g2 in zip(train_dict['diag_1_grp'], train_dict['diag_2_grp'])], dtype=object)
    diag_dyad_ev = np.array([f"{g1}_{g2}" for g1, g2 in zip(eval_dict['diag_1_grp'], eval_dict['diag_2_grp'])], dtype=object)
    tr_dyad_te, ev_dyad_te = oof_target_encode(diag_dyad_tr, y_train, diag_dyad_ev, n_folds=5, smoothing=25.0, seed=42)
    train_dict['te_diag_dyad'] = tr_dyad_te.astype(np.float64)
    eval_dict['te_diag_dyad'] = ev_dyad_te.astype(np.float64)
    extra_numeric.append('te_diag_dyad')

    pfx_dyad_tr = np.array([f"{p1}_{p2}" for p1, p2 in zip(train_dict['diag_1_prefix'], train_dict['diag_2_prefix'])], dtype=object)
    pfx_dyad_ev = np.array([f"{p1}_{p2}" for p1, p2 in zip(eval_dict['diag_1_prefix'], eval_dict['diag_2_prefix'])], dtype=object)
    tr_pfx_dyad_te, ev_pfx_dyad_te = oof_target_encode(pfx_dyad_tr, y_train, pfx_dyad_ev, n_folds=5, smoothing=30.0, seed=42)
    train_dict['te_pfx_dyad'] = tr_pfx_dyad_te.astype(np.float64)
    eval_dict['te_pfx_dyad'] = ev_pfx_dyad_te.astype(np.float64)
    extra_numeric.append('te_pfx_dyad')

    if 'admission_type_id' in train and 'discharge_disposition_id' in train:
        adm_disp_tr = np.array([f"{a}_{d}" for a, d in zip(train['admission_type_id'], train['discharge_disposition_id'])], dtype=object)
        adm_disp_ev = np.array([f"{a}_{d}" for a, d in zip(eval_cols['admission_type_id'], eval_cols['discharge_disposition_id'])], dtype=object)
        tr_adm_disp_te, ev_adm_disp_te = oof_target_encode(adm_disp_tr, y_train, adm_disp_ev, n_folds=5, smoothing=25.0, seed=42)
        train_dict['te_adm_x_disp'] = tr_adm_disp_te.astype(np.float64)
        eval_dict['te_adm_x_disp'] = ev_adm_disp_te.astype(np.float64)
        extra_numeric.append('te_adm_x_disp')

    # Cardiorenal & high risk comorbidity interactions
    train_dict['cm_chf_and_renal'] = ((cm_train['cm_chf'] > 0) & (cm_train['cm_renal'] > 0)).astype(np.float64)
    eval_dict['cm_chf_and_renal'] = ((cm_eval['cm_chf'] > 0) & (cm_eval['cm_renal'] > 0)).astype(np.float64)
    extra_numeric.append('cm_chf_and_renal')

    train_dict['cm_diab_and_renal'] = ((cm_train['cm_diabetes_complicated'] > 0) & (cm_train['cm_renal'] > 0)).astype(np.float64)
    eval_dict['cm_diab_and_renal'] = ((cm_eval['cm_diabetes_complicated'] > 0) & (cm_eval['cm_renal'] > 0)).astype(np.float64)
    extra_numeric.append('cm_diab_and_renal')

    train_dict['cm_high_risk_score'] = (
        cm_train['cm_chf'] * 2.0 + cm_train['cm_renal'] * 2.0 +
        cm_train['cm_pulmonary'] + cm_train['cm_diabetes_complicated']
    ).astype(np.float64)
    eval_dict['cm_high_risk_score'] = (
        cm_eval['cm_chf'] * 2.0 + cm_eval['cm_renal'] * 2.0 +
        cm_eval['cm_pulmonary'] + cm_eval['cm_diabetes_complicated']
    ).astype(np.float64)
    extra_numeric.append('cm_high_risk_score')

    # 4. Utilization & Encounter Intensity
    tr_inp = np.nan_to_num(train['number_inpatient'], nan=0.0)
    ev_inp = np.nan_to_num(eval_cols['number_inpatient'], nan=0.0)
    tr_emg = np.nan_to_num(train['number_emergency'], nan=0.0)
    ev_emg = np.nan_to_num(eval_cols['number_emergency'], nan=0.0)
    tr_out = np.nan_to_num(train['number_outpatient'], nan=0.0)
    ev_out = np.nan_to_num(eval_cols['number_outpatient'], nan=0.0)
    tr_los = np.maximum(np.nan_to_num(train['time_in_hospital'], nan=1.0), 1.0)
    ev_los = np.maximum(np.nan_to_num(eval_cols['time_in_hospital'], nan=1.0), 1.0)
    tr_med = np.nan_to_num(train['num_medications'], nan=0.0)
    ev_med = np.nan_to_num(eval_cols['num_medications'], nan=0.0)
    tr_lab = np.nan_to_num(train['num_lab_procedures'], nan=0.0)
    ev_lab = np.nan_to_num(eval_cols['num_lab_procedures'], nan=0.0)
    tr_prc = np.nan_to_num(train['num_procedures'], nan=0.0)
    ev_prc = np.nan_to_num(eval_cols['num_procedures'], nan=0.0)
    tr_dx = np.nan_to_num(train['number_diagnoses'], nan=0.0)
    ev_dx = np.nan_to_num(eval_cols['number_diagnoses'], nan=0.0)

    tr_tot_prior = tr_inp + tr_emg + tr_out
    ev_tot_prior = ev_inp + ev_emg + ev_out
    train_dict['total_prior_visits'] = tr_tot_prior
    eval_dict['total_prior_visits'] = ev_tot_prior
    extra_numeric.append('total_prior_visits')

    train_dict['inpatient_ratio'] = tr_inp / (tr_tot_prior + 1.0)
    eval_dict['inpatient_ratio'] = ev_inp / (ev_tot_prior + 1.0)
    extra_numeric.append('inpatient_ratio')

    train_dict['emergency_ratio'] = tr_emg / (tr_tot_prior + 1.0)
    eval_dict['emergency_ratio'] = ev_emg / (ev_tot_prior + 1.0)
    extra_numeric.append('emergency_ratio')

    train_dict['outpatient_ratio'] = tr_out / (tr_tot_prior + 1.0)
    eval_dict['outpatient_ratio'] = ev_out / (ev_tot_prior + 1.0)
    extra_numeric.append('outpatient_ratio')

    train_dict['acute_utilization'] = tr_inp * 2.0 + tr_emg
    eval_dict['acute_utilization'] = ev_inp * 2.0 + ev_emg
    extra_numeric.append('acute_utilization')

    train_dict['log1p_inpatient'] = np.log1p(tr_inp)
    eval_dict['log1p_inpatient'] = np.log1p(ev_inp)
    extra_numeric.append('log1p_inpatient')

    train_dict['log1p_emergency'] = np.log1p(tr_emg)
    eval_dict['log1p_emergency'] = np.log1p(ev_emg)
    extra_numeric.append('log1p_emergency')

    train_dict['inp_x_los'] = tr_inp * tr_los
    eval_dict['inp_x_los'] = ev_inp * ev_los
    extra_numeric.append('inp_x_los')

    train_dict['inp_x_cm_count'] = tr_inp * cm_train['cm_count']
    eval_dict['inp_x_cm_count'] = ev_inp * cm_eval['cm_count']
    extra_numeric.append('inp_x_cm_count')

    train_dict['inp_x_chf'] = tr_inp * cm_train['cm_chf']
    eval_dict['inp_x_chf'] = ev_inp * cm_eval['cm_chf']
    extra_numeric.append('inp_x_chf')

    train_dict['has_inpatient'] = (tr_inp > 0).astype(np.float64)
    eval_dict['has_inpatient'] = (ev_inp > 0).astype(np.float64)
    extra_numeric.append('has_inpatient')

    train_dict['has_emergency'] = (tr_emg > 0).astype(np.float64)
    eval_dict['has_emergency'] = (ev_emg > 0).astype(np.float64)
    extra_numeric.append('has_emergency')

    train_dict['has_prior_visit'] = (tr_tot_prior > 0).astype(np.float64)
    eval_dict['has_prior_visit'] = (ev_tot_prior > 0).astype(np.float64)
    extra_numeric.append('has_prior_visit')

    train_dict['frequent_inpatient'] = (tr_inp >= 2).astype(np.float64)
    eval_dict['frequent_inpatient'] = (ev_inp >= 2).astype(np.float64)
    extra_numeric.append('frequent_inpatient')

    train_dict['meds_per_day'] = tr_med / tr_los
    eval_dict['meds_per_day'] = ev_med / ev_los
    extra_numeric.append('meds_per_day')

    train_dict['labs_per_day'] = tr_lab / tr_los
    eval_dict['labs_per_day'] = ev_lab / ev_los
    extra_numeric.append('labs_per_day')

    train_dict['procs_per_day'] = tr_prc / tr_los
    eval_dict['procs_per_day'] = ev_prc / ev_los
    extra_numeric.append('procs_per_day')

    train_dict['dx_per_day'] = tr_dx / tr_los
    eval_dict['dx_per_day'] = ev_dx / ev_los
    extra_numeric.append('dx_per_day')

    train_dict['total_services'] = tr_med + tr_lab + tr_prc
    eval_dict['total_services'] = ev_med + ev_lab + ev_prc
    extra_numeric.append('total_services')

    train_dict['services_per_day'] = train_dict['total_services'] / tr_los
    eval_dict['services_per_day'] = eval_dict['total_services'] / ev_los
    extra_numeric.append('services_per_day')

    # Codified LACE Index (Length of stay, Acuity, Comorbidity, Emergency)
    def _lace_los(los_arr):
        pts = np.zeros_like(los_arr, dtype=np.float64)
        pts[los_arr == 1] = 1.0
        pts[los_arr == 2] = 2.0
        pts[los_arr == 3] = 3.0
        pts[(los_arr >= 4) & (los_arr <= 6)] = 4.0
        pts[(los_arr >= 7) & (los_arr <= 13)] = 5.0
        pts[los_arr >= 14] = 7.0
        return pts

    tr_lace_l = _lace_los(tr_los)
    ev_lace_l = _lace_los(ev_los)
    tr_lace_a = (((train['admission_type_id'] == '1') | (train['admission_source_id'] == '7')).astype(np.float64)) * 3.0
    ev_lace_a = (((eval_cols['admission_type_id'] == '1') | (eval_cols['admission_source_id'] == '7')).astype(np.float64)) * 3.0

    def _compute_cci(cm):
        raw = (cm['cm_mi'] + cm['cm_chf'] + cm['cm_pvd'] + cm['cm_cerebrovascular'] + cm['cm_dementia'] +
               cm['cm_pulmonary'] + cm['cm_rheumatic'] + cm['cm_peptic_ulcer'] + cm['cm_liver'] +
               2.0 * cm['cm_hemiplegia'] + 2.0 * cm['cm_renal'] + 2.0 * cm['cm_diabetes_complicated'] +
               2.0 * cm['cm_cancer'] + 6.0 * cm['cm_metastatic'] + 6.0 * cm['cm_hiv'])
        pts = np.clip(raw, 0.0, 3.0)
        pts[raw >= 4.0] = 5.0
        return raw, pts

    tr_cci_raw, tr_lace_c = _compute_cci(cm_train)
    ev_cci_raw, ev_lace_c = _compute_cci(cm_eval)
    tr_lace_e = np.clip(tr_emg, 0.0, 4.0)
    ev_lace_e = np.clip(ev_emg, 0.0, 4.0)

    tr_lace = tr_lace_l + tr_lace_a + tr_lace_c + tr_lace_e
    ev_lace = ev_lace_l + ev_lace_a + ev_lace_c + ev_lace_e
    train_dict['lace_score'] = tr_lace
    eval_dict['lace_score'] = ev_lace
    extra_numeric.append('lace_score')
    train_dict['lace_high'] = (tr_lace >= 10.0).astype(np.float64)
    eval_dict['lace_high'] = (ev_lace >= 10.0).astype(np.float64)
    extra_numeric.append('lace_high')
    train_dict['charlson_raw'] = tr_cci_raw
    eval_dict['charlson_raw'] = ev_cci_raw
    extra_numeric.append('charlson_raw')

    # Clinical Instability & Premature Discharge Risk
    tr_burden = cm_train['cm_count'] * 2.0 + (tr_inp > 0) * 3.0 + tr_dx + tr_med / 5.0
    ev_burden = cm_eval['cm_count'] * 2.0 + (ev_inp > 0) * 3.0 + ev_dx + ev_med / 5.0
    train_dict['clinical_instability_ratio'] = tr_burden / tr_los
    eval_dict['clinical_instability_ratio'] = ev_burden / ev_los
    extra_numeric.append('clinical_instability_ratio')

    train_dict['premature_discharge'] = ((cm_train['cm_count'] >= 2) & (tr_los <= 2.0)).astype(np.float64)
    eval_dict['premature_discharge'] = ((cm_eval['cm_count'] >= 2) & (ev_los <= 2.0)).astype(np.float64)
    extra_numeric.append('premature_discharge')

    # 5. Age Numeric & Interactions
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    tr_age_num = np.array([age_map.get(x, 65.0) for x in train['age']], dtype=np.float64)
    ev_age_num = np.array([age_map.get(x, 65.0) for x in eval_cols['age']], dtype=np.float64)
    train_dict['age_num'] = tr_age_num
    eval_dict['age_num'] = ev_age_num
    extra_numeric.append('age_num')

    train_dict['is_elderly'] = (tr_age_num >= 70).astype(np.float64)
    eval_dict['is_elderly'] = (ev_age_num >= 70).astype(np.float64)
    extra_numeric.append('is_elderly')

    train_dict['age_x_inpatient'] = tr_age_num * tr_inp
    eval_dict['age_x_inpatient'] = ev_age_num * ev_inp
    extra_numeric.append('age_x_inpatient')

    train_dict['very_elderly'] = (tr_age_num >= 80).astype(np.float64)
    eval_dict['very_elderly'] = (ev_age_num >= 80).astype(np.float64)
    extra_numeric.append('very_elderly')

    # 6. Medication Dynamics
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    active_med_cols = [c for c in med_names if c in task.categorical_columns]

    tr_active = np.zeros(n_train, dtype=np.float64)
    ev_active = np.zeros(n_eval, dtype=np.float64)
    tr_up = np.zeros(n_train, dtype=np.float64)
    ev_up = np.zeros(n_eval, dtype=np.float64)
    tr_down = np.zeros(n_train, dtype=np.float64)
    ev_down = np.zeros(n_eval, dtype=np.float64)

    for m in active_med_cols:
        tr_m = train[m]
        ev_m = eval_cols[m]
        tr_active += (tr_m != 'No') & (tr_m is not None)
        ev_active += (ev_m != 'No') & (ev_m is not None)
        tr_up += (tr_m == 'Up')
        ev_up += (ev_m == 'Up')
        tr_down += (tr_m == 'Down')
        ev_down += (ev_m == 'Down')

    train_dict['num_active_diabetes_meds'] = tr_active
    eval_dict['num_active_diabetes_meds'] = ev_active
    extra_numeric.append('num_active_diabetes_meds')

    train_dict['num_meds_up'] = tr_up
    eval_dict['num_meds_up'] = ev_up
    extra_numeric.append('num_meds_up')

    train_dict['num_meds_down'] = tr_down
    eval_dict['num_meds_down'] = ev_down
    extra_numeric.append('num_meds_down')

    train_dict['num_meds_changed'] = tr_up + tr_down
    eval_dict['num_meds_changed'] = ev_up + ev_down
    extra_numeric.append('num_meds_changed')

    if 'insulin' in train:
        tr_ins = train['insulin']
        ev_ins = eval_cols['insulin']
        train_dict['insulin_active'] = ((tr_ins != 'No') & (tr_ins is not None)).astype(np.float64)
        eval_dict['insulin_active'] = ((ev_ins != 'No') & (ev_ins is not None)).astype(np.float64)
        extra_numeric.append('insulin_active')

        train_dict['insulin_up'] = (tr_ins == 'Up').astype(np.float64)
        eval_dict['insulin_up'] = (ev_ins == 'Up').astype(np.float64)
        extra_numeric.append('insulin_up')

        train_dict['insulin_down'] = (tr_ins == 'Down').astype(np.float64)
        eval_dict['insulin_down'] = (ev_ins == 'Down').astype(np.float64)
        extra_numeric.append('insulin_down')

        train_dict['insulin_changed'] = ((tr_ins == 'Up') | (tr_ins == 'Down')).astype(np.float64)
        eval_dict['insulin_changed'] = ((ev_ins == 'Up') | (ev_ins == 'Down')).astype(np.float64)
        extra_numeric.append('insulin_changed')

        train_dict['insulin_steady'] = (tr_ins == 'Steady').astype(np.float64)
        eval_dict['insulin_steady'] = (ev_ins == 'Steady').astype(np.float64)
        extra_numeric.append('insulin_steady')

    if 'metformin' in train:
        tr_met = train['metformin']
        ev_met = eval_cols['metformin']
        train_dict['metformin_active'] = ((tr_met != 'No') & (tr_met is not None)).astype(np.float64)
        eval_dict['metformin_active'] = ((ev_met != 'No') & (ev_met is not None)).astype(np.float64)
        extra_numeric.append('metformin_active')

    if 'insulin_active' in train_dict and 'metformin_active' in train_dict:
        train_dict['insulin_no_metformin'] = (
            (train_dict['insulin_active'] == 1.0) & (train_dict['metformin_active'] == 0.0)
        ).astype(np.float64)
        eval_dict['insulin_no_metformin'] = (
            (eval_dict['insulin_active'] == 1.0) & (eval_dict['metformin_active'] == 0.0)
        ).astype(np.float64)
        extra_numeric.append('insulin_no_metformin')

    if 'change' in train:
        train_dict['change_numeric'] = (train['change'] == 'Ch').astype(np.float64)
        eval_dict['change_numeric'] = (eval_cols['change'] == 'Ch').astype(np.float64)
        extra_numeric.append('change_numeric')

    if 'diabetesMed' in train:
        train_dict['diabetesMed_numeric'] = (train['diabetesMed'] == 'Yes').astype(np.float64)
        eval_dict['diabetesMed_numeric'] = (eval_cols['diabetesMed'] == 'Yes').astype(np.float64)
        extra_numeric.append('diabetesMed_numeric')

    train_dict['polypharmacy'] = (tr_med >= 16).astype(np.float64)
    eval_dict['polypharmacy'] = (ev_med >= 16).astype(np.float64)
    extra_numeric.append('polypharmacy')

    train_dict['med_volatility_per_day'] = (train_dict['num_meds_changed'] / tr_los).astype(np.float64)
    eval_dict['med_volatility_per_day'] = (eval_dict['num_meds_changed'] / ev_los).astype(np.float64)
    extra_numeric.append('med_volatility_per_day')

    # 7. Lab Indicators
    if 'A1Cresult' in train:
        tr_a1c = train['A1Cresult']
        ev_a1c = eval_cols['A1Cresult']
        train_dict['a1c_tested'] = ((tr_a1c != 'None') & (tr_a1c is not None)).astype(np.float64)
        eval_dict['a1c_tested'] = ((ev_a1c != 'None') & (ev_a1c is not None)).astype(np.float64)
        extra_numeric.append('a1c_tested')

        train_dict['a1c_high'] = ((tr_a1c == '>8') | (tr_a1c == '>7')).astype(np.float64)
        eval_dict['a1c_high'] = ((ev_a1c == '>8') | (ev_a1c == '>7')).astype(np.float64)
        extra_numeric.append('a1c_high')

    if 'max_glu_serum' in train:
        tr_glu = train['max_glu_serum']
        ev_glu = eval_cols['max_glu_serum']
        train_dict['glu_tested'] = ((tr_glu != 'None') & (tr_glu is not None)).astype(np.float64)
        eval_dict['glu_tested'] = ((ev_glu != 'None') & (ev_glu is not None)).astype(np.float64)
        extra_numeric.append('glu_tested')

        train_dict['glu_high'] = ((tr_glu == '>200') | (tr_glu == '>300')).astype(np.float64)
        eval_dict['glu_high'] = ((ev_glu == '>200') | (ev_glu == '>300')).astype(np.float64)
        extra_numeric.append('glu_high')

    if 'a1c_high' in train_dict and 'glu_high' in train_dict:
        train_dict['uncontrolled_diabetes'] = (
            (train_dict['a1c_high'] == 1.0) | (train_dict['glu_high'] == 1.0)
        ).astype(np.float64)
        eval_dict['uncontrolled_diabetes'] = (
            (eval_dict['a1c_high'] == 1.0) | (eval_dict['glu_high'] == 1.0)
        ).astype(np.float64)
        extra_numeric.append('uncontrolled_diabetes')

    # Clinical Inertia & Glycemic Change Patterns (Strack et al. 2014)
    if 'a1c_high' in train_dict and 'change_numeric' in train_dict:
        train_dict['a1c_high_no_change'] = ((train_dict['a1c_high'] == 1.0) & (train_dict['change_numeric'] == 0.0)).astype(np.float64)
        eval_dict['a1c_high_no_change'] = ((eval_dict['a1c_high'] == 1.0) & (eval_dict['change_numeric'] == 0.0)).astype(np.float64)
        extra_numeric.append('a1c_high_no_change')

        train_dict['a1c_high_med_changed'] = ((train_dict['a1c_high'] == 1.0) & (train_dict['change_numeric'] == 1.0)).astype(np.float64)
        eval_dict['a1c_high_med_changed'] = ((eval_dict['a1c_high'] == 1.0) & (eval_dict['change_numeric'] == 1.0)).astype(np.float64)
        extra_numeric.append('a1c_high_med_changed')

    # 8. Admission & Discharge Indicators
    if 'discharge_disposition_id' in train:
        tr_disp = train['discharge_disposition_id']
        ev_disp = eval_cols['discharge_disposition_id']
        train_dict['disp_home'] = (tr_disp == '1').astype(np.float64)
        eval_dict['disp_home'] = (ev_disp == '1').astype(np.float64)
        extra_numeric.append('disp_home')

        train_dict['disp_not_home'] = (tr_disp != '1').astype(np.float64)
        eval_dict['disp_not_home'] = (ev_disp != '1').astype(np.float64)
        extra_numeric.append('disp_not_home')

        train_dict['inp_and_not_home'] = ((tr_inp > 0) & (train_dict['disp_not_home'] == 1.0)).astype(np.float64)
        eval_dict['inp_and_not_home'] = ((ev_inp > 0) & (eval_dict['disp_not_home'] == 1.0)).astype(np.float64)
        extra_numeric.append('inp_and_not_home')

        train_dict['disp_snf'] = (tr_disp == '3').astype(np.float64)
        eval_dict['disp_snf'] = (ev_disp == '3').astype(np.float64)
        extra_numeric.append('disp_snf')

        train_dict['disp_home_health'] = (tr_disp == '6').astype(np.float64)
        eval_dict['disp_home_health'] = (ev_disp == '6').astype(np.float64)
        extra_numeric.append('disp_home_health')

        train_dict['disp_ama'] = ((tr_disp == '7') | (tr_disp == '18')).astype(np.float64)
        eval_dict['disp_ama'] = ((ev_disp == '7') | (ev_disp == '18')).astype(np.float64)
        extra_numeric.append('disp_ama')

    if 'admission_source_id' in train:
        train_dict['adm_from_er'] = (train['admission_source_id'] == '7').astype(np.float64)
        eval_dict['adm_from_er'] = (eval_cols['admission_source_id'] == '7').astype(np.float64)
        extra_numeric.append('adm_from_er')

    if 'admission_type_id' in train:
        train_dict['adm_type_emergency'] = (train['admission_type_id'] == '1').astype(np.float64)
        eval_dict['adm_type_emergency'] = (eval_cols['admission_type_id'] == '1').astype(np.float64)
        extra_numeric.append('adm_type_emergency')

    # 9. Missingness
    for miss_col in ('weight', 'medical_specialty', 'payer_code'):
        if miss_col in train:
            train_dict[f'{miss_col}_missing'] = np.array([(x is None) for x in train[miss_col]], dtype=np.float64)
            eval_dict[f'{miss_col}_missing'] = np.array([(x is None) for x in eval_cols[miss_col]], dtype=np.float64)
            extra_numeric.append(f'{miss_col}_missing')

    # 10. Multi-Way Clinical Tensor Decomposition (HOSVD + Non-Negative Matrix Factorization)
    from sklearn.decomposition import TruncatedSVD, NMF

    D_tr = np.column_stack([
        cm_train['cm_mi'], cm_train['cm_chf'], cm_train['cm_renal'], cm_train['cm_diabetes_complicated'],
        cm_train['cm_pulmonary'], cm_train['cm_cerebrovascular'], cm_train['cm_liver'], cm_train['cm_cancer'],
        train_dict['diag1_is_circulatory'], train_dict['diag1_is_diabetes'],
        train_dict['has_grp_respiratory'], train_dict['has_grp_digestive']
    ]).astype(np.float64)
    D_ev = np.column_stack([
        cm_eval['cm_mi'], cm_eval['cm_chf'], cm_eval['cm_renal'], cm_eval['cm_diabetes_complicated'],
        cm_eval['cm_pulmonary'], cm_eval['cm_cerebrovascular'], cm_eval['cm_liver'], cm_eval['cm_cancer'],
        eval_dict['diag1_is_circulatory'], eval_dict['diag1_is_diabetes'],
        eval_dict['has_grp_respiratory'], eval_dict['has_grp_digestive']
    ]).astype(np.float64)

    M_tr = np.column_stack([
        train_dict.get('insulin_active', np.zeros(n_train)),
        train_dict.get('insulin_up', np.zeros(n_train)),
        train_dict.get('insulin_down', np.zeros(n_train)),
        train_dict.get('metformin_active', np.zeros(n_train)),
        (train_dict['num_meds_changed'] > 0).astype(np.float64),
        train_dict['polypharmacy'],
        train_dict.get('a1c_tested', np.zeros(n_train))
    ]).astype(np.float64)
    M_ev = np.column_stack([
        eval_dict.get('insulin_active', np.zeros(n_eval)),
        eval_dict.get('insulin_up', np.zeros(n_eval)),
        eval_dict.get('insulin_down', np.zeros(n_eval)),
        eval_dict.get('metformin_active', np.zeros(n_eval)),
        (eval_dict['num_meds_changed'] > 0).astype(np.float64),
        eval_dict['polypharmacy'],
        eval_dict.get('a1c_tested', np.zeros(n_eval))
    ]).astype(np.float64)

    U_tr = np.column_stack([
        train_dict['has_inpatient'],
        train_dict['has_emergency'],
        train_dict['frequent_inpatient'],
        train_dict['disp_not_home'],
        train_dict['adm_type_emergency'],
        (tr_los >= 5.0).astype(np.float64),
        train_dict['is_elderly']
    ]).astype(np.float64)
    U_ev = np.column_stack([
        eval_dict['has_inpatient'],
        eval_dict['has_emergency'],
        eval_dict['frequent_inpatient'],
        eval_dict['disp_not_home'],
        eval_dict['adm_type_emergency'],
        (ev_los >= 5.0).astype(np.float64),
        eval_dict['is_elderly']
    ]).astype(np.float64)

    # Tensor outer-product matricization (D x M, D x U, M x U)
    Z_DM_tr = (D_tr[:, :, None] * M_tr[:, None, :]).reshape(n_train, -1)
    Z_DM_ev = (D_ev[:, :, None] * M_ev[:, None, :]).reshape(n_eval, -1)
    Z_DU_tr = (D_tr[:, :, None] * U_tr[:, None, :]).reshape(n_train, -1)
    Z_DU_ev = (D_ev[:, :, None] * U_ev[:, None, :]).reshape(n_eval, -1)
    Z_MU_tr = (M_tr[:, :, None] * U_tr[:, None, :]).reshape(n_train, -1)
    Z_MU_ev = (M_ev[:, :, None] * U_ev[:, None, :]).reshape(n_eval, -1)

    X_tensor_tr = np.hstack([D_tr, M_tr, U_tr, Z_DM_tr, Z_DU_tr, Z_MU_tr])
    X_tensor_ev = np.hstack([D_ev, M_ev, U_ev, Z_DM_ev, Z_DU_ev, Z_MU_ev])

    # 10a. Higher-Order SVD: Principal Orthogonal Tensor Modes
    svd = TruncatedSVD(n_components=6, algorithm='randomized', random_state=42)
    svd_tr = svd.fit_transform(X_tensor_tr)
    svd_ev = svd.transform(X_tensor_ev)
    for i in range(6):
        col_name = f'tensor_svd_{i}'
        train_dict[col_name] = svd_tr[:, i].astype(np.float64)
        eval_dict[col_name] = svd_ev[:, i].astype(np.float64)
        extra_numeric.append(col_name)

    # 10b. Non-negative Matrix Factorization: Additive Latent Clinical Phenotypes
    nmf = NMF(n_components=4, init='nndsvd', max_iter=40, random_state=42)
    nmf_tr = nmf.fit_transform(X_tensor_tr)
    nmf_ev = nmf.transform(X_tensor_ev)
    for i in range(4):
        col_name = f'pheno_nmf_{i}'
        train_dict[col_name] = nmf_tr[:, i].astype(np.float64)
        eval_dict[col_name] = nmf_ev[:, i].astype(np.float64)
        extra_numeric.append(col_name)

    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(
        task, train_dict, extra_numeric=tuple(extra_numeric), extra_categorical=tuple(extra_categorical)
    )
    x_eval = encoder.transform(eval_dict)

    # Diverse Dual-Engine Ensemble (LightGBM + XGBoost)
    model1 = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.024,
        num_leaves=31,
        min_child_samples=160,
        subsample=0.72,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=0.5,
        cat_smooth=50.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p1 = model1.predict_proba(x_eval)[:, 1]

    model2 = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.024,
        num_leaves=18,
        min_child_samples=120,
        subsample=0.68,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=2.0,
        cat_smooth=70.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    model2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p2 = model2.predict_proba(x_eval)[:, 1]

    model3 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.025,
        num_leaves=45,
        min_child_samples=200,
        subsample=0.78,
        subsample_freq=1,
        colsample_bytree=0.72,
        reg_lambda=0.5,
        reg_alpha=0.1,
        cat_smooth=35.0,
        random_state=789,
        n_jobs=1,
        verbose=-1,
    )
    model3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p3 = model3.predict_proba(x_eval)[:, 1]

    model4 = lgb.LGBMClassifier(
        n_estimators=270,
        learning_rate=0.024,
        num_leaves=25,
        min_child_samples=140,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.48,
        reg_lambda=1.0,
        reg_alpha=0.3,
        cat_smooth=55.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    model4.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p4 = model4.predict_proba(x_eval)[:, 1]

    xgb_mod = globals().get('xgb') or globals().get('xgboost') or __import__('sys').modules.get('xgboost')
    p_xgb = None
    if xgb_mod is not None and hasattr(xgb_mod, 'XGBClassifier'):
        try:
            # Exclude raw integer-encoded categorical columns to prevent ordinal split artifacts in XGBoost
            num_indices = [i for i in range(x_train.shape[1]) if i not in encoder.categorical_indices]
            xgb_clf = xgb_mod.XGBClassifier(
                n_estimators=240,
                learning_rate=0.024,
                max_depth=5,
                subsample=0.75,
                colsample_bytree=0.55,
                reg_lambda=2.0,
                reg_alpha=0.5,
                tree_method='hist',
                random_state=42,
                n_jobs=1,
                verbosity=0,
            )
            xgb_clf.fit(x_train[:, num_indices], y_train)
            p_xgb = xgb_clf.predict_proba(x_eval[:, num_indices])[:, 1]
        except Exception:
            p_xgb = None

    # Dual-Space Ensemble Optimization: Convex Combination of Linear Expectation and Log-Odds Consensus
    models_p = [p1, p2, p3, p4]
    if p_xgb is not None and np.all(np.isfinite(p_xgb)):
        models_p.append(p_xgb)
        weights = np.array([0.22, 0.22, 0.20, 0.16, 0.20], dtype=np.float64)
    else:
        weights = np.array([0.28, 0.28, 0.24, 0.20], dtype=np.float64)
    weights = weights / np.sum(weights)

    p_linear = sum(w * p for w, p in zip(weights, models_p))

    eps = 1e-5
    z_blend = sum(w * np.log(np.clip(p, eps, 1.0 - eps) / (1.0 - np.clip(p, eps, 1.0 - eps))) for w, p in zip(weights, models_p))
    p_logit = 1.0 / (1.0 + np.exp(-z_blend))

    preds = 0.50 * p_linear + 0.50 * p_logit
    return np.clip(preds, 0.0, 1.0)



# EVOLVE-BLOCK-END
# =====================================================================
