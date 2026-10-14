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
    train_cols = dict(train)
    eval_cols_dict = dict(eval_cols)

    extra_numeric = []
    extra_categorical = []

    # 1. Comorbidity features & Weighted Charlson Comorbidity Index (CCI)
    cm_train = icd9_comorbidities(train_cols['diag_1'], train_cols['diag_2'], train_cols['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols_dict['diag_1'], eval_cols_dict['diag_2'], eval_cols_dict['diag_3'])
    for k in cm_train:
        train_cols[k] = cm_train[k]
        eval_cols_dict[k] = cm_eval[k]
        extra_numeric.append(k)

    # Clinically weighted Charlson Comorbidity Index:
    # 1 pt: MI, CHF, PVD, Cerebrovascular, Dementia, Pulmonary, Rheumatic, Peptic ulcer
    # 2 pts: Hemiplegia, Renal, Cancer, Complicated diabetes
    # 3 pts: Liver disease
    # 6 pts: Metastatic cancer, HIV
    charlson_w_tr = (
        cm_train['cm_mi'] + cm_train['cm_chf'] + cm_train['cm_pvd'] +
        cm_train['cm_cerebrovascular'] + cm_train['cm_dementia'] +
        cm_train['cm_pulmonary'] + cm_train['cm_rheumatic'] + cm_train['cm_peptic_ulcer'] +
        2.0 * (cm_train['cm_hemiplegia'] + cm_train['cm_renal'] + cm_train['cm_cancer'] + cm_train['cm_diabetes_complicated']) +
        3.0 * cm_train['cm_liver'] +
        6.0 * (cm_train['cm_metastatic'] + cm_train['cm_hiv'])
    ).astype(np.float64)
    charlson_w_ev = (
        cm_eval['cm_mi'] + cm_eval['cm_chf'] + cm_eval['cm_pvd'] +
        cm_eval['cm_cerebrovascular'] + cm_eval['cm_dementia'] +
        cm_eval['cm_pulmonary'] + cm_eval['cm_rheumatic'] + cm_eval['cm_peptic_ulcer'] +
        2.0 * (cm_eval['cm_hemiplegia'] + cm_eval['cm_renal'] + cm_eval['cm_cancer'] + cm_eval['cm_diabetes_complicated']) +
        3.0 * cm_eval['cm_liver'] +
        6.0 * (cm_eval['cm_metastatic'] + cm_eval['cm_hiv'])
    ).astype(np.float64)
    train_cols['charlson_weighted'] = charlson_w_tr
    eval_cols_dict['charlson_weighted'] = charlson_w_ev
    extra_numeric.append('charlson_weighted')

    # Cardiorenal & high-risk comorbidity interactions
    train_cols['cm_cardiorenal'] = (cm_train['cm_chf'] * cm_train['cm_renal']).astype(np.float64)
    eval_cols_dict['cm_cardiorenal'] = (cm_eval['cm_chf'] * cm_eval['cm_renal']).astype(np.float64)
    extra_numeric.append('cm_cardiorenal')

    train_cols['cm_cardiopulmonary'] = (cm_train['cm_chf'] * cm_train['cm_pulmonary']).astype(np.float64)
    eval_cols_dict['cm_cardiopulmonary'] = (cm_eval['cm_chf'] * cm_eval['cm_pulmonary']).astype(np.float64)
    extra_numeric.append('cm_cardiopulmonary')

    train_cols['cm_diabetic_renal'] = (cm_train['cm_diabetes_complicated'] * cm_train['cm_renal']).astype(np.float64)
    eval_cols_dict['cm_diabetic_renal'] = (cm_eval['cm_diabetes_complicated'] * cm_eval['cm_renal']).astype(np.float64)
    extra_numeric.append('cm_diabetic_renal')

    train_cols['cm_chf_diabetes'] = (cm_train['cm_chf'] * cm_train['cm_diabetes_complicated']).astype(np.float64)
    eval_cols_dict['cm_chf_diabetes'] = (cm_eval['cm_chf'] * cm_eval['cm_diabetes_complicated']).astype(np.float64)
    extra_numeric.append('cm_chf_diabetes')

    # 2. ICD-9 Chapter Groups & 3-Digit Prefixes
    def _get_prefix(val):
        if val is None or not isinstance(val, str) or len(val) == 0 or val == '?':
            return 'missing'
        parts = val.split('.')
        p = parts[0]
        return p[:3] if len(p) >= 3 else p

    for col in ('diag_1', 'diag_2', 'diag_3'):
        g_name = f'{col}_group'
        train_cols[g_name] = np.array([icd9_group(x) for x in train_cols[col]], dtype=object)
        eval_cols_dict[g_name] = np.array([icd9_group(x) for x in eval_cols_dict[col]], dtype=object)
        extra_categorical.append(g_name)

        p_name = f'{col}_prefix'
        train_cols[p_name] = np.array([_get_prefix(x) for x in train_cols[col]], dtype=object)
        eval_cols_dict[p_name] = np.array([_get_prefix(x) for x in eval_cols_dict[col]], dtype=object)
        extra_categorical.append(p_name)

    # Core HRRP conditions & clinical decompensations
    def _check_diag_condition(diag_cols_list, prefix_set):
        n = len(diag_cols_list[0])
        out = np.zeros(n, dtype=np.float64)
        for d_col in diag_cols_list:
            for i, v in enumerate(d_col):
                if v is not None and isinstance(v, str):
                    if v.split('.')[0][:3] in prefix_set:
                        out[i] = 1.0
        return out

    diags_tr = [train_cols['diag_1'], train_cols['diag_2'], train_cols['diag_3']]
    diags_ev = [eval_cols_dict['diag_1'], eval_cols_dict['diag_2'], eval_cols_dict['diag_3']]

    cond_map = {
        'flag_hf': {'428'},
        'flag_ami': {'410'},
        'flag_pneumonia': {'480', '481', '482', '483', '484', '485', '486'},
        'flag_copd': {'491', '492', '494', '496'},
        'flag_renal': {'584', '585', '586'},
        'flag_sepsis': {'038', '995'},
        'flag_electrolyte': {'276'},
        'flag_afib': {'427'},
        'flag_hypertension': {'401', '402', '403', '404'},
        'flag_uti': {'599'},
        'flag_complications': {'996', '997', '998', '999'},
    }
    for cond_name, p_set in cond_map.items():
        train_cols[cond_name] = _check_diag_condition(diags_tr, p_set)
        eval_cols_dict[cond_name] = _check_diag_condition(diags_ev, p_set)
        extra_numeric.append(cond_name)

    # Primary diagnosis condition flags (index reason for admission)
    for c_flag, p_set in [
        ('diag1_is_hf', {'428'}),
        ('diag1_is_ami', {'410'}),
        ('diag1_is_renal', {'584', '585', '586'}),
        ('diag1_is_sepsis', {'038', '995'}),
    ]:
        train_cols[c_flag] = np.array([1.0 if _get_prefix(x) in p_set else 0.0 for x in train_cols['diag_1']], dtype=np.float64)
        eval_cols_dict[c_flag] = np.array([1.0 if _get_prefix(x) in p_set else 0.0 for x in eval_cols_dict['diag_1']], dtype=np.float64)
        extra_numeric.append(c_flag)

    # 3. High-cardinality & 3-Digit Prefix Target Encodings
    te_cols = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_prefix', 'diag_2_prefix', 'diag_3_prefix',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'medical_specialty', 'diag_1_group',
        'payer_code'
    ]
    for col in te_cols:
        if col in train_cols and col in eval_cols_dict:
            te_name = f'{col}_te'
            tr_te, ev_te = oof_target_encode(
                train_cols[col], y_train, eval_cols_dict[col],
                n_folds=5, smoothing=20.0, seed=42
            )
            train_cols[te_name] = tr_te.astype(np.float64)
            eval_cols_dict[te_name] = ev_te.astype(np.float64)
            extra_numeric.append(te_name)

    # High-signal interaction target encodings
    # a) Disease group x Discharge disposition
    inter_tr1 = np.array([f"{g}_{d}" for g, d in zip(train_cols['diag_1_group'], train_cols['discharge_disposition_id'])], dtype=object)
    inter_ev1 = np.array([f"{g}_{d}" for g, d in zip(eval_cols_dict['diag_1_group'], eval_cols_dict['discharge_disposition_id'])], dtype=object)
    tr_ite1, ev_ite1 = oof_target_encode(inter_tr1, y_train, inter_ev1, n_folds=5, smoothing=25.0, seed=42)
    train_cols['diag_disposition_te'] = tr_ite1.astype(np.float64)
    eval_cols_dict['diag_disposition_te'] = ev_ite1.astype(np.float64)
    extra_numeric.append('diag_disposition_te')

    # b) Admission type x Discharge disposition
    inter_tr2 = np.array([f"{a}_{d}" for a, d in zip(train_cols['admission_type_id'], train_cols['discharge_disposition_id'])], dtype=object)
    inter_ev2 = np.array([f"{a}_{d}" for a, d in zip(eval_cols_dict['admission_type_id'], eval_cols_dict['discharge_disposition_id'])], dtype=object)
    tr_ite2, ev_ite2 = oof_target_encode(inter_tr2, y_train, inter_ev2, n_folds=5, smoothing=20.0, seed=42)
    train_cols['adm_disposition_te'] = tr_ite2.astype(np.float64)
    eval_cols_dict['adm_disposition_te'] = ev_ite2.astype(np.float64)
    extra_numeric.append('adm_disposition_te')

    # c) Age bracket x Discharge disposition
    inter_tr3 = np.array([f"{a}_{d}" for a, d in zip(train_cols['age'], train_cols['discharge_disposition_id'])], dtype=object)
    inter_ev3 = np.array([f"{a}_{d}" for a, d in zip(eval_cols_dict['age'], eval_cols_dict['discharge_disposition_id'])], dtype=object)
    tr_ite3, ev_ite3 = oof_target_encode(inter_tr3, y_train, inter_ev3, n_folds=5, smoothing=25.0, seed=42)
    train_cols['age_disposition_te'] = tr_ite3.astype(np.float64)
    eval_cols_dict['age_disposition_te'] = ev_ite3.astype(np.float64)
    extra_numeric.append('age_disposition_te')

    # d) Primary x Secondary Diagnosis chapter dyad
    inter_tr4 = np.array([f"{g1}_{g2}" for g1, g2 in zip(train_cols['diag_1_group'], train_cols['diag_2_group'])], dtype=object)
    inter_ev4 = np.array([f"{g1}_{g2}" for g1, g2 in zip(eval_cols_dict['diag_1_group'], eval_cols_dict['diag_2_group'])], dtype=object)
    tr_ite4, ev_ite4 = oof_target_encode(inter_tr4, y_train, inter_ev4, n_folds=5, smoothing=25.0, seed=42)
    train_cols['diag1_diag2_te'] = tr_ite4.astype(np.float64)
    eval_cols_dict['diag1_diag2_te'] = ev_ite4.astype(np.float64)
    extra_numeric.append('diag1_diag2_te')

    # 4. Utilization, LACE Index & Intensity features
    total_vis_tr = train_cols['number_outpatient'] + train_cols['number_emergency'] + train_cols['number_inpatient']
    total_vis_ev = eval_cols_dict['number_outpatient'] + eval_cols_dict['number_emergency'] + eval_cols_dict['number_inpatient']

    train_cols['total_prior_visits'] = total_vis_tr.astype(np.float64)
    eval_cols_dict['total_prior_visits'] = total_vis_ev.astype(np.float64)
    extra_numeric.append('total_prior_visits')

    # Log-transformed utilizations to smooth extreme tails
    train_cols['log_inpatient'] = np.log1p(train_cols['number_inpatient']).astype(np.float64)
    eval_cols_dict['log_inpatient'] = np.log1p(eval_cols_dict['number_inpatient']).astype(np.float64)
    extra_numeric.append('log_inpatient')

    train_cols['log_emergency'] = np.log1p(train_cols['number_emergency']).astype(np.float64)
    eval_cols_dict['log_emergency'] = np.log1p(eval_cols_dict['number_emergency']).astype(np.float64)
    extra_numeric.append('log_emergency')

    train_cols['log_total_visits'] = np.log1p(total_vis_tr).astype(np.float64)
    eval_cols_dict['log_total_visits'] = np.log1p(total_vis_ev).astype(np.float64)
    extra_numeric.append('log_total_visits')

    train_cols['inpatient_ratio'] = (train_cols['number_inpatient'] / (total_vis_tr + 1.0)).astype(np.float64)
    eval_cols_dict['inpatient_ratio'] = (eval_cols_dict['number_inpatient'] / (total_vis_ev + 1.0)).astype(np.float64)
    extra_numeric.append('inpatient_ratio')

    train_cols['emergency_ratio'] = (train_cols['number_emergency'] / (total_vis_tr + 1.0)).astype(np.float64)
    eval_cols_dict['emergency_ratio'] = (eval_cols_dict['number_emergency'] / (total_vis_ev + 1.0)).astype(np.float64)
    extra_numeric.append('emergency_ratio')

    train_cols['has_inpatient'] = (train_cols['number_inpatient'] > 0).astype(np.float64)
    eval_cols_dict['has_inpatient'] = (eval_cols_dict['number_inpatient'] > 0).astype(np.float64)
    extra_numeric.append('has_inpatient')

    train_cols['frequent_inpatient'] = (train_cols['number_inpatient'] >= 2).astype(np.float64)
    eval_cols_dict['frequent_inpatient'] = (eval_cols_dict['number_inpatient'] >= 2).astype(np.float64)
    extra_numeric.append('frequent_inpatient')

    train_cols['has_emergency'] = (train_cols['number_emergency'] > 0).astype(np.float64)
    eval_cols_dict['has_emergency'] = (eval_cols_dict['number_emergency'] > 0).astype(np.float64)
    extra_numeric.append('has_emergency')

    acute_vis_tr = train_cols['number_inpatient'] + train_cols['number_emergency']
    acute_vis_ev = eval_cols_dict['number_inpatient'] + eval_cols_dict['number_emergency']
    train_cols['acute_prior_visits'] = acute_vis_tr.astype(np.float64)
    eval_cols_dict['acute_prior_visits'] = acute_vis_ev.astype(np.float64)
    extra_numeric.append('acute_prior_visits')

    train_cols['log_outpatient'] = np.log1p(train_cols['number_outpatient']).astype(np.float64)
    eval_cols_dict['log_outpatient'] = np.log1p(eval_cols_dict['number_outpatient']).astype(np.float64)
    extra_numeric.append('log_outpatient')

    train_cols['acute_visit_ratio'] = (acute_vis_tr / (total_vis_tr + 1.0)).astype(np.float64)
    eval_cols_dict['acute_visit_ratio'] = (acute_vis_ev / (total_vis_ev + 1.0)).astype(np.float64)
    extra_numeric.append('acute_visit_ratio')

    train_cols['super_inpatient'] = (train_cols['number_inpatient'] >= 3).astype(np.float64)
    eval_cols_dict['super_inpatient'] = (eval_cols_dict['number_inpatient'] >= 3).astype(np.float64)
    extra_numeric.append('super_inpatient')

    # Validated LACE Readmission Risk Index:
    # L: Length of stay (1:1, 2:2, 3:3, 4-6:4, 7-13:5, >=14:7)
    # A: Acuity (Emergency/Urgent admission_type_id in 1, 2 = 3 pts)
    # C: Charlson Comorbidity (0:0, 1:1, 2:2, 3:3, >=4:5)
    # E: Emergency department visits in prior year (0:0, 1:1, 2:2, 3:3, >=4:4)
    def _compute_lace(los, adm_type, cci, ed_vis):
        l_pts = np.zeros(len(los), dtype=np.float64)
        l_pts[los == 1] = 1.0
        l_pts[los == 2] = 2.0
        l_pts[los == 3] = 3.0
        l_pts[(los >= 4) & (los <= 6)] = 4.0
        l_pts[(los >= 7) & (los <= 13)] = 5.0
        l_pts[los >= 14] = 7.0

        a_pts = np.array([3.0 if str(a) in ('1', '2') else 0.0 for a in adm_type], dtype=np.float64)

        c_pts = np.zeros(len(cci), dtype=np.float64)
        c_pts[cci == 1] = 1.0
        c_pts[cci == 2] = 2.0
        c_pts[cci == 3] = 3.0
        c_pts[cci >= 4] = 5.0

        e_pts = np.minimum(ed_vis, 4.0).astype(np.float64)

        return l_pts + a_pts + c_pts + e_pts

    train_cols['lace_score'] = _compute_lace(
        train_cols['time_in_hospital'], train_cols['admission_type_id'],
        cm_train['cm_count'], train_cols['number_emergency']
    )
    eval_cols_dict['lace_score'] = _compute_lace(
        eval_cols_dict['time_in_hospital'], eval_cols_dict['admission_type_id'],
        cm_eval['cm_count'], eval_cols_dict['number_emergency']
    )
    extra_numeric.append('lace_score')

    train_cols['lace_high_risk'] = (train_cols['lace_score'] >= 10.0).astype(np.float64)
    eval_cols_dict['lace_high_risk'] = (eval_cols_dict['lace_score'] >= 10.0).astype(np.float64)
    extra_numeric.append('lace_high_risk')

    # Validated HOSPITAL Readmission Risk Score (Donze et al. 2013)
    def _compute_hospital(los, adm_type, inpat, spec, num_proc):
        pts = np.zeros(len(los), dtype=np.float64)
        pts += (los >= 5).astype(np.float64) * 2.0
        pts += np.array([1.0 if str(a) in ('1', '2') else 0.0 for a in adm_type], dtype=np.float64)
        pts += (num_proc > 0).astype(np.float64) * 1.0
        pts += ((inpat >= 2) & (inpat <= 5)).astype(np.float64) * 2.0
        pts += (inpat > 5).astype(np.float64) * 5.0
        if spec is not None:
            pts += np.array([2.0 if s is not None and 'Oncology' in str(s) else 0.0 for s in spec], dtype=np.float64)
        return pts

    train_cols['hospital_score'] = _compute_hospital(
        train_cols['time_in_hospital'], train_cols['admission_type_id'],
        train_cols['number_inpatient'], train_cols.get('medical_specialty'),
        train_cols['num_procedures']
    )
    eval_cols_dict['hospital_score'] = _compute_hospital(
        eval_cols_dict['time_in_hospital'], eval_cols_dict['admission_type_id'],
        eval_cols_dict['number_inpatient'], eval_cols_dict.get('medical_specialty'),
        eval_cols_dict['num_procedures']
    )
    extra_numeric.append('hospital_score')

    # Admission source and type phenotypes
    adm_src_tr = train_cols['admission_source_id']
    adm_src_ev = eval_cols_dict['admission_source_id']
    train_cols['adm_src_er'] = (adm_src_tr == '7').astype(np.float64)
    eval_cols_dict['adm_src_er'] = (adm_src_ev == '7').astype(np.float64)
    extra_numeric.append('adm_src_er')

    train_cols['adm_src_transfer'] = np.array([1.0 if str(s) in ('4', '5', '6') else 0.0 for s in adm_src_tr], dtype=np.float64)
    eval_cols_dict['adm_src_transfer'] = np.array([1.0 if str(s) in ('4', '5', '6') else 0.0 for s in adm_src_ev], dtype=np.float64)
    extra_numeric.append('adm_src_transfer')

    adm_type_tr = train_cols['admission_type_id']
    adm_type_ev = eval_cols_dict['admission_type_id']
    train_cols['adm_type_emergency'] = (adm_type_tr == '1').astype(np.float64)
    eval_cols_dict['adm_type_emergency'] = (adm_type_ev == '1').astype(np.float64)
    extra_numeric.append('adm_type_emergency')

    train_cols['adm_type_elective'] = (adm_type_tr == '3').astype(np.float64)
    eval_cols_dict['adm_type_elective'] = (adm_type_ev == '3').astype(np.float64)
    extra_numeric.append('adm_type_elective')

    train_cols['acute_er_admission'] = ((adm_type_tr == '1') & (adm_src_tr == '7')).astype(np.float64)
    eval_cols_dict['acute_er_admission'] = ((adm_type_ev == '1') & (adm_src_ev == '7')).astype(np.float64)
    extra_numeric.append('acute_er_admission')

    # Payer phenotypes
    if 'payer_code' in train_cols:
        pc_tr = train_cols['payer_code']
        pc_ev = eval_cols_dict['payer_code']
        train_cols['payer_is_mc'] = (pc_tr == 'MC').astype(np.float64)
        eval_cols_dict['payer_is_mc'] = (pc_ev == 'MC').astype(np.float64)
        extra_numeric.append('payer_is_mc')

        train_cols['payer_is_md'] = (pc_tr == 'MD').astype(np.float64)
        eval_cols_dict['payer_is_md'] = (pc_ev == 'MD').astype(np.float64)
        extra_numeric.append('payer_is_md')

        train_cols['payer_is_sp'] = (pc_tr == 'SP').astype(np.float64)
        eval_cols_dict['payer_is_sp'] = (pc_ev == 'SP').astype(np.float64)
        extra_numeric.append('payer_is_sp')

    # Discharge disposition phenotypes
    disp_tr = train_cols['discharge_disposition_id']
    disp_ev = eval_cols_dict['discharge_disposition_id']
    train_cols['disch_home'] = np.array([1.0 if str(d) == '1' else 0.0 for d in disp_tr], dtype=np.float64)
    eval_cols_dict['disch_home'] = np.array([1.0 if str(d) == '1' else 0.0 for d in disp_ev], dtype=np.float64)
    extra_numeric.append('disch_home')

    train_cols['disch_snf'] = np.array([1.0 if str(d) == '3' else 0.0 for d in disp_tr], dtype=np.float64)
    eval_cols_dict['disch_snf'] = np.array([1.0 if str(d) == '3' else 0.0 for d in disp_ev], dtype=np.float64)
    extra_numeric.append('disch_snf')

    train_cols['disch_home_health'] = np.array([1.0 if str(d) == '6' else 0.0 for d in disp_tr], dtype=np.float64)
    eval_cols_dict['disch_home_health'] = np.array([1.0 if str(d) == '6' else 0.0 for d in disp_ev], dtype=np.float64)
    extra_numeric.append('disch_home_health')

    train_cols['disch_ama'] = np.array([1.0 if str(d) in ('7', '18') else 0.0 for d in disp_tr], dtype=np.float64)
    eval_cols_dict['disch_ama'] = np.array([1.0 if str(d) in ('7', '18') else 0.0 for d in disp_ev], dtype=np.float64)
    extra_numeric.append('disch_ama')

    los_tr = np.maximum(train_cols['time_in_hospital'], 1.0)
    los_ev = np.maximum(eval_cols_dict['time_in_hospital'], 1.0)

    train_cols['med_per_day'] = (train_cols['num_medications'] / los_tr).astype(np.float64)
    eval_cols_dict['med_per_day'] = (eval_cols_dict['num_medications'] / los_ev).astype(np.float64)
    extra_numeric.append('med_per_day')

    train_cols['lab_per_day'] = (train_cols['num_lab_procedures'] / los_tr).astype(np.float64)
    eval_cols_dict['lab_per_day'] = (eval_cols_dict['num_lab_procedures'] / los_ev).astype(np.float64)
    extra_numeric.append('lab_per_day')

    train_cols['proc_per_day'] = (train_cols['num_procedures'] / los_tr).astype(np.float64)
    eval_cols_dict['proc_per_day'] = (eval_cols_dict['num_procedures'] / los_ev).astype(np.float64)
    extra_numeric.append('proc_per_day')

    train_cols['diag_per_day'] = (train_cols['number_diagnoses'] / los_tr).astype(np.float64)
    eval_cols_dict['diag_per_day'] = (eval_cols_dict['number_diagnoses'] / los_ev).astype(np.float64)
    extra_numeric.append('diag_per_day')

    train_cols['total_procedures'] = (train_cols['num_lab_procedures'] + train_cols['num_procedures']).astype(np.float64)
    eval_cols_dict['total_procedures'] = (eval_cols_dict['num_lab_procedures'] + eval_cols_dict['num_procedures']).astype(np.float64)
    extra_numeric.append('total_procedures')

    train_cols['lab_per_med'] = (train_cols['num_lab_procedures'] / (train_cols['num_medications'] + 1.0)).astype(np.float64)
    eval_cols_dict['lab_per_med'] = (eval_cols_dict['num_lab_procedures'] / (eval_cols_dict['num_medications'] + 1.0)).astype(np.float64)
    extra_numeric.append('lab_per_med')

    train_cols['med_per_diag'] = (train_cols['num_medications'] / (train_cols['number_diagnoses'] + 1.0)).astype(np.float64)
    eval_cols_dict['med_per_diag'] = (eval_cols_dict['num_medications'] / (eval_cols_dict['number_diagnoses'] + 1.0)).astype(np.float64)
    extra_numeric.append('med_per_diag')

    train_cols['high_los'] = (train_cols['time_in_hospital'] >= 7).astype(np.float64)
    eval_cols_dict['high_los'] = (eval_cols_dict['time_in_hospital'] >= 7).astype(np.float64)
    extra_numeric.append('high_los')

    train_cols['proc_per_diag'] = (train_cols['num_procedures'] / (train_cols['number_diagnoses'] + 1.0)).astype(np.float64)
    eval_cols_dict['proc_per_diag'] = (eval_cols_dict['num_procedures'] / (eval_cols_dict['number_diagnoses'] + 1.0)).astype(np.float64)
    extra_numeric.append('proc_per_diag')

    train_cols['is_max_diagnoses'] = (train_cols['number_diagnoses'] >= 9.0).astype(np.float64)
    eval_cols_dict['is_max_diagnoses'] = (eval_cols_dict['number_diagnoses'] >= 9.0).astype(np.float64)
    extra_numeric.append('is_max_diagnoses')

    # 5. Age numeric midpoint
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    train_cols['age_num'] = np.array([age_map.get(a, 65.0) for a in train_cols['age']], dtype=np.float64)
    eval_cols_dict['age_num'] = np.array([age_map.get(a, 65.0) for a in eval_cols_dict['age']], dtype=np.float64)
    extra_numeric.append('age_num')

    # 6. Medication patterns & changes
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    present_meds = [m for m in med_names if m in train_cols and m in eval_cols_dict]

    n_tr = len(y_train)
    n_ev = len(eval_cols_dict['age'])
    act_tr, act_ev = np.zeros(n_tr, dtype=np.float64), np.zeros(n_ev, dtype=np.float64)
    chg_tr, chg_ev = np.zeros(n_tr, dtype=np.float64), np.zeros(n_ev, dtype=np.float64)
    up_tr, up_ev = np.zeros(n_tr, dtype=np.float64), np.zeros(n_ev, dtype=np.float64)
    dn_tr, dn_ev = np.zeros(n_tr, dtype=np.float64), np.zeros(n_ev, dtype=np.float64)

    for m in present_meds:
        c_tr = train_cols[m]
        c_ev = eval_cols_dict[m]
        act_tr += ((c_tr == 'Steady') | (c_tr == 'Up') | (c_tr == 'Down')).astype(np.float64)
        act_ev += ((c_ev == 'Steady') | (c_ev == 'Up') | (c_ev == 'Down')).astype(np.float64)
        chg_tr += ((c_tr == 'Up') | (c_tr == 'Down')).astype(np.float64)
        chg_ev += ((c_ev == 'Up') | (c_ev == 'Down')).astype(np.float64)
        up_tr += (c_tr == 'Up').astype(np.float64)
        up_ev += (c_ev == 'Up').astype(np.float64)
        dn_tr += (c_tr == 'Down').astype(np.float64)
        dn_ev += (c_ev == 'Down').astype(np.float64)

    train_cols['med_count_active'] = act_tr
    eval_cols_dict['med_count_active'] = act_ev
    extra_numeric.append('med_count_active')

    train_cols['med_count_changes'] = chg_tr
    eval_cols_dict['med_count_changes'] = chg_ev
    extra_numeric.append('med_count_changes')

    train_cols['med_count_up'] = up_tr
    eval_cols_dict['med_count_up'] = up_ev
    extra_numeric.append('med_count_up')

    train_cols['med_count_down'] = dn_tr
    eval_cols_dict['med_count_down'] = dn_ev
    extra_numeric.append('med_count_down')

    if 'insulin' in train_cols:
        ins_tr = train_cols['insulin']
        ins_ev = eval_cols_dict['insulin']
        on_ins_tr = (ins_tr != 'No') & (ins_tr is not None)
        on_ins_ev = (ins_ev != 'No') & (ins_ev is not None)
        train_cols['on_insulin'] = on_ins_tr.astype(np.float64)
        eval_cols_dict['on_insulin'] = on_ins_ev.astype(np.float64)
        extra_numeric.append('on_insulin')

        train_cols['insulin_up'] = (ins_tr == 'Up').astype(np.float64)
        eval_cols_dict['insulin_up'] = (ins_ev == 'Up').astype(np.float64)
        extra_numeric.append('insulin_up')

        train_cols['insulin_down'] = (ins_tr == 'Down').astype(np.float64)
        eval_cols_dict['insulin_down'] = (ins_ev == 'Down').astype(np.float64)
        extra_numeric.append('insulin_down')

        train_cols['insulin_steady'] = (ins_tr == 'Steady').astype(np.float64)
        eval_cols_dict['insulin_steady'] = (ins_ev == 'Steady').astype(np.float64)
        extra_numeric.append('insulin_steady')

        # Insulin titration (unstable dosage during encounter)
        train_cols['insulin_titrated'] = ((ins_tr == 'Up') | (ins_tr == 'Down')).astype(np.float64)
        eval_cols_dict['insulin_titrated'] = ((ins_ev == 'Up') | (ins_ev == 'Down')).astype(np.float64)
        extra_numeric.append('insulin_titrated')

        # Oral + Insulin complex polypharmacy
        non_ins_active_tr = act_tr - train_cols['on_insulin']
        non_ins_active_ev = act_ev - eval_cols_dict['on_insulin']
        train_cols['insulin_and_oral'] = ((train_cols['on_insulin'] > 0) & (non_ins_active_tr > 0)).astype(np.float64)
        eval_cols_dict['insulin_and_oral'] = ((eval_cols_dict['on_insulin'] > 0) & (non_ins_active_ev > 0)).astype(np.float64)
        extra_numeric.append('insulin_and_oral')

        train_cols['insulin_monotherapy'] = ((train_cols['on_insulin'] > 0) & (non_ins_active_tr == 0)).astype(np.float64)
        eval_cols_dict['insulin_monotherapy'] = ((eval_cols_dict['on_insulin'] > 0) & (non_ins_active_ev == 0)).astype(np.float64)
        extra_numeric.append('insulin_monotherapy')

    # Sulfonylurea class indicators & hypoglycemia interaction
    sulfo_names = ['glimepiride', 'glipizide', 'glyburide', 'chlorpropamide', 'tolbutamide', 'acetohexamide', 'tolazamide']
    sulfo_tr = np.zeros(n_tr, dtype=np.float64)
    sulfo_ev = np.zeros(n_ev, dtype=np.float64)
    for sm in sulfo_names:
        if sm in train_cols and sm in eval_cols_dict:
            sulfo_tr += (train_cols[sm] != 'No').astype(np.float64)
            sulfo_ev += (eval_cols_dict[sm] != 'No').astype(np.float64)
    train_cols['on_sulfonylurea'] = (sulfo_tr > 0).astype(np.float64)
    eval_cols_dict['on_sulfonylurea'] = (sulfo_ev > 0).astype(np.float64)
    extra_numeric.append('on_sulfonylurea')

    if 'metformin' in train_cols and 'metformin' in eval_cols_dict:
        train_cols['on_metformin'] = (train_cols['metformin'] != 'No').astype(np.float64)
        eval_cols_dict['on_metformin'] = (eval_cols_dict['metformin'] != 'No').astype(np.float64)
        extra_numeric.append('on_metformin')

    if 'insulin' in train_cols:
        train_cols['insulin_and_sulfo'] = (train_cols['on_insulin'] * train_cols['on_sulfonylurea']).astype(np.float64)
        eval_cols_dict['insulin_and_sulfo'] = (eval_cols_dict['on_insulin'] * eval_cols_dict['on_sulfonylurea']).astype(np.float64)
        extra_numeric.append('insulin_and_sulfo')

    if 'change' in train_cols:
        train_cols['had_change'] = (train_cols['change'] == 'Ch').astype(np.float64)
        eval_cols_dict['had_change'] = (eval_cols_dict['change'] == 'Ch').astype(np.float64)
        extra_numeric.append('had_change')

    if 'diabetesMed' in train_cols:
        train_cols['on_diabetes_med'] = (train_cols['diabetesMed'] == 'Yes').astype(np.float64)
        eval_cols_dict['on_diabetes_med'] = (eval_cols_dict['diabetesMed'] == 'Yes').astype(np.float64)
        extra_numeric.append('on_diabetes_med')

    # 7. Lab testing indicators
    if 'A1Cresult' in train_cols:
        a1c_tr = train_cols['A1Cresult']
        a1c_ev = eval_cols_dict['A1Cresult']
        train_cols['a1c_tested'] = (a1c_tr != 'None').astype(np.float64)
        eval_cols_dict['a1c_tested'] = (a1c_ev != 'None').astype(np.float64)
        extra_numeric.append('a1c_tested')

        train_cols['a1c_high'] = ((a1c_tr == '>8') | (a1c_tr == '>7')).astype(np.float64)
        eval_cols_dict['a1c_high'] = ((a1c_ev == '>8') | (a1c_ev == '>7')).astype(np.float64)
        extra_numeric.append('a1c_high')

        train_cols['a1c_normal'] = (a1c_tr == 'Norm').astype(np.float64)
        eval_cols_dict['a1c_normal'] = (a1c_ev == 'Norm').astype(np.float64)
        extra_numeric.append('a1c_normal')

        chg_tr_flag = (train_cols['change'] == 'Ch') if 'change' in train_cols else np.zeros(n_tr, dtype=bool)
        chg_ev_flag = (eval_cols_dict['change'] == 'Ch') if 'change' in eval_cols_dict else np.zeros(n_ev, dtype=bool)
        train_cols['a1c_high_no_change'] = ((train_cols['a1c_high'] == 1.0) & (~chg_tr_flag)).astype(np.float64)
        eval_cols_dict['a1c_high_no_change'] = ((eval_cols_dict['a1c_high'] == 1.0) & (~chg_ev_flag)).astype(np.float64)
        extra_numeric.append('a1c_high_no_change')

    if 'max_glu_serum' in train_cols:
        glu_tr = train_cols['max_glu_serum']
        glu_ev = eval_cols_dict['max_glu_serum']
        train_cols['glu_tested'] = (glu_tr != 'None').astype(np.float64)
        eval_cols_dict['glu_tested'] = (glu_ev != 'None').astype(np.float64)
        extra_numeric.append('glu_tested')

        train_cols['glu_high'] = ((glu_tr == '>300') | (glu_tr == '>200')).astype(np.float64)
        eval_cols_dict['glu_high'] = ((glu_ev == '>300') | (glu_ev == '>200')).astype(np.float64)
        extra_numeric.append('glu_high')

    # Fit encoder
    encoder = TabularEncoder(min_count=35)
    encoder.fit(
        task, train_cols,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical)
    )
    x_train = encoder.transform(train_cols)
    x_eval = encoder.transform(eval_cols_dict)

    # Structurally diverse LightGBM ensemble:
    # 1. Balanced standard leaf-wise model with path_smooth
    # 2. Extra-Trees model (random cutpoints to explore split space)
    # 3. Depth-constrained model (max_depth=5, prevents deep spurious interactions)
    # 4. Aggressive feature-subsampled model (colsample=0.45)
    # 5. Asymmetric positive-class-weighted model (scale_pos_weight=1.5, calibrated back via odds ratio)
    model_configs = [
        (lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.024,
            num_leaves=31,
            min_child_samples=130,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=2.5,
            reg_alpha=0.2,
            path_smooth=5.0,
            cat_smooth=45.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ), 1.0),
        (lgb.LGBMClassifier(
            n_estimators=290,
            learning_rate=0.022,
            num_leaves=35,
            min_child_samples=110,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=3.0,
            reg_alpha=0.1,
            extra_trees=True,
            path_smooth=3.0,
            cat_smooth=35.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ), 1.0),
        (lgb.LGBMClassifier(
            n_estimators=270,
            learning_rate=0.025,
            max_depth=5,
            num_leaves=24,
            min_child_samples=160,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=2.0,
            reg_alpha=0.3,
            cat_smooth=50.0,
            random_state=456,
            n_jobs=1,
            verbose=-1,
        ), 1.0),
        (lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.026,
            num_leaves=28,
            min_child_samples=140,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.45,
            reg_lambda=4.0,
            reg_alpha=0.5,
            path_smooth=5.0,
            cat_smooth=40.0,
            random_state=789,
            n_jobs=1,
            verbose=-1,
        ), 1.0),
        (lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.022,
            num_leaves=30,
            min_child_samples=150,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            scale_pos_weight=1.5,
            reg_lambda=3.0,
            reg_alpha=0.3,
            cat_smooth=45.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        ), 1.5),
    ]

    preds = np.zeros(len(x_eval), dtype=np.float64)
    for model, pos_w in model_configs:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        p = model.predict_proba(x_eval)[:, 1]
        if pos_w != 1.0:
            # Calibrate back to true prevalence scale via Bayes odds inversion
            p = p / (p + pos_w * (1.0 - p))
        preds += p
    preds /= len(model_configs)

    preds = np.clip(preds, 0.0, 1.0)
    if np.isnan(preds).any():
        preds = np.nan_to_num(preds, nan=0.114)

    return preds


# EVOLVE-BLOCK-END
# =====================================================================
