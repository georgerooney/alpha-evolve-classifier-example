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
    import lightgbm as lgb
    from sklearn.decomposition import TruncatedSVD

    # 1. Comorbidities via clinical toolkit
    cm_tr = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_ev = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])

    def compute_comorbidity_syndromes(cm):
        chf = cm['cm_chf']
        renal = cm['cm_renal']
        mi = cm['cm_mi']
        pvd = cm['cm_pvd']
        cevd = cm['cm_cerebrovascular']
        cancer = cm['cm_cancer']
        met = cm['cm_metastatic']
        liver = cm['cm_liver']
        diab_comp = cm['cm_diabetes_complicated']

        cardiorenal = ((chf > 0.5) & (renal > 0.5)).astype(np.float64)
        vascular = ((mi > 0.5) | (pvd > 0.5) | (cevd > 0.5)).astype(np.float64)
        high_burden = ((cancer > 0.5) | (met > 0.5) | (liver > 0.5)).astype(np.float64)
        diabetic_triad = ((diab_comp > 0.5) & (renal > 0.5) & (chf > 0.5)).astype(np.float64)

        charlson_weighted = (
            1.0 * (mi + chf + pvd + cevd + cm['cm_dementia'] + cm['cm_pulmonary'] +
                   cm['cm_rheumatic'] + cm['cm_peptic_ulcer'] + diab_comp) +
            2.0 * (cm['cm_hemiplegia'] + renal + cancer) +
            3.0 * liver +
            6.0 * (met + cm['cm_hiv'])
        ).astype(np.float64)

        return {
            'cm_cardiorenal': cardiorenal,
            'cm_vascular': vascular,
            'cm_high_burden': high_burden,
            'cm_diabetic_triad': diabetic_triad,
            'cm_charlson_weighted': charlson_weighted,
        }

    cm_synd_tr = compute_comorbidity_syndromes(cm_tr)
    cm_synd_ev = compute_comorbidity_syndromes(cm_ev)

    # 2. Low-rank decomposition (TruncatedSVD) of the comorbidity multi-condition matrix
    cm_keys = [
        'cm_mi', 'cm_chf', 'cm_pvd', 'cm_cerebrovascular', 'cm_dementia',
        'cm_pulmonary', 'cm_rheumatic', 'cm_peptic_ulcer', 'cm_liver',
        'cm_hemiplegia', 'cm_renal', 'cm_cancer', 'cm_metastatic', 'cm_hiv',
        'cm_diabetes_complicated'
    ]
    cm_mat_tr = np.column_stack([cm_tr[k] for k in cm_keys])
    cm_mat_ev = np.column_stack([cm_ev[k] for k in cm_keys])

    svd_cm = TruncatedSVD(n_components=3, random_state=42)
    cm_latent_tr = svd_cm.fit_transform(cm_mat_tr)
    cm_latent_ev = svd_cm.transform(cm_mat_ev)

    cm_svd_dict_tr = {f'cm_svd_{k}': cm_latent_tr[:, k].astype(np.float64) for k in range(3)}
    cm_svd_dict_ev = {f'cm_svd_{k}': cm_latent_ev[:, k].astype(np.float64) for k in range(3)}

    # 3. ICD-9 Chapter groups and 3-digit category prefix extraction
    grp1_tr = np.array([icd9_group(c) for c in train['diag_1']], dtype=object)
    grp1_ev = np.array([icd9_group(c) for c in eval_cols['diag_1']], dtype=object)
    grp2_tr = np.array([icd9_group(c) for c in train['diag_2']], dtype=object)
    grp2_ev = np.array([icd9_group(c) for c in eval_cols['diag_2']], dtype=object)
    grp3_tr = np.array([icd9_group(c) for c in train['diag_3']], dtype=object)
    grp3_ev = np.array([icd9_group(c) for c in eval_cols['diag_3']], dtype=object)

    def extract_icd9_prefix(diag_arr):
        n = len(diag_arr)
        out = np.empty(n, dtype=object)
        for i, val in enumerate(diag_arr):
            if val is None or val == '?' or val == 'None':
                out[i] = 'None'
            else:
                s = str(val).strip()
                dot_idx = s.find('.')
                if dot_idx != -1:
                    out[i] = s[:dot_idx]
                else:
                    out[i] = s[:3] if len(s) > 3 else s
        return out

    pfx1_tr = extract_icd9_prefix(train['diag_1'])
    pfx1_ev = extract_icd9_prefix(eval_cols['diag_1'])
    pfx2_tr = extract_icd9_prefix(train['diag_2'])
    pfx2_ev = extract_icd9_prefix(eval_cols['diag_2'])
    pfx3_tr = extract_icd9_prefix(train['diag_3'])
    pfx3_ev = extract_icd9_prefix(eval_cols['diag_3'])

    # 4. Out-of-fold target encodings on high-cardinality codes, 3-digit prefixes & interactions
    te_diag1_tr, te_diag1_ev = oof_target_encode(train['diag_1'], y_train, eval_cols['diag_1'], smoothing=20.0, seed=42)
    te_diag2_tr, te_diag2_ev = oof_target_encode(train['diag_2'], y_train, eval_cols['diag_2'], smoothing=25.0, seed=42)
    te_diag3_tr, te_diag3_ev = oof_target_encode(train['diag_3'], y_train, eval_cols['diag_3'], smoothing=25.0, seed=42)

    te_pfx1_tr, te_pfx1_ev = oof_target_encode(pfx1_tr, y_train, pfx1_ev, smoothing=20.0, seed=42)
    te_pfx2_tr, te_pfx2_ev = oof_target_encode(pfx2_tr, y_train, pfx2_ev, smoothing=20.0, seed=42)
    te_pfx3_tr, te_pfx3_ev = oof_target_encode(pfx3_tr, y_train, pfx3_ev, smoothing=20.0, seed=42)

    te_disch_tr, te_disch_ev = oof_target_encode(train['discharge_disposition_id'], y_train, eval_cols['discharge_disposition_id'], smoothing=15.0, seed=42)
    te_admit_tr, te_admit_ev = oof_target_encode(train['admission_source_id'], y_train, eval_cols['admission_source_id'], smoothing=15.0, seed=42)
    te_admtype_tr, te_admtype_ev = oof_target_encode(train['admission_type_id'], y_train, eval_cols['admission_type_id'], smoothing=15.0, seed=42)
    te_spec_tr, te_spec_ev = oof_target_encode(train['medical_specialty'], y_train, eval_cols['medical_specialty'], smoothing=25.0, seed=42)
    te_payer_tr, te_payer_ev = oof_target_encode(train['payer_code'], y_train, eval_cols['payer_code'], smoothing=20.0, seed=42)

    # Encounter flow interactions:
    da_tr = np.array([f"{d}_{s}" for d, s in zip(train['discharge_disposition_id'], train['admission_source_id'])], dtype=object)
    da_ev = np.array([f"{d}_{s}" for d, s in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_source_id'])], dtype=object)
    te_da_tr, te_da_ev = oof_target_encode(da_tr, y_train, da_ev, smoothing=20.0, seed=42)

    dt_tr = np.array([f"{d}_{t}" for d, t in zip(train['discharge_disposition_id'], train['admission_type_id'])], dtype=object)
    dt_ev = np.array([f"{d}_{t}" for d, t in zip(eval_cols['discharge_disposition_id'], eval_cols['admission_type_id'])], dtype=object)
    te_dt_tr, te_dt_ev = oof_target_encode(dt_tr, y_train, dt_ev, smoothing=20.0, seed=42)

    gd_tr = np.array([f"{g}_{d}" for g, d in zip(grp1_tr, train['discharge_disposition_id'])], dtype=object)
    gd_ev = np.array([f"{g}_{d}" for g, d in zip(grp1_ev, eval_cols['discharge_disposition_id'])], dtype=object)
    te_gd_tr, te_gd_ev = oof_target_encode(gd_tr, y_train, gd_ev, smoothing=20.0, seed=42)

    pfx_disch_tr = np.array([f"{p}_{d}" for p, d in zip(pfx1_tr, train['discharge_disposition_id'])], dtype=object)
    pfx_disch_ev = np.array([f"{p}_{d}" for p, d in zip(pfx1_ev, eval_cols['discharge_disposition_id'])], dtype=object)
    te_pfx_disch_tr, te_pfx_disch_ev = oof_target_encode(pfx_disch_tr, y_train, pfx_disch_ev, smoothing=20.0, seed=42)

    age_disch_tr = np.array([f"{a}_{d}" for a, d in zip(train['age'], train['discharge_disposition_id'])], dtype=object)
    age_disch_ev = np.array([f"{a}_{d}" for a, d in zip(eval_cols['age'], eval_cols['discharge_disposition_id'])], dtype=object)
    te_age_disch_tr, te_age_disch_ev = oof_target_encode(age_disch_tr, y_train, age_disch_ev, smoothing=20.0, seed=42)

    g12_tr = np.array([f"{g1}_{g2}" for g1, g2 in zip(grp1_tr, grp2_tr)], dtype=object)
    g12_ev = np.array([f"{g1}_{g2}" for g1, g2 in zip(grp1_ev, grp2_ev)], dtype=object)
    te_g12_tr, te_g12_ev = oof_target_encode(g12_tr, y_train, g12_ev, smoothing=20.0, seed=42)

    # 5. Frequency (count) encodings on high-cardinality encounter attributes
    def compute_freq_encoding(tr_arr, ev_arr):
        counts = {}
        for x in tr_arr:
            counts[x] = counts.get(x, 0) + 1
        n = float(len(tr_arr))
        tr_f = np.array([counts.get(x, 0) / n for x in tr_arr], dtype=np.float64)
        ev_f = np.array([counts.get(x, 0) / n for x in ev_arr], dtype=np.float64)
        return tr_f, ev_f

    freq_diag1_tr, freq_diag1_ev = compute_freq_encoding(train['diag_1'], eval_cols['diag_1'])
    freq_pfx1_tr, freq_pfx1_ev = compute_freq_encoding(pfx1_tr, pfx1_ev)
    freq_spec_tr, freq_spec_ev = compute_freq_encoding(train['medical_specialty'], eval_cols['medical_specialty'])
    freq_disch_tr, freq_disch_ev = compute_freq_encoding(train['discharge_disposition_id'], eval_cols['discharge_disposition_id'])

    # 6. Low-rank decomposition (TruncatedSVD) of the 23-medication therapeutic regimen matrix
    med_names = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]

    def build_med_matrix(data):
        n = len(data['time_in_hospital'])
        mat = np.zeros((n, len(med_names)), dtype=np.float64)
        for j, col in enumerate(med_names):
            if col in data:
                arr = data[col]
                mat[arr == 'Steady', j] = 1.0
                mat[arr == 'Up', j] = 2.0
                mat[arr == 'Down', j] = -1.0
        return mat

    med_mat_tr = build_med_matrix(train)
    med_mat_ev = build_med_matrix(eval_cols)

    svd_med = TruncatedSVD(n_components=4, random_state=42)
    med_latent_tr = svd_med.fit_transform(med_mat_tr)
    med_latent_ev = svd_med.transform(med_mat_ev)

    med_svd_dict_tr = {f'med_svd_{k}': med_latent_tr[:, k].astype(np.float64) for k in range(4)}
    med_svd_dict_ev = {f'med_svd_{k}': med_latent_ev[:, k].astype(np.float64) for k in range(4)}

    # 7. Clinical condition flags across diagnoses
    def extract_clinical_flags(data, p1, p2, p3):
        n = len(data['time_in_hospital'])

        def check_condition(prefix_set):
            res = np.zeros(n, dtype=np.float64)
            for pfx in (p1, p2, p3):
                for i, v in enumerate(pfx):
                    if v in prefix_set:
                        res[i] = 1.0
            return res

        has_chf = check_condition({'428'})
        has_ckd = check_condition({'585', '586'})
        has_aki = check_condition({'584'})
        has_cad_ami = check_condition({'410', '411', '412', '413', '414'})
        has_dysrhythmia = check_condition({'427'})
        has_htn = check_condition({'401', '402', '403', '404', '405'})
        has_copd = check_condition({'490', '491', '492', '493', '494', '495', '496'})
        has_pna = check_condition({'480', '481', '482', '483', '484', '485', '486'})
        has_fluid_lyte = check_condition({'276'})
        has_uti = check_condition({'599'})
        has_cellulitis = check_condition({'682'})
        has_dm = check_condition({'250'})

        primary_chf = (p1 == '428').astype(np.float64)
        primary_cad = np.isin(p1, ['410', '411', '412', '413', '414']).astype(np.float64)
        primary_dm = (p1 == '250').astype(np.float64)
        primary_resp = np.isin(p1, ['480', '481', '482', '483', '484', '485', '486', '490', '491', '492', '493', '494', '496']).astype(np.float64)

        chf_renal = ((has_chf > 0.5) & ((has_ckd > 0.5) | (has_aki > 0.5))).astype(np.float64)
        chf_lyte = ((has_chf > 0.5) & (has_fluid_lyte > 0.5)).astype(np.float64)
        dm_renal = ((has_dm > 0.5) & (has_ckd > 0.5)).astype(np.float64)

        return {
            'clin_has_chf': has_chf,
            'clin_has_ckd': has_ckd,
            'clin_has_aki': has_aki,
            'clin_has_cad_ami': has_cad_ami,
            'clin_has_dysrhythmia': has_dysrhythmia,
            'clin_has_htn': has_htn,
            'clin_has_copd': has_copd,
            'clin_has_pna': has_pna,
            'clin_has_fluid_lyte': has_fluid_lyte,
            'clin_has_uti': has_uti,
            'clin_has_cellulitis': has_cellulitis,
            'clin_primary_chf': primary_chf,
            'clin_primary_cad': primary_cad,
            'clin_primary_dm': primary_dm,
            'clin_primary_resp': primary_resp,
            'clin_chf_renal': chf_renal,
            'clin_chf_lyte': chf_lyte,
            'clin_dm_renal': dm_renal,
        }

    clin_tr = extract_clinical_flags(train, pfx1_tr, pfx2_tr, pfx3_tr)
    clin_ev = extract_clinical_flags(eval_cols, pfx1_ev, pfx2_ev, pfx3_ev)

    # 8. Medication class features and therapy combinations
    def extract_med_class_features(data):
        n = len(data['time_in_hospital'])
        su_meds = ['glimepiride', 'glipizide', 'glyburide', 'chlorpropamide', 'tolbutamide', 'acetohexamide', 'tolazamide']
        tzd_meds = ['pioglitazone', 'rosiglitazone', 'troglitazone']
        met_meds = ['metformin', 'glyburide-metformin', 'glipizide-metformin', 'metformin-rosiglitazone', 'metformin-pioglitazone']

        on_su = np.zeros(n, dtype=np.float64)
        for m in su_meds:
            if m in data:
                on_su = np.maximum(on_su, (data[m] != 'No').astype(np.float64))

        on_tzd = np.zeros(n, dtype=np.float64)
        for m in tzd_meds:
            if m in data:
                on_tzd = np.maximum(on_tzd, (data[m] != 'No').astype(np.float64))

        on_met = np.zeros(n, dtype=np.float64)
        for m in met_meds:
            if m in data:
                on_met = np.maximum(on_met, (data[m] != 'No').astype(np.float64))

        ins = data.get('insulin')
        if ins is not None:
            on_ins = (ins != 'No').astype(np.float64)
        else:
            on_ins = np.zeros(n, dtype=np.float64)

        num_classes = on_su + on_tzd + on_met + on_ins
        ins_plus_oral = ((on_ins > 0.5) & ((on_su + on_tzd + on_met) > 0.5)).astype(np.float64)
        ins_su_hypo_risk = ((on_ins > 0.5) & (on_su > 0.5)).astype(np.float64)
        oral_only = ((on_ins < 0.5) & ((on_su + on_tzd + on_met) > 0.5)).astype(np.float64)

        return {
            'med_on_su': on_su,
            'med_on_tzd': on_tzd,
            'med_on_met': on_met,
            'med_num_classes': num_classes,
            'med_ins_plus_oral': ins_plus_oral,
            'med_ins_su_hypo_risk': ins_su_hypo_risk,
            'med_oral_only': oral_only,
        }

    med_cls_tr = extract_med_class_features(train)
    med_cls_ev = extract_med_class_features(eval_cols)

    # 9. Encounter flow, discharge destination, and missingness signals
    def extract_flow_features(data):
        disch = data['discharge_disposition_id']
        admit_src = data['admission_source_id']
        admit_typ = data['admission_type_id']
        n_inp = np.nan_to_num(data['number_inpatient'], nan=0.0)

        disch_home = (disch == '1').astype(np.float64)
        disch_snf = (disch == '3').astype(np.float64)
        disch_home_health = (disch == '6').astype(np.float64)
        disch_ama = (disch == '7').astype(np.float64)
        disch_facility = np.isin(disch, ['2', '3', '4', '5', '22']).astype(np.float64)
        disch_post_acute = np.isin(disch, ['3', '4', '5', '6', '22']).astype(np.float64)

        admit_er = ((admit_src == '7') | (admit_typ == '1')).astype(np.float64)
        admit_elective = (admit_typ == '3').astype(np.float64)
        admit_transfer = np.isin(admit_src, ['4', '5', '6']).astype(np.float64)

        disch_facility_with_inp = disch_facility * (n_inp > 0).astype(np.float64)
        disch_post_acute_with_inp = disch_post_acute * (n_inp > 0).astype(np.float64)
        disch_home_no_inp = disch_home * (n_inp == 0).astype(np.float64)
        admit_er_with_inp = admit_er * (n_inp > 0).astype(np.float64)

        missing_spec = np.isin(data.get('medical_specialty', ''), [None, '?', 'None']).astype(np.float64)
        missing_payer = np.isin(data.get('payer_code', ''), [None, '?', 'None']).astype(np.float64)
        has_weight = (~np.isin(data.get('weight', ''), [None, '?', 'None'])).astype(np.float64)

        return {
            'flow_disch_home': disch_home,
            'flow_disch_snf': disch_snf,
            'flow_disch_home_health': disch_home_health,
            'flow_disch_ama': disch_ama,
            'flow_disch_facility': disch_facility,
            'flow_disch_post_acute': disch_post_acute,
            'flow_admit_er': admit_er,
            'flow_admit_elective': admit_elective,
            'flow_admit_transfer': admit_transfer,
            'flow_disch_facility_inp': disch_facility_with_inp,
            'flow_disch_post_acute_inp': disch_post_acute_with_inp,
            'flow_disch_home_no_inp': disch_home_no_inp,
            'flow_admit_er_inp': admit_er_with_inp,
            'flag_missing_spec': missing_spec,
            'flag_missing_payer': missing_payer,
            'flag_has_weight': has_weight,
        }

    flow_tr = extract_flow_features(train)
    flow_ev = extract_flow_features(eval_cols)

    # 10. Clinical domain velocities, utilization, and discordance features
    def extract_domain_features(data, cm):
        los = np.maximum(np.nan_to_num(data['time_in_hospital'], nan=1.0), 0.5)
        n_labs = np.nan_to_num(data['num_lab_procedures'], nan=0.0)
        n_procs = np.nan_to_num(data['num_procedures'], nan=0.0)
        n_meds = np.nan_to_num(data['num_medications'], nan=0.0)
        n_diags = np.nan_to_num(data['number_diagnoses'], nan=0.0)

        n_inp = np.nan_to_num(data['number_inpatient'], nan=0.0)
        n_emg = np.nan_to_num(data['number_emergency'], nan=0.0)
        n_out = np.nan_to_num(data['number_outpatient'], nan=0.0)

        labs_per_day = n_labs / los
        procs_per_day = n_procs / los
        meds_per_day = n_meds / los
        diags_per_day = n_diags / los
        total_events = n_labs + n_procs + n_meds
        events_per_day = total_events / los
        med_per_diag = n_meds / np.maximum(n_diags, 1.0)
        lab_proc_ratio = n_labs / np.maximum(n_procs, 1.0)

        total_prior = n_inp + n_emg + n_out
        log_total_prior = np.log1p(total_prior)
        log_inpatient = np.log1p(n_inp)
        log_emergency = np.log1p(n_emg)
        weighted_util = 3.0 * n_inp + 1.5 * n_emg + 0.5 * n_out
        inpatient_ratio = n_inp / (total_prior + 1.0)
        acute_util_ratio = (n_inp + n_emg) / (total_prior + 1.0)
        has_prior_inp = (n_inp > 0).astype(np.float64)
        frequent_inpatient = (n_inp >= 2).astype(np.float64)
        high_inpatient = (n_inp >= 3).astype(np.float64)
        zero_prior_visits = (total_prior == 0).astype(np.float64)
        emergency_ratio = n_emg / (total_prior + 1.0)
        inp_squared = (n_inp ** 2).astype(np.float64)
        inp_per_day = n_inp / los

        cm_count = cm['cm_count']
        discharge_acuity_gap = (n_diags + 2.0 * cm_count) / los
        premature_discharge = ((los <= 2.0) & ((n_diags >= 8.0) | (cm_count >= 2.0))).astype(np.float64)

        age_map = {
            '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
            '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
            '[80-90)': 85.0, '[90-100)': 95.0
        }
        age_col = data['age']
        age_num = np.array([age_map.get(a, 65.0) for a in age_col], dtype=np.float64)
        age_x_inpatient = age_num * n_inp
        age_x_cm = age_num * cm_count
        is_elderly = (age_num >= 75.0).astype(np.float64)
        elderly_prior_inp = ((age_num >= 75.0) & (n_inp > 0)).astype(np.float64)

        n_rows = len(los)
        num_active_meds = np.zeros(n_rows, dtype=np.float64)
        num_up_meds = np.zeros(n_rows, dtype=np.float64)
        num_down_meds = np.zeros(n_rows, dtype=np.float64)

        for m in med_names:
            if m in data:
                arr = data[m]
                num_active_meds += (arr != 'No').astype(np.float64)
                num_up_meds += (arr == 'Up').astype(np.float64)
                num_down_meds += (arr == 'Down').astype(np.float64)

        num_changed_meds = num_up_meds + num_down_meds
        polypharmacy = (num_active_meds >= 3).astype(np.float64)
        med_dose_net_change = num_up_meds - num_down_meds
        med_intensified = (num_up_meds > 0).astype(np.float64)

        ins = data.get('insulin')
        if ins is not None:
            insulin_is_up = (ins == 'Up').astype(np.float64)
            insulin_is_down = (ins == 'Down').astype(np.float64)
            insulin_is_steady = (ins == 'Steady').astype(np.float64)
            insulin_is_active = (ins != 'No').astype(np.float64)
            insulin_changed = ((ins == 'Up') | (ins == 'Down')).astype(np.float64)
        else:
            insulin_is_up = np.zeros(n_rows, dtype=np.float64)
            insulin_is_down = np.zeros(n_rows, dtype=np.float64)
            insulin_is_steady = np.zeros(n_rows, dtype=np.float64)
            insulin_is_active = np.zeros(n_rows, dtype=np.float64)
            insulin_changed = np.zeros(n_rows, dtype=np.float64)

        change_flag = (data.get('change') == 'Ch').astype(np.float64)
        dm_med_flag = (data.get('diabetesMed') == 'Yes').astype(np.float64)

        a1c = data.get('A1Cresult')
        if a1c is not None:
            a1c_tested = (a1c != 'None').astype(np.float64)
            a1c_high = ((a1c == '>8') | (a1c == '>7')).astype(np.float64)
            a1c_norm = (a1c == 'Norm').astype(np.float64)
        else:
            a1c_tested = np.zeros(n_rows, dtype=np.float64)
            a1c_high = np.zeros(n_rows, dtype=np.float64)
            a1c_norm = np.zeros(n_rows, dtype=np.float64)

        glu = data.get('max_glu_serum')
        if glu is not None:
            glu_tested = (glu != 'None').astype(np.float64)
            glu_high = ((glu == '>300') | (glu == '>200')).astype(np.float64)
        else:
            glu_tested = np.zeros(n_rows, dtype=np.float64)
            glu_high = np.zeros(n_rows, dtype=np.float64)

        unmanaged_hyperglycemia = ((a1c_high > 0.5) & (change_flag < 0.5)).astype(np.float64)

        return {
            'labs_per_day': labs_per_day,
            'procs_per_day': procs_per_day,
            'meds_per_day': meds_per_day,
            'diags_per_day': diags_per_day,
            'total_events': total_events,
            'events_per_day': events_per_day,
            'med_per_diag': med_per_diag,
            'lab_proc_ratio': lab_proc_ratio,
            'total_prior': total_prior,
            'log_total_prior': log_total_prior,
            'log_inpatient': log_inpatient,
            'log_emergency': log_emergency,
            'weighted_util': weighted_util,
            'inpatient_ratio': inpatient_ratio,
            'acute_util_ratio': acute_util_ratio,
            'has_prior_inp': has_prior_inp,
            'frequent_inpatient': frequent_inpatient,
            'high_inpatient': high_inpatient,
            'zero_prior_visits': zero_prior_visits,
            'emergency_ratio': emergency_ratio,
            'inp_squared': inp_squared,
            'inp_per_day': inp_per_day,
            'discharge_acuity_gap': discharge_acuity_gap,
            'premature_discharge': premature_discharge,
            'age_num': age_num,
            'age_x_inpatient': age_x_inpatient,
            'age_x_cm': age_x_cm,
            'is_elderly': is_elderly,
            'elderly_prior_inp': elderly_prior_inp,
            'num_active_meds': num_active_meds,
            'num_up_meds': num_up_meds,
            'num_down_meds': num_down_meds,
            'num_changed_meds': num_changed_meds,
            'polypharmacy': polypharmacy,
            'med_dose_net_change': med_dose_net_change,
            'med_intensified': med_intensified,
            'insulin_is_up': insulin_is_up,
            'insulin_is_down': insulin_is_down,
            'insulin_is_steady': insulin_is_steady,
            'insulin_is_active': insulin_is_active,
            'insulin_changed': insulin_changed,
            'change_flag': change_flag,
            'dm_med_flag': dm_med_flag,
            'a1c_tested': a1c_tested,
            'a1c_high': a1c_high,
            'a1c_norm': a1c_norm,
            'glu_tested': glu_tested,
            'glu_high': glu_high,
            'unmanaged_hyperglycemia': unmanaged_hyperglycemia,
        }

    dom_tr = extract_domain_features(train, cm_tr)
    dom_ev = extract_domain_features(eval_cols, cm_ev)

    # 11. Populate feature dictionaries
    train_dict = dict(train)
    eval_dict = dict(eval_cols)

    for k, v in cm_tr.items():
        train_dict[k] = v
        eval_dict[k] = cm_ev[k]

    for k, v in cm_synd_tr.items():
        train_dict[k] = v
        eval_dict[k] = cm_synd_ev[k]

    for k, v in cm_svd_dict_tr.items():
        train_dict[k] = v
        eval_dict[k] = cm_svd_dict_ev[k]

    for k, v in med_svd_dict_tr.items():
        train_dict[k] = v
        eval_dict[k] = med_svd_dict_ev[k]

    for k, v in clin_tr.items():
        train_dict[k] = v
        eval_dict[k] = clin_ev[k]

    for k, v in med_cls_tr.items():
        train_dict[k] = v
        eval_dict[k] = med_cls_ev[k]

    for k, v in flow_tr.items():
        train_dict[k] = v
        eval_dict[k] = flow_ev[k]

    for k, v in dom_tr.items():
        train_dict[k] = v
        eval_dict[k] = dom_ev[k]

    te_features = {
        'te_diag1': (te_diag1_tr, te_diag1_ev),
        'te_diag2': (te_diag2_tr, te_diag2_ev),
        'te_diag3': (te_diag3_tr, te_diag3_ev),
        'te_pfx1': (te_pfx1_tr, te_pfx1_ev),
        'te_pfx2': (te_pfx2_tr, te_pfx2_ev),
        'te_pfx3': (te_pfx3_tr, te_pfx3_ev),
        'te_discharge': (te_disch_tr, te_disch_ev),
        'te_admit': (te_admit_tr, te_admit_ev),
        'te_admtype': (te_admtype_tr, te_admtype_ev),
        'te_spec': (te_spec_tr, te_spec_ev),
        'te_payer': (te_payer_tr, te_payer_ev),
        'te_da': (te_da_tr, te_da_ev),
        'te_dt': (te_dt_tr, te_dt_ev),
        'te_gd': (te_gd_tr, te_gd_ev),
        'te_pfx_disch': (te_pfx_disch_tr, te_pfx_disch_ev),
        'te_age_disch': (te_age_disch_tr, te_age_disch_ev),
        'te_g12': (te_g12_tr, te_g12_ev),
    }
    for k, (v_tr, v_ev) in te_features.items():
        train_dict[k] = v_tr
        eval_dict[k] = v_ev

    freq_features = {
        'freq_diag1': (freq_diag1_tr, freq_diag1_ev),
        'freq_pfx1': (freq_pfx1_tr, freq_pfx1_ev),
        'freq_spec': (freq_spec_tr, freq_spec_ev),
        'freq_disch': (freq_disch_tr, freq_disch_ev),
    }
    for k, (v_tr, v_ev) in freq_features.items():
        train_dict[k] = v_tr
        eval_dict[k] = v_ev

    train_dict['diag_1_group'] = grp1_tr
    eval_dict['diag_1_group'] = grp1_ev
    train_dict['diag_2_group'] = grp2_tr
    eval_dict['diag_2_group'] = grp2_ev
    train_dict['diag_3_group'] = grp3_tr
    eval_dict['diag_3_group'] = grp3_ev

    train_dict['diag_1_prefix'] = pfx1_tr
    eval_dict['diag_1_prefix'] = pfx1_ev
    train_dict['diag_2_prefix'] = pfx2_tr
    eval_dict['diag_2_prefix'] = pfx2_ev
    train_dict['diag_3_prefix'] = pfx3_tr
    eval_dict['diag_3_prefix'] = pfx3_ev

    extra_numeric = tuple(
        list(cm_tr.keys()) +
        list(cm_synd_tr.keys()) +
        list(cm_svd_dict_tr.keys()) +
        list(med_svd_dict_tr.keys()) +
        list(clin_tr.keys()) +
        list(med_cls_tr.keys()) +
        list(flow_tr.keys()) +
        list(dom_tr.keys()) +
        list(te_features.keys()) +
        list(freq_features.keys())
    )
    extra_categorical = (
        'diag_1_group', 'diag_2_group', 'diag_3_group',
        'diag_1_prefix', 'diag_2_prefix', 'diag_3_prefix'
    )

    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_dict, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_dict)
    x_eval = encoder.transform(eval_dict)

    cat_idx = encoder.categorical_indices

    # 12. Heterogeneous Ensemble Architecture with Shallow Anchor
    # Engine A: Deep interaction specialist
    model_deep = lgb.LGBMClassifier(
        n_estimators=210,
        learning_rate=0.018,
        num_leaves=28,
        max_depth=6,
        min_child_samples=80,
        subsample=0.72,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_alpha=0.2,
        reg_lambda=1.2,
        cat_smooth=40.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model_deep.fit(x_train, y_train, categorical_feature=cat_idx)
    p_deep = model_deep.predict_proba(x_eval)[:, 1]

    # Engine B: Broad robust shallow generalizer
    model_shallow = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.020,
        num_leaves=15,
        max_depth=4,
        min_child_samples=160,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.3,
        reg_lambda=2.0,
        cat_smooth=70.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    model_shallow.fit(x_train, y_train, categorical_feature=cat_idx)
    p_shallow = model_shallow.predict_proba(x_eval)[:, 1]

    # Engine C: Readmission boundary margin-weighted learner
    model_margin = lgb.LGBMClassifier(
        n_estimators=190,
        learning_rate=0.020,
        num_leaves=22,
        max_depth=5,
        min_child_samples=110,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.55,
        scale_pos_weight=1.30,
        reg_alpha=0.2,
        reg_lambda=1.0,
        cat_smooth=60.0,
        random_state=999,
        n_jobs=1,
        verbose=-1,
    )
    model_margin.fit(x_train, y_train, categorical_feature=cat_idx)
    p_margin = model_margin.predict_proba(x_eval)[:, 1]

    # Engine D: Extremely Randomized Trees booster (structural diversity)
    model_extra = lgb.LGBMClassifier(
        n_estimators=190,
        learning_rate=0.022,
        num_leaves=26,
        max_depth=6,
        min_child_samples=85,
        extra_trees=True,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_alpha=0.2,
        reg_lambda=1.5,
        cat_smooth=50.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    model_extra.fit(x_train, y_train, categorical_feature=cat_idx)
    p_extra = model_extra.predict_proba(x_eval)[:, 1]

    # Engine E: Low-depth regularized anchor learner (stabilizes interaction noise)
    model_anchor = lgb.LGBMClassifier(
        n_estimators=220,
        learning_rate=0.022,
        num_leaves=10,
        max_depth=3,
        min_child_samples=200,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_alpha=0.5,
        reg_lambda=3.0,
        cat_smooth=90.0,
        random_state=777,
        n_jobs=1,
        verbose=-1,
    )
    model_anchor.fit(x_train, y_train, categorical_feature=cat_idx)
    p_anchor = model_anchor.predict_proba(x_eval)[:, 1]

    # Engine F: XGBoost histogram booster
    p_xgb = None
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            n_estimators=140,
            max_depth=4,
            learning_rate=0.026,
            subsample=0.78,
            colsample_bytree=0.60,
            reg_alpha=0.2,
            reg_lambda=2.5,
            min_child_weight=7.0,
            tree_method='hist',
            random_state=42,
            n_jobs=1,
            eval_metric='logloss',
        )
        xgb_model.fit(x_train, y_train)
        p_xgb = xgb_model.predict_proba(x_eval)[:, 1]
    except Exception:
        p_xgb = None

    # 13. Latent Logit Standardization and Recalibrated Blending
    def logit(p):
        p_c = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(p_c / (1.0 - p_c))

    def standardize(z):
        s = np.std(z)
        if s < 1e-7:
            return z - np.mean(z)
        return (z - np.mean(z)) / s

    if p_xgb is not None:
        weights = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
        preds = [p_deep, p_shallow, p_margin, p_extra, p_anchor, p_xgb]
    else:
        weights = [0.28, 0.22, 0.17, 0.17, 0.16]
        preds = [p_deep, p_shallow, p_margin, p_extra, p_anchor]

    std_logits = [standardize(logit(p)) for p in preds]

    ensemble_latent = np.zeros_like(std_logits[0])
    for w, z in zip(weights, std_logits):
        ensemble_latent += w * z

    # Empirical calibration: center at prior logit (~0.114) with natural risk dispersion
    prior_logit = np.log(0.114 / (1.0 - 0.114))
    calibrated_logit = prior_logit + 0.55 * standardize(ensemble_latent)
    return 1.0 / (1.0 + np.exp(-np.clip(calibrated_logit, -20.0, 20.0)))


# EVOLVE-BLOCK-END
# =====================================================================
