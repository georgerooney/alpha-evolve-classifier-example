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
    from sklearn.model_selection import StratifiedKFold
    
    # 1. Initialize lists for new features
    extra_numeric = []
    extra_categorical = []
    
    # Make shallow copies to avoid modifying caller's dict keys
    train = dict(train)
    eval_cols = dict(eval_cols)
    
    # 2. Extract comorbidities using icd9_comorbidities
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    train_diags = [train[col] for col in diag_cols if col in train]
    eval_diags = [eval_cols[col] for col in diag_cols if col in eval_cols]
    
    train_cm = icd9_comorbidities(*train_diags)
    eval_cm = icd9_comorbidities(*eval_diags)
    
    for k in train_cm:
        train[k] = train_cm[k]
        eval_cols[k] = eval_cm[k]
        extra_numeric.append(k)
        
    # 3. Grouped ICD-9 chapters
    for col in diag_cols:
        grouped_col = col + "_grp"
        train[grouped_col] = np.array([icd9_group(x) for x in train[col]], dtype=object)
        eval_cols[grouped_col] = np.array([icd9_group(x) for x in eval_cols[col]], dtype=object)
        extra_categorical.append(grouped_col)
        
    # 4. Target encoding of high-cardinality columns
    te_cols = ['diag_1', 'diag_2', 'diag_3', 'medical_specialty', 'discharge_disposition_id', 'admission_source_id', 'payer_code']
    for col in te_cols:
        if col in train and col in eval_cols:
            tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42)
            te_col = col + "_te"
            train[te_col] = tr_enc
            eval_cols[te_col] = ev_enc
            extra_numeric.append(te_col)
            
    # 5. Utilisation intensity features
    train_total_visits = train['number_outpatient'] + train['number_emergency'] + train['number_inpatient']
    eval_total_visits = eval_cols['number_outpatient'] + eval_cols['number_emergency'] + eval_cols['number_inpatient']
    train['total_visits'] = train_total_visits
    eval_cols['total_visits'] = eval_total_visits
    extra_numeric.append('total_visits')
    
    train['inpatient_ratio'] = train['number_inpatient'] / (train_total_visits + 1.0)
    eval_cols['inpatient_ratio'] = eval_cols['number_inpatient'] / (eval_total_visits + 1.0)
    extra_numeric.append('inpatient_ratio')
    
    # 6. Medication active and change counts
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    train_active_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_active_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    train_change_meds = np.zeros(len(y_train), dtype=np.float64)
    eval_change_meds = np.zeros(len(eval_cols['insulin']), dtype=np.float64)
    
    for col in med_cols:
        if col in train:
            train_active_meds += (train[col] != "No") & (train[col] != None)
            eval_active_meds += (eval_cols[col] != "No") & (eval_cols[col] != None)
            train_change_meds += (train[col] == "Up") | (train[col] == "Down")
            eval_change_meds += (eval_cols[col] == "Up") | (eval_cols[col] == "Down")
            
    train['active_meds_count'] = train_active_meds
    eval_cols['active_meds_count'] = eval_active_meds
    extra_numeric.append('active_meds_count')
    
    train['change_meds_count'] = train_change_meds
    eval_cols['change_meds_count'] = eval_change_meds
    extra_numeric.append('change_meds_count')
    
    # 7. Insulin change interaction
    train['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(train['insulin'], train['change'])], dtype=object)
    eval_cols['insulin_change'] = np.array([f"{ins}_{chg}" for ins, chg in zip(eval_cols['insulin'], eval_cols['change'])], dtype=object)
    extra_categorical.append('insulin_change')
    
    # 8. Numeric age representation
    def age_to_numeric(age_str):
        if age_str is None:
            return np.nan
        try:
            cleaned = age_str.replace('[', '').replace(')', '').replace(']', '')
            parts = cleaned.split('-')
            return (float(parts[0]) + float(parts[1])) / 2.0
        except Exception:
            return np.nan

    train['age_num'] = np.array([age_to_numeric(x) for x in train['age']], dtype=np.float64)
    eval_cols['age_num'] = np.array([age_to_numeric(x) for x in eval_cols['age']], dtype=np.float64)
    extra_numeric.append('age_num')

    # 9. Comorbidity weighted Charlson proxy and syndromic interactions
    train_charlson = (
        train_cm['cm_mi'] + train_cm['cm_chf'] + train_cm['cm_pvd'] +
        train_cm['cm_cerebrovascular'] + train_cm['cm_dementia'] +
        train_cm['cm_pulmonary'] + train_cm['cm_rheumatic'] +
        train_cm['cm_peptic_ulcer'] + train_cm['cm_liver'] +
        2.0 * (train_cm['cm_hemiplegia'] + train_cm['cm_renal'] +
               train_cm['cm_diabetes_complicated'] + train_cm['cm_cancer']) +
        4.0 * (train_cm['cm_metastatic'] + train_cm['cm_hiv'])
    )
    eval_charlson = (
        eval_cm['cm_mi'] + eval_cm['cm_chf'] + eval_cm['cm_pvd'] +
        eval_cm['cm_cerebrovascular'] + eval_cm['cm_dementia'] +
        eval_cm['cm_pulmonary'] + eval_cm['cm_rheumatic'] +
        eval_cm['cm_peptic_ulcer'] + eval_cm['cm_liver'] +
        2.0 * (eval_cm['cm_hemiplegia'] + eval_cm['cm_renal'] +
               eval_cm['cm_diabetes_complicated'] + eval_cm['cm_cancer']) +
        4.0 * (eval_cm['cm_metastatic'] + eval_cm['cm_hiv'])
    )
    train['charlson_proxy'] = train_charlson
    eval_cols['charlson_proxy'] = eval_charlson
    extra_numeric.append('charlson_proxy')

    for d, cm in [(train, train_cm), (eval_cols, eval_cm)]:
        d['cm_chf_renal'] = cm['cm_chf'] * cm['cm_renal']
        d['cm_chf_dm'] = cm['cm_chf'] * cm['cm_diabetes_complicated']
        d['cm_renal_dm'] = cm['cm_renal'] * cm['cm_diabetes_complicated']
        d['cm_pulmonary_chf'] = cm['cm_pulmonary'] * cm['cm_chf']
    extra_numeric.extend(['cm_chf_renal', 'cm_chf_dm', 'cm_renal_dm', 'cm_pulmonary_chf'])

    # 10. Prior utilization details & acute care ratios
    for d in (train, eval_cols):
        tot = d['number_outpatient'] + d['number_emergency'] + d['number_inpatient']
        d['has_inpatient'] = (d['number_inpatient'] > 0).astype(np.float64)
        d['has_emergency'] = (d['number_emergency'] > 0).astype(np.float64)
        d['has_outpatient'] = (d['number_outpatient'] > 0).astype(np.float64)
        d['frequent_inpatient'] = (d['number_inpatient'] >= 2).astype(np.float64)
        d['emergency_ratio'] = d['number_emergency'] / (tot + 1.0)
        d['acute_ratio'] = (d['number_inpatient'] + d['number_emergency']) / (tot + 1.0)
        d['weighted_prior_visits'] = 3.0 * d['number_inpatient'] + 2.0 * d['number_emergency'] + 1.0 * d['number_outpatient']
        d['log_inpatient'] = np.log1p(np.maximum(0.0, d['number_inpatient']))
        d['log_emergency'] = np.log1p(np.maximum(0.0, d['number_emergency']))
        d['log_total_visits'] = np.log1p(np.maximum(0.0, tot))
    extra_numeric.extend([
        'has_inpatient', 'has_emergency', 'has_outpatient', 'frequent_inpatient',
        'emergency_ratio', 'acute_ratio', 'weighted_prior_visits',
        'log_inpatient', 'log_emergency', 'log_total_visits'
    ])

    # 11. Hospital stay intensity & procedures ratios
    for d in (train, eval_cols):
        t_hosp = np.maximum(1.0, d['time_in_hospital'])
        d['labs_per_day'] = d['num_lab_procedures'] / t_hosp
        d['meds_per_day'] = d['num_medications'] / t_hosp
        d['procs_per_day'] = d['num_procedures'] / t_hosp
        d['diagnoses_per_day'] = d['number_diagnoses'] / t_hosp
        d['total_procedures'] = d['num_lab_procedures'] + d['num_procedures']
        d['meds_per_diag'] = d['num_medications'] / (d['number_diagnoses'] + 1.0)
        d['procs_per_med'] = d['total_procedures'] / (d['num_medications'] + 1.0)
        d['long_stay'] = (d['time_in_hospital'] >= 7).astype(np.float64)
    extra_numeric.extend([
        'labs_per_day', 'meds_per_day', 'procs_per_day', 'diagnoses_per_day',
        'total_procedures', 'meds_per_diag', 'procs_per_med', 'long_stay'
    ])

    # 12. Discharge and admission indicators
    for d in (train, eval_cols):
        disp = d['discharge_disposition_id']
        d['disch_home'] = (disp == '1').astype(np.float64)
        d['disch_snf'] = (disp == '3').astype(np.float64)
        d['disch_home_health'] = (disp == '6').astype(np.float64)
        d['disch_ama'] = (disp == '7').astype(np.float64)
        d['disch_transfer'] = np.isin(disp, ['2', '3', '4', '5', '22']).astype(np.float64)

        adm_typ = d['admission_type_id']
        adm_src = d['admission_source_id']
        d['adm_is_emergency'] = np.isin(adm_typ, ['1', '2']).astype(np.float64)
        d['adm_from_er'] = (adm_src == '7').astype(np.float64)
        d['adm_high_acuity'] = (d['adm_is_emergency'] * d['adm_from_er']).astype(np.float64)
    extra_numeric.extend([
        'disch_home', 'disch_snf', 'disch_home_health', 'disch_ama', 'disch_transfer',
        'adm_is_emergency', 'adm_from_er', 'adm_high_acuity'
    ])

    # 13. LACE Index bedside score proxy
    def calc_lace(los, is_er, charlson, ed_visits):
        l_score = np.zeros_like(los, dtype=np.float64)
        l_score[los == 2] = 1.0
        l_score[los == 3] = 2.0
        l_score[(los >= 4) & (los <= 6)] = 3.0
        l_score[(los >= 7) & (los <= 13)] = 4.0
        l_score[los >= 14] = 5.0
        a_score = 3.0 * is_er
        c_score = np.clip(charlson, 0.0, 5.0)
        e_score = np.clip(ed_visits, 0.0, 4.0)
        return l_score + a_score + c_score + e_score

    train['lace_score'] = calc_lace(train['time_in_hospital'], train['adm_is_emergency'], train_charlson, train['number_emergency'])
    eval_cols['lace_score'] = calc_lace(eval_cols['time_in_hospital'], eval_cols['adm_is_emergency'], eval_charlson, eval_cols['number_emergency'])
    extra_numeric.append('lace_score')

    # 14. Lab test indicators & Glycemic control clinical inertia
    for d in (train, eval_cols):
        a1c = d['A1Cresult']
        glu = d['max_glu_serum']
        chg = d['change']

        a1c_tested = (a1c != 'None').astype(np.float64)
        a1c_high = ((a1c == '>7') | (a1c == '>8')).astype(np.float64)
        glu_tested = (glu != 'None').astype(np.float64)
        glu_high = ((glu == '>200') | (glu == '>300')).astype(np.float64)

        d['a1c_tested'] = a1c_tested
        d['a1c_high'] = a1c_high
        d['glu_tested'] = glu_tested
        d['glu_high'] = glu_high
        d['diabetes_uncontrolled'] = ((a1c_high == 1.0) | (glu_high == 1.0)).astype(np.float64)
        d['high_a1c_no_med_change'] = ((a1c_high == 1.0) & (chg == 'No')).astype(np.float64)
        d['high_a1c_med_change'] = ((a1c_high == 1.0) & (chg == 'Ch')).astype(np.float64)
    extra_numeric.extend([
        'a1c_tested', 'a1c_high', 'glu_tested', 'glu_high',
        'diabetes_uncontrolled', 'high_a1c_no_med_change', 'high_a1c_med_change'
    ])

    # 15. Medication directionality & polypharmacy
    for d in (train, eval_cols):
        ins = d['insulin']
        d['on_insulin'] = (ins != 'No').astype(np.float64)
        d['insulin_up'] = (ins == 'Up').astype(np.float64)
        d['insulin_down'] = (ins == 'Down').astype(np.float64)
        d['insulin_steady'] = (ins == 'Steady').astype(np.float64)
        d['insulin_changed'] = ((ins == 'Up') | (ins == 'Down')).astype(np.float64)
        d['on_metformin'] = (d['metformin'] != 'No').astype(np.float64)
        d['polypharmacy'] = (d['num_medications'] >= 16).astype(np.float64)
        d['diabetes_med_flag'] = (d['diabetesMed'] == 'Yes').astype(np.float64)
        d['med_change_flag'] = (d['change'] == 'Ch').astype(np.float64)
    extra_numeric.extend([
        'on_insulin', 'insulin_up', 'insulin_down', 'insulin_steady', 'insulin_changed',
        'on_metformin', 'polypharmacy', 'diabetes_med_flag', 'med_change_flag'
    ])

    # 16. Missingness flags
    for d in (train, eval_cols):
        d['missing_specialty'] = (d['medical_specialty'] == None).astype(np.float64)
        d['missing_payer'] = (d['payer_code'] == None).astype(np.float64)
    extra_numeric.extend(['missing_specialty', 'missing_payer'])

    # 17. High-order categorical interactions & additional target encodings
    train['diag1_disch'] = np.array([f"{g}_{disp}" for g, disp in zip(train['diag_1_grp'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['diag1_disch'] = np.array([f"{g}_{disp}" for g, disp in zip(eval_cols['diag_1_grp'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('diag1_disch')

    train['admsrc_disch'] = np.array([f"{s}_{disp}" for s, disp in zip(train['admission_source_id'], train['discharge_disposition_id'])], dtype=object)
    eval_cols['admsrc_disch'] = np.array([f"{s}_{disp}" for s, disp in zip(eval_cols['admission_source_id'], eval_cols['discharge_disposition_id'])], dtype=object)
    extra_categorical.append('admsrc_disch')

    more_te_cols = ['admission_type_id', 'diag_1_grp', 'diag1_disch', 'admsrc_disch']
    for col in more_te_cols:
        if col in train and col in eval_cols:
            tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42)
            te_col = col + "_te"
            train[te_col] = tr_enc
            eval_cols[te_col] = ev_enc
            extra_numeric.append(te_col)

    # 18. ICD-9 3-Digit Category Prefix & Primary Diagnosis Profiling
    def get_icd9_prefix(code):
        if code is None:
            return "missing"
        s = str(code).strip()
        if not s or s == "?":
            return "missing"
        return s.split(".")[0]

    for d in (train, eval_cols):
        p1 = np.array([get_icd9_prefix(x) for x in d["diag_1"]], dtype=object)
        p2 = np.array([get_icd9_prefix(x) for x in d["diag_2"]], dtype=object)
        p3 = np.array([get_icd9_prefix(x) for x in d["diag_3"]], dtype=object)

        d["diag_1_pfx"] = p1
        d["diag_2_pfx"] = p2
        d["diag_3_pfx"] = p3

        d["diag1_is_chf"] = (p1 == "428").astype(np.float64)
        d["diag1_is_dm"] = (p1 == "250").astype(np.float64)
        d["diag1_is_circulatory"] = np.isin(p1, ["428", "410", "411", "414", "427", "434", "401", "402"]).astype(np.float64)
        d["diag1_is_resp"] = np.isin(p1, ["486", "491", "492", "493", "496"]).astype(np.float64)
        d["diag1_is_renal"] = np.isin(p1, ["584", "585", "586"]).astype(np.float64)

        d["any_diag_dm"] = ((p1 == "250") | (p2 == "250") | (p3 == "250")).astype(np.float64)
        d["dm_is_secondary"] = ((p1 != "250") & ((p2 == "250") | (p3 == "250"))).astype(np.float64)

    extra_categorical.extend(["diag_1_pfx", "diag_2_pfx", "diag_3_pfx"])
    extra_numeric.extend([
        "diag1_is_chf", "diag1_is_dm", "diag1_is_circulatory", "diag1_is_resp", "diag1_is_renal",
        "any_diag_dm", "dm_is_secondary"
    ])

    # 19. Frequency Encodings (Rare vs Common case volume)
    def compute_freq_encoding(tr_arr, ev_arr):
        counts = {}
        for x in tr_arr:
            if x is not None:
                counts[x] = counts.get(x, 0) + 1
        n = float(len(tr_arr))
        tr_f = np.array([counts.get(x, 0) / n for x in tr_arr], dtype=np.float64)
        ev_f = np.array([counts.get(x, 0) / n for x in ev_arr], dtype=np.float64)
        return tr_f, ev_f

    for f_col in ["medical_specialty", "payer_code", "diag_1_pfx", "discharge_disposition_id"]:
        if f_col in train and f_col in eval_cols:
            tr_f, ev_f = compute_freq_encoding(train[f_col], eval_cols[f_col])
            col_name = f_col + "_freq"
            train[col_name] = tr_f
            eval_cols[col_name] = ev_f
            extra_numeric.append(col_name)

    # 20. Target encodings on category prefixes and age-discharge interactions
    train["diag2_grp"] = np.array([icd9_group(x) for x in train["diag_2"]], dtype=object)
    eval_cols["diag2_grp"] = np.array([icd9_group(x) for x in eval_cols["diag_2"]], dtype=object)
    extra_categorical.append("diag2_grp")

    train["age_disch"] = np.array([f"{a}_{d}" for a, d in zip(train["age"], train["discharge_disposition_id"])], dtype=object)
    eval_cols["age_disch"] = np.array([f"{a}_{d}" for a, d in zip(eval_cols["age"], eval_cols["discharge_disposition_id"])], dtype=object)
    extra_categorical.append("age_disch")

    pfx_te_cols = ["diag_1_pfx", "diag_2_pfx", "diag2_grp", "age_disch", "race"]
    for col in pfx_te_cols:
        if col in train and col in eval_cols:
            tr_enc, ev_enc = oof_target_encode(train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42)
            te_col = col + "_te"
            train[te_col] = tr_enc
            eval_cols[te_col] = ev_enc
            extra_numeric.append(te_col)

    # 21. Medication Directional Titration & ADA Stepped Care Regimens
    for d in (train, eval_cols):
        num_up = np.zeros(len(d["insulin"]), dtype=np.float64)
        num_down = np.zeros(len(d["insulin"]), dtype=np.float64)
        for col in med_cols:
            if col in d:
                num_up += (d[col] == "Up")
                num_down += (d[col] == "Down")

        d["num_meds_up"] = num_up
        d["num_meds_down"] = num_down
        d["any_med_up"] = (num_up > 0).astype(np.float64)
        d["any_med_down"] = (num_down > 0).astype(np.float64)
        d["complex_med_change"] = ((num_up > 0) & (num_down > 0)).astype(np.float64)

        has_oral = np.zeros(len(d["insulin"]), dtype=bool)
        for col in med_cols:
            if col != "insulin" and col in d:
                has_oral = has_oral | ((d[col] != "No") & (d[col] != None))

        on_ins = (d["insulin"] != "No") & (d["insulin"] != None)
        no_med = (d["diabetesMed"] == "No")

        d["regimen_none"] = no_med.astype(np.float64)
        d["regimen_insulin_only"] = (on_ins & (~has_oral)).astype(np.float64)
        d["regimen_insulin_plus_oral"] = (on_ins & has_oral).astype(np.float64)
        d["regimen_oral_only"] = ((~on_ins) & has_oral).astype(np.float64)

    extra_numeric.extend([
        "num_meds_up", "num_meds_down", "any_med_up", "any_med_down", "complex_med_change",
        "regimen_none", "regimen_insulin_only", "regimen_insulin_plus_oral", "regimen_oral_only"
    ])

    # 22. Clinical Syndromes & Stay Utilization Refinements
    for d, cm in [(train, train_cm), (eval_cols, eval_cm)]:
        d["cm_cardiorenal"] = (cm["cm_chf"] * cm["cm_renal"]).astype(np.float64)
        d["cm_vascular"] = ((cm["cm_pvd"] == 1) | (cm["cm_cerebrovascular"] == 1) | (cm["cm_mi"] == 1)).astype(np.float64)
        d["cm_severe"] = ((cm["cm_renal"] == 1) | (cm["cm_chf"] == 1) | (cm["cm_pulmonary"] == 1)).astype(np.float64)
        d["multimorbid"] = (cm["cm_count"] >= 2).astype(np.float64)
        d["highly_multimorbid"] = (cm["cm_count"] >= 3).astype(np.float64)
        d["cm_density"] = cm["cm_count"] / (d["time_in_hospital"] + 1.0)

        inpatient_cnt = d["number_inpatient"]
        d["inp_0"] = (inpatient_cnt == 0).astype(np.float64)
        d["inp_1"] = (inpatient_cnt == 1).astype(np.float64)
        d["inp_2plus"] = (inpatient_cnt >= 2).astype(np.float64)
        d["inp_3plus"] = (inpatient_cnt >= 3).astype(np.float64)
        d["revolving_door"] = ((inpatient_cnt >= 1) & (d["time_in_hospital"] >= 4)).astype(np.float64)
        d["total_care_actions"] = d["num_lab_procedures"] + 3.0 * d["num_procedures"] + 2.0 * d["num_medications"]
        d["care_intensity"] = d["total_care_actions"] / (d["time_in_hospital"] + 1.0)
        d["er_to_facility"] = (d["adm_from_er"] * d["disch_transfer"]).astype(np.float64)

    extra_numeric.extend([
        "cm_cardiorenal", "cm_vascular", "cm_severe", "multimorbid", "highly_multimorbid", "cm_density",
        "inp_0", "inp_1", "inp_2plus", "inp_3plus", "revolving_door",
        "total_care_actions", "care_intensity", "er_to_facility"
    ])

    # 23. Tabular encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train)
    x_eval = encoder.transform(eval_cols)

    # 24. Heterogeneous Stratified 5-Fold Ensemble: LGBM-Deep + LGBM-Shallow + XGBoost
    import xgboost as xgb
    from scipy.stats import rankdata

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    eval_preds_lgb1 = np.zeros(x_eval.shape[0], dtype=np.float64)
    eval_preds_lgb2 = np.zeros(x_eval.shape[0], dtype=np.float64)
    eval_preds_xgb = np.zeros(x_eval.shape[0], dtype=np.float64)

    for train_idx, val_idx in skf.split(x_train, y_train):
        fold_x_train, fold_y_train = x_train[train_idx], y_train[train_idx]

        # Model 1: LightGBM (leaf-wise 31 leaves for complex interaction discovery)
        lgb1 = lgb.LGBMClassifier(
            n_estimators=230,
            learning_rate=0.02,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=0.1,
            reg_lambda=1.0,
            cat_smooth=30.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
        lgb1.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        eval_preds_lgb1 += lgb1.predict_proba(x_eval)[:, 1] / 5.0

        # Model 2: LightGBM (depth-constrained max_depth=5, 16 leaves with strong L2 for generalization)
        lgb2 = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.02,
            num_leaves=16,
            max_depth=5,
            min_child_samples=140,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.4,
            reg_lambda=2.5,
            cat_smooth=40.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        )
        lgb2.fit(fold_x_train, fold_y_train, categorical_feature=encoder.categorical_indices)
        eval_preds_lgb2 += lgb2.predict_proba(x_eval)[:, 1] / 5.0

        # Model 3: XGBoost (depth-wise histogram trees with regularized split scoring)
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.025,
            max_depth=5,
            subsample=0.75,
            colsample_bytree=0.65,
            reg_alpha=0.3,
            reg_lambda=2.0,
            min_child_weight=5,
            tree_method='hist',
            random_state=42,
            n_jobs=1,
            eval_metric='logloss',
        )
        xgb_model.fit(fold_x_train, fold_y_train)
        eval_preds_xgb += xgb_model.predict_proba(x_eval)[:, 1] / 5.0

    # 25. Percentile Rank-Averaging Blend (strictly optimizes ROC-AUC invariance)
    n_eval_rows = float(len(eval_preds_lgb1))
    rank_lgb1 = rankdata(eval_preds_lgb1) / n_eval_rows
    rank_lgb2 = rankdata(eval_preds_lgb2) / n_eval_rows
    rank_xgb = rankdata(eval_preds_xgb) / n_eval_rows

    final_preds = 0.40 * rank_lgb1 + 0.30 * rank_lgb2 + 0.30 * rank_xgb
    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
