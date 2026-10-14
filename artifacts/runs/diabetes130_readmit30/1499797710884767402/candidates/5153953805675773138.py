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

Strategy only: an untuned single LightGBM on the raw columns. Encoding mechanics live in `clinical_toolkit`.
"""

from __future__ import annotations

from typing import Dict

import lightgbm as lgb
import numpy as np

from clinical_models import TabularTask
from clinical_toolkit import TabularEncoder, icd9_group  # noqa: F401  (icd9_group available for feature ideas)


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
    try:
        from clinical_toolkit import icd9_group, TabularEncoder
    except ImportError:
        pass
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    train_data = dict(train)
    eval_data = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Clean continuous age and related features
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    for col_dict in [train_data, eval_data]:
        raw_age = col_dict.get('age', np.array([]))
        num_age = np.array([age_map.get(a, np.nan) for a in raw_age], dtype=np.float64)
        col_dict['num_age'] = num_age
        col_dict['age_elderly'] = (num_age >= 75.0).astype(np.float64)
    extra_numeric.extend(['num_age', 'age_elderly'])

    # 2. Prior Utilisation & Hospital Stay intensity
    for col_dict in [train_data, eval_data]:
        inp = np.nan_to_num(col_dict.get('number_inpatient', 0.0), nan=0.0)
        emg = np.nan_to_num(col_dict.get('number_emergency', 0.0), nan=0.0)
        outp = np.nan_to_num(col_dict.get('number_outpatient', 0.0), nan=0.0)
        tot_visits = inp + emg + outp

        col_dict['util_total_visits'] = tot_visits
        col_dict['util_inpatient_ratio'] = inp / (tot_visits + 1.0)
        col_dict['util_acute_ratio'] = (inp + emg) / (tot_visits + 1.0)
        col_dict['util_weighted_score'] = 3.0 * inp + 1.5 * emg + 0.5 * outp
        col_dict['util_frequent_inpatient'] = (inp >= 2.0).astype(np.float64)

        # Log transforms to reduce skew
        col_dict['log_number_inpatient'] = np.log1p(inp)
        col_dict['log_number_emergency'] = np.log1p(emg)
        col_dict['log_number_outpatient'] = np.log1p(outp)
        col_dict['log_total_visits'] = np.log1p(tot_visits)

        col_dict['has_prior_inpatient'] = (inp > 0.0).astype(np.float64)
        col_dict['has_prior_emergency'] = (emg > 0.0).astype(np.float64)
        col_dict['has_prior_outpatient'] = (outp > 0.0).astype(np.float64)

        time_safe = np.maximum(np.nan_to_num(col_dict.get('time_in_hospital', 1.0), nan=1.0), 1.0)
        num_labs = np.nan_to_num(col_dict.get('num_lab_procedures', 0.0), nan=0.0)
        num_proc = np.nan_to_num(col_dict.get('num_procedures', 0.0), nan=0.0)
        num_meds = np.nan_to_num(col_dict.get('num_medications', 0.0), nan=0.0)
        num_diag = np.nan_to_num(col_dict.get('number_diagnoses', 0.0), nan=0.0)

        col_dict['hosp_labs_per_day'] = num_labs / time_safe
        col_dict['hosp_meds_per_day'] = num_meds / time_safe
        col_dict['hosp_proc_per_day'] = num_proc / time_safe
        col_dict['hosp_service_intensity'] = (num_labs + 2.0 * num_proc + 1.5 * num_meds) / time_safe
        col_dict['hosp_diag_per_day'] = num_diag / time_safe
        col_dict['hosp_meds_per_diag'] = num_meds / (num_diag + 1.0)
        col_dict['hosp_rapid_readmit_risk'] = ((time_safe <= 2.0) & (inp > 0.0)).astype(np.float64)

        num_age_safe = np.nan_to_num(col_dict['num_age'], nan=60.0)
        col_dict['age_times_inpatient'] = num_age_safe * inp

    extra_numeric.extend([
        'util_total_visits', 'util_inpatient_ratio', 'util_acute_ratio',
        'util_weighted_score', 'util_frequent_inpatient',
        'log_number_inpatient', 'log_number_emergency', 'log_number_outpatient', 'log_total_visits',
        'has_prior_inpatient', 'has_prior_emergency', 'has_prior_outpatient',
        'hosp_labs_per_day', 'hosp_meds_per_day', 'hosp_proc_per_day',
        'hosp_service_intensity', 'hosp_diag_per_day', 'hosp_meds_per_diag',
        'hosp_rapid_readmit_risk', 'age_times_inpatient'
    ])

    # 3. Medication Dynamics, Polypharmacy & Therapeutic Inertia
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    sulfonylureas = {'glimepiride', 'glipizide', 'glyburide', 'chlorpropamide', 'tolbutamide', 'tolazamide', 'acetohexamide'}
    tzds = {'pioglitazone', 'rosiglitazone'}

    for col_dict in [train_data, eval_data]:
        n_rows = len(col_dict['time_in_hospital'])
        active = np.zeros(n_rows, dtype=np.float64)
        up = np.zeros(n_rows, dtype=np.float64)
        down = np.zeros(n_rows, dtype=np.float64)
        on_sulfonylurea = np.zeros(n_rows, dtype=np.float64)
        on_metformin = np.zeros(n_rows, dtype=np.float64)
        on_tzd = np.zeros(n_rows, dtype=np.float64)

        for m in med_cols:
            if m in col_dict:
                arr = col_dict[m]
                is_active = np.isin(arr, ['Steady', 'Up', 'Down'])
                active += is_active.astype(np.float64)
                up += (arr == 'Up').astype(np.float64)
                down += (arr == 'Down').astype(np.float64)
                if m in sulfonylureas:
                    on_sulfonylurea = np.maximum(on_sulfonylurea, is_active.astype(np.float64))
                elif m == 'metformin':
                    on_metformin = is_active.astype(np.float64)
                elif m in tzds:
                    on_tzd = np.maximum(on_tzd, is_active.astype(np.float64))

        insulin_arr = col_dict.get('insulin', np.full(n_rows, 'No', dtype=object))
        ins_active = np.isin(insulin_arr, ['Steady', 'Up', 'Down']).astype(np.float64)
        ins_up = (insulin_arr == 'Up').astype(np.float64)
        ins_down = (insulin_arr == 'Down').astype(np.float64)
        oral_meds = np.maximum(0.0, active - ins_active)

        col_dict['med_num_active'] = active
        col_dict['med_num_changed'] = up + down
        col_dict['med_num_up'] = up
        col_dict['med_num_down'] = down
        col_dict['med_num_steady'] = active - (up + down)
        col_dict['med_insulin_active'] = ins_active
        col_dict['med_insulin_up'] = ins_up
        col_dict['med_insulin_down'] = ins_down
        col_dict['med_oral_active'] = oral_meds
        col_dict['med_complex_regimen'] = ((ins_active > 0.0) & (oral_meds >= 2.0)).astype(np.float64)
        col_dict['med_insulin_titration'] = (ins_up + ins_down > 0.0).astype(np.float64)
        col_dict['med_churn_ratio'] = (up + down) / np.maximum(1.0, active)
        col_dict['on_sulfonylurea'] = on_sulfonylurea
        col_dict['on_metformin'] = on_metformin
        col_dict['on_tzd'] = on_tzd
        col_dict['med_risk_insulin_sulfonylurea'] = (ins_active * on_sulfonylurea).astype(np.float64)
        col_dict['med_risk_tzd_insulin'] = (on_tzd * ins_active).astype(np.float64)

    extra_numeric.extend([
        'med_num_active', 'med_num_changed', 'med_num_up', 'med_num_down', 'med_num_steady',
        'med_insulin_active', 'med_insulin_up', 'med_insulin_down', 'med_oral_active',
        'med_complex_regimen', 'med_insulin_titration', 'med_churn_ratio', 'on_sulfonylurea',
        'on_metformin', 'on_tzd', 'med_risk_insulin_sulfonylurea', 'med_risk_tzd_insulin'
    ])

    # 4. Lab Results (A1C and Glucose) & Clinical Inertia
    for col_dict in [train_data, eval_data]:
        a1c = col_dict.get('A1Cresult', np.array([]))
        glu = col_dict.get('max_glu_serum', np.array([]))

        col_dict['a1c_performed'] = (~np.isin(a1c, ['None', None])).astype(np.float64)
        col_dict['a1c_high'] = np.isin(a1c, ['>7', '>8']).astype(np.float64)
        col_dict['glu_performed'] = (~np.isin(glu, ['None', None])).astype(np.float64)
        col_dict['glu_high'] = np.isin(glu, ['>200', '>300']).astype(np.float64)
        col_dict['therapeutic_inertia'] = (
            (col_dict['a1c_high'] > 0.0) & (col_dict['med_num_changed'] == 0.0)
        ).astype(np.float64)

    extra_numeric.extend(['a1c_performed', 'a1c_high', 'glu_performed', 'glu_high', 'therapeutic_inertia'])

    # 5. Admission, Discharge & SDOH Payer Groupings
    def group_discharge(disch_arr):
        res = np.full(len(disch_arr), 'other', dtype=object)
        disch_str = np.array([str(x).strip() for x in disch_arr])
        res[np.isin(disch_str, ['1', '8'])] = 'routine_home'
        res[disch_str == '6'] = 'home_health'
        res[disch_str == '3'] = 'snf'
        res[disch_str == '7'] = 'ama'
        res[np.isin(disch_str, ['2', '4', '5', '10', '12', '15', '16', '17', '22', '23', '24', '27', '28', '29', '30'])] = 'facility_transfer'
        return res

    def group_admission_source(source_arr):
        res = np.full(len(source_arr), 'other', dtype=object)
        src_str = np.array([str(x).strip() for x in source_arr])
        res[src_str == '7'] = 'emergency'
        res[np.isin(src_str, ['1', '2', '3'])] = 'referral'
        res[src_str == '5'] = 'snf_transfer'
        res[np.isin(src_str, ['4', '6', '10', '18', '22', '25', '26'])] = 'other_transfer'
        return res

    def group_admission_type(type_arr):
        res = np.full(len(type_arr), 'other', dtype=object)
        type_str = np.array([str(x).strip() for x in type_arr])
        res[np.isin(type_str, ['1', '2', '5'])] = 'emergency_urgent'
        res[type_str == '3'] = 'elective'
        return res

    for col_dict in [train_data, eval_data]:
        disch_raw = col_dict.get('discharge_disposition_id', [])
        adm_src_raw = col_dict.get('admission_source_id', [])
        adm_type_raw = col_dict.get('admission_type_id', [])
        payer_raw = col_dict.get('payer_code', np.array([]))

        g_disch = group_discharge(disch_raw)
        g_src = group_admission_source(adm_src_raw)
        g_type = group_admission_type(adm_type_raw)

        col_dict['grouped_discharge'] = g_disch
        col_dict['grouped_adm_source'] = g_src
        col_dict['grouped_adm_type'] = g_type

        # Clinical discordance flags
        col_dict['is_home_health'] = (g_disch == 'home_health').astype(np.float64)
        col_dict['is_snf'] = (g_disch == 'snf').astype(np.float64)
        col_dict['is_ama'] = (g_disch == 'ama').astype(np.float64)
        col_dict['is_emergency_adm'] = (g_src == 'emergency').astype(np.float64)

        # Premature discharge discordance: acute/emergency admission, short stay (<=2 days), complex (>=7 diagnoses), discharged home without support
        time_stay = np.nan_to_num(col_dict.get('time_in_hospital', 1.0), nan=1.0)
        n_diag = np.nan_to_num(col_dict.get('number_diagnoses', 0.0), nan=0.0)
        col_dict['risk_premature_discharge'] = (
            (col_dict['is_emergency_adm'] > 0.0) & (time_stay <= 2.0) & (n_diag >= 7.0) & (g_disch == 'routine_home')
        ).astype(np.float64)

        # SDOH Payer features
        payer_str = np.array([str(x).strip() for x in payer_raw])
        col_dict['payer_is_medicare'] = (payer_str == 'MC').astype(np.float64)
        col_dict['payer_is_medicaid'] = (payer_str == 'MD').astype(np.float64)
        col_dict['payer_is_selfpay'] = np.isin(payer_str, ['SP', 'UN']).astype(np.float64)
        col_dict['payer_is_missing'] = np.isin(payer_str, ['?', 'None', 'nan']).astype(np.float64)

        # Medication adherence / change context
        chg_arr = col_dict.get('change', np.array([]))
        med_presc = col_dict.get('diabetesMed', np.array([]))
        col_dict['med_had_change'] = (chg_arr == 'Ch').astype(np.float64)
        col_dict['med_diabetes_prescribed'] = (med_presc == 'Yes').astype(np.float64)
        col_dict['med_polypharmacy_high'] = (col_dict['num_medications'] >= 15.0).astype(np.float64)

    extra_categorical.extend(['grouped_discharge', 'grouped_adm_source', 'grouped_adm_type'])
    extra_numeric.extend([
        'is_home_health', 'is_snf', 'is_ama', 'is_emergency_adm', 'risk_premature_discharge',
        'payer_is_medicare', 'payer_is_medicaid', 'payer_is_selfpay', 'payer_is_missing',
        'med_had_change', 'med_diabetes_prescribed', 'med_polypharmacy_high'
    ])

    # 6. Diagnosis Preprocessing, Diagnostic Entropy, Charlson & Geriatric Frailty Index
    def extract_prefix(code):
        if code is None or code == '?' or code == 'None':
            return 'missing'
        s = str(code).strip()
        return s.split('.')[0] if s else 'missing'

    def compute_diag_entropy(g1, g2, g3):
        n = len(g1)
        entropies = np.zeros(n, dtype=np.float64)
        distinct_counts = np.zeros(n, dtype=np.float64)
        for i in range(n):
            c1, c2, c3 = g1[i], g2[i], g3[i]
            v1 = c1 not in ('missing', 'None', None, '?')
            v2 = c2 not in ('missing', 'None', None, '?')
            v3 = c3 not in ('missing', 'None', None, '?')
            tot = v1 + v2 + v3
            if tot == 0:
                continue
            
            v_list = []
            if v1: v_list.append(c1)
            if v2: v_list.append(c2)
            if v3: v_list.append(c3)
            
            distinct = len(set(v_list))
            distinct_counts[i] = float(distinct)
            
            if tot == 1:
                entropies[i] = 0.0
            elif tot == 2:
                if distinct == 1:
                    entropies[i] = 0.0
                else:
                    entropies[i] = 0.6931471805599453
            elif tot == 3:
                if distinct == 1:
                    entropies[i] = 0.0
                elif distinct == 2:
                    entropies[i] = 0.6365141682948128
                else:
                    entropies[i] = 1.0986122886681098
        return entropies, distinct_counts

    def compute_comorbidity_and_frailty(p1, p2, p3, d1_raw, d2_raw, d3_raw, elderly_arr):
        n = len(p1)
        charlson_scores = np.zeros(n, dtype=np.float64)
        frailty_scores = np.zeros(n, dtype=np.float64)
        chf = np.zeros(n, dtype=np.float64)
        cad = np.zeros(n, dtype=np.float64)
        copd = np.zeros(n, dtype=np.float64)
        ckd = np.zeros(n, dtype=np.float64)
        stroke = np.zeros(n, dtype=np.float64)
        cancer = np.zeros(n, dtype=np.float64)
        liver = np.zeros(n, dtype=np.float64)
        sepsis = np.zeros(n, dtype=np.float64)
        ulcer = np.zeros(n, dtype=np.float64)
        electrolyte = np.zeros(n, dtype=np.float64)
        dementia = np.zeros(n, dtype=np.float64)
        pneumonia = np.zeros(n, dtype=np.float64)
        htn = np.zeros(n, dtype=np.float64)
        diab_complicated = np.zeros(n, dtype=np.float64)

        for i in range(n):
            prefs = (p1[i], p2[i], p3[i])
            raws = (str(d1_raw[i]).strip(), str(d2_raw[i]).strip(), str(d3_raw[i]).strip())

            for p in prefs:
                if not p or p == 'missing':
                    continue
                if p[0].isdigit():
                    val = int(p)
                    if val == 428:
                        chf[i] = 1.0
                    elif 410 <= val <= 414:
                        cad[i] = 1.0
                    elif (490 <= val <= 496) or (500 <= val <= 505):
                        copd[i] = 1.0
                    elif val in (582, 583, 585, 586, 588):
                        ckd[i] = 1.0
                    elif 430 <= val <= 438:
                        stroke[i] = 1.0
                    elif (140 <= val <= 199) or (200 <= val <= 208):
                        cancer[i] = 1.0
                    elif val in (571, 572):
                        liver[i] = 1.0
                    elif val in (38, 995):
                        sepsis[i] = 1.0
                    elif 531 <= val <= 534:
                        ulcer[i] = 1.0
                    elif val == 276:
                        electrolyte[i] = 1.0
                    elif val in (290, 294, 331):
                        dementia[i] = 1.0
                    elif 480 <= val <= 486:
                        pneumonia[i] = 1.0
                    elif 401 <= val <= 405:
                        htn[i] = 1.0

            for r in raws:
                if r.startswith('250.') and not r.startswith('250.0'):
                    diab_complicated[i] = 1.0
                if r.startswith('707'):
                    ulcer[i] = 1.0

            charlson = (
                1.0 * chf[i] + 1.0 * cad[i] + 1.0 * copd[i] + 1.0 * ulcer[i] +
                2.0 * ckd[i] + 2.0 * stroke[i] + 2.0 * cancer[i] + 3.0 * liver[i] + 2.0 * sepsis[i]
            )
            charlson_scores[i] = charlson

            # Hospital Frailty Risk Score (Gilbert et al., Lancet 2018 adaptation)
            frailty = (
                1.0 * elderly_arr[i] + 2.0 * dementia[i] + 1.5 * ulcer[i] +
                1.5 * ckd[i] + 1.5 * chf[i] + 1.0 * electrolyte[i] + 1.5 * stroke[i]
            )
            frailty_scores[i] = frailty

        return (
            charlson_scores, frailty_scores, chf, cad, copd, ckd, stroke, cancer,
            liver, sepsis, electrolyte, dementia, pneumonia, htn, diab_complicated
        )

    for col_dict in [train_data, eval_data]:
        d1 = col_dict.get('diag_1', np.array([]))
        d2 = col_dict.get('diag_2', np.array([]))
        d3 = col_dict.get('diag_3', np.array([]))

        g1 = np.array([icd9_group(c) for c in d1], dtype=object)
        g2 = np.array([icd9_group(c) for c in d2], dtype=object)
        g3 = np.array([icd9_group(c) for c in d3], dtype=object)
        p1 = np.array([extract_prefix(c) for c in d1], dtype=object)
        p2 = np.array([extract_prefix(c) for c in d2], dtype=object)
        p3 = np.array([extract_prefix(c) for c in d3], dtype=object)

        col_dict['diag_1_group'] = g1
        col_dict['diag_2_group'] = g2
        col_dict['diag_3_group'] = g3
        col_dict['diag_1_prefix'] = p1
        col_dict['diag_2_prefix'] = p2
        col_dict['diag_3_prefix'] = p3

        # Entropy of diagnostic chapters
        ent, distinct_g = compute_diag_entropy(g1, g2, g3)
        col_dict['diag_chapter_entropy'] = ent
        col_dict['diag_distinct_chapters'] = distinct_g

        # Charlson Comorbidities & Geriatric Frailty
        (
            charlson, frailty, chf, cad, copd, ckd, stroke, cancer, liver, sepsis,
            electrolyte, dementia, pneumonia, htn, diab_comp
        ) = compute_comorbidity_and_frailty(p1, p2, p3, d1, d2, d3, col_dict['age_elderly'])

        col_dict['charlson_score'] = charlson
        col_dict['frailty_score'] = frailty
        col_dict['comorb_has_chf'] = chf
        col_dict['comorb_has_cad'] = cad
        col_dict['comorb_has_copd'] = copd
        col_dict['comorb_has_ckd'] = ckd
        col_dict['comorb_has_stroke'] = stroke
        col_dict['comorb_has_cancer'] = cancer
        col_dict['comorb_has_liver'] = liver
        col_dict['comorb_has_sepsis'] = sepsis
        col_dict['comorb_has_electrolyte'] = electrolyte
        col_dict['comorb_has_dementia'] = dementia
        col_dict['comorb_has_pneumonia'] = pneumonia
        col_dict['comorb_has_htn'] = htn
        col_dict['comorb_diab_complicated'] = diab_comp

        circ = (g1 == 'circulatory') | (g2 == 'circulatory') | (g3 == 'circulatory')
        diab = (g1 == 'diabetes') | (g2 == 'diabetes') | (g3 == 'diabetes')
        resp = (g1 == 'respiratory') | (g2 == 'respiratory') | (g3 == 'respiratory')
        digest = (g1 == 'digestive') | (g2 == 'digestive') | (g3 == 'digestive')
        genit = (g1 == 'genitourinary') | (g2 == 'genitourinary') | (g3 == 'genitourinary')

        col_dict['comorb_has_circ'] = circ.astype(np.float64)
        col_dict['comorb_has_diab'] = diab.astype(np.float64)
        col_dict['comorb_has_resp'] = resp.astype(np.float64)
        col_dict['comorb_has_genit'] = genit.astype(np.float64)
        col_dict['comorb_num_systems'] = (
            circ.astype(int) + diab.astype(int) + resp.astype(int) + digest.astype(int) + genit.astype(int)
        ).astype(np.float64)

        # High-risk syndromic intersections
        col_dict['syndrome_cardiorenal'] = (chf * ckd).astype(np.float64)
        col_dict['syndrome_cardiodiabetic'] = (chf * diab.astype(np.float64)).astype(np.float64)
        col_dict['syndrome_pulmorenal'] = (copd * ckd).astype(np.float64)

        # Geriatric hypoglycemia risk: elderly + (insulin or sulfonylurea) + renal impairment
        elderly = col_dict['age_elderly']
        med_risk = (col_dict['med_insulin_active'] > 0.0) | (col_dict['on_sulfonylurea'] > 0.0)
        col_dict['syndrome_hypoglycemia_risk'] = ((elderly > 0.0) & med_risk & (ckd > 0.0)).astype(np.float64)

        # Untreated diabetic crisis: diabetic primary diagnosis but zero diabetes medications prescribed
        col_dict['syndrome_untreated_diabetes'] = (
            (col_dict['comorb_has_diab'] > 0.0) & (col_dict['med_diabetes_prescribed'] == 0.0)
        ).astype(np.float64)

        # Multi-morbidity and cardiovascular burden indices
        col_dict['comorb_multimorbidity_count'] = (
            chf + cad + copd + ckd + stroke + cancer + sepsis + dementia + pneumonia + htn + diab_comp
        ).astype(np.float64)
        col_dict['comorb_cv_burden'] = (chf + cad + stroke + htn).astype(np.float64)

        # Age and Comorbidity Interaction features
        num_age_safe = np.nan_to_num(col_dict['num_age'], nan=60.0)
        col_dict['age_times_charlson'] = num_age_safe * charlson
        col_dict['age_times_frailty'] = num_age_safe * frailty

    extra_categorical.extend([
        'diag_1_group', 'diag_2_group', 'diag_3_group',
        'diag_1_prefix', 'diag_2_prefix', 'diag_3_prefix'
    ])
    extra_numeric.extend([
        'diag_chapter_entropy', 'diag_distinct_chapters', 'charlson_score', 'frailty_score',
        'comorb_has_chf', 'comorb_has_cad', 'comorb_has_copd', 'comorb_has_ckd',
        'comorb_has_stroke', 'comorb_has_cancer', 'comorb_has_liver', 'comorb_has_sepsis',
        'comorb_has_electrolyte', 'comorb_has_dementia', 'comorb_has_pneumonia', 'comorb_has_htn', 'comorb_diab_complicated',
        'comorb_has_circ', 'comorb_has_diab', 'comorb_has_resp', 'comorb_has_genit', 'comorb_num_systems',
        'syndrome_cardiorenal', 'syndrome_cardiodiabetic', 'syndrome_pulmorenal', 'syndrome_hypoglycemia_risk',
        'syndrome_untreated_diabetes', 'comorb_multimorbidity_count', 'comorb_cv_burden',
        'age_times_charlson', 'age_times_frailty'
    ])

    # 7. CRAZY IDEA: ICD-9 Co-occurrence Graph Risk Diffusion (Network Medicine)
    # We construct a graph G=(V,E) where nodes are diagnosis prefixes, and edges are co-occurrence counts.
    # Out-of-fold empirical risk is propagated across graph edges via 1-hop Laplacian diffusion.
    def compute_graph_risk_diffusion(tr_p1, tr_p2, tr_p3, y_tr, ev_p1, ev_p2, ev_p3, smoothing=15.0):
        n_tr = len(tr_p1)
        prior = float(np.mean(y_tr))

        # Node empirical stats
        node_counts = {}
        node_sums = {}
        for i in range(n_tr):
            y = y_tr[i]
            seen = set()
            for p in (tr_p1[i], tr_p2[i], tr_p3[i]):
                if p and p != 'missing':
                    seen.add(p)
            for c in seen:
                node_counts[c] = node_counts.get(c, 0) + 1
                node_sums[c] = node_sums.get(c, 0.0) + y

        base_risk = {}
        for c, cnt in node_counts.items():
            base_risk[c] = (node_sums[c] + smoothing * prior) / (cnt + smoothing)

        # Graph co-occurrence adjacency
        cooccur = {}
        for i in range(n_tr):
            codes = [p for p in (tr_p1[i], tr_p2[i], tr_p3[i]) if p and p != 'missing']
            k = len(codes)
            for a in range(k):
                ca = codes[a]
                if ca not in cooccur:
                    cooccur[ca] = {}
                for b in range(a + 1, k):
                    cb = codes[b]
                    if cb not in cooccur:
                        cooccur[cb] = {}
                    cooccur[ca][cb] = cooccur[ca].get(cb, 0) + 1
                    cooccur[cb][ca] = cooccur[cb].get(ca, 0) + 1

        # Laplacian 1-hop diffused neighborhood risk
        diffused_risk = {}
        degrees = {}
        for c, nbrs in cooccur.items():
            deg = sum(nbrs.values())
            degrees[c] = float(deg)
            if deg > 0:
                weighted_risk = sum(w * base_risk.get(nbr, prior) for nbr, w in nbrs.items())
                diffused_risk[c] = (weighted_risk + smoothing * base_risk[c]) / (deg + smoothing)
            else:
                diffused_risk[c] = base_risk.get(c, prior)

        def extract_graph_features(p1_arr, p2_arr, p3_arr):
            n = len(p1_arr)
            out_mean_diff = np.full(n, prior, dtype=np.float64)
            out_max_diff = np.full(n, prior, dtype=np.float64)
            out_log_degree = np.zeros(n, dtype=np.float64)
            out_triad_w = np.zeros(n, dtype=np.float64)
            out_atypical = np.zeros(n, dtype=np.float64)

            for i in range(n):
                p1_val, p2_val, p3_val = p1_arr[i], p2_arr[i], p3_arr[i]
                codes = []
                if p1_val and p1_val != 'missing': codes.append(p1_val)
                if p2_val and p2_val != 'missing': codes.append(p2_val)
                if p3_val and p3_val != 'missing': codes.append(p3_val)
                
                if not codes:
                    continue
                risks = [diffused_risk.get(c, prior) for c in codes]
                degs = [degrees.get(c, 0.0) for c in codes]
                out_mean_diff[i] = float(np.mean(risks))
                out_max_diff[i] = float(np.max(risks))
                out_log_degree[i] = float(np.log1p(np.sum(degs)))

                # Internal triad edge weight
                internal_w = 0.0
                k = len(codes)
                for a in range(k):
                    ca = codes[a]
                    ca_nbrs = cooccur.get(ca, {})
                    for b in range(a + 1, k):
                        cb = codes[b]
                        internal_w += ca_nbrs.get(cb, 0)
                out_triad_w[i] = float(np.log1p(internal_w))
                # Flag atypical comorbidity bundle (2+ conditions that rarely/never co-occur in hospital practice)
                if k >= 2 and internal_w == 0.0:
                    out_atypical[i] = 1.0

            return out_mean_diff, out_max_diff, out_log_degree, out_triad_w, out_atypical

        tr_feats = extract_graph_features(tr_p1, tr_p2, tr_p3)
        ev_feats = extract_graph_features(ev_p1, ev_p2, ev_p3)
        return tr_feats, ev_feats

    tr_gf, ev_gf = compute_graph_risk_diffusion(
        train_data['diag_1_prefix'], train_data['diag_2_prefix'], train_data['diag_3_prefix'],
        y_train,
        eval_data['diag_1_prefix'], eval_data['diag_2_prefix'], eval_data['diag_3_prefix'],
    )
    graph_feat_names = [
        'graph_diffused_risk_mean', 'graph_diffused_risk_max',
        'graph_log_degree_sum', 'graph_triad_cooccur_w', 'graph_atypical_syndrome'
    ]
    for idx, f_name in enumerate(graph_feat_names):
        train_data[f_name] = tr_gf[idx]
        eval_data[f_name] = ev_gf[idx]
        extra_numeric.append(f_name)

    # 8. Latent Clinical Phenotyping via TruncatedSVD on EHR Bag-of-Words
    train_docs = []
    eval_docs = []
    for cd, doc_list in [(train_data, train_docs), (eval_data, eval_docs)]:
        n_rows = len(cd['time_in_hospital'])
        d1g = cd['diag_1_group']
        d2g = cd['diag_2_group']
        d3g = cd['diag_3_group']
        p1 = cd['diag_1_prefix']
        p2 = cd['diag_2_prefix']
        p3 = cd['diag_3_prefix']
        disch = cd['grouped_discharge']
        src = cd['grouped_adm_source']
        ins = cd['med_insulin_active']
        ins_up = cd['med_insulin_up']
        a1c_hi = cd['a1c_high']
        chg = cd['med_had_change']
        frail = cd['frailty_score']
        chf = cd['comorb_has_chf']
        ckd = cd['comorb_has_ckd']

        for i in range(n_rows):
            tokens = [
                f"d1_{d1g[i]}", f"d2_{d2g[i]}", f"d3_{d3g[i]}",
                f"p1_{p1[i]}", f"p2_{p2[i]}", f"p3_{p3[i]}",
                f"disch_{disch[i]}", f"src_{src[i]}"
            ]
            if ins[i] > 0.0:
                tokens.append("med_insulin")
            if ins_up[i] > 0.0:
                tokens.append("med_insulin_up")
            if a1c_hi[i] > 0.0:
                tokens.append("lab_a1c_high")
            if chg[i] > 0.0:
                tokens.append("med_change")
            if frail[i] >= 2.0:
                tokens.append("geriatric_frail")
            if cd['util_total_visits'][i] >= 5.0:
                tokens.append("visits_high")
            if cd['time_in_hospital'][i] >= 7.0:
                tokens.append("hosp_long")
            if cd['num_medications'][i] >= 15.0:
                tokens.append("polypharmacy")
            if chf[i] > 0.0:
                tokens.append("chf")
            if ckd[i] > 0.0:
                tokens.append("ckd")
            doc_list.append(" ".join(tokens))

    tfidf = TfidfVectorizer(max_features=160, token_pattern=r'(?u)\b\w+\b', norm='l2')
    svd = TruncatedSVD(n_components=12, random_state=42)

    tr_tfidf = tfidf.fit_transform(train_docs)
    ev_tfidf = tfidf.transform(eval_docs)

    tr_svd = svd.fit_transform(tr_tfidf)
    ev_svd = svd.transform(ev_tfidf)

    for k in range(12):
        k_name = f'svd_phenotype_{k}'
        train_data[k_name] = tr_svd[:, k].astype(np.float64)
        eval_data[k_name] = ev_svd[:, k].astype(np.float64)
        extra_numeric.append(k_name)

    # 9. Out-of-fold Smoothed Target Encoding on key and interaction features
    for col_dict in [train_data, eval_data]:
        col_dict['cross_diag1_disch'] = (
            col_dict['diag_1_group'].astype(str) + '__' + col_dict['grouped_discharge'].astype(str)
        )
        col_dict['cross_adm_disch'] = (
            col_dict['grouped_adm_source'].astype(str) + '__' + col_dict['grouped_discharge'].astype(str)
        )
        col_dict['cross_age_disch'] = (
            col_dict['age'].astype(str) + '__' + col_dict['grouped_discharge'].astype(str)
        )
        col_dict['cross_diag1_diag2'] = (
            col_dict['diag_1_prefix'].astype(str) + '__' + col_dict['diag_2_prefix'].astype(str)
        )
        col_dict['cross_diag1_diag3'] = (
            col_dict['diag_1_prefix'].astype(str) + '__' + col_dict['diag_3_prefix'].astype(str)
        )
        col_dict['cross_diag2_diag3'] = (
            col_dict['diag_2_prefix'].astype(str) + '__' + col_dict['diag_3_prefix'].astype(str)
        )
        col_dict['cross_payer_specialty'] = (
            col_dict['payer_code'].astype(str) + '__' + col_dict['medical_specialty'].astype(str)
        )
        col_dict['cross_age_specialty'] = (
            col_dict['age'].astype(str) + '__' + col_dict['medical_specialty'].astype(str)
        )
        col_dict['cross_disch_specialty'] = (
            col_dict['grouped_discharge'].astype(str) + '__' + col_dict['medical_specialty'].astype(str)
        )

    def compute_oof_te(tr_arr, y_tr, ev_arr, smoothing=30.0, n_splits=5, seed=42):
        prior = float(np.mean(y_tr))
        tr_te = np.full(len(tr_arr), prior, dtype=np.float64)
        ev_te = np.full(len(ev_arr), prior, dtype=np.float64)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(tr_arr, y_tr):
            sub_c = tr_arr[train_idx]
            sub_y = y_tr[train_idx]
            v_c = tr_arr[val_idx]

            counts = {}
            sums = {}
            for c, y in zip(sub_c, sub_y):
                counts[c] = counts.get(c, 0) + 1
                sums[c] = sums.get(c, 0.0) + y

            vals = np.empty(len(val_idx), dtype=np.float64)
            for i, c in enumerate(v_c):
                cnt = counts.get(c, 0)
                vals[i] = (sums[c] + smoothing * prior) / (cnt + smoothing) if cnt > 0 else prior
            tr_te[val_idx] = vals

        full_counts = {}
        full_sums = {}
        for c, y in zip(tr_arr, y_tr):
            full_counts[c] = full_counts.get(c, 0) + 1
            full_sums[c] = full_sums.get(c, 0.0) + y

        for i, c in enumerate(ev_arr):
            cnt = full_counts.get(c, 0)
            ev_te[i] = (full_sums[c] + smoothing * prior) / (cnt + smoothing) if cnt > 0 else prior

        return tr_te, ev_te

    te_targets = [
        'discharge_disposition_id', 'admission_source_id', 'admission_type_id',
        'payer_code', 'medical_specialty',
        'diag_1_prefix', 'diag_2_prefix', 'diag_3_prefix',
        'grouped_discharge', 'grouped_adm_source',
        'cross_diag1_disch', 'cross_adm_disch', 'cross_age_disch',
        'cross_diag1_diag2', 'cross_diag1_diag3', 'cross_diag2_diag3',
        'cross_payer_specialty', 'cross_age_specialty', 'cross_disch_specialty'
    ]
    for c_name in te_targets:
        if c_name in train_data:
            tr_te, ev_te = compute_oof_te(train_data[c_name], y_train, eval_data[c_name], smoothing=30.0)
            te_col_name = f'te_{c_name}'
            train_data[te_col_name] = tr_te
            eval_data[te_col_name] = ev_te
            extra_numeric.append(te_col_name)

    # 10. Tabular Encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_data, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)

    # 11. Heterogeneous Multi-Model Gradient Boosting Ensemble with Rank Blending
    from scipy.stats import rankdata

    lgb_models = [
        lgb.LGBMClassifier(
            n_estimators=450,
            learning_rate=0.02,
            num_leaves=18,
            min_child_samples=45,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        # Extremely Randomized Trees boosting mode for tree diversity
        lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.02,
            num_leaves=26,
            min_child_samples=55,
            extra_trees=True,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.5,
            reg_lambda=8.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=360,
            learning_rate=0.02,
            num_leaves=34,
            min_child_samples=80,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=1.0,
            reg_lambda=12.0,
            random_state=456,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=380,
            learning_rate=0.025,
            num_leaves=22,
            min_child_samples=65,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=3.0,
            random_state=789,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    xgb_models = [
        xgb.XGBClassifier(
            n_estimators=380,
            learning_rate=0.02,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=1,
            eval_metric='logloss',
        ),
        xgb.XGBClassifier(
            n_estimators=320,
            learning_rate=0.02,
            max_depth=5,
            min_child_weight=6,
            subsample=0.85,
            colsample_bytree=0.6,
            reg_alpha=1.2,
            reg_lambda=10.0,
            random_state=123,
            n_jobs=1,
            eval_metric='logloss',
        )
    ]

    all_models = lgb_models + xgb_models
    model_weights = [0.20, 0.20, 0.18, 0.14, 0.15, 0.13]

    n_eval = len(x_eval)
    blended_rank = np.zeros(n_eval, dtype=np.float64)
    blended_prob = np.zeros(n_eval, dtype=np.float64)

    for i, model in enumerate(all_models):
        w = model_weights[i]
        if i < len(lgb_models):
            model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        else:
            model.fit(x_train, y_train)

        raw_pred = model.predict_proba(x_eval)[:, 1]
        r_pred = (rankdata(raw_pred) - 0.5) / n_eval
        blended_rank += w * r_pred
        blended_prob += w * raw_pred

    # Rank-percentile calibrated blend: preserves non-parametric rank alignment while preserving probability tails
    preds = 0.70 * blended_rank + 0.30 * blended_prob
    preds = np.nan_to_num(preds, nan=0.114, posinf=1.0, neginf=0.0)
    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
