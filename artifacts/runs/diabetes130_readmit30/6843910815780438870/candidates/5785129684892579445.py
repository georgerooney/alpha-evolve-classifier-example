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


from sklearn.decomposition import TruncatedSVD
from scipy.stats import rankdata
import lightgbm as lgb
import xgboost as xgb

try:
    from clinical_toolkit import TabularEncoder, icd9_group, icd9_comorbidities, oof_target_encode
except ImportError:
    pass


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

    # 1. Comorbidity extraction (Charlson index features)
    cm_train = icd9_comorbidities(train['diag_1'], train['diag_2'], train['diag_3'])
    cm_eval = icd9_comorbidities(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    for k, v in cm_train.items():
        train_dict[k] = v.astype(float)
        eval_dict[k] = cm_eval[k].astype(float)
        extra_numeric.append(k)

    # Weighted Charlson Comorbidity Index
    def compute_cci(d: Dict[str, np.ndarray]) -> np.ndarray:
        w1 = (
            d['cm_mi'] + d['cm_chf'] + d['cm_pvd'] + d['cm_cerebrovascular'] +
            d['cm_dementia'] + d['cm_pulmonary'] + d['cm_rheumatic'] +
            d['cm_peptic_ulcer'] + d['cm_liver']
        )
        w2 = 2.0 * (d['cm_hemiplegia'] + d['cm_renal'] + d['cm_cancer'] + d['cm_diabetes_complicated'])
        w3 = 4.0 * (d['cm_metastatic'] + d['cm_hiv'])
        return (w1 + w2 + w3).astype(float)

    train_dict['charlson_weighted_score'] = compute_cci(train_dict)
    eval_dict['charlson_weighted_score'] = compute_cci(eval_dict)
    extra_numeric.append('charlson_weighted_score')

    # Continuous age mapping for clinical indices
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    age_tr = np.array([age_map.get(a, 60.0) for a in train['age']], dtype=float)
    age_ev = np.array([age_map.get(a, 60.0) for a in eval_cols['age']], dtype=float)

    # Charlson Age-Comorbidity Index (CACI)
    def compute_caci(cci: np.ndarray, age_v: np.ndarray) -> np.ndarray:
        age_pts = np.clip(np.floor((age_v - 40.0) / 10.0), 0.0, 4.0)
        return cci + age_pts

    train_dict['caci_score'] = compute_caci(train_dict['charlson_weighted_score'], age_tr)
    eval_dict['caci_score'] = compute_caci(eval_dict['charlson_weighted_score'], age_ev)
    extra_numeric.append('caci_score')

    train_dict['cardiorenal_risk'] = ((train_dict['cm_chf'] > 0) & (train_dict['cm_renal'] > 0)).astype(float)
    eval_dict['cardiorenal_risk'] = ((eval_dict['cm_chf'] > 0) & (eval_dict['cm_renal'] > 0)).astype(float)
    extra_numeric.append('cardiorenal_risk')

    train_dict['is_multimorbid'] = (train_dict['cm_count'] >= 3.0).astype(float)
    eval_dict['is_multimorbid'] = (eval_dict['cm_count'] >= 3.0).astype(float)
    extra_numeric.append('is_multimorbid')

    train_dict['is_severely_multimorbid'] = (train_dict['cm_count'] >= 5.0).astype(float)
    eval_dict['is_severely_multimorbid'] = (eval_dict['cm_count'] >= 5.0).astype(float)
    extra_numeric.append('is_severely_multimorbid')

    train_dict['cardiorenal_metabolic_triad'] = (
        (train_dict['cm_chf'] > 0) & (train_dict['cm_renal'] > 0) & (train_dict['cm_diabetes_complicated'] > 0)
    ).astype(float)
    eval_dict['cardiorenal_metabolic_triad'] = (
        (eval_dict['cm_chf'] > 0) & (eval_dict['cm_renal'] > 0) & (eval_dict['cm_diabetes_complicated'] > 0)
    ).astype(float)
    extra_numeric.append('cardiorenal_metabolic_triad')

    train_dict['copd_chf_multimorbid'] = (
        (train_dict['cm_chf'] > 0) & (train_dict['cm_pulmonary'] > 0)
    ).astype(float)
    eval_dict['copd_chf_multimorbid'] = (
        (eval_dict['cm_chf'] > 0) & (eval_dict['cm_pulmonary'] > 0)
    ).astype(float)
    extra_numeric.append('copd_chf_multimorbid')

    train_dict['vascular_diabetes_risk'] = (
        ((train_dict['cm_pvd'] > 0) | (train_dict['cm_cerebrovascular'] > 0)) & (train_dict['cm_diabetes_complicated'] > 0)
    ).astype(float)
    eval_dict['vascular_diabetes_risk'] = (
        ((eval_dict['cm_pvd'] > 0) | (eval_dict['cm_cerebrovascular'] > 0)) & (eval_dict['cm_diabetes_complicated'] > 0)
    ).astype(float)
    extra_numeric.append('vascular_diabetes_risk')

    # 2. ICD-9 Chapter groups and 3-digit prefixes
    def extract_icd9_prefix(val: Any) -> str:
        if val is None or not isinstance(val, str) or val in ('None', '?'):
            return 'missing'
        dot_idx = val.find('.')
        return val[:dot_idx] if dot_idx != -1 else val

    for col in ['diag_1', 'diag_2', 'diag_3']:
        grp_col = f"{col}_group"
        train_dict[grp_col] = np.array([icd9_group(c) for c in train[col]], dtype=object)
        eval_dict[grp_col] = np.array([icd9_group(c) for c in eval_cols[col]], dtype=object)
        extra_categorical.append(grp_col)

        pfx_col = f"{col}_prefix"
        train_dict[pfx_col] = np.array([extract_icd9_prefix(c) for c in train[col]], dtype=object)
        eval_dict[pfx_col] = np.array([extract_icd9_prefix(c) for c in eval_cols[col]], dtype=object)
        extra_categorical.append(pfx_col)

    # Granular clinical condition matching across diagnoses
    def match_prefix_tuple(d1: np.ndarray, d2: np.ndarray, d3: np.ndarray, pfxs: tuple) -> (np.ndarray, np.ndarray):
        n = len(d1)
        any_match = np.zeros(n, dtype=float)
        prim_match = np.zeros(n, dtype=float)
        for i in range(n):
            c1 = d1[i]
            c2 = d2[i]
            c3 = d3[i]
            if c1 is not None and isinstance(c1, str) and c1.startswith(pfxs):
                prim_match[i] = 1.0
                any_match[i] = 1.0
                continue
            if (c2 is not None and isinstance(c2, str) and c2.startswith(pfxs)) or \
               (c3 is not None and isinstance(c3, str) and c3.startswith(pfxs)):
                any_match[i] = 1.0
        return any_match, prim_match

    clinical_conditions = {
        'cond_heart_failure': ('428',),
        'cond_diabetes': ('250',),
        'cond_dka_hyperosmolar': ('250.1', '250.2', '250.3'),
        'cond_dm_renal': ('250.4',),
        'cond_dm_neuro': ('250.6',),
        'cond_dm_pvd': ('250.7',),
        'cond_dm_other': ('250.8',),
        'cond_hypoglycemia': ('251.0', '251.1', '251.2'),
        'cond_pneumonia': ('480', '481', '482', '483', '484', '485', '486'),
        'cond_copd': ('491', '492', '493', '494', '496'),
        'cond_renal': ('584', '585', '586'),
        'cond_aki': ('584',),
        'cond_ckd': ('585',),
        'cond_esrd': ('585.6',),
        'cond_fluid_electrolyte': ('276',),
        'cond_hypertension': ('401', '402', '403', '404', '405'),
        'cond_dysrhythmia': ('427',),
        'cond_cad_ami': ('410', '411', '412', '413', '414'),
        'cond_uti': ('599',),
        'cond_cellulitis_ulcer': ('681', '682', '707'),
        'cond_septicemia': ('038', '995.9'),
    }

    for cname, pfxs in clinical_conditions.items():
        tr_any, tr_prim = match_prefix_tuple(train['diag_1'], train['diag_2'], train['diag_3'], pfxs)
        ev_any, ev_prim = match_prefix_tuple(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'], pfxs)
        train_dict[cname] = tr_any
        eval_dict[cname] = ev_any
        extra_numeric.append(cname)

        p_name = f"{cname}_primary"
        train_dict[p_name] = tr_prim
        eval_dict[p_name] = ev_prim
        extra_numeric.append(p_name)

    # Diabetic control flags (ICD-9 250.x2 / 250.x3 uncontrolled, 250.x1 / 250.x3 Type 1)
    def check_dm_subtypes(d1: np.ndarray, d2: np.ndarray, d3: np.ndarray) -> (np.ndarray, np.ndarray):
        n = len(d1)
        uncontrolled = np.zeros(n, dtype=float)
        type1 = np.zeros(n, dtype=float)
        for i in range(n):
            for code in (d1[i], d2[i], d3[i]):
                if code is not None and isinstance(code, str) and code.startswith('250.'):
                    last_ch = code[-1]
                    if last_ch in ('2', '3'):
                        uncontrolled[i] = 1.0
                    if last_ch in ('1', '3'):
                        type1[i] = 1.0
        return uncontrolled, type1

    tr_unc, tr_t1 = check_dm_subtypes(train['diag_1'], train['diag_2'], train['diag_3'])
    ev_unc, ev_t1 = check_dm_subtypes(eval_cols['diag_1'], eval_cols['diag_2'], eval_cols['diag_3'])
    train_dict['is_uncontrolled_dm'] = tr_unc
    eval_dict['is_uncontrolled_dm'] = ev_unc
    extra_numeric.append('is_uncontrolled_dm')
    train_dict['is_type1_dm'] = tr_t1
    eval_dict['is_type1_dm'] = ev_t1
    extra_numeric.append('is_type1_dm')

    # Diagnostic chapter group indicators
    for grp in ['circulatory', 'respiratory', 'diabetes', 'digestive', 'genitourinary', 'injury', 'musculoskeletal', 'neoplasms']:
        col_name = f"has_group_{grp}"
        train_dict[col_name] = (
            (train_dict['diag_1_group'] == grp) |
            (train_dict['diag_2_group'] == grp) |
            (train_dict['diag_3_group'] == grp)
        ).astype(float)
        eval_dict[col_name] = (
            (eval_dict['diag_1_group'] == grp) |
            (eval_dict['diag_2_group'] == grp) |
            (eval_dict['diag_3_group'] == grp)
        ).astype(float)
        extra_numeric.append(col_name)

        prim_name = f"is_primary_{grp}"
        train_dict[prim_name] = (train_dict['diag_1_group'] == grp).astype(float)
        eval_dict[prim_name] = (eval_dict['diag_1_group'] == grp).astype(float)
        extra_numeric.append(prim_name)

    # 3. High-cardinality out-of-fold target encoding
    te_candidates = [
        'diag_1', 'diag_2', 'diag_3',
        'diag_1_prefix', 'diag_2_prefix', 'diag_3_prefix',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'medical_specialty', 'payer_code', 'race'
    ]
    for col in te_candidates:
        if col in train_dict:
            te_tr, te_ev = oof_target_encode(
                train_dict[col], y_train, eval_dict[col], n_folds=5, smoothing=25.0, seed=42
            )
            cname = f"{col}_te"
            train_dict[cname] = te_tr.astype(float)
            eval_dict[cname] = te_ev.astype(float)
            extra_numeric.append(cname)

    # 4. Clinical domain features: utilization, LACE index, stay intensity
    def derive_clinical_numeric(d: Dict[str, np.ndarray], age_vals: np.ndarray, cci_scores: np.ndarray) -> Dict[str, np.ndarray]:
        feats = {}
        stay = np.maximum(np.nan_to_num(d['time_in_hospital'].astype(float), nan=1.0), 1.0)
        n_inpatient = np.nan_to_num(d['number_inpatient'].astype(float), nan=0.0)
        n_emergency = np.nan_to_num(d['number_emergency'].astype(float), nan=0.0)
        n_outpatient = np.nan_to_num(d['number_outpatient'].astype(float), nan=0.0)

        total_prior = n_inpatient + n_emergency + n_outpatient
        feats['total_prior_visits'] = total_prior
        feats['inpatient_ratio'] = n_inpatient / (total_prior + 1.0)
        feats['emergency_ratio'] = n_emergency / (total_prior + 1.0)
        feats['outpatient_ratio'] = n_outpatient / (total_prior + 1.0)
        feats['acute_prior_visits'] = n_inpatient + n_emergency
        feats['acute_utilization_ratio'] = feats['acute_prior_visits'] / (total_prior + 1.0)

        feats['has_prior_inpatient'] = (n_inpatient > 0.0).astype(float)
        feats['has_multiple_inpatient'] = (n_inpatient >= 2.0).astype(float)
        feats['frequent_inpatient'] = (n_inpatient >= 3.0).astype(float)
        feats['has_prior_emergency'] = (n_emergency > 0.0).astype(float)
        feats['has_multiple_emergency'] = (n_emergency >= 2.0).astype(float)
        feats['has_prior_outpatient'] = (n_outpatient > 0.0).astype(float)
        feats['has_prior_visits'] = (total_prior > 0.0).astype(float)
        feats['high_utilizer'] = ((n_inpatient >= 2.0) | (total_prior >= 5.0)).astype(float)
        feats['is_first_visit'] = (total_prior == 0.0).astype(float)
        feats['has_only_acute_visits'] = (((n_inpatient > 0.0) | (n_emergency > 0.0)) & (n_outpatient == 0.0)).astype(float)

        feats['log_inpatient'] = np.log1p(n_inpatient)
        feats['log_emergency'] = np.log1p(n_emergency)
        feats['log_total_prior'] = np.log1p(total_prior)
        feats['inpatient_times_stay'] = n_inpatient * stay
        feats['inpatient_times_cci'] = n_inpatient * cci_scores

        num_meds = np.nan_to_num(d['num_medications'].astype(float), nan=0.0)
        num_labs = np.nan_to_num(d['num_lab_procedures'].astype(float), nan=0.0)
        num_procs = np.nan_to_num(d['num_procedures'].astype(float), nan=0.0)
        num_diags = np.nan_to_num(d['number_diagnoses'].astype(float), nan=0.0)

        feats['labs_per_day'] = num_labs / stay
        feats['meds_per_day'] = num_meds / stay
        feats['procs_per_day'] = num_procs / stay
        feats['diags_per_day'] = num_diags / stay
        feats['med_proc_ratio'] = num_meds / (num_procs + 1.0)
        feats['lab_med_ratio'] = num_labs / (num_meds + 1.0)
        feats['meds_per_diag'] = num_meds / (num_diags + 1.0)
        feats['stay_per_diag'] = stay / (num_diags + 1.0)
        feats['total_clinical_events'] = num_labs + num_meds + num_procs
        feats['events_per_day'] = feats['total_clinical_events'] / stay
        feats['clinical_complexity'] = (stay / 4.0) + (n_inpatient * 2.0) + (num_meds / 15.0) + (num_diags / 8.0)

        feats['long_stay'] = (stay >= 7.0).astype(float)
        feats['short_stay'] = (stay <= 2.0).astype(float)

        # LACE Readmission Risk Index (Length of stay, Acuity, Comorbidity, Emergency visits)
        l_pts = np.select([stay < 2, stay < 3, stay < 4, stay < 7, stay < 14], [1.0, 2.0, 3.0, 4.0, 5.0], default=7.0)
        a_pts = np.where(d['admission_type_id'] == '1', 3.0, 0.0)
        c_pts = np.select([cci_scores < 1, cci_scores < 2, cci_scores < 3, cci_scores < 4], [0.0, 1.0, 2.0, 3.0], default=5.0)
        e_pts = np.select([n_emergency < 1, n_emergency < 2, n_emergency < 3, n_emergency < 4], [0.0, 1.0, 2.0, 3.0], default=4.0)
        lace_score = l_pts + a_pts + c_pts + e_pts
        feats['lace_index'] = lace_score
        feats['high_lace_risk'] = (lace_score >= 10.0).astype(float)
        feats['lace_times_inpatient'] = (lace_score / 10.0) * n_inpatient

        # Demographics
        feats['age_num'] = age_vals
        feats['is_elderly'] = (age_vals >= 70.0).astype(float)
        feats['is_very_elderly'] = (age_vals >= 80.0).astype(float)
        feats['is_nonagenarian'] = (age_vals >= 90.0).astype(float)
        feats['age_times_inpatient'] = (age_vals / 50.0) * n_inpatient
        feats['is_female'] = (d['gender'] == 'Female').astype(float)

        # Admission & discharge flags
        feats['is_discharge_home'] = (d['discharge_disposition_id'] == '1').astype(float)
        feats['is_discharge_snf'] = (d['discharge_disposition_id'] == '3').astype(float)
        feats['is_discharge_home_health'] = (d['discharge_disposition_id'] == '6').astype(float)
        feats['is_discharge_ama'] = (d['discharge_disposition_id'] == '7').astype(float)
        feats['is_discharge_rehab_ltc'] = np.isin(d['discharge_disposition_id'], ['4', '5', '22']).astype(float)
        feats['is_discharge_transfer'] = np.isin(d['discharge_disposition_id'], ['2', '4', '5', '18', '22', '25']).astype(float)
        feats['is_high_risk_discharge'] = np.isin(d['discharge_disposition_id'], ['3', '6', '7', '22', '28']).astype(float)
        feats['high_risk_disch_inpatient'] = feats['is_high_risk_discharge'] * feats['has_prior_inpatient']
        feats['snf_elderly'] = feats['is_discharge_snf'] * feats['is_elderly']
        feats['home_health_elderly'] = feats['is_discharge_home_health'] * feats['is_elderly']

        feats['is_emergency_admission'] = (d['admission_type_id'] == '1').astype(float)
        feats['is_urgent_admission'] = (d['admission_type_id'] == '2').astype(float)
        feats['is_elective'] = (d['admission_type_id'] == '3').astype(float)
        feats['is_emergency_source'] = (d['admission_source_id'] == '7').astype(float)
        feats['is_transfer_source'] = np.isin(d['admission_source_id'], ['4', '5', '6']).astype(float)

        # Lab test indicators and monotonic ordinal values
        feats['a1c_tested'] = (d['A1Cresult'] != 'None').astype(float)
        feats['a1c_abnormal'] = np.isin(d['A1Cresult'], ['>7', '>8']).astype(float)
        feats['a1c_severe'] = (d['A1Cresult'] == '>8').astype(float)
        feats['a1c_normal'] = (d['A1Cresult'] == 'Norm').astype(float)
        a1c_num_map = {'None': 0.0, 'Norm': 1.0, '>7': 2.0, '>8': 3.0}
        feats['a1c_ordinal'] = np.array([a1c_num_map.get(v, 0.0) for v in d['A1Cresult']], dtype=float)

        feats['glu_tested'] = (d['max_glu_serum'] != 'None').astype(float)
        feats['glu_abnormal'] = np.isin(d['max_glu_serum'], ['>200', '>300']).astype(float)
        feats['glu_severe'] = (d['max_glu_serum'] == '>300').astype(float)
        feats['glu_normal'] = (d['max_glu_serum'] == 'Norm').astype(float)
        glu_num_map = {'None': 0.0, 'Norm': 1.0, '>200': 2.0, '>300': 3.0}
        feats['glu_ordinal'] = np.array([glu_num_map.get(v, 0.0) for v in d['max_glu_serum']], dtype=float)

        # Informative payer & specialty clusters
        feats['payer_medicare'] = (d['payer_code'] == 'MC').astype(float)
        feats['payer_medicaid'] = (d['payer_code'] == 'MD').astype(float)
        feats['payer_commercial'] = np.isin(d['payer_code'], ['BC', 'CP', 'HM', 'OG', 'UN', 'PO']).astype(float)
        feats['payer_self_pay'] = np.isin(d['payer_code'], ['SP', 'SI']).astype(float)
        feats['payer_missing'] = np.isin(d['payer_code'], [None, '?', 'missing']).astype(float)

        spec = d['medical_specialty']
        feats['spec_cardio'] = np.array([1.0 if s is not None and 'Cardio' in str(s) else 0.0 for s in spec], dtype=float)
        feats['spec_internal'] = np.isin(spec, ['InternalMedicine', 'Family/GeneralPractice']).astype(float)
        feats['spec_surgery'] = np.array([1.0 if s is not None and 'Surg' in str(s) else 0.0 for s in spec], dtype=float)
        feats['spec_emergency'] = np.array([1.0 if s is not None and 'Emergency' in str(s) else 0.0 for s in spec], dtype=float)
        feats['spec_missing'] = np.isin(spec, [None, '?', 'missing']).astype(float)

        return feats

    clin_tr = derive_clinical_numeric(train, age_tr, train_dict['charlson_weighted_score'])
    clin_ev = derive_clinical_numeric(eval_cols, age_ev, eval_dict['charlson_weighted_score'])
    for k in clin_tr:
        train_dict[k] = clin_tr[k]
        eval_dict[k] = clin_ev[k]
        extra_numeric.append(k)

    # 5. Medication features
    med_list = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    present_meds = [m for m in med_list if m in train]

    def derive_med_features(d: Dict[str, np.ndarray], cm_renal_col: np.ndarray) -> Dict[str, np.ndarray]:
        res = {}
        if present_meds:
            m_stack = np.column_stack([d[m] for m in present_meds])
            is_active = (m_stack != 'No') & (m_stack != None)
            res['num_active_meds'] = np.sum(is_active, axis=1).astype(float)
            res['num_med_up'] = np.sum(m_stack == 'Up', axis=1).astype(float)
            res['num_med_down'] = np.sum(m_stack == 'Down', axis=1).astype(float)
            res['num_med_steady'] = np.sum(m_stack == 'Steady', axis=1).astype(float)
            res['num_med_changes'] = res['num_med_up'] + res['num_med_down']

        has_ins = np.zeros(len(d['age']), dtype=float)
        if 'insulin' in d:
            has_ins = ((d['insulin'] != 'No') & (d['insulin'] != None)).astype(float)
            res['has_insulin'] = has_ins
            res['insulin_up'] = (d['insulin'] == 'Up').astype(float)
            res['insulin_down'] = (d['insulin'] == 'Down').astype(float)
            res['insulin_steady'] = (d['insulin'] == 'Steady').astype(float)
            res['insulin_changed'] = ((d['insulin'] == 'Up') | (d['insulin'] == 'Down')).astype(float)

        has_met = np.zeros(len(d['age']), dtype=float)
        if 'metformin' in d:
            has_met = ((d['metformin'] != 'No') & (d['metformin'] != None)).astype(float)
            res['has_metformin'] = has_met
            if 'insulin' in d:
                res['insulin_and_metformin'] = has_ins * has_met

        su_names = [
            'glimepiride', 'glipizide', 'glyburide', 'chlorpropamide',
            'tolbutamide', 'acetohexamide', 'tolazamide',
            'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone'
        ]
        pres_su = [m for m in su_names if m in d]
        has_su = np.zeros(len(d['age']), dtype=float)
        if pres_su:
            su_stack = np.column_stack([d[m] for m in pres_su])
            has_su = np.any((su_stack != 'No') & (su_stack != None), axis=1).astype(float)
            res['has_sulfonylurea'] = has_su
            if 'insulin' in d:
                res['insulin_and_sulfonylurea'] = has_ins * has_su
            res['su_renal_hypoglycemia_risk'] = has_su * (cm_renal_col > 0).astype(float)

        has_chg = (d['change'] == 'Ch').astype(float) if 'change' in d else np.zeros(len(d['age']), dtype=float)
        res['has_med_change'] = has_chg
        res['has_med_down'] = (res.get('num_med_down', np.zeros(len(d['age']))) > 0.0).astype(float)
        res['has_med_up'] = (res.get('num_med_up', np.zeros(len(d['age']))) > 0.0).astype(float)

        if 'diabetesMed' in d:
            res['on_diabetes_med'] = (d['diabetesMed'] == 'Yes').astype(float)

        # Regimen complexity profiles
        res['insulin_only'] = (has_ins > 0.0) & (res.get('num_active_meds', np.zeros(len(d['age']))) == 1.0)
        res['insulin_only'] = res['insulin_only'].astype(float)
        res['oral_only'] = (has_ins == 0.0) & (res.get('num_active_meds', np.zeros(len(d['age']))) > 0.0)
        res['oral_only'] = res['oral_only'].astype(float)

        num_medications = np.nan_to_num(d['num_medications'].astype(float), nan=0.0)
        res['polypharmacy'] = (num_medications >= 16.0).astype(float)
        res['severe_polypharmacy'] = (num_medications >= 25.0).astype(float)

        # Clinical inertia marker (severe A1C but no medication adjustment)
        a1c_sev = (d['A1Cresult'] == '>8').astype(float)
        a1c_tst = (d['A1Cresult'] != 'None').astype(float)
        res['a1c_severe_no_change'] = a1c_sev * (1.0 - has_chg)
        res['a1c_tested_and_change'] = a1c_tst * has_chg

        return res

    med_tr = derive_med_features(train, train_dict['cm_renal'])
    med_ev = derive_med_features(eval_cols, eval_dict['cm_renal'])
    for k in med_tr:
        train_dict[k] = med_tr[k]
        eval_dict[k] = med_ev[k]
        extra_numeric.append(k)

    # Admission-Discharge cross interaction out-of-fold target encoding
    train_dict['adm_disch_interaction'] = np.array(
        [f"{s}_{d}" for s, d in zip(train['admission_source_id'], train['discharge_disposition_id'])],
        dtype=object
    )
    eval_dict['adm_disch_interaction'] = np.array(
        [f"{s}_{d}" for s, d in zip(eval_cols['admission_source_id'], eval_cols['discharge_disposition_id'])],
        dtype=object
    )
    extra_categorical.append('adm_disch_interaction')

    te_tr_adi, te_ev_adi = oof_target_encode(
        train_dict['adm_disch_interaction'], y_train, eval_dict['adm_disch_interaction'],
        n_folds=5, smoothing=25.0, seed=42
    )
    train_dict['adm_disch_interaction_te'] = te_tr_adi.astype(float)
    eval_dict['adm_disch_interaction_te'] = te_ev_adi.astype(float)
    extra_numeric.append('adm_disch_interaction_te')

    # Diagnosis-Discharge and Age-Discharge cross interaction target encodings
    train_dict['diag1_disch_interaction'] = np.array(
        [f"{g}_{d}" for g, d in zip(train_dict['diag_1_group'], train['discharge_disposition_id'])],
        dtype=object
    )
    eval_dict['diag1_disch_interaction'] = np.array(
        [f"{g}_{d}" for g, d in zip(eval_dict['diag_1_group'], eval_cols['discharge_disposition_id'])],
        dtype=object
    )
    te_tr_d1d, te_ev_d1d = oof_target_encode(
        train_dict['diag1_disch_interaction'], y_train, eval_dict['diag1_disch_interaction'],
        n_folds=5, smoothing=25.0, seed=42
    )
    train_dict['diag1_disch_interaction_te'] = te_tr_d1d.astype(float)
    eval_dict['diag1_disch_interaction_te'] = te_ev_d1d.astype(float)
    extra_numeric.append('diag1_disch_interaction_te')

    train_dict['age_disch_interaction'] = np.array(
        [f"{a}_{d}" for a, d in zip(train['age'], train['discharge_disposition_id'])],
        dtype=object
    )
    eval_dict['age_disch_interaction'] = np.array(
        [f"{a}_{d}" for a, d in zip(eval_cols['age'], eval_cols['discharge_disposition_id'])],
        dtype=object
    )
    te_tr_ad, te_ev_ad = oof_target_encode(
        train_dict['age_disch_interaction'], y_train, eval_dict['age_disch_interaction'],
        n_folds=5, smoothing=25.0, seed=42
    )
    train_dict['age_disch_interaction_te'] = te_tr_ad.astype(float)
    eval_dict['age_disch_interaction_te'] = te_ev_ad.astype(float)
    extra_numeric.append('age_disch_interaction_te')

    train_dict['diag1_diag2_group'] = np.array(
        [f"{g1}_{g2}" for g1, g2 in zip(train_dict['diag_1_group'], train_dict['diag_2_group'])],
        dtype=object
    )
    eval_dict['diag1_diag2_group'] = np.array(
        [f"{g1}_{g2}" for g1, g2 in zip(eval_dict['diag_1_group'], eval_dict['diag_2_group'])],
        dtype=object
    )
    te_tr_d12, te_ev_d12 = oof_target_encode(
        train_dict['diag1_diag2_group'], y_train, eval_dict['diag1_diag2_group'],
        n_folds=5, smoothing=25.0, seed=42
    )
    train_dict['diag1_diag2_group_te'] = te_tr_d12.astype(float)
    eval_dict['diag1_diag2_group_te'] = te_ev_d12.astype(float)
    extra_numeric.append('diag1_diag2_group_te')

    # 6. Standardized Low-Rank Phenotype Factorization (TruncatedSVD/PCA)
    svd_cols = [
        'cm_mi', 'cm_chf', 'cm_pvd', 'cm_cerebrovascular', 'cm_dementia',
        'cm_pulmonary', 'cm_rheumatic', 'cm_peptic_ulcer', 'cm_liver',
        'cm_hemiplegia', 'cm_renal', 'cm_cancer', 'cm_metastatic', 'cm_diabetes_complicated',
        'cardiorenal_risk', 'cond_heart_failure', 'cond_diabetes', 'cond_pneumonia',
        'cond_copd', 'cond_renal', 'cond_aki', 'cond_ckd', 'cond_dka_hyperosmolar',
        'has_group_circulatory', 'has_group_respiratory', 'has_group_diabetes',
        'has_group_digestive', 'has_group_genitourinary',
        'has_prior_inpatient', 'has_multiple_inpatient', 'has_prior_emergency',
        'is_discharge_home', 'is_discharge_snf', 'is_discharge_ama', 'is_discharge_home_health',
        'is_emergency_admission', 'a1c_tested', 'a1c_abnormal', 'glu_tested'
    ]
    if present_meds:
        for m in present_meds:
            tr_m_active = ((train[m] != 'No') & (train[m] != None)).astype(float)
            ev_m_active = ((eval_cols[m] != 'No') & (eval_cols[m] != None)).astype(float)
            m_col = f"svd_in_{m}"
            train_dict[m_col] = tr_m_active
            eval_dict[m_col] = ev_m_active
            svd_cols.append(m_col)

    M_tr = np.column_stack([train_dict[c] for c in svd_cols])
    M_ev = np.column_stack([eval_dict[c] for c in svd_cols])

    # Standardize columns so SVD extracts true multi-factor phenotype variance
    m_mean = np.nanmean(M_tr, axis=0)
    m_std = np.nanstd(M_tr, axis=0)
    m_std[m_std < 1e-6] = 1.0
    M_tr_norm = np.nan_to_num((M_tr - m_mean) / m_std, nan=0.0)
    M_ev_norm = np.nan_to_num((M_ev - m_mean) / m_std, nan=0.0)

    n_svd_components = 8
    svd = TruncatedSVD(n_components=n_svd_components, algorithm='randomized', random_state=42)
    svd_tr = svd.fit_transform(M_tr_norm)
    svd_ev = svd.transform(M_ev_norm)

    for i in range(n_svd_components):
        sname = f"svd_phenotype_{i}"
        train_dict[sname] = svd_tr[:, i].astype(float)
        eval_dict[sname] = svd_ev[:, i].astype(float)
        extra_numeric.append(sname)

    # 7. Tabular encoding
    encoder = TabularEncoder(min_count=40)
    encoder.fit(task, train_dict, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_dict)
    x_eval = encoder.transform(eval_dict)

    # Prepare numeric-only feature matrix for XGBoost (excludes arbitrary integer codes)
    xgb_feature_mask = np.ones(x_train.shape[1], dtype=bool)
    xgb_feature_mask[encoder.categorical_indices] = False
    x_train_xgb = x_train[:, xgb_feature_mask]
    x_eval_xgb = x_eval[:, xgb_feature_mask]

    # 8. Diverse multi-model gradient boosting ensemble (LightGBM + XGBoost)
    lgb_models = [
        lgb.LGBMClassifier(
            n_estimators=360,
            learning_rate=0.024,
            num_leaves=31,
            min_child_samples=100,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=2.0,
            reg_alpha=0.1,
            cat_smooth=40.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.020,
            num_leaves=42,
            min_child_samples=80,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_lambda=3.5,
            reg_alpha=0.5,
            cat_smooth=30.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=420,
            learning_rate=0.020,
            num_leaves=22,
            min_child_samples=140,
            subsample=0.70,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=2.5,
            reg_alpha=0.2,
            cat_smooth=50.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=360,
            learning_rate=0.024,
            num_leaves=35,
            min_child_samples=90,
            extra_trees=True,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=2.0,
            reg_alpha=0.2,
            cat_smooth=35.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
    ]

    xgb_models = [
        xgb.XGBClassifier(
            n_estimators=260,
            learning_rate=0.028,
            max_depth=4,
            subsample=0.80,
            colsample_bytree=0.65,
            reg_lambda=4.0,
            reg_alpha=0.5,
            tree_method='hist',
            random_state=42,
            n_jobs=1,
            eval_metric='logloss',
        ),
        xgb.XGBClassifier(
            n_estimators=260,
            learning_rate=0.022,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.55,
            reg_lambda=2.5,
            reg_alpha=0.3,
            tree_method='hist',
            random_state=2024,
            n_jobs=1,
            eval_metric='logloss',
        ),
    ]

    n_eval = len(x_eval)
    preds_rank = np.zeros(n_eval, dtype=float)

    # Fit LightGBM models with native categorical splits
    for model in lgb_models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        p = model.predict_proba(x_eval)[:, 1]
        preds_rank += (rankdata(p) - 0.5) / n_eval

    # Fit XGBoost models on numeric-only features (target encodings & clinical domain signals)
    xgb_weight = 0.85
    for model in xgb_models:
        model.fit(x_train_xgb, y_train)
        p = model.predict_proba(x_eval_xgb)[:, 1]
        preds_rank += xgb_weight * ((rankdata(p) - 0.5) / n_eval)

    total_weight = len(lgb_models) + (len(xgb_models) * xgb_weight)
    preds = preds_rank / total_weight
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
