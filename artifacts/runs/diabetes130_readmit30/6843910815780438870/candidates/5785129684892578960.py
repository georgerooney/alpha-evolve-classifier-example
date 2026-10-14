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
from sklearn.ensemble import HistGradientBoostingClassifier
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

    # Weighted Charlson Comorbidity Index & cardiorenal syndrome flags
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

    train_dict['cardiorenal_risk'] = ((train_dict['cm_chf'] > 0) & (train_dict['cm_renal'] > 0)).astype(float)
    eval_dict['cardiorenal_risk'] = ((eval_dict['cm_chf'] > 0) & (eval_dict['cm_renal'] > 0)).astype(float)
    extra_numeric.append('cardiorenal_risk')

    train_dict['is_multimorbid'] = (train_dict['cm_count'] >= 3.0).astype(float)
    eval_dict['is_multimorbid'] = (eval_dict['cm_count'] >= 3.0).astype(float)
    extra_numeric.append('is_multimorbid')

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

    # Additional high-risk comorbidity combinations
    train_dict['cardiorenal_anemia_syndrome'] = (
        (train_dict['cm_chf'] > 0) & (train_dict['cm_renal'] > 0) & (train_dict['cm_rheumatic'] == 0)
    ).astype(float)
    eval_dict['cardiorenal_anemia_syndrome'] = (
        (eval_dict['cm_chf'] > 0) & (eval_dict['cm_renal'] > 0) & (eval_dict['cm_rheumatic'] == 0)
    ).astype(float)
    extra_numeric.append('cardiorenal_anemia_syndrome')

    train_dict['diabetic_renal_syndrome'] = (
        (train_dict['cm_diabetes_complicated'] > 0) & (train_dict['cm_renal'] > 0)
    ).astype(float)
    eval_dict['diabetic_renal_syndrome'] = (
        (eval_dict['cm_diabetes_complicated'] > 0) & (eval_dict['cm_renal'] > 0)
    ).astype(float)
    extra_numeric.append('diabetic_renal_syndrome')

    train_dict['diabetic_cardiac_syndrome'] = (
        (train_dict['cm_diabetes_complicated'] > 0) & (train_dict['cm_chf'] > 0)
    ).astype(float)
    eval_dict['diabetic_cardiac_syndrome'] = (
        (eval_dict['cm_diabetes_complicated'] > 0) & (eval_dict['cm_chf'] > 0)
    ).astype(float)
    extra_numeric.append('diabetic_cardiac_syndrome')

    train_dict['neurovascular_syndrome'] = (
        (train_dict['cm_cerebrovascular'] > 0) & (train_dict['cm_pvd'] > 0)
    ).astype(float)
    eval_dict['neurovascular_syndrome'] = (
        (eval_dict['cm_cerebrovascular'] > 0) & (eval_dict['cm_pvd'] > 0)
    ).astype(float)
    extra_numeric.append('neurovascular_syndrome')

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

    # 4-character ICD-9 subprefix (e.g. 250.1 DKA vs 250.0 uncomplicated)
    def extract_icd9_subprefix(val: Any) -> str:
        if val is None or not isinstance(val, str) or val in ('None', '?'):
            return 'missing'
        dot_idx = val.find('.')
        if dot_idx != -1 and len(val) > dot_idx + 1:
            return val[:dot_idx + 2]
        return val

    for col in ['diag_1', 'diag_2', 'diag_3']:
        sub_col = f"{col}_subprefix"
        train_dict[sub_col] = np.array([extract_icd9_subprefix(c) for c in train[col]], dtype=object)
        eval_dict[sub_col] = np.array([extract_icd9_subprefix(c) for c in eval_cols[col]], dtype=object)
        extra_categorical.append(sub_col)

    # Diabetes-specific acute complication markers
    def extract_diabetes_complications(d_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        res = {}
        for pfx, tag in [('250.1', 'dka'), ('250.2', 'hyperosmolar'), ('250.4', 'diab_renal'),
                         ('250.6', 'diab_neuro'), ('250.7', 'diab_pvd')]:
            m1 = np.array([isinstance(c, str) and c.startswith(pfx) for c in d_dict['diag_1_subprefix']])
            m2 = np.array([isinstance(c, str) and c.startswith(pfx) for c in d_dict['diag_2_subprefix']])
            m3 = np.array([isinstance(c, str) and c.startswith(pfx) for c in d_dict['diag_3_subprefix']])
            res[f"has_{tag}"] = (m1 | m2 | m3).astype(float)
            res[f"primary_{tag}"] = m1.astype(float)

        # Hypoglycemia (251.0, 251.1, 251.2)
        m1_hypo = np.isin(d_dict['diag_1_prefix'], ['251'])
        m2_hypo = np.isin(d_dict['diag_2_prefix'], ['251'])
        m3_hypo = np.isin(d_dict['diag_3_prefix'], ['251'])
        res['has_hypoglycemia'] = (m1_hypo | m2_hypo | m3_hypo).astype(float)

        # Uncontrolled diabetes (ICD-9 ending in 2 or 3)
        def is_uncontrolled(col: np.ndarray) -> np.ndarray:
            return np.array([
                1.0 if (c is not None and isinstance(c, str) and c.startswith('250') and c[-1] in ('2', '3')) else 0.0
                for c in col
            ], dtype=float)

        res['has_uncontrolled_dm'] = ((is_uncontrolled(d_dict['diag_1']) +
                                       is_uncontrolled(d_dict['diag_2']) +
                                       is_uncontrolled(d_dict['diag_3'])) > 0).astype(float)
        return res

    dm_comp_tr = extract_diabetes_complications(train_dict)
    dm_comp_ev = extract_diabetes_complications(eval_dict)
    for k in dm_comp_tr:
        train_dict[k] = dm_comp_tr[k]
        eval_dict[k] = dm_comp_ev[k]
        extra_numeric.append(k)

    # HRRP high-risk condition flags across all 3 diagnoses
    def check_diag_condition(d_dict: Dict[str, np.ndarray], codes: set) -> np.ndarray:
        m1 = np.isin(d_dict['diag_1_prefix'], list(codes))
        m2 = np.isin(d_dict['diag_2_prefix'], list(codes))
        m3 = np.isin(d_dict['diag_3_prefix'], list(codes))
        return (m1 | m2 | m3).astype(float)

    hrrp_conditions = {
        'cond_heart_failure': {'428'},
        'cond_diabetes': {'250'},
        'cond_pneumonia': {'480', '481', '482', '483', '484', '485', '486'},
        'cond_copd': {'491', '492', '493', '494', '496'},
        'cond_renal': {'584', '585', '586'},
        'cond_fluid_electrolyte': {'276'},
        'cond_hypertension': {'401', '402', '403', '404', '405'},
        'cond_dysrhythmia': {'427'},
        'cond_cad_ami': {'410', '411', '412', '413', '414'},
        'cond_uti': {'599'},
        'cond_cellulitis_ulcer': {'681', '682', '707'},
        'cond_septicemia': {'038'},
        'cond_anemia': {'280', '281', '282', '283', '284', '285'},
    }
    for cond_name, codes in hrrp_conditions.items():
        train_dict[cond_name] = check_diag_condition(train_dict, codes)
        eval_dict[cond_name] = check_diag_condition(eval_dict, codes)
        extra_numeric.append(cond_name)

        # Primary diagnosis condition flag (HRRP focus on primary admission cause)
        p_name = f"{cond_name}_primary"
        train_dict[p_name] = np.isin(train_dict['diag_1_prefix'], list(codes)).astype(float)
        eval_dict[p_name] = np.isin(eval_dict['diag_1_prefix'], list(codes)).astype(float)
        extra_numeric.append(p_name)

    # Aggregate HRRP burden score and condition count
    tr_hrrp_sum = np.sum([train_dict[c] for c in hrrp_conditions], axis=0).astype(float)
    ev_hrrp_sum = np.sum([eval_dict[c] for c in hrrp_conditions], axis=0).astype(float)
    train_dict['num_hrrp_conditions'] = tr_hrrp_sum
    eval_dict['num_hrrp_conditions'] = ev_hrrp_sum
    extra_numeric.append('num_hrrp_conditions')

    train_dict['has_multiple_hrrp'] = (tr_hrrp_sum >= 2.0).astype(float)
    eval_dict['has_multiple_hrrp'] = (ev_hrrp_sum >= 2.0).astype(float)
    extra_numeric.append('has_multiple_hrrp')

    tr_prim_hrrp = np.sum([train_dict[f"{c}_primary"] for c in hrrp_conditions], axis=0).astype(float)
    ev_prim_hrrp = np.sum([eval_dict[f"{c}_primary"] for c in hrrp_conditions], axis=0).astype(float)
    train_dict['is_primary_hrrp'] = (tr_prim_hrrp > 0.0).astype(float)
    eval_dict['is_primary_hrrp'] = (ev_prim_hrrp > 0.0).astype(float)
    extra_numeric.append('is_primary_hrrp')

    # CHF and Hypertension co-occurrence
    train_dict['chf_and_hypertension'] = (
        (train_dict['cond_heart_failure'] > 0) & (train_dict['cond_hypertension'] > 0)
    ).astype(float)
    eval_dict['chf_and_hypertension'] = (
        (eval_dict['cond_heart_failure'] > 0) & (eval_dict['cond_hypertension'] > 0)
    ).astype(float)
    extra_numeric.append('chf_and_hypertension')

    # Diagnostic chapter group indicators: any slot and primary slot
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
        'diag_1_subprefix', 'diag_2_subprefix', 'diag_3_subprefix',
        'discharge_disposition_id', 'admission_source_id',
        'admission_type_id', 'medical_specialty', 'payer_code',
        'age', 'race'
    ]
    for col in te_candidates:
        if col in train_dict:
            te_tr, te_ev = oof_target_encode(
                train_dict[col], y_train, eval_dict[col], n_folds=5, smoothing=20.0, seed=42
            )
            cname = f"{col}_te"
            train_dict[cname] = te_tr.astype(float)
            eval_dict[cname] = te_ev.astype(float)
            extra_numeric.append(cname)

    # 4. Clinical domain features: utilization & stay intensity
    def derive_clinical_numeric(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        feats = {}
        stay = np.maximum(np.nan_to_num(d['time_in_hospital'].astype(float), nan=1.0), 1.0)
        n_inpatient = np.nan_to_num(d['number_inpatient'].astype(float), nan=0.0)
        n_emergency = np.nan_to_num(d['number_emergency'].astype(float), nan=0.0)
        n_outpatient = np.nan_to_num(d['number_outpatient'].astype(float), nan=0.0)

        total_prior = n_inpatient + n_emergency + n_outpatient
        feats['total_prior_visits'] = total_prior
        feats['inpatient_ratio'] = n_inpatient / (total_prior + 1.0)
        feats['emergency_ratio'] = n_emergency / (total_prior + 1.0)
        feats['has_prior_inpatient'] = (n_inpatient > 0.0).astype(float)
        feats['has_multiple_inpatient'] = (n_inpatient >= 2.0).astype(float)
        feats['has_prior_emergency'] = (n_emergency > 0.0).astype(float)
        feats['has_prior_visits'] = (total_prior > 0.0).astype(float)
        feats['frequent_inpatient'] = (n_inpatient >= 3.0).astype(float)
        feats['high_utilizer'] = ((n_inpatient >= 2.0) | (total_prior >= 5.0)).astype(float)

        feats['log_inpatient'] = np.log1p(n_inpatient)
        feats['log_emergency'] = np.log1p(n_emergency)
        feats['log_total_prior'] = np.log1p(total_prior)
        feats['inpatient_times_stay'] = n_inpatient * stay
        feats['has_prior_outpatient'] = (n_outpatient > 0.0).astype(float)
        feats['outpatient_ratio'] = n_outpatient / (total_prior + 1.0)
        feats['has_only_acute_visits'] = (((n_inpatient > 0.0) | (n_emergency > 0.0)) & (n_outpatient == 0.0)).astype(float)
        feats['acute_prior_visits'] = n_inpatient + n_emergency
        feats['acute_utilization_ratio'] = feats['acute_prior_visits'] / (total_prior + 1.0)
        feats['is_first_visit'] = (total_prior == 0.0).astype(float)

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
        feats['clinical_complexity'] = (stay / 4.0) + (n_inpatient * 2.0) + (num_meds / 15.0) + (num_diags / 8.0)
        feats['stay_per_diag'] = stay / (num_diags + 1.0)
        feats['total_clinical_events'] = num_labs + num_meds + num_procs
        feats['events_per_day'] = feats['total_clinical_events'] / stay
        feats['has_no_procedures'] = (num_procs == 0.0).astype(float)
        feats['is_surgical'] = (num_procs >= 2.0).astype(float)
        feats['total_acute_visits'] = n_inpatient + n_emergency
        feats['has_both_inpatient_and_emergency'] = ((n_inpatient > 0.0) & (n_emergency > 0.0)).astype(float)
        feats['multiple_prior_acute'] = ((n_inpatient + n_emergency) >= 2.0).astype(float)
        feats['chronic_high_utilizer'] = ((n_inpatient >= 2.0) & (stay >= 5.0)).astype(float)
        feats['inpatient_times_comorbidities'] = n_inpatient * np.maximum(num_diags, 1.0)
        feats['is_high_diagnoses'] = (num_diags >= 9.0).astype(float)
        feats['diag_density'] = num_diags / stay

        # Monotonic continuous age & age interactions
        age_map = {
            '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
            '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
            '[80-90)': 85.0, '[90-100)': 95.0
        }
        age_vals = np.array([age_map.get(a, 60.0) for a in d['age']], dtype=float)
        feats['age_num'] = age_vals
        feats['is_elderly'] = (age_vals >= 70.0).astype(float)
        feats['is_very_elderly'] = (age_vals >= 80.0).astype(float)
        feats['is_nonagenarian'] = (np.array(d['age']) == '[90-100)').astype(float)
        feats['is_young_adult'] = np.isin(d['age'], ['[0-10)', '[10-20)', '[20-30)']).astype(float)
        feats['nonagenarian_inpatient'] = feats['is_nonagenarian'] * n_inpatient
        feats['age_times_inpatient'] = (age_vals / 50.0) * n_inpatient

        # Socioeconomic & Payer flags
        p_code = np.array(d['payer_code'])
        feats['payer_code_missing'] = np.isin(p_code, [None, '?', 'None']).astype(float)
        feats['medical_specialty_missing'] = np.isin(d['medical_specialty'], [None, '?', 'None']).astype(float)
        feats['has_weight_recorded'] = (~np.isin(d['weight'], [None, '?', 'None'])).astype(float)
        feats['is_medicare'] = (p_code == 'MC').astype(float)
        feats['is_medicaid'] = (p_code == 'MD').astype(float)
        feats['is_self_pay'] = np.isin(p_code, ['UN', 'SP']).astype(float)
        feats['is_commercial_payer'] = np.isin(p_code, ['BC', 'HM', 'CP', 'CI']).astype(float)
        feats['medicare_under_65'] = ((age_vals < 65.0) & (feats['is_medicare'] > 0)).astype(float)

        # Admission & discharge flags
        feats['is_discharge_home'] = (d['discharge_disposition_id'] == '1').astype(float)
        feats['is_discharge_snf'] = (d['discharge_disposition_id'] == '3').astype(float)
        feats['is_discharge_home_health'] = (d['discharge_disposition_id'] == '6').astype(float)
        feats['is_discharge_ama'] = (d['discharge_disposition_id'] == '7').astype(float)
        feats['is_discharge_transfer'] = np.isin(d['discharge_disposition_id'], ['2', '4', '5', '18', '22', '25']).astype(float)
        feats['is_high_risk_discharge'] = np.isin(d['discharge_disposition_id'], ['3', '6', '7', '22', '28']).astype(float)
        feats['is_discharged_not_home'] = (d['discharge_disposition_id'] != '1').astype(float)
        feats['high_risk_disch_inpatient'] = feats['is_high_risk_discharge'] * feats['has_prior_inpatient']
        feats['non_home_with_prior_inpatient'] = feats['is_discharged_not_home'] * feats['has_prior_inpatient']
        feats['snf_with_prior_inpatient'] = feats['is_discharge_snf'] * feats['has_prior_inpatient']
        feats['is_emergency_admission'] = (d['admission_type_id'] == '1').astype(float)
        feats['is_urgent_admission'] = (d['admission_type_id'] == '2').astype(float)
        feats['is_elective'] = (d['admission_type_id'] == '3').astype(float)
        feats['is_emergency_source'] = (d['admission_source_id'] == '7').astype(float)
        feats['is_transfer_admission'] = np.isin(d['admission_source_id'], ['4', '5', '6', '10', '18', '22', '25', '26']).astype(float)
        feats['is_emergency_from_er'] = ((d['admission_type_id'] == '1') & (d['admission_source_id'] == '7')).astype(float)
        feats['is_elective_to_home'] = ((d['admission_type_id'] == '3') & (d['discharge_disposition_id'] == '1')).astype(float)

        # Geriatric polypharmacy and clinical frailty index
        feats['geriatric_polypharmacy'] = ((age_vals >= 70.0) & (num_meds >= 15.0)).astype(float)
        feats['geriatric_risk_score'] = (age_vals >= 75.0).astype(float) * (
            feats['is_discharge_snf'] + feats['has_prior_inpatient'] + (num_meds >= 15.0).astype(float)
        )

        # Lab test indicators
        feats['a1c_tested'] = (d['A1Cresult'] != 'None').astype(float)
        feats['a1c_abnormal'] = np.isin(d['A1Cresult'], ['>7', '>8']).astype(float)
        feats['glu_tested'] = (d['max_glu_serum'] != 'None').astype(float)
        feats['glu_abnormal'] = np.isin(d['max_glu_serum'], ['>200', '>300']).astype(float)

        return feats

    clin_tr = derive_clinical_numeric(train)
    clin_ev = derive_clinical_numeric(eval_cols)
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

    def derive_med_features(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        res = {}
        if present_meds:
            m_stack = np.column_stack([d[m] for m in present_meds])
            is_active = (m_stack != 'No') & (m_stack != None)
            res['num_active_meds'] = np.sum(is_active, axis=1).astype(float)
            res['num_med_up'] = np.sum(m_stack == 'Up', axis=1).astype(float)
            res['num_med_down'] = np.sum(m_stack == 'Down', axis=1).astype(float)
            res['num_med_steady'] = np.sum(m_stack == 'Steady', axis=1).astype(float)
            res['num_med_changes'] = res['num_med_up'] + res['num_med_down']

        if 'insulin' in d:
            has_ins = ((d['insulin'] != 'No') & (d['insulin'] != None)).astype(float)
            res['has_insulin'] = has_ins
            res['insulin_up'] = (d['insulin'] == 'Up').astype(float)
            res['insulin_down'] = (d['insulin'] == 'Down').astype(float)
            res['insulin_steady'] = (d['insulin'] == 'Steady').astype(float)
            res['insulin_changed'] = ((d['insulin'] == 'Up') | (d['insulin'] == 'Down')).astype(float)

        if 'metformin' in d and 'insulin' in d:
            has_met = ((d['metformin'] != 'No') & (d['metformin'] != None)).astype(float)
            res['insulin_and_metformin'] = has_ins * has_met

        # Medication drug classes
        biguanides = ['metformin', 'glyburide-metformin', 'glipizide-metformin', 'metformin-rosiglitazone', 'metformin-pioglitazone']
        sulfonylureas = [
            'glimepiride', 'glipizide', 'glyburide', 'chlorpropamide',
            'tolbutamide', 'acetohexamide', 'tolazamide',
            'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone'
        ]
        tzds = ['pioglitazone', 'rosiglitazone', 'troglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']

        pres_big = [m for m in biguanides if m in d]
        has_big = np.any(np.column_stack([d[m] for m in pres_big]) != 'No', axis=1).astype(float) if pres_big else np.zeros(len(d['age']), dtype=float)

        pres_su = [m for m in sulfonylureas if m in d]
        has_su = np.any(np.column_stack([d[m] for m in pres_su]) != 'No', axis=1).astype(float) if pres_su else np.zeros(len(d['age']), dtype=float)

        pres_tzd = [m for m in tzds if m in d]
        has_tzd = np.any(np.column_stack([d[m] for m in pres_tzd]) != 'No', axis=1).astype(float) if pres_tzd else np.zeros(len(d['age']), dtype=float)

        res['has_biguanide'] = has_big
        res['has_sulfonylurea'] = has_su
        res['has_tzd'] = has_tzd

        has_ins_arr = res.get('has_insulin', np.zeros(len(d['age']), dtype=float))
        num_classes = has_big + has_su + has_tzd + has_ins_arr
        res['num_med_classes'] = num_classes
        res['is_insulin_only'] = ((has_ins_arr > 0) & (has_big == 0) & (has_su == 0) & (has_tzd == 0)).astype(float)
        res['is_oral_only'] = ((has_ins_arr == 0) & (num_classes > 0)).astype(float)
        res['is_combo_insulin_oral'] = ((has_ins_arr > 0) & ((has_big + has_su + has_tzd) > 0)).astype(float)

        if 'insulin' in d:
            res['insulin_and_sulfonylurea'] = has_ins * has_su

        has_chg = (d['change'] == 'Ch').astype(float) if 'change' in d else np.zeros(len(d['age']), dtype=float)
        res['has_med_change'] = has_chg
        res['has_med_down'] = (res.get('num_med_down', np.zeros(len(d['age']))) > 0.0).astype(float)
        res['has_med_up'] = (res.get('num_med_up', np.zeros(len(d['age']))) > 0.0).astype(float)

        if 'diabetesMed' in d:
            res['on_diabetes_med'] = (d['diabetesMed'] == 'Yes').astype(float)

        # Clinical inertia marker (abnormal A1C but no medication change)
        a1c_abn = np.isin(d['A1Cresult'], ['>7', '>8']).astype(float)
        a1c_tst = (d['A1Cresult'] != 'None').astype(float)
        res['a1c_high_no_change'] = a1c_abn * (1.0 - has_chg)
        res['a1c_tested_and_change'] = a1c_tst * has_chg

        return res

    med_tr = derive_med_features(train)
    med_ev = derive_med_features(eval_cols)
    for k in med_tr:
        train_dict[k] = med_tr[k]
        eval_dict[k] = med_ev[k]
        extra_numeric.append(k)

    # Admission-Discharge interaction category & out-of-fold target encoding
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
        n_folds=5, smoothing=20.0, seed=42
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
        n_folds=5, smoothing=20.0, seed=42
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
        n_folds=5, smoothing=20.0, seed=42
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
        n_folds=5, smoothing=20.0, seed=42
    )
    train_dict['diag1_diag2_group_te'] = te_tr_d12.astype(float)
    eval_dict['diag1_diag2_group_te'] = te_ev_d12.astype(float)
    extra_numeric.append('diag1_diag2_group_te')

    # Inpatient Utilization Bin x Discharge Disposition Interaction
    def bin_inp(n_inp_arr: np.ndarray) -> np.ndarray:
        bins = []
        for v in n_inp_arr:
            val = float(v) if v is not None and not np.isnan(v) else 0.0
            if val == 0:
                bins.append('0')
            elif val == 1:
                bins.append('1')
            elif val == 2:
                bins.append('2')
            else:
                bins.append('3p')
        return np.array(bins, dtype=object)

    train_inp_bin = bin_inp(train['number_inpatient'])
    eval_inp_bin = bin_inp(eval_cols['number_inpatient'])

    train_dict['inp_disch_interaction'] = np.array(
        [f"{b}_{d}" for b, d in zip(train_inp_bin, train['discharge_disposition_id'])],
        dtype=object
    )
    eval_dict['inp_disch_interaction'] = np.array(
        [f"{b}_{d}" for b, d in zip(eval_inp_bin, eval_cols['discharge_disposition_id'])],
        dtype=object
    )
    te_tr_id, te_ev_id = oof_target_encode(
        train_dict['inp_disch_interaction'], y_train, eval_dict['inp_disch_interaction'],
        n_folds=5, smoothing=20.0, seed=42
    )
    train_dict['inp_disch_interaction_te'] = te_tr_id.astype(float)
    eval_dict['inp_disch_interaction_te'] = te_ev_id.astype(float)
    extra_numeric.append('inp_disch_interaction_te')

    # Additional high-risk cross interactions: Admission Type x Source, Diag1 Prefix x Disch, Payer x Age
    train_dict['adm_type_source'] = np.array(
        [f"{t}_{s}" for t, s in zip(train['admission_type_id'], train['admission_source_id'])],
        dtype=object
    )
    eval_dict['adm_type_source'] = np.array(
        [f"{t}_{s}" for t, s in zip(eval_cols['admission_type_id'], eval_cols['admission_source_id'])],
        dtype=object
    )
    te_tr_ats, te_ev_ats = oof_target_encode(
        train_dict['adm_type_source'], y_train, eval_dict['adm_type_source'],
        n_folds=5, smoothing=20.0, seed=42
    )
    train_dict['adm_type_source_te'] = te_tr_ats.astype(float)
    eval_dict['adm_type_source_te'] = te_ev_ats.astype(float)
    extra_numeric.append('adm_type_source_te')

    train_dict['diag1_pfx_disch'] = np.array(
        [f"{p}_{d}" for p, d in zip(train_dict['diag_1_prefix'], train['discharge_disposition_id'])],
        dtype=object
    )
    eval_dict['diag1_pfx_disch'] = np.array(
        [f"{p}_{d}" for p, d in zip(eval_dict['diag_1_prefix'], eval_cols['discharge_disposition_id'])],
        dtype=object
    )
    te_tr_dpd, te_ev_dpd = oof_target_encode(
        train_dict['diag1_pfx_disch'], y_train, eval_dict['diag1_pfx_disch'],
        n_folds=5, smoothing=20.0, seed=42
    )
    train_dict['diag1_pfx_disch_te'] = te_tr_dpd.astype(float)
    eval_dict['diag1_pfx_disch_te'] = te_ev_dpd.astype(float)
    extra_numeric.append('diag1_pfx_disch_te')

    train_dict['payer_age'] = np.array(
        [f"{p}_{a}" for p, a in zip(train['payer_code'], train['age'])],
        dtype=object
    )
    eval_dict['payer_age'] = np.array(
        [f"{p}_{a}" for p, a in zip(eval_cols['payer_code'], eval_cols['age'])],
        dtype=object
    )
    te_tr_pa, te_ev_pa = oof_target_encode(
        train_dict['payer_age'], y_train, eval_dict['payer_age'],
        n_folds=5, smoothing=20.0, seed=42
    )
    train_dict['payer_age_te'] = te_tr_pa.astype(float)
    eval_dict['payer_age_te'] = te_ev_pa.astype(float)
    extra_numeric.append('payer_age_te')

    # Insulin decrease in renal failure interaction
    train_dict['insulin_down_renal'] = (train_dict['insulin_down'] * (train_dict['cm_renal'] > 0)).astype(float)
    eval_dict['insulin_down_renal'] = (eval_dict['insulin_down'] * (eval_dict['cm_renal'] > 0)).astype(float)
    extra_numeric.append('insulin_down_renal')

    # 6. Tensor-inspired Low-Rank Phenotype Factorization (TruncatedSVD)
    # Decompose multi-way comorbidity-medication-utilization space into orthogonal latent factors
    svd_cols = [
        'cm_mi', 'cm_chf', 'cm_pvd', 'cm_cerebrovascular', 'cm_dementia',
        'cm_pulmonary', 'cm_rheumatic', 'cm_peptic_ulcer', 'cm_liver',
        'cm_hemiplegia', 'cm_renal', 'cm_cancer', 'cm_metastatic', 'cm_diabetes_complicated',
        'cardiorenal_risk',
        'cond_heart_failure', 'cond_diabetes', 'cond_pneumonia', 'cond_copd', 'cond_renal', 'cond_anemia',
        'cond_fluid_electrolyte', 'cond_hypertension', 'cond_dysrhythmia', 'cond_septicemia', 'cond_uti',
        'has_group_circulatory', 'has_group_respiratory', 'has_group_diabetes',
        'has_group_digestive', 'has_group_genitourinary',
        'has_prior_inpatient', 'has_multiple_inpatient', 'has_prior_emergency',
        'is_discharge_home', 'is_discharge_snf', 'is_discharge_ama', 'is_discharge_home_health',
        'is_emergency_admission', 'is_transfer_admission', 'is_medicare', 'is_medicaid',
        'a1c_tested', 'a1c_abnormal', 'glu_tested'
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

    n_svd_components = 10
    svd = TruncatedSVD(n_components=n_svd_components, algorithm='randomized', random_state=42)
    svd_tr = svd.fit_transform(M_tr)
    svd_ev = svd.transform(M_ev)

    for i in range(n_svd_components):
        sname = f"svd_phenotype_{i}"
        train_dict[sname] = svd_tr[:, i].astype(float)
        eval_dict[sname] = svd_ev[:, i].astype(float)
        extra_numeric.append(sname)

    # 7. Tabular encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_dict, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_dict)
    x_eval = encoder.transform(eval_dict)

    # Numeric & target-encoded subset for XGBoost (excludes raw categorical integer codes)
    xgb_feat_indices = [i for i in range(x_train.shape[1]) if i not in encoder.categorical_indices]
    x_train_xgb = x_train[:, xgb_feat_indices]
    x_eval_xgb = x_eval[:, xgb_feat_indices]

    # 8. Tri-family regularized ensemble (LightGBM + XGBoost + HistGradientBoosting)
    lgb_models = [
        lgb.LGBMClassifier(
            n_estimators=380,
            learning_rate=0.024,
            num_leaves=31,
            min_child_samples=120,
            min_split_gain=0.01,
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
            n_estimators=420,
            learning_rate=0.020,
            num_leaves=44,
            min_child_samples=80,
            min_split_gain=0.01,
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
            n_estimators=440,
            learning_rate=0.020,
            num_leaves=22,
            min_child_samples=160,
            min_split_gain=0.01,
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
            n_estimators=380,
            learning_rate=0.024,
            num_leaves=35,
            min_child_samples=100,
            min_split_gain=0.01,
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
            n_estimators=280,
            learning_rate=0.025,
            max_depth=5,
            subsample=0.80,
            colsample_bytree=0.60,
            reg_lambda=4.0,
            reg_alpha=0.5,
            tree_method='hist',
            random_state=42,
            n_jobs=1,
            eval_metric='logloss',
        ),
        xgb.XGBClassifier(
            n_estimators=260,
            learning_rate=0.028,
            max_depth=4,
            subsample=0.75,
            colsample_bytree=0.50,
            reg_lambda=5.0,
            reg_alpha=0.2,
            min_child_weight=15,
            tree_method='hist',
            random_state=2024,
            n_jobs=1,
            eval_metric='logloss',
        ),
    ]

    hgb_models = [
        HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.030,
            max_leaf_nodes=31,
            min_samples_leaf=80,
            l2_regularization=3.0,
            random_state=42,
        ),
        HistGradientBoostingClassifier(
            max_iter=240,
            learning_rate=0.032,
            max_leaf_nodes=22,
            min_samples_leaf=120,
            l2_regularization=5.0,
            random_state=2024,
        ),
    ]

    n_eval = len(x_eval)
    preds_rank = np.zeros(n_eval, dtype=float)
    total_weight = 0.0

    for model in lgb_models:
        model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        p = model.predict_proba(x_eval)[:, 1]
        preds_rank += 1.0 * ((rankdata(p) - 0.5) / n_eval)
        total_weight += 1.0

    for model in xgb_models:
        model.fit(x_train_xgb, y_train)
        p = model.predict_proba(x_eval_xgb)[:, 1]
        preds_rank += 0.75 * ((rankdata(p) - 0.5) / n_eval)
        total_weight += 0.75

    for model in hgb_models:
        model.fit(x_train_xgb, y_train)
        p = model.predict_proba(x_eval_xgb)[:, 1]
        preds_rank += 0.60 * ((rankdata(p) - 0.5) / n_eval)
        total_weight += 0.60

    preds = preds_rank / total_weight
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
