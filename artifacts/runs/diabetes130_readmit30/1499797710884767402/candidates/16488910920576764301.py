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
    from scipy.stats import rankdata
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    train_data = dict(train)
    eval_data = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    # 1. Clean continuous age
    age_map = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0
    }
    for col_dict in [train_data, eval_data]:
        raw_age = col_dict.get('age', np.array([]))
        col_dict['num_age'] = np.array([age_map.get(a, np.nan) for a in raw_age], dtype=np.float64)
    extra_numeric.append('num_age')

    # 2. Utilisation, Hospital Stay Dynamics & Admission/Discharge Trajectories
    for col_dict in [train_data, eval_data]:
        inp = np.nan_to_num(col_dict.get('number_inpatient', 0.0), nan=0.0)
        emg = np.nan_to_num(col_dict.get('number_emergency', 0.0), nan=0.0)
        outp = np.nan_to_num(col_dict.get('number_outpatient', 0.0), nan=0.0)
        tot_visits = inp + emg + outp

        col_dict['util_total_visits'] = tot_visits
        col_dict['util_inpatient_ratio'] = inp / (tot_visits + 1.0)
        col_dict['util_emergency_ratio'] = emg / (tot_visits + 1.0)
        col_dict['util_visit_severity'] = inp * 3.0 + emg * 1.5 + outp * 0.5
        col_dict['util_inp_to_outp'] = inp / (outp + 1.0)
        col_dict['util_emg_to_outp'] = emg / (outp + 1.0)

        col_dict['has_prior_inpatient'] = (inp > 0.0).astype(np.float64)
        col_dict['has_prior_emergency'] = (emg > 0.0).astype(np.float64)
        col_dict['log_number_inpatient'] = np.log1p(inp)
        col_dict['log_total_visits'] = np.log1p(tot_visits)

        time_safe = np.maximum(np.nan_to_num(col_dict.get('time_in_hospital', 1.0), nan=1.0), 1.0)
        num_labs = np.nan_to_num(col_dict.get('num_lab_procedures', 0.0), nan=0.0)
        num_proc = np.nan_to_num(col_dict.get('num_procedures', 0.0), nan=0.0)
        num_meds = np.nan_to_num(col_dict.get('num_medications', 0.0), nan=0.0)
        num_diag = np.nan_to_num(col_dict.get('number_diagnoses', 1.0), nan=1.0)

        col_dict['hosp_labs_per_day'] = num_labs / time_safe
        col_dict['hosp_meds_per_day'] = num_meds / time_safe
        col_dict['hosp_service_intensity'] = (num_labs + num_proc + num_meds) / time_safe
        col_dict['hosp_proc_to_labs'] = num_proc / (num_labs + 1.0)
        col_dict['hosp_meds_to_labs'] = num_meds / (num_labs + 1.0)
        col_dict['hosp_interaction_log'] = np.log1p(num_labs * num_proc * num_meds)
        col_dict['hosp_diag_to_days'] = num_diag / time_safe
        col_dict['hosp_meds_to_diag'] = num_meds / (num_diag + 1.0)

        # Discharge and Admission grouping
        disp = col_dict.get('discharge_disposition_id', np.array([]))
        adm_s = col_dict.get('admission_source_id', np.array([]))
        adm_t = col_dict.get('admission_type_id', np.array([]))

        col_dict['feat_disp_home'] = np.isin(disp, ['1', '8']).astype(np.float64)
        col_dict['feat_disp_home_health'] = (disp == '6').astype(np.float64)
        col_dict['feat_disp_facility'] = np.isin(disp, ['2', '3', '4', '5', '22', '23', '24', '25']).astype(np.float64)
        col_dict['feat_disp_ama'] = (disp == '7').astype(np.float64)
        col_dict['feat_adm_emergency'] = (np.isin(adm_s, ['7']) | np.isin(adm_t, ['1'])).astype(np.float64)
        col_dict['feat_adm_elective'] = (adm_t == '3').astype(np.float64)
        col_dict['feat_disp_adm_comb'] = np.array([f"{d}_{s}" for d, s in zip(disp, adm_s)], dtype=object)

        # Lab results
        a1c = col_dict.get('A1Cresult', np.array([]))
        glu = col_dict.get('max_glu_serum', np.array([]))
        col_dict['feat_high_a1c'] = np.isin(a1c, ['>7', '>8']).astype(np.float64)
        col_dict['feat_high_glucose'] = np.isin(glu, ['>200', '>300']).astype(np.float64)
        col_dict['feat_performed_a1c'] = (a1c != 'None').astype(np.float64)
        col_dict['feat_performed_glucose'] = (glu != 'None').astype(np.float64)

        # Missingness features
        spec = col_dict.get('medical_specialty', np.array([]))
        payer = col_dict.get('payer_code', np.array([]))
        weight = col_dict.get('weight', np.array([]))
        col_dict['feat_spec_missing'] = np.isin(spec, [None, 'None', '?', 'missing']).astype(np.float64)
        col_dict['feat_payer_missing'] = np.isin(payer, [None, 'None', '?', 'missing']).astype(np.float64)
        col_dict['feat_weight_missing'] = np.isin(weight, [None, 'None', '?', 'missing']).astype(np.float64)

    extra_categorical.append('feat_disp_adm_comb')
    extra_numeric.extend([
        'util_total_visits', 'util_inpatient_ratio', 'util_emergency_ratio',
        'util_visit_severity', 'util_inp_to_outp', 'util_emg_to_outp',
        'has_prior_inpatient', 'has_prior_emergency', 'log_number_inpatient', 'log_total_visits',
        'hosp_labs_per_day', 'hosp_meds_per_day', 'hosp_service_intensity',
        'hosp_proc_to_labs', 'hosp_meds_to_labs', 'hosp_interaction_log',
        'hosp_diag_to_days', 'hosp_meds_to_diag',
        'feat_disp_home', 'feat_disp_home_health', 'feat_disp_facility', 'feat_disp_ama',
        'feat_adm_emergency', 'feat_adm_elective',
        'feat_high_a1c', 'feat_high_glucose', 'feat_performed_a1c', 'feat_performed_glucose',
        'feat_spec_missing', 'feat_payer_missing', 'feat_weight_missing'
    ])

    # 3. Medication Summary Features
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    for col_dict in [train_data, eval_data]:
        n_rows = len(col_dict['time_in_hospital'])
        active = np.zeros(n_rows, dtype=np.float64)
        up = np.zeros(n_rows, dtype=np.float64)
        down = np.zeros(n_rows, dtype=np.float64)

        for m in med_cols:
            if m in col_dict:
                arr = col_dict[m]
                active += np.isin(arr, ['Steady', 'Up', 'Down']).astype(np.float64)
                up += (arr == 'Up').astype(np.float64)
                down += (arr == 'Down').astype(np.float64)

        insulin_arr = col_dict.get('insulin', np.full(n_rows, 'No', dtype=object))
        ins_active = np.isin(insulin_arr, ['Steady', 'Up', 'Down']).astype(np.float64)
        ins_up = (insulin_arr == 'Up').astype(np.float64)
        ins_down = (insulin_arr == 'Down').astype(np.float64)
        oral_meds = np.maximum(0.0, active - ins_active)

        met_arr = col_dict.get('metformin', np.full(n_rows, 'No', dtype=object))
        col_dict['med_metformin_active'] = np.isin(met_arr, ['Steady', 'Up', 'Down']).astype(np.float64)

        col_dict['med_num_active'] = active
        col_dict['med_num_changed'] = up + down
        col_dict['med_insulin_active'] = ins_active
        col_dict['med_insulin_up'] = ins_up
        col_dict['med_insulin_down'] = ins_down
        col_dict['med_insulin_plus_oral'] = ((ins_active > 0.0) & (oral_meds > 0.0)).astype(np.float64)
        col_dict['med_only_insulin'] = ((ins_active > 0.0) & (oral_meds == 0.0)).astype(np.float64)
        col_dict['med_only_oral'] = ((ins_active == 0.0) & (oral_meds > 0.0)).astype(np.float64)
        col_dict['med_multiple_oral'] = (oral_meds > 1.0).astype(np.float64)
        col_dict['med_no_active'] = (active == 0.0).astype(np.float64)

        dm = col_dict.get('diabetesMed', np.array([]))
        ch = col_dict.get('change', np.array([]))
        col_dict['feat_diabetesMed_num'] = (dm == 'Yes').astype(np.float64)
        col_dict['feat_change_num'] = (ch == 'Ch').astype(np.float64)

    extra_numeric.extend([
        'med_num_active', 'med_num_changed', 'med_insulin_active',
        'med_insulin_up', 'med_insulin_down', 'med_insulin_plus_oral',
        'med_only_insulin', 'med_only_oral', 'med_multiple_oral', 'med_no_active',
        'feat_diabetesMed_num', 'feat_change_num', 'med_metformin_active'
    ])

    # 4. Vectorized Diagnosis Grouping & Comorbidity Identification
    def extract_prefix(code):
        if code is None or code == '?' or code == 'None':
            return 'missing'
        s = str(code).strip()
        return s.split('.')[0] if s else 'missing'

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

        chf = (p1 == '428') | (p2 == '428') | (p3 == '428')
        ckd = (p1 == '585') | (p2 == '585') | (p3 == '585') | (p1 == '586') | (p2 == '586') | (p3 == '586')
        ht = np.isin(p1, ['401', '402', '403', '404', '405']) | \
             np.isin(p2, ['401', '402', '403', '404', '405']) | \
             np.isin(p3, ['401', '402', '403', '404', '405'])
        copd = np.isin(p1, ['490', '491', '492', '493', '494', '495', '496']) | \
               np.isin(p2, ['490', '491', '492', '493', '494', '495', '496']) | \
               np.isin(p3, ['490', '491', '492', '493', '494', '495', '496'])
        cad = np.isin(p1, ['410', '411', '412', '413', '414']) | \
              np.isin(p2, ['410', '411', '412', '413', '414']) | \
              np.isin(p3, ['410', '411', '412', '413', '414'])
        aki = (p1 == '584') | (p2 == '584') | (p3 == '584')
        stroke = np.isin(p1, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | \
                 np.isin(p2, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | \
                 np.isin(p3, ['430', '431', '432', '433', '434', '435', '436', '437', '438'])

        col_dict['comorb_has_chf'] = chf.astype(np.float64)
        col_dict['comorb_has_ckd'] = ckd.astype(np.float64)
        col_dict['comorb_has_ht'] = ht.astype(np.float64)
        col_dict['comorb_has_copd'] = copd.astype(np.float64)
        col_dict['comorb_has_cad'] = cad.astype(np.float64)
        col_dict['comorb_has_aki'] = aki.astype(np.float64)
        col_dict['comorb_has_stroke'] = stroke.astype(np.float64)

    extra_categorical.extend(['diag_1_group', 'diag_2_group', 'diag_3_group', 'diag_1_prefix'])
    extra_numeric.extend([
        'comorb_has_circ', 'comorb_has_diab', 'comorb_has_resp', 'comorb_has_genit',
        'comorb_has_chf', 'comorb_has_ckd', 'comorb_has_ht', 'comorb_has_copd',
        'comorb_has_cad', 'comorb_has_aki', 'comorb_has_stroke', 'comorb_num_systems'
    ])

    # 5. Rockwood Clinical Frailty & Deficit Saturation Index
    for col_dict in [train_data, eval_data]:
        age_val = np.nan_to_num(col_dict.get('num_age', 50.0), nan=50.0)
        time_val = np.nan_to_num(col_dict.get('time_in_hospital', 1.0), nan=1.0)
        inp_val = np.nan_to_num(col_dict.get('number_inpatient', 0.0), nan=0.0)
        emg_val = np.nan_to_num(col_dict.get('number_emergency', 0.0), nan=0.0)
        diag_val = np.nan_to_num(col_dict.get('number_diagnoses', 1.0), nan=1.0)
        med_val = np.nan_to_num(col_dict.get('num_medications', 0.0), nan=0.0)
        tot_visits = np.nan_to_num(col_dict.get('util_total_visits', 0.0), nan=0.0)

        d1 = (age_val >= 70.0).astype(np.float64)
        d2 = (time_val >= 6.0).astype(np.float64)
        d3 = (inp_val >= 1.0).astype(np.float64)
        d4 = (emg_val >= 1.0).astype(np.float64)
        d5 = (diag_val >= 8.0).astype(np.float64)
        d6 = (med_val >= 15.0).astype(np.float64)
        d7 = col_dict.get('comorb_has_chf', np.zeros_like(age_val))
        d8 = col_dict.get('comorb_has_ckd', np.zeros_like(age_val))
        d9 = col_dict.get('comorb_has_copd', np.zeros_like(age_val))
        d10 = (col_dict.get('comorb_num_systems', np.zeros_like(age_val)) >= 3.0).astype(np.float64)
        d11 = (col_dict.get('med_num_changed', np.zeros_like(age_val)) >= 1.0).astype(np.float64)
        d12 = col_dict.get('feat_disp_facility', np.zeros_like(age_val))
        d13 = col_dict.get('feat_adm_emergency', np.zeros_like(age_val))
        d14 = ((col_dict.get('feat_high_a1c', np.zeros_like(age_val)) == 1.0) |
               (col_dict.get('feat_high_glucose', np.zeros_like(age_val)) == 1.0)).astype(np.float64)
        d15 = ((col_dict.get('med_insulin_up', np.zeros_like(age_val)) == 1.0) |
               (col_dict.get('med_insulin_down', np.zeros_like(age_val)) == 1.0)).astype(np.float64)

        deficits = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9 + d10 + d11 + d12 + d13 + d14 + d15
        col_dict['frailty_deficit_count'] = deficits
        col_dict['frailty_deficit_index'] = deficits / 15.0
        col_dict['frailty_saturation'] = np.exp(2.0 * col_dict['frailty_deficit_index']) - 1.0
        col_dict['cardiorenal_syndrome'] = (col_dict.get('comorb_has_chf', 0.0) * col_dict.get('comorb_has_ckd', 0.0)).astype(np.float64)

        col_dict['inter_age_visits'] = age_val * tot_visits
        col_dict['inter_age_diagnoses'] = age_val * diag_val
        col_dict['inter_visits_diagnoses'] = tot_visits * diag_val
        col_dict['inter_inp_time'] = inp_val * time_val

    extra_numeric.extend([
        'frailty_deficit_count', 'frailty_deficit_index', 'frailty_saturation', 'cardiorenal_syndrome',
        'inter_age_visits', 'inter_age_diagnoses', 'inter_visits_diagnoses', 'inter_inp_time'
    ])

    # 6. Latent Clinical Phenotyping via TruncatedSVD on Clinical Bag-of-Entities
    n_tr = len(train_data['time_in_hospital'])
    n_ev = len(eval_data['time_in_hospital'])

    def build_clinical_tokens(col_dict, n_rows):
        tokens = [[] for _ in range(n_rows)]
        for diag_col in ['diag_1', 'diag_2', 'diag_3']:
            if diag_col in col_dict:
                arr = col_dict[diag_col]
                for i, code in enumerate(arr):
                    if code is not None and code != '?' and code != 'None' and str(code).strip() != '':
                        s = str(code).strip()
                        tokens[i].append(f'{diag_col}_{s}')
                        tokens[i].append(f'{diag_col}_p_{s.split(".")[0]}')
                        tokens[i].append(f'{diag_col}_g_{icd9_group(s)}')
        for m in med_cols:
            if m in col_dict:
                arr = col_dict[m]
                for i, v in enumerate(arr):
                    if v in ('Steady', 'Up', 'Down'):
                        tokens[i].append(f'med_{m}_{v}')
        disp = col_dict.get('discharge_disposition_id', np.array([]))
        adm_s = col_dict.get('admission_source_id', np.array([]))
        for i in range(n_rows):
            if i < len(disp) and disp[i] is not None:
                tokens[i].append(f'disp_{disp[i]}')
            if i < len(adm_s) and adm_s[i] is not None:
                tokens[i].append(f'adm_{adm_s[i]}')
        return [' '.join(t) for t in tokens]

    tr_text = build_clinical_tokens(train_data, n_tr)
    ev_text = build_clinical_tokens(eval_data, n_ev)

    bow_vec = CountVectorizer(min_df=8, binary=True, token_pattern=r'\S+')
    X_bow_tr = bow_vec.fit_transform(tr_text)
    X_bow_ev = bow_vec.transform(ev_text)

    svd = TruncatedSVD(n_components=6, random_state=42)
    X_svd_tr = svd.fit_transform(X_bow_tr)
    X_svd_ev = svd.transform(X_bow_ev)

    for k in range(6):
        col_name = f'svd_pheno_{k}'
        train_data[col_name] = X_svd_tr[:, k].astype(np.float64)
        eval_data[col_name] = X_svd_ev[:, k].astype(np.float64)
        extra_numeric.append(col_name)

    # 7. Taxonomy-Aware Hierarchical Empirical Bayes Target Encoding for Diagnoses
    def compute_hierarchical_diag_te(tr_d, ev_d, y_tr, n_splits=5, s_glob=40.0, s_grp=15.0, s_pref=8.0, seed=42):
        tr_clean = np.array([str(x).strip() if x is not None and x != '?' and x != 'None' and str(x).strip() != '' else 'missing' for x in tr_d], dtype=object)
        ev_clean = np.array([str(x).strip() if x is not None and x != '?' and x != 'None' and str(x).strip() != '' else 'missing' for x in ev_d], dtype=object)

        tr_grp = np.array([icd9_group(c) for c in tr_clean], dtype=object)
        ev_grp = np.array([icd9_group(c) for c in ev_clean], dtype=object)

        tr_pref = np.array([c.split('.')[0] if c != 'missing' else 'missing' for c in tr_clean], dtype=object)
        ev_pref = np.array([c.split('.')[0] if c != 'missing' else 'missing' for c in ev_clean], dtype=object)

        prior = float(np.mean(y_tr))
        tr_te = np.full(len(tr_d), prior, dtype=np.float64)
        ev_te = np.full(len(ev_d), prior, dtype=np.float64)

        def build_hier_rates(codes, prefs, grps, y, prior_val):
            grp_counts, grp_sums = {}, {}
            pref_counts, pref_sums, pref_to_grp = {}, {}, {}
            code_counts, code_sums, code_to_pref = {}, {}, {}

            for c, p, g, val in zip(codes, prefs, grps, y):
                grp_counts[g] = grp_counts.get(g, 0) + 1
                grp_sums[g] = grp_sums.get(g, 0.0) + val

                pref_counts[p] = pref_counts.get(p, 0) + 1
                pref_sums[p] = pref_sums.get(p, 0.0) + val
                pref_to_grp[p] = g

                code_counts[c] = code_counts.get(c, 0) + 1
                code_sums[c] = code_sums.get(c, 0.0) + val
                code_to_pref[c] = p

            grp_rate = {g: (grp_sums[g] + s_glob * prior_val) / (grp_counts[g] + s_glob) for g in grp_counts}
            pref_rate = {p: (pref_sums[p] + s_grp * grp_rate.get(pref_to_grp[p], prior_val)) / (pref_counts[p] + s_grp) for p in pref_counts}
            code_rate = {c: (code_sums[c] + s_pref * pref_rate.get(code_to_pref[c], prior_val)) / (code_counts[c] + s_pref) for c in code_counts}
            return code_rate, pref_rate, grp_rate

        def lookup(c, p, g, code_rate, pref_rate, grp_rate, prior_val):
            if c in code_rate:
                return code_rate[c]
            if p in pref_rate:
                return pref_rate[p]
            if g in grp_rate:
                return grp_rate[g]
            return prior_val

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(tr_clean, y_tr):
            sub_c, sub_p, sub_g, sub_y = tr_clean[train_idx], tr_pref[train_idx], tr_grp[train_idx], y_tr[train_idx]
            v_c, v_p, v_g = tr_clean[val_idx], tr_pref[val_idx], tr_grp[val_idx]
            sub_prior = float(np.mean(sub_y))

            c_r, p_r, g_r = build_hier_rates(sub_c, sub_p, sub_g, sub_y, sub_prior)
            tr_te[val_idx] = [lookup(c, p, g, c_r, p_r, g_r, sub_prior) for c, p, g in zip(v_c, v_p, v_g)]

        full_c_r, full_p_r, full_g_r = build_hier_rates(tr_clean, tr_pref, tr_grp, y_tr, prior)
        ev_te[:] = [lookup(c, p, g, full_c_r, full_p_r, full_g_r, prior) for c, p, g in zip(ev_clean, ev_pref, ev_grp)]
        return tr_te, ev_te

    h_tr1, h_ev1 = compute_hierarchical_diag_te(train_data['diag_1'], eval_data['diag_1'], y_train)
    h_tr2, h_ev2 = compute_hierarchical_diag_te(train_data['diag_2'], eval_data['diag_2'], y_train)
    h_tr3, h_ev3 = compute_hierarchical_diag_te(train_data['diag_3'], eval_data['diag_3'], y_train)

    train_data['hier_te_diag_1'] = h_tr1
    eval_data['hier_te_diag_1'] = h_ev1
    train_data['hier_te_diag_2'] = h_tr2
    eval_data['hier_te_diag_2'] = h_ev2
    train_data['hier_te_diag_3'] = h_tr3
    eval_data['hier_te_diag_3'] = h_ev3

    train_data['hier_te_diag_max'] = np.maximum(np.maximum(h_tr1, h_tr2), h_tr3)
    eval_data['hier_te_diag_max'] = np.maximum(np.maximum(h_ev1, h_ev2), h_ev3)
    train_data['hier_te_diag_mean'] = (h_tr1 + h_tr2 + h_tr3) / 3.0
    eval_data['hier_te_diag_mean'] = (h_ev1 + h_ev2 + h_ev3) / 3.0
    extra_numeric.extend(['hier_te_diag_1', 'hier_te_diag_2', 'hier_te_diag_3', 'hier_te_diag_max', 'hier_te_diag_mean'])

    # Standard OOF target encoding on non-hierarchical high-cardinality features
    def compute_oof_te(tr_arr, y_tr, ev_arr, smoothing=25.0, n_splits=5, seed=42):
        tr_arr_str = np.array([str(x) if x is not None else 'missing' for x in tr_arr], dtype=object)
        ev_arr_str = np.array([str(x) if x is not None else 'missing' for x in ev_arr], dtype=object)
        prior = float(np.mean(y_tr))
        tr_te = np.full(len(tr_arr_str), prior, dtype=np.float64)
        ev_te = np.full(len(ev_arr_str), prior, dtype=np.float64)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(tr_arr_str, y_tr):
            sub_c = tr_arr_str[train_idx]
            sub_y = y_tr[train_idx]
            v_c = tr_arr_str[val_idx]

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
        for c, y in zip(tr_arr_str, y_tr):
            full_counts[c] = full_counts.get(c, 0) + 1
            full_sums[c] = full_sums.get(c, 0.0) + y

        for i, c in enumerate(ev_arr_str):
            cnt = full_counts.get(c, 0)
            ev_te[i] = (full_sums[c] + smoothing * prior) / (cnt + smoothing) if cnt > 0 else prior

        return tr_te, ev_te

    te_targets = [
        'discharge_disposition_id',
        'admission_source_id',
        'medical_specialty',
        'feat_disp_adm_comb',
        'payer_code'
    ]
    for c_name in te_targets:
        if c_name in train_data:
            tr_te, ev_te = compute_oof_te(train_data[c_name], y_train, eval_data[c_name], smoothing=25.0)
            te_col_name = f'te_{c_name}'
            train_data[te_col_name] = tr_te
            eval_data[te_col_name] = ev_te
            extra_numeric.append(te_col_name)

    # 8. Non-parametric Topological Manifold Geometry (k-NN Density & Contrastive Boundary)
    manifold_cols = [
        'num_age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_diagnoses', 'util_total_visits',
        'util_inpatient_ratio', 'util_visit_severity', 'frailty_deficit_index',
        'hier_te_diag_max', 'svd_pheno_0'
    ]

    def extract_manifold_matrix(col_dict, cols, n_rows):
        mat = np.zeros((n_rows, len(cols)), dtype=np.float64)
        for j, col in enumerate(cols):
            arr = np.nan_to_num(col_dict.get(col, np.zeros(n_rows)), nan=0.0)
            mat[:, j] = arr
        return mat

    X_m_tr_raw = extract_manifold_matrix(train_data, manifold_cols, n_tr)
    X_m_ev_raw = extract_manifold_matrix(eval_data, manifold_cols, n_ev)

    scaler = StandardScaler()
    X_m_tr = scaler.fit_transform(X_m_tr_raw)
    X_m_ev = scaler.transform(X_m_ev_raw)

    knn_risk_oof = np.full(n_tr, np.mean(y_train), dtype=np.float64)
    knn_contrastive_oof = np.zeros(n_tr, dtype=np.float64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf.split(X_m_tr, y_train):
        sub_X = X_m_tr[tr_idx]
        sub_y = y_train[tr_idx]
        v_X = X_m_tr[val_idx]

        nn_all = NearestNeighbors(n_neighbors=30, algorithm='kd_tree', metric='euclidean', n_jobs=1)
        nn_all.fit(sub_X)
        indices = nn_all.kneighbors(v_X, return_distance=False)
        knn_risk_oof[val_idx] = np.mean(sub_y[indices], axis=1)

        nn_pos = NearestNeighbors(n_neighbors=3, algorithm='kd_tree', metric='euclidean', n_jobs=1)
        nn_pos.fit(sub_X[sub_y == 1])
        d_pos = nn_pos.kneighbors(v_X, return_distance=True)[0].mean(axis=1)

        nn_neg = NearestNeighbors(n_neighbors=3, algorithm='kd_tree', metric='euclidean', n_jobs=1)
        nn_neg.fit(sub_X[sub_y == 0])
        d_neg = nn_neg.kneighbors(v_X, return_distance=True)[0].mean(axis=1)

        knn_contrastive_oof[val_idx] = (d_neg - d_pos) / (d_neg + d_pos + 1e-4)

    nn_all_full = NearestNeighbors(n_neighbors=30, algorithm='kd_tree', metric='euclidean', n_jobs=1)
    nn_all_full.fit(X_m_tr)
    ev_indices = nn_all_full.kneighbors(X_m_ev, return_distance=False)
    knn_risk_ev = np.mean(y_train[ev_indices], axis=1)

    nn_pos_full = NearestNeighbors(n_neighbors=3, algorithm='kd_tree', metric='euclidean', n_jobs=1)
    nn_pos_full.fit(X_m_tr[y_train == 1])
    ev_d_pos = nn_pos_full.kneighbors(X_m_ev, return_distance=True)[0].mean(axis=1)

    nn_neg_full = NearestNeighbors(n_neighbors=3, algorithm='kd_tree', metric='euclidean', n_jobs=1)
    nn_neg_full.fit(X_m_tr[y_train == 0])
    ev_d_neg = nn_neg_full.kneighbors(X_m_ev, return_distance=True)[0].mean(axis=1)
    knn_contrastive_ev = (ev_d_neg - ev_d_pos) / (ev_d_neg + ev_d_pos + 1e-4)

    train_data['knn_risk_density'] = knn_risk_oof
    eval_data['knn_risk_density'] = knn_risk_ev
    train_data['knn_contrastive_ratio'] = knn_contrastive_oof
    eval_data['knn_contrastive_ratio'] = knn_contrastive_ev
    extra_numeric.extend(['knn_risk_density', 'knn_contrastive_ratio'])

    # 9. Tabular Encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_data, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_data)
    x_eval = encoder.transform(eval_data)

    # 10. Diverse Heterogeneous Multi-Model Gradient Boosting Ensemble
    models = [
        # Model 1: Moderate depth, conservative
        lgb.LGBMClassifier(
            n_estimators=380,
            learning_rate=0.022,
            num_leaves=18,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=0.2,
            reg_lambda=8.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 2: Extremely Randomized Trees mode (extra_trees=True)
        lgb.LGBMClassifier(
            n_estimators=360,
            learning_rate=0.025,
            num_leaves=24,
            min_child_samples=50,
            extra_trees=True,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=10.0,
            random_state=123,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 3: Deeper, regularized tree
        lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.018,
            num_leaves=31,
            min_child_samples=75,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.6,
            reg_alpha=1.5,
            reg_lambda=20.0,
            random_state=456,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 4: Shallow, fast bagging
        lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.035,
            num_leaves=12,
            min_child_samples=30,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.5,
            reg_alpha=0.05,
            reg_lambda=2.0,
            random_state=789,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 5: Cost-sensitive minority focusing (scale_pos_weight=1.5)
        lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.022,
            num_leaves=20,
            scale_pos_weight=1.5,
            min_child_samples=60,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.3,
            reg_lambda=12.0,
            random_state=999,
            n_jobs=1,
            verbose=-1,
        ),
        # Model 6: XGBoost depth-5
        xgb.XGBClassifier(
            n_estimators=280,
            learning_rate=0.025,
            max_depth=5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=10.0,
            random_state=101,
            n_jobs=1,
            verbosity=0,
        ),
        # Model 7: XGBoost shallow depth-3
        xgb.XGBClassifier(
            n_estimators=260,
            learning_rate=0.032,
            max_depth=3,
            min_child_weight=2,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=202,
            n_jobs=1,
            verbosity=0,
        ),
    ]

    n_eval_rows = len(x_eval)
    rank_sum = np.zeros(n_eval_rows, dtype=np.float64)
    raw_prob_sum = np.zeros(n_eval_rows, dtype=np.float64)

    for model in models:
        if isinstance(model, lgb.LGBMClassifier):
            model.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        else:
            model.fit(x_train, y_train)

        p = model.predict_proba(x_eval)[:, 1]
        r = rankdata(p) / float(n_eval_rows)
        rank_sum += r
        raw_prob_sum += p

    avg_rank = rank_sum / len(models)
    mean_raw_prob = raw_prob_sum / len(models)

    # 11. Quantile Matching Consensus Calibration: Enforces exact rank consensus with calibrated tails
    sorted_raw = np.sort(mean_raw_prob)
    rank_indices = np.argsort(np.argsort(avg_rank))
    calibrated_preds = sorted_raw[rank_indices]

    calibrated_preds = np.nan_to_num(calibrated_preds, nan=0.114, posinf=1.0, neginf=0.0)
    return np.clip(calibrated_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
