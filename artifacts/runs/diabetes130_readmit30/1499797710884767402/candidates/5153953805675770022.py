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
    """Fit on `train`/`y_train` using Clinical LACE+ Scoring, Comorbidity Indices & Simplex Quad-GBDTs."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from scipy.optimize import minimize
    import xgboost as xgb

    non_med_cols = {
        'race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty',
        'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id'
    }
    med_cols = [col for col in task.categorical_columns if col not in non_med_cols]

    age_midpoints = {
        '[0-10)': 5.0, '[10-20)': 15.0, '[20-30)': 25.0, '[30-40)': 35.0,
        '[40-50)': 45.0, '[50-60)': 55.0, '[60-70)': 65.0, '[70-80)': 75.0,
        '[80-90)': 85.0, '[90-100)': 95.0,
    }

    def safe_str(val):
        if val is None or val == '?' or val == '':
            return '__missing__'
        try:
            if isinstance(val, float) and np.isnan(val):
                return '__missing__'
        except Exception:
            pass
        return str(val)

    def clean_icd(code):
        if code is None or code == '?' or code == '':
            return 'missing'
        try:
            if isinstance(code, float) and np.isnan(code):
                return 'missing'
        except Exception:
            pass
        return str(code).strip().split('.')[0]

    def check_neoplasm(codes):
        res = np.zeros(len(codes), dtype=bool)
        for idx, c in enumerate(codes):
            try:
                v = int(c)
                if 140 <= v <= 208:
                    res[idx] = True
            except Exception:
                pass
        return res

    # 1. Domain Feature Engineering: Clinical Scores, Biomarkers, and Dynamics
    for d in [train, eval_cols]:
        n_rows = len(d['gender'])
        d['age_num'] = np.array([age_midpoints.get(str(x), 65.0) for x in d['age']], dtype=float)

        # Diagnoses cleaning & chapter groupings
        for col in ['diag_1', 'diag_2', 'diag_3']:
            d[f'{col}_grp'] = np.array([icd9_group(x) for x in d[col]], dtype=object)
            d[f'{col}_3d'] = np.array([clean_icd(x) for x in d[col]], dtype=object)

        c1, c2, c3 = d['diag_1_3d'], d['diag_2_3d'], d['diag_3_3d']

        # Clinical Condition Flags (ICD-9)
        has_mi = np.isin(c1, ['410', '412']) | np.isin(c2, ['410', '412']) | np.isin(c3, ['410', '412'])
        has_chf = (c1 == '428') | (c2 == '428') | (c3 == '428')
        has_cad = np.isin(c1, ['410', '411', '412', '413', '414']) | np.isin(c2, ['410', '411', '412', '413', '414']) | np.isin(c3, ['410', '411', '412', '413', '414'])
        has_htn = np.isin(c1, ['401', '402', '403', '404', '405']) | np.isin(c2, ['401', '402', '403', '404', '405']) | np.isin(c3, ['401', '402', '403', '404', '405'])
        has_pvd = np.isin(c1, ['440', '441', '442', '443', '444', '445', '446', '447']) | np.isin(c2, ['440', '441', '442', '443', '444', '445', '446', '447']) | np.isin(c3, ['440', '441', '442', '443', '444', '445', '446', '447'])
        has_stroke = np.isin(c1, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | np.isin(c2, ['430', '431', '432', '433', '434', '435', '436', '437', '438']) | np.isin(c3, ['430', '431', '432', '433', '434', '435', '436', '437', '438'])
        has_dementia = np.isin(c1, ['290', '294', '331']) | np.isin(c2, ['290', '294', '331']) | np.isin(c3, ['290', '294', '331'])
        has_copd = np.isin(c1, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c2, ['490', '491', '492', '493', '494', '495', '496']) | np.isin(c3, ['490', '491', '492', '493', '494', '495', '496'])
        has_rheum = np.isin(c1, ['710', '714', '725']) | np.isin(c2, ['710', '714', '725']) | np.isin(c3, ['710', '714', '725'])
        has_pud = np.isin(c1, ['531', '532', '533', '534']) | np.isin(c2, ['531', '532', '533', '534']) | np.isin(c3, ['531', '532', '533', '534'])
        has_mild_liver = (c1 == '571') | (c2 == '571') | (c3 == '571')
        has_sev_liver = np.isin(c1, ['572', '456']) | np.isin(c2, ['572', '456']) | np.isin(c3, ['572', '456'])
        has_hemi = np.isin(c1, ['342', '344']) | np.isin(c2, ['342', '344']) | np.isin(c3, ['342', '344'])
        has_ckd = np.isin(c1, ['582', '583', '585', '586', '588']) | np.isin(c2, ['582', '583', '585', '586', '588']) | np.isin(c3, ['582', '583', '585', '586', '588'])
        has_aki = (c1 == '584') | (c2 == '584') | (c3 == '584')
        has_renal = has_ckd | has_aki
        has_met_cancer = np.isin(c1, ['196', '197', '198', '199']) | np.isin(c2, ['196', '197', '198', '199']) | np.isin(c3, ['196', '197', '198', '199'])
        has_hiv = (c1 == '042') | (c2 == '042') | (c3 == '042')
        has_neoplasm = check_neoplasm(c1) | check_neoplasm(c2) | check_neoplasm(c3)
        has_fluid_elect = (c1 == '276') | (c2 == '276') | (c3 == '276')
        has_pneumonia = np.isin(c1, ['480', '481', '482', '483', '484', '485', '486']) | np.isin(c2, ['480', '481', '482', '483', '484', '485', '486']) | np.isin(c3, ['480', '481', '482', '483', '484', '485', '486'])
        has_uti = (c1 == '599') | (c2 == '599') | (c3 == '599')
        has_sepsis = np.isin(c1, ['038', '995']) | np.isin(c2, ['038', '995']) | np.isin(c3, ['038', '995'])
        has_anemia = np.isin(c1, ['280', '281', '282', '283', '284', '285']) | np.isin(c2, ['280', '281', '282', '283', '284', '285']) | np.isin(c3, ['280', '281', '282', '283', '284', '285'])
        has_afib = (c1 == '427') | (c2 == '427') | (c3 == '427')
        has_ulcer_skin = np.isin(c1, ['680', '681', '682', '683', '684', '685', '686', '707']) | np.isin(c2, ['680', '681', '682', '683', '684', '685', '686', '707']) | np.isin(c3, ['680', '681', '682', '683', '684', '685', '686', '707'])

        # Diabetic manifestations
        has_diab_uncomp = np.zeros(n_rows, dtype=bool)
        has_diab_acute = np.zeros(n_rows, dtype=bool)
        has_diab_comp = np.zeros(n_rows, dtype=bool)
        has_diab_renal = np.zeros(n_rows, dtype=bool)
        has_diab_neuro = np.zeros(n_rows, dtype=bool)
        has_diab_vasc = np.zeros(n_rows, dtype=bool)

        for col in ['diag_1', 'diag_2', 'diag_3']:
            raw_codes = np.array([str(x) for x in d[col]])
            has_diab_uncomp |= np.isin(raw_codes, ['250', '250.0', '250.00', '250.02'])
            has_diab_acute |= np.array([s.startswith(('250.1', '250.2', '250.3')) for s in raw_codes])
            has_diab_comp |= np.array([s.startswith(('250.4', '250.5', '250.6', '250.7', '250.8', '250.9')) for s in raw_codes])
            has_diab_renal |= np.array([s.startswith('250.4') for s in raw_codes])
            has_diab_neuro |= np.array([s.startswith('250.6') for s in raw_codes])
            has_diab_vasc |= np.array([s.startswith('250.7') for s in raw_codes])

        d['has_chf'] = has_chf.astype(float)
        d['has_ckd'] = has_ckd.astype(float)
        d['has_aki'] = has_aki.astype(float)
        d['has_cad'] = has_cad.astype(float)
        d['has_htn'] = has_htn.astype(float)
        d['has_copd'] = has_copd.astype(float)
        d['has_pvd'] = has_pvd.astype(float)
        d['has_stroke'] = has_stroke.astype(float)
        d['has_fluid_elect'] = has_fluid_elect.astype(float)
        d['has_pneumonia'] = has_pneumonia.astype(float)
        d['has_uti'] = has_uti.astype(float)
        d['has_sepsis'] = has_sepsis.astype(float)
        d['has_anemia'] = has_anemia.astype(float)
        d['has_afib'] = has_afib.astype(float)
        d['has_ulcer_skin'] = has_ulcer_skin.astype(float)
        d['has_diab_comp'] = has_diab_comp.astype(float)
        d['has_diab_acute'] = has_diab_acute.astype(float)
        d['cardiorenal_risk'] = ((has_chf | has_cad) & has_renal).astype(float)
        d['cardiorenal_anemia'] = (d['cardiorenal_risk'] > 0.5) & has_anemia
        d['cardiorenal_anemia'] = d['cardiorenal_anemia'].astype(float)
        d['cardiorespiratory'] = (has_chf & has_copd).astype(float)
        d['metabolic_sepsis'] = ((has_diab_acute | has_fluid_elect) & (has_sepsis | has_pneumonia | has_uti)).astype(float)
        d['vascular_burden'] = (has_cad.astype(float) + has_pvd.astype(float) + has_stroke.astype(float) + has_htn.astype(float))

        # Charlson Comorbidity Index (CCI)
        age_pts = np.clip(np.floor((d['age_num'] - 40.0) / 10.0), 0.0, 4.0)
        charlson = (
            1.0 * (has_mi | has_chf | has_pvd | has_stroke | has_dementia | has_copd | has_rheum | has_pud | has_mild_liver | has_diab_uncomp).astype(float)
            + 1.0 * has_diab_comp.astype(float)
            + 2.0 * (has_hemi | has_renal | has_neoplasm).astype(float)
            + 3.0 * has_sev_liver.astype(float)
            + 6.0 * (has_met_cancer | has_hiv).astype(float)
            + age_pts
        )
        d['charlson_score'] = charlson

        # Admission & Discharge Context
        disp = np.array([safe_str(x) for x in d['discharge_disposition_id']])
        d['disch_is_home'] = (disp == '1').astype(float)
        d['disch_is_snf'] = (disp == '3').astype(float)
        d['disch_is_home_health'] = np.isin(disp, ['6', '8']).astype(float)
        d['disch_is_transfer'] = np.isin(disp, ['2', '4', '5', '27', '29', '30']).astype(float)
        d['disch_is_rehab'] = (disp == '22').astype(float)
        d['disch_is_ama'] = np.isin(disp, ['7', '23', '24']).astype(float)
        d['disch_is_psych'] = (disp == '28').astype(float)
        d['disch_is_facility'] = np.isin(disp, ['2', '3', '4', '5', '6', '8', '22', '27', '28', '29', '30']).astype(float)

        adm_type = np.array([safe_str(x) for x in d['admission_type_id']])
        d['is_emergency_adm'] = (adm_type == '1').astype(float)
        d['is_urgent_adm'] = (adm_type == '2').astype(float)
        d['is_elective_adm'] = (adm_type == '3').astype(float)
        d['is_acute_adm'] = np.isin(adm_type, ['1', '2', '7']).astype(float)

        adm_src = np.array([safe_str(x) for x in d['admission_source_id']])
        d['adm_from_er'] = (adm_src == '7').astype(float)
        d['adm_from_referral'] = np.isin(adm_src, ['1', '2', '3']).astype(float)
        d['adm_from_transfer'] = np.isin(adm_src, ['4', '5', '6', '10', '17', '20']).astype(float)
        d['adm_from_snf'] = (adm_src == '5').astype(float)

        d['er_to_facility'] = (d['adm_from_er'] * d['disch_is_facility']).astype(float)
        d['transfer_in_and_out'] = (d['adm_from_transfer'] * d['disch_is_facility']).astype(float)
        d['elective_to_home'] = (d['is_elective_adm'] * d['disch_is_home']).astype(float)

        # LACE & LACE+ Index
        stay = np.maximum(0.0, d['time_in_hospital'])
        l_score = np.zeros(n_rows, dtype=float)
        l_score += (stay == 1).astype(float) * 1.0
        l_score += (stay == 2).astype(float) * 2.0
        l_score += (stay == 3).astype(float) * 3.0
        l_score += ((stay >= 4) & (stay <= 6)).astype(float) * 4.0
        l_score += ((stay >= 7) & (stay <= 13)).astype(float) * 5.0
        l_score += (stay >= 14).astype(float) * 7.0

        a_score = d['is_acute_adm'] * 3.0
        c_score = np.where(charlson >= 4.0, 5.0, np.clip(charlson, 0.0, 3.0))
        e_score = np.clip(d['number_emergency'], 0.0, 4.0)

        lace = l_score + a_score + c_score + e_score
        d['lace_score'] = lace
        d['lace_plus'] = lace + 2.0 * np.log1p(d['number_inpatient']) + 0.5 * (d['age_num'] / 10.0) + 1.5 * d['disch_is_facility']

        # Utilization & Intensity Dynamics
        tot_v = d['number_outpatient'] + d['number_emergency'] + d['number_inpatient']
        acute_v = d['number_emergency'] + d['number_inpatient']
        d['total_visits'] = tot_v.astype(float)
        d['acute_visits'] = acute_v.astype(float)
        d['has_prior_inpatient'] = (d['number_inpatient'] > 0).astype(float)
        d['frequent_inpatient'] = (d['number_inpatient'] >= 2).astype(float)
        d['super_utilizer'] = (acute_v >= 3).astype(float)
        d['zero_prior_visits'] = (tot_v == 0).astype(float)

        d['inpatient_ratio'] = d['number_inpatient'] / (tot_v + 1.0)
        d['emergency_ratio'] = d['number_emergency'] / (tot_v + 1.0)
        d['outpatient_ratio'] = d['number_outpatient'] / (tot_v + 1.0)

        log_inp = np.log1p(np.maximum(0.0, d['number_inpatient']))
        log_er = np.log1p(np.maximum(0.0, d['number_emergency']))
        log_out = np.log1p(np.maximum(0.0, d['number_outpatient']))
        log_tot = np.log1p(np.maximum(0.0, tot_v))

        d['log_inpatient'] = log_inp
        d['log_emergency'] = log_er
        d['log_outpatient'] = log_out
        d['log_total_visits'] = log_tot
        d['inpatient_squared'] = (d['number_inpatient'] ** 2).astype(float)

        d['med_per_day'] = d['num_medications'] / (stay + 1.0)
        d['lab_per_day'] = d['num_lab_procedures'] / (stay + 1.0)
        d['proc_per_day'] = d['num_procedures'] / (stay + 1.0)
        d['diag_per_day'] = d['number_diagnoses'] / (stay + 1.0)
        d['clinical_intensity'] = (d['num_lab_procedures'] + 4.0 * d['num_procedures'] + 2.0 * d['num_medications']) / (stay + 1.0)
        d['lab_per_med'] = d['num_lab_procedures'] / (d['num_medications'] + 1.0)
        d['lab_per_diag'] = d['num_lab_procedures'] / (d['number_diagnoses'] + 1.0)

        d['high_util_long_stay'] = ((d['number_inpatient'] > 0) & (stay >= 7)).astype(float)
        d['short_stay_acute'] = ((stay <= 2) & (d['is_emergency_adm'] > 0.5)).astype(float)
        d['utilization_x_stay'] = log_inp * np.log1p(stay)
        d['utilization_x_facility'] = log_inp * d['disch_is_facility']
        d['utilization_x_charlson'] = log_inp * (charlson / 5.0)
        d['utilization_x_age'] = log_inp * (d['age_num'] / 50.0)

        # Medication Profile & Regimen Complexity
        active = np.zeros(n_rows, dtype=float)
        changes = np.zeros(n_rows, dtype=float)
        ups = np.zeros(n_rows, dtype=float)
        downs = np.zeros(n_rows, dtype=float)
        for col in med_cols:
            if col in d:
                val = d[col]
                active += (val != 'No').astype(float)
                changes += np.isin(val, ['Up', 'Down']).astype(float)
                ups += (val == 'Up').astype(float)
                downs += (val == 'Down').astype(float)

        d['active_med_count'] = active
        d['med_change_count'] = changes
        d['med_up_count'] = ups
        d['med_down_count'] = downs

        insulin = d.get('insulin', np.array(['No'] * n_rows))
        d['on_insulin'] = (insulin != 'No').astype(float)
        d['insulin_up'] = (insulin == 'Up').astype(float)
        d['insulin_down'] = (insulin == 'Down').astype(float)
        d['insulin_steady'] = (insulin == 'Steady').astype(float)

        met = d.get('metformin', np.array(['No'] * n_rows))
        d['on_metformin'] = (met != 'No').astype(float)

        sulf = np.zeros(n_rows, dtype=bool)
        for s_col in ['glipizide', 'glyburide', 'glimepiride', 'chlorpropamide', 'tolbutamide', 'acetohexamide', 'tolazamide']:
            if s_col in d:
                sulf |= (d[s_col] != 'No')
        d['on_sulfonylurea'] = sulf.astype(float)

        tzd = np.zeros(n_rows, dtype=bool)
        for t_col in ['pioglitazone', 'rosiglitazone', 'troglitazone']:
            if t_col in d:
                tzd |= (d[t_col] != 'No')
        d['on_tzd'] = tzd.astype(float)

        other_oral = np.zeros(n_rows, dtype=bool)
        for o_col in ['repaglinide', 'nateglinide', 'acarbose', 'miglitol']:
            if o_col in d:
                other_oral |= (d[o_col] != 'No')
        d['on_other_oral'] = other_oral.astype(float)

        d['num_drug_classes'] = d['on_insulin'] + d['on_metformin'] + d['on_sulfonylurea'] + d['on_tzd'] + d['on_other_oral']
        d['insulin_plus_oral'] = ((d['on_insulin'] > 0.5) & ((d['on_metformin'] + d['on_sulfonylurea'] + d['on_tzd'] + d['on_other_oral']) > 0.5)).astype(float)
        d['polypharmacy'] = (d['num_medications'] >= 16).astype(float)
        d['high_med_intensity'] = ((d['num_medications'] / (stay + 1.0)) >= 4.0).astype(float)
        d['change_flag'] = (d['change'] == 'Ch').astype(float)
        d['diabetes_med_flag'] = (d['diabetesMed'] == 'Yes').astype(float)

        # Biomarkers & Care Quality
        a1c_res = np.array([safe_str(x) for x in d['A1Cresult']])
        glu_res = np.array([safe_str(x) for x in d['max_glu_serum']])
        d['a1c_tested'] = (a1c_res != 'None').astype(float)
        d['a1c_high'] = (a1c_res == '>8').astype(float)
        d['a1c_normal'] = (a1c_res == 'Norm').astype(float)
        d['glu_high'] = np.isin(glu_res, ['>200', '>300']).astype(float)
        d['glu_tested'] = (glu_res != 'None').astype(float)
        d['therapeutic_inertia'] = ((a1c_res == '>8') & (d['change'] == 'No')).astype(float)

        # Pathway Interaction Strings
        d['diag1_disch'] = np.array([f"{g}_{p}" for g, p in zip(d['diag_1_grp'], disp)], dtype=object)
        d['adm_disch'] = np.array([f"{a}_{p}" for a, p in zip(adm_src, disp)], dtype=object)
        d['age_disch'] = np.array([f"{a}_{p}" for a, p in zip(d['age'], disp)], dtype=object)

        # Hypoglycemia and Acute Diabetic Crises flags
        d['has_hypoglycemia'] = np.isin(c1, ['251']) | (c1 == '250.8') | (c2 == '250.8') | (c3 == '250.8')
        d['has_hypoglycemia'] = d['has_hypoglycemia'].astype(float)
        d['frailty_index'] = (
            d['charlson_score'] + (d['time_in_hospital'] >= 4).astype(float) +
            d['disch_is_facility'] + (d['number_inpatient'] > 0).astype(float)
        )

    # 1B. Low-Rank Tensor SVD Decomposition of Multi-Modal Clinical Profile
    svd_feature_keys = [
        'has_chf', 'has_ckd', 'has_aki', 'has_cad', 'has_htn', 'has_copd', 'has_pvd', 'has_stroke',
        'has_fluid_elect', 'has_pneumonia', 'has_uti', 'has_sepsis', 'has_anemia', 'has_afib',
        'has_diab_comp', 'has_diab_acute', 'has_hypoglycemia', 'cardiorenal_risk', 'vascular_burden',
        'disch_is_home', 'disch_is_snf', 'disch_is_facility', 'disch_is_ama',
        'is_emergency_adm', 'is_acute_adm', 'adm_from_er', 'adm_from_transfer',
        'log_inpatient', 'log_emergency', 'log_outpatient', 'time_in_hospital',
        'on_insulin', 'insulin_up', 'insulin_down', 'on_metformin', 'on_sulfonylurea',
        'active_med_count', 'med_change_count', 'a1c_tested', 'a1c_high'
    ]
    x_svd_tr = np.column_stack([np.nan_to_num(train[k], nan=0.0) for k in svd_feature_keys]).astype(float)
    x_svd_ev = np.column_stack([np.nan_to_num(eval_cols[k], nan=0.0) for k in svd_feature_keys]).astype(float)

    mu_svd = np.mean(x_svd_tr, axis=0)
    std_svd = np.std(x_svd_tr, axis=0)
    std_svd[std_svd < 1e-6] = 1.0

    x_svd_tr_std = (x_svd_tr - mu_svd) / std_svd
    x_svd_ev_std = (x_svd_ev - mu_svd) / std_svd

    cov_svd = (x_svd_tr_std.T @ x_svd_tr_std) / float(len(x_svd_tr_std))
    evals_s, evecs_s = np.linalg.eigh(cov_svd)
    sort_idx = np.argsort(evals_s)[::-1]
    n_svd_comps = 8
    top_v = evecs_s[:, sort_idx[:n_svd_comps]]

    z_tr_svd = x_svd_tr_std @ top_v
    z_ev_svd = x_svd_ev_std @ top_v
    recon_tr = z_tr_svd @ top_v.T
    recon_ev = z_ev_svd @ top_v.T

    for i in range(n_svd_comps):
        train[f'svd_comp_{i}'] = z_tr_svd[:, i]
        eval_cols[f'svd_comp_{i}'] = z_ev_svd[:, i]

    train['svd_recon_err'] = np.sum((x_svd_tr_std - recon_tr) ** 2, axis=1)
    eval_cols['svd_recon_err'] = np.sum((x_svd_ev_std - recon_ev) ** 2, axis=1)

    # 2. Out-of-Fold Hierarchical Empirical Bayes & Frequency Encodings
    prior_glob = float(np.mean(y_train))
    n_tr = len(y_train)

    def compute_smoothed_map(cats, targets, smooth=25.0, prior=0.114):
        counts, sums = {}, {}
        for c, y in zip(cats, targets):
            c_str = safe_str(c)
            counts[c_str] = counts.get(c_str, 0) + 1
            sums[c_str] = sums.get(c_str, 0.0) + y
        return {c: (sums[c] + smooth * prior) / (counts[c] + smooth) for c in counts}

    def compute_hierarchical_eb_map(full_codes, grp_codes, targets, m_grp=15.0, m_code=25.0, prior=0.114):
        # Level 1: Group / Chapter shrinkage to global prior
        grp_counts, grp_sums = {}, {}
        for g, y in zip(grp_codes, targets):
            gs = safe_str(g)
            grp_counts[gs] = grp_counts.get(gs, 0) + 1
            grp_sums[gs] = grp_sums.get(gs, 0.0) + y
        grp_post = {g: (grp_sums[g] + m_grp * prior) / (grp_counts[g] + m_grp) for g in grp_counts}

        # Level 2: Diagnosis code shrinkage to group posterior
        code_counts, code_sums, code_grp_map = {}, {}, {}
        for c, g, y in zip(full_codes, grp_codes, targets):
            cs = safe_str(c)
            code_counts[cs] = code_counts.get(cs, 0) + 1
            code_sums[cs] = code_sums.get(cs, 0.0) + y
            code_grp_map[cs] = safe_str(g)

        code_post = {}
        for c in code_counts:
            parent_prior = grp_post.get(code_grp_map[c], prior)
            code_post[c] = (code_sums[c] + m_code * parent_prior) / (code_counts[c] + m_code)
        return code_post, grp_post

    def compute_freq_map(arr):
        n = len(arr)
        counts = {}
        for x in arr:
            s = safe_str(x)
            counts[s] = counts.get(s, 0) + 1
        return {k: v / n for k, v in counts.items()}

    # Frequency mappings
    for f_name, col_key in [
        ('freq_diag_1', 'diag_1_3d'),
        ('freq_specialty', 'medical_specialty'),
        ('freq_discharge', 'discharge_disposition_id'),
        ('freq_adm_source', 'admission_source_id'),
        ('freq_payer', 'payer_code'),
    ]:
        f_map = compute_freq_map(train[col_key])
        train[f_name] = np.array([f_map.get(safe_str(x), 0.0) for x in train[col_key]], dtype=float)
        eval_cols[f_name] = np.array([f_map.get(safe_str(x), 0.0) for x in eval_cols[col_key]], dtype=float)

    skf_te = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    te_features = [
        'te_diag_1', 'te_diag_1_grp', 'te_diag_max', 'te_diag_mean', 'te_diag_sum', 'te_diag_min',
        'te_discharge', 'te_specialty', 'te_adm_source', 'te_adm_type', 'te_payer',
        'te_diag1_disch', 'te_adm_disch', 'te_age_disch'
    ]
    for tf in te_features:
        train[tf] = np.zeros(n_tr, dtype=float)

    for tr_idx, val_idx in skf_te.split(train['gender'], y_train):
        y_tr_f = y_train[tr_idx]

        heb_d1, grp_d1 = compute_hierarchical_eb_map(
            train['diag_1_3d'][tr_idx], train['diag_1_grp'][tr_idx], y_tr_f, m_grp=15.0, m_code=25.0, prior=prior_glob
        )
        train['te_diag_1'][val_idx] = np.array([
            heb_d1.get(safe_str(c), grp_d1.get(safe_str(g), prior_glob))
            for c, g in zip(train['diag_1_3d'][val_idx], train['diag_1_grp'][val_idx])
        ])
        train['te_diag_1_grp'][val_idx] = np.array([grp_d1.get(safe_str(c), prior_glob) for c in train['diag_1_grp'][val_idx]])

        pooled_d = np.concatenate([train['diag_1_3d'][tr_idx], train['diag_2_3d'][tr_idx], train['diag_3_3d'][tr_idx]])
        pooled_g = np.concatenate([train['diag_1_grp'][tr_idx], train['diag_2_grp'][tr_idx], train['diag_3_grp'][tr_idx]])
        pooled_y = np.concatenate([y_tr_f, y_tr_f, y_tr_f])
        heb_pool, grp_pool = compute_hierarchical_eb_map(pooled_d, pooled_g, pooled_y, m_grp=15.0, m_code=25.0, prior=prior_glob)

        pd1 = np.array([heb_pool.get(safe_str(c), grp_pool.get(safe_str(g), prior_glob)) for c, g in zip(train['diag_1_3d'][val_idx], train['diag_1_grp'][val_idx])])
        pd2 = np.array([heb_pool.get(safe_str(c), grp_pool.get(safe_str(g), prior_glob)) for c, g in zip(train['diag_2_3d'][val_idx], train['diag_2_grp'][val_idx])])
        pd3 = np.array([heb_pool.get(safe_str(c), grp_pool.get(safe_str(g), prior_glob)) for c, g in zip(train['diag_3_3d'][val_idx], train['diag_3_grp'][val_idx])])
        train['te_diag_max'][val_idx] = np.maximum(np.maximum(pd1, pd2), pd3)
        train['te_diag_min'][val_idx] = np.minimum(np.minimum(pd1, pd2), pd3)
        train['te_diag_mean'][val_idx] = (pd1 + pd2 + pd3) / 3.0
        train['te_diag_sum'][val_idx] = pd1 + pd2 + pd3

        m_disp = compute_smoothed_map(train['discharge_disposition_id'][tr_idx], y_tr_f, smooth=20.0, prior=prior_glob)
        train['te_discharge'][val_idx] = np.array([m_disp.get(safe_str(c), prior_glob) for c in train['discharge_disposition_id'][val_idx]])

        m_spec = compute_smoothed_map(train['medical_specialty'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_specialty'][val_idx] = np.array([m_spec.get(safe_str(c), prior_glob) for c in train['medical_specialty'][val_idx]])

        m_src = compute_smoothed_map(train['admission_source_id'][tr_idx], y_tr_f, smooth=20.0, prior=prior_glob)
        train['te_adm_source'][val_idx] = np.array([m_src.get(safe_str(c), prior_glob) for c in train['admission_source_id'][val_idx]])

        m_adm = compute_smoothed_map(train['admission_type_id'][tr_idx], y_tr_f, smooth=20.0, prior=prior_glob)
        train['te_adm_type'][val_idx] = np.array([m_adm.get(safe_str(c), prior_glob) for c in train['admission_type_id'][val_idx]])

        m_pay = compute_smoothed_map(train['payer_code'][tr_idx], y_tr_f, smooth=25.0, prior=prior_glob)
        train['te_payer'][val_idx] = np.array([m_pay.get(safe_str(c), prior_glob) for c in train['payer_code'][val_idx]])

        m_d1_disch = compute_smoothed_map(train['diag1_disch'][tr_idx], y_tr_f, smooth=30.0, prior=prior_glob)
        train['te_diag1_disch'][val_idx] = np.array([m_d1_disch.get(safe_str(c), prior_glob) for c in train['diag1_disch'][val_idx]])

        m_adm_disch = compute_smoothed_map(train['adm_disch'][tr_idx], y_tr_f, smooth=30.0, prior=prior_glob)
        train['te_adm_disch'][val_idx] = np.array([m_adm_disch.get(safe_str(c), prior_glob) for c in train['adm_disch'][val_idx]])

        m_age_disch = compute_smoothed_map(train['age_disch'][tr_idx], y_tr_f, smooth=30.0, prior=prior_glob)
        train['te_age_disch'][val_idx] = np.array([m_age_disch.get(safe_str(c), prior_glob) for c in train['age_disch'][val_idx]])

    # Full train-to-eval target mappings
    heb_d1_all, grp_d1_all = compute_hierarchical_eb_map(
        train['diag_1_3d'], train['diag_1_grp'], y_train, m_grp=15.0, m_code=25.0, prior=prior_glob
    )
    eval_cols['te_diag_1'] = np.array([
        heb_d1_all.get(safe_str(c), grp_d1_all.get(safe_str(g), prior_glob))
        for c, g in zip(eval_cols['diag_1_3d'], eval_cols['diag_1_grp'])
    ], dtype=float)
    eval_cols['te_diag_1_grp'] = np.array([grp_d1_all.get(safe_str(c), prior_glob) for c in eval_cols['diag_1_grp']], dtype=float)

    pooled_d_all = np.concatenate([train['diag_1_3d'], train['diag_2_3d'], train['diag_3_3d']])
    pooled_g_all = np.concatenate([train['diag_1_grp'], train['diag_2_grp'], train['diag_3_grp']])
    pooled_y_all = np.concatenate([y_train, y_train, y_train])
    heb_pool_all, grp_pool_all = compute_hierarchical_eb_map(pooled_d_all, pooled_g_all, pooled_y_all, m_grp=15.0, m_code=25.0, prior=prior_glob)
    epd1 = np.array([heb_pool_all.get(safe_str(c), grp_pool_all.get(safe_str(g), prior_glob)) for c, g in zip(eval_cols['diag_1_3d'], eval_cols['diag_1_grp'])])
    epd2 = np.array([heb_pool_all.get(safe_str(c), grp_pool_all.get(safe_str(g), prior_glob)) for c, g in zip(eval_cols['diag_2_3d'], eval_cols['diag_2_grp'])])
    epd3 = np.array([heb_pool_all.get(safe_str(c), grp_pool_all.get(safe_str(g), prior_glob)) for c, g in zip(eval_cols['diag_3_3d'], eval_cols['diag_3_grp'])])
    eval_cols['te_diag_max'] = np.maximum(np.maximum(epd1, epd2), epd3)
    eval_cols['te_diag_min'] = np.minimum(np.minimum(epd1, epd2), epd3)
    eval_cols['te_diag_mean'] = (epd1 + epd2 + epd3) / 3.0
    eval_cols['te_diag_sum'] = epd1 + epd2 + epd3

    m_disp_all = compute_smoothed_map(train['discharge_disposition_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_discharge'] = np.array([m_disp_all.get(safe_str(c), prior_glob) for c in eval_cols['discharge_disposition_id']], dtype=float)

    m_spec_all = compute_smoothed_map(train['medical_specialty'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_specialty'] = np.array([m_spec_all.get(safe_str(c), prior_glob) for c in eval_cols['medical_specialty']], dtype=float)

    m_src_all = compute_smoothed_map(train['admission_source_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_adm_source'] = np.array([m_src_all.get(safe_str(c), prior_glob) for c in eval_cols['admission_source_id']], dtype=float)

    m_adm_all = compute_smoothed_map(train['admission_type_id'], y_train, smooth=20.0, prior=prior_glob)
    eval_cols['te_adm_type'] = np.array([m_adm_all.get(safe_str(c), prior_glob) for c in eval_cols['admission_type_id']], dtype=float)

    m_pay_all = compute_smoothed_map(train['payer_code'], y_train, smooth=25.0, prior=prior_glob)
    eval_cols['te_payer'] = np.array([m_pay_all.get(safe_str(c), prior_glob) for c in eval_cols['payer_code']], dtype=float)

    m_d1_disch_all = compute_smoothed_map(train['diag1_disch'], y_train, smooth=30.0, prior=prior_glob)
    eval_cols['te_diag1_disch'] = np.array([m_d1_disch_all.get(safe_str(c), prior_glob) for c in eval_cols['diag1_disch']], dtype=float)

    m_adm_disch_all = compute_smoothed_map(train['adm_disch'], y_train, smooth=30.0, prior=prior_glob)
    eval_cols['te_adm_disch'] = np.array([m_adm_disch_all.get(safe_str(c), prior_glob) for c in eval_cols['adm_disch']], dtype=float)

    m_age_disch_all = compute_smoothed_map(train['age_disch'], y_train, smooth=30.0, prior=prior_glob)
    eval_cols['te_age_disch'] = np.array([m_age_disch_all.get(safe_str(c), prior_glob) for c in eval_cols['age_disch']], dtype=float)

    extra_num = [
        'age_num', 'total_visits', 'acute_visits', 'has_prior_inpatient', 'frequent_inpatient', 'super_utilizer', 'zero_prior_visits',
        'inpatient_ratio', 'emergency_ratio', 'outpatient_ratio',
        'log_inpatient', 'log_emergency', 'log_outpatient', 'log_total_visits', 'inpatient_squared',
        'med_per_day', 'lab_per_day', 'proc_per_day', 'diag_per_day', 'clinical_intensity',
        'lab_per_med', 'lab_per_diag', 'high_util_long_stay', 'short_stay_acute',
        'utilization_x_stay', 'utilization_x_facility', 'utilization_x_charlson', 'utilization_x_age',
        'active_med_count', 'med_change_count', 'med_up_count', 'med_down_count',
        'on_insulin', 'insulin_up', 'insulin_down', 'insulin_steady',
        'on_metformin', 'on_sulfonylurea', 'on_tzd', 'on_other_oral',
        'num_drug_classes', 'insulin_plus_oral', 'polypharmacy', 'high_med_intensity',
        'change_flag', 'diabetes_med_flag',
        'a1c_tested', 'a1c_high', 'a1c_normal', 'glu_high', 'glu_tested', 'therapeutic_inertia',
        'disch_is_home', 'disch_is_snf', 'disch_is_home_health', 'disch_is_transfer', 'disch_is_rehab',
        'disch_is_ama', 'disch_is_psych', 'disch_is_facility',
        'is_emergency_adm', 'is_urgent_adm', 'is_elective_adm', 'is_acute_adm',
        'adm_from_er', 'adm_from_referral', 'adm_from_transfer', 'adm_from_snf',
        'er_to_facility', 'transfer_in_and_out', 'elective_to_home',
        'has_chf', 'has_ckd', 'has_aki', 'has_cad', 'has_htn', 'has_copd', 'has_pvd', 'has_stroke',
        'has_fluid_elect', 'has_pneumonia', 'has_uti', 'has_sepsis', 'has_anemia', 'has_afib', 'has_ulcer_skin',
        'has_diab_comp', 'has_diab_acute', 'cardiorenal_risk', 'cardiorenal_anemia', 'cardiorespiratory', 'metabolic_sepsis', 'vascular_burden',
        'has_hypoglycemia', 'frailty_index', 'charlson_score', 'lace_score', 'lace_plus',
        'svd_comp_0', 'svd_comp_1', 'svd_comp_2', 'svd_comp_3', 'svd_comp_4', 'svd_comp_5', 'svd_comp_6', 'svd_comp_7', 'svd_recon_err',
        'freq_diag_1', 'freq_specialty', 'freq_discharge', 'freq_adm_source', 'freq_payer',
    ] + te_features

    extra_cat = [
        'diag_1_grp', 'diag_2_grp', 'diag_3_grp', 'diag_1_3d', 'diag_2_3d', 'diag_3_3d',
        'diag1_disch', 'adm_disch', 'age_disch'
    ]

    # 3. Encoding Matrix Assembly
    encoder = TabularEncoder(min_count=10)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(eval_cols)

    # 4. Quad-Engine GBDT Models across 4 Stratified Folds
    skf_models = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    oof_lgb1 = np.zeros(n_tr, dtype=float)
    oof_lgb2 = np.zeros(n_tr, dtype=float)
    oof_hgb = np.zeros(n_tr, dtype=float)
    oof_xgb = np.zeros(n_tr, dtype=float)

    n_eval_rows = len(eval_cols['gender'])
    eval_logits_lgb1 = np.zeros(n_eval_rows, dtype=float)
    eval_logits_lgb2 = np.zeros(n_eval_rows, dtype=float)
    eval_logits_hgb = np.zeros(n_eval_rows, dtype=float)
    eval_logits_xgb = np.zeros(n_eval_rows, dtype=float)

    def to_logit(p, eps=1e-6):
        p_c = np.clip(p, eps, 1.0 - eps)
        return np.log(p_c / (1.0 - p_c))

    for fold, (tr_idx, val_idx) in enumerate(skf_models.split(x_train, y_train)):
        x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
        x_va = x_train[val_idx]

        # Model 1: Standard Leaf-Wise LightGBM
        m_lgb1 = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.033,
            num_leaves=31,
            min_child_samples=55,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.8,
            reg_lambda=4.0,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m_lgb1.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_lgb1[val_idx] = m_lgb1.predict_proba(x_va)[:, 1]
        eval_logits_lgb1 += to_logit(m_lgb1.predict_proba(x_eval)[:, 1]) / 4.0

        # Model 2: Depth-Constrained Regularized LightGBM (Prevents spurious EHR branch overfit)
        m_lgb2 = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.035,
            max_depth=6,
            num_leaves=24,
            min_child_samples=75,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_alpha=2.0,
            reg_lambda=8.0,
            random_state=142 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m_lgb2.fit(x_tr, y_tr, categorical_feature=encoder.categorical_indices)
        oof_lgb2[val_idx] = m_lgb2.predict_proba(x_va)[:, 1]
        eval_logits_lgb2 += to_logit(m_lgb2.predict_proba(x_eval)[:, 1]) / 4.0

        # Model 3: HistGradientBoostingClassifier (Exact Histogram Binning)
        m_hgb = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.033,
            max_leaf_nodes=31,
            min_samples_leaf=55,
            l2_regularization=4.0,
            categorical_features=encoder.categorical_indices,
            random_state=42 + fold,
        )
        m_hgb.fit(x_tr, y_tr)
        oof_hgb[val_idx] = m_hgb.predict_proba(x_va)[:, 1]
        eval_logits_hgb += to_logit(m_hgb.predict_proba(x_eval)[:, 1]) / 4.0

        # Model 4: XGBoost with histogram tree method
        m_xgb = xgb.XGBClassifier(
            n_estimators=190,
            learning_rate=0.033,
            max_depth=5,
            min_child_weight=30,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.8,
            reg_lambda=4.0,
            random_state=42 + fold,
            n_jobs=1,
            tree_method='hist',
            eval_metric='logloss',
        )
        m_xgb.fit(x_tr, y_tr)
        oof_xgb[val_idx] = m_xgb.predict_proba(x_va)[:, 1]
        eval_logits_xgb += to_logit(m_xgb.predict_proba(x_eval)[:, 1]) / 4.0

    # 5. Smooth Wilcoxon Pairwise Rank-Surrogate Optimization via SLSQP
    z_oofs = np.column_stack([to_logit(oof_lgb1), to_logit(oof_lgb2), to_logit(oof_hgb), to_logit(oof_xgb)])

    # Sample balanced pairs for smooth Wilcoxon ranking surrogate
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    rng = np.random.RandomState(42)
    sample_pos = rng.choice(pos_idx, size=min(1500, len(pos_idx)), replace=False)
    sample_neg = rng.choice(neg_idx, size=min(2500, len(neg_idx)), replace=False)

    z_pos = z_oofs[sample_pos]
    z_neg = z_oofs[sample_neg]
    n_p = len(sample_pos)
    n_n = len(sample_neg)
    tau = 0.25
    lam_reg = 0.02

    def rank_surrogate_loss_and_grad(w):
        u_p = z_pos @ w
        u_n = z_neg @ w
        diff = (u_n[:, None] - u_p[None, :]) / tau
        diff_clipped = np.clip(diff, -20.0, 20.0)
        sig = 1.0 / (1.0 + np.exp(-diff_clipped))
        loss = np.mean(sig) + 0.5 * lam_reg * np.sum((w - 0.25) ** 2)

        g_matrix = (sig * (1.0 - sig)) / (tau * n_p * n_n)
        g_neg = np.sum(g_matrix, axis=1)
        g_pos = -np.sum(g_matrix, axis=0)
        grad = z_neg.T @ g_neg + z_pos.T @ g_pos + lam_reg * (w - 0.25)
        return float(loss), grad.astype(float)

    w_init = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    res = minimize(
        rank_surrogate_loss_and_grad,
        w_init,
        method='SLSQP',
        jac=True,
        bounds=[(0.02, 0.85)] * 4,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        options={'maxiter': 50, 'ftol': 1e-6}
    )
    if res.success and np.all(res.x >= 0):
        best_w = res.x / np.sum(res.x)
    else:
        best_w = w_init

    # Reconstruct final probability prediction in calibrated logit evidence space
    z_final = (
        best_w[0] * eval_logits_lgb1 +
        best_w[1] * eval_logits_lgb2 +
        best_w[2] * eval_logits_hgb +
        best_w[3] * eval_logits_xgb
    )
    pred = 1.0 / (1.0 + np.exp(-z_final))
    return np.clip(np.nan_to_num(pred, nan=prior_glob), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
