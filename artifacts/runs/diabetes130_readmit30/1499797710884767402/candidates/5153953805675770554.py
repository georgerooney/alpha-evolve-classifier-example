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
    """Fit on `train`/`y_train` with Holographic Clinical Phenotyping and Tri-Model Rank Ensemble."""
    from sklearn.model_selection import StratifiedKFold
    try:
        from scipy.stats import rankdata
    except Exception:
        def rankdata(a):
            order = np.argsort(a)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(a) + 1)
            return ranks

    # Resolve ICD-9 grouper
    icd9_fn = globals().get("icd9_group")
    if icd9_fn is None:
        try:
            from clinical_toolkit import icd9_group as icd9_fn
        except Exception:
            def icd9_fn(code):
                if code is None or code == "" or code == "?":
                    return "missing"
                s = str(code).strip()
                if not s or s == "None":
                    return "missing"
                if s[0] in ("V", "v", "E", "e"):
                    return "other"
                try:
                    val = float(s.split(".")[0])
                except ValueError:
                    return "other"
                if (390 <= val <= 459) or val == 785:
                    return "circulatory"
                if (460 <= val <= 519) or val == 786:
                    return "respiratory"
                if (520 <= val <= 579) or val == 787:
                    return "digestive"
                if val == 250:
                    return "diabetes"
                if 800 <= val <= 999:
                    return "injury"
                if 710 <= val <= 739:
                    return "musculoskeletal"
                if (580 <= val <= 629) or val == 788:
                    return "genitourinary"
                if 140 <= val <= 239:
                    return "neoplasms"
                return "other"

    # Shallow copies of input dictionaries for feature attachment
    train_ext = dict(train)
    eval_ext = dict(eval_cols)
    n_train = len(y_train)
    any_eval_key = next(iter(eval_cols.keys()))
    n_eval = len(eval_cols[any_eval_key])

    def parse_prefix(code):
        if code is None or code == "" or code == "?":
            return "missing"
        s = str(code).strip()
        if not s or s == "None":
            return "missing"
        return s.split(".")[0].upper()

    # 1. Holographic Clinical Feature Extraction
    def extract_phenotypic_features(data_dict: Dict[str, np.ndarray], n_rows: int):
        t_hosp = np.nan_to_num(data_dict.get("time_in_hospital", np.zeros(n_rows)).astype(float), nan=4.0)
        n_labs = np.nan_to_num(data_dict.get("num_lab_procedures", np.zeros(n_rows)).astype(float), nan=43.0)
        n_procs = np.nan_to_num(data_dict.get("num_procedures", np.zeros(n_rows)).astype(float), nan=1.0)
        n_meds = np.nan_to_num(data_dict.get("num_medications", np.zeros(n_rows)).astype(float), nan=16.0)
        n_outp = np.nan_to_num(data_dict.get("number_outpatient", np.zeros(n_rows)).astype(float), nan=0.0)
        n_emerg = np.nan_to_num(data_dict.get("number_emergency", np.zeros(n_rows)).astype(float), nan=0.0)
        n_inpat = np.nan_to_num(data_dict.get("number_inpatient", np.zeros(n_rows)).astype(float), nan=0.0)
        n_diag = np.nan_to_num(data_dict.get("number_diagnoses", np.zeros(n_rows)).astype(float), nan=7.0)

        # Acute utilization dynamics & velocity
        tot_prior = n_inpat + n_emerg + n_outp
        acute_prior = n_inpat + n_emerg
        inpat_ratio = n_inpat / (tot_prior + 1.0)
        emerg_ratio = n_emerg / (tot_prior + 1.0)
        has_inpatient = (n_inpat > 0.0).astype(float)
        frequent_inpatient = (n_inpat >= 2.0).astype(float)
        heavy_inpatient = (n_inpat >= 3.0).astype(float)
        has_emergency = (n_emerg > 0.0).astype(float)
        acute_care_score = 3.0 * n_inpat + 1.5 * n_emerg + 0.2 * n_outp
        inpat_velocity = (n_inpat ** 2) / (t_hosp + 1.0)

        # Non-linear log transforms
        log_inpatient = np.log1p(n_inpat)
        log_emergency = np.log1p(n_emerg)
        log_outpatient = np.log1p(n_outp)
        log_tot_prior = np.log1p(tot_prior)
        log_time_hosp = np.log1p(t_hosp)
        log_num_meds = np.log1p(n_meds)
        log_num_labs = np.log1p(n_labs)

        # Encounter care intensity ratios
        lab_intensity = n_labs / (t_hosp + 1.0)
        med_intensity = n_meds / (t_hosp + 1.0)
        proc_intensity = n_procs / (t_hosp + 1.0)
        diag_intensity = n_diag / (t_hosp + 1.0)
        action_intensity = (n_labs + 5.0 * n_procs + 2.0 * n_meds) / (t_hosp + 1.0)
        med_per_diag = n_meds / (n_diag + 1.0)
        lab_per_proc = n_labs / (n_procs + 1.0)
        short_stay = (t_hosp <= 2.0).astype(float)
        long_stay = (t_hosp >= 7.0).astype(float)
        high_med_count = (n_meds >= 20.0).astype(float)

        # Continuous age parsing
        age_map = {
            "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
            "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
            "[80-90)": 85.0, "[90-100)": 95.0
        }
        raw_age = data_dict.get("age", np.empty(n_rows, dtype=object))
        age_continuous = np.array([age_map.get(a, 65.0) if a is not None else 65.0 for a in raw_age], dtype=float)

        # Expanded Comorbidity Syndemic Screening across diag 1-3
        diag1 = data_dict.get("diag_1", np.empty(n_rows, dtype=object))
        diag2 = data_dict.get("diag_2", np.empty(n_rows, dtype=object))
        diag3 = data_dict.get("diag_3", np.empty(n_rows, dtype=object))

        diag_1_p = np.array([parse_prefix(c) for c in diag1], dtype=object)
        diag_2_p = np.array([parse_prefix(c) for c in diag2], dtype=object)
        diag_3_p = np.array([parse_prefix(c) for c in diag3], dtype=object)

        has_chf = np.zeros(n_rows, dtype=float)
        has_cad = np.zeros(n_rows, dtype=float)
        has_ckd = np.zeros(n_rows, dtype=float)
        has_copd = np.zeros(n_rows, dtype=float)
        has_htn = np.zeros(n_rows, dtype=float)
        has_diab_comp = np.zeros(n_rows, dtype=float)
        has_diab_acute = np.zeros(n_rows, dtype=float)
        has_infection = np.zeros(n_rows, dtype=float)
        has_stroke = np.zeros(n_rows, dtype=float)
        has_pvd = np.zeros(n_rows, dtype=float)
        has_dysrhythmia = np.zeros(n_rows, dtype=float)
        has_fluid_electrolyte = np.zeros(n_rows, dtype=float)
        has_anemia = np.zeros(n_rows, dtype=float)
        has_cancer = np.zeros(n_rows, dtype=float)
        has_liver = np.zeros(n_rows, dtype=float)

        for i in range(n_rows):
            codes = (diag1[i], diag2[i], diag3[i])
            for c in codes:
                if c is None or c == "" or c == "?":
                    continue
                s = str(c).strip()
                p = s.split(".")[0].upper()
                try:
                    val = int(p)
                except ValueError:
                    val = -1

                if p == "428":
                    has_chf[i] = 1.0
                if p in ("410", "411", "412", "413", "414"):
                    has_cad[i] = 1.0
                if p == "427":
                    has_dysrhythmia[i] = 1.0
                if p in ("585", "586", "584", "582", "583"):
                    has_ckd[i] = 1.0
                if p in ("491", "492", "493", "494", "496"):
                    has_copd[i] = 1.0
                if p in ("401", "402", "403", "404", "405"):
                    has_htn[i] = 1.0
                if 430 <= val <= 438:
                    has_stroke[i] = 1.0
                if 440 <= val <= 445:
                    has_pvd[i] = 1.0
                if p in ("038", "480", "481", "482", "485", "486", "599", "681", "682"):
                    has_infection[i] = 1.0
                if p == "276":
                    has_fluid_electrolyte[i] = 1.0
                if 280 <= val <= 285:
                    has_anemia[i] = 1.0
                if (140 <= val <= 208) or (230 <= val <= 239):
                    has_cancer[i] = 1.0
                if p in ("571", "572"):
                    has_liver[i] = 1.0
                if p == "250":
                    if any(s.startswith(x) for x in ("250.1", "250.2", "250.3")):
                        has_diab_acute[i] = 1.0
                    if not any(s.endswith(x) for x in ("00", "01", "02", "03")):
                        has_diab_comp[i] = 1.0

        comorbidity_count = (
            has_chf + has_cad + has_ckd + has_copd + has_htn +
            has_diab_comp + has_infection + has_stroke + has_pvd +
            has_dysrhythmia + has_fluid_electrolyte + has_anemia +
            has_cancer + has_liver
        )
        # Weighted Charlson Index proxy
        charlson_index = (
            2.0 * has_chf + 2.0 * has_ckd + 2.0 * has_cancer + 2.0 * has_stroke +
            1.5 * has_liver + 1.0 * has_cad + 1.0 * has_copd + 1.0 * has_pvd +
            1.0 * has_diab_comp + 1.0 * has_infection + 1.0 * has_fluid_electrolyte +
            0.5 * has_htn + 0.5 * has_anemia
        )

        # Antidiabetic Polypharmacy & Medication Regimen Complexity
        med_columns = [
            "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
            "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
            "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
            "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
        ]
        num_active_meds = np.zeros(n_rows, dtype=float)
        num_meds_up = np.zeros(n_rows, dtype=float)
        num_meds_down = np.zeros(n_rows, dtype=float)
        num_meds_changed = np.zeros(n_rows, dtype=float)

        for m_col in med_columns:
            if m_col in data_dict:
                m_arr = data_dict[m_col]
                for i, v in enumerate(m_arr):
                    if v is not None and v not in ("No", "", "?", "None"):
                        num_active_meds[i] += 1.0
                        if v == "Up":
                            num_meds_up[i] += 1.0
                            num_meds_changed[i] += 1.0
                        elif v == "Down":
                            num_meds_down[i] += 1.0
                            num_meds_changed[i] += 1.0

        polypharmacy_flag = (num_active_meds >= 3.0).astype(float)
        raw_ins = data_dict.get("insulin")
        insulin_on = np.zeros(n_rows, dtype=float)
        insulin_up = np.zeros(n_rows, dtype=float)
        insulin_down = np.zeros(n_rows, dtype=float)
        insulin_changed = np.zeros(n_rows, dtype=float)
        if raw_ins is not None:
            for i, v in enumerate(raw_ins):
                if v is not None and v not in ("No", "", "?", "None"):
                    insulin_on[i] = 1.0
                    if v == "Up":
                        insulin_up[i] = 1.0
                        insulin_changed[i] = 1.0
                    elif v == "Down":
                        insulin_down[i] = 1.0
                        insulin_changed[i] = 1.0

        raw_met = data_dict.get("metformin")
        metformin_on = np.zeros(n_rows, dtype=float)
        if raw_met is not None:
            for i, v in enumerate(raw_met):
                if v is not None and v not in ("No", "", "?", "None"):
                    metformin_on[i] = 1.0

        # Regimen complexity flags
        insulin_and_metformin = (insulin_on * metformin_on)
        insulin_only = (insulin_on * (num_active_meds == 1.0).astype(float))
        oral_only = ((1.0 - insulin_on) * (num_active_meds > 0.0).astype(float))

        # Glycemic tracking flags & clinical inertia
        raw_a1c = data_dict.get("A1Cresult")
        a1c_tested = np.zeros(n_rows, dtype=float)
        a1c_high = np.zeros(n_rows, dtype=float)
        a1c_very_high = np.zeros(n_rows, dtype=float)
        if raw_a1c is not None:
            for i, v in enumerate(raw_a1c):
                if v is not None and v not in ("None", "", "?"):
                    a1c_tested[i] = 1.0
                    if v in (">7", ">8"):
                        a1c_high[i] = 1.0
                    if v == ">8":
                        a1c_very_high[i] = 1.0

        raw_glu = data_dict.get("max_glu_serum")
        glu_tested = np.zeros(n_rows, dtype=float)
        glu_high = np.zeros(n_rows, dtype=float)
        if raw_glu is not None:
            for i, v in enumerate(raw_glu):
                if v is not None and v not in ("None", "", "?"):
                    glu_tested[i] = 1.0
                    if v in (">200", ">300"):
                        glu_high[i] = 1.0

        # Clinical inertia: severely uncontrolled A1C without any med change
        clinical_inertia = (a1c_very_high * (num_meds_changed == 0.0).astype(float))

        # Discharge & Admission Pathway Markers
        raw_disp = data_dict.get("discharge_disposition_id")
        disp_is_home = np.array([1.0 if str(d) == "1" else 0.0 for d in raw_disp], dtype=float)
        disp_is_snf = np.array([1.0 if str(d) == "3" else 0.0 for d in raw_disp], dtype=float)
        disp_is_home_health = np.array([1.0 if str(d) == "6" else 0.0 for d in raw_disp], dtype=float)
        disp_is_ama = np.array([1.0 if str(d) == "7" else 0.0 for d in raw_disp], dtype=float)
        disp_is_post_acute = np.array([1.0 if str(d) in ("3", "4", "5", "6", "22") else 0.0 for d in raw_disp], dtype=float)

        raw_adm_src = data_dict.get("admission_source_id")
        adm_from_er = np.array([1.0 if str(s) == "7" else 0.0 for s in raw_adm_src], dtype=float)
        adm_from_referral = np.array([1.0 if str(s) in ("1", "2") else 0.0 for s in raw_adm_src], dtype=float)
        adm_is_transfer = np.array([1.0 if str(s) in ("4", "5", "6") else 0.0 for s in raw_adm_src], dtype=float)

        # Cross interactions with prior utilization
        inpat_x_emerg = n_inpat * n_emerg
        inpat_x_age = n_inpat * (age_continuous / 10.0)
        inpat_x_post_acute = n_inpat * disp_is_post_acute
        inpat_x_diag = n_inpat * n_diag

        # Missingness indicators
        spec_arr = data_dict.get("medical_specialty")
        spec_missing = np.array([1.0 if (s is None or s in ("?", "None", "")) else 0.0 for s in spec_arr], dtype=float)
        payer_arr = data_dict.get("payer_code")
        payer_missing = np.array([1.0 if (p is None or p in ("?", "None", "")) else 0.0 for p in payer_arr], dtype=float)

        # ICD-9 Chapters
        diag_1_grp = np.array([icd9_fn(c) for c in diag1], dtype=object)
        diag_2_grp = np.array([icd9_fn(c) for c in diag2], dtype=object)
        diag_3_grp = np.array([icd9_fn(c) for c in diag3], dtype=object)

        numeric_feats = {
            "feat_tot_prior": tot_prior,
            "feat_acute_prior": acute_prior,
            "feat_inpat_ratio": inpat_ratio,
            "feat_emerg_ratio": emerg_ratio,
            "feat_has_inpatient": has_inpatient,
            "feat_frequent_inpatient": frequent_inpatient,
            "feat_heavy_inpatient": heavy_inpatient,
            "feat_has_emergency": has_emergency,
            "feat_acute_care_score": acute_care_score,
            "feat_inpat_velocity": inpat_velocity,
            "feat_log_inpatient": log_inpatient,
            "feat_log_emergency": log_emergency,
            "feat_log_outpatient": log_outpatient,
            "feat_log_tot_prior": log_tot_prior,
            "feat_log_time_hosp": log_time_hosp,
            "feat_log_num_meds": log_num_meds,
            "feat_log_num_labs": log_num_labs,
            "feat_lab_intensity": lab_intensity,
            "feat_med_intensity": med_intensity,
            "feat_proc_intensity": proc_intensity,
            "feat_diag_intensity": diag_intensity,
            "feat_action_intensity": action_intensity,
            "feat_med_per_diag": med_per_diag,
            "feat_lab_per_proc": lab_per_proc,
            "feat_short_stay": short_stay,
            "feat_long_stay": long_stay,
            "feat_high_med_count": high_med_count,
            "feat_age_continuous": age_continuous,
            "feat_has_chf": has_chf,
            "feat_has_cad": has_cad,
            "feat_has_ckd": has_ckd,
            "feat_has_copd": has_copd,
            "feat_has_htn": has_htn,
            "feat_has_stroke": has_stroke,
            "feat_has_pvd": has_pvd,
            "feat_has_dysrhythmia": has_dysrhythmia,
            "feat_has_diab_comp": has_diab_comp,
            "feat_has_diab_acute": has_diab_acute,
            "feat_has_infection": has_infection,
            "feat_has_fluid_electrolyte": has_fluid_electrolyte,
            "feat_has_anemia": has_anemia,
            "feat_has_cancer": has_cancer,
            "feat_has_liver": has_liver,
            "feat_comorbidity_count": comorbidity_count,
            "feat_charlson_index": charlson_index,
            "feat_num_active_meds": num_active_meds,
            "feat_num_meds_up": num_meds_up,
            "feat_num_meds_down": num_meds_down,
            "feat_num_meds_changed": num_meds_changed,
            "feat_polypharmacy_flag": polypharmacy_flag,
            "feat_insulin_on": insulin_on,
            "feat_insulin_up": insulin_up,
            "feat_insulin_down": insulin_down,
            "feat_insulin_changed": insulin_changed,
            "feat_insulin_and_metformin": insulin_and_metformin,
            "feat_insulin_only": insulin_only,
            "feat_oral_only": oral_only,
            "feat_a1c_tested": a1c_tested,
            "feat_a1c_high": a1c_high,
            "feat_a1c_very_high": a1c_very_high,
            "feat_glu_tested": glu_tested,
            "feat_glu_high": glu_high,
            "feat_clinical_inertia": clinical_inertia,
            "feat_disp_is_home": disp_is_home,
            "feat_disp_is_snf": disp_is_snf,
            "feat_disp_is_home_health": disp_is_home_health,
            "feat_disp_is_ama": disp_is_ama,
            "feat_disp_is_post_acute": disp_is_post_acute,
            "feat_adm_from_er": adm_from_er,
            "feat_adm_from_referral": adm_from_referral,
            "feat_adm_is_transfer": adm_is_transfer,
            "feat_inpat_x_emerg": inpat_x_emerg,
            "feat_inpat_x_age": inpat_x_age,
            "feat_inpat_x_post_acute": inpat_x_post_acute,
            "feat_inpat_x_diag": inpat_x_diag,
            "feat_spec_missing": spec_missing,
            "feat_payer_missing": payer_missing,
        }

        categorical_feats = {
            "diag_1_group": diag_1_grp,
            "diag_2_group": diag_2_grp,
            "diag_3_group": diag_3_grp,
            "diag_1_prefix": diag_1_p,
            "diag_2_prefix": diag_2_p,
            "diag_3_prefix": diag_3_p,
        }

        return numeric_feats, categorical_feats

    train_num, train_cat = extract_phenotypic_features(train, n_train)
    eval_num, eval_cat = extract_phenotypic_features(eval_cols, n_eval)

    for k, v in train_num.items():
        train_ext[k] = v
        eval_ext[k] = eval_num[k]
    for k, v in train_cat.items():
        train_ext[k] = v
        eval_ext[k] = eval_cat[k]

    # Clinical interaction pairs
    disp_tr = train.get("discharge_disposition_id")
    disp_ev = eval_cols.get("discharge_disposition_id")
    adm_src_tr = train.get("admission_source_id")
    adm_src_ev = eval_cols.get("admission_source_id")
    d1_grp_tr = train_cat["diag_1_group"]
    d1_grp_ev = eval_cat["diag_1_group"]

    inter_d_grp_tr = np.array([f"{d}_{g}" for d, g in zip(disp_tr, d1_grp_tr)], dtype=object)
    inter_d_grp_ev = np.array([f"{d}_{g}" for d, g in zip(disp_ev, d1_grp_ev)], dtype=object)
    train_ext["discharge_x_diag1"] = inter_d_grp_tr
    eval_ext["discharge_x_diag1"] = inter_d_grp_ev

    inter_d_s_tr = np.array([f"{d}_{s}" for d, s in zip(disp_tr, adm_src_tr)], dtype=object)
    inter_d_s_ev = np.array([f"{d}_{s}" for d, s in zip(disp_ev, adm_src_ev)], dtype=object)
    train_ext["discharge_x_source"] = inter_d_s_tr
    eval_ext["discharge_x_source"] = inter_d_s_ev

    # 2. Bayesian Out-of-Fold Target Encoding & Frequency Encoding
    def compute_oof_target_enc(tr_arr: np.ndarray, y: np.ndarray, ev_arr: np.ndarray, m: float = 30.0):
        mu = float(np.mean(y))
        oof_tr = np.full(len(tr_arr), mu, dtype=np.float64)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for tr_idx, val_idx in skf.split(tr_arr, y):
            f_tr_v = tr_arr[tr_idx]
            f_tr_y = y[tr_idx]
            f_val_v = tr_arr[val_idx]
            sums = {}
            counts = {}
            for val, lbl in zip(f_tr_v, f_tr_y):
                s_key = str(val) if val is not None else "missing"
                sums[s_key] = sums.get(s_key, 0.0) + lbl
                counts[s_key] = counts.get(s_key, 0) + 1
            fold_res = np.empty(len(f_val_v), dtype=np.float64)
            for i, val in enumerate(f_val_v):
                s_key = str(val) if val is not None else "missing"
                c = counts.get(s_key, 0)
                s = sums.get(s_key, 0.0)
                fold_res[i] = (s + m * mu) / (c + m)
            oof_tr[val_idx] = fold_res

        full_sums = {}
        full_counts = {}
        for val, lbl in zip(tr_arr, y):
            s_key = str(val) if val is not None else "missing"
            full_sums[s_key] = full_sums.get(s_key, 0.0) + lbl
            full_counts[s_key] = full_counts.get(s_key, 0) + 1
        oof_ev = np.empty(len(ev_arr), dtype=np.float64)
        for i, val in enumerate(ev_arr):
            s_key = str(val) if val is not None else "missing"
            c = full_counts.get(s_key, 0)
            s = full_sums.get(s_key, 0.0)
            oof_ev[i] = (s + m * mu) / (c + m)
        return oof_tr, oof_ev

    def compute_freq_enc(tr_arr: np.ndarray, ev_arr: np.ndarray):
        counts = {}
        for val in tr_arr:
            s_key = str(val) if val is not None else "missing"
            counts[s_key] = counts.get(s_key, 0) + 1
        n = float(len(tr_arr))
        freq_tr = np.array([counts.get(str(val) if val is not None else "missing", 0) / n for val in tr_arr], dtype=float)
        freq_ev = np.array([counts.get(str(val) if val is not None else "missing", 0) / n for val in ev_arr], dtype=float)
        return freq_tr, freq_ev

    te_configs = [
        ("discharge_disposition_id", 30.0),
        ("admission_source_id", 30.0),
        ("admission_type_id", 30.0),
        ("medical_specialty", 35.0),
        ("diag_1_prefix", 25.0),
        ("diag_1_group", 20.0),
        ("diag_2_prefix", 30.0),
        ("discharge_x_diag1", 25.0),
        ("discharge_x_source", 25.0),
    ]

    added_numeric_cols = []
    for te_col, m_val in te_configs:
        te_tr, te_ev = compute_oof_target_enc(train_ext[te_col], y_train, eval_ext[te_col], m=m_val)
        out_name = f"te_{te_col}"
        train_ext[out_name] = te_tr
        eval_ext[out_name] = te_ev
        added_numeric_cols.append(out_name)

    freq_cols = ["discharge_disposition_id", "admission_source_id", "medical_specialty", "diag_1_prefix"]
    for f_col in freq_cols:
        fr_tr, fr_ev = compute_freq_enc(train_ext[f_col], eval_ext[f_col])
        f_name = f"freq_{f_col}"
        train_ext[f_name] = fr_tr
        eval_ext[f_name] = fr_ev
        added_numeric_cols.append(f_name)

    extra_numeric = list(train_num.keys()) + added_numeric_cols
    extra_categorical = list(train_cat.keys()) + ["discharge_x_diag1", "discharge_x_source"]

    # 3. Fit TabularEncoder
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_ext, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_ext)
    x_eval = encoder.transform(eval_ext)

    # 4. Quad-Model Diverse Ensemble
    # Model 1: Deep Regularized LightGBM (complex interaction learning)
    m1 = lgb.LGBMClassifier(
        n_estimators=360,
        learning_rate=0.03,
        num_leaves=32,
        min_child_samples=35,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=3.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p1 = np.nan_to_num(m1.predict_proba(x_eval)[:, 1], nan=0.114)

    # Model 2: Shallow Conservative LightGBM (high-regularization anchor)
    m2 = lgb.LGBMClassifier(
        n_estimators=420,
        learning_rate=0.022,
        num_leaves=16,
        min_child_samples=70,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=0.8,
        reg_lambda=6.0,
        random_state=202,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p2 = np.nan_to_num(m2.predict_proba(x_eval)[:, 1], nan=0.114)

    # Model 3: Extremely Randomized Trees LightGBM (variance reduction via random thresholding)
    m3 = lgb.LGBMClassifier(
        n_estimators=380,
        learning_rate=0.03,
        num_leaves=28,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.65,
        extra_trees=True,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=303,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p3 = np.nan_to_num(m3.predict_proba(x_eval)[:, 1], nan=0.114)

    # Model 4: XGBoost with histogram binning (or LightGBM fallback)
    xgb_mod = globals().get("xgb")
    p4 = None
    if xgb_mod is not None:
        try:
            m4 = xgb_mod.XGBClassifier(
                n_estimators=280,
                learning_rate=0.035,
                max_depth=5,
                min_child_weight=22,
                subsample=0.8,
                colsample_bytree=0.65,
                reg_alpha=0.3,
                reg_lambda=3.5,
                tree_method="hist",
                random_state=404,
                n_jobs=1,
                eval_metric="auc",
            )
            m4.fit(x_train, y_train)
            p4 = np.nan_to_num(m4.predict_proba(x_eval)[:, 1], nan=0.114)
        except Exception:
            p4 = None

    if p4 is None:
        m4 = lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=22,
            min_child_samples=50,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.3,
            reg_lambda=2.5,
            random_state=404,
            n_jobs=1,
            verbose=-1,
        )
        m4.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
        p4 = np.nan_to_num(m4.predict_proba(x_eval)[:, 1], nan=0.114)

    # 5. Percentile Rank-Norm Fusion
    def rank_norm(preds: np.ndarray) -> np.ndarray:
        n = len(preds)
        if n <= 1:
            return preds
        r = rankdata(preds)
        return (r - 1.0) / (n - 1.0)

    final_preds = (
        0.30 * rank_norm(p1) +
        0.25 * rank_norm(p2) +
        0.25 * rank_norm(p3) +
        0.20 * rank_norm(p4)
    )
    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
