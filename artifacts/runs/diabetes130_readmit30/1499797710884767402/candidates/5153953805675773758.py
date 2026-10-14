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
    """Fit on `train`/`y_train` with Multi-Resolution Clinical Phenotyping and Quad-Model Rank Ensemble."""
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
        inpat_ratio = n_inpat / (tot_prior + 1.0)
        emerg_ratio = n_emerg / (tot_prior + 1.0)
        has_inpatient = (n_inpat > 0.0).astype(float)
        frequent_inpatient = (n_inpat >= 2.0).astype(float)
        acute_care_score = 3.0 * n_inpat + 1.5 * n_emerg + 0.2 * n_outp
        inpat_velocity = (n_inpat ** 2) / (t_hosp + 1.0)

        # Encounter care intensity ratios
        lab_intensity = n_labs / (t_hosp + 1.0)
        med_intensity = n_meds / (t_hosp + 1.0)
        proc_intensity = n_procs / (t_hosp + 1.0)
        diag_intensity = n_diag / (t_hosp + 1.0)
        action_intensity = (n_labs + n_procs + n_meds) / (t_hosp + 1.0)
        med_per_diag = n_meds / (n_diag + 1.0)

        # Continuous age parsing
        age_map = {
            "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
            "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
            "[80-90)": 85.0, "[90-100)": 95.0
        }
        raw_age = data_dict.get("age", np.empty(n_rows, dtype=object))
        age_continuous = np.array([age_map.get(a, 65.0) if a is not None else 65.0 for a in raw_age], dtype=float)

        # Comorbidity Syndemic Screening across diag 1-3
        diag1 = data_dict.get("diag_1", np.empty(n_rows, dtype=object))
        diag2 = data_dict.get("diag_2", np.empty(n_rows, dtype=object))
        diag3 = data_dict.get("diag_3", np.empty(n_rows, dtype=object))

        has_chf = np.zeros(n_rows, dtype=float)
        has_cad = np.zeros(n_rows, dtype=float)
        has_ckd = np.zeros(n_rows, dtype=float)
        has_copd = np.zeros(n_rows, dtype=float)
        has_htn = np.zeros(n_rows, dtype=float)
        has_diab_comp = np.zeros(n_rows, dtype=float)
        has_infection = np.zeros(n_rows, dtype=float)

        for i in range(n_rows):
            c1 = str(diag1[i]) if diag1[i] is not None else ""
            c2 = str(diag2[i]) if diag2[i] is not None else ""
            c3 = str(diag3[i]) if diag3[i] is not None else ""
            for c in (c1, c2, c3):
                if c.startswith("428"):
                    has_chf[i] = 1.0
                if any(c.startswith(p) for p in ("410", "411", "412", "413", "414")):
                    has_cad[i] = 1.0
                if any(c.startswith(p) for p in ("585", "586", "584")):
                    has_ckd[i] = 1.0
                if any(c.startswith(p) for p in ("491", "492", "493", "494", "496")):
                    has_copd[i] = 1.0
                if any(c.startswith(p) for p in ("401", "402", "403", "404", "405")):
                    has_htn[i] = 1.0
                if c.startswith("250.") and not (c.endswith("00") or c.endswith("02")):
                    has_diab_comp[i] = 1.0
                if any(c.startswith(p) for p in ("038", "480", "481", "482", "485", "486", "599")):
                    has_infection[i] = 1.0

        # Expanded Comorbidity & Charlson Phenotyping across all 3 diagnoses
        has_pvd = np.zeros(n_rows, dtype=float)
        has_stroke = np.zeros(n_rows, dtype=float)
        has_cancer = np.zeros(n_rows, dtype=float)
        has_anemia = np.zeros(n_rows, dtype=float)
        has_electrolyte = np.zeros(n_rows, dtype=float)
        primary_is_chf = np.zeros(n_rows, dtype=float)
        primary_is_diab = np.zeros(n_rows, dtype=float)
        primary_is_cad = np.zeros(n_rows, dtype=float)
        primary_is_infection = np.zeros(n_rows, dtype=float)

        for i in range(n_rows):
            d1 = str(diag1[i]).strip() if diag1[i] is not None else ""
            d2 = str(diag2[i]).strip() if diag2[i] is not None else ""
            d3 = str(diag3[i]).strip() if diag3[i] is not None else ""
            if d1 in ("None", "?"): d1 = ""
            if d2 in ("None", "?"): d2 = ""
            if d3 in ("None", "?"): d3 = ""

            if d1.startswith("428"):
                primary_is_chf[i] = 1.0
            if d1.startswith("250"):
                primary_is_diab[i] = 1.0
            if any(d1.startswith(p) for p in ("410", "411", "412", "413", "414")):
                primary_is_cad[i] = 1.0
            if any(d1.startswith(p) for p in ("038", "480", "481", "482", "485", "486", "599")):
                primary_is_infection[i] = 1.0

            for c in (d1, d2, d3):
                if not c:
                    continue
                if any(c.startswith(p) for p in ("440", "441", "442", "443", "444", "447", "785.4")):
                    has_pvd[i] = 1.0
                if any(c.startswith(p) for p in ("430", "431", "432", "433", "434", "435", "436", "437", "438")):
                    has_stroke[i] = 1.0
                if any(c.startswith(p) for p in ("14", "15", "16", "17", "18", "19", "200", "201", "202", "203", "204", "205", "206", "207", "208")):
                    has_cancer[i] = 1.0
                if any(c.startswith(p) for p in ("280", "281", "282", "283", "284", "285")):
                    has_anemia[i] = 1.0
                if c.startswith("276"):
                    has_electrolyte[i] = 1.0

        # Weighted Charlson Comorbidity Index (CCI)
        charlson_score = (
            1.0 * has_cad +
            1.0 * has_chf +
            1.0 * has_pvd +
            1.0 * has_stroke +
            1.0 * has_copd +
            2.0 * has_ckd +
            2.0 * has_diab_comp +
            2.0 * has_cancer
        )
        comorbidity_score = (
            has_chf + has_cad + has_ckd + has_copd + has_htn +
            has_diab_comp + has_infection + has_pvd + has_stroke +
            has_cancer + has_anemia + has_electrolyte
        )

        # Antidiabetic Polypharmacy Instability Vector
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
        insulin_changed = np.zeros(n_rows, dtype=float)
        insulin_up = np.zeros(n_rows, dtype=float)
        insulin_down = np.zeros(n_rows, dtype=float)
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

        raw_ch = data_dict.get("change")
        change_flag = np.zeros(n_rows, dtype=float)
        if raw_ch is not None:
            for i, v in enumerate(raw_ch):
                if v == "Ch":
                    change_flag[i] = 1.0

        raw_dm = data_dict.get("diabetesMed")
        dm_med_flag = np.zeros(n_rows, dtype=float)
        if raw_dm is not None:
            for i, v in enumerate(raw_dm):
                if v == "Yes":
                    dm_med_flag[i] = 1.0

        # Glycemic tracking flags
        raw_a1c = data_dict.get("A1Cresult")
        a1c_tested = np.zeros(n_rows, dtype=float)
        a1c_high = np.zeros(n_rows, dtype=float)
        if raw_a1c is not None:
            for i, v in enumerate(raw_a1c):
                if v is not None and v not in ("None", "", "?"):
                    a1c_tested[i] = 1.0
                    if v in (">7", ">8"):
                        a1c_high[i] = 1.0

        raw_glu = data_dict.get("max_glu_serum")
        glu_tested = np.zeros(n_rows, dtype=float)
        glu_high = np.zeros(n_rows, dtype=float)
        if raw_glu is not None:
            for i, v in enumerate(raw_glu):
                if v is not None and v not in ("None", "", "?"):
                    glu_tested[i] = 1.0
                    if v in (">200", ">300"):
                        glu_high[i] = 1.0

        # Discharge, Admission & Missingness Indicators
        raw_disp = data_dict.get("discharge_disposition_id")
        disp_home = np.zeros(n_rows, dtype=float)
        disp_snf = np.zeros(n_rows, dtype=float)
        disp_home_health = np.zeros(n_rows, dtype=float)
        disp_ama = np.zeros(n_rows, dtype=float)
        disp_short_hosp = np.zeros(n_rows, dtype=float)
        if raw_disp is not None:
            for i in range(n_rows):
                d_val = str(raw_disp[i]) if raw_disp[i] is not None else ""
                if d_val in ("1", "8"):
                    disp_home[i] = 1.0
                elif d_val in ("3", "4", "5", "22", "23", "24"):
                    disp_snf[i] = 1.0
                elif d_val == "6":
                    disp_home_health[i] = 1.0
                elif d_val == "7":
                    disp_ama[i] = 1.0
                elif d_val == "2":
                    disp_short_hosp[i] = 1.0

        raw_adm = data_dict.get("admission_type_id")
        adm_emergency = np.zeros(n_rows, dtype=float)
        adm_elective = np.zeros(n_rows, dtype=float)
        adm_urgent = np.zeros(n_rows, dtype=float)
        if raw_adm is not None:
            for i in range(n_rows):
                a_val = str(raw_adm[i]) if raw_adm[i] is not None else ""
                if a_val in ("1", "7"):
                    adm_emergency[i] = 1.0
                elif a_val == "2":
                    adm_urgent[i] = 1.0
                elif a_val == "3":
                    adm_elective[i] = 1.0

        raw_src = data_dict.get("admission_source_id")
        adm_src_er = np.zeros(n_rows, dtype=float)
        adm_src_transfer = np.zeros(n_rows, dtype=float)
        adm_src_clinic = np.zeros(n_rows, dtype=float)
        if raw_src is not None:
            for i in range(n_rows):
                s_val = str(raw_src[i]) if raw_src[i] is not None else ""
                if s_val == "7":
                    adm_src_er[i] = 1.0
                elif s_val in ("4", "5", "6"):
                    adm_src_transfer[i] = 1.0
                elif s_val in ("1", "2", "3"):
                    adm_src_clinic[i] = 1.0

        raw_spec = data_dict.get("medical_specialty")
        spec_missing = np.zeros(n_rows, dtype=float)
        if raw_spec is not None:
            for i in range(n_rows):
                sv = raw_spec[i]
                if sv is None or sv in ("?", "None", ""):
                    spec_missing[i] = 1.0
        else:
            spec_missing.fill(1.0)

        raw_payer = data_dict.get("payer_code")
        payer_missing = np.zeros(n_rows, dtype=float)
        if raw_payer is not None:
            for i in range(n_rows):
                pv = raw_payer[i]
                if pv is None or pv in ("?", "None", ""):
                    payer_missing[i] = 1.0
        else:
            payer_missing.fill(1.0)

        # ICD-9 Disease Prefixes (3-digit clinical family)
        def get_diag_prefix(code):
            if code is None or code == "" or code == "?":
                return "missing"
            s = str(code).strip()
            if not s or s == "None":
                return "missing"
            return s.split(".")[0]

        diag_1_pref = np.array([get_diag_prefix(c) for c in diag1], dtype=object)
        diag_2_pref = np.array([get_diag_prefix(c) for c in diag2], dtype=object)
        diag_3_pref = np.array([get_diag_prefix(c) for c in diag3], dtype=object)

        # ICD-9 Chapters
        diag_1_grp = np.array([icd9_fn(c) for c in diag1], dtype=object)
        diag_2_grp = np.array([icd9_fn(c) for c in diag2], dtype=object)
        diag_3_grp = np.array([icd9_fn(c) for c in diag3], dtype=object)

        numeric_feats = {
            "feat_tot_prior": tot_prior,
            "feat_inpat_ratio": inpat_ratio,
            "feat_emerg_ratio": emerg_ratio,
            "feat_has_inpatient": has_inpatient,
            "feat_frequent_inpatient": frequent_inpatient,
            "feat_acute_care_score": acute_care_score,
            "feat_inpat_velocity": inpat_velocity,
            "feat_lab_intensity": lab_intensity,
            "feat_med_intensity": med_intensity,
            "feat_proc_intensity": proc_intensity,
            "feat_diag_intensity": diag_intensity,
            "feat_action_intensity": action_intensity,
            "feat_med_per_diag": med_per_diag,
            "feat_age_continuous": age_continuous,
            "feat_has_chf": has_chf,
            "feat_has_cad": has_cad,
            "feat_has_ckd": has_ckd,
            "feat_has_copd": has_copd,
            "feat_has_htn": has_htn,
            "feat_has_diab_comp": has_diab_comp,
            "feat_has_infection": has_infection,
            "feat_comorbidity_score": comorbidity_score,
            "feat_num_active_meds": num_active_meds,
            "feat_num_meds_up": num_meds_up,
            "feat_num_meds_down": num_meds_down,
            "feat_num_meds_changed": num_meds_changed,
            "feat_polypharmacy_flag": polypharmacy_flag,
            "feat_insulin_on": insulin_on,
            "feat_insulin_changed": insulin_changed,
            "feat_insulin_up": insulin_up,
            "feat_insulin_down": insulin_down,
            "feat_metformin_on": metformin_on,
            "feat_change_flag": change_flag,
            "feat_dm_med_flag": dm_med_flag,
            "feat_a1c_tested": a1c_tested,
            "feat_a1c_high": a1c_high,
            "feat_glu_tested": glu_tested,
            "feat_glu_high": glu_high,
            "feat_has_pvd": has_pvd,
            "feat_has_stroke": has_stroke,
            "feat_has_cancer": has_cancer,
            "feat_has_anemia": has_anemia,
            "feat_has_electrolyte": has_electrolyte,
            "feat_primary_is_chf": primary_is_chf,
            "feat_primary_is_diab": primary_is_diab,
            "feat_primary_is_cad": primary_is_cad,
            "feat_primary_is_infection": primary_is_infection,
            "feat_charlson_score": charlson_score,
            "feat_disp_home": disp_home,
            "feat_disp_snf": disp_snf,
            "feat_disp_home_health": disp_home_health,
            "feat_disp_ama": disp_ama,
            "feat_disp_short_hosp": disp_short_hosp,
            "feat_adm_emergency": adm_emergency,
            "feat_adm_elective": adm_elective,
            "feat_adm_urgent": adm_urgent,
            "feat_adm_src_er": adm_src_er,
            "feat_adm_src_transfer": adm_src_transfer,
            "feat_adm_src_clinic": adm_src_clinic,
            "feat_spec_missing": spec_missing,
            "feat_payer_missing": payer_missing,
            "feat_log_inpat": np.log1p(n_inpat),
            "feat_log_emerg": np.log1p(n_emerg),
            "feat_log_outp": np.log1p(n_outp),
            "feat_log_tot_prior": np.log1p(tot_prior),
            "feat_chf_x_inpatient": has_chf * n_inpat,
            "feat_ckd_x_inpatient": has_ckd * n_inpat,
            "feat_snf_x_inpatient": disp_snf * n_inpat,
            "feat_ama_x_inpatient": disp_ama * (n_inpat + 1.0),
            "feat_age_x_inpatient": age_continuous * n_inpat,
            "feat_age_x_charlson": age_continuous * charlson_score,
        }

        categorical_feats = {
            "diag_1_prefix": diag_1_pref,
            "diag_2_prefix": diag_2_pref,
            "diag_3_prefix": diag_3_pref,
            "diag_1_group": diag_1_grp,
            "diag_2_group": diag_2_grp,
            "diag_3_group": diag_3_grp,
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

    # Care Continuum & Clinical Interaction Tuples
    disp_tr = train.get("discharge_disposition_id")
    disp_ev = eval_cols.get("discharge_disposition_id")
    adm_tr = train.get("admission_source_id")
    adm_ev = eval_cols.get("admission_source_id")
    adm_type_tr = train.get("admission_type_id")
    adm_type_ev = eval_cols.get("admission_type_id")
    d1_grp_tr = train_cat["diag_1_group"]
    d1_grp_ev = eval_cat["diag_1_group"]

    train_ext["discharge_x_diag1"] = np.array([f"{d}_{g}" for d, g in zip(disp_tr, d1_grp_tr)], dtype=object)
    eval_ext["discharge_x_diag1"] = np.array([f"{d}_{g}" for d, g in zip(disp_ev, d1_grp_ev)], dtype=object)

    train_ext["discharge_x_admsrc"] = np.array([f"{d}_{s}" for d, s in zip(disp_tr, adm_tr)], dtype=object)
    eval_ext["discharge_x_admsrc"] = np.array([f"{d}_{s}" for d, s in zip(disp_ev, adm_ev)], dtype=object)

    train_ext["admtype_x_admsrc"] = np.array([f"{t}_{s}" for t, s in zip(adm_type_tr, adm_tr)], dtype=object)
    eval_ext["admtype_x_admsrc"] = np.array([f"{t}_{s}" for t, s in zip(adm_type_ev, adm_ev)], dtype=object)

    train_ext["diag1_x_admsrc"] = np.array([f"{g}_{s}" for g, s in zip(d1_grp_tr, adm_tr)], dtype=object)
    eval_ext["diag1_x_admsrc"] = np.array([f"{g}_{s}" for g, s in zip(d1_grp_ev, adm_ev)], dtype=object)

    # Frequency Prevalence Encoding for key categoricals
    def compute_freq_enc(tr_arr: np.ndarray, ev_arr: np.ndarray):
        n = float(len(tr_arr))
        counts = {}
        for v in tr_arr:
            s_key = str(v) if v is not None else "missing"
            counts[s_key] = counts.get(s_key, 0) + 1
        tr_f = np.array([counts.get(str(v) if v is not None else "missing", 0) / n for v in tr_arr], dtype=float)
        ev_f = np.array([counts.get(str(v) if v is not None else "missing", 0) / n for v in ev_arr], dtype=float)
        return tr_f, ev_f

    freq_cols = ["diag_1_prefix", "discharge_disposition_id", "admission_source_id", "medical_specialty"]
    for fc in freq_cols:
        f_tr, f_ev = compute_freq_enc(train_ext[fc], eval_ext[fc])
        train_ext[f"freq_{fc}"] = f_tr
        eval_ext[f"freq_{fc}"] = f_ev

    # 2. Bayesian Out-of-Fold Target Encoding
    def compute_oof_target_enc(tr_arr: np.ndarray, y: np.ndarray, ev_arr: np.ndarray, m: float = 25.0):
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

    te_configs = [
        ("discharge_disposition_id", 20.0),
        ("admission_source_id", 25.0),
        ("admission_type_id", 25.0),
        ("payer_code", 25.0),
        ("medical_specialty", 35.0),
        ("diag_1_group", 25.0),
        ("diag_1_prefix", 35.0),
        ("discharge_x_diag1", 25.0),
        ("discharge_x_admsrc", 25.0),
    ]
    for te_col, m_shrink in te_configs:
        te_tr, te_ev = compute_oof_target_enc(train_ext[te_col], y_train, eval_ext[te_col], m=m_shrink)
        out_name = f"te_{te_col}"
        train_ext[out_name] = te_tr
        eval_ext[out_name] = te_ev

    extra_numeric = (
        list(train_num.keys()) +
        [f"te_{c}" for c, _ in te_configs] +
        [f"freq_{c}" for c in freq_cols]
    )
    extra_categorical = (
        list(train_cat.keys()) +
        ["discharge_x_diag1", "discharge_x_admsrc", "admtype_x_admsrc", "diag1_x_admsrc"]
    )

    # 3. Fit TabularEncoder
    encoder = TabularEncoder(min_count=20)
    encoder.fit(task, train_ext, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(train_ext)
    x_eval = encoder.transform(eval_ext)

    # 4. Quad-Model Multi-Inductive Ensemble
    # Model 1: Balanced LightGBM
    m1 = lgb.LGBMClassifier(
        n_estimators=360,
        learning_rate=0.03,
        num_leaves=32,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=2.5,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p1 = np.nan_to_num(m1.predict_proba(x_eval)[:, 1], nan=0.114)

    # Model 2: Shallow Conservative LightGBM
    m2 = lgb.LGBMClassifier(
        n_estimators=420,
        learning_rate=0.025,
        num_leaves=16,
        min_child_samples=75,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=0.8,
        reg_lambda=6.0,
        random_state=202,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p2 = np.nan_to_num(m2.predict_proba(x_eval)[:, 1], nan=0.114)

    # Model 3: High-Capacity Subsampled LightGBM
    m3 = lgb.LGBMClassifier(
        n_estimators=340,
        learning_rate=0.028,
        num_leaves=48,
        min_child_samples=50,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=0.5,
        reg_lambda=4.0,
        random_state=303,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p3 = np.nan_to_num(m3.predict_proba(x_eval)[:, 1], nan=0.114)

    # Model 4: XGBoost with histogram binning (or diverse LightGBM fallback)
    xgb_mod = globals().get("xgb") or globals().get("xgboost")
    if xgb_mod is None:
        try:
            import xgboost as xgb_mod
        except Exception:
            xgb_mod = None

    p4 = None
    if xgb_mod is not None:
        try:
            m4 = xgb_mod.XGBClassifier(
                n_estimators=280,
                learning_rate=0.035,
                max_depth=5,
                min_child_weight=25,
                subsample=0.8,
                colsample_bytree=0.65,
                reg_alpha=0.4,
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
            learning_rate=0.032,
            num_leaves=24,
            min_child_samples=60,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.1,
            reg_lambda=1.5,
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
        0.28 * rank_norm(p2) +
        0.22 * rank_norm(p3) +
        0.20 * rank_norm(p4)
    )
    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
