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
    import lightgbm as lgb
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    train_copy = dict(train)
    eval_copy = dict(eval_cols)

    extra_numeric = []
    extra_categorical = []

    def add_num(name: str, trn_arr: np.ndarray, val_arr: np.ndarray) -> None:
        train_copy[name] = np.asarray(trn_arr, dtype=np.float64)
        eval_copy[name] = np.asarray(val_arr, dtype=np.float64)
        extra_numeric.append(name)

    def add_cat(name: str, trn_arr: np.ndarray, val_arr: np.ndarray) -> None:
        train_copy[name] = np.asarray(trn_arr, dtype=object)
        eval_copy[name] = np.asarray(val_arr, dtype=object)
        extra_categorical.append(name)

    def safe_num(arr: np.ndarray, default: float = 0.0) -> np.ndarray:
        res = np.array(arr, dtype=np.float64)
        return np.nan_to_num(res, nan=default)

    # 1. Demographics & Age Features
    age_map = {
        "[0-10)": 5.0,
        "[10-20)": 15.0,
        "[20-30)": 25.0,
        "[30-40)": 35.0,
        "[40-50)": 45.0,
        "[50-60)": 55.0,
        "[60-70)": 65.0,
        "[70-80)": 75.0,
        "[80-90)": 85.0,
        "[90-100)": 95.0,
    }
    age_trn = np.array([age_map.get(x, 65.0) for x in train["age"]], dtype=np.float64)
    age_eval = np.array([age_map.get(x, 65.0) for x in eval_cols["age"]], dtype=np.float64)
    add_num("age_num", age_trn, age_eval)
    add_num("age_sq", (age_trn / 50.0) ** 2, (age_eval / 50.0) ** 2)
    add_num("is_young", (age_trn < 40.0).astype(float), (age_eval < 40.0).astype(float))
    add_num("is_elderly", (age_trn >= 70.0).astype(float), (age_eval >= 70.0).astype(float))
    add_num("is_very_elderly", (age_trn >= 80.0).astype(float), (age_eval >= 80.0).astype(float))
    add_num("is_nonagenarian", (age_trn >= 90.0).astype(float), (age_eval >= 90.0).astype(float))

    add_num("is_female", (train["gender"] == "Female").astype(float), (eval_cols["gender"] == "Female").astype(float))
    add_num("is_caucasian", (train["race"] == "Caucasian").astype(float), (eval_cols["race"] == "Caucasian").astype(float))
    add_num("is_african_american", (train["race"] == "AfricanAmerican").astype(float), (eval_cols["race"] == "AfricanAmerican").astype(float))
    add_num("race_missing", np.isin(train["race"], [None, "?", "None"]).astype(float), np.isin(eval_cols["race"], [None, "?", "None"]).astype(float))

    # 2. Prior Utilization Signals
    inp_trn = safe_num(train["number_inpatient"])
    inp_eval = safe_num(eval_cols["number_inpatient"])
    emg_trn = safe_num(train["number_emergency"])
    emg_eval = safe_num(eval_cols["number_emergency"])
    out_trn = safe_num(train["number_outpatient"])
    out_eval = safe_num(eval_cols["number_outpatient"])

    tot_trn = inp_trn + emg_trn + out_trn
    tot_eval = inp_eval + emg_eval + out_eval
    acute_trn = inp_trn + emg_trn
    acute_eval = inp_eval + emg_eval

    add_num("total_prior_visits", tot_trn, tot_eval)
    add_num("acute_prior_visits", acute_trn, acute_eval)
    add_num("has_prior_inpatient", (inp_trn > 0).astype(float), (inp_eval > 0).astype(float))
    add_num("has_multi_inpatient", (inp_trn > 1).astype(float), (inp_eval > 1).astype(float))
    add_num("has_frequent_inpatient", (inp_trn >= 3).astype(float), (inp_eval >= 3).astype(float))
    add_num("has_prior_emergency", (emg_trn > 0).astype(float), (emg_eval > 0).astype(float))
    add_num("has_prior_outpatient", (out_trn > 0).astype(float), (out_eval > 0).astype(float))
    add_num("inpatient_ratio", inp_trn / (tot_trn + 1.0), inp_eval / (tot_eval + 1.0))
    add_num("emergency_ratio", emg_trn / (tot_trn + 1.0), emg_eval / (tot_eval + 1.0))
    add_num("acute_ratio", acute_trn / (tot_trn + 1.0), acute_eval / (tot_eval + 1.0))
    add_num("log_inpatient", np.log1p(inp_trn), np.log1p(inp_eval))
    add_num("log_emergency", np.log1p(emg_trn), np.log1p(emg_eval))
    add_num("log_total_visits", np.log1p(tot_trn), np.log1p(tot_eval))

    # 3. Hospital Stay & Care Intensity Metrics
    los_trn = safe_num(train["time_in_hospital"], default=1.0)
    los_eval = safe_num(eval_cols["time_in_hospital"], default=1.0)
    los_trn_clip = np.maximum(los_trn, 1.0)
    los_eval_clip = np.maximum(los_eval, 1.0)

    lab_trn = safe_num(train["num_lab_procedures"])
    lab_eval = safe_num(eval_cols["num_lab_procedures"])
    prc_trn = safe_num(train["num_procedures"])
    prc_eval = safe_num(eval_cols["num_procedures"])
    med_trn = safe_num(train["num_medications"])
    med_eval = safe_num(eval_cols["num_medications"])
    diag_trn = safe_num(train["number_diagnoses"], default=1.0)
    diag_eval = safe_num(eval_cols["number_diagnoses"], default=1.0)

    tot_care_trn = lab_trn + prc_trn + med_trn
    tot_care_eval = lab_eval + prc_eval + med_eval
    add_num("total_care_load", tot_care_trn, tot_care_eval)
    add_num("med_per_day", med_trn / los_trn_clip, med_eval / los_eval_clip)
    add_num("labs_per_day", lab_trn / los_trn_clip, lab_eval / los_eval_clip)
    add_num("procs_per_day", prc_trn / los_trn_clip, prc_eval / los_eval_clip)
    add_num("care_per_day", tot_care_trn / los_trn_clip, tot_care_eval / los_eval_clip)
    add_num("diag_per_day", diag_trn / los_trn_clip, diag_eval / los_eval_clip)
    add_num("med_per_diag", med_trn / np.maximum(diag_trn, 1.0), med_eval / np.maximum(diag_eval, 1.0))
    add_num("polypharmacy", (med_trn >= 16).astype(float), (med_eval >= 16).astype(float))
    add_num("high_diagnoses", (diag_trn >= 9).astype(float), (diag_eval >= 9).astype(float))

    # Cross-feature interactions
    add_num("inpatient_x_los", inp_trn * los_trn, inp_eval * los_eval)
    add_num("inpatient_x_diag", inp_trn * diag_trn, inp_eval * diag_eval)
    add_num("inpatient_x_age", inp_trn * age_trn, inp_eval * age_eval)
    add_num("med_x_diag", med_trn * diag_trn, med_eval * diag_eval)
    add_num("acute_x_los", acute_trn * los_trn, acute_eval * los_eval)

    # 4. ICD-9 Diagnosis Grouping & High-Risk Clinical Comorbidities
    _icd9_fn = globals().get("icd9_group", None)
    if _icd9_fn is None:
        try:
            from clinical_toolkit import icd9_group as _icd9_fn
        except Exception:
            pass

    def safe_icd9(code):
        if _icd9_fn is not None:
            try:
                return _icd9_fn(code)
            except Exception:
                pass
        if code is None or code == "?" or str(code).strip() == "" or code != code:
            return "missing"
        s = str(code).strip()
        if s.startswith(("V", "v", "E", "e")):
            return "other"
        try:
            val = float(s)
        except ValueError:
            return "other"
        if (390 <= val <= 459) or val == 785:
            return "circulatory"
        if (460 <= val <= 519) or val == 786:
            return "respiratory"
        if (520 <= val <= 579) or val == 787:
            return "digestive"
        if 250 <= val < 251:
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

    def get_3digit(code):
        if code is None or code == "?" or str(code).strip() == "" or code != code:
            return "missing"
        s = str(code).strip().split(".")[0]
        return s if s else "missing"

    diag_3d_trn = {}
    diag_3d_eval = {}
    for d in ["diag_1", "diag_2", "diag_3"]:
        add_cat(f"{d}_group", [safe_icd9(c) for c in train[d]], [safe_icd9(c) for c in eval_cols[d]])
        d3_t = [get_3digit(c) for c in train[d]]
        d3_e = [get_3digit(c) for c in eval_cols[d]]
        diag_3d_trn[d] = d3_t
        diag_3d_eval[d] = d3_e
        add_cat(f"{d}_3digit", d3_t, d3_e)

    def extract_comorbidities(cols_dict):
        n = len(cols_dict["time_in_hospital"])
        has_chf = np.zeros(n, dtype=np.float64)
        has_ckd = np.zeros(n, dtype=np.float64)
        has_copd = np.zeros(n, dtype=np.float64)
        has_htn = np.zeros(n, dtype=np.float64)
        has_cad = np.zeros(n, dtype=np.float64)
        has_stroke = np.zeros(n, dtype=np.float64)
        has_fluid = np.zeros(n, dtype=np.float64)
        has_anemia = np.zeros(n, dtype=np.float64)
        has_pvd = np.zeros(n, dtype=np.float64)
        has_sepsis = np.zeros(n, dtype=np.float64)
        has_diab = np.zeros(n, dtype=np.float64)
        has_comp_diab = np.zeros(n, dtype=np.float64)
        has_acute_diab = np.zeros(n, dtype=np.float64)
        circ_cnt = np.zeros(n, dtype=np.float64)
        resp_cnt = np.zeros(n, dtype=np.float64)
        diab_cnt = np.zeros(n, dtype=np.float64)

        copd_pref = ("490", "491", "492", "493", "494", "496", "480", "481", "482", "486")
        htn_pref = ("401", "402", "403", "404", "405")
        ckd_pref = ("585", "586", "582", "583", "584", "V42.0", "V45.1", "V56")
        cad_pref = ("410", "411", "412", "413", "414")
        stroke_pref = ("430", "431", "432", "433", "434", "435", "436", "437", "438")
        pvd_pref = ("440", "441", "442", "443", "444", "447")
        anemia_pref = ("280", "281", "282", "283", "284", "285")

        for d in ["diag_1", "diag_2", "diag_3"]:
            for i, c in enumerate(cols_dict[d]):
                if c is None or c == "?" or c != c:
                    continue
                s = str(c).strip()
                grp = safe_icd9(s)
                if grp == "circulatory":
                    circ_cnt[i] += 1.0
                elif grp == "respiratory":
                    resp_cnt[i] += 1.0
                elif grp == "diabetes":
                    diab_cnt[i] += 1.0

                if s.startswith("428"):
                    has_chf[i] = 1.0
                if any(s.startswith(p) for p in ckd_pref):
                    has_ckd[i] = 1.0
                if any(s.startswith(p) for p in copd_pref):
                    has_copd[i] = 1.0
                if any(s.startswith(p) for p in htn_pref):
                    has_htn[i] = 1.0
                if any(s.startswith(p) for p in cad_pref):
                    has_cad[i] = 1.0
                if any(s.startswith(p) for p in stroke_pref):
                    has_stroke[i] = 1.0
                if s.startswith("276"):
                    has_fluid[i] = 1.0
                if any(s.startswith(p) for p in anemia_pref):
                    has_anemia[i] = 1.0
                if any(s.startswith(p) for p in pvd_pref):
                    has_pvd[i] = 1.0
                if s.startswith(("038", "995.91", "995.92")):
                    has_sepsis[i] = 1.0

                if s.startswith("250"):
                    has_diab[i] = 1.0
                    if s.startswith(("250.1", "250.2", "250.3")):
                        has_acute_diab[i] = 1.0
                    if "." in s and not s.split(".")[1].startswith("0"):
                        has_comp_diab[i] = 1.0

        chronic_cnt = has_chf + has_ckd + has_copd + has_htn + has_cad + has_stroke + has_fluid + has_anemia + has_diab
        return {
            "has_chf": has_chf,
            "has_ckd": has_ckd,
            "has_copd": has_copd,
            "has_htn": has_htn,
            "has_cad": has_cad,
            "has_stroke": has_stroke,
            "has_fluid_electrolyte": has_fluid,
            "has_anemia": has_anemia,
            "has_pvd": has_pvd,
            "has_sepsis": has_sepsis,
            "has_diabetes_diag": has_diab,
            "has_comp_diabetes": has_comp_diab,
            "has_acute_diabetes": has_acute_diab,
            "chronic_count": chronic_cnt,
            "circulatory_count": circ_cnt,
            "respiratory_count": resp_cnt,
            "diabetes_group_count": diab_cnt,
            "cardiorenal": has_chf * has_ckd,
        }

    comorb_trn = extract_comorbidities(train)
    comorb_eval = extract_comorbidities(eval_cols)
    for k in comorb_trn:
        add_num(k, comorb_trn[k], comorb_eval[k])

    add_num("inpatient_x_chronic", inp_trn * comorb_trn["chronic_count"], inp_eval * comorb_eval["chronic_count"])
    add_num("inpatient_x_chf", inp_trn * comorb_trn["has_chf"], inp_eval * comorb_eval["has_chf"])
    add_num("inpatient_x_ckd", inp_trn * comorb_trn["has_ckd"], inp_eval * comorb_eval["has_ckd"])

    # 5. Out-of-Fold Bayesian Target Encoding for Diagnoses & High-Cardinality Categoricals
    def compute_oof_target_encoding(train_series, eval_series, y_tr, m=25.0, n_splits=5, seed=2024):
        n_tr = len(train_series)
        n_ev = len(eval_series)
        global_mean = float(np.mean(y_tr))
        oof_train = np.full(n_tr, global_mean, dtype=np.float64)
        eval_enc = np.full(n_ev, global_mean, dtype=np.float64)

        def clean_val(v):
            if v is None:
                return "__MISSING__"
            if isinstance(v, float) and (v != v or np.isnan(v)):
                return "__MISSING__"
            s = str(v).strip()
            return "__MISSING__" if s in ("", "?", "None", "nan") else s

        tr_s = [clean_val(x) for x in train_series]
        ev_s = [clean_val(x) for x in eval_series]

        skf_te = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for t_idx, v_idx in skf_te.split(train_series, y_tr):
            counts = {}
            sums = {}
            for idx in t_idx:
                val = tr_s[idx]
                counts[val] = counts.get(val, 0) + 1
                sums[val] = sums.get(val, 0.0) + y_tr[idx]

            for idx in v_idx:
                val = tr_s[idx]
                c = counts.get(val, 0)
                if c > 0:
                    oof_train[idx] = (sums[val] + m * global_mean) / (c + m)
                else:
                    oof_train[idx] = global_mean

        all_counts = {}
        all_sums = {}
        for idx in range(n_tr):
            val = tr_s[idx]
            all_counts[val] = all_counts.get(val, 0) + 1
            all_sums[val] = all_sums.get(val, 0.0) + y_tr[idx]

        for idx in range(n_ev):
            val = ev_s[idx]
            c = all_counts.get(val, 0)
            if c > 0:
                eval_enc[idx] = (all_sums[val] + m * global_mean) / (c + m)
            else:
                eval_enc[idx] = global_mean

        return oof_train, eval_enc

    # Diagnosis target encodings
    te_d1_trn, te_d1_eval = compute_oof_target_encoding(train["diag_1"], eval_cols["diag_1"], y_train, m=25.0)
    te_d2_trn, te_d2_eval = compute_oof_target_encoding(train["diag_2"], eval_cols["diag_2"], y_train, m=25.0)
    te_d3_trn, te_d3_eval = compute_oof_target_encoding(train["diag_3"], eval_cols["diag_3"], y_train, m=25.0)
    add_num("te_diag_1", te_d1_trn, te_d1_eval)
    add_num("te_diag_2", te_d2_trn, te_d2_eval)
    add_num("te_diag_3", te_d3_trn, te_d3_eval)
    add_num("te_diag_max", np.maximum.reduce([te_d1_trn, te_d2_trn, te_d3_trn]), np.maximum.reduce([te_d1_eval, te_d2_eval, te_d3_eval]))
    add_num("te_diag_mean", (te_d1_trn + te_d2_trn + te_d3_trn) / 3.0, (te_d1_eval + te_d2_eval + te_d3_eval) / 3.0)
    add_num("te_diag_sum", te_d1_trn + te_d2_trn + te_d3_trn, te_d1_eval + te_d2_eval + te_d3_eval)

    te_3d1_trn, te_3d1_eval = compute_oof_target_encoding(diag_3d_trn["diag_1"], diag_3d_eval["diag_1"], y_train, m=30.0)
    te_3d2_trn, te_3d2_eval = compute_oof_target_encoding(diag_3d_trn["diag_2"], diag_3d_eval["diag_2"], y_train, m=30.0)
    te_3d3_trn, te_3d3_eval = compute_oof_target_encoding(diag_3d_trn["diag_3"], diag_3d_eval["diag_3"], y_train, m=30.0)
    add_num("te_diag_3d_1", te_3d1_trn, te_3d1_eval)
    add_num("te_diag_3d_2", te_3d2_trn, te_3d2_eval)
    add_num("te_diag_3d_3", te_3d3_trn, te_3d3_eval)
    add_num("te_diag_3d_max", np.maximum.reduce([te_3d1_trn, te_3d2_trn, te_3d3_trn]), np.maximum.reduce([te_3d1_eval, te_3d2_eval, te_3d3_eval]))
    add_num("te_diag_3d_mean", (te_3d1_trn + te_3d2_trn + te_3d3_trn) / 3.0, (te_3d1_eval + te_3d2_eval + te_3d3_eval) / 3.0)

    # Administrative and specialty target encodings
    te_spec_trn, te_spec_eval = compute_oof_target_encoding(train["medical_specialty"], eval_cols["medical_specialty"], y_train, m=25.0)
    te_disc_trn, te_disc_eval = compute_oof_target_encoding(train["discharge_disposition_id"], eval_cols["discharge_disposition_id"], y_train, m=20.0)
    te_src_trn, te_src_eval = compute_oof_target_encoding(train["admission_source_id"], eval_cols["admission_source_id"], y_train, m=20.0)
    te_adm_trn, te_adm_eval = compute_oof_target_encoding(train["admission_type_id"], eval_cols["admission_type_id"], y_train, m=20.0)
    add_num("te_medical_specialty", te_spec_trn, te_spec_eval)
    add_num("te_discharge_disposition", te_disc_trn, te_disc_eval)
    add_num("te_admission_source", te_src_trn, te_src_eval)
    add_num("te_admission_type", te_adm_trn, te_adm_eval)

    # 6. Medication Features & Glycemic Control
    diabetes_meds = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    ]

    def extract_med_stats(cols_dict):
        n = len(cols_dict["time_in_hospital"])
        active = np.zeros(n, dtype=np.float64)
        up = np.zeros(n, dtype=np.float64)
        down = np.zeros(n, dtype=np.float64)
        for med in diabetes_meds:
            if med in cols_dict:
                arr = cols_dict[med]
                u = (arr == "Up")
                d = (arr == "Down")
                s = (arr == "Steady")
                up += u
                down += d
                active += (u | d | s)
        return active, up, down, up + down

    act_t, up_t, dn_t, chg_t = extract_med_stats(train)
    act_e, up_e, dn_e, chg_e = extract_med_stats(eval_cols)
    add_num("num_active_meds", act_t, act_e)
    add_num("num_meds_up", up_t, up_e)
    add_num("num_meds_down", dn_t, dn_e)
    add_num("num_meds_changed", chg_t, chg_e)
    add_num("has_med_change", (chg_t > 0).astype(float), (chg_e > 0).astype(float))

    # Insulin specific details
    ins_t = train["insulin"]
    ins_e = eval_cols["insulin"]
    ins_presc_t = (ins_t != "No").astype(float)
    ins_presc_e = (ins_e != "No").astype(float)
    add_num("insulin_prescribed", ins_presc_t, ins_presc_e)
    add_num("insulin_changed", np.isin(ins_t, ["Up", "Down"]).astype(float), np.isin(ins_e, ["Up", "Down"]).astype(float))
    add_num("insulin_up", (ins_t == "Up").astype(float), (ins_e == "Up").astype(float))
    add_num("insulin_down", (ins_t == "Down").astype(float), (ins_e == "Down").astype(float))
    add_num("insulin_steady", (ins_t == "Steady").astype(float), (ins_e == "Steady").astype(float))

    # Oral diabetes medications
    oral_t = np.maximum(act_t - ins_presc_t, 0.0)
    oral_e = np.maximum(act_e - ins_presc_e, 0.0)
    add_num("num_oral_meds", oral_t, oral_e)
    add_num("insulin_and_oral", ((ins_presc_t > 0) & (oral_t > 0)).astype(float), ((ins_presc_e > 0) & (oral_e > 0)).astype(float))
    add_num("insulin_only", ((ins_presc_t > 0) & (oral_t == 0)).astype(float), ((ins_presc_e > 0) & (oral_e == 0)).astype(float))
    add_num("oral_only", ((ins_presc_t == 0) & (oral_t > 0)).astype(float), ((ins_presc_e == 0) & (oral_e > 0)).astype(float))

    for key in ["metformin", "glipizide", "glyburide", "pioglitazone"]:
        if key in train and key in eval_cols:
            add_num(f"{key}_prescribed", (train[key] != "No").astype(float), (eval_cols[key] != "No").astype(float))
            add_num(f"{key}_changed", np.isin(train[key], ["Up", "Down"]).astype(float), np.isin(eval_cols[key], ["Up", "Down"]).astype(float))

    add_num("change_flag", (train["change"] == "Ch").astype(float), (eval_cols["change"] == "Ch").astype(float))
    add_num("diabetesMed_flag", (train["diabetesMed"] == "Yes").astype(float), (eval_cols["diabetesMed"] == "Yes").astype(float))

    # 7. Admission, Discharge Dispositions & Clinical Risk Scores
    disch_snf_t = (train["discharge_disposition_id"] == "3").astype(float)
    disch_snf_e = (eval_cols["discharge_disposition_id"] == "3").astype(float)
    disch_home_t = (train["discharge_disposition_id"] == "1").astype(float)
    disch_home_e = (eval_cols["discharge_disposition_id"] == "1").astype(float)
    disch_hh_t = (train["discharge_disposition_id"] == "6").astype(float)
    disch_hh_e = (eval_cols["discharge_disposition_id"] == "6").astype(float)
    disch_ama_t = (train["discharge_disposition_id"] == "7").astype(float)
    disch_ama_e = (eval_cols["discharge_disposition_id"] == "7").astype(float)

    add_num("discharged_home", disch_home_t, disch_home_e)
    add_num("discharged_snf", disch_snf_t, disch_snf_e)
    add_num("discharged_home_health", disch_hh_t, disch_hh_e)
    add_num("discharged_ama", disch_ama_t, disch_ama_e)
    add_num("discharged_rehab", (train["discharge_disposition_id"] == "22").astype(float), (eval_cols["discharge_disposition_id"] == "22").astype(float))
    add_num("discharged_transfer", np.isin(train["discharge_disposition_id"], ["2", "4", "5", "22", "23", "24"]).astype(float),
            np.isin(eval_cols["discharge_disposition_id"], ["2", "4", "5", "22", "23", "24"]).astype(float))
    add_num("discharged_facility", np.isin(train["discharge_disposition_id"], ["2", "3", "4", "5", "6", "22", "23", "24"]).astype(float),
            np.isin(eval_cols["discharge_disposition_id"], ["2", "3", "4", "5", "6", "22", "23", "24"]).astype(float))

    add_num("admitted_er", (train["admission_source_id"] == "7").astype(float), (eval_cols["admission_source_id"] == "7").astype(float))
    add_num("admitted_transfer", np.isin(train["admission_source_id"], ["4", "5", "6"]).astype(float), np.isin(eval_cols["admission_source_id"], ["4", "5", "6"]).astype(float))
    add_num("admitted_referral", np.isin(train["admission_source_id"], ["1", "2"]).astype(float), np.isin(eval_cols["admission_source_id"], ["1", "2"]).astype(float))

    add_num("admission_emergency", (train["admission_type_id"] == "1").astype(float), (eval_cols["admission_type_id"] == "1").astype(float))
    add_num("admission_urgent", (train["admission_type_id"] == "2").astype(float), (eval_cols["admission_type_id"] == "2").astype(float))
    add_num("admission_elective", (train["admission_type_id"] == "3").astype(float), (eval_cols["admission_type_id"] == "3").astype(float))
    add_num("admission_acute", np.isin(train["admission_type_id"], ["1", "2"]).astype(float), np.isin(eval_cols["admission_type_id"], ["1", "2"]).astype(float))

    # Lab tests
    a1c_tested_t = (train["A1Cresult"] != "None").astype(float)
    a1c_tested_e = (eval_cols["A1Cresult"] != "None").astype(float)
    glu_tested_t = (train["max_glu_serum"] != "None").astype(float)
    glu_tested_e = (eval_cols["max_glu_serum"] != "None").astype(float)
    add_num("a1c_tested", a1c_tested_t, a1c_tested_e)
    add_num("a1c_high", np.isin(train["A1Cresult"], [">7", ">8"]).astype(float), np.isin(eval_cols["A1Cresult"], [">7", ">8"]).astype(float))
    add_num("glu_tested", glu_tested_t, glu_tested_e)
    add_num("glu_high", np.isin(train["max_glu_serum"], [">200", ">300"]).astype(float), np.isin(eval_cols["max_glu_serum"], [">200", ">300"]).astype(float))
    add_num("no_glycemic_testing", ((1.0 - a1c_tested_t) * (1.0 - glu_tested_t)), ((1.0 - a1c_tested_e) * (1.0 - glu_tested_e)))

    # Missingness indicators
    for col in ["payer_code", "medical_specialty"]:
        trn_m = np.isin(train[col], [None, "?", "None"]).astype(float)
        eval_m = np.isin(eval_cols[col], [None, "?", "None"]).astype(float)
        add_num(f"{col}_missing", trn_m, eval_m)

    # Composite Clinical Risk Score
    comp_risk_t = (2.0 * inp_trn) + (1.2 * emg_trn) + (0.3 * out_trn) + (1.5 * comorb_trn["has_chf"]) + \
                  (1.5 * comorb_trn["has_ckd"]) + (1.2 * disch_snf_t) + (0.8 * disch_hh_t) + (1.5 * disch_ama_t)
    comp_risk_e = (2.0 * inp_eval) + (1.2 * emg_eval) + (0.3 * out_eval) + (1.5 * comorb_eval["has_chf"]) + \
                  (1.5 * comorb_eval["has_ckd"]) + (1.2 * disch_snf_e) + (0.8 * disch_hh_e) + (1.5 * disch_ama_e)
    add_num("composite_risk_score", comp_risk_t, comp_risk_e)
    add_num("log_composite_risk", np.log1p(comp_risk_t), np.log1p(comp_risk_e))

    # 8. Fit & Transform via TabularEncoder
    encoder = TabularEncoder(min_count=15)
    encoder.fit(
        task,
        train_copy,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_copy)
    x_eval = encoder.transform(eval_copy)

    # 9. Cross-Validated Bagging Ensemble with Structurally Diverse LightGBM Architectures
    n_eval = len(los_eval)
    eval_preds = np.zeros(n_eval, dtype=np.float64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cat_idx = encoder.categorical_indices

    for fold, (trn_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        xt, yt = x_train[trn_idx], y_train[trn_idx]

        # Model 1: Regularized, shallow depth-constrained tree for robust main effects
        m1 = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.032,
            num_leaves=20,
            max_depth=5,
            min_child_samples=55,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.68,
            reg_alpha=0.15,
            reg_lambda=5.0,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m1.fit(xt, yt, categorical_feature=cat_idx)
        p1 = m1.predict_proba(x_eval)[:, 1]

        # Model 2: Extra-Trees randomized GBDT for orthogonal split exploration
        m2 = lgb.LGBMClassifier(
            n_estimators=280,
            learning_rate=0.035,
            num_leaves=28,
            max_depth=6,
            min_child_samples=45,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.72,
            extra_trees=True,
            reg_alpha=0.1,
            reg_lambda=4.0,
            random_state=101 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m2.fit(xt, yt, categorical_feature=cat_idx)
        p2 = m2.predict_proba(x_eval)[:, 1]

        # Model 3: Deeper interaction tree capturing non-linear combinations
        m3 = lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.038,
            num_leaves=34,
            max_depth=7,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.78,
            min_split_gain=0.01,
            reg_alpha=0.05,
            reg_lambda=3.0,
            random_state=202 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m3.fit(xt, yt, categorical_feature=cat_idx)
        p3 = m3.predict_proba(x_eval)[:, 1]

        fold_pred = 0.35 * p1 + 0.35 * p2 + 0.30 * p3
        eval_preds += fold_pred / 5.0

    return np.clip(eval_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
