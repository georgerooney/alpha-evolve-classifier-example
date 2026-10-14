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
    import xgboost as xgb
    from sklearn.decomposition import TruncatedSVD
    from sklearn.model_selection import StratifiedKFold
    from collections import Counter

    n_tr = len(y_train)
    n_ev = len(eval_cols["time_in_hospital"])

    train_ext = dict(train)
    eval_ext = dict(eval_cols)
    new_num_train = {}
    new_num_eval = {}
    new_cat_train = {}
    new_cat_eval = {}

    # Helper Grouping Functions
    def group_discharge(disp):
        if disp is None:
            return "missing"
        s = str(disp).strip()
        if s in ("1", "8"):
            return "home"
        elif s in ("3", "4", "5", "6", "22", "23", "24"):
            return "facility"
        elif s in ("2", "9", "10", "12", "15", "16", "17"):
            return "hospital_transfer"
        elif s in ("18", "25", "26"):
            return "unknown"
        else:
            return "other"

    def group_admission_source(src):
        if src is None:
            return "missing"
        s = str(src).strip()
        if s in ("1", "2", "3"):
            return "referral"
        elif s in ("7",):
            return "emergency"
        elif s in ("4", "5", "6", "10", "22", "25"):
            return "transfer"
        else:
            return "other"

    def group_admission_type(adm_type):
        if adm_type is None:
            return "missing"
        s = str(adm_type).strip()
        if s in ("1", "2"):
            return "urgent"
        elif s == "3":
            return "elective"
        elif s == "4":
            return "newborn"
        elif s in ("5", "6", "7", "8"):
            return "other"
        else:
            return "missing"

    # 1. ICD-9 Chapter Groupings and 3-Digit Prefixes
    def icd9_prefix(code):
        if code is None:
            return "missing"
        s = str(code).strip()
        if not s or s == "?":
            return "missing"
        return s.split(".")[0]

    for d_col in ["diag_1", "diag_2", "diag_3"]:
        d_tr = train[d_col]
        d_ev = eval_cols[d_col]
        new_cat_train[f"{d_col}_group"] = np.array([icd9_group(c) for c in d_tr], dtype=object)
        new_cat_eval[f"{d_col}_group"] = np.array([icd9_group(c) for c in d_ev], dtype=object)
        new_cat_train[f"{d_col}_prefix"] = np.array([icd9_prefix(c) for c in d_tr], dtype=object)
        new_cat_eval[f"{d_col}_prefix"] = np.array([icd9_prefix(c) for c in d_ev], dtype=object)

    d1_grp_tr = new_cat_train["diag_1_group"]
    d2_grp_tr = new_cat_train["diag_2_group"]
    d3_grp_tr = new_cat_train["diag_3_group"]
    d1_grp_ev = new_cat_eval["diag_1_group"]
    d2_grp_ev = new_cat_eval["diag_2_group"]
    d3_grp_ev = new_cat_eval["diag_3_group"]

    # Comorbidity presence indicators
    for cond in ["circulatory", "respiratory", "digestive", "diabetes", "genitourinary"]:
        new_num_train[f"has_{cond}_diag"] = (
            (d1_grp_tr == cond) | (d2_grp_tr == cond) | (d3_grp_tr == cond)
        ).astype(np.float64)
        new_num_eval[f"has_{cond}_diag"] = (
            (d1_grp_ev == cond) | (d2_grp_ev == cond) | (d3_grp_ev == cond)
        ).astype(np.float64)

    new_num_train["num_chronic_conditions"] = (
        new_num_train["has_circulatory_diag"]
        + new_num_train["has_respiratory_diag"]
        + new_num_train["has_digestive_diag"]
        + new_num_train["has_diabetes_diag"]
        + new_num_train["has_genitourinary_diag"]
    )
    new_num_eval["num_chronic_conditions"] = (
        new_num_eval["has_circulatory_diag"]
        + new_num_eval["has_respiratory_diag"]
        + new_num_eval["has_digestive_diag"]
        + new_num_eval["has_diabetes_diag"]
        + new_num_eval["has_genitourinary_diag"]
    )

    # 2. Utilization and Clinical Intensity Features
    def extract_utilization(cols):
        inp = np.nan_to_num(cols["number_inpatient"], nan=0.0).astype(np.float64)
        emg = np.nan_to_num(cols["number_emergency"], nan=0.0).astype(np.float64)
        out = np.nan_to_num(cols["number_outpatient"], nan=0.0).astype(np.float64)
        dur = np.maximum(np.nan_to_num(cols["time_in_hospital"], nan=1.0).astype(np.float64), 1.0)
        n_med = np.nan_to_num(cols["num_medications"], nan=0.0).astype(np.float64)
        n_lab = np.nan_to_num(cols["num_lab_procedures"], nan=0.0).astype(np.float64)
        n_proc = np.nan_to_num(cols["num_procedures"], nan=0.0).astype(np.float64)
        n_diag = np.nan_to_num(cols["number_diagnoses"], nan=0.0).astype(np.float64)

        tot_prior = inp + emg + out
        return {
            "total_prior_visits": tot_prior,
            "has_inpatient": (inp > 0.0).astype(np.float64),
            "has_emergency": (emg > 0.0).astype(np.float64),
            "has_outpatient": (out > 0.0).astype(np.float64),
            "inpatient_ratio": inp / (tot_prior + 1.0),
            "emergency_ratio": emg / (tot_prior + 1.0),
            "inpatient_per_day": inp / dur,
            "emergency_per_day": emg / dur,
            "med_per_day": n_med / dur,
            "lab_per_day": n_lab / dur,
            "proc_per_day": n_proc / dur,
            "diag_per_day": n_diag / dur,
            "intensity_index": (n_med + n_lab + n_proc) / dur,
            "service_intensity": n_diag * n_med,
            "med_times_lab": n_med * n_lab,
            "stay_per_diag": dur / (n_diag + 1e-5),
            "has_no_prior": (tot_prior == 0.0).astype(np.float64),
            "log1p_inpatient": np.log1p(inp),
            "log1p_emergency": np.log1p(emg),
            "log1p_outpatient": np.log1p(out),
            "log1p_total_prior": np.log1p(tot_prior),
        }

    util_tr = extract_utilization(train)
    util_ev = extract_utilization(eval_cols)
    new_num_train.update(util_tr)
    new_num_eval.update(util_ev)

    # Utilization score (weighted inpatient/emergency/outpatient visits)
    inp_tr = np.nan_to_num(train["number_inpatient"], nan=0.0).astype(np.float64)
    inp_ev = np.nan_to_num(eval_cols["number_inpatient"], nan=0.0).astype(np.float64)
    emg_tr = np.nan_to_num(train["number_emergency"], nan=0.0).astype(np.float64)
    emg_ev = np.nan_to_num(eval_cols["number_emergency"], nan=0.0).astype(np.float64)
    out_tr = np.nan_to_num(train["number_outpatient"], nan=0.0).astype(np.float64)
    out_ev = np.nan_to_num(eval_cols["number_outpatient"], nan=0.0).astype(np.float64)

    new_num_train["utilization_score"] = inp_tr * 2.0 + emg_tr * 1.0 + out_tr * 0.5
    new_num_eval["utilization_score"] = inp_ev * 2.0 + emg_ev * 1.0 + out_ev * 0.5

    # LACE Index Proxy calculation
    L_score_tr = np.minimum(np.nan_to_num(train["time_in_hospital"], nan=1.0).astype(np.float64), 14.0)
    L_score_ev = np.minimum(np.nan_to_num(eval_cols["time_in_hospital"], nan=1.0).astype(np.float64), 14.0)

    grp_type_tr_lace = np.array([group_admission_type(x) for x in train["admission_type_id"]], dtype=object)
    grp_type_ev_lace = np.array([group_admission_type(x) for x in eval_cols["admission_type_id"]], dtype=object)
    A_score_tr = np.where(grp_type_tr_lace == "urgent", 3.0, 0.0)
    A_score_ev = np.where(grp_type_ev_lace == "urgent", 3.0, 0.0)

    C_score_tr = new_num_train["num_chronic_conditions"]
    C_score_ev = new_num_eval["num_chronic_conditions"]

    E_score_tr = np.minimum(new_num_train["total_prior_visits"], 4.0)
    E_score_ev = np.minimum(new_num_eval["total_prior_visits"], 4.0)

    new_num_train["lace_proxy"] = L_score_tr + A_score_tr + C_score_tr + E_score_tr
    new_num_eval["lace_proxy"] = L_score_ev + A_score_ev + C_score_ev + E_score_ev

    # 3. Continuous Age Parsing & Demographics
    def parse_age(age_arr):
        res = np.empty(len(age_arr), dtype=np.float64)
        for i, val in enumerate(age_arr):
            if val is None:
                res[i] = 65.0
            else:
                s = str(val)
                if s.startswith("[") and "-" in s:
                    try:
                        low = float(s[1:].split("-")[0])
                        res[i] = low + 5.0
                    except Exception:
                        res[i] = 65.0
                else:
                    res[i] = 65.0
        return res

    age_tr = parse_age(train["age"])
    age_ev = parse_age(eval_cols["age"])
    new_num_train["age_numeric"] = age_tr
    new_num_eval["age_numeric"] = age_ev
    new_num_train["age_x_inpatient"] = age_tr * new_num_train["has_inpatient"]
    new_num_eval["age_x_inpatient"] = age_ev * new_num_eval["has_inpatient"]

    gender_male_tr = (train["gender"] == "Male").astype(np.float64)
    gender_male_ev = (eval_cols["gender"] == "Male").astype(np.float64)
    new_num_train["age_x_gender"] = age_tr * gender_male_tr
    new_num_eval["age_x_gender"] = age_ev * gender_male_ev

    new_num_train["age_x_emergency"] = age_tr * new_num_train["has_emergency"]
    new_num_eval["age_x_emergency"] = age_ev * new_num_eval["has_emergency"]

    new_num_train["age_x_total_prior"] = age_tr * new_num_train["total_prior_visits"]
    new_num_eval["age_x_total_prior"] = age_ev * new_num_eval["total_prior_visits"]

    n_med_tr = np.nan_to_num(train["num_medications"], nan=0.0).astype(np.float64)
    n_med_ev = np.nan_to_num(eval_cols["num_medications"], nan=0.0).astype(np.float64)
    new_num_train["age_x_meds"] = age_tr * n_med_tr
    new_num_eval["age_x_meds"] = age_ev * n_med_ev

    # Diabetic complications tracking
    def count_comp_occurrences(prefix, d1, d2, d3):
        cnt = np.zeros(len(d1), dtype=np.float64)
        for d in [d1, d2, d3]:
            has_pfx = np.array([
                False if (c is None or (isinstance(c, float) and np.isnan(c))) 
                else str(c).strip().startswith(prefix) 
                for c in d
            ], dtype=bool)
            cnt += has_pfx.astype(np.float64)
        return cnt

    def count_comp_starts_with(prefixes, d1, d2, d3):
        cnt = np.zeros(len(d1), dtype=np.float64)
        for d in [d1, d2, d3]:
            has_pfx = np.array([
                False if (c is None or (isinstance(c, float) and np.isnan(c))) 
                else any(str(c).strip().startswith(pfx) for pfx in prefixes)
                for c in d
            ], dtype=bool)
            cnt += has_pfx.astype(np.float64)
        return cnt

    d1_raw, d2_raw, d3_raw = train["diag_1"], train["diag_2"], train["diag_3"]
    d1_raw_ev, d2_raw_ev, d3_raw_ev = eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"]

    new_num_train["comp_renal"] = count_comp_occurrences("250.4", d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_renal"] = count_comp_occurrences("250.4", d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_ophthalmic"] = count_comp_occurrences("250.5", d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_ophthalmic"] = count_comp_occurrences("250.5", d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_neurological"] = count_comp_occurrences("250.6", d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_neurological"] = count_comp_occurrences("250.6", d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_circulatory"] = count_comp_occurrences("250.7", d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_circulatory"] = count_comp_occurrences("250.7", d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_ketoacidosis"] = count_comp_occurrences("250.1", d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_ketoacidosis"] = count_comp_occurrences("250.1", d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_hypertension"] = count_comp_starts_with(["401", "402", "403", "404", "405"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_hypertension"] = count_comp_starts_with(["401", "402", "403", "404", "405"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_ihd"] = count_comp_starts_with(["410", "411", "412", "413", "414"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_ihd"] = count_comp_starts_with(["410", "411", "412", "413", "414"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_chf"] = count_comp_starts_with(["428"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_chf"] = count_comp_starts_with(["428"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    # 4. Medication Dynamics & Guideline Compliance Interactions
    med_names = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone"
    ]
    available_meds = [m for m in med_names if m in train and m in eval_cols]

    def extract_med_counts(cols, n_rows):
        active = np.zeros(n_rows, dtype=np.float64)
        up = np.zeros(n_rows, dtype=np.float64)
        down = np.zeros(n_rows, dtype=np.float64)
        steady = np.zeros(n_rows, dtype=np.float64)
        for m in available_meds:
            arr = cols[m]
            is_act = np.array([0.0 if (v is None or v == "No") else 1.0 for v in arr], dtype=np.float64)
            is_up = np.array([1.0 if v == "Up" else 0.0 for v in arr], dtype=np.float64)
            is_down = np.array([1.0 if v == "Down" else 0.0 for v in arr], dtype=np.float64)
            is_steady = np.array([1.0 if v == "Steady" else 0.0 for v in arr], dtype=np.float64)
            active += is_act
            up += is_up
            down += is_down
            steady += is_steady
        return active, up, down, steady, up + down

    (new_num_train["num_active_meds"], new_num_train["num_up_meds"],
     new_num_train["num_down_meds"], new_num_train["num_steady_meds"],
     new_num_changed_meds_tmp) = extract_med_counts(train, n_tr)
    new_num_train["num_changed_meds"] = new_num_changed_meds_tmp

    (new_num_eval["num_active_meds"], new_num_eval["num_up_meds"],
     new_num_eval["num_down_meds"], new_num_eval["num_steady_meds"],
     new_num_changed_meds_tmp_ev) = extract_med_counts(eval_cols, n_ev)
    new_num_eval["num_changed_meds"] = new_num_changed_meds_tmp_ev

    # Non-diabetes medications and medication ratios
    new_num_train["non_diabetes_meds"] = np.maximum(n_med_tr - new_num_train["num_active_meds"], 0.0)
    new_num_eval["non_diabetes_meds"] = np.maximum(n_med_ev - new_num_eval["num_active_meds"], 0.0)

    new_num_train["active_meds_ratio"] = new_num_train["num_active_meds"] / (n_med_tr + 1e-5)
    new_num_eval["active_meds_ratio"] = new_num_eval["num_active_meds"] / (n_med_ev + 1e-5)

    new_num_train["up_ratio"] = new_num_train["num_up_meds"] / (new_num_train["num_changed_meds"] + 1e-5)
    new_num_eval["up_ratio"] = new_num_eval["num_up_meds"] / (new_num_eval["num_changed_meds"] + 1e-5)

    new_num_train["down_ratio"] = new_num_train["num_down_meds"] / (new_num_train["num_changed_meds"] + 1e-5)
    new_num_eval["down_ratio"] = new_num_eval["num_down_meds"] / (new_num_eval["num_changed_meds"] + 1e-5)

    def get_med_active(cols, med):
        if med in cols:
            arr = cols[med]
            return np.array([0.0 if (v is None or v == "No") else 1.0 for v in arr], dtype=np.float64)
        return np.zeros(len(cols["time_in_hospital"]), dtype=np.float64)

    # Specific key drugs and A1C guideline interactions
    for prefix, data_dict, num_dict in [("tr", train, new_num_train), ("ev", eval_cols, new_num_eval)]:
        ins = data_dict.get("insulin")
        if ins is not None:
            num_dict["insulin_active"] = np.array([0.0 if (v is None or v == "No") else 1.0 for v in ins], dtype=np.float64)
            num_dict["insulin_up"] = np.array([1.0 if v == "Up" else 0.0 for v in ins], dtype=np.float64)
            num_dict["insulin_down"] = np.array([1.0 if v == "Down" else 0.0 for v in ins], dtype=np.float64)
            num_dict["insulin_changed"] = np.array([1.0 if v in ("Up", "Down") else 0.0 for v in ins], dtype=np.float64)
        else:
            num_dict["insulin_active"] = np.zeros(len(data_dict["time_in_hospital"]), dtype=np.float64)
            num_dict["insulin_up"] = np.zeros(len(data_dict["time_in_hospital"]), dtype=np.float64)
            num_dict["insulin_down"] = np.zeros(len(data_dict["time_in_hospital"]), dtype=np.float64)
            num_dict["insulin_changed"] = np.zeros(len(data_dict["time_in_hospital"]), dtype=np.float64)

        metf = data_dict.get("metformin")
        if metf is not None:
            num_dict["metformin_active"] = np.array([0.0 if (v is None or v == "No") else 1.0 for v in metf], dtype=np.float64)
        else:
            num_dict["metformin_active"] = np.zeros(len(data_dict["time_in_hospital"]), dtype=np.float64)

        glip_act = get_med_active(data_dict, "glipizide")
        glyb_act = get_med_active(data_dict, "glyburide")
        glim_act = get_med_active(data_dict, "glimepiride")
        piog_act = get_med_active(data_dict, "pioglitazone")
        ros_act = get_med_active(data_dict, "rosiglitazone")

        num_dict["sulf_active"] = ((glip_act + glyb_act + glim_act) > 0).astype(np.float64)
        num_dict["tzd_active"] = ((piog_act + ros_act) > 0).astype(np.float64)

        ins_act = num_dict["insulin_active"]
        met_act = num_dict["metformin_active"]
        num_dict["insulin_and_metformin"] = ins_act * met_act
        num_dict["insulin_and_sulf"] = ins_act * num_dict["sulf_active"]
        num_dict["metformin_and_sulf"] = met_act * num_dict["sulf_active"]
        num_dict["insulin_and_oral"] = ins_act * ((num_dict["num_active_meds"] - ins_act) > 0).astype(np.float64)

        a1c = data_dict.get("A1Cresult")
        chg_arr = data_dict.get("change")
        if chg_arr is not None:
            chg = np.array([1.0 if v == "Ch" else 0.0 for v in chg_arr], dtype=np.float64)
        else:
            chg = np.zeros(len(data_dict["time_in_hospital"]), dtype=np.float64)

        num_dict["a1c_tested"] = np.array([0.0 if (v is None or v == "None") else 1.0 for v in a1c], dtype=np.float64)
        a1c_high = np.array([1.0 if (v == ">7" or v == ">8") else 0.0 for v in a1c], dtype=np.float64)
        num_dict["a1c_high"] = a1c_high
        num_dict["a1c_high_no_change"] = a1c_high * (1.0 - chg)
        num_dict["a1c_high_changed"] = a1c_high * chg
        num_dict["a1c_normal"] = np.array([1.0 if v == "Norm" else 0.0 for v in a1c], dtype=np.float64)

        glu = data_dict.get("max_glu_serum")
        if glu is not None:
            num_dict["glu_tested"] = np.array([0.0 if (v is None or v == "None") else 1.0 for v in glu], dtype=np.float64)
            num_dict["glu_high"] = np.array([1.0 if (v == ">200" or v == ">300") else 0.0 for v in glu], dtype=np.float64)
            num_dict["glu_normal"] = np.array([1.0 if v == "Norm" else 0.0 for v in glu], dtype=np.float64)
        else:
            num_dict["glu_tested"] = np.zeros(len(data_dict["time_in_hospital"]), dtype=np.float64)
            num_dict["glu_high"] = np.zeros(len(data_dict["time_in_hospital"]), dtype=np.float64)
            num_dict["glu_normal"] = np.zeros(len(data_dict["time_in_hospital"]), dtype=np.float64)

    # 5. Tensor Decomposition: Latent Clinical Phenotypes & CP Bilinear Interaction
    # Multi-hot clinical concept matrix
    chapter_list = ["circulatory", "respiratory", "digestive", "diabetes", "injury", "musculoskeletal", "genitourinary", "neoplasms", "other"]
    D_train = np.column_stack([
        ((d1_grp_tr == grp).astype(np.float64) + (d2_grp_tr == grp).astype(np.float64) + (d3_grp_tr == grp).astype(np.float64))
        for grp in chapter_list
    ])
    D_eval = np.column_stack([
        ((d1_grp_ev == grp).astype(np.float64) + (d2_grp_ev == grp).astype(np.float64) + (d3_grp_ev == grp).astype(np.float64))
        for grp in chapter_list
    ])

    # Top 30 most frequent ICD-9 3-digit prefixes in train
    p1_tr, p2_tr, p3_tr = new_cat_train["diag_1_prefix"], new_cat_train["diag_2_prefix"], new_cat_train["diag_3_prefix"]
    p1_ev, p2_ev, p3_ev = new_cat_eval["diag_1_prefix"], new_cat_eval["diag_2_prefix"], new_cat_eval["diag_3_prefix"]
    counts = Counter([p for p in np.concatenate([p1_tr, p2_tr, p3_tr]) if p != "missing"])
    top_prefixes = [p for p, _ in counts.most_common(30)]

    C_tr_parts = [D_train]
    C_ev_parts = [D_eval]
    if top_prefixes:
        pref_tr = np.column_stack([((p1_tr == p) | (p2_tr == p) | (p3_tr == p)).astype(np.float64) for p in top_prefixes])
        pref_ev = np.column_stack([((p1_ev == p) | (p2_ev == p) | (p3_ev == p)).astype(np.float64) for p in top_prefixes])
        C_tr_parts.append(pref_tr)
        C_ev_parts.append(pref_ev)

    if available_meds:
        med_act_tr = np.column_stack([((train[m] != "No") & (train[m] != None)).astype(np.float64) for m in available_meds])  # noqa: E711
        med_act_ev = np.column_stack([((eval_cols[m] != "No") & (eval_cols[m] != None)).astype(np.float64) for m in available_meds])  # noqa: E711
        C_tr_parts.append(med_act_tr)
        C_ev_parts.append(med_act_ev)

    top_disp = ["1", "3", "6", "2", "7", "18", "22", "25"]
    disp_tr = np.column_stack([(train["discharge_disposition_id"] == v).astype(np.float64) for v in top_disp])
    disp_ev = np.column_stack([(eval_cols["discharge_disposition_id"] == v).astype(np.float64) for v in top_disp])
    C_tr_parts.append(disp_tr)
    C_ev_parts.append(disp_ev)

    C_train = np.column_stack(C_tr_parts)
    C_eval = np.column_stack(C_ev_parts)

    svd = TruncatedSVD(n_components=10, random_state=42)
    z_train = svd.fit_transform(C_train)
    z_eval = svd.transform(C_eval)
    for k in range(10):
        new_num_train[f"svd_phenotype_{k}"] = z_train[:, k]
        new_num_eval[f"svd_phenotype_{k}"] = z_eval[:, k]

    # Canonical Polyadic (CP) rank-1 tensor interaction projections
    key_meds = [m for m in ["insulin", "metformin", "glipizide", "glyburide", "pioglitazone", "rosiglitazone"] if m in train and m in eval_cols]
    if key_meds:
        M_train = np.column_stack([((train[m] != "No") & (train[m] != None)).astype(np.float64) for m in key_meds])  # noqa: E711
        M_eval = np.column_stack([((eval_cols[m] != "No") & (eval_cols[m] != None)).astype(np.float64) for m in key_meds])  # noqa: E711
        svd_d = TruncatedSVD(n_components=2, random_state=42)
        svd_m = TruncatedSVD(n_components=2, random_state=42)
        zd_tr = svd_d.fit_transform(D_train)
        zd_ev = svd_d.transform(D_eval)
        zm_tr = svd_m.fit_transform(M_train)
        zm_ev = svd_m.transform(M_eval)
        for i in range(2):
            for j in range(2):
                new_num_train[f"tensor_cp_d{i}_m{j}"] = zd_tr[:, i] * zm_tr[:, j]
                new_num_eval[f"tensor_cp_d{i}_m{j}"] = zd_ev[:, i] * zm_ev[:, j]

    # 6. Empirical Bayes Out-of-Fold Target Encoding
    def compute_bayes_target_encoding(train_arr, eval_arr, y, m_smooth=20.0, n_splits=5, seed=42):
        clean_tr = np.array([str(c) if c is not None else "Missing" for c in train_arr], dtype=object)
        clean_ev = np.array([str(c) if c is not None else "Missing" for c in eval_arr], dtype=object)
        global_mean = float(np.mean(y))
        oof_train = np.full(len(clean_tr), global_mean, dtype=np.float64)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for tr_idx, val_idx in skf.split(clean_tr, y):
            cats_tr, y_tr = clean_tr[tr_idx], y[tr_idx]
            uniq, inv, cnts = np.unique(cats_tr, return_inverse=True, return_counts=True)
            sums = np.bincount(inv, weights=y_tr)
            smoothed = (sums + m_smooth * global_mean) / (cnts + m_smooth)
            cat_map = dict(zip(uniq, smoothed))
            oof_train[val_idx] = [cat_map.get(c, global_mean) for c in clean_tr[val_idx]]

        uniq, inv, cnts = np.unique(clean_tr, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=y)
        smoothed = (sums + m_smooth * global_mean) / (cnts + m_smooth)
        full_map = dict(zip(uniq, smoothed))
        eval_out = np.array([full_map.get(c, global_mean) for c in clean_ev], dtype=np.float64)
        return oof_train, eval_out

    grp_disp_tr = np.array([group_discharge(x) for x in train["discharge_disposition_id"]], dtype=object)
    grp_disp_ev = np.array([group_discharge(x) for x in eval_cols["discharge_disposition_id"]], dtype=object)
    grp_src_tr = np.array([group_admission_source(x) for x in train["admission_source_id"]], dtype=object)
    grp_src_ev = np.array([group_admission_source(x) for x in eval_cols["admission_source_id"]], dtype=object)
    grp_type_tr = np.array([group_admission_type(x) for x in train["admission_type_id"]], dtype=object)
    grp_type_ev = np.array([group_admission_type(x) for x in eval_cols["admission_type_id"]], dtype=object)

    new_cat_train["grp_discharge"] = grp_disp_tr
    new_cat_eval["grp_discharge"] = grp_disp_ev
    new_cat_train["grp_adm_source"] = grp_src_tr
    new_cat_eval["grp_adm_source"] = grp_src_ev
    new_cat_train["grp_adm_type"] = grp_type_tr
    new_cat_eval["grp_adm_type"] = grp_type_ev

    grp_disp_adm_tr = np.array([f"{d}_{a}" for d, a in zip(grp_disp_tr, grp_src_tr)], dtype=object)
    grp_disp_adm_ev = np.array([f"{d}_{a}" for d, a in zip(grp_disp_ev, grp_src_ev)], dtype=object)
    new_cat_train["grp_disp_adm"] = grp_disp_adm_tr
    new_cat_eval["grp_disp_adm"] = grp_disp_adm_ev

    disp_adm = np.array([f"{d}_{a}" for d, a in zip(train["discharge_disposition_id"], train["admission_source_id"])], dtype=object)
    disp_adm_ev = np.array([f"{d}_{a}" for d, a in zip(eval_cols["discharge_disposition_id"], eval_cols["admission_source_id"])], dtype=object)

    diag_disp = np.array([f"{g}_{d}" for g, d in zip(new_cat_train["diag_1_group"], train["discharge_disposition_id"])], dtype=object)
    diag_disp_ev = np.array([f"{g}_{d}" for g, d in zip(new_cat_eval["diag_1_group"], eval_cols["discharge_disposition_id"])], dtype=object)

    new_cat_train["disp_adm"] = disp_adm
    new_cat_eval["disp_adm"] = disp_adm_ev
    new_cat_train["diag_disp"] = diag_disp
    new_cat_eval["diag_disp"] = diag_disp_ev

    te_cols = [
        ("discharge_disposition_id", train["discharge_disposition_id"], eval_cols["discharge_disposition_id"]),
        ("admission_source_id", train["admission_source_id"], eval_cols["admission_source_id"]),
        ("admission_type_id", train["admission_type_id"], eval_cols["admission_type_id"]),
        ("medical_specialty", train["medical_specialty"], eval_cols["medical_specialty"]),
        ("diag_1_prefix", new_cat_train["diag_1_prefix"], new_cat_eval["diag_1_prefix"]),
        ("diag_2_prefix", new_cat_train["diag_2_prefix"], new_cat_eval["diag_2_prefix"]),
        ("diag_3_prefix", new_cat_train["diag_3_prefix"], new_cat_eval["diag_3_prefix"]),
        ("diag_1_group", new_cat_train["diag_1_group"], new_cat_eval["diag_1_group"]),
        ("diag_2_group", new_cat_train["diag_2_group"], new_cat_eval["diag_2_group"]),
        ("diag_3_group", new_cat_train["diag_3_group"], new_cat_eval["diag_3_group"]),
        ("payer_code", train["payer_code"], eval_cols["payer_code"]),
        ("age", train["age"], eval_cols["age"]),
        ("disp_adm", disp_adm, disp_adm_ev),
        ("diag_disp", diag_disp, diag_disp_ev),
        ("grp_discharge", grp_disp_tr, grp_disp_ev),
        ("grp_adm_source", grp_src_tr, grp_src_ev),
        ("grp_adm_type", grp_type_tr, grp_type_ev),
        ("grp_disp_adm", grp_disp_adm_tr, grp_disp_adm_ev),
    ]
    for col_name, arr_tr, arr_ev in te_cols:
        te_tr, te_ev = compute_bayes_target_encoding(arr_tr, arr_ev, y_train, m_smooth=20.0)
        new_num_train[f"te_{col_name}"] = te_tr
        new_num_eval[f"te_{col_name}"] = te_ev

    # Register all derived features
    for k, v in new_num_train.items():
        train_ext[k] = v
        eval_ext[k] = new_num_eval[k]
    for k, v in new_cat_train.items():
        train_ext[k] = v
        eval_ext[k] = new_cat_eval[k]

    extra_numeric = list(new_num_train.keys())
    extra_categorical = list(new_cat_train.keys())

    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(task, train_ext, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_eval = encoder.transform(eval_ext)

    # 7. Stratified 5-Fold Cross-Validated Ensemble & Full Train Blend
    skf_model = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    p1_cv_total = np.zeros(n_ev)
    p2_cv_total = np.zeros(n_ev)
    p3_cv_total = np.zeros(n_ev)
    p4_cv_total = np.zeros(n_ev)
    
    for fold, (tr_idx, val_idx) in enumerate(skf_model.split(x_train, y_train)):
        X_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
        
        # Model 1: LGBM Leaf-wise
        m1_fold = lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.045,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=5.0,
            reg_alpha=0.5,
            random_state=42 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m1_fold.fit(X_tr, y_tr, categorical_feature=encoder.categorical_indices)
        p1_cv_total += m1_fold.predict_proba(x_eval)[:, 1]
        
        # Model 2: LGBM Depth-wise / Regularized
        m2_fold = lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.04,
            max_depth=5,
            num_leaves=22,
            min_child_samples=50,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=8.0,
            reg_alpha=1.0,
            random_state=101 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m2_fold.fit(X_tr, y_tr, categorical_feature=encoder.categorical_indices)
        p2_cv_total += m2_fold.predict_proba(x_eval)[:, 1]

        # Model 3: LGBM Extra Trees
        m3_fold = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=35,
            min_child_samples=45,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            extra_trees=True,
            reg_lambda=10.0,
            reg_alpha=2.0,
            random_state=555 + fold,
            n_jobs=1,
            verbose=-1,
        )
        m3_fold.fit(X_tr, y_tr, categorical_feature=encoder.categorical_indices)
        p3_cv_total += m3_fold.predict_proba(x_eval)[:, 1]
        
        # Model 4: XGBoost Hist-based
        try:
            m4_fold = xgb.XGBClassifier(
                n_estimators=180,
                learning_rate=0.045,
                max_depth=4,
                min_child_weight=25,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_lambda=6.0,
                reg_alpha=1.0,
                tree_method="hist",
                random_state=2023 + fold,
                n_jobs=1,
                eval_metric="logloss",
            )
            m4_fold.fit(X_tr, y_tr)
            p4_cv_total += m4_fold.predict_proba(x_eval)[:, 1]
        except Exception:
            pass

    p1_cv = p1_cv_total / 5.0
    p2_cv = p2_cv_total / 5.0
    p3_cv = p3_cv_total / 5.0
    p4_cv = p4_cv_total / 5.0

    # Train on 100% of data
    m1_full = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1_full.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p1_full = m1_full.predict_proba(x_eval)[:, 1]

    m2_full = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.035,
        max_depth=5,
        num_leaves=22,
        min_child_samples=50,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=8.0,
        reg_alpha=1.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m2_full.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p2_full = m2_full.predict_proba(x_eval)[:, 1]

    m3_full = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.04,
        num_leaves=35,
        min_child_samples=45,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.6,
        extra_trees=True,
        reg_lambda=10.0,
        reg_alpha=2.0,
        random_state=555,
        n_jobs=1,
        verbose=-1,
    )
    m3_full.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p3_full = m3_full.predict_proba(x_eval)[:, 1]

    try:
        m4_full = xgb.XGBClassifier(
            n_estimators=220,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=25,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=6.0,
            reg_alpha=1.0,
            tree_method="hist",
            random_state=2023,
            n_jobs=1,
            eval_metric="logloss",
        )
        m4_full.fit(x_train, y_train)
        p4_full = m4_full.predict_proba(x_eval)[:, 1]
        
        preds_cv = 0.30 * p1_cv + 0.30 * p2_cv + 0.20 * p3_cv + 0.20 * p4_cv
        preds_full = 0.30 * p1_full + 0.30 * p2_full + 0.20 * p3_full + 0.20 * p4_full
    except Exception:
        preds_cv = 0.40 * p1_cv + 0.35 * p2_cv + 0.25 * p3_cv
        preds_full = 0.40 * p1_full + 0.35 * p2_full + 0.25 * p3_full

    preds = 0.60 * preds_cv + 0.40 * preds_full
    preds = np.clip(preds, 0.0, 1.0)
    preds = np.nan_to_num(preds, nan=float(np.mean(y_train)))
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
