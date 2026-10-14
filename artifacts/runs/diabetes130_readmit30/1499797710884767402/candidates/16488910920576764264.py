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
    from sklearn.cluster import KMeans
    from scipy.special import logsumexp, softmax
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
        elif s == "3":
            return "snf"
        elif s == "6":
            return "home_health"
        elif s == "7":
            return "ama"
        elif s in ("4", "5", "22", "23", "24"):
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
            "frequent_inpatient": (inp >= 2.0).astype(np.float64),
            "has_emergency": (emg > 0.0).astype(np.float64),
            "has_outpatient": (out > 0.0).astype(np.float64),
            "inpatient_ratio": inp / (tot_prior + 1.0),
            "emergency_ratio": emg / (tot_prior + 1.0),
            "emergency_inpatient_ratio": (emg + inp) / (tot_prior + 1.0),
            "revolving_door": ((inp >= 2.0) & (emg >= 1.0)).astype(np.float64),
            "inpatient_per_day": inp / dur,
            "emergency_per_day": emg / dur,
            "med_per_day": n_med / dur,
            "lab_per_day": n_lab / dur,
            "proc_per_day": n_proc / dur,
            "diag_per_day": n_diag / dur,
            "intensity_index": (n_med + n_lab + n_proc) / dur,
            "kinetic_intensity": (n_lab + 2.0 * n_proc + 1.5 * n_med) / (dur + 0.5),
            "polypharmacy": (n_med >= 15.0).astype(np.float64),
            "meds_per_diag": n_med / (n_diag + 1e-5),
            "complexity_score": (n_diag * 2.0 + n_med + dur * 1.5 + inp * 5.0),
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

    new_num_train["comp_chf"] = count_comp_starts_with(["428", "398", "402", "404"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_chf"] = count_comp_starts_with(["428", "398", "402", "404"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_ckd"] = count_comp_starts_with(["585", "586", "584", "403", "404"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_ckd"] = count_comp_starts_with(["585", "586", "584", "403", "404"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_copd"] = count_comp_starts_with(["490", "491", "492", "493", "494", "495", "496", "518"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_copd"] = count_comp_starts_with(["490", "491", "492", "493", "494", "495", "496", "518"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_fluid"] = count_comp_starts_with(["276"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_fluid"] = count_comp_starts_with(["276"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_infection"] = count_comp_starts_with(["038", "482", "486", "995", "682"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_infection"] = count_comp_starts_with(["038", "482", "486", "995", "682"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_pvd"] = count_comp_starts_with(["443", "707"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_pvd"] = count_comp_starts_with(["443", "707"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    def is_complicated_diabetes(val):
        if val is None:
            return False
        s = str(val).strip()
        if s.startswith("250."):
            parts = s.split(".")
            if len(parts) > 1 and len(parts[1]) > 0:
                return parts[1][0] not in ("0",)
        return False

    def count_complicated_diabetes(d1, d2, d3):
        cnt = np.zeros(len(d1), dtype=np.float64)
        for d in [d1, d2, d3]:
            has_cd = np.array([is_complicated_diabetes(c) for c in d], dtype=bool)
            cnt += has_cd.astype(np.float64)
        return cnt

    new_num_train["has_comp_diabetes"] = (count_complicated_diabetes(d1_raw, d2_raw, d3_raw) > 0).astype(np.float64)
    new_num_eval["has_comp_diabetes"] = (count_complicated_diabetes(d1_raw_ev, d2_raw_ev, d3_raw_ev) > 0).astype(np.float64)

    p1_tr_diag = new_cat_train["diag_1_prefix"]
    p1_ev_diag = new_cat_eval["diag_1_prefix"]
    new_num_train["primary_diag_chf"] = (p1_tr_diag == "428").astype(np.float64)
    new_num_eval["primary_diag_chf"] = (p1_ev_diag == "428").astype(np.float64)
    new_num_train["primary_diag_diabetes"] = (p1_tr_diag == "250").astype(np.float64)
    new_num_eval["primary_diag_diabetes"] = (p1_ev_diag == "250").astype(np.float64)
    new_num_train["primary_diag_circulatory"] = (d1_grp_tr == "circulatory").astype(np.float64)
    new_num_eval["primary_diag_circulatory"] = (d1_grp_ev == "circulatory").astype(np.float64)
    new_num_train["primary_diag_respiratory"] = (d1_grp_tr == "respiratory").astype(np.float64)
    new_num_eval["primary_diag_respiratory"] = (d1_grp_ev == "respiratory").astype(np.float64)

    # Re-evaluate comprehensive Charlson/Elixhauser chronic comorbidity count and update LACE proxy
    for pfx, num_dict in [("tr", new_num_train), ("ev", new_num_eval)]:
        num_dict["num_chronic_conditions"] = (
            num_dict["has_diabetes_diag"]
            + (num_dict["comp_chf"] > 0).astype(np.float64)
            + (num_dict["comp_ihd"] > 0).astype(np.float64)
            + (num_dict["comp_ckd"] > 0).astype(np.float64)
            + (num_dict["comp_copd"] > 0).astype(np.float64)
            + (num_dict["comp_fluid"] > 0).astype(np.float64)
            + (num_dict["comp_infection"] > 0).astype(np.float64)
            + (num_dict["comp_hypertension"] > 0).astype(np.float64)
            + (num_dict["comp_pvd"] > 0).astype(np.float64)
        )
    new_num_train["lace_proxy"] = L_score_tr + A_score_tr + new_num_train["num_chronic_conditions"] + E_score_tr
    new_num_eval["lace_proxy"] = L_score_ev + A_score_ev + new_num_eval["num_chronic_conditions"] + E_score_ev

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

    # Clinical syndromic risk & admission/discharge indicators
    for pfx, data_dict, num_dict in [("tr", train, new_num_train), ("ev", eval_cols, new_num_eval)]:
        has_ckd = ((num_dict["comp_ckd"] > 0) | (num_dict["comp_renal"] > 0)).astype(np.float64)
        has_chf = (num_dict["comp_chf"] > 0).astype(np.float64)
        has_diab = num_dict["has_diabetes_diag"]
        ins_act = num_dict["insulin_active"]

        num_dict["cardiorenal_syndrome"] = (has_diab * has_chf * has_ckd).astype(np.float64)
        num_dict["chf_and_ckd"] = (has_chf * has_ckd).astype(np.float64)
        num_dict["cardiorenal_score"] = (has_diab + 1.5 * has_chf + 1.5 * has_ckd).astype(np.float64)
        num_dict["hypoglycemia_risk"] = (ins_act * has_ckd).astype(np.float64)
        num_dict["sulf_in_ckd"] = (num_dict["sulf_active"] * has_ckd).astype(np.float64)
        num_dict["metformin_in_renal"] = (num_dict["metformin_active"] * has_ckd).astype(np.float64)
        age_arr_ref = age_tr if pfx == "tr" else age_ev
        num_dict["hypoglycemia_prone"] = (((num_dict["sulf_active"] > 0) | (ins_act > 0)) & ((has_ckd > 0) | (age_arr_ref >= 75.0))).astype(np.float64)
        num_dict["clinical_inertia"] = (((num_dict["a1c_high"] > 0) | (num_dict["glu_high"] > 0)) & (num_dict["num_changed_meds"] == 0)).astype(np.float64)

        d_str = np.array([str(x) if x is not None else "missing" for x in data_dict.get("discharge_disposition_id", [])], dtype=object)
        s_str = np.array([str(x) if x is not None else "missing" for x in data_dict.get("admission_source_id", [])], dtype=object)
        t_str = np.array([str(x) if x is not None else "missing" for x in data_dict.get("admission_type_id", [])], dtype=object)

        num_dict["discharged_to_home"] = np.isin(d_str, ["1", "8"]).astype(np.float64)
        num_dict["discharged_to_snf"] = (d_str == "3").astype(np.float64)
        num_dict["discharged_to_home_health"] = (d_str == "6").astype(np.float64)
        num_dict["discharged_left_ama"] = (d_str == "7").astype(np.float64)
        facility_set = ["2", "3", "4", "5", "9", "10", "12", "15", "16", "17", "22", "23", "24", "27", "28", "29", "30"]
        num_dict["discharged_to_facility"] = np.isin(d_str, facility_set).astype(np.float64)
        num_dict["discharged_high_risk"] = np.isin(d_str, ["3", "6", "7", "18", "22", "23", "28"]).astype(np.float64)

        num_dict["admit_from_emergency"] = (s_str == "7").astype(np.float64)
        num_dict["admit_from_referral"] = np.isin(s_str, ["1", "2", "3"]).astype(np.float64)
        num_dict["admit_from_transfer"] = np.isin(s_str, ["4", "5", "6", "10", "18", "22"]).astype(np.float64)

        num_dict["admit_type_emergency"] = (t_str == "1").astype(np.float64)
        num_dict["admit_type_urgent"] = (t_str == "2").astype(np.float64)
        num_dict["admit_type_elective"] = (t_str == "3").astype(np.float64)

    # 5. Tensor Decomposition: Structured Multi-Mode Factorization & CP Bilinear Coupling
    # Mode 1: Diagnosis Space (ICD-9 Chapters + Top Prefixes)
    chapter_list = ["circulatory", "respiratory", "digestive", "diabetes", "injury", "musculoskeletal", "genitourinary", "neoplasms", "other"]
    D_tr_list = [((d1_grp_tr == grp).astype(np.float64) + (d2_grp_tr == grp).astype(np.float64) + (d3_grp_tr == grp).astype(np.float64)) for grp in chapter_list]
    D_ev_list = [((d1_grp_ev == grp).astype(np.float64) + (d2_grp_ev == grp).astype(np.float64) + (d3_grp_ev == grp).astype(np.float64)) for grp in chapter_list]

    p1_tr, p2_tr, p3_tr = new_cat_train["diag_1_prefix"], new_cat_train["diag_2_prefix"], new_cat_train["diag_3_prefix"]
    p1_ev, p2_ev, p3_ev = new_cat_eval["diag_1_prefix"], new_cat_eval["diag_2_prefix"], new_cat_eval["diag_3_prefix"]
    counts = Counter([p for p in np.concatenate([p1_tr, p2_tr, p3_tr]) if p != "missing"])
    top_prefixes = [p for p, _ in counts.most_common(16)]
    if top_prefixes:
        D_tr_list.extend([((p1_tr == p) | (p2_tr == p) | (p3_tr == p)).astype(np.float64) for p in top_prefixes])
        D_ev_list.extend([((p1_ev == p) | (p2_ev == p) | (p3_ev == p)).astype(np.float64) for p in top_prefixes])
    D_train = np.column_stack(D_tr_list)
    D_eval = np.column_stack(D_ev_list)

    # Mode 2: Pharmacotherapy Space (Key Agents, Regimen Dynamics & Polypharmacy)
    top_med_names = ["insulin", "metformin", "glipizide", "glyburide", "glimepiride", "pioglitazone", "rosiglitazone"]
    m_tr_cols = []
    m_ev_cols = []
    for m in top_med_names:
        if m in train:
            m_tr_cols.append(((train[m] != "No") & (train[m] != None)).astype(np.float64))  # noqa: E711
            m_ev_cols.append(((eval_cols[m] != "No") & (eval_cols[m] != None)).astype(np.float64))  # noqa: E711
        else:
            m_tr_cols.append(np.zeros(n_tr, dtype=np.float64))
            m_ev_cols.append(np.zeros(n_ev, dtype=np.float64))
    m_tr_cols.append((train.get("change") == "Ch").astype(np.float64))
    m_ev_cols.append((eval_cols.get("change") == "Ch").astype(np.float64))
    m_tr_cols.append((train.get("insulin") == "Up").astype(np.float64))
    m_ev_cols.append((eval_cols.get("insulin") == "Up").astype(np.float64))
    m_tr_cols.append((train.get("insulin") == "Down").astype(np.float64))
    m_ev_cols.append((eval_cols.get("insulin") == "Down").astype(np.float64))
    m_tr_cols.append((new_num_train["num_active_meds"] / 5.0).astype(np.float64))
    m_ev_cols.append((new_num_eval["num_active_meds"] / 5.0).astype(np.float64))
    m_tr_cols.append((np.nan_to_num(train["num_medications"], nan=0.0) >= 15.0).astype(np.float64))
    m_ev_cols.append((np.nan_to_num(eval_cols["num_medications"], nan=0.0) >= 15.0).astype(np.float64))
    M_train = np.column_stack(m_tr_cols)
    M_eval = np.column_stack(m_ev_cols)

    # Mode 3: Care Transition & Acuity Space (Discharge Disposition, Admission Source & Type)
    d_tr_id = np.array([str(x) if x is not None else "" for x in train["discharge_disposition_id"]], dtype=object)
    d_ev_id = np.array([str(x) if x is not None else "" for x in eval_cols["discharge_disposition_id"]], dtype=object)
    s_tr_id = np.array([str(x) if x is not None else "" for x in train["admission_source_id"]], dtype=object)
    s_ev_id = np.array([str(x) if x is not None else "" for x in eval_cols["admission_source_id"]], dtype=object)
    t_tr_id = np.array([str(x) if x is not None else "" for x in train["admission_type_id"]], dtype=object)
    t_ev_id = np.array([str(x) if x is not None else "" for x in eval_cols["admission_type_id"]], dtype=object)

    u_tr_cols = [
        np.isin(d_tr_id, ["1", "8"]).astype(np.float64),
        (d_tr_id == "3").astype(np.float64),
        (d_tr_id == "6").astype(np.float64),
        np.isin(d_tr_id, ["2", "4", "5", "22", "23", "24"]).astype(np.float64),
        (d_tr_id == "7").astype(np.float64),
        (s_tr_id == "7").astype(np.float64),
        np.isin(s_tr_id, ["1", "2", "3"]).astype(np.float64),
        np.isin(s_tr_id, ["4", "5", "6", "10", "22"]).astype(np.float64),
        np.isin(t_tr_id, ["1", "2"]).astype(np.float64),
        (t_tr_id == "3").astype(np.float64),
    ]
    u_ev_cols = [
        np.isin(d_ev_id, ["1", "8"]).astype(np.float64),
        (d_ev_id == "3").astype(np.float64),
        (d_ev_id == "6").astype(np.float64),
        np.isin(d_ev_id, ["2", "4", "5", "22", "23", "24"]).astype(np.float64),
        (d_ev_id == "7").astype(np.float64),
        (s_ev_id == "7").astype(np.float64),
        np.isin(s_ev_id, ["1", "2", "3"]).astype(np.float64),
        np.isin(s_ev_id, ["4", "5", "6", "10", "22"]).astype(np.float64),
        np.isin(t_ev_id, ["1", "2"]).astype(np.float64),
        (t_ev_id == "3").astype(np.float64),
    ]
    U_train = np.column_stack(u_tr_cols)
    U_eval = np.column_stack(u_ev_cols)

    # Low-rank factor extraction per individual mode (Unsupervised Baseline)
    k_d = min(4, D_train.shape[1])
    k_m = min(3, M_train.shape[1])
    k_u = min(2, U_train.shape[1])

    svd_d = TruncatedSVD(n_components=k_d, random_state=42)
    zd_tr = svd_d.fit_transform(D_train)
    zd_ev = svd_d.transform(D_eval)
    for k in range(k_d):
        new_num_train[f"latent_disease_{k}"] = zd_tr[:, k]
        new_num_eval[f"latent_disease_{k}"] = zd_ev[:, k]

    svd_m = TruncatedSVD(n_components=k_m, random_state=42)
    zm_tr = svd_m.fit_transform(M_train)
    zm_ev = svd_m.transform(M_eval)
    for k in range(k_m):
        new_num_train[f"latent_treatment_{k}"] = zm_tr[:, k]
        new_num_eval[f"latent_treatment_{k}"] = zm_ev[:, k]

    svd_u = TruncatedSVD(n_components=k_u, random_state=42)
    zu_tr = svd_u.fit_transform(U_train)
    zu_ev = svd_u.transform(U_eval)
    for k in range(k_u):
        new_num_train[f"latent_transition_{k}"] = zu_tr[:, k]
        new_num_eval[f"latent_transition_{k}"] = zu_ev[:, k]

    # Regularized Alternating Least Squares for 3-Way Canonical Polyadic Tensor Factorization
    def cp_als_3way(T_in, R=4, n_iter=8, reg=1e-3, seed=42):
        rng = np.random.RandomState(seed)
        I_dim, J_dim, K_dim = T_in.shape
        A = rng.randn(I_dim, R)
        A /= (np.linalg.norm(A, axis=0, keepdims=True) + 1e-8)
        B = rng.randn(J_dim, R)
        B /= (np.linalg.norm(B, axis=0, keepdims=True) + 1e-8)
        C = rng.randn(K_dim, R)
        C /= (np.linalg.norm(C, axis=0, keepdims=True) + 1e-8)

        for _ in range(n_iter):
            # Update A factor matrix
            G_bc = (B.T @ B) * (C.T @ C) + reg * np.eye(R)
            num_A = np.einsum("ijk,jr,kr->ir", T_in, B, C)
            A = num_A @ np.linalg.inv(G_bc)
            A /= (np.linalg.norm(A, axis=0, keepdims=True) + 1e-8)

            # Update B factor matrix
            G_ac = (A.T @ A) * (C.T @ C) + reg * np.eye(R)
            num_B = np.einsum("ijk,ir,kr->jr", T_in, A, C)
            B = num_B @ np.linalg.inv(G_ac)
            B /= (np.linalg.norm(B, axis=0, keepdims=True) + 1e-8)

            # Update C factor matrix
            G_ab = (A.T @ A) * (B.T @ B) + reg * np.eye(R)
            num_C = np.einsum("ijk,ir,jr->kr", T_in, A, B)
            C = num_C @ np.linalg.inv(G_ab)
            norm_c = np.linalg.norm(C, axis=0, keepdims=True) + 1e-8
            C /= norm_c

        return A, B, C

    # Out-of-fold Cross-Fitting of Supervised CP-ALS 3-Way Tensor & Bilinear SVD Models
    skf_tensor = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    R_cp = 4

    cp_3w_tr = np.zeros((n_tr, R_cp), dtype=np.float64)
    cp_dm_tr = np.zeros((n_tr, R_cp), dtype=np.float64)
    cp_du_tr = np.zeros((n_tr, R_cp), dtype=np.float64)
    cp_3w_ev = np.zeros((n_ev, R_cp), dtype=np.float64)
    cp_dm_ev = np.zeros((n_ev, R_cp), dtype=np.float64)
    cp_du_ev = np.zeros((n_ev, R_cp), dtype=np.float64)

    bilin_dm0_tr = np.zeros(n_tr, dtype=np.float64)
    bilin_dm1_tr = np.zeros(n_tr, dtype=np.float64)
    bilin_du0_tr = np.zeros(n_tr, dtype=np.float64)
    bilin_dm0_ev = np.zeros(n_ev, dtype=np.float64)
    bilin_dm1_ev = np.zeros(n_ev, dtype=np.float64)
    bilin_du0_ev = np.zeros(n_ev, dtype=np.float64)

    for tr_idx, val_idx in skf_tensor.split(D_train, y_train):
        D_f_tr, M_f_tr, U_f_tr = D_train[tr_idx], M_train[tr_idx], U_train[tr_idx]
        y_f_tr = y_train[tr_idx]
        y_cent = y_f_tr - float(np.mean(y_f_tr))
        n_f = len(y_f_tr)

        mu_d, std_d = np.mean(D_f_tr, axis=0), np.std(D_f_tr, axis=0) + 1e-4
        mu_m, std_m = np.mean(M_f_tr, axis=0), np.std(M_f_tr, axis=0) + 1e-4
        mu_u, std_u = np.mean(U_f_tr, axis=0), np.std(U_f_tr, axis=0) + 1e-4

        D_tr_s = (D_f_tr - mu_d) / std_d
        M_tr_s = (M_f_tr - mu_m) / std_m
        U_tr_s = (U_f_tr - mu_u) / std_u

        D_val_s = (D_train[val_idx] - mu_d) / std_d
        M_val_s = (M_train[val_idx] - mu_m) / std_m
        U_val_s = (U_train[val_idx] - mu_u) / std_u

        D_ev_s = (D_eval - mu_d) / std_d
        M_ev_s = (M_eval - mu_m) / std_m
        U_ev_s = (U_eval - mu_u) / std_u

        # Compute supervised risk-weighted 3-way interaction tensor
        T_risk = np.einsum("i,ia,ib,ic->abc", y_cent, D_tr_s, M_tr_s, U_tr_s) / n_f
        A_cp, B_cp, C_cp = cp_als_3way(T_risk, R=R_cp, n_iter=8, reg=1e-3, seed=42)

        # Tri-linear and bi-linear tensor projections on validation fold
        alpha_v = D_val_s @ A_cp
        beta_v = M_val_s @ B_cp
        gamma_v = U_val_s @ C_cp

        cp_3w_tr[val_idx] = alpha_v * beta_v * gamma_v
        cp_dm_tr[val_idx] = alpha_v * beta_v
        cp_du_tr[val_idx] = alpha_v * gamma_v

        # Accumulate out-of-fold evaluation projections
        alpha_e = D_ev_s @ A_cp
        beta_e = M_ev_s @ B_cp
        gamma_e = U_ev_s @ C_cp

        cp_3w_ev += (alpha_e * beta_e * gamma_e) / 5.0
        cp_dm_ev += (alpha_e * beta_e) / 5.0
        cp_du_ev += (alpha_e * gamma_e) / 5.0

        # Coupled 2-Way Bilinear Risk Cross-Covariance via SVD
        B_dm = (D_tr_s.T @ (y_cent[:, None] * M_tr_s)) / n_f
        u_dm, _, vt_dm = np.linalg.svd(B_dm, full_matrices=False)
        v_dm = vt_dm.T

        bilin_dm0_tr[val_idx] = (D_val_s @ u_dm[:, 0]) * (M_val_s @ v_dm[:, 0])
        bilin_dm1_tr[val_idx] = (D_val_s @ u_dm[:, 1]) * (M_val_s @ v_dm[:, 1])
        bilin_dm0_ev += ((D_ev_s @ u_dm[:, 0]) * (M_ev_s @ v_dm[:, 0])) / 5.0
        bilin_dm1_ev += ((D_ev_s @ u_dm[:, 1]) * (M_ev_s @ v_dm[:, 1])) / 5.0

        B_du = (D_tr_s.T @ (y_cent[:, None] * U_tr_s)) / n_f
        u_du, _, vt_du = np.linalg.svd(B_du, full_matrices=False)
        v_du = vt_du.T

        bilin_du0_tr[val_idx] = (D_val_s @ u_du[:, 0]) * (U_val_s @ v_du[:, 0])
        bilin_du0_ev += ((D_ev_s @ u_du[:, 0]) * (U_ev_s @ v_du[:, 0])) / 5.0

    for r in range(R_cp):
        new_num_train[f"tensor_cp_3w_{r}"] = cp_3w_tr[:, r]
        new_num_eval[f"tensor_cp_3w_{r}"] = cp_3w_ev[:, r]
        new_num_train[f"tensor_cp_dm_{r}"] = cp_dm_tr[:, r]
        new_num_eval[f"tensor_cp_dm_{r}"] = cp_dm_ev[:, r]
        new_num_train[f"tensor_cp_du_{r}"] = cp_du_tr[:, r]
        new_num_eval[f"tensor_cp_du_{r}"] = cp_du_ev[:, r]

    new_num_train["bilinear_risk_dm_0"] = bilin_dm0_tr
    new_num_eval["bilinear_risk_dm_0"] = bilin_dm0_ev
    new_num_train["bilinear_risk_dm_1"] = bilin_dm1_tr
    new_num_eval["bilinear_risk_dm_1"] = bilin_dm1_ev
    new_num_train["bilinear_risk_du_0"] = bilin_du0_tr
    new_num_eval["bilinear_risk_du_0"] = bilin_du0_ev

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

    diag_pair_tr = np.array([f"{a}_{b}" for a, b in zip(new_cat_train["diag_1_prefix"], new_cat_train["diag_2_prefix"])], dtype=object)
    diag_pair_ev = np.array([f"{a}_{b}" for a, b in zip(new_cat_eval["diag_1_prefix"], new_cat_eval["diag_2_prefix"])], dtype=object)

    age_disp_tr = np.array([f"{a}_{d}" for a, d in zip(train["age"], train["discharge_disposition_id"])], dtype=object)
    age_disp_ev = np.array([f"{a}_{d}" for a, d in zip(eval_cols["age"], eval_cols["discharge_disposition_id"])], dtype=object)

    new_cat_train["disp_adm"] = disp_adm
    new_cat_eval["disp_adm"] = disp_adm_ev
    new_cat_train["diag_disp"] = diag_disp
    new_cat_eval["diag_disp"] = diag_disp_ev
    new_cat_train["diag_pair_1_2"] = diag_pair_tr
    new_cat_eval["diag_pair_1_2"] = diag_pair_ev
    new_cat_train["age_disch"] = age_disp_tr
    new_cat_eval["age_disch"] = age_disp_ev

    te_cols = [
        ("discharge_disposition_id", train["discharge_disposition_id"], eval_cols["discharge_disposition_id"]),
        ("admission_source_id", train["admission_source_id"], eval_cols["admission_source_id"]),
        ("admission_type_id", train["admission_type_id"], eval_cols["admission_type_id"]),
        ("medical_specialty", train["medical_specialty"], eval_cols["medical_specialty"]),
        ("diag_1", train["diag_1"], eval_cols["diag_1"]),
        ("diag_2", train["diag_2"], eval_cols["diag_2"]),
        ("diag_3", train["diag_3"], eval_cols["diag_3"]),
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
        ("diag_pair_1_2", diag_pair_tr, diag_pair_ev),
        ("age_disch", age_disp_tr, age_disp_ev),
        ("grp_discharge", grp_disp_tr, grp_disp_ev),
        ("grp_adm_source", grp_src_tr, grp_src_ev),
        ("grp_adm_type", grp_type_tr, grp_type_ev),
        ("grp_disp_adm", grp_disp_adm_tr, grp_disp_adm_ev),
    ]
    for col_name, arr_tr, arr_ev in te_cols:
        te_tr, te_ev = compute_bayes_target_encoding(arr_tr, arr_ev, y_train, m_smooth=20.0)
        new_num_train[f"te_{col_name}"] = te_tr
        new_num_eval[f"te_{col_name}"] = te_ev

    # Diagnosis target encoding summary metrics across all 3 diagnoses
    tr_d1, tr_d2, tr_d3 = new_num_train["te_diag_1_prefix"], new_num_train["te_diag_2_prefix"], new_num_train["te_diag_3_prefix"]
    ev_d1, ev_d2, ev_d3 = new_num_eval["te_diag_1_prefix"], new_num_eval["te_diag_2_prefix"], new_num_eval["te_diag_3_prefix"]
    new_num_train["te_diag_prefix_max"] = np.maximum(tr_d1, np.maximum(tr_d2, tr_d3))
    new_num_train["te_diag_prefix_mean"] = (tr_d1 + tr_d2 + tr_d3) / 3.0
    new_num_train["te_diag_prefix_min"] = np.minimum(tr_d1, np.minimum(tr_d2, tr_d3))
    new_num_eval["te_diag_prefix_max"] = np.maximum(ev_d1, np.maximum(ev_d2, ev_d3))
    new_num_eval["te_diag_prefix_mean"] = (ev_d1 + ev_d2 + ev_d3) / 3.0
    new_num_eval["te_diag_prefix_min"] = np.minimum(ev_d1, np.minimum(ev_d2, ev_d3))

    raw_d1, raw_d2, raw_d3 = new_num_train["te_diag_1"], new_num_train["te_diag_2"], new_num_train["te_diag_3"]
    raw_ev_d1, raw_ev_d2, raw_ev_d3 = new_num_eval["te_diag_1"], new_num_eval["te_diag_2"], new_num_eval["te_diag_3"]
    new_num_train["te_diag_max"] = np.maximum(raw_d1, np.maximum(raw_d2, raw_d3))
    new_num_train["te_diag_mean"] = (raw_d1 + raw_d2 + raw_d3) / 3.0
    new_num_train["te_diag_min"] = np.minimum(raw_d1, np.minimum(raw_d2, raw_d3))
    new_num_eval["te_diag_max"] = np.maximum(raw_ev_d1, np.maximum(raw_ev_d2, raw_ev_d3))
    new_num_eval["te_diag_mean"] = (raw_ev_d1 + raw_ev_d2 + raw_ev_d3) / 3.0
    new_num_eval["te_diag_min"] = np.minimum(raw_ev_d1, np.minimum(raw_ev_d2, raw_ev_d3))

    new_num_train["disp_x_inp"] = new_num_train["te_discharge_disposition_id"] * new_num_train["log1p_inpatient"]
    new_num_eval["disp_x_inp"] = new_num_eval["te_discharge_disposition_id"] * new_num_eval["log1p_inpatient"]
    new_num_train["disp_x_lace"] = new_num_train["te_discharge_disposition_id"] * new_num_train["lace_proxy"]
    new_num_eval["disp_x_lace"] = new_num_eval["te_discharge_disposition_id"] * new_num_eval["lace_proxy"]

    # Continuous Hopfield Energy Landscape & Attractor Dynamics (CHEL-AD)
    hop_feats = [
        "log1p_inpatient",
        "log1p_emergency",
        "utilization_score",
        "lace_proxy",
        "num_chronic_conditions",
        "num_changed_meds",
        "insulin_active",
        "comp_chf",
        "comp_ckd",
        "te_discharge_disposition_id",
        "te_diag_max",
    ]
    Z_tr_raw = np.column_stack([np.nan_to_num(new_num_train[f], nan=0.0) for f in hop_feats])
    Z_ev_raw = np.column_stack([np.nan_to_num(new_num_eval[f], nan=0.0) for f in hop_feats])

    # Ising Hamiltonian Clinical Stress Features
    B_tr = np.column_stack([
        (np.nan_to_num(train["time_in_hospital"], nan=1.0) >= 4).astype(np.float64),
        new_num_train["has_inpatient"],
        (new_num_train["comp_chf"] > 0).astype(np.float64),
        (new_num_train["comp_ckd"] > 0).astype(np.float64),
        (new_num_train["num_changed_meds"] > 0).astype(np.float64),
        (new_num_train["discharged_high_risk"] > 0).astype(np.float64),
        (age_tr >= 70.0).astype(np.float64),
        new_num_train["a1c_high"],
    ])
    B_ev = np.column_stack([
        (np.nan_to_num(eval_cols["time_in_hospital"], nan=1.0) >= 4).astype(np.float64),
        new_num_eval["has_inpatient"],
        (new_num_eval["comp_chf"] > 0).astype(np.float64),
        (new_num_eval["comp_ckd"] > 0).astype(np.float64),
        (new_num_eval["num_changed_meds"] > 0).astype(np.float64),
        (new_num_eval["discharged_high_risk"] > 0).astype(np.float64),
        (age_ev >= 70.0).astype(np.float64),
        new_num_eval["a1c_high"],
    ])
    S_tr = 2.0 * B_tr - 1.0
    S_ev = 2.0 * B_ev - 1.0
    J_mat = np.corrcoef(S_tr, rowvar=False)
    J_mat = np.nan_to_num(J_mat, nan=0.0)
    np.fill_diagonal(J_mat, 0.0)

    new_num_train["ising_energy"] = -0.5 * np.sum((S_tr @ J_mat) * S_tr, axis=1)
    new_num_eval["ising_energy"] = -0.5 * np.sum((S_ev @ J_mat) * S_ev, axis=1)

    # Out-of-fold Modern Hopfield Projections
    skf_hop = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    hop_delta_e_tr = np.zeros(n_tr, dtype=np.float64)
    hop_delta_align_tr = np.zeros(n_tr, dtype=np.float64)
    hop_entropy_pos_tr = np.zeros(n_tr, dtype=np.float64)

    hop_delta_e_ev = np.zeros(n_ev, dtype=np.float64)
    hop_delta_align_ev = np.zeros(n_ev, dtype=np.float64)
    hop_entropy_pos_ev = np.zeros(n_ev, dtype=np.float64)

    K_hop = 8
    beta_hop = 6.0

    for tr_idx, val_idx in skf_hop.split(Z_tr_raw, y_train):
        mu_fold = np.mean(Z_tr_raw[tr_idx], axis=0)
        std_fold = np.std(Z_tr_raw[tr_idx], axis=0) + 1e-6

        Z_tr_scaled = (Z_tr_raw[tr_idx] - mu_fold) / std_fold
        Z_val_scaled = (Z_tr_raw[val_idx] - mu_fold) / std_fold
        Z_ev_scaled = (Z_ev_raw - mu_fold) / std_fold

        Z_tr_norm = Z_tr_scaled / (np.linalg.norm(Z_tr_scaled, axis=1, keepdims=True) + 1e-8)
        Z_val_norm = Z_val_scaled / (np.linalg.norm(Z_val_scaled, axis=1, keepdims=True) + 1e-8)
        Z_ev_norm = Z_ev_scaled / (np.linalg.norm(Z_ev_scaled, axis=1, keepdims=True) + 1e-8)

        pos_mask = (y_train[tr_idx] == 1)
        neg_mask = (y_train[tr_idx] == 0)

        km_pos = KMeans(n_clusters=K_hop, n_init=2, max_iter=30, random_state=42)
        km_pos.fit(Z_tr_norm[pos_mask])
        proto_pos = km_pos.cluster_centers_
        proto_pos /= (np.linalg.norm(proto_pos, axis=1, keepdims=True) + 1e-8)

        km_neg = KMeans(n_clusters=K_hop, n_init=2, max_iter=30, random_state=42)
        km_neg.fit(Z_tr_norm[neg_mask])
        proto_neg = km_neg.cluster_centers_
        proto_neg /= (np.linalg.norm(proto_neg, axis=1, keepdims=True) + 1e-8)

        def eval_hopfield(Z_query):
            sim_pos = Z_query @ proto_pos.T
            sim_neg = Z_query @ proto_neg.T

            E_pos = -(1.0 / beta_hop) * logsumexp(beta_hop * sim_pos, axis=1)
            E_neg = -(1.0 / beta_hop) * logsumexp(beta_hop * sim_neg, axis=1)
            delta_e = E_neg - E_pos

            W_pos = softmax(beta_hop * sim_pos, axis=1)
            W_neg = softmax(beta_hop * sim_neg, axis=1)

            recon_pos = W_pos @ proto_pos
            recon_neg = W_neg @ proto_neg

            align_pos = np.sum(Z_query * recon_pos, axis=1)
            align_neg = np.sum(Z_query * recon_neg, axis=1)
            delta_align = align_pos - align_neg

            H_pos = -np.sum(W_pos * np.log(W_pos + 1e-12), axis=1)
            return delta_e, delta_align, H_pos

        de_v, da_v, hp_v = eval_hopfield(Z_val_norm)
        hop_delta_e_tr[val_idx] = de_v
        hop_delta_align_tr[val_idx] = da_v
        hop_entropy_pos_tr[val_idx] = hp_v

        de_ev, da_ev, hp_ev = eval_hopfield(Z_ev_norm)
        hop_delta_e_ev += de_ev / 5.0
        hop_delta_align_ev += da_ev / 5.0
        hop_entropy_pos_ev += hp_ev / 5.0

    new_num_train["hopfield_energy_differential"] = hop_delta_e_tr
    new_num_eval["hopfield_energy_differential"] = hop_delta_e_ev
    new_num_train["hopfield_alignment_differential"] = hop_delta_align_tr
    new_num_eval["hopfield_alignment_differential"] = hop_delta_align_ev
    new_num_train["hopfield_entropy_pos"] = hop_entropy_pos_tr
    new_num_eval["hopfield_entropy_pos"] = hop_entropy_pos_ev

    tr_g1, tr_g2, tr_g3 = new_num_train["te_diag_1_group"], new_num_train["te_diag_2_group"], new_num_train["te_diag_3_group"]
    ev_g1, ev_g2, ev_g3 = new_num_eval["te_diag_1_group"], new_num_eval["te_diag_2_group"], new_num_eval["te_diag_3_group"]
    new_num_train["te_diag_grp_max"] = np.maximum(tr_g1, np.maximum(tr_g2, tr_g3))
    new_num_eval["te_diag_grp_max"] = np.maximum(ev_g1, np.maximum(ev_g2, ev_g3))

    # Register all derived features
    for k, v in new_num_train.items():
        train_ext[k] = v
        eval_ext[k] = new_num_eval[k]
    for k, v in new_cat_train.items():
        train_ext[k] = v
        eval_ext[k] = new_cat_eval[k]

    extra_numeric = list(new_num_train.keys())
    extra_categorical = list(new_cat_train.keys())

    encoder = TabularEncoder(min_count=15)
    x_train = encoder.fit_transform(task, train_ext, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_eval = encoder.transform(eval_ext)

    # 7. Stratified 5-Fold Cross-Validated Stacking & AUC-Optimal Weight Optimization
    from scipy.optimize import minimize
    from sklearn.metrics import roc_auc_score

    skf_model = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_factories = [
        # Model 1: LGBM Leaf-wise
        ("lgb_leaf", lambda s: lgb.LGBMClassifier(
            n_estimators=210,
            learning_rate=0.042,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_lambda=5.0,
            reg_alpha=0.5,
            random_state=s,
            n_jobs=1,
            verbose=-1,
        )),
        # Model 2: LGBM Depth-wise / Regularized
        ("lgb_depth", lambda s: lgb.LGBMClassifier(
            n_estimators=230,
            learning_rate=0.038,
            max_depth=5,
            num_leaves=22,
            min_child_samples=50,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=8.0,
            reg_alpha=1.0,
            random_state=s,
            n_jobs=1,
            verbose=-1,
        )),
        # Model 3: LGBM Extra Trees
        ("lgb_extra", lambda s: lgb.LGBMClassifier(
            n_estimators=190,
            learning_rate=0.042,
            num_leaves=35,
            min_child_samples=45,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.6,
            extra_trees=True,
            reg_lambda=10.0,
            reg_alpha=2.0,
            random_state=s,
            n_jobs=1,
            verbose=-1,
        )),
        # Model 4: LGBM Shallow / Regularized
        ("lgb_shallow", lambda s: lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.038,
            max_depth=4,
            num_leaves=15,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=8.0,
            reg_alpha=1.0,
            random_state=s,
            n_jobs=1,
            verbose=-1,
        )),
    ]

    try:
        xgb.XGBClassifier()
        model_factories.append(
            ("xgb_hist", lambda s: xgb.XGBClassifier(
                n_estimators=170,
                learning_rate=0.042,
                max_depth=4,
                min_child_weight=25,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_lambda=6.0,
                reg_alpha=1.0,
                tree_method="hist",
                random_state=s,
                n_jobs=1,
                eval_metric="logloss",
            ))
        )
    except Exception:
        pass

    n_models = len(model_factories)
    oof_preds = np.zeros((n_tr, n_models), dtype=np.float64)
    eval_preds = np.zeros((n_ev, n_models), dtype=np.float64)

    for m_idx, (m_name, m_ctor) in enumerate(model_factories):
        is_lgb = m_name.startswith("lgb")
        for fold, (tr_idx, val_idx) in enumerate(skf_model.split(x_train, y_train)):
            model = m_ctor(42 + fold * 13 + m_idx * 101)
            X_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
            X_va = x_train[val_idx]
            if is_lgb:
                model.fit(X_tr, y_tr, categorical_feature=encoder.categorical_indices)
            else:
                model.fit(X_tr, y_tr)
            oof_preds[val_idx, m_idx] = model.predict_proba(X_va)[:, 1]
            eval_preds[:, m_idx] += model.predict_proba(x_eval)[:, 1] / 5.0

    from scipy.stats import rankdata

    # Map model predictions to uniform rank percentiles to eliminate calibration variance distortion
    oof_ranks = np.column_stack([rankdata(oof_preds[:, m]) / n_tr for m in range(n_models)])
    eval_ranks = np.column_stack([rankdata(eval_preds[:, m]) / n_ev for m in range(n_models)])

    # Multi-point initialization across candidate strategies
    candidates = []
    # 1. Uniform baseline
    candidates.append(np.ones(n_models, dtype=np.float64) / n_models)
    # 2. Individual single models
    for m in range(n_models):
        e_m = np.zeros(n_models, dtype=np.float64)
        e_m[m] = 1.0
        candidates.append(e_m)
    # 3. Supervised AUC-power weights
    individual_aucs = np.array([roc_auc_score(y_train, oof_preds[:, m]) for m in range(n_models)])
    auc_diff = np.maximum(individual_aucs - 0.5, 0.001)
    w_power = (auc_diff ** 3) / np.sum(auc_diff ** 3)
    candidates.append(w_power)

    def eval_auc(w, preds_mat):
        w_p = np.maximum(w, 0.0)
        s_val = np.sum(w_p)
        if s_val <= 0:
            return 0.5
        return roc_auc_score(y_train, preds_mat @ (w_p / s_val))

    best_init_w = candidates[0]
    best_score = eval_auc(best_init_w, oof_ranks)
    for c in candidates[1:]:
        score = eval_auc(c, oof_ranks)
        if score > best_score:
            best_score = score
            best_init_w = c

    # Local Nelder-Mead optimization starting from the globally best simplex candidate
    def neg_auc_obj(w):
        return -eval_auc(w, oof_ranks)

    res = minimize(neg_auc_obj, best_init_w, method="Nelder-Mead", options={"maxiter": 100})
    opt_w = np.maximum(res.x, 0.0)
    s_opt = np.sum(opt_w)
    if s_opt > 0:
        opt_w /= s_opt
    else:
        opt_w = best_init_w

    # Regularized shrinkage towards uniform weights to guard against out-of-fold overfitting
    w_final = 0.80 * opt_w + 0.20 * (np.ones(n_models, dtype=np.float64) / n_models)

    # Dual-Space Consensus Blending: Combine calibrated probability blend and rank-consensus blend
    p_blend_ev = eval_preds @ w_final
    r_blend_ev = eval_ranks @ w_final

    p_blend_rank = rankdata(p_blend_ev) / n_ev
    preds = 0.50 * p_blend_rank + 0.50 * r_blend_ev

    preds = np.clip(preds, 0.0, 1.0)
    preds = np.nan_to_num(preds, nan=float(np.mean(y_train)))
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
