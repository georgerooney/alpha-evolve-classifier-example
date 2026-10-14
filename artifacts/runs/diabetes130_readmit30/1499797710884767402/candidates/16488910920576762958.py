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
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from scipy.optimize import nnls
    from scipy.stats import rankdata

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

    # Quantum Information & Ginzburg-Landau Thermodynamics Helpers
    def compute_quantum_entanglement(m1, m2, m3, m4):
        V = np.column_stack([m1, m2, m3, m4]).astype(np.float64)
        norm = np.linalg.norm(V, axis=1, keepdims=True) + 1e-7
        U = V / norm

        a = U[:, 0] ** 2 + U[:, 1] ** 2
        b = U[:, 0] * U[:, 2] + U[:, 1] * U[:, 3]
        c = U[:, 2] ** 2 + U[:, 3] ** 2

        tau = a + c
        delta = np.maximum(a * c - b ** 2, 0.0)

        disc = np.maximum(tau ** 2 - 4.0 * delta, 0.0)
        sqrt_disc = np.sqrt(disc)
        l1 = np.clip((tau + sqrt_disc) / 2.0, 1e-12, 1.0)
        l2 = np.clip((tau - sqrt_disc) / 2.0, 1e-12, 1.0)

        vn_entropy = -(l1 * np.log2(l1) + l2 * np.log2(l2))
        purity = l1 ** 2 + l2 ** 2
        concurrence = 2.0 * np.sqrt(delta)
        return vn_entropy, purity, concurrence

    def compute_ginzburg_landau(t, h):
        Q = t / 3.0
        R = h / 2.0
        D = Q ** 3 + R ** 2

        psi = np.zeros_like(t, dtype=np.float64)

        mask_pos = D > 0
        if np.any(mask_pos):
            sqrt_D = np.sqrt(np.maximum(D[mask_pos], 0.0))
            u = np.cbrt(R[mask_pos] + sqrt_D)
            v = np.cbrt(R[mask_pos] - sqrt_D)
            psi[mask_pos] = u + v

        mask_neg = ~mask_pos
        if np.any(mask_neg):
            Q_neg = Q[mask_neg]
            R_neg = R[mask_neg]
            neg_Q3 = np.maximum(-(Q_neg ** 3), 1e-12)
            val = np.clip(R_neg / np.sqrt(neg_Q3), -1.0, 1.0)
            theta = np.arccos(val)
            psi[mask_neg] = 2.0 * np.sqrt(np.maximum(-Q_neg, 0.0)) * np.cos(theta / 3.0)

        f_star = 0.5 * t * (psi ** 2) + 0.25 * (psi ** 4) - h * psi
        chi = 1.0 / (np.maximum(t + 3.0 * (psi ** 2), 0.01) + 1e-4)
        fluc_entropy = chi * np.abs(t)
        return psi, f_star, chi, fluc_entropy

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

    new_num_train["comp_dysrhythmia"] = count_comp_starts_with(["427"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_dysrhythmia"] = count_comp_starts_with(["427"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_pneumonia"] = count_comp_starts_with(["480", "481", "482", "483", "484", "485", "486"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_pneumonia"] = count_comp_starts_with(["480", "481", "482", "483", "484", "485", "486"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_aki"] = count_comp_starts_with(["584"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_aki"] = count_comp_starts_with(["584"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_uti"] = count_comp_starts_with(["599"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_uti"] = count_comp_starts_with(["599"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_sepsis"] = count_comp_starts_with(["038", "995"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_sepsis"] = count_comp_starts_with(["038", "995"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_cellulitis"] = count_comp_starts_with(["681", "682"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_cellulitis"] = count_comp_starts_with(["681", "682"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    new_num_train["comp_anemia"] = count_comp_starts_with(["280", "281", "282", "283", "284", "285"], d1_raw, d2_raw, d3_raw)
    new_num_eval["comp_anemia"] = count_comp_starts_with(["280", "281", "282", "283", "284", "285"], d1_raw_ev, d2_raw_ev, d3_raw_ev)

    def compute_charlson(d1, d2, d3):
        score = np.zeros(len(d1), dtype=np.float64)
        for d in [d1, d2, d3]:
            for i, c in enumerate(d):
                if c is None or (isinstance(c, float) and np.isnan(c)):
                    continue
                s = str(c).strip()
                if not s or s == "?":
                    continue
                pfx = s.split(".")[0]
                if pfx in ("410", "412"):
                    score[i] += 1.0
                elif pfx in ("428", "398", "402", "404"):
                    score[i] += 1.0
                elif pfx in ("440", "441", "442", "443", "707"):
                    score[i] += 1.0
                elif pfx.isdigit() and 430 <= int(pfx) <= 438:
                    score[i] += 1.0
                elif pfx == "290":
                    score[i] += 1.0
                elif (pfx.isdigit() and 490 <= int(pfx) <= 496) or pfx == "518":
                    score[i] += 1.0
                elif pfx in ("710", "714", "725"):
                    score[i] += 1.0
                elif pfx in ("531", "532", "533", "534"):
                    score[i] += 1.0
                elif pfx == "571":
                    score[i] += 1.0
                elif pfx in ("342", "344"):
                    score[i] += 2.0
                elif pfx in ("582", "583", "585", "586"):
                    score[i] += 2.0
                elif pfx.isdigit() and (140 <= int(pfx) <= 195 or 200 <= int(pfx) <= 208):
                    score[i] += 2.0
                elif pfx.isdigit() and 196 <= int(pfx) <= 199:
                    score[i] += 6.0
        return score

    new_num_train["charlson_score"] = compute_charlson(d1_raw, d2_raw, d3_raw)
    new_num_eval["charlson_score"] = compute_charlson(d1_raw_ev, d2_raw_ev, d3_raw_ev)

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
            + (num_dict["comp_dysrhythmia"] > 0).astype(np.float64)
            + (num_dict["comp_pneumonia"] > 0).astype(np.float64)
            + (num_dict["comp_aki"] > 0).astype(np.float64)
            + (num_dict["comp_uti"] > 0).astype(np.float64)
            + (num_dict["comp_sepsis"] > 0).astype(np.float64)
            + (num_dict["comp_cellulitis"] > 0).astype(np.float64)
            + (num_dict["comp_anemia"] > 0).astype(np.float64)
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

        # Discharge and admission disposition indicators (resolved before dependencies)
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

        # Advanced syndrome and interaction indicators
        n_lab_val = np.nan_to_num(data_dict["num_lab_procedures"], nan=0.0).astype(np.float64)
        n_med_val = np.nan_to_num(data_dict["num_medications"], nan=0.0).astype(np.float64)
        dur_val = np.maximum(np.nan_to_num(data_dict["time_in_hospital"], nan=1.0).astype(np.float64), 1.0)

        num_dict["crm_severity"] = (
            (num_dict["comp_chf"] > 0).astype(np.float64) * 2.0
            + (num_dict["comp_ckd"] > 0).astype(np.float64) * 2.0
            + (num_dict["has_comp_diabetes"] > 0).astype(np.float64) * 1.5
            + (num_dict["comp_hypertension"] > 0).astype(np.float64) * 1.0
            + (num_dict["comp_aki"] > 0).astype(np.float64) * 1.5
        )
        num_dict["acuity_discharge"] = (
            (n_lab_val > 50.0).astype(np.float64)
            * ((num_dict["discharged_to_snf"] > 0) | (num_dict["discharged_to_home_health"] > 0)).astype(np.float64)
        )
        num_dict["snf_elderly"] = (num_dict["discharged_to_snf"] * (age_arr_ref >= 75.0)).astype(np.float64)
        num_dict["home_health_elderly"] = (num_dict["discharged_to_home_health"] * (age_arr_ref >= 70.0)).astype(np.float64)
        num_dict["emergency_repeat"] = num_dict["admit_from_emergency"] * num_dict["log1p_emergency"]
        num_dict["lab_to_med_ratio"] = n_lab_val / (n_med_val + 1.0)
        num_dict["med_regimen_instability"] = num_dict["num_changed_meds"] / (num_dict["num_active_meds"] + 1.0)
        num_dict["insulin_escalation"] = num_dict["insulin_up"] * num_dict["a1c_high"]

        # Quantum Information Entanglement across Multi-Organ Axes
        m_meta = (num_dict["has_diabetes_diag"] + num_dict["a1c_high"] + num_dict["insulin_active"] + 0.5 * num_dict["has_comp_diabetes"])
        m_card = (num_dict["comp_chf"] + num_dict["comp_ihd"] + num_dict["comp_hypertension"] + num_dict["comp_dysrhythmia"])
        m_rena = (num_dict["comp_ckd"] + num_dict["comp_renal"] + num_dict["comp_aki"] + num_dict["comp_fluid"] + num_dict["comp_copd"])
        m_inst = (num_dict["num_changed_meds"] + num_dict["log1p_inpatient"] * 1.5 + num_dict["discharged_high_risk"] * 1.5)
        q_ent, q_pur, q_con = compute_quantum_entanglement(m_meta, m_card, m_rena, m_inst)
        num_dict["quantum_von_neumann_entropy"] = q_ent
        num_dict["quantum_state_purity"] = q_pur
        num_dict["quantum_concurrence"] = q_con

        # Ginzburg-Landau Critical Phase Transition
        gl_temp = (age_arr_ref / 50.0) + (n_lab_val / 40.0) + (dur_val / 5.0)
        gl_t = gl_temp - 3.2
        gl_h = (
            0.5 * num_dict["log1p_inpatient"]
            + 0.3 * num_dict["log1p_emergency"]
            + 0.3 * num_dict["discharged_high_risk"]
            + 0.2 * num_dict["admit_from_emergency"]
        )
        gl_psi, gl_f, gl_chi, gl_fluc = compute_ginzburg_landau(gl_t, gl_h)
        num_dict["gl_order_parameter"] = gl_psi
        num_dict["gl_free_energy"] = gl_f
        num_dict["gl_susceptibility"] = gl_chi
        num_dict["gl_fluctuation_entropy"] = gl_fluc

        # Symplectic Hamiltonian Action & Phase Coordinates
        q_coord = dur_val / 7.0
        p_coord = (n_lab_val + 2.0 * n_med_val) / 50.0
        num_dict["hamiltonian_energy"] = 0.5 * (p_coord ** 2 + q_coord ** 2)
        num_dict["hamiltonian_action"] = p_coord * q_coord
        num_dict["hamiltonian_phase"] = np.arctan2(p_coord, q_coord)

    # 5. Tensor Decomposition: Structured Multi-Mode Factorization & CP Bilinear Coupling
    # Mode 1: Diagnosis Space (ICD-9 Chapters + Top Prefixes)
    chapter_list = ["circulatory", "respiratory", "digestive", "diabetes", "injury", "musculoskeletal", "genitourinary", "neoplasms", "other"]
    D_tr_list = [((d1_grp_tr == grp).astype(np.float64) + (d2_grp_tr == grp).astype(np.float64) + (d3_grp_tr == grp).astype(np.float64)) for grp in chapter_list]
    D_ev_list = [((d1_grp_ev == grp).astype(np.float64) + (d2_grp_ev == grp).astype(np.float64) + (d3_grp_ev == grp).astype(np.float64)) for grp in chapter_list]

    p1_tr, p2_tr, p3_tr = new_cat_train["diag_1_prefix"], new_cat_train["diag_2_prefix"], new_cat_train["diag_3_prefix"]
    p1_ev, p2_ev, p3_ev = new_cat_eval["diag_1_prefix"], new_cat_eval["diag_2_prefix"], new_cat_eval["diag_3_prefix"]
    counts = Counter([p for p in np.concatenate([p1_tr, p2_tr, p3_tr]) if p != "missing"])
    top_prefixes = [p for p, _ in counts.most_common(25)]
    if top_prefixes:
        D_tr_list.extend([((p1_tr == p) | (p2_tr == p) | (p3_tr == p)).astype(np.float64) for p in top_prefixes])
        D_ev_list.extend([((p1_ev == p) | (p2_ev == p) | (p3_ev == p)).astype(np.float64) for p in top_prefixes])
    D_train = np.column_stack(D_tr_list)
    D_eval = np.column_stack(D_ev_list)

    # Mode 2: Pharmacotherapy Space (Active Medications)
    if available_meds:
        M_train = np.column_stack([((train[m] != "No") & (train[m] != None)).astype(np.float64) for m in available_meds])  # noqa: E711
        M_eval = np.column_stack([((eval_cols[m] != "No") & (eval_cols[m] != None)).astype(np.float64) for m in available_meds])  # noqa: E711
    else:
        M_train = np.zeros((n_tr, 1), dtype=np.float64)
        M_eval = np.zeros((n_ev, 1), dtype=np.float64)

    # Mode 3: Care Transition Space (Discharge Disposition & Admission Source)
    top_disp = ["1", "3", "6", "2", "7", "18", "22", "25"]
    top_src = ["7", "1", "2", "4", "5", "6"]
    U_train = np.column_stack(
        [(train["discharge_disposition_id"] == v).astype(np.float64) for v in top_disp] +
        [(train["admission_source_id"] == v).astype(np.float64) for v in top_src]
    )
    U_eval = np.column_stack(
        [(eval_cols["discharge_disposition_id"] == v).astype(np.float64) for v in top_disp] +
        [(eval_cols["admission_source_id"] == v).astype(np.float64) for v in top_src]
    )

    # Low-rank factor extraction per mode
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

    # Canonical Polyadic (CP) rank-1 disease-treatment coupling and disease-acuity coupling
    for k in range(min(k_d, k_m)):
        new_num_train[f"tensor_cp_dm_{k}"] = zd_tr[:, k] * zm_tr[:, k]
        new_num_eval[f"tensor_cp_dm_{k}"] = zd_ev[:, k] * zm_ev[:, k]
    new_num_train["tensor_cp_du_0"] = zd_tr[:, 0] * zu_tr[:, 0]
    new_num_eval["tensor_cp_du_0"] = zd_ev[:, 0] * zu_ev[:, 0]

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

    # Frequency (prevalence) encoding for high-cardinality categoricals
    def compute_freq_encoding(tr_arr, ev_arr):
        s_tr = [str(x) if x is not None else "missing" for x in tr_arr]
        s_ev = [str(x) if x is not None else "missing" for x in ev_arr]
        c = Counter(s_tr)
        n = float(len(s_tr))
        return (
            np.array([c.get(x, 0) / n for x in s_tr], dtype=np.float64),
            np.array([c.get(x, 0) / n for x in s_ev], dtype=np.float64),
        )

    for col_key, tr_val, ev_val in [
        ("diag_1_prefix", new_cat_train["diag_1_prefix"], new_cat_eval["diag_1_prefix"]),
        ("diag_2_prefix", new_cat_train["diag_2_prefix"], new_cat_eval["diag_2_prefix"]),
        ("diag_3_prefix", new_cat_train["diag_3_prefix"], new_cat_eval["diag_3_prefix"]),
        ("discharge_disposition_id", train["discharge_disposition_id"], eval_cols["discharge_disposition_id"]),
        ("admission_source_id", train["admission_source_id"], eval_cols["admission_source_id"]),
        ("medical_specialty", train["medical_specialty"], eval_cols["medical_specialty"]),
        ("payer_code", train["payer_code"], eval_cols["payer_code"]),
    ]:
        f_tr, f_ev = compute_freq_encoding(tr_val, ev_val)
        new_num_train[f"freq_{col_key}"] = f_tr
        new_num_eval[f"freq_{col_key}"] = f_ev

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
        "charlson_score",
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

        km_pos = KMeans(n_clusters=K_hop, n_init=1, max_iter=15, random_state=42)
        km_pos.fit(Z_tr_norm[pos_mask])
        proto_pos = km_pos.cluster_centers_
        proto_pos /= (np.linalg.norm(proto_pos, axis=1, keepdims=True) + 1e-8)

        km_neg = KMeans(n_clusters=K_hop, n_init=1, max_iter=15, random_state=42)
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

    # 7. Stratified 5-Fold Cross-Validated Stacking & Scale-Invariant Rank Ensembling
    skf_model = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Clean numeric feature matrix (continuous + target-encodings + frequencies) without raw integer codes
    num_indices = [i for i in range(x_train.shape[1]) if i not in encoder.categorical_indices]
    x_train_num = x_train[:, num_indices]
    x_eval_num = x_eval[:, num_indices]

    med_vals = np.nanmedian(x_train_num, axis=0)
    med_vals = np.nan_to_num(med_vals, nan=0.0)

    model_factories = [
        # Model 1: LGBM Leaf-wise
        ("lgb_leaf", "lgb", lambda s: lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.035,
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
        ("lgb_depth", "lgb", lambda s: lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.035,
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
        # Model 3: LGBM Pos-weighted (Prioritizes high-risk readmissions)
        ("lgb_posweight", "lgb", lambda s: lgb.LGBMClassifier(
            n_estimators=240,
            learning_rate=0.035,
            num_leaves=26,
            min_child_samples=45,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            scale_pos_weight=1.5,
            reg_lambda=6.0,
            reg_alpha=1.0,
            random_state=s,
            n_jobs=1,
            verbose=-1,
        )),
        # Model 4: LGBM Extra Trees
        ("lgb_extra", "lgb", lambda s: lgb.LGBMClassifier(
            n_estimators=230,
            learning_rate=0.035,
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
        # Model 5: XGBoost Hist on clean continuous / target-encoded feature space
        ("xgb_hist", "xgb", lambda s: xgb.XGBClassifier(
            n_estimators=170,
            learning_rate=0.035,
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
        )),
        # Model 6: Regularized Linear Model on Standardized Features
        ("linear_ridge", "linear", lambda s: LogisticRegression(
            C=0.05,
            max_iter=150,
            solver="lbfgs",
            random_state=s,
        )),
    ]

    n_models = len(model_factories)
    oof_preds = np.zeros((n_tr, n_models), dtype=np.float64)
    eval_preds = np.zeros((n_ev, n_models), dtype=np.float64)

    for m_idx, (m_name, m_type, m_ctor) in enumerate(model_factories):
        for fold, (tr_idx, val_idx) in enumerate(skf_model.split(x_train, y_train)):
            model = m_ctor(42 + fold * 13 + m_idx * 101)
            y_tr = y_train[tr_idx]

            if m_type == "lgb":
                X_tr, X_va = x_train[tr_idx], x_train[val_idx]
                model.fit(X_tr, y_tr, categorical_feature=encoder.categorical_indices)
                oof_preds[val_idx, m_idx] = model.predict_proba(X_va)[:, 1]
                eval_preds[:, m_idx] += model.predict_proba(x_eval)[:, 1] / 5.0
            elif m_type == "xgb":
                X_tr, X_va = x_train_num[tr_idx], x_train_num[val_idx]
                model.fit(X_tr, y_tr)
                oof_preds[val_idx, m_idx] = model.predict_proba(X_va)[:, 1]
                eval_preds[:, m_idx] += model.predict_proba(x_eval_num)[:, 1] / 5.0
            elif m_type == "linear":
                X_tr_raw, X_va_raw = x_train_num[tr_idx], x_train_num[val_idx]
                X_tr_imp = np.where(np.isnan(X_tr_raw), med_vals, X_tr_raw)
                X_va_imp = np.where(np.isnan(X_va_raw), med_vals, X_va_raw)
                X_ev_imp = np.where(np.isnan(x_eval_num), med_vals, x_eval_num)
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr_imp)
                X_va_sc = scaler.transform(X_va_imp)
                X_ev_sc = scaler.transform(X_ev_imp)
                model.fit(X_tr_sc, y_tr)
                oof_preds[val_idx, m_idx] = model.predict_proba(X_va_sc)[:, 1]
                eval_preds[:, m_idx] += model.predict_proba(X_ev_sc)[:, 1] / 5.0

    # Scale-invariant rank transformation for ROC-AUC ensembling
    oof_ranks = np.column_stack([rankdata(oof_preds[:, m]) / float(n_tr) for m in range(n_models)])
    eval_ranks = np.column_stack([rankdata(eval_preds[:, m]) / float(n_ev) for m in range(n_models)])

    # Compute optimal non-negative ensemble weights via NNLS on ranks
    w_nnls, _ = nnls(oof_ranks, y_train)
    if w_nnls.sum() > 0:
        w_nnls /= w_nnls.sum()
    else:
        w_nnls = np.ones(n_models, dtype=np.float64) / n_models

    uniform_w = np.ones(n_models, dtype=np.float64) / n_models

    # Regularized blend: 80% NNLS optimal weights + 20% uniform weights
    blend_weights = 0.80 * w_nnls + 0.20 * uniform_w
    blend_weights /= blend_weights.sum()

    preds = eval_ranks @ blend_weights

    preds = np.clip(preds, 0.0, 1.0)
    preds = np.nan_to_num(preds, nan=float(np.mean(y_train)))
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
