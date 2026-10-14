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


def fit_and_score_risk(
    task: TabularTask,
    train: Dict[str, np.ndarray],
    y_train: np.ndarray,
    eval_cols: Dict[str, np.ndarray],
) -> np.ndarray:
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols`."""
    import sys
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import StandardScaler

    def get_discharge_group(disp):
        d = str(disp).strip()
        if d in ("1", "8"):
            return "home"
        elif d in ("2", "3", "4", "5", "22", "23", "24", "30"):
            return "facility"
        elif d in ("6", "25"):
            return "home_health"
        elif d == "7":
            return "ama"
        else:
            return "other"

    def get_admission_source_group(src):
        s = str(src).strip()
        if s in ("7", "8", "25"):
            return "emergency"
        elif s in ("1", "2", "3"):
            return "referral"
        elif s in ("4", "5", "6", "10", "22"):
            return "transfer"
        else:
            return "other"

    def get_admission_type_group(typ):
        t = str(typ).strip()
        if t in ("1", "2"):
            return "urgent"
        elif t == "3":
            return "elective"
        else:
            return "other"

    def get_icd9_prefix(code):
        if code is None or code == "" or code == "missing" or code == "?" or str(code).strip() == "None":
            return "missing"
        code_str = str(code).strip()
        if "." in code_str:
            return code_str.split(".")[0]
        return code_str[:3]

    def map_age_to_numeric(age_col):
        mapping = {
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
        res = np.array([mapping.get(str(x).strip(), 65.0) for x in age_col], dtype=np.float64)
        return res

    # Retrieve toolkit helpers from globals or loaded module
    toolkit = sys.modules.get("clinical_toolkit")
    icd9_grp = globals().get("icd9_group") or (getattr(toolkit, "icd9_group", None) if toolkit else None)
    icd9_cm = globals().get("icd9_comorbidities") or (getattr(toolkit, "icd9_comorbidities", None) if toolkit else None)
    oof_te = globals().get("oof_target_encode") or (getattr(toolkit, "oof_target_encode", None) if toolkit else None)
    EncoderClass = globals().get("TabularEncoder") or (getattr(toolkit, "TabularEncoder", None) if toolkit else None)

    n_tr = len(y_train)
    n_ev = len(eval_cols["time_in_hospital"])

    tr = dict(train)
    ev = dict(eval_cols)
    extra_num = []
    extra_cat = []

    def add_num(name: str, arr_tr, arr_ev):
        t = np.nan_to_num(np.asarray(arr_tr, dtype=np.float64), nan=np.nan, posinf=0.0, neginf=0.0)
        e = np.nan_to_num(np.asarray(arr_ev, dtype=np.float64), nan=np.nan, posinf=0.0, neginf=0.0)
        tr[name] = t
        ev[name] = e
        extra_num.append(name)

    def add_cat(name: str, arr_tr, arr_ev):
        tr[name] = np.asarray(arr_tr, dtype=object)
        ev[name] = np.asarray(arr_ev, dtype=object)
        extra_cat.append(name)

    # Grouped clinical categories
    dg_tr = np.array([get_discharge_group(x) for x in train["discharge_disposition_id"]], dtype=object)
    dg_ev = np.array([get_discharge_group(x) for x in eval_cols["discharge_disposition_id"]], dtype=object)
    add_cat("discharge_group", dg_tr, dg_ev)

    as_tr = np.array([get_admission_source_group(x) for x in train["admission_source_id"]], dtype=object)
    as_ev = np.array([get_admission_source_group(x) for x in eval_cols["admission_source_id"]], dtype=object)
    add_cat("admission_source_group", as_tr, as_ev)

    at_tr = np.array([get_admission_type_group(x) for x in train["admission_type_id"]], dtype=object)
    at_ev = np.array([get_admission_type_group(x) for x in eval_cols["admission_type_id"]], dtype=object)
    add_cat("admission_type_group", at_tr, at_ev)

    # Define base inpatient variables early
    inp_tr = np.nan_to_num(train["number_inpatient"].astype(np.float64), nan=0.0)
    inp_ev = np.nan_to_num(eval_cols["number_inpatient"].astype(np.float64), nan=0.0)

    # 1. Charlson Comorbidities & Multi-System Interactions
    if icd9_cm is not None:
        cm_tr = icd9_cm(train["diag_1"], train["diag_2"], train["diag_3"])
        cm_ev = icd9_cm(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
        for k, v in cm_tr.items():
            add_num(k, v, cm_ev[k])

        # Cardio-Renal-Metabolic (CRM) syndrome dynamics
        crm_tr = cm_tr["cm_chf"] * cm_tr["cm_renal"] * cm_tr["cm_diabetes_complicated"]
        crm_ev = cm_ev["cm_chf"] * cm_ev["cm_renal"] * cm_ev["cm_diabetes_complicated"]
        add_num("crm_triad", crm_tr, crm_ev)

        cardiorenal_tr = cm_tr["cm_chf"] * cm_tr["cm_renal"]
        cardiorenal_ev = cm_ev["cm_chf"] * cm_ev["cm_renal"]
        add_num("cardiorenal", cardiorenal_tr, cardiorenal_ev)

        vasc_tr = cm_tr["cm_mi"] + cm_tr["cm_pvd"] + cm_tr["cm_cerebrovascular"]
        vasc_ev = cm_ev["cm_mi"] + cm_ev["cm_pvd"] + cm_ev["cm_cerebrovascular"]
        add_num("vascular_burden", vasc_tr, vasc_ev)

        pulmo_renal_tr = cm_tr["cm_pulmonary"] * cm_tr["cm_renal"]
        pulmo_renal_ev = cm_ev["cm_pulmonary"] * cm_ev["cm_renal"]
        add_num("pulmo_renal", pulmo_renal_tr, pulmo_renal_ev)

    # 2. ICD-9 Chapter Mapping & Systemic Diversity
    if icd9_grp is not None:
        g1_tr = np.array([icd9_grp(c) for c in train["diag_1"]], dtype=object)
        g1_ev = np.array([icd9_grp(c) for c in eval_cols["diag_1"]], dtype=object)
        add_cat("diag_1_group", g1_tr, g1_ev)

        g2_tr = np.array([icd9_grp(c) for c in train["diag_2"]], dtype=object)
        g2_ev = np.array([icd9_grp(c) for c in eval_cols["diag_2"]], dtype=object)
        add_cat("diag_2_group", g2_tr, g2_ev)

        g3_tr = np.array([icd9_grp(c) for c in train["diag_3"]], dtype=object)
        g3_ev = np.array([icd9_grp(c) for c in eval_cols["diag_3"]], dtype=object)
        add_cat("diag_3_group", g3_tr, g3_ev)

        pair_tr = np.array([f"{a}_{b}" for a, b in zip(g1_tr, g2_tr)], dtype=object)
        pair_ev = np.array([f"{a}_{b}" for a, b in zip(g1_ev, g2_ev)], dtype=object)
        add_cat("diag_pair_group", pair_tr, pair_ev)

        # Multi-system diagnosis entropy / dispersion
        def system_diversity(a1, a2, a3):
            res = np.zeros(len(a1), dtype=np.float64)
            for i in range(len(a1)):
                active = {x for x in (a1[i], a2[i], a3[i]) if x not in ("missing", None, "")}
                res[i] = len(active)
            return res

        add_num("diag_system_diversity", system_diversity(g1_tr, g2_tr, g3_tr), system_diversity(g1_ev, g2_ev, g3_ev))
    else:
        g1_tr = np.array(["missing"] * n_tr, dtype=object)
        g1_ev = np.array(["missing"] * n_ev, dtype=object)
        g2_tr = np.array(["missing"] * n_tr, dtype=object)
        g2_ev = np.array(["missing"] * n_ev, dtype=object)
        g3_tr = np.array(["missing"] * n_tr, dtype=object)
        g3_ev = np.array(["missing"] * n_ev, dtype=object)
        pair_tr = np.array(["missing"] * n_tr, dtype=object)
        pair_ev = np.array(["missing"] * n_ev, dtype=object)

    # 2.5 ICD-9 Code Prefix and Prefix Pair Grouping
    pref1_tr = np.array([get_icd9_prefix(c) for c in train["diag_1"]], dtype=object)
    pref1_ev = np.array([get_icd9_prefix(c) for c in eval_cols["diag_1"]], dtype=object)
    add_cat("diag_1_pref", pref1_tr, pref1_ev)

    pref2_tr = np.array([get_icd9_prefix(c) for c in train["diag_2"]], dtype=object)
    pref2_ev = np.array([get_icd9_prefix(c) for c in eval_cols["diag_2"]], dtype=object)
    add_cat("diag_2_pref", pref2_tr, pref2_ev)

    pref3_tr = np.array([get_icd9_prefix(c) for c in train["diag_3"]], dtype=object)
    pref3_ev = np.array([get_icd9_prefix(c) for c in eval_cols["diag_3"]], dtype=object)
    add_cat("diag_3_pref", pref3_tr, pref3_ev)

    pref_pair_tr = np.array([f"{a}_{b}" for a, b in zip(pref1_tr, pref2_tr)], dtype=object)
    pref_pair_ev = np.array([f"{a}_{b}" for a, b in zip(pref1_ev, pref2_ev)], dtype=object)
    add_cat("diag_pref_pair", pref_pair_tr, pref_pair_ev)

    # 3. Healthcare Utilization Intensity & Velocity
    def extract_utilization(d):
        inp = np.nan_to_num(d["number_inpatient"].astype(np.float64), nan=0.0)
        emg = np.nan_to_num(d["number_emergency"].astype(np.float64), nan=0.0)
        out = np.nan_to_num(d["number_outpatient"].astype(np.float64), nan=0.0)
        time_hosp = np.maximum(np.nan_to_num(d["time_in_hospital"].astype(np.float64), nan=1.0), 1.0)
        labs = np.nan_to_num(d["num_lab_procedures"].astype(np.float64), nan=0.0)
        procs = np.nan_to_num(d["num_procedures"].astype(np.float64), nan=0.0)
        meds = np.nan_to_num(d["num_medications"].astype(np.float64), nan=0.0)
        diags = np.maximum(np.nan_to_num(d["number_diagnoses"].astype(np.float64), nan=1.0), 1.0)

        total_prior = inp + emg + out
        has_prior_inp = (inp > 0).astype(np.float64)
        has_multi_inp = (inp >= 2).astype(np.float64)
        has_prior_emg = (emg > 0).astype(np.float64)
        has_multi_emg = (emg >= 2).astype(np.float64)
        frequent_flyer = ((inp + emg) >= 3).astype(np.float64)
        inp_ratio = inp / (total_prior + 1.0)
        emg_ratio = emg / (total_prior + 1.0)
        acute_score = 3.0 * inp + 1.5 * emg + 0.5 * out

        return {
            "total_prior_visits": total_prior,
            "has_prior_inp": has_prior_inp,
            "has_multi_inp": has_multi_inp,
            "has_prior_emg": has_prior_emg,
            "has_multi_emg": has_multi_emg,
            "frequent_flyer": frequent_flyer,
            "inp_ratio": inp_ratio,
            "emg_ratio": emg_ratio,
            "acute_severity_score": acute_score,
            "log_acute_severity": np.log1p(acute_score),
            "labs_per_day": labs / time_hosp,
            "meds_per_day": meds / time_hosp,
            "procs_per_day": procs / time_hosp,
            "diags_per_day": diags / time_hosp,
            "meds_per_diag": meds / diags,
            "service_density": (labs + 4.0 * procs + 2.0 * meds) / time_hosp,
        }

    u_tr = extract_utilization(train)
    u_ev = extract_utilization(eval_cols)
    for k in u_tr:
        add_num(k, u_tr[k], u_ev[k])

    # 4. Pharmacological Regimen Complexity & Titration Volatility
    med_names = [
        c for c in task.categorical_columns if c in {
            "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
            "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
            "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
            "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
            "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
        }
    ]

    def extract_meds(d):
        n = len(d["time_in_hospital"])
        act = np.zeros(n, dtype=np.float64)
        up = np.zeros(n, dtype=np.float64)
        dn = np.zeros(n, dtype=np.float64)
        st = np.zeros(n, dtype=np.float64)
        for m in med_names:
            if m in d:
                col = d[m]
                act += ((col != "No") & (col is not None)).astype(np.float64)
                up += (col == "Up").astype(np.float64)
                dn += (col == "Down").astype(np.float64)
                st += (col == "Steady").astype(np.float64)

        ins = d.get("insulin")
        if ins is not None:
            ins_act = (ins != "No").astype(np.float64)
            ins_up = (ins == "Up").astype(np.float64)
            ins_dn = (ins == "Down").astype(np.float64)
            ins_st = (ins == "Steady").astype(np.float64)
        else:
            ins_act = np.zeros(n, dtype=np.float64)
            ins_up = np.zeros(n, dtype=np.float64)
            ins_dn = np.zeros(n, dtype=np.float64)
            ins_st = np.zeros(n, dtype=np.float64)

        met = d.get("metformin")
        if met is not None:
            met_act = (met != "No").astype(np.float64)
        else:
            met_act = np.zeros(n, dtype=np.float64)

        oral_act = np.maximum(act - ins_act, 0.0)

        change_col = d.get("change")
        change_med = (change_col == "Ch").astype(np.float64) if change_col is not None else np.zeros(n, dtype=np.float64)

        diab_col = d.get("diabetesMed")
        has_diab_med = (diab_col == "Yes").astype(np.float64) if diab_col is not None else np.zeros(n, dtype=np.float64)

        a1c = d.get("A1Cresult")
        if a1c is not None:
            a1c_tested = (a1c != "None").astype(np.float64)
            a1c_high = ((a1c == ">7") | (a1c == ">8")).astype(np.float64)
        else:
            a1c_tested = np.zeros(n, dtype=np.float64)
            a1c_high = np.zeros(n, dtype=np.float64)

        glu = d.get("max_glu_serum")
        if glu is not None:
            glu_tested = (glu != "None").astype(np.float64)
            glu_high = ((glu == ">200") | (glu == ">300")).astype(np.float64)
        else:
            glu_tested = np.zeros(n, dtype=np.float64)
            glu_high = np.zeros(n, dtype=np.float64)

        return {
            "active_med_count": act,
            "med_up_count": up,
            "med_down_count": dn,
            "med_steady_count": st,
            "dosage_volatility": up + dn,
            "insulin_active": ins_act,
            "insulin_up": ins_up,
            "insulin_down": ins_dn,
            "insulin_steady": ins_st,
            "metformin_active": met_act,
            "oral_active_count": oral_act,
            "insulin_oral_combo": ins_act * (oral_act > 0).astype(np.float64),
            "change_med": change_med,
            "has_diabetes_med": has_diab_med,
            "a1c_tested": a1c_tested,
            "a1c_high": a1c_high,
            "glu_tested": glu_tested,
            "glu_high": glu_high,
        }

    m_tr = extract_meds(train)
    m_ev = extract_meds(eval_cols)
    for k in m_tr:
        add_num(k, m_tr[k], m_ev[k])

    # Polypharmacy & numeric interactions
    non_diab_meds_tr = np.maximum(train["num_medications"].astype(np.float64) - m_tr["active_med_count"], 0.0)
    non_diab_meds_ev = np.maximum(eval_cols["num_medications"].astype(np.float64) - m_ev["active_med_count"], 0.0)
    add_num("non_diabetes_med_count", non_diab_meds_tr, non_diab_meds_ev)

    diab_ratio_tr = m_tr["active_med_count"] / (train["num_medications"].astype(np.float64) + 1.0)
    diab_ratio_ev = m_ev["active_med_count"] / (eval_cols["num_medications"].astype(np.float64) + 1.0)
    add_num("diabetes_med_ratio", diab_ratio_tr, diab_ratio_ev)

    hosp_tr = np.nan_to_num(train["time_in_hospital"].astype(np.float64), nan=1.0)
    hosp_ev = np.nan_to_num(eval_cols["time_in_hospital"].astype(np.float64), nan=1.0)
    add_num("inp_hosp_interaction", inp_tr * hosp_tr, inp_ev * hosp_ev)

    diags_tr = np.nan_to_num(train["number_diagnoses"].astype(np.float64), nan=1.0)
    diags_ev = np.nan_to_num(eval_cols["number_diagnoses"].astype(np.float64), nan=1.0)
    add_num("inp_diags_interaction", inp_tr * diags_tr, inp_ev * diags_ev)

    # 5. Out-of-Fold Target Encodings
    def encode_target(tr_col, ev_col, smoothing=25.0, seed=42):
        if oof_te is not None:
            return oof_te(tr_col, y_train, ev_col, smoothing=smoothing, seed=seed)
        prior = float(np.mean(y_train))
        return np.full(len(tr_col), prior), np.full(len(ev_col), prior)

    for col_name, smooth in [
        ("diag_1", 25.0),
        ("diag_2", 25.0),
        ("diag_3", 25.0),
        ("discharge_disposition_id", 15.0),
        ("admission_source_id", 20.0),
        ("medical_specialty", 25.0),
        ("payer_code", 20.0),
        ("admission_type_id", 20.0),
    ]:
        te_tr, te_ev = encode_target(train[col_name], eval_cols[col_name], smoothing=smooth)
        add_num(f"te_{col_name}", te_tr, te_ev)

    te_pair_tr, te_pair_ev = encode_target(pair_tr, pair_ev, smoothing=25.0)
    add_num("te_diag_pair", te_pair_tr, te_pair_ev)

    # Diagnosis Prefix target encodings
    te_pref1_tr, te_pref1_ev = encode_target(pref1_tr, pref1_ev, smoothing=25.0)
    add_num("te_diag_1_pref", te_pref1_tr, te_pref1_ev)

    te_pref2_tr, te_pref2_ev = encode_target(pref2_tr, pref2_ev, smoothing=25.0)
    add_num("te_diag_2_pref", te_pref2_tr, te_pref2_ev)

    te_pref3_tr, te_pref3_ev = encode_target(pref3_tr, pref3_ev, smoothing=25.0)
    add_num("te_diag_3_pref", te_pref3_tr, te_pref3_ev)

    te_pref_pair_tr, te_pref_pair_ev = encode_target(pref_pair_tr, pref_pair_ev, smoothing=30.0)
    add_num("te_pref_pair", te_pref_pair_tr, te_pref_pair_ev)

    # Complex clinical interaction encodings
    age_disch_tr = np.array([f"{a}_{d}" for a, d in zip(train["age"], train["discharge_disposition_id"])], dtype=object)
    age_disch_ev = np.array([f"{a}_{d}" for a, d in zip(eval_cols["age"], eval_cols["discharge_disposition_id"])], dtype=object)
    te_ad_tr, te_ad_ev = encode_target(age_disch_tr, age_disch_ev, smoothing=20.0)
    add_num("te_age_discharge", te_ad_tr, te_ad_ev)

    diag_1_disch_tr = np.array([f"{g}_{d}" for g, d in zip(g1_tr, train["discharge_disposition_id"])], dtype=object)
    diag_1_disch_ev = np.array([f"{g}_{d}" for g, d in zip(g1_ev, eval_cols["discharge_disposition_id"])], dtype=object)
    te_d1_disch_tr, te_d1_disch_ev = encode_target(diag_1_disch_tr, diag_1_disch_ev, smoothing=25.0)
    add_num("te_diag1g_discharge", te_d1_disch_tr, te_d1_disch_ev)

    adm_type_disch_tr = np.array([f"{a}_{d}" for a, d in zip(train["admission_type_id"], train["discharge_disposition_id"])], dtype=object)
    adm_type_disch_ev = np.array([f"{a}_{d}" for a, d in zip(eval_cols["admission_type_id"], eval_cols["discharge_disposition_id"])], dtype=object)
    te_atd_tr, te_atd_ev = encode_target(adm_type_disch_tr, adm_type_disch_ev, smoothing=20.0)
    add_num("te_adm_type_discharge", te_atd_tr, te_atd_ev)

    spec_disch_tr = np.array([f"{s}_{d}" for s, d in zip(train["medical_specialty"], train["discharge_disposition_id"])], dtype=object)
    spec_disch_ev = np.array([f"{s}_{d}" for s, d in zip(eval_cols["medical_specialty"], eval_cols["discharge_disposition_id"])], dtype=object)
    te_sd_tr, te_sd_ev = encode_target(spec_disch_tr, spec_disch_ev, smoothing=25.0)
    add_num("te_spec_discharge", te_sd_tr, te_sd_ev)

    # Demographic baseline risk and excess individual divergence
    demo_tr = np.array([f"{a}_{g}_{r}" for a, g, r in zip(train["age"], train["gender"], train["race"])], dtype=object)
    demo_ev = np.array([f"{a}_{g}_{r}" for a, g, r in zip(eval_cols["age"], eval_cols["gender"], eval_cols["race"])], dtype=object)
    te_demo_tr, te_demo_ev = encode_target(demo_tr, demo_ev, smoothing=35.0)
    add_num("te_demographic_risk", te_demo_tr, te_demo_ev)
    add_num("excess_individual_risk", u_tr["acute_severity_score"] * te_demo_tr, u_ev["acute_severity_score"] * te_demo_ev)

    # Coarser and interaction target encodings
    te_dg_tr, te_dg_ev = encode_target(dg_tr, dg_ev, smoothing=20.0)
    add_num("te_discharge_group", te_dg_tr, te_dg_ev)

    te_as_tr, te_as_ev = encode_target(as_tr, as_ev, smoothing=20.0)
    add_num("te_admission_source_group", te_as_tr, te_as_ev)

    te_at_tr, te_at_ev = encode_target(at_tr, at_ev, smoothing=20.0)
    add_num("te_admission_type_group", te_at_tr, te_at_ev)

    # Inpatient visit bin interactions
    inp_bin_tr = np.array(["0" if x == 0 else "1" if x == 1 else "2+" for x in inp_tr], dtype=object)
    inp_bin_ev = np.array(["0" if x == 0 else "1" if x == 1 else "2+" for x in inp_ev], dtype=object)

    disch_inp_tr = np.array([f"{d}_{ib}" for d, ib in zip(train["discharge_disposition_id"], inp_bin_tr)], dtype=object)
    disch_inp_ev = np.array([f"{d}_{ib}" for d, ib in zip(eval_cols["discharge_disposition_id"], inp_bin_ev)], dtype=object)
    te_di_tr, te_di_ev = encode_target(disch_inp_tr, disch_inp_ev, smoothing=25.0)
    add_num("te_disch_inp", te_di_tr, te_di_ev)

    spec_inp_tr = np.array([f"{s}_{ib}" for s, ib in zip(train["medical_specialty"], inp_bin_tr)], dtype=object)
    spec_inp_ev = np.array([f"{s}_{ib}" for s, ib in zip(eval_cols["medical_specialty"], inp_bin_ev)], dtype=object)
    te_si_tr, te_si_ev = encode_target(spec_inp_tr, spec_inp_ev, smoothing=25.0)
    add_num("te_spec_inp", te_si_tr, te_si_ev)

    diag1g_inp_tr = np.array([f"{g}_{ib}" for g, ib in zip(g1_tr, inp_bin_tr)], dtype=object)
    diag1g_inp_ev = np.array([f"{g}_{ib}" for g, ib in zip(g1_ev, inp_bin_ev)], dtype=object)
    te_d1gi_tr, te_d1gi_ev = encode_target(diag1g_inp_tr, diag1g_inp_ev, smoothing=25.0)
    add_num("te_diag1g_inp", te_d1gi_tr, te_d1gi_ev)

    age_inp_tr = np.array([f"{a}_{ib}" for a, ib in zip(train["age"], inp_bin_tr)], dtype=object)
    age_inp_ev = np.array([f"{a}_{ib}" for a, ib in zip(eval_cols["age"], inp_bin_ev)], dtype=object)
    te_ai_tr, te_ai_ev = encode_target(age_inp_tr, age_inp_ev, smoothing=20.0)
    add_num("te_age_inp", te_ai_tr, te_ai_ev)

    # Medication clinical interaction
    ins_a1c_tr = np.array([f"{ins}_{a1c}" for ins, a1c in zip(train["insulin"], train["A1Cresult"])], dtype=object)
    ins_a1c_ev = np.array([f"{ins}_{a1c}" for ins, a1c in zip(eval_cols["insulin"], eval_cols["A1Cresult"])], dtype=object)
    te_ia_tr, te_ia_ev = encode_target(ins_a1c_tr, ins_a1c_ev, smoothing=20.0)
    add_num("te_insulin_a1c", te_ia_tr, te_ia_ev)

    # 5.5 Missingness Indicators and Continuous Age Midpoint Representation
    age_num_tr = map_age_to_numeric(train["age"])
    age_num_ev = map_age_to_numeric(eval_cols["age"])
    add_num("age_numeric", age_num_tr, age_num_ev)

    weight_tr = train.get("weight")
    weight_ev = eval_cols.get("weight")
    if weight_tr is not None and weight_ev is not None:
        has_w_tr = ((weight_tr != None) & (weight_tr != "None") & (weight_tr != "?")).astype(np.float64)
        has_w_ev = ((weight_ev != None) & (weight_ev != "None") & (weight_ev != "?")).astype(np.float64)
        add_num("has_weight", has_w_tr, has_w_ev)

    spec_tr = train.get("medical_specialty")
    spec_ev = eval_cols.get("medical_specialty")
    if spec_tr is not None and spec_ev is not None:
        has_spec_tr = ((spec_tr != None) & (spec_tr != "None") & (spec_tr != "?")).astype(np.float64)
        has_spec_ev = ((spec_ev != None) & (spec_ev != "None") & (spec_ev != "?")).astype(np.float64)
        add_num("has_medical_specialty", has_spec_tr, has_spec_ev)

    # 6. Unsupervised Clinical Phenotype Archetype Projection (PARF)
    archetype_keys = [
        "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
        "number_inpatient", "number_emergency", "number_diagnoses",
        "active_med_count", "acute_severity_score"
    ]
    if "cm_count" in tr:
        archetype_keys.append("cm_count")

    X_arch_tr = np.column_stack([np.nan_to_num(tr[k].astype(np.float64), nan=0.0) for k in archetype_keys])
    X_arch_ev = np.column_stack([np.nan_to_num(ev[k].astype(np.float64), nan=0.0) for k in archetype_keys])

    scaler = StandardScaler()
    X_scaled_tr = scaler.fit_transform(X_arch_tr)
    X_scaled_ev = scaler.transform(X_arch_ev)

    n_archetypes = 6
    kmeans = MiniBatchKMeans(n_clusters=n_archetypes, random_state=42, batch_size=1024, n_init=3)
    kmeans.fit(X_scaled_tr)

    dists_tr = kmeans.transform(X_scaled_tr)
    dists_ev = kmeans.transform(X_scaled_ev)
    for c_i in range(n_archetypes):
        add_num(f"archetype_dist_{c_i}", dists_tr[:, c_i], dists_ev[:, c_i])

    clust_tr = kmeans.predict(X_scaled_tr).astype(str)
    clust_ev = kmeans.predict(X_scaled_ev).astype(str)
    te_cl_tr, te_cl_ev = encode_target(clust_tr, clust_ev, smoothing=20.0)
    add_num("te_phenotype_archetype", te_cl_tr, te_cl_ev)

    # 7. Tabular Encoding
    encoder = EncoderClass(min_count=20)
    x_train = encoder.fit_transform(task, tr, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_eval = encoder.transform(ev)
    cat_indices = encoder.categorical_indices

    # 8. Diverse Gradient Boosting Ensemble
    # LightGBM Model 1: Balanced capacity
    m1 = lgb.LGBMClassifier(
        n_estimators=360,
        learning_rate=0.022,
        num_leaves=31,
        max_depth=6,
        min_child_samples=100,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=0.2,
        reg_lambda=2.0,
        cat_smooth=35.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=cat_indices)
    p1 = m1.predict_proba(x_eval)[:, 1]

    # LightGBM Model 2: Deeper, fine-grained interactions
    m2 = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.024,
        num_leaves=45,
        max_depth=7,
        min_child_samples=120,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=0.5,
        reg_lambda=4.0,
        cat_smooth=45.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=cat_indices)
    p2 = m2.predict_proba(x_eval)[:, 1]

    # LightGBM Model 3: Shallow, conservative regularizer
    m3 = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.018,
        num_leaves=20,
        max_depth=5,
        min_child_samples=180,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.5,
        cat_smooth=25.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=cat_indices)
    p3 = m3.predict_proba(x_eval)[:, 1]

    # LightGBM Model 4: Shallow, highly regularized stabilizer
    m4 = lgb.LGBMClassifier(
        n_estimators=440,
        learning_rate=0.015,
        num_leaves=15,
        max_depth=4,
        min_child_samples=250,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=1.5,
        reg_lambda=6.0,
        cat_smooth=50.0,
        random_state=999,
        n_jobs=1,
        verbose=-1,
    )
    m4.fit(x_train, y_train, categorical_feature=cat_indices)
    p4 = m4.predict_proba(x_eval)[:, 1]

    # XGBoost Model: Histogram-based tree splits
    xgb_preds = None
    try:
        m_xgb = xgb.XGBClassifier(
            n_estimators=280,
            learning_rate=0.025,
            max_depth=5,
            min_child_weight=25,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.3,
            reg_lambda=3.0,
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        m_xgb.fit(x_train, y_train)
        xgb_preds = m_xgb.predict_proba(x_eval)[:, 1]
    except Exception:
        xgb_preds = None

    if xgb_preds is not None:
        final_preds = 0.28 * p1 + 0.24 * p2 + 0.20 * p3 + 0.13 * p4 + 0.15 * xgb_preds
    else:
        final_preds = 0.35 * p1 + 0.30 * p2 + 0.20 * p3 + 0.15 * p4

    final_preds = np.nan_to_num(final_preds, nan=0.114)
    return np.clip(final_preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
