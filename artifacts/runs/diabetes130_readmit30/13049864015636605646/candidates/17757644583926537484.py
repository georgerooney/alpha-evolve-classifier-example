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
    """Fit on `train`/`y_train` and return one readmission probability per row of `eval_cols` (same order)."""
    # 1. Comorbidities across all diagnoses
    cm_tr = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
    cm_ev = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])

    tr_cols = dict(train)
    ev_cols = dict(eval_cols)
    tr_cols.update(cm_tr)
    ev_cols.update(cm_ev)
    extra_num = list(cm_tr.keys())

    # Weighted Charlson Comorbidity score and key multi-morbidity interactions
    charlson_tr = (
        cm_tr["cm_mi"] + cm_tr["cm_chf"] + cm_tr["cm_pvd"] + cm_tr["cm_cerebrovascular"] +
        cm_tr["cm_dementia"] + cm_tr["cm_pulmonary"] + cm_tr["cm_rheumatic"] +
        cm_tr["cm_peptic_ulcer"] + cm_tr["cm_liver"] + 2.0 * cm_tr["cm_hemiplegia"] +
        2.0 * cm_tr["cm_renal"] + 2.0 * cm_tr["cm_cancer"] +
        2.0 * cm_tr["cm_diabetes_complicated"] + 4.0 * cm_tr["cm_metastatic"] + 6.0 * cm_tr["cm_hiv"]
    )
    charlson_ev = (
        cm_ev["cm_mi"] + cm_ev["cm_chf"] + cm_ev["cm_pvd"] + cm_ev["cm_cerebrovascular"] +
        cm_ev["cm_dementia"] + cm_ev["cm_pulmonary"] + cm_ev["cm_rheumatic"] +
        cm_ev["cm_peptic_ulcer"] + cm_ev["cm_liver"] + 2.0 * cm_ev["cm_hemiplegia"] +
        2.0 * cm_ev["cm_renal"] + 2.0 * cm_ev["cm_cancer"] +
        2.0 * cm_ev["cm_diabetes_complicated"] + 4.0 * cm_ev["cm_metastatic"] + 6.0 * cm_ev["cm_hiv"]
    )
    tr_cols["charlson_score"] = charlson_tr
    ev_cols["charlson_score"] = charlson_ev
    tr_cols["cardiorenal"] = cm_tr["cm_chf"] * cm_tr["cm_renal"]
    ev_cols["cardiorenal"] = cm_ev["cm_chf"] * cm_ev["cm_renal"]
    tr_cols["cardiopulmonary"] = cm_tr["cm_chf"] * cm_tr["cm_pulmonary"]
    ev_cols["cardiopulmonary"] = cm_ev["cm_chf"] * cm_ev["cm_pulmonary"]
    extra_num.extend(["charlson_score", "cardiorenal", "cardiopulmonary"])

    # ICD-9 chapter groups & 3-digit diagnostic prefixes
    tr_grp1 = np.array([icd9_group(c) for c in train["diag_1"]], dtype=object)
    ev_grp1 = np.array([icd9_group(c) for c in eval_cols["diag_1"]], dtype=object)
    tr_grp2 = np.array([icd9_group(c) for c in train["diag_2"]], dtype=object)
    ev_grp2 = np.array([icd9_group(c) for c in eval_cols["diag_2"]], dtype=object)
    tr_grp3 = np.array([icd9_group(c) for c in train["diag_3"]], dtype=object)
    ev_grp3 = np.array([icd9_group(c) for c in eval_cols["diag_3"]], dtype=object)

    tr_pfx1 = np.array([str(c)[:3] if c is not None else "None" for c in train["diag_1"]], dtype=object)
    ev_pfx1 = np.array([str(c)[:3] if c is not None else "None" for c in eval_cols["diag_1"]], dtype=object)
    tr_pfx2 = np.array([str(c)[:3] if c is not None else "None" for c in train["diag_2"]], dtype=object)
    ev_pfx2 = np.array([str(c)[:3] if c is not None else "None" for c in eval_cols["diag_2"]], dtype=object)
    tr_pfx3 = np.array([str(c)[:3] if c is not None else "None" for c in train["diag_3"]], dtype=object)
    ev_pfx3 = np.array([str(c)[:3] if c is not None else "None" for c in eval_cols["diag_3"]], dtype=object)

    tr_cols["diag_1_group"] = tr_grp1
    ev_cols["diag_1_group"] = ev_grp1
    tr_cols["diag_2_group"] = tr_grp2
    ev_cols["diag_2_group"] = ev_grp2
    tr_cols["diag_3_group"] = tr_grp3
    ev_cols["diag_3_group"] = ev_grp3

    tr_cols["diag_1_prefix"] = tr_pfx1
    ev_cols["diag_1_prefix"] = ev_pfx1
    tr_cols["diag_2_prefix"] = tr_pfx2
    ev_cols["diag_2_prefix"] = ev_pfx2
    tr_cols["diag_3_prefix"] = tr_pfx3
    ev_cols["diag_3_prefix"] = ev_pfx3

    extra_cat = [
        "diag_1_group", "diag_2_group", "diag_3_group",
        "diag_1_prefix", "diag_2_prefix", "diag_3_prefix"
    ]

    for grp in ["circulatory", "diabetes", "respiratory", "genitourinary", "digestive"]:
        tr_cols[f"num_{grp}_diag"] = (
            (tr_grp1 == grp).astype(float) + (tr_grp2 == grp).astype(float) + (tr_grp3 == grp).astype(float)
        )
        ev_cols[f"num_{grp}_diag"] = (
            (ev_grp1 == grp).astype(float) + (ev_grp2 == grp).astype(float) + (ev_grp3 == grp).astype(float)
        )
        extra_num.append(f"num_{grp}_diag")

    # High-impact CMS HRRP primary admission diagnoses
    tr_cols["prim_chf"] = (tr_pfx1 == "428").astype(np.float64)
    ev_cols["prim_chf"] = (ev_pfx1 == "428").astype(np.float64)
    tr_cols["prim_ami"] = (tr_pfx1 == "410").astype(np.float64)
    ev_cols["prim_ami"] = (ev_pfx1 == "410").astype(np.float64)
    tr_cols["prim_pneumonia"] = np.isin(tr_pfx1, ["480", "481", "482", "483", "485", "486"]).astype(np.float64)
    ev_cols["prim_pneumonia"] = np.isin(ev_pfx1, ["480", "481", "482", "483", "485", "486"]).astype(np.float64)
    tr_cols["prim_copd"] = np.isin(tr_pfx1, ["491", "492", "493", "496"]).astype(np.float64)
    ev_cols["prim_copd"] = np.isin(ev_pfx1, ["491", "492", "493", "496"]).astype(np.float64)
    tr_cols["prim_diabetes"] = (tr_pfx1 == "250").astype(np.float64)
    ev_cols["prim_diabetes"] = (ev_pfx1 == "250").astype(np.float64)
    tr_cols["prim_renal"] = np.isin(tr_pfx1, ["584", "585", "586"]).astype(np.float64)
    ev_cols["prim_renal"] = np.isin(ev_pfx1, ["584", "585", "586"]).astype(np.float64)
    tr_cols["is_cms_penalty_cond"] = (
        tr_cols["prim_chf"] + tr_cols["prim_ami"] + tr_cols["prim_pneumonia"] + tr_cols["prim_copd"]
    ).clip(0.0, 1.0)
    ev_cols["is_cms_penalty_cond"] = (
        ev_cols["prim_chf"] + ev_cols["prim_ami"] + ev_cols["prim_pneumonia"] + ev_cols["prim_copd"]
    ).clip(0.0, 1.0)
    extra_num.extend(["prim_chf", "prim_ami", "prim_pneumonia", "prim_copd", "prim_diabetes", "prim_renal", "is_cms_penalty_cond"])

    # 2. Target encode high-cardinality categorical features & 3-digit prefixes
    te_cols = [
        "diag_1",
        "diag_2",
        "diag_3",
        "discharge_disposition_id",
        "admission_source_id",
        "medical_specialty",
        "payer_code",
        "admission_type_id",
    ]
    for col in te_cols:
        tr_te, ev_te = oof_target_encode(train[col], y_train, eval_cols[col], smoothing=20.0)
        te_name = f"{col}_te"
        tr_cols[te_name] = tr_te
        ev_cols[te_name] = ev_te
        extra_num.append(te_name)

    # 3-digit ICD-9 prefix target encodings (dense signal across clinical clusters)
    for pfx_name, pfx_tr, pfx_ev in [
        ("diag_1_prefix", tr_pfx1, ev_pfx1),
        ("diag_2_prefix", tr_pfx2, ev_pfx2),
        ("diag_3_prefix", tr_pfx3, ev_pfx3),
    ]:
        tr_pte, ev_pte = oof_target_encode(pfx_tr, y_train, pfx_ev, smoothing=20.0, seed=17)
        tr_cols[f"{pfx_name}_te"] = tr_pte
        ev_cols[f"{pfx_name}_te"] = ev_pte
        extra_num.append(f"{pfx_name}_te")

    # High-order interaction target encodings
    tr_grp_te, ev_grp_te = oof_target_encode(tr_grp1, y_train, ev_grp1, smoothing=30.0, seed=42)
    tr_cols["diag_1_group_te"] = tr_grp_te
    ev_cols["diag_1_group_te"] = ev_grp_te
    extra_num.append("diag_1_group_te")

    adm_disch_tr = np.array([f"{s}_{d}" for s, d in zip(train["admission_source_id"], train["discharge_disposition_id"])], dtype=object)
    adm_disch_ev = np.array([f"{s}_{d}" for s, d in zip(eval_cols["admission_source_id"], eval_cols["discharge_disposition_id"])], dtype=object)
    tr_ad_te, ev_ad_te = oof_target_encode(adm_disch_tr, y_train, adm_disch_ev, smoothing=25.0, seed=42)
    tr_cols["adm_disch_combo_te"] = tr_ad_te
    ev_cols["adm_disch_combo_te"] = ev_ad_te
    extra_num.append("adm_disch_combo_te")

    diag_disch_tr = np.array([f"{g}_{d}" for g, d in zip(tr_grp1, train["discharge_disposition_id"])], dtype=object)
    diag_disch_ev = np.array([f"{g}_{d}" for g, d in zip(ev_grp1, eval_cols["discharge_disposition_id"])], dtype=object)
    tr_dd_te, ev_dd_te = oof_target_encode(diag_disch_tr, y_train, diag_disch_ev, smoothing=25.0, seed=42)
    tr_cols["diag_disch_combo_te"] = tr_dd_te
    ev_cols["diag_disch_combo_te"] = ev_dd_te
    extra_num.append("diag_disch_combo_te")

    # 3. Utilization intensity
    inpatient_tr = np.nan_to_num(train["number_inpatient"], nan=0.0)
    inpatient_ev = np.nan_to_num(eval_cols["number_inpatient"], nan=0.0)
    emergency_tr = np.nan_to_num(train["number_emergency"], nan=0.0)
    emergency_ev = np.nan_to_num(eval_cols["number_emergency"], nan=0.0)
    outpatient_tr = np.nan_to_num(train["number_outpatient"], nan=0.0)
    outpatient_ev = np.nan_to_num(eval_cols["number_outpatient"], nan=0.0)

    total_prior_tr = inpatient_tr + emergency_tr + outpatient_tr
    total_prior_ev = inpatient_ev + emergency_ev + outpatient_ev

    tr_cols["total_prior_visits"] = total_prior_tr
    ev_cols["total_prior_visits"] = total_prior_ev
    tr_cols["inpatient_ratio"] = inpatient_tr / (total_prior_tr + 1.0)
    ev_cols["inpatient_ratio"] = inpatient_ev / (total_prior_ev + 1.0)
    tr_cols["emergency_ratio"] = emergency_tr / (total_prior_tr + 1.0)
    ev_cols["emergency_ratio"] = emergency_ev / (total_prior_ev + 1.0)
    tr_cols["has_prior_inpatient"] = (inpatient_tr > 0).astype(np.float64)
    ev_cols["has_prior_inpatient"] = (inpatient_ev > 0).astype(np.float64)
    tr_cols["has_mult_inpatient"] = (inpatient_tr >= 2).astype(np.float64)
    ev_cols["has_mult_inpatient"] = (inpatient_ev >= 2).astype(np.float64)
    tr_cols["util_severity"] = 3.0 * inpatient_tr + 1.5 * emergency_tr + 0.5 * outpatient_tr
    ev_cols["util_severity"] = 3.0 * inpatient_ev + 1.5 * emergency_ev + 0.5 * outpatient_ev
    tr_cols["acute_prior_visits"] = inpatient_tr + emergency_tr
    ev_cols["acute_prior_visits"] = inpatient_ev + emergency_ev
    tr_cols["acute_ratio"] = tr_cols["acute_prior_visits"] / (total_prior_tr + 1.0)
    ev_cols["acute_ratio"] = ev_cols["acute_prior_visits"] / (total_prior_ev + 1.0)
    tr_cols["has_frequent_inpatient"] = (inpatient_tr >= 3).astype(np.float64)
    ev_cols["has_frequent_inpatient"] = (inpatient_ev >= 3).astype(np.float64)
    tr_cols["no_prior_visits"] = (total_prior_tr == 0).astype(np.float64)
    ev_cols["no_prior_visits"] = (total_prior_ev == 0).astype(np.float64)
    tr_cols["log1p_inpatient"] = np.log1p(inpatient_tr)
    ev_cols["log1p_inpatient"] = np.log1p(inpatient_ev)

    extra_num.extend([
        "total_prior_visits", "inpatient_ratio", "emergency_ratio",
        "has_prior_inpatient", "has_mult_inpatient", "has_frequent_inpatient",
        "no_prior_visits", "util_severity", "acute_prior_visits", "acute_ratio",
        "log1p_inpatient"
    ])

    # Demographics & Clinical Indicators
    age_map = {"[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0, "[40-50)": 45.0,
               "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0, "[80-90)": 85.0, "[90-100)": 95.0}
    tr_cols["age_midpoint"] = np.array([age_map.get(str(x), 65.0) for x in train["age"]], dtype=np.float64)
    ev_cols["age_midpoint"] = np.array([age_map.get(str(x), 65.0) for x in eval_cols["age"]], dtype=np.float64)
    tr_cols["is_elderly"] = (tr_cols["age_midpoint"] >= 70.0).astype(np.float64)
    ev_cols["is_elderly"] = (ev_cols["age_midpoint"] >= 70.0).astype(np.float64)
    tr_cols["is_very_elderly"] = (tr_cols["age_midpoint"] >= 80.0).astype(np.float64)
    ev_cols["is_very_elderly"] = (ev_cols["age_midpoint"] >= 80.0).astype(np.float64)

    disch_tr = train["discharge_disposition_id"]
    disch_ev = eval_cols["discharge_disposition_id"]
    tr_cols["disch_home"] = (disch_tr == "1").astype(np.float64)
    ev_cols["disch_home"] = (disch_ev == "1").astype(np.float64)
    tr_cols["disch_snf"] = (disch_tr == "3").astype(np.float64)
    ev_cols["disch_snf"] = (disch_ev == "3").astype(np.float64)
    tr_cols["disch_homehealth"] = (disch_tr == "6").astype(np.float64)
    ev_cols["disch_homehealth"] = (disch_ev == "6").astype(np.float64)
    tr_cols["disch_ama"] = (disch_tr == "7").astype(np.float64)
    ev_cols["disch_ama"] = (disch_ev == "7").astype(np.float64)
    tr_cols["disch_transfer"] = np.isin(disch_tr, ["2", "4", "5", "22"]).astype(np.float64)
    ev_cols["disch_transfer"] = np.isin(disch_ev, ["2", "4", "5", "22"]).astype(np.float64)
    tr_cols["disch_not_home"] = (disch_tr != "1").astype(np.float64)
    ev_cols["disch_not_home"] = (disch_ev != "1").astype(np.float64)

    adm_src_tr = train["admission_source_id"]
    adm_src_ev = eval_cols["admission_source_id"]
    tr_cols["adm_er"] = (adm_src_tr == "7").astype(np.float64)
    ev_cols["adm_er"] = (adm_src_ev == "7").astype(np.float64)
    tr_cols["adm_transfer"] = np.isin(adm_src_tr, ["4", "5", "6"]).astype(np.float64)
    ev_cols["adm_transfer"] = np.isin(adm_src_ev, ["4", "5", "6"]).astype(np.float64)
    tr_cols["adm_referral"] = np.isin(adm_src_tr, ["1", "2"]).astype(np.float64)
    ev_cols["adm_referral"] = np.isin(adm_src_ev, ["1", "2"]).astype(np.float64)

    adm_typ_tr = train["admission_type_id"]
    adm_typ_ev = eval_cols["admission_type_id"]
    tr_cols["adm_emergency"] = (adm_typ_tr == "1").astype(np.float64)
    ev_cols["adm_emergency"] = (adm_typ_ev == "1").astype(np.float64)
    tr_cols["adm_elective"] = (adm_typ_tr == "3").astype(np.float64)
    ev_cols["adm_elective"] = (adm_typ_ev == "3").astype(np.float64)
    tr_cols["er_to_facility"] = tr_cols["adm_er"] * tr_cols["disch_not_home"]
    ev_cols["er_to_facility"] = ev_cols["adm_er"] * ev_cols["disch_not_home"]

    los_tr = np.maximum(np.nan_to_num(train["time_in_hospital"], nan=1.0), 1.0)
    los_ev = np.maximum(np.nan_to_num(eval_cols["time_in_hospital"], nan=1.0), 1.0)
    diag_num_tr = np.nan_to_num(train["number_diagnoses"], nan=0.0)
    diag_num_ev = np.nan_to_num(eval_cols["number_diagnoses"], nan=0.0)
    proc_num_tr = np.nan_to_num(train["num_procedures"], nan=0.0)
    proc_num_ev = np.nan_to_num(eval_cols["num_procedures"], nan=0.0)
    med_num_tr = np.nan_to_num(train["num_medications"], nan=0.0)
    med_num_ev = np.nan_to_num(eval_cols["num_medications"], nan=0.0)
    lab_num_tr = np.nan_to_num(train["num_lab_procedures"], nan=0.0)
    lab_num_ev = np.nan_to_num(eval_cols["num_lab_procedures"], nan=0.0)

    tr_cols["labs_per_day"] = lab_num_tr / los_tr
    ev_cols["labs_per_day"] = lab_num_ev / los_ev
    tr_cols["meds_per_day"] = med_num_tr / los_tr
    ev_cols["meds_per_day"] = med_num_ev / los_ev
    tr_cols["procs_per_day"] = proc_num_tr / los_tr
    ev_cols["procs_per_day"] = proc_num_ev / los_ev
    tr_cols["services_per_day"] = tr_cols["labs_per_day"] + tr_cols["meds_per_day"] + tr_cols["procs_per_day"]
    ev_cols["services_per_day"] = ev_cols["labs_per_day"] + ev_cols["meds_per_day"] + ev_cols["procs_per_day"]
    tr_cols["meds_per_diag"] = med_num_tr / np.maximum(diag_num_tr, 1.0)
    ev_cols["meds_per_diag"] = med_num_ev / np.maximum(diag_num_ev, 1.0)
    tr_cols["labs_per_med"] = lab_num_tr / np.maximum(med_num_tr, 1.0)
    ev_cols["labs_per_med"] = lab_num_ev / np.maximum(med_num_ev, 1.0)
    tr_cols["diag_per_day"] = diag_num_tr / los_tr
    ev_cols["diag_per_day"] = diag_num_ev / los_ev
    tr_cols["long_los"] = (los_tr >= 7.0).astype(np.float64)
    ev_cols["long_los"] = (los_ev >= 7.0).astype(np.float64)

    # Inpatient cross-feature interactions (prior utilization x current episode acuity)
    tr_cols["inpatient_x_los"] = inpatient_tr * los_tr
    ev_cols["inpatient_x_los"] = inpatient_ev * los_ev
    tr_cols["inpatient_x_charlson"] = inpatient_tr * charlson_tr
    ev_cols["inpatient_x_charlson"] = inpatient_ev * charlson_ev
    tr_cols["inpatient_x_diagnoses"] = inpatient_tr * diag_num_tr
    ev_cols["inpatient_x_diagnoses"] = inpatient_ev * diag_num_ev
    tr_cols["emergency_x_inpatient"] = emergency_tr * inpatient_tr
    ev_cols["emergency_x_inpatient"] = emergency_ev * inpatient_ev

    # Validated Clinical Indices: LACE Index and HOSPITAL Score
    def calc_lace(los, adm_type, charlson, emergency):
        l_pts = np.where(los < 1, 0.0,
                np.where(los == 1, 1.0,
                np.where(los == 2, 2.0,
                np.where(los == 3, 3.0,
                np.where(los <= 6, 4.0,
                np.where(los <= 13, 5.0, 7.0))))))
        a_pts = np.where(adm_type == "1", 3.0, 0.0)
        c_pts = np.where(charlson == 0, 0.0,
                np.where(charlson == 1, 1.0,
                np.where(charlson == 2, 2.0,
                np.where(charlson == 3, 3.0, 5.0))))
        e_pts = np.clip(emergency, 0.0, 4.0)
        return l_pts + a_pts + c_pts + e_pts

    tr_lace = calc_lace(los_tr, adm_typ_tr, charlson_tr, emergency_tr)
    ev_lace = calc_lace(los_ev, adm_typ_ev, charlson_ev, emergency_ev)
    tr_cols["lace_score"] = tr_lace
    ev_cols["lace_score"] = ev_lace
    tr_cols["lace_high_risk"] = (tr_lace >= 10.0).astype(np.float64)
    ev_cols["lace_high_risk"] = (ev_lace >= 10.0).astype(np.float64)

    # Partial HOSPITAL Score (admissions, stay >= 5d, procedures, urgency)
    hosp_adm_tr = np.where(inpatient_tr > 5, 5.0, np.where(inpatient_tr >= 2, 2.0, 0.0))
    hosp_adm_ev = np.where(inpatient_ev > 5, 5.0, np.where(inpatient_ev >= 2, 2.0, 0.0))
    tr_hosp = hosp_adm_tr + (los_tr >= 5.0).astype(np.float64) * 2.0 + (proc_num_tr > 0).astype(np.float64) + np.isin(adm_typ_tr, ["1", "2"]).astype(np.float64)
    ev_hosp = hosp_adm_ev + (los_ev >= 5.0).astype(np.float64) * 2.0 + (proc_num_ev > 0).astype(np.float64) + np.isin(adm_typ_ev, ["1", "2"]).astype(np.float64)
    tr_cols["hospital_score"] = tr_hosp
    ev_cols["hospital_score"] = ev_hosp

    tr_cols["a1c_tested"] = (train["A1Cresult"] != "None").astype(np.float64)
    ev_cols["a1c_tested"] = (eval_cols["A1Cresult"] != "None").astype(np.float64)
    tr_cols["a1c_abnormal"] = np.isin(train["A1Cresult"], [">7", ">8"]).astype(np.float64)
    ev_cols["a1c_abnormal"] = np.isin(eval_cols["A1Cresult"], [">7", ">8"]).astype(np.float64)
    tr_cols["a1c_uncontrolled_no_change"] = (tr_cols["a1c_abnormal"] == 1.0) & (train["change"] == "No")
    ev_cols["a1c_uncontrolled_no_change"] = (ev_cols["a1c_abnormal"] == 1.0) & (eval_cols["change"] == "No")
    tr_cols["a1c_uncontrolled_no_change"] = tr_cols["a1c_uncontrolled_no_change"].astype(np.float64)
    ev_cols["a1c_uncontrolled_no_change"] = ev_cols["a1c_uncontrolled_no_change"].astype(np.float64)

    tr_cols["glu_tested"] = (train["max_glu_serum"] != "None").astype(np.float64)
    ev_cols["glu_tested"] = (eval_cols["max_glu_serum"] != "None").astype(np.float64)
    tr_cols["glu_high"] = np.isin(train["max_glu_serum"], [">200", ">300"]).astype(np.float64)
    ev_cols["glu_high"] = np.isin(eval_cols["max_glu_serum"], [">200", ">300"]).astype(np.float64)

    med_list = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
    ]
    valid_meds = [m for m in med_list if m in train]
    n_tr, n_ev = len(y_train), len(los_ev)
    act_tr, act_ev = np.zeros(n_tr, dtype=np.float64), np.zeros(n_ev, dtype=np.float64)
    up_tr, up_ev = np.zeros(n_tr, dtype=np.float64), np.zeros(n_ev, dtype=np.float64)
    dn_tr, dn_ev = np.zeros(n_tr, dtype=np.float64), np.zeros(n_ev, dtype=np.float64)
    for m in valid_meds:
        act_tr += (train[m] != "No").astype(np.float64)
        act_ev += (eval_cols[m] != "No").astype(np.float64)
        up_tr += (train[m] == "Up").astype(np.float64)
        up_ev += (eval_cols[m] == "Up").astype(np.float64)
        dn_tr += (train[m] == "Down").astype(np.float64)
        dn_ev += (eval_cols[m] == "Down").astype(np.float64)

    tr_cols["num_active_meds"] = act_tr
    ev_cols["num_active_meds"] = act_ev
    tr_cols["num_meds_up"] = up_tr
    ev_cols["num_meds_up"] = up_ev
    tr_cols["num_meds_down"] = dn_tr
    ev_cols["num_meds_down"] = dn_ev
    tr_cols["num_meds_changed"] = up_tr + dn_tr
    ev_cols["num_meds_changed"] = up_ev + dn_ev

    ins_tr = train["insulin"]
    ins_ev = eval_cols["insulin"]
    tr_cols["insulin_active"] = (ins_tr != "No").astype(np.float64)
    ev_cols["insulin_active"] = (ins_ev != "No").astype(np.float64)
    tr_cols["insulin_up"] = (ins_tr == "Up").astype(np.float64)
    ev_cols["insulin_up"] = (ins_ev == "Up").astype(np.float64)
    tr_cols["insulin_down"] = (ins_tr == "Down").astype(np.float64)
    ev_cols["insulin_down"] = (ins_ev == "Down").astype(np.float64)
    tr_cols["insulin_changed"] = np.isin(ins_tr, ["Up", "Down"]).astype(np.float64)
    ev_cols["insulin_changed"] = np.isin(ins_ev, ["Up", "Down"]).astype(np.float64)
    tr_cols["has_med_change"] = (train["change"] == "Ch").astype(np.float64)
    ev_cols["has_med_change"] = (eval_cols["change"] == "Ch").astype(np.float64)
    tr_cols["metformin_active"] = (train["metformin"] != "No").astype(np.float64)
    ev_cols["metformin_active"] = (eval_cols["metformin"] != "No").astype(np.float64)
    tr_cols["insulin_plus_oral"] = ((tr_cols["insulin_active"] == 1.0) & (act_tr > 1.0)).astype(np.float64)
    ev_cols["insulin_plus_oral"] = ((ev_cols["insulin_active"] == 1.0) & (act_ev > 1.0)).astype(np.float64)

    extra_num.extend([
        "age_midpoint", "is_elderly", "is_very_elderly", "disch_home", "disch_snf",
        "disch_homehealth", "disch_ama", "disch_transfer", "disch_not_home",
        "adm_er", "adm_transfer", "adm_referral", "adm_emergency", "adm_elective",
        "er_to_facility", "labs_per_day", "meds_per_day", "procs_per_day", "services_per_day",
        "meds_per_diag", "labs_per_med", "diag_per_day", "long_los",
        "inpatient_x_los", "inpatient_x_charlson", "inpatient_x_diagnoses", "emergency_x_inpatient",
        "lace_score", "lace_high_risk", "hospital_score",
        "a1c_tested", "a1c_abnormal", "a1c_uncontrolled_no_change", "glu_tested", "glu_high",
        "num_active_meds", "num_meds_up", "num_meds_down", "num_meds_changed",
        "insulin_active", "insulin_up", "insulin_down", "insulin_changed",
        "has_med_change", "metformin_active", "insulin_plus_oral"
    ])

    # 4. Tabular encoding with min_count=15 to retain granular categorical categories
    encoder = TabularEncoder(min_count=15)
    encoder.fit(task, tr_cols, extra_numeric=extra_num, extra_categorical=extra_cat)
    x_train = encoder.transform(tr_cols)
    x_eval = encoder.transform(ev_cols)

    # 5. Blend 4 complementary LightGBM models with diverse architectures
    # Model 1: Deeper, expressive model capturing complex clinical interactions
    model1 = lgb.LGBMClassifier(
        n_estimators=380,
        learning_rate=0.022,
        num_leaves=32,
        min_child_samples=130,
        subsample=0.72,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=1.5,
        reg_alpha=0.2,
        cat_smooth=25.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred1 = model1.predict_proba(x_eval)[:, 1]

    # Model 2: Conservative, regularized model with higher category smoothing
    model2 = lgb.LGBMClassifier(
        n_estimators=420,
        learning_rate=0.018,
        num_leaves=20,
        min_child_samples=180,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_lambda=3.0,
        reg_alpha=0.5,
        cat_smooth=45.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    model2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred2 = model2.predict_proba(x_eval)[:, 1]

    # Model 3: Shallow, depth-constrained tree preventing over-partitioning
    model3 = lgb.LGBMClassifier(
        n_estimators=340,
        learning_rate=0.025,
        num_leaves=24,
        max_depth=5,
        min_child_samples=160,
        subsample=0.78,
        subsample_freq=1,
        colsample_bytree=0.40,
        reg_lambda=2.0,
        reg_alpha=1.0,
        cat_smooth=35.0,
        random_state=777,
        n_jobs=1,
        verbose=-1,
    )
    model3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred3 = model3.predict_proba(x_eval)[:, 1]

    # Model 4: Broad feature coverage with high child sample threshold
    model4 = lgb.LGBMClassifier(
        n_estimators=360,
        learning_rate=0.020,
        num_leaves=26,
        min_child_samples=220,
        subsample=0.68,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=4.0,
        reg_alpha=0.3,
        cat_smooth=50.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    model4.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    pred4 = model4.predict_proba(x_eval)[:, 1]

    return 0.30 * pred1 + 0.28 * pred2 + 0.22 * pred3 + 0.20 * pred4


# EVOLVE-BLOCK-END
# =====================================================================
