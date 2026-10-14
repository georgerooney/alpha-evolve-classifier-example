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
    """Fit an ensemble of gradient-boosted trees with rich clinical feature engineering."""
    import lightgbm as lgb

    try:
        import xgboost as xgb
    except Exception:
        xgb = None

    # Retrieve pre-imported toolkit functions
    _icd9_comorbidities = globals().get("icd9_comorbidities")
    _icd9_group = globals().get("icd9_group")
    _oof_target_encode = globals().get("oof_target_encode")
    _TabularEncoder = globals().get("TabularEncoder")

    if _icd9_comorbidities is None or _oof_target_encode is None:
        try:
            import clinical_toolkit

            if _icd9_comorbidities is None:
                _icd9_comorbidities = getattr(
                    clinical_toolkit, "icd9_comorbidities", None
                )
            if _icd9_group is None:
                _icd9_group = getattr(clinical_toolkit, "icd9_group", None)
            if _oof_target_encode is None:
                _oof_target_encode = getattr(
                    clinical_toolkit, "oof_target_encode", None
                )
            if _TabularEncoder is None:
                _TabularEncoder = getattr(
                    clinical_toolkit, "TabularEncoder", None
                )
        except Exception:
            pass

    n_train = len(y_train)
    n_eval = len(eval_cols["time_in_hospital"])

    train_ext = dict(train)
    eval_ext = dict(eval_cols)
    extra_num = {}
    extra_cat = {}

    # 1. Comorbidities & Canonical Charlson Comorbidity Index (Rank-1 Clinical Risk Projection)
    if _icd9_comorbidities is not None:
        cm_tr = _icd9_comorbidities(
            train["diag_1"], train["diag_2"], train["diag_3"]
        )
        cm_ev = _icd9_comorbidities(
            eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"]
        )
        for k in cm_tr:
            extra_num[k] = (
                cm_tr[k].astype(np.float64),
                cm_ev[k].astype(np.float64),
            )

        # Standard weighted Charlson Comorbidity Index
        cci_tr = (
            cm_tr["cm_mi"]
            + cm_tr["cm_chf"]
            + cm_tr["cm_pvd"]
            + cm_tr["cm_cerebrovascular"]
            + cm_tr["cm_dementia"]
            + cm_tr["cm_pulmonary"]
            + cm_tr["cm_rheumatic"]
            + cm_tr["cm_peptic_ulcer"]
            + cm_tr["cm_liver"]
            + 2.0 * cm_tr["cm_diabetes_complicated"]
            + 2.0 * cm_tr["cm_hemiplegia"]
            + 2.0 * cm_tr["cm_renal"]
            + 2.0 * cm_tr["cm_cancer"]
            + 4.0 * cm_tr["cm_metastatic"]
            + 4.0 * cm_tr["cm_hiv"]
        )
        cci_ev = (
            cm_ev["cm_mi"]
            + cm_ev["cm_chf"]
            + cm_ev["cm_pvd"]
            + cm_ev["cm_cerebrovascular"]
            + cm_ev["cm_dementia"]
            + cm_ev["cm_pulmonary"]
            + cm_ev["cm_rheumatic"]
            + cm_ev["cm_peptic_ulcer"]
            + cm_ev["cm_liver"]
            + 2.0 * cm_ev["cm_diabetes_complicated"]
            + 2.0 * cm_ev["cm_hemiplegia"]
            + 2.0 * cm_ev["cm_renal"]
            + 2.0 * cm_ev["cm_cancer"]
            + 4.0 * cm_ev["cm_metastatic"]
            + 4.0 * cm_ev["cm_hiv"]
        )
        extra_num["cm_cci"] = (cci_tr, cci_ev)

        # TruncatedSVD for latent comorbidity representation
        try:
            from sklearn.decomposition import TruncatedSVD
            cm_keys = sorted([k for k in cm_tr.keys() if k.startswith("cm_")])
            cm_matrix_tr = np.column_stack([cm_tr[k].astype(np.float64) for k in cm_keys])
            cm_matrix_ev = np.column_stack([cm_ev[k].astype(np.float64) for k in cm_keys])
            svd = TruncatedSVD(n_components=3, random_state=42)
            svd_tr = svd.fit_transform(cm_matrix_tr)
            svd_ev = svd.transform(cm_matrix_ev)
            extra_num["cm_svd_0"] = (svd_tr[:, 0], svd_ev[:, 0])
            extra_num["cm_svd_1"] = (svd_tr[:, 1], svd_ev[:, 1])
            extra_num["cm_svd_2"] = (svd_tr[:, 2], svd_ev[:, 2])
        except Exception:
            pass
    else:
        cci_tr = np.zeros(n_train, dtype=np.float64)
        cci_ev = np.zeros(n_eval, dtype=np.float64)

    # 2. ICD-9 Chapter Groups & Clinical pathways
    if _icd9_group is not None:
        g1_tr = np.array([_icd9_group(c) for c in train["diag_1"]], dtype=object)
        g1_ev = np.array([_icd9_group(c) for c in eval_cols["diag_1"]], dtype=object)
        g2_tr = np.array([_icd9_group(c) for c in train["diag_2"]], dtype=object)
        g2_ev = np.array([_icd9_group(c) for c in eval_cols["diag_2"]], dtype=object)
        g3_tr = np.array([_icd9_group(c) for c in train["diag_3"]], dtype=object)
        g3_ev = np.array([_icd9_group(c) for c in eval_cols["diag_3"]], dtype=object)

        extra_cat["diag_1_group"] = (g1_tr, g1_ev)
        extra_cat["diag_2_group"] = (g2_tr, g2_ev)
        extra_cat["diag_3_group"] = (g3_tr, g3_ev)

        diag_12_tr = np.char.add(np.char.add(g1_tr.astype(str), "_"), g2_tr.astype(str))
        diag_12_ev = np.char.add(np.char.add(g1_ev.astype(str), "_"), g2_ev.astype(str))
        extra_cat["diag_12_group"] = (diag_12_tr, diag_12_ev)

    # Transition pathways (Admission Source to Discharge Disposition)
    src_tr = np.array([str(x) if x is not None else "None" for x in train["admission_source_id"]], dtype=object)
    src_ev = np.array([str(x) if x is not None else "None" for x in eval_cols["admission_source_id"]], dtype=object)
    dis_tr = np.array([str(x) if x is not None else "None" for x in train["discharge_disposition_id"]], dtype=object)
    dis_ev = np.array([str(x) if x is not None else "None" for x in eval_cols["discharge_disposition_id"]], dtype=object)

    path_tr = np.char.add(np.char.add(src_tr.astype(str), "_"), dis_tr.astype(str))
    path_ev = np.char.add(np.char.add(src_ev.astype(str), "_"), dis_ev.astype(str))
    extra_cat["path_src_disch"] = (path_tr, path_ev)

    # Primary acute clinical admission causes
    def is_prefix(arr, prefixes):
        res = np.zeros(len(arr), dtype=np.float64)
        for i, v in enumerate(arr):
            if v is not None:
                s = str(v)
                for p in prefixes:
                    if s.startswith(p):
                        res[i] = 1.0
                        break
        return res

    extra_num["diag1_chf"] = (
        is_prefix(train["diag_1"], ["428"]),
        is_prefix(eval_cols["diag_1"], ["428"]),
    )
    extra_num["diag1_ihd"] = (
        is_prefix(train["diag_1"], ["410", "411", "412", "413", "414"]),
        is_prefix(eval_cols["diag_1"], ["410", "411", "412", "413", "414"]),
    )
    extra_num["diag1_pna"] = (
        is_prefix(train["diag_1"], ["480", "481", "482", "483", "484", "485", "486"]),
        is_prefix(eval_cols["diag_1"], ["480", "481", "482", "483", "484", "485", "486"]),
    )
    extra_num["diag1_renal"] = (
        is_prefix(train["diag_1"], ["584", "585", "586"]),
        is_prefix(eval_cols["diag_1"], ["584", "585", "586"]),
    )
    extra_num["diag1_sepsis"] = (
        is_prefix(train["diag_1"], ["038", "995.9"]),
        is_prefix(eval_cols["diag_1"], ["038", "995.9"]),
    )
    extra_num["diag1_dka"] = (
        is_prefix(train["diag_1"], ["250.1", "250.2", "250.3"]),
        is_prefix(eval_cols["diag_1"], ["250.1", "250.2", "250.3"]),
    )

    # 3. High-Risk Clinical Discharges & Admissions
    disch_tr = train["discharge_disposition_id"]
    disch_ev = eval_cols["discharge_disposition_id"]
    adm_type_tr = train["admission_type_id"]
    adm_type_ev = eval_cols["admission_type_id"]
    adm_src_tr = train["admission_source_id"]
    adm_src_ev = eval_cols["admission_source_id"]

    extra_num["disch_is_home"] = (
        (disch_tr == "1").astype(np.float64),
        (disch_ev == "1").astype(np.float64),
    )
    extra_num["disch_is_snf"] = (
        (disch_tr == "3").astype(np.float64),
        (disch_ev == "3").astype(np.float64),
    )
    extra_num["disch_is_home_health"] = (
        (disch_tr == "6").astype(np.float64),
        (disch_ev == "6").astype(np.float64),
    )
    extra_num["disch_is_facility"] = (
        np.isin(disch_tr, ["2", "3", "4", "5", "22", "23", "24"]).astype(np.float64),
        np.isin(disch_ev, ["2", "3", "4", "5", "22", "23", "24"]).astype(np.float64),
    )

    is_emerg_adm_tr = (adm_type_tr == "1").astype(np.float64)
    is_emerg_adm_ev = (adm_type_ev == "1").astype(np.float64)
    extra_num["adm_is_emergency"] = (is_emerg_adm_tr, is_emerg_adm_ev)
    extra_num["adm_src_er"] = (
        (adm_src_tr == "7").astype(np.float64),
        (adm_src_ev == "7").astype(np.float64),
    )

    # 3. Out-of-Fold Bayes Target Encodings (Rank-1 Conditional Expectation Projections)
    te_values = {}
    if _oof_target_encode is not None:
        te_spec = [
            ("diag_1", 15.0, 42),
            ("diag_2", 15.0, 43),
            ("diag_3", 15.0, 51),
            ("discharge_disposition_id", 10.0, 44),
            ("admission_source_id", 15.0, 45),
            ("admission_type_id", 20.0, 46),
            ("medical_specialty", 25.0, 47),
            ("payer_code", 25.0, 48),
            ("path_src_disch", 15.0, 49),
            ("diag_12_group", 20.0, 50),
        ]
        for col_name, smoothing_val, seed_v in te_spec:
            tr_col = train.get(col_name)
            ev_col = eval_cols.get(col_name)
            if tr_col is None and col_name in extra_cat:
                tr_col = extra_cat[col_name][0]
                ev_col = extra_cat[col_name][1]

            if tr_col is not None:
                tr_te, ev_te = _oof_target_encode(
                    tr_col,
                    y_train,
                    ev_col,
                    n_folds=5,
                    smoothing=smoothing_val,
                    seed=seed_v,
                )
                extra_num[f"te_{col_name}"] = (
                    tr_te.astype(np.float64),
                    ev_te.astype(np.float64),
                )
                te_values[col_name] = (tr_te.astype(np.float64), ev_te.astype(np.float64))

    # 4. Continuous Midpoint Demographics
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
    age_tr = np.array([age_map.get(str(a), 65.0) for a in train["age"]], dtype=np.float64)
    age_ev = np.array([age_map.get(str(a), 65.0) for a in eval_cols["age"]], dtype=np.float64)
    extra_num["age_num"] = (age_tr, age_ev)
    extra_num["is_elderly"] = (
        (age_tr >= 75.0).astype(np.float64),
        (age_ev >= 75.0).astype(np.float64),
    )
    extra_num["is_extreme_elderly"] = (
        (age_tr >= 90.0).astype(np.float64),
        (age_ev >= 90.0).astype(np.float64),
    )

    # 5. Prior Healthcare Utilization: Non-Redundant Log-Scale Basis
    inp_tr = np.nan_to_num(train["number_inpatient"], nan=0.0)
    inp_ev = np.nan_to_num(eval_cols["number_inpatient"], nan=0.0)
    emg_tr = np.nan_to_num(train["number_emergency"], nan=0.0)
    emg_ev = np.nan_to_num(eval_cols["number_emergency"], nan=0.0)
    out_tr = np.nan_to_num(train["number_outpatient"], nan=0.0)
    out_ev = np.nan_to_num(eval_cols["number_outpatient"], nan=0.0)

    tot_vis_tr = inp_tr + emg_tr + out_tr
    tot_vis_ev = inp_ev + emg_ev + out_ev

    log_inp_tr = np.log1p(inp_tr)
    log_inp_ev = np.log1p(inp_ev)

    extra_num["log_inpatient"] = (log_inp_tr, log_inp_ev)
    extra_num["log_emergency"] = (np.log1p(emg_tr), np.log1p(emg_ev))
    extra_num["log_outpatient"] = (np.log1p(out_tr), np.log1p(out_ev))
    extra_num["log_total_visits"] = (np.log1p(tot_vis_tr), np.log1p(tot_vis_ev))
    extra_num["has_prior_inpatient"] = (
        (inp_tr > 0.0).astype(np.float64),
        (inp_ev > 0.0).astype(np.float64),
    )

    # Weighted utilization risk score
    extra_num["utilization_risk_score"] = (
        3.0 * log_inp_tr + 1.0 * np.log1p(emg_tr) + 0.5 * np.log1p(out_tr),
        3.0 * log_inp_ev + 1.0 * np.log1p(emg_ev) + 0.5 * np.log1p(out_ev),
    )

    # Extreme elderly interaction with inpatient visits
    extra_num["extreme_elderly_x_inpatient"] = (
        (age_tr >= 90.0).astype(np.float64) * log_inp_tr,
        (age_ev >= 90.0).astype(np.float64) * log_inp_ev,
    )

    # 6. Stay Intensity & Canonical LACE Index
    t_hosp_tr = np.nan_to_num(train["time_in_hospital"], nan=1.0)
    t_hosp_ev = np.nan_to_num(eval_cols["time_in_hospital"], nan=1.0)
    n_med_tr = np.nan_to_num(train["num_medications"], nan=0.0)
    n_med_ev = np.nan_to_num(eval_cols["num_medications"], nan=0.0)
    n_lab_tr = np.nan_to_num(train["num_lab_procedures"], nan=0.0)
    n_lab_ev = np.nan_to_num(eval_cols["num_lab_procedures"], nan=0.0)
    n_proc_tr = np.nan_to_num(train["num_procedures"], nan=0.0)
    n_proc_ev = np.nan_to_num(eval_cols["num_procedures"], nan=0.0)

    extra_num["med_per_day"] = (
        n_med_tr / (t_hosp_tr + 1.0),
        n_med_ev / (t_hosp_ev + 1.0),
    )
    extra_num["lab_per_day"] = (
        n_lab_tr / (t_hosp_tr + 1.0),
        n_lab_ev / (t_hosp_ev + 1.0),
    )
    extra_num["events_per_day"] = (
        (n_lab_tr + n_med_tr + n_proc_tr) / (t_hosp_tr + 1.0),
        (n_lab_ev + n_med_ev + n_proc_ev) / (t_hosp_ev + 1.0),
    )

    # LACE Index (Length of stay, Acuity, Comorbidities, Emergency visits)
    def compute_lace(los, is_emerg, cci, emg):
        los_pts = np.select(
            [los <= 1, los == 2, los == 3, los <= 6, los <= 13],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            default=7.0,
        )
        acuity_pts = np.where(is_emerg > 0, 3.0, 0.0)
        cci_pts = np.clip(cci, 0.0, 6.0)
        emg_pts = np.clip(emg, 0.0, 4.0)
        return los_pts + acuity_pts + cci_pts + emg_pts

    extra_num["lace_index"] = (
        compute_lace(t_hosp_tr, is_emerg_adm_tr, cci_tr, emg_tr),
        compute_lace(t_hosp_ev, is_emerg_adm_ev, cci_ev, emg_ev),
    )

    # 7. Target-Encoded Interaction Features with Prior Inpatient stays
    for col_name in ["diag_1", "discharge_disposition_id", "path_src_disch", "diag_12_group"]:
        if col_name in te_values:
            tr_te, ev_te = te_values[col_name]
            extra_num[f"te_{col_name}_x_inpatient"] = (
                tr_te * log_inp_tr,
                ev_te * log_inp_ev,
              )

    # 9. Medication Regimen Dynamics
    med_list = [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]
    avail_meds = [m for m in med_list if m in train]

    def process_meds(cols_dict, size):
        active = np.zeros(size, dtype=np.float64)
        up = np.zeros(size, dtype=np.float64)
        down = np.zeros(size, dtype=np.float64)
        for m in avail_meds:
            arr = cols_dict[m]
            active += (arr != "No") & (arr is not None)
            up += arr == "Up"
            down += arr == "Down"
        changed = up + down
        insulin_col = cols_dict.get("insulin", np.array(["No"] * size))
        insulin_no = (insulin_col == "No").astype(np.float64)
        insulin_steady = (insulin_col == "Steady").astype(np.float64)
        insulin_up = (insulin_col == "Up").astype(np.float64)
        insulin_down = (insulin_col == "Down").astype(np.float64)
        insulin_changed = np.isin(insulin_col, ["Up", "Down"]).astype(
            np.float64
        )
        return (
            active,
            up,
            down,
            changed,
            insulin_no,
            insulin_steady,
            insulin_up,
            insulin_down,
            insulin_changed,
        )

    (
        act_tr,
        up_tr,
        dn_tr,
        chg_tr,
        ins_no_tr,
        ins_st_tr,
        ins_up_tr,
        ins_dn_tr,
        ins_chg_tr,
    ) = process_meds(train, n_train)
    (
        act_ev,
        up_ev,
        dn_ev,
        chg_ev,
        ins_no_ev,
        ins_st_ev,
        ins_up_ev,
        ins_dn_ev,
        ins_chg_ev,
    ) = process_meds(eval_cols, n_eval)

    extra_num["num_active_meds"] = (act_tr, act_ev)
    extra_num["num_meds_up"] = (up_tr, up_ev)
    extra_num["num_meds_down"] = (dn_tr, dn_ev)
    extra_num["num_meds_changed"] = (chg_tr, chg_ev)
    extra_num["active_med_ratio"] = (
        act_tr / (n_med_tr + 1.0),
        act_ev / (n_med_ev + 1.0),
    )
    extra_num["insulin_no"] = (ins_no_tr, ins_no_ev)
    extra_num["insulin_steady"] = (ins_st_tr, ins_st_ev)
    extra_num["insulin_up"] = (ins_up_tr, ins_up_ev)
    extra_num["insulin_down"] = (ins_dn_tr, ins_dn_ev)
    extra_num["insulin_changed"] = (ins_chg_tr, ins_chg_ev)
    extra_num["insulin_alone"] = (
        ((ins_no_tr == 0.0) & (act_tr == 1.0)).astype(np.float64),
        ((ins_no_ev == 0.0) & (act_ev == 1.0)).astype(np.float64),
    )
    extra_num["polypharmacy"] = (
        (n_med_tr >= 16.0).astype(np.float64),
        (n_med_ev >= 16.0).astype(np.float64),
    )

    met_tr = np.isin(train["metformin"], ["Steady", "Up", "Down"]).astype(
        np.float64
    )
    met_ev = np.isin(eval_cols["metformin"], ["Steady", "Up", "Down"]).astype(
        np.float64
    )
    extra_num["metformin_active"] = (met_tr, met_ev)

    extra_num["change_flag"] = (
        (train["change"] == "Ch").astype(np.float64),
        (eval_cols["change"] == "Ch").astype(np.float64),
    )
    extra_num["diabetes_med_flag"] = (
        (train["diabetesMed"] == "Yes").astype(np.float64),
        (eval_cols["diabetesMed"] == "Yes").astype(np.float64),
    )

    # 10. Glycemic Labs & Missingness Flags
    a1c_tested_tr = (train["A1Cresult"] != "None").astype(np.float64)
    a1c_tested_ev = (eval_cols["A1Cresult"] != "None").astype(np.float64)
    glu_tested_tr = (train["max_glu_serum"] != "None").astype(np.float64)
    glu_tested_ev = (eval_cols["max_glu_serum"] != "None").astype(np.float64)

    a1c_high_tr = np.isin(train["A1Cresult"], [">7", ">8"]).astype(np.float64)
    a1c_high_ev = np.isin(eval_cols["A1Cresult"], [">7", ">8"]).astype(np.float64)
    glu_high_tr = np.isin(train["max_glu_serum"], [">200", ">300"]).astype(np.float64)
    glu_high_ev = np.isin(eval_cols["max_glu_serum"], [">200", ">300"]).astype(np.float64)

    extra_num["a1c_tested"] = (a1c_tested_tr, a1c_tested_ev)
    extra_num["a1c_high"] = (a1c_high_tr, a1c_high_ev)
    extra_num["glu_tested"] = (glu_tested_tr, glu_tested_ev)
    extra_num["glu_high"] = (glu_high_tr, glu_high_ev)

    extra_num["glycemic_both_tested"] = (
        a1c_tested_tr * glu_tested_tr,
        a1c_tested_ev * glu_tested_ev,
    )
    extra_num["glycemic_any_high"] = (
        np.maximum(a1c_high_tr, glu_high_tr),
        np.maximum(a1c_high_ev, glu_high_ev),
    )

    for c in ["weight", "payer_code", "medical_specialty"]:
        if c in train:
            extra_num[f"{c}_missing"] = (
                np.isin(train[c], [None, "?"]).astype(np.float64),
                np.isin(eval_cols[c], [None, "?"]).astype(np.float64),
            )

    # Register derived columns
    extra_num_names = []
    for k, (tr_v, ev_v) in extra_num.items():
        train_ext[k] = tr_v
        eval_ext[k] = ev_v
        extra_num_names.append(k)

    extra_cat_names = []
    for k, (tr_v, ev_v) in extra_cat.items():
        train_ext[k] = tr_v
        eval_ext[k] = ev_v
        extra_cat_names.append(k)

    encoder = _TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(
        task,
        train_ext,
        extra_numeric=extra_num_names,
        extra_categorical=extra_cat_names,
    )
    x_eval = encoder.transform(eval_ext)
    cat_indices = encoder.categorical_indices

    # Model 1: Canonical Greedy Leaf-Wise GBDT (LightGBM)
    m1 = lgb.LGBMClassifier(
        n_estimators=380,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.70,
        reg_lambda=3.0,
        reg_alpha=0.4,
        cat_smooth=30.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=cat_indices)
    p1 = m1.predict_proba(x_eval)[:, 1]

    # Model 2: Randomized Threshold Leaf-Wise GBDT (LightGBM ExtraTrees)
    try:
        m2 = lgb.LGBMClassifier(
            n_estimators=340,
            learning_rate=0.03,
            extra_trees=True,
            num_leaves=35,
            min_child_samples=90,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=2.5,
            reg_alpha=0.3,
            cat_smooth=30.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        )
        m2.fit(x_train, y_train, categorical_feature=cat_indices)
        p2 = m2.predict_proba(x_eval)[:, 1]
    except Exception:
        p2 = None

    # Prepare numerical and target-encoded subsets for XGBoost and Logistic Regression
    cat_set = set(cat_indices)
    num_indices = [
        i for i in range(x_train.shape[1]) if i not in cat_set
    ]
    x_train_num = x_train[:, num_indices]
    x_eval_num = x_eval[:, num_indices]

    # Model 3: Orthogonal Depth-Wise Symmetric GBDT (XGBoost)
    p_xgb = None
    if xgb is not None:
        try:
            m_xgb = xgb.XGBClassifier(
                n_estimators=280,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=8,
                subsample=0.80,
                colsample_bytree=0.65,
                reg_lambda=4.0,
                reg_alpha=0.5,
                random_state=42,
                n_jobs=1,
                verbosity=0,
                eval_metric="logloss",
            )
            m_xgb.fit(x_train_num, y_train)
            p_xgb = m_xgb.predict_proba(x_eval_num)[:, 1]
        except Exception:
            p_xgb = None

    # Model 4: Orthogonal Regularized Linear Model (Logistic Regression)
    p_lr = None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline

        m_lr = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            LogisticRegression(C=0.02, max_iter=400, random_state=42, penalty='l2')
        )
        m_lr.fit(x_train_num, y_train)
        p_lr = m_lr.predict_proba(x_eval_num)[:, 1]
    except Exception:
        p_lr = None

    # Robust multi-family ensemble blending across diverse models
    preds_list = [p1]
    weights_list = [0.40]

    if p2 is not None:
        preds_list.append(p2)
        weights_list.append(0.20)

    if p_xgb is not None:
        preds_list.append(p_xgb)
        weights_list.append(0.25)

    if p_lr is not None:
        preds_list.append(p_lr)
        weights_list.append(0.15)

    total_w = sum(weights_list)
    preds = np.zeros_like(p1)
    for p, w in zip(preds_list, weights_list):
        preds += (w / total_w) * p

    preds = np.nan_to_num(preds, nan=0.114)
    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
