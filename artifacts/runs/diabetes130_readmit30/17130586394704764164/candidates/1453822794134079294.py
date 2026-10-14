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

    # Helper: truncate ICD-9 codes to primary 3-digit categories
    def to_icd9_3(diag_arr):
        res = np.empty(len(diag_arr), dtype=object)
        for i, v in enumerate(diag_arr):
            if v is None or v == "" or v == "?":
                res[i] = "None"
            else:
                s = str(v).strip()
                res[i] = s[:3] if len(s) >= 3 else s
        return res

    diag1_3_tr = to_icd9_3(train["diag_1"])
    diag1_3_ev = to_icd9_3(eval_cols["diag_1"])
    diag2_3_tr = to_icd9_3(train["diag_2"])
    diag2_3_ev = to_icd9_3(eval_cols["diag_2"])
    diag3_3_tr = to_icd9_3(train["diag_3"])
    diag3_3_ev = to_icd9_3(eval_cols["diag_3"])

    extra_cat["diag_1_3digit"] = (diag1_3_tr, diag1_3_ev)
    extra_cat["diag_2_3digit"] = (diag2_3_tr, diag2_3_ev)
    extra_cat["diag_3_3digit"] = (diag3_3_tr, diag3_3_ev)

    # 1. Comorbidities & Charlson Index
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

        # Weighted Charlson Comorbidity Index
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
        extra_num["cardiorenal"] = (
            (cm_tr["cm_chf"] * cm_tr["cm_renal"]).astype(np.float64),
            (cm_ev["cm_chf"] * cm_ev["cm_renal"]).astype(np.float64),
        )
        extra_num["diabetic_renal"] = (
            (cm_tr["cm_diabetes_complicated"] * cm_tr["cm_renal"]).astype(
                np.float64
            ),
            (cm_ev["cm_diabetes_complicated"] * cm_ev["cm_renal"]).astype(
                np.float64
            ),
        )
        extra_num["multimorbid"] = (
            (cm_tr["cm_count"] >= 3.0).astype(np.float64),
            (cm_ev["cm_count"] >= 3.0).astype(np.float64),
        )
        extra_num["triad_cardiorenal_diab"] = (
            (
                cm_tr["cm_chf"]
                * cm_tr["cm_renal"]
                * cm_tr["cm_diabetes_complicated"]
            ).astype(np.float64),
            (
                cm_ev["cm_chf"]
                * cm_ev["cm_renal"]
                * cm_ev["cm_diabetes_complicated"]
            ).astype(np.float64),
        )
        extra_num["respiratory_cardiac"] = (
            (
                cm_tr["cm_pulmonary"]
                * np.clip(cm_tr["cm_chf"] + cm_tr["cm_mi"], 0.0, 1.0)
            ).astype(np.float64),
            (
                cm_ev["cm_pulmonary"]
                * np.clip(cm_ev["cm_chf"] + cm_ev["cm_mi"], 0.0, 1.0)
            ).astype(np.float64),
        )
    else:
        cci_tr = np.zeros(n_train, dtype=np.float64)
        cci_ev = np.zeros(n_eval, dtype=np.float64)

    # 2. ICD-9 Chapter Groups
    diag1_g_tr = np.array(["missing"] * n_train, dtype=object)
    diag1_g_ev = np.array(["missing"] * n_eval, dtype=object)
    if _icd9_group is not None:
        diag1_g_tr = np.array(
            [_icd9_group(c) for c in train["diag_1"]], dtype=object
        )
        diag1_g_ev = np.array(
            [_icd9_group(c) for c in eval_cols["diag_1"]], dtype=object
        )
        diag2_g_tr = np.array(
            [_icd9_group(c) for c in train["diag_2"]], dtype=object
        )
        diag2_g_ev = np.array(
            [_icd9_group(c) for c in eval_cols["diag_2"]], dtype=object
        )
        diag3_g_tr = np.array(
            [_icd9_group(c) for c in train["diag_3"]], dtype=object
        )
        diag3_g_ev = np.array(
            [_icd9_group(c) for c in eval_cols["diag_3"]], dtype=object
        )

        extra_cat["diag_1_group"] = (diag1_g_tr, diag1_g_ev)
        extra_cat["diag_2_group"] = (diag2_g_tr, diag2_g_ev)
        extra_cat["diag_3_group"] = (diag3_g_tr, diag3_g_ev)
        extra_num["diag1_diag2_same"] = (
            (diag1_g_tr == diag2_g_tr).astype(np.float64),
            (diag1_g_ev == diag2_g_ev).astype(np.float64),
        )

    # Specific diagnosis prefixes (CHF, IHD, Pneumonia, Renal, Lyte, Acute Diabetes)
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
        is_prefix(
            train["diag_1"],
            ["480", "481", "482", "483", "484", "485", "486"],
        ),
        is_prefix(
            eval_cols["diag_1"],
            ["480", "481", "482", "483", "484", "485", "486"],
        ),
    )
    extra_num["diag1_renal"] = (
        is_prefix(train["diag_1"], ["584", "585", "586"]),
        is_prefix(eval_cols["diag_1"], ["584", "585", "586"]),
    )
    extra_num["diag1_fluid_lyte"] = (
        is_prefix(train["diag_1"], ["276"]),
        is_prefix(eval_cols["diag_1"], ["276"]),
    )
    extra_num["diag1_diab_acute"] = (
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
    extra_num["disch_is_ama"] = (
        (disch_tr == "7").astype(np.float64),
        (disch_ev == "7").astype(np.float64),
    )
    extra_num["disch_is_facility"] = (
        np.isin(disch_tr, ["2", "3", "4", "5", "22", "23", "24"]).astype(
            np.float64
        ),
        np.isin(disch_ev, ["2", "3", "4", "5", "22", "23", "24"]).astype(
            np.float64
        ),
    )

    is_emerg_adm_tr = (adm_type_tr == "1").astype(np.float64)
    is_emerg_adm_ev = (adm_type_ev == "1").astype(np.float64)
    extra_num["adm_is_emergency"] = (is_emerg_adm_tr, is_emerg_adm_ev)
    extra_num["adm_is_urgent"] = (
        (adm_type_tr == "2").astype(np.float64),
        (adm_type_ev == "2").astype(np.float64),
    )
    extra_num["adm_is_elective"] = (
        (adm_type_tr == "3").astype(np.float64),
        (adm_type_ev == "3").astype(np.float64),
    )
    extra_num["adm_src_er"] = (
        (adm_src_tr == "7").astype(np.float64),
        (adm_src_ev == "7").astype(np.float64),
    )

    # 4. Out-of-fold Target Encoding for Categoricals and Combinations
    if _oof_target_encode is not None:
        te_spec = [
            ("diag_1", 15.0, 42),
            ("diag_2", 15.0, 43),
            ("diag_3", 15.0, 44),
            ("discharge_disposition_id", 10.0, 45),
            ("admission_source_id", 15.0, 46),
            ("admission_type_id", 20.0, 47),
            ("medical_specialty", 25.0, 48),
            ("payer_code", 25.0, 49),
        ]
        for col_name, smoothing_val, seed_v in te_spec:
            if col_name in train:
                tr_te, ev_te = _oof_target_encode(
                    train[col_name],
                    y_train,
                    eval_cols[col_name],
                    n_folds=5,
                    smoothing=smoothing_val,
                    seed=seed_v,
                )
                extra_num[f"te_{col_name}"] = (
                    tr_te.astype(np.float64),
                    ev_te.astype(np.float64),
                )

        # 3-digit diagnosis target encodings
        for name, tr_col, ev_col in [
            ("diag_1_3digit", diag1_3_tr, diag1_3_ev),
            ("diag_2_3digit", diag2_3_tr, diag2_3_ev),
            ("diag_3_3digit", diag3_3_tr, diag3_3_ev),
        ]:
            tr_te, ev_te = _oof_target_encode(
                tr_col, y_train, ev_col, n_folds=5, smoothing=20.0, seed=51
            )
            extra_num[f"te_{name}"] = (
                tr_te.astype(np.float64),
                ev_te.astype(np.float64),
            )

        # Trajectory combination: discharge disposition + admission type
        da_tr = np.array(
            [f"{d}_{a}" for d, a in zip(disch_tr, adm_type_tr)], dtype=object
        )
        da_ev = np.array(
            [f"{d}_{a}" for d, a in zip(disch_ev, adm_type_ev)], dtype=object
        )
        tr_te_da, ev_te_da = _oof_target_encode(
            da_tr, y_train, da_ev, n_folds=5, smoothing=15.0, seed=52
        )
        extra_num["te_disch_adm_combo"] = (
            tr_te_da.astype(np.float64),
            ev_te_da.astype(np.float64),
        )

        # Interaction combination: diag1 group + discharge disposition
        dd_tr = np.array(
            [f"{g}_{d}" for g, d in zip(diag1_g_tr, disch_tr)], dtype=object
        )
        dd_ev = np.array(
            [f"{g}_{d}" for g, d in zip(diag1_g_ev, disch_ev)], dtype=object
        )
        tr_te_dd, ev_te_dd = _oof_target_encode(
            dd_tr, y_train, dd_ev, n_folds=5, smoothing=20.0, seed=53
        )
        extra_num["te_diag1_disch_combo"] = (
            tr_te_dd.astype(np.float64),
            ev_te_dd.astype(np.float64),
        )

    # 5. Frequency Encoding for High-Cardinality Categoricals
    def freq_encode(tr_arr, ev_arr):
        counts = {}
        for v in tr_arr:
            counts[v] = counts.get(v, 0) + 1
        n = float(len(tr_arr))
        tr_freq = np.array(
            [counts.get(v, 0) / n for v in tr_arr], dtype=np.float64
        )
        ev_freq = np.array(
            [counts.get(v, 0) / n for v in ev_arr], dtype=np.float64
        )
        return tr_freq, ev_freq

    f_med_tr, f_med_ev = freq_encode(
        train["medical_specialty"], eval_cols["medical_specialty"]
    )
    extra_num["freq_medical_specialty"] = (f_med_tr, f_med_ev)

    f_disch_tr, f_disch_ev = freq_encode(disch_tr, disch_ev)
    extra_num["freq_discharge_disposition"] = (f_disch_tr, f_disch_ev)

    f_d1_tr, f_d1_ev = freq_encode(diag1_3_tr, diag1_3_ev)
    extra_num["freq_diag1_3digit"] = (f_d1_tr, f_d1_ev)

    # 6. Demographics & Age Continuous Midpoint + Geriatric Interactions
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
    age_tr = np.array(
        [age_map.get(str(a), 65.0) for a in train["age"]], dtype=np.float64
    )
    age_ev = np.array(
        [age_map.get(str(a), 65.0) for a in eval_cols["age"]], dtype=np.float64
    )
    extra_num["age_num"] = (age_tr, age_ev)
    is_eld_tr = (age_tr >= 75.0).astype(np.float64)
    is_eld_ev = (age_ev >= 75.0).astype(np.float64)
    extra_num["is_elderly"] = (is_eld_tr, is_eld_ev)

    # 7. Prior Healthcare Utilization & Ratios
    inp_tr = np.nan_to_num(train["number_inpatient"], nan=0.0)
    inp_ev = np.nan_to_num(eval_cols["number_inpatient"], nan=0.0)
    emg_tr = np.nan_to_num(train["number_emergency"], nan=0.0)
    emg_ev = np.nan_to_num(eval_cols["number_emergency"], nan=0.0)
    out_tr = np.nan_to_num(train["number_outpatient"], nan=0.0)
    out_ev = np.nan_to_num(eval_cols["number_outpatient"], nan=0.0)

    tot_vis_tr = inp_tr + emg_tr + out_tr
    tot_vis_ev = inp_ev + emg_ev + out_ev
    extra_num["total_prior_visits"] = (tot_vis_tr, tot_vis_ev)
    extra_num["has_prior_inpatient"] = (
        (inp_tr > 0.0).astype(np.float64),
        (inp_ev > 0.0).astype(np.float64),
    )
    extra_num["has_prior_emergency"] = (
        (emg_tr > 0.0).astype(np.float64),
        (emg_ev > 0.0).astype(np.float64),
    )
    extra_num["has_any_prior_visit"] = (
        (tot_vis_tr > 0.0).astype(np.float64),
        (tot_vis_ev > 0.0).astype(np.float64),
    )
    extra_num["frequent_inpatient"] = (
        (inp_tr >= 2.0).astype(np.float64),
        (inp_ev >= 2.0).astype(np.float64),
    )
    extra_num["log_inpatient"] = (np.log1p(inp_tr), np.log1p(inp_ev))
    extra_num["log_total_visits"] = (np.log1p(tot_vis_tr), np.log1p(tot_vis_ev))
    extra_num["inpatient_ratio"] = (
        (inp_tr + 0.1) / (tot_vis_tr + 1.0),
        (inp_ev + 0.1) / (tot_vis_ev + 1.0),
    )
    extra_num["emergency_ratio"] = (
        (emg_tr + 0.1) / (tot_vis_tr + 1.0),
        (emg_ev + 0.1) / (tot_vis_ev + 1.0),
    )
    extra_num["inpatient_squared"] = (inp_tr**2, inp_ev**2)
    extra_num["prior_util_weighted"] = (
        inp_tr * 3.0 + emg_tr * 1.5 + out_tr * 0.5,
        inp_ev * 3.0 + emg_ev * 1.5 + out_ev * 0.5,
    )
    extra_num["high_utilizer"] = (
        ((inp_tr >= 2.0) | (tot_vis_tr >= 5.0)).astype(np.float64),
        ((inp_ev >= 2.0) | (tot_vis_ev >= 5.0)).astype(np.float64),
    )

    # 8. Stay Intensity, Clinical Events & LACE Index
    t_hosp_tr = np.nan_to_num(train["time_in_hospital"], nan=1.0)
    t_hosp_ev = np.nan_to_num(eval_cols["time_in_hospital"], nan=1.0)
    n_med_tr = np.nan_to_num(train["num_medications"], nan=0.0)
    n_med_ev = np.nan_to_num(eval_cols["num_medications"], nan=0.0)
    n_lab_tr = np.nan_to_num(train["num_lab_procedures"], nan=0.0)
    n_lab_ev = np.nan_to_num(eval_cols["num_lab_procedures"], nan=0.0)
    n_proc_tr = np.nan_to_num(train["num_procedures"], nan=0.0)
    n_proc_ev = np.nan_to_num(eval_cols["num_procedures"], nan=0.0)
    n_diag_tr = np.nan_to_num(train["number_diagnoses"], nan=0.0)
    n_diag_ev = np.nan_to_num(eval_cols["number_diagnoses"], nan=0.0)

    extra_num["med_per_day"] = (
        n_med_tr / (t_hosp_tr + 1.0),
        n_med_ev / (t_hosp_ev + 1.0),
    )
    extra_num["lab_per_day"] = (
        n_lab_tr / (t_hosp_tr + 1.0),
        n_lab_ev / (t_hosp_ev + 1.0),
    )
    extra_num["proc_per_day"] = (
        n_proc_tr / (t_hosp_tr + 1.0),
        n_proc_ev / (t_hosp_ev + 1.0),
    )
    extra_num["diag_per_day"] = (
        n_diag_tr / (t_hosp_tr + 1.0),
        n_diag_ev / (t_hosp_ev + 1.0),
    )
    extra_num["inpatient_per_day"] = (
        inp_tr / (t_hosp_tr + 1.0),
        inp_ev / (t_hosp_ev + 1.0),
    )
    extra_num["intensity_index"] = (
        np.log1p(n_med_tr * n_lab_tr) / (t_hosp_tr + 1.0),
        np.log1p(n_med_ev * n_lab_ev) / (t_hosp_ev + 1.0),
    )
    extra_num["tot_clinical_events"] = (
        n_lab_tr + n_med_tr + n_proc_tr,
        n_lab_ev + n_med_ev + n_proc_ev,
    )
    extra_num["events_per_day"] = (
        (n_lab_tr + n_med_tr + n_proc_tr) / (t_hosp_tr + 1.0),
        (n_lab_ev + n_med_ev + n_proc_ev) / (t_hosp_ev + 1.0),
    )
    extra_num["has_procedures"] = (
        (n_proc_tr > 0.0).astype(np.float64),
        (n_proc_ev > 0.0).astype(np.float64),
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

    lace_tr = compute_lace(t_hosp_tr, is_emerg_adm_tr, cci_tr, emg_tr)
    lace_ev = compute_lace(t_hosp_ev, is_emerg_adm_ev, cci_ev, emg_ev)
    extra_num["lace_index"] = (lace_tr, lace_ev)

    # Geriatric interactions
    if _icd9_comorbidities is not None:
        extra_num["elderly_dementia"] = (
            is_eld_tr * cm_tr["cm_dementia"],
            is_eld_ev * cm_ev["cm_dementia"],
        )
    extra_num["elderly_snf"] = (
        is_eld_tr * (disch_tr == "3").astype(np.float64),
        is_eld_ev * (disch_ev == "3").astype(np.float64),
    )
    extra_num["elderly_inpatient"] = (is_eld_tr * inp_tr, is_eld_ev * inp_ev)
    extra_num["age_x_inpatient"] = (
        (age_tr / 50.0) * np.log1p(inp_tr),
        (age_ev / 50.0) * np.log1p(inp_ev),
    )
    extra_num["age_x_cci"] = (
        (age_tr / 50.0) * cci_tr,
        (age_ev / 50.0) * cci_ev,
    )
    extra_num["age_x_los"] = (
        (age_tr / 50.0) * t_hosp_tr,
        (age_ev / 50.0) * t_hosp_ev,
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
    extra_num["a1c_tested"] = (
        (train["A1Cresult"] != "None").astype(np.float64),
        (eval_cols["A1Cresult"] != "None").astype(np.float64),
    )
    extra_num["a1c_high"] = (
        np.isin(train["A1Cresult"], [">7", ">8"]).astype(np.float64),
        np.isin(eval_cols["A1Cresult"], [">7", ">8"]).astype(np.float64),
    )
    extra_num["glu_tested"] = (
        (train["max_glu_serum"] != "None").astype(np.float64),
        (eval_cols["max_glu_serum"] != "None").astype(np.float64),
    )
    extra_num["glu_high"] = (
        np.isin(train["max_glu_serum"], [">200", ">300"]).astype(np.float64),
        np.isin(eval_cols["max_glu_serum"], [">200", ">300"]).astype(
            np.float64
        ),
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

    # Model 1: Balanced LightGBM
    m1 = lgb.LGBMClassifier(
        n_estimators=360,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=2.0,
        reg_alpha=0.2,
        cat_smooth=30.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    m1.fit(x_train, y_train, categorical_feature=cat_indices)
    p1 = m1.predict_proba(x_eval)[:, 1]

    # Model 2: Deeper, High-Interaction LightGBM
    m2 = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.025,
        num_leaves=45,
        max_depth=7,
        min_child_samples=90,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=3.0,
        reg_alpha=0.5,
        cat_smooth=25.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    m2.fit(x_train, y_train, categorical_feature=cat_indices)
    p2 = m2.predict_proba(x_eval)[:, 1]

    # Model 3: Conservative, Regularized Shallow LightGBM
    m3 = lgb.LGBMClassifier(
        n_estimators=380,
        learning_rate=0.025,
        num_leaves=18,
        max_depth=5,
        min_child_samples=70,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=5.0,
        reg_alpha=1.0,
        cat_smooth=45.0,
        random_state=777,
        n_jobs=1,
        verbose=-1,
    )
    m3.fit(x_train, y_train, categorical_feature=cat_indices)
    p3 = m3.predict_proba(x_eval)[:, 1]

    # Model 4: Extremely Randomized Trees LightGBM (ExtraTrees for structural variance reduction)
    try:
        m4 = lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.03,
            extra_trees=True,
            num_leaves=35,
            min_child_samples=100,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=2.5,
            reg_alpha=0.3,
            cat_smooth=35.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        )
        m4.fit(x_train, y_train, categorical_feature=cat_indices)
        p4 = m4.predict_proba(x_eval)[:, 1]
    except Exception:
        m4 = lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.03,
            num_leaves=35,
            min_child_samples=100,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=2.5,
            reg_alpha=0.3,
            cat_smooth=35.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        )
        m4.fit(x_train, y_train, categorical_feature=cat_indices)
        p4 = m4.predict_proba(x_eval)[:, 1]

    # Model 5: Complementary XGBoost on Numeric + Target Encoded features
    p_xgb = None
    if xgb is not None:
        try:
            cat_set = set(cat_indices)
            num_indices = [
                i for i in range(x_train.shape[1]) if i not in cat_set
            ]
            x_train_num = x_train[:, num_indices]
            x_eval_num = x_eval[:, num_indices]
            m_xgb = xgb.XGBClassifier(
                n_estimators=260,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=8,
                subsample=0.80,
                colsample_bytree=0.60,
                reg_lambda=4.0,
                reg_alpha=0.6,
                random_state=42,
                n_jobs=1,
                verbosity=0,
                eval_metric="logloss",
            )
            m_xgb.fit(x_train_num, y_train)
            p_xgb = m_xgb.predict_proba(x_eval_num)[:, 1]
        except Exception:
            p_xgb = None

    if p_xgb is not None:
        preds = 0.28 * p1 + 0.24 * p2 + 0.20 * p3 + 0.16 * p4 + 0.12 * p_xgb
    else:
        preds = 0.32 * p1 + 0.28 * p2 + 0.22 * p3 + 0.18 * p4

    preds = np.nan_to_num(preds, nan=0.114)
    return np.clip(preds, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
