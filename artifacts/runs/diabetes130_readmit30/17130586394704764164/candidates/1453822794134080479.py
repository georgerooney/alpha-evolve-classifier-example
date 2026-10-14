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
    from scipy.stats import rankdata

    # Safe lookup of toolkit utilities
    fn_comorbidities = globals().get("icd9_comorbidities", None)
    fn_group = globals().get("icd9_group", None)
    fn_te = globals().get("oof_target_encode", None)
    if fn_comorbidities is None or fn_group is None or fn_te is None:
        try:
            import clinical_toolkit

            if fn_comorbidities is None:
                fn_comorbidities = getattr(
                    clinical_toolkit, "icd9_comorbidities", None
                )
            if fn_group is None:
                fn_group = getattr(clinical_toolkit, "icd9_group", None)
            if fn_te is None:
                fn_te = getattr(clinical_toolkit, "oof_target_encode", None)
        except Exception:
            pass

    def target_encode(
        tr_col: np.ndarray,
        y_tr: np.ndarray,
        ev_col: np.ndarray,
        n_folds: int = 5,
        smoothing: float = 25.0,
        seed: int = 42,
    ):
        if fn_te is not None:
            try:
                return fn_te(
                    tr_col,
                    y_tr,
                    ev_col,
                    n_folds=n_folds,
                    smoothing=smoothing,
                    seed=seed,
                )
            except Exception:
                pass
        # Robust fallback out-of-fold target encoding
        global_mean = float(np.mean(y_tr))
        tr_enc = np.full(len(tr_col), global_mean, dtype=np.float64)
        ev_enc = np.full(len(ev_col), global_mean, dtype=np.float64)
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for tr_idx, val_idx in kf.split(tr_col):
            counts, sums = {}, {}
            for idx in tr_idx:
                val = tr_col[idx]
                counts[val] = counts.get(val, 0) + 1
                sums[val] = sums.get(val, 0.0) + y_tr[idx]
            for idx in val_idx:
                val = tr_col[idx]
                n = counts.get(val, 0)
                s = sums.get(val, 0.0)
                tr_enc[idx] = (s + smoothing * global_mean) / (n + smoothing)

        total_counts, total_sums = {}, {}
        for idx in range(len(tr_col)):
            val = tr_col[idx]
            total_counts[val] = total_counts.get(val, 0) + 1
            total_sums[val] = total_sums.get(val, 0.0) + y_tr[idx]
        for idx in range(len(ev_col)):
            val = ev_col[idx]
            n = total_counts.get(val, 0)
            s = total_sums.get(val, 0.0)
            ev_enc[idx] = (s + smoothing * global_mean) / (n + smoothing)
        return tr_enc, ev_enc

    def extract_icd9_prefix(val):
        if val is None:
            return "missing"
        s = str(val).strip()
        if s in ("", "?", "None", "nan", "NaN"):
            return "missing"
        if s.startswith(("V", "E", "v", "e")):
            return s[:3].upper()
        dot_idx = s.find(".")
        prefix = s[:dot_idx] if dot_idx != -1 else s[:3]
        return prefix[:3]

    train_copy = dict(train)
    eval_copy = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    n_tr = len(y_train)
    n_ev = len(eval_cols[task.numeric_columns[0]])

    # 1. Charlson-style Comorbidity Mapping
    if fn_comorbidities is not None and "diag_1" in train and "diag_2" in train:
        try:
            d1_tr, d2_tr = train["diag_1"], train["diag_2"]
            d3_tr = train.get("diag_3", np.full(len(d1_tr), None, dtype=object))
            d1_ev, d2_ev = eval_cols["diag_1"], eval_cols["diag_2"]
            d3_ev = eval_cols.get(
                "diag_3", np.full(len(d1_ev), None, dtype=object)
            )

            cm_tr = fn_comorbidities(d1_tr, d2_tr, d3_tr)
            cm_ev = fn_comorbidities(d1_ev, d2_ev, d3_ev)
            for k in cm_tr:
                train_copy[k] = np.asarray(cm_tr[k], dtype=np.float64)
                eval_copy[k] = np.asarray(cm_ev[k], dtype=np.float64)
                extra_numeric.append(k)
        except Exception:
            pass

    # 2. Cardiorenal-Metabolic Disease Cluster & Multimorbidity Synergies
    cm_chf_tr = train_copy.get("cm_chf", np.zeros(n_tr, dtype=np.float64))
    cm_chf_ev = eval_copy.get("cm_chf", np.zeros(n_ev, dtype=np.float64))
    cm_renal_tr = train_copy.get("cm_renal", np.zeros(n_tr, dtype=np.float64))
    cm_renal_ev = eval_copy.get("cm_renal", np.zeros(n_ev, dtype=np.float64))
    cm_diab_tr = train_copy.get(
        "cm_diabetes_complicated", np.zeros(n_tr, dtype=np.float64)
    )
    cm_diab_ev = eval_copy.get(
        "cm_diabetes_complicated", np.zeros(n_ev, dtype=np.float64)
    )
    cm_pulm_tr = train_copy.get(
        "cm_pulmonary", np.zeros(n_tr, dtype=np.float64)
    )
    cm_pulm_ev = eval_copy.get(
        "cm_pulmonary", np.zeros(n_ev, dtype=np.float64)
    )
    cm_mi_tr = train_copy.get("cm_mi", np.zeros(n_tr, dtype=np.float64))
    cm_mi_ev = eval_copy.get("cm_mi", np.zeros(n_ev, dtype=np.float64))
    cm_pvd_tr = train_copy.get("cm_pvd", np.zeros(n_tr, dtype=np.float64))
    cm_pvd_ev = eval_copy.get("cm_pvd", np.zeros(n_ev, dtype=np.float64))
    cm_cvd_tr = train_copy.get(
        "cm_cerebrovascular", np.zeros(n_tr, dtype=np.float64)
    )
    cm_cvd_ev = eval_copy.get(
        "cm_cerebrovascular", np.zeros(n_ev, dtype=np.float64)
    )
    cm_cancer_tr = train_copy.get("cm_cancer", np.zeros(n_tr, dtype=np.float64))
    cm_cancer_ev = eval_copy.get("cm_cancer", np.zeros(n_ev, dtype=np.float64))

    train_copy["cardiorenal_syndrome"] = cm_chf_tr * cm_renal_tr
    eval_copy["cardiorenal_syndrome"] = cm_chf_ev * cm_renal_ev
    extra_numeric.append("cardiorenal_syndrome")

    train_copy["cardiometabolic_syndrome"] = cm_chf_tr * cm_diab_tr
    eval_copy["cardiometabolic_syndrome"] = cm_chf_ev * cm_diab_ev
    extra_numeric.append("cardiometabolic_syndrome")

    train_copy["nephrometabolic_syndrome"] = cm_renal_tr * cm_diab_tr
    eval_copy["nephrometabolic_syndrome"] = cm_renal_ev * cm_diab_ev
    extra_numeric.append("nephrometabolic_syndrome")

    train_copy["vascular_burden"] = (
        cm_mi_tr + cm_chf_tr + cm_pvd_tr + cm_cvd_tr
    )
    eval_copy["vascular_burden"] = (
        cm_mi_ev + cm_chf_ev + cm_pvd_ev + cm_cvd_ev
    )
    extra_numeric.append("vascular_burden")

    train_copy["high_risk_cm_score"] = (
        cm_chf_tr + cm_renal_tr + cm_pulm_tr + cm_diab_tr + cm_cancer_tr
    )
    eval_copy["high_risk_cm_score"] = (
        cm_chf_ev + cm_renal_ev + cm_pulm_ev + cm_diab_ev + cm_cancer_ev
    )
    extra_numeric.append("high_risk_cm_score")


    # 3. ICD-9 Chapter Grouping & 3-Digit Category Hierarchy
    if fn_group is not None:
        try:
            for diag_name in ["diag_1", "diag_2", "diag_3"]:
                if diag_name in train and diag_name in eval_cols:
                    grp_key = f"{diag_name}_grp"
                    train_copy[grp_key] = np.array(
                        [fn_group(c) for c in train[diag_name]], dtype=object
                    )
                    eval_copy[grp_key] = np.array(
                        [fn_group(c) for c in eval_cols[diag_name]],
                        dtype=object,
                    )
                    extra_categorical.append(grp_key)
        except Exception:
            pass


    # Extract 3-digit ICD-9 disease categories
    d1_pfx_tr = np.array(
        [extract_icd9_prefix(x) for x in train.get("diag_1", [])], dtype=object
    )
    d1_pfx_ev = np.array(
        [extract_icd9_prefix(x) for x in eval_cols.get("diag_1", [])],
        dtype=object,
    )
    d2_pfx_tr = np.array(
        [extract_icd9_prefix(x) for x in train.get("diag_2", [])], dtype=object
    )
    d2_pfx_ev = np.array(
        [extract_icd9_prefix(x) for x in eval_cols.get("diag_2", [])],
        dtype=object,
    )
    d3_pfx_tr = np.array(
        [extract_icd9_prefix(x) for x in train.get("diag_3", [])], dtype=object
    )
    d3_pfx_ev = np.array(
        [extract_icd9_prefix(x) for x in eval_cols.get("diag_3", [])],
        dtype=object,
    )

    train_copy["diag_1_3digit"] = d1_pfx_tr
    eval_copy["diag_1_3digit"] = d1_pfx_ev
    extra_categorical.append("diag_1_3digit")

    train_copy["diag_2_3digit"] = d2_pfx_tr
    eval_copy["diag_2_3digit"] = d2_pfx_ev
    extra_categorical.append("diag_2_3digit")

    train_copy["diag_3_3digit"] = d3_pfx_tr
    eval_copy["diag_3_3digit"] = d3_pfx_ev
    extra_categorical.append("diag_3_3digit")

    # High-risk clinical phenotype indicators
    train_copy["has_diag_chf"] = (
        (d1_pfx_tr == "428") | (d2_pfx_tr == "428") | (d3_pfx_tr == "428")
    ).astype(np.float64)
    eval_copy["has_diag_chf"] = (
        (d1_pfx_ev == "428") | (d2_pfx_ev == "428") | (d3_pfx_ev == "428")
    ).astype(np.float64)
    extra_numeric.append("has_diag_chf")

    train_copy["has_diag_renal"] = (
        np.isin(d1_pfx_tr, ["585", "586"])
        | np.isin(d2_pfx_tr, ["585", "586"])
        | np.isin(d3_pfx_tr, ["585", "586"])
    ).astype(np.float64)
    eval_copy["has_diag_renal"] = (
        np.isin(d1_pfx_ev, ["585", "586"])
        | np.isin(d2_pfx_ev, ["585", "586"])
        | np.isin(d3_pfx_ev, ["585", "586"])
    ).astype(np.float64)
    extra_numeric.append("has_diag_renal")

    train_copy["has_diag_cad"] = (
        np.isin(d1_pfx_tr, ["410", "411", "412", "413", "414"])
        | np.isin(d2_pfx_tr, ["410", "411", "412", "413", "414"])
        | np.isin(d3_pfx_tr, ["410", "411", "412", "413", "414"])
    ).astype(np.float64)
    eval_copy["has_diag_cad"] = (
        np.isin(d1_pfx_ev, ["410", "411", "412", "413", "414"])
        | np.isin(d2_pfx_ev, ["410", "411", "412", "413", "414"])
        | np.isin(d3_pfx_ev, ["410", "411", "412", "413", "414"])
    ).astype(np.float64)
    extra_numeric.append("has_diag_cad")

    train_copy["primary_is_diabetes"] = (d1_pfx_tr == "250").astype(np.float64)
    eval_copy["primary_is_diabetes"] = (d1_pfx_ev == "250").astype(np.float64)
    extra_numeric.append("primary_is_diabetes")

    train_copy["primary_is_chf"] = (d1_pfx_tr == "428").astype(np.float64)
    eval_copy["primary_is_chf"] = (d1_pfx_ev == "428").astype(np.float64)
    extra_numeric.append("primary_is_chf")

    # High-Risk HRRP Conditions: COPD, Pneumonia, Sepsis
    copd_codes = ["491", "492", "493", "494", "496"]
    train_copy["has_diag_copd"] = (
        np.isin(d1_pfx_tr, copd_codes)
        | np.isin(d2_pfx_tr, copd_codes)
        | np.isin(d3_pfx_tr, copd_codes)
    ).astype(np.float64)
    eval_copy["has_diag_copd"] = (
        np.isin(d1_pfx_ev, copd_codes)
        | np.isin(d2_pfx_ev, copd_codes)
        | np.isin(d3_pfx_ev, copd_codes)
    ).astype(np.float64)
    extra_numeric.append("has_diag_copd")

    pna_codes = ["480", "481", "482", "483", "484", "485", "486"]
    train_copy["has_diag_pneumonia"] = (
        np.isin(d1_pfx_tr, pna_codes)
        | np.isin(d2_pfx_tr, pna_codes)
        | np.isin(d3_pfx_tr, pna_codes)
    ).astype(np.float64)
    eval_copy["has_diag_pneumonia"] = (
        np.isin(d1_pfx_ev, pna_codes)
        | np.isin(d2_pfx_ev, pna_codes)
        | np.isin(d3_pfx_ev, pna_codes)
    ).astype(np.float64)
    extra_numeric.append("has_diag_pneumonia")

    sepsis_codes = ["038", "995"]
    train_copy["has_diag_sepsis"] = (
        np.isin(d1_pfx_tr, sepsis_codes)
        | np.isin(d2_pfx_tr, sepsis_codes)
        | np.isin(d3_pfx_tr, sepsis_codes)
    ).astype(np.float64)
    eval_copy["has_diag_sepsis"] = (
        np.isin(d1_pfx_ev, sepsis_codes)
        | np.isin(d2_pfx_ev, sepsis_codes)
        | np.isin(d3_pfx_ev, sepsis_codes)
    ).astype(np.float64)
    extra_numeric.append("has_diag_sepsis")

    train_copy["cardiopulmonary_syndrome"] = cm_chf_tr * train_copy["has_diag_copd"]
    eval_copy["cardiopulmonary_syndrome"] = cm_chf_ev * eval_copy["has_diag_copd"]
    extra_numeric.append("cardiopulmonary_syndrome")

    # 4. Admission-to-Discharge Trajectory Transitions
    adm_src_tr = train.get("admission_source_id", np.array([""] * n_tr, dtype=object))
    adm_src_ev = eval_cols.get("admission_source_id", np.array([""] * n_ev, dtype=object))
    adm_type_tr = train.get("admission_type_id", np.array([""] * n_tr, dtype=object))
    adm_type_ev = eval_cols.get("admission_type_id", np.array([""] * n_ev, dtype=object))
    dd_tr = train.get("discharge_disposition_id", np.array([""] * n_tr, dtype=object))
    dd_ev = eval_cols.get("discharge_disposition_id", np.array([""] * n_ev, dtype=object))

    src_disch_tr = np.array([f"{s}_{d}" for s, d in zip(adm_src_tr, dd_tr)], dtype=object)
    src_disch_ev = np.array([f"{s}_{d}" for s, d in zip(adm_src_ev, dd_ev)], dtype=object)
    type_disch_tr = np.array([f"{t}_{d}" for t, d in zip(adm_type_tr, dd_tr)], dtype=object)
    type_disch_ev = np.array([f"{t}_{d}" for t, d in zip(adm_type_ev, dd_ev)], dtype=object)

    train_copy["src_disch_cross"] = src_disch_tr
    eval_copy["src_disch_cross"] = src_disch_ev
    extra_categorical.append("src_disch_cross")

    train_copy["type_disch_cross"] = type_disch_tr
    eval_copy["type_disch_cross"] = type_disch_ev
    extra_categorical.append("type_disch_cross")

    # Joint Primary Diagnosis Category x Discharge Disposition Trajectory
    diag1_dd_tr = np.array([f"{p}_{d}" for p, d in zip(d1_pfx_tr, dd_tr)], dtype=object)
    diag1_dd_ev = np.array([f"{p}_{d}" for p, d in zip(d1_pfx_ev, dd_ev)], dtype=object)
    train_copy["diag1_disch_cross"] = diag1_dd_tr
    eval_copy["diag1_disch_cross"] = diag1_dd_ev
    extra_categorical.append("diag1_disch_cross")

    # 5. Out-of-Fold Target Encoding (Single & Joint Interactions)
    te_target_cols = [
        "discharge_disposition_id",
        "medical_specialty",
        "admission_source_id",
        "admission_type_id",
        "diag_1",
        "diag_2",
        "diag_3",
        "diag_1_3digit",
        "diag_2_3digit",
        "diag_3_3digit",
        "payer_code",
        "src_disch_cross",
        "type_disch_cross",
        "diag1_disch_cross",
    ]
    for col in te_target_cols:
        if col in train_copy and col in eval_copy:
            tr_te, ev_te = target_encode(
                train_copy[col],
                y_train,
                eval_copy[col],
                n_folds=5,
                smoothing=25.0,
                seed=42,
            )
            te_col_name = f"te_{col}"
            train_copy[te_col_name] = tr_te
            eval_copy[te_col_name] = ev_te
            extra_numeric.append(te_col_name)

    # 6. Frequency Density Encoding
    freq_cols = [
        "discharge_disposition_id",
        "medical_specialty",
        "admission_source_id",
        "admission_type_id",
        "payer_code",
        "diag_1_3digit",
        "diag_2_3digit",
        "diag_3_3digit",
    ]
    for fc in freq_cols:
        if fc in train_copy and fc in eval_copy:
            tr_vals = train_copy[fc]
            ev_vals = eval_copy[fc]
            counts = {}
            for v in tr_vals:
                counts[v] = counts.get(v, 0) + 1
            n_total = float(len(tr_vals))
            freq_name = f"freq_{fc}"
            train_copy[freq_name] = np.array(
                [counts.get(v, 0) / n_total for v in tr_vals], dtype=np.float64
            )
            eval_copy[freq_name] = np.array(
                [counts.get(v, 0) / n_total for v in ev_vals], dtype=np.float64
            )
            extra_numeric.append(freq_name)

    # 7. Prior Utilisation Signals & Hospital Score Proxy
    n_in_tr = np.nan_to_num(train.get("number_inpatient", 0.0), nan=0.0)
    n_in_ev = np.nan_to_num(eval_cols.get("number_inpatient", 0.0), nan=0.0)
    n_em_tr = np.nan_to_num(train.get("number_emergency", 0.0), nan=0.0)
    n_em_ev = np.nan_to_num(eval_cols.get("number_emergency", 0.0), nan=0.0)
    n_op_tr = np.nan_to_num(train.get("number_outpatient", 0.0), nan=0.0)
    n_op_ev = np.nan_to_num(eval_cols.get("number_outpatient", 0.0), nan=0.0)

    tot_tr = n_in_tr + n_em_tr + n_op_tr
    tot_ev = n_in_ev + n_em_ev + n_op_ev

    train_copy["total_prior_visits"] = tot_tr
    eval_copy["total_prior_visits"] = tot_ev
    extra_numeric.append("total_prior_visits")

    train_copy["has_prior_inpatient"] = (n_in_tr > 0).astype(np.float64)
    eval_copy["has_prior_inpatient"] = (n_in_ev > 0).astype(np.float64)
    extra_numeric.append("has_prior_inpatient")

    train_copy["has_multiple_inpatient"] = (n_in_tr >= 2).astype(np.float64)
    eval_copy["has_multiple_inpatient"] = (n_in_ev >= 2).astype(np.float64)
    extra_numeric.append("has_multiple_inpatient")

    train_copy["has_prior_emergency"] = (n_em_tr > 0).astype(np.float64)
    eval_copy["has_prior_emergency"] = (n_em_ev > 0).astype(np.float64)
    extra_numeric.append("has_prior_emergency")

    train_copy["has_multiple_emergency"] = (n_em_tr >= 2).astype(np.float64)
    eval_copy["has_multiple_emergency"] = (n_em_ev >= 2).astype(np.float64)
    extra_numeric.append("has_multiple_emergency")

    train_copy["zero_prior_visits"] = (tot_tr == 0).astype(np.float64)
    eval_copy["zero_prior_visits"] = (tot_ev == 0).astype(np.float64)
    extra_numeric.append("zero_prior_visits")

    train_copy["inpatient_ratio"] = n_in_tr / (tot_tr + 1.0)
    eval_copy["inpatient_ratio"] = n_in_ev / (tot_ev + 1.0)
    extra_numeric.append("inpatient_ratio")

    train_copy["acute_prior_visits"] = n_in_tr + n_em_tr
    eval_copy["acute_prior_visits"] = n_in_ev + n_em_ev
    extra_numeric.append("acute_prior_visits")

    train_copy["log_inpatient"] = np.log1p(np.maximum(0.0, n_in_tr))
    eval_copy["log_inpatient"] = np.log1p(np.maximum(0.0, n_in_ev))
    extra_numeric.append("log_inpatient")

    train_copy["log_total_prior"] = np.log1p(np.maximum(0.0, tot_tr))
    eval_copy["log_total_prior"] = np.log1p(np.maximum(0.0, tot_ev))
    extra_numeric.append("log_total_prior")

    # 8. Encounter Intensity & Hospital Burden
    t_hosp_tr = np.maximum(
        1.0, np.nan_to_num(train.get("time_in_hospital", 1.0), nan=1.0)
    )
    t_hosp_ev = np.maximum(
        1.0, np.nan_to_num(eval_cols.get("time_in_hospital", 1.0), nan=1.0)
    )

    n_lab_tr = np.nan_to_num(train.get("num_lab_procedures", 0.0), nan=0.0)
    n_lab_ev = np.nan_to_num(eval_cols.get("num_lab_procedures", 0.0), nan=0.0)
    n_proc_tr = np.nan_to_num(train.get("num_procedures", 0.0), nan=0.0)
    n_proc_ev = np.nan_to_num(eval_cols.get("num_procedures", 0.0), nan=0.0)
    n_med_tr = np.nan_to_num(train.get("num_medications", 0.0), nan=0.0)
    n_med_ev = np.nan_to_num(eval_cols.get("num_medications", 0.0), nan=0.0)
    n_diag_tr = np.nan_to_num(train.get("number_diagnoses", 0.0), nan=0.0)
    n_diag_ev = np.nan_to_num(eval_cols.get("number_diagnoses", 0.0), nan=0.0)

    train_copy["labs_per_day"] = n_lab_tr / t_hosp_tr
    eval_copy["labs_per_day"] = n_lab_ev / t_hosp_ev
    extra_numeric.append("labs_per_day")

    train_copy["meds_per_day"] = n_med_tr / t_hosp_tr
    eval_copy["meds_per_day"] = n_med_ev / t_hosp_ev
    extra_numeric.append("meds_per_day")

    train_copy["procs_per_day"] = n_proc_tr / t_hosp_tr
    eval_copy["procs_per_day"] = n_proc_ev / t_hosp_ev
    extra_numeric.append("procs_per_day")

    train_copy["diags_per_day"] = n_diag_tr / t_hosp_tr
    eval_copy["diags_per_day"] = n_diag_ev / t_hosp_ev
    extra_numeric.append("diags_per_day")

    train_copy["service_intensity"] = (
        n_lab_tr + 5.0 * n_proc_tr + n_med_tr
    ) / t_hosp_tr
    eval_copy["service_intensity"] = (
        n_lab_ev + 5.0 * n_proc_ev + n_med_ev
    ) / t_hosp_ev
    extra_numeric.append("service_intensity")

    train_copy["med_x_diag"] = n_med_tr * n_diag_tr
    eval_copy["med_x_diag"] = n_med_ev * n_diag_ev
    extra_numeric.append("med_x_diag")

    train_copy["complexity_index"] = (n_med_tr * n_lab_tr) / t_hosp_tr
    eval_copy["complexity_index"] = (n_med_ev * n_lab_ev) / t_hosp_ev
    extra_numeric.append("complexity_index")

    # HOSPITAL Score proxy
    is_urgent_tr = (
        np.isin(adm_type_tr, ["1", "2"]) | (adm_src_tr == "7")
    ).astype(np.float64)
    is_urgent_ev = (
        np.isin(adm_type_ev, ["1", "2"]) | (adm_src_ev == "7")
    ).astype(np.float64)
    los_ge_5_tr = (t_hosp_tr >= 5.0).astype(np.float64)
    los_ge_5_ev = (t_hosp_ev >= 5.0).astype(np.float64)
    prior_inp_pts_tr = np.where(
        n_in_tr >= 2.0, 5.0, np.where(n_in_tr == 1.0, 2.0, 0.0)
    )
    prior_inp_pts_ev = np.where(
        n_in_ev >= 2.0, 5.0, np.where(n_in_ev == 1.0, 2.0, 0.0)
    )

    train_copy["hospital_score_proxy"] = (
        2.0 * cm_cancer_tr
        + (n_proc_tr > 0).astype(np.float64)
        + is_urgent_tr
        + prior_inp_pts_tr
        + 2.0 * los_ge_5_tr
    )
    eval_copy["hospital_score_proxy"] = (
        2.0 * cm_cancer_ev
        + (n_proc_ev > 0).astype(np.float64)
        + is_urgent_ev
        + prior_inp_pts_ev
        + 2.0 * los_ge_5_ev
    )
    extra_numeric.append("hospital_score_proxy")

    # 9. Age Ordinal Mapping & Interactions
    age_midpoint_map = {
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
        [age_midpoint_map.get(str(x), 65.0) for x in train.get("age", [])],
        dtype=np.float64,
    )
    age_ev = np.array(
        [age_midpoint_map.get(str(x), 65.0) for x in eval_cols.get("age", [])],
        dtype=np.float64,
    )
    train_copy["age_numeric"] = age_tr
    eval_copy["age_numeric"] = age_ev
    extra_numeric.append("age_numeric")

    train_copy["age_x_inpatient"] = age_tr * n_in_tr
    eval_copy["age_x_inpatient"] = age_ev * n_in_ev
    extra_numeric.append("age_x_inpatient")

    train_copy["age_x_meds"] = age_tr * n_med_tr
    eval_copy["age_x_meds"] = age_ev * n_med_ev
    extra_numeric.append("age_x_meds")


    # 10. Medication Regimen Dynamics
    med_candidates = [
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
    avail_meds = [
        m
        for m in med_candidates
        if m in task.categorical_columns and m in train and m in eval_cols
    ]

    num_active_tr = np.zeros(n_tr, dtype=np.float64)
    num_active_ev = np.zeros(n_ev, dtype=np.float64)
    num_up_tr = np.zeros(n_tr, dtype=np.float64)
    num_up_ev = np.zeros(n_ev, dtype=np.float64)
    num_down_tr = np.zeros(n_tr, dtype=np.float64)
    num_down_ev = np.zeros(n_ev, dtype=np.float64)

    for m in avail_meds:
        tr_m = train[m]
        ev_m = eval_cols[m]
        num_active_tr += (
            (tr_m != "No") & (tr_m != None) & (tr_m != "")
        ).astype(np.float64)
        num_active_ev += (
            (ev_m != "No") & (ev_m != None) & (ev_m != "")
        ).astype(np.float64)
        num_up_tr += (tr_m == "Up").astype(np.float64)
        num_up_ev += (ev_m == "Up").astype(np.float64)
        num_down_tr += (tr_m == "Down").astype(np.float64)
        num_down_ev += (ev_m == "Down").astype(np.float64)

    train_copy["num_meds_active"] = num_active_tr
    eval_copy["num_meds_active"] = num_active_ev
    extra_numeric.append("num_meds_active")

    train_copy["num_meds_up"] = num_up_tr
    eval_copy["num_meds_up"] = num_up_ev
    extra_numeric.append("num_meds_up")

    train_copy["num_meds_down"] = num_down_tr
    eval_copy["num_meds_down"] = num_down_ev
    extra_numeric.append("num_meds_down")

    train_copy["num_meds_changed"] = num_up_tr + num_down_tr
    eval_copy["num_meds_changed"] = num_up_ev + num_down_ev
    extra_numeric.append("num_meds_changed")

    train_copy["polypharmacy_flag"] = (num_active_tr >= 3).astype(np.float64)
    eval_copy["polypharmacy_flag"] = (num_active_ev >= 3).astype(np.float64)
    extra_numeric.append("polypharmacy_flag")


    if "insulin" in train and "insulin" in eval_cols:
        ins_tr, ins_ev = train["insulin"], eval_cols["insulin"]
        train_copy["insulin_is_up"] = (ins_tr == "Up").astype(np.float64)
        eval_copy["insulin_is_up"] = (ins_ev == "Up").astype(np.float64)
        extra_numeric.append("insulin_is_up")

        train_copy["insulin_is_down"] = (ins_tr == "Down").astype(np.float64)
        eval_copy["insulin_is_down"] = (ins_ev == "Down").astype(np.float64)
        extra_numeric.append("insulin_is_down")

        train_copy["insulin_is_steady"] = (ins_tr == "Steady").astype(
            np.float64
        )
        eval_copy["insulin_is_steady"] = (ins_ev == "Steady").astype(np.float64)
        extra_numeric.append("insulin_is_steady")

        train_copy["insulin_is_active"] = (
            (ins_tr != "No") & (ins_tr != None)
        ).astype(np.float64)
        eval_copy["insulin_is_active"] = (
            (ins_ev != "No") & (ins_ev != None)
        ).astype(np.float64)
        extra_numeric.append("insulin_is_active")

        train_copy["insulin_titrated"] = (
            (ins_tr == "Up") | (ins_tr == "Down")
        ).astype(np.float64)
        eval_copy["insulin_titrated"] = (
            (ins_ev == "Up") | (ins_ev == "Down")
        ).astype(np.float64)
        extra_numeric.append("insulin_titrated")

        # Combination antidiabetic therapy (Insulin + oral agents)
        train_copy["oral_plus_insulin"] = (
            (num_active_tr >= 2) & (train_copy["insulin_is_active"] == 1.0)
        ).astype(np.float64)
        eval_copy["oral_plus_insulin"] = (
            (num_active_ev >= 2) & (eval_copy["insulin_is_active"] == 1.0)
        ).astype(np.float64)
        extra_numeric.append("oral_plus_insulin")

    if "change" in train and "change" in eval_cols:
        train_copy["med_changed_flag"] = (train["change"] == "Ch").astype(
            np.float64
        )
        eval_copy["med_changed_flag"] = (eval_cols["change"] == "Ch").astype(
            np.float64
        )
        extra_numeric.append("med_changed_flag")

    if "diabetesMed" in train and "diabetesMed" in eval_cols:
        train_copy["diabetes_med_flag"] = (
            train["diabetesMed"] == "Yes"
        ).astype(np.float64)
        eval_copy["diabetes_med_flag"] = (
            eval_cols["diabetesMed"] == "Yes"
        ).astype(np.float64)
        extra_numeric.append("diabetes_med_flag")

    # 11. High-Risk Discharge Disposition Indicators
    train_copy["disch_to_home"] = (dd_tr == "1").astype(np.float64)
    eval_copy["disch_to_home"] = (dd_ev == "1").astype(np.float64)
    extra_numeric.append("disch_to_home")

    train_copy["disch_to_snf"] = (dd_tr == "3").astype(np.float64)
    eval_copy["disch_to_snf"] = (dd_ev == "3").astype(np.float64)
    extra_numeric.append("disch_to_snf")

    train_copy["disch_to_home_health"] = (dd_tr == "6").astype(np.float64)
    eval_copy["disch_to_home_health"] = (dd_ev == "6").astype(np.float64)
    extra_numeric.append("disch_to_home_health")

    train_copy["disch_ama"] = (dd_tr == "7").astype(np.float64)
    eval_copy["disch_ama"] = (dd_ev == "7").astype(np.float64)
    extra_numeric.append("disch_ama")

    # Inpatient Utilization Interactions with Disposition & Acuity
    train_copy["inpatient_x_snf"] = n_in_tr * train_copy["disch_to_snf"]
    eval_copy["inpatient_x_snf"] = n_in_ev * eval_copy["disch_to_snf"]
    extra_numeric.append("inpatient_x_snf")

    train_copy["inpatient_x_urgent"] = n_in_tr * is_urgent_tr
    eval_copy["inpatient_x_urgent"] = n_in_ev * is_urgent_ev
    extra_numeric.append("inpatient_x_urgent")

    # 12. Glycemic Testing Flags
    if "A1Cresult" in train and "A1Cresult" in eval_cols:
        a1c_tr = train["A1Cresult"]
        a1c_ev = eval_cols["A1Cresult"]
        train_copy["a1c_tested"] = (
            (a1c_tr != "None") & (a1c_tr != None)
        ).astype(np.float64)
        eval_copy["a1c_tested"] = (
            (a1c_ev != "None") & (a1c_ev != None)
        ).astype(np.float64)
        extra_numeric.append("a1c_tested")

        train_copy["a1c_abnormal"] = (
            (a1c_tr == ">7") | (a1c_tr == ">8")
        ).astype(np.float64)
        eval_copy["a1c_abnormal"] = (
            (a1c_ev == ">7") | (a1c_ev == ">8")
        ).astype(np.float64)
        extra_numeric.append("a1c_abnormal")

    if "max_glu_serum" in train and "max_glu_serum" in eval_cols:
        glu_tr = train["max_glu_serum"]
        glu_ev = eval_cols["max_glu_serum"]
        train_copy["glu_tested"] = (
            (glu_tr != "None") & (glu_tr != None)
        ).astype(np.float64)
        eval_copy["glu_tested"] = (
            (glu_ev != "None") & (glu_ev != None)
        ).astype(np.float64)
        extra_numeric.append("glu_tested")

        train_copy["glu_abnormal"] = (
            (glu_tr == ">200") | (glu_tr == ">300")
        ).astype(np.float64)
        eval_copy["glu_abnormal"] = (
            (glu_ev == ">200") | (glu_ev == ">300")
        ).astype(np.float64)
        extra_numeric.append("glu_abnormal")

    # 13. Missingness Indicators
    for m_col in ["weight", "medical_specialty", "payer_code"]:
        if m_col in train and m_col in eval_cols:
            train_copy[f"is_missing_{m_col}"] = np.array(
                [
                    x is None or str(x).strip() in ("?", "", "None")
                    for x in train[m_col]
                ],
                dtype=np.float64,
            )
            eval_copy[f"is_missing_{m_col}"] = np.array(
                [
                    x is None or str(x).strip() in ("?", "", "None")
                    for x in eval_cols[m_col]
                ],
                dtype=np.float64,
            )
            extra_numeric.append(f"is_missing_{m_col}")


    # Remove duplicates and existing task columns
    existing_cols = set(task.numeric_columns) | set(task.categorical_columns)
    extra_numeric = [
        c for c in dict.fromkeys(extra_numeric) if c not in existing_cols
    ]
    extra_categorical = [
        c for c in dict.fromkeys(extra_categorical) if c not in existing_cols
    ]

    # TabularEncoder with min_count=20
    encoder = TabularEncoder(min_count=20)
    x_train = encoder.fit_transform(
        task,
        train_copy,
        extra_numeric=extra_numeric,
        extra_categorical=extra_categorical,
    )
    x_eval = encoder.transform(eval_copy)

    # 14. Purified Continuous Subspace for XGBoost & Linear Anchor
    cat_indices_set = set(encoder.categorical_indices)
    num_indices = [
        i for i in range(x_train.shape[1]) if i not in cat_indices_set
    ]
    x_train_num = x_train[:, num_indices]
    x_eval_num = x_eval[:, num_indices]

    # 15. LightGBM Multi-Scale Ensemble
    # Model 1: Balanced leaf-wise with subsampling
    model1 = lgb.LGBMClassifier(
        n_estimators=260,
        learning_rate=0.022,
        num_leaves=31,
        min_child_samples=140,
        subsample=0.74,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_alpha=0.05,
        reg_lambda=1.5,
        cat_smooth=45.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model1.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p1 = model1.predict_proba(x_eval)[:, 1]

    # Model 2: Shallow, high-regularization architecture (main effects focus)
    model2 = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.020,
        num_leaves=20,
        max_depth=5,
        min_child_samples=120,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_alpha=0.25,
        reg_lambda=4.0,
        cat_smooth=35.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    model2.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p2 = model2.predict_proba(x_eval)[:, 1]

    # Model 3: Deeper interaction capacity
    model3 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.022,
        num_leaves=44,
        min_child_samples=170,
        subsample=0.70,
        subsample_freq=1,
        colsample_bytree=0.72,
        reg_alpha=0.02,
        reg_lambda=2.0,
        cat_smooth=65.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )
    model3.fit(x_train, y_train, categorical_feature=encoder.categorical_indices)
    p3 = model3.predict_proba(x_eval)[:, 1]

    lgb_preds = (p1 + p2 + p3) / 3.0

    # 16. Purified XGBoost Model
    xgb_preds = None
    try:
        import xgboost as xgb

        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.025,
            colsample_bytree=0.60,
            subsample=0.80,
            reg_alpha=0.1,
            reg_lambda=2.5,
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
        )
        xgb_model.fit(x_train_num, y_train)
        xgb_preds = xgb_model.predict_proba(x_eval_num)[:, 1]
    except Exception:
        xgb_preds = None

    # 17. Regularized Linear Anchor Model
    lr_preds = None
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        x_tr_dense = scaler.fit_transform(imputer.fit_transform(x_train_num))
        x_ev_dense = scaler.transform(imputer.transform(x_eval_num))

        lr_model = LogisticRegression(
            C=0.05,
            max_iter=200,
            penalty="l2",
            solver="lbfgs",
            random_state=42,
        )
        lr_model.fit(x_tr_dense, y_train)
        lr_preds = lr_model.predict_proba(x_ev_dense)[:, 1]
    except Exception:
        lr_preds = None

    # 18. Concordance-Preserving Quantile Calibration & Rank Blending
    n_ev_len = len(p1)
    r1 = rankdata(p1) / n_ev_len
    r2 = rankdata(p2) / n_ev_len
    r3 = rankdata(p3) / n_ev_len

    preds_list = [p1, p2, p3]
    ranks_list = [r1, r2, r3]
    weights = [0.24, 0.22, 0.22]

    if xgb_preds is not None:
        r_xgb = rankdata(xgb_preds) / n_ev_len
        preds_list.append(xgb_preds)
        ranks_list.append(r_xgb)
        weights.append(0.22)

    if lr_preds is not None:
        r_lr = rankdata(lr_preds) / n_ev_len
        preds_list.append(lr_preds)
        ranks_list.append(r_lr)
        weights.append(0.10)

    w_arr = np.array(weights, dtype=np.float64)
    w_arr /= w_arr.sum()

    blended_rank = np.zeros(n_ev_len, dtype=np.float64)
    consensus_prob = np.zeros(n_ev_len, dtype=np.float64)
    for w, r, p in zip(w_arr, ranks_list, preds_list):
        blended_rank += w * r
        consensus_prob += w * p

    sorted_consensus = np.sort(consensus_prob)
    grid = np.linspace(
        1.0 / (2.0 * n_ev_len), 1.0 - 1.0 / (2.0 * n_ev_len), n_ev_len
    )
    final_preds = np.interp(blended_rank, grid, sorted_consensus)

    return np.clip(np.nan_to_num(final_preds, nan=0.114), 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
