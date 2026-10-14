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
    import sys
    from collections import Counter
    import numpy as np
    import lightgbm as lgb
    from sklearn.decomposition import TruncatedSVD

    def _get_tool(name: str):
        if name in globals():
            return globals()[name]
        tk = sys.modules.get("clinical_toolkit")
        if tk and hasattr(tk, name):
            return getattr(tk, name)
        try:
            mod = __import__("clinical_toolkit", fromlist=[name])
            return getattr(mod, name, None)
        except Exception:
            return None

    _TabularEncoder = _get_tool("TabularEncoder")
    _icd9_group = _get_tool("icd9_group")
    _icd9_comorbidities = _get_tool("icd9_comorbidities")
    _oof_target_encode = _get_tool("oof_target_encode")

    tr = dict(train)
    ev = dict(eval_cols)
    n_tr = len(y_train)
    n_ev = len(ev[next(iter(ev.keys()))])
    extra_numeric = []
    extra_categorical = []

    # 1. Comorbidities & Charlson Index
    if _icd9_comorbidities is not None and "diag_1" in tr and "diag_2" in tr and "diag_3" in tr:
        try:
            cm_tr = _icd9_comorbidities(tr["diag_1"], tr["diag_2"], tr["diag_3"])
            cm_ev = _icd9_comorbidities(ev["diag_1"], ev["diag_2"], ev["diag_3"])
            for cm_name, cm_arr in cm_tr.items():
                tr[cm_name] = cm_arr
                ev[cm_name] = cm_ev[cm_name]
                extra_numeric.append(cm_name)

            cm_weights = {
                "cm_mi": 1.0, "cm_chf": 2.0, "cm_pvd": 1.0, "cm_cerebrovascular": 1.0,
                "cm_dementia": 1.0, "cm_pulmonary": 1.0, "cm_rheumatic": 1.0, "cm_peptic_ulcer": 1.0,
                "cm_liver": 1.0, "cm_hemiplegia": 2.0, "cm_renal": 2.0, "cm_cancer": 2.0,
                "cm_diabetes_complicated": 2.0, "cm_metastatic": 4.0, "cm_hiv": 4.0,
            }
            charlson_tr = np.zeros(n_tr, dtype=np.float64)
            charlson_ev = np.zeros(n_ev, dtype=np.float64)
            for cm_k, w in cm_weights.items():
                if cm_k in tr:
                    charlson_tr += w * tr[cm_k]
                    charlson_ev += w * ev[cm_k]
            tr["charlson_index"] = charlson_tr
            ev["charlson_index"] = charlson_ev
            extra_numeric.append("charlson_index")

            if "cm_chf" in tr and "cm_renal" in tr:
                tr["cardiorenal"] = (tr["cm_chf"] * tr["cm_renal"]).astype(np.float64)
                ev["cardiorenal"] = (ev["cm_chf"] * ev["cm_renal"]).astype(np.float64)
                extra_numeric.append("cardiorenal")

            if "cm_chf" in tr and "cm_pulmonary" in tr:
                tr["cardiopulmonary"] = (tr["cm_chf"] * tr["cm_pulmonary"]).astype(np.float64)
                ev["cardiopulmonary"] = (ev["cm_chf"] * ev["cm_pulmonary"]).astype(np.float64)
                extra_numeric.append("cardiopulmonary")
        except Exception:
            pass

    # 2. ICD-9 Chapter Groups & 3-Digit Prefixes
    if _icd9_group is not None:
        for d_col in ["diag_1", "diag_2", "diag_3"]:
            if d_col in tr:
                try:
                    grp_col = d_col + "_group"
                    tr[grp_col] = np.array([_icd9_group(c) for c in tr[d_col]], dtype=object)
                    ev[grp_col] = np.array([_icd9_group(c) for c in ev[d_col]], dtype=object)
                    extra_categorical.append(grp_col)
                except Exception:
                    pass

    for d_col in ["diag_1", "diag_2", "diag_3"]:
        if d_col in tr:
            pfx_col = d_col + "_prefix"
            tr[pfx_col] = np.array([str(c).split(".")[0] if c is not None and c != "?" else "None" for c in tr[d_col]], dtype=object)
            ev[pfx_col] = np.array([str(c).split(".")[0] if c is not None and c != "?" else "None" for c in ev[d_col]], dtype=object)
            extra_categorical.append(pfx_col)

    # 3. Out-Of-Fold Target Encodings & Cross Encodings
    if _oof_target_encode is not None:
        te_candidates = [
            "discharge_disposition_id",
            "admission_source_id",
            "admission_type_id",
            "medical_specialty",
            "diag_1",
            "diag_2",
            "diag_3",
            "diag_1_prefix",
            "diag_2_prefix",
            "diag_3_prefix",
            "diag_1_group",
            "payer_code",
        ]
        for col in te_candidates:
            if col in tr and col in ev:
                try:
                    tr_te, ev_te = _oof_target_encode(
                        tr[col], y_train, ev[col], n_folds=5, smoothing=25.0, seed=42
                    )
                    te_name = col + "_te"
                    tr[te_name] = tr_te.astype(np.float64)
                    ev[te_name] = ev_te.astype(np.float64)
                    extra_numeric.append(te_name)
                except Exception:
                    pass

        # Interaction Target Encodings
        if "admission_source_id" in tr and "discharge_disposition_id" in tr:
            try:
                pair_tr = np.array([f"{s}_{d}" for s, d in zip(tr["admission_source_id"], tr["discharge_disposition_id"])], dtype=object)
                pair_ev = np.array([f"{s}_{d}" for s, d in zip(ev["admission_source_id"], ev["discharge_disposition_id"])], dtype=object)
                tr_te, ev_te = _oof_target_encode(pair_tr, y_train, pair_ev, n_folds=5, smoothing=30.0, seed=42)
                tr["adm_src_disch_te"] = tr_te.astype(np.float64)
                ev["adm_src_disch_te"] = ev_te.astype(np.float64)
                extra_numeric.append("adm_src_disch_te")
            except Exception:
                pass

        if "diag_1_group" in tr and "discharge_disposition_id" in tr:
            try:
                pair_tr = np.array([f"{g}_{d}" for g, d in zip(tr["diag_1_group"], tr["discharge_disposition_id"])], dtype=object)
                pair_ev = np.array([f"{g}_{d}" for g, d in zip(ev["diag_1_group"], ev["discharge_disposition_id"])], dtype=object)
                tr_te, ev_te = _oof_target_encode(pair_tr, y_train, pair_ev, n_folds=5, smoothing=30.0, seed=42)
                tr["diag1_grp_disch_te"] = tr_te.astype(np.float64)
                ev["diag1_grp_disch_te"] = ev_te.astype(np.float64)
                extra_numeric.append("diag1_grp_disch_te")
            except Exception:
                pass

        if "age" in tr and "discharge_disposition_id" in tr:
            try:
                pair_tr = np.array([f"{a}_{d}" for a, d in zip(tr["age"], tr["discharge_disposition_id"])], dtype=object)
                pair_ev = np.array([f"{a}_{d}" for a, d in zip(ev["age"], ev["discharge_disposition_id"])], dtype=object)
                tr_te, ev_te = _oof_target_encode(pair_tr, y_train, pair_ev, n_folds=5, smoothing=30.0, seed=42)
                tr["age_disch_te"] = tr_te.astype(np.float64)
                ev["age_disch_te"] = ev_te.astype(np.float64)
                extra_numeric.append("age_disch_te")
            except Exception:
                pass

    # Frequency Encoding
    freq_cols = ["diag_1_prefix", "medical_specialty", "discharge_disposition_id", "admission_source_id"]
    for col in freq_cols:
        if col in tr and col in ev:
            cnts = Counter(tr[col])
            tr[f"{col}_freq"] = np.array([cnts.get(x, 0) / float(n_tr) for x in tr[col]], dtype=np.float64)
            ev[f"{col}_freq"] = np.array([cnts.get(x, 0) / float(n_tr) for x in ev[col]], dtype=np.float64)
            extra_numeric.append(f"{col}_freq")

    # 4. Utilization Intensity, Pharmacotherapy & Clinical Derived Features
    med_names = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]
    sulfo_names = ["glimepiride", "glipizide", "glyburide", "chlorpropamide", "tolbutamide", "acetohexamide", "tolazamide"]
    tzd_names = ["pioglitazone", "rosiglitazone", "troglitazone"]

    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }

    for d, n_rows in [(tr, n_tr), (ev, n_ev)]:
        n_inp = np.nan_to_num(d.get("number_inpatient", np.zeros(n_rows, dtype=np.float64)), nan=0.0)
        n_emg = np.nan_to_num(d.get("number_emergency", np.zeros(n_rows, dtype=np.float64)), nan=0.0)
        n_out = np.nan_to_num(d.get("number_outpatient", np.zeros(n_rows, dtype=np.float64)), nan=0.0)
        t_hosp = np.nan_to_num(d.get("time_in_hospital", np.ones(n_rows, dtype=np.float64)), nan=1.0)
        n_meds = np.nan_to_num(d.get("num_medications", np.zeros(n_rows, dtype=np.float64)), nan=0.0)
        n_labs = np.nan_to_num(d.get("num_lab_procedures", np.zeros(n_rows, dtype=np.float64)), nan=0.0)
        n_prc = np.nan_to_num(d.get("num_procedures", np.zeros(n_rows, dtype=np.float64)), nan=0.0)
        n_diag = np.nan_to_num(d.get("number_diagnoses", np.ones(n_rows, dtype=np.float64)), nan=1.0)

        tot_prior = n_inp + n_emg + n_out
        d["total_prior_visits"] = tot_prior
        d["has_prior_inpatient"] = (n_inp > 0).astype(np.float64)
        d["multi_prior_inpatient"] = (n_inp >= 2).astype(np.float64)
        d["has_prior_emergency"] = (n_emg > 0).astype(np.float64)
        d["multi_prior_emergency"] = (n_emg >= 2).astype(np.float64)
        d["inpatient_ratio"] = n_inp / (tot_prior + 1.0)
        d["emergency_ratio"] = n_emg / (tot_prior + 1.0)
        d["acute_visits"] = n_inp + n_emg
        d["acute_ratio"] = (n_inp + n_emg) / (tot_prior + 1.0)
        d["prior_acute_severity"] = (2.0 * n_inp + n_emg) / (tot_prior + 1.0)

        # Log-transformed utilizations
        d["log_inpatient"] = np.log1p(n_inp)
        d["log_emergency"] = np.log1p(n_emg)
        d["log_total_visits"] = np.log1p(tot_prior)
        d["log_time_hosp"] = np.log1p(t_hosp)
        d["inpatient_x_stay"] = n_inp * t_hosp
        d["inpatient_x_emergency"] = n_inp * n_emg
        d["inp_per_day"] = n_inp / (t_hosp + 1.0)

        d["service_intensity"] = (n_meds + n_labs + n_prc) / (t_hosp + 1.0)
        d["labs_per_day"] = n_labs / (t_hosp + 1.0)
        d["meds_per_day"] = n_meds / (t_hosp + 1.0)
        d["procs_per_day"] = n_prc / (t_hosp + 1.0)
        d["meds_per_diag"] = n_meds / (n_diag + 1.0)
        d["labs_per_diag"] = n_labs / (n_diag + 1.0)
        d["diag_per_day"] = n_diag / (t_hosp + 1.0)
        d["short_stay"] = (t_hosp <= 2).astype(np.float64)
        d["long_stay"] = (t_hosp >= 7).astype(np.float64)
        d["high_diagnoses"] = (n_diag >= 9).astype(np.float64)

        if "age" in d:
            age_num = np.array([age_map.get(a, np.nan) for a in d["age"]], dtype=np.float64)
        else:
            age_num = np.full(n_rows, np.nan, dtype=np.float64)
        d["age_numeric"] = age_num
        d["is_elderly"] = (np.nan_to_num(age_num, nan=0.0) >= 70).astype(np.float64)
        d["age_x_inpatient"] = np.nan_to_num(age_num, nan=50.0) * n_inp

        n_act = np.zeros(n_rows, dtype=np.float64)
        n_chg = np.zeros(n_rows, dtype=np.float64)
        n_up = np.zeros(n_rows, dtype=np.float64)
        n_dn = np.zeros(n_rows, dtype=np.float64)
        for m in med_names:
            if m in d:
                m_arr = d[m]
                for i in range(n_rows):
                    v = m_arr[i]
                    if v is not None and v != "No":
                        n_act[i] += 1.0
                        if v == "Up":
                            n_chg[i] += 1.0
                            n_up[i] += 1.0
                        elif v == "Down":
                            n_chg[i] += 1.0
                            n_dn[i] += 1.0
        d["num_active_meds"] = n_act
        d["num_med_changes"] = n_chg
        d["num_meds_up"] = n_up
        d["num_meds_down"] = n_dn

        # Pharmacological classes
        n_sulfo = np.zeros(n_rows, dtype=np.float64)
        for m in sulfo_names:
            if m in d:
                m_arr = d[m]
                for i in range(n_rows):
                    if m_arr[i] is not None and m_arr[i] != "No":
                        n_sulfo[i] += 1.0
        d["num_sulfonylureas"] = n_sulfo

        n_tzd = np.zeros(n_rows, dtype=np.float64)
        for m in tzd_names:
            if m in d:
                m_arr = d[m]
                for i in range(n_rows):
                    if m_arr[i] is not None and m_arr[i] != "No":
                        n_tzd[i] += 1.0
        d["num_tzds"] = n_tzd

        ins = d.get("insulin", np.array([None] * n_rows))
        ins_act = np.fromiter((1.0 if v is not None and v != "No" else 0.0 for v in ins), dtype=np.float64, count=n_rows)
        d["insulin_active"] = ins_act
        d["insulin_up"] = np.fromiter((1.0 if v == "Up" else 0.0 for v in ins), dtype=np.float64, count=n_rows)
        d["insulin_down"] = np.fromiter((1.0 if v == "Down" else 0.0 for v in ins), dtype=np.float64, count=n_rows)
        d["insulin_changed"] = np.fromiter((1.0 if v in ("Up", "Down") else 0.0 for v in ins), dtype=np.float64, count=n_rows)
        d["insulin_steady"] = np.fromiter((1.0 if v == "Steady" else 0.0 for v in ins), dtype=np.float64, count=n_rows)
        d["insulin_plus_oral"] = ((ins_act > 0) & ((n_act - ins_act) > 0)).astype(np.float64)
        d["hypo_risk_combo"] = ((ins_act > 0) & (n_sulfo > 0)).astype(np.float64)

        met = d.get("metformin", np.array([None] * n_rows))
        d["metformin_active"] = np.fromiter((1.0 if v is not None and v != "No" else 0.0 for v in met), dtype=np.float64, count=n_rows)

        chg_col = d.get("change", np.array([None] * n_rows))
        d["has_med_change"] = np.fromiter((1.0 if v == "Ch" else 0.0 for v in chg_col), dtype=np.float64, count=n_rows)

        disp = d.get("discharge_disposition_id", np.array([None] * n_rows))
        disp_str = np.array([str(v) if v is not None else "" for v in disp])
        d["discharged_home"] = (disp_str == "1").astype(np.float64)
        d["discharged_snf"] = (disp_str == "3").astype(np.float64)
        d["discharged_home_health"] = (disp_str == "6").astype(np.float64)
        d["discharged_ama"] = (disp_str == "7").astype(np.float64)
        d["discharged_high_risk"] = np.isin(disp_str, ["3", "6", "7", "22", "28"]).astype(np.float64)

        adm_s = d.get("admission_source_id", np.array([None] * n_rows))
        adm_s_str = np.array([str(v) if v is not None else "" for v in adm_s])
        d["admitted_from_er"] = (adm_s_str == "7").astype(np.float64)
        d["admitted_transfer"] = np.isin(adm_s_str, ["4", "5", "6"]).astype(np.float64)
        d["er_to_high_risk_disch"] = d["admitted_from_er"] * d["discharged_high_risk"]

        adm_t = d.get("admission_type_id", np.array([None] * n_rows))
        adm_t_str = np.array([str(v) if v is not None else "" for v in adm_t])
        d["admission_emergency"] = (adm_t_str == "1").astype(np.float64)
        d["admission_elective"] = (adm_t_str == "3").astype(np.float64)

        a1c = d.get("A1Cresult", np.array([None] * n_rows))
        a1c_hi = np.fromiter((1.0 if v in (">8", ">7") else 0.0 for v in a1c), dtype=np.float64, count=n_rows)
        d["a1c_measured"] = np.fromiter((0.0 if v in ("None", None) else 1.0 for v in a1c), dtype=np.float64, count=n_rows)
        d["a1c_high"] = a1c_hi
        d["a1c_normal"] = np.fromiter((1.0 if v == "Norm" else 0.0 for v in a1c), dtype=np.float64, count=n_rows)
        d["a1c_high_and_change"] = a1c_hi * d["has_med_change"]

        glu = d.get("max_glu_serum", np.array([None] * n_rows))
        d["glu_high"] = np.fromiter((1.0 if v in (">200", ">300") else 0.0 for v in glu), dtype=np.float64, count=n_rows)

        # Specialty & Payer missingness/flags
        med_spec = d.get("medical_specialty", np.array([None] * n_rows))
        d["specialty_is_missing"] = np.fromiter((1.0 if s is None or s == "?" else 0.0 for s in med_spec), dtype=np.float64, count=n_rows)

        payer = d.get("payer_code", np.array([None] * n_rows))
        d["payer_is_missing"] = np.fromiter((1.0 if p is None or p == "?" else 0.0 for p in payer), dtype=np.float64, count=n_rows)
        d["is_medicare"] = np.fromiter((1.0 if p == "MC" else 0.0 for p in payer), dtype=np.float64, count=n_rows)
        d["is_medicaid"] = np.fromiter((1.0 if p == "MD" else 0.0 for p in payer), dtype=np.float64, count=n_rows)

        # Uncontrolled diabetes flag
        d1 = d.get("diag_1", np.array([None] * n_rows))
        d2 = d.get("diag_2", np.array([None] * n_rows))
        d3 = d.get("diag_3", np.array([None] * n_rows))
        uncontrolled = np.zeros(n_rows, dtype=np.float64)
        diag1_chf = np.zeros(n_rows, dtype=np.float64)
        diag1_dm = np.zeros(n_rows, dtype=np.float64)
        for i in range(n_rows):
            for cd in (d1[i], d2[i], d3[i]):
                if cd is not None and isinstance(cd, str) and cd.startswith("250"):
                    parts = cd.split(".")
                    if len(parts) > 1 and len(parts[1]) >= 2 and parts[1][1] in ("2", "3"):
                        uncontrolled[i] = 1.0
            if d1[i] is not None and isinstance(d1[i], str):
                if d1[i].startswith("428"):
                    diag1_chf[i] = 1.0
                elif d1[i].startswith("250"):
                    diag1_dm[i] = 1.0
        d["uncontrolled_diabetes"] = uncontrolled
        d["diag1_is_chf"] = diag1_chf
        d["diag1_is_diabetes"] = diag1_dm

        if "cm_chf" in d:
            d["tzd_heart_failure_risk"] = ((n_tzd > 0) & (d["cm_chf"] > 0)).astype(np.float64)
        else:
            d["tzd_heart_failure_risk"] = np.zeros(n_rows, dtype=np.float64)

        if "cm_renal" in d:
            d["metformin_renal_risk"] = ((d["metformin_active"] > 0) & (d["cm_renal"] > 0)).astype(np.float64)
        else:
            d["metformin_renal_risk"] = np.zeros(n_rows, dtype=np.float64)

        if "cm_chf" in d and "cm_renal" in d and "cm_diabetes_complicated" in d:
            d["cardiorenal_metabolic_triad"] = (d["cm_chf"] * d["cm_renal"] * d["cm_diabetes_complicated"]).astype(np.float64)
        else:
            d["cardiorenal_metabolic_triad"] = np.zeros(n_rows, dtype=np.float64)

    extra_numeric.extend([
        "total_prior_visits", "has_prior_inpatient", "multi_prior_inpatient",
        "has_prior_emergency", "multi_prior_emergency", "inpatient_ratio",
        "emergency_ratio", "acute_visits", "acute_ratio", "prior_acute_severity",
        "log_inpatient", "log_emergency", "log_total_visits", "log_time_hosp",
        "inpatient_x_stay", "inpatient_x_emergency", "inp_per_day", "service_intensity",
        "labs_per_day", "meds_per_day", "procs_per_day", "meds_per_diag", "labs_per_diag",
        "diag_per_day", "short_stay", "long_stay", "high_diagnoses", "age_numeric",
        "is_elderly", "age_x_inpatient", "num_active_meds", "num_med_changes",
        "num_meds_up", "num_meds_down", "num_sulfonylureas", "num_tzds", "insulin_active",
        "insulin_up", "insulin_down", "insulin_changed", "insulin_steady", "insulin_plus_oral",
        "hypo_risk_combo", "metformin_active", "has_med_change", "discharged_home",
        "discharged_snf", "discharged_home_health", "discharged_ama", "discharged_high_risk",
        "admitted_from_er", "admitted_transfer", "er_to_high_risk_disch", "admission_emergency",
        "admission_elective", "a1c_measured", "a1c_high", "a1c_normal", "a1c_high_and_change",
        "glu_high", "specialty_is_missing", "payer_is_missing", "is_medicare", "is_medicaid",
        "uncontrolled_diabetes", "diag1_is_chf", "diag1_is_diabetes", "tzd_heart_failure_risk",
        "metformin_renal_risk", "cardiorenal_metabolic_triad",
    ])

    # 5. Latent Clinical Phenotyping via Multi-Aspect Tensor Decomposition
    # Aspect 1: Pathology Profile Matrix
    # Aspect 2: Pharmacotherapy Regimen Matrix
    # Aspect 3: Tensor Tucker/CP Bilinear Cross-Coupling
    diag_counter = Counter()
    for d_col in ["diag_1", "diag_2", "diag_3"]:
        if d_col in tr:
            for c in tr[d_col]:
                if c is not None and c != "?" and isinstance(c, str):
                    diag_counter[c.split(".")[0]] += 1
    top_diag_prefixes = [p for p, cnt in diag_counter.most_common(45) if cnt >= 40]

    def _build_pathology_matrix(cols, n_rows):
        ind = []
        for cm_name in [
            "cm_mi", "cm_chf", "cm_pvd", "cm_cerebrovascular", "cm_dementia",
            "cm_pulmonary", "cm_rheumatic", "cm_peptic_ulcer", "cm_liver",
            "cm_hemiplegia", "cm_renal", "cm_cancer", "cm_metastatic",
            "cm_hiv", "cm_diabetes_complicated",
        ]:
            if cm_name in cols:
                ind.append(cols[cm_name].astype(np.float32))

        for k in ["uncontrolled_diabetes", "diag1_is_chf", "diag1_is_diabetes"]:
            if k in cols:
                ind.append(cols[k].astype(np.float32))

        if top_diag_prefixes:
            d1 = cols.get("diag_1", np.array([None] * n_rows))
            d2 = cols.get("diag_2", np.array([None] * n_rows))
            d3 = cols.get("diag_3", np.array([None] * n_rows))
            row_prefixes = []
            for i in range(n_rows):
                s = set()
                for code in (d1[i], d2[i], d3[i]):
                    if code is not None and code != "?" and isinstance(code, str):
                        s.add(code.split(".")[0])
                row_prefixes.append(s)
            for pfx in top_diag_prefixes:
                has_p = np.fromiter((1.0 if pfx in row_prefixes[i] else 0.0 for i in range(n_rows)), dtype=np.float32, count=n_rows)
                ind.append(has_p)

        if ind:
            return np.column_stack(ind)
        return np.zeros((n_rows, 1), dtype=np.float32)

    def _build_pharmacotherapy_matrix(cols, n_rows):
        ind = []
        for m in med_names:
            if m in cols:
                m_arr = cols[m]
                act = np.fromiter((1.0 if v is not None and v != "No" else 0.0 for v in m_arr), dtype=np.float32, count=n_rows)
                chg = np.fromiter((1.0 if v in ("Up", "Down") else 0.0 for v in m_arr), dtype=np.float32, count=n_rows)
                up = np.fromiter((1.0 if v == "Up" else 0.0 for v in m_arr), dtype=np.float32, count=n_rows)
                ind.append(act)
                ind.append(chg)
                ind.append(up)

        for k in ["insulin_active", "insulin_up", "insulin_down", "insulin_plus_oral", "metformin_active"]:
            if k in cols:
                ind.append(cols[k].astype(np.float32))

        if ind:
            return np.column_stack(ind)
        return np.zeros((n_rows, 1), dtype=np.float32)

    X_path_tr = _build_pathology_matrix(tr, n_tr)
    X_path_ev = _build_pathology_matrix(ev, n_ev)
    X_rx_tr = _build_pharmacotherapy_matrix(tr, n_tr)
    X_rx_ev = _build_pharmacotherapy_matrix(ev, n_ev)

    n_latent_path = min(6, max(1, X_path_tr.shape[1] - 1))
    n_latent_rx = min(4, max(1, X_rx_tr.shape[1] - 1))

    try:
        # Pathology Subspace Factorization
        svd_path = TruncatedSVD(n_components=n_latent_path, algorithm="randomized", random_state=42)
        Z_path_tr = svd_path.fit_transform(X_path_tr)
        Z_path_ev = svd_path.transform(X_path_ev)
        for j in range(n_latent_path):
            tr[f"pathology_latent_{j}"] = Z_path_tr[:, j].astype(np.float64)
            ev[f"pathology_latent_{j}"] = Z_path_ev[:, j].astype(np.float64)
            extra_numeric.append(f"pathology_latent_{j}")

        X_rec_p_tr = np.dot(Z_path_tr, svd_path.components_)
        X_rec_p_ev = np.dot(Z_path_ev, svd_path.components_)
        tr["pathology_anomaly_score"] = np.sqrt(np.mean((X_path_tr - X_rec_p_tr) ** 2, axis=1)).astype(np.float64)
        ev["pathology_anomaly_score"] = np.sqrt(np.mean((X_path_ev - X_rec_p_ev) ** 2, axis=1)).astype(np.float64)
        extra_numeric.append("pathology_anomaly_score")

        # Pharmacotherapy Subspace Factorization
        svd_rx = TruncatedSVD(n_components=n_latent_rx, algorithm="randomized", random_state=42)
        Z_rx_tr = svd_rx.fit_transform(X_rx_tr)
        Z_rx_ev = svd_rx.transform(X_rx_ev)
        for k in range(n_latent_rx):
            tr[f"treatment_latent_{k}"] = Z_rx_tr[:, k].astype(np.float64)
            ev[f"treatment_latent_{k}"] = Z_rx_ev[:, k].astype(np.float64)
            extra_numeric.append(f"treatment_latent_{k}")

        X_rec_r_tr = np.dot(Z_rx_tr, svd_rx.components_)
        X_rec_r_ev = np.dot(Z_rx_ev, svd_rx.components_)
        tr["treatment_anomaly_score"] = np.sqrt(np.mean((X_rx_tr - X_rec_r_tr) ** 2, axis=1)).astype(np.float64)
        ev["treatment_anomaly_score"] = np.sqrt(np.mean((X_rx_ev - X_rec_r_ev) ** 2, axis=1)).astype(np.float64)
        extra_numeric.append("treatment_anomaly_score")

        # Bilinear Tensor CP Outer Products: pathology mode x treatment mode
        n_cp_p = min(3, n_latent_path)
        n_cp_r = min(3, n_latent_rx)
        for j in range(n_cp_p):
            for k in range(n_cp_r):
                feat_name = f"tensor_cp_{j}_{k}"
                tr[feat_name] = (Z_path_tr[:, j] * Z_rx_tr[:, k]).astype(np.float64)
                ev[feat_name] = (Z_path_ev[:, j] * Z_rx_ev[:, k]).astype(np.float64)
                extra_numeric.append(feat_name)

        # Phenotype-Treatment Cosine Alignment
        norm_ptr = np.linalg.norm(Z_path_tr[:, :n_cp_p], axis=1) + 1e-6
        norm_rtr = np.linalg.norm(Z_rx_tr[:, :n_cp_r], axis=1) + 1e-6
        min_k = min(n_cp_p, n_cp_r)
        dot_tr = np.sum(Z_path_tr[:, :min_k] * Z_rx_tr[:, :min_k], axis=1)
        tr["phenotype_tx_alignment"] = (dot_tr / (norm_ptr * norm_rtr)).astype(np.float64)

        norm_pev = np.linalg.norm(Z_path_ev[:, :n_cp_p], axis=1) + 1e-6
        norm_rev = np.linalg.norm(Z_rx_ev[:, :n_cp_r], axis=1) + 1e-6
        dot_ev = np.sum(Z_path_ev[:, :min_k] * Z_rx_ev[:, :min_k], axis=1)
        ev["phenotype_tx_alignment"] = (dot_ev / (norm_pev * norm_rev)).astype(np.float64)
        extra_numeric.append("phenotype_tx_alignment")
    except Exception:
        pass

    extra_numeric = list(dict.fromkeys(extra_numeric))
    extra_categorical = list(dict.fromkeys(extra_categorical))

    # 6. Tabular Encoding (min_count=20 retains informative clinical categories)
    encoder = _TabularEncoder(min_count=20)
    encoder.fit(task, tr, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(tr)
    x_eval = encoder.transform(ev)
    cat_indices = list(encoder.categorical_indices)

    # 7. Multi-Scale LightGBM + XGBoost Optimization Ensemble with Log-Odds Consensus
    model_balanced = lgb.LGBMClassifier(
        n_estimators=320,
        learning_rate=0.020,
        num_leaves=31,
        max_depth=6,
        min_child_samples=160,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=2.0,
        reg_alpha=0.2,
        cat_smooth=50.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model_shallow = lgb.LGBMClassifier(
        n_estimators=360,
        learning_rate=0.016,
        num_leaves=18,
        max_depth=5,
        min_child_samples=220,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=6.0,
        reg_alpha=0.8,
        cat_smooth=80.0,
        random_state=101,
        n_jobs=1,
        verbose=-1,
    )
    model_deep = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.022,
        num_leaves=42,
        max_depth=7,
        min_child_samples=120,
        subsample=0.68,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_lambda=1.5,
        reg_alpha=0.15,
        cat_smooth=35.0,
        random_state=2024,
        n_jobs=1,
        verbose=-1,
    )

    model_balanced.fit(x_train, y_train, categorical_feature=cat_indices)
    model_shallow.fit(x_train, y_train, categorical_feature=cat_indices)
    model_deep.fit(x_train, y_train, categorical_feature=cat_indices)

    p1 = model_balanced.predict_proba(x_eval)[:, 1]
    p2 = model_shallow.predict_proba(x_eval)[:, 1]
    p3 = model_deep.predict_proba(x_eval)[:, 1]

    # ExtraTrees Randomized Split LightGBM
    p4 = None
    try:
        model_extra = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.020,
            num_leaves=28,
            max_depth=6,
            min_child_samples=150,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=3.0,
            reg_alpha=0.3,
            cat_smooth=60.0,
            extra_trees=True,
            random_state=888,
            n_jobs=1,
            verbose=-1,
        )
        model_extra.fit(x_train, y_train, categorical_feature=cat_indices)
        p4 = model_extra.predict_proba(x_eval)[:, 1]
    except Exception:
        model_extra_fb = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.020,
            num_leaves=28,
            max_depth=6,
            min_child_samples=150,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=3.0,
            reg_alpha=0.3,
            cat_smooth=60.0,
            random_state=888,
            n_jobs=1,
            verbose=-1,
        )
        model_extra_fb.fit(x_train, y_train, categorical_feature=cat_indices)
        p4 = model_extra_fb.predict_proba(x_eval)[:, 1]

    # Level-wise XGBoost model for structural ensemble diversity
    p5 = None
    try:
        import xgboost as xgb
        model_xgb = xgb.XGBClassifier(
            n_estimators=260,
            learning_rate=0.022,
            max_depth=5,
            min_child_weight=20,
            subsample=0.75,
            colsample_bytree=0.65,
            reg_lambda=4.0,
            reg_alpha=0.3,
            tree_method="hist",
            random_state=777,
            n_jobs=1,
            eval_metric="logloss",
        )
        model_xgb.fit(x_train, y_train)
        p5 = model_xgb.predict_proba(x_eval)[:, 1]
    except Exception:
        p5 = None

    def _to_logit(p):
        p_c = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(p_c / (1.0 - p_c))

    def _to_prob(logit):
        return 1.0 / (1.0 + np.exp(-logit))

    if p5 is not None:
        l_ens = (
            0.26 * _to_logit(p1)
            + 0.23 * _to_logit(p2)
            + 0.18 * _to_logit(p3)
            + 0.18 * _to_logit(p4)
            + 0.15 * _to_logit(p5)
        )
    else:
        l_ens = (
            0.30 * _to_logit(p1)
            + 0.28 * _to_logit(p2)
            + 0.22 * _to_logit(p3)
            + 0.20 * _to_logit(p4)
        )

    preds = _to_prob(l_ens)
    preds = np.clip(preds, 0.0, 1.0)
    preds = np.nan_to_num(preds, nan=float(np.mean(y_train)))
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
