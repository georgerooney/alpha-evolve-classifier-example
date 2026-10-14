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
    """Predict 30-day readmission risk using clinical factors, transition indicators, and a tri-GBDT ensemble."""
    train_ext = dict(train)
    eval_ext = dict(eval_cols)
    extra_numeric: List[str] = []
    extra_categorical: List[str] = []

    # 1. Comorbidity Projections & High-Risk Clinical Interactions
    if "icd9_comorbidities" in globals():
        cm_train = icd9_comorbidities(train.get("diag_1"), train.get("diag_2"), train.get("diag_3"))
        cm_eval = icd9_comorbidities(eval_cols.get("diag_1"), eval_cols.get("diag_2"), eval_cols.get("diag_3"))
        for k, v in cm_train.items():
            train_ext[k] = v.astype(np.float64)
            eval_ext[k] = cm_eval[k].astype(np.float64)
            extra_numeric.append(k)

        if "cm_chf" in train_ext and "cm_renal" in train_ext:
            train_ext["feat_cardiorenal"] = (train_ext["cm_chf"] * train_ext["cm_renal"]).astype(np.float64)
            eval_ext["feat_cardiorenal"] = (eval_ext["cm_chf"] * eval_ext["cm_renal"]).astype(np.float64)
            extra_numeric.append("feat_cardiorenal")

        if "cm_chf" in train_ext and "cm_pulmonary" in train_ext:
            train_ext["feat_cardiopulmonary"] = (train_ext["cm_chf"] * train_ext["cm_pulmonary"]).astype(np.float64)
            eval_ext["feat_cardiopulmonary"] = (eval_ext["cm_chf"] * eval_ext["cm_pulmonary"]).astype(np.float64)
            extra_numeric.append("feat_cardiopulmonary")

        if "cm_renal" in train_ext and "cm_diabetes_complicated" in train_ext:
            train_ext["feat_renaldiabetes"] = (train_ext["cm_renal"] * train_ext["cm_diabetes_complicated"]).astype(np.float64)
            eval_ext["feat_renaldiabetes"] = (eval_ext["cm_renal"] * eval_ext["cm_diabetes_complicated"]).astype(np.float64)
            extra_numeric.append("feat_renaldiabetes")

        if "cm_count" in train_ext:
            train_ext["feat_multi_comorbid"] = (train_ext["cm_count"] >= 2.0).astype(np.float64)
            eval_ext["feat_multi_comorbid"] = (eval_ext["cm_count"] >= 2.0).astype(np.float64)
            extra_numeric.append("feat_multi_comorbid")

    # 2. Diagnosis Chapters Across Primary, Secondary, and Tertiary Positions
    if "icd9_group" in globals():
        for d_col in ["diag_1", "diag_2", "diag_3"]:
            if d_col in train and d_col in eval_cols:
                grp_col = f"{d_col}_grp"
                train_ext[grp_col] = np.array([icd9_group(c) for c in train[d_col]], dtype=object)
                eval_ext[grp_col] = np.array([icd9_group(c) for c in eval_cols[d_col]], dtype=object)
                extra_categorical.append(grp_col)

        if "diag_1_grp" in train_ext:
            tr_d1 = train_ext["diag_1_grp"]
            ev_d1 = eval_ext["diag_1_grp"]
            train_ext["feat_d1_circulatory"] = (tr_d1 == "circulatory").astype(np.float64)
            eval_ext["feat_d1_circulatory"] = (ev_d1 == "circulatory").astype(np.float64)
            extra_numeric.append("feat_d1_circulatory")

            train_ext["feat_d1_diabetes"] = (tr_d1 == "diabetes").astype(np.float64)
            eval_ext["feat_d1_diabetes"] = (ev_d1 == "diabetes").astype(np.float64)
            extra_numeric.append("feat_d1_diabetes")

            train_ext["feat_d1_respiratory"] = (tr_d1 == "respiratory").astype(np.float64)
            eval_ext["feat_d1_respiratory"] = (ev_d1 == "respiratory").astype(np.float64)
            extra_numeric.append("feat_d1_respiratory")

        if all(f"{c}_grp" in train_ext for c in ["diag_1", "diag_2", "diag_3"]):
            tr_any_circ = (
                (train_ext["diag_1_grp"] == "circulatory")
                | (train_ext["diag_2_grp"] == "circulatory")
                | (train_ext["diag_3_grp"] == "circulatory")
            ).astype(np.float64)
            ev_any_circ = (
                (eval_ext["diag_1_grp"] == "circulatory")
                | (eval_ext["diag_2_grp"] == "circulatory")
                | (eval_ext["diag_3_grp"] == "circulatory")
            ).astype(np.float64)
            train_ext["feat_any_circulatory"] = tr_any_circ
            eval_ext["feat_any_circulatory"] = ev_any_circ
            extra_numeric.append("feat_any_circulatory")

            tr_any_diab = (
                (train_ext["diag_1_grp"] == "diabetes")
                | (train_ext["diag_2_grp"] == "diabetes")
                | (train_ext["diag_3_grp"] == "diabetes")
            ).astype(np.float64)
            ev_any_diab = (
                (eval_ext["diag_1_grp"] == "diabetes")
                | (eval_ext["diag_2_grp"] == "diabetes")
                | (eval_ext["diag_3_grp"] == "diabetes")
            ).astype(np.float64)
            train_ext["feat_any_diabetes"] = tr_any_diab
            eval_ext["feat_any_diabetes"] = ev_any_diab
            extra_numeric.append("feat_any_diabetes")

    # 3. Out-Of-Fold Target Encoding on High-Cardinality Diagnoses and Transition IDs
    te_cols = [
        "discharge_disposition_id",
        "admission_source_id",
        "admission_type_id",
        "diag_1",
        "diag_2",
        "diag_3",
        "medical_specialty",
    ]
    for col in te_cols:
        if col in train and col in eval_cols:
            te_name = f"te_{col}"
            tr_te, ev_te = oof_target_encode(
                train[col], y_train, eval_cols[col], n_folds=5, smoothing=25.0, seed=42
            )
            train_ext[te_name] = tr_te.astype(np.float64)
            eval_ext[te_name] = ev_te.astype(np.float64)
            extra_numeric.append(te_name)

    # 4. Inpatient Utilization History & Daily Care Intensity Metrics
    def _extract_intensity(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        time_hosp = np.maximum(1.0, np.nan_to_num(d["time_in_hospital"], nan=1.0))
        labs = np.maximum(0.0, np.nan_to_num(d["num_lab_procedures"], nan=0.0))
        procs = np.maximum(0.0, np.nan_to_num(d["num_procedures"], nan=0.0))
        meds = np.maximum(0.0, np.nan_to_num(d["num_medications"], nan=0.0))
        outp = np.maximum(0.0, np.nan_to_num(d["number_outpatient"], nan=0.0))
        emerg = np.maximum(0.0, np.nan_to_num(d["number_emergency"], nan=0.0))
        inpat = np.maximum(0.0, np.nan_to_num(d["number_inpatient"], nan=0.0))
        num_diag = np.maximum(1.0, np.nan_to_num(d["number_diagnoses"], nan=1.0))

        total_prior = outp + emerg + inpat
        return {
            "feat_total_prior": total_prior,
            "feat_log_inpatient": np.log1p(inpat),
            "feat_log_emergency": np.log1p(emerg),
            "feat_log_total_prior": np.log1p(total_prior),
            "feat_has_inpatient": (inpat > 0.0).astype(np.float64),
            "feat_frequent_inpatient": (inpat >= 2.0).astype(np.float64),
            "feat_inpatient_ratio": (inpat + 0.05) / (total_prior + 0.1),
            "feat_emergency_ratio": (emerg + 0.05) / (total_prior + 0.1),
            "feat_service_intensity": (labs + procs + meds) / time_hosp,
            "feat_meds_per_day": meds / time_hosp,
            "feat_labs_per_day": labs / time_hosp,
            "feat_procs_per_day": procs / time_hosp,
            "feat_diag_per_day": num_diag / time_hosp,
            "feat_inpatient_x_diag": inpat * num_diag,
            "feat_long_stay": (time_hosp >= 7.0).astype(np.float64),
        }

    tr_intens = _extract_intensity(train)
    ev_intens = _extract_intensity(eval_cols)
    for k in tr_intens:
        train_ext[k] = tr_intens[k].astype(np.float64)
        eval_ext[k] = ev_intens[k].astype(np.float64)
        extra_numeric.append(k)

    # 5. Medication Regimen Complexity & Glycemic Volatility (Strack et al.)
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    ]

    def _extract_meds_and_labs(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        n = len(d["time_in_hospital"])
        active_cnt = np.zeros(n, dtype=np.float64)
        change_cnt = np.zeros(n, dtype=np.float64)
        for m in med_cols:
            if m in d:
                arr = d[m]
                active_cnt += np.isin(arr, ["Steady", "Up", "Down"]).astype(np.float64)
                change_cnt += np.isin(arr, ["Up", "Down"]).astype(np.float64)

        feats: Dict[str, np.ndarray] = {
            "feat_num_active_meds": active_cnt,
            "feat_num_med_changes": change_cnt,
            "feat_polypharmacy": (active_cnt >= 3.0).astype(np.float64),
        }

        if "insulin" in d:
            ins = d["insulin"]
            on_ins = np.isin(ins, ["Steady", "Up", "Down"]).astype(np.float64)
            feats["feat_on_insulin"] = on_ins
            feats["feat_insulin_up"] = (ins == "Up").astype(np.float64)
            feats["feat_insulin_down"] = (ins == "Down").astype(np.float64)
            feats["feat_insulin_steady"] = (ins == "Steady").astype(np.float64)
            feats["feat_insulin_and_oral"] = on_ins * ((active_cnt - on_ins) > 0.0).astype(np.float64)

        if "metformin" in d:
            met = d["metformin"]
            feats["feat_on_metformin"] = np.isin(met, ["Steady", "Up", "Down"]).astype(np.float64)

        if "change" in d:
            feats["feat_med_change"] = (d["change"] == "Ch").astype(np.float64)
        if "diabetesMed" in d:
            feats["feat_diabetes_med_yes"] = (d["diabetesMed"] == "Yes").astype(np.float64)

        if "A1Cresult" in d:
            a1c = d["A1Cresult"]
            feats["feat_a1c_tested"] = (~np.isin(a1c, ["None", None])).astype(np.float64)
            feats["feat_a1c_high"] = (a1c == ">8").astype(np.float64)
            feats["feat_a1c_norm"] = (a1c == "Norm").astype(np.float64)
            if "change" in d:
                feats["feat_a1c_high_no_change"] = ((a1c == ">8") & (d["change"] != "Ch")).astype(np.float64)
                feats["feat_a1c_high_changed"] = ((a1c == ">8") & (d["change"] == "Ch")).astype(np.float64)

        if "max_glu_serum" in d:
            glu = d["max_glu_serum"]
            feats["feat_glu_tested"] = (~np.isin(glu, ["None", None])).astype(np.float64)
            feats["feat_glu_high"] = np.isin(glu, [">200", ">300"]).astype(np.float64)

        return feats

    tr_meds = _extract_meds_and_labs(train)
    ev_meds = _extract_meds_and_labs(eval_cols)
    for k in tr_meds:
        train_ext[k] = tr_meds[k].astype(np.float64)
        eval_ext[k] = ev_meds[k].astype(np.float64)
        extra_numeric.append(k)

    # 6. Discharge Disposition, Admission Category & Administrative Missingness
    def _extract_adm_disch_demo(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        feats: Dict[str, np.ndarray] = {}
        if "discharge_disposition_id" in d:
            disch = d["discharge_disposition_id"]
            feats["feat_disch_home"] = (disch == "1").astype(np.float64)
            feats["feat_disch_snf"] = (disch == "3").astype(np.float64)
            feats["feat_disch_homehealth"] = (disch == "6").astype(np.float64)
            feats["feat_disch_ama"] = (disch == "7").astype(np.float64)
            feats["feat_disch_facility"] = np.isin(disch, ["2", "3", "4", "5", "22"]).astype(np.float64)

        if "admission_type_id" in d:
            adm = d["admission_type_id"]
            feats["feat_adm_emergency"] = (adm == "1").astype(np.float64)
            feats["feat_adm_urgent"] = (adm == "2").astype(np.float64)
            feats["feat_adm_elective"] = (adm == "3").astype(np.float64)

        if "medical_specialty" in d:
            feats["feat_missing_specialty"] = (d["medical_specialty"] == None).astype(np.float64)
        if "payer_code" in d:
            feats["feat_missing_payer"] = (d["payer_code"] == None).astype(np.float64)

        return feats

    tr_adm = _extract_adm_disch_demo(train)
    ev_adm = _extract_adm_disch_demo(eval_cols)
    for k in tr_adm:
        train_ext[k] = tr_adm[k].astype(np.float64)
        eval_ext[k] = ev_adm[k].astype(np.float64)
        extra_numeric.append(k)

    # 7. Age Midpoint & Geriatric Frailty Interaction
    age_map = {
        "[0-10)": 5.0, "[10-20)": 15.0, "[20-30)": 25.0, "[30-40)": 35.0,
        "[40-50)": 45.0, "[50-60)": 55.0, "[60-70)": 65.0, "[70-80)": 75.0,
        "[80-90)": 85.0, "[90-100)": 95.0,
    }
    tr_age = np.array([age_map.get(a, 65.0) for a in train["age"]], dtype=np.float64)
    ev_age = np.array([age_map.get(a, 65.0) for a in eval_cols["age"]], dtype=np.float64)
    train_ext["feat_age_num"] = tr_age
    eval_ext["feat_age_num"] = ev_age
    extra_numeric.append("feat_age_num")

    train_ext["feat_elderly"] = (tr_age >= 70.0).astype(np.float64)
    eval_ext["feat_elderly"] = (ev_age >= 70.0).astype(np.float64)
    extra_numeric.append("feat_elderly")

    if "number_inpatient" in train:
        tr_inpat = np.maximum(0.0, np.nan_to_num(train["number_inpatient"], nan=0.0))
        ev_inpat = np.maximum(0.0, np.nan_to_num(eval_cols["number_inpatient"], nan=0.0))
        train_ext["feat_age_x_inpatient"] = (tr_age / 50.0) * np.log1p(tr_inpat)
        eval_ext["feat_age_x_inpatient"] = (ev_age / 50.0) * np.log1p(ev_inpat)
        extra_numeric.append("feat_age_x_inpatient")

    # 8. Tabular Encoding
    encoder = TabularEncoder(min_count=20)
    encoder.fit(
        task,
        train_ext,
        extra_numeric=tuple(extra_numeric),
        extra_categorical=tuple(extra_categorical),
    )
    x_train = encoder.transform(train_ext)
    x_eval = encoder.transform(eval_ext)
    cats = encoder.categorical_indices

    # 9. Structurally Diverse Tri-GBDT Ensemble (Depth-Constrained, Deeper Interaction, and Heavy Bagging)
    model1 = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.02,
        max_depth=5,
        num_leaves=22,
        min_child_samples=180,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.65,
        reg_lambda=4.0,
        reg_alpha=0.3,
        cat_smooth=40.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model2 = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.02,
        max_depth=7,
        num_leaves=34,
        min_child_samples=130,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.60,
        reg_lambda=2.0,
        reg_alpha=0.15,
        cat_smooth=25.0,
        random_state=123,
        n_jobs=1,
        verbose=-1,
    )
    model3 = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.025,
        max_depth=4,
        num_leaves=15,
        min_child_samples=160,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.50,
        reg_lambda=5.0,
        reg_alpha=0.4,
        cat_smooth=35.0,
        random_state=999,
        n_jobs=1,
        verbose=-1,
    )

    model1.fit(x_train, y_train, categorical_feature=cats)
    model2.fit(x_train, y_train, categorical_feature=cats)
    model3.fit(x_train, y_train, categorical_feature=cats)

    p1 = model1.predict_proba(x_eval)[:, 1]
    p2 = model2.predict_proba(x_eval)[:, 1]
    p3 = model3.predict_proba(x_eval)[:, 1]

    pred = 0.35 * p1 + 0.35 * p2 + 0.30 * p3
    return np.clip(pred, 1e-6, 1.0 - 1e-6)


# EVOLVE-BLOCK-END
# =====================================================================
