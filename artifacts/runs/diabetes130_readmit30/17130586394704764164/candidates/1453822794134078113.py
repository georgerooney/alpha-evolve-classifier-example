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
    train_dict = dict(train)
    eval_dict = dict(eval_cols)
    extra_numeric = []
    extra_categorical = []

    n_train = len(y_train)
    n_eval = len(next(iter(eval_cols.values())))

    # 1. Charlson Comorbidities via icd9_comorbidities
    if "icd9_comorbidities" in globals():
        cm_train = icd9_comorbidities(train["diag_1"], train["diag_2"], train["diag_3"])
        cm_eval = icd9_comorbidities(eval_cols["diag_1"], eval_cols["diag_2"], eval_cols["diag_3"])
        for k, v in cm_train.items():
            train_dict[k] = v.astype(np.float64)
            eval_dict[k] = cm_eval[k].astype(np.float64)
            extra_numeric.append(k)

        # High-risk clinical interactions
        chf_tr = cm_train.get("cm_chf", np.zeros(n_train))
        chf_ev = cm_eval.get("cm_chf", np.zeros(n_eval))
        renal_tr = cm_train.get("cm_renal", np.zeros(n_train))
        renal_ev = cm_eval.get("cm_renal", np.zeros(n_eval))
        cnt_tr = cm_train.get("cm_count", np.zeros(n_train))
        cnt_ev = cm_eval.get("cm_count", np.zeros(n_eval))

        train_dict["chf_or_renal"] = np.maximum(chf_tr, renal_tr).astype(np.float64)
        eval_dict["chf_or_renal"] = np.maximum(chf_ev, renal_ev).astype(np.float64)
        extra_numeric.append("chf_or_renal")

        inp_tr = train["number_inpatient"].astype(np.float64)
        inp_ev = eval_cols["number_inpatient"].astype(np.float64)

        train_dict["chf_x_inp"] = (chf_tr * inp_tr).astype(np.float64)
        eval_dict["chf_x_inp"] = (chf_ev * inp_ev).astype(np.float64)
        extra_numeric.append("chf_x_inp")

        train_dict["renal_x_inp"] = (renal_tr * inp_tr).astype(np.float64)
        eval_dict["renal_x_inp"] = (renal_ev * inp_ev).astype(np.float64)
        extra_numeric.append("renal_x_inp")

        train_dict["cm_count_x_inp"] = (cnt_tr * inp_tr).astype(np.float64)
        eval_dict["cm_count_x_inp"] = (cnt_ev * inp_ev).astype(np.float64)
        extra_numeric.append("cm_count_x_inp")

    # 2. ICD-9 Chapter Groupings via icd9_group
    if "icd9_group" in globals():
        for diag in ("diag_1", "diag_2", "diag_3"):
            grp_name = f"{diag}_group"
            train_dict[grp_name] = np.array([icd9_group(c) for c in train[diag]], dtype=object)
            eval_dict[grp_name] = np.array([icd9_group(c) for c in eval_cols[diag]], dtype=object)
            extra_categorical.append(grp_name)

        d1_tr = train_dict["diag_1_group"]
        d1_ev = eval_dict["diag_1_group"]
        for domain in ("circulatory", "diabetes", "respiratory"):
            feat_name = f"diag1_is_{domain}"
            train_dict[feat_name] = (d1_tr == domain).astype(np.float64)
            eval_dict[feat_name] = (d1_ev == domain).astype(np.float64)
            extra_numeric.append(feat_name)

    # 3. Out-Of-Fold Target Encoding on high-cardinality codes
    if "oof_target_encode" in globals():
        te_cols = [
            "diag_1",
            "diag_2",
            "diag_3",
            "discharge_disposition_id",
            "admission_source_id",
            "medical_specialty",
            "payer_code",
        ]
        prior_mean = float(np.mean(y_train))
        for col_name in te_cols:
            if col_name in train_dict:
                tr_te, ev_te = oof_target_encode(
                    train_dict[col_name], y_train, eval_dict[col_name], n_folds=5, smoothing=25.0, seed=42
                )
                te_name = f"te_{col_name}"
                train_dict[te_name] = np.nan_to_num(tr_te, nan=prior_mean).astype(np.float64)
                eval_dict[te_name] = np.nan_to_num(ev_te, nan=prior_mean).astype(np.float64)
                extra_numeric.append(te_name)

    # 4. Ordinal Age Mapping
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
    age_tr = np.array([age_map.get(str(x), np.nan) for x in train["age"]], dtype=np.float64)
    age_ev = np.array([age_map.get(str(x), np.nan) for x in eval_cols["age"]], dtype=np.float64)
    train_dict["age_num"] = age_tr
    eval_dict["age_num"] = age_ev
    extra_numeric.append("age_num")

    # 5. Prior Utilization Intensity Features
    for d, split_name in ((train, "train"), (eval_cols, "eval")):
        inp = d["number_inpatient"].astype(np.float64)
        emg = d["number_emergency"].astype(np.float64)
        out = d["number_outpatient"].astype(np.float64)
        total_prior = inp + emg + out

        target_dict = train_dict if split_name == "train" else eval_dict
        target_dict["total_prior_visits"] = total_prior
        target_dict["has_inpatient"] = (inp > 0).astype(np.float64)
        target_dict["has_emergency"] = (emg > 0).astype(np.float64)
        target_dict["has_prior_visits"] = (total_prior > 0).astype(np.float64)
        target_dict["inpatient_ratio"] = inp / (total_prior + 1.0)
        target_dict["emergency_ratio"] = emg / (total_prior + 1.0)
        target_dict["frequent_inpatient"] = (inp >= 2).astype(np.float64)
        target_dict["frequent_emergency"] = (emg >= 2).astype(np.float64)

    extra_numeric.extend([
        "total_prior_visits",
        "has_inpatient",
        "has_emergency",
        "has_prior_visits",
        "inpatient_ratio",
        "emergency_ratio",
        "frequent_inpatient",
        "frequent_emergency",
    ])

    # 6. Hospital Stay Intensity Ratios & Interactions
    for d, age_arr, split_name in ((train, age_tr, "train"), (eval_cols, age_ev, "eval")):
        target_dict = train_dict if split_name == "train" else eval_dict
        time_hosp = np.maximum(d["time_in_hospital"].astype(np.float64), 1.0)
        num_labs = d["num_lab_procedures"].astype(np.float64)
        num_procs = d["num_procedures"].astype(np.float64)
        num_meds = d["num_medications"].astype(np.float64)
        num_diags = np.maximum(d["number_diagnoses"].astype(np.float64), 1.0)
        inp = d["number_inpatient"].astype(np.float64)
        clean_age = np.nan_to_num(age_arr, nan=65.0)

        target_dict["meds_per_day"] = num_meds / time_hosp
        target_dict["labs_per_day"] = num_labs / time_hosp
        target_dict["procs_per_day"] = num_procs / time_hosp
        target_dict["meds_per_diag"] = num_meds / num_diags
        target_dict["service_density"] = (num_labs + num_procs + num_meds) / time_hosp
        target_dict["total_procedures"] = num_labs + num_procs

        target_dict["inp_x_age"] = inp * clean_age
        target_dict["inp_x_diags"] = inp * num_diags
        target_dict["inp_x_time"] = inp * time_hosp
        target_dict["inp_x_meds"] = inp * num_meds
        target_dict["age_x_meds"] = clean_age * num_meds

    extra_numeric.extend([
        "meds_per_day",
        "labs_per_day",
        "procs_per_day",
        "meds_per_diag",
        "service_density",
        "total_procedures",
        "inp_x_age",
        "inp_x_diags",
        "inp_x_time",
        "inp_x_meds",
        "age_x_meds",
    ])

    # 7. Medication Changes & Insulin Dynamics
    medication_names = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    med_cols = [c for c in medication_names if c in train]

    for d, row_count, split_name in ((train, n_train, "train"), (eval_cols, n_eval, "eval")):
        target_dict = train_dict if split_name == "train" else eval_dict
        active_count = np.zeros(row_count, dtype=np.float64)
        change_count = np.zeros(row_count, dtype=np.float64)
        up_count = np.zeros(row_count, dtype=np.float64)
        down_count = np.zeros(row_count, dtype=np.float64)

        for med in med_cols:
            arr = d[med]
            active_count += (arr != "No").astype(np.float64)
            up_count += (arr == "Up").astype(np.float64)
            down_count += (arr == "Down").astype(np.float64)
            change_count += np.isin(arr, ["Up", "Down"]).astype(np.float64)

        insulin_arr = d.get("insulin", np.array(["No"] * row_count))
        ch_arr = d.get("change", np.array(["No"] * row_count))
        dm_arr = d.get("diabetesMed", np.array(["No"] * row_count))

        target_dict["num_active_meds"] = active_count
        target_dict["num_med_changes"] = change_count
        target_dict["num_med_up"] = up_count
        target_dict["num_med_down"] = down_count
        target_dict["insulin_active"] = (insulin_arr != "No").astype(np.float64)
        target_dict["insulin_up"] = (insulin_arr == "Up").astype(np.float64)
        target_dict["insulin_down"] = (insulin_arr == "Down").astype(np.float64)
        target_dict["insulin_steady"] = (insulin_arr == "Steady").astype(np.float64)
        target_dict["insulin_changed"] = np.isin(insulin_arr, ["Up", "Down"]).astype(np.float64)
        target_dict["change_flag"] = (ch_arr == "Ch").astype(np.float64)
        target_dict["diabetes_med_flag"] = (dm_arr == "Yes").astype(np.float64)

    extra_numeric.extend([
        "num_active_meds",
        "num_med_changes",
        "num_med_up",
        "num_med_down",
        "insulin_active",
        "insulin_up",
        "insulin_down",
        "insulin_steady",
        "insulin_changed",
        "change_flag",
        "diabetes_med_flag",
    ])

    # 8. Lab Result Flags & Clinically Proven Interplay
    for d, row_count, split_name in ((train, n_train, "train"), (eval_cols, n_eval, "eval")):
        target_dict = train_dict if split_name == "train" else eval_dict
        a1c = d.get("A1Cresult", np.array(["None"] * row_count))
        glu = d.get("max_glu_serum", np.array(["None"] * row_count))
        ch = d.get("change", np.array(["No"] * row_count))

        a1c_tested = (a1c != "None").astype(np.float64)
        a1c_high = np.isin(a1c, [">7", ">8"]).astype(np.float64)
        glu_tested = (glu != "None").astype(np.float64)
        glu_high = np.isin(glu, [">200", ">300"]).astype(np.float64)

        target_dict["a1c_tested"] = a1c_tested
        target_dict["a1c_high"] = a1c_high
        target_dict["a1c_norm"] = (a1c == "Norm").astype(np.float64)
        target_dict["a1c_high_no_change"] = (a1c_high * (ch != "Ch")).astype(np.float64)
        target_dict["glu_tested"] = glu_tested
        target_dict["glu_high"] = glu_high
        target_dict["glu_norm"] = (glu == "Norm").astype(np.float64)

    extra_numeric.extend([
        "a1c_tested",
        "a1c_high",
        "a1c_norm",
        "a1c_high_no_change",
        "glu_tested",
        "glu_high",
        "glu_norm",
    ])

    # 9. Discharge Disposition & Admission Source Hazards
    for d, row_count, split_name in ((train, n_train, "train"), (eval_cols, n_eval, "eval")):
        target_dict = train_dict if split_name == "train" else eval_dict
        disch = d.get("discharge_disposition_id", np.array(["1"] * row_count))
        adm_src = d.get("admission_source_id", np.array(["1"] * row_count))
        adm_type = d.get("admission_type_id", np.array(["1"] * row_count))

        target_dict["disch_home"] = (disch == "1").astype(np.float64)
        target_dict["disch_snf"] = (disch == "3").astype(np.float64)
        target_dict["disch_homehealth"] = (disch == "6").astype(np.float64)
        target_dict["disch_snf_or_hh"] = np.isin(disch, ["3", "6"]).astype(np.float64)
        target_dict["disch_ama"] = (disch == "7").astype(np.float64)
        target_dict["disch_rehab"] = (disch == "22").astype(np.float64)
        target_dict["adm_emergency"] = (adm_src == "7").astype(np.float64)
        target_dict["adm_transfer"] = np.isin(adm_src, ["1", "2", "4", "5", "6"]).astype(np.float64)
        target_dict["adm_type_emergency"] = (adm_type == "1").astype(np.float64)
        target_dict["adm_type_urgent"] = (adm_type == "2").astype(np.float64)
        target_dict["adm_type_elective"] = (adm_type == "3").astype(np.float64)

    extra_numeric.extend([
        "disch_home",
        "disch_snf",
        "disch_homehealth",
        "disch_snf_or_hh",
        "disch_ama",
        "disch_rehab",
        "adm_emergency",
        "adm_transfer",
        "adm_type_emergency",
        "adm_type_urgent",
        "adm_type_elective",
    ])

    # Tabular Encoding
    encoder = TabularEncoder(min_count=40)
    x_train = encoder.fit_transform(
        task,
        train_dict,
        extra_numeric=extra_numeric,
        extra_categorical=extra_categorical,
    )
    x_eval = encoder.transform(eval_dict)
    cat_indices = encoder.categorical_indices

    # Diverse Multi-Model LightGBM Ensemble
    models = [
        lgb.LGBMClassifier(
            n_estimators=260,
            learning_rate=0.020,
            num_leaves=31,
            max_depth=6,
            min_child_samples=150,
            subsample=0.72,
            subsample_freq=1,
            colsample_bytree=0.60,
            reg_lambda=1.5,
            reg_alpha=0.2,
            cat_smooth=50.0,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.018,
            num_leaves=18,
            max_depth=5,
            min_child_samples=250,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.50,
            reg_lambda=4.0,
            reg_alpha=0.5,
            cat_smooth=80.0,
            random_state=101,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=220,
            learning_rate=0.022,
            num_leaves=48,
            max_depth=8,
            min_child_samples=100,
            subsample=0.68,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_lambda=0.5,
            reg_alpha=0.05,
            cat_smooth=35.0,
            random_state=2024,
            n_jobs=1,
            verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.020,
            num_leaves=25,
            max_depth=6,
            min_child_samples=180,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_lambda=2.0,
            reg_alpha=0.3,
            cat_smooth=60.0,
            random_state=777,
            n_jobs=1,
            verbose=-1,
        ),
    ]
    weights = [0.30, 0.25, 0.25, 0.20]

    final_pred = np.zeros(n_eval, dtype=np.float64)
    for model, w in zip(models, weights):
        model.fit(x_train, y_train, categorical_feature=cat_indices)
        final_pred += w * model.predict_proba(x_eval)[:, 1]

    return np.clip(final_pred, 0.0, 1.0)


# EVOLVE-BLOCK-END
# =====================================================================
