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
    import numpy as np
    import lightgbm as lgb
    import xgboost as xgb
    
    # 1. Feature Engineering
    extra_numeric = []
    extra_categorical = []
    
    n_train = len(y_train)
    n_eval = len(next(iter(eval_cols.values())))
    
    # Comorbidities & Charlson Comorbidity Index
    diag_cols_present = [col for col in ["diag_1", "diag_2", "diag_3"] if col in train]
    if diag_cols_present:
        comorbs_train = icd9_comorbidities(*(train[col] for col in diag_cols_present))
        comorbs_eval = icd9_comorbidities(*(eval_cols[col] for col in diag_cols_present))
        for k in comorbs_train.keys():
            train[k] = comorbs_train[k]
            eval_cols[k] = comorbs_eval[k]
            extra_numeric.append(k)
            
        cci_weights = {
            "cm_mi": 1.0, "cm_chf": 1.0, "cm_pvd": 1.0, "cm_cerebrovascular": 1.0,
            "cm_dementia": 1.0, "cm_pulmonary": 1.0, "cm_rheumatic": 1.0, "cm_peptic_ulcer": 1.0,
            "cm_liver": 1.0, "cm_hemiplegia": 2.0, "cm_renal": 2.0, "cm_diabetes_complicated": 2.0,
            "cm_cancer": 2.0, "cm_metastatic": 6.0, "cm_hiv": 6.0
        }
        cci_train = np.zeros(n_train)
        cci_eval = np.zeros(n_eval)
        for k, w in cci_weights.items():
            if k in train:
                cci_train += train[k] * w
                cci_eval += eval_cols[k] * w
        train["charlson_index"] = cci_train
        eval_cols["charlson_index"] = cci_eval
        extra_numeric.append("charlson_index")
            
    # Grouped ICD9 chapters
    def safe_icd9_group(x):
        if x is None or x == "" or x == "?" or (isinstance(x, float) and np.isnan(x)):
            return "missing"
        return icd9_group(str(x))
        
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in train:
            grp_train = np.array([safe_icd9_group(x) for x in train[col]], dtype=object)
            grp_eval = np.array([safe_icd9_group(x) for x in eval_cols[col]], dtype=object)
            grp_name = f"{col}_group"
            train[grp_name] = grp_train
            eval_cols[grp_name] = grp_eval
            extra_categorical.append(grp_name)
            
    # Medication Features
    med_cols = [
        col for col in task.categorical_columns 
        if col in train and any(val in ["Steady", "Up", "Down"] for val in np.unique(train[col].astype(str)))
    ]
    if med_cols:
        active_train = np.zeros(n_train, dtype=np.float64)
        up_train = np.zeros(n_train, dtype=np.float64)
        down_train = np.zeros(n_train, dtype=np.float64)
        steady_train = np.zeros(n_train, dtype=np.float64)
        
        active_eval = np.zeros(n_eval, dtype=np.float64)
        up_eval = np.zeros(n_eval, dtype=np.float64)
        down_eval = np.zeros(n_eval, dtype=np.float64)
        steady_eval = np.zeros(n_eval, dtype=np.float64)
        
        for col in med_cols:
            vt = train[col]
            ve = eval_cols[col]
            active_train += (vt != "No") & (vt != None)
            up_train += (vt == "Up")
            down_train += (vt == "Down")
            steady_train += (vt == "Steady")
            
            active_eval += (ve != "No") & (ve != None)
            up_eval += (ve == "Up")
            down_eval += (ve == "Down")
            steady_eval += (ve == "Steady")
            
        train["num_active_meds"] = active_train
        train["num_up_meds"] = up_train
        train["num_down_meds"] = down_train
        train["num_steady_meds"] = steady_train
        
        eval_cols["num_active_meds"] = active_eval
        eval_cols["num_up_meds"] = up_eval
        eval_cols["num_down_meds"] = down_eval
        eval_cols["num_steady_meds"] = steady_eval

        train["med_volatility"] = (up_train + down_train) / (active_train + 1e-5)
        eval_cols["med_volatility"] = (up_eval + down_eval) / (active_eval + 1e-5)
        train["med_torque"] = up_train - down_train
        eval_cols["med_torque"] = up_eval - down_eval
        
        extra_numeric.extend(["num_active_meds", "num_up_meds", "num_down_meds", "num_steady_meds", "med_volatility", "med_torque"])

        if "insulin" in train:
            for d in [train, eval_cols]:
                ins = d["insulin"]
                d["on_insulin"] = ((ins != "No") & (ins != None)).astype(float)
                d["insulin_up"] = (ins == "Up").astype(float)
                d["insulin_down"] = (ins == "Down").astype(float)
            extra_numeric.extend(["on_insulin", "insulin_up", "insulin_down"])

        # Low-Rank Medication Matrix Factorization (PCA)
        from sklearn.decomposition import PCA
        med_matrix_train = np.zeros((n_train, len(med_cols)))
        med_matrix_eval = np.zeros((n_eval, len(med_cols)))
        mapping = {"No": 0, "Steady": 1, "Down": -1, "Up": 2}
        for i, col in enumerate(med_cols):
            val_train = train[col]
            val_eval = eval_cols[col]
            med_matrix_train[:, i] = np.array([mapping.get(x, 0) for x in val_train])
            med_matrix_eval[:, i] = np.array([mapping.get(x, 0) for x in val_eval])

        pca = PCA(n_components=4, random_state=42)
        pca_train = pca.fit_transform(med_matrix_train)
        pca_eval = pca.transform(med_matrix_eval)

        for j in range(4):
            col_name = f"med_pca_{j}"
            train[col_name] = pca_train[:, j]
            eval_cols[col_name] = pca_eval[:, j]
            extra_numeric.append(col_name)

    # Comorbidity Synergies (Cardiorenal, Cardiopulmonary, Cardiometabolic)
    cm_pairs = [
        ("cm_chf", "cm_renal", "cm_cardiorenal"),
        ("cm_chf", "cm_pulmonary", "cm_cardiopulmonary"),
        ("cm_chf", "cm_diabetes_complicated", "cm_cardiometabolic"),
        ("cm_cerebrovascular", "cm_pvd", "cm_vascular_disease"),
    ]
    for c1, c2, cname in cm_pairs:
        if c1 in train and c2 in train:
            train[cname] = (train[c1] * train[c2]).astype(float)
            eval_cols[cname] = (eval_cols[c1] * eval_cols[c2]).astype(float)
            extra_numeric.append(cname)

    # Diagnostic Triad Syndrome & Shannon Multidisciplinary Entropy
    if all(f"{col}_group" in train for col in ["diag_1", "diag_2", "diag_3"]):
        g1_tr, g2_tr, g3_tr = train["diag_1_group"], train["diag_2_group"], train["diag_3_group"]
        g1_ev, g2_ev, g3_ev = eval_cols["diag_1_group"], eval_cols["diag_2_group"], eval_cols["diag_3_group"]
        
        syndrome_tr = np.array(["_".join(sorted([str(a), str(b), str(c)])) for a, b, c in zip(g1_tr, g2_tr, g3_tr)], dtype=object)
        syndrome_ev = np.array(["_".join(sorted([str(a), str(b), str(c)])) for a, b, c in zip(g1_ev, g2_ev, g3_ev)], dtype=object)
        train["diag_syndrome"] = syndrome_tr
        eval_cols["diag_syndrome"] = syndrome_ev
        extra_categorical.append("diag_syndrome")

        syn_te_tr, syn_te_ev = oof_target_encode(syndrome_tr, y_train, syndrome_ev, n_folds=5, smoothing=20.0, seed=42)
        train["diag_syndrome_te"] = syn_te_tr
        eval_cols["diag_syndrome_te"] = syn_te_ev
        extra_numeric.append("diag_syndrome_te")

        def compute_diag_entropy(g1, g2, g3):
            n = len(g1)
            ent = np.zeros(n, dtype=np.float64)
            for i in range(n):
                items = [g1[i], g2[i], g3[i]]
                counts = {}
                for item in items:
                    counts[item] = counts.get(item, 0) + 1
                e = 0.0
                for c in counts.values():
                    p = c / 3.0
                    e -= p * np.log2(p)
                ent[i] = e
            return ent

        train["diag_entropy"] = compute_diag_entropy(g1_tr, g2_tr, g3_tr)
        eval_cols["diag_entropy"] = compute_diag_entropy(g1_ev, g2_ev, g3_ev)
        extra_numeric.append("diag_entropy")

    # Glycemic Treatment Inaction Paradox & Lab-Medication Synergy
    if "A1Cresult" in train and "change" in train:
        for d in [train, eval_cols]:
            a1c = d["A1Cresult"]
            chg = d["change"]
            glu = d.get("max_glu_serum")
            
            high_a1c = (a1c == ">8") | (a1c == ">7")
            d["a1c_tested"] = (a1c != "None").astype(float)
            d["a1c_uncontrolled_neglected"] = (high_a1c & (chg == "No")).astype(float)
            d["a1c_uncontrolled_addressed"] = (high_a1c & (chg == "Ch")).astype(float)
            
            if glu is not None:
                high_glu = (glu == ">200") | (glu == ">300")
                d["glu_tested"] = (glu != "None").astype(float)
                d["glu_uncontrolled_neglected"] = (high_glu & (chg == "No")).astype(float)
                d["glu_uncontrolled_addressed"] = (high_glu & (chg == "Ch")).astype(float)

        extra_numeric.extend(["a1c_tested", "a1c_uncontrolled_neglected", "a1c_uncontrolled_addressed"])
        if "max_glu_serum" in train:
            extra_numeric.extend(["glu_tested", "glu_uncontrolled_neglected", "glu_uncontrolled_addressed"])

    # Contraindication: Metformin in Renal Failure (MALA Risk)
    if "metformin" in train and "cm_renal" in train:
        train["metformin_renal_contra"] = ((train["metformin"] != "No") & (train["metformin"] != None) & (train["cm_renal"] > 0)).astype(float)
        eval_cols["metformin_renal_contra"] = ((eval_cols["metformin"] != "No") & (eval_cols["metformin"] != None) & (eval_cols["cm_renal"] > 0)).astype(float)
        extra_numeric.append("metformin_renal_contra")

    # Demographics and Continuous Age Parsing
    def parse_age(age_str):
        if age_str is None or not isinstance(age_str, str):
            return np.nan
        try:
            cleaned = age_str.replace("[", "").replace(")", "").replace("]", "")
            parts = cleaned.split("-")
            if len(parts) == 2:
                return (float(parts[0]) + float(parts[1])) / 2.0
        except Exception:
            pass
        return np.nan

    if "age" in train:
        age_tr = np.array([parse_age(x) for x in train["age"]])
        age_ev = np.array([parse_age(x) for x in eval_cols["age"]])
        train["age_numeric"] = age_tr
        eval_cols["age_numeric"] = age_ev
        train["is_elderly"] = (age_tr >= 75.0).astype(float)
        eval_cols["is_elderly"] = (age_ev >= 75.0).astype(float)
        extra_numeric.extend(["age_numeric", "is_elderly"])

    # Informative Missingness indicators
    if "weight" in train:
        train["has_weight"] = np.array([float(x is not None and x != "?") for x in train["weight"]])
        eval_cols["has_weight"] = np.array([float(x is not None and x != "?") for x in eval_cols["weight"]])
        extra_numeric.append("has_weight")
        
    for col in ["medical_specialty", "payer_code"]:
        if col in train:
            col_name = f"has_{col}"
            train[col_name] = np.array([float(x is not None and x != "?") for x in train[col]])
            eval_cols[col_name] = np.array([float(x is not None and x != "?") for x in eval_cols[col]])
            extra_numeric.append(col_name)

    # Key discharge destination indicators
    if "discharge_disposition_id" in train:
        for dest in ["1", "3", "6", "18"]:
            col_name = f"discharged_to_{dest}"
            train[col_name] = (train["discharge_disposition_id"] == dest).astype(float)
            eval_cols[col_name] = (eval_cols["discharge_disposition_id"] == dest).astype(float)
            extra_numeric.append(col_name)
        for d in [train, eval_cols]:
            disp = d["discharge_disposition_id"]
            d["discharged_facility"] = np.isin(disp, ["3", "4", "5", "22", "23", "24"]).astype(float)
            d["discharged_ama"] = np.isin(disp, ["7", "18"]).astype(float)
        extra_numeric.extend(["discharged_facility", "discharged_ama"])
        
    # Interaction, Utilization & Care Transition Friction Features
    num_cols = ["number_outpatient", "number_emergency", "number_inpatient", "num_lab_procedures", "num_procedures", "num_medications", "time_in_hospital"]
    if all(col in train for col in num_cols):
        for d in [train, eval_cols]:
            d["total_visits"] = d["number_outpatient"] + d["number_emergency"] + d["number_inpatient"]
            d["has_inpatient"] = (d["number_inpatient"] > 0).astype(float)
            d["has_emergency"] = (d["number_emergency"] > 0).astype(float)
            d["has_outpatient"] = (d["number_outpatient"] > 0).astype(float)
            d["severity_idx"] = d["num_lab_procedures"] + d["num_procedures"] + d["num_medications"]
            d["log_num_lab_procedures"] = np.log1p(d["num_lab_procedures"])
            d["log_num_medications"] = np.log1p(d["num_medications"])
            d["log_total_visits"] = np.log1p(d["total_visits"])
            
            time_h = d["time_in_hospital"] + 1e-5
            d["procedures_per_day"] = d["num_procedures"] / time_h
            d["meds_per_day"] = d["num_medications"] / time_h
            d["labs_per_day"] = d["num_lab_procedures"] / time_h
            d["discharge_pressure"] = d["severity_idx"] / (d["time_in_hospital"] + 0.5)
            
            d["labs_per_med"] = d["num_lab_procedures"] / (d["num_medications"] + 1e-5)
            d["inpatient_ratio"] = d["number_inpatient"] / (d["total_visits"] + 1e-5)
            d["emergency_ratio"] = d["number_emergency"] / (d["total_visits"] + 1e-5)
            
            d["recidivism_velocity"] = 3.0 * d["number_inpatient"] + 1.5 * d["number_emergency"] + 0.5 * d["number_outpatient"]
            d["inpatient_per_stay"] = d["number_inpatient"] / (d["time_in_hospital"] + 1.0)
            d["is_frequent_inpatient"] = (d["number_inpatient"] >= 2).astype(float)

        extra_numeric.extend([
            "total_visits", "has_inpatient", "has_emergency", "has_outpatient", 
            "severity_idx", "log_num_lab_procedures", "log_num_medications", 
            "log_total_visits", "procedures_per_day", "meds_per_day", "labs_per_day",
            "discharge_pressure", "labs_per_med", "inpatient_ratio", "emergency_ratio",
            "recidivism_velocity", "inpatient_per_stay", "is_frequent_inpatient"
        ])
        
    # 3-Digit ICD-9 Category Prefix Extraction & Hierarchy
    def extract_prefix(val):
        if val is None or val == "" or val == "?" or (isinstance(val, float) and np.isnan(val)):
            return "missing"
        s = str(val).strip()
        return s.split(".")[0][:3] if s else "missing"

    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in train:
            p_tr = np.array([extract_prefix(x) for x in train[col]], dtype=object)
            p_ev = np.array([extract_prefix(x) for x in eval_cols[col]], dtype=object)
            p_name = f"{col}_prefix"
            train[p_name] = p_tr
            eval_cols[p_name] = p_ev
            extra_categorical.append(p_name)

            p_te_tr, p_te_ev = oof_target_encode(p_tr, y_train, p_ev, n_folds=5, smoothing=20.0, seed=42)
            train[f"{p_name}_te"] = p_te_tr
            eval_cols[f"{p_name}_te"] = p_te_ev
            extra_numeric.append(f"{p_name}_te")

    # Max and Mean Diagnosis Risk Profile across all 3 diagnoses
    if all(f"diag_{i}_prefix_te" in train for i in [1, 2, 3]):
        train["max_diag_risk"] = np.maximum(train["diag_1_prefix_te"], np.maximum(train["diag_2_prefix_te"], train["diag_3_prefix_te"]))
        eval_cols["max_diag_risk"] = np.maximum(eval_cols["diag_1_prefix_te"], np.maximum(eval_cols["diag_2_prefix_te"], eval_cols["diag_3_prefix_te"]))
        train["mean_diag_risk"] = (train["diag_1_prefix_te"] + train["diag_2_prefix_te"] + train["diag_3_prefix_te"]) / 3.0
        eval_cols["mean_diag_risk"] = (eval_cols["diag_1_prefix_te"] + eval_cols["diag_2_prefix_te"] + eval_cols["diag_3_prefix_te"]) / 3.0
        extra_numeric.extend(["max_diag_risk", "mean_diag_risk"])

    # Detailed Diabetic Pathophysiology (Uncontrolled & Complicated Staging)
    def parse_diabetes_status(raw_val):
        if raw_val is None:
            return 0.0, 0.0
        s = str(raw_val).strip()
        if not s.startswith("250"):
            return 0.0, 0.0
        uncontrolled, complication = 0.0, 0.0
        if "." in s:
            parts = s.split(".")
            if len(parts) > 1 and len(parts[1]) > 0:
                dec = parts[1]
                if dec[0] != "0":
                    complication = 1.0
                if len(dec) > 1 and dec[1] in ["2", "3"]:
                    uncontrolled = 1.0
        elif len(s) >= 4:
            if s[3] != "0":
                complication = 1.0
            if len(s) >= 5 and s[4] in ["2", "3"]:
                uncontrolled = 1.0
        return uncontrolled, complication

    uncontrolled_tr = np.zeros(n_train, dtype=np.float64)
    uncontrolled_ev = np.zeros(n_eval, dtype=np.float64)
    comp_tr = np.zeros(n_train, dtype=np.float64)
    comp_ev = np.zeros(n_eval, dtype=np.float64)
    for col in diag_cols_present:
        for i, x in enumerate(train[col]):
            u, c = parse_diabetes_status(x)
            if u: uncontrolled_tr[i] = 1.0
            if c: comp_tr[i] = 1.0
        for i, x in enumerate(eval_cols[col]):
            u, c = parse_diabetes_status(x)
            if u: uncontrolled_ev[i] = 1.0
            if c: comp_ev[i] = 1.0

    train["diag_uncontrolled_dm"] = uncontrolled_tr
    eval_cols["diag_uncontrolled_dm"] = uncontrolled_ev
    train["diag_complicated_dm"] = comp_tr
    eval_cols["diag_complicated_dm"] = comp_ev
    extra_numeric.extend(["diag_uncontrolled_dm", "diag_complicated_dm"])

    # Major Clinical Phenotype Flags Across Any Diagnostic Position
    clinical_phenotypes = {
        "cond_heart_failure": ["428"],
        "cond_copd_asthma": ["491", "492", "493", "494", "496"],
        "cond_kidney_disease": ["584", "585", "586"],
        "cond_ischemic_heart": ["410", "411", "412", "413", "414"],
        "cond_hypertension": ["401", "402", "403", "404", "405"],
        "cond_acute_infection": ["038", "486", "482", "599", "682"]
    }
    for pheno_name, target_prefixes in clinical_phenotypes.items():
        tr_flag = np.zeros(n_train, dtype=np.float64)
        ev_flag = np.zeros(n_eval, dtype=np.float64)
        for col in ["diag_1_prefix", "diag_2_prefix", "diag_3_prefix"]:
            if col in train:
                tr_flag = np.maximum(tr_flag, np.isin(train[col], target_prefixes).astype(float))
                ev_flag = np.maximum(ev_flag, np.isin(eval_cols[col], target_prefixes).astype(float))
        train[pheno_name] = tr_flag
        eval_cols[pheno_name] = ev_flag
        extra_numeric.append(pheno_name)

    # Validated Clinical Decision Score: LACE Index
    if all(k in train for k in ["time_in_hospital", "admission_type_id", "charlson_index", "number_emergency"]):
        def compute_lace(stay, adm_type, cci, ed):
            n = len(stay)
            lace = np.zeros(n, dtype=np.float64)
            for i in range(n):
                s = stay[i]
                if s < 1: l = 0.0
                elif s == 1: l = 1.0
                elif s == 2: l = 2.0
                elif s == 3: l = 3.0
                elif s <= 6: l = 4.0
                elif s <= 13: l = 5.0
                else: l = 7.0

                a = 3.0 if str(adm_type[i]) == "1" else 0.0

                c = cci[i]
                if c == 0: cc = 0.0
                elif c == 1: cc = 1.0
                elif c == 2: cc = 2.0
                elif c == 3: cc = 3.0
                else: cc = 5.0

                e = ed[i]
                if e == 0: ee = 0.0
                elif e == 1: ee = 1.0
                elif e == 2: ee = 2.0
                elif e == 3: ee = 3.0
                else: ee = 4.0

                lace[i] = l + a + cc + ee
            return lace

        train["lace_score"] = compute_lace(train["time_in_hospital"], train["admission_type_id"], train["charlson_index"], train["number_emergency"])
        eval_cols["lace_score"] = compute_lace(eval_cols["time_in_hospital"], eval_cols["admission_type_id"], eval_cols["charlson_index"], eval_cols["number_emergency"])
        train["lace_high_risk"] = (train["lace_score"] >= 10.0).astype(float)
        eval_cols["lace_high_risk"] = (eval_cols["lace_score"] >= 10.0).astype(float)
        extra_numeric.extend(["lace_score", "lace_high_risk"])

    # Pharmacotherapy Class Breadth, Hypoglycemia Risk & Drug Interactions
    sulfonylureas = {"chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "tolazamide"}
    tzds = {"pioglitazone", "rosiglitazone"}
    biguanides = {"metformin"}
    meglitinides = {"repaglinide", "nateglinide"}
    agis = {"acarbose", "miglitol"}

    for d in [train, eval_cols]:
        on_su = np.zeros(len(d["age"]), dtype=np.float64)
        on_tzd = np.zeros(len(d["age"]), dtype=np.float64)
        on_big = np.zeros(len(d["age"]), dtype=np.float64)
        on_meg = np.zeros(len(d["age"]), dtype=np.float64)
        on_agi = np.zeros(len(d["age"]), dtype=np.float64)

        for med in sulfonylureas:
            if med in d:
                on_su = np.maximum(on_su, ((d[med] != "No") & (d[med] != None)).astype(float))
        for med in tzds:
            if med in d:
                on_tzd = np.maximum(on_tzd, ((d[med] != "No") & (d[med] != None)).astype(float))
        for med in biguanides:
            if med in d:
                on_big = np.maximum(on_big, ((d[med] != "No") & (d[med] != None)).astype(float))
        for med in meglitinides:
            if med in d:
                on_meg = np.maximum(on_meg, ((d[med] != "No") & (d[med] != None)).astype(float))
        for med in agis:
            if med in d:
                on_agi = np.maximum(on_agi, ((d[med] != "No") & (d[med] != None)).astype(float))

        num_oral = on_su + on_tzd + on_big + on_meg + on_agi
        d["num_oral_classes"] = num_oral
        d["num_total_classes"] = num_oral + d.get("on_insulin", 0.0)
        d["insulin_and_oral"] = (d.get("on_insulin", 0.0) * (num_oral > 0)).astype(float)
        d["hypo_risk"] = (d.get("on_insulin", 0.0) * on_su).astype(float)
        if "cm_chf" in d:
            d["tzd_chf_contra"] = (on_tzd * (d["cm_chf"] > 0)).astype(float)

    extra_numeric.extend(["num_oral_classes", "num_total_classes", "insulin_and_oral", "hypo_risk"])
    if "cm_chf" in train:
        extra_numeric.append("tzd_chf_contra")

    # Additional Clinical Utilization & Intensity Ratios
    for d in [train, eval_cols]:
        d["inpatient_to_stay_ratio"] = d["number_inpatient"] / (d["time_in_hospital"] + 1.0)
        d["acute_to_outpatient_ratio"] = (d["number_inpatient"] + d["number_emergency"]) / (d["number_outpatient"] + 1.0)
        d["meds_per_diagnosis"] = d["num_medications"] / (d["number_diagnoses"] + 1e-5)
        d["labs_per_diagnosis"] = d["num_lab_procedures"] / (d["number_diagnoses"] + 1e-5)
        d["procedures_per_diagnosis"] = d["num_procedures"] / (d["number_diagnoses"] + 1e-5)
        d["diagnoses_per_day"] = d["number_diagnoses"] / (d["time_in_hospital"] + 0.5)
        d["lab_intensity_per_med"] = d["num_lab_procedures"] / (d["num_medications"] + 1.0)

    extra_numeric.extend([
        "inpatient_to_stay_ratio", "acute_to_outpatient_ratio", "meds_per_diagnosis",
        "labs_per_diagnosis", "procedures_per_diagnosis", "diagnoses_per_day", "lab_intensity_per_med"
    ])

    # Discharge Disposition Transition Friction & Interactions
    if "discharge_disposition_id" in train:
        for d in [train, eval_cols]:
            disp = d["discharge_disposition_id"]
            snf = (disp == "3").astype(float)
            hh = (disp == "6").astype(float)
            ama = np.isin(disp, ["7", "18"]).astype(float)
            prior_inp = (d["number_inpatient"] > 0).astype(float)
            d["discharged_snf"] = snf
            d["discharged_home_health"] = hh
            d["snf_prior_inpatient"] = snf * prior_inp
            d["home_health_prior_inpatient"] = hh * prior_inp
            d["high_risk_discharge_prior_inp"] = (snf | hh | ama) * prior_inp

        extra_numeric.extend([
            "discharged_snf", "discharged_home_health", "snf_prior_inpatient",
            "home_health_prior_inpatient", "high_risk_discharge_prior_inp"
        ])

    # Inpatient Recidivism x Discharge Disposition Interaction
    if "number_inpatient" in train and "discharge_disposition_id" in train:
        inp_binned_tr = np.clip(train["number_inpatient"], 0, 3).astype(int).astype(str)
        inp_binned_ev = np.clip(eval_cols["number_inpatient"], 0, 3).astype(int).astype(str)
        train["inpatient_discharge_comb"] = np.array([f"{a}_{str(b)}" for a, b in zip(inp_binned_tr, train["discharge_disposition_id"])], dtype=object)
        eval_cols["inpatient_discharge_comb"] = np.array([f"{a}_{str(b)}" for a, b in zip(inp_binned_ev, eval_cols["discharge_disposition_id"])], dtype=object)
        extra_categorical.append("inpatient_discharge_comb")

        tr_enc, ev_enc = oof_target_encode(train["inpatient_discharge_comb"], y_train, eval_cols["inpatient_discharge_comb"], n_folds=5, smoothing=20.0, seed=42)
        train["inpatient_discharge_comb_te"] = tr_enc
        eval_cols["inpatient_discharge_comb_te"] = ev_enc
        extra_numeric.append("inpatient_discharge_comb_te")

    # Frequency / Prevalence Rarity Encodings
    freq_cols = ["diag_1_prefix", "medical_specialty", "discharge_disposition_id", "admission_source_id", "payer_code"]
    for col in freq_cols:
        if col in train:
            counts = {}
            for x in train[col]:
                sx = str(x)
                counts[sx] = counts.get(sx, 0) + 1
            train[f"{col}_freq"] = np.array([counts.get(str(x), 0) / float(n_train) for x in train[col]], dtype=np.float64)
            eval_cols[f"{col}_freq"] = np.array([counts.get(str(x), 0) / float(n_train) for x in eval_cols[col]], dtype=np.float64)
            extra_numeric.append(f"{col}_freq")

    # Cartesian Interaction Pairs for High-Cardinality Encodings
    pairs = [
        ("discharge_disposition_id", "admission_source_id"),
        ("admission_type_id", "discharge_disposition_id"),
        ("discharge_disposition_id", "diag_1_group"),
        ("admission_source_id", "diag_1_group"),
        ("discharge_disposition_id", "age"),
        ("discharge_disposition_id", "diag_1_prefix")
    ]
    for col1, col2 in pairs:
        if col1 in train and col2 in train:
            comb_train = np.array([f"{str(x1)}_{str(x2)}" for x1, x2 in zip(train[col1], train[col2])], dtype=object)
            comb_eval = np.array([f"{str(x1)}_{str(x2)}" for x1, x2 in zip(eval_cols[col1], eval_cols[col2])], dtype=object)
            comb_name = f"{col1}_{col2}_comb"
            train[comb_name] = comb_train
            eval_cols[comb_name] = comb_eval
            extra_categorical.append(comb_name)

            train_enc, eval_enc = oof_target_encode(
                comb_train, y_train, comb_eval, n_folds=5, smoothing=20.0, seed=42
            )
            te_name = f"{comb_name}_te"
            train[te_name] = train_enc
            eval_cols[te_name] = eval_enc
            extra_numeric.append(te_name)

    # Target Encoding of standard High Cardinality columns
    te_cols = [
        "diag_1", "diag_2", "diag_3",
        "medical_specialty", "discharge_disposition_id",
        "admission_source_id", "admission_type_id",
        "payer_code", "age"
    ]
    for col in te_cols:
        if col in train:
            train_enc, eval_enc = oof_target_encode(
                train[col], y_train, eval_cols[col], n_folds=5, smoothing=20.0, seed=42
            )
            te_name = f"{col}_te"
            train[te_name] = train_enc
            eval_cols[te_name] = eval_enc
            extra_numeric.append(te_name)

    # 2. Encode all columns with TabularEncoder
    encoder = TabularEncoder(min_count=25)
    x_train = encoder.fit_transform(task, train, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_eval = encoder.transform(eval_cols)

    # 3. Fit Multi-Engine Model Ensemble
    from sklearn.ensemble import HistGradientBoostingClassifier

    model1 = lgb.LGBMClassifier(
        n_estimators=240,
        learning_rate=0.018,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=1.5,
        reg_alpha=0.3,
        cat_smooth=40.0,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    model2 = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.012,
        num_leaves=63,
        min_child_samples=60,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_lambda=3.0,
        reg_alpha=1.0,
        cat_smooth=70.0,
        random_state=100,
        n_jobs=1,
        verbose=-1,
    )

    # Extremely Randomized Trees GBDT for low prediction correlation
    model3 = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.022,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.7,
        extra_trees=True,
        reg_lambda=1.0,
        cat_smooth=25.0,
        random_state=2023,
        n_jobs=1,
        verbose=-1,
    )

    model4 = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_lambda=2.5,
        reg_alpha=0.5,
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )

    model5 = xgb.XGBClassifier(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.016,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        reg_alpha=1.0,
        tree_method="hist",
        random_state=2024,
        n_jobs=1,
        verbosity=0,
    )

    model6 = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.025,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=2.0,
        random_state=42,
    )

    # Fit models
    cat_feats = encoder.categorical_indices
    model1.fit(x_train, y_train, categorical_feature=cat_feats)
    model2.fit(x_train, y_train, categorical_feature=cat_feats)
    model3.fit(x_train, y_train, categorical_feature=cat_feats)
    model4.fit(x_train, y_train)
    model5.fit(x_train, y_train)
    model6.fit(x_train, y_train)

    # Predict probabilities
    p1 = model1.predict_proba(x_eval)[:, 1]
    p2 = model2.predict_proba(x_eval)[:, 1]
    p3 = model3.predict_proba(x_eval)[:, 1]
    p4 = model4.predict_proba(x_eval)[:, 1]
    p5 = model5.predict_proba(x_eval)[:, 1]
    p6 = model6.predict_proba(x_eval)[:, 1]

    weights = [0.22, 0.20, 0.15, 0.18, 0.13, 0.12]
    p_raw = (
        weights[0] * p1 +
        weights[1] * p2 +
        weights[2] * p3 +
        weights[3] * p4 +
        weights[4] * p5 +
        weights[5] * p6
    )

    # Rank-Quantile Congruence Coupling
    from scipy.stats import rankdata
    r1 = rankdata(p1)
    r2 = rankdata(p2)
    r3 = rankdata(p3)
    r4 = rankdata(p4)
    r5 = rankdata(p5)
    r6 = rankdata(p6)
    r_blend = (
        weights[0] * r1 +
        weights[1] * r2 +
        weights[2] * r3 +
        weights[3] * r4 +
        weights[4] * r5 +
        weights[5] * r6
    )

    order = np.argsort(np.argsort(r_blend))
    p_final = np.sort(p_raw)[order]

    return p_final


# EVOLVE-BLOCK-END
# =====================================================================
