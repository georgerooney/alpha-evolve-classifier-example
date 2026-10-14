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

    # 1. Comorbidities
    if _icd9_comorbidities is not None and "diag_1" in tr and "diag_2" in tr and "diag_3" in tr:
        try:
            cm_tr = _icd9_comorbidities(tr["diag_1"], tr["diag_2"], tr["diag_3"])
            cm_ev = _icd9_comorbidities(ev["diag_1"], ev["diag_2"], ev["diag_3"])
            for cm_name, cm_arr in cm_tr.items():
                tr[cm_name] = cm_arr
                ev[cm_name] = cm_ev[cm_name]
                extra_numeric.append(cm_name)
        except Exception:
            pass

    # 2. ICD-9 Chapter Groups
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

    # 3. Out-Of-Fold Target Encodings
    if _oof_target_encode is not None:
        te_candidates = [
            "discharge_disposition_id",
            "admission_source_id",
            "admission_type_id",
            "medical_specialty",
            "diag_1",
            "diag_2",
            "diag_3",
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

    # 4. Latent Clinical Phenotyping via Low-Rank Matrix/Tensor Decomposition
    med_names = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin",
        "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]

    diag_counter = Counter()
    for d_col in ["diag_1", "diag_2", "diag_3"]:
        if d_col in tr:
            for c in tr[d_col]:
                if c is not None and c != "?" and isinstance(c, str):
                    diag_counter[c.split(".")[0]] += 1
    top_diag_prefixes = [p for p, cnt in diag_counter.most_common(50) if cnt >= 50]

    def _build_clinical_indicators(cols, n_rows):
        indicators = []
        for cm_name in [
            "cm_mi", "cm_chf", "cm_pvd", "cm_cerebrovascular", "cm_dementia",
            "cm_pulmonary", "cm_rheumatic", "cm_peptic_ulcer", "cm_liver",
            "cm_hemiplegia", "cm_renal", "cm_cancer", "cm_metastatic",
            "cm_hiv", "cm_diabetes_complicated",
        ]:
            if cm_name in cols:
                indicators.append(cols[cm_name].astype(np.float32))

        for m in med_names:
            if m in cols:
                m_arr = cols[m]
                act = np.fromiter((1.0 if v is not None and v != "No" else 0.0 for v in m_arr), dtype=np.float32, count=n_rows)
                chg = np.fromiter((1.0 if v in ("Up", "Down") else 0.0 for v in m_arr), dtype=np.float32, count=n_rows)
                indicators.append(act)
                indicators.append(chg)

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
                indicators.append(has_p)

        if indicators:
            return np.column_stack(indicators)
        return np.zeros((n_rows, 1), dtype=np.float32)

    X_clin_tr = _build_clinical_indicators(tr, n_tr)
    X_clin_ev = _build_clinical_indicators(ev, n_ev)

    n_latent = min(8, X_clin_tr.shape[1] - 1)
    if n_latent >= 2:
        try:
            svd = TruncatedSVD(n_components=n_latent, algorithm="randomized", random_state=42)
            Z_tr = svd.fit_transform(X_clin_tr)
            Z_ev = svd.transform(X_clin_ev)
            for j in range(n_latent):
                feat_name = f"latent_phenotype_{j}"
                tr[feat_name] = Z_tr[:, j].astype(np.float64)
                ev[feat_name] = Z_ev[:, j].astype(np.float64)
                extra_numeric.append(feat_name)

            # Low-rank reconstruction anomaly score
            X_rec_tr = np.dot(Z_tr, svd.components_)
            X_rec_ev = np.dot(Z_ev, svd.components_)
            tr["clinical_anomaly_score"] = np.sqrt(np.mean((X_clin_tr - X_rec_tr) ** 2, axis=1)).astype(np.float64)
            ev["clinical_anomaly_score"] = np.sqrt(np.mean((X_clin_ev - X_rec_ev) ** 2, axis=1)).astype(np.float64)
            extra_numeric.append("clinical_anomaly_score")
        except Exception:
            pass

    # 5. Utilization Intensity & Clinical Derived Features
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
        d["has_prior_emergency"] = (n_emg > 0).astype(np.float64)
        d["inpatient_ratio"] = n_inp / (tot_prior + 1.0)
        d["emergency_ratio"] = n_emg / (tot_prior + 1.0)

        d["service_intensity"] = (n_meds + n_labs + n_prc) / (t_hosp + 1.0)
        d["labs_per_day"] = n_labs / (t_hosp + 1.0)
        d["meds_per_day"] = n_meds / (t_hosp + 1.0)
        d["procs_per_day"] = n_prc / (t_hosp + 1.0)
        d["meds_per_diag"] = n_meds / (n_diag + 1.0)
        d["labs_per_diag"] = n_labs / (n_diag + 1.0)

        if "age" in d:
            d["age_numeric"] = np.array([age_map.get(a, np.nan) for a in d["age"]], dtype=np.float64)
        else:
            d["age_numeric"] = np.full(n_rows, np.nan, dtype=np.float64)

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

        ins = d.get("insulin", np.array([None] * n_rows))
        d["insulin_active"] = np.fromiter((1.0 if v is not None and v != "No" else 0.0 for v in ins), dtype=np.float64, count=n_rows)
        d["insulin_up"] = np.fromiter((1.0 if v == "Up" else 0.0 for v in ins), dtype=np.float64, count=n_rows)
        d["insulin_down"] = np.fromiter((1.0 if v == "Down" else 0.0 for v in ins), dtype=np.float64, count=n_rows)

        disp = d.get("discharge_disposition_id", np.array([None] * n_rows))
        d["discharged_home"] = np.fromiter((1.0 if str(v) == "1" else 0.0 for v in disp), dtype=np.float64, count=n_rows)

        a1c = d.get("A1Cresult", np.array([None] * n_rows))
        d["a1c_measured"] = np.fromiter((0.0 if v in ("None", None) else 1.0 for v in a1c), dtype=np.float64, count=n_rows)
        d["a1c_high"] = np.fromiter((1.0 if v in (">8", ">7") else 0.0 for v in a1c), dtype=np.float64, count=n_rows)

    extra_numeric.extend([
        "total_prior_visits", "has_prior_inpatient", "has_prior_emergency",
        "inpatient_ratio", "emergency_ratio", "service_intensity",
        "labs_per_day", "meds_per_day", "procs_per_day", "meds_per_diag",
        "labs_per_diag", "age_numeric", "num_active_meds", "num_med_changes",
        "num_meds_up", "num_meds_down", "insulin_active", "insulin_up",
        "insulin_down", "discharged_home", "a1c_measured", "a1c_high",
    ])

    extra_numeric = list(dict.fromkeys(extra_numeric))
    extra_categorical = list(dict.fromkeys(extra_categorical))

    # 6. Tabular Encoding
    encoder = _TabularEncoder(min_count=40)
    encoder.fit(task, tr, extra_numeric=extra_numeric, extra_categorical=extra_categorical)
    x_train = encoder.transform(tr)
    x_eval = encoder.transform(ev)
    cat_indices = list(encoder.categorical_indices)

    # 7. Multi-Scale LightGBM Optimization Ensemble
    model_balanced = lgb.LGBMClassifier(
        n_estimators=280,
        learning_rate=0.022,
        num_leaves=31,
        max_depth=6,
        min_child_samples=160,
        subsample=0.72,
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
        n_estimators=320,
        learning_rate=0.018,
        num_leaves=16,
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
        n_estimators=240,
        learning_rate=0.026,
        num_leaves=38,
        max_depth=7,
        min_child_samples=120,
        subsample=0.65,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_lambda=0.5,
        reg_alpha=0.05,
        cat_smooth=30.0,
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

    preds = 0.40 * p1 + 0.35 * p2 + 0.25 * p3
    preds = np.clip(preds, 0.0, 1.0)
    preds = np.nan_to_num(preds, nan=float(np.mean(y_train)))
    return preds


# EVOLVE-BLOCK-END
# =====================================================================
