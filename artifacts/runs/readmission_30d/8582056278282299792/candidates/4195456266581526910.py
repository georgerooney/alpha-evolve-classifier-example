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

"""Initial baseline seed program for clinical Health Clinical Risk & Care Allocation.

Contains the two-stage strategy evolved by AlphaEvolve:
1. `fit_and_score_risk`: Refines patient event probabilities over legacy baseline API scores + EHR/claims features.
2. `allocate_interventions`: Assigns care-management interventions under slot, budget, and clinical eligibility rules.
"""

from __future__ import annotations

from typing import Dict, Sequence

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from clinical_models import (
    CareAllocationPlan,
    CohortSpec,
    InterventionTier,
    PatientRecord,
)
from clinical_toolkit import (
    CareAllocationState,
    ClinicalFeatureBuilder,
    safe_logit,
    safe_sigmoid,
)


# =====================================================================
# EVOLVE-BLOCK-START
# Strategy only. Domain mechanics and feasibility checks live in `clinical_toolkit` (`ClinicalFeatureBuilder`,
# `RiskBlender`, `CareAllocationState`, `safe_logit`, `safe_sigmoid`). Pre-imported libraries available in sandbox:
# `numpy`, `scipy`, `sklearn`, `lightgbm`, `xgboost`.


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Fit multi-modal gradient boosting and tensor-decomposed models with convex optimization blending."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    import lightgbm as lgb
    import xgboost as xgb

    def to_logit(p_arr: np.ndarray) -> np.ndarray:
        p_clipped = np.clip(p_arr, 1e-5, 1.0 - 1e-5)
        return np.log(p_clipped / (1.0 - p_clipped))

    def extract_submodel_features(records: Sequence[PatientRecord]) -> np.ndarray:
        feats = []
        for r in records:
            api = float(getattr(r, "baseline_api_prob", 0.5) or 0.5)
            api = max(1e-4, min(1.0 - 1e-4, api))
            sub = getattr(r, "baseline_submodel_scores", {}) or {}
            claims = float(sub.get("claims_linear", api) or api)
            claims = max(1e-4, min(1.0 - 1e-4, claims))
            vitals = float(sub.get("ehr_vitals", api) or api)
            vitals = max(1e-4, min(1.0 - 1e-4, vitals))
            util = float(sub.get("utilization_rules", api) or api)
            util = max(1e-4, min(1.0 - 1e-4, util))
            acuity = float(getattr(r, "acuity_score", 0.5) or 0.5)

            l_api = float(np.log(api / (1.0 - api)))
            l_claims = float(np.log(claims / (1.0 - claims)))
            l_vitals = float(np.log(vitals / (1.0 - vitals)))
            l_util = float(np.log(util / (1.0 - util)))

            feats.append([
                api, l_api,
                claims, l_claims,
                vitals, l_vitals,
                util, l_util,
                acuity,
            ])
        return np.array(feats, dtype=np.float64)

    # 1. Feature extraction with non-linear clinical syndrome interactions
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train_raw = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)

    x_train = np.nan_to_num(x_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    x_eval = np.nan_to_num(x_eval_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.array(y_train_raw, dtype=np.int32)

    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        out: Dict[str, float] = {}
        for p in eval_records:
            out[p.patient_id] = round(float(getattr(p, "baseline_api_prob", 0.5)), 6)
        return out

    # 2. Latent Clinical Factor & Tensor Decomposition (Truncated SVD)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_eval_scaled = scaler.transform(x_eval)

    n_samples, n_features = x_train_scaled.shape
    k_factors = min(8, n_features, max(1, n_samples - 1))

    mean_vec = np.mean(x_train_scaled, axis=0)
    xc_train = x_train_scaled - mean_vec
    xc_eval = x_eval_scaled - mean_vec

    try:
        u, s, vt = np.linalg.svd(xc_train, full_matrices=False)
        vk = vt[:k_factors, :].T
        z_train = xc_train @ vk
        z_eval = xc_eval @ vk
        rec_train = z_train @ vk.T
        anom_train = np.linalg.norm(xc_train - rec_train, axis=1, keepdims=True)
        rec_eval = z_eval @ vk.T
        anom_eval = np.linalg.norm(xc_eval - rec_eval, axis=1, keepdims=True)
    except Exception:
        z_train = np.zeros((n_samples, k_factors), dtype=np.float64)
        z_eval = np.zeros((len(eval_records), k_factors), dtype=np.float64)
        anom_train = np.zeros((n_samples, 1), dtype=np.float64)
        anom_eval = np.zeros((len(eval_records), 1), dtype=np.float64)

    # 3. Bilinear cross-domain tensor interaction terms
    sub_train = extract_submodel_features(train_records)
    sub_eval = extract_submodel_features(eval_records)

    legacy_logits_tr = sub_train[:, [1, 3, 5, 7]]
    legacy_logits_ev = sub_eval[:, [1, 3, 5, 7]]

    k_tensor = min(4, k_factors)
    tensor_tr_list = []
    tensor_ev_list = []
    for ki in range(k_tensor):
        for mi in range(4):
            tensor_tr_list.append(z_train[:, ki] * legacy_logits_tr[:, mi])
            tensor_ev_list.append(z_eval[:, ki] * legacy_logits_ev[:, mi])

    tensor_feats_tr = np.column_stack(tensor_tr_list) if tensor_tr_list else np.zeros((n_samples, 0))
    tensor_feats_ev = np.column_stack(tensor_ev_list) if tensor_ev_list else np.zeros((len(eval_records), 0))

    X_train_full = np.hstack([x_train_scaled, z_train, anom_train, tensor_feats_tr, sub_train])
    X_eval_full = np.hstack([x_eval_scaled, z_eval, anom_eval, tensor_feats_ev, sub_eval])

    # 4. Out-of-fold Cross-Validation & Stacking Ensemble
    min_class_count = min(np.sum(y_train == 0), np.sum(y_train == 1))
    n_splits = max(2, min(5, min_class_count))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lr = np.zeros(n_samples, dtype=np.float64)
    oof_lgb = np.zeros(n_samples, dtype=np.float64)
    oof_xgb = np.zeros(n_samples, dtype=np.float64)

    eval_lr = np.zeros(len(eval_records), dtype=np.float64)
    eval_lgb = np.zeros(len(eval_records), dtype=np.float64)
    eval_xgb = np.zeros(len(eval_records), dtype=np.float64)

    scaler_full = StandardScaler()
    X_tr_full_norm = scaler_full.fit_transform(X_train_full)
    X_ev_full_norm = scaler_full.transform(X_eval_full)

    for tr_idx, val_idx in skf.split(X_train_full, y_train):
        X_f_tr, y_f_tr = X_tr_full_norm[tr_idx], y_train[tr_idx]
        X_f_val = X_tr_full_norm[val_idx]

        # Model A: Regularized Logistic Regression
        lr = LogisticRegression(C=0.25, max_iter=300, random_state=42)
        lr.fit(X_f_tr, y_f_tr)
        oof_lr[val_idx] = lr.predict_proba(X_f_val)[:, 1]
        eval_lr += lr.predict_proba(X_ev_full_norm)[:, 1] / n_splits

        # Model B: LightGBM Classifier
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=130,
            learning_rate=0.035,
            max_depth=4,
            num_leaves=15,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.80,
            min_child_samples=15,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        lgb_clf.fit(X_train_full[tr_idx], y_f_tr)
        oof_lgb[val_idx] = lgb_clf.predict_proba(X_train_full[val_idx])[:, 1]
        eval_lgb += lgb_clf.predict_proba(X_eval_full)[:, 1] / n_splits

        # Model C: XGBoost Classifier
        xgb_clf = xgb.XGBClassifier(
            n_estimators=110,
            learning_rate=0.035,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.2,
            reg_lambda=1.5,
            random_state=42,
            eval_metric="logloss",
        )
        xgb_clf.fit(X_train_full[tr_idx], y_f_tr)
        oof_xgb[val_idx] = xgb_clf.predict_proba(X_train_full[val_idx])[:, 1]
        eval_xgb += xgb_clf.predict_proba(X_eval_full)[:, 1] / n_splits

    # 5. Meta-learner: Convex optimization on out-of-fold log-odds
    M_train = np.column_stack([
        to_logit(oof_lr),
        to_logit(oof_lgb),
        to_logit(oof_xgb),
        sub_train[:, 1],  # logit(api)
        sub_train[:, 3],  # logit(claims)
        sub_train[:, 5],  # logit(vitals)
        sub_train[:, 7],  # logit(util)
    ])

    M_eval = np.column_stack([
        to_logit(eval_lr),
        to_logit(eval_lgb),
        to_logit(eval_xgb),
        sub_eval[:, 1],   # logit(api)
        sub_eval[:, 3],   # logit(claims)
        sub_eval[:, 5],   # logit(vitals)
        sub_eval[:, 7],   # logit(util)
    ])

    meta_clf = LogisticRegression(C=0.5, penalty="l2", max_iter=300, random_state=42)
    meta_clf.fit(M_train, y_train)
    meta_eval_probs = meta_clf.predict_proba(M_eval)[:, 1]

    # 6. Ensemble blending with RiskBlender if available
    rb_probs_eval = None
    try:
        blender = RiskBlender(
            linear_weight=0.35,
            lgbm_weight=0.35,
            xgb_weight=0.25,
            baseline_prior_weight=0.05,
            random_state=42,
        )
        rb_dict = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
        )
        if rb_dict and len(rb_dict) == len(eval_records):
            rb_probs_eval = np.array([rb_dict[p.patient_id] for p in eval_records])
    except Exception:
        rb_probs_eval = None

    if rb_probs_eval is not None:
        blended_logits = 0.70 * to_logit(meta_eval_probs) + 0.30 * to_logit(rb_probs_eval)
        z_clipped = np.clip(blended_logits, -30.0, 30.0)
        final_eval_probs = 1.0 / (1.0 + np.exp(-z_clipped))
    else:
        final_eval_probs = meta_eval_probs

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        val = float(final_eval_probs[idx])
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, val)), 6)
    return out





def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Optimal Knapsack allocation maximizing net expected savings under budget & slot constraints."""
    state = CareAllocationState(cohort_spec, eval_records)
    candidate_tiers = [
        t for t in cohort_spec.interventions.keys()
        if t != InterventionTier.NONE
    ]

    candidates = []
    for p in eval_records:
        r = float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5)))
        for tier in candidate_tiers:
            spec = cohort_spec.interventions[tier]
            if getattr(p, "acuity_score", 0.0) < spec.min_acuity_score:
                continue
            if set(getattr(p, "contraindications", [])) & set(spec.contraindicated_flags):
                continue
            net_ben = state.expected_net_benefit(p.patient_id, r, tier)
            if net_ben > 0.0:
                cost = float(spec.cost_usd)
                efficiency = net_ben / max(50.0, cost)
                score = net_ben * (efficiency ** 0.25)
                candidates.append((score, p.patient_id, tier))

    candidates.sort(key=lambda x: -x[0])

    assigned = set()
    for score, pid, tier in candidates:
        if pid in assigned:
            continue
        if state.fits(pid, tier):
            if state.assign(pid, tier):
                assigned.add(pid)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
