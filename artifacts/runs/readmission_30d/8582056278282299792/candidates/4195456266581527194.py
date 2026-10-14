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
from typing import Sequence, Dict, Tuple, List, Optional, Any
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb


def safe_logit(p: float, eps: float = 1e-5) -> float:
    p_clamped = min(max(float(p), eps), 1.0 - eps)
    return float(np.log(p_clamped / (1.0 - p_clamped)))


def safe_sigmoid(x: float) -> float:
    x_clamped = min(max(float(x), -20.0), 20.0)
    return float(1.0 / (1.0 + np.exp(-x_clamped)))


def _extract_patient_domain_features(records: Sequence[PatientRecord]) -> np.ndarray:
    """Extract legacy submodel signals, acuity, discordance, and interaction features."""
    feats = []
    for p in records:
        base_p = float(np.clip(p.baseline_api_prob, 1e-4, 1.0 - 1e-4))
        base_logit = safe_logit(base_p)
        sub = p.baseline_submodel_scores or {}
        c_lin = float(np.clip(sub.get("claims_linear", base_p), 1e-4, 1.0 - 1e-4))
        e_vit = float(np.clip(sub.get("ehr_vitals", base_p), 1e-4, 1.0 - 1e-4))
        u_rul = float(np.clip(sub.get("utilization_rules", base_p), 1e-4, 1.0 - 1e-4))
        c_logit = safe_logit(c_lin)
        e_logit = safe_logit(e_vit)
        u_logit = safe_logit(u_rul)
        acuity = float(p.acuity_score)
        n_contra = float(len(p.contraindications))

        # Cross-channel acute vs historical discordance
        diff_ehr_claims = e_vit - c_lin
        diff_ehr_base = e_vit - base_p
        diff_util_claims = u_rul - c_lin
        diff_util_ehr = u_rul - e_vit

        sub_vals = [c_lin, e_vit, u_rul]
        range_sub = max(sub_vals) - min(sub_vals)
        mean_sub = sum(sub_vals) / 3.0
        std_sub = float(np.std(sub_vals))

        acuity_x_base = acuity * base_p
        acuity_x_evit = acuity * e_vit
        acuity_x_diff = acuity * diff_ehr_claims

        feats.append([
            base_p,
            base_logit,
            c_lin,
            c_logit,
            e_vit,
            e_logit,
            u_rul,
            u_logit,
            acuity,
            n_contra,
            diff_ehr_claims,
            diff_ehr_base,
            diff_util_claims,
            diff_util_ehr,
            range_sub,
            mean_sub,
            std_sub,
            acuity_x_base,
            acuity_x_evit,
            acuity_x_diff,
        ])
    return np.array(feats, dtype=np.float64)


def _tensor_decompose_and_augment(
    x_tr_scaled: np.ndarray,
    x_ev_scaled: np.ndarray,
    dom_tr: np.ndarray,
    dom_ev: np.ndarray,
    k_components: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Latent syndromic SVD decomposition and multi-channel augmentation."""
    n_tr, d = x_tr_scaled.shape
    k = max(2, min(k_components, d, n_tr - 2))

    _, _, vt = np.linalg.svd(x_tr_scaled, full_matrices=False)
    v_k = vt[:k, :].T

    z_tr = x_tr_scaled @ v_k
    z_ev = x_ev_scaled @ v_k

    # Pairwise syndromic interaction terms
    t_tr_list, t_ev_list = [], []
    for i in range(k):
        for j in range(i, k):
            t_tr_list.append((z_tr[:, i] * z_tr[:, j])[:, None])
            t_ev_list.append((z_ev[:, i] * z_ev[:, j])[:, None])
    t_tr = np.hstack(t_tr_list)
    t_ev = np.hstack(t_ev_list)

    # Cross-modal coupling with legacy submodels
    k_sub = min(3, k)
    coupling_tr, coupling_ev = [], []
    for i in range(k_sub):
        for m in (2, 4, 6):  # c_lin, e_vit, u_rul in dom_tr
            coupling_tr.append((z_tr[:, i] * dom_tr[:, m])[:, None])
            coupling_ev.append((z_ev[:, i] * dom_ev[:, m])[:, None])
    cross_tr = np.hstack(coupling_tr)
    cross_ev = np.hstack(coupling_ev)

    aug_tr = np.hstack([x_tr_scaled, z_tr, t_tr, dom_tr, cross_tr])
    aug_ev = np.hstack([x_ev_scaled, z_ev, t_ev, dom_ev, cross_ev])
    return aug_tr, aug_ev


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Multi-model ensemble with low-rank tensor decomposition and SLSQP probability calibration."""
    # Build clinical features with clinically validated non-linear interactions
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_tr_raw, y_train = builder.fit_transform(cohort_spec, train_records)
    x_ev_raw = builder.transform(cohort_spec, eval_records)

    x_tr_clean = np.nan_to_num(x_tr_raw, nan=0.0, posinf=0.0, neginf=0.0)
    x_ev_clean = np.nan_to_num(x_ev_raw, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    x_tr_scaled = np.clip(scaler.fit_transform(x_tr_clean), -10.0, 10.0)
    x_ev_scaled = np.clip(scaler.transform(x_ev_clean), -10.0, 10.0)

    dom_tr = _extract_patient_domain_features(train_records)
    dom_ev = _extract_patient_domain_features(eval_records)

    aug_tr, aug_ev = _tensor_decompose_and_augment(x_tr_scaled, x_ev_scaled, dom_tr, dom_ev, k_components=4)

    meta_scaler = StandardScaler()
    aug_tr_scaled = np.clip(meta_scaler.fit_transform(aug_tr), -10.0, 10.0)
    aug_ev_scaled = np.clip(meta_scaler.transform(aug_ev), -10.0, 10.0)

    n_train = len(train_records)
    n_eval = len(eval_records)
    n_splits = 5

    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]
    rng = np.random.RandomState(42)
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    pos_folds = np.array_split(pos_indices, n_splits)
    neg_folds = np.array_split(neg_indices, n_splits)

    oof_lgb = np.zeros(n_train, dtype=np.float64)
    oof_xgb = np.zeros(n_train, dtype=np.float64)
    oof_hgb = np.zeros(n_train, dtype=np.float64)
    oof_lr = np.zeros(n_train, dtype=np.float64)
    oof_et = np.zeros(n_train, dtype=np.float64)

    eval_preds_lgb = np.zeros(n_eval, dtype=np.float64)
    eval_preds_xgb = np.zeros(n_eval, dtype=np.float64)
    eval_preds_hgb = np.zeros(n_eval, dtype=np.float64)
    eval_preds_lr = np.zeros(n_eval, dtype=np.float64)
    eval_preds_et = np.zeros(n_eval, dtype=np.float64)

    for fold in range(n_splits):
        val_idx = np.concatenate([pos_folds[fold], neg_folds[fold]])
        tr_mask = np.ones(n_train, dtype=bool)
        tr_mask[val_idx] = False

        x_fold_tr, y_fold_tr = aug_tr[tr_mask], y_train[tr_mask]
        x_fold_val = aug_tr[val_idx]

        x_fold_tr_s = aug_tr_scaled[tr_mask]
        x_fold_val_s = aug_tr_scaled[val_idx]

        # 1. LightGBM
        model_lgb = lgb.LGBMClassifier(
            n_estimators=160,
            learning_rate=0.035,
            num_leaves=16,
            max_depth=4,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=42 + fold,
            verbose=-1,
        )
        model_lgb.fit(x_fold_tr, y_fold_tr)
        oof_lgb[val_idx] = model_lgb.predict_proba(x_fold_val)[:, 1]
        eval_preds_lgb += model_lgb.predict_proba(aug_ev)[:, 1] / float(n_splits)

        # 2. XGBoost
        model_xgb = xgb.XGBClassifier(
            n_estimators=140,
            learning_rate=0.035,
            max_depth=3,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.15,
            reg_lambda=1.5,
            eval_metric="logloss",
            random_state=42 + fold,
        )
        model_xgb.fit(x_fold_tr, y_fold_tr)
        oof_xgb[val_idx] = model_xgb.predict_proba(x_fold_val)[:, 1]
        eval_preds_xgb += model_xgb.predict_proba(aug_ev)[:, 1] / float(n_splits)

        # 3. HistGradientBoosting
        model_hgb = HistGradientBoostingClassifier(
            max_iter=130,
            learning_rate=0.035,
            max_leaf_nodes=15,
            max_depth=3,
            min_samples_leaf=20,
            l2_regularization=1.5,
            random_state=42 + fold,
        )
        model_hgb.fit(x_fold_tr, y_fold_tr)
        oof_hgb[val_idx] = model_hgb.predict_proba(x_fold_val)[:, 1]
        eval_preds_hgb += model_hgb.predict_proba(aug_ev)[:, 1] / float(n_splits)

        # 4. Regularized Logistic Regression
        model_lr = LogisticRegression(C=0.2, max_iter=300, random_state=42 + fold)
        model_lr.fit(x_fold_tr_s, y_fold_tr)
        oof_lr[val_idx] = model_lr.predict_proba(x_fold_val_s)[:, 1]
        eval_preds_lr += model_lr.predict_proba(aug_ev_scaled)[:, 1] / float(n_splits)

        # 5. ExtraTrees
        model_et = ExtraTreesClassifier(
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=15,
            max_features="sqrt",
            random_state=42 + fold,
        )
        model_et.fit(x_fold_tr, y_fold_tr)
        oof_et[val_idx] = model_et.predict_proba(x_fold_val)[:, 1]
        eval_preds_et += model_et.predict_proba(aug_ev)[:, 1] / float(n_splits)

    # Legacy baseline API probabilities
    oof_base = np.array([float(np.clip(p.baseline_api_prob, 1e-4, 1.0 - 1e-4)) for p in train_records])
    eval_base = np.array([float(np.clip(p.baseline_api_prob, 1e-4, 1.0 - 1e-4)) for p in eval_records])

    # Convert to logit space for linear ensemble blending
    z_oof = np.column_stack([
        [safe_logit(v) for v in np.clip(oof_lgb, 1e-4, 1.0 - 1e-4)],
        [safe_logit(v) for v in np.clip(oof_xgb, 1e-4, 1.0 - 1e-4)],
        [safe_logit(v) for v in np.clip(oof_hgb, 1e-4, 1.0 - 1e-4)],
        [safe_logit(v) for v in np.clip(oof_lr, 1e-4, 1.0 - 1e-4)],
        [safe_logit(v) for v in np.clip(oof_et, 1e-4, 1.0 - 1e-4)],
        [safe_logit(v) for v in np.clip(oof_base, 1e-4, 1.0 - 1e-4)],
    ])

    z_eval = np.column_stack([
        [safe_logit(v) for v in np.clip(eval_preds_lgb, 1e-4, 1.0 - 1e-4)],
        [safe_logit(v) for v in np.clip(eval_preds_xgb, 1e-4, 1.0 - 1e-4)],
        [safe_logit(v) for v in np.clip(eval_preds_hgb, 1e-4, 1.0 - 1e-4)],
        [safe_logit(v) for v in np.clip(eval_preds_lr, 1e-4, 1.0 - 1e-4)],
        [safe_logit(v) for v in np.clip(eval_preds_et, 1e-4, 1.0 - 1e-4)],
        [safe_logit(v) for v in np.clip(eval_base, 1e-4, 1.0 - 1e-4)],
    ])

    # SLSQP optimization of ensemble weights to minimize OOF Brier score
    def brier_objective(weights: np.ndarray) -> float:
        w = weights / np.sum(weights)
        combined_logits = z_oof @ w
        p = np.array([safe_sigmoid(val) for val in combined_logits])
        return float(np.mean((p - y_train) ** 2))

    w_init = np.array([0.25, 0.25, 0.20, 0.10, 0.10, 0.10])
    bounds = [(0.02, 0.50)] * 6
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    try:
        res = minimize(brier_objective, w_init, method="SLSQP", bounds=bounds, constraints=constraints)
        opt_weights = res.x / np.sum(res.x) if res.success else w_init
    except Exception:
        opt_weights = w_init

    oof_blend_logits = z_oof @ opt_weights
    eval_blend_logits = z_eval @ opt_weights

    # Monotonic Platt calibration to minimize Brier score while preserving exact AUC ranking
    calibrator = LogisticRegression(C=1e3, max_iter=200)
    calibrator.fit(oof_blend_logits.reshape(-1, 1), y_train)

    if calibrator.coef_[0, 0] > 0.0:
        eval_probs = calibrator.predict_proba(eval_blend_logits.reshape(-1, 1))[:, 1]
    else:
        eval_probs = np.array([safe_sigmoid(val) for val in eval_blend_logits])

    # Safely anchor with RiskBlender if available
    try:
        from clinical_toolkit import RiskBlender  # type: ignore
        blender = RiskBlender(
            linear_weight=0.35,
            lgbm_weight=0.35,
            xgb_weight=0.20,
            baseline_prior_weight=0.10,
            random_state=42,
        )
        rb_dict = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_tr_clean,
            y_train=y_train,
            x_eval=x_ev_clean,
        )
        rb_probs = np.array([rb_dict.get(p.patient_id, eval_probs[i]) for i, p in enumerate(eval_records)])
        final_logits = 0.82 * np.array([safe_logit(v) for v in np.clip(eval_probs, 1e-4, 1.0 - 1e-4)]) + \
                       0.18 * np.array([safe_logit(v) for v in np.clip(rb_probs, 1e-4, 1.0 - 1e-4)])
        final_probs = np.array([safe_sigmoid(val) for val in final_logits])
    except Exception:
        final_probs = eval_probs

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        out[p.patient_id] = round(float(np.clip(final_probs[idx], 1e-4, 1.0 - 1e-4)), 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Optimal care allocation maximizing expected net benefit subject to budget and clinical constraints."""
    state = CareAllocationState(cohort_spec, eval_records)

    active_tiers = [
        t
        for t, s in cohort_spec.interventions.items()
        if t != InterventionTier.NONE and s.max_slots > 0 and s.relative_risk_reduction > 0.0
    ]
    if not active_tiers:
        return state.to_plan()

    # Patient priority: maximum positive net benefit achievable across all feasible active tiers
    def patient_priority(p: PatientRecord) -> float:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        max_b = 0.0
        for tier in active_tiers:
            if state.fits(p.patient_id, tier):
                b = state.expected_net_benefit(p.patient_id, risk, tier)
                if b > max_b:
                    max_b = b
        return max_b

    ranked_patients = sorted(eval_records, key=lambda p: -patient_priority(p))

    for p in ranked_patients:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        candidate_tiers = []
        for tier in active_tiers:
            if state.fits(p.patient_id, tier):
                b = state.expected_net_benefit(p.patient_id, risk, tier)
                if b > 0.0:
                    candidate_tiers.append((tier, b))

        # Assign tier with maximum expected net benefit
        candidate_tiers.sort(key=lambda x: -x[1])
        for tier, _ in candidate_tiers:
            if state.fits(p.patient_id, tier):
                if state.assign(p.patient_id, tier):
                    break

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
