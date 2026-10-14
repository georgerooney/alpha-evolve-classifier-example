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
import numpy as np
from typing import Dict, Sequence, List, Tuple, Optional, Any

from clinical_toolkit import (
    ClinicalFeatureBuilder,
    RiskBlender,
    CareAllocationState,
)


def _safe_logit(p: float, eps: float = 1e-4) -> float:
    """Numerically stable logit transformation."""
    p_clamped = max(eps, min(1.0 - eps, float(p)))
    return float(np.log(p_clamped / (1.0 - p_clamped)))


def _extract_clinical_meta_features(records: Sequence[Any]) -> np.ndarray:
    """Extract clinical divergence, cross-submodel dispersion, and syndrome indicators."""
    rows = []
    for p in records:
        b_prob = float(getattr(p, "baseline_api_prob", 0.5))
        b_prob = max(1e-4, min(1.0 - 1e-4, b_prob))
        b_logit = _safe_logit(b_prob)

        sub = getattr(p, "baseline_submodel_scores", None) or {}
        claims = float(sub.get("claims_linear", b_prob))
        ehr = float(sub.get("ehr_vitals", b_prob))
        util = float(sub.get("utilization_rules", b_prob))

        claims = max(1e-4, min(1.0 - 1e-4, claims))
        ehr = max(1e-4, min(1.0 - 1e-4, ehr))
        util = max(1e-4, min(1.0 - 1e-4, util))

        claims_logit = _safe_logit(claims)
        ehr_logit = _safe_logit(ehr)
        util_logit = _safe_logit(util)

        acuity = float(getattr(p, "acuity_score", 0.0))

        # Divergence: acute vitals vs chronic claims baseline
        acute_vs_claims = ehr - claims
        acute_vs_claims_logit = ehr_logit - claims_logit

        # Frequent utilization vs chronic claims baseline
        util_vs_claims = util - claims
        util_vs_claims_logit = util_logit - claims_logit

        # Acute vitals vs utilization
        ehr_vs_util = ehr - util
        ehr_vs_util_logit = ehr_logit - util_logit

        # Acuity interactions
        acuity_x_bprob = acuity * b_prob
        acuity_x_ehr = acuity * ehr
        acuity_x_util = acuity * util
        acuity_x_diff = acuity * acute_vs_claims

        # Cross-submodel dispersion
        sub_vals = [claims, ehr, util]
        sub_max = max(sub_vals)
        sub_min = min(sub_vals)
        sub_spread = sub_max - sub_min
        sub_mean = float(np.mean(sub_vals))
        sub_std = float(np.std(sub_vals))
        sub_geom = float((claims * ehr * util) ** (1.0 / 3.0))

        contra = getattr(p, "contraindications", ())
        n_contra = float(len(contra)) if contra else 0.0

        # Elderly elective overprediction correction indicator:
        # High claims baseline but low acute vitals and low acuity
        elderly_elective_proxy = max(0.0, claims - ehr) / (acuity + 0.5)

        # Additional non-linear interactions
        acuity_sq = acuity * acuity
        contra_acuity = n_contra * acuity
        util_x_ehr = util * ehr
        claims_x_ehr = claims * ehr
        acuity_vs_bprob = acuity / (b_prob + 0.1)

        rows.append([
            b_prob, b_logit,
            claims, claims_logit,
            ehr, ehr_logit,
            util, util_logit,
            acuity, acuity_sq,
            acute_vs_claims, acute_vs_claims_logit,
            util_vs_claims, util_vs_claims_logit,
            ehr_vs_util, ehr_vs_util_logit,
            acuity_x_bprob, acuity_x_ehr, acuity_x_util, acuity_x_diff,
            sub_max, sub_min, sub_spread, sub_mean, sub_std, sub_geom,
            n_contra, contra_acuity, elderly_elective_proxy,
            util_x_ehr, claims_x_ehr, acuity_vs_bprob
        ])
    return np.array(rows, dtype=np.float32)


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Low-Rank Tensor Augmented Multi-Model Ensemble with Optimal Convex Stacking."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.decomposition import TruncatedSVD
    from sklearn.model_selection import StratifiedKFold, KFold
    from sklearn.preprocessing import StandardScaler
    from scipy.optimize import minimize

    # 1. Feature extraction: clinical feature builder + domain meta-features
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train_raw = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)

    meta_train = _extract_clinical_meta_features(train_records)
    meta_eval = _extract_clinical_meta_features(eval_records)

    x_train_init = np.nan_to_num(np.hstack([x_train_raw, meta_train]), nan=0.0, posinf=1.0, neginf=-1.0)
    x_eval_init = np.nan_to_num(np.hstack([x_eval_raw, meta_eval]), nan=0.0, posinf=1.0, neginf=-1.0)
    y_train = np.array(y_train_raw, dtype=np.float32)

    # 2. Low-Rank Tensor / Matrix Decomposition (TruncatedSVD + Tucker bilinear projection)
    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train_init)
    x_eval_sc = scaler.transform(x_eval_init)

    n_svd_comp = min(6, x_train_sc.shape[1] - 1, max(2, len(train_records) - 1))
    svd = TruncatedSVD(n_components=n_svd_comp, random_state=42)
    latent_train = svd.fit_transform(x_train_sc)
    latent_eval = svd.transform(x_eval_sc)

    # Atypicality / reconstruction residual error:
    recon_train = svd.inverse_transform(latent_train)
    recon_eval = svd.inverse_transform(latent_eval)
    resid_train = np.linalg.norm(x_train_sc - recon_train, axis=1, keepdims=True)
    resid_eval = np.linalg.norm(x_eval_sc - recon_eval, axis=1, keepdims=True)

    # Submodel scores for bilinear tensor interaction
    def _get_sub_triplet(recs: Sequence[Any]) -> np.ndarray:
        arr = []
        for p in recs:
            b = float(getattr(p, "baseline_api_prob", 0.5))
            sub = getattr(p, "baseline_submodel_scores", None) or {}
            c = float(sub.get("claims_linear", b))
            e = float(sub.get("ehr_vitals", b))
            u = float(sub.get("utilization_rules", b))
            arr.append([c, e, u])
        return np.array(arr, dtype=np.float32)

    sub_tr = _get_sub_triplet(train_records)
    sub_ev = _get_sub_triplet(eval_records)

    # Bilinear tensor outer product: top 2 latent syndromic coordinates x 3 submodels
    k_lat = min(2, latent_train.shape[1])
    bilinear_train = np.zeros((len(train_records), k_lat * 3), dtype=np.float32)
    bilinear_eval = np.zeros((len(eval_records), k_lat * 3), dtype=np.float32)
    for i in range(k_lat):
        for j in range(3):
            col_idx = i * 3 + j
            bilinear_train[:, col_idx] = latent_train[:, i] * sub_tr[:, j]
            bilinear_eval[:, col_idx] = latent_eval[:, i] * sub_ev[:, j]

    # Combined feature representation
    x_train = np.nan_to_num(
        np.hstack([x_train_sc, latent_train, resid_train, bilinear_train]),
        nan=0.0, posinf=1.0, neginf=-1.0
    )
    x_eval = np.nan_to_num(
        np.hstack([x_eval_sc, latent_eval, resid_eval, bilinear_eval]),
        nan=0.0, posinf=1.0, neginf=-1.0
    )

    # Sample weights emphasizing acute readmission events
    sample_weights = np.ones(len(train_records), dtype=np.float32)
    for i, p in enumerate(train_records):
        y_val = float(getattr(p, "outcome_label", 0.0) or 0.0)
        t_val = getattr(p, "time_to_event_days", None)
        if y_val > 0.5:
            if t_val is not None:
                try:
                    t_f = float(t_val)
                    t_f = max(1.0, min(30.0, t_f))
                except Exception:
                    t_f = 15.0
            else:
                t_f = 15.0
            sample_weights[i] = 1.0 + 0.3 * (30.0 - t_f) / 30.0

    # 3. K-Fold Cross Validation Setup
    n_pos = int(np.sum(y_train == 1))
    min_class = min(n_pos, len(y_train) - n_pos)
    if min_class >= 5:
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds = list(skf.split(x_train, y_train))
    elif min_class >= 2:
        n_splits = min_class
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds = list(skf.split(x_train, y_train))
    else:
        n_splits = 2
        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        folds = list(kf.split(x_train))

    oof_lgb = np.zeros(len(y_train), dtype=np.float64)
    oof_xgb = np.zeros(len(y_train), dtype=np.float64)
    oof_et = np.zeros(len(y_train), dtype=np.float64)
    oof_lr = np.zeros(len(y_train), dtype=np.float64)

    eval_lgb = np.zeros(len(eval_records), dtype=np.float64)
    eval_xgb = np.zeros(len(eval_records), dtype=np.float64)
    eval_et = np.zeros(len(eval_records), dtype=np.float64)
    eval_lr = np.zeros(len(eval_records), dtype=np.float64)

    for fold, (tr_idx, val_idx) in enumerate(folds):
        x_tr_f, y_tr_f = x_train[tr_idx], y_train[tr_idx]
        x_va_f = x_train[val_idx]
        sw_tr_f = sample_weights[tr_idx]

        # Model 1: LightGBM
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.04,
            num_leaves=16,
            max_depth=4,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=1.0,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=1,
        )
        clf_lgb.fit(x_tr_f, y_tr_f, sample_weight=sw_tr_f)
        oof_lgb[val_idx] = clf_lgb.predict_proba(x_va_f)[:, 1]
        eval_lgb += clf_lgb.predict_proba(x_eval)[:, 1] / float(n_splits)

        # Model 2: XGBoost
        clf_xgb = xgb.XGBClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42 + fold,
            n_jobs=1,
        )
        clf_xgb.fit(x_tr_f, y_tr_f, sample_weight=sw_tr_f)
        oof_xgb[val_idx] = clf_xgb.predict_proba(x_va_f)[:, 1]
        eval_xgb += clf_xgb.predict_proba(x_eval)[:, 1] / float(n_splits)

        # Model 3: ExtraTrees (orthogonal variance reduction)
        clf_et = ExtraTreesClassifier(
            n_estimators=120,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42 + fold,
            n_jobs=1,
        )
        clf_et.fit(x_tr_f, y_tr_f, sample_weight=sw_tr_f)
        oof_et[val_idx] = clf_et.predict_proba(x_va_f)[:, 1]
        eval_et += clf_et.predict_proba(x_eval)[:, 1] / float(n_splits)

        # Model 4: Regularized Logistic Regression
        clf_lr = LogisticRegression(
            C=0.15,
            max_iter=500,
            penalty="l2",
            solver="lbfgs",
            random_state=42 + fold,
        )
        clf_lr.fit(x_tr_f, y_tr_f)
        oof_lr[val_idx] = clf_lr.predict_proba(x_va_f)[:, 1]
        eval_lr += clf_lr.predict_proba(x_eval)[:, 1] / float(n_splits)

    # Legacy baseline predictions
    base_tr = np.array([float(getattr(p, "baseline_api_prob", 0.5)) for p in train_records], dtype=np.float64)
    base_ev = np.array([float(getattr(p, "baseline_api_prob", 0.5)) for p in eval_records], dtype=np.float64)

    # 4. Optimal Simplex-Constrained Ensemble Optimization
    p_oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_et, oof_lr, base_tr])
    p_oof_matrix = np.clip(p_oof_matrix, 1e-4, 1.0 - 1e-4)

    p_eval_matrix = np.column_stack([eval_lgb, eval_xgb, eval_et, eval_lr, base_ev])
    p_eval_matrix = np.clip(p_eval_matrix, 1e-4, 1.0 - 1e-4)

    w_prior = np.array([0.30, 0.30, 0.18, 0.14, 0.08], dtype=np.float64)
    eps = 1e-6

    def _loss_fn(w: np.ndarray) -> float:
        p_blend = np.clip(np.dot(p_oof_matrix, w), eps, 1.0 - eps)
        # Log-loss + prior regularization
        bce = -np.mean(y_train * np.log(p_blend) + (1.0 - y_train) * np.log(1.0 - p_blend))
        reg = 0.05 * np.sum((w - w_prior) ** 2)
        return float(bce + reg)

    def _grad_fn(w: np.ndarray) -> np.ndarray:
        p_blend = np.clip(np.dot(p_oof_matrix, w), eps, 1.0 - eps)
        err = (p_blend - y_train) / (p_blend * (1.0 - p_blend))
        grad = np.dot(p_oof_matrix.T, err) / float(len(y_train))
        grad += 0.10 * (w - w_prior)
        return grad

    opt_res = minimize(
        _loss_fn,
        w_prior,
        jac=_grad_fn,
        bounds=[(0.01, 0.60)] * 5,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        method="SLSQP",
        options={"maxiter": 100, "ftol": 1e-7},
    )

    if opt_res.success and np.all(opt_res.x >= 0.0):
        w_opt = opt_res.x / np.sum(opt_res.x)
    else:
        w_opt = w_prior / np.sum(w_prior)

    pred_oof_blend = np.clip(np.dot(p_oof_matrix, w_opt), 1e-4, 1.0 - 1e-4)
    pred_eval_blend = np.clip(np.dot(p_eval_matrix, w_opt), 1e-4, 1.0 - 1e-4)

    z_oof = np.log(pred_oof_blend / (1.0 - pred_oof_blend))
    z_eval = np.log(pred_eval_blend / (1.0 - pred_eval_blend))

    # 5. Strictly Monotonic Platt Calibration
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    calibrator.fit(z_oof.reshape(-1, 1), y_train)

    if calibrator.coef_[0][0] > 0.01:
        eval_probs = calibrator.predict_proba(z_eval.reshape(-1, 1))[:, 1]
    else:
        y_mean = float(np.mean(y_train))
        delta = np.log(max(1e-4, y_mean) / max(1e-4, 1.0 - y_mean)) - float(np.mean(z_oof))
        eval_probs = 1.0 / (1.0 + np.exp(-(z_eval + delta)))

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        prob = float(eval_probs[idx])
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, prob)), 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Maximize expected net clinical benefit under budget and slot constraints."""
    state = CareAllocationState(cohort_spec, eval_records)
    active_tiers = [
        t for t in cohort_spec.interventions.keys()
        if getattr(t, "name", str(t)).upper() != "NONE" and str(t) != "NONE"
    ]

    # Identify promising intervention-patient pairs
    # We create a list of all feasible (patient, tier, nb, cost) options
    options = []
    for p in eval_records:
        risk = float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5)))
        for tier in active_tiers:
            if state.fits(p.patient_id, tier):
                nb = state.expected_net_benefit(p.patient_id, risk, tier)
                if nb > 0.0:
                    cost = float(getattr(cohort_spec.interventions[tier], "cost_usd", 1000.0))
                    options.append({
                        "patient_id": p.patient_id,
                        "tier": tier,
                        "nb": nb,
                        "cost": cost,
                        "eff": nb / (cost + 1e-3),
                        "risk": risk
                    })

    # Sort options globally by a composite score of efficiency and absolute NB
    # This prioritizes high ROI options but also ensures high absolute value is captured.
    options.sort(key=lambda x: (-(x["nb"] * (1.0 + 0.5 * x["eff"])), -x["risk"]))

    assigned_patients = set()
    for opt in options:
        pid = opt["patient_id"]
        if pid in assigned_patients:
            continue
        tier = opt["tier"]
        
        # Check if it still fits within budget and slots
        if state.fits(pid, tier):
            if state.assign(pid, tier):
                assigned_patients.add(pid)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
