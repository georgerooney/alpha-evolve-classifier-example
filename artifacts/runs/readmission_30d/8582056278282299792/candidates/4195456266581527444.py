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
from typing import Dict, Sequence, Any, List, Tuple
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

try:
    from clinical_toolkit import (
        ClinicalFeatureBuilder,
        CareAllocationState,
        InterventionTier,
        RiskBlender,
    )
except ImportError:
    pass


def _extract_legacy_discrepancy_features(records: Sequence[Any]) -> np.ndarray:
    """Extract legacy scores, non-linear submodel dissonance, and acuity interactions."""
    feats: List[List[float]] = []
    for p in records:
        base_p = float(getattr(p, "baseline_api_prob", 0.5))
        base_p = float(np.clip(base_p, 1e-4, 1.0 - 1e-4))
        logit_base = float(np.log(base_p / (1.0 - base_p)))

        subscores = getattr(p, "baseline_submodel_scores", {}) or {}
        claims = float(subscores.get("claims_linear", base_p))
        ehr = float(subscores.get("ehr_vitals", base_p))
        util = float(subscores.get("utilization_rules", base_p))

        acuity = float(getattr(p, "acuity_score", 0.0))
        contra_cnt = float(len(getattr(p, "contraindications", [])))

        # Cross-organ / acute-on-chronic discrepancy signals
        d_ehr_claims = ehr - claims
        d_util_claims = util - claims
        d_ehr_util = ehr - util
        sub_mean = (claims + ehr + util) / 3.0
        sub_std = float(np.std([claims, ehr, util]))
        sub_max = max(claims, ehr, util)
        sub_min = min(claims, ehr, util)

        feats.append([
            base_p,
            logit_base,
            claims,
            ehr,
            util,
            acuity,
            contra_cnt,
            d_ehr_claims,
            d_util_claims,
            d_ehr_util,
            sub_mean,
            sub_std,
            sub_max,
            sub_min,
            acuity * base_p,
            acuity * ehr,
            acuity * util,
        ])
    return np.asarray(feats, dtype=np.float32)


def _decompose_and_augment_features(
    x_tr: np.ndarray,
    x_ev: np.ndarray,
    train_records: Sequence[Any],
    eval_records: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Low-rank tensor decomposition: extracts latent syndromic modes and 2nd/3rd-order synergies."""
    leg_tr = _extract_legacy_discrepancy_features(train_records)
    leg_ev = _extract_legacy_discrepancy_features(eval_records)

    x_tr_comb = np.hstack([x_tr, leg_tr])
    x_ev_comb = np.hstack([x_ev, leg_ev])
    x_tr_comb = np.nan_to_num(x_tr_comb, nan=0.0, posinf=0.0, neginf=0.0)
    x_ev_comb = np.nan_to_num(x_ev_comb, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    z_tr = scaler.fit_transform(x_tr_comb)
    z_ev = scaler.transform(x_ev_comb)
    z_tr = np.nan_to_num(z_tr, nan=0.0, posinf=0.0, neginf=0.0)
    z_ev = np.nan_to_num(z_ev, nan=0.0, posinf=0.0, neginf=0.0)

    # Latent orthogonal syndromic modes (K=8 principal axes)
    k_components = min(8, z_tr.shape[1], max(1, z_tr.shape[0] - 1))
    svd = TruncatedSVD(n_components=k_components, random_state=42)
    svd.fit(z_tr)
    v = svd.components_.T  # Shape: (D, K)

    # 1st-order latent syndrome score F = Z @ V
    f_tr = z_tr @ v
    f_ev = z_ev @ v

    # 2nd-order symmetric CP tensor bilinear interaction energy
    # P_{i, k} = 0.5 * [(Z V)_{i,k}^2 - (Z^2 V^2)_{i,k}]
    z2_tr = z_tr ** 2
    z2_ev = z_ev ** 2
    v2 = v ** 2
    z2_v2_tr = z2_tr @ v2
    z2_v2_ev = z2_ev @ v2
    p_tr = 0.5 * (f_tr ** 2 - z2_v2_tr)
    p_ev = 0.5 * (f_ev ** 2 - z2_v2_ev)

    # 3rd-order symmetric tensor trilinear multi-organ collapse energy
    # T_{i, k} = (1/6) * [F^3 - 3*F*(Z^2 V^2) + 2*(Z^3 V^3)]
    z3_tr = z_tr ** 3
    z3_ev = z_ev ** 3
    v3 = v ** 3
    z3_v3_tr = z3_tr @ v3
    z3_v3_ev = z3_ev @ v3
    t_tr = (1.0 / 6.0) * (f_tr ** 3 - 3.0 * f_tr * z2_v2_tr + 2.0 * z3_v3_tr)
    t_ev = (1.0 / 6.0) * (f_ev ** 3 - 3.0 * f_ev * z2_v2_ev + 2.0 * z3_v3_ev)

    # Numerical outlier damping on higher-order tensor powers
    p_tr = np.clip(p_tr, -50.0, 50.0)
    p_ev = np.clip(p_ev, -50.0, 50.0)
    t_tr = np.clip(t_tr, -100.0, 100.0)
    t_ev = np.clip(t_ev, -100.0, 100.0)

    # Non-linear physiological saturation and deviation magnitude
    s_tr = np.tanh(f_tr)
    s_ev = np.tanh(f_ev)
    m_tr = np.abs(f_tr)
    m_ev = np.abs(f_ev)

    aug_tr = np.hstack([x_tr_comb, f_tr, p_tr, t_tr, s_tr, m_tr])
    aug_ev = np.hstack([x_ev_comb, f_ev, p_ev, t_ev, s_ev, m_ev])
    aug_tr = np.nan_to_num(aug_tr, nan=0.0, posinf=0.0, neginf=0.0)
    aug_ev = np.nan_to_num(aug_ev, nan=0.0, posinf=0.0, neginf=0.0)
    return aug_tr, aug_ev


def fit_and_score_risk(
    cohort_spec: Any,
    train_records: Sequence[Any],
    eval_records: Sequence[Any],
) -> Dict[str, float]:
    """Stage 1: Low-rank tensor decomposition, cross-validated ensemble, and Platt calibration."""
    try:
        builder = ClinicalFeatureBuilder(include_interactions=True)
        x_train, y_train = builder.fit_transform(cohort_spec, train_records)
        x_eval = builder.transform(cohort_spec, eval_records)

        x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
        x_eval = np.nan_to_num(x_eval, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.asarray(y_train, dtype=np.int32)

        if len(np.unique(y_train)) < 2:
            return {p.patient_id: float(getattr(p, "baseline_api_prob", 0.5)) for p in eval_records}

        # 1. Feature augmentation via low-rank tensor decomposition
        x_train_aug, x_eval_aug = _decompose_and_augment_features(
            x_train, x_eval, train_records, eval_records
        )

        # 2. RiskBlender evaluation on augmented representations
        blender = RiskBlender(
            linear_weight=0.35,
            lgbm_weight=0.40,
            xgb_weight=0.25,
            baseline_prior_weight=0.06,
            random_state=42,
        )
        preds_blender_dict = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train_aug,
            y_train=y_train,
            x_eval=x_eval_aug,
        )
        blender_eval_probs = np.array([
            float(preds_blender_dict.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5)))
            for p in eval_records
        ], dtype=np.float64)

        # 3. Stratified 5-Fold CV Ensemble (LGBM, XGBoost, Elastic Logistic Regression)
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        n_splits = min(5, n_pos, n_neg)

        train_base_probs = np.array([
            float(getattr(p, "baseline_api_prob", 0.5)) for p in train_records
        ], dtype=np.float64)
        eval_base_probs = np.array([
            float(getattr(p, "baseline_api_prob", 0.5)) for p in eval_records
        ], dtype=np.float64)

        if n_splits >= 3:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            n_tr = len(y_train)
            n_ev = len(eval_records)

            oof_lgb = np.zeros(n_tr, dtype=np.float64)
            oof_xgb = np.zeros(n_tr, dtype=np.float64)
            oof_lr = np.zeros(n_tr, dtype=np.float64)

            ev_lgb = np.zeros(n_ev, dtype=np.float64)
            ev_xgb = np.zeros(n_ev, dtype=np.float64)
            ev_lr = np.zeros(n_ev, dtype=np.float64)

            for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train_aug, y_train)):
                xt_f, yt_f = x_train_aug[tr_idx], y_train[tr_idx]
                xv_f = x_train_aug[val_idx]

                # Model A: LightGBM
                model_lgb = lgb.LGBMClassifier(
                    objective="binary",
                    n_estimators=130,
                    learning_rate=0.035,
                    num_leaves=15,
                    max_depth=4,
                    min_child_samples=15,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42 + fold,
                    n_jobs=2,
                    verbose=-1,
                )
                model_lgb.fit(xt_f, yt_f)
                oof_lgb[val_idx] = model_lgb.predict_proba(xv_f)[:, 1]
                ev_lgb += model_lgb.predict_proba(x_eval_aug)[:, 1] / n_splits

                # Model B: XGBoost
                model_xgb = xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=120,
                    learning_rate=0.035,
                    max_depth=3,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42 + fold,
                    n_jobs=2,
                    verbosity=0,
                )
                model_xgb.fit(xt_f, yt_f)
                oof_xgb[val_idx] = model_xgb.predict_proba(xv_f)[:, 1]
                ev_xgb += model_xgb.predict_proba(x_eval_aug)[:, 1] / n_splits

                # Model C: Regularized Logistic Regression on standardized latent features
                scaler_f = StandardScaler()
                xt_sc = scaler_f.fit_transform(xt_f)
                xv_sc = scaler_f.transform(xv_f)
                xev_sc = scaler_f.transform(x_eval_aug)

                model_lr = LogisticRegression(
                    C=0.20,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=400,
                    random_state=42 + fold,
                )
                model_lr.fit(xt_sc, yt_f)
                oof_lr[val_idx] = model_lr.predict_proba(xv_sc)[:, 1]
                ev_lr += model_lr.predict_proba(xev_sc)[:, 1] / n_splits

            # 4. Non-Negative Constrained Quadratic Programming on OOF predictions
            p_oof_mat = np.column_stack([oof_lgb, oof_xgb, oof_lr, train_base_probs])
            p_ev_mat = np.column_stack([ev_lgb, ev_xgb, ev_lr, eval_base_probs])
            n_models = p_oof_mat.shape[1]

            def brier_objective(w: np.ndarray) -> float:
                pred = p_oof_mat @ w
                return float(np.mean((pred - y_train) ** 2))

            w0 = np.ones(n_models) / n_models
            bnds = [(0.0, 1.0) for _ in range(n_models)]
            cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
            opt_res = minimize(brier_objective, w0, method="SLSQP", bounds=bnds, constraints=cons)

            opt_weights = opt_res.x if opt_res.success else w0
            opt_weights = np.clip(opt_weights, 0.0, 1.0)
            opt_weights = opt_weights / (np.sum(opt_weights) + 1e-8)

            custom_oof = p_oof_mat @ opt_weights
            custom_eval = p_ev_mat @ opt_weights

            # Ensemble RiskBlender and OOF-optimized models
            combined_eval = 0.50 * blender_eval_probs + 0.50 * custom_eval

            # 5. Strictly Monotonic Platt Calibration to Minimize Brier Score
            logit_oof = np.log(np.clip(custom_oof, 1e-4, 1.0 - 1e-4) / (1.0 - np.clip(custom_oof, 1e-4, 1.0 - 1e-4)))
            logit_eval = np.log(np.clip(combined_eval, 1e-4, 1.0 - 1e-4) / (1.0 - np.clip(combined_eval, 1e-4, 1.0 - 1e-4)))

            calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            calibrator.fit(logit_oof.reshape(-1, 1), y_train)

            # Monotonicity check ensures exact rank invariance (zero loss on AUC & Sens@Spec80)
            if calibrator.coef_[0, 0] > 0.01:
                calibrated_eval = calibrator.predict_proba(logit_eval.reshape(-1, 1))[:, 1]
                final_eval_probs = 0.60 * calibrated_eval + 0.40 * combined_eval
            else:
                final_eval_probs = combined_eval
        else:
            final_eval_probs = blender_eval_probs

        result: Dict[str, float] = {}
        for i, p in enumerate(eval_records):
            val = float(final_eval_probs[i])
            if np.isnan(val) or np.isinf(val):
                val = float(getattr(p, "baseline_api_prob", 0.5))
            result[p.patient_id] = float(np.clip(val, 0.001, 0.999))
        return result
    except Exception:
        return {
            p.patient_id: float(np.clip(getattr(p, "baseline_api_prob", 0.5), 0.001, 0.999))
            for p in eval_records
        }


def allocate_interventions(
    cohort_spec: Any,
    eval_records: Sequence[Any],
    predicted_risks: Dict[str, float],
) -> Any:
    """Stage 2: Optimal knapsack allocation with alpha-sweep and upgrade hill-climbing."""
    ref_state = CareAllocationState(cohort_spec, eval_records)

    # 1. Deduce tier costs and slot capacities robustly
    candidate_tiers = [
        tier for tier in cohort_spec.interventions.keys()
        if tier != InterventionTier.NONE
    ]
    tier_costs: Dict[Any, float] = {}
    tier_slots: Dict[Any, int] = {}

    for tier in candidate_tiers:
        spec = cohort_spec.interventions[tier]
        tier_slots[tier] = int(spec.max_slots)
        cost = getattr(spec, "cost_usd", getattr(spec, "cost", None))
        if cost is None:
            for p in eval_records:
                if ref_state.fits(p.patient_id, tier):
                    cost = -ref_state.expected_net_benefit(p.patient_id, 0.0, tier)
                    break
        tier_costs[tier] = max(1.0, float(cost if cost is not None else 0.0))

    total_budget = float(cohort_spec.total_budget_usd)

    # 2. Build candidate patient-tier options with clinical eligibility verified
    patient_options: Dict[str, Dict[Any, Tuple[float, float]]] = {}
    all_pairs = []

    for p in eval_records:
        pid = p.patient_id
        risk = float(predicted_risks.get(pid, getattr(p, "baseline_api_prob", 0.5)))
        patient_options[pid] = {}

        for tier in candidate_tiers:
            if ref_state.fits(pid, tier):
                enb = ref_state.expected_net_benefit(pid, risk, tier)
                if enb > 0.0:
                    cost = tier_costs[tier]
                    patient_options[pid][tier] = (enb, cost)
                    all_pairs.append((pid, tier, enb, cost))

    # 3. Multi-pass optimization over efficiency-yield tradeoff exponents alpha in [0.0, 1.0]
    best_assignment: Dict[str, Any] = {}
    best_total_benefit = -1.0

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for alpha in alphas:
        # Rank by marginal benefit weighted by cost^alpha
        ranked_pairs = sorted(
            all_pairs,
            key=lambda x: (x[2] / (x[3] ** alpha + 1e-6), x[2]),
            reverse=True,
        )

        curr_assigned: Dict[str, Any] = {}
        slots_used = {t: 0 for t in candidate_tiers}
        budget_used = 0.0

        for pid, tier, enb, cost in ranked_pairs:
            if pid not in curr_assigned:
                if (slots_used[tier] + 1 <= tier_slots[tier]) and (budget_used + cost <= total_budget):
                    curr_assigned[pid] = tier
                    slots_used[tier] += 1
                    budget_used += cost

        # Upgrade Hill-Climbing: upgrade assigned patients to higher-benefit tiers if budget and slots permit
        for pid in list(curr_assigned.keys()):
            curr_tier = curr_assigned[pid]
            curr_enb, curr_cost = patient_options[pid][curr_tier]

            upgrade_candidates = [
                (t, patient_options[pid][t][0], patient_options[pid][t][1])
                for t in candidate_tiers
                if t in patient_options[pid] and t != curr_tier and patient_options[pid][t][0] > curr_enb
            ]
            upgrade_candidates.sort(key=lambda x: (x[1] - curr_enb) / max(1.0, x[2] - curr_cost), reverse=True)

            for cand_tier, cand_enb, cand_cost in upgrade_candidates:
                delta_cost = cand_cost - curr_cost
                if (slots_used[cand_tier] + 1 <= tier_slots[cand_tier]) and (budget_used + delta_cost <= total_budget):
                    slots_used[curr_tier] -= 1
                    slots_used[cand_tier] += 1
                    budget_used += delta_cost
                    curr_assigned[pid] = cand_tier
                    curr_tier = cand_tier
                    curr_enb, curr_cost = cand_enb, cand_cost
                    break

        total_enb = sum(patient_options[pid][tier][0] for pid, tier in curr_assigned.items())
        if total_enb > best_total_benefit:
            best_total_benefit = total_enb
            best_assignment = curr_assigned

    # 4. Commit optimal plan into CareAllocationState to guarantee 0 violations
    final_state = CareAllocationState(cohort_spec, eval_records)
    sorted_assignments = sorted(
        best_assignment.items(),
        key=lambda item: patient_options[item[0]][item[1]][0],
        reverse=True,
    )

    for pid, tier in sorted_assignments:
        if final_state.fits(pid, tier):
            final_state.assign(pid, tier)

    plan = final_state.to_plan()
    for p in eval_records:
        if p.patient_id not in plan.assignments:
            plan.assignments[p.patient_id] = InterventionTier.NONE

    return plan


# EVOLVE-BLOCK-END
# =====================================================================
