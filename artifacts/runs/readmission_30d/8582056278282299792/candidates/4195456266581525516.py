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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb

try:
    from clinical_toolkit import ClinicalFeatureBuilder, CareAllocationState, InterventionTier
except ImportError:
    pass


def _extract_patient_features(records: Sequence[Any]) -> np.ndarray:
    """Extract baseline risk scores, model discordance signals, and clinical acuity."""
    feats = []
    for p in records:
        b_prob = float(getattr(p, "baseline_api_prob", 0.5))
        b_prob_clip = np.clip(b_prob, 1e-5, 1.0 - 1e-5)
        b_logit = float(np.log(b_prob_clip / (1.0 - b_prob_clip)))

        sub = getattr(p, "baseline_submodel_scores", {}) or {}
        claims = float(sub.get("claims_linear", b_prob))
        vitals = float(sub.get("ehr_vitals", b_prob))
        util = float(sub.get("utilization_rules", b_prob))

        acuity = float(getattr(p, "acuity_score", 0.0))
        n_contra = float(len(getattr(p, "contraindications", []) or []))

        # Cross-modality discrepancies: discordance between claims and acute vitals
        claims_vs_vitals = claims - vitals
        claims_vs_util = claims - util
        vitals_vs_util = vitals - util

        # Cross-modality interactions
        claims_x_vitals = claims * vitals
        sub_std = float(np.std([claims, vitals, util]))
        sub_max = max(claims, vitals, util)
        sub_min = min(claims, vitals, util)

        feats.append([
            b_prob,
            b_logit,
            claims,
            vitals,
            util,
            acuity,
            n_contra,
            claims_vs_vitals,
            claims_vs_util,
            vitals_vs_util,
            claims_x_vitals,
            sub_std,
            sub_max,
            sub_min,
            acuity * b_prob,
            acuity * vitals,
        ])
    return np.array(feats, dtype=np.float32)


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Use ClinicalFeatureBuilder with interactions and RiskBlender for optimal predictions."""
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)

    try:
        # 1. Base clinical syndrome feature extraction with interactions
        builder = ClinicalFeatureBuilder(include_interactions=True)
        x_train_raw, y_train_raw = builder.fit_transform(cohort_spec, train_records)
        x_eval_raw = builder.transform(cohort_spec, eval_records)

        y_train = np.asarray(y_train_raw, dtype=np.int32)
        n_samples = len(y_train)

        # Fallback for degenerate training cohorts
        if len(np.unique(y_train)) < 2:
            return {p.patient_id: float(getattr(p, "baseline_api_prob", 0.5)) for p in eval_records}

        x_train_clean = np.nan_to_num(x_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
        x_eval_clean = np.nan_to_num(x_eval_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Extract baseline model priors, discrepancies, and acuity features
        p_train_feats = _extract_patient_features(train_records)
        p_eval_feats = _extract_patient_features(eval_records)

        X_tr_dense = np.hstack([x_train_clean, p_train_feats])
        X_ev_dense = np.hstack([x_eval_clean, p_eval_feats])

        # 3. Low-rank factor decomposition (Truncated SVD) to extract latent comorbidity modes
        scaler = StandardScaler()
        X_tr_std = scaler.fit_transform(X_tr_dense)
        X_ev_std = scaler.transform(X_ev_dense)

        n_svd = min(8, max(2, X_tr_std.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_svd, random_state=42)
        Z_tr = svd.fit_transform(X_tr_std)
        Z_ev = svd.transform(X_ev_std)

        # Bilinear tensor interaction contractions between top 4 latent clinical modes
        m_modes = min(4, n_svd)
        int_tr_list, int_ev_list = [], []
        for i in range(m_modes):
            for j in range(i + 1, m_modes):
                int_tr_list.append(Z_tr[:, i] * Z_tr[:, j])
                int_ev_list.append(Z_ev[:, i] * Z_ev[:, j])

        Z_int_tr = np.column_stack(int_tr_list) if int_tr_list else np.empty((n_samples, 0))
        Z_int_ev = np.column_stack(int_ev_list) if int_ev_list else np.empty((len(eval_records), 0))

        X_final_tr = np.hstack([X_tr_dense, Z_tr, Z_int_tr])
        X_final_ev = np.hstack([X_ev_dense, Z_ev, Z_int_ev])

        base_prior_tr = p_train_feats[:, 0]
        base_prior_ev = p_eval_feats[:, 0]

        # 4. Out-of-fold stratified cross-validation ensembling
        n_splits = 5 if n_samples >= 50 else max(2, n_samples // 10)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof_lgb1 = np.zeros(n_samples, dtype=np.float64)
        oof_lgb2 = np.zeros(n_samples, dtype=np.float64)
        oof_xgb = np.zeros(n_samples, dtype=np.float64)
        oof_lin = np.zeros(n_samples, dtype=np.float64)

        pred_lgb1 = np.zeros(len(eval_records), dtype=np.float64)
        pred_lgb2 = np.zeros(len(eval_records), dtype=np.float64)
        pred_xgb = np.zeros(len(eval_records), dtype=np.float64)
        pred_lin = np.zeros(len(eval_records), dtype=np.float64)

        for train_idx, val_idx in skf.split(X_final_tr, y_train):
            X_fold_tr, y_fold_tr = X_final_tr[train_idx], y_train[train_idx]
            X_fold_val = X_final_tr[val_idx]

            # Model 1: Leaf-wise LightGBM
            clf_lgb1 = lgb.LGBMClassifier(
                n_estimators=140,
                learning_rate=0.035,
                max_depth=4,
                num_leaves=15,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
            clf_lgb1.fit(X_fold_tr, y_fold_tr)
            oof_lgb1[val_idx] = clf_lgb1.predict_proba(X_fold_val)[:, 1]
            pred_lgb1 += clf_lgb1.predict_proba(X_final_ev)[:, 1] / n_splits

            # Model 2: Conservative regularized LightGBM
            clf_lgb2 = lgb.LGBMClassifier(
                n_estimators=110,
                learning_rate=0.04,
                max_depth=3,
                num_leaves=8,
                min_child_samples=15,
                subsample=0.85,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=1.0,
                random_state=123,
                verbose=-1,
                n_jobs=-1,
            )
            clf_lgb2.fit(X_fold_tr, y_fold_tr)
            oof_lgb2[val_idx] = clf_lgb2.predict_proba(X_fold_val)[:, 1]
            pred_lgb2 += clf_lgb2.predict_proba(X_final_ev)[:, 1] / n_splits

            # Model 3: XGBoost with second-order tree splitting
            clf_xgb = xgb.XGBClassifier(
                n_estimators=120,
                learning_rate=0.035,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_alpha=0.2,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
                n_jobs=-1,
            )
            clf_xgb.fit(X_fold_tr, y_fold_tr)
            oof_xgb[val_idx] = clf_xgb.predict_proba(X_fold_val)[:, 1]
            pred_xgb += clf_xgb.predict_proba(X_final_ev)[:, 1] / n_splits

            # Model 4: Calibrated L2 Logistic Regression on standardized latent features
            fold_scaler = StandardScaler()
            X_fold_tr_s = fold_scaler.fit_transform(X_fold_tr)
            X_fold_val_s = fold_scaler.transform(X_fold_val)
            X_ev_s = fold_scaler.transform(X_final_ev)

            log_reg = LogisticRegression(C=0.2, penalty="l2", solver="lbfgs", max_iter=400, random_state=42)
            cal_reg = CalibratedClassifierCV(estimator=log_reg, method="sigmoid", cv=3)
            cal_reg.fit(X_fold_tr_s, y_fold_tr)
            oof_lin[val_idx] = cal_reg.predict_proba(X_fold_val_s)[:, 1]
            pred_lin += cal_reg.predict_proba(X_ev_s)[:, 1] / n_splits

        # 5. Brier Score Minimization on Probability Simplex via SLSQP
        oof_matrix = np.column_stack([oof_lgb1, oof_lgb2, oof_xgb, oof_lin, base_prior_tr])
        eval_matrix = np.column_stack([pred_lgb1, pred_lgb2, pred_xgb, pred_lin, base_prior_ev])

        def brier_objective(w: np.ndarray) -> float:
            p_blend = oof_matrix @ w
            return float(np.mean((p_blend - y_train) ** 2))

        init_weights = np.array([0.28, 0.22, 0.25, 0.15, 0.10], dtype=np.float64)
        init_weights /= np.sum(init_weights)

        res = minimize(
            brier_objective,
            x0=init_weights,
            bounds=[(0.01, 0.55)] * 5,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            method="SLSQP",
        )
        opt_weights = res.x if res.success else init_weights

        final_probs = eval_matrix @ opt_weights
        final_probs = np.clip(final_probs, 0.001, 0.999)

        return {p.patient_id: float(final_probs[i]) for i, p in enumerate(eval_records)}

    except Exception:
        # Resilient fallback to legacy API probabilities in any unanticipated edge case
        return {p.patient_id: float(getattr(p, "baseline_api_prob", 0.5)) for p in eval_records}


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Optimal multi-choice knapsack allocation with alpha-sweep and upgrade hill-climbing."""
    ref_state = CareAllocationState(cohort_spec, eval_records)

    # 1. Deduce tier costs and slot capacities exactly from state and spec
    dummy_pid = eval_records[0].patient_id if eval_records else None
    tier_costs: Dict[Any, float] = {}
    tier_slots: Dict[Any, int] = {}
    candidate_tiers = []

    for tier, spec in cohort_spec.interventions.items():
        if tier != InterventionTier.NONE:
            c = max(1e-5, -ref_state.expected_net_benefit(dummy_pid, 0.0, tier)) if dummy_pid else 0.0
            tier_costs[tier] = c
            tier_slots[tier] = spec.max_slots
            candidate_tiers.append(tier)

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

            for cand_tier in candidate_tiers:
                if cand_tier in patient_options[pid] and cand_tier != curr_tier:
                    cand_enb, cand_cost = patient_options[pid][cand_tier]
                    delta_enb = cand_enb - curr_enb
                    delta_cost = cand_cost - curr_cost

                    if delta_enb > 0.0:
                        has_slot = (slots_used[cand_tier] + 1 <= tier_slots[cand_tier])
                        has_budget = (budget_used + delta_cost <= total_budget)
                        if has_slot and has_budget:
                            slots_used[curr_tier] -= 1
                            slots_used[cand_tier] += 1
                            budget_used += delta_cost
                            curr_assigned[pid] = cand_tier
                            curr_tier = cand_tier
                            curr_enb, curr_cost = cand_enb, cand_cost

        # Residual Fill: assign remaining unassigned patients to any remaining feasible slots
        for p in eval_records:
            pid = p.patient_id
            if pid not in curr_assigned:
                eligible_tiers = sorted(
                    [t for t in patient_options[pid].keys()],
                    key=lambda t: patient_options[pid][t][0],
                    reverse=True,
                )
                for cand_tier in eligible_tiers:
                    cand_enb, cand_cost = patient_options[pid][cand_tier]
                    if (slots_used[cand_tier] + 1 <= tier_slots[cand_tier]) and (budget_used + cand_cost <= total_budget):
                        curr_assigned[pid] = cand_tier
                        slots_used[cand_tier] += 1
                        budget_used += cand_cost
                        break

        total_enb = sum(patient_options[pid][tier][0] for pid, tier in curr_assigned.items())
        if total_enb > best_total_benefit:
            best_total_benefit = total_enb
            best_assignment = curr_assigned

    # 4. Atomically commit the optimal plan into fresh CareAllocationState to guarantee 0 violations
    final_state = CareAllocationState(cohort_spec, eval_records)
    sorted_assignments = sorted(
        best_assignment.items(),
        key=lambda item: patient_options[item[0]][item[1]][0],
        reverse=True,
    )

    for pid, tier in sorted_assignments:
        if final_state.fits(pid, tier):
            final_state.assign(pid, tier)

    return final_state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
