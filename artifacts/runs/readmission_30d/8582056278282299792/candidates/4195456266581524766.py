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
    """Stage 1: Multi-model 5-fold CV stacking with syndrome residual features and monotonic Platt calibration."""
    import numpy as np
    from scipy.optimize import minimize
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
    import lightgbm as lgb
    import xgboost as xgb

    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train_raw = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)

    y_train = np.asarray(y_train_raw, dtype=int)
    if x_train_raw.ndim == 1:
        x_train_raw = x_train_raw.reshape(-1, 1)
    if x_eval_raw.ndim == 1:
        x_eval_raw = x_eval_raw.reshape(-1, 1)

    # Extract domain residual features capturing legacy submodel discrepancies and clinical acuity
    def extract_extra_features(records: Sequence[PatientRecord]) -> np.ndarray:
        rows = []
        for p in records:
            base_p = float(getattr(p, "baseline_api_prob", 0.5))
            base_p = max(1e-4, min(1.0 - 1e-4, base_p))
            base_l = safe_logit(base_p)

            sub = getattr(p, "baseline_submodel_scores", None) or {}
            c_lin = max(1e-4, min(1.0 - 1e-4, float(sub.get("claims_linear", base_p))))
            e_vit = max(1e-4, min(1.0 - 1e-4, float(sub.get("ehr_vitals", base_p))))
            u_rul = max(1e-4, min(1.0 - 1e-4, float(sub.get("utilization_rules", base_p))))

            acuity = float(getattr(p, "acuity_score", 0.0))
            contra_cnt = float(len(getattr(p, "contraindications", ())))

            sub_mean = (c_lin + e_vit + u_rul) / 3.0
            sub_std = float(np.std([c_lin, e_vit, u_rul]))
            sub_max = max(c_lin, e_vit, u_rul)
            sub_min = min(c_lin, e_vit, u_rul)
            discrepancy = base_p - sub_mean

            v_c_diff = e_vit - c_lin
            u_c_diff = u_rul - c_lin

            rows.append([
                base_p,
                base_l,
                c_lin,
                safe_logit(c_lin),
                e_vit,
                safe_logit(e_vit),
                u_rul,
                safe_logit(u_rul),
                acuity,
                contra_cnt,
                sub_mean,
                sub_std,
                sub_max,
                sub_min,
                discrepancy,
                v_c_diff,
                u_c_diff,
                acuity * base_p,
                acuity * e_vit,
                acuity * discrepancy,
            ])
        return np.asarray(rows, dtype=np.float32)

    extra_train = extract_extra_features(train_records)
    extra_eval = extract_extra_features(eval_records)

    x_train_aug = np.column_stack([x_train_raw, extra_train])
    x_eval_aug = np.column_stack([x_eval_raw, extra_eval])

    imputer = SimpleImputer(strategy="median")
    x_train_imp = imputer.fit_transform(x_train_aug)
    x_eval_imp = imputer.transform(x_eval_aug)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_imp)
    x_eval_scaled = scaler.transform(x_eval_imp)
    np.clip(x_train_scaled, -10.0, 10.0, out=x_train_scaled)
    np.clip(x_eval_scaled, -10.0, 10.0, out=x_eval_scaled)

    def get_models(seed: int = 42):
        models = {}
        try:
            models["lgb"] = (
                lgb.LGBMClassifier(
                    n_estimators=120,
                    learning_rate=0.04,
                    max_depth=4,
                    num_leaves=15,
                    min_child_samples=8,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=seed,
                    verbose=-1,
                    n_jobs=1,
                ),
                "tree",
            )
        except Exception:
            pass

        try:
            models["xgb"] = (
                xgb.XGBClassifier(
                    n_estimators=120,
                    learning_rate=0.04,
                    max_depth=3,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    eval_metric="logloss",
                    random_state=seed,
                    verbosity=0,
                    n_jobs=1,
                ),
                "tree",
            )
        except Exception:
            pass

        try:
            models["hgb"] = (
                HistGradientBoostingClassifier(
                    max_iter=100,
                    learning_rate=0.04,
                    max_depth=4,
                    min_samples_leaf=8,
                    l2_regularization=1.0,
                    random_state=seed,
                ),
                "tree",
            )
        except Exception:
            pass

        try:
            models["et"] = (
                ExtraTreesClassifier(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_leaf=4,
                    random_state=seed,
                    n_jobs=1,
                ),
                "tree",
            )
        except Exception:
            pass

        try:
            models["lr"] = (
                LogisticRegression(
                    C=0.5,
                    max_iter=300,
                    random_state=seed,
                ),
                "scaled",
            )
        except Exception:
            pass

        return models

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    initial_models = get_models(42)
    model_names = list(initial_models.keys())
    oof_preds = {name: np.zeros(len(train_records), dtype=float) for name in model_names}
    eval_preds = {name: np.zeros(len(eval_records), dtype=float) for name in model_names}

    for fold, (trn_idx, val_idx) in enumerate(skf.split(x_train_imp, y_train)):
        fold_models = get_models(42 + fold)
        for name, (mdl, mtype) in fold_models.items():
            if mtype == "scaled":
                X_tr, X_va, X_te = x_train_scaled[trn_idx], x_train_scaled[val_idx], x_eval_scaled
            else:
                X_tr, X_va, X_te = x_train_imp[trn_idx], x_train_imp[val_idx], x_eval_imp

            try:
                mdl.fit(X_tr, y_train[trn_idx])
                oof_preds[name][val_idx] = mdl.predict_proba(X_va)[:, 1]
                eval_preds[name] += mdl.predict_proba(X_te)[:, 1] / n_splits
            except Exception:
                pass

    # Calibrated legacy baseline prediction stream
    y_mean = max(1e-4, min(1.0 - 1e-4, float(np.mean(y_train))))
    base_train = np.array([float(getattr(p, "baseline_api_prob", 0.5)) for p in train_records])
    base_eval = np.array([float(getattr(p, "baseline_api_prob", 0.5)) for p in eval_records])
    shift = safe_logit(y_mean) - float(np.mean([safe_logit(p) for p in base_train]))

    oof_preds["base"] = np.array([safe_sigmoid(safe_logit(p) + shift) for p in base_train])
    eval_preds["base"] = np.array([safe_sigmoid(safe_logit(p) + shift) for p in base_eval])
    model_names.append("base")

    valid_names = [
        name for name in model_names
        if np.std(oof_preds[name]) > 1e-6 and np.all(np.isfinite(oof_preds[name]))
    ]
    if not valid_names:
        valid_names = ["base"]

    P = np.column_stack([oof_preds[name] for name in valid_names])
    Q = np.column_stack([eval_preds[name] for name in valid_names])
    K = len(valid_names)

    # Solve constrained NNLS for optimal out-of-fold Brier weights
    def brier_objective(weights):
        w = weights / np.sum(weights)
        pred = P @ w
        return float(np.mean((pred - y_train) ** 2))

    init_w = np.ones(K) / K
    bounds = [(0.0, 1.0) for _ in range(K)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    try:
        res = minimize(
            brier_objective,
            init_w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-6},
        )
        if res.success and np.sum(res.x) > 1e-3:
            best_w = np.maximum(0.0, res.x)
            best_w /= np.sum(best_w)
        else:
            best_w = init_w
    except Exception:
        best_w = init_w

    oof_blend = P @ best_w
    eval_blend = Q @ best_w

    # Strictly monotonic Platt calibration (preserves exact ROC-AUC, PR-AUC, Sens@Spec80)
    oof_logits = np.array([safe_logit(float(p)) for p in oof_blend]).reshape(-1, 1)
    eval_logits = np.array([safe_logit(float(p)) for p in eval_blend]).reshape(-1, 1)

    try:
        platt = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        platt.fit(oof_logits, y_train)
        if platt.coef_[0][0] > 0.05:
            calib_eval = platt.predict_proba(eval_logits)[:, 1]
            calib_oof = platt.predict_proba(oof_logits)[:, 1]
        else:
            calib_eval = eval_blend
            calib_oof = oof_blend
    except Exception:
        calib_eval = eval_blend
        calib_oof = oof_blend

    # Monotonic logit shift to align with incidence base rate
    calib_oof_logits = [safe_logit(float(p)) for p in calib_oof]
    mean_oof_logit = float(np.mean(calib_oof_logits))
    shift_amount = 0.5 * (safe_logit(y_mean) - mean_oof_logit)
    shifted_eval_logits = [safe_logit(float(p)) + shift_amount for p in calib_eval]
    final_eval_probs = np.array([safe_sigmoid(l) for l in shifted_eval_logits])

    # Blend with RiskBlender baseline if available
    try:
        from clinical_toolkit import RiskBlender
        blender = RiskBlender(random_state=42)
        blender_dict = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train_raw,
            y_train=y_train_raw,
            x_eval=x_eval_raw,
        )
        if blender_dict and len(blender_dict) == len(eval_records):
            blender_probs = np.array([float(blender_dict[p.patient_id]) for p in eval_records])
            final_eval_probs = 0.85 * final_eval_probs + 0.15 * blender_probs
    except Exception:
        pass

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        prob = float(final_eval_probs[idx])
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, prob)), 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Cost-benefit marginal efficiency allocation with multi-tier upgrade optimization."""
    probe_state = CareAllocationState(cohort_spec, eval_records)
    costs = {t: float(s.cost_usd) for t, s in cohort_spec.interventions.items()}
    max_slots = {t: int(s.max_slots) for t, s in cohort_spec.interventions.items()}
    total_budget = float(cohort_spec.total_budget_usd)

    # Pre-calculate eligible tiers with positive net expected savings
    patient_tier_benefit = {}
    for p in eval_records:
        pid = p.patient_id
        risk = float(predicted_risks.get(pid, getattr(p, "baseline_api_prob", 0.5)))
        patient_tier_benefit[pid] = {}
        for tier in cohort_spec.interventions:
            if tier == InterventionTier.NONE:
                continue
            if probe_state.fits(pid, tier):
                ben = probe_state.expected_net_benefit(pid, risk, tier)
                if ben > 0.0:
                    patient_tier_benefit[pid][tier] = float(ben)

    # Candidate moves ranked by marginal return on investment
    candidate_moves = []
    for pid, tiers in patient_tier_benefit.items():
        for tier, ben in tiers.items():
            cost = max(costs[tier], 1.0)
            eff = ben / cost
            candidate_moves.append((eff, ben, pid, tier))

    candidate_moves.sort(key=lambda x: (x[0], x[1]), reverse=True)

    assigned_tier = {p.patient_id: InterventionTier.NONE for p in eval_records}
    tier_counts = {t: 0 for t in cohort_spec.interventions}
    spent_budget = 0.0

    # Phase 1: High-efficiency greedy assignment
    for eff, ben, pid, tier in candidate_moves:
        if assigned_tier[pid] != InterventionTier.NONE:
            continue
        c = costs[tier]
        if tier_counts[tier] + 1 <= max_slots[tier] and spent_budget + c <= total_budget:
            assigned_tier[pid] = tier
            tier_counts[tier] += 1
            spent_budget += c

    # Phase 2: Upgrade passes for surplus budget
    for _ in range(50):
        best_upgrade = None
        best_delta_eff = 0.0
        for p in eval_records:
            pid = p.patient_id
            cur_tier = assigned_tier[pid]
            cur_ben = patient_tier_benefit[pid].get(cur_tier, 0.0)
            cur_cost = costs.get(cur_tier, 0.0)
            for new_tier, new_ben in patient_tier_benefit[pid].items():
                if new_tier == cur_tier:
                    continue
                delta_ben = new_ben - cur_ben
                delta_cost = costs[new_tier] - cur_cost
                if delta_ben > 0.0:
                    if tier_counts[new_tier] + 1 <= max_slots[new_tier] and spent_budget + delta_cost <= total_budget:
                        delta_eff = delta_ben / max(delta_cost, 1.0)
                        if delta_eff > best_delta_eff:
                            best_delta_eff = delta_eff
                            best_upgrade = (pid, cur_tier, new_tier, delta_cost)

        if best_upgrade is not None:
            pid, cur_tier, new_tier, delta_cost = best_upgrade
            if cur_tier != InterventionTier.NONE:
                tier_counts[cur_tier] -= 1
            tier_counts[new_tier] += 1
            spent_budget += delta_cost
            assigned_tier[pid] = new_tier
        else:
            break

    # Build validated CareAllocationState safely
    final_state = CareAllocationState(cohort_spec, eval_records)
    sorted_pids = sorted(
        eval_records,
        key=lambda p: -float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5))),
    )

    for p in sorted_pids:
        pid = p.patient_id
        target_tier = assigned_tier[pid]
        if target_tier != InterventionTier.NONE:
            if final_state.fits(pid, target_tier):
                final_state.assign(pid, target_tier)
            else:
                for alt_tier in sorted(patient_tier_benefit[pid].keys(), key=lambda t: -patient_tier_benefit[pid][t]):
                    if final_state.fits(pid, alt_tier):
                        final_state.assign(pid, alt_tier)
                        break

    if final_state.all_ok():
        return final_state.to_plan()

    # Fallback to conservative greedy allocation to strictly avoid constraint violations
    fallback_state = CareAllocationState(cohort_spec, eval_records)
    tiers_by_rrr = [
        t for t, s in sorted(
            cohort_spec.interventions.items(),
            key=lambda kv: -kv[1].relative_risk_reduction,
        ) if t != InterventionTier.NONE
    ]
    for p in sorted_pids:
        risk = float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5)))
        for tier in tiers_by_rrr:
            if fallback_state.fits(p.patient_id, tier) and fallback_state.expected_net_benefit(p.patient_id, risk, tier) > 0.0:
                fallback_state.assign(p.patient_id, tier)
                break
    return fallback_state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
