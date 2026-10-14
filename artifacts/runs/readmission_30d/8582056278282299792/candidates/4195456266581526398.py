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
from typing import Dict, Sequence, Optional, List, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb

try:
    from clinical_toolkit import (
        ClinicalFeatureBuilder,
        CareAllocationState,
        CohortSpec,
        PatientRecord,
        InterventionTier,
        CareAllocationPlan,
    )
except ImportError:
    pass

try:
    from clinical_toolkit import RiskBlender
except (ImportError, AttributeError):
    RiskBlender = None


def safe_logit(p: float, eps: float = 1e-6) -> float:
    p_clamped = max(eps, min(1.0 - eps, float(p)))
    return float(np.log(p_clamped / (1.0 - p_clamped)))


def safe_sigmoid(z: float) -> float:
    z_clamped = max(-35.0, min(35.0, float(z)))
    return float(1.0 / (1.0 + np.exp(-z_clamped)))


def extract_meta_features(records: Sequence[Any]) -> np.ndarray:
    feats = []
    for p in records:
        b_prob = float(getattr(p, 'baseline_api_prob', 0.5))
        b_logit = safe_logit(b_prob)
        acuity = float(getattr(p, 'acuity_score', 1.0))
        sub = getattr(p, 'baseline_submodel_scores', {}) or {}
        c_prob = float(sub.get('claims_linear', b_prob))
        v_prob = float(sub.get('ehr_vitals', b_prob))
        u_prob = float(sub.get('utilization_rules', b_prob))
        c_logit = safe_logit(c_prob)
        v_logit = safe_logit(v_prob)
        u_logit = safe_logit(u_prob)

        # 1. Divergence across submodels in logit and probability space
        acute_decomp_gap = v_logit - c_logit
        acute_prob_gap = v_prob - c_prob
        util_surge = u_logit - c_logit
        util_prob_surge = u_prob - c_prob
        util_v_gap = u_logit - v_logit
        util_b_gap = u_logit - b_logit

        # 2. Consensus and dispersion
        sub_probs = [c_prob, v_prob, u_prob]
        sub_mean = float(np.mean(sub_probs))
        sub_std = float(np.std(sub_probs))
        sub_max = max(sub_probs)
        sub_min = min(sub_probs)
        sub_spread = max(c_logit, v_logit, u_logit) - min(c_logit, v_logit, u_logit)
        api_consensus_gap = b_prob - sub_mean
        api_logit_gap = b_logit - safe_logit(sub_mean)

        # 3. Acuity-weighted interactions
        acuity_risk = acuity * b_logit
        acuity_vitals = acuity * v_logit
        acuity_claims = acuity * c_logit
        acuity_util = acuity * u_logit
        acuity_sq = acuity ** 2
        log_acuity = float(np.log(max(1e-3, acuity)))
        vitals_flare = acute_prob_gap * acuity

        # 4. SDoH adherence spiral proxies
        adherence_spiral = u_logit * c_logit
        sdoh_churn = u_prob * max(0.0, 1.0 - c_prob)

        # 5. Elderly elective low-acuity dampeners
        elective_dampener = b_logit / (1.0 + np.exp(-acuity))
        claims_acuity_ratio = c_prob / (acuity + 0.5)

        # 6. Geometric & harmonic means
        geo_sub = float((max(1e-4, c_prob) * max(1e-4, v_prob) * max(1e-4, u_prob)) ** (1.0 / 3.0))
        harm_sub = float(3.0 / (1.0 / max(1e-4, c_prob) + 1.0 / max(1e-4, v_prob) + 1.0 / max(1e-4, u_prob)))

        # 7. Contraindications burden & clinical complexity
        contra_cnt = float(len(getattr(p, 'contraindications', []) or []))
        contra_acuity = contra_cnt * acuity
        has_contra = 1.0 if contra_cnt > 0 else 0.0

        feats.append([
            b_prob,
            b_logit,
            acuity,
            c_prob,
            v_prob,
            u_prob,
            c_logit,
            v_logit,
            u_logit,
            acute_decomp_gap,
            acute_prob_gap,
            util_surge,
            util_prob_surge,
            util_v_gap,
            util_b_gap,
            sub_mean,
            sub_std,
            sub_max,
            sub_min,
            sub_spread,
            api_consensus_gap,
            api_logit_gap,
            acuity_risk,
            acuity_vitals,
            acuity_claims,
            acuity_util,
            acuity_sq,
            log_acuity,
            vitals_flare,
            adherence_spiral,
            sdoh_churn,
            elective_dampener,
            claims_acuity_ratio,
            geo_sub,
            harm_sub,
            contra_cnt,
            contra_acuity,
            has_contra,
        ])
    return np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)


def get_tier_cost(spec: Any, tier: Any) -> float:
    interv = spec.interventions.get(tier)
    if interv is None:
        return 1.0
    for attr in ('cost_usd', 'cost', 'unit_cost', 'cost_per_slot'):
        if hasattr(interv, attr):
            return float(getattr(interv, attr))
    if isinstance(interv, dict):
        for k in ('cost_usd', 'cost', 'unit_cost'):
            if k in interv:
                return float(interv[k])
    return 1000.0


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Multi-model RiskBlender with 10 non-linear syndrome interaction features."""
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)

    x_train = np.nan_to_num(x_train, nan=0.0, posinf=1.0, neginf=-1.0)
    x_eval = np.nan_to_num(x_eval, nan=0.0, posinf=1.0, neginf=-1.0)
    y_train = np.array(y_train, dtype=np.int32)

    meta_train = extract_meta_features(train_records)
    meta_eval = extract_meta_features(eval_records)

    x_tr_full = np.hstack([x_train, meta_train])
    x_ev_full = np.hstack([x_eval, meta_eval])

    if len(np.unique(y_train)) < 2:
        return {
            p.patient_id: round(max(1e-4, min(1.0 - 1e-4, float(getattr(p, 'baseline_api_prob', 0.5)))), 6)
            for p in eval_records
        }

    # 1. Toolkit RiskBlender if available
    blender_preds = None
    blender_cls = globals().get('RiskBlender', None)
    if blender_cls is None:
        try:
            import clinical_toolkit
            blender_cls = getattr(clinical_toolkit, 'RiskBlender', None)
        except Exception:
            blender_cls = None

    if blender_cls is not None:
        try:
            blender = blender_cls(
                linear_weight=0.35,
                lgbm_weight=0.40,
                xgb_weight=0.25,
                baseline_prior_weight=0.08,
                random_state=42,
            )
            try:
                blender_preds = blender.fit_predict(
                    cohort_spec,
                    train_records,
                    eval_records,
                    x_train=x_tr_full,
                    y_train=y_train,
                    x_eval=x_ev_full,
                )
            except Exception:
                blender_preds = blender.fit_predict(
                    cohort_spec,
                    train_records,
                    eval_records,
                    x_train=x_train,
                    y_train=y_train,
                    x_eval=x_eval,
                )
        except Exception:
            blender_preds = None

    # 2. Multi-Model Bagged Cross-Validation Ensemble with OOF tracking
    n_train = len(y_train)
    min_class_count = int(np.min(np.bincount(y_train)))
    n_splits = 5 if min_class_count >= 5 else 1

    oof_lgb = np.zeros(n_train, dtype=np.float32)
    oof_xgb = np.zeros(n_train, dtype=np.float32)
    oof_hgb = np.zeros(n_train, dtype=np.float32)
    oof_et = np.zeros(n_train, dtype=np.float32)
    oof_lr = np.zeros(n_train, dtype=np.float32)
    oof_base = np.array([float(getattr(p, 'baseline_api_prob', 0.5)) for p in train_records], dtype=np.float32)

    p_lgb_folds: List[np.ndarray] = []
    p_xgb_folds: List[np.ndarray] = []
    p_hgb_folds: List[np.ndarray] = []
    p_et_folds: List[np.ndarray] = []
    p_lr_folds: List[np.ndarray] = []

    if n_splits > 1:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(x_tr_full, y_train))
    else:
        splits = [(np.arange(n_train), np.arange(n_train))]

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        x_tr_fold = x_tr_full[tr_idx]
        y_tr_fold = y_train[tr_idx]
        x_val_fold = x_tr_full[val_idx]

        # LightGBM (subsample_freq=1 activates row subsampling bagging!)
        lgb_clf = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=135,
            learning_rate=0.038,
            num_leaves=15,
            max_depth=5,
            subsample=0.82,
            subsample_freq=1,
            colsample_bytree=0.78,
            min_child_samples=8,
            reg_alpha=0.15,
            reg_lambda=1.2,
            random_state=42 + fold_idx,
            verbosity=-1,
            n_jobs=1,
        )
        lgb_clf.fit(x_tr_fold, y_tr_fold)
        oof_lgb[val_idx] = lgb_clf.predict_proba(x_val_fold)[:, 1]
        p_lgb_folds.append(lgb_clf.predict_proba(x_ev_full)[:, 1])

        # XGBoost
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=115,
            learning_rate=0.042,
            max_depth=4,
            subsample=0.82,
            colsample_bytree=0.78,
            min_child_weight=2,
            gamma=0.1,
            reg_alpha=0.15,
            reg_lambda=1.2,
            random_state=42 + fold_idx,
            use_label_encoder=False,
            n_jobs=1,
        )
        xgb_clf.fit(x_tr_fold, y_tr_fold)
        oof_xgb[val_idx] = xgb_clf.predict_proba(x_val_fold)[:, 1]
        p_xgb_folds.append(xgb_clf.predict_proba(x_ev_full)[:, 1])

        # HistGradientBoosting (binned gradient boosting diversity)
        hgb_clf = HistGradientBoostingClassifier(
            loss='log_loss',
            learning_rate=0.04,
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=8,
            l2_regularization=1.0,
            random_state=42 + fold_idx,
        )
        hgb_clf.fit(x_tr_fold, y_tr_fold)
        oof_hgb[val_idx] = hgb_clf.predict_proba(x_val_fold)[:, 1]
        p_hgb_folds.append(hgb_clf.predict_proba(x_ev_full)[:, 1])

        # ExtraTrees (randomized tree bagging - variance reduction orthogonal to boosting)
        et_clf = ExtraTreesClassifier(
            n_estimators=85,
            max_depth=6,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42 + fold_idx,
            n_jobs=1,
        )
        et_clf.fit(x_tr_fold, y_tr_fold)
        oof_et[val_idx] = et_clf.predict_proba(x_val_fold)[:, 1]
        p_et_folds.append(et_clf.predict_proba(x_ev_full)[:, 1])

        # Calibrated Logistic Regression
        scaler = StandardScaler()
        x_tr_s = np.nan_to_num(scaler.fit_transform(x_tr_fold), nan=0.0)
        x_val_s = np.nan_to_num(scaler.transform(x_val_fold), nan=0.0)
        x_ev_s = np.nan_to_num(scaler.transform(x_ev_full), nan=0.0)
        lr_clf = LogisticRegression(
            C=0.20,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42 + fold_idx,
        )
        lr_clf.fit(x_tr_s, y_tr_fold)
        oof_lr[val_idx] = lr_clf.predict_proba(x_val_s)[:, 1]
        p_lr_folds.append(lr_clf.predict_proba(x_ev_s)[:, 1])

    p_lgb = np.mean(p_lgb_folds, axis=0)
    p_xgb = np.mean(p_xgb_folds, axis=0)
    p_hgb = np.mean(p_hgb_folds, axis=0)
    p_et = np.mean(p_et_folds, axis=0)
    p_lr = np.mean(p_lr_folds, axis=0)
    p_base = np.array([float(getattr(p, 'baseline_api_prob', 0.5)) for p in eval_records], dtype=np.float32)

    # 3. Dynamic SLSQP Weight Optimization on Out-Of-Fold Predictions
    oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_hgb, oof_et, oof_lr, oof_base])
    eval_matrix = np.column_stack([p_lgb, p_xgb, p_hgb, p_et, p_lr, p_base])

    def oof_objective(weights: np.ndarray) -> float:
        w = np.maximum(0.0, np.array(weights))
        s = np.sum(w)
        if s > 0:
            w = w / s
        pred = np.clip(oof_matrix @ w, 1e-6, 1.0 - 1e-6)
        logloss = -np.mean(y_train * np.log(pred) + (1.0 - y_train) * np.log(1.0 - pred))
        brier = np.mean((pred - y_train) ** 2)
        return float(logloss + brier)

    init_weights = np.array([0.28, 0.24, 0.18, 0.12, 0.10, 0.08], dtype=np.float64)
    bounds = [(0.02, 0.60)] * 6
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    try:
        opt_res = minimize(oof_objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        if opt_res.success and np.all(np.isfinite(opt_res.x)):
            best_w = np.maximum(0.0, opt_res.x)
            best_w = best_w / np.sum(best_w)
        else:
            best_w = init_weights / np.sum(init_weights)
    except Exception:
        best_w = init_weights / np.sum(init_weights)

    oof_ens = oof_matrix @ best_w
    p_ens = eval_matrix @ best_w

    if blender_preds is not None:
        p_blender = np.array([float(blender_preds.get(p.patient_id, p.baseline_api_prob)) for p in eval_records])
        p_ens = 0.82 * p_ens + 0.18 * p_blender

    # 4. Strictly Monotonic Platt Calibration + Prior Matching (Preserves 100% of ranks while minimizing Brier)
    try:
        z_oof = np.array([safe_logit(p) for p in oof_ens]).reshape(-1, 1)
        z_eval = np.array([safe_logit(p) for p in p_ens]).reshape(-1, 1)
        calibrator = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=500)
        calibrator.fit(z_oof, y_train)
        if calibrator.coef_[0][0] > 0:
            p_cal = calibrator.predict_proba(z_eval)[:, 1]
        else:
            p_cal = p_ens
    except Exception:
        p_cal = p_ens

    train_prior = float(np.mean(y_train))
    eval_mean = float(np.mean(p_cal))
    shift = 0.5 * (safe_logit(train_prior) - safe_logit(eval_mean))
    logits_calibrated = np.array([safe_logit(p) + shift for p in p_cal])
    p_calibrated = np.array([safe_sigmoid(z) for z in logits_calibrated])

    out: Dict[str, float] = {}
    for i, p in enumerate(eval_records):
        prob = float(p_calibrated[i])
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, prob)), 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Multi-policy marginal ROI knapsack allocation maximizing actuarial net ROI."""
    none_tier = None
    if 'InterventionTier' in globals():
        none_tier = getattr(InterventionTier, 'NONE', None)
    if none_tier is None:
        for t in cohort_spec.interventions.keys():
            if getattr(t, 'name', '') == 'NONE' or str(t) == 'NONE':
                none_tier = t
                break

    active_tiers = [
        t for t in cohort_spec.interventions.keys()
        if t != none_tier and getattr(t, 'name', '') != 'NONE' and str(t) != 'NONE'
    ]
    tier_costs = {t: get_tier_cost(cohort_spec, t) for t in active_tiers}

    # Simulate competing knapsack allocation strategies and choose the one maximizing expected net benefit
    candidates: List[Tuple[str, Any]] = [
        ('greedy_benefit', None),
        ('roi_0.5', 0.5),
        ('roi_0.7', 0.7),
        ('roi_0.85', 0.85),
        ('roi_1.0', 1.0),
        ('acuity_roi', 'acuity'),
    ]

    best_plan: Optional[CareAllocationPlan] = None
    best_total_benefit = -1e9

    for name, param in candidates:
        cand_state = CareAllocationState(cohort_spec, eval_records)
        assigned: set = set()
        cand_total_benefit = 0.0

        if name == 'greedy_benefit':
            ranked = sorted(
                eval_records,
                key=lambda p: -float(predicted_risks.get(p.patient_id, p.baseline_api_prob)),
            )
            for p in ranked:
                risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
                b_tier = None
                b_val = 0.0
                for tier in active_tiers:
                    if cand_state.fits(p.patient_id, tier):
                        val = cand_state.expected_net_benefit(p.patient_id, risk, tier)
                        if val > b_val:
                            b_val = val
                            b_tier = tier
                if b_tier is not None:
                    if cand_state.assign(p.patient_id, b_tier):
                        assigned.add(p.patient_id)
                        cand_total_benefit += b_val
        elif name == 'acuity_roi':
            pairs = []
            for p in eval_records:
                risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
                for tier in active_tiers:
                    if cand_state.fits(p.patient_id, tier):
                        val = cand_state.expected_net_benefit(p.patient_id, risk, tier)
                        if val > 0:
                            score = (val * (getattr(p, 'acuity_score', 1.0) ** 0.5)) / (tier_costs[tier] ** 0.6 + 5.0)
                            pairs.append((score, val, p, tier))
            pairs.sort(key=lambda x: -x[0])
            for score, val, p, tier in pairs:
                if p.patient_id not in assigned and cand_state.fits(p.patient_id, tier):
                    if cand_state.assign(p.patient_id, tier):
                        assigned.add(p.patient_id)
                        cand_total_benefit += val
        else:
            exp = float(param)
            pairs = []
            for p in eval_records:
                risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
                for tier in active_tiers:
                    if cand_state.fits(p.patient_id, tier):
                        val = cand_state.expected_net_benefit(p.patient_id, risk, tier)
                        if val > 0:
                            score = val / (tier_costs[tier] ** exp + 1.0)
                            pairs.append((score, val, p, tier))
            pairs.sort(key=lambda x: -x[0])
            for score, val, p, tier in pairs:
                if p.patient_id not in assigned and cand_state.fits(p.patient_id, tier):
                    if cand_state.assign(p.patient_id, tier):
                        assigned.add(p.patient_id)
                        cand_total_benefit += val

        plan = cand_state.to_plan()
        if cand_total_benefit > best_total_benefit or best_plan is None:
            best_total_benefit = cand_total_benefit
            best_plan = plan

    # Hard constraint verification: every patient must be assigned
    if best_plan is not None and none_tier is not None:
        for p in eval_records:
            if p.patient_id not in best_plan.assignments:
                best_plan.assignments[p.patient_id] = none_tier

    return best_plan if best_plan is not None else CareAllocationState(cohort_spec, eval_records).to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
