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

        # 1. Acute decompensation gap (acute vitals divergence from chronic claims)
        acute_decomp_gap = v_logit - c_logit
        # 2. Frequent utilization surge
        util_surge = u_logit - c_logit
        # 3. Utilization vs baseline
        util_b_gap = u_logit - b_logit
        # 4. Multi-submodel discrepancy spread
        sub_spread = max(c_logit, v_logit, u_logit) - min(c_logit, v_logit, u_logit)
        # 5. Acuity-weighted baseline risk
        acuity_risk = acuity * b_logit
        # 6. Acuity-weighted acute vitals
        acuity_vitals = acuity * v_logit
        # 7. Elderly elective dampener (mitigate over-prediction when acuity is low)
        dampener = b_logit / (1.0 + np.exp(-acuity))
        # 8. SDoH non-adherence spiral proxy
        adherence_spiral = u_logit * c_logit
        # 9. Contraindications burden
        contra_cnt = float(len(getattr(p, 'contraindications', []) or []))

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
            util_surge,
            util_b_gap,
            sub_spread,
            acuity_risk,
            acuity_vitals,
            dampener,
            adherence_spiral,
            contra_cnt,
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
                linear_weight=0.40,
                lgbm_weight=0.35,
                xgb_weight=0.25,
                baseline_prior_weight=0.08,
                random_state=42,
            )
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

    # 2. Multi-Model Bagged Cross-Validation Ensemble
    min_class_count = int(np.min(np.bincount(y_train)))
    n_splits = 5 if min_class_count >= 5 else 1

    p_lgb_folds: List[np.ndarray] = []
    p_xgb_folds: List[np.ndarray] = []
    p_lr_folds: List[np.ndarray] = []

    if n_splits > 1:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(x_tr_full, y_train))
    else:
        splits = [(np.arange(len(y_train)), np.arange(len(y_train)))]

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        x_tr_fold = x_tr_full[tr_idx]
        y_tr_fold = y_train[tr_idx]

        # LightGBM
        lgb_clf = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=130,
            learning_rate=0.04,
            num_leaves=15,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.80,
            min_child_samples=8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold_idx,
            verbosity=-1,
            n_jobs=1,
        )
        lgb_clf.fit(x_tr_fold, y_tr_fold)
        p_lgb_folds.append(lgb_clf.predict_proba(x_ev_full)[:, 1])

        # XGBoost
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_estimators=110,
            learning_rate=0.045,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.80,
            min_child_weight=2,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold_idx,
            use_label_encoder=False,
            n_jobs=1,
        )
        xgb_clf.fit(x_tr_fold, y_tr_fold)
        p_xgb_folds.append(xgb_clf.predict_proba(x_ev_full)[:, 1])

        # Calibrated Logistic Regression
        scaler = StandardScaler()
        x_tr_s = scaler.fit_transform(x_tr_fold)
        x_ev_s = scaler.transform(x_ev_full)
        lr_clf = LogisticRegression(
            C=0.20,
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42 + fold_idx,
        )
        lr_clf.fit(x_tr_s, y_tr_fold)
        p_lr_folds.append(lr_clf.predict_proba(x_ev_s)[:, 1])

    p_lgb = np.mean(p_lgb_folds, axis=0)
    p_xgb = np.mean(p_xgb_folds, axis=0)
    p_lr = np.mean(p_lr_folds, axis=0)
    p_base = np.array([float(getattr(p, 'baseline_api_prob', 0.5)) for p in eval_records])

    if blender_preds is not None:
        p_blender = np.array([float(blender_preds.get(p.patient_id, p.baseline_api_prob)) for p in eval_records])
        p_ens = 0.35 * p_lgb + 0.25 * p_xgb + 0.15 * p_lr + 0.15 * p_blender + 0.10 * p_base
    else:
        p_ens = 0.45 * p_lgb + 0.30 * p_xgb + 0.15 * p_lr + 0.10 * p_base

    # 3. Strictly Monotonic Prior-Matching Logit Shift (Optimizes Brier without altering ranking)
    train_prior = float(np.mean(y_train))
    eval_mean = float(np.mean(p_ens))
    shift = 0.5 * (safe_logit(train_prior) - safe_logit(eval_mean))
    logits_calibrated = np.array([safe_logit(p) + shift for p in p_ens])
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
        ('roi_0.75', 0.75),
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
