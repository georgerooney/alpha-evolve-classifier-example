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


def _bounded_logit(val: float) -> float:
    return safe_logit(max(1e-4, min(1.0 - 1e-4, float(val))))


def _extract_dissonance_features(records: Sequence[PatientRecord]) -> np.ndarray:
    """Extract clinical tension, divergence, and non-linear discordance among legacy submodels."""
    feats = []
    for p in records:
        sub = getattr(p, "baseline_submodel_scores", {}) or {}
        api_prob = float(getattr(p, "baseline_api_prob", 0.2))
        claims = float(sub.get("claims_linear", api_prob))
        vitals = float(sub.get("ehr_vitals", api_prob))
        util = float(sub.get("utilization_rules", api_prob))
        acuity = float(getattr(p, "acuity_score", 0.0) or 0.0)

        # Logit-space submodel projections
        l_api = _bounded_logit(api_prob)
        l_claims = _bounded_logit(claims)
        l_vitals = _bounded_logit(vitals)
        l_util = _bounded_logit(util)

        # Pairwise clinical tensions
        diff_vitals_claims = vitals - claims      # Acute decompensation vs chronic baseline
        diff_util_claims = util - claims          # Utilization/SDoH vs chronic coding
        diff_vitals_util = vitals - util          # Acute vitals vs routine utilization
        diff_api_submean = api_prob - (claims + vitals + util) / 3.0

        # Submodel divergence dispersion
        sub_arr = np.array([claims, vitals, util], dtype=np.float32)
        spread = float(np.ptp(sub_arr))
        variance = float(np.var(sub_arr))

        # Syndrome interaction cross-terms
        acuity_vitals = acuity * vitals
        acuity_claims = acuity * claims
        shock_divergence = max(0.0, diff_vitals_claims) * (1.0 + acuity)
        fragility_illusion = max(0.0, claims - vitals) * (1.0 / (acuity + 0.1))

        feats.append([
            api_prob, claims, vitals, util, acuity,
            l_api, l_claims, l_vitals, l_util,
            diff_vitals_claims, diff_util_claims, diff_vitals_util, diff_api_submean,
            spread, variance,
            acuity_vitals, acuity_claims, shock_divergence, fragility_illusion
        ])
    return np.array(feats, dtype=np.float32)


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Chrono-Physiological Dissonance & Stratified Bagged Tri-Ensemble with Platt Stacking."""
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    # 1. Feature Engineering with Interaction Syndromes + Discordance Tensor
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)

    x_train_extra = _extract_dissonance_features(train_records)
    x_eval_extra = _extract_dissonance_features(eval_records)

    x_tr_full = np.nan_to_num(np.hstack([x_train_raw, x_train_extra]), nan=0.0, posinf=1.0, neginf=-1.0)
    x_ev_full = np.nan_to_num(np.hstack([x_eval_raw, x_eval_extra]), nan=0.0, posinf=1.0, neginf=-1.0)
    y_train = np.asarray(y_train, dtype=np.int32)

    # 2. Accelerated Failure-Time (AFT) Hazard Kernel Sample Weights
    sample_weights = np.ones(len(train_records), dtype=np.float32)
    for i, p in enumerate(train_records):
        if int(getattr(p, "outcome_label", 0) or 0) == 1:
            tt = float(getattr(p, "time_to_event_days", 15.0) or 15.0)
            sample_weights[i] = 1.0 + 0.35 * np.exp(-max(0.0, tt) / 12.0)

    # 3. Stratified 5-Fold Cross-Validated Stacking (OOF Generation & Bagging)
    n_pos = int(np.sum(y_train == 1))
    n_splits = max(2, min(5, n_pos))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_lgb = np.zeros(len(train_records), dtype=np.float32)
    oof_xgb = np.zeros(len(train_records), dtype=np.float32)
    oof_lr = np.zeros(len(train_records), dtype=np.float32)

    test_lgb = np.zeros(len(eval_records), dtype=np.float32)
    test_xgb = np.zeros(len(eval_records), dtype=np.float32)
    test_lr = np.zeros(len(eval_records), dtype=np.float32)

    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_tr_full)
    x_ev_scaled = scaler.transform(x_ev_full)

    for tr_idx, va_idx in skf.split(x_tr_full, y_train):
        x_tr_f, y_tr_f = x_tr_full[tr_idx], y_train[tr_idx]
        x_va_f = x_tr_full[va_idx]
        w_tr_f = sample_weights[tr_idx]

        # LightGBM Classifier
        clf_lgb = lgb.LGBMClassifier(
            n_estimators=170,
            learning_rate=0.035,
            num_leaves=24,
            max_depth=5,
            min_child_samples=16,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        clf_lgb.fit(x_tr_f, y_tr_f, sample_weight=w_tr_f)
        oof_lgb[va_idx] = clf_lgb.predict_proba(x_va_f)[:, 1]
        test_lgb += clf_lgb.predict_proba(x_ev_full)[:, 1] / float(n_splits)

        # XGBoost Classifier
        clf_xgb = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.035,
            max_depth=4,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=1,
        )
        clf_xgb.fit(x_tr_f, y_tr_f, sample_weight=w_tr_f)
        oof_xgb[va_idx] = clf_xgb.predict_proba(x_va_f)[:, 1]
        test_xgb += clf_xgb.predict_proba(x_ev_full)[:, 1] / float(n_splits)

        # Scaled Logistic Regression
        clf_lr = LogisticRegression(C=0.6, max_iter=300, random_state=42)
        clf_lr.fit(x_tr_scaled[tr_idx], y_tr_f, sample_weight=w_tr_f)
        oof_lr[va_idx] = clf_lr.predict_proba(x_tr_scaled[va_idx])[:, 1]
        test_lr += clf_lr.predict_proba(x_ev_scaled)[:, 1] / float(n_splits)

    # 4. Out-of-Fold Platt-Calibrated Logit Meta-Learner
    train_api = np.array([p.baseline_api_prob for p in train_records], dtype=np.float32)
    eval_api = np.array([p.baseline_api_prob for p in eval_records], dtype=np.float32)

    meta_x_tr = np.column_stack([
        [_bounded_logit(v) for v in oof_lgb],
        [_bounded_logit(v) for v in oof_xgb],
        [_bounded_logit(v) for v in oof_lr],
        [_bounded_logit(v) for v in train_api],
    ])
    meta_x_ev = np.column_stack([
        [_bounded_logit(v) for v in test_lgb],
        [_bounded_logit(v) for v in test_xgb],
        [_bounded_logit(v) for v in test_lr],
        [_bounded_logit(v) for v in eval_api],
    ])

    meta_clf = LogisticRegression(C=1.0, fit_intercept=True, random_state=42)
    meta_clf.fit(meta_x_tr, y_train)
    stacked_probs = meta_clf.predict_proba(meta_x_ev)[:, 1]

    # 5. Native Toolkit RiskBlender Hybridization
    blender_preds = None
    try:
        blender = RiskBlender(
            linear_weight=0.35,
            lgbm_weight=0.38,
            xgb_weight=0.22,
            baseline_prior_weight=0.05,
            random_state=42,
        )
        blender_preds = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train_raw,
            y_train=y_train,
            x_eval=x_eval_raw,
        )
    except Exception:
        blender_preds = None

    # 6. Ensemble Synthesis & Prevalence Calibration Centering
    train_prevalence = float(np.mean(y_train))
    comb_logits = []
    for idx, p in enumerate(eval_records):
        p_stk = float(stacked_probs[idx])
        p_api = float(p.baseline_api_prob)
        if blender_preds is not None and p.patient_id in blender_preds:
            p_bln = float(blender_preds[p.patient_id])
            l_val = 0.50 * _bounded_logit(p_stk) + 0.40 * _bounded_logit(p_bln) + 0.10 * _bounded_logit(p_api)
        else:
            l_val = 0.85 * _bounded_logit(p_stk) + 0.15 * _bounded_logit(p_api)
        comb_logits.append(l_val)

    comb_probs = [safe_sigmoid(lv) for lv in comb_logits]
    mean_comb = float(np.mean(comb_probs))
    if abs(mean_comb - train_prevalence) > 0.005:
        shift = _bounded_logit(train_prevalence) - _bounded_logit(mean_comb)
        comb_probs = [safe_sigmoid(lv + 0.5 * shift) for lv in comb_logits]

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        prob = float(comb_probs[idx])
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, prob)), 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Tri-heuristic simulated dispatch maximizing cohort Net ROI subject to hard constraints."""
    strategies = ["damped_roi", "pure_benefit", "baseline_rrr"]
    best_state = None
    best_total_benefit = -1.0

    for strat in strategies:
        state = CareAllocationState(cohort_spec, eval_records)
        assigned = set()
        total_benefit = 0.0

        if strat == "baseline_rrr":
            ranked_patients = sorted(
                eval_records,
                key=lambda p: -float(predicted_risks.get(p.patient_id, p.baseline_api_prob)),
            )
            tiers_by_rrr = [
                t
                for t, s in sorted(
                    cohort_spec.interventions.items(),
                    key=lambda kv: -kv[1].relative_risk_reduction,
                )
                if t != InterventionTier.NONE
            ]
            for p in ranked_patients:
                risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
                for tier in tiers_by_rrr:
                    if state.fits(p.patient_id, tier):
                        enb = state.expected_net_benefit(p.patient_id, risk, tier)
                        if enb > 0.0:
                            if state.assign(p.patient_id, tier):
                                assigned.add(p.patient_id)
                                total_benefit += enb
                            break
        else:
            candidates = []
            for p in eval_records:
                risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
                for tier, spec in cohort_spec.interventions.items():
                    if tier == InterventionTier.NONE:
                        continue
                    enb = state.expected_net_benefit(p.patient_id, risk, tier)
                    if enb > 0.0:
                        cost = max(1.0, float(getattr(spec, "cost_usd", 1.0) or 1.0))
                        prio = enb / (cost ** 0.5) if strat == "damped_roi" else enb
                        candidates.append((prio, enb, p.patient_id, tier))

            candidates.sort(key=lambda x: -x[0])
            for prio, enb, pid, tier in candidates:
                if pid in assigned:
                    continue
                if state.fits(pid, tier):
                    if state.assign(pid, tier):
                        assigned.add(pid)
                        total_benefit += enb

        if state.all_ok() and total_benefit > best_total_benefit:
            best_total_benefit = total_benefit
            best_state = state

    if best_state is not None:
        return best_state.to_plan()

    fallback = CareAllocationState(cohort_spec, eval_records)
    return fallback.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
