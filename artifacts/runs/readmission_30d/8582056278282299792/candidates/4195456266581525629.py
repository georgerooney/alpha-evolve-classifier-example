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
    """Stage 1: Multi-modal clinical ensemble with syndrome interactions, CV-bagging, and monotonic calibration."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb
    import xgboost as xgb

    def _logit(p_arr: np.ndarray) -> np.ndarray:
        p_c = np.clip(p_arr, 1e-5, 1.0 - 1e-5)
        return np.log(p_c / (1.0 - p_c))

    def _sigmoid(z_arr: np.ndarray) -> np.ndarray:
        z_c = np.clip(z_arr, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z_c))

    if not eval_records:
        return {}

    # Extract non-linear syndrome interaction features from frozen toolkit
    builder = ClinicalFeatureBuilder(include_interactions=True)
    try:
        x_train_raw, y_train_raw = builder.fit_transform(cohort_spec, train_records)
        x_eval_raw = builder.transform(cohort_spec, eval_records)
    except Exception:
        x_train_raw = np.zeros((len(train_records), 1), dtype=np.float64)
        y_train_raw = np.array([getattr(r, "outcome_label", 0) or 0 for r in train_records], dtype=np.int32)
        x_eval_raw = np.zeros((len(eval_records), 1), dtype=np.float64)

    x_train = np.nan_to_num(x_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    x_eval = np.nan_to_num(x_eval_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.array(y_train_raw, dtype=np.int32)

    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2 or len(train_records) < 4:
        out: Dict[str, float] = {}
        for p in eval_records:
            val = float(getattr(p, "baseline_api_prob", 0.5) or 0.5)
            out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, val)), 6)
        return out

    # Dynamic extraction of any additional scalar attributes present on records
    extra_keys = []
    if train_records:
        sample = train_records[0]
        for k in dir(sample):
            if k.startswith("_") or k in {
                "outcome_label", "time_to_event_days", "patient_id",
                "baseline_api_prob", "baseline_submodel_scores",
                "contraindications", "acuity_score"
            }:
                continue
            val = getattr(sample, k, None)
            if isinstance(val, (int, float, np.number)) and not isinstance(val, bool):
                extra_keys.append(k)
            elif isinstance(val, bool):
                extra_keys.append(k)

    # Advanced clinical feature extraction incorporating HRRP cross-organ interactions
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

            # Submodel consensus and dispersion statistics
            sub_logits = [l_api, l_claims, l_vitals, l_util]
            l_mean = float(np.mean(sub_logits))
            l_std = float(np.std(sub_logits))
            l_max = float(np.max(sub_logits))
            l_min = float(np.min(sub_logits))
            l_range = l_max - l_min

            # Cross-system discrepancy signals
            disc_ac_chr = l_vitals - l_claims
            disc_ac_utl = l_vitals - l_util
            disc_chr_utl = l_claims - l_util
            disc_api_chr = l_api - l_claims
            disc_api_vit = l_api - l_vitals

            # Acuity interactions
            acuity_api = acuity * l_api
            acuity_vitals = acuity * l_vitals
            acuity_claims = acuity * l_claims
            acuity_util = acuity * l_util
            acuity_range = acuity * l_range
            acuity_sq = acuity ** 2

            # Clinical phenotype syndrome indicators
            contra_count = float(len(getattr(r, "contraindications", []) or []))
            has_contra = 1.0 if contra_count > 0.0 else 0.0
            contra_acuity = contra_count * acuity
            extreme_vitals = (1.0 if vitals > 0.55 else 0.0) * acuity
            chronic_frailty = claims * acuity
            sdoh_spiral = 1.0 if (util > 0.50 and claims < 0.45) else 0.0
            elderly_elective = max(0.0, api - 2.0 * acuity)

            row = [
                api, l_api,
                claims, l_claims,
                vitals, l_vitals,
                util, l_util,
                acuity, acuity_sq,
                l_mean, l_std, l_max, l_min, l_range,
                disc_ac_chr, disc_ac_utl, disc_chr_utl, disc_api_chr, disc_api_vit,
                acuity_api, acuity_vitals, acuity_claims, acuity_util, acuity_range,
                contra_count, has_contra, contra_acuity,
                extreme_vitals, chronic_frailty, sdoh_spiral, elderly_elective,
            ]
            for ek in extra_keys:
                row.append(float(getattr(r, ek, 0.0) or 0.0))

            feats.append(row)
        return np.array(feats, dtype=np.float64)

    sub_train = extract_submodel_features(train_records)
    sub_eval = extract_submodel_features(eval_records)

    x_train_aug = np.nan_to_num(np.hstack([x_train, sub_train]), nan=0.0, posinf=0.0, neginf=0.0)
    x_eval_aug = np.nan_to_num(np.hstack([x_eval, sub_eval]), nan=0.0, posinf=0.0, neginf=0.0)

    n_eval = len(eval_records)
    n_train = len(train_records)

    # 1. Primary Toolkit Model: RiskBlender (blends LightGBM, XGBoost, Linear, and Prior)
    rb_probs_eval = None
    try:
        blender = RiskBlender(
            linear_weight=0.35,
            lgbm_weight=0.40,
            xgb_weight=0.25,
            baseline_prior_weight=0.03,
            random_state=42,
        )
        rb_dict = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train_aug,
            y_train=y_train,
            x_eval=x_eval_aug,
        )
        if rb_dict and len(rb_dict) == n_eval:
            rb_probs_eval = np.array([float(rb_dict[p.patient_id]) for p in eval_records], dtype=np.float64)
    except Exception:
        rb_probs_eval = None

    # Fallback to standard features if augmented call failed
    if rb_probs_eval is None:
        try:
            blender = RiskBlender(
                linear_weight=0.40,
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
            if rb_dict and len(rb_dict) == n_eval:
                rb_probs_eval = np.array([float(rb_dict[p.patient_id]) for p in eval_records], dtype=np.float64)
        except Exception:
            rb_probs_eval = None

    # 2. Stratified Cross-Validated Multi-Model Ensemble
    min_class_count = int(np.min(np.bincount(y_train)))
    n_splits = max(2, min(5, min_class_count))

    cv_models = {
        "lgbm": [],
        "xgb": [],
        "hist": [],
        "lr": [],
        "et": [],
    }

    oof_preds = {m: np.zeros(n_train, dtype=np.float64) for m in cv_models}
    eval_preds = {m: np.zeros(n_eval, dtype=np.float64) for m in cv_models}
    model_success = {m: False for m in cv_models}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train_aug, y_train)):
        X_tr, y_tr = x_train_aug[tr_idx], y_train[tr_idx]
        X_val = x_train_aug[val_idx]

        # Model A: Tuned LightGBM
        try:
            clf_lgb = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.035,
                num_leaves=15,
                max_depth=4,
                min_child_samples=8,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42 + fold,
                verbose=-1,
                n_jobs=1,
            )
            clf_lgb.fit(X_tr, y_tr)
            p_val = clf_lgb.predict_proba(X_val)[:, 1]
            p_ev = clf_lgb.predict_proba(x_eval_aug)[:, 1]
            oof_preds["lgbm"][val_idx] = p_val
            eval_preds["lgbm"] += p_ev / n_splits
            model_success["lgbm"] = True
        except Exception:
            pass

        # Model B: Tuned XGBoost
        try:
            clf_xgb = xgb.XGBClassifier(
                n_estimators=80,
                learning_rate=0.04,
                max_depth=3,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.80,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=42 + fold,
                n_jobs=1,
            )
            clf_xgb.fit(X_tr, y_tr)
            p_val = clf_xgb.predict_proba(X_val)[:, 1]
            p_ev = clf_xgb.predict_proba(x_eval_aug)[:, 1]
            oof_preds["xgb"][val_idx] = p_val
            eval_preds["xgb"] += p_ev / n_splits
            model_success["xgb"] = True
        except Exception:
            pass

        # Model C: HistGradientBoosting
        try:
            clf_hist = HistGradientBoostingClassifier(
                max_iter=90,
                learning_rate=0.04,
                max_depth=4,
                min_samples_leaf=8,
                l2_regularization=1.0,
                random_state=42 + fold,
            )
            clf_hist.fit(X_tr, y_tr)
            p_val = clf_hist.predict_proba(X_val)[:, 1]
            p_ev = clf_hist.predict_proba(x_eval_aug)[:, 1]
            oof_preds["hist"][val_idx] = p_val
            eval_preds["hist"] += p_ev / n_splits
            model_success["hist"] = True
        except Exception:
            pass

        # Model D: Regularized Logistic Regression with Fold-Specific Scaling
        try:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)
            X_ev_s = scaler.transform(x_eval_aug)

            clf_lr = LogisticRegression(
                C=0.20,
                class_weight="balanced",
                max_iter=400,
                random_state=42 + fold,
            )
            clf_lr.fit(X_tr_s, y_tr)
            p_val = clf_lr.predict_proba(X_val_s)[:, 1]
            p_ev = clf_lr.predict_proba(X_ev_s)[:, 1]
            oof_preds["lr"][val_idx] = p_val
            eval_preds["lr"] += p_ev / n_splits
            model_success["lr"] = True
        except Exception:
            pass

        # Model E: ExtraTreesClassifier
        try:
            clf_et = ExtraTreesClassifier(
                n_estimators=70,
                max_depth=5,
                min_samples_leaf=6,
                class_weight="balanced",
                random_state=42 + fold,
                n_jobs=1,
            )
            clf_et.fit(X_tr, y_tr)
            p_val = clf_et.predict_proba(X_val)[:, 1]
            p_ev = clf_et.predict_proba(x_eval_aug)[:, 1]
            oof_preds["et"][val_idx] = p_val
            eval_preds["et"] += p_ev / n_splits
            model_success["et"] = True
        except Exception:
            pass

    # 3. Dynamic Weighting Based on Out-of-Fold Discrimination
    weights = {}
    for m in cv_models:
        if model_success[m]:
            try:
                auc_m = float(roc_auc_score(y_train, oof_preds[m]))
                weights[m] = max(0.05, (auc_m - 0.50) ** 2)
            except Exception:
                weights[m] = 0.15

    total_cv_weight = sum(weights.values())
    if total_cv_weight > 0.0:
        for m in weights:
            weights[m] /= total_cv_weight

    # Standardize individual model logits for robust ensemble combination
    def _std_logits(z_arr: np.ndarray) -> np.ndarray:
        s = float(np.std(z_arr))
        m = float(np.mean(z_arr))
        return (z_arr - m) / (s + 1e-5)

    eval_cv_logits = np.zeros(n_eval, dtype=np.float64)
    oof_cv_logits = np.zeros(n_train, dtype=np.float64)
    has_cv_ensemble = total_cv_weight > 0.0

    if has_cv_ensemble:
        for m, w in weights.items():
            if model_success[m]:
                z_ev_m = _std_logits(_logit(eval_preds[m]))
                z_oof_m = _std_logits(_logit(oof_preds[m]))
                eval_cv_logits += w * z_ev_m
                oof_cv_logits += w * z_oof_m

    # 4. Ensembling RiskBlender Backbone with CV Ensemble
    if rb_probs_eval is not None and has_cv_ensemble:
        z_rb = _std_logits(_logit(rb_probs_eval))
        blended_eval_logits = 0.40 * z_rb + 0.60 * eval_cv_logits
    elif rb_probs_eval is not None:
        blended_eval_logits = _std_logits(_logit(rb_probs_eval))
    elif has_cv_ensemble:
        blended_eval_logits = eval_cv_logits
    else:
        api_eval = np.array([float(getattr(p, "baseline_api_prob", 0.5) or 0.5) for p in eval_records], dtype=np.float64)
        blended_eval_logits = _logit(api_eval)

    # 5. Strictly Monotonic OOF Platt Calibration and Cohort Prevalence Alignment
    # Preserves ROC-AUC, PR-AUC, and Sens@Spec_0.80 exactly while optimizing Brier score
    y_prev = float(np.mean(y_train))
    p_final = None

    if has_cv_ensemble:
        try:
            calibrator = LogisticRegression(C=1.0, max_iter=200, random_state=42)
            calibrator.fit(oof_cv_logits.reshape(-1, 1), y_train)
            slope = float(calibrator.coef_[0][0])
            intercept = float(calibrator.intercept_[0])
            # Constrain slope to positive range to ensure strict monotonicity
            slope = max(0.40, min(2.50, slope))
            calibrated_eval_logits = slope * blended_eval_logits + intercept
            p_initial = _sigmoid(calibrated_eval_logits)
            p_mean = float(np.mean(p_initial))

            if 0.01 < y_prev < 0.99 and 0.01 < p_mean < 0.99:
                shift = float(_logit(np.array([y_prev]))[0] - _logit(np.array([p_mean]))[0])
                sample_shrinkage = min(1.0, n_train / 40.0)
                final_logits = calibrated_eval_logits + (shift * 0.45 * sample_shrinkage)
                p_final = _sigmoid(final_logits)
            else:
                p_final = p_initial
        except Exception:
            p_final = None

    if p_final is None:
        p_raw = _sigmoid(blended_eval_logits)
        p_mean = float(np.mean(p_raw))
        if 0.01 < y_prev < 0.99 and 0.01 < p_mean < 0.99:
            shift = float(_logit(np.array([y_prev]))[0] - _logit(np.array([p_mean]))[0])
            sample_shrinkage = min(1.0, n_train / 40.0)
            p_final = _sigmoid(blended_eval_logits + (shift * sample_shrinkage))
        else:
            p_final = p_raw

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        val = float(p_final[idx])
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, val)), 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Adaptive multi-policy knapsack allocation maximizing expected net savings with 0 violations."""
    candidate_tiers = [
        t for t in cohort_spec.interventions.keys()
        if t != InterventionTier.NONE
    ]

    ref_state = CareAllocationState(cohort_spec, eval_records)
    if not candidate_tiers or not eval_records:
        return ref_state.to_plan()

    # Pre-filter eligible patient-tier pairs and compute expected net benefit
    tier_candidates = []
    patient_best_tiers = {}

    for p in eval_records:
        r = float(predicted_risks.get(p.patient_id, getattr(p, "baseline_api_prob", 0.5) or 0.5))
        r = max(1e-4, min(1.0 - 1e-4, r))
        acuity = float(getattr(p, "acuity_score", 0.0) or 0.0)
        contra = set(getattr(p, "contraindications", []) or [])

        best_nb = 0.0
        best_tier_info = None

        for tier in candidate_tiers:
            spec = cohort_spec.interventions[tier]
            if acuity < spec.min_acuity_score:
                continue
            if contra & set(spec.contraindicated_flags):
                continue

            net_ben = ref_state.expected_net_benefit(p.patient_id, r, tier)
            if net_ben > 0.0:
                cost = max(1.0, float(spec.cost_usd))
                cand = {
                    "pid": p.patient_id,
                    "tier": tier,
                    "net_ben": net_ben,
                    "eff": net_ben / cost,
                    "risk": r,
                }
                tier_candidates.append(cand)
                if net_ben > best_nb:
                    best_nb = net_ben
                    best_tier_info = cand

        if best_tier_info is not None:
            patient_best_tiers[p.patient_id] = best_tier_info

    if not tier_candidates:
        return ref_state.to_plan()

    best_state = None
    best_total_benefit = -1.0

    # Policy Set 1: Multi-exponent benefit-efficiency greedy knapsack exploration
    alphas = [0.0, 0.15, 0.35, 0.55, 0.75, 1.0, 1.25, 1.50, 2.0]
    for alpha in alphas:
        trial_state = CareAllocationState(cohort_spec, eval_records)
        scored = []
        for c in tier_candidates:
            score = c["net_ben"] * (c["eff"] ** alpha)
            scored.append((score, c["pid"], c["tier"], c["net_ben"]))
        scored.sort(key=lambda x: -x[0])

        assigned_pids = set()
        tot_benefit = 0.0
        for _, pid, tier, ben in scored:
            if pid in assigned_pids:
                continue
            if trial_state.fits(pid, tier):
                if trial_state.assign(pid, tier):
                    assigned_pids.add(pid)
                    tot_benefit += ben

        if tot_benefit > best_total_benefit or best_state is None:
            best_total_benefit = tot_benefit
            best_state = trial_state

    # Policy Set 2: Patient-centric best tier priority sorted by risk
    if patient_best_tiers:
        trial_state = CareAllocationState(cohort_spec, eval_records)
        sorted_pats = sorted(patient_best_tiers.values(), key=lambda x: -x["risk"])
        tot_benefit = 0.0
        assigned_pids = set()
        for cand in sorted_pats:
            pid = cand["pid"]
            tier = cand["tier"]
            if trial_state.fits(pid, tier):
                if trial_state.assign(pid, tier):
                    assigned_pids.add(pid)
                    tot_benefit += cand["net_ben"]

        if tot_benefit > best_total_benefit:
            best_total_benefit = tot_benefit
            best_state = trial_state

    plan = best_state.to_plan() if best_state is not None else ref_state.to_plan()

    # Guarantee all eval patients have an assigned tier
    for p in eval_records:
        if p.patient_id not in plan.assignments or plan.assignments[p.patient_id] is None:
            plan.assignments[p.patient_id] = InterventionTier.NONE

    return plan


# EVOLVE-BLOCK-END
# =====================================================================
