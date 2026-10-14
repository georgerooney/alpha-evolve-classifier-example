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
    """Stage 1: Decoupled Rank-Preserving Multi-Model Stacking & Strictly Monotonic Brier Calibration."""
    import numpy as np
    from scipy.optimize import minimize
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
    from sklearn.ensemble import HistGradientBoostingClassifier
    import lightgbm as lgb
    import xgboost as xgb

    def _extract_engineered_features(records):
        feats = []
        contra_spec_set = set(cohort_spec.contraindicated_flags or []) if hasattr(cohort_spec, 'contraindicated_flags') else set()
        for p in records:
            b_prob = float(p.baseline_api_prob) if hasattr(p, 'baseline_api_prob') and p.baseline_api_prob is not None else 0.5
            b_prob = float(np.clip(b_prob, 1e-4, 1.0 - 1e-4))
            b_logit = float(np.log(b_prob / (1.0 - b_prob)))

            submodels = getattr(p, 'baseline_submodel_scores', {}) or {}
            c_lin = float(submodels.get('claims_linear', b_prob))
            e_vit = float(submodels.get('ehr_vitals', b_prob))
            u_rul = float(submodels.get('utilization_rules', b_prob))

            c_lin = float(np.clip(c_lin, 1e-4, 1.0 - 1e-4))
            e_vit = float(np.clip(e_vit, 1e-4, 1.0 - 1e-4))
            u_rul = float(np.clip(u_rul, 1e-4, 1.0 - 1e-4))

            c_logit = float(np.log(c_lin / (1.0 - c_lin)))
            e_logit = float(np.log(e_vit / (1.0 - e_vit)))
            u_logit = float(np.log(u_rul / (1.0 - u_rul)))

            acuity = float(getattr(p, 'acuity_score', 0.0) or 0.0)

            diff_vit_claims = e_vit - c_lin
            diff_rul_claims = u_rul - c_lin
            diff_vit_rul = e_vit - u_rul

            scores = [c_lin, e_vit, u_rul]
            s_max = max(scores)
            s_min = min(scores)
            s_std = float(np.std(scores))
            s_mean = float(np.mean(scores))

            p_contra = set(getattr(p, 'contraindications', []) or [])
            contra_count = float(len(p_contra))
            contra_spec_match = float(len(p_contra & contra_spec_set))

            feats.append([
                b_prob, b_logit,
                c_lin, e_vit, u_rul,
                c_logit, e_logit, u_logit,
                acuity,
                diff_vit_claims, diff_rul_claims, diff_vit_rul,
                s_max, s_min, s_std, s_mean,
                contra_count, contra_spec_match,
                b_prob * acuity,
                e_vit * acuity,
                diff_vit_claims * acuity,
                s_max - s_min
            ])
        return np.array(feats, dtype=np.float32)

    try:
        builder = ClinicalFeatureBuilder(include_interactions=True)
        x_tr_raw, y_tr_raw = builder.fit_transform(cohort_spec, train_records)
        x_ev_raw = builder.transform(cohort_spec, eval_records)

        x_tr_raw = np.nan_to_num(x_tr_raw, nan=0.0, posinf=1e4, neginf=-1e4)
        x_ev_raw = np.nan_to_num(x_ev_raw, nan=0.0, posinf=1e4, neginf=-1e4)

        extra_tr = _extract_engineered_features(train_records)
        extra_ev = _extract_engineered_features(eval_records)

        x_train = np.hstack([x_tr_raw, extra_tr])
        x_eval = np.hstack([x_ev_raw, extra_ev])
        y_train = np.array(y_tr_raw, dtype=np.float32)

        base_train = np.array([float(np.clip(p.baseline_api_prob, 1e-4, 1.0 - 1e-4)) for p in train_records], dtype=np.float32)
        base_eval = np.array([float(np.clip(p.baseline_api_prob, 1e-4, 1.0 - 1e-4)) for p in eval_records], dtype=np.float32)

        unique_labels = np.unique(y_train)
        if len(unique_labels) < 2:
            return {p.patient_id: round(float(p.baseline_api_prob), 6) for p in eval_records}

        min_class_count = int(np.min(np.bincount(y_train.astype(int))))
        n_splits = max(2, min(5, min_class_count))

        model_keys = ['lgb1', 'lgb2', 'xgb', 'hgb', 'lr', 'base']
        M = len(model_keys)
        N_train = len(train_records)
        N_eval = len(eval_records)

        oof_preds = {k: np.zeros(N_train, dtype=np.float32) for k in model_keys}
        eval_preds = {k: np.zeros(N_eval, dtype=np.float32) for k in model_keys}

        oof_preds['base'] = base_train.copy()
        eval_preds['base'] = base_eval.copy()

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for tr_idx, val_idx in skf.split(x_train, y_train.astype(int)):
            x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
            x_val = x_train[val_idx]

            # Logistic Regression on standardized features
            try:
                scaler = StandardScaler()
                x_tr_sc = scaler.fit_transform(x_tr)
                x_val_sc = scaler.transform(x_val)
                x_ev_sc = scaler.transform(x_eval)
                lr = LogisticRegression(C=0.15, penalty='l2', solver='lbfgs', max_iter=300, random_state=42)
                lr.fit(x_tr_sc, y_tr)
                oof_preds['lr'][val_idx] = lr.predict_proba(x_val_sc)[:, 1]
                eval_preds['lr'] += lr.predict_proba(x_ev_sc)[:, 1] / n_splits
            except Exception:
                oof_preds['lr'][val_idx] = base_train[val_idx]
                eval_preds['lr'] += base_eval / n_splits

            # Gradient Boosted Trees
            fold_models = {
                'lgb1': lgb.LGBMClassifier(
                    n_estimators=110, learning_rate=0.035, max_depth=4, num_leaves=15,
                    min_child_samples=15, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=2
                ),
                'lgb2': lgb.LGBMClassifier(
                    n_estimators=80, learning_rate=0.05, max_depth=3, num_leaves=7,
                    min_child_samples=20, subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=0.5, reg_lambda=2.0, random_state=123, verbose=-1, n_jobs=2
                ),
                'xgb': xgb.XGBClassifier(
                    n_estimators=90, learning_rate=0.04, max_depth=3, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=0.3, reg_lambda=1.5,
                    eval_metric='logloss', random_state=42, n_jobs=2
                ),
                'hgb': HistGradientBoostingClassifier(
                    max_iter=80, learning_rate=0.04, max_depth=4, l2_regularization=1.5,
                    min_samples_leaf=15, random_state=42
                )
            }

            for k, model in fold_models.items():
                try:
                    model.fit(x_tr, y_tr)
                    oof_preds[k][val_idx] = model.predict_proba(x_val)[:, 1]
                    eval_preds[k] += model.predict_proba(x_eval)[:, 1] / n_splits
                except Exception:
                    oof_preds[k][val_idx] = base_train[val_idx]
                    eval_preds[k] += base_eval / n_splits

        def _to_logit(p_arr):
            p_cl = np.clip(p_arr, 1e-4, 1.0 - 1e-4)
            return np.log(p_cl / (1.0 - p_cl))

        oof_logits = np.column_stack([_to_logit(oof_preds[k]) for k in model_keys])
        eval_logits = np.column_stack([_to_logit(eval_preds[k]) for k in model_keys])

        def _rank_fitness(y_true, scores):
            try:
                auc = roc_auc_score(y_true, scores)
            except Exception:
                auc = 0.5
            try:
                pr_auc = average_precision_score(y_true, scores)
            except Exception:
                pr_auc = 0.0
            try:
                fpr, tpr, _ = roc_curve(y_true, scores)
                valid_tpr = tpr[fpr <= 0.200001]
                sens80 = float(np.max(valid_tpr)) if len(valid_tpr) > 0 else 0.0
            except Exception:
                sens80 = 0.0
            return 0.50 * auc + 0.25 * pr_auc + 0.15 * sens80

        # Model weight search directly maximizing rank-based clinical fitness
        indiv_scores = [_rank_fitness(y_train, oof_logits[:, i]) for i in range(M)]
        best_idx = int(np.argmax(indiv_scores))
        best_w = np.zeros(M, dtype=np.float32)
        best_w[best_idx] = 1.0
        best_score = indiv_scores[best_idx]

        s_exp = np.exp((np.array(indiv_scores) - np.max(indiv_scores)) * 8.0)
        soft_w = s_exp / np.sum(s_exp)
        soft_sc = _rank_fitness(y_train, oof_logits @ soft_w)
        if soft_sc > best_score:
            best_score = soft_sc
            best_w = soft_w.copy()

        rng = np.random.RandomState(42)
        for _ in range(40):
            cand_w = rng.dirichlet(best_w * 8.0 + 0.5)
            cand_sc = _rank_fitness(y_train, oof_logits @ cand_w)
            if cand_sc > best_score:
                best_score = cand_sc
                best_w = cand_w.copy()

        def _obj(theta):
            exp_t = np.exp(theta - np.max(theta))
            w = exp_t / np.sum(exp_t)
            return -_rank_fitness(y_train, oof_logits @ w)

        theta_init = np.log(np.clip(best_w, 1e-3, 1.0))
        res = minimize(_obj, theta_init, method='Nelder-Mead', options={'maxiter': 50, 'disp': False})
        if res.fun < -best_score:
            exp_res = np.exp(res.x - np.max(res.x))
            final_w = exp_res / np.sum(exp_res)
        else:
            final_w = best_w

        blended_oof = oof_logits @ final_w
        blended_eval = eval_logits @ final_w

        # Stage 1b: Strictly monotonic probability calibration for optimal Brier score
        def _brier_loss(params):
            alpha, b = params
            a = np.exp(alpha)
            p = 1.0 / (1.0 + np.exp(-(a * blended_oof + b)))
            return np.mean((p - y_train) ** 2)

        cal_res = minimize(_brier_loss, [0.0, 0.0], method='L-BFGS-B', bounds=[(-2.5, 2.5), (-8.0, 8.0)])
        opt_a = float(np.exp(cal_res.x[0]))
        opt_b = float(cal_res.x[1])

        calibrated_eval_probs = 1.0 / (1.0 + np.exp(-(opt_a * blended_eval + opt_b)))
        calibrated_eval_probs = np.clip(calibrated_eval_probs, 1e-4, 1.0 - 1e-4)

        raw_preds = {p.patient_id: float(calibrated_eval_probs[idx]) for idx, p in enumerate(eval_records)}

    except Exception:
        builder = ClinicalFeatureBuilder(include_interactions=False)
        x_tr_raw, y_tr_raw = builder.fit_transform(cohort_spec, train_records)
        x_ev_raw = builder.transform(cohort_spec, eval_records)
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_tr_raw)
        x_ev = scaler.transform(x_ev_raw)
        clf = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        clf.fit(x_tr, y_tr_raw)
        lr_probs = clf.predict_proba(x_ev)[:, 1]
        raw_preds = {}
        for idx, p in enumerate(eval_records):
            blended = 0.50 * safe_logit(float(lr_probs[idx])) + 0.50 * safe_logit(float(p.baseline_api_prob))
            raw_preds[p.patient_id] = safe_sigmoid(blended)

    out: Dict[str, float] = {}
    for p in eval_records:
        prob = raw_preds.get(p.patient_id, p.baseline_api_prob)
        if not np.isfinite(prob):
            prob = float(p.baseline_api_prob)
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, float(prob))), 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: First-fit highest-risk patients into the highest-efficacy feasible intervention tier."""
    state = CareAllocationState(cohort_spec, eval_records)
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
            if state.fits(p.patient_id, tier) and state.expected_net_benefit(p.patient_id, risk, tier) > 0.0:
                state.assign(p.patient_id, tier)
                break

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
