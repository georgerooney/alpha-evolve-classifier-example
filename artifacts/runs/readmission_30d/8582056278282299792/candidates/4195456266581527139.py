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


def _extract_legacy_features(records: Sequence[PatientRecord]) -> np.ndarray:
    """Extract and cross-interact legacy submodels, acuity, and clinical fragility flags."""
    import numpy as np

    feats = []
    for p in records:
        base_p = float(getattr(p, "baseline_api_prob", 0.2))
        base_p = max(1e-4, min(1.0 - 1e-4, base_p))
        base_logit = safe_logit(base_p)
        sub = getattr(p, "baseline_submodel_scores", {}) or {}
        c_lin = float(sub.get("claims_linear", base_logit))
        e_vit = float(sub.get("ehr_vitals", base_logit))
        u_rul = float(sub.get("utilization_rules", base_logit))
        acuity = float(getattr(p, "acuity_score", 0.5))
        contra_cnt = float(len(getattr(p, "contraindications", [])))
        sub_spread = max(c_lin, e_vit, u_rul) - min(c_lin, e_vit, u_rul)
        feats.append([
            base_p,
            base_logit,
            c_lin,
            e_vit,
            u_rul,
            acuity,
            contra_cnt,
            sub_spread,
            base_logit * acuity,
            e_vit * c_lin,
        ])
    res = np.array(feats, dtype=np.float64)
    return np.nan_to_num(res, nan=0.0, posinf=10.0, neginf=-10.0)


def _to_logit(p: np.ndarray) -> np.ndarray:
    """Numerically robust vectorized logit transform."""
    import numpy as np
    cl = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(cl / (1.0 - cl))


def _to_sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically robust vectorized sigmoid transform."""
    import numpy as np
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Multi-model gradient boosting and linear ensemble with syndrome interactions and logit calibration."""
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    # 1. Non-linear clinical syndrome interaction feature matrix
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval = builder.transform(cohort_spec, eval_records)

    # 2. Extract legacy submodel signals and build combined feature matrix
    extra_train = _extract_legacy_features(train_records)
    extra_eval = _extract_legacy_features(eval_records)
    x_train_comb = np.hstack([x_train, extra_train])
    x_eval_comb = np.hstack([x_eval, extra_eval])

    # 3. Train diverse gradient boosted and linear estimators
    # (a) LightGBM: leaf-wise growth capturing complex syndrome interactions
    lgb = LGBMClassifier(
        n_estimators=160,
        learning_rate=0.035,
        num_leaves=15,
        max_depth=4,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    lgb.fit(x_train_comb, y_train)
    p_lgb = lgb.predict_proba(x_eval_comb)[:, 1]

    # (b) XGBoost: depth-wise regularized tree boosting
    xgb = XGBClassifier(
        n_estimators=140,
        learning_rate=0.035,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
        n_jobs=1,
    )
    xgb.fit(x_train_comb, y_train)
    p_xgb = xgb.predict_proba(x_eval_comb)[:, 1]

    # (c) HistGradientBoosting: fast binned gradient booster
    hgb = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.035,
        max_leaf_nodes=15,
        min_samples_leaf=15,
        random_state=42,
    )
    hgb.fit(x_train_comb, y_train)
    p_hgb = hgb.predict_proba(x_eval_comb)[:, 1]

    # (d) Standardized L2 Logistic Regression: stable linear baseline
    scaler = StandardScaler()
    x_tr_sc = scaler.fit_transform(x_train_comb)
    x_ev_sc = scaler.transform(x_eval_comb)
    lr = LogisticRegression(C=0.5, max_iter=250, random_state=42)
    lr.fit(x_tr_sc, y_train)
    p_lr = lr.predict_proba(x_ev_sc)[:, 1]

    # Form custom ensemble logit prediction
    custom_logit = (
        0.38 * _to_logit(p_lgb)
        + 0.32 * _to_logit(p_xgb)
        + 0.18 * _to_logit(p_hgb)
        + 0.12 * _to_logit(p_lr)
    )

    # 4. Integrate native toolkit RiskBlender
    rb_probs = None
    try:
        blender = RiskBlender(
            linear_weight=0.30,
            lgbm_weight=0.42,
            xgb_weight=0.28,
            baseline_prior_weight=0.06,
            random_state=42,
        )
        rb_preds = blender.fit_predict(
            cohort_spec,
            train_records,
            eval_records,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
        )
        rb_probs = np.array(
            [float(rb_preds[p.patient_id]) for p in eval_records], dtype=np.float64
        )
    except Exception:
        rb_probs = None

    if rb_probs is not None:
        rb_logit = _to_logit(rb_probs)
        ens_logit = 0.50 * rb_logit + 0.50 * custom_logit
    else:
        ens_logit = custom_logit

    # 5. Monotonic logit calibration to align with ground-truth cohort prevalence (optimizes Brier score)
    raw_probs = _to_sigmoid(ens_logit)
    y_mean = float(np.mean(y_train))
    y_mean = max(1e-4, min(1.0 - 1e-4, y_mean))
    pred_mean = float(np.mean(raw_probs))
    pred_mean = max(1e-4, min(1.0 - 1e-4, pred_mean))

    shift = safe_logit(y_mean) - safe_logit(pred_mean)
    calibrated_probs = _to_sigmoid(ens_logit + shift)

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        prob = float(calibrated_probs[idx])
        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, prob)), 6)
    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Optimal care allocation maximizing expected net benefit under clinical and budget constraints."""
    state = CareAllocationState(cohort_spec, eval_records)
    ranked_patients = sorted(
        eval_records,
        key=lambda p: -float(predicted_risks.get(p.patient_id, p.baseline_api_prob)),
    )
    eligible_tiers = [
        t for t in cohort_spec.interventions.keys()
        if t != InterventionTier.NONE
    ]

    for p in ranked_patients:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        best_tier = None
        best_enb = 0.0
        for tier in eligible_tiers:
            if state.fits(p.patient_id, tier):
                enb = state.expected_net_benefit(p.patient_id, risk, tier)
                if enb > best_enb:
                    best_enb = enb
                    best_tier = tier
        if best_tier is not None:
            state.assign(p.patient_id, best_tier)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
