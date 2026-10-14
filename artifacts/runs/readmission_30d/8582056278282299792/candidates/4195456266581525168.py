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


def _extract_multiview_metadata(records: Sequence[PatientRecord]) -> np.ndarray:
    """Extract legacy submodel projections, clinical acuity, and tensor cross-view interactions."""
    rows = []
    for p in records:
        sub = getattr(p, "baseline_submodel_scores", {}) or {}
        cl = float(sub.get("claims_linear", 0.0))
        ev = float(sub.get("ehr_vitals", 0.0))
        ur = float(sub.get("utilization_rules", 0.0))
        acuity = float(getattr(p, "acuity_score", 0.0))
        base_p = float(getattr(p, "baseline_api_prob", 0.5))
        base_logit = safe_logit(base_p)
        contra_count = float(len(getattr(p, "contraindications", [])))

        # Bilinear cross-view discordance and interaction features
        disc_raw = ev - cl
        disc_abs = abs(ev - cl)
        util_acuity = ur * acuity

        rows.append([
            base_p, base_logit, cl, ev, ur, acuity, contra_count,
            disc_raw, disc_abs, util_acuity
        ])
    return np.asarray(rows, dtype=np.float32)


def fit_and_score_risk(
    cohort_spec: CohortSpec,
    train_records: Sequence[PatientRecord],
    eval_records: Sequence[PatientRecord],
) -> Dict[str, float]:
    """Stage 1: Multi-view low-rank latent factorization and calibrated ensemble scoring."""
    # 1. Non-linear interaction feature extraction
    builder = ClinicalFeatureBuilder(include_interactions=True)
    x_train_raw, y_train = builder.fit_transform(cohort_spec, train_records)
    x_eval_raw = builder.transform(cohort_spec, eval_records)

    y_train = np.asarray(y_train, dtype=int)
    if len(np.unique(y_train)) < 2:
        return {p.patient_id: float(p.baseline_api_prob) for p in eval_records}

    # Clean missing or infinite values
    x_train_clean = np.nan_to_num(x_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    x_eval_clean = np.nan_to_num(x_eval_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Robust scaling
    scaler = RobustScaler()
    x_tr_scaled = np.nan_to_num(scaler.fit_transform(x_train_clean), nan=0.0, posinf=3.0, neginf=-3.0)
    x_ev_scaled = np.nan_to_num(scaler.transform(x_eval_clean), nan=0.0, posinf=3.0, neginf=-3.0)

    # 3. Latent Low-Rank Clinical Tensor Subspace (Truncated SVD)
    n_latents = min(8, max(2, x_tr_scaled.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_latents, random_state=42)
    z_train = svd.fit_transform(x_tr_scaled)
    z_eval = svd.transform(x_ev_scaled)

    # Reconstruction error represents atypical clinical syndrome anomalies
    x_tr_recon = svd.inverse_transform(z_train)
    x_ev_recon = svd.inverse_transform(z_eval)
    recon_err_tr = np.linalg.norm(x_tr_scaled - x_tr_recon, axis=1, keepdims=True)
    recon_err_ev = np.linalg.norm(x_ev_scaled - x_ev_recon, axis=1, keepdims=True)

    # 4. Multi-view submodel extraction and bilinear tensor interactions
    meta_tr = _extract_multiview_metadata(train_records)
    meta_ev = _extract_multiview_metadata(eval_records)

    cross_tr_cols = [z_train[:, 0:1] * meta_tr[:, 1:2]]  # Frailty x baseline logit
    cross_ev_cols = [z_eval[:, 0:1] * meta_ev[:, 1:2]]
    if n_latents >= 2:
        cross_tr_cols.append(z_train[:, 1:2] * meta_tr[:, 3:4])  # Acute stress x EHR vitals
        cross_ev_cols.append(z_eval[:, 1:2] * meta_ev[:, 3:4])
    cross_tr = np.column_stack(cross_tr_cols)
    cross_ev = np.column_stack(cross_ev_cols)

    # Augmented multi-view feature representations
    x_train_aug = np.column_stack([x_tr_scaled, z_train, recon_err_tr, meta_tr, cross_tr])
    x_eval_aug = np.column_stack([x_ev_scaled, z_eval, recon_err_ev, meta_ev, cross_ev])

    # 5. Fit Calibrated Multi-Model Ensemble
    predictions = []
    weights = []

    # Model A: LightGBM Classifier
    try:
        import lightgbm as lgb
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=130,
            learning_rate=0.035,
            max_depth=4,
            num_leaves=15,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=2,
        )
        lgb_clf.fit(x_train_aug, y_train)
        predictions.append((lgb_clf.predict_proba(x_eval_aug)[:, 1], 0.40))
    except Exception:
        pass

    # Model B: XGBoost Classifier
    try:
        import xgboost as xgb
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.035,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
            n_jobs=2,
        )
        xgb_clf.fit(x_train_aug, y_train)
        predictions.append((xgb_clf.predict_proba(x_eval_aug)[:, 1], 0.30))
    except Exception:
        pass

    # Model C: Calibrated Regularized Logistic Regression (Platt scaling)
    try:
        std_scaler = StandardScaler()
        x_tr_std = std_scaler.fit_transform(x_train_aug)
        x_ev_std = std_scaler.transform(x_eval_aug)

        lr_base = LogisticRegression(C=0.5, penalty="l2", max_iter=300, random_state=42)
        try:
            calibrated_lr = CalibratedClassifierCV(lr_base, method="sigmoid", cv=3)
            calibrated_lr.fit(x_tr_std, y_train)
            lr_prob = calibrated_lr.predict_proba(x_ev_std)[:, 1]
        except Exception:
            lr_base.fit(x_tr_std, y_train)
            lr_prob = lr_base.predict_proba(x_ev_std)[:, 1]
        predictions.append((lr_prob, 0.20))
    except Exception:
        pass

    # 6. Ensemble Logit Blending with Legacy Prior
    prior_weight = 0.10
    total_model_weight = sum(w for _, w in predictions) if predictions else 0.0

    out: Dict[str, float] = {}
    for idx, p in enumerate(eval_records):
        base_p = float(getattr(p, "baseline_api_prob", 0.5))
        base_logit = safe_logit(base_p)

        if predictions and total_model_weight > 0.0:
            model_logit = sum(w * safe_logit(float(probs[idx])) for probs, w in predictions) / total_model_weight
            blended_logit = (1.0 - prior_weight) * model_logit + prior_weight * base_logit
            final_prob = safe_sigmoid(blended_logit)
        else:
            final_prob = base_p

        out[p.patient_id] = round(max(1e-4, min(1.0 - 1e-4, float(final_prob))), 6)

    return out


def allocate_interventions(
    cohort_spec: CohortSpec,
    eval_records: Sequence[PatientRecord],
    predicted_risks: Dict[str, float],
) -> CareAllocationPlan:
    """Stage 2: Knapsack Care Allocation prioritizing maximum net benefit and ROI."""
    state = CareAllocationState(cohort_spec, eval_records)
    active_tiers = [t for t in cohort_spec.interventions.keys() if t != InterventionTier.NONE]

    # Evaluate all eligible (patient, tier) candidates with positive expected net benefit
    candidates = []
    for p in eval_records:
        risk = float(predicted_risks.get(p.patient_id, p.baseline_api_prob))
        for tier in active_tiers:
            if state.fits(p.patient_id, tier):
                benefit = state.expected_net_benefit(p.patient_id, risk, tier)
                if benefit > 0.0:
                    cost = cohort_spec.interventions[tier].cost_usd
                    roi = benefit / max(1.0, cost)
                    candidates.append((p.patient_id, tier, benefit, roi))

    # Prioritize assignments offering maximal net benefit and cost-efficiency
    candidates.sort(key=lambda item: (-item[2], -item[3]))

    assigned_patients = set()
    for patient_id, tier, _, _ in candidates:
        if patient_id in assigned_patients:
            continue
        if state.fits(patient_id, tier):
            if state.assign(patient_id, tier):
                assigned_patients.add(patient_id)

    return state.to_plan()


# EVOLVE-BLOCK-END
# =====================================================================
