# AlphaEvolve Clinical Classifier — Real-Data Tabular & Care-Allocation Portfolio

End-to-end Google Cloud AlphaEvolve repository for evolving clinical risk classifiers in a zero-leakage sandboxed evaluation harness, featuring:

1. **Real-Data Tabular Track ([`problems/diabetes130_readmit30`](problems/diabetes130_readmit30))**: 30-day hospital readmission prediction on the public UCI *Diabetes 130-US Hospitals (1999–2008)* cohort (`99,343` encounters across `~70,000` patients; `11.4%` readmission rate), evaluated with patient-grouped splits, paired bootstrap confidence intervals against tuned GBDT and human-stacked baselines, and permutation leakage checks.
2. **Two-Stage Synthetic Portfolio Harness (`readmission_30d`, `inpatient_admission`, `sepsis_90d`, `chf_30d`)**: Pluggable multi-model harness and constrained care-management allocation simulator (`Stage 1` risk scoring + `Stage 2` budget/slot/contraindication optimization).

---

## 1. Real-Data Benchmark: UCI Diabetes 130-US Hospitals (`diabetes130_readmit30`)

### Evaluation Design
- **Dataset**: UCI *Diabetes 130-US Hospitals for Years 1999–2008* (Strack et al., 2014). After excluding expired and hospice discharge dispositions, `99,343` inpatient encounters remain (`11.4%` readmitted within 30 days).
- **Patient-Grouped Partitioning**: Patients (`patient_nbr`) are hashed (`partition_salt`) into disjoint **60% train / 20% selection / 20% locked test (`n = 19,734`)** partitions so no patient ever crosses splits.
- **Evolution Fitness (`split="all"`)**: $\text{Fitness} = 10{,}000 \times \overline{\text{ROC-AUC}}$ across 3 rounds. Each round draws a fresh `20,000`-row training subsample and scores a disjoint `~6,600`-row selection shard controlled by private `AE_PERTURB_SEED`.
- **Strong Seed & Baselines ([`scripts/tabular_report.py`](scripts/tabular_report.py))**:
  - The seed program ([`problems/diabetes130_readmit30/initial_program.py`](problems/diabetes130_readmit30/initial_program.py)) is **already a tuned LightGBM** (hyper-parameters selected by 48-trial random search on the fitness rounds), so hyperparameter re-tuning alone cannot win.
  - Finalists are compared on the **locked test set (`n = 19,734`)** using 1,000-resample paired bootstrap CIs against:
    1. **Tuned LightGBM** (`= seed`, 48-trial random search)
    2. **Tuned XGBoost** (48-trial random search)
    3. **Human-Engineered Stack** (Charlson ICD-9 comorbidities, ICD-9 chapter groupings, medication steady/up/down counts, total utilization, 5-fold out-of-fold target encoding on high-cardinality codes, and Tuned LightGBM + Tuned XGBoost stacked with a logistic meta-learner).

### Locked-Test Results (3 Independent Replicates × 120 Programs)

Finalist selection rule fixed in advance: `rank1` by selection fitness from each independent AlphaEvolve run ([`artifacts/reports/diabetes130_readmit30_20261002_233459.json`](artifacts/reports/diabetes130_readmit30_20261002_233459.json)).

| Arm | Selection Fitness AUC | Locked-Test ROC-AUC (`n = 19,734`) | $\Delta$ vs Tuned LightGBM [95% CI] | $\Delta$ vs Human Stack [95% CI] | PR-AUC | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned LightGBM (`= seed`)** | 0.6646 | 0.6808 | — | $-0.0026$ $[-0.0056, +0.0004]$ | 0.2337 | 0.0968 |
| **Tuned XGBoost** | 0.6606 | 0.6730 | $-0.0079$ $[-0.0128, -0.0028]$ | $-0.0105$ $[-0.0153, -0.0059]$ | 0.2327 | 0.0969 |
| **Human Stack** | 0.6708 | 0.6835 | $+0.0026$ $[-0.0004, +0.0056]$ | — | 0.2389 | 0.0962 |
| **AlphaEvolve Replicate A `rank1`** | 0.6763 | **0.6903** | **$+0.0095$ $[+0.0055, +0.0133]$** | **$+0.0069$ $[+0.0037, +0.0100]$** | **0.2449** | **0.0957** |
| **AlphaEvolve Replicate B `rank1`** | 0.6745 | **0.6876** | **$+0.0067$ $[+0.0030, +0.0107]$** | **$+0.0041$ $[+0.0009, +0.0074]$** | 0.2414 | 0.2942* |
| **AlphaEvolve Replicate C `rank1`** | 0.6752 | **0.6900** | **$+0.0091$ $[+0.0052, +0.0131]$** | **$+0.0065$ $[+0.0032, +0.0096]$** | **0.2454** | **0.0957** |

*\*Replicate B uses rank-averaged ensembling (optimizing ranking/ROC-AUC without post-hoc probability calibration).*

- **Consistent Lift Across Replicates**: All 3 replicate finalists beat both the tuned LightGBM (`mean Δ = +0.0084`, `SD = 0.0015`) and the human-engineered stack (`mean Δ = +0.0058`, `SD = 0.0015`) with 95% paired bootstrap confidence intervals strictly above zero.
- **Label-Shuffle Leakage Check ([`scripts/leakage_check.py`](scripts/leakage_check.py))**: All finalists pass permutation testing on the fitness rounds (`shuffled AUC = 0.489–0.495`, null SE `0.0037`), confirming zero evaluation-label or row-order leakage.

---

## 2. Repository Structure

```text
src/
  evaluator.py            Sandboxed worker (python -P -s, sys.addaudithook, label-stripped eval pipe)
  tabular_benchmarks.py   Real-data CSV loader, patient-grouped hash splits, and multi-round sampler
  clinical_toolkit.py     Frozen candidate toolkit (TabularEncoder, icd9_group, icd9_comorbidities, oof_target_encode, RiskBlender, CareAllocationState)
  clinical_metrics.py     Exact Mann-Whitney U ROC-AUC, paired bootstrap CI, PR-AUC, Sens@Spec, Brier, Stage 2 ROI
  clinical_benchmarks.py  Synthetic multi-scenario clinical cohort generator
  clinical_models.py      Dataclasses (TabularTask, PatientRecord, CohortSpec, CareAllocationPlan)
  data_adapters.py        CSV/Kaggle & AWS Snowflake cohort adapters + baseline model adapter
problems/
  diabetes130_readmit30/  Real-data UCI 30-day hospital readmission problem (TabularTask contract)
  readmission_30d/        Synthetic two-stage harness problem (30-day readmission + care allocation)
  inpatient_admission/    Synthetic two-stage harness problem (inpatient admission + bed capacity)
  sepsis_90d/             Synthetic two-stage harness problem (sepsis trajectory + intervention triage)
  chf_30d/                Synthetic two-stage harness problem (CHF 30-day readmission)
  _template/              Template used by new_problem.py to scaffold additional problems
scripts/
  fetch_diabetes130.sh    Downloads & SHA-256 verifies UCI Diabetes 130-US Hospitals into data/diabetes130/
  tabular_report.py       Tunes LightGBM/XGBoost baselines, scores locked test, runs paired bootstrap CIs
  leakage_check.py        Refits finalists on permuted training labels to verify AUC collapses to ~0.50
  provision_gcp.sh        Local setup script (enables APIs, generates local .env seeds, provisions engine, fetches data)
artifacts/
  reports/                Locked-test JSON evaluation reports with paired bootstrap CIs
  best_evolved_program.py Hand-written reference candidate for the synthetic two-stage harness track
simulator/                Interactive portfolio & live real-data run diagnostics UI (simulator_ui.html)
tests/                    Behavioral and differential pytest suites
```

---

## 3. Quickstart & Verification

### Fetch the UCI Diabetes 130-US Hospitals Dataset
Downloads the public archive from UCI and verifies the pinned SHA-256 checksum into `data/diabetes130/diabetic_data.csv`:
```bash
bash scripts/fetch_diabetes130.sh
```

### Run the Behavioral Test Suite
```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
```

### Evaluate the Real-Data Seed & Baselines
Score the `diabetes130_readmit30` seed program in the sandbox on the 3 fitness rounds:
```bash
.venv/bin/python src/evaluator.py --program-dir problems/diabetes130_readmit30 --no-perturb
```

Run the label-shuffle leakage check on any candidate or seed program:
```bash
.venv/bin/python scripts/leakage_check.py --problem diabetes130_readmit30 problems/diabetes130_readmit30/initial_program.py
```

Tune the LightGBM + XGBoost baselines, evaluate the human stack, and score finalist candidates on the locked test set with 1,000-resample paired bootstrap CIs:
```bash
.venv/bin/python scripts/tabular_report.py --problem diabetes130_readmit30 --candidates artifacts/runs/diabetes130_readmit30/<exp_id>/rank1.py
```

### Launch the Diagnostic UI
Serves both the live/historical real-data benchmark tables and the two-stage synthetic portfolio simulator:
```bash
GOOGLE_API_USE_CLIENT_CERTIFICATE=false .venv/bin/python simulator/portfolio_simulator.py --serve
```

---

## 4. Local Setup & Running AlphaEvolve

All candidate evaluation runs locally on your machine using your own `gcloud` user credentials:

```bash
# 1. Authenticate with gcloud and Application Default Credentials
gcloud auth login
gcloud auth application-default login

# 2. Enable APIs, generate local private evaluation seeds in .env, create the Gemini Enterprise engine, and fetch data
bash scripts/provision_gcp.sh <YOUR_GCP_PROJECT_ID>

# 3. Launch a live AlphaEvolve run on the real-data UCI problem
uv run python run_experiment.py --problem diabetes130_readmit30
```

---

## 5. Synthetic Two-Stage Portfolio Track (Harness & Care-Allocation Demo)

The four two-stage problems (`readmission_30d`, `inpatient_admission`, `sepsis_90d`, `chf_30d`) test combined `fit_and_score_risk` + `allocate_interventions` evolution under hard budget, slot capacity, acuity, and contraindication constraints.

> **Note on Synthetic vs. Real-Data Lift**: [`artifacts/best_evolved_program.py`](artifacts/best_evolved_program.py) is a hand-written reference candidate whose feature builder (`ClinicalFeatureBuilder(include_interactions=True)`) mirrors the synthetic cohort generator's latent interaction terms. Use the synthetic track to test two-stage constraint plumbing and care-allocation ROI simulation; use [`problems/diabetes130_readmit30`](problems/diabetes130_readmit30) to measure genuine out-of-sample classification lift.

Evaluate the synthetic portfolio scorecard locally:
```bash
.venv/bin/python run_experiment.py --portfolio-report
```

Scaffold a new two-stage problem directory:
```bash
.venv/bin/python new_problem.py oncology_30d --title "Oncology 30-Day Acute Toxicity Admission" --target-auc 0.75
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
