# AlphaEvolve Clinical Classifier — Current Status & Next Steps

## 1. What Is Built & Verified
- **Shared Core Engine (`src/`)**:
  - [`src/clinical_models.py`](src/clinical_models.py): `PatientRecord`, `CohortSpec`, `InterventionSpec`, `InterventionTier`, `CareAllocationPlan`, `ClinicalBenchmarkInstance`.
  - [`src/clinical_metrics.py`](src/clinical_metrics.py): Exact Mann-Whitney U `ROC-AUC`, `PR-AUC`, `Sensitivity@Specificity`, `Brier` calibration, and Stage 2 care-management financial ROI + CMS HRRP penalty calculation.
  - [`src/clinical_toolkit.py`](src/clinical_toolkit.py): Frozen candidate toolkit (`ClinicalFeatureBuilder` with submodel disagreement features, `compute_urgency_sample_weights`, 4-model Stratified K-Fold OOF `RiskBlender` combining L2 Logistic + LightGBM + XGBoost + GradientBoosting + meta-calibrator, and constraint-safe `CareAllocationState` with atomic `assign_all`).
  - [`src/evaluator.py`](src/evaluator.py): Sandboxed two-phase subprocess worker (`python -P -s`, `PYTHONHASHSEED=0`, `sys.addaudithook`), stripping ground-truth outcomes from `eval_records` before they cross the pipe so label leakage is impossible.
  - [`src/data_adapters.py`](src/data_adapters.py): Drop-in CSV/Kaggle loader (`CSVDatasetAdapter`), AWS Snowflake cohort adapter (`SnowflakeDatasetAdapter`, `SnowflakeConfig`), and legacy baseline model API adapter (`SimulatedVertexModelAdapter`).
- **Split-Cloud AWS Snowflake × GCP AlphaEvolve Architecture (`docs/`)**:
  - [`docs/aws_snowflake_split_cloud_architecture.svg`](docs/aws_snowflake_split_cloud_architecture.svg): Architecture diagram showing AWS ECS/EKS evaluation workers + Snowflake PHI perimeter communicating with GCP AlphaEvolve via Workload Identity Federation (`sts.googleapis.com`).
  - [`docs/AWS_SNOWFLAKE_MIGRATION.md`](docs/AWS_SNOWFLAKE_MIGRATION.md): Step-by-step migration guide.
- **4 Synthetic Clinical Portfolio Problems (`problems/`)**. These are a harness/demo track, not a performance claim:
  - `problems/readmission_30d`, `problems/inpatient_admission`, `problems/sepsis_90d`, `problems/chf_30d`.
  - Plus `problems/_template` and `new_problem.py` for rapid onboarding of additional models.
  - The AUC numbers previously quoted here (">0.78", ">0.88", ...) came from the **hand-written**
    `artifacts/best_evolved_program.py`, not from an AlphaEvolve run.
  - That reference program scores within about 0.01 AUC of the generator's oracle, because
    `ClinicalFeatureBuilder(include_interactions=True)` reproduces the generator's latent log-odds terms. It
    demonstrates the plumbing, not AlphaEvolve's discovery ability.
  - Synthetic prevalence (35–69%) and n_eval≈170 (±0.08 AUC CI) also make it unsuitable for measuring lift.
- **Stakeholder & Engineering Diagnostic Simulator (`simulator/`)**:
  - `simulator/portfolio_simulator.py` and `simulator/simulator_ui.html`. The UI includes:
    - a candidate table with status filters (`GRADED`/`HARD_VIOLATION`/`RUNTIME_ERROR`/`PENDING`);
    - diff summaries vs. `initial_program.py`;
    - a full `.py` source inspector.
  - **Real-data card** (`/api/real_data`, `/api/live_run`):
    - live progress and a best-so-far fitness chart, polled every 15 s from the AE API with a 30 s server cache;
    - seed vs. tuned LightGBM vs. AE best;
    - the locked-test table with paired-bootstrap CIs, read from the latest `artifacts/reports/<pid>_*.json`.
  - **How a live run is found:** `run_experiment.py` writes `artifacts/runs/<pid>/<exp_id>/experiment.json` at start. Once `candidates.json` exists, the UI switches to it.
  - **What is never fabricated:**
    - Real-data problems never get a fabricated trace.
    - The synthetic 48-candidate trace is labelled "SYNTHETIC reference trace".
    - The UI has no hardcoded or offline fallback data.
  - Serve: `GOOGLE_API_USE_CLIENT_CERTIFICATE=false .venv/bin/python simulator/portfolio_simulator.py --serve`.
    - Why the env var: on this workstation the ECP cert provider segfaults (`TransportError: Cert provider command returns non-zero status code -11`) for newly started processes.
    - The read-only list call works fine over plain TLS with ADC.

## 2. Real-Data Track: `problems/diabetes130_readmit30` (use this to measure AlphaEvolve)
- **Data:** UCI *Diabetes 130-US Hospitals 1999–2008*. After dropping expired/hospice dispositions, that leaves
  99,343 encounters from about 70k patients, with 11.4% readmitted within 30 days.
  - Fetch with `scripts/fetch_diabetes130.sh`, which pins the checksum. The data lands in `data/`, which is
    gitignored.
- **Splits:** patients are hashed (`partition_salt`) into train 60% / selection 20% / **locked test** 20%.
  - Fitness = `10,000 × mean ROC-AUC` over 3 rounds. Each round trains on a fresh 20k-row draw of train patients and
    scores a disjoint selection shard of about 6.6k rows.
  - The draw is controlled by `AE_PERTURB_SEED`.
  - `--split test` trains on all train patients and scores the locked test. It is for finalists only.
- **Code:**
  - [`src/tabular_benchmarks.py`](src/tabular_benchmarks.py) handles loading and splits. It is generic: any CSV
    classification set needs only a `problem_config.json`.
  - [`src/evaluator.py`](src/evaluator.py) has the `tabular_round` sandbox protocol and dispatches on `task_type`.
  - [`src/clinical_toolkit.py`](src/clinical_toolkit.py) adds `TabularEncoder` and `icd9_group`, which are generic
    only.
  - [`scripts/tabular_report.py`](scripts/tabular_report.py) provides the tuned baseline, the locked-test scoring and
    the paired bootstrap.
- **Baseline numbers** (commit after `0140b3d`, public default seeds, 24-trial random search):

  | Arm | Fitness (selection) mean AUC | Locked-test AUC | Δ vs tuned [95% CI] |
  |---|---|---|---|
  | Seed (`initial_program.py`, untuned LightGBM) | 0.629 (round SD 0.004) | 0.652 | −0.028 [−0.034, −0.020] |
  | Tuned LightGBM (random search on fitness rounds) | 0.664 | **0.679** | — |

  - **The bar for AlphaEvolve is the tuned LightGBM's 0.679 on the locked test**, not the seed.
  - Round-to-round SD is about 0.004, so fitness differences under about 0.005 AUC are noise.
  - One sandboxed fitness evaluation takes about 13 s wall time.

- **Live run 1 (2026-10-02)**
  - **Setup:**
    - experiment `8582056278282301190`, run from commit `afe9fa3`;
    - 79/80 evaluated over about 2.5 h with 1 evaluator;
    - private seeds;
    - report `artifacts/reports/diabetes130_readmit30_20261002_151026.json`.

  | Arm | Fitness AUC | Locked-test AUC (n=19,734) | Δ vs tuned LightGBM [95% CI] | PR-AUC | Brier |
  |---|---|---|---|---|---|
  | Tuned LightGBM (24 trials, private-seed fitness rounds) | 0.664 | 0.6793 | — | 0.233 | 0.0967 |
  | Seed | 0.633 | 0.6518 | −0.0275 [−0.0344, −0.0201] | 0.212 | 0.0983 |
  | AE rank1 | 0.676 | 0.6871 | **+0.0077 [+0.0035, +0.0121]** | 0.243 | 0.0959 |
  | AE rank2 | 0.675 | 0.6895 | **+0.0102 [+0.0061, +0.0143]** | 0.245 | 0.0958 |
  | AE rank3 | 0.674 | 0.6876 | **+0.0083 [+0.0044, +0.0127]** | 0.243 | 0.0959 |

  - **What evolved** (all 3 finalists converged on it):
    - ICD-9 comorbidity features;
    - 5-fold out-of-fold target encoding;
    - a 5-fold stacked ensemble of 4 LightGBM models (one DART) and 2 XGBoost models, with a logistic meta-learner.
  - This is strong, standard tabular practice, which AE rediscovered. The fair next bar is therefore a human-built
    stack or AutoML (e.g. AutoGluon), not a single tuned GBDT.
  - **Caveats:**
    - **Cost:** finalists take about 90–105 s per round on 20k rows, against a 120 s timeout and about 3 s for the
      seed. The timeout is effectively the complexity cap.
    - **Why `report.json` has `heldout: null` for the finalists:** `split=test` trains on all 59k train rows, so they
      exceed 120 s. `tabular_report.py` scores in-process with no timeout.
    - **Fairness:** subgroup AUC gaps are large and unchanged from the seed: age 90–100 ≈ 0.53–0.54, and a race gap of
      about 0.10–0.11.
    - **Wasted candidates:**
      - 7 of 85 programs died on `Sandbox blocked file open: __init__.py`. These were lazy sklearn/scipy submodule
        imports (`sklearn.decomposition`, `.cluster`, `.feature_extraction.text`, `scipy.sparse`) that are not
        pre-imported before the audit hook.
      - 2 programs hit the timeout.
      - 9 programs had their own bugs.
    - The locked test has now been used once for these finalists. Do not iterate against it.
  - **Fixed after run 1:**
    - **Sandbox imports:** the sandbox now allows read-only opens of `.py`/`.pyc`/`.so` files under the stdlib and
      site-packages.
      - Paths are resolved with `realpath`, so `..` and symlinks can't escape to the repo.
      - Repo sources, `.env`, data and system files stay blocked, and tests pin this.
      - Result: 5 of the 7 blocked run-1 candidates now grade (0.638–0.650). One has its own bug and one times out.
    - **New leakage check:** `scripts/leakage_check.py` refits a candidate on permuted `y_train` over the fitness rounds
      and passes if the AUC is within 4 null standard errors of 0.5. It never touches the locked test.
      - Run-1 finalists all **PASS**: true AUC about 0.673, and shuffled AUC 0.4895 / 0.4902 / 0.4896 (null SE 0.0037).
      - The shuffled AUC is slightly *below* 0.5 (z ≈ −2.7), not above it. That is the opposite of leakage, which would
        push it above 0.5. A plausible cause is out-of-fold target encoding: on noise labels, a row's encoding is
        anti-correlated with its own label, and the stacker learns from that. Worth a look if it grows.
  - **Next run:**
    - `.env` now sets `PARALLEL_EVALUATION=True`, `WORKER_CONCURRENCY=6` and `CONCURRENCY=8`.
    - Benchmark on 24 cores: 4 parallel evaluations gave 2.9× the throughput with bit-identical scores.

- **Live run 2 (2026-10-02)**
  - **Setup:**
    - experiment `8703075698965655118`, run from commit `f39a42b` (sandbox fix and leakage check in place);
    - `PARALLEL_EVALUATION=True` with 6 workers;
    - 79/80 evaluated in **about 40 min** (run 1 took about 2.5 h);
    - report `artifacts/reports/diabetes130_readmit30_20261002_174539.json`.

  | Arm | Fitness AUC | Locked-test AUC | Δ vs tuned LightGBM [95% CI] | PR-AUC | Brier |
  |---|---|---|---|---|---|
  | Tuned LightGBM | 0.664 | 0.6793 | — | 0.233 | 0.0967 |
  | Run-2 rank1 | 0.667 | 0.6818 | +0.0025 [−0.0028, +0.0085] | 0.242 | 0.0961 |
  | Run-2 rank2 | 0.667 | 0.6821 | +0.0028 [−0.0029, +0.0087] | 0.242 | 0.0961 |
  | Run-2 rank3 | 0.666 | 0.6813 | +0.0019 [−0.0034, +0.0077] | 0.239 | 0.0962 |
  | Run-1 rank2 (re-scored for reference) | 0.675 | 0.6895 | +0.0102 [+0.0061, +0.0143] | 0.245 | 0.0958 |

  - **Run 2 is not significantly better than the tuned baseline.** All three CIs cross 0.
  - **What evolved:**
    - ICD-9 chapter grouping;
    - shallower, more regularised LightGBM trees;
    - averaging over 3 seeds.

    The programs are about 68 lines (run 1's were about 765) and take 17–24 s per full evaluation (run 1's took
    about 300 s). Run 2 found a cheap tuned-GBDT variant and never reached the stacking/target-encoding basin.
  - **Hypothesis, not established (n = 1 run each):**
    - With 6 evaluators and `CONCURRENCY=8`, many children are generated before their parents are scored.
    - That makes the search broader and shallower per wall-clock minute.
    - Next time, run ≥ 2 seeds per configuration, or a larger budget (≥ 200 programs), before concluding that
      parallelism hurts quality.
  - **Health checks:**
    - Leakage check: all 3 **PASS** (shuffled AUC about 0.486, null SE 0.0037).
    - Errors: 0 sandbox import blocks, and 5 candidate bugs (dtype errors).
    - The finalists now have `heldout` scores, because they fit in the timeout.
  - **Fairness is unchanged:** age 90–100 AUC is about 0.49–0.50, and the race gap is about 0.12.
  - **Bug fixed:** `tabular_report.py` keyed arms by file stem, so passing `run1/rank2.py` and `run2/rank2.py`
    together silently dropped one of them. Colliding stems are now prefixed with their run directory, and a test
    pins this.

- **Problem v2 (2026-10-02): stronger baselines and a tuned seed**
  - **Why:** run 2 mostly re-tuned LightGBM, and run 1 rediscovered standard stacking. Neither tells us whether AE
    finds something a careful data scientist wouldn't.
  - **Changes:**
    - `tabular_report.py` now has three baseline arms:
      - tuned LightGBM;
      - tuned XGBoost (same random-search harness, 48 trials each);
      - a **human stack**: comorbidity flags, ICD-9 chapters, medication Up/Down/Steady counts, total
        utilisation, out-of-fold target encoding of high-cardinality codes, and both tuned GBDTs stacked with a
        logistic meta-learner.
    - It reports Δ vs tuned LightGBM *and* vs the human stack, plus the mean and SD of Δ across candidates.
      `--selection-only` scores the baselines without touching the locked test.
    - New generic toolkit helpers: `icd9_comorbidities` (a fixed Charlson ICD-9 mapping) and `oof_target_encode`
      (`y_train` only, out-of-fold). Both are documented in `problem_description.md`.
    - **The seed is now the tuned LightGBM** (fitness 6646, 11.5 s for 3 rounds), so re-tuning alone can't win.
  - **Selection bar (fitness rounds, private seeds):**

    | Baseline | Selection AUC |
    |---|---|
    | Tuned LightGBM (= seed) | 0.6646 |
    | Tuned XGBoost | 0.6606 |
    | Human stack | **0.6708** |

    Run 1's best candidate scored 0.6756 on this measure.
  - **v2 replicate results (3 × 120 programs, 6 evaluators each; locked test n = 19,734):**
    - Report: `artifacts/reports/diabetes130_readmit30_20261002_233459.json`.
    - Finalist rule, fixed in advance: each run's rank1 by fitness.

    | Arm | Test AUC | Δ vs tuned LightGBM [95% CI] | Δ vs human stack [95% CI] | Brier |
    |---|---|---|---|---|
    | Tuned LightGBM (= seed) | 0.6808 | — | −0.0026 [−0.0056, +0.0004] | 0.0968 |
    | Tuned XGBoost | 0.6730 | −0.0079 [−0.0128, −0.0028] | −0.0105 [−0.0153, −0.0059] | 0.0969 |
    | Human stack | 0.6835 | +0.0026 [−0.0004, +0.0056] | — | 0.0962 |
    | Replicate A rank1 (`17130586394704764164`) | 0.6903 | +0.0095 [+0.0055, +0.0133] | **+0.0069 [+0.0037, +0.0100]** | 0.0957 |
    | Replicate B rank1 (`6843910815780438870`) | 0.6876 | +0.0067 [+0.0030, +0.0107] | **+0.0041 [+0.0009, +0.0074]** | 0.2942 |
    | Replicate C rank1 (`13049864015636605646`) | 0.6900 | +0.0091 [+0.0052, +0.0131] | **+0.0065 [+0.0032, +0.0096]** | 0.0957 |
    | Run-1 rank2 (reference) | 0.6895 | +0.0087 [+0.0043, +0.0128] | +0.0061 [+0.0027, +0.0095] | 0.0958 |

    - **Headline:** all 3 replicates beat the human stack with CIs above 0. Across replicates, mean Δ vs the human
      stack is **+0.0058 (SD 0.0015)** and vs tuned LightGBM **+0.0084 (SD 0.0015)**.
      - The script's printed "candidate mean" also includes run 3 and run-1 rank2, so use these replicate-only
        numbers instead.
    - **Leakage check:** all 3 PASS (shuffled AUC 0.489–0.495).
    - **What the replicates found:** comorbidity flags plus out-of-fold target encoding (the new helpers), and
      LightGBM + XGBoost.
      - A and C use a K-fold stack with a logistic meta-learner. B uses a rank-average blend instead.
      - Programs are 665–1,170 lines.
    - **Caveats:**
      - **Replicate B's Brier score is 0.294:** rank-averaged outputs are not probabilities. AUC is unaffected,
        but any deployment would need a calibration step.
      - **Heavy candidates:** all 3 runs had 11–15 candidates time out at 120 s, and A's finalists also time out
        on full-train heldout scoring. The 120 s cap is now binding.
      - **Fairness gaps** were not re-examined for these finalists.
  - **Run 3 (v1 seed, 200 programs):**
    - Best fitness was 0.6772, but **its rank1 crashes at full-train scale**: `ValueError: categorical
      cardinality 333 > 255`. With 59k rows, more categories pass the frequency threshold than in the 20k-row
      fitness rounds. Locked-test AUC is therefore 0.50.
    - `report.json` already flagged this (heldout = −995,000 for the top 3).
    - **Lesson: choose finalists among candidates with a valid heldout (full-train) score, not by fitness alone.**
  - **Depth test (2 runs at 2 evaluators / `CONCURRENCY=2`):** best-so-far fitness at equal program counts was
    within the replicate spread:

    | Run | @40 | @60 | @100 |
    |---|---|---|---|
    | Depth 1 | 0.6712 | 0.6744 | 0.6744 |
    | Depth 2 | 0.6746 | 0.6746 | 0.6754 |
    | Parallel replicates | 0.6725–0.6735 | 0.6728–0.6745 | 0.6728–0.6763 |

    - **No evidence that parallel evaluation hurts search quality per program**, and it is about 4× faster per
      wall-clock hour. Keep `WORKER_CONCURRENCY=6`.
    - Caveat: from about 20:30–22:08 the depth runs were contaminated by CPU oversubscription (next bullet), so
      only the early checkpoints are clean.
  - **Incident: CPU oversubscription.**
    - `leakage_check.py` and `tabular_report.py` ran candidates in-process without the sandbox's
      `OMP_NUM_THREADS=1`. Candidates using `n_jobs=-1` drove the load to 50 on 24 cores, and the depth runs went
      from 0–1 timeouts to 12 and 9.
    - Fixed in `2386fc5`, with a test. A side benefit: single-threaded leakage checks are about 5× faster than
      the thrashing multithreaded ones.

## 3. Local Setup Checklist
1. Authenticate locally with `gcloud auth login` and `gcloud auth application-default login`.
2. Run `bash scripts/provision_gcp.sh <YOUR_GCP_PROJECT_ID>` to enable `discoveryengine.googleapis.com` and `aiplatform.googleapis.com`, generate local private `AE_PERTURB_SEED` and `AE_HELDOUT_SEED` values in `.env`, provision the Discovery Engine / Gemini Enterprise engine and assistant (`setup_alphaevolve.py`), and download the UCI dataset.
3. **Launch a live run on real data:**
   - Run `uv run python run_experiment.py --problem diabetes130_readmit30`.
   - Re-run `scripts/tabular_report.py --candidates <top-3 programs>` with the private seeds, and report Δ vs tuned
     LightGBM and the human stack with CIs.
   - Every evaluated candidate is written to `artifacts/runs/<problem_id>/<exp_id>/candidates.json`.
4. Connect Clinical Silver/GCP CSV or AWS Snowflake cohorts (`SnowflakeDatasetAdapter`) via `problems/<problem_id>/problem_config.json`.

## 4. Remaining Maturity Backlog
- **Scale the transport:** columnar JSON is fine at 20k–60k rows. Above ~200k rows, switch to `numpy`/Arrow buffers.
- **Multi-fidelity:** add a CPU-time penalty or budget to the fitness if candidates drift toward heavy ensembles.
- **Leakage audits:** run label-shuffled sanity runs (fitness should collapse to about 0.5) and check feature
  importance on finalists.
- **Synthetic track:** either retire it, or turn it into a capability test:
  - strip the oracle features from `ClinicalFeatureBuilder`;
  - set prevalence to about 15% and n_eval ≥ 2k;
  - report *oracle-gap closed*.
- **Next open datasets:**
  - PhysioNet/CinC 2019 Sepsis (open; time-series, so it needs a different toolkit).
  - MIMIC-IV (credentialed; check PhysioNet's LLM-use terms first).
