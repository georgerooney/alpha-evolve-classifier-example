# AlphaEvolve Clinical Classifier — Agent Rules

- Read `NEXT_STEPS.md` first. It is the current handoff (portfolio baseline vs. evolved scores, GCP setup checklist, and Silver environment integration roadmap).
- Never edit evaluator, benchmark, or toolkit files while an AlphaEvolve experiment is running. The worker re-imports them from disk. Use a scratch copy.
- Never use `--no-perturb` for evolution fitness. Keep `AE_PERTURB_SEED` and `AE_HELDOUT_SEED` private and fixed per run, and never write them into committed files or insights.
- Any change to `src/evaluator.py`, `src/clinical_metrics.py`, `src/clinical_benchmarks.py`, or `src/tabular_benchmarks.py` requires a fresh experiment and re-scoring of the seed and finalists at HEAD.
- `problems/<problem_id>/initial_program.py` contains strategy only (`fit_and_score_risk` and `allocate_interventions`). Domain mechanics, feature builders, GBDT blenders, and constraint checks belong in `src/clinical_toolkit.py` and must be documented in `problems/<problem_id>/problem_description.md`.
- Real-data tabular problems (`task_type: tabular_classification`, e.g. `diabetes130_readmit30`) define only `fit_and_score_risk(task, train, y_train, eval_cols) -> probs`. Their toolkit helpers must stay generic (encoding, code grouping). Never add features derived from analysing the evaluation data. That would recreate the synthetic track's circularity, where the toolkit encodes the generator's formula.
- The locked test split (`--split test`) is for finalists only. Never use it inside a fitness loop or to tune the baseline. Report lift with `scripts/tabular_report.py` (paired bootstrap vs. a tuned LightGBM), not as "evolved vs. seed".
- Before reporting finalists, run `scripts/leakage_check.py` on them. It refits on shuffled labels using fitness rounds only, and the AUC must collapse to about 0.5.
- Pick finalists only from candidates with a valid `heldout` (full-train) score in `report.json`. Run 3's top 3 by fitness all crashed on 59k rows (category cardinality above 255).
- `src/` modules stay flat (not a package): candidates import them by bare name (`from clinical_toolkit import ...`), so never rename or prefix them.
- Tests: `.venv/bin/python -m pytest -q -p no:cacheprovider`. Always write the failing behavioral test first.
- `simulator/simulator_ui.html` must strictly use DOM construction APIs (`createElement`, `textContent`, `replaceChildren`) and never `innerHTML`.
