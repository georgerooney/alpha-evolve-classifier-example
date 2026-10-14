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

"""Post-run LLM narration for AlphaEvolve clinical classifier runs (Vertex AI Gemini).

Writes two things next to each run on disk (`artifacts/runs/<problem_id>/<run_id>/`), neither of which touches the
evaluator or any fitness score:
  * `candidates.json[*].summary` — 1–2 sentence plain-English summary of each candidate (`gemini-3.5-flash-lite`),
    cached in place so re-runs only fill gaps (pass `--refresh` to re-generate).
  * `analysis.json` — structured commentary on the run (`gemini-3.8-flash`): headline, what changed vs. the seed,
    why it works clinically/statistically, what didn't work (failed candidates, regressions, and lessons across runs),
    risks/validation checks, and concrete ideas for the next run.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import difflib
import json
import os
import pathlib
import re
import sys
import urllib.request
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUMMARY_MODEL = os.environ.get("AE_SUMMARY_MODEL", "gemini-3.5-flash-lite")
ANALYSIS_MODEL = os.environ.get("AE_ANALYSIS_MODEL", "gemini-3.8-flash")
SEED_SUMMARY = "Baseline seed program (unchanged)."

_SECTIONS = ("headline", "what_changed", "why_it_works", "what_didnt_work", "risks", "next_steps")
ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "required": list(_SECTIONS),
    "properties": {
        "headline": {"type": "STRING", "description": "One sentence: what the best program does differently and the net result."},
        "what_changed": {"type": "STRING", "description": "Markdown bullets: the main algorithmic/feature changes vs. the seed in plain English."},
        "why_it_works": {"type": "STRING", "description": "Markdown bullets: the clinical/statistical reason each change helps, citing measured metrics."},
        "what_didnt_work": {"type": "STRING", "description": "Markdown bullets: failed candidates, regressions, constraint violations, and lessons across runs."},
        "risks": {"type": "STRING", "description": "Markdown bullets: caveats, calibration/generalization risks, and what to validate before deployment."},
        "next_steps": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "3-6 concrete, specific ideas for the next AlphaEvolve run."},
    },
}

GenerateFn = Callable[[str, str, dict[str, Any] | None], Any]


def _block(src: str) -> str:
    m = re.search(r"# EVOLVE-BLOCK-START\s*\n(.*?)# EVOLVE-BLOCK-END", src, re.DOTALL)
    return m.group(1).strip() if m else src.strip()


def _diff(seed: str, cand: str) -> str:
    return "".join(
        difflib.unified_diff(
            _block(seed).splitlines(True),
            _block(cand).splitlines(True),
            fromfile="seed",
            tofile="candidate",
            n=2,
        )
    )


def _offending_line(cand_src: str, insights: list[str]) -> str:
    """Pulls the line number from a `<candidate_program>` traceback and shows that source line."""
    lines = cand_src.splitlines()
    hits = [int(n) for ins in insights for n in re.findall(r'"<candidate_program>", line (\d+)', ins)]
    for ln in reversed(hits):
        if 1 <= ln <= len(lines):
            return f"Offending source line {ln}: {lines[ln - 1].strip()}"
    return ""


def _seed_source(run_dir: pathlib.Path, cands: list[dict], fallback_seed: str | None = None) -> str:
    seed = next((c for c in cands if c.get("generation") == 0), None)
    if seed:
        p = run_dir / seed.get("source_file", f"candidates/{seed['candidate_id']}.py")
        if p.is_file():
            return p.read_text(encoding="utf-8")
    if fallback_seed is not None:
        return fallback_seed
    problem_id = run_dir.parent.name
    init_p = ROOT / "problems" / problem_id / "initial_program.py"
    if init_p.is_file():
        return init_p.read_text(encoding="utf-8")
    return ""


def _metrics_line(c: dict[str, Any]) -> str:
    parts = []
    if c.get("roc_auc") is not None:
        parts.append(f"ROC-AUC {c['roc_auc']:.4f}")
    if c.get("pr_auc") is not None:
        parts.append(f"PR-AUC {c['pr_auc']:.4f}")
    if c.get("brier") is not None:
        parts.append(f"Brier {c['brier']:.4f}")
    if c.get("subgroup_gap") is not None:
        parts.append(f"Subgroup gap {c['subgroup_gap']:.4f}")
    if c.get("stage2_net_savings_usd"):
        parts.append(f"Stage-2 Net ROI ${c['stage2_net_savings_usd']:,.0f}")
    return ", ".join(parts) if parts else "no sub-metrics recorded"


def vertex_generate(model: str, prompt: str, schema: dict[str, Any] | None = None) -> Any:
    """Calls the Vertex AI `generateContent` REST endpoint using Application Default Credentials."""
    import google.auth
    import google.auth.transport.requests
    import setup_alphaevolve

    env_file = ROOT / ".env"
    env = {**(setup_alphaevolve.load_dotenv(env_file) if env_file.is_file() else {}), **os.environ}
    creds, default_project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    project = env.get("PROJECT_ID") or default_project or getattr(creds, "quota_project_id", None)
    if not project:
        raise RuntimeError("PROJECT_ID must be set in .env / environment or configured via gcloud ADC.")
    creds.refresh(google.auth.transport.requests.Request())
    if hasattr(creds, "quota_project_id") and not creds.quota_project_id:
        creds._quota_project_id = project

    gen_cfg: dict[str, Any] = {"temperature": 0.2}
    if schema is not None:
        gen_cfg.update(responseMimeType="application/json", responseSchema=schema)
    url = f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/publishers/google/models/{model}:generateContent"
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gen_cfg}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
            "x-goog-user-project": project,
        },
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    parts = data["candidates"][0]["content"]["parts"]
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    return json.loads(text) if schema is not None else text


def summarize_candidates(
    run_dir: pathlib.Path,
    problem_id: str | None = None,
    seed_src: str | None = None,
    *,
    generate: GenerateFn = vertex_generate,
    refresh: bool = False,
) -> int:
    """Fills `summary` (1–2 sentences) on each entry in `run_dir/candidates.json`. Returns count written."""
    cat_file = run_dir / "candidates.json"
    if not cat_file.is_file():
        return 0
    cands = json.loads(cat_file.read_text(encoding="utf-8"))
    prob_id = problem_id or run_dir.parent.name
    seed = _seed_source(run_dir, cands, fallback_seed=seed_src)
    seed_row = next((c for c in cands if c.get("generation") == 0), {})
    seed_score = seed_row.get("score")
    wrote = 0
    to_run: list[tuple[dict[str, Any], str]] = []
    for c in cands:
        cid = c.get("candidate_id")
        if c.get("generation") == 0:
            if c.get("summary") != SEED_SUMMARY:
                c["summary"] = SEED_SUMMARY
                c["summary_model"] = SUMMARY_MODEL
            continue
        if c.get("summary") and not refresh:
            continue
        src_p = run_dir / c.get("source_file", f"candidates/{cid}.py")
        cand_src = src_p.read_text(encoding="utf-8") if src_p.is_file() else ""
        diff = _diff(seed, cand_src) if cand_src else ""
        ins = c.get("insights") or []
        status, score = c.get("status"), c.get("score")
        if status == "GRADED" and score is not None:
            rel = (
                f" ({score - seed_score:+.2f} vs. seed {seed_score:.2f} — "
                f"{'better' if score >= seed_score else 'worse'} than the seed)"
                if seed_score is not None
                else ""
            )
            outcome = (
                f"Outcome: FEASIBLE (GRADED) — the candidate ran successfully on all evaluation folds and scored "
                f"{score:.2f}{rel}. Sub-metrics: {_metrics_line(c)}. Do NOT describe it as failed, crashed, or broken."
            )
        else:
            err = " | ".join(s for s in ins if "ERROR" in s or "VIOLATION" in s or "Violation" in s or "!" in s) or " ".join(ins[:2])
            off = _offending_line(cand_src, ins)
            outcome = f"Outcome: {status} (score {score}). Error/violation: {err[:700]}" + (f"\n{off}" if off else "")
        prompt = (
            f"You are summarizing one candidate program from an AlphaEvolve evolutionary coding run on `{prob_id}` "
            f"(clinical risk stratification & care allocation).\n"
            f"Candidate `{cid}` (generation {c.get('generation')}, rank #{c.get('rank')}).\n{outcome}\n\n"
            f"Unified diff of EVOLVE-BLOCK vs. the seed program:\n```diff\n{diff or '(no code diff from seed)'}\n```\n\n"
            "Write 1–2 concise sentences (max 45 words, plain text, no markdown or bullets) stating what this candidate "
            "changed in the algorithm/features relative to the seed and the resulting outcome (if it was feasible, say whether it "
            "improved or regressed vs. the seed; if it failed or violated a constraint, state the actual cause given above — "
            "never invent a cause not shown in the error)."
        )
        to_run.append((c, prompt))

    if to_run:
        from concurrent.futures import ThreadPoolExecutor

        workers = 12 if generate is vertex_generate else 1

        def _call(item: tuple[dict[str, Any], str]) -> tuple[dict[str, Any], str | None]:
            cand_row, pr = item
            try:
                res = str(generate(SUMMARY_MODEL, pr, None)).strip().replace("\n", " ")
                return cand_row, res
            except Exception as exc:  # noqa: BLE001 — fail soft: leave `summary` unset so a later run can retry
                print(f"  [warn] summary failed for {cand_row.get('candidate_id')}: {exc}", file=sys.stderr)
                return cand_row, None

        if workers == 1:
            results = [_call(it) for it in to_run]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_call, to_run))
        for cand_row, summary_txt in results:
            if summary_txt:
                cand_row["summary"] = summary_txt
                cand_row["summary_model"] = SUMMARY_MODEL
                wrote += 1

    cat_file.write_text(json.dumps(cands, indent=2) + "\n", encoding="utf-8")
    return wrote


def failure_clusters(cands: list[dict[str, Any]]) -> str:
    """Groups every non-GRADED candidate by its error class or constraint violation so no failure mode is hidden."""
    buckets: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for c in cands:
        st = c.get("status")
        if st in ("GRADED", "PENDING", None):
            continue
        text = " ".join(c.get("insights") or [])
        if st == "RUNTIME_ERROR":
            m = re.search(r"([A-Za-z_]\w*(?:Error|Exception)): ([^|\n]+)", text)
            cause = f"{m.group(1)}: {m.group(2).strip()[:120]}" if m else (text[:120] or "RuntimeError")
        else:
            m = re.search(r"\[[^\]]+\]\s*([^:|\n]+)", text)
            raw = m.group(1).strip() if m else re.sub(r"^HARD CONSTRAINT VIOLATIONS?[^:]*:\s*", "", text)[:120]
            cause = re.sub(r"-?\d+(?:\.\d+)?", "#", raw)
        key = f"{cause} [{st}]"
        buckets[key] += 1
        examples.setdefault(key, f"`{c.get('candidate_id')}` (gen {c.get('generation')})")
    if not buckets:
        return "  (none — every evaluated candidate was GRADED)"
    return "\n".join(f"  - {n} × {k} (e.g. {examples[k]})" for k, n in buckets.most_common(12))


def run_history(runs_dir: pathlib.Path, current_run_id: str) -> str:
    """Builds a chronological table of every run on disk for this problem (active and archived)."""
    rows: list[tuple[bool, float, str]] = []
    best_fit: tuple[float, str] | None = None
    best_ho: tuple[float, str] | None = None
    candidates_dirs: list[tuple[bool, pathlib.Path]] = []
    if runs_dir.is_dir():
        for p in runs_dir.iterdir():
            if p.is_dir() and not p.name.startswith("_"):
                candidates_dirs.append((True, p))
        arch = runs_dir / "_archive"
        if arch.is_dir():
            for p in arch.rglob("candidates.json"):
                candidates_dirs.append((False, p.parent))

    for active, d in candidates_dirs:
        cfile = d / "candidates.json"
        if not cfile.is_file():
            continue
        cands = json.loads(cfile.read_text(encoding="utf-8"))
        rep = json.loads((d / "report.json").read_text(encoding="utf-8")) if (d / "report.json").is_file() else {}
        graded = [c for c in cands if c.get("status") == "GRADED" and c.get("score") is not None]
        top_score = max((float(c["score"]) for c in graded), default=None)
        base = (rep.get("baseline") or {}).get("fitness")
        ho_base = (rep.get("baseline") or {}).get("heldout")
        top0 = (rep.get("top") or [{}])[0]
        top_scores = top0.get("scores") if isinstance(top0.get("scores"), dict) else top0
        ho_top = (top_scores or {}).get("heldout")
        if active and top_score is not None and (best_fit is None or top_score > best_fit[0]):
            best_fit = (top_score, f"run {d.name} rank 1")
        if active and isinstance(ho_base, (int, float)) and (best_ho is None or float(ho_base) > best_ho[0]):
            best_ho = (float(ho_base), f"run {d.name} seed")
        if active and isinstance(ho_top, (int, float)) and ho_top > 0 and (best_ho is None or ho_top > best_ho[0]):
            best_ho = (float(ho_top), f"run {d.name} rank 1")
        tag = "THIS RUN" if d.name == current_run_id else ("active" if active else f"archived:{d.parent.name}")
        rows.append(
            (
                active,
                cfile.stat().st_mtime,
                f"  - run `{d.name}` [{tag}, commit {rep.get('git_commit', '?')}]: "
                f"{len(cands)} candidates ({len(graded)} feasible, "
                f"{sum(1 for c in cands if c.get('status') == 'HARD_VIOLATION')} hard-violation, "
                f"{sum(1 for c in cands if c.get('status') == 'RUNTIME_ERROR')} runtime-error) | "
                f"seed fitness={base}, best fitness={top_score}, seed heldout={ho_base}, rank1 heldout={ho_top}",
            )
        )
    rows.sort(key=lambda r: r[1])
    hdr = []
    if best_fit:
        hdr.append(f"  Best fitness among active runs: {best_fit[0]:.2f} ({best_fit[1]})")
    if best_ho:
        hdr.append(f"  Best held-out among active runs: {best_ho[0]:.2f} ({best_ho[1]})")
    if any(not a for a, _, _ in rows):
        hdr.append("  (Archived runs used earlier seed baselines or split configurations; their numbers are not comparable.)")
    return "\n".join(hdr + [line for _, _, line in rows])


def explain_run(
    run_dir: pathlib.Path,
    problem_id: str | None = None,
    seed_src: str | None = None,
    *,
    generate: GenerateFn = vertex_generate,
) -> dict[str, Any]:
    """Generates `run_dir/analysis.json` with `gemini-3.8-flash`."""
    cands = json.loads((run_dir / "candidates.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8")) if (run_dir / "report.json").is_file() else {}
    prob_id = problem_id or report.get("problem_id") or run_dir.parent.name
    seed = _seed_source(run_dir, cands, fallback_seed=seed_src)
    best_src = (
        (run_dir / "rank1.py").read_text(encoding="utf-8")
        if (run_dir / "rank1.py").is_file()
        else _seed_source(run_dir, cands, fallback_seed=seed_src)
    )
    prob_md_path = ROOT / "problems" / prob_id / "problem_description.md"
    prob_md = prob_md_path.read_text(encoding="utf-8") if prob_md_path.is_file() else f"Problem: {prob_id}"

    # Check if there is a locked-test report for this problem in artifacts/reports/
    reports_dir = run_dir.parents[1] / "reports" if run_dir.parent.name == prob_id else ROOT / "artifacts" / "reports"
    locked_reports = sorted(reports_dir.glob(f"{prob_id}_*.json")) if reports_dir.is_dir() else []
    locked_summary = ""
    if locked_reports:
        try:
            lr = json.loads(locked_reports[-1].read_text(encoding="utf-8"))
            arms_txt = "\n".join(
                f"  - {a.get('arm')}: locked-test ROC-AUC={a.get('test_auc')}, PR-AUC={a.get('test_pr_auc')}, "
                f"Brier={a.get('test_brier')}, delta_vs_tuned={a.get('vs_tuned_delta')} "
                f"(95% CI [{a.get('vs_tuned_ci_low')}, {a.get('vs_tuned_ci_high')}], p={a.get('vs_tuned_p_value_one_sided')}), "
                f"delta_vs_stack={a.get('vs_stack_delta')} (p={a.get('vs_stack_p_value_one_sided')})"
                for a in (lr.get("arms") or [])
            )
            locked_summary = f"\nLocked-Test Paired Bootstrap Report (N={lr.get('locked_test_n')}):\n{arms_txt}\n"
        except Exception:
            locked_summary = ""

    seed_row = next((c for c in cands if c.get("generation") == 0), {})
    seed_line = (
        f"Seed program (the baseline every claim compares with): fitness {seed_row.get('score')}, "
        f"{_metrics_line(seed_row)} | {seed_row.get('summary') or SEED_SUMMARY}"
    )
    graded = [c for c in cands if c.get("status") == "GRADED" and c.get("score") is not None]
    seed_fit = (report.get("baseline") or {}).get("fitness") or seed_row.get("score")
    worse = [c for c in graded if seed_fit is not None and float(c["score"]) < float(seed_fit)]
    top_lines = "\n".join(
        f"  - rank #{c.get('rank')} `{c['candidate_id']}` (gen {c.get('generation')}): fitness {c.get('score')} | "
        f"{_metrics_line(c)} | {c.get('summary') or c.get('what_changed', '')}"
        for c in graded[:8]
    )
    regress_lines = "\n".join(
        f"  - `{c['candidate_id']}` (gen {c.get('generation')}): fitness {c.get('score')} "
        f"({float(c['score']) - float(seed_fit):+.2f} vs. seed) | {_metrics_line(c)} | "
        f"{c.get('summary') or c.get('what_changed', '')}"
        for c in worse[:5]
    ) or "  (none — every feasible candidate beat or matched the seed)"

    prompt = (
        f"You are a staff clinical machine learning scientist and biostatistician writing the post-run analysis of "
        f"AlphaEvolve run `{run_dir.name}` on `{prob_id}` for a technical & executive audience.\n\n"
        f"## Problem description\n{prob_md[:3500]}\n\n"
        f"## Run history across this repository (compare against this — only call a score 'best to date' if it beats "
        f"the active-run best lines below)\n{run_history(run_dir.parent, run_dir.name)}\n\n"
        f"## This run's scoreboard (`report.json`)\n```json\n{json.dumps(report, indent=2)}\n```\n"
        f"{locked_summary}\n"
        f"## Seed baseline & top feasible candidates (of {len(cands)} total; {len(graded)} feasible)\n"
        f"  - {seed_line}\n{top_lines}\n\n"
        f"## Feasible candidates that regressed below the seed ({len(worse)} total)\n{regress_lines}\n\n"
        f"## Non-feasible candidates grouped by cause (all {len(cands) - len(graded)} of them)\n"
        f"{failure_clusters(cands)}\n\n"
        f"## Seed EVOLVE-BLOCK\n```python\n{_block(seed)}\n```\n\n"
        f"## Rank #1 EVOLVE-BLOCK (`rank1.py`)\n```python\n{_block(best_src)}\n```\n\n"
        f"## Unified diff (seed -> rank #1)\n```diff\n{_diff(seed, best_src)}\n```\n\n"
        "Produce the structured JSON commentary. Ground every claim in the diff, insights, and numbers above: "
        "compare rank #1's metrics against the SEED line (not rank #2), cite exact failure counts from the "
        "grouped list above, explain in `what_didnt_work` which candidate ideas crashed, violated constraints, or "
        "regressed (including lessons across runs like category cardinality > 255 on full train if applicable), "
        "and make `next_steps` concrete algorithmic or feature-engineering experiments for the next run."
    )
    for secret in ("AE_PERTURB_SEED", "AE_HELDOUT_SEED"):
        val = os.environ.get(secret)
        if val:
            prompt = prompt.replace(val, "<redacted>")
    raw = generate(ANALYSIS_MODEL, prompt, ANALYSIS_SCHEMA)
    if not isinstance(raw, dict) or not all(k in raw for k in _SECTIONS) or not isinstance(raw["next_steps"], list):
        raise ValueError(f"Malformed analysis response from {ANALYSIS_MODEL}: {raw!r}")
    out = {
        **{k: raw[k] for k in _SECTIONS},
        "model": ANALYSIS_MODEL,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    (run_dir / "analysis.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def narrate(
    run_dir: pathlib.Path,
    problem_id: str | None = None,
    seed_src: str | None = None,
    *,
    generate: GenerateFn = vertex_generate,
    refresh: bool = False,
) -> None:
    """Best-effort hook for `run_experiment.py` — logs and swallows any LLM/network error."""
    try:
        n = summarize_candidates(run_dir, problem_id, seed_src, generate=generate, refresh=refresh)
        explain_run(run_dir, problem_id, seed_src, generate=generate)
        print(f"Narrated {run_dir.name}: {n} candidate summaries ({SUMMARY_MODEL}) + analysis.json ({ANALYSIS_MODEL})")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] narration skipped for {run_dir.name}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", help="Problem ID (defaults to all problems with runs)")
    ap.add_argument("--run", help="Run ID under artifacts/runs/<problem_id>/ (defaults to all runs)")
    ap.add_argument("--refresh", action="store_true", help="Re-summarize candidates even if a summary is cached")
    args = ap.parse_args(argv)

    runs_root = ROOT / "artifacts" / "runs"
    if not runs_root.is_dir():
        return 0
    prob_dirs = [runs_root / args.problem] if args.problem else sorted(p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("_"))
    for pdir in prob_dirs:
        if not pdir.is_dir():
            continue
        prob_id = pdir.name
        rdirs = [pdir / args.run] if args.run else sorted(
            (d for d in pdir.iterdir() if d.is_dir() and not d.name.startswith("_") and (d / "candidates.json").is_file()),
            key=lambda d: (d / "candidates.json").stat().st_mtime,
        )
        for rdir in rdirs:
            print(f"[{prob_id}/{rdir.name}] summarizing candidates with {SUMMARY_MODEL}...")
            n = summarize_candidates(rdir, prob_id, refresh=args.refresh)
            print(f"[{prob_id}/{rdir.name}] wrote {n} new summaries; generating run analysis with {ANALYSIS_MODEL}...")
            explain_run(rdir, prob_id)
            print(f"[{prob_id}/{rdir.name}] -> {rdir / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
