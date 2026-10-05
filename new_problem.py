#!/usr/bin/env python3
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

"""Scaffolding CLI to generate a new clinical problem folder in `problems/<problem_id>/`.

Usage:
    python3 new_problem.py oncology_30d --title "Oncology 30-Day Acute Toxicity Admission" --target-auc 0.75
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil

HERE = pathlib.Path(__file__).resolve().parent
TEMPLATE_DIR = HERE / "problems" / "_template"


def create_problem(
    problem_id: str,
    *,
    title: str,
    horizon_days: int = 30,
    target_auc: float = 0.75,
    event_cost_usd: float = 18000.0,
    roi_per_1pct_auc_usd: float = 450000.0,
    output_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    if not re.fullmatch(r"[a-z0-9_]+", problem_id):
        raise ValueError(f"Invalid problem_id '{problem_id}': use lowercase letters, digits, and underscores.")

    dest = (output_dir.resolve() if output_dir else (HERE / "problems" / problem_id).resolve())
    dest.mkdir(parents=True, exist_ok=True)

    prefix = problem_id.upper()[:6]
    cfg = {
        "problem_id": problem_id,
        "title": title,
        "horizon_days": horizon_days,
        "target_auc": target_auc,
        "target_specificity": 0.80,
        "min_sensitivity_floor": 0.55,
        "event_cost_usd": event_cost_usd,
        "hrrp_penalty_multiplier": 1.15,
        "roi_per_1pct_auc_usd": roi_per_1pct_auc_usd,
        "base_seed": 55000,
        "scenarios": [
            {
                "scenario_id": f"{prefix}-01",
                "scenario_name": f"{title} — Standard Clinical Cohort",
                "description": f"Primary benchmark cohort for {title}.",
                "n_train": 340,
                "n_eval": 170,
                "budget_usd": 26000.0,
                "shift": "standard",
            },
            {
                "scenario_id": f"{prefix}-02",
                "scenario_name": f"{title} — High-SDoH & Adherence Risk Cohort",
                "description": "High social deprivation and post-discharge medication adherence barriers.",
                "n_train": 320,
                "n_eval": 160,
                "budget_usd": 24000.0,
                "shift": "high_sdoh",
            },
            {
                "scenario_id": f"{prefix}-03",
                "scenario_name": f"{title} — Multi-Organ Renal/Metabolic Complexity",
                "description": "Concurrent renal and metabolic instability subpopulation.",
                "n_train": 320,
                "n_eval": 160,
                "budget_usd": 24000.0,
                "shift": "renal_metabolic",
            },
            {
                "scenario_id": f"{prefix}-04",
                "scenario_name": f"{title} — Capacity-Constrained Care Network",
                "description": "Tight care-management slot limits requiring high specificity.",
                "n_train": 320,
                "n_eval": 160,
                "budget_usd": 20000.0,
                "shift": "capacity_crunch",
            },
        ],
    }
    (dest / "problem_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    seed_src = TEMPLATE_DIR / "initial_program.py"
    desc_src = TEMPLATE_DIR / "problem_description.md"
    if seed_src.is_file():
        shutil.copyfile(seed_src, dest / "initial_program.py")
    if desc_src.is_file():
        raw_desc = desc_src.read_text(encoding="utf-8")
        rendered = (
            raw_desc.replace("{{PROBLEM_ID}}", problem_id)
            .replace("{{TITLE}}", title)
            .replace("{{HORIZON_DAYS}}", str(horizon_days))
            .replace("{{TARGET_AUC}}", f"{target_auc:.2f}")
            .replace("{{EVENT_COST_USD}}", f"{event_cost_usd:,.0f}")
            .replace("{{ROI_PER_1PCT_USD}}", f"{roi_per_1pct_auc_usd:,.0f}")
        )
        (dest / "problem_description.md").write_text(rendered, encoding="utf-8")

    return dest


def main() -> None:
    p = argparse.ArgumentParser(description="Scaffold a new AlphaEvolve clinical problem directory.")
    p.add_argument("problem_id", help="Short snake_case identifier, e.g. oncology_30d")
    p.add_argument("--title", required=True, help="Human-readable clinical model title")
    p.add_argument("--horizon-days", type=int, default=30, help="Prediction window in days")
    p.add_argument("--target-auc", type=float, default=0.75, help="Target ROC-AUC gate")
    p.add_argument("--event-cost-usd", type=float, default=18000.0, help="Cost per acute clinical event (USD)")
    p.add_argument("--roi-per-1pct-usd", type=float, default=450000.0, help="Actuarial ROI per +1%% AUC uplift (USD)")
    p.add_argument("--output-dir", type=pathlib.Path, default=None, help="Optional custom output directory")
    args = p.parse_args()

    out = create_problem(
        args.problem_id,
        title=args.title,
        horizon_days=args.horizon_days,
        target_auc=args.target_auc,
        event_cost_usd=args.event_cost_usd,
        roi_per_1pct_auc_usd=args.roi_per_1pct_usd,
        output_dir=args.output_dir,
    )
    print(f"Created problem template at: {out}")


if __name__ == "__main__":
    main()
