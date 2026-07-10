r"""
evaluation/run_all_ablations.py — subprocess-isolated orchestrator.
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STAGES = [
    ("cca",           "CCA Layer Ablation",                       "evaluation.ablation_cca"),
    ("attention",     "Attention-Module Comparison",              "evaluation.ablation_attention_variants"),
    ("loss",          "Loss-Function Ablation",                   "evaluation.ablation_loss"),
    ("augmentation",  "Augmentation-Strategy Ablation",           "evaluation.ablation_augmentation"),
    ("graph",         "Graph-Construction Ablation",              "evaluation.ablation_graph"),
    ("unet",          "Segmentation-Backbone Ablation",           "evaluation.ablation_unet_variants"),
    ("incremental",   "Incremental CCA+Graph+KG Ablation",        "evaluation.ablation_incremental"),
    ("neurosymbolic", "Neurosymbolic Gate Isolation",             "evaluation.ablation_neurosymbolic"),
    ("other",         "Other-Modules Ablation",                   "evaluation.ablation_other_modules"),
    ("cost",          "Computational Cost & Resource Study",      "evaluation.compute_cost"),
    ("report",        "Consolidated Report & Master Figures",     "evaluation.generate_visuals"),
]


def run_stage_subprocess(module_path: str) -> int:
    cmd = [sys.executable, "-m", module_path]
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--skip", nargs="+", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name, desc, module in STAGES:
            print(f"  {name:15s} {desc}   [{module}]")
        return

    stages_to_run = STAGES
    if args.only:
        wanted = set(args.only)
        stages_to_run = [s for s in STAGES if s[0] in wanted]
    if args.skip:
        skip = set(args.skip)
        stages_to_run = [s for s in stages_to_run if s[0] not in skip]

    results = {}
    for i, (name, desc, module) in enumerate(stages_to_run, 1):
        print(f"\n{'#' * 70}\n# STAGE {i}/{len(stages_to_run)}: {name} — {desc}\n{'#' * 70}")
        t0 = time.time()
        try:
            returncode = run_stage_subprocess(module)
        except Exception as e:
            print(f"  [ERROR] Could not launch stage '{name}': {e}")
            returncode = -1
        elapsed = time.time() - t0

        if returncode == 0:
            results[name] = ("OK", elapsed)
        else:
            results[name] = (f"FAILED (exit code {returncode})", elapsed)
            print(f"  [ERROR] Stage '{name}' exited with code {returncode}.")
            if args.stop_on_error:
                break

    print("\n" + "=" * 70 + "\n  ABLATION SUITE SUMMARY\n" + "=" * 70)
    for name, (status, elapsed) in results.items():
        marker = "OK " if status == "OK" else "!! "
        print(f"  [{marker}] {name:15s} {status:30s} ({elapsed/60:.1f} min)")

    if any(s[0] != "OK" for s in results.values()):
        failed = [n for n, (s, _) in results.items() if s != "OK"]
        print(f"\n  Re-run just the failed ones:")
        print(f"    python evaluation/run_all_ablations.py --only {' '.join(failed)}")


if __name__ == "__main__":
    main()