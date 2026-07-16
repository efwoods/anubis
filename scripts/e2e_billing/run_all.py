#!/usr/bin/env python
# scripts/e2e_billing/run_all.py

"""Run every end-to-end billing scenario in sequence and summarize.

Usage (from the repo root; the scripts import their neighbors, so run from
this directory or let this orchestrator handle the paths):

    export STRIPE_SECRET_KEY=sk_test_...
    export STRIPE_BILLING_CONFIG_JSON='...'        # from provision_stripe_billing.py
    export E2E_API_BASE_URL=http://localhost:8124  # optional, API-side checks
    export E2E_API_KEY=...                         # optional
    export E2E_ASSISTANT_ID=...                    # optional, upload scenario
    python scripts/e2e_billing/run_all.py

Each scenario is an independent process so one crash cannot mask the rest.
"""

from __future__ import annotations

import os
import subprocess
import sys

SCENARIO_FILES = [
    "scenario_messaging_allotment.py",
    "scenario_document_uploads.py",
    "scenario_adapters.py",
    "scenario_period_reset.py",
    "scenario_tier_changes.py",
    "scenario_trial_paths.py",
]


def main() -> int:
    """Run each scenario as a subprocess; return non-zero when any fails."""
    scenario_directory = os.path.dirname(os.path.abspath(__file__))
    results: dict[str, int] = {}
    for scenario_file in SCENARIO_FILES:
        completed = subprocess.run(
            [sys.executable, os.path.join(scenario_directory, scenario_file)],
            cwd=scenario_directory,
        )
        results[scenario_file] = completed.returncode

    print("\n=== end-to-end billing summary ===")
    for scenario_file, return_code in results.items():
        outcome = "PASS" if return_code == 0 else f"FAIL (exit {return_code})"
        print(f"  {outcome:>14}  {scenario_file}")
    return 0 if all(code == 0 for code in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
