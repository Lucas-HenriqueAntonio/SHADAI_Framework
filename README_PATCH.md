# SHADAI final-analysis compatibility patch

## Why this patch exists

The frozen final simulator writes:

- `rng_scheme = split_v2`
- `telemetry_scheme = c18_flood_intensity_v1`

Some experiment runners/analyzers retained historical validation labels:

- `c9_regret_v1`
- `c14_capacity_v1`
- `c15_runtime_v1`

Those historical labels caused valid final-campaign outputs to be rejected even though the current simulator produced them correctly.

## What this patch changes

It does **not** change:

- `shadai_sumo.py`
- `veiculo.py`
- selection logic
- pricing logic
- RNG behavior
- any simulation result

It only updates compatibility checks in experiment/analyzer scripts to the telemetry schema emitted by the frozen final core.

It also replaces `run_final_analysis.py` with a version that runs every analyzer and reports all failures instead of stopping at the first one.

## Installation

Copy all `.py` files from this patch into the root of `SHADAI_IEEE_FINAL`, replacing files with the same names.

Then run:

    python validate_final_outputs.py

If workload/capacity/runtime are present and compatible, do NOT rerun simulations.

Then run:

    python run_final_campaign.py --phase analysis

If the validator reports a genuinely incomplete experiment directory, rerun only that runner, e.g.:

    python ablation_load.py

and then rerun analysis.

## Important

The name `c18_flood_intensity_v1` is retained as a historical telemetry-schema identifier in the frozen core. It does not mean that non-flood experiments are being run as flood attacks. Attack behavior is controlled separately by `attack_type` and related fields.
