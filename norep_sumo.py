# norep_sumo.py

import argparse
import os
import subprocess
import sys


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "C23 corrected No-Reputation baseline for SHADAI. "
        "The reputation component is removed and the remaining "
        "energy and proximity weights are renormalized equally: "
        "w1=0.0, w2=0.5, w3=0.5."
    )
)


parser.add_argument(
    "--seed",
    type=int,
    default=42
)


parser.add_argument(
    "--experiment",
    type=str,
    default="c23_norep"
)


parser.add_argument(
    "--output-dir",
    type=str,
    default="norep_alignment"
)


parser.add_argument(
    "--lambda1",
    type=float,
    default=0.5
)

parser.add_argument(
    "--lambda2",
    type=float,
    default=0.3
)

parser.add_argument(
    "--lambda3",
    type=float,
    default=0.2
)


parser.add_argument(
    "--epsilon",
    type=float,
    default=0.2
)


parser.add_argument(
    "--cooldown-steps",
    type=int,
    default=40
)


parser.add_argument(
    "--request-interval",
    type=int,
    default=20
)


parser.add_argument(
    "--headless",
    action="store_true"
)


parser.add_argument(
    "--shadai-script",
    type=str,
    default="shadai_sumo.py"
)


args = parser.parse_args()


# ============================================================
# VALIDATION
# ============================================================

if abs(
    args.lambda1
    +
    args.lambda2
    +
    args.lambda3
    -
    1.0
) > 1e-6:

    raise ValueError(
        "lambda1 + lambda2 + lambda3 must sum to 1.0."
    )


if not (
    0.0
    <=
    args.epsilon
    <=
    1.0
):

    raise ValueError(
        "epsilon must be between 0 and 1."
    )


if not os.path.exists(
    args.shadai_script
):

    raise FileNotFoundError(
        f"SHADAI simulator not found: "
        f"{args.shadai_script}"
    )


os.makedirs(
    args.output_dir,
    exist_ok=True
)


# ============================================================
# C23 NO-REPUTATION WEIGHTS
# ============================================================

#
# Original:
#
#   w1 = 0.4 reputation
#   w2 = 0.3 energy
#   w3 = 0.3 proximity
#
# No-Reputation:
#
#   w1 = 0.0
#   w2 = 0.5
#   w3 = 0.5
#
# The remaining weight mass is renormalized so that:
#
#   w1 + w2 + w3 = 1
#
# This isolates removal of reputation without changing the
# total score-weight mass.
#

W1_NOREP = 0.0
W2_NOREP = 0.5
W3_NOREP = 0.5


# ============================================================
# COMMAND
# ============================================================

cmd = [

    sys.executable,

    args.shadai_script,

    "--lambda1",
    str(args.lambda1),

    "--lambda2",
    str(args.lambda2),

    "--lambda3",
    str(args.lambda3),

    "--w1",
    str(W1_NOREP),

    "--w2",
    str(W2_NOREP),

    "--w3",
    str(W3_NOREP),

    "--epsilon",
    str(args.epsilon),

    "--cooldown-steps",
    str(args.cooldown_steps),

    "--request-interval",
    str(args.request_interval),

    "--malicious-ratio",
    "0.0",

    "--attack-type",
    "none",

    "--flood-intensity",
    "0",

    "--seed",
    str(args.seed),

    "--experiment",
    args.experiment,

    "--output-dir",
    args.output_dir
]


if args.headless:

    cmd.append(
        "--headless"
    )


# ============================================================
# HEADER
# ============================================================

print("\n")

print("=" * 120)

print(
    "C23 - CORRECTED NO-REPUTATION BASELINE"
)

print("=" * 120)

print(
    "Original weights      : "
    "(0.4, 0.3, 0.3)"
)

print(
    "No-Reputation weights : "
    "(0.0, 0.5, 0.5)"
)

print(
    "Reputation component  : disabled"
)

print(
    "Remaining weights     : renormalized"
)

print(
    f"Epsilon               : "
    f"{args.epsilon}"
)

print(
    f"Cooldown              : "
    f"{args.cooldown_steps}"
)

print(
    f"Request interval      : "
    f"{args.request_interval}"
)

print(
    "Attack                : none"
)

print(
    f"Seed                  : "
    f"{args.seed}"
)

print(
    f"Experiment            : "
    f"{args.experiment}"
)

print(
    f"Output directory      : "
    f"{args.output_dir}"
)

print("=" * 120)

print()


# ============================================================
# EXECUTE
# ============================================================

result = subprocess.run(
    cmd,
    check=False
)


if result.returncode != 0:

    raise RuntimeError(
        "No-Reputation simulation failed with "
        f"return code {result.returncode}."
    )


print("\n")

print("=" * 120)

print(
    "NO-REPUTATION RUN COMPLETED"
)

print("=" * 120)

print()