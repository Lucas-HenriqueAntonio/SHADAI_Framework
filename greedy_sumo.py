# greedy_sumo.py

import argparse
import os
import subprocess
import sys


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "C22 corrected Greedy baseline for SHADAI. "
        "Greedy is defined as pure exploitation of the same "
        "SHADAI final score: argmax_i Score_i, equivalent to "
        "running SHADAI with epsilon = 0."
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
    default="c22_greedy"
)


parser.add_argument(
    "--output-dir",
    type=str,
    default="greedy_alignment"
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
    "--w1",
    type=float,
    default=0.4
)

parser.add_argument(
    "--w2",
    type=float,
    default=0.3
)

parser.add_argument(
    "--w3",
    type=float,
    default=0.3
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


if abs(
    args.w1
    +
    args.w2
    +
    args.w3
    -
    1.0
) > 1e-6:

    raise ValueError(
        "w1 + w2 + w3 must sum to 1.0."
    )


if not os.path.exists(
    args.shadai_script
):

    raise FileNotFoundError(
        f"SHADAI simulator not found: {args.shadai_script}"
    )


os.makedirs(
    args.output_dir,
    exist_ok=True
)


# ============================================================
# GREEDY DEFINITION
# ============================================================

#
# IMPORTANT:
#
# Greedy is NOT:
#
#     argmin C_i
#
# Greedy is:
#
#     argmax Score_i_final
#
# where Score_i_final is the same score used by SHADAI,
# including the same cooldown mechanism.
#
# Since Plataforma.selecionar already performs exploitation
# by choosing the highest final score, epsilon = 0 makes the
# selection purely greedy.
#


EPSILON_GREEDY = 0.0


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
    str(args.w1),

    "--w2",
    str(args.w2),

    "--w3",
    str(args.w3),

    "--epsilon",
    str(EPSILON_GREEDY),

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
# PRINT
# ============================================================

print("\n")

print("=" * 120)

print(
    "C22 - CORRECTED GREEDY BASELINE"
)

print("=" * 120)

print(
    "Definition          : argmax final Score_i"
)

print(
    "Implementation      : SHADAI selector with epsilon = 0"
)

print(
    "Exploration         : disabled"
)

print(
    "Cooldown            : preserved"
)

print(
    "Eligibility         : identical to SHADAI"
)

print(
    "Score weights       : identical to SHADAI"
)

print(
    "Attack              : none"
)

print(
    f"Seed                : {args.seed}"
)

print(
    f"Experiment          : {args.experiment}"
)

print(
    f"Output directory    : {args.output_dir}"
)

print("=" * 120)

print()


# ============================================================
# EXECUTION
# ============================================================

result = subprocess.run(
    cmd,
    check=False
)


if result.returncode != 0:

    raise RuntimeError(
        "Greedy simulation failed with return code "
        f"{result.returncode}."
    )


print("\n")

print("=" * 120)

print(
    "GREEDY RUN COMPLETED"
)

print("=" * 120)

print()