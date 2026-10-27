# validar_c16.py

import os
import subprocess
import sys

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "c16_tests"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


TESTES = [

    {
        "name":
            "c16_no_attack_default",

        "attack_enabled":
            False,

        "malicious_ratio":
            0.20
    },

    {
        "name":
            "c16_attack_10",

        "attack_enabled":
            True,

        "malicious_ratio":
            0.10
    },

    {
        "name":
            "c16_attack_20",

        "attack_enabled":
            True,

        "malicious_ratio":
            0.20
    },

    {
        "name":
            "c16_attack_40",

        "attack_enabled":
            True,

        "malicious_ratio":
            0.40
    }
]


SEED = 42

SHADAI_SCRIPT = "shadai_sumo.py"

EXPECTED_RNG = "split_v1"

EXPECTED_TELEMETRY = "c15_runtime_v1"

EXPECTED_ATTACK_SCHEME = "c16_param_v1"


# ============================================================
# BOOL HELPER
# ============================================================

def converter_bool(
    valor
):

    return str(
        valor
    ).strip().lower() in [

        "true",
        "1",
        "yes"
    ]


# ============================================================
# EXECUTA TESTES
# ============================================================

print("\n")

print("=" * 110)

print(
    "SHADAI - C16 MALICIOUS-RATIO PARAMETERIZATION VALIDATION"
)

print("=" * 110)


falhas = []


for teste in TESTES:

    nome = teste[
        "name"
    ]

    ataque = teste[
        "attack_enabled"
    ]

    ratio = teste[
        "malicious_ratio"
    ]


    comando = [

        sys.executable,

        SHADAI_SCRIPT,

        "--headless",

        "--seed",
        str(
            SEED
        ),

        "--experiment",
        nome,

        "--output-dir",
        OUTPUT_DIR,

        "--malicious-ratio",
        str(
            ratio
        )
    ]


    if ataque:

        comando.append(
            "--attack-enabled"
        )


    print(
        f"\nRUN: {nome}"
    )

    print(
        f"  attack_enabled = {ataque}"
    )

    print(
        f"  malicious_ratio = {ratio:.2f}"
    )


    resultado = subprocess.run(
        comando
    )


    if resultado.returncode != 0:

        falhas.append(
            (
                nome,
                "simulation_failed"
            )
        )

        continue


    arquivos = [

        os.path.join(

            OUTPUT_DIR,

            f"metricas_shadai_400_{nome}.csv"
        ),

        os.path.join(

            OUTPUT_DIR,

            f"eligibilidade_shadai_400_{nome}.csv"
        ),

        os.path.join(

            OUTPUT_DIR,

            f"requests_shadai_400_{nome}.csv"
        ),

        os.path.join(

            OUTPUT_DIR,

            f"runtime_shadai_400_{nome}.csv"
        )
    ]


    for arquivo in arquivos:

        if not os.path.exists(
            arquivo
        ):

            falhas.append(
                (
                    nome,
                    f"missing:{arquivo}"
                )
            )

            continue


        df = pd.read_csv(
            arquivo
        )


        if df.empty:

            falhas.append(
                (
                    nome,
                    f"empty:{arquivo}"
                )
            )

            continue


        obrigatorias = [

            "rng_scheme",
            "telemetry_scheme",
            "attack_scheme",
            "attack_enabled",
            "malicious_ratio"
        ]


        for coluna in obrigatorias:

            if coluna not in df.columns:

                falhas.append(
                    (
                        nome,
                        f"missing_column:{coluna}"
                    )
                )


        if not set(
            obrigatorias
        ).issubset(
            df.columns
        ):

            continue


        # ====================================================
        # RNG
        # ====================================================

        if not (
            df[
                "rng_scheme"
            ]
            .astype(str)
            .eq(
                EXPECTED_RNG
            )
            .all()
        ):

            falhas.append(
                (
                    nome,
                    "wrong_rng_scheme"
                )
            )


        # ====================================================
        # TELEMETRY
        # ====================================================

        if not (
            df[
                "telemetry_scheme"
            ]
            .astype(str)
            .eq(
                EXPECTED_TELEMETRY
            )
            .all()
        ):

            falhas.append(
                (
                    nome,
                    "wrong_telemetry_scheme"
                )
            )


        # ====================================================
        # ATTACK SCHEME
        # ====================================================

        if not (
            df[
                "attack_scheme"
            ]
            .astype(str)
            .eq(
                EXPECTED_ATTACK_SCHEME
            )
            .all()
        ):

            falhas.append(
                (
                    nome,
                    "wrong_attack_scheme"
                )
            )


        # ====================================================
        # ATTACK ENABLED
        # ====================================================

        attack_values = (

            df[
                "attack_enabled"
            ]
            .map(
                converter_bool
            )
        )


        if not (
            attack_values
            ==
            ataque
        ).all():

            falhas.append(
                (
                    nome,
                    "wrong_attack_enabled"
                )
            )


        # ====================================================
        # MALICIOUS RATIO
        # ====================================================

        ratios = pd.to_numeric(

            df[
                "malicious_ratio"
            ],

            errors="coerce"
        )


        if not (
            (
                ratios
                -
                ratio
            )
            .abs()
            <
            1e-9
        ).all():

            falhas.append(
                (
                    nome,
                    "wrong_malicious_ratio"
                )
            )


    print(
        f"[OK] {nome} outputs inspected."
    )


# ============================================================
# TESTA LIMITES INVÁLIDOS
# ============================================================

print("\n")

print(
    "VALIDATING INVALID MALICIOUS RATIOS"
)

print("-" * 110)


for invalido in [

    -0.1,
    1.1
]:

    comando = [

        sys.executable,

        SHADAI_SCRIPT,

        "--headless",

        "--experiment",
        "c16_invalid",

        "--malicious-ratio",
        str(
            invalido
        )
    ]


    resultado = subprocess.run(
        comando
    )


    if resultado.returncode == 0:

        falhas.append(
            (
                "invalid_ratio",
                f"accepted:{invalido}"
            )
        )


        print(
            f"[FAIL] Invalid ratio accepted: "
            f"{invalido}"
        )

    else:

        print(
            f"[PASS] Invalid ratio rejected: "
            f"{invalido}"
        )


# ============================================================
# RESULTADO FINAL
# ============================================================

print("\n")

print("=" * 110)

print(
    "C16 VALIDATION SUMMARY"
)

print("=" * 110)


if not falhas:

    print(
        "[PASS] All C16 parameterization checks passed."
    )

    print(
        "[PASS] malicious_ratio accepts values in [0, 1]."
    )

    print(
        "[PASS] attack_enabled is independently configurable."
    )

    print(
        "[PASS] attack metadata is present in all output types."
    )

    print(
        "[PASS] existing split RNG scheme remains unchanged."
    )

else:

    print(
        f"[FAIL] {len(falhas)} problem(s) detected."
    )

    for falha in falhas:

        print(
            " -",
            falha
        )


print("=" * 110)

print()