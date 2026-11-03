import pandas as pd


ARQUIVO = "requests_shadai_400_c14_test.csv"


df = pd.read_csv(
    ARQUIVO
)


print("\n")
print("=" * 90)

print(
    "C14 - REQUEST OUTCOME BREAKDOWN"
)

print("=" * 90)


# ============================================================
# TOTAL
# ============================================================

total = len(
    df
)


print(
    f"Total requests: {total}"
)


# ============================================================
# ALLOCATED / UNSERVED
# ============================================================

allocated = (

    df[
        "allocated"
    ]
    .astype(str)
    .str.lower()
    .isin(
        [
            "true",
            "1"
        ]
    )
)


n_allocated = allocated.sum()

n_unserved = (
    total
    - n_allocated
)


print(
    f"Allocated: {n_allocated}"
)

print(
    f"Unserved : {n_unserved}"
)


# ============================================================
# FAILURE REASONS
# ============================================================

resultado = (

    df[
        "failure_reason"
    ]

    .fillna(
        "allocated"
    )

    .replace(
        "",
        "allocated"
    )

    .value_counts()
)


print("\n")

print(
    "ABSOLUTE COUNTS"
)

print("-" * 90)

print(
    resultado
)


print("\n")

print(
    "PERCENTAGE OF ALL OFFERED REQUESTS"
)

print("-" * 90)


percentuais = (

    resultado
    /
    total
    *
    100.0
)


for motivo, valor in percentuais.items():

    print(
        f"{motivo:<30}: "
        f"{valor:>7.2f}%"
    )


# ============================================================
# UNSERVED-ONLY BREAKDOWN
# ============================================================

df_unserved = df[
    ~allocated
].copy()


print("\n")

print(
    "BREAKDOWN AMONG UNSERVED REQUESTS ONLY"
)

print("-" * 90)


if len(
    df_unserved
) > 0:

    falhas = (

        df_unserved[
            "failure_reason"
        ]

        .fillna(
            "unknown"
        )

        .replace(
            "",
            "unknown"
        )

        .value_counts()
    )


    for motivo, quantidade in falhas.items():

        percentual = (

            quantidade
            /
            len(
                df_unserved
            )
            *
            100.0
        )


        print(

            f"{motivo:<30}: "

            f"{quantidade:>5} "

            f"({percentual:>6.2f}%)"
        )


# ============================================================
# SANITY CHECK
# ============================================================

print("\n")

print(
    "SANITY CHECK"
)

print("-" * 90)


if (
    n_allocated
    +
    n_unserved
    ==
    total
):

    print(
        "[PASS] allocated + unserved = total requests"
    )

else:

    print(
        "[FAIL] Request accounting mismatch"
    )


if (
    "no_active_vehicles"
    in resultado.index
):

    print(
        f"No-active-vehicle requests: "
        f"{resultado['no_active_vehicles']}"
    )


if (
    "no_eligible_candidates"
    in resultado.index
):

    print(
        f"No-eligible-candidate requests: "
        f"{resultado['no_eligible_candidates']}"
    )


if (
    "selection_failed"
    in resultado.index
):

    print(
        f"Selection failures: "
        f"{resultado['selection_failed']}"
    )


print("\n")

print("=" * 90)

print(
    "C14 TEST ANALYSIS COMPLETED"
)

print("=" * 90)

print()