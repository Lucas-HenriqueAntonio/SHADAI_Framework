# lider.py

import random


class Plataforma:

    def __init__(
        self,
        preco_inicial=10.0,
        epsilon=0.2,
        w1=0.4,
        w2=0.3,
        w3=0.3,
        cooldown_steps=40,
        rng_policy_gate=None,
        rng_policy_choice=None
    ):

        self.preco = float(
            preco_inicial
        )

        # =====================================================
        # EPSILON
        # =====================================================

        self.epsilon = float(
            epsilon
        )

        if not (
            0.0 <= self.epsilon <= 1.0
        ):

            raise ValueError(
                "epsilon must be between 0.0 and 1.0."
            )

        # =====================================================
        # PESOS
        # =====================================================

        self.w1 = float(
            w1
        )

        self.w2 = float(
            w2
        )

        self.w3 = float(
            w3
        )

        soma_w = (
            self.w1
            + self.w2
            + self.w3
        )

        if abs(
            soma_w - 1.0
        ) > 1e-6:

            raise ValueError(
                "w1 + w2 + w3 must sum to 1.0."
            )

        if any(
            valor < 0
            for valor in [
                self.w1,
                self.w2,
                self.w3
            ]
        ):

            raise ValueError(
                "Score weights cannot be negative."
            )

        # =====================================================
        # RNGs EXCLUSIVOS DA POLÍTICA
        # =====================================================

        self.rng_policy_gate = (
            rng_policy_gate
            if rng_policy_gate is not None
            else random.Random()
        )

        self.rng_policy_choice = (
            rng_policy_choice
            if rng_policy_choice is not None
            else random.Random()
        )

        # =====================================================
        # COOLDOWN EM TEMPO DE SIMULAÇÃO
        # =====================================================

        self.cooldown = {}

        self.cooldown_steps = int(
            cooldown_steps
        )

        if self.cooldown_steps < 0:

            raise ValueError(
                "cooldown_steps cannot be negative."
            )


    # =========================================================
    # OFERTA
    # =========================================================

    def ofertar(
        self,
        veiculos,
        servico,
        pos_tarefa
    ):

        candidatos = []

        for v in veiculos:

            (
                utilidade,
                custo_total,
                dist,
                detalhes
            ) = v.calcular_utilidade(

                self.preco,
                servico,
                pos_tarefa
            )

            if (
                utilidade > 0
                and
                v.energia >= servico["energia"]
            ):

                candidatos.append({

                    "veiculo":
                        v,

                    "utilidade":
                        utilidade,

                    "custo_total":
                        custo_total,

                    "dist":
                        dist,

                    "detalhes_utilidade":
                        detalhes
                })

        return candidatos


    # =========================================================
    # COMPONENTES NORMALIZADOS
    # =========================================================

    def calcular_componentes_score(
        self,
        candidato
    ):

        v = candidato[
            "veiculo"
        ]

        distancia = candidato[
            "dist"
        ]

        ENERGIA_MAX = 120.0
        DIST_MAX = 1000.0
        REPUT_MAX = 10.0

        energia_norm = (
            v.energia
            / ENERGIA_MAX
        )

        proximidade_norm = (
            1.0
            - (
                distancia
                / DIST_MAX
            )
        )

        reputacao_norm = (
            v.reputacao
            / REPUT_MAX
        )

        energia_norm = min(
            1.0,
            max(
                0.0,
                energia_norm
            )
        )

        proximidade_norm = min(
            1.0,
            max(
                0.0,
                proximidade_norm
            )
        )

        reputacao_norm = min(
            1.0,
            max(
                0.0,
                reputacao_norm
            )
        )

        return {

            "reputacao_norm":
                reputacao_norm,

            "energia_norm":
                energia_norm,

            "proximidade_norm":
                proximidade_norm,

            "distancia":
                distancia
        }


    # =========================================================
    # SCORE
    # =========================================================

    def calcular_score(
        self,
        candidato
    ):

        componentes = (
            self.calcular_componentes_score(
                candidato
            )
        )

        score = (

            self.w1
            * componentes[
                "reputacao_norm"
            ]

            + self.w2
            * componentes[
                "energia_norm"
            ]

            + self.w3
            * componentes[
                "proximidade_norm"
            ]
        )

        return (
            score,
            componentes
        )


    # =========================================================
    # SELEÇÃO
    # =========================================================

    def selecionar(
        self,
        candidatos,
        pos_tarefa,
        step_atual
    ):

        if not candidatos:

            return []

        for candidato in candidatos:

            v = candidato[
                "veiculo"
            ]

            ultimo_step = (
                self.cooldown.get(
                    v.id,
                    None
                )
            )

            penalizado = False

            steps_desde_ultima_selecao = None

            if ultimo_step is not None:

                steps_desde_ultima_selecao = (
                    step_atual
                    - ultimo_step
                )

                if (
                    steps_desde_ultima_selecao
                    < self.cooldown_steps
                ):

                    penalizado = True

            (
                score_base,
                componentes
            ) = self.calcular_score(
                candidato
            )

            score_final = score_base

            if penalizado:

                score_final *= 0.6

            candidato[
                "score_base"
            ] = score_base

            candidato[
                "score"
            ] = score_final

            candidato[
                "reputacao_norm"
            ] = componentes[
                "reputacao_norm"
            ]

            candidato[
                "energia_norm"
            ] = componentes[
                "energia_norm"
            ]

            candidato[
                "proximidade_norm"
            ] = componentes[
                "proximidade_norm"
            ]

            candidato[
                "distancia"
            ] = componentes[
                "distancia"
            ]

            candidato[
                "cooldown_penalty"
            ] = penalizado

            candidato[
                "last_selection_step"
            ] = (
                ultimo_step
                if ultimo_step is not None
                else -1
            )

            candidato[
                "steps_since_last_selection"
            ] = (
                steps_desde_ultima_selecao
                if steps_desde_ultima_selecao is not None
                else -1
            )

        # =====================================================
        # ORDENA PELO SCORE FINAL
        # =====================================================

        candidatos.sort(

            key=lambda x:
                x["score"],

            reverse=True
        )

        # =====================================================
        # EPSILON-GREEDY
        #
        # O teste de exploration possui stream próprio.
        # A escolha aleatória possui outro stream próprio.
        # =====================================================

        sorteio_exploracao = (
            self.rng_policy_gate.random()
        )

        if sorteio_exploracao < self.epsilon:

            escolhido = (
                self.rng_policy_choice.choice(
                    candidatos
                )
            )

            escolhido[
                "selection_mode"
            ] = "exploration"

        else:

            escolhido = candidatos[
                0
            ]

            escolhido[
                "selection_mode"
            ] = "exploitation"

        # =====================================================
        # REGISTRA COOLDOWN
        # =====================================================

        self.cooldown[
            escolhido[
                "veiculo"
            ].id
        ] = step_atual

        return [
            escolhido
        ]