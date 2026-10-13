# veiculo.py

import random


class Veiculo:

    def __init__(
        self,
        id,
        custo_base,
        capacidade,
        pos=0.0,
        energia=100.0,
        lambda1=0.5,
        lambda2=0.3,
        lambda3=0.2,
        custo_max=8.0,
        preco_max=60.0,
        energia_servico_max=25.0,
        distancia_max=1000.0,
        tempo_max=100.0
    ):

        self.id = id

        self.custo_base = float(custo_base)

        self.capacidade = float(capacidade)

        self.posicao = float(pos)

        self.energia = float(energia)

        self.malicioso = False

        self.reputacao = random.uniform(
            5,
            10
        )

        self.servicos_concluidos = 0

        self.ganho_total = 0.0


        # =====================================================
        # PESOS DA FUNÇÃO DE UTILIDADE
        # =====================================================

        self.lambda1 = float(lambda1)

        self.lambda2 = float(lambda2)

        self.lambda3 = float(lambda3)


        # =====================================================
        # REFERÊNCIAS DE NORMALIZAÇÃO
        # =====================================================

        self.custo_max = float(
            custo_max
        )

        self.preco_max = float(
            preco_max
        )

        self.energia_servico_max = float(
            energia_servico_max
        )

        self.distancia_max = float(
            distancia_max
        )

        self.tempo_max = float(
            tempo_max
        )


        # =====================================================
        # VALIDAÇÕES
        # =====================================================

        soma_lambda = (
            self.lambda1
            + self.lambda2
            + self.lambda3
        )


        if abs(
            soma_lambda - 1.0
        ) > 1e-6:

            raise ValueError(
                "lambda1 + lambda2 + lambda3 "
                "must sum to 1.0."
            )


        if any(
            valor < 0
            for valor in [
                self.lambda1,
                self.lambda2,
                self.lambda3
            ]
        ):

            raise ValueError(
                "Lambda weights cannot be negative."
            )


        if self.custo_max <= 0:

            raise ValueError(
                "custo_max must be greater than zero."
            )


        if self.preco_max <= 0:

            raise ValueError(
                "preco_max must be greater than zero."
            )


    # =========================================================
    # CLAMP [0,1]
    # =========================================================

    @staticmethod
    def _clamp01(valor):

        return min(
            1.0,
            max(
                0.0,
                float(valor)
            )
        )


    # =========================================================
    # DISTÂNCIA
    # =========================================================

    def distancia_para(
        self,
        destino
    ):

        return abs(
            self.posicao
            - destino
        )


    # =========================================================
    # FUNÇÃO DE UTILIDADE NORMALIZADA
    # =========================================================

    def calcular_utilidade(
        self,
        preco,
        servico,
        posicao_tarefa
    ):

        distancia = (
            self.distancia_para(
                posicao_tarefa
            )
        )


        energia_servico = float(
            servico["energia"]
        )


        tempo_servico = float(
            servico["duracao"]
        )


        # =====================================================
        # PREÇO NORMALIZADO
        # =====================================================

        preco_norm = (
            float(preco)
            / self.preco_max
        )


        preco_norm = self._clamp01(
            preco_norm
        )


        # =====================================================
        # CUSTO-BASE NORMALIZADO
        # =====================================================

        custo_base_norm = (
            self.custo_base
            / self.custo_max
        )


        custo_base_norm = self._clamp01(
            custo_base_norm
        )


        # =====================================================
        # FATORES OPERACIONAIS NORMALIZADOS
        # =====================================================

        energia_norm = (
            energia_servico
            / self.energia_servico_max
        )


        energia_norm = self._clamp01(
            energia_norm
        )


        distancia_norm = (
            distancia
            / self.distancia_max
        )


        distancia_norm = self._clamp01(
            distancia_norm
        )


        tempo_norm = (
            tempo_servico
            / self.tempo_max
        )


        tempo_norm = self._clamp01(
            tempo_norm
        )


        # =====================================================
        # OPERATIONAL BURDEN
        # =====================================================

        operational_burden = (

            self.lambda1
            * energia_norm

            + self.lambda2
            * distancia_norm

            + self.lambda3
            * tempo_norm
        )


        # =====================================================
        # CUSTO TOTAL NORMALIZADO
        # =====================================================

        denominador = (

            1.0

            + self.lambda1
            + self.lambda2
            + self.lambda3
        )


        custo_total_norm = (

            custo_base_norm
            + operational_burden

        ) / denominador


        custo_total_norm = self._clamp01(
            custo_total_norm
        )


        # =====================================================
        # UTILIDADE
        # =====================================================

        utilidade = (

            preco_norm
            - custo_total_norm
        )


        # =====================================================
        # DETALHES PARA DIAGNÓSTICO
        # =====================================================

        detalhes = {

            "preco_norm":
                preco_norm,

            "custo_base_norm":
                custo_base_norm,

            "energia_norm":
                energia_norm,

            "distancia_norm":
                distancia_norm,

            "tempo_norm":
                tempo_norm,

            "operational_burden":
                operational_burden,

            "custo_total_norm":
                custo_total_norm
        }


        return (
            utilidade,
            custo_total_norm,
            distancia,
            detalhes
        )


    # =========================================================
    # EXECUÇÃO DO SERVIÇO
    # =========================================================

    def executar_servico(
        self,
        servico
    ):

        consumo = servico.get(
            "energia_consumo",
            servico["energia"]
        )


        variacao = random.uniform(
            0.9,
            1.1
        )


        consumo_real = (
            consumo
            * variacao
        )


        self.energia = max(
            0.0,
            self.energia
            - consumo_real
        )


        return consumo_real


    # =========================================================
    # REPUTAÇÃO
    # =========================================================

    def atualizar_reputacao(
        self,
        sucesso
    ):

        if self.malicioso:

            sucesso = (

                False

                if random.random() < 0.7

                else True
            )


        if sucesso:

            self.reputacao += 1

        else:

            self.reputacao -= 1


        self.reputacao = max(
            0,
            self.reputacao
        )


    # =========================================================
    # GANHO
    # =========================================================

    def registrar_ganho(
        self,
        valor
    ):

        self.ganho_total += float(
            valor
        )