# veiculo.py
import random


class Veiculo:

    def __init__(self, id, custo_base, capacidade, pos=0.0, energia=100.0):
        self.id = id
        self.custo_base = float(custo_base)
        self.capacidade = float(capacidade)
        self.posicao = float(pos)
        self.energia = float(energia)
        self.malicioso = False
        self.reputacao = random.uniform(5, 10)
        self.servicos_concluidos = 0
        self.ganho_total = 0.0

    def distancia_para(self, destino):
        return abs(self.posicao - destino)

    # =====================================================
    # FUNÇÃO DE UTILIDADE NORMALIZADA (SHADAI)
    # =====================================================
    def calcular_utilidade(self, preco, servico, posicao_tarefa):

        distancia = self.distancia_para(posicao_tarefa)

        energia_servico = servico["energia"]
        tempo_servico = servico["duracao"]

        # normalização
        ENERGIA_MAX = 25.0
        DIST_MAX = 1000.0
        TEMPO_MAX = 100.0

        energia_norm = energia_servico / ENERGIA_MAX
        dist_norm = distancia / DIST_MAX
        tempo_norm = tempo_servico / TEMPO_MAX

        # pesos
        lambda1 = 0.5
        lambda2 = 0.3
        lambda3 = 0.2

        custo_total = (
            self.custo_base +
            lambda1 * energia_norm +
            lambda2 * dist_norm +
            lambda3 * tempo_norm
        )

        utilidade = float(preco) - custo_total

        return utilidade, custo_total, distancia

    # =====================================================
    # EXECUÇÃO DO SERVIÇO
    # =====================================================
    def executar_servico(self, servico):

        consumo = servico.get("energia_consumo", servico["energia"])
        variacao = random.uniform(0.9, 1.1)

        consumo_real = consumo * variacao
        self.energia = max(0.0, self.energia - consumo_real)

        return consumo_real

    # =====================================================
    # REPUTAÇÃO
    # =====================================================
    def atualizar_reputacao(self, sucesso):

    # comportamento malicioso
        if self.malicioso:
            sucesso = False if random.random() < 0.7 else True

        if sucesso:
            self.reputacao += 1
        else:
            self.reputacao -= 1

        self.reputacao = max(0, self.reputacao)

    # =====================================================
    # GANHO
    # =====================================================
    def registrar_ganho(self, valor):
        self.ganho_total += float(valor)