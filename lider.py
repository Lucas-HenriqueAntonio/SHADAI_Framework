# lider.py
import random
import time


class Plataforma:

    def __init__(self, preco_inicial=10.0, epsilon=0.2):

        self.preco = float(preco_inicial)
        self.epsilon = float(epsilon)

        self.cooldown = {}
        self.cooldown_steps = 40

        # pesos
        self.w1 = 0.4   # reputação
        self.w2 = 0.3   # energia
        self.w3 = 0.3   # proximidade

    # =====================================================
    # CONSULTA VEÍCULOS
    # =====================================================
    def ofertar(self, veiculos, servico, pos_tarefa):

        candidatos = []

        for v in veiculos:

            if hasattr(v, "calcular_utilidade"):

                utilidade, custo_total, dist = v.calcular_utilidade(
                    self.preco,
                    servico,
                    pos_tarefa
                )
            else:
                dist = v.distancia_para(pos_tarefa)
                custo_total = 0
                utilidade = 0

            if utilidade > 0 and v.energia >= servico["energia"]:

                candidatos.append({
                    "veiculo": v,
                    "utilidade": utilidade,
                    "custo_total": custo_total,
                    "dist": dist
                })

        return candidatos

    # =====================================================
    # SCORE NORMALIZADO
    # =====================================================
    def calcular_score(self, candidato):

        v = candidato["veiculo"]
        d = candidato["dist"]

        ENERGIA_MAX = 120.0
        DIST_MAX = 1000.0
        REPUT_MAX = 10.0

        energia_norm = v.energia / ENERGIA_MAX
        dist_norm = 1.0 - (d / DIST_MAX)
        reput_norm = v.reputacao / REPUT_MAX

        # clamp
        energia_norm = min(1.0, max(0.0, energia_norm))
        dist_norm = min(1.0, max(0.0, dist_norm))
        reput_norm = min(1.0, max(0.0, reput_norm))

        score = (
            self.w1 * reput_norm +
            self.w2 * energia_norm +
            self.w3 * dist_norm
        )

        return score

    # =====================================================
    # SELEÇÃO FINAL
    # =====================================================
    def selecionar(self, candidatos, pos_tarefa):

        if not candidatos:
            return []

        now = time.time()

        for c in candidatos:

            v = c["veiculo"]

            last = self.cooldown.get(v.id, None)
            penalizado = False

            if last is not None and (now - last) < self.cooldown_steps:
                penalizado = True

            score = self.calcular_score(c)

            if penalizado:
                score *= 0.6

            c["score"] = score

        candidatos.sort(key=lambda x: x["score"], reverse=True)

        if random.random() < self.epsilon:
            escolhido = random.choice(candidatos)
        else:
            escolhido = candidatos[0]

        self.cooldown[escolhido["veiculo"].id] = now

        return [escolhido]