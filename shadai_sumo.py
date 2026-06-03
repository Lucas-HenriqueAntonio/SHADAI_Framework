import traci
import xml.etree.ElementTree as ET
import random
import csv
import os

from veiculo import Veiculo
from lider import Plataforma

# ============================================================
# MODO
# ============================================================

MODO = "shadai" # "random | greedy| no_reputation

# ============================================================
# CENÁRIO
# ============================================================

MAX_VEICULOS = 400

ATAQUE = True
TIPO_ATAQUE = "flood"  # "ganancia" | "flood"

if ATAQUE:
    PERCENTUAL_MALICIOSOS = 0.2
    if TIPO_ATAQUE == "ganancia":
        SUFIXO = "ganancia"
    elif TIPO_ATAQUE == "flood":
        SUFIXO = "flood"
else:
    PERCENTUAL_MALICIOSOS = 0.0
    SUFIXO = "normal"

# ============================================================
# CONFIG SUMO
# ============================================================

def ler_tempo_final_sumocfg(sumocfg_path):
    tree = ET.parse(sumocfg_path)
    root = tree.getroot()
    end = root.find("./time/end")

    if end is not None:
        return int(end.attrib["value"])

    return 3600

# ============================================================
# SERVIÇOS
# ============================================================

SERVICOS = [
    {"nome": "Processamento de Imagem Urbana", "energia": 18, "duracao": 60, "preco": 45.0},
    {"nome": "Agregação de Dados de Sensores", "energia": 8, "duracao": 30, "preco": 22.0},
    {"nome": "Reencaminhamento de Dados (Edge Relay)", "energia": 5, "duracao": 25, "preco": 15.0},
    {"nome": "Atualização de Modelos Locais (ML)", "energia": 22, "duracao": 80, "preco": 60.0},
    {"nome": "Diagnóstico Veicular", "energia": 10, "duracao": 40, "preco": 28.0}
]

# ============================================================
# ESTRUTURAS
# ============================================================

veiculos = {}
servicos_ativos = []
plataforma = Plataforma()

# ============================================================
# CSV
# ============================================================

if ATAQUE:
    METRICS_FILE = f"metricas_{MODO}_{MAX_VEICULOS}_{SUFIXO}.csv"
else:
    METRICS_FILE = f"metricas_{MODO}_{MAX_VEICULOS}.csv"

if not os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "modo", "step", "veiculo", "servico",
            "score", "utilidade", "reputacao",
            "energia", "malicioso"
        ])

# ============================================================
# FUNÇÕES
# ============================================================

def registrar_metricas(step, veiculo, servico, score, utilidade):
    with open(METRICS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            MODO,
            step,
            veiculo.id,
            servico["nome"],
            round(score, 4),
            round(utilidade, 4),
            round(veiculo.reputacao, 4),
            round(veiculo.energia, 4),
            getattr(veiculo, "malicioso", False)
        ])

def atualizar_veiculos():

    ids = traci.vehicle.getIDList()

    if len(ids) > MAX_VEICULOS:
        ids = random.sample(ids, MAX_VEICULOS)

    for vid in ids:

        pos = traci.vehicle.getPosition(vid)[0]

        if vid not in veiculos:

            v = Veiculo(
                vid,
                custo_base=random.uniform(3,8),
                capacidade=random.uniform(1,10),
                pos=pos,
                energia=random.uniform(60,120)
            )

            # marcar malicioso
            if random.random() < PERCENTUAL_MALICIOSOS:
                v.malicioso = True
                print(f"[MALICIOSO] Veículo {vid}")

            veiculos[vid] = v

        else:
            veiculos[vid].posicao = pos

def finalizar_servicos(step):

    finalizados = [s for s in servicos_ativos if s["fim"] <= step]

    for s in finalizados:

        v = s["veiculo"]

        sucesso = True if random.random() > 0.1 else False

        v.atualizar_reputacao(sucesso)

        print(f"[Fim Serviço] {v.id} | Rep={v.reputacao}")

        servicos_ativos.remove(s)

def nova_requisicao(step):

    servico = random.choice(SERVICOS)
    plataforma.preco = servico["preco"]
    pos_tarefa = random.uniform(0,1000)

    print(f"\n=== Step {step} | Serviço: {servico['nome']} ===")

    lista_veiculos = list(veiculos.values())
    candidatos = plataforma.ofertar(lista_veiculos, servico, pos_tarefa)

    # ========================================================
    # ATAQUES
    # ========================================================
    if ATAQUE:

        # ---------------------------
        # ATAQUE 1: GANÂNCIA
        # ---------------------------
        if TIPO_ATAQUE == "ganancia":

            for v in lista_veiculos:

                if getattr(v, "malicioso", False):

                    ja_existe = any(c["veiculo"].id == v.id for c in candidatos)

                    if not ja_existe:
                        dist = v.distancia_para(pos_tarefa)

                        candidatos.append({
                            "veiculo": v,
                            "utilidade": 1.0,
                            "custo_total": 50.0,
                            "dist": dist
                        })

                        print(f"[ATAQUE GANÂNCIA] {v.id} entrou como candidato")

        # ---------------------------
        # ATAQUE 2: FLOOD
        # ---------------------------
        elif TIPO_ATAQUE == "flood":

            novos_candidatos = []

            for c in candidatos:

                novos_candidatos.append(c)

                v = c["veiculo"]

                if getattr(v, "malicioso", False):

                    # duplica candidato várias vezes
                    for _ in range(3):  # intensidade do flood
                        novos_candidatos.append({
                            "veiculo": v,
                            "utilidade": c["utilidade"],
                            "custo_total": c["custo_total"],
                            "dist": c["dist"]
                        })

                    print(f"[ATAQUE FLOOD] {v.id} multiplicado")

            candidatos = novos_candidatos

    if not candidatos:
        print("Nenhum candidato")
        return

    # ========================================================
    # SELEÇÃO (INALTERADA)
    # ========================================================

    if MODO == "shadai":

        escolhidos = plataforma.selecionar(candidatos, pos_tarefa)

        if not escolhidos:
            return

        escolhido = escolhidos[0]

    elif MODO == "random":

        escolhido = random.choice(candidatos)
        escolhido["score"] = plataforma.calcular_score(escolhido)

    elif MODO == "greedy":

        escolhido = min(candidatos, key=lambda x: x["custo_total"])
        escolhido["score"] = plataforma.calcular_score(escolhido)

    elif MODO == "no_reputation":

        w1 = plataforma.w1
        plataforma.w1 = 0.0

        escolhidos = plataforma.selecionar(candidatos, pos_tarefa)

        plataforma.w1 = w1

        if not escolhidos:
            return

        escolhido = escolhidos[0]

    else:
        raise ValueError("Modo inválido")

    # ========================================================
    # EXECUÇÃO
    # ========================================================

    v = escolhido["veiculo"]

    print(f"Selecionado: {v.id} | Score={escolhido['score']:.2f}")

    registrar_metricas(
        step,
        v,
        servico,
        escolhido["score"],
        escolhido["utilidade"]
    )

    fim = step + servico["duracao"]

    servicos_ativos.append({
        "veiculo": v,
        "fim": fim,
        "nome": servico["nome"]
    })

    v.executar_servico(servico)

# ============================================================
# LOOP
# ============================================================

def executar_shadai(sumocfg):

    tempo_final = ler_tempo_final_sumocfg(sumocfg)

    traci.start(["sumo-gui","-c",sumocfg])

    for step in range(tempo_final):

        traci.simulationStep()

        atualizar_veiculos()

        finalizar_servicos(step)

        if step > 0 and step % 20 == 0 and len(veiculos) > 0:
            nova_requisicao(step)

    traci.close()
    print("Fim")

if __name__ == "__main__":
    executar_shadai("cidade.sumocfg")