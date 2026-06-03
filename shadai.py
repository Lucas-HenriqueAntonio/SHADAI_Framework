# shadai_prototipo.py

import numpy as np
import random
from veiculo import Veiculo
from lider import Plataforma

np.random.seed(42)
random.seed(42)

# ===== Config inicial =====
num_veiculos = 10
custos = np.random.uniform(5, 50, num_veiculos)
capacidades = np.random.uniform(1, 10, num_veiculos)
posicoes = np.random.uniform(0, 100, num_veiculos)  # posição inicial em "km"
energias = np.random.uniform(50, 100, num_veiculos) # energia inicial

veiculos = [Veiculo(i, custos[i], capacidades[i], posicoes[i], energias[i]) for i in range(num_veiculos)]

plataforma = Plataforma(preco_inicial=5.0, alvo_participantes=3, executores_por_tarefa=1)

rodadas = 5
for rodada in range(1, rodadas + 1):
    print(f"\n=== Rodada {rodada} ===")
    # posição da tarefa (ex: lugar onde precisa executar serviço)
    pos_tarefa = np.random.uniform(0, 100)

    # encontrar preço ótimo para ter candidatos
    limiares = [v.custo / v.capacidade for v in veiculos]
    preco_max_estimado = max(limiares) + 10
    preco_encontrado, participantes, _ = plataforma.buscar_preco_binario(
        veiculos, preco_min=0.0, preco_max=preco_max_estimado, eps=0.01, max_iter=50
    )
    if preco_encontrado is not None:
        plataforma.preco = preco_encontrado

    candidatos = plataforma.ofertar(veiculos)
    escolhidos = plataforma.selecionar(candidatos, posicao_tarefa=pos_tarefa)

    print(f"Preço={plataforma.preco:.2f} | Candidatos={len(candidatos)} | Executores={len(escolhidos)}")
    for esc in escolhidos:
        v = esc["veiculo"]
        print(f" -> Veículo {v.id} | Reputação={v.reputacao} | Energia={v.energia:.1f} | Ganho Acum={v.ganhos_acumulados:.2f}")

# ===== Ranking Final =====
print("\n=== Ranking Final ===")
for v in sorted(veiculos, key=lambda x: (x.reputacao, x.ganhos_acumulados), reverse=True):
    print(f"Veículo {v.id} | Reputação={v.reputacao} | Ganho={v.ganhos_acumulados:.2f} | Sucessos={v.tarefas_realizadas} | Falhas={v.tarefas_falhas}")
