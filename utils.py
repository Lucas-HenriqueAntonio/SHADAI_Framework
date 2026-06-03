# utils.py

def mostrar_resultados(resultados):
    """
    Exibe no console os resultados da interação líder vs veículos.
    """
    for r in resultados:
        status = "Participa" if r["participa"] else "Recusa"
        print(f"Veículo {r['veiculo']} -> Payoff={r['payoff']:.2f} -> {status}")
