from gettext import find
import hashlib
from collections import Counter
import matplotlib.pyplot as plt


def sha1_hex(data: str) -> str:
    """Calcula SHA1 e devolve em hex, 40 caracteres."""
    return hashlib.sha1(data.encode("utf-8", errors="replace")).hexdigest()


def criar_grafico(counts: Counter):
    """Cria gráfico de barras com o número de coins por categoria."""
    categorias = sorted(counts.keys())
    valores = [counts[cat] for cat in categorias]

    plt.figure(figsize=(10, 6))
    plt.bar(categorias, valores)

    plt.title("Número de Coins por Categoria")
    plt.xlabel("Categoria da Coin")
    plt.ylabel("Quantidade")
    plt.ylim(bottom=0)  # garante que começa no zero
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for i, v in enumerate(valores):
        plt.text(i, v, str(v), ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


def main():
    filename = "deti_coins_v2_vault_2.txt"
    counts = Counter()

    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            # Tipo da moeda (V00, V01, …)
            if "V" in line[:1]:
                coin_type = line[:3]
                counts[coin_type] += 1
                continue

    print("\n========= RESUMO DOS TIPOS DE MOEDAS =========")
    for key in sorted(counts.keys()):
        print(f"{key}: {counts[key]}")
    print("==============================================")

    # Criação do gráfico
    criar_grafico(counts)


if __name__ == "__main__":
    main()
