"""Conferência do teste externo (README 3.3), feita ANTES de qualquer avaliação.

Remove linhas apenas por conteúdo (nunca por previsão de modelo) e gera teste_externo_2026.csv.
Cada remoção fica registrada com o motivo em teste_externo_2026_remocoes.csv.

Uso: .venv\\Scripts\\python.exe conferir_teste_externo.py
"""
import pandas as pd

# Remoções por conteúdo, identificadas pelo link (estável entre execuções).
REMOCOES = {
    # E-farsas: o maior blockquote é o desmentido/nota oficial, não o boato.
    "http://www.e-farsas.com/o-ibge-vai-incluir-o-trafico-de-drogas-no-calculo-do-pib.html":
        "trecho é nota oficial do IBGE (desmentido), não o boato",
    "http://www.e-farsas.com/a-inteligencia-artificial-pode-alterar-votos-na-urna-eletronica.html":
        "trecho é explicação oficial sobre a urna (desmentido), não o boato",
    "http://www.e-farsas.com/a-coca-cola-diminuiu-seu-produto-por-causa-da-politica-de-lula.html":
        "trecho é nota da Coca-Cola (desmentido), não o boato",
    # Boatos.org: versões traduzidas de um boato que já está no conjunto em português.
    "https://www.boatos.org/espanol/un-hombre-compraria-dos-asientos-en-un-avion-por-su-esposa-fallecida-y-lloro-en-el-vuelo.html":
        "trecho em espanhol (tradução de boato já presente em português)",
    "https://www.boatos.org/english/did-a-man-buy-two-airplane-seats-because-of-his-deceased-wife-and-cry-on-a-flight.html":
        "trecho em inglês (tradução de boato já presente em português)",
}

COLUNAS_FINAIS = ["texto", "label", "fonte", "link", "data"]

if __name__ == "__main__":
    bruto = pd.read_csv("teste_externo_2026_bruto.csv")

    # Confere se todo link da lista de remoção existe no bruto (evita remoção "silenciosa" errada).
    faltando = [l for l in REMOCOES if l not in set(bruto["link"])]
    if faltando:
        print("ATENÇÃO: links da lista de remoção que não estão no bruto:")
        for l in faltando:
            print("  ", l)

    # Separa removidos e mantidos.
    mask = bruto["link"].isin(REMOCOES)
    removidos = bruto[mask].assign(motivo=bruto.loc[mask, "link"].map(REMOCOES))
    final = bruto[~mask][COLUNAS_FINAIS].reset_index(drop=True)

    # Links duplicados não deveriam existir; confere.
    assert final["link"].is_unique, "links duplicados no teste final"

    final.to_csv("teste_externo_2026.csv", index=False, encoding="utf-8-sig")
    removidos[["id", "fonte", "link", "motivo", "texto"]].to_csv(
        "teste_externo_2026_remocoes.csv", index=False, encoding="utf-8-sig")

    print(f"Bruto: {len(bruto)} | removidos: {len(removidos)} | final: {len(final)}")
    print(final.groupby(["label", "fonte"]).size())
