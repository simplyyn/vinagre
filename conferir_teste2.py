"""Conferência do 2º teste externo, feita ANTES de qualquer avaliação (mesma regra do teste 1).

Remove linhas apenas por conteúdo (nunca por previsão de modelo) e gera teste2_2026.csv.
Cada remoção fica registrada com o motivo em teste2_2026_remocoes.csv.

Uso: .venv\\Scripts\\python.exe conferir_teste2.py
"""
import pandas as pd

COLUNAS_FINAIS = ["texto", "label", "fonte", "link", "data"]

# Remoções por link (conteúdo conferido à mão).
REMOCOES_LINK = {
    "https://agenciabrasil.ebc.com.br/geral/noticia/2026-10/mega-sena-acumula-e-proximo-concurso-pode-pagar-r-92-milhoes":
        "trecho é cabeçalho da página (editoria, título, assinatura), não o início da matéria",
}


def motivo(r):
    # Regras por conteúdo; as fake (17 boatos citados) foram lidas e nenhuma é texto da agência.
    if r["link"] in REMOCOES_LINK:
        return REMOCOES_LINK[r["link"]]
    if "/especial-publicitario/" in r["link"]:
        return "G1 especial publicitário (anúncio, não notícia)"
    if "foi o candidato a presidente mais votado em" in r["texto"]:
        return "G1 página automática de resultado por cidade (texto-modelo, não matéria)"
    return ""


if __name__ == "__main__":
    bruto = pd.read_csv("teste2_2026_bruto.csv")
    bruto = bruto[bruto["motivo"].isna()].copy()   # descartes automáticos da coleta ficam no bruto
    bruto["motivo"] = bruto.apply(motivo, axis=1)
    mask = bruto["motivo"] != ""
    final = bruto[~mask][COLUNAS_FINAIS].reset_index(drop=True)
    assert final["link"].is_unique, "links duplicados no teste 2"

    final.to_csv("teste2_2026.csv", index=False, encoding="utf-8-sig")
    bruto[mask][["id", "fonte", "link", "motivo", "texto"]].to_csv("teste2_2026_remocoes.csv", index=False,
                                                                   encoding="utf-8-sig")
    print(f"Após coleta: {len(bruto)} | removidos na conferência: {mask.sum()} | final: {len(final)}")
    print(bruto[mask]["motivo"].value_counts())
    print(final.groupby(["label", "fonte"]).size())
