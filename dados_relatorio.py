"""Junta os números do relatório (validação, testes e exemplos) e gera relatorio.html a partir do modelo.

Os números vêm dos CSVs de resultado; nada é digitado à mão (exceto os resultados históricos do README, citados
com a seção de origem).

Uso: .venv\\Scripts\\python.exe dados_relatorio.py
"""
import json

import pandas as pd

ESCOLHIDO = "TF-IDF caracteres [ESCOLHIDO]"


def linhas(df, grupo, cols=("experimento", "f1_macro", "bal_acc", "recall_fake", "recall_true")):
    d = df[df["grupo"] == grupo]
    fontes = [c for c in d.columns if c.startswith("acerto_") and d[c].notna().any()]
    return [{**{c: r[c] for c in cols}, "fontes": {f[7:]: r[f] for f in fontes if pd.notna(r[f])}}
            for _, r in d.iterrows()]


if __name__ == "__main__":
    val = pd.read_csv("resultados_experimentos.csv")
    teste = pd.read_csv("resultados_teste_final.csv")
    prev = pd.read_csv("previsoes_teste_final.csv")

    # Exemplos de erro do modelo escolhido (para mostrar limites reais no relatório).
    p = prev[prev["metodo"] == ESCOLHIDO]
    erros = p[p["label"] != p["previsto"]][["teste", "fonte", "label", "previsto", "texto_50"]]

    dados = {
        # Históricos do README (seções 3.1, 5.2 e 5.8), já publicados no projeto.
        "historico": {
            "tamanho": {"fake": [696, 956, 1354, 1122], "true": [3873, 5583, 8594, 6675]},
            "truncamento": [{"N": 30, "so_tamanho": 55.77, "tfidf": 87.35}, {"N": 50, "so_tamanho": 56.90, "tfidf": 89.88},
                            {"N": 100, "so_tamanho": 55.45, "tfidf": 90.94}],
            "baseline_tamanho_texto_completo": 94.17,
            "treino_completo_recall_true_30": 0.15,
        },
        # E8 descritivo (saída de `exp_numeros.py descritivo`, registrada no EXPERIMENTOS.md).
        "numeros": [{"corpus": "Fake.br", "fake": 2.13, "true": 3.12, "auc": 0.605},
                    {"corpus": "FakeRecogna 2020–21", "fake": 3.40, "true": 3.79, "auc": 0.564},
                    {"corpus": "Corpus externo", "fake": 2.41, "true": 5.21, "auc": 0.758}],
        "e3": linhas(val, "E3 base"), "e4": linhas(val, "E4 dados"), "e5": linhas(val, "E5 modelo"),
        "e6": linhas(val, "E6 fontes"), "e7": linhas(val, "E7 embeddings"), "e8": linhas(val, "E8 números"),
        "teste": teste.to_dict(orient="records"),
        "erros": erros.to_dict(orient="records"),
        "n_testes": prev.drop_duplicates(["teste", "link"]).groupby(["teste", "label", "fonte"]).size()
                        .reset_index(name="n").to_dict(orient="records"),
    }
    js = json.dumps(dados, ensure_ascii=False, default=float).replace("NaN", "null")
    with open("relatorio_modelo.html", encoding="utf-8") as f:
        html = f.read().replace("/*__DADOS__*/null", js)
    with open("relatorio.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("relatorio.html gerado;", len(erros), "erros do modelo escolhido listados")
