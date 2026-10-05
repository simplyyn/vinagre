"""Avaliação final ÚNICA nos testes externos (autorizada pelo responsável em 05/10/2026).

Etapas (nesta ordem, uma vez só):
  congelar -> treina e salva em modelos_finais/ todos os métodos pré-registrados no EXPERIMENTOS.md.
              Não lê nenhum teste.
  avaliar  -> carrega os modelos congelados e avalia em teste_externo_2026.csv (teste 1) e teste2_2026.csv
              (teste 2), com a entrada cortada em 50 palavras (formato do produto). Não treina nada.

Os modelos são treinados exatamente como na validação (fr2021 + ext, só o split de treino do corpus externo).

Uso: .venv\\Scripts\\python.exe avaliacao_final.py <congelar|avaliar>
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

from anls import SEED, novo_tfidf_bal, truncar
from experimentos import N_PRODUTO, embeddings, fabricas_modelo, fontes_de_treino, juntar, treinar
from exp_numeros import so_numeros, tfidf_mais_numeros

PASTA_MODELOS = "modelos_finais"
DADOS = ["fr2021", "ext"]
TESTES = {"Teste 1 (30/09)": "teste_externo_2026.csv", "Teste 2 (02–05/10)": "teste2_2026.csv"}


class EmbLogReg:
    # Embeddings de sentence-transformers + regressão logística (E7), com a mesma interface de predict.
    def __init__(self, nome_modelo, prefixo, C):
        self.nome_modelo, self.prefixo, self.C = nome_modelo, prefixo, C

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        E = embeddings(truncar(pd.Series(list(X)), N_PRODUTO).tolist(), self.nome_modelo, self.prefixo)
        self.lr_ = LogisticRegression(C=self.C, max_iter=5000, class_weight="balanced").fit(E, y)
        return self

    def predict(self, X):
        return self.lr_.predict(embeddings(list(X), self.nome_modelo, self.prefixo))


# Métodos pré-registrados (ordem = ordem dos gráficos). None = modelo salvo antigo, só carregado.
METODOS = {
    "Fake.br (modelo_final salvo)": ("historico", "modelo_final_tfidf.joblib"),
    "Fake.br + FakeRecogna (candidato salvo)": ("historico", "modelo_candidato_fakebr_fakerecogna.joblib"),
    "TF-IDF palavras [base]": ("tfidf", novo_tfidf_bal),
    "TF-IDF caracteres [ESCOLHIDO]": ("tfidf", fabricas_modelo()["char_wb(2,5)"]),
    "TF-IDF palavras + caracteres (sem lowercase)": ("tfidf", fabricas_modelo()["word(1,2) + char_wb(2,5) sem lowercase"]),
    "Embeddings e5-small + LogReg": ("emb", lambda: EmbLogReg("intfloat/multilingual-e5-small", "query: ", 10)),
    "Só números (E8)": ("tfidf", so_numeros("categorias")),
    "TF-IDF palavras + números (E8)": ("tfidf", tfidf_mais_numeros("categorias", 1.0)),
}


def arquivo(nome):
    return os.path.join(PASTA_MODELOS, f"{list(METODOS).index(nome):02d}.joblib")


def congelar():
    # Treina e salva cada método. Não lê nenhum teste.
    os.makedirs(PASTA_MODELOS, exist_ok=True)
    fontes, _, _ = fontes_de_treino()
    X, y = juntar(fontes, DADOS)
    for nome, (tipo, fab) in METODOS.items():
        if tipo == "historico":
            continue
        # Embeddings: treino em 50 palavras sem aumento misto (como na E7); demais: treino misto 30/50/100.
        modelo = fab().fit(X, y) if tipo == "emb" else treinar(X, y, fabrica=fab)
        joblib.dump(modelo, arquivo(nome))
        print("congelado:", nome, "->", arquivo(nome))


def wilson(acertos, n, z=1.96):
    # Intervalo de confiança de 95% (Wilson) para uma proporção.
    p = acertos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    meia = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centro - meia), 100 * (centro + meia)


def ic_f1_bootstrap(y, pred, n_boot=2000):
    # IC 95% do F1 macro por bootstrap (reamostragem dos itens do teste).
    rng = np.random.default_rng(SEED)
    y, pred = np.asarray(y), np.asarray(pred)
    v = [f1_score(y[i], pred[i], average="macro") for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return np.percentile(v, 2.5) * 100, np.percentile(v, 97.5) * 100


def metricas(y, pred, fonte):
    acertos = int((y == pred).sum())
    lo, hi = wilson(acertos, len(y))
    f_lo, f_hi = ic_f1_bootstrap(y, pred)
    (tn_fake, fp_true), (fn_fake, tp_true) = confusion_matrix(y, pred, labels=["fake", "true"])
    linha = {"n": len(y), "accuracy": acertos / len(y) * 100, "acc_ic_baixo": lo, "acc_ic_alto": hi,
             "f1_macro": f1_score(y, pred, average="macro") * 100, "f1_ic_baixo": f_lo, "f1_ic_alto": f_hi,
             "bal_acc": balanced_accuracy_score(y, pred) * 100,
             "recall_fake": recall_score(y, pred, pos_label="fake") * 100,
             "recall_true": recall_score(y, pred, pos_label="true") * 100,
             "fake_como_fake": tn_fake, "fake_como_true": fp_true, "true_como_fake": fn_fake, "true_como_true": tp_true}
    for f, ok in pd.Series(y == pred).groupby(np.asarray(fonte)):
        linha[f"acerto_{f}"] = ok.mean() * 100
        linha[f"n_{f}"] = len(ok)
    return linha


def avaliar():
    # Avaliação única. Recusa rodar de novo se já existir resultado (o teste só pode ser avaliado uma vez).
    if os.path.exists("resultados_teste_final.csv"):
        sys.exit("resultados_teste_final.csv já existe: a avaliação final já foi feita.")
    testes = {k: pd.read_csv(v) for k, v in TESTES.items()}
    testes["Testes 1 + 2"] = pd.concat(testes.values(), ignore_index=True)
    linhas, previsoes = [], []
    for nome, (tipo, fab) in METODOS.items():
        modelo = joblib.load(fab if tipo == "historico" else arquivo(nome))
        for teste, df in testes.items():
            X = truncar(df["texto"], N_PRODUTO)
            pred = np.asarray(modelo.predict(X))
            linhas.append({"metodo": nome, "teste": teste, **metricas(df["label"].values, pred, df["fonte"])})
            if teste != "Testes 1 + 2":
                previsoes.append(df.assign(teste=teste, metodo=nome, previsto=pred, texto_50=X))
        r = [l for l in linhas if l["metodo"] == nome]
        print(f"{nome}: " + " | ".join(f"{l['teste']}: F1 {l['f1_macro']:.1f} acc {l['accuracy']:.1f} "
                                        f"[{l['acc_ic_baixo']:.0f}-{l['acc_ic_alto']:.0f}]" for l in r))
    pd.DataFrame(linhas).to_csv("resultados_teste_final.csv", index=False, encoding="utf-8-sig")
    pd.concat(previsoes).to_csv("previsoes_teste_final.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    {"congelar": congelar, "avaliar": avaliar}[sys.argv[1]]()
