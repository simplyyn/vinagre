"""Treina o modelo escolhido do zero, exatamente como foi feito no projeto.

Receita (detalhes e motivos em COMO_FOI_TREINADO.md):
  1. Dados: FakeRecogna 2.0 (anos 2020–2021) + parte de treino do corpus externo (dados/corpus_externo.csv).
  2. Aumento de dados: cada texto entra 3 vezes, cortado em 30, 50 e 100 palavras.
  3. Modelo: TF-IDF de n-gramas de caracteres (char_wb, 2 a 5) + LinearSVC com classes balanceadas.

O FakeRecogna é baixado do Hugging Face na primeira execução (precisa de internet e do pacote `datasets`).
Saída: modelo/modelo_retreinado.joblib (não sobrescreve o modelo original).

Uso: python treinar_do_zero.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED = 42
PASTA = os.path.dirname(os.path.abspath(__file__))
ARQ_FR2 = os.path.join(PASTA, "dados", "fakerecogna2_abstrativa.parquet")


# ---------------------------------------------------------------------------
# 1. Dados
# ---------------------------------------------------------------------------

def carregar_fakerecogna():
    # Baixa uma vez do Hugging Face e guarda em parquet.
    if not os.path.exists(ARQ_FR2):
        from datasets import load_dataset
        ds = load_dataset("recogna-nlp/fakerecogna2-abstrativa")
        pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True).to_parquet(ARQ_FR2)
    df = pd.read_parquet(ARQ_FR2)

    # Label do FakeRecogna: 1 = fake, 0 = true (invertido em relação à intuição).
    df["label"] = df["Label"].map({1: "fake", 0: "true"})
    txt = df["Noticia"].astype(str).str.strip()

    # Tipos de texto fake:
    #   boato_citado    -> começa com o boato entre aspas (>= 20 palavras): usa só o que está nas aspas;
    #   texto_checagem  -> tem vocabulário de agência de checagem ("falso", "boato"...): DESCARTADO,
    #                      porque descreve o boato em vez de ser o boato;
    #   outros          -> demais fake, usadas inteiras.
    marcadores = r"\b(?:fake|falso|falsa|boato|checagem|verifica|enganos[oa]|desmentid[oa]|viraliz\w*|circula\w*|mentira)\b"
    df["tem_checagem"] = txt.str.lower().str.contains(marcadores, regex=True)
    df["rumor_citado"] = txt.str.extract(r'^[“"](.+?)[”"]')[0]
    n_pal_citado = df["rumor_citado"].str.split().str.len().fillna(0)
    df["tipo"] = np.select(
        [df["label"] == "true", n_pal_citado >= 20, df["tem_checagem"]],
        ["true_resumida", "fake_boato_citado", "fake_texto_checagem"], default="fake_outros")
    df["texto_base"] = np.where(df["tipo"] == "fake_boato_citado", df["rumor_citado"], txt)

    # Ano: da coluna Data ou, se faltar, da URL.
    data_txt = df["Data"].astype(str).str.extract(r"(\d{2}/\d{2}/\d{4})")[0]
    df["ano"] = pd.to_datetime(data_txt, format="%d/%m/%Y", errors="coerce").dt.year
    df["ano"] = df["ano"].fillna(df["URL"].astype(str).str.extract(r"/(20\d{2})/")[0].astype(float))
    return df


def fakerecogna_2020_2021(df):
    # Só 2020–2021: são os únicos anos com as duas classes em quantidade (evita o atalho "época = fake").
    fr = df[df["ano"].isin([2020, 2021])]
    fake = fr[fr["tipo"].isin(["fake_boato_citado", "fake_outros"])]
    true = fr[fr["tipo"] == "true_resumida"].sample(len(fake), random_state=SEED)  # mesmo número de true
    return (pd.concat([fake["texto_base"], true["texto_base"]], ignore_index=True),
            pd.concat([fake["label"], true["label"]], ignore_index=True))


def corpus_externo():
    # Boatos.org e E-farsas (fake) x G1 e Agência Brasil (true), 2022–2026. Coluna `split` já definida.
    c = pd.read_csv(os.path.join(PASTA, "dados", "corpus_externo.csv"))
    return c[c["split"] == "treino"].reset_index(drop=True), c[c["split"] == "val"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Aumento de dados por truncamento
# ---------------------------------------------------------------------------

def truncar(s, n):
    return s.str.split().str[:n].str.join(" ")


def aumentar(textos, labels):
    # Cada texto aparece cortado em 30, 50 e 100 palavras: o modelo aprende com trechos de vários tamanhos
    # e não pode usar o tamanho do texto como pista.
    t, l = pd.Series(list(textos)), pd.Series(list(labels))
    return (pd.concat([truncar(t, n) for n in (30, 50, 100)], ignore_index=True),
            pd.concat([l] * 3, ignore_index=True))


# ---------------------------------------------------------------------------
# 3. Modelo
# ---------------------------------------------------------------------------

def novo_modelo():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",      # n-gramas de caracteres, sem atravessar espaços entre palavras
            ngram_range=(2, 5),      # pedaços de 2 a 5 caracteres
            max_features=200000,     # os 200 mil pedaços mais frequentes
            sublinear_tf=True,       # 1 + log(contagem): repetir um termo pesa menos
            dtype=np.float32)),      # (lowercase=True é o padrão: tudo vira minúscula)
        ("svm", LinearSVC(C=1.0, max_iter=20000, class_weight="balanced")),
    ])


if __name__ == "__main__":
    print("Carregando FakeRecogna (download na primeira vez)...")
    X_fr, y_fr = fakerecogna_2020_2021(carregar_fakerecogna())
    ext_tr, ext_val = corpus_externo()
    print(f"FakeRecogna 2020–21: {len(X_fr)} textos {y_fr.value_counts().to_dict()}")
    print(f"Corpus externo (treino): {len(ext_tr)} textos {ext_tr['label'].value_counts().to_dict()}")

    # Mesma ordem do projeto: FakeRecogna primeiro, depois o externo.
    X = pd.concat([X_fr, ext_tr["texto"]], ignore_index=True)
    y = pd.concat([y_fr, ext_tr["label"]], ignore_index=True)
    X_aum, y_aum = aumentar(X, y)
    print(f"Treino após aumento 30/50/100: {len(X_aum)} trechos. Treinando...")
    modelo = novo_modelo().fit(X_aum, y_aum)

    # Conferência na validação externa (formato do produto: até 50 palavras). Esperado: F1 macro ~91,4.
    pred = modelo.predict(truncar(ext_val["texto"], 50))
    print(f"F1 macro na validação externa: {f1_score(ext_val['label'], pred, average='macro') * 100:.2f}")

    saida = os.path.join(PASTA, "modelo", "modelo_retreinado.joblib")
    joblib.dump(modelo, saida)
    print("Salvo em", saida)
