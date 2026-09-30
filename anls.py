"""Pipeline do projeto de classificação de notícias falsas (Fake.br).

Segue a seção "Pipeline" do README. Por enquanto executa só os números de controle:
  - baseline só tamanho (CV agrupada, texto completo): ~94,17% de accuracy;
  - TF-IDF com N=50 palavras (CV agrupada): ~89,89% de F1 macro.

Uso: .venv\\Scripts\\python.exe anls.py
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

SEED = 42
ARQ_FAKEBR = "dataset_original(1).csv"


# ---------------------------------------------------------------------------
# Carregamento e limpeza do Fake.br (README 3.1)
# ---------------------------------------------------------------------------

def carregar_fakebr(caminho=ARQ_FAKEBR):
    # Leitura dos dados originais.
    df = pd.read_csv(caminho, encoding="utf-8")

    # Labels em minúsculas; "false" vira "fake".
    df["label"] = df["label"].astype(str).str.strip().str.lower().replace({"false": "fake"})

    # Remove BOM e colapsa espaços múltiplos.
    df["text"] = (df["text"].astype(str)
                  .str.replace("﻿", "", regex=False)
                  .str.replace(r"\s+", " ", regex=True)
                  .str.strip())

    # index -> pair_id (cada fake i tem uma true i da mesma categoria).
    df = df.rename(columns={"index": "pair_id"})

    # Par 69 removido: a true dos pares 61 e 69 é o mesmo texto.
    df = df[df["pair_id"] != 69].reset_index(drop=True)
    return df


def conferir_fakebr(df):
    # Checagens de sanidade contra os números do README.
    print(f"Linhas: {len(df)} (esperado 7198)")
    print("Classes:", df["label"].value_counts().to_dict(), "(esperado 3599 cada)")
    pares = df.groupby("pair_id")["label"].agg(lambda s: tuple(sorted(s)))
    print("Pares completos fake/true:", (pares == ("fake", "true")).sum(), "de", len(pares))
    cat_igual = df.groupby("pair_id")["category"].nunique().eq(1).mean()
    print(f"Pares com mesma categoria: {cat_igual:.1%}")
    print("Textos com BOM restantes:", df["text"].str.contains("﻿").sum())


# ---------------------------------------------------------------------------
# Funções principais (README 4.1)
# ---------------------------------------------------------------------------

# Truncamento em N palavras.
def truncar(s, N):
    return s.str.split().str[:N].str.join(" ")


# Truncamento misto (aumento de dados): cada texto aparece cortado em 30, 50 e 100 palavras.
def aumentar(textos, labels):
    t, l = pd.Series(list(textos)), pd.Series(list(labels))
    return (pd.concat([truncar(t, n) for n in (30, 50, 100)], ignore_index=True),
            pd.concat([l] * 3, ignore_index=True))


# Modelo padrão do projeto.
def novo_tfidf():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), dtype=np.float32)),
        ("svm", LinearSVC(C=1.0, max_iter=20000))
    ])


# Mesmo modelo com classes balanceadas (usado quando o treino mistura corpora).
def novo_tfidf_bal():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), dtype=np.float32)),
        ("svm", LinearSVC(C=1.0, max_iter=20000, class_weight="balanced"))
    ])


# CV oficial: pares fake/true sempre no mesmo fold.
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)


# Filtra pares em que as duas notícias têm pelo menos N palavras e trunca em N.
def base_truncada(df, N):
    n_palavras = df["text"].str.split().str.len()
    pares_ok = n_palavras.groupby(df["pair_id"]).min() >= N
    d = df[df["pair_id"].isin(pares_ok[pares_ok].index)].copy()
    d["text"] = truncar(d["text"], N)
    return d.reset_index(drop=True)


# Roda a CV agrupada e devolve média e desvio de cada métrica (em %).
def rodar_cv(modelo, X, y, groups):
    r = cross_validate(modelo, X, y, groups=groups, cv=cv,
                       scoring=["accuracy", "f1_macro"], n_jobs=-1)
    return {m: (100 * r[f"test_{m}"].mean(), 100 * r[f"test_{m}"].std())
            for m in ("accuracy", "f1_macro")}


def mostrar(nome, res):
    acc, f1 = res["accuracy"], res["f1_macro"]
    print(f"{nome}: accuracy {acc[0]:.2f} (std {acc[1]:.2f}) | F1 macro {f1[0]:.2f} (std {f1[1]:.2f})")


# ---------------------------------------------------------------------------
# FakeRecogna 2.0 (README 3.2; notebook células 42–55)
# ---------------------------------------------------------------------------

ARQ_FR2 = "dados_externos/fakerecogna2_abstrativa.parquet"


def carregar_fakerecogna():
    # Baixa uma vez do Hugging Face e guarda em parquet para não depender da rede.
    import os
    if not os.path.exists(ARQ_FR2):
        from datasets import load_dataset
        ds = load_dataset("recogna-nlp/fakerecogna2-abstrativa")
        os.makedirs(os.path.dirname(ARQ_FR2), exist_ok=True)
        pd.concat([ds[s].to_pandas() for s in ds.keys()], ignore_index=True).to_parquet(ARQ_FR2)
    df_rec2 = pd.read_parquet(ARQ_FR2)

    # Label: 1 = fake, 0 = true (invertido).
    df_rec2["label"] = df_rec2["Label"].map({1: "fake", 0: "true"})
    txt_r2 = df_rec2["Noticia"].astype(str).str.strip()

    # Vocabulário típico de checagem e boato citado entre aspas (>= 20 palavras).
    marcadores = r"\b(?:fake|falso|falsa|boato|checagem|verifica|enganos[oa]|desmentid[oa]|viraliz\w*|circula\w*|mentira)\b"
    df_rec2["tem_checagem"] = txt_r2.str.lower().str.contains(marcadores, regex=True)
    df_rec2["rumor_citado"] = txt_r2.str.extract(r'^[“"](.+?)[”"]')[0]
    n_pal_citado = df_rec2["rumor_citado"].str.split().str.len().fillna(0)
    df_rec2["tipo"] = np.select(
        [df_rec2["label"] == "true", n_pal_citado >= 20, df_rec2["tem_checagem"]],
        ["true_resumida", "fake_boato_citado", "fake_texto_checagem"], default="fake_outros")

    # Texto usado: só o conteúdo das aspas no boato citado; nos demais, a notícia inteira.
    df_rec2["texto_base"] = np.where(df_rec2["tipo"] == "fake_boato_citado", df_rec2["rumor_citado"], txt_r2)

    # Ano: da coluna Data ou, se faltar, da URL.
    data_txt = df_rec2["Data"].astype(str).str.extract(r"(\d{2}/\d{2}/\d{4})")[0]
    df_rec2["ano"] = pd.to_datetime(data_txt, format="%d/%m/%Y", errors="coerce").dt.year
    df_rec2["ano"] = df_rec2["ano"].fillna(df_rec2["URL"].astype(str).str.extract(r"/(20\d{2})/")[0].astype(float))
    df_rec2["dominio"] = df_rec2["URL"].astype(str).str.extract(r"https?://(?:www\.)?([^/]+)")[0].str.lower()
    return df_rec2


def treino_candidato(df_modelo, df_rec2):
    # Treino do candidato: Fake.br completo + FakeRecogna 2020–2021 (fake sem checagem + true balanceadas).
    fr = df_rec2[df_rec2["ano"].isin([2020, 2021])]
    fr_fake = fr[fr["tipo"].isin(["fake_boato_citado", "fake_outros"])]
    fr_true = fr[fr["tipo"] == "true_resumida"].sample(len(fr_fake), random_state=SEED)
    X = pd.concat([df_modelo["text"], fr_fake["texto_base"], fr_true["texto_base"]], ignore_index=True)
    y = pd.concat([df_modelo["label"], fr_fake["label"], fr_true["label"]], ignore_index=True)
    return X, y


# ---------------------------------------------------------------------------
# Números de controle (README 5.1 e 5.2)
# ---------------------------------------------------------------------------

def controle_tamanho(df_modelo):
    # Baseline só tamanho: log1p(n_caracteres) + StandardScaler + LinearSVC, texto completo.
    modelo = Pipeline([
        ("log", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(C=1.0, max_iter=20000))
    ])
    y, g = df_modelo["label"], df_modelo["pair_id"]

    # n_caracteres = len(text) já limpo (com espaços e pontuação).
    X_len = df_modelo[["text"]].assign(n=df_modelo["text"].str.len())[["n"]].to_numpy(float)
    res = rodar_cv(modelo, X_len, y, g)
    mostrar("Controle 1 - só tamanho (len(text))", res)

    # Variante com a coluna n_characters do Fake.br (só alfanuméricos), para comparação.
    X_col = df_modelo[["n_characters"]].to_numpy(float)
    mostrar("           - só tamanho (n_characters)", rodar_cv(modelo, X_col, y, g))
    return res


def controle_tfidf_n50(df_modelo):
    # TF-IDF em textos truncados em 50 palavras (só pares com as duas >= 50 palavras).
    d = base_truncada(df_modelo, 50)
    print(f"N=50: {d['pair_id'].nunique()} pares (esperado 3500)")
    res = rodar_cv(novo_tfidf(), d["text"], d["label"], d["pair_id"])
    mostrar("Controle 2 - TF-IDF N=50", res)
    return res


# ---------------------------------------------------------------------------
# Inspeção dos modelos salvos (README 4.2)
# ---------------------------------------------------------------------------

def inspecionar_modelo(caminho):
    # Mostra os passos e parâmetros relevantes do Pipeline salvo.
    m = joblib.load(caminho)
    print(f"\n{caminho}: {type(m).__name__}")
    if isinstance(m, Pipeline):
        for nome, passo in m.steps:
            p = passo.get_params()
            chaves = ["max_features", "ngram_range", "min_df", "sublinear_tf", "lowercase",
                      "C", "class_weight", "max_iter"]
            print(f"  {nome}: {type(passo).__name__}", {k: p[k] for k in chaves if k in p})
        tfidf = m.steps[0][1]
        if hasattr(tfidf, "vocabulary_"):
            print("  vocabulário:", len(tfidf.vocabulary_))
        clf = m.steps[-1][1]
        if hasattr(clf, "classes_"):
            print("  classes:", list(clf.classes_))
    return m


if __name__ == "__main__":
    df = carregar_fakebr()
    conferir_fakebr(df)

    # Base dos modelos.
    df_modelo = df.copy()
    df_modelo["n_links"] = df_modelo["n_links"].fillna(0)

    etapa = sys.argv[1] if len(sys.argv) > 1 else "controle"
    if etapa in ("controle", "tudo"):
        controle_tamanho(df_modelo)
        controle_tfidf_n50(df_modelo)
    if etapa in ("modelos", "tudo"):
        inspecionar_modelo("modelo_final_tfidf.joblib")
        inspecionar_modelo("modelo_candidato_fakebr_fakerecogna.joblib")
