"""Experimentos para aumentar o F1 em notícias externas (ver EXPERIMENTOS.md).

Toda escolha é feita na VALIDAÇÃO EXTERNA (dados_externos/). O teste_externo_2026.csv não é lido aqui,
exceto os textos (sem labels) para descontaminar o corpus de quase-duplicatas.

Uso: .venv\\Scripts\\python.exe experimentos.py <etapa>
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity

from anls import (SEED, aumentar, carregar_fakebr, carregar_fakerecogna, novo_tfidf, novo_tfidf_bal,
                  treino_candidato, truncar)

PASTA = "dados_externos"
ARQ_RESULTADOS = "resultados_experimentos.csv"
N_PRODUTO = 50


# ---------------------------------------------------------------------------
# Corpus externo: limpeza, descontaminação e divisão treino/validação
# ---------------------------------------------------------------------------

def preparar_corpus_externo():
    c = pd.read_csv(os.path.join(PASTA, "corpus_externo_bruto.csv"))
    c["texto"] = c["texto"].fillna("")
    n0 = len(c)

    # Mínimos do notebook (célula 59): boato >= 20 palavras, início de matéria >= 30.
    minimo = np.where(c["label"] == "fake", 20, 30)
    c = c[c["n_palavras"] >= minimo]
    n1 = len(c)

    # Textos repetidos (mesmo boato em posts diferentes) ficam uma vez só.
    c = c.drop_duplicates("texto")
    n2 = len(c)

    # Descontaminação: remove itens quase idênticos a qualquer texto do teste (cosseno >= 0,8).
    # Usa apenas os textos do teste (sem label e sem modelo), só para evitar vazamento de conteúdo.
    teste = pd.read_csv("teste_externo_2026_bruto.csv")["texto"]
    vec = TfidfVectorizer().fit(pd.concat([c["texto"], teste]))
    sim = cosine_similarity(vec.transform(c["texto"]), vec.transform(teste)).max(axis=1)
    c = c[sim < 0.8]
    n3 = len(c)

    # Data de publicação.
    c["dt"] = pd.to_datetime(c["data"], utc=True, errors="coerce", format="mixed")

    # Divisão: fake -> 30% mais recentes de cada agência na validação; true -> 30% aleatório por fonte.
    c["split"] = "treino"
    for fonte, g in c.groupby("fonte"):
        n_val = int(round(0.3 * len(g)))
        if g["label"].iloc[0] == "fake":
            idx_val = g.sort_values("dt", ascending=False).index[:n_val]
        else:
            idx_val = g.sample(n_val, random_state=SEED).index
        c.loc[idx_val, "split"] = "val"

    print(f"Corpus externo: {n0} brutos -> {n1} com mínimo de palavras -> {n2} sem duplicados "
          f"-> {n3} após descontaminação")
    print(c.groupby(["split", "label", "fonte"]).agg(n=("link", "size"),
                                                     de=("dt", "min"), ate=("dt", "max")))
    c.to_csv(os.path.join(PASTA, "corpus_externo.csv"), index=False, encoding="utf-8-sig")
    return c


def carregar_corpus_externo():
    c = pd.read_csv(os.path.join(PASTA, "corpus_externo.csv"))
    return c[c["split"] == "treino"].reset_index(drop=True), c[c["split"] == "val"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Avaliação e registro
# ---------------------------------------------------------------------------

def avaliar(modelo, val, nome, grupo, obs="", pred_fn=None):
    # Avalia no formato do produto (até 50 palavras) e registra no CSV de resultados.
    X = truncar(val["texto"], N_PRODUTO)
    pred = pred_fn(X) if pred_fn else modelo.predict(X)
    y = val["label"]
    linha = {
        "grupo": grupo, "experimento": nome,
        "f1_macro": f1_score(y, pred, average="macro") * 100,
        "bal_acc": balanced_accuracy_score(y, pred) * 100,
        "recall_fake": recall_score(y, pred, pos_label="fake") * 100,
        "recall_true": recall_score(y, pred, pos_label="true") * 100,
        "n_val": len(val), "obs": obs, "quando": time.strftime("%Y-%m-%d %H:%M"),
    }
    # Acerto por fonte (para ver se o ganho vem de uma fonte só).
    for fonte, g in val.assign(ok=(pred == y.values)).groupby("fonte"):
        linha[f"acerto_{fonte}"] = g["ok"].mean() * 100
    df = pd.DataFrame([linha])
    if os.path.exists(ARQ_RESULTADOS):
        df = pd.concat([pd.read_csv(ARQ_RESULTADOS), df], ignore_index=True)
    df.to_csv(ARQ_RESULTADOS, index=False, encoding="utf-8-sig")
    print(f"[{grupo}] {nome}: F1 {linha['f1_macro']:.2f} | bal_acc {linha['bal_acc']:.2f} | "
          f"rec_fake {linha['recall_fake']:.2f} | rec_true {linha['recall_true']:.2f} | "
          + " ".join(f"{k[7:]} {v:.1f}" for k, v in linha.items() if k.startswith("acerto_")))
    return linha


# ---------------------------------------------------------------------------
# Fontes de dados de treino
# ---------------------------------------------------------------------------

def fontes_de_treino():
    # Fake.br completo (todas as fontes), FakeRecogna 2020–21 (como no candidato), FakeRecogna todos os anos
    # e o treino do corpus externo. Cada uma vira (X, y).
    df_modelo = carregar_fakebr()
    df_rec2 = carregar_fakerecogna()
    ext_tr, ext_val = carregar_corpus_externo()

    fb = (df_modelo["text"], df_modelo["label"])

    X_c, y_c = treino_candidato(df_modelo, df_rec2)
    fr2021 = (X_c.iloc[len(df_modelo):], y_c.iloc[len(df_modelo):])

    fake_all = df_rec2[df_rec2["tipo"].isin(["fake_boato_citado", "fake_outros"])]
    true_all = df_rec2[df_rec2["tipo"] == "true_resumida"].sample(len(fake_all), random_state=SEED)
    frall = (pd.concat([fake_all["texto_base"], true_all["texto_base"]]),
             pd.concat([fake_all["label"], true_all["label"]]))

    ext = (ext_tr["texto"], ext_tr["label"])
    return {"fb": fb, "fr2021": fr2021, "frall": frall, "ext": ext}, ext_tr, ext_val


def juntar(fontes, nomes, repetir_ext=1):
    # Concatena as fontes pedidas; "ext" pode ser repetido para ganhar peso frente aos corpora grandes.
    Xs, ys = [], []
    for n in nomes:
        X, y = fontes[n]
        k = repetir_ext if n == "ext" else 1
        Xs += [X] * k
        ys += [y] * k
    return pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)


def treinar(X, y, fabrica=novo_tfidf_bal, misto=True):
    # Treino no esquema vigente: truncamento misto 30/50/100 (decisão da seção 5.8).
    if misto:
        X, y = aumentar(X, y)
    return fabrica().fit(X, y)


# ---------------------------------------------------------------------------
# Etapas
# ---------------------------------------------------------------------------

def etapa_base():
    # E3: ponto de partida — modelos salvos na validação externa.
    import joblib
    _, val = carregar_corpus_externo()
    avaliar(joblib.load("modelo_final_tfidf.joblib"), val, "modelo_final (só Fake.br)", "E3 base")
    avaliar(joblib.load("modelo_candidato_fakebr_fakerecogna.joblib"), val,
            "modelo_candidato (Fake.br + FR 2020-21)", "E3 base")


def etapa_dados():
    # E4: qual combinação de dados de treino generaliza melhor para o formato externo.
    fontes, _, val = fontes_de_treino()
    combos = [["fb"], ["fb", "fr2021"], ["fb", "frall"], ["fr2021"], ["frall"], ["ext"],
              ["fb", "ext"], ["fr2021", "ext"], ["frall", "ext"], ["fb", "fr2021", "ext"], ["fb", "frall", "ext"]]
    for nomes in combos:
        X, y = juntar(fontes, nomes)
        avaliar(treinar(X, y), val, " + ".join(nomes), "E4 dados", obs=f"n_treino={len(X)}")
    # Peso do corpus externo (repetição) nas combinações com corpora grandes.
    for nomes in (["fb", "fr2021", "ext"], ["fb", "frall", "ext"], ["frall", "ext"]):
        for k in (3, 5):
            X, y = juntar(fontes, nomes, repetir_ext=k)
            avaliar(treinar(X, y), val, " + ".join(nomes) + f" (ext x{k})", "E4 dados", obs=f"n_treino={len(X)}")


def fabricas_modelo():
    # E5: variações de representação e classificador (todas com class_weight balanceado).
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC

    def tfidf(**kw):
        base = dict(max_features=50000, ngram_range=(1, 2), dtype=np.float32)
        base.update(kw)
        return TfidfVectorizer(**base)

    def svm(C=1.0):
        return LinearSVC(C=C, max_iter=20000, class_weight="balanced")

    return {
        "word(1,2) C=1 [base]": lambda: Pipeline([("tfidf", tfidf()), ("svm", svm())]),
        "word(1,2) C=0.1": lambda: Pipeline([("tfidf", tfidf()), ("svm", svm(0.1))]),
        "word(1,2) C=0.3": lambda: Pipeline([("tfidf", tfidf()), ("svm", svm(0.3))]),
        "word(1,2) C=3": lambda: Pipeline([("tfidf", tfidf()), ("svm", svm(3))]),
        "word(1,2) sublinear": lambda: Pipeline([("tfidf", tfidf(sublinear_tf=True)), ("svm", svm())]),
        "word(1,2) sem max_features, min_df=2": lambda: Pipeline([("tfidf", tfidf(max_features=None, min_df=2)),
                                                                  ("svm", svm())]),
        "word(1,2) mantém maiúsculas": lambda: Pipeline([("tfidf", tfidf(lowercase=False)), ("svm", svm())]),
        "word(1,1)": lambda: Pipeline([("tfidf", tfidf(ngram_range=(1, 1))), ("svm", svm())]),
        "char_wb(2,5)": lambda: Pipeline([("tfidf", tfidf(analyzer="char_wb", ngram_range=(2, 5),
                                                          max_features=200000, sublinear_tf=True)),
                                          ("svm", svm())]),
        "word(1,2) + char_wb(2,5)": lambda: Pipeline([
            ("feat", FeatureUnion([
                ("w", tfidf(sublinear_tf=True)),
                ("c", tfidf(analyzer="char_wb", ngram_range=(2, 5), max_features=200000, sublinear_tf=True))])),
            ("svm", svm())]),
        "word(1,2) + char_wb(2,5) sem lowercase": lambda: Pipeline([
            ("feat", FeatureUnion([
                ("w", tfidf(sublinear_tf=True)),
                ("c", tfidf(analyzer="char_wb", ngram_range=(2, 5), max_features=200000, sublinear_tf=True,
                            lowercase=False))])),
            ("svm", svm())]),
        "LogReg word(1,2) C=10": lambda: Pipeline([("tfidf", tfidf(sublinear_tf=True)),
                                                   ("lr", LogisticRegression(C=10, max_iter=3000,
                                                                             class_weight="balanced"))]),
    }


def etapa_modelo(nomes_dados):
    # E5: com os dados escolhidos na E4, varia representação e classificador.
    fontes, _, val = fontes_de_treino()
    rep = 1
    nomes = []
    for n in nomes_dados.split("+"):
        if n.startswith("ext") and "x" in n:
            rep = int(n.split("x")[1]); n = "ext"
        nomes.append(n)
    X, y = juntar(fontes, nomes, repetir_ext=rep)
    for nome, fab in fabricas_modelo().items():
        t0 = time.time()
        avaliar(treinar(X, y, fabrica=fab), val, nome, "E5 modelo",
                obs=f"dados={nomes_dados}; {time.time() - t0:.0f}s")


def etapa_fontes(nomes_dados):
    # E6: robustez entre fontes dentro do corpus externo. Treina sem uma fonte de cada classe e valida nela.
    # Pergunta: o ganho do corpus externo vem de "boato x jornalismo" ou de assinatura de cada site?
    fontes, ext_tr, ext_val = fontes_de_treino()
    base = [n for n in nomes_dados.split("+") if n != "ext"]
    for fora_fake, fora_true in (("E-farsas", "Agência Brasil"), ("Boatos.org", "G1")):
        tr = ext_tr[~ext_tr["fonte"].isin([fora_fake, fora_true])]
        val = pd.concat([ext_tr, ext_val])
        val = val[val["fonte"].isin([fora_fake, fora_true])].reset_index(drop=True)
        fontes_local = dict(fontes, ext=(tr["texto"], tr["label"]))
        for nomes in (base, base + ["ext"]):
            if not nomes:
                continue
            X, y = juntar(fontes_local, nomes)
            avaliar(treinar(X, y), val, f"{'+'.join(nomes)} | sem {fora_fake}/{fora_true} no treino",
                    "E6 fontes", obs=f"valida só em {fora_fake} e {fora_true} (treino+val)")


def embeddings(textos, nome_modelo, prefixo=""):
    # Codifica textos com um modelo sentence-transformers, com cache em disco (é a parte cara, na CPU).
    import hashlib
    from sentence_transformers import SentenceTransformer
    chave = hashlib.md5((nome_modelo + prefixo + "\x00".join(textos)).encode()).hexdigest()[:16]
    arq = os.path.join(PASTA, f"emb_{chave}.npy")
    if os.path.exists(arq):
        return np.load(arq)
    st = SentenceTransformer(nome_modelo, device="cpu")
    st.max_seq_length = 128
    E = st.encode([prefixo + t for t in textos], batch_size=64, show_progress_bar=True,
                  normalize_embeddings=True, convert_to_numpy=True)
    np.save(arq, E)
    return E


def etapa_embeddings(nomes_dados, nome_modelo, prefixo=""):
    # E7: embeddings de modelo pré-treinado + regressão logística. Treino em 50 palavras (sem aumento misto,
    # para caber no tempo de CPU); validação no formato do produto.
    from sklearn.linear_model import LogisticRegression
    fontes, _, val = fontes_de_treino()
    X, y = juntar(fontes, nomes_dados.split("+"))
    Xtr = truncar(X, N_PRODUTO).tolist()
    Xv = truncar(val["texto"], N_PRODUTO).tolist()
    t0 = time.time()
    Etr, Ev = embeddings(Xtr, nome_modelo, prefixo), embeddings(Xv, nome_modelo, prefixo)
    for C in (0.3, 1, 3, 10):
        lr = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Etr, y)
        avaliar(None, val, f"emb {nome_modelo.split('/')[-1]} + LogReg C={C}", "E7 embeddings",
                obs=f"dados={nomes_dados}; {time.time() - t0:.0f}s", pred_fn=lambda _: lr.predict(Ev))


if __name__ == "__main__":
    etapa = sys.argv[1] if len(sys.argv) > 1 else "preparar"
    if etapa == "embeddings":
        etapa_embeddings(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
    elif etapa == "modelo":
        etapa_modelo(sys.argv[2])
    elif etapa == "fontes":
        etapa_fontes(sys.argv[2])
    elif etapa == "preparar":
        preparar_corpus_externo()
    elif etapa == "base":
        etapa_base()
    elif etapa == "dados":
        etapa_dados()
