"""E8: a quantidade de números no texto ajuda a separar true de fake? (ver EXPERIMENTOS.md)

Hipótese: notícia verdadeira cita mais números (datas, valores, percentuais, quantidades) porque fala de
coisas exatas. Tudo é medido em trechos de até 50 palavras (formato do produto) e toda escolha usa só a
validação externa. O teste_externo_2026.csv não é lido aqui.

Uso: .venv\\Scripts\\python.exe exp_numeros.py <descritivo|modelo|fontes|fakebr>
"""
import re
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from anls import base_truncada, carregar_fakebr, mostrar, rodar_cv, truncar
from experimentos import N_PRODUTO, avaliar, fontes_de_treino, juntar, treinar

DADOS = ["fr2021", "ext"]   # combinação sugerida no CONTINUAR.md (empatada com a maior da E4)
GRUPO = "E8 números"


# ---------------------------------------------------------------------------
# Extração das features de números
# ---------------------------------------------------------------------------

# Números por extenso ("um"/"uma" ficam de fora: quase sempre são artigo).
EXTENSO = set("""dois duas três quatro cinco seis sete oito nove dez onze doze treze catorze quatorze quinze
dezesseis dezessete dezoito dezenove vinte trinta quarenta cinquenta sessenta setenta oitenta noventa cem
cento duzentos trezentos quatrocentos quinhentos mil milhão milhões bilhão bilhões trilhão trilhões
metade dobro triplo dúzia dezena dezenas centena centenas""".split())

RE_DIGITO = re.compile(r"\d")
RE_ANO = re.compile(r"^\(?(?:19|20)\d{2}\b")
RE_PCT = re.compile(r"\d.*%|^%")
RE_HORA = re.compile(r"^\d{1,2}(?:h\d{0,2}|:\d{2})\b")
RE_DATA = re.compile(r"^\(?\d{1,2}(?:º|/\d{1,2})")
RE_PONTA = re.compile(r"^\W+|\W+$")

CATEGORIAS = ["ano", "percentual", "dinheiro", "hora_data", "outro_digito", "extenso"]


def contar_numeros(texto):
    # Conta palavras com dígito, por categoria, e números por extenso. Devolve contagens absolutas.
    toks = texto.split()
    c = dict.fromkeys(CATEGORIAS, 0)
    c["n_palavras"] = len(toks)
    for i, t in enumerate(toks):
        limpo = RE_PONTA.sub("", t.lower())
        if RE_DIGITO.search(t):
            anterior = toks[i - 1] if i else ""
            if RE_PCT.search(t) or " ".join(toks[i + 1:i + 3]).lower().startswith("por cento"):
                c["percentual"] += 1
            elif "$" in t or anterior.endswith("$"):
                c["dinheiro"] += 1
            elif RE_ANO.match(t):
                c["ano"] += 1
            elif RE_HORA.match(t) or RE_DATA.match(t):
                c["hora_data"] += 1
            else:
                c["outro_digito"] += 1
        elif limpo in EXTENSO:
            c["extenso"] += 1
    c["n_numeros"] = sum(c[k] for k in CATEGORIAS)
    return c


def tabela_numeros(textos):
    return pd.DataFrame([contar_numeros(t) for t in textos])


class FeaturesNumeros(BaseEstimator, TransformerMixin):
    """Transforma texto em features de números.

    modo:
      - "contagem":   número absoluto de números (depende do tamanho do trecho);
      - "pct":        números / palavras (o "percentual" da ideia original);
      - "pct_suave":  percentual "relativizado": (n_numeros + m*p0) / (n_palavras + m). Em trecho curto o
                      percentual cru é instável (1 número em 8 palavras = 12,5%); a suavização puxa o valor para
                      a média do treino (p0) e só confia no percentual quando há palavras suficientes;
      - "categorias": pct_suave do total + pct_suave de cada categoria (ano, percentual, dinheiro...).
    """

    def __init__(self, modo="pct", m=20):
        self.modo = modo
        self.m = m

    def _suave(self, cont, n, p0):
        return (cont + self.m * p0) / (n + self.m)

    def fit(self, X, y=None):
        # p0 = média de cada taxa no treino (fit só no treino).
        t = tabela_numeros(X)
        n = t["n_palavras"].sum()
        self.p0_ = {k: t[k].sum() / max(n, 1) for k in CATEGORIAS + ["n_numeros"]}
        return self

    def transform(self, X):
        t = tabela_numeros(X)
        n = t["n_palavras"].clip(lower=1)
        if self.modo == "contagem":
            cols = [t["n_numeros"]]
        elif self.modo == "pct":
            cols = [t["n_numeros"] / n]
        elif self.modo == "pct_suave":
            cols = [self._suave(t["n_numeros"], n, self.p0_["n_numeros"])]
        elif self.modo == "categorias":
            cols = [self._suave(t[k], n, self.p0_[k]) for k in ["n_numeros"] + CATEGORIAS]
        elif self.modo == "tamanho":
            # Controle: só o número de palavras do trecho (mede o atalho de tamanho).
            cols = [np.log1p(t["n_palavras"])]
        return np.column_stack(cols).astype(np.float32)


# ---------------------------------------------------------------------------
# Fábricas de modelo
# ---------------------------------------------------------------------------

def tfidf(**kw):
    base = dict(max_features=50000, ngram_range=(1, 2), dtype=np.float32)
    base.update(kw)
    return TfidfVectorizer(**base)


def svm():
    return LinearSVC(C=1.0, max_iter=20000, class_weight="balanced")


def so_numeros(modo):
    # Só as features de números (sem texto): mede o sinal isolado da hipótese.
    return lambda: Pipeline([("num", FeaturesNumeros(modo)), ("sc", StandardScaler()),
                             ("lr", LogisticRegression(class_weight="balanced", max_iter=2000))])


def tfidf_mais_numeros(modo, peso):
    # TF-IDF + features de números padronizadas, lado a lado no mesmo LinearSVC.
    return lambda: Pipeline([
        ("feat", FeatureUnion([("w", tfidf()),
                               ("n", Pipeline([("num", FeaturesNumeros(modo)), ("sc", StandardScaler())]))],
                              transformer_weights={"w": 1.0, "n": peso})),
        ("svm", svm())])


def mascarar_digitos(s):
    # "2026" -> "0000", "R$ 3,5" -> "R$ 0,0": o modelo vê o formato do número, não o valor.
    return re.sub(r"\d", "0", s.lower())


def tfidf_digitos_mascarados():
    # Alternativa sem feature nova: números viram formato. Evita aprender "2026" (atalho de época).
    return lambda: Pipeline([("tfidf", tfidf(preprocessor=mascarar_digitos)), ("svm", svm())])


def modelos():
    return {
        "só tamanho (controle)": so_numeros("tamanho"),
        "só números: contagem": so_numeros("contagem"),
        "só números: pct": so_numeros("pct"),
        "só números: pct_suave": so_numeros("pct_suave"),
        "só números: categorias": so_numeros("categorias"),
        "TF-IDF [base]": lambda: Pipeline([("tfidf", tfidf()), ("svm", svm())]),
        "TF-IDF + pct_suave (peso 1)": tfidf_mais_numeros("pct_suave", 1.0),
        "TF-IDF + categorias (peso 0.3)": tfidf_mais_numeros("categorias", 0.3),
        "TF-IDF + categorias (peso 1)": tfidf_mais_numeros("categorias", 1.0),
        "TF-IDF dígitos mascarados": tfidf_digitos_mascarados(),
    }


# ---------------------------------------------------------------------------
# Etapas
# ---------------------------------------------------------------------------

def resumo(textos, labels, rotulo):
    # Estatística descritiva por classe (trechos de até 50 palavras) e AUC de cada taxa isolada.
    t = tabela_numeros(truncar(pd.Series(list(textos)), N_PRODUTO))
    n = t["n_palavras"].clip(lower=1)
    y = pd.Series(list(labels))
    d = pd.DataFrame({"label": y, "palavras": t["n_palavras"], "tem_numero": (t["n_numeros"] > 0) * 100.0,
                      "pct_numeros": t["n_numeros"] / n * 100})
    for k in CATEGORIAS:
        d[f"pct_{k}"] = t[k] / n * 100
    print(f"\n=== {rotulo} (n={len(d)}) — médias por classe; pct_* = % das palavras ===")
    print(d.groupby("label").mean().round(2).T.to_string())
    # AUC: 0,5 = não separa; > 0,5 = mais números nas true; < 0,5 = mais números nas fake.
    auc = {c: roc_auc_score(y == "true", d[c]) for c in d.columns if c not in ("label",)}
    print("AUC (true como positivo):", {k: round(v, 3) for k, v in auc.items()})
    return d


def etapa_descritivo():
    # Antes de modelar: a diferença existe? Em quais corpora? É só de uma fonte?
    fontes, ext_tr, _ = fontes_de_treino()
    d = base_truncada(carregar_fakebr(), N_PRODUTO)
    resumo(d["text"], d["label"], "Fake.br (pares com >= 50 palavras)")
    resumo(*fontes["fr2021"], "FakeRecogna 2020-21 (treino do candidato)")
    resumo(*fontes["ext"], "Corpus externo — TREINO")
    # Por fonte no corpus externo: separa "true tem mais número" de "G1 escreve com mais número".
    print("\n=== Corpus externo (treino) por fonte ===")
    t = tabela_numeros(truncar(ext_tr["texto"], N_PRODUTO))
    n = t["n_palavras"].clip(lower=1)
    por_fonte = pd.DataFrame({"fonte": ext_tr["fonte"], "label": ext_tr["label"], "palavras": t["n_palavras"],
                              "tem_numero": (t["n_numeros"] > 0) * 100.0, "pct_numeros": t["n_numeros"] / n * 100,
                              **{f"pct_{k}": t[k] / n * 100 for k in CATEGORIAS}})
    print(por_fonte.groupby(["label", "fonte"]).mean().round(2).T.to_string())


def etapa_modelo():
    # Validação externa com os dados escolhidos (fr2021 + ext) e treino misto 30/50/100.
    fontes, _, val = fontes_de_treino()
    X, y = juntar(fontes, DADOS)
    for nome, fab in modelos().items():
        avaliar(treinar(X, y, fabrica=fab), val, nome, GRUPO, obs=f"dados={'+'.join(DADOS)}")


def etapa_fontes():
    # Como a E6: treina sem uma fonte de cada classe e valida só nelas (assinatura de site x sinal real).
    fontes, ext_tr, ext_val = fontes_de_treino()
    escolhidos = {k: v for k, v in modelos().items()
                  if k in ("só números: categorias", "TF-IDF [base]", "TF-IDF + categorias (peso 0.3)",
                           "TF-IDF + categorias (peso 1)", "TF-IDF dígitos mascarados")}
    for fora_fake, fora_true in (("E-farsas", "Agência Brasil"), ("Boatos.org", "G1")):
        tr = ext_tr[~ext_tr["fonte"].isin([fora_fake, fora_true])]
        val = pd.concat([ext_tr, ext_val])
        val = val[val["fonte"].isin([fora_fake, fora_true])].reset_index(drop=True)
        X, y = juntar(dict(fontes, ext=(tr["texto"], tr["label"])), DADOS)
        for nome, fab in escolhidos.items():
            avaliar(treinar(X, y, fabrica=fab), val, f"{nome} | sem {fora_fake}/{fora_true} no treino",
                    GRUPO + " (fontes)", obs=f"dados={'+'.join(DADOS)}; valida só em {fora_fake} e {fora_true}")


def etapa_fakebr():
    # Protocolo de referência (CV agrupada, Fake.br N=50), sem aumento misto, como o controle 2.
    d = base_truncada(carregar_fakebr(), N_PRODUTO)
    for nome in ("TF-IDF [base]", "TF-IDF + categorias (peso 0.3)", "TF-IDF + categorias (peso 1)",
                 "só números: categorias"):
        mostrar(f"Fake.br CV N=50 - {nome}", rodar_cv(modelos()[nome](), d["text"], d["label"], d["pair_id"]))


if __name__ == "__main__":
    {"descritivo": etapa_descritivo, "modelo": etapa_modelo, "fontes": etapa_fontes,
     "fakebr": etapa_fakebr}[sys.argv[1]]()
