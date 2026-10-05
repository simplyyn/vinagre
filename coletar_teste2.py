"""Coleta do 2º teste externo (teste2_2026), com notícias publicadas DEPOIS do corpus externo (a partir de 02/10/2026).

Por quê: o teste 1 tem só 39 itens (IC de ~±15 pontos). Um 2º conjunto, mais novo e coletado da mesma forma,
dá uma medida independente. Mesmas fontes e mesma extração do teste 1 (coletar_teste_externo.py).

Regras contra contaminação (só por conteúdo/link/data, nunca por previsão de modelo):
  - só itens publicados a partir de 02/10/2026 (o corpus externo vai até 01/10);
  - nenhum link do teste 1 nem do corpus externo;
  - mínimo de palavras igual ao do corpus (fake >= 20, true >= 30) e textos repetidos uma vez só;
  - remove quase-duplicatas (cosseno >= 0,8) de textos do corpus externo e do teste 1 (boato antigo republicado).

Gera teste2_2026_bruto.csv. O teste2_2026.csv final só sai depois da conferência por conteúdo.
Este script NÃO carrega nenhum modelo e NÃO avalia nada.

Uso: .venv\\Scripts\\python.exe coletar_teste2.py
"""
import time

import feedparser
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from coletar_corpus_externo import AB_SECOES, G1_SECOES
from coletar_teste_externo import HEADERS, extrair_fake, extrair_true

DESDE = pd.Timestamp("2026-10-02", tz="America/Sao_Paulo")
# Fake: a partir de 28/09 (as fake do corpus vão até 27/09; links do corpus e do teste 1 continuam excluídos).
# Motivo: de 02/10 em diante só havia 9 boatos com texto (eleição; E-farsas sem publicações).
DESDE_FAKE = pd.Timestamp("2026-09-28", tz="America/Sao_Paulo")
MAX_TRUE_POR_FONTE = 40

FEEDS = {
    # Boatos.org: após a eleição de 04/10 o feed foi tomado por milhares de páginas automáticas de resultado;
    # as páginas são lidas até chegar em DESDE (ver paginas_boatos).
    "Boatos.org": ("fake", None),
    "E-farsas": ("fake", [f"https://www.e-farsas.com/feed?paged={p}" for p in (1, 2, 3)]),
    "G1": ("true", ["https://g1.globo.com/rss/g1/"] + [f"https://g1.globo.com/rss/g1/{s}/" for s in G1_SECOES]),
    "Agência Brasil": ("true", ["https://agenciabrasil.ebc.com.br/rss/ultimasnoticias/feed.xml"]
                       + [f"https://agenciabrasil.ebc.com.br/rss/{s}/feed.xml" for s in AB_SECOES]),
}


def paginas_boatos(max_paginas=300):
    # Lê o feed paginado do Boatos.org até a página cujo item mais antigo é anterior a DESDE.
    urls = []
    for p in range(1, max_paginas + 1):
        url = f"https://www.boatos.org/feed?paged={p}"
        es = feedparser.parse(url, agent=HEADERS["User-Agent"]).entries
        if not es:
            break
        urls.append(url)
        if pd.to_datetime(es[-1].get("published", ""), utc=True, errors="coerce") < DESDE_FAKE:
            break
    return urls


def listar(fonte, label, urls, excluir):
    # Itens dos feeds publicados a partir de DESDE, sem links já usados.
    itens, vistos = [], set()
    for url in urls:
        for e in feedparser.parse(url, agent=HEADERS["User-Agent"]).entries:
            l = e.get("link", "")
            dt = pd.to_datetime(e.get("published", ""), utc=True, errors="coerce")
            if l in vistos or l in excluir or pd.isna(dt) or dt < (DESDE_FAKE if label == "fake" else DESDE):
                continue
            if fonte == "G1" and ("/noticia/" not in l or "/fato-ou-fake/" in l):
                continue
            # Boatos.org: sem traduções e sem páginas automáticas de resultado de eleição (não são boatos).
            if fonte == "Boatos.org" and ("/espanol/" in l or "/english/" in l or "/resultado-eleicoes-" in l):
                continue
            vistos.add(l)
            itens.append({"fonte": fonte, "label": label, "link": l, "dt": dt,
                          "data": e.get("published", ""), "titulo": e.get("title", "")})
    # Mais recentes primeiro (ordem do feed, sem escolha manual).
    itens.sort(key=lambda it: it["dt"], reverse=True)
    return itens


def motivo_descarte(texto, label, ja_mantidos, ref):
    # Filtros por conteúdo: mínimo de palavras, quase-duplicata de item já mantido (textos-modelo, como as
    # páginas de resultado por cidade do G1) e quase-duplicata do corpus externo ou do teste 1.
    if len(texto.split()) < (20 if label == "fake" else 30):
        return "abaixo do mínimo de palavras"
    if ja_mantidos and cosine_similarity(ref["vec"].transform([texto]),
                                         ref["vec"].transform(ja_mantidos)).max() >= 0.8:
        return "quase idêntico a outro item do teste 2 (texto-modelo/repetido)"
    if cosine_similarity(ref["vec"].transform([texto]), ref["X"]).max() >= 0.8:
        return "quase idêntico a texto do corpus ou do teste 1"
    return ""


if __name__ == "__main__":
    corpus = pd.read_csv("dados_externos/corpus_externo_bruto.csv")
    teste1 = pd.read_csv("teste_externo_2026_bruto.csv")
    excluir = set(corpus["link"]) | set(teste1["link"])
    textos_ref = pd.concat([corpus["texto"].dropna(), teste1["texto"].dropna()])
    vec = TfidfVectorizer().fit(textos_ref)
    ref = {"vec": vec, "X": vec.transform(textos_ref)}

    linhas = []
    for fonte, (label, urls) in FEEDS.items():
        itens = listar(fonte, label, urls or paginas_boatos(), excluir)
        mantidos = []
        for it in itens:
            # True: para quando a fonte atinge o limite de itens válidos. Fake: todos.
            if label == "true" and len(mantidos) >= MAX_TRUE_POR_FONTE:
                break
            try:
                html = requests.get(it["link"], headers=HEADERS, timeout=30).text
                texto = extrair_true(html, it["titulo"]) if label == "true" else extrair_fake(html)[0]
                motivo = motivo_descarte(texto, label, mantidos, ref)
            except requests.RequestException as err:
                texto, motivo = "", f"erro de download: {err}"[:100]
            linhas.append({**it, "texto": texto, "motivo": motivo})
            if not motivo:
                mantidos.append(texto)
            time.sleep(1)
        print(f"{fonte}: {len(itens)} links desde {DESDE.date()}, {len(mantidos)} mantidos")

    b = pd.DataFrame(linhas)
    b["n_palavras"] = b["texto"].fillna("").str.split().str.len()

    b.insert(0, "id", range(len(b)))
    b.drop(columns="dt").to_csv("teste2_2026_bruto.csv", index=False, encoding="utf-8-sig")
    print(b.assign(mantido=b["motivo"] == "").groupby(["label", "fonte"])["mantido"].agg(["size", "sum"]))
    print(b.loc[b["motivo"] != "", "motivo"].value_counts())
