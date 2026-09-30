"""Coleta um corpus externo (2024–2026) para TREINO e VALIDAÇÃO dos experimentos de generalização.

Por quê: o Fake.br (2016–2018, fake quase todas de um site) e o FakeRecogna (true resumidas) não se
parecem com o formato do teste externo (boato citado por agência x início de matéria jornalística).
Para melhorar o F1 externo sem olhar o teste final, precisamos de dados do mesmo tipo de fonte.

Regras para não contaminar o teste final (teste_externo_2026.csv):
  - nenhum link presente em teste_externo_2026_bruto.csv entra no corpus;
  - a página 1 dos feeds usados no teste é ignorada (o teste saiu dela);
  - o feed fato-ou-fake do G1 não entra como true (é checagem).

O HTML baixado fica em cache (dados_externos/html) para permitir mudar a extração sem novo download.
Uso: .venv\\Scripts\\python.exe coletar_corpus_externo.py
"""
import gzip
import hashlib
import os
import threading
import time

import feedparser
import pandas as pd
import requests

from coletar_teste_externo import HEADERS, extrair_fake, extrair_true

PASTA = "dados_externos"
PASTA_HTML = os.path.join(PASTA, "html")
os.makedirs(PASTA_HTML, exist_ok=True)

# Feeds de fake: páginas antigas dos feeds WordPress das agências (a página 1 é a do teste).
PAGINAS_BOATOS = range(2, 101)
PAGINAS_EFARSAS = range(2, 41)

# Feeds de true: editorias e estados do G1 e editorias da Agência Brasil.
G1_SECOES = ["economia", "politica", "mundo", "tecnologia", "educacao", "pop-arte", "sp/sao-paulo",
             "rj/rio-de-janeiro", "mg/minas-gerais", "sp/campinas-regiao", "pr/parana",
             "rs/rio-grande-do-sul", "ba/bahia", "pe/pernambuco", "go/goias", "ce/ceara", "pa/para",
             "am/amazonas", "df/distrito-federal", "sc/santa-catarina", "es/espirito-santo"]
AB_SECOES = ["economia", "politica", "geral", "internacional", "saude", "educacao", "justica",
             "direitos-humanos", "esportes", "meio-ambiente", "cultura"]


# Baixa (ou lê do cache) o HTML de uma página.
def baixar(link):
    arq = os.path.join(PASTA_HTML, hashlib.md5(link.encode()).hexdigest() + ".html.gz")
    if os.path.exists(arq):
        with gzip.open(arq, "rt", encoding="utf-8") as f:
            return f.read(), True
    html = requests.get(link, headers=HEADERS, timeout=30).text
    with gzip.open(arq, "wt", encoding="utf-8") as f:
        f.write(html)
    return html, False


# Lista (fonte, label, link, data, titulo) de todos os feeds de uma fonte.
def listar(fonte, label, urls):
    itens = []
    for url in urls:
        feed = feedparser.parse(url, agent=HEADERS["User-Agent"])
        for e in feed.entries:
            itens.append({"fonte": fonte, "label": label, "link": e.get("link", ""),
                          "data": e.get("published", ""), "titulo": e.get("title", ""), "feed": url})
    return itens


# Processa os itens de uma fonte em sequência (uma thread por domínio, com pausa entre downloads).
def processar_fonte(itens, saida, lock):
    for it in itens:
        try:
            html, cache = baixar(it["link"])
        except requests.RequestException as err:
            with lock:
                saida.append({**it, "texto": "", "erro": str(err)[:100]})
            continue
        if it["label"] == "true":
            texto, n_bq = extrair_true(html, it["titulo"]), None
        else:
            texto, n_bq = extrair_fake(html)
        with lock:
            saida.append({**it, "texto": texto, "n_blockquotes": n_bq, "erro": ""})
        if not cache:
            time.sleep(1)


if __name__ == "__main__":
    teste_links = set(pd.read_csv("teste_externo_2026_bruto.csv")["link"])

    # Monta a lista de itens por fonte.
    por_fonte = {
        "Boatos.org": listar("Boatos.org", "fake", [f"https://www.boatos.org/feed?paged={p}" for p in PAGINAS_BOATOS]),
        "E-farsas": listar("E-farsas", "fake", [f"https://www.e-farsas.com/feed?paged={p}" for p in PAGINAS_EFARSAS]),
        "G1": listar("G1", "true", [f"https://g1.globo.com/rss/g1/{s}/" for s in G1_SECOES]),
        "Agência Brasil": listar("Agência Brasil", "true",
                                 [f"https://agenciabrasil.ebc.com.br/rss/{s}/feed.xml" for s in AB_SECOES]),
    }

    # Remove duplicados, links do teste e páginas que não são matérias de texto.
    for fonte, itens in por_fonte.items():
        vistos, filtrados = set(), []
        for it in itens:
            l = it["link"]
            if l in vistos or l in teste_links:
                continue
            if fonte == "G1" and ("/noticia/" not in l or "/fato-ou-fake/" in l):
                continue
            if fonte == "Boatos.org" and ("/espanol/" in l or "/english/" in l):
                continue
            vistos.add(l)
            filtrados.append(it)
        por_fonte[fonte] = filtrados
        print(f"{fonte}: {len(filtrados)} links")

    # Uma thread por domínio: respeita cada site e roda os quatro em paralelo.
    saida, lock = [], threading.Lock()
    threads = [threading.Thread(target=processar_fonte, args=(itens, saida, lock)) for itens in por_fonte.values()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    corpus = pd.DataFrame(saida)
    corpus["n_palavras"] = corpus["texto"].fillna("").str.split().str.len()
    corpus.to_csv(os.path.join(PASTA, "corpus_externo_bruto.csv"), index=False, encoding="utf-8-sig")
    print(corpus.groupby(["label", "fonte"]).agg(n=("link", "size"),
                                                  com_texto=("n_palavras", lambda s: (s > 0).sum())))
