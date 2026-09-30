"""Coleta automática do teste externo via RSS (README 3.3).

Gera teste_externo_2026_bruto.csv (com colunas extras para a conferência).
O teste_externo_2026.csv final só é gerado depois da conferência manual por conteúdo.
Este script NÃO carrega nenhum modelo e NÃO avalia nada.

Uso: .venv\\Scripts\\python.exe coletar_teste_externo.py
"""
import re
import time

import feedparser
import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup

MAX_PALAVRAS = 60
MAX_POR_FONTE = 15
HEADERS = {"User-Agent": "Mozilla/5.0"}

FEEDS = {
    "Agência Brasil": ("true", "https://agenciabrasil.ebc.com.br/rss/ultimasnoticias/feed.xml"),
    "G1": ("true", "https://g1.globo.com/rss/g1/"),
    "Boatos.org": ("fake", "https://www.boatos.org/feed"),
    "E-farsas": ("fake", "https://www.e-farsas.com/feed"),
}

# Classes de blockquote que são embeds de redes sociais (não são o boato em texto).
CLASSES_EMBED = ("instagram-media", "tiktok-embed", "twitter-tweet", "wp-embedded-content")


# Normaliza espaços e corta em N palavras.
def limpar_e_cortar(texto, n=MAX_PALAVRAS):
    palavras = " ".join(str(texto).split()).split()
    return " ".join(palavras[:n])


# Trecho de notícia true: início da matéria (sem título), sem linhas de navegação.
def extrair_true(html, titulo):
    corpo = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    linhas = [l.strip() for l in corpo.split("\n") if l.strip()]

    # Remove o título quando o extrator o inclui (Agência Brasil inclui, G1 não).
    if linhas and linhas[0].strip().lower() == titulo.strip().lower():
        linhas = linhas[1:]

    # Remove linhas de navegação/lixo de página e marcadores gráficos.
    lixo = re.compile(r"^(leia (também|mais)|veja (também|mais)|assista|📲|siga o|clique aqui)", re.I)
    linhas = [re.sub(r"^[➡▶►•✅📌🔴]+\s*", "", l) for l in linhas if not lixo.match(l)]

    # Remove legendas de foto ("... — Foto: crédito").
    linhas = [l for l in linhas if not re.search(r"—\s*Foto:", l)]

    # Remove assinatura com data ("Por Fulano, ... 30/09/2026 14h03 Atualizado ..."), que vem quebrada em linhas.
    texto = " ".join(linhas)
    texto = re.sub(r"Por\s[^.]*?\d{2}/\d{2}/\d{4}\s+\d{2}h\d{2}(\s+Atualizado(\s+há)?\s+\S+(\s+\d{2}/\d{2}/\d{4})?)?\s*",
                   "", texto)
    return limpar_e_cortar(texto)


# Trecho de notícia fake: maior blockquote da página que não seja embed de rede social.
def extrair_fake(html):
    sopa = BeautifulSoup(html, "html.parser")
    candidatos = []
    for bq in sopa.find_all("blockquote"):
        classes = " ".join(bq.get("class") or [])
        texto = " ".join(bq.get_text(" ").split())
        if any(c in classes for c in CLASSES_EMBED) or texto.startswith("@"):
            continue
        candidatos.append(texto)
    if not candidatos:
        return "", 0
    maior = max(candidatos, key=lambda t: len(t.split()))
    # Remove marcadores inseridos pela agência ("Versão 1:", "Versão 2:"), que não fazem parte do boato.
    maior = re.sub(r"Vers[ãa]o \d+\s*:\s*", " ", maior, flags=re.I)
    # Remove aspas nas pontas (o boato costuma vir entre aspas).
    maior = maior.strip(" \"“”'‘’")
    return limpar_e_cortar(maior), len(candidatos)


def coletar():
    linhas, descartes = [], []
    for fonte, (label, url) in FEEDS.items():
        feed = feedparser.parse(url, agent=HEADERS["User-Agent"])
        n_ok = 0
        # Itens na ordem do feed (mais recentes primeiro), sem escolha manual.
        for e in feed.entries:
            if n_ok >= MAX_POR_FONTE:
                break
            link, titulo = e.get("link", ""), e.get("title", "")
            # No G1, só matérias de texto (vídeos e "ao vivo" não têm corpo de notícia).
            if fonte == "G1" and "/noticia/" not in link:
                descartes.append((fonte, link, "não é /noticia/"))
                continue
            try:
                html = requests.get(link, headers=HEADERS, timeout=30).text
            except requests.RequestException as err:
                descartes.append((fonte, link, f"erro de download: {err}"))
                continue
            if label == "true":
                texto, n_bq = extrair_true(html, titulo), None
            else:
                texto, n_bq = extrair_fake(html)
            if not texto:
                descartes.append((fonte, link, "sem trecho extraído"))
                continue
            linhas.append({"texto": texto, "label": label, "fonte": fonte, "link": link,
                           "data": e.get("published", ""), "titulo": titulo,
                           "n_palavras": len(texto.split()), "n_blockquotes": n_bq})
            n_ok += 1
            time.sleep(1)
        print(f"{fonte}: {len(feed.entries)} itens no feed, {n_ok} coletados")
    return pd.DataFrame(linhas), pd.DataFrame(descartes, columns=["fonte", "link", "motivo"])


if __name__ == "__main__":
    bruto, descartes = coletar()
    bruto.insert(0, "id", range(len(bruto)))
    bruto.to_csv("teste_externo_2026_bruto.csv", index=False, encoding="utf-8-sig")
    descartes.to_csv("teste_externo_2026_descartes_automaticos.csv", index=False, encoding="utf-8-sig")
    print(f"\nTotal: {len(bruto)} itens -> teste_externo_2026_bruto.csv")
    print(bruto.groupby(["label", "fonte"]).size())
    print(f"Descartes automáticos: {len(descartes)}")
