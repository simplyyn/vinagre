"""Teste manual: digite ou cole uma notícia e veja a previsão de vários modelos lado a lado.

Uso:
  .venv\\Scripts\\python.exe testar_noticia.py                      (modo interativo: cola o texto, Enter vazio para enviar)
  .venv\\Scripts\\python.exe testar_noticia.py "texto da notícia"   (uma notícia só)
  .venv\\Scripts\\python.exe testar_noticia.py --arquivo noticias.txt  (uma notícia por linha)
  Acrescente --emb para incluir o modelo de embeddings (e5-small; mais lento para carregar).

A entrada é cortada em 50 palavras (formato do produto). "Confiança" é a distância até a fronteira de decisão
(score do SVM): perto de 0 = o modelo está em dúvida; quanto maior, mais certo. Não é probabilidade, e a escala
muda de um modelo para outro (a regressão logística dos embeddings dá números bem maiores): compare a confiança só
dentro do mesmo modelo.
Isto é só para demonstração: não use os resultados para escolher modelo (ver regras no README).
"""
import sys

import joblib
import numpy as np
import pandas as pd

from anls import truncar
from exp_numeros import contar_numeros

N_PRODUTO = 50

# Modelos mostrados (nome curto -> arquivo). O escolhido vem primeiro.
MODELOS = {
    "TF-IDF caracteres [ESCOLHIDO]": "modelos_finais/03.joblib",
    "TF-IDF palavras (fr2021+ext)": "modelos_finais/02.joblib",
    "Palavras + caracteres": "modelos_finais/04.joblib",
    "Candidato antigo (Fake.br+FR)": "modelo_candidato_fakebr_fakerecogna.joblib",
    "Só Fake.br (antigo)": "modelo_final_tfidf.joblib",
}


def carregar(usar_emb):
    modelos = {nome: joblib.load(arq) for nome, arq in MODELOS.items()}
    if usar_emb:
        # O modelo de embeddings foi salvo pelo avaliacao_final.py; a classe precisa existir em __main__ para carregar.
        import avaliacao_final
        sys.modules["__main__"].EmbLogReg = avaliacao_final.EmbLogReg
        from sentence_transformers import SentenceTransformer
        emb = joblib.load("modelos_finais/05.joblib")
        st = SentenceTransformer(emb.nome_modelo, device="cpu")
        modelos["Embeddings e5-small"] = (st, emb)
    return modelos


def prever(modelo, trecho):
    # Devolve (classe, score). Score > 0 -> true; < 0 -> fake.
    if isinstance(modelo, tuple):
        st, emb = modelo
        E = st.encode([emb.prefixo + trecho], normalize_embeddings=True)
        score = float(emb.lr_.decision_function(E)[0])
    else:
        score = float(modelo.decision_function([trecho])[0])
    return ("true" if score >= 0 else "fake"), score


def analisar(texto, modelos):
    trecho = truncar(pd.Series([" ".join(texto.split())]), N_PRODUTO).iloc[0]
    n = len(trecho.split())
    print("\n" + "─" * 78)
    print(f"Trecho analisado ({n} palavras):\n  {trecho[:300]}{'…' if len(trecho) > 300 else ''}")
    if n < 20:
        print("  ⚠ Trecho curto (< 20 palavras): os modelos foram treinados com 30–100 palavras; a previsão é menos confiável.")
    print()
    votos = []
    for nome, m in modelos.items():
        classe, score = prever(m, trecho)
        votos.append(classe)
        barra = "█" * min(20, int(abs(score) * 10))
        print(f"  {nome:<32} {classe.upper():<5} confiança {abs(score):4.2f} {barra}")
    # Explicação simples (E8): proporção de números no trecho, só como informação.
    c = contar_numeros(trecho)
    print(f"\n  Votos: {votos.count('true')} TRUE × {votos.count('fake')} FAKE"
          f" | números no trecho: {c['n_numeros']} ({c['n_numeros'] / max(n, 1):.0%} das palavras;"
          f" média em notícia true ~5%, em boato ~2%)")
    print("  (confiança = distância da fronteira; a escala muda entre modelos, compare só dentro de cada um)")


if __name__ == "__main__":
    args = sys.argv[1:]
    usar_emb = "--emb" in args
    args = [a for a in args if a != "--emb"]
    print("Carregando modelos...")
    modelos = carregar(usar_emb)

    if args and args[0] == "--arquivo":
        with open(args[1], encoding="utf-8") as f:
            for linha in f:
                if linha.strip():
                    analisar(linha, modelos)
    elif args:
        analisar(" ".join(args), modelos)
    else:
        # Modo interativo: cola o texto (pode ter várias linhas); linha vazia envia; "sair" encerra.
        while True:
            print("\nCole a notícia (linha vazia para analisar, 'sair' para encerrar):")
            linhas = []
            while True:
                try:
                    l = input()
                except EOFError:
                    l = "sair" if not linhas else ""
                if l.strip().lower() == "sair":
                    sys.exit(0)
                if not l.strip():
                    break
                linhas.append(l)
            if linhas:
                analisar(" ".join(linhas), modelos)
