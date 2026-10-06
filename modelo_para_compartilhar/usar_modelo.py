"""Classifica trechos de notícia em português como TRUE (verdadeira) ou FAKE (boato).

Não depende de nenhum outro arquivo do projeto: só do modelo em modelo/modelo_vinagre.joblib.

Uso:
  python usar_modelo.py                          (modo interativo: cola o texto, Enter em linha vazia analisa)
  python usar_modelo.py "texto da notícia"       (uma notícia só)
  python usar_modelo.py --arquivo noticias.txt   (uma notícia por linha; linhas com # são ignoradas)

Como ler a saída:
  - A entrada é cortada nas primeiras 50 palavras (o modelo foi feito para trechos curtos).
  - "score" é a distância até a fronteira de decisão do SVM: >= 0 -> TRUE, < 0 -> FAKE.
    Perto de 0 = caso de dúvida; quanto mais longe de 0, mais "certo" o modelo está.
    NÃO é probabilidade (não dá para ler 0,8 como "80%").
"""
import os
import sys

import joblib

ARQ_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo", "modelo_vinagre.joblib")
N_PALAVRAS = 50


# Corta o texto nas primeiras N palavras (mesmo formato usado no treino e na avaliação).
def truncar(texto, n=N_PALAVRAS):
    return " ".join(texto.split()[:n])


# Devolve (classe, score, trecho analisado).
def classificar(modelo, texto):
    trecho = truncar(texto)
    score = float(modelo.decision_function([trecho])[0])
    return ("TRUE" if score >= 0 else "FAKE"), score, trecho


def mostrar(modelo, texto):
    classe, score, trecho = classificar(modelo, texto)
    n = len(trecho.split())
    if abs(score) < 0.2:
        certeza = "dúvida (perto da fronteira)"
    elif abs(score) < 0.6:
        certeza = "moderada"
    else:
        certeza = "alta"
    print("\n" + "-" * 70)
    print(f"Trecho ({n} palavras): {trecho[:200]}{'...' if len(trecho) > 200 else ''}")
    if n < 20:
        print("  Aviso: trecho curto (< 20 palavras); o modelo foi treinado com 30 a 100 palavras.")
    print(f"Resultado: {classe}   score {score:+.3f}   certeza {certeza}")


if __name__ == "__main__":
    modelo = joblib.load(ARQ_MODELO)
    args = sys.argv[1:]
    if args and args[0] == "--arquivo":
        with open(args[1], encoding="utf-8") as f:
            for linha in f:
                # Linhas vazias e comentários (#) são ignorados.
                if linha.strip() and not linha.startswith("#"):
                    mostrar(modelo, linha)
    elif args:
        mostrar(modelo, " ".join(args))
    else:
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
                mostrar(modelo, " ".join(linhas))
