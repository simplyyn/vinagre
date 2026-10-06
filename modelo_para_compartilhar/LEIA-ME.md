# Vinagre: detector de boatos em português

Modelo que lê um **trecho de notícia (até 50 palavras)** e diz se ele parece **notícia verdadeira (TRUE)** ou
**boato (FAKE)**. Projeto acadêmico da Residência em IA.

**Resultado no teste final** (116 notícias publicadas depois do treino, avaliadas uma única vez): F1 macro
**94,2**. Acertou **100%** das notícias verdadeiras e **84%** dos boatos.

> ⚠️ O modelo reconhece **estilo de escrita**, não verifica fatos. Um boato escrito com cara de notícia pode
> passar como verdadeiro. Use como curiosidade e estudo, não como checagem.

## O que tem nesta pasta

| Arquivo | Para quê |
|---|---|
| `LEIA-ME.md` | este guia |
| `COMO_O_MODELO_FUNCIONA.md` | explicação didática: como o texto vira números e como a decisão é tomada |
| `COMO_FOI_TREINADO.md` | **ficha técnica completa**: datasets, pré-processamento, features, hiperparâmetros, por que cada escolha, avaliação e limitações |
| `usar_modelo.py` | testar o modelo com suas próprias notícias |
| `treinar_do_zero.py` | reproduzir o treino do zero |
| `exemplos.txt` | algumas notícias e boatos reais para testar |
| `requirements.txt` | bibliotecas necessárias |
| `modelo/modelo_vinagre.joblib` | o modelo treinado (TF-IDF + LinearSVC, scikit-learn 1.6.1) |
| `dados/corpus_externo.csv` | corpus coletado pelo projeto (Boatos.org, E-farsas, G1, Agência Brasil), com a coluna `split` (treino/val) |
| `dados/teste_externo_2026.csv`, `dados/teste2_2026.csv` | os dois testes finais (116 itens) |
| `resultados/relatorio.html` | relatório técnico com gráficos (abrir no navegador) |
| `resultados/apresentacao.html` | mini-site da apresentação |
| `resultados/*.csv` | números de todos os experimentos e do teste final |

## Como testar (5 minutos)

Precisa do **Python 3.12** (https://www.python.org/downloads/). Abra um terminal **dentro desta pasta**:

```bash
# 1. Criar um ambiente isolado e instalar as bibliotecas
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

# 2. Testar
python usar_modelo.py --arquivo exemplos.txt        # roda os exemplos prontos
python usar_modelo.py "cole aqui o texto da notícia" # uma notícia
python usar_modelo.py                                # modo interativo
```

Exemplo de saída:

```
Trecho (22 palavras): Atenção, aposentado: se você não votar, seu benefício poderá ser cancelado. ...
Resultado: FAKE   score -1.854   certeza alta
```

O `exemplos.txt` traz 7 casos reais, incluindo **um erro de propósito**: um boato escrito como notícia ("ZANIN
ASSUME O TSE...") que o modelo classifica como TRUE com confiança. É a principal limitação do modelo.

- **score ≥ 0 → TRUE, score < 0 → FAKE.** Perto de 0 = dúvida. Não é porcentagem.
- Para notícia de jornal: cole o **começo da matéria, sem o título**.
- Para boato: cole o **texto do boato** (a mensagem que circula), não o texto da agência que desmentiu.

**Atenção à versão:** o modelo foi salvo com o **scikit-learn 1.6.1**. Com outra versão ele pode dar aviso ou
não carregar. Por isso o `requirements.txt` fixa essa versão.

## Retreinar do zero (opcional)

```bash
python treinar_do_zero.py
```

Baixa o FakeRecogna 2.0 do Hugging Face (precisa de internet), treina com a mesma receita e confere o F1 na
validação (~91,4). Leva poucos minutos.

## Código completo do projeto

https://github.com/simplyyn/vinagre (todos os experimentos, coleta de dados e o histórico das decisões em
`EXPERIMENTOS.md`).
