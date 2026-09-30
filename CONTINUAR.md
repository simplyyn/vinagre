# Como retomar o trabalho

## 1. Preparar o ambiente (só se for um PC novo ou se o `.venv` não existir)

```powershell
# Python 3.12 (se não houver): winget install -e --id Python.Python.3.12 --scope user
py -3.12 -m venv .venv   # ou: & "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install sentence-transformers "scikit-learn==1.6.1"
.venv\Scripts\python.exe -m spacy download pt_core_news_sm
```

O scikit-learn **tem que ser 1.6.1** (versão em que os `.joblib` foram salvos).

## 2. Estado em 30/09/2026

- Ambiente, pipeline e números de controle: feitos (ver `EXPERIMENTOS.md`, E0).
- Teste externo final coletado e conferido: `teste_externo_2026.csv` (39 itens). **NÃO avaliado.**
- Corpus externo para treino/validação: `dados_externos/corpus_externo_bruto.csv` (textos já extraídos).
  O cache de HTML (`dados_externos/html/`, ~480 MB) não vai para o Git; só é necessário para mudar a extração.
  Se o CSV bruto estiver faltando, rode `.venv\Scripts\python.exe coletar_corpus_externo.py` (~1h).
- Próximo passo: rodar as etapas de `experimentos.py` (preparar → base → dados → modelo → fontes → embeddings).

## 3. Prompt para colar no Claude Code

```
Leia README.md, EXPERIMENTOS.md e CONTINUAR.md antes de qualquer ação. Estamos na fase de experimentos para
aumentar o F1 em notícias externas. Regras: responda em português, didático, explicando o motivo de cada decisão;
registre cada ação, motivo e resultado em EXPERIMENTOS.md; NÃO use o teste_externo_2026.csv para escolher nada
(ele só será avaliado uma vez, no final, com minha autorização); commits sem coautoria do Claude.

Continue de onde parou:
1. Confira o ambiente (.venv, sklearn 1.6.1). Se faltar algo, siga a seção 1 do CONTINUAR.md.
2. Rode `.venv\Scripts\python.exe experimentos.py preparar` e revise o corpus externo (tamanhos, datas, divisão
   treino/validação, descontaminação).
3. Rode as etapas `base`, `dados`, depois `modelo <melhor combinação de dados>`, `fontes <combinação>` e
   `embeddings <combinação> <modelo>` (modelos: intfloat/multilingual-e5-small com prefixo "query: ",
   sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
4. Analise os resultados (diferenças < 1–2 pontos são empate), escolha o modelo final justificando, e me mostre
   o resumo antes de qualquer avaliação no teste final.
```
