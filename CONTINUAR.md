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

## 2. Estado em 01/10/2026

- Ambiente, pipeline e números de controle: feitos e conferidos de novo em outro PC (ver `EXPERIMENTOS.md`, E0 e E2b).
- Teste externo final coletado e conferido: `teste_externo_2026.csv` (39 itens). **NÃO avaliado.**
- Corpus externo **recoletado em 01/10** e agora versionado no Git:
  - `dados_externos/corpus_externo_bruto.csv` (textos extraídos) e `dados_externos/corpus_externo.csv`
    (já preparado: mínimo de palavras, duplicados, descontaminação, coluna `split` treino/val).
  - **Não rode `coletar_corpus_externo.py` nem `experimentos.py preparar` de novo**: isso mudaria o corpus e a
    divisão, e os resultados já registrados deixariam de ser comparáveis.
  - O cache de HTML (`dados_externos/html/`) e o parquet do FakeRecogna não vão para o Git; o parquet é baixado
    sozinho na primeira execução (`anls.carregar_fakerecogna`).
- Etapas feitas: **E3 base** e **E4 dados (17 combinações, completa)**. Resultados em
  `resultados_experimentos.csv` e `EXPERIMENTOS.md`. Topo em empate técnico: `frall + ext x3` (F1 90,69),
  `fr2021 + ext` (90,47), `frall + ext x5` (90,09); peso do externo muda < 1 ponto. **Não rode `dados` de novo**
  (duplicaria linhas no CSV).

## 2.0 Estado em 05/10/2026 (mais recente)

- Ambiente recriado em outro PC (controles iguais ao E0).
- Feitos: escolha de dados (`fr2021+ext`), **E5**, **E6** (agora também compara modelos da E5), **E7** e **E8**
  (feature de números: não entra). Análise e escolha registradas no `EXPERIMENTOS.md`.
- **Modelo escolhido:** `fr2021+ext`, TF-IDF `char_wb(2,5)` + LinearSVC balanceado, treino misto. Salvo em
  `modelos_finais/03.joblib` (os 8 métodos avaliados estão em `modelos_finais/`, ordem do `avaliacao_final.py`).
- **Avaliação final FEITA (autorizada):** teste 1 (39) + teste 2 novo (77, `teste2_2026.csv`). Escolhido: F1 94,2
  [89–99] nos dois juntos. Detalhes no `EXPERIMENTOS.md`. **Os testes não podem mais ser usados para escolher nada.**
- Teste manual (demonstração): `testar_noticia.py` (interativo, texto como argumento ou `--arquivo`; `--emb` inclui
  os embeddings). Mostra a previsão de vários modelos lado a lado. Não serve para escolher modelo.
  Versão no navegador: `testador_web.py [--emb]` e abrir http://127.0.0.1:8000 (cola a notícia e clica em Analisar).
- Relatório da apresentação: `relatorio.html` (gerado por `dados_relatorio.py` a partir de `relatorio_modelo.html`).
- Não rodar de novo `modelo`, `fontes`, `embeddings`, `exp_numeros.py modelo/fontes` (duplicaria linhas no CSV) nem
  `coletar_teste2.py` (mudaria o teste 2).
- Próximo: escrever o relatório final; opcional, limpar os 42 especiais publicitários do corpus externo (exigiria
  nova validação e um teste novo).

## 2.1 O que fazer amanhã (em ordem) — versão de 01/10 (passos 3 a 7 feitos em 05/10)

1. Conferir o ambiente: `.venv\Scripts\python.exe -c "import sklearn; print(sklearn.__version__)"` → 1.6.1.
2. ~~Terminar a E4~~ (feito).
3. Escolher a combinação de dados e registrar no `EXPERIMENTOS.md` (empates de < 1–2 pontos: preferir a mais
   simples). Sugestão: `fr2021+ext` (sem peso, mesmo FakeRecogna do candidato, empatada com a maior).
4. Rodar `experimentos.py modelo <combinação>` (E5), ex.: `modelo fr2021+ext` (com peso: `fr2021+extx3`).
5. Rodar `experimentos.py fontes <combinação>` (E6).
6. Rodar E7 (embeddings, demora na CPU):
   `experimentos.py embeddings <combinação> intfloat/multilingual-e5-small "query: "` e
   `experimentos.py embeddings <combinação> sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
7. Analisar, escolher o modelo final com justificativa e mostrar o resumo **antes** de qualquer avaliação no teste.

Obs.: `etapa_fontes` e `etapa_embeddings` não entendem o sufixo de peso (`extx3`); use nelas a combinação sem
peso (ex.: `fr2021+ext`). O parser da `etapa_modelo` foi corrigido em 01/10 ("ext" quebrava por conter "x").

Ponto de atenção registrado no E2b: no corpus externo as fake de treino vão até jul/2026 e as true são de
ago–out/2026 (possível atalho de "assunto da época"); comentar isso na análise final.

## 3. Prompt para colar no Claude Code

```
Leia README.md, EXPERIMENTOS.md e CONTINUAR.md antes de qualquer ação. Estamos na fase de experimentos para
aumentar o F1 em notícias externas. Regras: responda em português, didático, explicando o motivo de cada decisão;
registre cada ação, motivo e resultado em EXPERIMENTOS.md; NÃO use o teste_externo_2026.csv para escolher nada
(ele só será avaliado uma vez, no final, com minha autorização); commits sem coautoria do Claude.

Continue de onde parou, seguindo a seção 2.1 do CONTINUAR.md:
1. Confira o ambiente (.venv, sklearn 1.6.1). Se faltar algo, siga a seção 1 do CONTINUAR.md.
2. NÃO recolete nem rode `preparar` (o corpus e a divisão já estão no Git). Termine a E4 (pesos x3/x5) se faltar.
3. Rode `modelo <melhor combinação de dados>`, `fontes <combinação>` e `embeddings <combinação> <modelo>`
   (modelos: intfloat/multilingual-e5-small com prefixo "query: ",
   sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
4. Analise os resultados (diferenças < 1–2 pontos são empate), escolha o modelo final justificando, e me mostre
   o resumo antes de qualquer avaliação no teste final.
```
