# Como retomar o trabalho

## 1. Preparar o ambiente (só se for um PC novo ou se o `.venv` não existir)

```powershell
# Python 3.12 (se não houver): winget install -e --id Python.Python.3.12 --scope user
# Git (se não houver):         winget install -e --id Git.Git --scope user
py -3.12 -m venv .venv   # ou: & "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install sentence-transformers "scikit-learn==1.6.1"
.venv\Scripts\python.exe -m spacy download pt_core_news_sm
```

O scikit-learn **tem que ser 1.6.1** (versão em que os `.joblib` foram salvos). Conferir:
`.venv\Scripts\python.exe -c "import sklearn; print(sklearn.__version__)"`.

No PC usado em 05/10 a pasta do projeto é
`C:\Users\24018503\vinagre-src\vinagre-6f8fe7b8d85e567213a7831a10de6049d12ee37a` (repositório Git ligado ao GitHub;
`git push` funciona pelo Git Credential Manager). Em outro PC: `git clone https://github.com/simplyyn/vinagre`.

## 2. Estado em 05/10/2026 (fase de experimentos ENCERRADA)

- **Modelo escolhido:** `fr2021+ext`, TF-IDF `char_wb(2,5)` + LinearSVC balanceado, treino misto 30/50/100.
  Arquivo: `modelos_finais/03.joblib`. Entrada no produto: até 50 palavras; `decision_function` ≥ 0 → true.
- **Validação externa:** F1 91,4. **Testes finais (avaliados uma vez, autorizados):** teste 1 (39 itens) + teste 2
  (77 itens, notícias de 28/09 a 05/10) → **F1 94,2 [IC 89–99]**, recall true 100% (85/85), recall fake 83,9%.
  Os 5 erros são boatos com redação de notícia ou narrativa. Fontes novas (E6): ~80–83.
- Tudo registrado no `EXPERIMENTOS.md` (E0–E8, escolha, pré-registro, avaliação final).
- **Relatório da apresentação (sexta, 09/10):** `relatorio.html` (abrir no navegador) ou o link publicado
  https://claude.ai/artifact/VYGKPhFndmkPikg617fFXg. Regenerar: `.venv\Scripts\python.exe dados_relatorio.py`.
- **Os testes 1 e 2 já foram usados:** não servem mais para escolher ou ajustar nada.

### Não rodar de novo
- `coletar_corpus_externo.py`, `experimentos.py preparar/dados/modelo/fontes/embeddings`, `exp_numeros.py modelo/fontes`
  (mudaria o corpus ou duplicaria linhas no CSV);
- `coletar_teste2.py` (mudaria o teste 2); `avaliacao_final.py congelar/avaliar` (a avaliação é única; o `avaliar`
  recusa rodar de novo).

## 3. Testar mais o modelo escolhido

### 3.1 Teste manual (demonstração, à vontade)

```powershell
.venv\Scripts\python.exe testador_web.py --emb      # abrir http://127.0.0.1:8000, colar a notícia, Analisar
.venv\Scripts\python.exe testar_noticia.py           # mesmo teste no terminal (linha vazia analisa, "sair" encerra)
.venv\Scripts\python.exe testar_noticia.py --arquivo noticias.txt   # uma notícia por linha
```

- A primeira linha é o modelo escolhido; as outras são comparação. Confiança perto de 0 = caso de dúvida.
- Usar trechos de 30–50 palavras (true: início da matéria sem título; fake: o boato citado, não o texto da agência).
- **Isto é demonstração, não medida de desempenho:** notícias escolhidas à mão não são amostra representativa.
  Nunca mudar o modelo por causa de acertos/erros aqui.
- Casos bons para a apresentação: boato de WhatsApp (CAIXA ALTA, "URGENTE") → FAKE com confiança alta; boato com
  redação de notícia → tende a passar como TRUE (limitação esperada).

### 3.2 Teste 3 (medida nova e válida do modelo escolhido)

Para medir de novo com rigor, coletar um **teste 3** com notícias publicadas depois de 05/10 (de preferência daqui a
1–2 semanas, para ter mais boatos):
1. Copiar `coletar_teste2.py` e `conferir_teste2.py` para `coletar_teste3.py` e `conferir_teste3.py`, trocando
   `DESDE`/`DESDE_FAKE` para 06/10/2026, os nomes de saída para `teste3_2026*` e acrescentando os links do teste 2 na
   lista de exclusão.
2. Conferir só por conteúdo (as mesmas regras: boato citado, sem anúncios, sem páginas automáticas, sem cabeçalho).
3. **Antes de avaliar**, pré-registrar no `EXPERIMENTOS.md` que só o modelo `modelos_finais/03.joblib` será avaliado,
   sem mudanças. Avaliar uma vez (reaproveitar `metricas()` do `avaliacao_final.py`) e registrar.

### 3.3 Melhorias possíveis (só com validação nova, nunca olhando os testes)
- Remover os 42 especiais publicitários do G1 do corpus externo (26 treino, 16 validação) e revalidar.
- Mais boatos do E-farsas em texto (hoje quase todos são imagem/vídeo).
- Medir o atalho de época (fake até jul/2026 × true ago–out/2026) com um split temporal.
- Calibrar o score em probabilidade (para o produto).
Qualquer mudança no modelo exige um teste novo depois (o 3 ou outro), pois os testes 1 e 2 já foram usados.

## 4. Prompt para colar no Claude Code

```
Leia README.md, EXPERIMENTOS.md e CONTINUAR.md antes de qualquer ação. A fase de experimentos terminou: o modelo
escolhido é modelos_finais/03.joblib (TF-IDF char_wb(2,5), fr2021+ext) e os testes 1 e 2 já foram avaliados uma vez.
Regras: responda em português, didático, explicando o motivo de cada decisão; registre cada ação, motivo e resultado
em EXPERIMENTOS.md; NÃO use os testes 1 e 2 para escolher ou ajustar nada; qualquer teste novo só é avaliado uma vez,
com minha autorização e pré-registro; commits sem coautoria do Claude.

Confira o ambiente (.venv, sklearn 1.6.1; seção 1 do CONTINUAR.md se faltar algo). Quero continuar testando o modelo
escolhido seguindo a seção 3 do CONTINUAR.md: <descreva aqui o que quer fazer — teste manual, teste 3 ou melhoria>.
```
