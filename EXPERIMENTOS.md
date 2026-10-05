# Log de experimentos — generalização para notícias externas

Registro cronológico de cada ação, com o **motivo** e o **resultado**, para uso no relatório.
Protocolos são sempre identificados; números de protocolos diferentes não são comparados entre si.

**Regra fixa:** o `teste_externo_2026.csv` (39 itens, coletado em 30/09/2026) **não é usado em nenhum
experimento desta fase**. Toda escolha de modelo usa a validação externa descrita abaixo. O teste só será
avaliado uma vez, no final, com autorização.

---

## E0. Ambiente e reprodução (30/09/2026)

- **Ação:** Python 3.12 + venv; scikit-learn fixado em **1.6.1** (versão em que os `.joblib` foram salvos).
- **Por quê:** com sklearn 1.9.1 o carregamento dos modelos gerava `InconsistentVersionWarning`; além disso, a
  divisão dos folds do `StratifiedGroupKFold` muda entre versões.
- **Resultado (CV agrupada, sklearn 1.6.1):** só tamanho = 94,15% acc (std 0,68); TF-IDF N=50 = 89,88% F1 macro.
  Batem com o notebook (94,17% / 89,88%).
- **Modelos salvos conferidos:** `modelo_final_tfidf.joblib` é idêntico a um retreino pelo pipeline (vocabulário
  e decisões 100% iguais). O candidato retreinado tem 97,4% de vocabulário em comum e correlação de scores 0,9998
  com o salvo; diferença de 40 textos (0,16%) na marcação de "texto de checagem" do FakeRecogna, provavelmente
  versão do dataset/pandas do Colab. Efeito desprezível.

## E1. Teste externo: coleta e conferência (30/09/2026)

- **Ação:** coleta automática via RSS (`coletar_teste_externo.py`) e conferência só por conteúdo
  (`conferir_teste_externo.py`).
- **Correções automáticas em relação ao notebook (célula 59), e por quê:**
  - título removido nas duas fontes true: o trafilatura inclui o título na Agência Brasil e não no G1, o que criaria
    diferença sistemática entre fontes;
  - marcadores "Versão 1:", "Versão 2:" do Boatos.org removidos: são texto da agência, não do boato, e o candidato
    viu Boatos.org no treino (possível pista espúria);
  - legendas de foto ("— Foto: ...") e assinaturas com data do G1 removidas (lixo de página).
- **Remoções na conferência (5):** 3 trechos do E-farsas que eram desmentidos oficiais (IBGE, TSE, Coca-Cola) e
  2 traduções (espanhol/inglês) de boato já presente em português.
- **Resultado:** 39 itens (true: Agência Brasil 10, G1 15; fake: Boatos.org 13, E-farsas 1). **Não avaliado.**
- **Limitação registrada:** a regra "maior blockquote" do README não funciona no E-farsas (o boato quase sempre é
  vídeo/imagem e o blockquote costuma ser o desmentido).

## E2. Corpus externo para treino e validação

- **Ação:** coleta de um corpus 2024–2026 das mesmas fontes (`coletar_corpus_externo.py`): páginas antigas dos
  feeds do Boatos.org e do E-farsas (fake) e editorias/estados do G1 e da Agência Brasil (true).
- **Por quê:** nenhum dos corpora disponíveis tem o formato do teste. O Fake.br é de 2016–2018 com fake quase todas
  de um site; o FakeRecogna tem true resumidas automaticamente. Sem dados no formato do teste, não há como medir
  (nem melhorar) o F1 externo sem usar o próprio teste.
- **Cuidados contra contaminação:** nenhum link do teste; página 1 dos feeds do teste ignorada; feed
  `fato-ou-fake` do G1 excluído das true; versões em espanhol/inglês do Boatos.org excluídas.
- **Coleta:** links listados após os filtros — Boatos.org 1.768, E-farsas 585, G1 1.809, Agência Brasil 100.
  Textos extraídos em `dados_externos/corpus_externo_bruto.csv`.
- **Resultado:** a preparar (mínimo de palavras, duplicados, descontaminação e divisão treino/validação são feitos
  por `experimentos.py preparar`).

### E2b. Recoleta em outro PC e preparação (01/10/2026)

- **Ação:** em um PC novo, o `corpus_externo_bruto.csv` não estava no repositório; rodei de novo
  `coletar_corpus_externo.py` (mesmo código, mesmos filtros). Ambiente recriado (Python 3.12.10, sklearn 1.6.1) e
  números de controle conferidos de novo: só tamanho 94,15% acc; TF-IDF N=50 89,88% F1 macro (iguais ao E0).
- **Por quê recoletar e não esperar o CSV antigo:** o código e os filtros anti-contaminação são os mesmos; a
  diferença é só de um dia nos feeds.
- **Diferença em relação à coleta de 30/09:** links após filtros — Boatos.org 1.769 (+1), E-farsas 585 (=),
  G1 1.827 (+18), Agência Brasil 103 (+3). Os feeds andaram um dia, então o corpus **não é idêntico** ao
  original. Links do teste continuam excluídos (lista do `teste_externo_2026_bruto.csv`).
- **Extração:** 940 páginas do Boatos.org e 459 do E-farsas ficaram sem texto (sem blockquote; o boato é imagem ou
  vídeo — mesma limitação do E1). Nenhum erro de download.
- **`experimentos.py preparar`:** 4.284 brutos → 2.853 com mínimo de palavras → 2.849 sem duplicados → 2.849 após
  descontaminação (nenhum item com cosseno ≥ 0,8 com o teste).

| Split | Label | Fonte | n | De | Até |
|---|---|---|---:|---|---|
| treino | fake | Boatos.org | 567 | 2025-11-27 | 2026-07-19 |
| treino | fake | E-farsas | 78 | 2022-05-11 | 2024-07-20 |
| treino | true | Agência Brasil | 72 | 2026-09-24 | 2026-10-01 |
| treino | true | G1 | 1.277 | 2026-08-08 | 2026-10-01 |
| val | fake | Boatos.org | 243 | 2026-07-19 | 2026-09-27 |
| val | fake | E-farsas | 34 | 2024-07-23 | 2026-03-04 |
| val | true | Agência Brasil | 31 | 2026-09-26 | 2026-10-01 |
| val | true | G1 | 547 | 2026-08-07 | 2026-10-01 |

- **Observações para a análise:**
  - classes desbalanceadas (treino 645 fake × 1.349 true; validação 277 × 578) → por isso a métrica é F1 macro e
    os modelos usam `class_weight="balanced"`;
  - **descompasso temporal no treino:** as fake vão de 2022 a jul/2026, as true são todas de ago–out/2026. O modelo
    pode aprender "assunto da época" (mesmo risco da V2 na seção 5.10 do README). O teste tem a mesma estrutura
    (true de 30/09, fake recentes), então a validação não detecta esse atalho; a etapa E6 (fontes) ajuda só em parte;
  - E-farsas e Agência Brasil são pequenos (34 e 31 na validação): acerto por fonte nelas tem variação alta.

## E3. Base: modelos salvos na validação externa (01/10/2026)

- **Ação:** `experimentos.py base` — os dois `.joblib` avaliados na validação externa (855 itens, até 50 palavras).
- **Por quê:** ponto de partida para medir qualquer ganho.

| Modelo | F1 macro | Bal. acc | Rec. fake | Rec. true | Ag. Brasil | Boatos.org | E-farsas | G1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| modelo_final (só Fake.br) | 79,93 | 78,80 | 66,43 | 91,18 | 100,0 | 70,8 | 35,3 | 90,7 |
| modelo_candidato (Fake.br + FR 2020–21) | 82,94 | 83,01 | 77,26 | 88,75 | 96,8 | 81,1 | 50,0 | 88,3 |

- **Leitura:** o candidato é ~3 pontos melhor (no limite do empate), com ganho vindo do recall de fake (Boatos.org).
  O E-farsas é o ponto fraco dos dois (n=34).

## E4. Combinações de dados de treino (01/10/2026)

- **Ação:** `experimentos.py dados` — mesmo modelo (TF-IDF word (1,2) + LinearSVC balanceado, treino misto
  30/50/100), variando só os dados. `fb` = Fake.br; `fr2021` = FakeRecogna 2020–21 (como no candidato);
  `frall` = FakeRecogna todos os anos; `ext` = treino do corpus externo.

| Dados | F1 macro | Bal. acc | Rec. fake | Rec. true | Ag. Brasil | Boatos.org | E-farsas | G1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fb | 79,93 | 78,80 | 66,43 | 91,18 | 100,0 | 70,8 | 35,3 | 90,7 |
| fb + fr2021 | 82,18 | 82,30 | 76,53 | 88,06 | 96,8 | 80,7 | 47,1 | 87,6 |
| fb + frall | 70,21 | 73,35 | 78,70 | 67,99 | 96,8 | 84,0 | 41,2 | 66,4 |
| fr2021 | 75,11 | 77,33 | 79,06 | 75,61 | 90,3 | 81,9 | 58,8 | 74,8 |
| frall | 69,23 | 72,76 | 79,78 | 65,74 | 93,5 | 82,3 | 61,8 | 64,2 |
| ext | 88,34 | 87,00 | 77,98 | 96,02 | 100,0 | 81,9 | 50,0 | 95,8 |
| fb + ext | 87,64 | 86,29 | 76,90 | 95,67 | 100,0 | 82,3 | 38,2 | 95,4 |
| **fr2021 + ext** | **90,47** | **89,71** | 83,75 | 95,67 | 96,8 | 87,2 | 58,8 | 95,6 |
| frall + ext | 89,59 | 89,11 | 83,75 | 94,46 | 96,8 | 86,8 | 61,8 | 94,3 |
| fb + fr2021 + ext | 89,74 | 88,81 | 81,95 | 95,67 | 100,0 | 86,4 | 50,0 | 95,4 |
| fb + frall + ext | 89,09 | 88,29 | 81,59 | 94,98 | 96,8 | 87,2 | 41,2 | 94,9 |
| fb + fr2021 + ext (ext x3) | 89,86 | 88,90 | 81,95 | 95,85 | 96,8 | 86,4 | 50,0 | 95,8 |
| fb + fr2021 + ext (ext x5) | 88,91 | 88,01 | 80,87 | 95,16 | 96,8 | 85,2 | 50,0 | 95,1 |
| fb + frall + ext (ext x3) | 89,92 | 89,18 | 83,03 | 95,33 | 96,8 | 86,8 | 55,9 | 95,2 |
| fb + frall + ext (ext x5) | 89,36 | 88,55 | 81,95 | 95,16 | 96,8 | 86,8 | 47,1 | 95,1 |
| frall + ext (ext x3) | 90,69 | 90,27 | 85,56 | 94,98 | 96,8 | 87,2 | 73,5 | 94,9 |
| frall + ext (ext x5) | 90,09 | 89,46 | 83,75 | 95,16 | 96,8 | 86,4 | 64,7 | 95,1 |

- **Leitura:**
  - peso do externo (x3/x5): variações de < 1 ponto → **empate**; `frall + ext x3` é o maior número (90,69), mas a
    diferença para `fr2021 + ext` (90,47) é desprezível. A alta no E-farsas (73,5) é em 34 itens, não é confiável;
  - o corpus externo é o que mais ajuda: sozinho já vai de ~83 (candidato) para 88,3;
  - entre as combinações com `ext`, todas ficam entre 87,6 e 90,5 → **empate técnico** (diferença < ~2 pontos);
    `fr2021 + ext` está na frente numericamente;
  - `frall` sem o externo derruba o recall de true (~66–68%): mesmo atalho temporal já visto na V2 (fake de vários
    anos × true só de 2020–21);
  - o Fake.br não ajuda quando já há corpus externo (fb + ext ≈ ext; fb + fr2021 + ext ≈ fr2021 + ext);
  - E-farsas continua instável (34 itens; 38–62%).

### Escolha da combinação de dados (05/10/2026)

- **Decisão:** `fr2021 + ext` (sem peso) para E5, E6 e E7.
- **Por quê:** empata com a maior (`frall + ext x3`: 90,69 × 90,47; diferença < 1 ponto) e é a mais simples: sem
  repetição do externo e com o mesmo recorte do FakeRecogna do candidato. O `frall` mostrou o atalho temporal
  (recall de true ~66–68% sem o externo), então prefiro não depender dele. A alta do `frall + ext x3` no E-farsas
  (73,5) é em 34 itens e não sustenta a escolha.

## E5. Modelo (05/10/2026)

- **Ação:** `experimentos.py modelo fr2021+ext` — 12 variações de representação/classificador, mesmos dados e
  treino misto 30/50/100. Validação externa (855 itens, até 50 palavras).

| Modelo | F1 macro | Bal. acc | Rec. fake | Rec. true | Ag. Brasil | Boatos.org | E-farsas | G1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| word(1,2) C=1 [base] | 90,47 | 89,71 | 83,75 | 95,67 | 96,8 | 87,2 | 58,8 | 95,6 |
| word(1,2) C=0.1 | 90,31 | 90,01 | 85,56 | 94,46 | 96,8 | 88,9 | 61,8 | 94,3 |
| word(1,2) C=0.3 | 90,53 | 90,00 | 84,84 | 95,16 | 96,8 | 88,1 | 61,8 | 95,1 |
| word(1,2) C=3 | 89,61 | 88,73 | 81,95 | 95,50 | 96,8 | 86,0 | 52,9 | 95,4 |
| word(1,2) sublinear | 89,76 | 88,91 | 82,31 | 95,50 | 100,0 | 86,0 | 55,9 | 95,2 |
| word(1,2) sem max_features, min_df=2 | 89,88 | 88,99 | 82,31 | 95,67 | 96,8 | 86,0 | 55,9 | 95,6 |
| word(1,2) mantém maiúsculas | 89,76 | 88,91 | 82,31 | 95,50 | 100,0 | 86,4 | 52,9 | 95,2 |
| word(1,1) | 89,44 | 88,93 | 83,39 | 94,46 | 96,8 | 86,4 | 61,8 | 94,3 |
| char_wb(2,5) | 91,43 | 90,70 | 85,20 | 96,19 | 100,0 | 88,5 | 61,8 | 96,0 |
| word(1,2) + char_wb(2,5) | 91,29 | 90,52 | 84,84 | 96,19 | 100,0 | 87,7 | 64,7 | 96,0 |
| **word(1,2) + char_wb(2,5) sem lowercase** | **92,25** | **91,50** | 86,28 | 96,71 | 100,0 | 90,1 | 58,8 | 96,5 |
| LogReg word(1,2) C=10 | 90,07 | 89,36 | 83,39 | 95,33 | 96,8 | 86,8 | 58,8 | 95,2 |

- **Leitura:** tudo entre 89,4 e 92,3. Hiperparâmetros do word (C, sublinear, min_df, LogReg) → empate com a
  base, como no grid da seção 5.7 do README. As três variantes com n-gramas de caracteres ficam no topo
  (+0,8 a +1,8), mas dentro da faixa de empate; a E6 é usada para desempatar.

## E6. Fontes (05/10/2026)

- **Ação:** `experimentos.py fontes fr2021+ext "char_wb(2,5)" "word(1,2) + char_wb(2,5) sem lowercase"`. Treina sem
  uma fonte de cada classe do corpus externo e valida só nelas (treino + val dessas fontes).
- **Mudança no código:** `etapa_fontes` passou a aceitar nomes de modelos da E5 para comparar com a base no mesmo
  protocolo (antes só treinava a base).

| Treino | Sem E-farsas/Ag. Brasil (n=215) | Sem Boatos.org/G1 (n=2.634) |
|---|---:|---:|
| fr2021 (sem externo) | 76,87 | 75,60 |
| fr2021 + ext — base | 80,05 | 80,43 |
| fr2021 + ext — char_wb(2,5) | 80,20 | 82,99 |
| fr2021 + ext — word + char_wb sem lowercase | 76,55 | 81,82 |

- **Leitura:**
  - o corpus externo ajuda mesmo em fontes não vistas (+3 a +5): o ganho não é só assinatura de site. A queda de
    ~90 (mesmas fontes) para ~80 mostra que **parte** dele é;
  - `char_wb(2,5)` nunca fica abaixo da base (empate e +2,6);
  - o primeiro da E5 (sem lowercase) perde 3,5 pontos no primeiro cenário (E-farsas: 58,0): as maiúsculas
    parecem ser estilo de cada site. Menos robusto;
  - o primeiro cenário tem só 215 itens (112 E-farsas e 103 Ag. Brasil): diferenças de 3 pontos ainda são frágeis.

## E7. Embeddings (05/10/2026)

- **Ação:** `experimentos.py embeddings fr2021+ext intfloat/multilingual-e5-small "query: "` e
  `... sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Embeddings normalizados + regressão logística
  balanceada, treino em 50 palavras (sem aumento misto, por tempo de CPU).

| Modelo | C=0,3 | C=1 | C=3 | C=10 | Rec. fake (C=10) | Rec. true (C=10) | E-farsas (C=10) |
|---|---:|---:|---:|---:|---:|---:|---:|
| multilingual-e5-small | 89,28 | 89,75 | 90,46 | **91,77** | 87,00 | 95,67 | 61,8 |
| paraphrase-multilingual-MiniLM-L12-v2 | 83,79 | 85,81 | 86,12 | 87,01 | 76,90 | 94,81 | 41,2 |

- **Leitura:** o e5-small empata com o `char_wb(2,5)` (91,77 × 91,43). O melhor C está na borda da grade (10), então
  talvez suba um pouco mais, mas escolher C pela validação já é ajuste nela. O MiniLM fica abaixo da base TF-IDF.
  Os embeddings **não** passaram pela E6 (a etapa não suporta embeddings), então a robustez entre fontes deles é
  desconhecida.

## Análise e escolha do modelo final (05/10/2026) — antes de qualquer avaliação no teste

| Candidato | Val. externa (F1) | E6 sem E-farsas/AgB | E6 sem Boatos/G1 |
|---|---:|---:|---:|
| modelo_candidato salvo (E3) | 82,94 | — | — |
| fr2021 + ext, word(1,2) [base] | 90,47 | 80,05 | 80,43 |
| **fr2021 + ext, char_wb(2,5)** | **91,43** | **80,20** | **82,99** |
| fr2021 + ext, word + char_wb sem lowercase | 92,25 | 76,55 | 81,82 |
| fr2021 + ext, e5-small + LogReg C=10 | 91,77 | não medido | não medido |

- **Escolha: `fr2021 + ext`, TF-IDF `char_wb(2,5)` (200k features, sublinear) + LinearSVC C=1 balanceado,
  treino misto 30/50/100.**
- **Por quê:**
  1. está no grupo de topo da validação (empate técnico com os outros três, diferenças < 1 ponto);
  2. é o único do topo que **nunca perdeu para a base** na E6 (empate e +2,6). O primeiro numérico (sem lowercase)
     perde 3,5 pontos em fonte nova;
  3. entre empatados, o mais simples e já testado: o e5-small exige torch e modelo de 470 MB no produto, não usa o
     treino misto e não tem medida de robustez entre fontes;
  4. n-gramas de caracteres fazem sentido para boatos (grafia própria, erros de digitação, pontuação).
- **O ganho real está nos dados, não no modelo:** ~83 (candidato) → ~90,5 (corpus externo) → ~91,4 (char).
  O último passo é empate; o que muda a ordem de grandeza é o corpus externo.
- **Ressalvas para o relatório:**
  - descompasso temporal no corpus externo (fake até jul/2026 × true de ago–out/2026): possível atalho de "assunto
    da época". O teste tem a mesma estrutura, então nem a validação nem o teste detectam esse atalho;
  - fontes novas custam ~10 pontos (E6): parte do ganho é assinatura de site;
  - E-farsas é o ponto fraco de todos (34 itens na validação, 58–68%);
  - foram testadas ~40 configurações na mesma validação: o número de topo (~91–92) é otimista.
- **Estimativa antes do teste:** ~91 de F1 em fontes já vistas; ~80–83 em fontes novas.

## Pré-registro da avaliação final (05/10/2026) — escrito ANTES de ler qualquer label dos testes

- **Autorização:** o responsável autorizou em 05/10/2026 a avaliação única no `teste_externo_2026.csv` e a coleta de
  um 2º teste (`teste2_2026.csv`, notícias posteriores ao corpus; `coletar_teste2.py`).
- **Métodos avaliados (todos de uma vez, `avaliacao_final.py`):** 1) modelo_final salvo (Fake.br);
  2) candidato salvo (Fake.br + FR 2020–21); 3) TF-IDF palavras [base]; 4) **TF-IDF caracteres char_wb(2,5)
  [ESCOLHIDO]**; 5) palavras + caracteres sem lowercase; 6) e5-small + LogReg C=10; 7) só números (E8);
  8) TF-IDF palavras + números (E8). De 3 a 8: dados `fr2021 + ext` (split de treino), como na validação.
- **Métricas:** F1 macro (IC 95% por bootstrap), accuracy (IC 95% Wilson), balanced accuracy, recall por classe,
  matriz de confusão e acerto por fonte; teste 1, teste 2 e os dois juntos.
- **Teste 2 (coleta e conferência, antes da avaliação):** `coletar_teste2.py` + `conferir_teste2.py`. True publicadas
  a partir de 02/10 (40 mais recentes válidas por fonte); fake a partir de 28/09 (de 02/10 em diante só havia 9:
  semana da eleição; o E-farsas não publicou nada). Excluídos: links do corpus e do teste 1, quase-duplicatas
  (cosseno ≥ 0,8) do corpus, do teste 1 e do próprio teste 2. A eleição de 04/10 gerou milhares de páginas
  automáticas de resultado no Boatos.org (excluídas pela URL) e no G1 (12 removidas na conferência por serem
  texto-modelo). Também removidos: 7 especiais publicitários do G1 (anúncios) e 1 cabeçalho de página da Ag. Brasil.
  As 17 fake foram lidas: todas são o boato citado. **Final: 77 itens** (fake: Boatos.org 17; true: Ag. Brasil 39,
  G1 21).
- **Achado na conferência:** o corpus externo tem 42 especiais publicitários do G1 marcados como true (26 no treino,
  16 na validação). Não corrigido agora (corpus e modelos congelados); fica como limitação.
- **Regra de leitura definida antes:** o modelo escolhido **não muda** com o resultado do teste. Diferenças dentro
  dos ICs são empate. O teste não é usado para nenhuma decisão.

## Avaliação final ÚNICA (05/10/2026)

- **Ação:** `avaliacao_final.py avaliar` (uma vez; o script recusa rodar de novo). Saídas:
  `resultados_teste_final.csv` e `previsoes_teste_final.csv`. Entrada cortada em 50 palavras.

| Método | Teste 1 (n=39) | Teste 2 (n=77) | 1 + 2 (n=116) | Rec. fake (1+2) | Rec. true (1+2) |
|---|---:|---:|---:|---:|---:|
| Fake.br (modelo_final salvo) | 94,2 | 84,9 | 88,8 [81–95] | 80,6 | 95,3 |
| Fake.br + FR (candidato salvo) | 88,5 | 87,6 | 88,2 [80–94] | 74,2 | 97,6 |
| TF-IDF palavras [base] | 88,0 | 93,9 | 91,7 [85–97] | 77,4 | 100,0 |
| **TF-IDF caracteres [ESCOLHIDO]** | **88,0** | **98,1** | **94,2 [89–99]** | **83,9** | **100,0** |
| Palavras + caracteres sem lowercase | 84,6 | 96,1 | 91,7 [85–97] | 77,4 | 100,0 |
| e5-small + LogReg | 88,0 | 94,2 | 91,9 [85–97] | 80,6 | 98,8 |
| Só números (E8) | 61,4 | 69,8 | 67,3 [58–76] | 87,1 | 62,4 |
| TF-IDF palavras + números (E8) | 88,0 | 93,9 | 91,7 [85–97] | 77,4 | 100,0 |

(F1 macro; IC 95% por bootstrap entre colchetes.)

- **Leitura:**
  - o escolhido tem o maior F1 nos dois testes juntos (94,2), mas os ICs se sobrepõem com todos os modelos com
    corpus externo (91,7–91,9) e também com os modelos antigos (~88). A ordem confirma a validação; o teste, sozinho,
    não permite declarar vencedor;
  - **nenhuma true foi classificada como fake** pelo escolhido (85/85). Os 5 erros são fake → true: boatos com redação
    de notícia ("Zanin assume o TSE... tomou posse") ou narrativa (avião, "cientista preso entre linhas temporais").
    É a limitação 3 do README (o modelo aprende estilo, não fatos);
  - no teste 1 o modelo só Fake.br fica numericamente em primeiro (94,2), mas com 14 fake uma previsão vale ~4 pontos;
    no teste 2 ele é o pior (84,9). Instabilidade de amostra pequena, não evidência;
  - a feature de números isolada reproduz a validação (67 × 69): o sinal existe, mas é fraco sozinho e redundante com o
    TF-IDF (que empata com e sem ela, como na E8);
  - E-farsas tem 1 item nos testes: não dá para avaliar essa fonte.
- **Estimativa final para o relatório:** F1 ~94 [89–99] em boatos e notícias das mesmas fontes, na semana seguinte
  ao treino; ~80–83 esperado em fontes novas (E6).

## E8. Feature de números (05/10/2026, fora do plano original)

- **Hipótese (do responsável):** notícia verdadeira cita mais números (datas, valores, percentuais, quantidades)
  porque trata de fatos exatos. A feature é a **proporção** de palavras que são número, e não a contagem.
- **Ação:** `exp_numeros.py` (script separado; não altera o pipeline). Conta palavras com dígito, separadas em
  ano, percentual, dinheiro, hora/data e outros, mais números por extenso ("dez", "mil", "milhões"; "um/uma"
  ficam de fora por serem quase sempre artigo). Dados `fr2021 + ext`, treino misto 30/50/100, validação externa
  até 50 palavras. Ambiente recriado em PC novo antes (controles: 94,15% / 89,88%, iguais ao E0).
- **"Relativizar" o percentual:** `pct_suave = (n_numeros + m·p0) / (n_palavras + m)`, com m = 20 e p0 = média do
  treino. Em trecho curto o percentual cru é instável (1 número em 8 palavras = 12,5%); a suavização só confia no
  percentual quando há palavras suficientes. Com 50 palavras fixas, contagem e percentual são equivalentes.

**Descritivo (trechos de até 50 palavras; % das palavras que são número):**

| Corpus | fake | true | AUC do pct isolado |
|---|---:|---:|---:|
| Fake.br (N=50) | 2,13 | 3,12 | 0,61 |
| FakeRecogna 2020–21 | 3,40 | 3,79 | 0,56 |
| Externo (treino) | 2,41 | 5,21 | 0,76 |

- No externo, a diferença aparece nas duas fontes true (Agência Brasil 5,4; G1 5,2) e nas duas fake (Boatos.org
  2,6; E-farsas 1,2): não é assinatura de um site só. As categorias que mais separam são "outros dígitos"
  (quantidades) e hora/data (padrão de redação do G1: "nesta quarta-feira (1º)"). Percentual e dinheiro: ~nada.

**Validação externa (855 itens):**

| Modelo | F1 macro | Bal. acc | Rec. fake | Rec. true | Ag. Brasil | Boatos.org | E-farsas | G1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| só tamanho (controle) | 49,66 | 54,69 | 9,39 | 100,0 | 100,0 | 6,2 | 32,4 | 100,0 |
| só números: contagem | 69,16 | 71,41 | 72,92 | 69,90 | 54,8 | 71,6 | 82,4 | 70,7 |
| só números: pct | 68,87 | 71,05 | 72,20 | 69,90 | 54,8 | 71,2 | 79,4 | 70,7 |
| só números: pct_suave | 68,87 | 71,05 | 72,20 | 69,90 | 54,8 | 71,2 | 79,4 | 70,7 |
| só números: categorias | 69,45 | 72,45 | 77,26 | 67,65 | 48,4 | 77,0 | 79,4 | 68,7 |
| TF-IDF [base] | 90,47 | 89,71 | 83,75 | 95,67 | 96,8 | 87,2 | 58,8 | 95,6 |
| TF-IDF + pct_suave | 90,47 | 89,71 | 83,75 | 95,67 | 96,8 | 87,2 | 58,8 | 95,6 |
| TF-IDF + categorias (peso 0,3 e 1) | 90,49 | 89,81 | 84,12 | 95,50 | 96,8 | 87,2 | 61,8 | 95,4 |
| TF-IDF com dígitos mascarados (2026 → 0000) | 90,70 | 89,79 | 83,39 | 96,19 | 100,0 | 85,6 | 67,6 | 96,0 |

- **Por que não muda nada:** o LinearSVC dá peso ~0 às features de números (coeficiente −0,01 contra 0,20 de
  média por termo do TF-IDF; 0 a 4 previsões diferentes do base em 855). O TF-IDF já contém essa informação
  (tokens "2026", "14h30", "milhões" e as palavras que acompanham números). Mesmo padrão do stacking da
  seção 5.6 do README (features com peso ~0).
- **Fontes (como a E6):** sem E-farsas/Ag. Brasil no treino, base 80,05 e + categorias 80,05 (igual); sem
  Boatos.org/G1, base 80,43 e + categorias 78,53 (−1,9, no limite do empate, para pior). Dígitos mascarados:
  80,10 e 76,64 (−3,8 no segundo cenário).
- **Fake.br (CV agrupada, N=50):** base 89,88 e + categorias 89,73 (empate); só números 57,60.
- **Conclusão:** a hipótese é **verdadeira como observação** (true têm mais números nos três corpora), mas a
  feature **não acrescenta** nada ao TF-IDF: o sinal é real, só que já está no texto. Isolada, ela chega a
  ~69 de F1. **Decisão: não entra no modelo.** O mascaramento de dígitos empata na validação e piora entre fontes,
  por isso também não entra. Resultados registrados em `resultados_experimentos.csv` (grupos "E8 números" e
  "E8 números (fontes)"); **não rodar `exp_numeros.py modelo/fontes` de novo** (duplicaria linhas).

## Planejamento dos próximos experimentos (definido antes de ver qualquer resultado)

Métrica principal: **F1 macro na validação externa**, com a entrada cortada em 50 palavras (formato do produto).
Também: balanced accuracy, recall por classe e acerto por fonte.

| Etapa | O quê | Por quê |
|---|---|---|
| E3 base | modelos salvos na validação externa | ponto de partida |
| E4 dados | combinações de Fake.br, FakeRecogna (2020–21 e todos os anos) e corpus externo; peso do externo (x3, x5) | o limite identificado até aqui é de dados (fake de um site só), não de modelo |
| E5 modelo | C, sublinear, n-gramas de caracteres, maiúsculas, união word+char, regressão logística | boatos usam CAPS, pontuação e grafia próprias; char n-grams capturam isso e resistem a erros de digitação |
| E6 fontes | treinar sem E-farsas/Agência Brasil e validar nelas (e vice-versa com Boatos.org/G1) | separar generalização real de assinatura de site |
| E7 embeddings | multilingual-e5-small e MiniLM multilíngue + regressão logística | representação semântica pode depender menos do vocabulário de cada fonte |

Divisão do corpus externo: fake → 30% mais recentes de cada agência na validação (imita o teste, que é o mais
recente); true → 30% aleatório por fonte (todas as true são recentes). Itens quase idênticos (cosseno ≥ 0,8) a
textos do teste são removidos antes (usa só o texto do teste, sem label e sem modelo).

