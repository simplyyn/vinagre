# Classificação de notícias falsas em português — Residência em IA

Projeto acadêmico de classificação de notícias em português como **true** (verdadeira) ou **fake** (falsa).
O foco atual **não é maximizar a acurácia no Fake.br**, e sim chegar a um modelo tecnicamente defensável que **generalize para notícias de fora do corpus**, no formato do produto final: **trechos incompletos de notícia (até ~50 palavras)**.

Este README é o contexto oficial e acumulado do projeto. Ele substitui qualquer histórico anterior.

---

## 1. Prompt inicial para o Claude Code

Cole o texto abaixo na primeira mensagem ao Claude Code, com este README na raiz do projeto:

```
Leia o README.md inteiro antes de qualquer ação. Ele é o contexto oficial do projeto e contém
decisões já tomadas, resultados, o papel de cada conjunto de dados e regras metodológicas.

Regras:
- Responda em português, de forma direta e didática. Explique o motivo das decisões técnicas.
- Não reinicie o projeto, não altere decisões já tomadas sem me avisar e não invente resultados.
- Siga a seção "Regras de trabalho" do README. Em especial: NUNCA avalie o teste externo final
  (teste_externo_2026.csv) sem minha autorização explícita. Ele só pode ser avaliado uma vez.
- Sempre diferencie: CV interna, validação por fonte, validação FakeRecogna, teste externo final
  e resultados históricos inválidos.
- Aponte qualquer inconsistência, possível leakage ou resultado metodologicamente questionável.

Primeira tarefa:
1. Configure o ambiente (seção "Ambiente").
2. Reconstrua o pipeline em scripts/notebook seguindo a seção "Pipeline" do README.
3. Reproduza dois números de controle antes de qualquer experimento novo:
   - baseline só tamanho (CV agrupada, texto completo): ~94,17% de accuracy;
   - TF-IDF com N=50 palavras (CV agrupada): ~89,89% de F1 macro.
   Se não bater (diferença maior que ~0,5 ponto), pare e me explique a diferença.
4. Depois, siga a seção "Próximos passos", começando pela coleta e conferência do teste externo.
   Não rode a avaliação final.
```

---

## 2. Regras de trabalho

1. **Não ajustar nada olhando o teste.** Hiperparâmetros, limiar, pesos, features e escolha de modelo só podem ser decididos com dados de treino ou validação.
2. **Papéis fixos dos conjuntos:**
   - **CV interna agrupada (Fake.br):** referência.
   - **Split por fonte (Fake.br):** validação da generalização entre fontes.
   - **FakeRecogna 2021:** validação (já foi usado para escolher variantes, então **não é mais teste**).
   - **`teste_externo_2026.csv`:** **único teste final**, avaliado **uma única vez**, com autorização do responsável.
3. **Scaler, TF-IDF e qualquer `fit` só no treino de cada fold.** Usar `Pipeline`.
4. **CV sempre agrupada por par** (`StratifiedGroupKFold`, `groups=pair_id`).
5. **Nunca usar como feature:** `author`, `link`, `date`, `category`, `pair_id`, domínio/veículo.
6. **Não afirmar causalidade.** Diferenciar observação de interpretação.
7. **Não comparar números de protocolos diferentes** como se fossem equivalentes.
8. **Diferenças menores que ~1–2 pontos são empate**, principalmente depois de testar várias configurações.
9. **Código:** comentário curto antes de cada bloco explicando o que ele faz; usar os nomes de variáveis já existentes.
10. **Texto para relatório:** português brasileiro, tom de aluno de Engenharia de Software, objetivo, sem exageros, mencionando limitações e sem esconder resultados ruins.

---

## 3. Dados

### 3.1 Fake.br (treino principal)

- Arquivo: `dataset_original_1_.csv` (7.200 notícias, 3.600 fake e 3.600 true).
- Colunas relevantes: `text`, `label`, `index` (renomeado para `pair_id`), features linguísticas pré-calculadas (`n_tokens`, `n_verbs`, `pausality` etc.), `author`, `link`, `date`, `category`.
- Tratamento aplicado:
  - labels normalizados para minúsculas (`false` → `fake`);
  - remoção do caractere BOM (`\ufeff`), presente em 10 textos true;
  - `index` → `pair_id`: **cada fake `i` tem uma true `i` da mesma categoria (3.600 pares, 100% de categoria igual)**;
  - **par 69 removido** (a true dos pares 61 e 69 é o mesmo texto) → 7.198 notícias, 3.599 por classe;
  - `n_links` com NaN → `fillna(0)` em `df_modelo`;
  - espaços múltiplos colapsados (sem efeito sobre o TF-IDF).
- Convenção de variáveis: `df` = dados originais; `df_modelo = df.copy()` = base dos modelos; `features` = 20 features absolutas; `features_normalizadas` = 9 features proporcionais.
- **Não há quebras de linha:** o título é a primeira frase do texto, terminada em ponto.
- **Colunas que vazam o label:** `author` está vazio em 98% das fake e em 2% das true; `link` identifica o site.
- `n_characters` no Fake.br conta **apenas caracteres alfanuméricos** (por isso `characters_per_word` ≈ `avg_word_len`).

**Tamanho do texto (achado central):**

| Classe | Média (caracteres) | Mediana | 25% | 75% |
|---|---:|---:|---:|---:|
| fake | 1.122 | 956 | 696 | 1.354 |
| true | 6.675 | 5.583 | 3.873 | 8.594 |

**Fontes (achado central):**

| Classe | Veículo | n |
|---|---|---:|
| fake | diariodobrasil.org | 3.337 (92,7%) |
| fake | afolhabrasil.com.br | 173 |
| fake | thejornalbrasil.com.br | 66 |
| fake | ceticismopolitico.com | 16 |
| fake | topfivetv.com | 7 |
| true | g1.globo.com | 2.299 |
| true | estadao (todos os subdomínios) | 1.203 |
| true | folha | 95 |
| true | outros | 2 |

### 3.2 FakeRecogna 2.0 (treino complementar e validação)

- Hugging Face: `recogna-nlp/fakerecogna2-abstrativa` (52.800 linhas, 26.400 por classe).
- **`Label`: 1 = fake, 0 = true** (invertido em relação à intuição).
- Texto em `Noticia`, **todo em minúsculas** (sem lematização e sem remoção de stopwords).
- **As true são resumos automáticos (abstrativos) de notícias reais**, não texto original.
- As fake vêm de 9 agências de checagem: boatos.org 8.579, e-farsas 3.327, Lupa 3.140, Aos Fatos 2.656, UOL 2.567, Fato ou Fake (G1) 2.265, AFP Checamos 1.586, Estadão Verifica 1.301 + 98, Comprova 877.
- Tipos de texto fake (classificação própria):
  - `fake_boato_citado` (2.158): começa com boato entre aspas → usa-se **só o conteúdo das aspas** (≥20 palavras);
  - `fake_texto_checagem` (6.630): contém vocabulário de agência (fake, falso, boato, checagem, viralizou…);
  - `fake_outros` (17.612): demais;
  - `true_resumida` (26.400).
- Anos (extraídos de `Data` ou da URL; 18,4% sem ano): **as true estão ~95% em 2020–2021**, enquanto as fake vão de 2013 a 2023. **Só 2020 e 2021 têm as duas classes em quantidade.**
- A versão 1 do FakeRecogna (`recogna-nlp/FakeRecogna`) foi **descartada**: o texto é lematizado e sem stopwords.

### 3.3 Teste externo final (avaliado em 05/10/2026)

- `teste_externo_2026.csv`: coletado automaticamente via RSS, sempre com os itens mais recentes, **sem escolha manual**:
  - true: Agência Brasil e G1 (início da matéria, até 60 palavras);
  - fake: Boatos.org e E-farsas (maior `blockquote` da página, que é o boato citado, até 60 palavras).
- Colunas: `texto`, `label`, `fonte`, `link`, `data`.
- **Status:** coletado (39 itens), conferido e **avaliado uma única vez em 05/10/2026**, junto com um 2º teste
  (`teste2_2026.csv`, 77 itens, notícias de 28/09 a 05/10; `coletar_teste2.py` + `conferir_teste2.py`). Resultados
  na seção 5.11 e no `EXPERIMENTOS.md`. Os dois testes não podem mais ser usados para escolher nada.
- Conferência permitida **antes** da avaliação: remover linhas apenas por conteúdo (trecho de fake que é descrição da agência; trecho de true que é lixo da página). **Nunca** por previsão do modelo.
- Ressalvas: o G1 é fonte de true no Fake.br (reportar as fontes separadamente); o Boatos.org é agência do FakeRecogna (usado no treino do candidato, mas em outro período).

### 3.4 Conjuntos históricos invalidados

- `noticias_teste` e `noticias_teste_2` (100 itens cada) **são sintéticos** (gerados por IA em outro chat): textos genéricos, sem nomes nem datas, e as "fake" são **descrições de boatos no estilo de checagem**, não o boato em si. Todos têm 1–2 frases (~200–300 caracteres).
- Os conjuntos reais de 10 e 50 notícias foram perdidos.
- **Todos os resultados externos históricos com 100 notícias são inválidos como generalização**: 66%, 81%, 88% do híbrido, a tabela de pesos, o "distribution shift" das features e a "inversão" de `pct_nouns`. Parte deles também tinha leakage (StandardScaler dos scores e limiar ajustados no próprio conjunto externo).
- A diferença de `characters_per_word` (4,8 vs 6,7) era, em boa parte, artefato: a função spaCy usava `len(texto)`, com espaços, enquanto o Fake.br conta só alfanuméricos.

---

## 4. Pipeline

### 4.1 Funções principais

```python
# Truncamento em N palavras.
def truncar(s, N):
    return s.str.split().str[:N].str.join(" ")

# Truncamento misto (aumento de dados): cada texto aparece cortado em 30, 50 e 100 palavras.
def aumentar(textos, labels):
    t, l = pd.Series(list(textos)), pd.Series(list(labels))
    return (pd.concat([truncar(t, n) for n in (30, 50, 100)], ignore_index=True),
            pd.concat([l] * 3, ignore_index=True))

# Modelo padrão do projeto.
def novo_tfidf():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), dtype=np.float32)),
        ("svm", LinearSVC(C=1.0, max_iter=20000))
    ])

# Mesmo modelo com classes balanceadas (usado quando o treino mistura corpora).
def novo_tfidf_bal():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), dtype=np.float32)),
        ("svm", LinearSVC(C=1.0, max_iter=20000, class_weight="balanced"))
    ])

# CV oficial: pares fake/true sempre no mesmo fold.
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

- **Truncamento para experimentos por N:** só entram pares em que **as duas** notícias têm pelo menos N palavras (todo texto fica com exatamente N palavras).
- **Split por fonte:** treino = pares em que as duas notícias são de diariodobrasil (fake) + g1 (true); teste = todos os textos das demais fontes; nenhum par dividido. Métricas: balanced accuracy e recall por classe.
- **Função única de features spaCy** (`pt_core_news_sm`), usada só em diagnóstico: verbos = `VERB` + `AUX` (o spaCy marca poder/dever como AUX); `characters_per_word` = caracteres alfanuméricos / palavras alfabéticas.
- **Decisão de limiar:** score ≥ 0 → true (classe positiva do LinearSVC é `true`, ordem alfabética).

### 4.2 Modelos salvos

- `modelo_final_fakebr.joblib`: TF-IDF word (1,2) + LinearSVC C=1, treino misto 30/50/100 em **todo o Fake.br**.
- `modelo_candidato_fakebr_fakerecogna.joblib`: mesmo modelo com `class_weight="balanced"`, treino misto em Fake.br completo + FakeRecogna **2020–2021** (fake dos tipos `boato_citado` e `outros`, sem `texto_checagem`, e o mesmo número de true resumidas amostradas com `random_state=42`).
- **`modelos_finais/03.joblib` (MODELO ESCOLHIDO, 05/10/2026):** TF-IDF `char_wb(2,5)` (200k features, sublinear) +
  LinearSVC C=1 balanceado, treino misto em FakeRecogna 2020–21 + treino do corpus externo (`fr2021 + ext`). Os demais
  arquivos de `modelos_finais/` são os outros métodos da avaliação final (ordem do `avaliacao_final.py`).
- Uso no produto: cortar a entrada em até 50 palavras e aplicar `decision_function` (≥ 0 → true). Demonstração:
  `testador_web.py` (navegador) ou `testar_noticia.py` (terminal).

---

## 5. Resultados (em ordem cronológica)

### 5.1 Baseline só tamanho (CV agrupada, texto completo)
`log1p(n_caracteres)` + StandardScaler + LinearSVC → **Accuracy 94,17% (std 0,68), F1 macro 94,16%**.
Conclusão: o desempenho interno histórico (~97%) vem em grande parte do tamanho do texto.

### 5.2 Truncamento (CV agrupada, F1 macro %)

| N | Pares | Só tamanho | TF-IDF | 9 features (spaCy) | Híbrido 50/50 |
|---:|---:|---:|---:|---:|---:|
| 30 | 3.545 | 55,77 | 87,35 | 64,48 | 81,52 |
| 50 | 3.500 | 56,90 | 89,88 | 66,35 | 83,30 |
| 100 | 2.993 | 55,45 | 90,94 | 69,44 | 85,41 |
| 200 | 1.129 | 53,66 | 88,62 | 68,21 | 83,12 |

- O truncamento neutraliza o tamanho (baseline ~54–57%).
- O TF-IDF mantém 87–91%.
- O híbrido (scaler dos scores em OOF do treino, peso fixo 50/50) **piora** o TF-IDF.
- N=200 é outra população (só fake longas), então não comparar diretamente.

### 5.3 Remoção do título (N=50, 3.453 pares, título = primeira frase)
Com título 89,50% → sem título 88,13% (−1,37). O título contribui pouco.
Primeira frase média: fake 11,7 palavras, true 19,2 (padrão G1: título + linha fina).

### 5.4 Mascaramento progressivo (N=50, F1 macro %)

| Versão | F1 | Queda |
|---|---:|---:|
| A original | 89,50 | — |
| B entidades (NER) | 88,06 | −1,43 |
| C + números e veículos | 87,85 | −1,65 |
| D + marcadores temporais | 87,27 | −2,23 |
| E + PROPN + termos políticos | 87,19 | −2,30 |
| F só palavras funcionais + classe gramatical | 82,01 | −7,48 |
| G só sequência de classes gramaticais | 80,15 | −9,35 |

- Tema, pessoas e época pesam pouco; o modelo aprende principalmente **estilo de redação**.
- Termos fortes: fake → "hoje", "ontem", "de acordo com", "através", "irá", "poderá", "pra", "eu", "vamos", "coréia" (grafia antiga); true → "diz", "afirma", "segundo", "nesta quarta-feira", "g1".
- Defeito conhecido: no nível E, "segundo" (de "segundo ela") foi mascarado como número.

### 5.5 Split por fonte (N=50; treino 2.091 pares DdB + G1)

| Versão | F1 mesma fonte | Bal. acc fonte nova | Recall fake | Recall true |
|---|---:|---:|---:|---:|
| A original | 91,49 | **74,29** | 75,10 | 73,49 |
| E mascarado | 90,27 | 71,23 | 71,26 | 71,20 |
| F funcionais | 85,22 | 68,48 | 74,33 | 62,63 |
| G só POS | 83,33 | 64,43 | 65,52 | 63,34 |

- Mudar de fonte custa ~17 pontos.
- **O estilo "puro" generaliza pior**: boa parte dele é assinatura de redação de cada site. O mascaramento serviu para diagnóstico e **não** entra no modelo final.
- Por veículo (versão A): A Folha do Brasil 76,2% (n=172), The Jornal Brasil 75,8% (n=66), Estadão 72,9% (n=1.174), Folha 81,1% (n=95).

### 5.6 Features linguísticas vs TF-IDF (split por fonte, balanced accuracy)

| Modelo | Bal. acc | Recall fake | Recall true |
|---|---:|---:|---:|
| TF-IDF | 74,29 | 75,10 | 73,49 |
| 9 features | 62,60 | 63,98 | 61,21 |
| Híbrido 50/50 | 69,65 | 72,03 | 67,27 |
| Stacking (LogReg em scores OOF do treino) | 74,56 | 78,54 | 70,57 |

Pesos do stacking: TF-IDF 4,359 e features 0,282. **Decisão: modelo final só com TF-IDF.**

### 5.7 Grid de hiperparâmetros (validação por fonte, 48 configurações)
- Top 10 entre 74,2% e 74,9% (empate).
- Médias por representação: char_wb (3,5) 72,02; word (1,1) 72,63; word (1,2) 73,01; word (1,3) 72,96.
- Médias por C: 0,1 → 72,11; 1 → 73,24; 10 → 72,62. `sublinear_tf=True` não entrou no top 10, e `min_df` não fez diferença.
- **Decisão: manter word (1,2), C=1, min_df=1, sublinear=False.** O limite é dos dados (fake de um só site), não do modelo.

### 5.8 Tamanho de treino (validação por fonte, balanced accuracy)

| Treino | Teste 30 | Teste 50 | Teste 100 |
|---|---:|---:|---:|
| N=50 | 73,04 | 74,67 | 68,58 |
| **Misto 30/50/100** | **73,81** | **75,54** | **70,14** |
| Texto completo | 50,08 | 50,35 | 51,72 |

- **O treino com texto completo classifica quase todo trecho curto como FAKE** (recall true 0,15% em 30 palavras). Isso reproduz com dados reais o colapso histórico e explica sua causa.
- Resíduo em N=100: o recall de fake cai (as fake curtas continuam curtas no treino). Recomendação para o produto: **entrada de até ~50 palavras**.

### 5.9 FakeRecogna 2.0 com o `modelo_final` (sem ajuste; todos os anos; até 50 palavras)
- True resumida × boato citado: recall true 69,9%, recall fake 69,5%, **balanced accuracy 69,7%**.
- Por faixa de tamanho (balanced accuracy): 20–29 palavras 66,0; 30–49 → 66,7; 50 → 71,5.
- % previsto como fake: `fake_outros` 63,4%, `fake_texto_checagem` 58,5%.
- True resumidas com menos de 20 palavras: 44,3% viram fake (resíduo de tamanho).

### 5.10 Treino combinado (treino = pares DdB+G1 + FakeRecogna 2020; validação = FakeRecogna 2021)

| Variante | FR2021 boato citado | FR2021 fake outros | Fake.br fontes novas | Checagem 2021 prevista fake |
|---|---:|---:|---:|---:|
| Base (só Fake.br) | 69,16 | 66,48 | 75,54 | 61,81% |
| **V1: + FR2020 fake e true** | **83,08** | **82,12** | 75,48 | 77,52% |
| V2: + FR2020 só fake | 59,73 | 59,51 | 73,48 | 97,09% |
| V1b: V1 sem fake descritivas | 83,43 | 82,01 | 75,48 | 76,59% |

- FakeRecogna 2020 no treino: 2.330 fake + 2.330 true.
- **V2 descartada**: aprendeu "tema da época = fake" (recall true 2021 de 22,8%).
- **V1 adotada**: ganho de ~14 pontos no FR2021 sem perda no controle Fake.br. Parte do ganho **pode** ser estilo do resumidor (as true de 2021 também são resumos). Só o teste externo com true originais resolve essa dúvida.
- V1b ≈ V1: filtrar fake descritivas não reduziu o efeito da checagem.

### 5.11 Fase de generalização externa (30/09–05/10/2026; detalhes no `EXPERIMENTOS.md`)

- **Corpus externo** 2022–2026 (Boatos.org, E-farsas, G1, Agência Brasil; 2.849 itens; 70% treino / 30% validação):
  formato do produto (boato citado × início de matéria). Toda escolha desta fase usa essa validação.
- **Dados (E4):** o corpus externo é o que mais ajuda: validação de ~83 (candidato) para ~90 (`fr2021 + ext`).
- **Modelo (E5–E7):** variações de TF-IDF e embeddings empatam (89–92). Escolhido `char_wb(2,5)` (91,4): topo da
  validação e único do topo que nunca perdeu para a base em fontes novas (E6: ~80–83).
- **Feature de números (E8):** as true têm mais números (no externo, ~2×), mas a feature não acrescenta nada ao
  TF-IDF (peso ~0 no SVM). Não usada.
- **Teste final (único, testes 1 + 2, 116 itens):** escolhido F1 **94,2 [IC 89–99]**, recall true 100%, recall fake
  83,9%. Os ICs se sobrepõem com os outros modelos treinados com o externo (~92) e com os antigos (~88): o teste
  confirma a direção, mas não prova vencedor. Os erros são boatos com redação de notícia.

---

## 6. Decisões vigentes

| Decisão | Escolha | Evidência |
|---|---|---|
| Representação | **TF-IDF char_wb (2,5)**, 200k features, sublinear (antes: word (1,2)) | E5 empate no topo; E6 mais robusto em fontes novas |
| Classificador | LinearSVC C=1, `class_weight="balanced"` | C e LogReg empatam (E5) |
| Embeddings | não usar | e5-small empata, mas é mais pesado e sem medida entre fontes (E7) |
| Feature de números | não usar | peso ~0 junto do TF-IDF (E8) |
| Features linguísticas | não usar | 62,6% entre fontes; peso ~0 no stacking |
| Híbrido | não usar | piora em todos os protocolos limpos |
| Mascaramento | não usar | generaliza pior entre fontes |
| Formato de treino | truncamento misto 30/50/100 | melhor nos 3 tamanhos; texto completo colapsa |
| Entrada no produto | até ~50 palavras | faixa mais equilibrada |
| Dados de treino | **FakeRecogna 2020–21 + corpus externo** (`fr2021 + ext`; antes: Fake.br + FR 2020–21) | E4: +7,5 pontos na validação externa; Fake.br não ajuda quando há o externo |

**Estimativa atual de desempenho (modelo escolhido):** F1 ~94 [89–99] em notícias de 2026 das mesmas fontes
(testes 1 + 2); ~80–83 em fontes que o modelo nunca viu (E6). Históricos: ~89% na mesma fonte no Fake.br; ~75% em
fontes novas do Fake.br (modelo antigo).

---

## 7. Limitações (para o relatório)

1. **Fake.br dominado por uma fonte:** 92,7% das fake vêm do Diário do Brasil. O modelo aprende em parte a distinção "Diário do Brasil vs G1/Estadão".
2. **Tamanho desigual entre as classes** (true ~6× maiores): atalho que só o truncamento neutraliza.
3. **O modelo aprende registro/estilo de redação** (jornalismo profissional vs site opinativo), não veracidade factual. Uma fake escrita em padrão jornalístico tende a passar.
4. **Limitação conceitual do TF-IDF:** não distingue divulgar um boato de noticiar sua checagem (textos de checagem tendem a ser classificados como fake).
5. **FakeRecogna:** true resumidas automaticamente (possível atalho de estilo do resumidor), fake misturando boato e texto de agência, e concentração temporal das true em 2020–2021.
6. **Época:** Fake.br de 2016–2018; FakeRecogna de 2018–2021 na parte usada.
7. **Os testes externos históricos eram sintéticos** e foram descartados.

---

## 8. Próximos passos

1. ~~Coletar e conferir o teste externo~~ (feito; mais um 2º teste).
2. ~~Avaliar uma única vez no teste~~ (feito em 05/10/2026, 8 métodos; seção 5.11).
3. Escrever o relatório final (base: `relatorio.html` e `EXPERIMENTOS.md`), separando interno, validação e teste, e
   com as limitações da seção 7. Apresentação: 09/10/2026.
4. Para medir de novo: teste 3 com notícias posteriores a 05/10, pré-registrado, só o modelo escolhido
   (roteiro na seção 3.2 do `CONTINUAR.md`).
5. Opcionais (não testados, não assumir que ajudam; exigem nova validação e um teste novo):
   - remover os 42 especiais publicitários do G1 do corpus externo;
   - no treino misto, usar em cada nível só textos com pelo menos N palavras (reduz o resíduo de tamanho em N=100);
   - calibrar scores para probabilidade (produto);
   - split temporal adicional.

---

## 9. Ambiente

```bash
pip install pandas numpy scikit-learn matplotlib seaborn spacy nltk datasets joblib feedparser trafilatura beautifulsoup4
python -m spacy download pt_core_news_sm
```

Arquivos esperados na pasta do projeto:
- `dataset_original_1_.csv` (Fake.br)
- `modelo_final_fakebr.joblib`
- `modelo_candidato_fakebr_fakerecogna.joblib`
- `teste_externo_2026.csv` (após a coleta)

Seed padrão: `42`.
