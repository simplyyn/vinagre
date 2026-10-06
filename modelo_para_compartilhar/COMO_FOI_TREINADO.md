# Como o modelo foi treinado: dados, features, hiperparâmetros e motivos

Projeto acadêmico (Residência em IA) de classificação de notícias em português como **true** ou **fake**.
O objetivo **não era tirar a maior nota no dataset de referência**. Era chegar a um modelo que **funcione em
notícias de fora do dataset**, no formato do produto: **trechos de até ~50 palavras**.

> Resumo em uma linha: **TF-IDF de n-gramas de caracteres (2–5) + LinearSVC balanceado**, treinado com
> **FakeRecogna 2.0 (2020–21) + um corpus coletado de agências de checagem e portais de notícia (2022–2026)**,
> cada texto cortado em 30, 50 e 100 palavras. F1 macro **94,2** no teste final (116 notícias novas).

---

## 1. Ficha técnica

| Item | Valor |
|---|---|
| Arquivo | `modelo/modelo_vinagre.joblib` (no projeto: `modelos_finais/03.joblib`) |
| Tipo | `sklearn.pipeline.Pipeline` = `TfidfVectorizer` → `LinearSVC` |
| Biblioteca | **scikit-learn 1.6.1** (precisa ser essa versão para carregar o `.joblib` sem aviso) |
| Entrada | texto em português, cortado nas **primeiras 50 palavras** |
| Saída | `predict` → `"true"`/`"fake"`; `decision_function` → score (≥ 0 = true) |
| Representação | `analyzer="char_wb"`, `ngram_range=(2, 5)`, `max_features=200000`, `sublinear_tf=True`, minúsculas |
| Classificador | `LinearSVC(C=1.0, class_weight="balanced", max_iter=20000)` |
| Dados de treino | FakeRecogna 2.0 (2020–21) + corpus externo (parte de treino), aumento 30/50/100 palavras |
| Semente | 42 |
| Data | 05/10/2026 |

---

## 2. Dados

### 2.1 O que entrou no treino do modelo final

| Conjunto | O que é | Fake | True |
|---|---|---:|---:|
| **FakeRecogna 2.0, anos 2020–21** | dataset público (Hugging Face `recogna-nlp/fakerecogna2-abstrativa`) | 4.001 | 4.001 |
| **Corpus externo (treino)** | coletado pelo projeto via RSS, 2022–2026 | 645 | 1.349 |

Depois do aumento de dados (seção 3), o treino tem **29.988 trechos**.

**FakeRecogna 2.0.** São 52.800 notícias, metade fake e metade true. As fake vêm de 9 agências de checagem
(Boatos.org, E-farsas, Lupa, Aos Fatos etc.). As true são **resumos automáticos** de notícias reais. O
tratamento:

- **Label invertido:** no dataset, `1 = fake` e `0 = true`.
- **Só 2020 e 2021.** As true estão ~95% nesses dois anos, e as fake vão de 2013 a 2023. Treinar com todos os
  anos ensinou o modelo que "assunto de outra época = fake": o acerto em true caiu para ~66%.
- As fake foram separadas em três tipos:
  - **boato citado** (o texto começa com o boato entre aspas, ≥ 20 palavras): usamos **só o que está nas aspas**,
    porque é o boato em si;
  - **texto de checagem** (tem palavras como "falso", "boato", "checagem", "viralizou"): **descartado**, porque
    é a agência *descrevendo* o boato, não o boato;
  - **outros**: usados inteiros.
- As true foram sorteadas (`random_state=42`) em número igual ao de fake.

**Corpus externo** (`dados/corpus_externo.csv`). Nenhum dataset pronto tinha o formato do uso real (boato
citado × início de matéria jornalística atual), então o projeto coletou um:

| Classe | Fonte | O que é extraído | Treino | Validação |
|---|---|---|---:|---:|
| fake | Boatos.org | o boato citado na página (maior `blockquote`) | 567 | 243 |
| fake | E-farsas | idem | 78 | 34 |
| true | G1 | início da matéria, sem título | 1.277 | 547 |
| true | Agência Brasil | início da matéria, sem título | 72 | 31 |

- Foram 4.284 páginas brutas, depois remoção de textos curtos (fake < 20 palavras, true < 30), de duplicados e
  de qualquer texto parecido com o teste final (cosseno ≥ 0,8). Sobraram **2.849**.
- **Divisão treino/validação (70/30):** nas fake, os 30% **mais recentes** de cada agência vão para validação
  (imita o uso real, em que o boato é sempre novo). Nas true, 30% sorteados por fonte.
- A coluna `split` do CSV diz o que é treino e o que é validação.

### 2.2 O que **não** entrou (e por quê)

| Dataset | Por que ficou de fora do modelo final |
|---|---|
| **Fake.br** (7.200 notícias, 2016–2018) | Foi o ponto de partida do projeto, mas tem dois atalhos graves: **92,7% das fake vêm de um único site** (Diário do Brasil), e as true são **6× maiores** que as fake. Um modelo que só olha o tamanho do texto acerta 94%. Com o corpus externo no treino, somar o Fake.br **não ajudou** (89,7 × 90,5 de F1). |
| FakeRecogna 2.0, todos os anos | atalho de época (ver acima) |
| FakeRecogna versão 1 | texto lematizado e sem stopwords, diferente do uso real |

---

## 3. Pré-processamento e aumento de dados

1. **Truncamento misto 30/50/100.** Cada texto do treino entra **três vezes**, cortado nas primeiras 30, 50 e
   100 palavras.
   - *Por quê:* o tamanho do texto era o maior atalho dos dados. Um modelo treinado com texto inteiro
     classificou como FAKE quase todo trecho curto (acerto em true de 0,15% com 30 palavras). Treinar com
     trechos neutraliza o tamanho e prepara o modelo para o produto.
   - *Evidência:* o misto foi melhor que treinar só com 50 palavras nos três tamanhos de teste (73,8 / 75,5 /
     70,1 × 73,0 / 74,7 / 68,6 de acurácia balanceada).
2. **Minúsculas** (padrão do `TfidfVectorizer`).
   - *Por quê:* manter maiúsculas subiu a validação (92,3), mas **perdeu 3,5 pontos em fonte nova**. As
     maiúsculas parecem ser estilo de cada site, não de boato em geral.
3. **Nada mais.** Sem remoção de stopwords, sem lematização, sem mascarar nomes ou números. Testamos
   mascaramento (entidades, números, datas): ele **piorou** a generalização entre fontes.

---

## 4. Features

A única feature é o **vetor TF-IDF de n-gramas de caracteres** (explicação passo a passo em
`COMO_O_MODELO_FUNCIONA.md`).

### Features testadas e descartadas

| Feature | Resultado | Decisão |
|---|---|---|
| Tamanho do texto | 94% no Fake.br com texto inteiro, ~55% (chance) com trechos | atalho, não usar |
| 9 features linguísticas (spaCy: % verbos, % substantivos, pausalidade...) | 62,6% entre fontes × 74,3% do TF-IDF; peso ~0 quando combinadas | não usar |
| Proporção de números no texto | o sinal existe (true têm ~2× mais números), mas peso ~0 junto do TF-IDF | não usar |
| Embeddings (multilingual-e5-small) | empata (91,8 × 91,4), mas exige modelo de 470 MB e torch | não usar |
| Colunas `author`, `link`, `date`, `category` | vazam o label (ex.: `author` vazio em 98% das fake) | **proibido** |

---

## 5. Hiperparâmetros e por que cada um

| Hiperparâmetro | Valor | Por quê |
|---|---|---|
| `analyzer` | `char_wb` | pedaços de letras resistem a erros de digitação e grafias de boato; foi o mais estável em fontes novas |
| `ngram_range` | (2, 5) | de 2 letras (sílabas, pontuação) a 5 (quase palavras curtas inteiras) |
| `max_features` | 200.000 | char n-grams geram muito mais termos que palavras; 200k cobre o vocabulário útil |
| `sublinear_tf` | True | repetição ("URGENTE URGENTE") pesa, mas com retorno decrescente |
| `lowercase` | True (padrão) | maiúsculas são assinatura de site (ver seção 3) |
| `C` | 1,0 | 0,1 / 0,3 / 1 / 3 empataram (< 1 ponto); manteve-se o padrão |
| `class_weight` | `"balanced"` | o corpus externo tem 2× mais true que fake |
| limiar | score ≥ 0 → true | padrão do SVM; **não** foi ajustado no teste |

Foram ~40 configurações testadas na validação. As diferenças no topo eram menores que 2 pontos, ou seja,
empate. **O ganho grande veio dos dados, não do modelo:**

```
Fake.br + FakeRecogna (modelo antigo) ............ 82,9 F1
+ corpus externo (mesmo modelo, palavras) ......... 90,5 F1   ← o salto
+ n-gramas de caracteres ........................... 91,4 F1   ← empate técnico
```

---

## 6. Como o modelo foi escolhido e avaliado

Três conjuntos com papéis fixos, para nunca escolher nada olhando a resposta:

1. **Validação externa** (855 itens, a parte `val` do corpus externo): **todas** as escolhas de dados, modelo
   e hiperparâmetros.
2. **Teste por fonte (E6):** o modelo é treinado **sem** uma fonte de cada classe e testado só nela. A pergunta
   é se ele aprende "boato × jornalismo" ou só "site A × site B".
3. **Teste final** (`dados/teste_externo_2026.csv` + `dados/teste2_2026.csv`, 116 notícias de 28/09 a
   05/10/2026, coletadas depois do treino). A escolha foi **pré-registrada** antes, e o teste foi avaliado
   **uma única vez**. O script se recusa a rodar de novo.

### Por que este modelo e não o primeiro da validação

| Candidato | Validação (F1) | Fonte nova 1 | Fonte nova 2 |
|---|---:|---:|---:|
| palavras (1,2) [base] | 90,5 | 80,1 | 80,4 |
| **caracteres (2,5) [escolhido]** | **91,4** | **80,2** | **83,0** |
| palavras + caracteres, com maiúsculas | 92,3 | 76,6 | 81,8 |
| embeddings e5-small | 91,8 | não medido | não medido |

O escolhido está no grupo de topo e é o **único que nunca perdeu para a base em fontes novas**.

### Resultado no teste final (avaliado uma vez)

| Métrica | Valor |
|---|---|
| F1 macro | **94,2** (IC 95%: 89–99) |
| Acerto em notícias verdadeiras | **100%** (85 de 85) |
| Acerto em boatos | **83,9%** (26 de 31) |
| Erros | 5, todos boatos classificados como verdadeiros |

Os 5 erros são boatos **escritos como notícia** (ex.: "Zanin assume o TSE... tomou posse") ou como narrativa.
Os intervalos de confiança se sobrepõem com os outros modelos treinados com o corpus externo (~92), então o
teste **confirma a direção, mas não prova um vencedor**. Tabela completa em
`resultados/resultados_teste_final.csv`.

**Estimativa honesta:** F1 ~94 para notícias e boatos das mesmas fontes na semana seguinte ao treino; **~80–83
em fontes que o modelo nunca viu**.

---

## 7. Limitações

1. **Aprende estilo, não fatos.** Um boato escrito em padrão jornalístico tende a passar como verdadeiro.
2. **Não distingue boato de checagem.** Um texto "É falso que..." usa o vocabulário do boato e pode virar FAKE.
3. **Fontes concentradas.** True quase só do G1, fake quase só do Boatos.org. Em fontes novas o desempenho cai
   ~10 pontos (parte do que ele aprendeu é assinatura de site).
4. **Época.** No corpus externo, as fake vão até jul/2026 e as true são de ago–out/2026. O modelo pode ter
   aprendido em parte "assunto do momento = true". O teste final tem a mesma estrutura, então não detecta esse
   atalho.
5. **FakeRecogna:** as true são resumos automáticos, e o modelo aprende o estilo do resumidor. Prova concreta,
   achada ao preparar esta pasta: a marca `</s>` (fim de texto do resumidor) aparece em **14.003 das 26.400
   true e em nenhuma fake**, e virou um dos pesos mais fortes de TRUE. No uso real ninguém cola `</s>`, então a
   previsão não muda, mas o treino "gasta" parte do aprendizado nesse atalho. Removê-la é uma melhoria possível,
   e exigiria um teste novo.
6. **Amostra pequena no teste** (116 itens, só 1 do E-farsas): o intervalo de confiança é largo.
7. **Ruído:** o corpus externo tem 42 "especiais publicitários" do G1 marcados como true (achado depois do
   treino, não corrigido).
8. **Score não é probabilidade** (não foi calibrado).

---

## 8. Reproduzir

```bash
python treinar_do_zero.py
```

O script baixa o FakeRecogna, monta exatamente os mesmos dados, treina, mostra o F1 na validação externa
(esperado ~91,4) e salva `modelo/modelo_retreinado.joblib`.

**Conferido em 06/10/2026:** o modelo retreinado por este script saiu idêntico ao original (mesmo vocabulário,
mesmas previsões em todo o corpus, F1 de validação 91,43). Uma diferença de pesos só na 6ª casa decimal vem do
arredondamento de ponto flutuante.

Histórico completo dos experimentos (E0 a E8, com todas as tabelas): `EXPERIMENTOS.md` e `README.md` do
repositório https://github.com/simplyyn/vinagre. Relatório visual: `resultados/relatorio.html`.
