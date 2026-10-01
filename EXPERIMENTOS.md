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

