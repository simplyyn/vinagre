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

