# Como o modelo funciona (explicação sem jargão)

O modelo recebe um **trecho de notícia de até 50 palavras** e responde **TRUE** (parece notícia de veículo
jornalístico) ou **FAKE** (parece boato). Ele faz isso em duas etapas.

```
texto ──► corta em 50 palavras ──► TF-IDF de caracteres ──► LinearSVC ──► score ──► score ≥ 0 ? TRUE : FAKE
                                   (texto vira números)     (soma com pesos)
```

## Etapa 1: transformar texto em números (TF-IDF de n-gramas de caracteres)

Um computador não entende palavras, então o texto vira uma lista de números.

1. **Quebrar em pedaços de 2 a 5 letras.** A palavra `urgente` vira os pedaços ` u`, `ur`, `urg`, `urge`,
   `rg`, `rge`, `rgen`, `gente`… O espaço no começo e no fim marca a borda da palavra. Esses pedaços se chamam
   *n-gramas de caracteres*. O `char_wb` do scikit-learn faz isso **dentro de cada palavra**, sem misturar uma
   palavra com a próxima.
2. **Contar quantas vezes cada pedaço aparece** no trecho. Usamos `1 + log(contagem)` (`sublinear_tf`): repetir
   "URGENTE" cinco vezes conta mais que uma vez, mas não cinco vezes mais.
3. **Dar mais peso aos pedaços raros (IDF).** Um pedaço que aparece em todo texto, como `de`, não ajuda a
   separar nada. Um pedaço que aparece em poucos textos, como `compartilhe`, ajuda muito.
4. O resultado é um vetor com **200.000 posições** (uma por pedaço do vocabulário), quase todas zero.

**Por que caracteres e não palavras inteiras?** Boatos têm grafia e pontuação próprias: "pra", "vc", "!!",
espaço antes da vírgula, asteriscos. Um TF-IDF de palavras **joga fora a pontuação**; o de caracteres a mantém,
e ela acabou sendo o sinal mais forte (tabela abaixo). Além disso, "compartilhem", "compartilhe" e
"compartilhar" dividem o pedaço `tilha`, então uma forma que o modelo nunca viu ainda é reconhecida. Nos testes, essa
versão empatou com a de palavras nas fontes já conhecidas e foi a mais estável em fontes novas (detalhes em
`COMO_FOI_TREINADO.md`).

## Etapa 2: decidir (LinearSVC)

O LinearSVC (máquina de vetores de suporte linear) aprendeu, no treino, **um peso para cada um dos 200 mil
pedaços**:

- peso **positivo** → o pedaço é típico de notícia de veículo;
- peso **negativo** → o pedaço é típico de boato.

Pesos reais do modelo, tirados de `svm.coef_` (o espaço `␣` marca borda de palavra):

| Mais típicos de **FAKE** | peso | Mais típicos de **TRUE** | peso |
|---|---:|---|---:|
| `!␣` (exclamação) | −4,23 | `␣a␣` (artigo "a") | +3,65 |
| `␣-␣`, `␣–␣` (travessão solto) | −2,5 a −3,1 | `␣o␣` (artigo "o") | +3,04 |
| `␣[`, `␣*` (colchetes, asteriscos) | −2,1 a −2,5 | `.␣` (frase terminada normalmente) | +2,29 |
| `␣.`, `␣,` (espaço **antes** da pontuação) | −1,8 a −2,4 | `␣(` (parênteses: "nesta quarta-feira (1º)") | +1,91 |
| `␣post`, `ostag` (post, postagem) | −1,4 a −1,6 | `disse`, `isse␣` | +1,3 a +1,8 |
| `␣eu␣`, `mos␣` (1ª pessoa: "eu", "vamos") | −1,5 | `␣de␣`, `␣da␣`, `␣do␣` | +1,2 a +1,5 |
| `␣pra␣` | −1,40 | `casos` | +1,10 |
| `!!` | −1,30 | `-pref` (vice-prefeito, ex-prefeito) | +1,03 |
| `menti`, `storc`, `␣omit` (mentira, distorce, omite) | −1,3 a −1,4 | `</s>` ⚠️ artefato, ver abaixo | +1,28 |
| `tilha` (compartilha) | −1,23 | `nesta` ("nesta terça-feira") | +0,66 |
| `␣lula` | −1,16 | | |

Lendo a tabela: **pontuação e formatação pesam mais que qualquer palavra.** Boato tem exclamação, travessão
solto, asterisco e espaço antes da vírgula. Notícia tem frase completa com artigo, parênteses e "disse".

Três achados para ter cuidado:

- **`</s>` é um artefato.** É uma marca de "fim de texto" que o resumidor automático do FakeRecogna deixou
  em algumas notícias verdadeiras. O modelo aprendeu que ela indica TRUE. No uso real ninguém cola `</s>`,
  então ela não atrapalha a previsão, mas mostra que parte do que o modelo aprendeu com o FakeRecogna é
  estilo do resumidor (limitação já prevista no projeto).
- **Vocabulário de agência de checagem aparece como FAKE** ("distorce", "omite", "mentira", "postagem",
  "diz"). Ele vem dos textos fake do FakeRecogna que misturam boato e descrição da agência. Por isso um texto
  de checagem ("Post distorce fala de...") tende a ser classificado como FAKE.
- `␣lula` como FAKE reflete a época: muitos boatos recentes citam o presidente. É um atalho de assunto, não
  de estilo.

Para classificar, ele multiplica cada número do vetor pelo seu peso e soma tudo:

```
score = soma(peso de cada pedaço × valor do pedaço no texto) + constante
score ≥ 0  → TRUE
score < 0  → FAKE
```

O **score** diz o quanto o texto está longe da fronteira. Perto de 0 o modelo está em dúvida. Score não é
probabilidade: um score de 0,8 **não** significa 80% de chance.

`class_weight="balanced"` faz o erro em uma classe com menos exemplos pesar mais no treino. Assim o modelo não
fica puxado para a classe que tem mais textos.

## Por que cortar em 50 palavras?

No dataset original, as notícias verdadeiras eram **6 vezes maiores** que as falsas. Um modelo treinado com o
texto inteiro aprendia "texto longo = verdadeiro" e chegava a 94% de acerto **só olhando o tamanho**. Ao
receber um trecho curto de notícia real, ele dizia que era FAKE.

A solução foi treinar com cada texto cortado em 30, 50 e 100 palavras, e usar até 50 palavras na prática. Assim
o tamanho deixa de ser pista e o modelo precisa olhar **como o texto foi escrito**.

## O que o modelo aprende de verdade (e o que não aprende)

O modelo **não verifica fatos**. Ele não sabe quem é o presidente nem se uma vacina causa algo. Ele aprende
**estilo de redação**: como escreve um jornalista de agência e como escreve quem espalha boato no WhatsApp.

Por isso:

- boato escrito como mensagem de corrente ("URGENTE!!! Compartilhem antes que apaguem") → **FAKE** com folga;
- boato escrito com cara de notícia ("O ministro assinou nesta terça-feira o decreto...") → tende a passar como
  **TRUE**. Os 5 erros do teste final foram todos desse tipo;
- texto de agência de checagem que **descreve** um boato ("É falso que...") pode ser marcado como FAKE, porque
  usa o vocabulário do boato.

## Exemplo rápido em Python

```python
import joblib

modelo = joblib.load("modelo/modelo_vinagre.joblib")   # Pipeline: TF-IDF + LinearSVC
trecho = " ".join("URGENTE!!! Compartilhem antes que apaguem ...".split()[:50])

print(modelo.predict([trecho]))            # ['fake'] ou ['true']
print(modelo.decision_function([trecho]))  # score: >= 0 true, < 0 fake

# Inspecionar o modelo por dentro:
tfidf = modelo.named_steps["tfidf"]        # TfidfVectorizer
svm = modelo.named_steps["svm"]            # LinearSVC
print(len(tfidf.vocabulary_))              # 200000 pedaços
print(svm.coef_.shape)                     # (1, 200000) pesos
```
