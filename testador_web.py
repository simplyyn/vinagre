"""Testador no navegador: cole uma notícia e veja a previsão do modelo escolhido (versão web do testar_noticia.py).

Mostra o veredito do modelo escolhido (modelos_finais/03.joblib), quais palavras empurraram a decisão para FAKE
ou para TRUE e, para comparação, a previsão dos outros modelos.

Uso: .venv\\Scripts\\python.exe testador_web.py [--emb] [--porta 8765]
Depois abra http://localhost:8765 no navegador. Ctrl+C no terminal encerra.
Só roda na sua máquina (localhost); nada é enviado para fora.
"""
import json
import sys
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd

from anls import truncar
from exp_numeros import contar_numeros
from testar_noticia import N_PRODUTO, carregar, prever

PORTA = 8765  # a 8000 costuma estar ocupada por outros servidores de desenvolvimento

# Exemplos reais dos testes finais (já avaliados; servem só para demonstração).
EXEMPLOS = [
    ("Boato em CAIXA ALTA", "URGENTE! PASSAPORTE DE FLÁVIO BOLSONARO É APREENDIDO! KÁSSIO NUNES RECUA! FLÁVIO "
     "BOLSONARO EM PÂNICO Descubra a VERDADE que eles estavam ESCONDENDO de você A apreensão do passaporte de Flávio "
     "Bolsonaro é uma BOMBA que abala os pilares da política nacional!"),
    ("Boato de WhatsApp", "Atenção, aposentado: se você não votar, seu benefício poderá ser cancelado. O voto agora "
     "serve como prova de vida. Lula está desesperado."),
    ("Notícia do G1", "Belém registrou 37 °C de temperatura máxima no sábado (3), a maior temperatura para o mês de "
     "outubro em 25 anos, segundo o Instituto Nacional de Meteorologia (Inmet) informou ao g1 nesta segunda-feira (5)."),
    ("Notícia da Agência Brasil", "A projeção do mercado financeiro para a inflação oficial em 2026 subiu de 4,99% "
     "para 5,01%, segundo o Boletim Focus divulgado nesta segunda-feira (5) pelo Banco Central. Foi a terceira alta "
     "consecutiva da estimativa."),
    ("Boato com cara de notícia (o modelo erra)", "ZANIN ASSUME O TSE ÀS VÉSPERAS DAS ELEIÇÕES DE 2026 O ministro "
     "Cristiano Zanin, que já atuou como advogado de Lula, tomou posse na presidência do Tribunal Superior Eleitoral "
     "(TSE). A mudança acontece em um momento decisivo."),
]

PAGINA = r"""<!doctype html>
<html lang="pt-br"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vinagre · Detector de boatos</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&family=Literata:opsz,wght@7..72,400..600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root {
  --papel: #f6f7f4; --folha: #ffffff; --tinta: #14181c; --tinta-2: #4b5258; --tinta-3: #868d92;
  --fio: #dfe3e0; --fake: #d4561f; --fake-f: rgba(212, 86, 31, .16); --true: #2a6fd0; --true-f: rgba(42, 111, 208, .14);
  --destaque: #4a3aa7; --sombra: 0 1px 2px rgba(20,24,28,.06), 0 8px 24px rgba(20,24,28,.06);
  --display: "Archivo", system-ui, sans-serif; --texto: "Literata", Georgia, serif; --dado: "JetBrains Mono", Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root { --papel: #111416; --folha: #1a1e21; --tinta: #eef1ef; --tinta-2: #b4bbb8; --tinta-3: #7f8784; --fio: #2b3033;
          --fake: #f07a45; --fake-f: rgba(240, 122, 69, .2); --true: #5b9cf0; --true-f: rgba(91, 156, 240, .2);
          --destaque: #a197f0; --sombra: none; color-scheme: dark; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--papel); color: var(--tinta); font-family: var(--texto); font-size: 16px; line-height: 1.55; }
main { max-width: 820px; margin: 0 auto; padding: 40px 16px 80px; display: grid; gap: 22px; }
header { display: grid; gap: 10px; }
.carimbos { display: flex; gap: 8px; }
.carimbo { font-family: var(--display); font-weight: 800; font-stretch: 125%; font-size: .72rem; letter-spacing: .14em;
           padding: 3px 8px; border: 2px solid currentColor; border-radius: 3px; transform: rotate(-2deg); }
.carimbo.f { color: var(--fake); } .carimbo.t { color: var(--true); transform: rotate(1.5deg); }
h1 { font-family: var(--display); font-size: clamp(2.1rem, 6vw, 3.2rem); font-weight: 800; font-stretch: 75%;
     line-height: 1; margin: 0; letter-spacing: -.01em; }
.sub { color: var(--tinta-2); margin: 0; max-width: 60ch; }
.cartao { background: var(--folha); border: 1px solid var(--fio); border-radius: 10px; padding: 20px; box-shadow: var(--sombra); }
textarea { width: 100%; min-height: 150px; resize: vertical; font: inherit; font-size: 1rem; color: var(--tinta);
           background: transparent; border: 1px solid var(--fio); border-radius: 8px; padding: 12px 14px; outline: none; }
textarea:focus { border-color: var(--destaque); box-shadow: 0 0 0 3px color-mix(in srgb, var(--destaque) 18%, transparent); }
.linha { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 12px; flex-wrap: wrap; }
.contador { font-family: var(--dado); font-size: .8rem; color: var(--tinta-3); }
.contador.ok { color: var(--destaque); }
button { font-family: var(--display); font-weight: 700; font-stretch: 95%; font-size: 1rem; cursor: pointer; border-radius: 8px; }
#btn { background: var(--tinta); color: var(--papel); border: 0; padding: 11px 26px; }
#btn:hover { opacity: .88; } #btn:disabled { opacity: .5; cursor: wait; }
.exemplos { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.exemplos span { font-family: var(--display); font-size: .78rem; font-weight: 650; letter-spacing: .08em; text-transform: uppercase;
                 color: var(--tinta-3); margin-right: 4px; }
.exemplos button { background: transparent; color: var(--tinta-2); border: 1px solid var(--fio); padding: 5px 11px;
                   font-size: .82rem; font-weight: 600; }
.exemplos button:hover { border-color: var(--tinta-3); color: var(--tinta); }

#saida { display: grid; gap: 22px; }
.veredito { display: grid; gap: 16px; }
.topo { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.rotulo { font-family: var(--display); font-size: .75rem; font-weight: 650; letter-spacing: .1em; text-transform: uppercase; color: var(--tinta-2); }
.classe { font-family: var(--display); font-weight: 800; font-stretch: 120%; font-size: clamp(2.6rem, 9vw, 4rem); line-height: 1;
          letter-spacing: .06em; }
.classe.FAKE { color: var(--fake); } .classe.TRUE { color: var(--true); }
.certeza { font-family: var(--display); font-size: 1.05rem; color: var(--tinta-2); }
.medidor { position: relative; height: 12px; border-radius: 6px;
           background: linear-gradient(90deg, var(--fake) 0%, var(--fake-f) 46%, var(--fio) 50%, var(--true-f) 54%, var(--true) 100%); }
.medidor .meio { position: absolute; left: 50%; top: -5px; bottom: -5px; width: 2px; background: var(--tinta-3); }
.medidor .ponteiro { position: absolute; top: 50%; width: 22px; height: 22px; border-radius: 50%; background: var(--folha);
                     border: 3px solid var(--tinta); transform: translate(-50%, -50%); transition: left .5s cubic-bezier(.2,.8,.2,1); }
.escala { display: flex; justify-content: space-between; font-family: var(--dado); font-size: .72rem; color: var(--tinta-3); }
.aviso { font-family: var(--display); font-size: .9rem; color: var(--fake); background: var(--fake-f); padding: 8px 12px; border-radius: 6px; }

.texto-cor { font-size: 1.04rem; line-height: 2; }
.texto-cor span { padding: 2px 1px; border-radius: 3px; }
.legenda { display: flex; gap: 16px; flex-wrap: wrap; font-family: var(--display); font-size: .82rem; color: var(--tinta-2); }
.legenda i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.chips { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.chips h3 { font-family: var(--display); font-size: .8rem; font-weight: 650; margin: 0 0 6px; letter-spacing: .06em; text-transform: uppercase; }
.chips .f h3 { color: var(--fake); } .chips .t h3 { color: var(--true); }
.chip { display: inline-block; font-family: var(--dado); font-size: .8rem; padding: 2px 8px; border-radius: 4px; margin: 0 4px 6px 0; }
.f .chip { background: var(--fake-f); } .t .chip { background: var(--true-f); }

details summary { cursor: pointer; font-family: var(--display); font-weight: 650; list-style: none; display: flex; justify-content: space-between; }
details summary::after { content: "+"; color: var(--tinta-3); font-size: 1.2rem; line-height: 1; }
details[open] summary::after { content: "−"; }
table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: .9rem; }
td { padding: 9px 6px; border-bottom: 1px solid var(--fio); }
td:first-child { font-family: var(--display); }
td.num { font-family: var(--dado); font-size: .82rem; text-align: right; white-space: nowrap; }
.pill { font-family: var(--display); font-weight: 800; font-size: .72rem; letter-spacing: .08em; padding: 2px 7px; border-radius: 3px; border: 1.5px solid currentColor; }
.pill.FAKE { color: var(--fake); } .pill.TRUE { color: var(--true); }
tr.esc td:first-child { color: var(--destaque); font-weight: 700; }
.nota { font-size: .84rem; color: var(--tinta-2); }
footer { font-size: .82rem; color: var(--tinta-3); border-top: 1px solid var(--fio); padding-top: 14px; }
@media (max-width: 520px) { main { padding-top: 24px; } .cartao { padding: 16px; } }
</style></head><body><main>
<header>
  <div class="carimbos"><span class="carimbo f">FAKE</span><span class="carimbo t">TRUE</span></div>
  <h1>Vinagre · detector de boatos</h1>
  <p class="sub">Cole o trecho de uma notícia (30 a 50 palavras funciona melhor). O modelo diz se o texto tem
    <strong>cara de boato</strong> ou de <strong>notícia de veículo</strong>, e mostra quais palavras pesaram.</p>
</header>

<section class="cartao">
  <textarea id="texto" placeholder="Cole a notícia aqui... (Ctrl+Enter analisa)"></textarea>
  <div class="linha">
    <span class="contador" id="contador">0 palavras</span>
    <button id="btn">Analisar</button>
  </div>
</section>
<div class="exemplos" id="exemplos"><span>Exemplos</span></div>

<div id="saida"></div>

<footer>Modelo: TF-IDF de n-gramas de caracteres (2–5) + LinearSVC · treinado com FakeRecogna 2020–21 e corpus
  externo 2022–2026 · F1 94,2 no teste final. O modelo reconhece <strong>estilo de escrita, não fatos</strong>:
  use para estudo, não como checagem. Roda só no seu computador.</footer>
</main>
<script>
const EXEMPLOS = __EXEMPLOS__;
const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// Contador de palavras (o modelo usa só as 50 primeiras).
function contar() {
  const n = $("texto").value.trim().split(/\s+/).filter(Boolean).length;
  const c = $("contador");
  c.textContent = n === 0 ? "0 palavras" : n > 50 ? `${n} palavras · só as 50 primeiras serão usadas` : `${n} palavras`;
  c.className = "contador" + (n >= 30 && n <= 50 ? " ok" : "");
}
$("texto").addEventListener("input", contar);
$("texto").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analisar(); });

for (const [nome, texto] of EXEMPLOS) {
  const b = document.createElement("button");
  b.textContent = nome;
  b.onclick = () => { $("texto").value = texto; contar(); analisar(); };
  $("exemplos").appendChild(b);
}

function certeza(s) {
  const a = Math.abs(s);
  return a < 0.2 ? "dúvida: perto da fronteira" : a < 0.6 ? "certeza moderada" : "certeza alta";
}

async function analisar() {
  const texto = $("texto").value.trim();
  if (!texto) { $("saida").innerHTML = "<p class='aviso'>Cole uma notícia primeiro.</p>"; return; }
  $("btn").disabled = true; $("btn").textContent = "Analisando...";
  try {
    const r = await (await fetch("/prever", { method: "POST", body: JSON.stringify({ texto }) })).json();
    const m = r.modelos[0];
    // Ponteiro: score limitado a [-2, 2] mapeado em 0–100%.
    const pos = 50 + Math.max(-2, Math.min(2, m.score)) * 25;
    // Palavras coloridas pela contribuição (escala relativa à maior contribuição do texto).
    const maxc = Math.max(0.05, ...r.palavras.map((p) => Math.abs(p.c)));
    const palavras = r.palavras.map((p) => {
      const a = Math.min(1, Math.abs(p.c) / maxc);
      const cor = p.c < 0 ? `color-mix(in srgb, var(--fake) ${Math.round(a * 55)}%, transparent)`
                          : `color-mix(in srgb, var(--true) ${Math.round(a * 50)}%, transparent)`;
      return `<span style="background:${cor}" title="${p.c >= 0 ? "+" : ""}${p.c.toFixed(3)}">${esc(p.w)}</span>`;
    }).join(" ");
    const chips = (lista) => lista.length ? lista.map((p) => `<span class="chip">${esc(p.w)}</span>`).join("") : "<span class='nota'>nenhuma</span>";
    const linhas = r.modelos.map((x, i) => `<tr class="${i === 0 ? "esc" : ""}"><td>${esc(x.nome)}</td>
      <td><span class="pill ${x.classe}">${x.classe}</span></td><td class="num">${x.score >= 0 ? "+" : ""}${x.score.toFixed(2)}</td></tr>`).join("");
    $("saida").innerHTML = `
      <section class="cartao veredito">
        <div class="topo"><span class="rotulo">Modelo escolhido diz</span><span class="certeza">${certeza(m.score)} · score ${m.score >= 0 ? "+" : ""}${m.score.toFixed(2)}</span></div>
        <div class="classe ${m.classe}">${m.classe}</div>
        <div><div class="medidor"><div class="meio"></div><div class="ponteiro" id="ponteiro" style="left:50%"></div></div>
          <div class="escala" style="margin-top:8px"><span>← boato</span><span>fronteira (0)</span><span>notícia →</span></div></div>
        ${r.n_palavras < 20 ? "<p class='aviso'>Trecho curto (menos de 20 palavras): o modelo foi treinado com 30 a 100 palavras, então a previsão é menos confiável.</p>" : ""}
      </section>
      <section class="cartao" style="display:grid;gap:14px">
        <span class="rotulo">Por que: o que cada palavra puxou (${r.n_palavras} palavras analisadas)</span>
        <div class="texto-cor">${palavras}</div>
        <div class="legenda"><span><i style="background:var(--fake)"></i>puxa para FAKE</span><span><i style="background:var(--true)"></i>puxa para TRUE</span><span>passe o mouse para ver o peso</span></div>
        <div class="chips"><div class="f"><h3>Mais puxaram para FAKE</h3>${chips(r.top_fake)}</div>
          <div class="t"><h3>Mais puxaram para TRUE</h3>${chips(r.top_true)}</div></div>
        <p class="nota">O modelo olha pedaços de 2 a 5 letras, incluindo pontuação. Cada palavra recebe a soma dos pedaços dela.
          Números no trecho: ${r.n_numeros} (${r.pct_numeros}% das palavras; notícia true tem ~5%, boato ~2%).</p>
      </section>
      <details class="cartao"><summary>Comparar com os outros modelos (${r.votos_true} TRUE × ${r.votos_fake} FAKE)</summary>
        <table>${linhas}</table>
        <p class="nota">Score = distância da fronteira de decisão: positivo = TRUE, negativo = FAKE. Não é probabilidade, e a escala muda entre modelos.</p>
      </details>`;
    requestAnimationFrame(() => requestAnimationFrame(() => { $("ponteiro").style.left = pos + "%"; }));
  } catch (e) {
    $("saida").innerHTML = "<p class='aviso'>Erro ao falar com o servidor. Ele ainda está rodando no terminal?</p>";
  }
  $("btn").disabled = false; $("btn").textContent = "Analisar";
}
$("btn").onclick = analisar;
</script></body></html>"""


def contribuicao_palavras(modelo, trecho):
    # Divide o score do modelo escolhido entre as palavras: cada n-grama de caracteres contribui com
    # valor_tfidf × peso_svm, repartido igualmente entre as ocorrências dele; cada palavra soma os n-gramas que gera.
    tfidf, svm = modelo.named_steps["tfidf"], modelo.named_steps["svm"]
    analisar = tfidf.build_analyzer()
    x = tfidf.transform([trecho]).tocsr()
    contrib_ng = {}
    vocab_inv = {i: g for g, i in tfidf.vocabulary_.items()}
    for j, v in zip(x.indices, x.data):
        contrib_ng[vocab_inv[j]] = float(v * svm.coef_[0, j])
    palavras = trecho.split()
    ngs_por_palavra = [analisar(p) for p in palavras]
    ocorrencias = Counter(g for ngs in ngs_por_palavra for g in ngs)
    return [{"w": p, "c": sum(contrib_ng.get(g, 0.0) / ocorrencias[g] for g in ngs)}
            for p, ngs in zip(palavras, ngs_por_palavra)]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        pagina = PAGINA.replace("__EXEMPLOS__", json.dumps(EXEMPLOS, ensure_ascii=False))
        self._responder(200, "text/html; charset=utf-8", pagina.encode("utf-8"))

    def do_POST(self):
        texto = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["texto"]
        trecho = truncar(pd.Series([" ".join(texto.split())]), N_PRODUTO).iloc[0]
        n = len(trecho.split())
        res = []
        for nome, m in MODELOS.items():
            classe, score = prever(m, trecho)
            res.append({"nome": nome, "classe": classe.upper(), "score": score})
        palavras = contribuicao_palavras(next(iter(MODELOS.values())), trecho)
        ordenadas = sorted(palavras, key=lambda p: p["c"])
        c = contar_numeros(trecho)
        corpo = {"trecho": trecho, "n_palavras": n, "modelos": res, "palavras": palavras,
                 "top_fake": [p for p in ordenadas[:5] if p["c"] < 0],
                 "top_true": [p for p in ordenadas[::-1][:5] if p["c"] > 0],
                 "votos_true": sum(r["classe"] == "TRUE" for r in res),
                 "votos_fake": sum(r["classe"] == "FAKE" for r in res),
                 "n_numeros": c["n_numeros"], "pct_numeros": round(100 * c["n_numeros"] / max(n, 1))}
        self._responder(200, "application/json", json.dumps(corpo).encode("utf-8"))

    def _responder(self, status, tipo, corpo):
        self.send_response(status)
        self.send_header("Content-Type", tipo)
        self.end_headers()
        self.wfile.write(corpo)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("Carregando modelos...")
    MODELOS = carregar("--emb" in sys.argv)
    if "--porta" in sys.argv:
        PORTA = int(sys.argv[sys.argv.index("--porta") + 1])
    print(f"Pronto: abra http://localhost:{PORTA} no navegador (Ctrl+C encerra).")
    HTTPServer(("127.0.0.1", PORTA), Handler).serve_forever()
