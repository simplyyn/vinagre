"""Testador no navegador: cole uma notícia e veja a previsão dos modelos (versão web do testar_noticia.py).

Uso: .venv\\Scripts\\python.exe testador_web.py [--emb]
Depois abra http://localhost:8000 no navegador. Ctrl+C no terminal encerra.
Só roda na sua máquina (localhost); nada é enviado para fora.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd

from anls import truncar
from exp_numeros import contar_numeros
from testar_noticia import N_PRODUTO, carregar, prever

PORTA = 8000

PAGINA = """<!doctype html>
<html lang="pt-br"><head><meta charset="utf-8"><title>Testador de notícias</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; color: #1b1f23; }
  h1 { font-size: 1.5rem; margin-bottom: 4px; }
  p.sub { color: #555; margin-top: 0; }
  textarea { width: 100%; height: 160px; font: inherit; padding: 10px; box-sizing: border-box; }
  button { margin-top: 10px; padding: 10px 22px; font: inherit; font-weight: 600; cursor: pointer; }
  .trecho { background: #f3f4f6; padding: 10px; font-size: 0.9rem; margin: 16px 0; }
  .aviso { color: #9a5b00; }
  table { width: 100%; border-collapse: collapse; }
  td { padding: 8px 6px; border-bottom: 1px solid #e5e7eb; }
  tr.esc td { font-weight: 700; background: #eef2ff; }
  .FAKE { color: #c2410c; font-weight: 700; } .TRUE { color: #1d4ed8; font-weight: 700; }
  .barra { display: inline-block; height: 10px; background: #9ca3af; vertical-align: middle; }
  .nota { color: #666; font-size: 0.85rem; }
</style></head><body>
<h1>Testador de notícias</h1>
<p class="sub">Cole o trecho de uma notícia (de preferência 30 a 50 palavras) e clique em Analisar.</p>
<textarea id="texto" placeholder="Cole a notícia aqui..."></textarea>
<button id="btn">Analisar</button>
<div id="saida"></div>
<script>
const btn = document.getElementById("btn"), saida = document.getElementById("saida");
btn.onclick = async () => {
  const texto = document.getElementById("texto").value.trim();
  if (!texto) { saida.innerHTML = "<p class='aviso'>Cole uma notícia primeiro.</p>"; return; }
  btn.disabled = true; btn.textContent = "Analisando...";
  try {
    const r = await (await fetch("/prever", { method: "POST", body: JSON.stringify({ texto }) })).json();
    const linhas = r.modelos.map((m, i) => `<tr class="${i === 0 ? "esc" : ""}"><td>${m.nome}</td>
      <td class="${m.classe}">${m.classe}</td><td>${m.confianca.toFixed(2)}</td>
      <td><span class="barra" style="width:${Math.min(120, m.confianca * 60)}px"></span></td></tr>`).join("");
    saida.innerHTML = `<div class="trecho"><b>Trecho analisado (${r.n_palavras} palavras):</b> ${r.trecho.replace(/</g, "&lt;")}</div>
      ${r.n_palavras < 20 ? "<p class='aviso'>Trecho curto (&lt; 20 palavras): a previsão é menos confiável.</p>" : ""}
      <table><tr><td><b>Modelo</b></td><td><b>Previsão</b></td><td><b>Confiança</b></td><td></td></tr>${linhas}</table>
      <p><b>Votos:</b> ${r.votos_true} TRUE × ${r.votos_fake} FAKE &nbsp;·&nbsp; <b>Números no trecho:</b> ${r.n_numeros}
        (${r.pct_numeros}% das palavras; média em notícia true ~5%, em boato ~2%)</p>
      <p class="nota">A primeira linha é o modelo escolhido. Confiança = distância da fronteira de decisão (não é
        probabilidade; a escala muda entre modelos).</p>`;
  } catch (e) { saida.innerHTML = "<p class='aviso'>Erro ao falar com o servidor. Ele ainda está rodando no terminal?</p>"; }
  btn.disabled = false; btn.textContent = "Analisar";
};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._responder(200, "text/html; charset=utf-8", PAGINA.encode("utf-8"))

    def do_POST(self):
        texto = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["texto"]
        trecho = truncar(pd.Series([" ".join(texto.split())]), N_PRODUTO).iloc[0]
        n = len(trecho.split())
        res = []
        for nome, m in MODELOS.items():
            classe, score = prever(m, trecho)
            res.append({"nome": nome, "classe": classe.upper(), "confianca": abs(score)})
        c = contar_numeros(trecho)
        corpo = {"trecho": trecho, "n_palavras": n, "modelos": res,
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
    print(f"Pronto: abra http://localhost:{PORTA} no navegador (Ctrl+C encerra).")
    HTTPServer(("127.0.0.1", PORTA), Handler).serve_forever()
