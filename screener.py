"""
================================================================================
EQUITY RESEARCH SCORING DASHBOARD — Versión Interactiva (filtros + botones)
================================================================================
Genera un dashboard_clientes.html con:
  - Filtro por sector (dropdown)
  - Filtro por score mínimo (slider)
  - Botones para cambiar entre 3 gráficas (Score / Desglose por pilar / Mapa)
  - Tabla que se actualiza en vivo según los filtros
  - Todo funciona SIN internet una vez generado el archivo (solo carga
    Plotly.js desde CDN la primera vez que se abre, con internet).

--------------------------------------------------------------------------------
INSTALACIÓN (en Google Colab, primera celda)
--------------------------------------------------------------------------------
    !pip install yfinance pandas numpy -q

--------------------------------------------------------------------------------
CÓMO USARLO
--------------------------------------------------------------------------------
1. Pega este archivo completo en una celda de Colab y corre con Shift+Enter.
2. Se genera "dashboard_clientes.html" en el panel de archivos (📁 a la izq).
3. Descárgalo con clic derecho > Descargar, y ábrelo con doble clic en tu
   computadora — se abre en cualquier navegador, no necesita Colab.
4. Dentro del HTML: usa el dropdown de sector y el slider de score para
   filtrar, y los botones de arriba para cambiar entre las 3 vistas.

--------------------------------------------------------------------------------
NOTA IMPORTANTE PARA USO CON CLIENTES
--------------------------------------------------------------------------------
Este dashboard es un filtro cuantitativo de primera pasada, no una
recomendación de inversión personalizada. Antes de proponer una acción a
un cliente específico, evalúa su perfil de riesgo, horizonte, objetivos,
concentración de cartera y la normativa aplicable.
================================================================================
"""

import json
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ================================================================================
# 1. CONFIGURACIÓN
# ================================================================================

SP500_SAMPLE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "MA",
    "UNH", "HD", "PG", "XOM", "CVX", "PFE", "KO", "PEP", "MRK", "ABBV",
    "BAC", "WMT", "DIS", "CSCO", "ADBE", "CRM", "NFLX", "INTC", "T", "VZ",
    "NKE", "MCD", "COST", "TMO", "ABT", "LIN", "ORCL", "TXN", "QCOM", "HON",
]
MEXICAN_ADRS = ["AMX", "CX", "KOF", "FMX", "VIST"]
BMV_TICKERS_MX = ["GFNORTEO.MX", "WALMEX.MX", "GMEXICOB.MX", "AC.MX", "CEMEXCPO.MX"]
TICKERS_UNIVERSE = SP500_SAMPLE + MEXICAN_ADRS  # + BMV_TICKERS_MX

THRESHOLDS = {
    "peg_max": 1.5, "roe_min": 0.12, "roe_preferred": 0.15,
    "debt_ebitda_max": 3.0, "drawdown_min": 0.15, "drawdown_max": 0.45,
}
WEIGHTS = {"value": 25, "growth": 25, "quality": 30, "momentum": 20}
DRAWDOWN_BONUS_POINTS = 10
MAX_SCORE = 100
TOP_N = 30  # más alto que antes, para que el filtro por sector tenga con qué trabajar
HTML_OUTPUT_FILE = "docs/index.html"


# ================================================================================
# 2. ESTRUCTURA DE DATOS
# ================================================================================

@dataclass
class StockMetrics:
    ticker: str
    name: str = "N/A"
    sector: str = "N/A"
    price: Optional[float] = None
    week52_high: Optional[float] = None
    pct_from_high: Optional[float] = None
    pe_forward: Optional[float] = None
    peg: Optional[float] = None
    ev_ebitda: Optional[float] = None
    fcf_yield: Optional[float] = None
    rev_growth_yoy: Optional[float] = None
    eps_cagr_5y: Optional[float] = None
    roe: Optional[float] = None
    debt_ebitda: Optional[float] = None
    fcf_positive: Optional[bool] = None
    value_score: float = 0.0
    growth_score: float = 0.0
    quality_score: float = 0.0
    momentum_score: float = 0.0
    composite_score: float = 0.0
    reasons: list = field(default_factory=list)


# ================================================================================
# 3. OBTENCIÓN Y CÁLCULO DE MÉTRICAS
# ================================================================================

def safe_get(d, key, default=None):
    val = d.get(key, default)
    return default if val is None else val


def fetch_raw_data(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        return info if info else None
    except Exception as e:
        print(f"  [WARN] {ticker}: {e}")
        return None


def fetch_eps_history(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        earnings = tk.earnings
        if earnings is None or earnings.empty:
            fin = tk.financials
            if fin is not None and not fin.empty and "Net Income" in fin.index:
                return fin.loc["Net Income"].sort_index()
        return earnings
    except Exception:
        return None


def compute_eps_cagr(eps_series) -> Optional[float]:
    try:
        if eps_series is None:
            return None
        vals = eps_series.dropna()
        vals = vals[vals != 0]
        if len(vals) < 2:
            return None
        first, last = vals.iloc[0], vals.iloc[-1]
        n = len(vals) - 1
        if first <= 0 or last <= 0 or n <= 0:
            return None
        return (last / first) ** (1 / n) - 1
    except Exception:
        return None


def compute_metrics(ticker: str) -> Optional[StockMetrics]:
    info = fetch_raw_data(ticker)
    if info is None:
        return None

    m = StockMetrics(ticker=ticker)
    m.name = safe_get(info, "shortName", ticker)
    m.sector = safe_get(info, "sector", "N/A")
    m.price = safe_get(info, "currentPrice") or safe_get(info, "regularMarketPrice")
    m.week52_high = safe_get(info, "fiftyTwoWeekHigh")
    if m.price and m.week52_high and m.week52_high > 0:
        m.pct_from_high = (m.week52_high - m.price) / m.week52_high

    m.pe_forward = safe_get(info, "forwardPE")
    m.peg = safe_get(info, "pegRatio")
    m.ev_ebitda = safe_get(info, "enterpriseToEbitda")

    fcf = safe_get(info, "freeCashflow")
    mkt_cap = safe_get(info, "marketCap")
    if fcf is not None and mkt_cap:
        m.fcf_yield = fcf / mkt_cap
        m.fcf_positive = fcf > 0

    m.rev_growth_yoy = safe_get(info, "revenueGrowth")
    m.eps_cagr_5y = compute_eps_cagr(fetch_eps_history(ticker))

    m.roe = safe_get(info, "returnOnEquity")
    total_debt = safe_get(info, "totalDebt")
    ebitda = safe_get(info, "ebitda")
    if total_debt is not None and ebitda and ebitda > 0:
        m.debt_ebitda = total_debt / ebitda

    return m


def passes_first_pass_filters(m: StockMetrics) -> bool:
    if m.price is None or m.pct_from_high is None:
        return False
    return m.pct_from_high > 0


# ================================================================================
# 4. SCORING POR PILAR
# ================================================================================

def score_value(m):
    points, max_points = 0.0, 0.0
    if m.peg is not None:
        max_points += 40
        if m.peg <= THRESHOLDS["peg_max"]:
            points += min(40, 40 * max(0, (THRESHOLDS["peg_max"] - m.peg) / THRESHOLDS["peg_max"] + 0.5))
    if m.ev_ebitda is not None and m.ev_ebitda > 0:
        max_points += 30
        points += np.clip((20 - m.ev_ebitda) / (20 - 8), 0, 1) * 30
    if m.fcf_yield is not None:
        max_points += 30
        points += np.clip(m.fcf_yield / 0.08, 0, 1) * 30
    return 0.0 if max_points == 0 else min(100.0, (points / max_points) * 100)


def score_growth(m):
    points, max_points = 0.0, 0.0
    if m.rev_growth_yoy is not None:
        max_points += 50
        points += np.clip(m.rev_growth_yoy / 0.15, 0, 1) * 50
    if m.eps_cagr_5y is not None:
        max_points += 50
        points += np.clip(m.eps_cagr_5y / 0.15, 0, 1) * 50
    return 0.0 if max_points == 0 else min(100.0, (points / max_points) * 100)


def score_quality(m):
    points, max_points = 0.0, 0.0
    if m.roe is not None:
        max_points += 40
        if m.roe >= THRESHOLDS["roe_preferred"]:
            points += 40
        elif m.roe >= THRESHOLDS["roe_min"]:
            points += 40 * ((m.roe - THRESHOLDS["roe_min"]) / (THRESHOLDS["roe_preferred"] - THRESHOLDS["roe_min"]))
    if m.debt_ebitda is not None:
        max_points += 35
        points += np.clip((THRESHOLDS["debt_ebitda_max"] - m.debt_ebitda) / THRESHOLDS["debt_ebitda_max"], 0, 1) * 35
    if m.fcf_positive is not None:
        max_points += 25
        if m.fcf_positive:
            points += 25
    return 0.0 if max_points == 0 else min(100.0, (points / max_points) * 100)


def score_momentum(m):
    if m.pct_from_high is None:
        return 0.0
    dd = m.pct_from_high
    lo, hi = THRESHOLDS["drawdown_min"], THRESHOLDS["drawdown_max"]
    if lo <= dd <= hi:
        mid = (lo + hi) / 2
        dist = abs(dd - mid) / ((hi - lo) / 2)
        return max(60.0, 100.0 - dist * 40)
    elif dd < lo:
        return max(0.0, (dd / lo) * 50)
    else:
        return max(0.0, 50 - (dd - hi) * 100)


def compute_composite_score(m: StockMetrics) -> StockMetrics:
    m.value_score = score_value(m)
    m.growth_score = score_growth(m)
    m.quality_score = score_quality(m)
    m.momentum_score = score_momentum(m)

    composite = (
        m.value_score * WEIGHTS["value"] / 100
        + m.growth_score * WEIGHTS["growth"] / 100
        + m.quality_score * WEIGHTS["quality"] / 100
        + m.momentum_score * WEIGHTS["momentum"] / 100
    )

    in_sweet_spot = (
        m.pct_from_high is not None
        and THRESHOLDS["drawdown_min"] <= m.pct_from_high <= THRESHOLDS["drawdown_max"]
    )
    if in_sweet_spot and m.quality_score >= 60 and m.value_score >= 50:
        composite = min(MAX_SCORE, composite + DRAWDOWN_BONUS_POINTS)
        m.reasons.append("Caída de precio con fundamentos sólidos (bono aplicado)")

    if m.peg is not None and m.peg <= THRESHOLDS["peg_max"]:
        m.reasons.append(f"PEG atractivo ({m.peg:.2f})")
    if m.roe is not None and m.roe >= THRESHOLDS["roe_preferred"]:
        m.reasons.append(f"ROE alto ({m.roe:.1%})")
    if m.rev_growth_yoy is not None and m.rev_growth_yoy > 0.10:
        m.reasons.append(f"Crecimiento sólido ({m.rev_growth_yoy:.1%} YoY)")
    if m.debt_ebitda is not None and m.debt_ebitda < THRESHOLDS["debt_ebitda_max"]:
        m.reasons.append(f"Deuda controlada ({m.debt_ebitda:.1f}x EBITDA)")

    m.composite_score = round(composite, 1)
    return m


# ================================================================================
# 5. PIPELINE
# ================================================================================

def run_screener(tickers: list) -> pd.DataFrame:
    results = []
    print(f"Analizando {len(tickers)} tickers...\n")
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {ticker}...")
        m = compute_metrics(ticker)
        if m is None or not passes_first_pass_filters(m):
            continue
        m = compute_composite_score(m)
        results.append(m)

    if not results:
        print("\nSin resultados. Revisa el universo de tickers o los umbrales.")
        return pd.DataFrame()

    rows = [{
        "Ticker": m.ticker, "Nombre": m.name, "Sector": m.sector, "Precio": m.price,
        "% desde Máx 52s": m.pct_from_high, "P/E Fwd": m.pe_forward, "PEG": m.peg,
        "EV/EBITDA": m.ev_ebitda, "ROE": m.roe, "Rev Growth YoY": m.rev_growth_yoy,
        "EPS CAGR 5Y": m.eps_cagr_5y, "Debt/EBITDA": m.debt_ebitda, "FCF Yield": m.fcf_yield,
        "Value Score": m.value_score, "Growth Score": m.growth_score,
        "Quality Score": m.quality_score, "Momentum Score": m.momentum_score,
        "Composite Score": m.composite_score,
        "Por qué es oportunidad": "; ".join(m.reasons[:2]) if m.reasons else "Ver métricas",
    } for m in results]

    df = pd.DataFrame(rows).sort_values("Composite Score", ascending=False).reset_index(drop=True)
    return df


# ================================================================================
# 6. DASHBOARD HTML INTERACTIVO (filtros + botones, todo en JavaScript/Plotly.js)
# ================================================================================

def _safe(v, default=None):
    """Convierte NaN/None a null para que sea JSON válido."""
    if v is None:
        return default
    try:
        if isinstance(v, float) and np.isnan(v):
            return default
    except Exception:
        pass
    return v


def export_dashboard_html(df: pd.DataFrame, filename: str = HTML_OUTPUT_FILE, top_n: int = TOP_N):
    import os
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    data = df.head(top_n).copy()

    records = []
    for _, r in data.iterrows():
        records.append({
            "ticker": r["Ticker"], "name": r["Nombre"], "sector": r["Sector"],
            "price": _safe(r["Precio"]), "drawdown": _safe(r["% desde Máx 52s"]),
            "peg": _safe(r["PEG"]), "roe": _safe(r["ROE"]),
            "revGrowth": _safe(r["Rev Growth YoY"]), "debtEbitda": _safe(r["Debt/EBITDA"]),
            "valueScore": _safe(r["Value Score"], 0), "growthScore": _safe(r["Growth Score"], 0),
            "qualityScore": _safe(r["Quality Score"], 0), "momentumScore": _safe(r["Momentum Score"], 0),
            "compositeScore": _safe(r["Composite Score"], 0),
            "reasons": r["Por qué es oportunidad"],
        })

    sectors = sorted(set(r["sector"] for r in records if r["sector"] and r["sector"] != "N/A"))
    data_json = json.dumps(records, ensure_ascii=False)
    sectors_json = json.dumps(sectors, ensure_ascii=False)
    weights_json = json.dumps(WEIGHTS)
    drawdown_min = THRESHOLDS["drawdown_min"] * 100
    drawdown_max = THRESHOLDS["drawdown_max"] * 100

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Dashboard de Oportunidades de Inversión</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root {{
    --primary: #1a3c5e; --accent: #06A77D; --bg: #fafafa; --border: #ddd;
  }}
  body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; padding: 24px 32px;
         background: var(--bg); color: #222; }}
  h1 {{ color: var(--primary); margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-top: 0; margin-bottom: 20px; }}
  .controls {{ display: flex; flex-wrap: wrap; gap: 24px; align-items: center;
               background: white; padding: 16px 20px; border-radius: 8px;
               border: 1px solid var(--border); margin-bottom: 20px; }}
  .control-group {{ display: flex; flex-direction: column; gap: 4px; }}
  .control-group label {{ font-size: 12px; font-weight: bold; color: #555; }}
  select, input[type=range] {{ font-size: 14px; padding: 4px; }}
  #scoreValue {{ font-weight: bold; color: var(--accent); }}
  .tabs {{ display: flex; gap: 8px; margin-bottom: 12px; }}
  .tab-btn {{ padding: 8px 16px; border: 1px solid var(--primary); background: white;
              color: var(--primary); border-radius: 6px; cursor: pointer; font-size: 13px; }}
  .tab-btn.active {{ background: var(--primary); color: white; }}
  .chart-container {{ background: white; border: 1px solid var(--border); border-radius: 8px;
                       padding: 12px; margin-bottom: 24px; }}
  .chart-panel {{ display: none; }}
  .chart-panel.active {{ display: block; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: white; }}
  th {{ background: var(--primary); color: white; padding: 8px; text-align: left;
        position: sticky; top: 0; }}
  td {{ padding: 8px; border-bottom: 1px solid var(--border); }}
  tr:hover {{ background: #f0f7f4; }}
  .count-badge {{ font-size: 13px; color: #666; margin-bottom: 8px; }}
  .disclaimer {{ margin-top: 30px; padding: 15px; background: #fff3cd;
                 border-left: 4px solid #ffc107; font-size: 13px; }}
</style>
</head>
<body>

<h1>Dashboard de Oportunidades de Inversión</h1>
<p class="subtitle">Screening cuantitativo multifactorial — Value · Growth · Quality · Momentum</p>

<div class="controls">
  <div class="control-group">
    <label for="sectorFilter">Sector</label>
    <select id="sectorFilter" onchange="applyFilters()">
      <option value="">Todos los sectores</option>
    </select>
  </div>
  <div class="control-group">
    <label for="scoreFilter">Score mínimo: <span id="scoreValue">0</span></label>
    <input type="range" id="scoreFilter" min="0" max="100" value="0" step="5" oninput="applyFilters()">
  </div>
  <div class="control-group">
    <label for="sortFilter">Ordenar por</label>
    <select id="sortFilter" onchange="applyFilters()">
      <option value="compositeScore">Composite Score</option>
      <option value="drawdown">% Caída desde máximo</option>
      <option value="roe">ROE</option>
      <option value="revGrowth">Crecimiento de ingresos</option>
    </select>
  </div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('score')" id="tabBtn-score">Composite Score</button>
  <button class="tab-btn" onclick="showTab('pillars')" id="tabBtn-pillars">Desglose por Pilar</button>
  <button class="tab-btn" onclick="showTab('scatter')" id="tabBtn-scatter">Mapa de Oportunidades</button>
</div>

<div class="chart-container">
  <div id="panel-score" class="chart-panel active"><div id="chart-score"></div></div>
  <div id="panel-pillars" class="chart-panel"><div id="chart-pillars"></div></div>
  <div id="panel-scatter" class="chart-panel"><div id="chart-scatter"></div></div>
</div>

<div class="count-badge" id="countBadge"></div>
<table id="dataTable">
  <thead>
    <tr>
      <th>Ticker</th><th>Nombre</th><th>Sector</th><th>Precio</th>
      <th>% desde Máx 52s</th><th>PEG</th><th>ROE</th><th>Rev Growth YoY</th>
      <th>Debt/EBITDA</th><th>Score</th><th>Por qué es oportunidad</th>
    </tr>
  </thead>
  <tbody id="tableBody"></tbody>
</table>

<div class="disclaimer">
  <strong>Nota:</strong> Este dashboard es un filtro cuantitativo de primera pasada,
  no una recomendación de inversión personalizada. Antes de proponer una acción a un
  cliente, evalúa su perfil de riesgo, horizonte, objetivos, concentración de cartera
  y la normativa aplicable.
</div>

<script>
const ALL_DATA = {data_json};
const SECTORS = {sectors_json};
const WEIGHTS = {weights_json};
const DRAWDOWN_MIN = {drawdown_min};
const DRAWDOWN_MAX = {drawdown_max};

// --- Poblar el dropdown de sectores ---
const sectorSelect = document.getElementById('sectorFilter');
SECTORS.forEach(s => {{
  const opt = document.createElement('option');
  opt.value = s; opt.textContent = s;
  sectorSelect.appendChild(opt);
}});

function fmtPct(v) {{ return (v === null || v === undefined) ? 'N/A' : (v * 100).toFixed(1) + '%'; }}
function fmtNum(v) {{ return (v === null || v === undefined) ? 'N/A' : v.toFixed(2); }}

function getFilteredData() {{
  const sector = document.getElementById('sectorFilter').value;
  const minScore = parseFloat(document.getElementById('scoreFilter').value);
  const sortKey = document.getElementById('sortFilter').value;

  let filtered = ALL_DATA.filter(d =>
    (sector === '' || d.sector === sector) && d.compositeScore >= minScore
  );
  filtered.sort((a, b) => (b[sortKey] ?? -Infinity) - (a[sortKey] ?? -Infinity));
  return filtered;
}}

function renderScoreChart(data) {{
  const rev = [...data].reverse();
  Plotly.newPlot('chart-score', [{{
    x: rev.map(d => d.compositeScore), y: rev.map(d => d.ticker),
    type: 'bar', orientation: 'h',
    marker: {{ color: rev.map(d => d.compositeScore), colorscale: 'Tealgrn' }},
    text: rev.map(d => d.compositeScore), textposition: 'outside',
  }}], {{
    title: 'Composite Score por Ticker', xaxis: {{ title: 'Composite Score (0-100)' }},
    height: Math.max(400, 30 * rev.length), template: 'plotly_white',
    margin: {{ l: 80 }},
  }}, {{ responsive: true }});
}}

function renderPillarsChart(data) {{
  const rev = [...data].reverse();
  const pillars = [
    {{ key: 'valueScore', name: 'Value', color: '#2E86AB', w: WEIGHTS.value }},
    {{ key: 'growthScore', name: 'Growth', color: '#06A77D', w: WEIGHTS.growth }},
    {{ key: 'qualityScore', name: 'Quality', color: '#F1A208', w: WEIGHTS.quality }},
    {{ key: 'momentumScore', name: 'Momentum', color: '#D62246', w: WEIGHTS.momentum }},
  ];
  const traces = pillars.map(p => ({{
    x: rev.map(d => (d[p.key] || 0) * p.w / 100), y: rev.map(d => d.ticker),
    name: p.name, type: 'bar', orientation: 'h', marker: {{ color: p.color }},
  }}));
  Plotly.newPlot('chart-pillars', traces, {{
    barmode: 'stack', title: 'Desglose del Score por Pilar',
    xaxis: {{ title: 'Contribución al Composite Score' }},
    height: Math.max(400, 30 * rev.length), template: 'plotly_white',
    legend: {{ orientation: 'h', y: -0.15 }}, margin: {{ l: 80 }},
  }}, {{ responsive: true }});
}}

function renderScatterChart(data) {{
  Plotly.newPlot('chart-scatter', [{{
    x: data.map(d => d.drawdown * 100), y: data.map(d => d.compositeScore),
    text: data.map(d => d.ticker), mode: 'markers+text', textposition: 'top center',
    marker: {{ size: 14, color: data.map(d => d.compositeScore), colorscale: 'Tealgrn', showscale: true }},
    type: 'scatter',
  }}], {{
    title: 'Mapa de Oportunidades: Caída desde Máximo vs. Score',
    xaxis: {{ title: '% de caída desde máximo de 52 semanas' }},
    yaxis: {{ title: 'Composite Score' }}, template: 'plotly_white', height: 500,
    shapes: [{{
      type: 'rect', x0: DRAWDOWN_MIN, x1: DRAWDOWN_MAX, y0: 0, y1: 100,
      fillcolor: 'green', opacity: 0.08, line: {{ width: 0 }},
    }}],
  }}, {{ responsive: true }});
}}

function renderTable(data) {{
  const tbody = document.getElementById('tableBody');
  tbody.innerHTML = data.map(d => `
    <tr>
      <td><strong>${{d.ticker}}</strong></td><td>${{d.name}}</td><td>${{d.sector}}</td>
      <td>${{fmtNum(d.price)}}</td><td>${{fmtPct(d.drawdown)}}</td>
      <td>${{fmtNum(d.peg)}}</td><td>${{fmtPct(d.roe)}}</td><td>${{fmtPct(d.revGrowth)}}</td>
      <td>${{fmtNum(d.debtEbitda)}}</td><td>${{d.compositeScore}}</td><td>${{d.reasons}}</td>
    </tr>`).join('');
  document.getElementById('countBadge').textContent =
    `Mostrando ${{data.length}} de ${{ALL_DATA.length}} oportunidades`;
}}

function applyFilters() {{
  document.getElementById('scoreValue').textContent = document.getElementById('scoreFilter').value;
  const filtered = getFilteredData();
  renderScoreChart(filtered);
  renderPillarsChart(filtered);
  renderScatterChart(filtered);
  renderTable(filtered);
}}

function showTab(name) {{
  ['score', 'pillars', 'scatter'].forEach(n => {{
    document.getElementById('panel-' + n).classList.toggle('active', n === name);
    document.getElementById('tabBtn-' + n).classList.toggle('active', n === name);
  }});
}}

applyFilters();
</script>
</body>
</html>
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✅ Dashboard interactivo guardado como '{filename}'.")
    print("   Descárgalo desde el panel de archivos de Colab (📁 a la izquierda) y ábrelo con doble clic.")


# ================================================================================
# 7. MAIN
# ================================================================================

if __name__ == "__main__":
    df_results = run_screener(TICKERS_UNIVERSE)

    if not df_results.empty:
        df_results.to_csv("equity_screener_results.csv", index=False)
        export_dashboard_html(df_results)
    else:
        print("Sin resultados. Ajusta el universo de tickers o los umbrales.")
