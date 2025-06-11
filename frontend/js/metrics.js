// metrics.js
// ------------------------------
// Solicita métricas (_metrics_xgb_) al backend
// y actualiza los KPI cards en pantalla.
// Debe importarse después de upload.js y app.js.
// ------------------------------

/**
 * Hace fetch a /metrics_xgb y pinta los KPIs:
 *  - total_sales
 *  - avg_profit_pct
 *  - sale_count
 *  - avg_sales
 */
async function fetchAndRenderMetrics() {
  try {
    const res = await fetch('/metrics_xgb');
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || res.statusText);
    }
    const { metrics } = await res.json();

    // Actualizo los KPI cards
    document.getElementById('kpi-total-sales').innerText = metrics.total_sales.toFixed(2);
    document.getElementById('kpi-avg-profit').innerText   = (metrics.avg_profit_pct * 100).toFixed(2) + '%';
    document.getElementById('kpi-sale-count').innerText   = metrics.sale_count;
    document.getElementById('kpi-avg-sales').innerText    = metrics.avg_sales.toFixed(2);
  } catch (err) {
    // Si hay un error (p.ej. no subió CSV) lo muestro en el status
    const statusEl = document.getElementById('uploadStatus');
    statusEl.innerText = 'Error cargando métricas: ' + err.message;
    console.error(err);
  }
}

// Al hacer click en el botón, lanzamos la petición
document.getElementById('metricsBtn').addEventListener('click', fetchAndRenderMetrics);
