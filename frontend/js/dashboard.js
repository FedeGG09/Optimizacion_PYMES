// =================================================================
// dashboard.js (versión completa con sección de Predicción)
// CARGAR ESTE MÓDULO *DESPUÉS* DE upload.js, tras subir el CSV.
// Llama a initDashboard() desde upload.js cuando el upload sea exitoso.
// =================================================================

// Guardamos referencias globales a las instancias de los gráficos
let barChartInstance = null;
let lineChartInstance = null;
let scatterChartInstance = null;

/**
 * fetchKpis(filters): obtiene los KPI de negocio según filtros
 */
async function fetchKpis({ month, vendor, product }) {
  try {
    const query = new URLSearchParams();
    if (month)    query.append("month", month);
    if (vendor)   query.append("vendor", vendor);
    if (product)  query.append("product", product);

    const resp = await fetch(`/kpis?${query.toString()}`);
    if (!resp.ok) throw new Error("Error al obtener KPIs");
    return await resp.json();
  } catch (err) {
    console.error("fetchKpis():", err);
    return null;
  }
}

/**
 * updateKpisDisplay(data): actualiza el DOM con los KPIs
 */
function updateKpisDisplay(data) {
  if (!data) return;
  const fmtCurrency = num =>
    num.toLocaleString("es-AR", { style: "currency", currency: "USD", minimumFractionDigits: 0 });
  const fmtPercent = pct => `${(pct * 100).toFixed(0)}%`;

  document.getElementById("kpi-total-sales").innerText = fmtCurrency(data.total_sales);
  document.getElementById("kpi-avg-profit").innerText  = fmtPercent(data.avg_profit_pct);
  document.getElementById("kpi-sale-count").innerText  = data.sale_count.toLocaleString();
  document.getElementById("kpi-avg-sales").innerText   = fmtCurrency(data.avg_sales);
}

/**
 * fetchGrouped(field, filters): agrupa datos según campo + filtros
 */
async function fetchGrouped(field, { month, vendor, product }) {
  try {
    const query = new URLSearchParams({ field });
    if (month)    query.append("month", month);
    if (vendor)   query.append("vendor", vendor);
    if (product)  query.append("product", product);

    const resp = await fetch(`/grouped?${query.toString()}`);
    if (!resp.ok) throw new Error("Error al obtener datos agrupados");
    const json = await resp.json();
    return json.data;
  } catch (err) {
    console.error("fetchGrouped():", err);
    return [];
  }
}

/**
 * drawBarChart(groupedData): dibuja el gráfico de barras
 */
function drawBarChart(groupedData) {
  const ctx = document.getElementById("bar-chart").getContext("2d");
  if (barChartInstance) barChartInstance.destroy();

  const labels = groupedData.map(o => o.group);
  const values = groupedData.map(o => o.total_sales);

  barChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Ventas",
        data: values,
        borderRadius: 6,
        barThickness: 20
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `$${ctx.parsed.y.toLocaleString(undefined, { minimumFractionDigits: 2 })}`
          }
        }
      },
      scales: {
        x: { grid: { display: false } },
        y: {
          ticks: {
            callback: value => `$${(value / 1000).toFixed(0)}k`
          }
        }
      }
    }
  });
}

/**
 * fetchScatterData(): obtiene datos para el gráfico de dispersión
 */
async function fetchScatterData() {
  try {
    const resp = await fetch(`/grouped?field=${encodeURIComponent("Product Name")}`);
    if (!resp.ok) throw new Error("Error al obtener datos de productos");
    const json = await resp.json();
    return json.data;
  } catch (err) {
    console.error("fetchScatterData():", err);
    return [];
  }
}

/**
 * drawScatterChart(): dibuja el gráfico de dispersión
 */
async function drawScatterChart() {
  const productData = await fetchScatterData();
  const puntos = productData.map(p => {
    const ing = p.total_sales;
    const prof = p.total_profit;
    return { x: ing, y: ing ? prof / ing : 0, etiqueta: p.group };
  });

  const ctx = document.getElementById("scatter-chart").getContext("2d");
  if (scatterChartInstance) scatterChartInstance.destroy();

  scatterChartInstance = new Chart(ctx, {
    type: "scatter",
    data: { datasets: [{ label: "Productos", data: puntos, pointRadius: 6 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const p = ctx.raw;
              return `${p.etiqueta}: Ingresos $${p.x.toLocaleString()} | Utilidad ${(p.y * 100).toFixed(1)}%`;
            }
          }
        }
      },
      scales: {
        x: { title: { text: "Ingresos (USD)" }, ticks: { callback: v => `$${(v / 1000).toFixed(0)}k` } },
        y: { title: { text: "% Utilidad" }, ticks: { callback: v => `${(v * 100).toFixed(0)}%` } }
      }
    }
  });
}

/**
 * initLineChart(selectedVendor, selectedMonth): dibuja el gráfico de líneas
 */
async function initLineChart(selectedVendor = "Todos", selectedMonth = null) {
  const titleEl = document.getElementById("line-chart-title");
  if (selectedMonth) {
    const fecha = new Date(selectedMonth + "-01");
    titleEl.innerText = `Ventas diarias de ${fecha.toLocaleString("es-ES",{ month:"long",year:"numeric" })} por Cliente`;
  } else {
    titleEl.innerText = "Ventas 2020 por Cliente";
  }

  let url = `/sales_trend?year=2020&vendor=${encodeURIComponent(selectedVendor)}`;
  if (selectedMonth) {
    const anio = selectedMonth.split("-")[0];
    url = `/sales_trend?year=${anio}&month=${encodeURIComponent(selectedMonth)}&vendor=${encodeURIComponent(selectedVendor)}`;
  }

  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error("Error al obtener datos de ventas");
    const json = await resp.json();

    const ctx = document.getElementById("line-chart").getContext("2d");
    if (lineChartInstance) lineChartInstance.destroy();

    const datasets = json.datasets.map(ds => ({
      label: ds.vendor,
      data: ds.values,
      fill: false,
      tension: 0.3,
      pointRadius: 4,
      borderWidth: 2
    }));

    lineChartInstance = new Chart(ctx, {
      type: "line",
      data: { labels: json.labels, datasets },
      options: { responsive: true, maintainAspectRatio: false }
    });
  }
  catch (err) {
    console.error("initLineChart():", err);
  }
}

/**
 * populateDropdowns(): rellena selects de Cliente y Producto
 */
async function populateDropdowns() {
  const fields = [
    { field: "Customer Name", selectId: "vendor-select" },
    { field: "Product Name",  selectId: "product-select" }
  ];
  for (let {field, selectId} of fields) {
    try {
      const resp = await fetch(`/grouped?field=${encodeURIComponent(field)}`);
      if (!resp.ok) throw "";
      const { data } = await resp.json();
      const sel = document.getElementById(selectId);
      sel.innerHTML = `<option value="Todos">Todos</option>`;
      data.forEach(item => {
        if (item.group) {
          const opt = new Option(item.group, item.group);
          sel.append(opt);
        }
      });
    } catch {}
  }
}

/**
 * populatePredictionDropdowns(): rellena selects de predicción
 */
async function populatePredictionDropdowns() {
  const specs = [
    { f:"Region",        id:"pred-region",      ph:"Ej. West" },
    { f:"Product ID",    id:"pred-product-id",  ph:"Ej. P-1001" },
    { f:"Category",      id:"pred-category",    ph:"Ej. Technology" },
    { f:"Sub-Category",  id:"pred-sub-category",ph:"Ej. Phones" },
    { f:"Product Name",  id:"pred-product-name",ph:"Ej. iPhone 12" }
  ];
  for (let s of specs) {
    try {
      const resp = await fetch(`/grouped?field=${encodeURIComponent(s.f)}`);
      if (!resp.ok) throw "";
      const { data } = await resp.json();
      const sel = document.getElementById(s.id);
      sel.innerHTML = `<option value="">${s.ph}</option>`;
      data.forEach(r => sel.append(new Option(r.group, r.group)));
    } catch {}
  }
}

/**
 * submitPrediction(): envía /predict y muestra el resultado
 */
async function submitPrediction() {
  const getVal = id => document.getElementById(id).value;
  const payload = {
    "Region":        getVal("pred-region"),
    "Product ID":    getVal("pred-product-id"),
    "Category":      getVal("pred-category"),
    "Sub-Category":  getVal("pred-sub-category"),
    "Product Name":  getVal("pred-product-name"),
    "Quantity":      +getVal("pred-quantity") || 0,
    "Discount":      +getVal("pred-discount") || 0,
    "Profit":        +getVal("pred-profit")   || 0
  };

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify([payload])
    });
    if (!resp.ok) throw "";
    const { predictions } = await resp.json();
    document.getElementById("prediction-result").innerText =
      `🔮 Predicción de Ventas: $${predictions[0].toFixed(2)}`;
  } catch {
    document.getElementById("prediction-result").innerText =
      "❌ Ocurrió un error al predecir.";
  }
}

// ------------------------------
// initDashboard(): arranca todo
// ------------------------------
async function initDashboard() {
  await populateDropdowns();
  await populatePredictionDropdowns();

  const initial = { month: null, vendor: "Todos", product: "Todos" };
  updateKpisDisplay(await fetchKpis(initial));
  await initLineChart("Todos", null);
  await drawScatterChart();
  drawBarChart(await fetchGrouped("Category", initial));

  // listeners de filtros
  const els = ["month-range","vendor-select","product-select","group-by"];
  const monthEl = document.getElementById("month-range");
  const vendorEl= document.getElementById("vendor-select");
  const prodEl  = document.getElementById("product-select");
  const groupEl = document.getElementById("group-by");

  const onChange = async () => {
    const f = { month: monthEl.value||null, vendor: vendorEl.value, product: prodEl.value };
    updateKpisDisplay(await fetchKpis(f));
    await initLineChart(f.vendor, f.month);
    drawBarChart(await fetchGrouped(groupEl.value, f));
    await drawScatterChart();
  };

  [monthEl, vendorEl, prodEl, groupEl].forEach(el => el.addEventListener("change", onChange));

  // listener predicción
  document
    .getElementById("prediction-form")
    .addEventListener("submit", e => { e.preventDefault(); submitPrediction(); });
}

// Exportamos para que upload.js lo invoque
window.initDashboard = initDashboard;

