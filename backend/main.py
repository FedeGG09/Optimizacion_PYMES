from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import io
from pathlib import Path

from backend.model_utils import load_data, predict_from_dataframe, evaluate_model
from backend.models.util import parse_month

# -------------------------------------------------------
# INSTANCIA FastAPI Y CORS
# -------------------------------------------------------
app = FastAPI(
    title="Sales Forecasting API",
    description="API para predecir ventas y calcular KPIs.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# RUTAS Y VARIABLES GLOBALES
# -------------------------------------------------------
BASE_DIR         = Path(__file__).resolve().parent
PROJECT_DIR      = BASE_DIR.parent
uploaded_csv_path: Path | None = None

# Frontend estáticos
FRONTEND_DIR = PROJECT_DIR / "frontend"
app.mount("/static/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="static_css")
app.mount("/static/js",  StaticFiles(directory=str(FRONTEND_DIR / "js")),  name="static_js")

@app.get("/")
def serve_frontend():
    index_path = FRONTEND_DIR / "src" / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "index.html no encontrado en frontend/src/")
    return FileResponse(str(index_path))

# -------------------------------------------------------
# Upload CSV
# -------------------------------------------------------
@app.post("/upload_csv")
def upload_training_csv(file: UploadFile = File(...)):
    global uploaded_csv_path
    try:
        contents = file.file.read()
        target = PROJECT_DIR / "stores_sales_forecasting.csv"
        with open(target, "wb") as f:
            f.write(contents)
        uploaded_csv_path = target
        return {"detail": f"CSV cargado en {target.name}"}
    except Exception as e:
        raise HTTPException(500, str(e))

def _get_df() -> pd.DataFrame:
    if uploaded_csv_path:
        return pd.read_csv(str(uploaded_csv_path), encoding="latin1")
    else:
        raise HTTPException(400, "No se ha subido ningún CSV.")

# -------------------------------------------------------
# Métricas XGB
# -------------------------------------------------------
@app.get("/metrics_xgb")
def metrics_xgb_endpoint():
    try:
        df = load_data(str(uploaded_csv_path)) if uploaded_csv_path else _get_df()
        return {"metrics": evaluate_model(df)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

# -------------------------------------------------------
# Predicciones
# -------------------------------------------------------
@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):
    try:
        data = file.file.read()
        df   = pd.read_csv(io.BytesIO(data), encoding="latin1")
        return {"predictions": predict_from_dataframe(df)}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/predict")
def predict_json(data: list[dict]):
    try:
        df = pd.DataFrame(data)
        return {"predictions": predict_from_dataframe(df)}
    except Exception as e:
        raise HTTPException(400, str(e))

# -------------------------------------------------------
# KPIs con filtrado por mes (nombre), vendor y producto
# -------------------------------------------------------
@app.get("/kpis")
def get_kpis(
    month:   str = Query(None, description="Mes en español (p.ej. 'febrero')"),
    vendor:  str = Query("Todos", description="Customer Name"),
    product: str = Query("Todos", description="Product Name")
):
    df = _get_df()
    # Aseguramos datetime
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    else:
        raise HTTPException(500, "No existe la columna 'Order Date'.")

    # Filtrado por mes
    if month:
        try:
            mnum = parse_month(month)
        except ValueError as e:
            raise HTTPException(400, str(e))
        df = df[df["Order Date"].dt.month == mnum]

    # Filtrado por vendor/product
    if vendor != "Todos":
        df = df[df["Customer Name"] == vendor]
    if product != "Todos":
        df = df[df["Product Name"]  == product]

    # Verificación de columnas
    for col in ("Sales","Profit"):
        if col not in df.columns:
            raise HTTPException(500, f"Falta columna '{col}' para KPI.")

    total_sales    = df["Sales"].sum()
    avg_profit_pct = (df["Profit"] / df["Sales"]).mean() if total_sales else 0
    count          = len(df)
    avg_sales      = df["Sales"].mean() if count else 0

    return {
        "total_sales":    float(total_sales),
        "avg_profit_pct": float(avg_profit_pct),
        "sale_count":     count,
        "avg_sales":      float(avg_sales)
    }

# -------------------------------------------------------
# Datos agrupados
# -------------------------------------------------------
@app.get("/grouped")
def get_grouped_data(
    field:   str = Query(..., description="Campo para agrupar"),
    month:   str = Query(None, description="Mes en español (p.ej. 'marzo')"),
    vendor:  str = Query("Todos"),
    product: str = Query("Todos")
):
    df = _get_df()
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    else:
        raise HTTPException(500, "No existe la columna 'Order Date'.")

    if month:
        try:
            mnum = parse_month(month)
        except ValueError as e:
            raise HTTPException(400, str(e))
        df = df[df["Order Date"].dt.month == mnum]

    if vendor  != "Todos":
        df = df[df["Customer Name"] == vendor]
    if product != "Todos":
        df = df[df["Product Name"]  == product]

    if field not in df.columns:
        raise HTTPException(400, f"Campo '{field}' no existe.")

    for col in ("Sales","Quantity","Discount","Profit"):
        if col not in df.columns:
            raise HTTPException(500, f"Falta columna '{col}' para agrupamiento.")

    grouped = (
        df.groupby(field, dropna=False)
          .agg(
              total_sales   = ("Sales",    "sum"),
              total_quantity= ("Quantity", "sum"),
              avg_discount  = ("Discount", "mean"),
              total_profit  = ("Profit",   "sum")
          )
          .reset_index()
          .rename(columns={field: "group"})
          .sort_values("total_sales", ascending=False)
    )

    return {"data": grouped.to_dict(orient="records")}

# -------------------------------------------------------
# Tendencia de ventas
# -------------------------------------------------------
@app.get("/sales_trend")
def sales_trend(
    year:   int    = Query(..., description="Año (p.ej. 2020)"),
    month:  str    = Query(None, description="Mes en español (p.ej. 'abril')"),
    vendor: str    = Query("Todos")
):
    df = _get_df()
    if "Order Date" not in df.columns:
        raise HTTPException(500, "No existe 'Order Date'.")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df[df["Order Date"].dt.year == year]

    # Filtrar por mes si se indica
    if month:
        try:
            mnum = parse_month(month)
        except ValueError as e:
            raise HTTPException(400, str(e))
        df = df[df["Order Date"].dt.month == mnum]
        if vendor != "Todos":
            df = df[df["Customer Name"] == vendor]

        df["Day"] = df["Order Date"].dt.day
        pivot = (
            df.groupby(["Day","Customer Name"])["Sales"]
              .sum()
              .unstack(fill_value=0)
        )
        days = list(range(1, calendar.monthrange(year, mnum)[1] + 1))
        labels = [f"{month}-{d:02d}" for d in days]

        return {
            "labels": labels,
            "datasets": [
                {"vendor": v, "values": pivot.get(v, pd.Series([0]*len(days))).tolist()}
                for v in pivot.columns
            ]
        }

    # Serie mes a mes
    if vendor != "Todos":
        df = df[df["Customer Name"] == vendor]

    df["YearMonth"] = df["Order Date"].dt.month
    pivot = (
        df.groupby(["YearMonth","Customer Name"])["Sales"]
          .sum()
          .unstack(fill_value=0)
    )
    months = list(range(1,13))
    labels = [f"{year}-{m:02d}" for m in months]

    return {
        "labels": labels,
        "datasets": [
            {
                "vendor": v,
                "values": [float(pivot.get(v, pd.Series([0]*12)).get(m, 0)) for m in months]
            }
            for v in pivot.columns
        ]
    }
