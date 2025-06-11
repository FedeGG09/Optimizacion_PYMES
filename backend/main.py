from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import io
from pathlib import Path

from backend.model_utils import load_data, predict_from_dataframe, evaluate_model
# -------------------------------------------------------
# INSTANCIAMOS FastAPI Y CONFIGURAMOS CORS
# -------------------------------------------------------
app = FastAPI(
    title="Sales Forecasting API",
    description="API para predecir ventas con XGBoost y obtener métricas.",
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
# RUTAS Y ARCHIVOS BASE
# -------------------------------------------------------
BASE_DIR         = Path(__file__).resolve().parent
PROJECT_DIR      = BASE_DIR.parent

# Variable global para la ruta del CSV subido
uploaded_csv_path: Path | None = None

# Directorio del frontend:
FRONTEND_DIR = PROJECT_DIR / "frontend"
SRC_DIR      = FRONTEND_DIR / "src"
CSS_DIR      = FRONTEND_DIR / "css"
JS_DIR       = FRONTEND_DIR / "js"

# -------------------------------------------------------
# MONTAR ARCHIVOS ESTÁTICOS (Frontend)
# -------------------------------------------------------
app.mount("/static/css", StaticFiles(directory=str(CSS_DIR)), name="static_css")
app.mount("/static/js", StaticFiles(directory=str(JS_DIR)), name="static_js")

# -------------------------------------------------------
# SERVIR index.html CUANDO PIDAN "/"
# -------------------------------------------------------
@app.get("/")
def serve_frontend():
    index_path = SRC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html no encontrado en frontend/src/")
    return FileResponse(str(index_path))


# -------------------------------------------------------
# ENDPOINT: /upload_csv (Cargar CSV de entrenamiento)
# -------------------------------------------------------
@app.post("/upload_csv")
def upload_training_csv(file: UploadFile = File(...)):
    """
    Recibe el CSV de entrenamiento, lo guarda en PROJECT_DIR
    y actualiza la variable global `uploaded_csv_path`.
    """
    global uploaded_csv_path
    try:
        contents = file.file.read()
        target_path = PROJECT_DIR / "stores_sales_forecasting.csv"
        with open(target_path, "wb") as f:
            f.write(contents)
        uploaded_csv_path = target_path
        return {"detail": f"CSV cargado correctamente en {target_path.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_df():
    """
    Función auxiliar que carga el DataFrame desde el CSV subido.
    """
    if not uploaded_csv_path:
        raise HTTPException(status_code=400, detail="No se ha subido ningún CSV de entrenamiento.")
    # puedes usar load_data si tus utils lo requieren, o pd.read_csv directo:
    return pd.read_csv(str(uploaded_csv_path), encoding="latin1")


# -------------------------------------------------------
# ENDPOINT: /metrics_xgb (Obtener métricas del XGBoost)
# -------------------------------------------------------
@app.get("/metrics_xgb")
def metrics_xgb_endpoint():
    try:
        df = load_data(str(uploaded_csv_path)) if uploaded_csv_path else _get_df()
        metrics = evaluate_model(df)
        return JSONResponse(content={"metrics": metrics})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------
# ENDPOINT: /predict_csv (Predicción batch a partir de CSV subido)
# -------------------------------------------------------
@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents), encoding="latin1")
        preds = predict_from_dataframe(df)
        return JSONResponse(content={"predictions": preds})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------------------------------
# ENDPOINT: /predict (Predicción a partir de JSON)
# -------------------------------------------------------
@app.post("/predict")
def predict_json(data: list[dict]):
    try:
        df = pd.DataFrame(data)
        preds = predict_from_dataframe(df)
        return JSONResponse(content={"predictions": preds})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------------------------------
# ENDPOINT: /kpis (Obtener KPI de negocio con filtros opcionales)
# -------------------------------------------------------
@app.get("/kpis")
def get_kpis(
    month: str = Query(None, description="Filtrar por mes (YYYY-MM), opcional."),
    vendor: str = Query("Todos", description="Filtrar por Customer Name."),
    product: str = Query("Todos", description="Filtrar por Product Name.")
):
    df = _get_df()

    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    if month:
        df = df[df["Order Date"].dt.to_period("M") == pd.Period(month, freq="M")]
    if vendor and vendor != "Todos":
        df = df[df["Customer Name"] == vendor]
    if product and product != "Todos":
        df = df[df["Product Name"] == product]

    for col in ["Sales", "Profit"]:
        if col not in df.columns:
            raise HTTPException(status_code=500, detail=f"Columna '{col}' no encontrada para calcular KPIs.")

    total_sales    = df["Sales"].sum()
    avg_profit_pct = (df["Profit"] / df["Sales"]).mean() if total_sales != 0 else 0
    sale_count     = df.shape[0]
    avg_sales      = df["Sales"].mean() if sale_count > 0 else 0

    return {
        "total_sales": float(total_sales),
        "avg_profit_pct": float(avg_profit_pct),
        "sale_count": int(sale_count),
        "avg_sales": float(avg_sales)
    }


# -------------------------------------------------------
# ENDPOINT: /grouped (Agrupar datos según un campo + filtros)
# -------------------------------------------------------
@app.get("/grouped")
def get_grouped_data(
    field: str = Query(
        ...,
        description=(
            "Columna por la que agrupar. Puede ser: "
            "'State', 'Postal Code', 'Region', 'Product ID', 'Category', "
            "'Sub-Category', 'Product Name', 'Customer Name'."
        )
    ),
    month: str = Query(None, description="Filtrar por mes (YYYY-MM), opcional."),
    vendor: str = Query("Todos", description="Filtrar por Customer Name."),
    product: str = Query("Todos", description="Filtrar por Product Name.")
):
    df = _get_df()

    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    if month:
        df = df[df["Order Date"].dt.to_period("M") == pd.Period(month, freq="M")]
    if vendor and vendor != "Todos":
        df = df[df["Customer Name"] == vendor]
    if product and product != "Todos":
        df = df[df["Product Name"] == product]

    if field not in df.columns:
        raise HTTPException(status_code=400, detail=f"El campo '{field}' no existe en el CSV.")

    for col in ["Sales", "Quantity", "Discount", "Profit"]:
        if col not in df.columns:
            raise HTTPException(status_code=500, detail=f"Falta columna '{col}'.")

    grouped = (
        df.groupby(field, dropna=False)
          .agg(
              total_sales=pd.NamedAgg(column="Sales", aggfunc="sum"),
              total_quantity=pd.NamedAgg(column="Quantity", aggfunc="sum"),
              avg_discount=pd.NamedAgg(column="Discount", aggfunc="mean"),
              total_profit=pd.NamedAgg(column="Profit", aggfunc="sum")
          )
          .reset_index()
          .rename(columns={field: "group"})
          .sort_values("total_sales", ascending=False)
    )

    return {"data": [
        {
            "group": row["group"],
            "total_sales": float(row["total_sales"]),
            "total_quantity": int(row["total_quantity"]),
            "avg_discount": float(row["avg_discount"]),
            "total_profit": float(row["total_profit"])
        }
        for _, row in grouped.iterrows()
    ]}


# -------------------------------------------------------
# ENDPOINT: /sales_trend (Ventas por tiempo, mes o día, según filtros)
# -------------------------------------------------------
@app.get("/sales_trend")
def sales_trend(
    year: int = Query(2020, description="Año para el que se calculan las ventas (p.ej. 2020)"),
    month: str = Query(None, description="Filtrar por mes (YYYY-MM), opcional."),
    vendor: str = Query("Todos", description="Filtrar por Customer Name.")
):
    df = _get_df()

    if "Order Date" not in df.columns:
        raise HTTPException(status_code=500, detail="La columna 'Order Date' no existe en el CSV.")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df[df["Order Date"].dt.year == year]

    if month:
        try:
            periodo = pd.Period(month, freq="M")
        except:
            raise HTTPException(status_code=400, detail="Formato de month inválido. Debe ser 'YYYY-MM'.")
        df = df[df["Order Date"].dt.to_period("M") == periodo]
        if vendor and vendor != "Todos":
            df = df[df["Customer Name"] == vendor]

        df["Day"] = df["Order Date"].dt.day
        grouped = df.groupby(["Customer Name", "Day"], dropna=False)["Sales"].sum().reset_index()
        total_dias = periodo.days_in_month
        todos_dias = list(range(1, total_dias + 1))

        pivot = grouped.pivot(index="Day", columns="Customer Name", values="Sales").fillna(0)
        pivot = pivot.reindex(todos_dias, fill_value=0)
        labels = [f"{month}-{dia:02d}" for dia in todos_dias]

        return {
            "labels": labels,
            "datasets": [
                {"vendor": cliente, "values": [float(v) for v in pivot[cliente].tolist()]}
                for cliente in pivot.columns
            ]
        }

    # mes no especificado: serie mes a mes
    if vendor and vendor != "Todos":
        df = df[df["Customer Name"] == vendor]
    if "Sales" not in df.columns or "Customer Name" not in df.columns:
        raise HTTPException(status_code=500, detail="El CSV no contiene 'Sales' o 'Customer Name'.")

    df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)
    grouped = df.groupby(["Customer Name", "YearMonth"], dropna=False)["Sales"].sum().reset_index()
    pivot = grouped.pivot(index="YearMonth", columns="Customer Name", values="Sales").fillna(0)
    todos_meses = [f"{year}-{mes:02d}" for mes in range(1, 13)]
    pivot = pivot.reindex(todos_meses, fill_value=0)

    return {
        "labels": todos_meses,
        "datasets": [
            {"vendor": cliente, "values": [float(v) for v in pivot[cliente].tolist()]}
            for cliente in pivot.columns
        ]
    }
