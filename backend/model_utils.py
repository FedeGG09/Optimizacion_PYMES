import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------------------------------------------------
# 1) Rutas absolutas al modelo XGBoost y a los feature names
# -------------------------------------------------------------------
MODEL_PATH    = Path(r"D:\Repositorios\Proyecto Sales forecasting\best_xgb_model.pkl")
FEATURES_PATH = Path(r"D:\Repositorios\Proyecto Sales forecasting\feature_names.pkl")


def load_data(path: str) -> pd.DataFrame:
    """
    Carga el CSV con encoding latin1 y devuelve un DataFrame.
    Verifica si el archivo existe.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No encontré el archivo CSV en la ruta: {path}")
    return pd.read_csv(path, encoding="latin1")


def load_model():
    """
    Carga el modelo XGBoost serializado desde disco (best_xgb_model.pkl).
    Si no existe, lanza FileNotFoundError.
    """
    if MODEL_PATH.is_file():
        return joblib.load(str(MODEL_PATH))
    else:
        raise FileNotFoundError(f"El modelo no fue encontrado en: {MODEL_PATH}")


def load_feature_names():
    """
    Carga la lista de nombres de columnas (features) desde disco
    (feature_names.pkl).
    """
    if FEATURES_PATH.is_file():
        return joblib.load(str(FEATURES_PATH))
    else:
        raise FileNotFoundError(f"No encontré el archivo de features en: {FEATURES_PATH}")


def get_target_column_name(df: pd.DataFrame) -> str:
    """
    Devuelve el nombre válido de la columna objetivo (ventas) si existe.
    Lanza un KeyError si no se encuentra ninguna columna válida.
    """
    posibles_nombres = [
        "Sales", "sales", "Ventas", "ventas", "Total_Sales", "total_sales", "Total Ventas", "total ventas",
        "sale", "ventas_totales", "ventasTotal", "ventas total", "sales_total", "sales amount", "amount_sold",
        "revenue", "Revenue",
        "valor_ventas", "valor ventas", "monto_ventas", "monto ventas"
    ]
    for nombre in posibles_nombres:
        if nombre in df.columns:
            return nombre
    raise KeyError(f"No se encontró ninguna columna de ventas válida. Nombres esperados: {posibles_nombres}")


def predict_from_dataframe(df: pd.DataFrame):
    """
    Genera predicciones para un DataFrame df:
      1) Elimina columna de ventas si está presente.
      2) Crea dummies.
      3) Alinea columnas con las del entrenamiento.
      4) Retorna la lista de predicciones.
    """
    # 1) Cargar modelo
    model = load_model()

    # 2) Detectar y eliminar columna objetivo si está presente
    try:
        target_col = get_target_column_name(df)
        X_raw = df.drop([target_col], axis=1, errors="ignore")
    except KeyError:
        X_raw = df.copy()

    # 3) Codificación one-hot
    X_encoded = pd.get_dummies(X_raw, drop_first=True)

    # 4) Alinear columnas
    trained_features = load_feature_names()
    X_aligned = X_encoded.reindex(columns=trained_features, fill_value=0)

    # 5) Predicción
    preds = model.predict(X_aligned)
    return preds.tolist()

def evaluate_model(df: pd.DataFrame):
    """
    Calcula métricas R2, MAE, MSE, RMSE usando el modelo XGBoost precargado.
    """
    # 1) Detectar columna objetivo
    target_col = get_target_column_name(df)

    y_true = df[target_col]
    X_raw = df.drop([target_col], axis=1, errors="ignore")

    # 2) Crear dummies y alinear
    X_encoded = pd.get_dummies(X_raw, drop_first=True)
    trained_features = load_feature_names()
    X_aligned = X_encoded.reindex(columns=trained_features, fill_value=0)

    # 3) Cargar modelo y predecir
    model = load_model()
    y_pred = model.predict(X_aligned)

    # 4) Métricas
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5

    return {
        "r2": float(r2),
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse)
    }