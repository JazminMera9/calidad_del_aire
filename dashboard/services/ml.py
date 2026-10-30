"""
Servicio de predicción y explicabilidad.

Reproduce exactamente el flujo del notebook E2_Capitulo2:
  1. Depuración del dataset (las mismas cuatro reglas del Capítulo 1).
  2. Ingeniería de variables del escenario A (pronóstico a 24 horas, 21 variables).
  3. Predicción con los modelos XGBoost seleccionados (uno por contaminante).
  4. Explicaciones TreeSHAP exactas, calculadas con el algoritmo nativo de XGBoost
     (predict con pred_contribs=True), que produce los mismos valores que shap.TreeExplainer.

Todo se calcula una sola vez al iniciar el servidor y queda en memoria (función obtener_servicio()).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

BASE = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE / "data" / "air_quality_historical.csv"
CARPETA_MODELOS = BASE / "ml_models"

OBJETIVOS = {"PM2.5": "pm2_5", "PM10": "pm10"}
CONTAMINANTES = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
OPTICAS = ["aerosol_optical_depth", "dust", "uv_index"]
REZAGOS = [f"{c}_{s}" for c in CONTAMINANTES for s in ("lag1", "lag2")] + [
    "pm10_ma3", "pm10_delta", "pm2_5_ma3", "pm2_5_delta"]
FEATURES_A = REZAGOS + [f"{c}_lag1" for c in OPTICAS] + ["mes", "dia_semana"]

# Partición cronológica del Capítulo 2 (70 / 15 / 15 sobre los 1.294 días de modelado)
N_TRAIN, N_VAL = 906, 194

# Nombres legibles para la interfaz
NOMBRES = {
    "pm2_5": "PM2.5", "pm10": "PM10", "carbon_monoxide": "CO", "nitrogen_dioxide": "NO₂",
    "sulphur_dioxide": "SO₂", "ozone": "O₃", "aerosol_optical_depth": "AOD", "dust": "Polvo",
    "uv_index": "Índice UV",
}
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
         "octubre", "noviembre", "diciembre"]


def nombre_variable(v: str) -> str:
    """'pm2_5_lag1' -> 'PM2.5 de ayer (t−1)'"""
    if v == "mes":
        return "Mes del año"
    if v == "dia_semana":
        return "Día de la semana"
    for sufijo, texto in (("_lag1", "de ayer (t−1)"), ("_lag2", "de anteayer (t−2)"),
                          ("_ma3", "promedio últimos 3 días"), ("_delta", "variación ayer vs. anteayer")):
        if v.endswith(sufijo):
            base = v[: -len(sufijo)]
            return f"{NOMBRES.get(base, base)} {texto}"
    return NOMBRES.get(v, v)


def grupo_variable(v: str) -> str:
    if v in ("mes", "dia_semana"):
        return "Calendario (estacionalidad)"
    if v.startswith(("pm10", "pm2_5")):
        return "Persistencia del material particulado"
    if v.split("_lag")[0] in OPTICAS:
        return "Aerosoles y radiación (día previo)"
    return "Gases precursores y ozono (días previos)"


# ----------------------------------------------------------------------------------------------
# Índice de Calidad del Aire — Resolución 2254 de 2017 (concentraciones de 24 horas, µg/m³)
# ----------------------------------------------------------------------------------------------
ICA = {
    "pm2_5": [(12, "Buena", "#2e9e5b"), (37, "Aceptable", "#e3b505"), (55, "Dañina a la salud de grupos sensibles", "#f07f13"),
              (150, "Dañina a la salud", "#d7263d"), (250, "Muy dañina a la salud", "#7b2d8e"), (float("inf"), "Peligrosa", "#7e0023")],
    "pm10": [(54, "Buena", "#2e9e5b"), (154, "Aceptable", "#e3b505"), (254, "Dañina a la salud de grupos sensibles", "#f07f13"),
             (354, "Dañina a la salud", "#d7263d"), (424, "Muy dañina a la salud", "#7b2d8e"), (float("inf"), "Peligrosa", "#7e0023")],
}
MENSAJES = {
    "Buena": "La calidad del aire es satisfactoria y no representa riesgo para la salud.",
    "Aceptable": "Calidad aceptable; personas muy sensibles podrían presentar síntomas leves.",
    "Dañina a la salud de grupos sensibles": "Niños, adultos mayores y personas con enfermedades respiratorias deben "
                                             "reducir la actividad física al aire libre.",
    "Dañina a la salud": "Toda la población puede presentar efectos en la salud; se recomienda limitar la exposición.",
    "Muy dañina a la salud": "Alerta sanitaria: toda la población puede presentar efectos graves.",
    "Peligrosa": "Emergencia: riesgo grave para la salud de toda la población.",
}


def categoria_ica(columna: str, valor: float) -> dict:
    for limite, nombre, color in ICA[columna]:
        if valor <= limite:
            return {"nombre": nombre, "color": color, "mensaje": MENSAJES[nombre]}
    raise ValueError(valor)


# ----------------------------------------------------------------------------------------------
# Preparación de datos (idéntica al notebook)
# ----------------------------------------------------------------------------------------------
def depurar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # 1. Poda de borde: días iniciales sin ningún dato
    while len(df) and df.drop(columns="date").iloc[0].isna().all():
        df = df.iloc[1:].reset_index(drop=True)
    # 2. AQI nulos -> valor del día siguiente (no son predictores)
    df[["us_aqi", "european_aqi"]] = df[["us_aqi", "european_aqi"]].bfill()
    # 3. NO2 = 0 (dato inválido) -> interpolación lineal
    df.loc[df["nitrogen_dioxide"] == 0, "nitrogen_dioxide"] = np.nan
    df["nitrogen_dioxide"] = df["nitrogen_dioxide"].interpolate(method="linear")
    # 4. UV = 0 (registro incompleto) -> valor del día siguiente
    df.loc[df["uv_index"] == 0, "uv_index"] = np.nan
    df["uv_index"] = df["uv_index"].bfill()
    return df


def construir_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Variables del escenario A. Todas usan .shift(1) o mayor: ninguna contiene información del día predicho."""
    d = df.copy()
    for c in CONTAMINANTES:
        d[f"{c}_lag1"] = d[c].shift(1)
        d[f"{c}_lag2"] = d[c].shift(2)
    for c in OPTICAS:
        d[f"{c}_lag1"] = d[c].shift(1)
    for o in ("pm10", "pm2_5"):
        d[f"{o}_ma3"] = d[o].shift(1).rolling(window=3, min_periods=2).mean()
        d[f"{o}_delta"] = d[o].shift(1) - d[o].shift(2)
    d["mes"] = d["date"].dt.month
    d["dia_semana"] = d["date"].dt.dayofweek
    return d


# ----------------------------------------------------------------------------------------------
# Servicio
# ----------------------------------------------------------------------------------------------
def _cargar_modelo(columna: str):
    """Carga el modelo como xgboost.Booster desde el formato nativo JSON (recomendado) o desde el .joblib exportado."""
    for patron in (f"modelo_{columna}.json", f"modelo_{columna}_XGBoost.json"):
        ruta = CARPETA_MODELOS / patron
        if ruta.exists():
            booster = xgb.Booster()
            booster.load_model(ruta)
            return _validar(booster, ruta.name), ruta.name
    for ruta in sorted(CARPETA_MODELOS.glob(f"modelo_{columna}_*.joblib")):
        import joblib
        try:
            modelo = joblib.load(ruta)
        except xgb.core.XGBoostError as exc:
            # Un .joblib creado en Colab (Linux) puede fallar en Windows al leer el estado interno de entrenamiento
            raise FileNotFoundError(
                f"No se pudo abrir {ruta.name} en este equipo ({exc}). Exporte el modelo en formato nativo "
                f"modelo_{columna}.json desde el notebook E2_Capitulo2 (ver README.md).") from exc
        booster = modelo.get_booster() if hasattr(modelo, "get_booster") else modelo
        return _validar(booster, ruta.name), ruta.name
    raise FileNotFoundError(
        f"No se encontró el modelo de {columna} en {CARPETA_MODELOS}. Copie allí el archivo "
        f"modelo_{columna}.json exportado desde el notebook E2_Capitulo2 (ver README.md).")


def _validar(booster, nombre: str):
    """Comprueba que el modelo corresponda al escenario de pronóstico a 24 horas (21 variables en el mismo orden)."""
    if booster.feature_names is not None and list(booster.feature_names) != FEATURES_A:
        raise FileNotFoundError(
            f"El archivo {nombre} no corresponde al modelo de pronóstico a 24 horas: sus variables de entrada son "
            f"{booster.feature_names}. Exporte el modelo del escenario A desde el notebook E2_Capitulo2.")
    return booster


def _predecir(booster, X: pd.DataFrame) -> np.ndarray:
    return booster.predict(xgb.DMatrix(X[FEATURES_A].astype(float)))


@dataclass
class Servicio:
    datos: pd.DataFrame                      # serie depurada
    matriz: pd.DataFrame                     # serie con variables (sin las 2 filas iniciales)
    modelos: dict = field(default_factory=dict)        # xgboost.Booster por contaminante
    archivos: dict = field(default_factory=dict)
    umbral_p90: dict = field(default_factory=dict)
    prueba: pd.DataFrame | None = None
    contrib_prueba: dict = field(default_factory=dict)

    # ---------------- utilidades ----------------
    @property
    def primera_fecha(self) -> date:
        return self.matriz["date"].min().date()

    @property
    def ultima_fecha(self) -> date:
        return self.datos["date"].max().date()

    @property
    def fecha_pronostico(self) -> date:
        """Día siguiente al último dato disponible: el pronóstico operativo."""
        return self.ultima_fecha + timedelta(days=1)

    def _contribuciones(self, columna: str, X: pd.DataFrame) -> np.ndarray:
        booster = self.modelos[columna]
        return booster.predict(xgb.DMatrix(X[FEATURES_A].astype(float)), pred_contribs=True)  # última columna = valor base

    # ---------------- pronóstico de un día ----------------
    def fila_para(self, fecha: date) -> pd.Series:
        """Vector de entrada para predecir `fecha` usando solo los días anteriores."""
        fecha = pd.Timestamp(fecha)
        if fecha.date() == self.fecha_pronostico:
            extendida = pd.concat([self.datos, pd.DataFrame({"date": [fecha]})], ignore_index=True)
            return construir_variables(extendida).iloc[-1]
        filas = self.matriz[self.matriz["date"] == fecha]
        if filas.empty:
            raise ValueError("Fecha fuera del rango disponible")
        return filas.iloc[0]

    def pronosticar(self, fecha: date, top: int = 8) -> dict:
        fila = self.fila_para(fecha)
        X = pd.DataFrame([fila[FEATURES_A].astype(float)])
        resultado = {"fecha": pd.Timestamp(fecha).date(), "contaminantes": []}
        for obj, col in OBJETIVOS.items():
            pred = float(_predecir(self.modelos[col], X)[0])
            contrib = self._contribuciones(col, X)[0]
            base, aportes = float(contrib[-1]), contrib[:-1]
            orden = np.argsort(-np.abs(aportes))[:top]
            factores = [{
                "variable": FEATURES_A[i], "nombre": nombre_variable(FEATURES_A[i]),
                "valor": float(X.iloc[0, i]), "aporte": float(aportes[i]),
            } for i in orden]
            observado = None
            if col in fila.index and pd.notna(fila.get(col)):
                observado = float(fila[col])
            persistencia = float(fila[f"{col}_lag1"])
            resultado["contaminantes"].append({
                "nombre": obj, "columna": col, "prediccion": pred, "base": base,
                "suma_aportes": float(aportes.sum()), "observado": observado,
                "error": (pred - observado) if observado is not None else None,
                "persistencia": persistencia,
                "ica": categoria_ica(col, pred),
                "critico": pred > self.umbral_p90[obj],
                "umbral": self.umbral_p90[obj],
                "factores": factores,
                "suma_mostrados": float(sum(f["aporte"] for f in factores)),
                "resto": float(aportes.sum() - sum(f["aporte"] for f in factores)),
            })
        resultado["mes"] = MESES[pd.Timestamp(fecha).month - 1]
        resultado["dia_semana"] = DIAS[pd.Timestamp(fecha).dayofweek]
        return resultado

    # ---------------- evaluación en el conjunto de prueba ----------------
    def metricas_prueba(self) -> list[dict]:
        filas = []
        for obj, col in OBJETIVOS.items():
            y = self.prueba[col].to_numpy()
            for nombre, pred in (("Persistencia (t−1)", self.prueba[f"{col}_lag1"].to_numpy()),
                                 ("Modelo XGBoost (pronóstico 24 h)", self.prueba[f"pred_{col}"].to_numpy())):
                err = pred - y
                mae = float(np.mean(np.abs(err)))
                rmse = float(np.sqrt(np.mean(err ** 2)))
                r2 = float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))
                pico = y > self.umbral_p90[obj]
                filas.append({"contaminante": obj, "modelo": nombre, "mae": mae, "rmse": rmse, "r2": r2,
                              "mae_picos": float(np.mean(np.abs(err[pico]))) if pico.any() else None,
                              "n_picos": int(pico.sum()), "mejora": None})
            mod, per = filas[-1], filas[-2]
            mod["mejora"] = (1 - mod["mae"] / per["mae"]) * 100
        return filas

    def importancia_global(self, columna: str) -> pd.DataFrame:
        c = self.contrib_prueba[columna][:, :-1]
        X = self.prueba[FEATURES_A]
        imp = np.abs(c).mean(axis=0)
        corr = [float(np.corrcoef(X[v], c[:, i])[0, 1]) if X[v].std() > 0 else 0.0 for i, v in enumerate(FEATURES_A)]
        t = pd.DataFrame({"variable": FEATURES_A, "nombre": [nombre_variable(v) for v in FEATURES_A],
                          "grupo": [grupo_variable(v) for v in FEATURES_A], "importancia": imp,
                          "peso": imp / imp.sum() * 100, "correlacion": corr})
        return t.sort_values("importancia", ascending=False).reset_index(drop=True)

    def importancia_grupos(self, columna: str) -> pd.DataFrame:
        t = self.importancia_global(columna)
        return (t.groupby("grupo")[["importancia", "peso"]].sum()
                .sort_values("peso", ascending=False).reset_index())


def crear_servicio(ruta_datos: Path = RUTA_DATOS) -> Servicio:
    datos = depurar_datos(pd.read_csv(ruta_datos))
    matriz = construir_variables(datos).iloc[2:].reset_index(drop=True)
    s = Servicio(datos=datos, matriz=matriz)
    for col in OBJETIVOS.values():
        s.modelos[col], s.archivos[col] = _cargar_modelo(col)
    entrenamiento = matriz.iloc[:N_TRAIN]
    s.umbral_p90 = {obj: float(entrenamiento[col].quantile(0.90)) for obj, col in OBJETIVOS.items()}
    prueba = matriz.iloc[N_TRAIN + N_VAL:].copy()
    for col in OBJETIVOS.values():
        prueba[f"pred_{col}"] = _predecir(s.modelos[col], prueba)
        s.contrib_prueba[col] = s._contribuciones(col, prueba)
    s.prueba = prueba
    return s


@lru_cache(maxsize=1)
def obtener_servicio() -> Servicio:
    return crear_servicio()


def metadatos() -> dict:
    ruta = CARPETA_MODELOS / "metadatos_modelos.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return {}
