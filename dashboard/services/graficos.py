"""Gráficos generados en el servidor con matplotlib y entregados como imágenes PNG en base64."""
import base64
import io
from datetime import timedelta

import matplotlib

matplotlib.use("Agg")  # sin interfaz gráfica (servidor)
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROJO, AZUL, GRIS, TINTA = "#c8102e", "#1f5fa8", "#8a8f98", "#1d2430"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c9ced6", "axes.labelcolor": TINTA, "xtick.color": "#4a5260",
                     "ytick.color": "#4a5260"})


MESES_CORTOS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def _fmt_dia(x, _pos=None):
    d = mdates.num2date(x)
    return f"{d.day:02d}-{MESES_CORTOS[d.month - 1]}"


def _fmt_mes(x, _pos=None):
    d = mdates.num2date(x)
    return f"{MESES_CORTOS[d.month - 1]} {d.year}"


def _a_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def factores(cont: dict) -> str:
    """Aportes TreeSHAP del día: rojo sube el pronóstico, azul lo baja."""
    f = list(reversed(cont["factores"]))
    etiquetas = [x["nombre"] for x in f]
    valores = [x["aporte"] for x in f]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(f) + 0.9))
    ax.barh(etiquetas, valores, color=[ROJO if v > 0 else AZUL for v in valores], height=0.62)
    ax.axvline(0, color=GRIS, linewidth=0.8)
    lim = max(abs(v) for v in valores) * 1.35 or 1
    ax.set_xlim(-lim, lim)
    for i, v in enumerate(valores):
        ax.text(v + (0.03 * lim if v >= 0 else -0.03 * lim), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=TINTA)
    ax.set_xlabel("Aporte al pronóstico (µg/m³)")
    ax.set_title(f"Factores que explican el pronóstico de {cont['nombre']}", fontsize=10, color=TINTA, loc="left")
    return _a_base64(fig)


def reciente(servicio, columna: str, nombre: str, fecha, prediccion: float, umbral: float, dias: int = 30) -> str:
    """Últimos días observados y el valor pronosticado para la fecha consultada."""
    fecha = pd.Timestamp(fecha)
    d = servicio.datos[(servicio.datos["date"] < fecha) & (servicio.datos["date"] >= fecha - timedelta(days=dias))]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.plot(d["date"], d[columna], color=TINTA, linewidth=1.4, label="Observado")
    ax.plot([d["date"].iloc[-1], fecha], [d[columna].iloc[-1], prediccion], color=ROJO, linestyle="--", linewidth=1.2)
    ax.scatter([fecha], [prediccion], color=ROJO, zorder=3, s=36, label="Pronóstico")
    obs = servicio.datos.loc[servicio.datos["date"] == fecha, columna]
    if len(obs) and pd.notna(obs.iloc[0]):
        ax.scatter([fecha], [obs.iloc[0]], color=TINTA, marker="x", zorder=3, s=40, label="Observado ese día")
    ax.axhline(umbral, color="#f07f13", linewidth=0.9, linestyle=":", label="Umbral día crítico (P90)")
    ax.set_ylabel(f"{nombre} (µg/m³)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_dia))
    ax.legend(fontsize=7, loc="upper left", frameon=False, ncol=2)
    return _a_base64(fig)


def historial(prueba: pd.DataFrame, columna: str, nombre: str, umbral: float) -> str:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(prueba["date"], prueba[columna], color=TINTA, linewidth=1.3, label=f"{nombre} observado")
    ax.plot(prueba["date"], prueba[f"pred_{columna}"], color=ROJO, linewidth=1.3, linestyle="--",
            label="Pronóstico 24 h (XGBoost)")
    ax.axhline(umbral, color="#f07f13", linewidth=0.9, linestyle=":", label="Umbral día crítico (P90)")
    ax.set_ylabel("µg/m³")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_mes))
    ax.legend(fontsize=8, loc="upper left", frameon=False, ncol=3)
    ax.set_title(f"{nombre}: observado vs. pronosticado en el periodo de prueba", fontsize=10, color=TINTA, loc="left")
    return _a_base64(fig)


def importancia(tabla: pd.DataFrame, nombre: str, top: int = 12) -> str:
    t = tabla.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.36 * top + 0.9))
    ax.barh(t["nombre"], t["importancia"], color=AZUL, height=0.62)
    for i, (v, p) in enumerate(zip(t["importancia"], t["peso"])):
        ax.text(v, i, f"  {v:.2f} ({p:.1f} %)", va="center", fontsize=8, color=TINTA)
    ax.set_xlim(0, t["importancia"].max() * 1.35)
    ax.set_xlabel("Importancia media |SHAP| (µg/m³)")
    ax.set_title(f"Variables más influyentes en el pronóstico de {nombre}", fontsize=10, color=TINTA, loc="left")
    return _a_base64(fig)


def grupos(tabla: pd.DataFrame, nombre: str) -> str:
    t = tabla.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    ax.barh(t["grupo"], t["peso"], color=["#9ab7d8", "#6f97c6", "#3f76b3", AZUL][-len(t):], height=0.6)
    for i, v in enumerate(t["peso"]):
        ax.text(v, i, f"  {v:.1f} %", va="center", fontsize=8, color=TINTA)
    ax.set_xlim(0, t["peso"].max() * 1.25)
    ax.set_xlabel("Peso relativo en la explicación (%)")
    ax.set_title(f"Importancia por grupo de variables — {nombre}", fontsize=10, color=TINTA, loc="left")
    return _a_base64(fig)
