from datetime import date

from django.shortcuts import render

from .services import graficos
from .services.ml import OBJETIVOS, metadatos, obtener_servicio


def _servicio_o_error(request):
    """Devuelve el servicio de predicción o una página de error amigable si faltan archivos."""
    try:
        return obtener_servicio(), None
    except FileNotFoundError as exc:
        return None, render(request, "dashboard/error.html", {"mensaje": str(exc)}, status=503)


def inicio(request):
    servicio, error = _servicio_o_error(request)
    if error:
        return error

    mensaje = None
    fecha = servicio.fecha_pronostico
    texto = request.GET.get("fecha")
    if texto:
        try:
            pedida = date.fromisoformat(texto)
            if servicio.primera_fecha <= pedida <= servicio.fecha_pronostico:
                fecha = pedida
            else:
                mensaje = (f"La fecha debe estar entre {servicio.primera_fecha:%d/%m/%Y} y "
                           f"{servicio.fecha_pronostico:%d/%m/%Y}. Se muestra el pronóstico más reciente.")
        except ValueError:
            mensaje = "Formato de fecha no válido. Se muestra el pronóstico más reciente."

    resultado = servicio.pronosticar(fecha)
    for c in resultado["contaminantes"]:
        c["grafico_factores"] = graficos.factores(c)
        c["grafico_reciente"] = graficos.reciente(servicio, c["columna"], c["nombre"], fecha,
                                                  c["prediccion"], c["umbral"])
    contexto = {
        "seccion": "inicio",
        "resultado": resultado,
        "es_operativo": fecha == servicio.fecha_pronostico,
        "fecha_min": servicio.primera_fecha.isoformat(),
        "fecha_max": servicio.fecha_pronostico.isoformat(),
        "fecha_valor": fecha.isoformat(),
        "ultima_fecha": servicio.ultima_fecha,
        "mensaje": mensaje,
    }
    return render(request, "dashboard/inicio.html", contexto)


def historial(request):
    servicio, error = _servicio_o_error(request)
    if error:
        return error
    prueba = servicio.prueba
    graficas = [{"nombre": obj, "img": graficos.historial(prueba, col, obj, servicio.umbral_p90[obj])}
                for obj, col in OBJETIVOS.items()]
    contexto = {
        "seccion": "historial",
        "metricas": servicio.metricas_prueba(),
        "graficas": graficas,
        "desde": prueba["date"].min().date(),
        "hasta": prueba["date"].max().date(),
        "umbral_pm25": servicio.umbral_p90["PM2.5"],
        "umbral_pm10": servicio.umbral_p90["PM10"],
    }
    return render(request, "dashboard/historial.html", contexto)


def explicabilidad(request):
    servicio, error = _servicio_o_error(request)
    if error:
        return error
    bloques = []
    for obj, col in OBJETIVOS.items():
        tabla = servicio.importancia_global(col)
        grupos = servicio.importancia_grupos(col)
        bloques.append({
            "nombre": obj,
            "img_importancia": graficos.importancia(tabla, obj),
            "img_grupos": graficos.grupos(grupos, obj),
            "tabla": tabla.head(10).to_dict("records"),
        })
    return render(request, "dashboard/explicabilidad.html", {"seccion": "explicabilidad", "bloques": bloques})


def acerca(request):
    servicio, error = _servicio_o_error(request)
    if error:
        return error
    return render(request, "dashboard/acerca.html", {
        "seccion": "acerca",
        "servicio": servicio,
        "meta": metadatos(),
        "n_dias": len(servicio.datos),
    })
