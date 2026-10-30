"""Pruebas automáticas: ejecutar con  python manage.py test dashboard"""
from datetime import date, timedelta
from unittest import skipUnless

import numpy as np
import pandas as pd
from django.test import SimpleTestCase
from django.urls import reverse

from .services.ml import (CARPETA_MODELOS, FEATURES_A, RUTA_DATOS, categoria_ica, construir_variables,
                          depurar_datos)

HAY_MODELOS = any(CARPETA_MODELOS.glob("modelo_pm2_5*")) and any(CARPETA_MODELOS.glob("modelo_pm10*"))


class PreparacionDatosTests(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.datos = depurar_datos(pd.read_csv(RUTA_DATOS))

    def test_depuracion_igual_al_notebook(self):
        self.assertEqual(len(self.datos), 1296)
        self.assertEqual(self.datos["date"].min(), pd.Timestamp("2022-08-03"))
        self.assertEqual(int(self.datos.isnull().sum().sum()), 0)
        self.assertEqual(int((self.datos["nitrogen_dioxide"] == 0).sum()), 0)
        self.assertEqual(int((self.datos["uv_index"] == 0).sum()), 0)

    def test_variables_sin_fuga_de_informacion(self):
        m = construir_variables(self.datos).iloc[2:].reset_index(drop=True)
        self.assertEqual(len(m), 1294)
        self.assertEqual(len(FEATURES_A), 21)
        # el rezago 1 de cada día es exactamente el valor del día anterior
        np.testing.assert_allclose(m["pm2_5_lag1"].iloc[1:].to_numpy(), m["pm2_5"].iloc[:-1].to_numpy())
        # ninguna variable de entrada es una medición del mismo día
        for v in FEATURES_A:
            self.assertNotIn(v, ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
                                 "ozone", "aerosol_optical_depth", "dust", "uv_index", "us_aqi", "european_aqi"])

    def test_categorias_ica(self):
        self.assertEqual(categoria_ica("pm2_5", 10)["nombre"], "Buena")
        self.assertEqual(categoria_ica("pm2_5", 30)["nombre"], "Aceptable")
        self.assertEqual(categoria_ica("pm2_5", 40)["nombre"], "Dañina a la salud de grupos sensibles")
        self.assertEqual(categoria_ica("pm10", 60)["nombre"], "Aceptable")


@skipUnless(HAY_MODELOS, "Copie los modelos exportados en dashboard/ml_models/ para ejecutar estas pruebas")
class VistasTests(SimpleTestCase):
    def test_paginas_responden(self):
        for nombre in ("inicio", "historial", "explicabilidad", "acerca"):
            r = self.client.get(reverse(f"dashboard:{nombre}"))
            self.assertEqual(r.status_code, 200, nombre)

    def test_pronostico_historico_y_aditividad(self):
        from .services.ml import obtener_servicio
        s = obtener_servicio()
        res = s.pronosticar(date(2025, 12, 26))
        for c in res["contaminantes"]:
            self.assertAlmostEqual(c["base"] + c["suma_aportes"], c["prediccion"], places=3)
            self.assertIsNotNone(c["observado"])
        manana = s.pronosticar(s.fecha_pronostico)
        self.assertIsNone(manana["contaminantes"][0]["observado"])

    def test_fecha_fuera_de_rango(self):
        r = self.client.get(reverse("dashboard:inicio"), {"fecha": "2030-01-01"})
        self.assertContains(r, "La fecha debe estar entre")
