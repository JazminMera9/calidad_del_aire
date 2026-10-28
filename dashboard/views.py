import base64
import io
import os
import matplotlib
matplotlib.use('Agg')  # Evita errores de interfaz gráfica en el servidor
import matplotlib.pyplot as plt
import pandas as pd
import shap
import xgboost as xgb
from django.conf import settings
from django.shortcuts import render

# 1. Ruta absoluta y carga del modelo
MODEL_PATH = os.path.join(settings.BASE_DIR, 'dashboard', 'ml_models', 'mejor_modelo_pm25.json')
modelo_pm25 = xgb.XGBRegressor()
modelo_pm25.load_model(MODEL_PATH)

# 2. Inicializar el motor de explicabilidad TreeSHAP
explainer_pm25 = shap.TreeExplainer(modelo_pm25)

def clasificar_ica(pm25):
    """Clasificación según los rangos estándar de la Resolución 2254 de 2017."""
    if pm25 <= 12.0:
        return {"estado": "Buena", "color": "#198754", "alerta": "Condiciones óptimas sin riesgo para la salud."}
    elif pm25 <= 37.4:
        return {"estado": "Aceptable / Moderada", "color": "#ffc107", "alerta": "Aceptable, posibles síntomas en población muy sensible."}
    elif pm25 <= 55.4:
        return {"estado": "Dañina para grupos sensibles (Alerta Naranja)", "color": "#fd7e14", "alerta": "Población vulnerable debe evitar actividades al aire libre."}
    else:
        return {"estado": "Dañina para la salud (Alerta Roja)", "color": "#dc3545", "alerta": "Toda la población puede experimentar efectos adversos."}

def prediccion_dashboard(request):
    contexto = {
        'prediccion': None,
        'ica': None,
        'grafico_shap': None
    }

    if request.method == 'POST':
        # Captura de variables
        co = float(request.POST.get('carbon_monoxide', 414.96))
        o3 = float(request.POST.get('ozone', 55.67))
        aod = float(request.POST.get('aerosol_optical_depth', 0.24))

        # Vector de 23 características
        datos_entrada = pd.DataFrame([{
            'carbon_monoxide': co,
            'nitrogen_dioxide': 14.30,
            'sulphur_dioxide': 4.23,
            'ozone': o3,
            'aerosol_optical_depth': aod,
            'dust': 0.16,
            'uv_index': 1.81,
            'pm10_lag1': 17.33,
            'pm10_lag2': 17.33,
            'pm2_5_lag1': 13.64,
            'pm2_5_lag2': 13.64,
            'carbon_monoxide_lag1': 485.11,
            'carbon_monoxide_lag2': 485.11,
            'nitrogen_dioxide_lag1': 14.30,
            'nitrogen_dioxide_lag2': 14.30,
            'sulphur_dioxide_lag1': 4.23,
            'sulphur_dioxide_lag2': 4.23,
            'ozone_lag1': 55.21,
            'ozone_lag2': 55.21,
            'pm10_ma3': 17.33,
            'pm10_delta': 0.0,
            'pm2_5_ma3': 13.64,
            'pm2_5_delta': 0.0
        }])

        # 3. Predicción numérica
        resultado = modelo_pm25.predict(datos_entrada)
        prediccion_val = round(float(resultado[0]), 2)
        contexto['prediccion'] = prediccion_val
        contexto['ica'] = clasificar_ica(prediccion_val)

        # 4. Cálculo de TreeSHAP y generación de gráfico local
        shap_vals = explainer_pm25.shap_values(datos_entrada)[0]
        feature_names = datos_entrada.columns

        # Tomar las 6 variables con mayor impacto absoluto
        df_shap = pd.DataFrame({
            'feature': feature_names,
            'shap': shap_vals
        }).reindex(pd.Series(abs(shap_vals)).sort_values(ascending=False).index).head(6)

        df_shap = df_shap.sort_values(by='shap')  # Para orden en gráfico de barras

        fig, ax = plt.subplots(figsize=(7, 3.5))
        colores = ['#dc3545' if val > 0 else '#0d6efd' for val in df_shap['shap']]
        ax.barh(df_shap['feature'], df_shap['shap'], color=colores)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title("Explicabilidad Local TreeSHAP (Aporte marginal en µg/m³)")
        ax.set_xlabel("Impacto en la predicción (+ Incrementa / - Disminuye)")
        plt.tight_layout()

        # Convertir gráfico a string Base64 para HTML
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        contexto['grafico_shap'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)

    return render(request, 'dashboard/index.html', contexto)