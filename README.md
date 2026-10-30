# Pronóstico explicable de PM2.5 y PM10 — Valle de Aburrá

Aplicación web en Django del trabajo de grado *Modelo predictivo y explicable de contaminantes atmosféricos en Medellín y su área metropolitana*. Presenta el pronóstico a 24 horas de PM2.5 y PM10 generado por los modelos XGBoost seleccionados en el Capítulo 2 y sus explicaciones TreeSHAP.

## Páginas

| Ruta | Contenido |
|---|---|
| `/` | Pronóstico para el día siguiente al último dato (o para cualquier fecha histórica), categoría ICA, indicador de día crítico y explicación local TreeSHAP |
| `/historial/` | Métricas en el periodo de prueba frente a la persistencia y series observadas vs. pronosticadas |
| `/explicabilidad/` | Importancia global de variables (TreeSHAP) e importancia por grupos |
| `/acerca/` | Datos, modelo, variables, tabla del ICA (Resolución 2254 de 2017) y limitaciones |

## Estructura

```
dashboard/
├── data/air_quality_historical.csv   ← dataset original (Open-Meteo)
├── ml_models/                        ← modelos exportados desde el notebook
│   ├── modelo_pm2_5.json
│   ├── modelo_pm10.json
│   └── metadatos_modelos.json        (opcional)
├── services/ml.py                    ← limpieza, variables, predicción y TreeSHAP (igual al notebook)
├── services/graficos.py              ← gráficos con matplotlib
├── templates/dashboard/              ← páginas HTML
├── static/dashboard/estilos.css
└── tests.py                          ← pruebas automáticas
```

## Instalación (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Abrir <http://127.0.0.1:8000/>.

## Modelos

La aplicación carga los modelos en el **formato nativo de XGBoost (.json)**, que no depende de la versión de scikit-learn ni de pandas. En el notebook `E2_Capitulo2`, sección 11, agregue antes de `shutil.make_archive(...)`:

```python
for obj, col in OBJETIVOS.items():
    mod = MEJOR[(esc_a, obj)]
    if mod == 'XGBoost':
        modelos[(esc_a, obj, mod)].save_model(os.path.join(CARPETA_MODELOS, f'modelo_{col}.json'))
```

Copie `modelo_pm2_5.json`, `modelo_pm10.json` y `metadatos_modelos.json` del `.zip` descargado a `dashboard/ml_models/`. Si solo dispone de los archivos `.joblib`, la aplicación también los carga, siempre que las versiones de las librerías coincidan con las de Colab.

## Actualizar los datos

Reemplace `dashboard/data/air_quality_historical.csv` por una versión más reciente descargada de la misma fuente, con las mismas columnas, y reinicie el servidor. El pronóstico se calculará para el día siguiente al último registro.

## Pruebas

```bash
python manage.py test dashboard
```

Verifican que la depuración y las variables sean idénticas a las del notebook (1.296 registros, 1.294 filas de modelado, 21 variables sin fuga de información), las categorías del ICA, que las páginas respondan y que se cumpla la propiedad aditiva de SHAP.

## Notas

- Las explicaciones TreeSHAP se calculan con `predict(..., pred_contribs=True)` de XGBoost, que implementa el mismo algoritmo exacto que `shap.TreeExplainer`; por eso la aplicación no requiere la librería `shap`.
- Herramienta de apoyo a la decisión: no reemplaza los boletines oficiales del SIATA ni del Área Metropolitana del Valle de Aburrá.
