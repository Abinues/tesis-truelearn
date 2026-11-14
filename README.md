# TrueLearn — Visualización del perfil del estudiante y corrección del modelo en tiempo real

Proyecto Final de Carrera — Ingeniería en Informática, FPUNA (2026).

Tablero en Streamlit y backend en Flask sobre la biblioteca
[TrueLearn](https://github.com/sahanbull/TrueLearn) 1.1.0. Muestra el perfil
del estudiante (conocimiento e interés por tema) y permite que el estudiante
corrija las estimaciones del modelo; el modelo se actualiza al momento con una
regla bayesiana de corrección.

## Estructura

```
.
├── dashboard.py              # Punto de entrada de Streamlit (página de inicio)
├── shared.py                 # CSS, funciones HTTP y ventana de corrección
├── ajuste_letras.py          # Tamaños de letra compartidos por las páginas
├── pages/                    # Una página de Streamlit por pantalla
│   ├── login.py              #   elección del estudiante
│   ├── resumen.py            #   resumen del perfil
│   ├── temas.py              #   mapa de temas
│   ├── evolucion.py          #   evolución temporal
│   ├── prediccion.py         #   predicción y correcciones al modelo
│   ├── historial.py          #   historial de correcciones
│   └── pesos_ink.py          #   pesos del INKClassifier
├── logic/
│   ├── api_backend.py        # Backend Flask (puerto 5000)
│   ├── administracion.py     # Restablecer todo y correcciones ficticias
│   ├── entrenador.py         # Entrena el INKClassifier sobre PEEK
│   ├── dominio.py            # Rangos, umbrales y escalas del dominio
│   ├── peek_loader.py        # Carga del dataset PEEK
│   ├── modelo_utils.py       # Utilidades sobre el clasificador de TrueLearn
│   ├── extractor_peek.py
│   ├── preparar_datos.py
│   └── gestor_perfil.py
├── tests/                    # Pruebas automatizadas (pytest, AppTest y e2e)
├── data/                     # Perfiles JSON (se generan al entrenar)
├── models/                   # Modelos entrenados .pkl (se generan al entrenar)
├── analisis_sensibilidad/    # Resultados de los scripts de análisis (CSV/JSON)
├── requirements.txt          # Dependencias de ejecución (versiones exactas)
├── requirements-dev.txt      # Dependencias de desarrollo
└── pyproject.toml            # Configuración del proyecto
```

### Scripts de análisis y calibración

Estos scripts no forman parte de la aplicación. Respaldan con datos las
decisiones de diseño de `logic/dominio.py` y producen los resultados que usa
la tesis:

| Script | Qué hace |
|---|---|
| `entrenar_gran_escala.py` | Entrena a todos los usuarios del split de entrenamiento de PEEK con al menos un evento |
| `calibrar_dominio.py` | Estadísticas empíricas de la media y la varianza de los componentes de conocimiento |
| `calibrar_pesos.py` | Rangos de los pesos del clasificador con todos los usuarios |
| `calibracion_por_estrato.py` | Muestra que la calibración de la escala no depende del nivel de actividad |
| `calibrar_correcciones.py` | Respalda con datos la magnitud de las correcciones explícitas |
| `sensibilidad_correccion.py` | Cuánto cambia la corrección según el parámetro `EVENTOS_EQUIVALENTES` |
| `analisis_sensibilidad.py` | Respalda dos decisiones de diseño que no tienen fuente en la literatura |
| `analisis_confianza_equipo.py` | Respaldo de la confianza (1 − varianza) × 100 |
| `interes_masas.py` | Diagnóstico de la distribución de los terciles de interés |
| `llenar_correcciones.py` | Llena con correcciones explícitas a los estudiantes ya entrenados |
| `versiones.py` | Imprime las versiones usadas en el proyecto |

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Para desarrollo (pruebas, linter, verificador de tipos, cobertura, navegador
para las pruebas e2e):

```bash
pip install -r requirements-dev.txt
playwright install chromium
```

Si `pip install truelearn` falla (por ejemplo, en Python 3.12 o superior):

```bash
pip install --no-deps truelearn
pip install trueskill mpmath
```

## Datos

El dataset PEEK (Bulathwela et al., 2021) no se incluye: se descarga solo la
primera vez que se entrena (`logic/peek_loader.py`). Su repositorio,
[PEEKC-Dataset](https://github.com/sahanbull/PEEKC-Dataset), lo ofrece
únicamente para investigación no comercial. Los modelos de `models/` y los
perfiles de `data/` son resultados derivados de PEEK y se publican solo con
fines académicos, para reproducir la tesis, bajo esa misma condición de uso no
comercial.

## Licencia

Uso solo para investigación no comercial (ver `LICENSE`), igual que PEEK.

## Entrenar los modelos

Requiere descargar PEEK y puede tardar:

```bash
python -m logic.entrenador
```

## Levantar el sistema completo

Desde la raíz del proyecto y con el entorno virtual activado. En una terminal
(backend, puerto 5000):

```bash
python -m logic.api_backend
```

En otra terminal (tablero, puerto 8501 por defecto):

```bash
streamlit run dashboard.py
```

Por defecto el backend corre con `debug=False`. Para depurar localmente:

```bash
TRUELEARN_DEBUG=1 python -m logic.api_backend          # Linux / macOS
$env:TRUELEARN_DEBUG=1; python -m logic.api_backend    # Windows (PowerShell)
```

**No** activar `TRUELEARN_DEBUG` si el puerto va a quedar accesible más allá
de `localhost`: el modo debug de Flask permite ejecución de código arbitrario
desde el navegador ante cualquier error.

## Pruebas y calidad de código

```bash
pytest -v                                                 # pruebas
pytest --cov --cov-report=term-missing                    # cobertura
ruff check .                                              # linter
mypy logic/dominio.py logic/modelo_utils.py logic/entrenador.py logic/peek_loader.py
```

Algunas pruebas necesitan `truelearn` instalado y las pruebas e2e necesitan
Playwright con Chromium; el resto corre sin ellos.