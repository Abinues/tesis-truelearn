# TrueLearn — Visualización del perfil del estudiante y feedback explícito

Proyecto Final de Carrera — Ingeniería en Informática, FPUNA.

Este árbol de archivos es la versión corregida del proyecto: ver
[`CAMBIOS.md`](./CAMBIOS.md) para el detalle completo de cada fix, su
justificación y su test de regresión asociado.

## Estructura

```
tesis/
├── dashboard.py          # Entry point de Streamlit
├── shared.py              # CSS, helpers HTTP y popup de corrección
├── pages/                 # Una página de Streamlit por pantalla
├── logic/
│   ├── api_backend.py     # Backend Flask (puerto 5000)
│   ├── entrenador.py      # Entrena INKClassifier sobre PEEK
│   ├── dominio.py         # Única fuente de verdad: rangos/umbrales/%
│   ├── peek_loader.py     # Carga centralizada y thread-safe de PEEK
│   ├── modelo_utils.py    # Utilidades sobre el classifier de TrueLearn
│   ├── extractor_peek.py
│   ├── preparar_datos.py
│   └── gestor_perfil.py
├── tests/                 # 94 tests automatizados (pytest + AppTest)
├── data/                  # JSONs de perfiles (se genera al entrenar)
├── models/                # .pkl de modelos entrenados (se genera al entrenar)
├── requirements.txt
├── pytest.ini
└── CAMBIOS.md
```

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Para desarrollo (tests, linter, type-checker, cobertura):

```bash
pip install -r requirements-dev.txt
```

Si `pip install truelearn` falla en Python 3.12+ (ver nota en
`CAMBIOS.md`, sección final), usar:

```bash
pip install --no-deps truelearn
pip install trueskill mpmath
```

## Correr los tests (165 tests; 146 corren incluso sin truelearn instalado)

```bash
pytest -v
```

## Calidad de código

```bash
ruff check .                                              # linter
mypy logic/dominio.py logic/modelo_utils.py logic/entrenador.py logic/peek_loader.py
pytest --cov --cov-report=term-missing                    # cobertura (97.7%)
```

## Entrenar los modelos (requiere descargar PEEK, puede tardar)

```bash
python -m logic.entrenador
```

## Levantar el sistema completo

En una terminal:

```bash
python logic/api_backend.py
```

En otra terminal:

```bash
streamlit run dashboard.py
```

Por defecto el backend corre con `debug=False`. Para depurar localmente:

```bash
TRUELEARN_DEBUG=1 python logic/api_backend.py
```

**No** activar `TRUELEARN_DEBUG` si el puerto va a quedar accesible más
allá de `localhost`: el modo debug de Flask permite ejecución de código
arbitrario desde el navegador ante cualquier traceback.
