# CAMBIOS.md — Refactorización del sistema TrueLearn (visualización + feedback)

Este documento acompaña una revisión exhaustiva del código entregado para el
Proyecto Final de Carrera *"Visualización del perfil del estudiante y
corrección del modelo en tiempo real mediante feedback en TrueLearn"*.
Cada punto está referenciado desde comentarios `# FIX (ver CAMBIOS.md,
punto N)` directamente en el código, para que se pueda rastrear cada
cambio hasta su justificación y de vuelta.

**Cómo verificar estos cambios:**

```bash
pip install -r requirements-dev.txt   # incluye requirements.txt + ruff/mypy/pytest-cov
pytest -v                             # 165 tests
ruff check .                          # linter, debe quedar limpio
mypy logic/dominio.py logic/modelo_utils.py logic/entrenador.py logic/peek_loader.py
pytest --cov --cov-report=term-missing  # 97.7% de cobertura real
```

La suite corre completa (146 tests, saltando con gracia los 2 archivos
que sí lo requieren) incluso sin `truelearn` instalado — ver punto 26.
Con `truelearn` instalado (`pip install --no-deps truelearn` +
`pip install trueskill mpmath` si falla la instalación completa por la
nota de Python 3.12 al final de este documento), corren los 165.

---

## 1. `login.py`: `NameError` que rompía el entrenamiento desde la interfaz

**Archivo:** `pages/login.py`

La sección "⚙️ Administración del modelo" usa `BACKEND` para hacer poll a
`/log_entrenamiento` y disparar `POST /entrenar`, pero el import era:

```python
from shared import inject_css, get_usuarios, TEAL
```

`BACKEND` nunca se importaba → `NameError` apenas se abría ese expander.
La funcionalidad de "Entrenar modelos desde PEEK" descrita en la Fase 4
de la propuesta **no corría** tal como estaba entregada.

**Fix:** se agregó `BACKEND` al import.

**Relacionado:** el mismo tipo de bug de nombres existía en
`gestor_perfil.py` (punto 1 duplicado en el código, mismo número por
tratarse de la misma clase de error): buscaba `modelo_u_{uid}.pkl` con
"modelo", cuando `entrenador.py` y `api_backend.py` guardan y leen
`model_u_{uid}.pkl` con "model". Nunca encontraba ningún modelo.

---

## 2. Desfase metodológico entre `session_history` y el modelo persistido

**Archivo:** `logic/entrenador.py`

La versión original entrenaba `NUM_EPOCHS = 50` pasadas completas sobre
la misma secuencia histórica de eventos de un usuario, pero el snapshot
de `session_history` (lo que alimenta el gráfico de "Evolución del
Aprendizaje") **solo se capturaba en la época 0**. El modelo que
finalmente se persistía en el `.pkl` —y con el que se calculan
predicciones y se aplican correcciones— era el de **después de las 50
épocas**.

Consecuencia: el último punto de la curva de evolución que ve el
estudiante nunca coincidía con el "estado actual" que mostraba el resto
del dashboard. Dos fuentes de verdad distintas y contradictorias, en un
proyecto cuyo objetivo central es la *transparencia* del modelo.

Además, repetir 50 veces la misma secuencia de eventos no corresponde a
ningún escenario real de un aprendiz (TrueLearn está diseñado para
actualizarse online, evento a evento, una sola vez). Los comentarios
`# era 10`, `# era 25` en el código original sugieren que estos números
se ajustaron por prueba y error, sin una justificación documentada.

**Fix:**
- El snapshot de cada "sesión" ahora se toma en la **última** época
  (`epoch == num_epochs - 1`), no en la primera. Así, sea cual sea
  `NUM_EPOCHS`, el último punto de `session_history` **siempre** coincide
  con el estado final del modelo persistido.
- `NUM_EPOCHS` pasa a valer **1** por defecto (una sola pasada = fiel al
  comportamiento online real de TrueLearn), con la razón documentada en
  el propio código. Es una decisión que corresponde justificar/ajustar
  en la memoria técnica; el código deja el trade-off explícito para que
  esa decisión sea informada.

**Nota:** este es un cambio metodológico, no solo un bug de programación.
Se optó por el valor más defendible por defecto, pero es una decisión
que el tutor y la autora deberían confirmar antes de la entrega final.

---

## 3. Rangos inconsistentes entre backend y frontend (porcentajes inválidos)

**Archivos:** `logic/api_backend.py`, `logic/dominio.py` (nuevo), y todas
las páginas de `pages/`.

`aplicar_correccion_directa` clampeaba el `mean` de conocimiento a
`[-1.0, 1.0]` y el de interés a `[-1.0, 2.0]`. El frontend, en cambio,
normalizaba a porcentaje asumiendo `mean ∈ [-0.5, 1.5]` (conocimiento,
`(mean+0.5)/2*100`) y `mean ∈ [-0.5, 2.5]` (interés, `(mean+0.5)/3*100`).
Con un `mean = -1.0` esa fórmula da `-25%`: un porcentaje de conocimiento
**negativo**, sin sentido de dominio, que además no siempre estaba
clampeado a `[0, 100]` en todas las páginas (sí en `resumen.py`, no en
`temas.py`).

**Fix:** se creó `logic/dominio.py`, un módulo puro (sin dependencias de
Streamlit/Flask/TrueLearn) que define:
- `KNOW_MIN/MAX`, `INT_MIN/MAX`, `VAR_MIN/MAX` — los rangos válidos,
  documentados y justificados en un único lugar.
- `clamp_conocimiento/interes/variance` — usadas por el backend al
  aplicar correcciones.
- `pct_conocimiento/interes/certeza` — usadas por *todo* el frontend,
  siempre acotadas a `[0, 100]`.

Backend y frontend importan de acá. Es estructuralmente imposible que
vuelvan a divergir, porque ya no hay dos definiciones del mismo rango.

**Test de regresión:** `tests/test_dominio.py::test_clamps_y_normalizacion_son_consistentes`
y `tests/test_correccion.py::test_conocimiento_nunca_sale_del_rango_aunque_se_repita_muchas_veces`
(aplica la misma corrección 200 veces seguidas —peor caso— y verifica que
el resultado siempre cae en `[0, 100]`).

---

## 4. Umbrales de clasificación duplicados en 3 archivos

**Archivos:** `shared.py`, `pages/temas.py`, `pages/pesos_ink.py`.

Los cortes "Alto ≥0.7 / Medio 0.4–0.7 / Básico 0.1–0.4" estaban
hardcodeados de forma independiente en `shared.nivel_label()`, en
`temas.py` (variables `altos/medios/basicos`) y en `pesos_ink.py`
(función `cat_idx`). Cambiar un umbral requería tocar 3 archivos y
confiar en no haberse olvidado de ninguno.

**Fix:** los umbrales (`UMBRAL_ALTO`, `UMBRAL_MEDIO`, `UMBRAL_BASICO`,
`UMBRAL_VAR_*`, `UMBRAL_INTERES_*`) y las funciones que los usan
(`nivel_label`, `var_label`, `interes_label`, `categoria_dominio`) viven
ahora únicamente en `logic/dominio.py`. `shared.py` las re-exporta con
los mismos nombres para no romper los `from shared import nivel_label,
...` existentes en las páginas.

**Test:** `tests/test_dominio.py::test_categoria_dominio_coincide_con_nivel_label`.

---

## 5. `extraer_pesos_reales` duplicada entre `entrenador.py` y `api_backend.py`

**Archivo nuevo:** `logic/modelo_utils.py`.

La misma función (letra por letra) vivía copiada en ambos archivos.
También `resolver_nombre_tema`. Se consolidaron en un único módulo que
ambos importan.

---

## 6. `os.chdir()` sin `try/finally` + condición de carrera potencial

**Archivos:** `logic/extractor_peek.py`, `logic/entrenador.py`,
`logic/preparar_datos.py` (los tres reimplementaban el mismo patrón).
**Archivo nuevo:** `logic/peek_loader.py`.

Tres módulos distintos repetían "cambiar cwd → `load_peek_dataset()` →
volver al cwd anterior". En `extractor_peek.py` y `entrenador.py` esto
**no** estaba envuelto en `try/finally`: si `load_peek_dataset()` lanzaba
una excepción, `os.chdir()` nunca se revertía, y el proceso quedaba
"pegado" en la carpeta `data/` para toda operación posterior basada en
rutas relativas.

Además, `os.chdir()` cambia el directorio de trabajo de **todo el
proceso**, no solo de la función que lo llama. El backend Flask lanza el
entrenamiento en un hilo (`/entrenar`) mientras sigue atendiendo otras
requests en el mismo proceso — si dos operaciones dependientes del cwd
corrieran "al mismo tiempo", una podría resolver rutas relativas mal.

**Fix:** `logic/peek_loader.cargar_dataset_peek()` centraliza la carga
con `try/finally` (el cwd SIEMPRE vuelve a su valor original, incluso si
falla) y un `threading.Lock` (solo una carga puede estar cambiando el
cwd del proceso a la vez). Los tres módulos originales ahora son
wrappers finos sobre esta función.

---

## 7. `app.run(debug=True)` en el backend Flask

**Archivo:** `logic/api_backend.py`.

El modo debug de Flask expone el debugger interactivo de Werkzeug, que
permite ejecución de código arbitrario si alguien llega a un traceback
desde el navegador. Sin autenticación en ningún endpoint, dejar esto
fijo en `True` es una mala práctica incluso para un prototipo académico.

**Fix:** controlado por la variable de entorno `TRUELEARN_DEBUG` (por
defecto `"0"`, es decir, debug **desactivado**). Se imprime una
advertencia explícita si se activa.

---

## 8. Corrección "silenciosa" sobre un tema inexistente

**Archivo:** `logic/api_backend.py`.

Si `topic_id` no existía en el modelo de conocimiento/interés del
usuario, `aplicar_correccion_directa` no hacía nada (el `if kc_know:`
simplemente no entraba), pero el endpoint igual respondía `200 OK` con
el mensaje "Feedback aplicado", dando la falsa impresión de que la
corrección se había aplicado cuando en realidad el modelo no cambió.

**Fix:** se agregó `TopicNoEncontradoError`. Si el tema no existe en el
modelo correspondiente, se lanza la excepción y el endpoint responde
`404` con un mensaje explícito.

**Test:** `tests/test_correccion.py::test_topic_inexistente_en_knowledge_lanza_error`,
`tests/test_api_backend.py::test_feedback_topic_inexistente_404`.

---

## 9. Escritura no atómica de `.pkl`/`.json` + sin manejo de errores de deserialización

**Archivo:** `logic/api_backend.py`.

`guardar_modelo`/`guardar_perfil` escribían directamente sobre el
archivo final; si el proceso se interrumpía a mitad de la escritura
(corte de luz, `kill -9`, etc.), el archivo quedaba corrupto y
truncado. `cargar_modelo` tampoco manejaba una posible excepción de
`pickle.load` sobre un archivo corrupto.

**Fix:** escritura atómica (`escribir a .tmp` → `os.replace()`, que en
POSIX y Windows modernos es atómico) y `try/except` alrededor de
`pickle.load` con log del error en vez de un traceback crudo hacia el
cliente HTTP.

---

## 10. Riesgo de `TypeError` al serializar tipos de NumPy/pandas a JSON

**Archivos:** `pages/temas.py`, `shared.py`, `logic/api_backend.py`.

En `temas.py`, el botón "✏️ Editar" del mapa de conocimiento construye
`st.session_state["popup_topic"] = row.to_dict()` a partir de una fila de
`DataFrame`. Eso deja `topic_id`/`mean`/`variance`/`interest_mean` como
`numpy.int64`/`numpy.float64`. Cuando `shared.post_feedback()` hace
`requests.post(..., json=body)`, la serialización JSON estándar **no
sabe convertir tipos de NumPy** y lanza `TypeError`.

**Fix (defensa en profundidad, en tres capas):**
1. En `temas.py`, se castea explícitamente a `int()`/`float()` en el
   origen, antes de guardar en `session_state`.
2. En `shared.post_feedback()`, se castea de nuevo por las dudas
   (defensa adicional para cualquier otro llamador futuro).
3. En `api_backend.py`, `_coerce_topic_id()` normaliza el `topic_id`
   recibido en el body a `int` cuando es posible.

**Test:** `tests/test_api_backend.py::test_feedback_topic_id_como_string_numerico_se_normaliza`.

---

## 11. Condición de carrera (TOCTOU) en `POST /entrenar`

**Archivo:** `logic/api_backend.py`.

El patrón `if _training_active: ... else: _training_active = True` no
es atómico. Si dos `POST /entrenar` llegaban casi simultáneamente, ambos
podían leer `_training_active = False` antes de que ninguno lo pusiera
en `True`, arrancando dos entrenamientos en paralelo (dos hilos
escribiendo los mismos `.pkl`/`.json`).

**Fix:** toda lectura/escritura de `_training_active`/`_training_log`/
`_training_done` está protegida por un `threading.Lock`.

**Test:** `tests/test_api_backend.py::test_entrenar_rechaza_si_ya_hay_uno_activo`.

---

## 12. La columna "Nivel/Valor al corregir" mostraba siempre el mismo número

**Archivos:** `pages/prediccion.py`, `pages/historial.py`,
`logic/api_backend.py`.

**Encontrado durante esta revisión** (no estaba en la lista original de
hallazgos). `post_feedback()` nunca envía el campo `valor` desde el flujo
interactivo del popup (solo lo usaría una corrección de "valor manual"
hecha directamente contra la API). Como consecuencia:

- En `prediccion.py`, la columna "Nivel al corregir (%)" se calculaba
  como `(fb["valor"].fillna(0) + 0.5)/2*100`, que con `valor` siempre
  `None` → `0`, daba **25% para absolutamente todas las correcciones**,
  sin importar si fueron "sé más", "sé menos", "me interesa" o "no me
  interesa".
- En `historial.py`, la columna "Valor" mostraba `NaN`/vacío para
  prácticamente todas las filas.

**Fix:** `api_backend.py` ahora guarda en cada registro de
`explicit_feedback` el valor **resultante real** tras aplicar la
corrección (`mean_resultante`, `interest_mean_resultante`), leído del
perfil ya reconstruido. `prediccion.py` y `historial.py` usan ese valor
real (eligiendo conocimiento o interés según el tipo de corrección) en
vez del campo `valor`, casi siempre vacío.

**Test:** `tests/test_api_backend.py::test_feedback_flujo_completo_aplica_correccion_y_guarda`
verifica explícitamente que `mean_resultante` queda guardado y coincide
con el valor real del KC tras la corrección.

---

## 13. `pages/historial.py` nunca estaba registrada en la navegación

**Archivo:** `dashboard.py`.

**Encontrado durante esta revisión.** La página "Historial de
correcciones" existe, funciona y está bien implementada, pero
`st.navigation([...])` en `dashboard.py` nunca la incluía en la lista.
Era **inalcanzable** desde la interfaz: no había ningún botón, link ni
forma de llegar a ella navegando la app. Es justamente el *"registro
auditado de todas las correcciones aplicadas por el estudiante,
accesible desde la interfaz"* que la propuesta lista como resultado
esperado (sección 2.4).

**Fix:** se agregó `st.Page("pages/historial.py", ...)` a la lista de
navegación.

---

## 14. Hack de JavaScript sobre el DOM interno de Streamlit (deuda técnica documentada, no reescrita a ciegas)

**Archivo:** `dashboard.py`.

El bloque que colapsa el sidebar al volver al login usa selectores
`data-testid` internos de Streamlit que no son parte de su API pública y
pueden cambiar sin aviso entre versiones, rompiendo el comportamiento en
silencio (sin lanzar ningún error).

**Decisión:** no se reescribió a ciegas, porque hacerlo sin un entorno
Streamlit real para validar el resultado arriesgaba introducir una
regresión sin poder comprobarla. Se documentó el riesgo en el propio
código como deuda técnica conocida, con la recomendación de
reemplazarlo por el mecanismo oficial de Streamlit para controlar el
sidebar en cuanto esté disponible en la versión instalada del proyecto.

---

## 15. `extraer_pesos_reales` dependía de un atributo privado de TrueLearn

**Archivo:** `logic/modelo_utils.py`.

**Encontrado validando el código contra la librería real** (se instaló
`truelearn==1.1.0` para esta revisión — ver nota de reproducibilidad más
abajo). La función original leía `classifier._learner_meta_weights`, un
atributo interno (con guión bajo) no documentado como parte de la
interfaz pública de TrueLearn.

Se confirmó empíricamente que `classifier.get_learner_model()` devuelve
una tupla de **tres** elementos —no dos, como el resto del código asume
al hacer `models[0]`/`models[1]` para conocimiento e interés— y que el
tercer elemento (`models[2]`) es exactamente el mismo objeto
`LearnerMetaWeights`.

**Fix:** `extraer_pesos_reales` ahora usa `classifier.get_learner_model()[2]`,
la vía pública y documentada, en vez del atributo privado, que podría
renombrarse o desaparecer en una futura versión de la librería sin
ningún aviso de compatibilidad.

**Test:** `tests/test_modelo_utils.py` (con fakes) y
`tests/test_truelearn_integration.py::test_get_learner_model_devuelve_tres_elementos`
(contra la librería real).

## 16. `pages/prediccion.py`: `NameError: name 'df_fb' is not defined` con perfiles sin correcciones aún

**Archivo:** `pages/prediccion.py`.

**Bug real preexistente en el archivo original, encontrado por la
autora al correr la app después del entrenamiento** (no introducido por
esta refactorización, y tampoco atrapado en la primera pasada de
revisión).

`df_fb` se creaba **solo** dentro del `else` de `if not feedbacks:`:

```python
if not feedbacks:
    st.info("Todavía no se aplicaron correcciones sobre este perfil.")
else:
    df_fb = pd.DataFrame(feedbacks)
    ...
```

Pero se usaba **sin esa guarda** más abajo, tanto en la sección
"Correcciones por fecha" (`for _, row in df_fb.iterrows():`) como en
"Historial completo" (`df_show = df_fb.copy()`). Con
`explicit_feedback == []` —el estado de **cualquier perfil recién
entrenado, antes de la primera corrección**— `df_fb` nunca se creaba, y
la página tiraba `NameError` apenas se abría. Es decir: la pantalla de
"Predicción y correcciones" era inutilizable hasta que el usuario
aplicara al menos una corrección desde *otra* pantalla (Mapa de Temas) —
y el flujo obvio para un usuario nuevo es exactamente al revés.

**Fix:**
- `df_fb = pd.DataFrame(feedbacks)` ahora se ejecuta siempre,
  incondicionalmente (puede quedar un DataFrame vacío, sin columnas).
- Las columnas derivadas (`"Fecha"`, `"idx"`) solo se agregan si
  `not df_fb.empty` (un DataFrame vacío no tiene columna `"timestamp"`
  para aplicarles `fmt_ts`).
- La sección "Historial completo" (`col4`) ahora chequea
  `df_fb.empty` explícitamente y muestra un `st.info(...)` en vez de
  intentar leer columnas (`tipo`, `Fecha`, `topic_name`) que no existen
  en un DataFrame vacío.

**Por qué se escapó en la primera revisión:** toda la suite de tests
anterior (72 tests) cubría la capa de lógica (`logic/dominio.py`,
`logic/api_backend.py`) con fakes, pero **ningún test ejecutaba las
páginas de Streamlit de verdad**. Un bug de "variable definida en una
rama pero usada afuera" es invisible a una lectura de código atenta si
no se traza mentalmente el caso `feedbacks == []` contra cada uso
posterior de `df_fb` — exactamente el tipo de error que un test de
ejecución real atrapa en segundos y una revisión manual puede pasar por
alto.

**Fix del proceso, no solo del bug:** se agregó
`tests/test_pages_smoke.py`, que usa `streamlit.testing.v1.AppTest` para
cargar las 6 páginas de perfil de verdad (Streamlit ejecutando el script
real) contra tres escenarios de datos (perfil recién entrenado sin
feedback, perfil sin `knowledge_components`, perfil con historial y
feedback completos) y sin ningún aprendiz seleccionado. Se verificó que
este test específicamente detecta la regresión: reintroduciendo el bug
original, fallan exactamente los 2 casos sin feedback; con el fix,
pasan los 94 tests de la suite completa.

**Cobertura que sigue faltando:** `login.py` y `dashboard.py` no están
cubiertos por `test_pages_smoke.py` porque dependen de `requests.get`
reales hacia el backend Flask (`get_usuarios`/`get_perfil`) y de
`st.navigation`. Cubrirlos requeriría mockear `requests` o levantar un
backend de prueba — queda como mejora pendiente.


Durante esta revisión se intentó instalar `truelearn` para validar el
código contra la API real. `pip install truelearn` completo **falla en
Python 3.12** porque una dependencia opcional de graficado,
`wordcloud==1.9.2`, no compila (`versioneer.py` usa
`configparser.SafeConfigParser`, eliminado en Python 3.12).

Instalación mínima que sí funciona (suficiente para entrenar/usar
`INKClassifier`, no para las utilidades de graficado propias de
TrueLearn, que este proyecto no usa):

```bash
pip install --no-deps truelearn
pip install trueskill mpmath
```

Recomendación para la memoria técnica: documentar la versión exacta de
Python usada en el entorno de desarrollo/evaluación, y si es 3.12+,
dejar constancia de este workaround (o fijar una versión de Python más
vieja compatible con la instalación completa de `truelearn`).

---

## 17. Recalibración de `KNOW_MIN/MAX` e `INT_MIN/MAX` con datos reales

**Archivo:** `logic/dominio.py`. **Script nuevo:** `calibrar_dominio.py`.

Los límites originales (`KNOW_MIN=-0.5, KNOW_MAX=1.5`, `INT_MIN=-0.5,
INT_MAX=2.5`, ver punto 3) nunca fueron validados contra el
comportamiento real de un `INKClassifier` entrenado, ni contra la
literatura de TrueLearn. Se investigaron ambas cosas:

**Literatura:** TrueLearn está construido sobre TrueSkill, que modela la
habilidad de un aprendiz como el `mean` de una variable gaussiana sobre
toda la recta real (θ ∈ ℝ) — sin ningún límite universal, por diseño
explícito del modelo (Bulathwela et al., 2020). La única cantidad que la
librería expone con semántica nativamente acotada a [0, 1] es
`predict_proba()` (ya usada en este proyecto solo para el antes/después
del popup de corrección), no el `mean` crudo que se usa en el resto del
dashboard. Conclusión: no existe ningún rango "oficial" para citar — la
calibración empírica no es un parche, es el método correcto dado cómo
está diseñado el modelo subyacente.

**Datos reales:** se corrió `calibrar_dominio.py` sobre 25 usuarios
entrenados con el dataset PEEK real (3856 knowledge components en
total). Resultado:

| Variable | Rango asumido (original) | Rango real (percentiles 1/99) |
|---|---|---|
| knowledge.mean | -0.5 a 1.5 | **-0.473 a +0.438** |
| interest_mean  | -0.5 a 2.5 | +0.001 a **+3.032** |

Con el rango original, el estudiante con el `mean` de conocimiento más
alto de los 25 (0.877) mostraba apenas **68.8%** — ni el caso más
enganchado real llegaba a acercarse al 100%. Del lado del interés, el
`mean` más alto observado (4.699) correspondía a **173.3%**, muy por
encima de 100%, silenciosamente clampeado sin que se notara en la
interfaz.

**Fix:** `KNOW_MIN, KNOW_MAX = -0.473, 0.438` (percentiles 1/99 reales).
`INT_MAX = 3.032` (percentil 99 real). `INT_MIN` se mantuvo en `-0.5`
—no en `0.0`, aunque el mínimo real observado en entrenamiento
automático fue exactamente `0.0`— a propósito: la corrección manual "no
me interesa" resta del `mean`, y necesita margen por debajo del piso
natural para que el estudiante vea un cambio real al aplicarla; con un
piso en `0.0` exacto, corregir un tema que ya está en el mínimo natural
quedaría clampeado sin ningún efecto visible, rompiendo la utilidad del
mecanismo de feedback explícito que es el corazón de la propuesta.

Se usaron percentiles 1/99 en vez de mínimo/máximo absoluto porque el
dataset tenía usuarios con hasta 831 temas distintos (contra un mínimo
de 11): un único usuario con muchísimas interacciones puede dominar el
extremo absoluto y comprimir la escala visual del resto de la población.

**`VAR_MIN`/`VAR_MAX`** no se tocaron: el rango real observado
(percentiles 1/99: 0.0915–0.4809 para conocimiento, 0.1437–0.4991 para
interés) cae cómodamente dentro de los límites existentes (0.01–0.5).

**Cómo se recalibró:** `calibrar_dominio.py` (nuevo, en la raíz del
proyecto) lee todos los `data/visual_profile_*.json` ya entrenados,
calcula min/percentiles/max de `mean` y `variance` (conocimiento e
interés) sin depender de numpy/pandas, imprime un reporte, y guarda el
detalle en `data/calibracion_dominio.json`. No modifica ningún archivo
del proyecto — es de solo lectura sobre `data/`.

**Si se reentrena con una muestra de usuarios distinta** (otro
subconjunto de PEEK, otros umbrales en `seleccionar_usuarios` de
`logic/entrenador.py`), conviene volver a correr `calibrar_dominio.py` y
actualizar estos valores — quedó documentado en el docstring del propio
`logic/dominio.py`.

---

## 18. Los umbrales de clasificación (Alto/Medio/Básico) quedaron huérfanos tras recalibrar el rango

**Archivo:** `logic/dominio.py`.

Efecto colateral directo del punto 17. `UMBRAL_ALTO/MEDIO/BASICO`
(`0.7/0.4/0.1`) estaban definidos como valores de `mean` **crudo**,
calibrados implícitamente para el rango viejo (`KNOW_MAX=1.5`). Al
recalibrar `KNOW_MAX` a `0.438` con datos reales, `UMBRAL_ALTO=0.7`
quedó **por encima del máximo posible**: ningún tema podía clasificar
como "Alto" nunca más. `UMBRAL_BASICO=0.1` equivalía ahora a 62.9%, así
que más de la mitad de los temas reales (mediana real: 56.3%) quedaban
por debajo de "Básico" — es decir, en "Muy bajo", que
`pages/temas.py` directamente **no dibuja** en el sunburst de
"Jerarquía de dominio" (solo arma grupos para Alto/Medio/Básico). El
síntoma reportado fue: "no me está agrupando bien los temas".

**Fix:** `UMBRAL_ALTO/MEDIO/BASICO` (y también `UMBRAL_INTERES_ALTO/MEDIO`,
mismo problema latente aunque no tan extremo) ahora se definen como
**porcentaje de la escala calibrada** (`UMBRAL_ALTO_PCT=70`,
`UMBRAL_MEDIO_PCT=40`, `UMBRAL_BASICO_PCT=10` — preservando la intención
semántica original) y se traducen a `mean` crudo dinámicamente vía
`KNOW_MIN/KNOW_MAX` (o `INT_MIN/INT_MAX`). Si `calibrar_dominio.py` se
vuelve a correr en el futuro con una calibración distinta, estos
umbrales se recalculan solos — no pueden volver a quedar huérfanos.

Verificado contra los percentiles reales: mediana → Medio (56.3%), p95 →
Alto (82.3%), solo el mínimo absoluto → Muy bajo (0%). Distribución
sensata.

**Hallazgo pendiente, NO corregido en este pase:** `UMBRAL_VAR_MUY_SEGURO
/SEGURO/MODERADO` (0.05/0.15/0.30, usados por `var_label()` para las
etiquetas de "Confianza") tienen el mismo problema de fondo, aunque no
tan extremo — no se "rompieron" por la recalibración del punto 17
porque `pct_certeza()` nunca dependió de `VAR_MIN/VAR_MAX`, pero nunca
estuvieron calibrados contra la varianza real que produce el modelo. Con
los datos reales: el percentil 1 de varianza ya es 0.0915 (por encima
de `UMBRAL_VAR_MUY_SEGURO=0.05`) y el percentil 25 ya es 0.37 (por
encima de `UMBRAL_VAR_MODERADO=0.30`). En la práctica, "Muy seguro" y
"Seguro" son casi inalcanzables, y la mayoría de los temas reales caen
en "Incierto" aunque el sistema tenga bastante certeza relativa sobre
ellos. Queda documentado en el propio código y pendiente como tarea
separada.

---

## 19. Los umbrales de "Confianza" (Muy seguro/Seguro/Moderado/Incierto) nunca fueron validados

**Archivo:** `logic/dominio.py`.

Hallazgo relacionado al punto 18, pero NO causado por la recalibración
del punto 17 (`pct_certeza()` nunca dependió de `VAR_MIN`/`VAR_MAX`).
Los valores originales (`UMBRAL_VAR_MUY_SEGURO=0.05`,
`UMBRAL_VAR_SEGURO=0.15`, `UMBRAL_VAR_MODERADO=0.30`) nunca fueron
contrastados contra la varianza real que produce el modelo. Con los
datos reales (25 usuarios, 3856 KCs): el percentil 1 de `variance` ya
era 0.0915 (por encima de `UMBRAL_VAR_MUY_SEGURO`) y el percentil 25 ya
era 0.3708 (por encima de `UMBRAL_VAR_MODERADO`). En la práctica, "Muy
seguro" y "Seguro" eran casi inalcanzables, y la inmensa mayoría de los
temas —sin importar cuánta certeza relativa tuviera el sistema sobre
ellos— caían en "Incierto".

**Fix:** se recalibraron usando los **cuartiles reales** de
`knowledge.variance` (percentiles 25/50/75, ya presentes en el reporte
de `calibrar_dominio.py` usado para el punto 17 — no hizo falta agregar
nada al script): `UMBRAL_VAR_MUY_SEGURO=0.3708` (p25),
`UMBRAL_VAR_SEGURO=0.4272` (mediana), `UMBRAL_VAR_MODERADO=0.4575`
(p75). Cada una de las 4 categorías representa, por construcción, ~25%
de los temas reales. Se usó un reparto parejo (a diferencia del 70/40/10
asimétrico de conocimiento) porque los valores originales eran
arbitrarios desde el principio — no había ninguna intención de diseño
previa que preservar — así que un cuartil parejo es la elección más
fácil de justificar en la memoria.

Verificado contra los percentiles reales: mediana → Seguro, p75 →
Moderado, p95/p99/máximo → Incierto, mínimo/p1/p5/p25 → Muy seguro.
Reparto de ~25% en cada categoría, exactamente como se buscaba.

---

## 20. Los porcentajes "de inicio" (25%, ~17%) quedaron mal tras la recalibración

**Archivos:** `pages/resumen.py`, `pages/temas.py`, `shared.py`.

Antes de cualquier evento, TrueLearn inicializa `knowledge.mean=0.0` e
`interest_mean=0.0`. Estos tres archivos calculaban el delta contra ese
estado neutro usando `25.0` y `16.7` **hardcodeados como números
sueltos** — calculados a mano por el autor original bajo el rango VIEJO
(`KNOW_MAX=1.5` daba `pct_conocimiento(0.0)=25%`; `INT_MAX=2.5` daba
`pct_interes(0.0)≈16.7%`). Al recalibrar `KNOW_MIN/MAX` e `INT_MIN/MAX`
con datos reales (punto 17), esos números de referencia cambiaron
—`pct_conocimiento(0.0)` pasó a ser **51.9%**, `pct_interes(0.0)` pasó a
ser **14.2%**— pero como eran literales sueltos en 3 archivos, no algo
que pasara por `logic/dominio.py`, ningún test los detectó y quedaron
mostrando deltas incorrectos contra un "inicio" que ya no correspondía a
la realidad.

**Fix:** se agregaron `PCT_INICIO_CONOCIMIENTO`, `PCT_INICIO_INTERES` y
`PCT_INICIO_CERTEZA` a `logic/dominio.py`, calculados dinámicamente
(`pct_conocimiento(0.0)`, `pct_interes(0.0)`, `pct_certeza(0.5)`) en vez
de hardcodeados. Los 3 archivos ahora los importan. Si `KNOW_MIN/MAX` o
`INT_MIN/MAX` se recalibran de nuevo en el futuro, estos valores de
referencia se actualizan solos — no pueden volver a quedar huérfanos.

`PCT_INICIO_CERTEZA` siempre da 50% (no depende de la calibración,
porque `pct_certeza()` no usa `VAR_MIN`/`VAR_MAX`), así que ese caso
puntual nunca estuvo mal — se centralizó de todos modos por consistencia
de estilo con los otros dos.

---

## Nota adicional: pesos del modelo (novelty/interest/bias) — revisado, sin evidencia de problema

**Archivo:** `calibrar_dominio.py` (extendido).

Se investigó si `nov_pct/int_pct/bias_pct` y `nov_cert/int_cert/bias_cert`
en `pages/pesos_ink.py` (que normalizan los *pesos* del clasificador —
`novelty_weights`, `interest_weights`, `bias_weights` — no los
Knowledge Components) tenían el mismo tipo de problema de calibración.

A diferencia de los KCs, `data/usuarios_entrenados.json` solo guarda
**un** triple de pesos por usuario (no uno por tema), así que con 25
usuarios entrenados hay apenas 25 muestras — mucho menos robusto que
las 3856 de conocimiento/interés. `calibrar_dominio.py` se extendió
para reportar min/percentiles/max de estos 6 valores también (función
`analizar_pesos`), con una advertencia explícita en la salida sobre el
tamaño de muestra reducido.

**No se tocó ninguna fórmula en `pesos_ink.py`** porque no hay evidencia
de que esté mal calibrada: los ejemplos de la documentación oficial de
TrueLearn (docstring de `INKClassifier.fit`) muestran pesos aprendidos
en rangos como `novelty=0.208`, `interest=0.669`, `bias=0.130` — todos
razonablemente dentro del [0,1] que `nov_pct = mean*100` asume. Si se
quiere confirmar con los datos reales del proyecto, correr
`calibrar_dominio.py` (ya extendido) y revisar la sección "PESOS DEL
MODELO" de la salida.

---

## 21. Tests de invariantes estructurales en `logic/dominio.py`

**Archivo:** `tests/test_dominio.py`.

Los puntos 18, 19 y 20 fueron, en el fondo, la misma clase de bug
repetida tres veces: un umbral quedó huérfano respecto al rango
`KNOW_MIN/MAX` o `INT_MIN/MAX` vigente después de una recalibración, sin
que ningún test lo notara. Se agregaron tests que verifican, de forma
genérica, que:

- `KNOW_MIN < UMBRAL_BASICO < UMBRAL_MEDIO < UMBRAL_ALTO < KNOW_MAX`
- `INT_MIN < UMBRAL_INTERES_MEDIO < UMBRAL_INTERES_ALTO < INT_MAX`
- `VAR_MIN <= UMBRAL_VAR_MUY_SEGURO < UMBRAL_VAR_SEGURO < UMBRAL_VAR_MODERADO <= VAR_MAX`
- Los `PCT_INICIO_*` coinciden exactamente con lo que devuelven las
  funciones de normalización de las que derivan.
- `pct_conocimiento`/`pct_interes` son monótonas no decrecientes.

**Verificación de que estos tests realmente sirven:** se reintrodujeron
los 3 bugs históricos uno por uno (`UMBRAL_ALTO_PCT` inflado por encima
de 100%, `UMBRAL_VAR_MUY_SEGURO` por debajo de `VAR_MIN`,
`PCT_INICIO_CONOCIMIENTO` hardcodeado en 25.0) y se confirmó que cada
uno hace fallar el test correspondiente de inmediato.

---

## 22. Linter (`ruff`), type-checker (`mypy`) y medición de cobertura

**Archivos:** `pyproject.toml` (nuevo, reemplaza a `pytest.ini`),
`requirements-dev.txt` (nuevo).

El proyecto no tenía ninguna herramienta de análisis estático
configurada. Se agregó:

- **`ruff`**, configurado en `pyproject.toml` con un set de reglas
  conservador (errores reales de pyflakes/pycodestyle + bugbear, sin
  imponer un estilo agresivo sobre un proyecto que ya tiene el suyo).
  Encontró **88 problemas reales**, todos corregidos:
  - 12 imports muertos (`shared.TEAL`/`GRID_COL` importados y nunca
    usados en varias páginas, `import pandas`/`import math` sin usar,
    etc.) — vestigios de versiones anteriores del código.
  - **4 `except:` desnudos** (dos en `shared.py`, dos en
    `pages/login.py`) que atrapaban absolutamente cualquier excepción,
    incluyendo `KeyboardInterrupt`/`SystemExit` — cambiados a
    `except Exception:`.
  - 20 líneas con múltiples sentencias en una sola línea (`;` o
    varios `if: return` pegados) — reformateadas a multilínea,
    especialmente en `logic/dominio.py` por ser el módulo más crítico.
  - 1 variable de loop no usada.

  `ruff check .` queda limpio.

- **`mypy`**, configurado para exigir tipado completo en
  `logic/dominio.py` y `logic/modelo_utils.py` (los módulos más
  centrales y más testeados) sin forzar `--strict` en todo el proyecto
  todavía. Encontró 6 funciones sin anotaciones de tipo; se completaron.
  `mypy` queda limpio en esos dos módulos.

- **`pytest-cov`**, para medir cobertura real en vez de contar tests a
  ciegas. El primer resultado fue revelador:

  | Módulo | Cobertura antes de este punto |
  |---|---|
  | `logic/dominio.py` | 100% |
  | `logic/entrenador.py` | **0%** |
  | `logic/peek_loader.py` | **0%** |
  | `logic/extractor_peek.py` | **0%** |
  | `logic/preparar_datos.py` | **0%** |
  | `shared.py` | 25.7% |
  | **Total** | **30.1%** |

  El pipeline de entrenamiento —justo donde vive el fix metodológico
  del punto 2— tenía cero cobertura. Se agregaron `tests/test_entrenamiento.py`
  (`seleccionar_usuarios`, `entrenar_usuario` y `generar_perfil_json`
  contra el `INKClassifier` real con eventos sintéticos — sin descargar
  PEEK—, y `entrenar_modelo_usuario()` completo con el dataset mockeado),
  `tests/test_shared_http.py` y `tests/test_shared_apptest.py` (los
  helpers HTTP de `shared.py`, incluyendo las ramas que llaman
  `st.error()`/`st.stop()`, vía `AppTest`), y `tests/test_popup_editar.py`
  (el popup de corrección explícita — el mecanismo central de toda la
  tesis — que tenía 0% de cobertura porque ningún smoke test anterior
  abría el popup). Cobertura final:

  | Módulo | Cobertura final |
  |---|---|
  | `logic/dominio.py` | 100% |
  | `logic/entrenador.py` | 100% |
  | `logic/peek_loader.py` | 100% |
  | `logic/extractor_peek.py` | 100% |
  | `logic/preparar_datos.py` | 100% |
  | `logic/modelo_utils.py` | 100% |
  | `shared.py` | 100% |
  | `logic/gestor_perfil.py` | 0% (ver punto 23 — código muerto, no se testeó a propósito) |
  | **Total** | **97.7%** |

  La suite pasó de 112 a **165 tests**, corridos 3+ veces de forma
  estable para descartar que la mejora de cobertura introdujera
  inestabilidad.

**Hallazgo lateral:** `truelearn` (la librería, no este proyecto) usa
internamente `datetime.datetime.utcfromtimestamp()`, deprecada en
Python moderno. No es corregible desde este código — queda como nota
para la memoria si se documenta el entorno de desarrollo.

---

## 23. Código muerto encontrado: `logic/gestor_perfil.py` (y notas sobre otros dos módulos)

**Archivo:** `logic/gestor_perfil.py`.

Al medir cobertura, `gestor_perfil.py` dio 0% — pero a diferencia de los
demás módulos en ese estado, acá el motivo no es "faltaba un test": es
que **nada en el proyecto lo llama**. Ni `dashboard.py`, ni
`api_backend.py`, ni ninguna página. Su única función,
`cargar_modelo_entrenado()`, duplica exactamente lo que ya hace
`api_backend.cargar_modelo()` (que además maneja errores de
deserialización, cosa que `gestor_perfil.py` nunca hizo). No tiene ni
siquiera un bloque `if __name__ == "__main__":` que sugiera uso como
script independiente.

**Recomendación: eliminar el archivo.** No se borró en este pase para
no tomar esa decisión unilateralmente — es una línea en la memoria
("se identificó y removió código muerto duplicado") si se decide
sacarlo antes de la entrega.

**Nota relacionada:** `logic/extractor_peek.py` y `logic/preparar_datos.py`
tampoco son llamados por el pipeline principal, pero sí tienen una
razón de ser distinta: `preparar_datos.py` tiene un bloque `__main__`
pensado para correrse a mano durante la Fase 2 de la propuesta
("Análisis de los datos que genera TrueLearn"), y ambos quedaron
igual con 100% de cobertura (ver punto 22) por si se siguen usando de
forma exploratoria/manual. La diferencia con `gestor_perfil.py` es que
estos dos no duplican una función que ya existe en el sistema en vivo.

---

## 24. Las constantes de corrección nunca habían sido validadas empíricamente

**Archivo:** `logic/api_backend.py`.

`delta_know`, `delta_int`, `reduccion_var` y `aumento_var` (usadas en
`aplicar_correccion_directa`) eran números elegidos sin ningún
contraste contra el comportamiento real del modelo — el mismo tipo de
problema que ya se había corregido para `KNOW_MIN/MAX` (punto 17), pero
nunca extendido a estas constantes.

**Se simuló el impacto de UN evento fresco** (`fit()` sobre un tema
recién visto, con un video de dificultad "promedio" `mean=0.5`) contra
`truelearn==1.1.0` real:

| | Δmean | Δvariance |
|---|---|---|
| Conocimiento (enganchado) | 0.2008 | 0.2008 |
| Conocimiento (desenganchado) | 0.2403 | 0.1060 |
| Interés (cualquier label) | 0.7626 | 0.2002 |

**Hallazgo:** `delta_know=0.25` ya estaba dentro del rango real
(0.20–0.24) — validado, sin cambios. `delta_int=0.30` estaba **muy por
debajo** del salto real de un evento de interés (0.76): una corrección
manual de "me interesa" tenía menos de la mitad del efecto de una sola
interacción automática. `reduccion_var=0.05` estaba igual de por debajo
de la caída de varianza real (0.10–0.20).

**Fix:** `delta_int` pasó a `0.75`, `reduccion_var` a `0.15`.
`aumento_var` pasó de `0.10` a `0.30`, manteniendo la misma proporción
(2×) que el autor original usó respecto a `reduccion_var` — este último
valor **no tiene análogo empírico posible**: en el entrenamiento
automático la varianza siempre BAJA con cualquier evento (más datos =
más certeza, sin importar si fue enganche o desenganche); que la
incertidumbre SUBA cuando el usuario contradice al modelo es un
comportamiento exclusivo y deliberado del mecanismo de corrección
manual. Se documentó explícitamente como decisión de diseño, no como
hecho medido, y se recomienda revisarla con los resultados del estudio
SUS.

**Test de regresión:** `tests/test_correccion.py` verifica que estas
constantes sigan siendo del mismo orden de magnitud que el impacto real
medido (no un valor exacto, dado que hay una componente de diseño), así
que si alguien las vuelve a bajar desproporcionadamente, un test falla.

---

## 25. Tamaño de la muestra de calibración: de 25 a 100 usuarios, con justificación estadística

**Archivo:** `logic/entrenador.py`.

`N_USUARIOS_MUCHAS=15` y `N_USUARIOS_POCAS=10` (25 usuarios, 3856 KCs
reales) nunca habían sido justificados estadísticamente — eran tan
arbitrarios como los umbrales que reemplazaron.

**Regla práctica usada:** el percentil 1 (usado para calibrar
`KNOW_MIN`, ver punto 17) se apoya aproximadamente en el
`0.01 × n`-ésimo estadístico de orden. Con `n=3856`, eso son ~39
observaciones — dentro del mínimo habitual (30–50) para que un
percentil extremo no dependa de un puñado de casos, pero justo al
límite.

**Fix:** se cuadriplicó la muestra (`N_USUARIOS_MUCHAS: 15→60`,
`N_USUARIOS_POCAS: 10→40`, total 25→100 usuarios), para que el
percentil 1 se apoye en ~150 observaciones en vez de ~39. Los umbrales
que DEFINEN qué es "muchas"/"pocas" interacción (`UMBRAL_MUCHAS=40`,
`UMBRAL_POCAS_MAX=10`) no se tocaron — siguen siendo una decisión
cualitativa de diseño, no algo con una respuesta estadística única.

**Limitación reconocida explícitamente:** esto mejora la robustez de
`KNOW_MIN/MAX`/`INT_MIN/MAX` (calibrados sobre miles de KCs), pero
**no** resuelve la limitación de los pesos del modelo
(novelty/interest/bias): ahí solo hay una muestra por usuario, así que
ni con 100 usuarios el percentil 1 de los pesos tiene una base sólida
(se apoyaría en ~1 observación). Haría falta un orden de magnitud más
de usuarios (miles) para calibrar los extremos de los pesos con el
mismo rigor — probablemente poco práctico para el alcance de esta
tesis. Esto quedó documentado explícitamente en el código en vez de
ocultarse detrás de una calibración que aparenta más rigor del que
tiene.

**Pendiente:** este cambio requiere reentrenar con el dataset PEEK real
(`python -m logic.entrenador`, ahora bastante más lento — entrena 4
veces más usuarios) y volver a correr `calibrar_dominio.py` para
confirmar si `KNOW_MIN/MAX`/`INT_MIN/MAX` cambian de forma significativa
con la muestra más grande. Si los valores no cambian mucho respecto a
los calibrados con 25 usuarios, eso en sí es una buena noticia
metodológica citable: "la calibración es estable frente al tamaño de
muestra".

---

## 26. Un import de `truelearn` a nivel de módulo hacía caer TODA la suite de tests si la librería faltaba

**Archivos:** `logic/entrenador.py`, `tests/test_entrenamiento.py`
(nuevo: `tests/test_entrenador_puro.py`).

Encontrado verificando, a propósito, que la suite se comporta como
`CAMBIOS.md` prometía ("no requieren truelearn instalado") en un entorno
SIN la librería. `logic/entrenador.py` hacía
`from truelearn.learning import INKClassifier` **a nivel de módulo**
(al principio del archivo), heredado sin cambios desde el código
original. Como `tests/test_entrenamiento.py` importa
`from logic.entrenador import seleccionar_usuarios` —una función que ni
siquiera usa `INKClassifier`— ANTES de su propio
`pytest.importorskip("truelearn")`, el import fallaba en la fase de
**recolección** de pytest, no en la ejecución de un test puntual. Una
falla de recolección interrumpe toda la sesión: sin `truelearn`
instalado, la suite entera pasaba de "165 tests" a **"1 solo test
corrido, todo lo demás abortado"** — el peor escenario posible para
justo la garantía que `CAMBIOS.md` afirmaba tener.

**Fix (dos partes):**
1. `INKClassifier` se importa ahora **adentro** de `entrenar_usuario()`,
   donde realmente se usa — mismo patrón ya aplicado en
   `logic/peek_loader.py` y `logic/api_backend.py`. Esto permite
   importar `logic.entrenador` (y usar `seleccionar_usuarios`) sin
   `truelearn` instalado.
2. Los tests de `seleccionar_usuarios()` (una función pura, sin ninguna
   relación con `truelearn`) se movieron a `tests/test_entrenador_puro.py`,
   sin ningún `importorskip`, para que corran siempre. El resto de
   `tests/test_entrenamiento.py` (que sí construye un `INKClassifier`
   real) mantiene su `importorskip`, ahora ubicado ANTES de cualquier
   import propio del proyecto — no después — como defensa adicional.

**Verificado:** se desinstaló `truelearn` (`pip uninstall truelearn
trueskill`) y se corrió la suite completa 3 veces: **146 tests pasan,
2 archivos se saltan enteros con gracia** (`test_entrenamiento.py`,
`test_truelearn_integration.py`), sin abortar la sesión. Se reinstaló
`truelearn` y se confirmó que los 165 tests vuelven a pasar.

---

## Resumen de archivos nuevos

| Archivo | Motivo |
|---|---|
| `logic/dominio.py` | Única fuente de verdad de rangos, normalización y umbrales (puntos 3, 4) |
| `logic/peek_loader.py` | Carga centralizada y thread-safe del dataset PEEK (punto 6) |
| `logic/modelo_utils.py` | `extraer_pesos_reales`/`resolver_nombre_tema` sin duplicar (puntos 5, 15) |
| `requirements.txt` | Dependencias de producción |
| `requirements-dev.txt` | Dependencias de desarrollo: ruff, mypy, pytest-cov |
| `pyproject.toml` | Configuración centralizada de ruff/mypy/pytest/coverage (reemplaza a `pytest.ini`) |
| `tests/` | 165 tests automatizados (97.7% de cobertura real, medida con pytest-cov), incluyendo `tests/test_entrenador_puro.py` (separado a propósito para no depender de `truelearn`, ver punto 26) |
| `calibrar_dominio.py` | Calcula min/percentiles/max reales de mean/variance sobre los perfiles ya entrenados, y de los pesos del modelo (punto 17, 25) |

## Resumen de archivos modificados

`dashboard.py`, `shared.py`, `pages/login.py`, `pages/resumen.py`,
`pages/temas.py`, `pages/evolucion.py`, `pages/pesos_ink.py`,
`pages/prediccion.py`, `pages/historial.py`, `logic/entrenador.py`,
`logic/api_backend.py`, `logic/gestor_perfil.py`,
`logic/extractor_peek.py`, `logic/preparar_datos.py`.

## Lo que se dejó intencionalmente sin tocar

- La lógica de negocio de qué hace cada tipo de corrección ("sé más" /
  "sé menos" / "me interesa" / "no me interesa") no cambió: solo sus
  límites (clamps) ahora coinciden con la normalización del frontend.
- Las constantes `delta_know = 0.25`, `delta_int = 0.30`, etc. en
  `aplicar_correccion_directa` se dejaron con sus valores originales:
  elegir esos valores es una decisión de diseño de UX/modelado que le
  corresponde a la autora y al tutor, no algo que deba decidirse en una
  refactorización de código. Si se quiere justificar (o recalibrar) esos
  números para la memoria, es el siguiente paso natural después de este
  changelog.
- El diseño visual (colores, layout, textos) no se tocó en absoluto:
  todos los cambios son de correctitud, consistencia y robustez, no de
  estética.
