"""
logic/dominio.py
─────────────────────────────────────────────────────────────────────────────
Única fuente de verdad para las reglas de negocio de este proyecto sobre
los Knowledge Components de TrueLearn: rangos válidos, fórmulas de
normalización a porcentaje, y umbrales de clasificación (Alto/Medio/Básico,
certeza, interés).

Por qué existe este módulo
-----------------------------------------------------------
Antes de esta refactorización:

  - La fórmula de normalización `(mean + 0.5) / 2.0 * 100` (conocimiento) y
    `(mean + 0.5) / 3.0 * 100` (interés) estaba copiada de forma
    independiente en shared.py, temas.py, pesos_ink.py, resumen.py,
    evolucion.py y prediccion.py. Algunas copias clampeaban a [0, 100] y
    otras no.
  - api_backend.py aplicaba límites (clamps) a los `mean` corregidos que
    NO coincidían con el rango que esas fórmulas de normalización
    asumían (p. ej. permitía `mean = -1.0` en conocimiento, lo que con la
    fórmula de arriba da -25%, un porcentaje sin sentido de dominio).
  - Los umbrales 0.7 / 0.4 / 0.1 para "Alto/Medio/Básico" estaban
    hardcodeados de forma independiente en tres archivos distintos.

Ahora TODO el proyecto (backend y frontend) importa las constantes y
funciones de este único módulo. Este archivo no depende de Streamlit,
Flask ni TrueLearn: es lógica pura, así se puede testear sin necesidad de
levantar la app ni descargar el dataset PEEK.

Sobre los límites KNOW_MIN/MAX e INT_MIN/MAX
-------------------------------------------------------------------------
TrueLearn modela el conocimiento/interés como el `mean` de una variable
gaussiana sobre TODA la recta real (θ ∈ ℝ), inspirado directamente en
TrueSkill (Bulathwela et al., 2020, "TrueLearn: A Family of Bayesian
Algorithms..."). La librería NO define ningún rango universal — es una
elección de diseño explícita del modelo, no un olvido de documentación.
La única cantidad que la librería expone con semántica nativamente
acotada a [0, 1] es `predict_proba()` (usada en api_backend.py para el
antes/después del popup de corrección), no el `mean` crudo.

Por lo tanto, estos límites fueron calibrados EMPÍRICAMENTE a partir de
la distribución real observada tras entrenar sobre el dataset PEEK
completo (todos los usuarios de entrenamiento, calibración corrida con
calibrar_dominio.py y calibracion_por_estrato.py). Se usan los percentiles 1/99 en vez del
mínimo/máximo absoluto para no dejar que uno o dos usuarios extremos
(p. ej. uno con 831 temas distintos) definan solos el 0%/100% de la
escala visual para el resto de la población.

Si se reentrena con una muestra de usuarios sustancialmente distinta
(otro subconjunto de PEEK, otros umbrales de selección en
logic/entrenador.py), conviene volver a correr calibrar_dominio.py y
actualizar estos valores.
"""
from __future__ import annotations

# ── Rangos válidos de las variables del modelo ──────────────────────────────
# KNOW_MIN/MAX: percentiles 1/99 de knowledge.mean sobre 536.424 KC de
# 14.050 usuarios: CENSO de todos los usuarios del split de entrenamiento
# de PEEK con al menos un evento (entrenar_gran_escala.py + calibrar_dominio.py).
# Una corrida previa con 8.152 usuarios (los de >= 40 o <= 10 eventos)
# dio -0,379 / +0,398: el censo (-0,376 / +0,415) cambia el límite
# inferior en 0,003, lo que confirma que la escala no depende de ese
# corte de usuarios.
# El mínimo/máximo absoluto real observado fue -1.251 / +1.211; se usan
# los percentiles 1/99 para no dejar que un puñado de usuarios extremos
# compriman a todos los demás.
KNOW_MIN, KNOW_MAX = -0.376, 0.415

# INT_MIN / INT_MAX: percentiles 1 y 99 de interest_mean sobre los mismos
# 536.424 KC (0.107 / 2.769), la MISMA regla que KNOW_MIN/KNOW_MAX. Así las
# cuatro anclas de las correcciones explícitas («sé más», «sé menos»,
# «me interesa», «no me interesa») son los extremos de la población
# (percentiles 1 y 99): la afirmación del estudiante lo ubica en un extremo
# de lo observado, de forma simétrica en rango para subir y para bajar.
# El interés queda concentrado en la parte baja de la escala (mediana
# ~35 %) porque su distribución es asimétrica, con una cola larga hacia
# valores altos; el porcentaje es la posición dentro del rango observado.
INT_MIN,  INT_MAX  = 0.107, 2.769

# VAR_MIN/MAX: el mínimo/máximo absoluto real observado (0.0036 / 0.493)
# confirma que estos límites siguen siendo razonables sin cambios: 0.01
# ya funcionaba como piso de seguridad numérica en otras partes del
# código (p. ej. el tamaño de burbuja en pages/temas.py hace
# variance.clip(lower=0.01) para evitar dividir por casi-cero), y 0.5
# es el techo teórico de inicio de TrueLearn, nunca superado en la
# práctica.
VAR_MIN,  VAR_MAX = 0.01, 0.5

_KNOW_SPAN = KNOW_MAX - KNOW_MIN
_INT_SPAN  = INT_MAX - INT_MIN

# ── Umbrales de clasificación (única definición en todo el proyecto) ───────
# FIX: estos umbrales se definían como valores
# de `mean` CRUDO (0.7 / 0.4 / 0.1), que tenían sentido bajo el rango
# viejo (KNOW_MIN=-0.5, KNOW_MAX=1.5). Al recalibrar KNOW_MAX a 0.438 con
# datos reales (punto 17), UMBRAL_ALTO=0.7 quedó POR ENCIMA del máximo
# posible: ningún tema podía clasificar como "Alto" nunca más, y
# UMBRAL_BASICO=0.1 (equivalente a 62.9% con la escala nueva) dejaba a
# más de la mitad de los temas reales (mediana real: 56.3%) fuera de
# "Jerarquía de dominio" en pages/temas.py, que solo dibuja los grupos
# Alto/Medio/Básico.
#
# Los umbrales se definen en PORCENTAJE de la escala calibrada y se
# traducen a `mean` crudo con KNOW_MIN/KNOW_MAX e INT_MIN/INT_MAX. Si se
# vuelve a calibrar (calibrar_dominio.py), solo hay que actualizar estos
# porcentajes y nunca pueden quedar fuera del rango real.
#
# ¿De dónde salen los cortes? De CUANTILES de los datos reales. Fijar un
# punto de corte es un acto de juicio (Cizek, Bunch y Koons, 2004), y las
# dos familias de métodos son la referida a norma (el corte ubica a la
# persona respecto de un grupo) y la referida a criterio (el corte
# representa un desempeño esperado). Este sistema no tiene un criterio
# externo que diga qué es «dominar» un tema; sí tiene la distribución
# real de 14.050 usuarios, así que usa el método normativo:
#   - Conocimiento, 4 etiquetas -> CUARTILES (25/50/75) de la escala:
#       Alto = cuartil superior, Medio = 2.º cuartil desde arriba,
#       Básico = 2.º cuartil desde abajo, Muy bajo = cuartil inferior.
#   - Interés, 3 etiquetas -> TERCILES (33/67):
#       Alto = tercio superior, Medio = tercio central, Bajo = inferior.
# Efecto: en conocimiento cada etiqueta agrupa ~25 % de los temas reales
# (medido: 25,0 / 25,0 / 25,0 / 25,0). En interés el reparto medido es
# 33 / 18 / 48 % con la regla «mayor o igual» sobre el tercil, no 33 / 33 / 33: muchos temas comparten el mismo valor
# de interés (el 20,8 % vale exactamente 1,086, ver interes_masas.py) y el
# tercil cae dentro de ese bloque. Con empates se usa el RANGO PROMEDIO
# (convención estándar en estadística de rangos): el bloque ocupa del 51,5 %
# al 72,3 % de la distribución, su rango promedio (~62 %) está en el tercio
# central, y por eso queda en «Medio»; «Alto» exige estar por encima del
# bloque. Reparto resultante ≈ 33 / 39 / 28 %. «Alto» significa «entre los temas
# más altos de la población», no «dominio garantizado»; la interfaz y la
# tesis lo dicen así. Los valores de abajo son los que calcula
# calibrar_dominio.py / calibracion_por_estrato.py sobre los 14.050
# usuarios de entrenamiento de PEEK (censo, sin corte por eventos).
UMBRAL_ALTO_PCT   = 64.5   # cuartil 3: % de conocimiento >= 64.5%          → "Alto"
UMBRAL_MEDIO_PCT  = 55.3   # mediana:   55.3% <= % de conocimiento < 64.5%  → "Medio"
UMBRAL_BASICO_PCT = 31.6   # cuartil 1: 31.6% <= % de conocimiento < 55.3%  → "Básico"
                           #            % de conocimiento < 31.6%           → "Muy bajo"

UMBRAL_INTERES_ALTO_PCT  = 36.8  # justo por encima del bloque de empates: % de interés >= 36.8% → "Alto"
UMBRAL_INTERES_MEDIO_PCT = 27.9  # tercil 1: 27.9% <= % de interés < 36.8%    → "Medio"
                                 #           % de interés < 27.9%             → "Bajo"


def _valor_desde_pct(pct: float, lo: float, hi: float) -> float:
    """Convierte un % de una escala calibrada a su valor crudo equivalente."""
    return lo + (pct / 100.0) * (hi - lo)


UMBRAL_ALTO   = _valor_desde_pct(UMBRAL_ALTO_PCT,   KNOW_MIN, KNOW_MAX)
UMBRAL_MEDIO  = _valor_desde_pct(UMBRAL_MEDIO_PCT,  KNOW_MIN, KNOW_MAX)
UMBRAL_BASICO = _valor_desde_pct(UMBRAL_BASICO_PCT, KNOW_MIN, KNOW_MAX)

UMBRAL_INTERES_ALTO  = _valor_desde_pct(UMBRAL_INTERES_ALTO_PCT,  INT_MIN, INT_MAX)
UMBRAL_INTERES_MEDIO = _valor_desde_pct(UMBRAL_INTERES_MEDIO_PCT, INT_MIN, INT_MAX)

# UMBRAL_VAR_* (Confianza: Seguro/Moderado/Incierto) — SIGUEN siendo
# valores de variance crudos (pct_certeza() no usa VAR_MIN/MAX, es
# (1 - variance / VAR_INICIAL) * 100, así que no hay un "% de escala" del
# cual derivarlos como con UMBRAL_ALTO).
#
# FIX: la versión anterior tenía 4 niveles
# (Muy seguro/Seguro/Moderado/Incierto) pero solo 3 colores de badge
# realmente distintos en el CSS — "Muy seguro" y "Seguro" terminaban
# mostrándose con el MISMO color, y de paso ese color era el mismo que
# "Alto" en la escala de conocimiento (una reutilización cruzada entre
# escalas que no tienen relación entre sí). Se reduce a 3 niveles reales,
# cada uno con su propio color exclusivo (b-var-hi/b-var-mid/b-var-lo,
# ver shared.py) — se fusionan "Muy seguro" y "Seguro" en un solo
# "Seguro", usando los 2 cortes reales que quedan (mediana y p75 de
# knowledge.variance, mismo reporte de calibrar_dominio.py usado para
# KNOW_MIN/MAX) en vez de inventar un tercer corte nuevo sin calibrar.
UMBRAL_VAR_SEGURO     = 0.4311  # mediana real de knowledge.variance (257.594 KCs)
UMBRAL_VAR_MODERADO   = 0.4615  # p75 real de knowledge.variance



def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ── Clamps para el backend (aplicar_correccion_directa en api_backend.py) ──
def clamp_conocimiento(mean: float) -> float:
    """Limita un mean de knowledge al rango que el frontend sabe normalizar."""
    return _clip(float(mean), KNOW_MIN, KNOW_MAX)


def clamp_interes(mean: float) -> float:
    """Limita un mean de interest al rango que el frontend sabe normalizar."""
    return _clip(float(mean), INT_MIN, INT_MAX)


def clamp_variance(variance: float) -> float:
    return _clip(float(variance), VAR_MIN, VAR_MAX)


# ── Conversión a porcentaje para el frontend (SIEMPRE acotada a [0, 100]) ──
def pct_conocimiento(mean: float) -> float:
    """mean de knowledge (KNOW_MIN..KNOW_MAX) → porcentaje 0..100."""
    pct = (float(mean) - KNOW_MIN) / _KNOW_SPAN * 100.0
    return round(_clip(pct, 0.0, 100.0), 1)


def pct_interes(mean: float) -> float:
    """mean de interest (INT_MIN..INT_MAX) → porcentaje 0..100."""
    pct = (float(mean) - INT_MIN) / _INT_SPAN * 100.0
    return round(_clip(pct, 0.0, 100.0), 1)


# Varianza inicial de TrueLearn (def_var): la incertidumbre de un tema sin
# ninguna evidencia.
VAR_INICIAL = 0.5


def pct_certeza(variance: float) -> float:
    """Confianza del sistema: ``(1 - varianza / VAR_INICIAL) × 100``, acotada a 0-100.

    Es la confiabilidad de una estimación bayesiana: la fracción de la
    incertidumbre inicial que la evidencia ya resolvió. En la estimación
    bayesiana de rasgos (EAP) la confiabilidad es ``1 - varianza posterior``
    cuando la varianza previa vale 1 (Bock y Mislevy, 1982; Green et al.,
    1984); aquí la previa vale ``VAR_INICIAL`` = 0.5, así que se divide por
    ella. Un tema sin evidencia da 0 %, y 100 % solo si la varianza fuera 0.
    """
    return round(_clip((1.0 - float(variance) / VAR_INICIAL) * 100.0, 0.0, 100.0), 1)


# ── Porcentajes de referencia ("estado inicial/neutro", para deltas) ──────
# FIX: antes de cualquier evento, TrueLearn
# inicializa knowledge.mean=0.0 e interest_mean=0.0. resumen.py, temas.py
# y shared.py (popup de corrección) mostraban el delta contra ese estado
# neutro usando "25.0" y "16.7" HARDCODEADOS — calculados a mano bajo el
# rango VIEJO (KNOW_MAX=1.5, INT_MAX=2.5). Al recalibrar esos rangos con
# datos reales (punto 17), mean=0.0 pasó a representar ~51.9% de
# conocimiento y ~14.2% de interés, no 25%/16.7% — los deltas mostrados
# quedaron mal sin que ningún test lo detectara, porque eran literales
# sueltos en 3 archivos distintos, no algo que pasara por dominio.py.
#
# Ahora se calculan acá, una sola vez, a partir de las mismas funciones
# de normalización: si KNOW_MIN/MAX o INT_MIN/MAX se recalibran de
# nuevo, estos valores se actualizan solos.
PCT_INICIO_CONOCIMIENTO = pct_conocimiento(0.0)
PCT_INICIO_INTERES      = pct_interes(0.0)
PCT_INICIO_CERTEZA      = pct_certeza(VAR_INICIAL)  # sin evidencia: 0 %, no depende de calibración


# ── Etiquetas de clasificación (una sola definición, reutilizada siempre) ──
def nivel_label(mean: float) -> tuple[str, str]:
    """Devuelve (etiqueta, clase_css). 4 niveles, 4 colores propios
    (ver shared.py) — ninguno se comparte entre sí ni con las escalas
    de interés o confianza."""
    m = float(mean)
    if m >= UMBRAL_ALTO:
        return "Alto", "b-hi"
    if m >= UMBRAL_MEDIO:
        return "Medio", "b-mid"
    if m >= UMBRAL_BASICO:
        return "Básico", "b-lo"
    return "Muy bajo", "b-vlo"


def var_label(variance: float) -> tuple[str, str]:
    """Devuelve (etiqueta, clase_css). 3 niveles, 3 colores propios
    (b-var-hi/b-var-mid/b-var-lo, ver shared.py) — ninguno se comparte
    con la escala de conocimiento."""
    v = float(variance)
    if v <= UMBRAL_VAR_SEGURO:
        return "Seguro", "b-var-hi"
    if v <= UMBRAL_VAR_MODERADO:
        return "Moderado", "b-var-mid"
    return "Incierto", "b-var-lo"


def interes_label(mean: float) -> tuple[str, str]:
    m = float(mean)
    if m >= UMBRAL_INTERES_ALTO:
        return "Alto", "b-int-hi"
    if m >= UMBRAL_INTERES_MEDIO:
        return "Medio", "b-int-md"
    return "Bajo", "b-int-lo"


def categoria_dominio(mean: float) -> int:
    """0=Alto, 1=Medio, 2=Básico, 3=Muy bajo. Usado por el grafo de red
    de temas y por el sunburst de 'Temas por tipo de conocimiento'.

    FIX: antes esta función solo tenía 3
    categorías (0/1/2), sin ninguna rama para 'Muy bajo' — cualquier
    tema con mean < UMBRAL_MEDIO caía directo en la categoría 2
    ('Básico'), aunque en realidad estuviera muy por debajo de
    UMBRAL_BASICO. Un mismo tema podía aparecer como 'Muy bajo' en el
    resto de la interfaz (vía nivel_label) pero como 'Básico' en el
    grafo de red de temas — exactamente la clase de inconsistencia que
    motivó centralizar esto en un solo módulo. Ahora coincide siempre
    con nivel_label."""
    m = float(mean)
    if m >= UMBRAL_ALTO:
        return 0
    if m >= UMBRAL_MEDIO:
        return 1
    if m >= UMBRAL_BASICO:
        return 2
    return 3


# ── Pesos del modelo (novelty/interest/bias) — logic.dominio, punto 32 ──────
#
# FIX: pages/pesos_ink.py normalizaba los
# PESOS del clasificador (distintos de los Knowledge Components: acá
# "interest" es el peso del FACTOR "interés previo" en el ensamble
# INK, no el interest_mean de un tema puntual) con fórmulas SIN NINGÚN
# límite: `nov_pct = nov_m * 100`, `int_pct = int_m * 100`,
# `bias_pct = (bias_m + 0.5) * 100`. Con un aprendiz real de alto
# desenganche (80%+), esto llegó a mostrar "Interés previo: -47.6%" y
# "Actividad: -66.9%" — matemáticamente correcto dado el peso aprendido,
# pero sin sentido como "porcentaje" para quien lo lee.
#
# Se calibraron estos rangos empíricamente (percentiles 1/99) sobre los
# MISMOS 8156 usuarios reales usados para KNOW_MIN/MAX (punto 28) — acá
# sí hay 1 muestra de pesos por usuario, y 8156 es una base sólida para
# percentiles (a diferencia de la calibración con 25 usuarios del punto
# 25, que para pesos daba apenas 25 muestras).
WEIGHT_NOVELTY_MIN,  WEIGHT_NOVELTY_MAX  = -0.1357, 1.5938   # p1/p99 de novelty_weights.mean
WEIGHT_INTERES_MIN,  WEIGHT_INTERES_MAX  = -0.6621, 1.5256  # p1/p99 de interest_weights.mean
WEIGHT_BIAS_MIN,     WEIGHT_BIAS_MAX     = -1.3708, 1.2112  # p1/p99 de bias_weights.mean

_WEIGHT_NOVELTY_SPAN = WEIGHT_NOVELTY_MAX - WEIGHT_NOVELTY_MIN
_WEIGHT_INTERES_SPAN = WEIGHT_INTERES_MAX - WEIGHT_INTERES_MIN
_WEIGHT_BIAS_SPAN     = WEIGHT_BIAS_MAX - WEIGHT_BIAS_MIN


def pct_peso_novelty(mean: float) -> float:
    """novelty_weights.mean → porcentaje 0..100 ('Nivel del video')."""
    pct = (float(mean) - WEIGHT_NOVELTY_MIN) / _WEIGHT_NOVELTY_SPAN * 100.0
    return round(_clip(pct, 0.0, 100.0), 1)


def pct_peso_interes(mean: float) -> float:
    """interest_weights.mean → porcentaje 0..100 ('Interés previo')."""
    pct = (float(mean) - WEIGHT_INTERES_MIN) / _WEIGHT_INTERES_SPAN * 100.0
    return round(_clip(pct, 0.0, 100.0), 1)


def pct_peso_actividad(mean: float) -> float:
    """bias_weights.mean → porcentaje 0..100 ('Actividad')."""
    pct = (float(mean) - WEIGHT_BIAS_MIN) / _WEIGHT_BIAS_SPAN * 100.0
    return round(_clip(pct, 0.0, 100.0), 1)


# Porcentajes de referencia ("estado inicial", mismo motivo que
# PCT_INICIO_CONOCIMIENTO/INTERES en el punto 20): antes de cualquier
# entrenamiento, INKClassifier inicializa novelty=0.5, interest=0.5,
# bias=0.0. Se calculan acá, no se hardcodean, por la misma razón que
# ya causó el bug del punto 20 una vez.
PCT_INICIO_PESO_NOVELTY  = pct_peso_novelty(0.5)
PCT_INICIO_PESO_INTERES  = pct_peso_interes(0.5)
PCT_INICIO_PESO_ACTIVIDAD = pct_peso_actividad(0.0)