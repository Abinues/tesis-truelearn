"""
Reglas de negocio sobre los Knowledge Components (KC) de TrueLearn.

Es la única fuente de verdad del proyecto para los rangos válidos, las
fórmulas de normalización a porcentaje y los umbrales de clasificación.
Backend y frontend importan de aquí, de modo que nunca puedan divergir. No
depende de Streamlit, Flask ni TrueLearn, por lo que se puede probar sin
levantar la aplicación ni descargar el dataset PEEK.

Rangos
    TrueLearn modela el conocimiento y el interés como la media de una
    variable gaussiana sobre toda la recta real y no define un rango
    universal. Los límites ``KNOW_*``, ``INT_*`` y ``WEIGHT_*`` se calibraron
    de forma empírica con los percentiles 1 y 99 de los valores observados
    al entrenar con usuarios reales de PEEK (8152 usuarios y 257.594 KC), para
    que pocos usuarios extremos no definan solos el 0 % y el 100 % de la
    escala. ``INT_MIN`` se fija por debajo del mínimo real a propósito: así
    varias correcciones "no me interesa" seguidas siguen siendo visibles
    antes de saturar en 0 %. Si se reentrena con otra muestra, hay que
    volver a ejecutar ``calibrar_dominio.py`` y actualizar estos valores.

Umbrales
    Los umbrales de conocimiento e interés se definen en porcentaje y se
    traducen a valores crudos de ``mean``. Los umbrales de confianza
    (``UMBRAL_VAR_*``) son valores crudos de varianza: corresponden a la
    mediana y al percentil 75 de ``knowledge.variance``.

Porcentajes de inicio
    ``PCT_INICIO_*`` es el porcentaje que representa el estado inicial del
    modelo (antes de cualquier evento). Se calculan con las mismas funciones
    de normalización, por lo que se actualizan solos al recalibrar.
"""
from __future__ import annotations

KNOW_MIN, KNOW_MAX = -0.379, 0.398

INT_MIN,  INT_MAX  = -1.0, 2.720

VAR_MIN,  VAR_MAX = 0.01, 0.5

_KNOW_SPAN = KNOW_MAX - KNOW_MIN
_INT_SPAN  = INT_MAX - INT_MIN

UMBRAL_ALTO_PCT   = 70.0
UMBRAL_MEDIO_PCT  = 40.0
UMBRAL_BASICO_PCT = 10.0

UMBRAL_INTERES_ALTO_PCT  = 50.0
UMBRAL_INTERES_MEDIO_PCT = 30.0


def _valor_desde_pct(pct: float, lo: float, hi: float) -> float:
    """
    Convierte un porcentaje de una escala calibrada a su valor crudo.

    Args:
        pct (float): Porcentaje entre 0 y 100.
        lo (float): Valor crudo que corresponde al 0 %.
        hi (float): Valor crudo que corresponde al 100 %.

    Returns:
        float: Valor crudo equivalente al porcentaje dado.
    """
    return lo + (pct / 100.0) * (hi - lo)


UMBRAL_ALTO   = _valor_desde_pct(UMBRAL_ALTO_PCT,   KNOW_MIN, KNOW_MAX)
UMBRAL_MEDIO  = _valor_desde_pct(UMBRAL_MEDIO_PCT,  KNOW_MIN, KNOW_MAX)
UMBRAL_BASICO = _valor_desde_pct(UMBRAL_BASICO_PCT, KNOW_MIN, KNOW_MAX)

UMBRAL_INTERES_ALTO  = _valor_desde_pct(UMBRAL_INTERES_ALTO_PCT,  INT_MIN, INT_MAX)
UMBRAL_INTERES_MEDIO = _valor_desde_pct(UMBRAL_INTERES_MEDIO_PCT, INT_MIN, INT_MAX)

UMBRAL_VAR_SEGURO     = 0.4318
UMBRAL_VAR_MODERADO   = 0.4634


def _clip(v: float, lo: float, hi: float) -> float:
    """
    Limita un valor a un intervalo cerrado.

    Args:
        v (float): Valor a limitar.
        lo (float): Cota inferior.
        hi (float): Cota superior.

    Returns:
        float: ``v`` si está dentro de ``[lo, hi]``; en otro caso, la cota
        más cercana.
    """
    return max(lo, min(hi, v))


def clamp_conocimiento(mean: float) -> float:
    """
    Limita una media de conocimiento al rango que el frontend sabe normalizar.

    Args:
        mean (float): Media de conocimiento (``knowledge.mean``).

    Returns:
        float: Valor dentro de ``[KNOW_MIN, KNOW_MAX]``.
    """
    return _clip(float(mean), KNOW_MIN, KNOW_MAX)


def clamp_interes(mean: float) -> float:
    """
    Limita una media de interés al rango que el frontend sabe normalizar.

    Args:
        mean (float): Media de interés (``interest.mean``).

    Returns:
        float: Valor dentro de ``[INT_MIN, INT_MAX]``.
    """
    return _clip(float(mean), INT_MIN, INT_MAX)


def clamp_variance(variance: float) -> float:
    """
    Limita una varianza al rango válido del modelo.

    Args:
        variance (float): Varianza de un KC.

    Returns:
        float: Valor dentro de ``[VAR_MIN, VAR_MAX]``.
    """
    return _clip(float(variance), VAR_MIN, VAR_MAX)


def pct_conocimiento(mean: float) -> float:
    """
    Convierte una media de conocimiento a porcentaje.

    Args:
        mean (float): Media de conocimiento (``KNOW_MIN`` a ``KNOW_MAX``).

    Returns:
        float: Porcentaje entre 0 y 100, redondeado a un decimal.
    """
    pct = (float(mean) - KNOW_MIN) / _KNOW_SPAN * 100.0
    return round(_clip(pct, 0.0, 100.0), 1)


def pct_interes(mean: float) -> float:
    """
    Convierte una media de interés a porcentaje.

    Args:
        mean (float): Media de interés (``INT_MIN`` a ``INT_MAX``).

    Returns:
        float: Porcentaje entre 0 y 100, redondeado a un decimal.
    """
    pct = (float(mean) - INT_MIN) / _INT_SPAN * 100.0
    return round(_clip(pct, 0.0, 100.0), 1)


def pct_certeza(variance: float) -> float:
    """
    Convierte una varianza a porcentaje de certeza, ``(1 - varianza) * 100``.

    Args:
        variance (float): Varianza de un KC.

    Returns:
        float: Porcentaje entre 0 y 100, redondeado a un decimal.
    """
    return round(_clip((1.0 - float(variance)) * 100.0, 0.0, 100.0), 1)


PCT_INICIO_CONOCIMIENTO = pct_conocimiento(0.0)
PCT_INICIO_INTERES      = pct_interes(0.0)
PCT_INICIO_CERTEZA      = pct_certeza(0.5)


def nivel_label(mean: float) -> tuple[str, str]:
    """
    Clasifica una media de conocimiento en cuatro niveles.

    Args:
        mean (float): Media de conocimiento.

    Returns:
        tuple[str, str]: Par ``(etiqueta, clase_css)``. Uno de ``("Alto",
        "b-hi")``, ``("Medio", "b-mid")``, ``("Básico", "b-lo")`` o
        ``("Muy bajo", "b-vlo")``.
    """
    m = float(mean)
    if m >= UMBRAL_ALTO:
        return "Alto", "b-hi"
    if m >= UMBRAL_MEDIO:
        return "Medio", "b-mid"
    if m >= UMBRAL_BASICO:
        return "Básico", "b-lo"
    return "Muy bajo", "b-vlo"


def var_label(variance: float) -> tuple[str, str]:
    """
    Clasifica una varianza en tres niveles de confianza.

    Args:
        variance (float): Varianza de un KC.

    Returns:
        tuple[str, str]: Par ``(etiqueta, clase_css)``. Uno de ``("Seguro",
        "b-var-hi")``, ``("Moderado", "b-var-mid")`` o ``("Incierto",
        "b-var-lo")``.
    """
    v = float(variance)
    if v <= UMBRAL_VAR_SEGURO:
        return "Seguro", "b-var-hi"
    if v <= UMBRAL_VAR_MODERADO:
        return "Moderado", "b-var-mid"
    return "Incierto", "b-var-lo"


def interes_label(mean: float) -> tuple[str, str]:
    """
    Clasifica una media de interés en tres niveles.

    Args:
        mean (float): Media de interés.

    Returns:
        tuple[str, str]: Par ``(etiqueta, clase_css)``. Uno de ``("Alto",
        "b-int-hi")``, ``("Medio", "b-int-md")`` o ``("Bajo", "b-int-lo")``.
    """
    m = float(mean)
    if m >= UMBRAL_INTERES_ALTO:
        return "Alto", "b-int-hi"
    if m >= UMBRAL_INTERES_MEDIO:
        return "Medio", "b-int-md"
    return "Bajo", "b-int-lo"


def categoria_dominio(mean: float) -> int:
    """
    Devuelve la categoría numérica de dominio de una media de conocimiento.

    Usa los mismos umbrales que ``nivel_label``, de modo que ambas siempre
    coinciden. La usan el grafo de la red de temas y el sunburst de temas
    por tipo de conocimiento.

    Args:
        mean (float): Media de conocimiento.

    Returns:
        int: ``0`` Alto, ``1`` Medio, ``2`` Básico o ``3`` Muy bajo.
    """
    m = float(mean)
    if m >= UMBRAL_ALTO:
        return 0
    if m >= UMBRAL_MEDIO:
        return 1
    if m >= UMBRAL_BASICO:
        return 2
    return 3


WEIGHT_NOVELTY_MIN,  WEIGHT_NOVELTY_MAX  = 0.0058, 1.6579
WEIGHT_INTERES_MIN,  WEIGHT_INTERES_MAX  = -0.4115, 1.3684
WEIGHT_BIAS_MIN,     WEIGHT_BIAS_MAX     = -1.0615, 1.0199

_WEIGHT_NOVELTY_SPAN = WEIGHT_NOVELTY_MAX - WEIGHT_NOVELTY_MIN
_WEIGHT_INTERES_SPAN = WEIGHT_INTERES_MAX - WEIGHT_INTERES_MIN
_WEIGHT_BIAS_SPAN     = WEIGHT_BIAS_MAX - WEIGHT_BIAS_MIN


def pct_peso_novelty(mean: float) -> float:
    """
    Convierte el peso de novedad del INKClassifier a porcentaje.

    Args:
        mean (float): Valor de ``novelty_weights.mean``.

    Returns:
        float: Porcentaje entre 0 y 100 ("Nivel del video"), redondeado a un
        decimal.
    """
    pct = (float(mean) - WEIGHT_NOVELTY_MIN) / _WEIGHT_NOVELTY_SPAN * 100.0
    return round(_clip(pct, 0.0, 100.0), 1)


def pct_peso_interes(mean: float) -> float:
    """
    Convierte el peso de interés del INKClassifier a porcentaje.

    Es el peso del factor "interés previo" del ensamble, distinto del
    ``interest_mean`` de un tema puntual.

    Args:
        mean (float): Valor de ``interest_weights.mean``.

    Returns:
        float: Porcentaje entre 0 y 100 ("Interés previo"), redondeado a un
        decimal.
    """
    pct = (float(mean) - WEIGHT_INTERES_MIN) / _WEIGHT_INTERES_SPAN * 100.0
    return round(_clip(pct, 0.0, 100.0), 1)


def pct_peso_actividad(mean: float) -> float:
    """
    Convierte el peso de sesgo del INKClassifier a porcentaje.

    Args:
        mean (float): Valor de ``bias_weights.mean``.

    Returns:
        float: Porcentaje entre 0 y 100 ("Actividad"), redondeado a un
        decimal.
    """
    pct = (float(mean) - WEIGHT_BIAS_MIN) / _WEIGHT_BIAS_SPAN * 100.0
    return round(_clip(pct, 0.0, 100.0), 1)


PCT_INICIO_PESO_NOVELTY  = pct_peso_novelty(0.5)
PCT_INICIO_PESO_INTERES  = pct_peso_interes(0.5)
PCT_INICIO_PESO_ACTIVIDAD = pct_peso_actividad(0.0)
