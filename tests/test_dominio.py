"""
tests/test_dominio.py
─────────────────────────────────────────────────────────────────────────────
Tests de logic.dominio: la única fuente de verdad de rangos, normalización
a porcentaje y umbrales de clasificación. No dependen de Streamlit, Flask
ni TrueLearn — corren en cualquier entorno, sin dataset ni red.

Varios de estos tests son regresiones directas de bugs reales encontrados
en la versión original del proyecto:
  - pct_conocimiento/pct_interes podían devolver valores negativos o
    mayores a 100 porque el backend no clampeaba al mismo rango que el
    frontend asumía al normalizar.
"""
import pytest

from logic import dominio as D

# ── pct_conocimiento ─────────────────────────────────────────────────────

def test_pct_conocimiento_extremos():
    assert D.pct_conocimiento(D.KNOW_MIN) == 0.0
    assert D.pct_conocimiento(D.KNOW_MAX) == 100.0


def test_pct_conocimiento_punto_medio():
    medio = (D.KNOW_MIN + D.KNOW_MAX) / 2
    assert D.pct_conocimiento(medio) == 50.0


@pytest.mark.parametrize("mean", [-999, -10, -1.0, D.KNOW_MIN - 0.01])
def test_pct_conocimiento_nunca_negativo(mean):
    """Regresión: antes del fix, un mean fuera de rango podía dar % < 0."""
    assert D.pct_conocimiento(mean) >= 0.0


@pytest.mark.parametrize("mean", [999, 10, 2.0, D.KNOW_MAX + 0.01])
def test_pct_conocimiento_nunca_mayor_a_100(mean):
    assert D.pct_conocimiento(mean) <= 100.0


# ── pct_interes ──────────────────────────────────────────────────────────

def test_pct_interes_extremos():
    assert D.pct_interes(D.INT_MIN) == 0.0
    assert D.pct_interes(D.INT_MAX) == 100.0


@pytest.mark.parametrize("mean", [-999, -10, D.INT_MIN - 0.01])
def test_pct_interes_nunca_negativo(mean):
    assert D.pct_interes(mean) >= 0.0


@pytest.mark.parametrize("mean", [999, 10, D.INT_MAX + 0.01])
def test_pct_interes_nunca_mayor_a_100(mean):
    assert D.pct_interes(mean) <= 100.0


# ── pct_certeza ──────────────────────────────────────────────────────────

def test_pct_certeza_extremos():
    assert D.pct_certeza(D.VAR_MIN) == pytest.approx(98.0, abs=0.1)
    assert D.pct_certeza(0.0) == 100.0
    assert D.pct_certeza(1.0) == 0.0


def test_pct_certeza_es_la_confiabilidad_bayesiana():
    """1 - varianza/varianza inicial: 0 % sin evidencia, 100 % sin incertidumbre,
    y lineal en la varianza (la fracción de incertidumbre inicial ya resuelta)."""
    assert D.pct_certeza(D.VAR_INICIAL) == 0.0
    assert D.pct_certeza(D.VAR_INICIAL / 2) == pytest.approx(50.0, abs=0.05)
    assert D.pct_certeza(0.25) == pytest.approx(50.0, abs=0.05)
    assert D.pct_certeza(0.4330) == pytest.approx(13.4, abs=0.1)   # mediana real tras 1 evento


def test_interes_el_bloque_de_empates_cae_en_medio_y_no_en_alto():
    """El 20,8 % de los temas vale exactamente 1,086 (primera actualización
    de un tema en un video de 5 temas). Con rango promedio ese bloque queda
    en el tercio central; «Alto» exige estar por encima del bloque."""
    assert D.interes_label(1.0860296825626903)[0] == "Medio"
    assert D.interes_label(1.0861)[0] == "Medio"
    assert D.interes_label(1.15)[0] == "Alto"
    assert D.interes_label(0.85)[0] == "Medio"
    assert D.interes_label(0.80)[0] == "Bajo"


@pytest.mark.parametrize("variance", [-5, -0.5, 5, 10])
def test_pct_certeza_siempre_acotado(variance):
    pct = D.pct_certeza(variance)
    assert 0.0 <= pct <= 100.0


# ── clamps (usados por api_backend.aplicar_correccion_directa) ──────────

def test_clamp_conocimiento_respeta_rango():
    assert D.clamp_conocimiento(-999) == D.KNOW_MIN
    assert D.clamp_conocimiento(999) == D.KNOW_MAX
    assert D.clamp_conocimiento(0.0) == 0.0


def test_clamp_interes_respeta_rango():
    assert D.clamp_interes(-999) == D.INT_MIN
    assert D.clamp_interes(999) == D.INT_MAX


def test_clamp_variance_respeta_rango():
    assert D.clamp_variance(-1) == D.VAR_MIN
    assert D.clamp_variance(100) == D.VAR_MAX


def test_clamps_y_normalizacion_son_consistentes():
    """
    Regresión central: cualquier valor que el
    backend puede llegar a escribir (porque pasó por un clamp_*) debe
    normalizar a un porcentaje válido en el frontend. Antes de este fix
    esto NO se cumplía: los clamps del backend permitían valores que la
    fórmula de normalización del frontend convertía en negativos.
    """
    for raw in [-999, -5, -1, 0, 0.5, 1, 5, 999]:
        know = D.clamp_conocimiento(raw)
        inter = D.clamp_interes(raw)
        assert 0.0 <= D.pct_conocimiento(know) <= 100.0
        assert 0.0 <= D.pct_interes(inter) <= 100.0


# ── etiquetas de clasificación ───────────────────────────────────────────

@pytest.mark.parametrize("mean,esperado", [
    (0.9, "Alto"),
    (D.UMBRAL_ALTO, "Alto"),          # borde inclusive
    (D.UMBRAL_ALTO - 0.001, "Medio"),
    (D.UMBRAL_MEDIO, "Medio"),        # borde inclusive
    (D.UMBRAL_MEDIO - 0.001, "Básico"),
    (D.UMBRAL_BASICO, "Básico"),      # borde inclusive
    (D.UMBRAL_BASICO - 0.001, "Muy bajo"),
])
def test_nivel_label_bordes(mean, esperado):
    lbl, _ = D.nivel_label(mean)
    assert lbl == esperado


@pytest.mark.parametrize("variance,esperado", [
    (0.01, "Seguro"),
    (D.UMBRAL_VAR_SEGURO, "Seguro"),
    (D.UMBRAL_VAR_SEGURO + 0.001, "Moderado"),
    (D.UMBRAL_VAR_MODERADO, "Moderado"),
    (D.UMBRAL_VAR_MODERADO + 0.01, "Incierto"),
])
def test_var_label_bordes(variance, esperado):
    lbl, _ = D.var_label(variance)
    assert lbl == esperado


def test_interes_label_bordes():
    assert D.interes_label(D.UMBRAL_INTERES_ALTO)[0] == "Alto"
    assert D.interes_label(D.UMBRAL_INTERES_MEDIO)[0] == "Medio"
    assert D.interes_label(D.UMBRAL_INTERES_MEDIO - 0.01)[0] == "Bajo"


def test_categoria_dominio_coincide_con_nivel_label():
    """categoria_dominio (usada en el grafo de red de temas y en el
    sunburst de temas.py) y nivel_label (usada en el resto de la
    interfaz) deben clasificar el mismo mean de forma consistente, en
    las 4 categorías: si divergen, un mismo tema aparece con un color
    en un gráfico y con otro color -- o directamente desaparece de
    algún gráfico -- justo el bug real que motivó este fix: antes categoria_dominio solo tenía 3 categorías, y
    cualquier tema 'Muy bajo' quedaba mal clasificado como 'Básico'."""
    muestras = [0.95, 0.7, 0.55, 0.4, 0.25, 0.1, -0.2, -0.9]
    esperado_por_label = {"Alto": 0, "Medio": 1, "Básico": 2, "Muy bajo": 3}
    for m in muestras:
        lbl, _ = D.nivel_label(m)
        cat = D.categoria_dominio(m)
        assert cat == esperado_por_label[lbl], (
            f"mean={m}: nivel_label dice '{lbl}' pero categoria_dominio "
            f"devuelve {cat} (se esperaba {esperado_por_label[lbl]})"
        )


def test_nivel_label_tiene_4_colores_distintos():
    """Regresión directa: antes 'Básico' y 'Muy
    bajo' compartían la misma clase CSS ('b-lo'), así que un tema con
    muy poco conocimiento se veía IDÉNTICO a uno con conocimiento
    básico en cualquier badge de la interfaz."""
    muestras = [D.UMBRAL_ALTO + 0.01, D.UMBRAL_MEDIO + 0.001,
                D.UMBRAL_BASICO + 0.001, D.UMBRAL_BASICO - 0.001]
    clases = [D.nivel_label(m)[1] for m in muestras]
    assert len(set(clases)) == 4, f"Clases de nivel_label no son todas distintas: {clases}"


def test_var_label_tiene_3_colores_distintos_y_propios():
    """Regresión directa: antes 'Muy seguro' y
    'Seguro' compartían la misma clase CSS ('b-hi'), que ADEMÁS era la
    misma que 'Alto' de nivel_label -- una reutilización cruzada entre
    dos escalas sin relación entre sí."""
    muestras = [0.01, D.UMBRAL_VAR_MODERADO - 0.001, D.UMBRAL_VAR_MODERADO + 0.01]
    clases_var = [D.var_label(m)[1] for m in muestras]
    assert len(set(clases_var)) == 3, f"Clases de var_label no son todas distintas: {clases_var}"

    clases_nivel = {D.nivel_label(m)[1] for m in
                     [D.UMBRAL_ALTO + 0.01, D.UMBRAL_MEDIO + 0.001,
                      D.UMBRAL_BASICO + 0.001, D.UMBRAL_BASICO - 0.001]}
    assert not (set(clases_var) & clases_nivel), (
        "var_label reutiliza alguna clase CSS de nivel_label "
        f"(var: {clases_var}, nivel: {clases_nivel})"
    )


# ── Invariantes estructurales ─────────────────
#
# Motivo de esta sección: en la práctica, tres bugs reales fueron exactamente la misma clase de error: un umbral
# (UMBRAL_ALTO, UMBRAL_VAR_MODERADO, "25.0" hardcodeado) quedó por fuera
# o mal ubicado respecto al rango KNOW_MIN/MAX o INT_MIN/MAX vigente,
# después de recalibrar. Los tests anteriores de este archivo verifican
# el COMPORTAMIENTO de cada función por separado; estos verifican que
# las CONSTANTES sean consistentes ENTRE SÍ, sin importar qué valores
# tenga la calibración vigente. Si alguien cambia KNOW_MIN/MAX o
# INT_MIN/MAX en el futuro sin actualizar lo demás, estos tests fallan
# de inmediato — no hace falta que un usuario note visualmente que algo
# "no cierra".

def test_umbrales_conocimiento_estan_dentro_del_rango_y_ordenados():
    assert D.KNOW_MIN < D.UMBRAL_BASICO < D.UMBRAL_MEDIO < D.UMBRAL_ALTO < D.KNOW_MAX, (
        "Los umbrales de conocimiento deben caer estrictamente DENTRO de "
        "[KNOW_MIN, KNOW_MAX] y en orden creciente. Si esto falla, algún "
        "umbral quedó inalcanzable (por encima de KNOW_MAX) o redundante "
        "(por debajo de KNOW_MIN) tras una recalibración — el bug exacto "
        "del punto 18."
    )


def test_umbrales_interes_estan_dentro_del_rango_y_ordenados():
    assert D.INT_MIN < D.UMBRAL_INTERES_MEDIO < D.UMBRAL_INTERES_ALTO < D.INT_MAX, (
        "Los umbrales de interés deben caer estrictamente dentro de "
        "[INT_MIN, INT_MAX] y en orden creciente."
    )


def test_umbrales_varianza_estan_dentro_del_rango_y_ordenados():
    assert D.VAR_MIN <= D.UMBRAL_VAR_SEGURO < D.UMBRAL_VAR_MODERADO <= D.VAR_MAX, (
        "Los umbrales de confianza deben caer dentro de [VAR_MIN, VAR_MAX] "
        "y en orden creciente."
    )


def test_umbrales_pct_estan_en_rango_0_100():
    for pct in (D.UMBRAL_ALTO_PCT, D.UMBRAL_MEDIO_PCT, D.UMBRAL_BASICO_PCT,
                D.UMBRAL_INTERES_ALTO_PCT, D.UMBRAL_INTERES_MEDIO_PCT):
        assert 0.0 <= pct <= 100.0


def test_pct_inicio_estan_en_rango_0_100():
    """Los 'porcentajes de inicio' (mean=0.0, variance=0.5, el estado
    neutro antes de cualquier evento) deben normalizar a un valor válido,
    sin importar cómo se recalibren KNOW_MIN/MAX o INT_MIN/MAX. El bug
    del punto 20: estos valores estaban hardcodeados como '25.0'/'16.7'
    en 3 archivos, y ninguno se actualizó cuando cambió la calibración."""
    assert 0.0 <= D.PCT_INICIO_CONOCIMIENTO <= 100.0
    assert 0.0 <= D.PCT_INICIO_INTERES <= 100.0
    assert D.PCT_INICIO_CERTEZA == 0.0  # sin evidencia: 1 - 0.5/0.5, no depende de calibración


def test_pct_inicio_coincide_con_las_funciones_de_normalizacion():
    """Regresión directa del punto 20: si alguien vuelve a hardcodear un
    número en vez de usar estas constantes, este test no lo detecta (no
    puede detectar código que no usa las constantes), pero SÍ garantiza
    que las constantes mismas nunca queden desincronizadas de las
    funciones de las que derivan."""
    assert D.PCT_INICIO_CONOCIMIENTO == D.pct_conocimiento(0.0)
    assert D.PCT_INICIO_INTERES == D.pct_interes(0.0)
    assert D.PCT_INICIO_CERTEZA == D.pct_certeza(0.5)


def test_umbral_alto_clasifica_como_alto_y_basico_como_no_muy_bajo():
    """Sanity check end-to-end: los umbrales derivados deben producir
    exactamente la etiqueta que su nombre promete, no solo estar en el
    rango correcto numéricamente."""
    assert D.nivel_label(D.UMBRAL_ALTO)[0] == "Alto"
    assert D.nivel_label(D.UMBRAL_MEDIO)[0] == "Medio"
    assert D.nivel_label(D.UMBRAL_BASICO)[0] == "Básico"
    assert D.nivel_label(D.UMBRAL_BASICO - 1e-6)[0] == "Muy bajo"

    assert D.interes_label(D.UMBRAL_INTERES_ALTO)[0] == "Alto"
    assert D.interes_label(D.UMBRAL_INTERES_MEDIO)[0] == "Medio"
    assert D.interes_label(D.UMBRAL_INTERES_MEDIO - 1e-6)[0] == "Bajo"

    assert D.var_label(D.UMBRAL_VAR_SEGURO)[0] == "Seguro"
    assert D.var_label(D.UMBRAL_VAR_MODERADO)[0] == "Moderado"
    assert D.var_label(D.UMBRAL_VAR_MODERADO + 1e-6)[0] == "Incierto"


@pytest.mark.parametrize("mean_lo,mean_hi", [(-2, -1), (-0.5, 0), (0, 0.5), (0.5, 2)])
def test_pct_conocimiento_es_monotona_no_decreciente(mean_lo, mean_hi):
    """Si mean_hi > mean_lo, el porcentaje resultante nunca puede ser
    menor. Una regla básica de cualquier normalización lineal, pero vale
    la pena tenerla como test explícito: una recalibración que invierta
    KNOW_MIN/KNOW_MAX por error (ej. pasarlos al revés) rompería esto de
    inmediato en vez de producir un gráfico silenciosamente invertido."""
    assert D.pct_conocimiento(mean_hi) >= D.pct_conocimiento(mean_lo)


@pytest.mark.parametrize("mean_lo,mean_hi", [(-2, -1), (-0.5, 0), (0, 0.5), (0.5, 2)])
def test_pct_interes_es_monotona_no_decreciente(mean_lo, mean_hi):
    assert D.pct_interes(mean_hi) >= D.pct_interes(mean_lo)


def test_know_min_menor_que_know_max():
    assert D.KNOW_MIN < D.KNOW_MAX


def test_int_min_menor_que_int_max():
    assert D.INT_MIN < D.INT_MAX


def test_var_min_menor_que_var_max():
    assert D.VAR_MIN < D.VAR_MAX


# ── Pesos del modelo (novelty/interest/bias) ──────

def test_weight_min_menor_que_weight_max():
    assert D.WEIGHT_NOVELTY_MIN < D.WEIGHT_NOVELTY_MAX
    assert D.WEIGHT_INTERES_MIN < D.WEIGHT_INTERES_MAX
    assert D.WEIGHT_BIAS_MIN < D.WEIGHT_BIAS_MAX


@pytest.mark.parametrize("fn,lo,hi", [
    (D.pct_peso_novelty, D.WEIGHT_NOVELTY_MIN, D.WEIGHT_NOVELTY_MAX),
    (D.pct_peso_interes, D.WEIGHT_INTERES_MIN, D.WEIGHT_INTERES_MAX),
    (D.pct_peso_actividad, D.WEIGHT_BIAS_MIN, D.WEIGHT_BIAS_MAX),
])
def test_pct_peso_extremos_y_acotado(fn, lo, hi):
    assert fn(lo) == 0.0
    assert fn(hi) == 100.0


@pytest.mark.parametrize("fn", [D.pct_peso_novelty, D.pct_peso_interes, D.pct_peso_actividad])
def test_pct_peso_nunca_sale_de_0_100(fn):
    """Regresión directa: antes de este fix,
    nov_pct/int_pct/bias_pct en pages/pesos_ink.py no tenían NINGÚN
    límite. Con un aprendiz real de alto desenganche esto llegó a
    mostrar '-47.6%' y '-66.9%' como si fueran porcentajes válidos."""
    for mean in [-999, -10, -1, 0, 0.5, 1, 10, 999]:
        pct = fn(mean)
        assert 0.0 <= pct <= 100.0


def test_pct_inicio_pesos_estan_en_rango_0_100():
    assert 0.0 <= D.PCT_INICIO_PESO_NOVELTY <= 100.0
    assert 0.0 <= D.PCT_INICIO_PESO_INTERES <= 100.0
    assert 0.0 <= D.PCT_INICIO_PESO_ACTIVIDAD <= 100.0


def test_pct_inicio_pesos_coincide_con_las_funciones():
    """Mismo motivo que test_pct_inicio_coincide_con_las_funciones_de_normalizacion:
    los valores 'de inicio' se calculan a partir de las funciones, nunca
    se hardcodean (ese fue exactamente el bug del punto 20)."""
    assert D.PCT_INICIO_PESO_NOVELTY == D.pct_peso_novelty(0.5)
    assert D.PCT_INICIO_PESO_INTERES == D.pct_peso_interes(0.5)
    assert D.PCT_INICIO_PESO_ACTIVIDAD == D.pct_peso_actividad(0.0)


def test_ejemplo_real_aprendiz_34953_queda_acotado():
    """Regresión: un aprendiz con 80%+ de desenganche mostraba
    'Interés previo: -47.6%' y 'Actividad: -66.9%' porque no había recorte.
    Un valor por debajo del percentil 1 vigente debe mostrarse como 0 %,
    nunca negativo. (Con la calibración de 14.050 usuarios, -0,476 y -1,169
    ya caen dentro del rango, así que se prueba contra el piso vigente.)"""
    assert D.pct_peso_interes(D.WEIGHT_INTERES_MIN - 0.05) == 0.0
    assert D.pct_peso_actividad(D.WEIGHT_BIAS_MIN - 0.05) == 0.0
    # El de novelty NO estaba en el extremo -> no debería clampear a 0 ni a 100
    resultado_novelty = D.pct_peso_novelty(0.961)
    assert 0.0 < resultado_novelty < 100.0