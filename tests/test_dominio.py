"""
Tests de ``logic.dominio``, única fuente de verdad de rangos, normalización a
porcentaje y umbrales de clasificación.

No dependen de Streamlit, Flask ni TrueLearn. Varios son regresiones de bugs
reales de la versión original: ``pct_conocimiento`` y ``pct_interes``
devolvían valores negativos o mayores a 100 porque el backend no acotaba al
mismo rango que el frontend asumía al normalizar.

También hay invariantes estructurales: verifican que las constantes
(umbrales, rangos y porcentajes de inicio) sean consistentes entre sí sin
importar la calibración vigente, de modo que una recalibración que deje un
umbral fuera de rango o desordenado falle de inmediato.
"""
import pytest

from logic import dominio as D


def test_pct_conocimiento_extremos():
    """Los extremos del rango de conocimiento normalizan a 0 y 100."""
    assert D.pct_conocimiento(D.KNOW_MIN) == 0.0
    assert D.pct_conocimiento(D.KNOW_MAX) == 100.0


def test_pct_conocimiento_punto_medio():
    """El punto medio del rango de conocimiento normaliza a 50."""
    medio = (D.KNOW_MIN + D.KNOW_MAX) / 2
    assert D.pct_conocimiento(medio) == 50.0


@pytest.mark.parametrize("mean", [-999, -10, -1.0, D.KNOW_MIN - 0.01])
def test_pct_conocimiento_nunca_negativo(mean):
    """Una media fuera de rango por debajo nunca produce un porcentaje negativo.

    Args:
        mean (float): Media de conocimiento por debajo de ``KNOW_MIN``.
    """
    assert D.pct_conocimiento(mean) >= 0.0


@pytest.mark.parametrize("mean", [999, 10, 2.0, D.KNOW_MAX + 0.01])
def test_pct_conocimiento_nunca_mayor_a_100(mean):
    """Una media fuera de rango por encima nunca produce más de 100.

    Args:
        mean (float): Media de conocimiento por encima de ``KNOW_MAX``.
    """
    assert D.pct_conocimiento(mean) <= 100.0


def test_pct_interes_extremos():
    """Los extremos del rango de interés normalizan a 0 y 100."""
    assert D.pct_interes(D.INT_MIN) == 0.0
    assert D.pct_interes(D.INT_MAX) == 100.0


@pytest.mark.parametrize("mean", [-999, -10, D.INT_MIN - 0.01])
def test_pct_interes_nunca_negativo(mean):
    """Una media de interés por debajo del rango nunca produce un negativo.

    Args:
        mean (float): Media de interés por debajo de ``INT_MIN``.
    """
    assert D.pct_interes(mean) >= 0.0


@pytest.mark.parametrize("mean", [999, 10, D.INT_MAX + 0.01])
def test_pct_interes_nunca_mayor_a_100(mean):
    """Una media de interés por encima del rango nunca produce más de 100.

    Args:
        mean (float): Media de interés por encima de ``INT_MAX``.
    """
    assert D.pct_interes(mean) <= 100.0


def test_pct_certeza_extremos():
    """Verifica la certeza en varianza mínima, nula y máxima."""
    assert D.pct_certeza(D.VAR_MIN) == pytest.approx(99.0, abs=0.1)
    assert D.pct_certeza(0.0) == 100.0
    assert D.pct_certeza(1.0) == 0.0


@pytest.mark.parametrize("variance", [-5, -0.5, 5, 10])
def test_pct_certeza_siempre_acotado(variance):
    """La certeza siempre queda entre 0 y 100.

    Args:
        variance (float): Varianza, incluso fuera del rango válido.
    """
    pct = D.pct_certeza(variance)
    assert 0.0 <= pct <= 100.0


def test_clamp_conocimiento_respeta_rango():
    """``clamp_conocimiento`` acota al rango de conocimiento."""
    assert D.clamp_conocimiento(-999) == D.KNOW_MIN
    assert D.clamp_conocimiento(999) == D.KNOW_MAX
    assert D.clamp_conocimiento(0.0) == 0.0


def test_clamp_interes_respeta_rango():
    """``clamp_interes`` acota al rango de interés."""
    assert D.clamp_interes(-999) == D.INT_MIN
    assert D.clamp_interes(999) == D.INT_MAX


def test_clamp_variance_respeta_rango():
    """``clamp_variance`` acota al rango de varianza."""
    assert D.clamp_variance(-1) == D.VAR_MIN
    assert D.clamp_variance(100) == D.VAR_MAX


def test_clamps_y_normalizacion_son_consistentes():
    """Todo valor que pasa por un ``clamp_*`` normaliza a un porcentaje válido.

    Antes, los límites del backend permitían valores que la normalización del
    frontend convertía en porcentajes negativos.
    """
    for raw in [-999, -5, -1, 0, 0.5, 1, 5, 999]:
        know = D.clamp_conocimiento(raw)
        inter = D.clamp_interes(raw)
        assert 0.0 <= D.pct_conocimiento(know) <= 100.0
        assert 0.0 <= D.pct_interes(inter) <= 100.0


@pytest.mark.parametrize("mean,esperado", [
    (0.9, "Alto"),
    (D.UMBRAL_ALTO, "Alto"),
    (D.UMBRAL_ALTO - 0.001, "Medio"),
    (D.UMBRAL_MEDIO, "Medio"),
    (D.UMBRAL_MEDIO - 0.001, "Básico"),
    (D.UMBRAL_BASICO, "Básico"),
    (D.UMBRAL_BASICO - 0.001, "Muy bajo"),
])
def test_nivel_label_bordes(mean, esperado):
    """``nivel_label`` clasifica correctamente, con bordes inclusivos.

    Args:
        mean (float): Media de conocimiento a clasificar.
        esperado (str): Etiqueta esperada.
    """
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
    """``var_label`` clasifica correctamente la confianza en los bordes.

    Args:
        variance (float): Varianza a clasificar.
        esperado (str): Etiqueta esperada.
    """
    lbl, _ = D.var_label(variance)
    assert lbl == esperado


def test_interes_label_bordes():
    """``interes_label`` clasifica correctamente en los bordes de interés."""
    assert D.interes_label(D.UMBRAL_INTERES_ALTO)[0] == "Alto"
    assert D.interes_label(D.UMBRAL_INTERES_MEDIO)[0] == "Medio"
    assert D.interes_label(D.UMBRAL_INTERES_MEDIO - 0.01)[0] == "Bajo"


def test_categoria_dominio_coincide_con_nivel_label():
    """``categoria_dominio`` y ``nivel_label`` clasifican igual en las 4 categorías.

    ``categoria_dominio`` se usa en el grafo de temas y en el sunburst de
    ``temas.py``; ``nivel_label``, en el resto de la interfaz. Si divergen, un
    tema aparece con un color en un gráfico y con otro en otro, o desaparece de
    alguno. Antes ``categoria_dominio`` solo tenía 3 categorías y los temas
    "Muy bajo" quedaban como "Básico".
    """
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
    """Las cuatro etiquetas de ``nivel_label`` usan clases CSS distintas.

    Antes "Básico" y "Muy bajo" compartían la clase ``b-lo``, por lo que un
    tema con muy poco conocimiento se veía idéntico a uno básico.
    """
    muestras = [D.UMBRAL_ALTO + 0.01, D.UMBRAL_MEDIO + 0.001,
                D.UMBRAL_BASICO + 0.001, D.UMBRAL_BASICO - 0.001]
    clases = [D.nivel_label(m)[1] for m in muestras]
    assert len(set(clases)) == 4, f"Clases de nivel_label no son todas distintas: {clases}"


def test_var_label_tiene_3_colores_distintos_y_propios():
    """Las tres etiquetas de ``var_label`` usan clases CSS distintas y propias.

    Antes "Muy seguro" y "Seguro" compartían la clase ``b-hi``, que además era
    la de "Alto" en ``nivel_label``: una reutilización cruzada entre escalas
    sin relación.
    """
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


def test_umbrales_conocimiento_estan_dentro_del_rango_y_ordenados():
    """Los umbrales de conocimiento están dentro del rango y en orden creciente.

    Si falla, algún umbral quedó inalcanzable (sobre ``KNOW_MAX``) o redundante
    (bajo ``KNOW_MIN``) tras una recalibración.
    """
    assert D.KNOW_MIN < D.UMBRAL_BASICO < D.UMBRAL_MEDIO < D.UMBRAL_ALTO < D.KNOW_MAX, (
        "Los umbrales de conocimiento deben caer estrictamente DENTRO de "
        "[KNOW_MIN, KNOW_MAX] y en orden creciente. Si esto falla, algún "
        "umbral quedó inalcanzable (por encima de KNOW_MAX) o redundante "
        "(por debajo de KNOW_MIN) tras una recalibración — el bug exacto "
        "del punto 18."
    )


def test_umbrales_interes_estan_dentro_del_rango_y_ordenados():
    """Los umbrales de interés están dentro del rango y en orden creciente."""
    assert D.INT_MIN < D.UMBRAL_INTERES_MEDIO < D.UMBRAL_INTERES_ALTO < D.INT_MAX, (
        "Los umbrales de interés deben caer estrictamente dentro de "
        "[INT_MIN, INT_MAX] y en orden creciente."
    )


def test_umbrales_varianza_estan_dentro_del_rango_y_ordenados():
    """Los umbrales de confianza están dentro del rango de varianza y ordenados."""
    assert D.VAR_MIN <= D.UMBRAL_VAR_SEGURO < D.UMBRAL_VAR_MODERADO <= D.VAR_MAX, (
        "Los umbrales de confianza deben caer dentro de [VAR_MIN, VAR_MAX] "
        "y en orden creciente."
    )


def test_umbrales_pct_estan_en_rango_0_100():
    """Los umbrales expresados en porcentaje están entre 0 y 100."""
    for pct in (D.UMBRAL_ALTO_PCT, D.UMBRAL_MEDIO_PCT, D.UMBRAL_BASICO_PCT,
                D.UMBRAL_INTERES_ALTO_PCT, D.UMBRAL_INTERES_MEDIO_PCT):
        assert 0.0 <= pct <= 100.0


def test_pct_inicio_estan_en_rango_0_100():
    """Los porcentajes de inicio (media 0.0, varianza 0.5) son válidos.

    El estado neutro previo a cualquier evento debe normalizar a un valor entre
    0 y 100 sin importar la calibración. Antes estaban escritos a mano
    ("25.0" y "16.7") en tres archivos y ninguno se actualizó al recalibrar.
    """
    assert 0.0 <= D.PCT_INICIO_CONOCIMIENTO <= 100.0
    assert 0.0 <= D.PCT_INICIO_INTERES <= 100.0
    assert D.PCT_INICIO_CERTEZA == 50.0


def test_pct_inicio_coincide_con_las_funciones_de_normalizacion():
    """Los porcentajes de inicio coinciden con las funciones de las que derivan.

    No detecta valores escritos a mano en otros archivos, pero garantiza que
    las constantes nunca queden desincronizadas de sus funciones.
    """
    assert D.PCT_INICIO_CONOCIMIENTO == D.pct_conocimiento(0.0)
    assert D.PCT_INICIO_INTERES == D.pct_interes(0.0)
    assert D.PCT_INICIO_CERTEZA == D.pct_certeza(0.5)


def test_umbral_alto_clasifica_como_alto_y_basico_como_no_muy_bajo():
    """Cada umbral produce exactamente la etiqueta que su nombre promete."""
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
    """``pct_conocimiento`` no decrece al aumentar la media.

    Una recalibración que invierta ``KNOW_MIN`` y ``KNOW_MAX`` rompería esto de
    inmediato, en vez de producir un gráfico invertido sin aviso.

    Args:
        mean_lo (float): Media menor.
        mean_hi (float): Media mayor.
    """
    assert D.pct_conocimiento(mean_hi) >= D.pct_conocimiento(mean_lo)


@pytest.mark.parametrize("mean_lo,mean_hi", [(-2, -1), (-0.5, 0), (0, 0.5), (0.5, 2)])
def test_pct_interes_es_monotona_no_decreciente(mean_lo, mean_hi):
    """``pct_interes`` no decrece al aumentar la media.

    Args:
        mean_lo (float): Media menor.
        mean_hi (float): Media mayor.
    """
    assert D.pct_interes(mean_hi) >= D.pct_interes(mean_lo)


def test_know_min_menor_que_know_max():
    """``KNOW_MIN`` es menor que ``KNOW_MAX``."""
    assert D.KNOW_MIN < D.KNOW_MAX


def test_int_min_menor_que_int_max():
    """``INT_MIN`` es menor que ``INT_MAX``."""
    assert D.INT_MIN < D.INT_MAX


def test_var_min_menor_que_var_max():
    """``VAR_MIN`` es menor que ``VAR_MAX``."""
    assert D.VAR_MIN < D.VAR_MAX


def test_weight_min_menor_que_weight_max():
    """El mínimo de cada peso del modelo es menor que su máximo."""
    assert D.WEIGHT_NOVELTY_MIN < D.WEIGHT_NOVELTY_MAX
    assert D.WEIGHT_INTERES_MIN < D.WEIGHT_INTERES_MAX
    assert D.WEIGHT_BIAS_MIN < D.WEIGHT_BIAS_MAX


@pytest.mark.parametrize("fn,lo,hi", [
    (D.pct_peso_novelty, D.WEIGHT_NOVELTY_MIN, D.WEIGHT_NOVELTY_MAX),
    (D.pct_peso_interes, D.WEIGHT_INTERES_MIN, D.WEIGHT_INTERES_MAX),
    (D.pct_peso_actividad, D.WEIGHT_BIAS_MIN, D.WEIGHT_BIAS_MAX),
])
def test_pct_peso_extremos_y_acotado(fn, lo, hi):
    """Los extremos del rango de cada peso normalizan a 0 y 100.

    Args:
        fn (Callable[[float], float]): Función de normalización del peso.
        lo (float): Mínimo del rango del peso.
        hi (float): Máximo del rango del peso.
    """
    assert fn(lo) == 0.0
    assert fn(hi) == 100.0


@pytest.mark.parametrize("fn", [D.pct_peso_novelty, D.pct_peso_interes, D.pct_peso_actividad])
def test_pct_peso_nunca_sale_de_0_100(fn):
    """Los porcentajes de peso nunca salen de [0, 100].

    Antes, ``nov_pct``, ``int_pct`` y ``bias_pct`` en ``pages/pesos_ink.py`` no
    tenían límite y un aprendiz real con alto desenganche llegó a mostrar
    "-47.6%" y "-66.9%".

    Args:
        fn (Callable[[float], float]): Función de normalización del peso.
    """
    for mean in [-999, -10, -1, 0, 0.5, 1, 10, 999]:
        pct = fn(mean)
        assert 0.0 <= pct <= 100.0


def test_pct_inicio_pesos_estan_en_rango_0_100():
    """Los porcentajes de inicio de los pesos están entre 0 y 100."""
    assert 0.0 <= D.PCT_INICIO_PESO_NOVELTY <= 100.0
    assert 0.0 <= D.PCT_INICIO_PESO_INTERES <= 100.0
    assert 0.0 <= D.PCT_INICIO_PESO_ACTIVIDAD <= 100.0


def test_pct_inicio_pesos_coincide_con_las_funciones():
    """Los porcentajes de inicio de los pesos derivan de sus funciones.

    Se calculan a partir de las funciones y nunca se escriben a mano.
    """
    assert D.PCT_INICIO_PESO_NOVELTY == D.pct_peso_novelty(0.5)
    assert D.PCT_INICIO_PESO_INTERES == D.pct_peso_interes(0.5)
    assert D.PCT_INICIO_PESO_ACTIVIDAD == D.pct_peso_actividad(0.0)


def test_ejemplo_real_aprendiz_34953_queda_acotado():
    """Valores reales del aprendiz #34953 (80% o más de desenganche) quedan acotados.

    Mostraba "Interés previo: -47.6%" y "Actividad: -66.9%". Ahora ambos deben
    quedar en 0%, mientras que novelty, que no estaba en el extremo, no debe
    acotarse a 0 ni a 100.
    """
    assert D.pct_peso_interes(-0.476) == 0.0
    assert D.pct_peso_actividad(-1.169) == 0.0
    resultado_novelty = D.pct_peso_novelty(0.961)
    assert 0.0 < resultado_novelty < 100.0
