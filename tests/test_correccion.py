"""
Tests de ``logic.api_backend.aplicar_correccion_directa``.

Es la función más delicada del sistema: aplica correcciones bayesianas
manuales sobre los Knowledge Components de TrueLearn. Se usa un clasificador
falso (``FakeClassifier`` y ``FakeKC``) que imita solo la interfaz necesaria:
``get_learner_model()`` con ``knowledge.topic_kc_pairs()`` y, en cada KC,
``mean``, ``variance`` y ``update()``. Así se prueba la lógica sin instalar
``truelearn`` ni descargar el dataset PEEK.

``api_backend`` no importa ``truelearn`` a nivel de módulo (el import está
dentro del endpoint ``/prediccion``), por lo que importarlo aquí es seguro.

Los tests del final verifican que las constantes de corrección sigan en el
orden de magnitud del impacto de un evento real medido con ``truelearn==1.1.0``:
conocimiento con delta de media de 0.20 a 0.24 y de varianza de 0.11 a 0.20;
interés con delta de media de 0.76 y de varianza de 0.20.
"""
import pytest

from logic import dominio as D
from logic.api_backend import TopicNoEncontradoError, aplicar_correccion_directa


class FakeKC:
    """Knowledge Component falso con ``mean``, ``variance`` y ``update``."""

    def __init__(self, mean, variance, timestamp=0.0):
        """Crea el componente.

        Args:
            mean (float): Media inicial.
            variance (float): Varianza inicial.
            timestamp (float): Marca de tiempo inicial.
        """
        self.mean = mean
        self.variance = variance
        self.timestamp = timestamp

    def update(self, mean=None, variance=None, timestamp=None):
        """Actualiza solo los campos que se reciben.

        Args:
            mean (float | None): Nueva media.
            variance (float | None): Nueva varianza.
            timestamp (float | None): Nueva marca de tiempo.
        """
        if mean is not None:
            self.mean = mean
        if variance is not None:
            self.variance = variance
        if timestamp is not None:
            self.timestamp = timestamp


class FakeKnowledge:
    """Conjunto de pares (tema, KC) con la interfaz ``topic_kc_pairs``."""

    def __init__(self, pairs):
        """Crea el conjunto.

        Args:
            pairs (Iterable[tuple[int, FakeKC]]): Pares (topic_id, KC).
        """
        self._pairs = list(pairs)

    def topic_kc_pairs(self):
        """Devuelve los pares (tema, KC).

        Returns:
            list[tuple[int, FakeKC]]: Pares almacenados.
        """
        return self._pairs


class FakeModel:
    """Modelo falso que expone un atributo ``knowledge``."""

    def __init__(self, knowledge):
        """Crea el modelo.

        Args:
            knowledge (FakeKnowledge): Conocimiento asociado al modelo.
        """
        self.knowledge = knowledge


class FakeWeights:
    """Peso falso con media y varianza."""

    def __init__(self, mean, variance):
        """Crea el peso.

        Args:
            mean (float): Media del peso.
            variance (float): Varianza del peso.
        """
        self.mean = mean
        self.variance = variance


class FakeLearnerMetaWeights:
    """Imita ``truelearn.models._weights.LearnerMetaWeights``."""

    def __init__(self, novelty=(0.5, 0.5), interest=(0.5, 0.5), bias=(0.0, 0.5)):
        """Crea los pesos meta del aprendiz.

        Args:
            novelty (tuple[float, float]): (media, varianza) del peso de novedad.
            interest (tuple[float, float]): (media, varianza) del peso de interés.
            bias (tuple[float, float]): (media, varianza) del peso de sesgo.
        """
        self.novelty_weights  = FakeWeights(*novelty)
        self.interest_weights = FakeWeights(*interest)
        self.bias_weights     = FakeWeights(*bias)


class FakeClassifier:
    """Imita la parte de ``INKClassifier`` que usa este proyecto.

    Sigue la API real verificada contra ``truelearn==1.1.0``:
    ``get_learner_model()`` devuelve una tupla de tres elementos
    (modelo de conocimiento, modelo de interés, pesos meta del aprendiz).
    """

    def __init__(self, know_pairs=(), int_pairs=(), meta_weights=None):
        """Crea el clasificador falso.

        Args:
            know_pairs (Iterable[tuple[int, FakeKC]]): Pares (tema, KC) de conocimiento.
            int_pairs (Iterable[tuple[int, FakeKC]]): Pares (tema, KC) de interés.
            meta_weights (FakeLearnerMetaWeights | None): Pesos meta; si es None
                se usan los valores por defecto.
        """
        self.knowledge_model = FakeModel(FakeKnowledge(know_pairs))
        self.interest_model  = FakeModel(FakeKnowledge(int_pairs))
        self.meta_weights     = meta_weights or FakeLearnerMetaWeights()

    def get_learner_model(self):
        """Devuelve el modelo del aprendiz.

        Returns:
            tuple: (modelo de conocimiento, modelo de interés, pesos meta).
        """
        return (self.knowledge_model, self.interest_model, self.meta_weights)


TOPIC = 42


def _classifier_con_topic(mean_know=0.0, var_know=0.5, mean_int=0.0, var_int=0.5):
    """Crea un clasificador falso con un único tema (``TOPIC``).

    Args:
        mean_know (float): Media inicial de conocimiento.
        var_know (float): Varianza inicial de conocimiento.
        mean_int (float): Media inicial de interés.
        var_int (float): Varianza inicial de interés.

    Returns:
        tuple[FakeClassifier, FakeKC, FakeKC]: El clasificador y los KC de
        conocimiento e interés para inspeccionarlos.
    """
    kc_know = FakeKC(mean_know, var_know)
    kc_int  = FakeKC(mean_int, var_int)
    clf = FakeClassifier(
        know_pairs=[(TOPIC, kc_know)],
        int_pairs=[(TOPIC, kc_int)],
    )
    return clf, kc_know, kc_int


def test_se_mas_sube_mean_y_baja_varianza():
    """"sé más" aumenta la media y reduce la varianza de conocimiento."""
    clf, kc_know, _ = _classifier_con_topic(mean_know=0.0, var_know=0.3)
    aplicar_correccion_directa(clf, TOPIC, "sé más")
    assert kc_know.mean > 0.0
    assert kc_know.variance < 0.3


def test_se_menos_baja_mean_y_baja_varianza():
    """"sé menos" reduce la media y también la varianza de conocimiento."""
    clf, kc_know, _ = _classifier_con_topic(mean_know=0.0, var_know=0.3)
    aplicar_correccion_directa(clf, TOPIC, "sé menos")
    assert kc_know.mean < 0.0
    assert kc_know.variance < 0.3


@pytest.mark.parametrize("tipo", ["sé más", "sé menos"])
def test_conocimiento_nunca_sale_del_rango_aunque_se_repita_muchas_veces(tipo):
    """El conocimiento permanece en rango tras 200 correcciones consecutivas.

    Regresión: en la versión original los límites de la función no coincidían
    con el rango que el frontend asumía al normalizar, lo que podía producir
    porcentajes negativos o mayores a 100. En el peor caso, el resultado debe
    estar siempre en ``[KNOW_MIN, KNOW_MAX]`` y normalizar a un porcentaje
    válido.

    Args:
        tipo (str): Tipo de corrección de conocimiento a repetir.
    """
    clf, kc_know, _ = _classifier_con_topic(mean_know=0.0, var_know=0.5)
    for _ in range(200):
        aplicar_correccion_directa(clf, TOPIC, tipo)
        assert D.KNOW_MIN <= kc_know.mean <= D.KNOW_MAX
        assert 0.0 <= D.pct_conocimiento(kc_know.mean) <= 100.0
        assert D.VAR_MIN <= kc_know.variance <= D.VAR_MAX


def test_me_interesa_sube_interes_y_baja_varianza():
    """"me interesa" aumenta la media y reduce la varianza de interés."""
    clf, _, kc_int = _classifier_con_topic(mean_int=0.0, var_int=0.3)
    aplicar_correccion_directa(clf, TOPIC, "me interesa")
    assert kc_int.mean > 0.0
    assert kc_int.variance < 0.3


def test_no_me_interesa_baja_interes_y_baja_varianza():
    """"no me interesa" reduce la media de interés y, como toda corrección
    explícita, también la varianza (sube la confianza del sistema)."""
    clf, _, kc_int = _classifier_con_topic(mean_int=0.0, var_int=0.3)
    aplicar_correccion_directa(clf, TOPIC, "no me interesa")
    assert kc_int.mean < 0.0
    assert kc_int.variance < 0.3


@pytest.mark.parametrize("tipo", ["me interesa", "no me interesa"])
def test_interes_nunca_sale_del_rango_aunque_se_repita_muchas_veces(tipo):
    """El interés permanece en rango tras 200 correcciones consecutivas.

    Args:
        tipo (str): Tipo de corrección de interés a repetir.
    """
    clf, _, kc_int = _classifier_con_topic(mean_int=0.0, var_int=0.3)
    for _ in range(200):
        aplicar_correccion_directa(clf, TOPIC, tipo)
        assert D.INT_MIN <= kc_int.mean <= D.INT_MAX
        assert 0.0 <= D.pct_interes(kc_int.mean) <= 100.0
        assert D.VAR_MIN <= kc_int.variance <= D.VAR_MAX


@pytest.mark.parametrize("tipo", ["sé más", "sé menos"])
def test_topic_inexistente_en_knowledge_lanza_error(tipo):
    """Un tema ausente del modelo de conocimiento lanza ``TopicNoEncontradoError``.

    Regresión: antes la función no hacía nada y el endpoint respondía 200 OK,
    dando la falsa impresión de que la corrección se había aplicado.

    Args:
        tipo (str): Tipo de corrección de conocimiento a aplicar.
    """
    clf = FakeClassifier(know_pairs=[], int_pairs=[])
    with pytest.raises(TopicNoEncontradoError):
        aplicar_correccion_directa(clf, TOPIC, tipo)


@pytest.mark.parametrize("tipo", ["me interesa", "no me interesa"])
def test_topic_inexistente_en_interest_lanza_error(tipo):
    """Un tema ausente del modelo de interés lanza ``TopicNoEncontradoError``.

    Args:
        tipo (str): Tipo de corrección de interés a aplicar.
    """
    clf = FakeClassifier(know_pairs=[], int_pairs=[])
    with pytest.raises(TopicNoEncontradoError):
        aplicar_correccion_directa(clf, TOPIC, tipo)


def test_no_afecta_otros_temas():
    """Una corrección sobre ``TOPIC`` no modifica los KC de otros temas."""
    kc_know_a = FakeKC(0.0, 0.5)
    kc_know_b = FakeKC(0.2, 0.4)
    clf = FakeClassifier(know_pairs=[(TOPIC, kc_know_a), (99, kc_know_b)])
    aplicar_correccion_directa(clf, TOPIC, "sé más")
    assert kc_know_b.mean == 0.2
    assert kc_know_b.variance == 0.4


def test_delta_conocimiento_es_del_orden_del_impacto_de_un_evento_real():
    """El delta de media de conocimiento es al menos 0.15.

    Un evento real fresco ronda 0.20; la corrección manual no debe ser una
    fracción minúscula de eso.
    """
    clf, kc_know, _ = _classifier_con_topic(mean_know=0.0, var_know=0.5)
    aplicar_correccion_directa(clf, TOPIC, "sé más")
    assert abs(kc_know.mean) >= 0.15


def test_delta_interes_es_del_orden_del_impacto_de_un_evento_real():
    """El delta de media de interés es al menos 0.5.

    Un evento real de interés ronda 0.76; con el valor anterior (0.30) la
    corrección quedaba desproporcionadamente chica frente al canal automático.
    """
    clf, _, kc_int = _classifier_con_topic(mean_int=0.0, var_int=0.5)
    aplicar_correccion_directa(clf, TOPIC, "me interesa")
    assert kc_int.mean >= 0.5


def test_reduccion_de_varianza_es_del_orden_del_impacto_de_un_evento_real():
    """La reducción de varianza es al menos 0.10.

    La caída real de varianza de un evento fresco ronda 0.10 a 0.20.
    """
    clf, kc_know, _ = _classifier_con_topic(mean_know=0.0, var_know=0.5)
    aplicar_correccion_directa(clf, TOPIC, "sé más")
    assert (0.5 - kc_know.variance) >= 0.10
