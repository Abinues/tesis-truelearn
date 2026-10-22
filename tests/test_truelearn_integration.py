"""
Tests de integración contra la librería real ``truelearn``.

A diferencia del resto de la suite, que usa objetos falsos, aquí se validan los
supuestos de ``logic.modelo_utils`` y ``logic.api_backend`` con el
``INKClassifier`` y el ``KnowledgeComponent`` reales. El módulo se omite
automáticamente si ``truelearn`` no está instalado.

Instalación mínima probada en Python 3.12, donde ``pip install truelearn``
completo falla porque la dependencia opcional ``wordcloud`` no compila:

    pip install --no-deps truelearn
    pip install trueskill mpmath

Alcanza para entrenar y usar ``INKClassifier``, pero no para las utilidades de
graficado de ``truelearn``.
"""
import pytest

truelearn = pytest.importorskip("truelearn", reason="truelearn no está instalado")

from truelearn.learning import INKClassifier  # noqa: E402

from logic.modelo_utils import extraer_pesos_reales  # noqa: E402


def test_get_learner_model_devuelve_tres_elementos():
    """Verifica que ``get_learner_model()`` devuelve tres elementos.

    La tupla es (modelo de conocimiento, modelo de interés, pesos meta del
    aprendiz). Si una versión futura de ``truelearn`` cambia su orden o su
    cantidad, este test es el primero en fallar.
    """
    clf = INKClassifier()
    models = clf.get_learner_model()
    assert len(models) == 3
    assert hasattr(models[0], "knowledge")
    assert hasattr(models[1], "knowledge")
    assert hasattr(models[2], "novelty_weights")
    assert hasattr(models[2], "interest_weights")
    assert hasattr(models[2], "bias_weights")


def test_extraer_pesos_reales_funciona_con_el_classifier_real():
    """Verifica ``extraer_pesos_reales`` con un ``INKClassifier`` sin entrenar.

    La versión original leía el atributo privado ``_learner_meta_weights``;
    ahora se usa la vía pública ``get_learner_model()[2]``.
    """
    clf = INKClassifier()
    pesos = extraer_pesos_reales(clf)

    for factor in ("novelty", "interest", "bias"):
        assert "mean" in pesos[factor]
        assert "variance" in pesos[factor]
        assert isinstance(pesos[factor]["mean"], float)
        assert isinstance(pesos[factor]["variance"], float)


def test_knowledge_component_tiene_la_interfaz_que_usa_aplicar_correccion():
    """Verifica la interfaz de ``KnowledgeComponent`` que usa la corrección.

    ``aplicar_correccion_directa`` llama a
    ``kc.update(mean=..., variance=..., timestamp=...)`` y lee ``kc.mean`` y
    ``kc.variance``.
    """
    from truelearn.models import KnowledgeComponent

    kc = KnowledgeComponent(mean=0.2, variance=0.4)
    assert hasattr(kc, "update")
    kc.update(mean=0.9, variance=0.05, timestamp=123.0)
    assert kc.mean == 0.9
    assert kc.variance == 0.05
