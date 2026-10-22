"""
Tests de ``logic.modelo_utils``.

Cubre ``extraer_pesos_reales`` (lee ``get_learner_model()[2]``, la API pública,
en lugar del atributo privado ``_learner_meta_weights``), ``resolver_nombre_tema``
y ``resolver_metadata_tema`` con los tres formatos de metadatos de tema:
tupla (url, nombre), tupla (nombre, url) y diccionario.
"""
from logic.modelo_utils import extraer_pesos_reales, resolver_metadata_tema, resolver_nombre_tema
from tests.test_correccion import FakeClassifier, FakeLearnerMetaWeights


def test_extraer_pesos_reales_usa_la_api_publica():
    """Verifica que los pesos se leen de ``get_learner_model()[2]``."""
    meta = FakeLearnerMetaWeights(novelty=(0.62, 0.11), interest=(0.44, 0.2), bias=(0.05, 0.3))
    clf = FakeClassifier(meta_weights=meta)

    pesos = extraer_pesos_reales(clf)

    assert pesos["novelty"]  == {"mean": 0.62, "variance": 0.11}
    assert pesos["interest"] == {"mean": 0.44, "variance": 0.2}
    assert pesos["bias"]     == {"mean": 0.05, "variance": 0.3}


def test_extraer_pesos_reales_no_depende_del_atributo_privado():
    """Detecta si se reintroduce el acceso a ``_learner_meta_weights``.

    El fake no define ese atributo a propósito, por lo que
    ``extraer_pesos_reales`` debe funcionar sin él.
    """
    clf = FakeClassifier()
    assert not hasattr(clf, "_learner_meta_weights")
    pesos = extraer_pesos_reales(clf)
    assert set(pesos.keys()) == {"novelty", "interest", "bias"}


def test_resolver_nombre_tema_con_tupla_url_primero():
    """Resuelve el nombre cuando los metadatos son una tupla (url, nombre)."""
    mapping = {1: ("http://wikipedia.org/x", "Álgebra")}
    assert resolver_nombre_tema(1, mapping) == "Álgebra"


def test_resolver_nombre_tema_con_tupla_nombre_primero():
    """Resuelve el nombre cuando los metadatos son una tupla (nombre, url)."""
    mapping = {1: ("Álgebra", "http://wikipedia.org/x")}
    assert resolver_nombre_tema(1, mapping) == "Álgebra"


def test_resolver_nombre_tema_con_dict():
    """Resuelve el nombre cuando los metadatos son un diccionario."""
    mapping = {1: {"name": "Álgebra", "url": "http://wikipedia.org/x"}}
    assert resolver_nombre_tema(1, mapping) == "Álgebra"


def test_resolver_nombre_tema_desconocido():
    """Devuelve ``ID_<n>`` cuando el tema no está en el mapeo."""
    assert resolver_nombre_tema(999, {}) == "ID_999"


def test_resolver_metadata_tema_con_tupla_url_primero():
    """Devuelve (nombre, url) desde una tupla (url, nombre)."""
    mapping = {1: ("http://wikipedia.org/x", "Álgebra")}
    nombre, url = resolver_metadata_tema(1, mapping)
    assert nombre == "Álgebra"
    assert url == "http://wikipedia.org/x"


def test_resolver_metadata_tema_con_tupla_nombre_primero_sin_url():
    """Devuelve la url vacía cuando la tupla solo trae el nombre."""
    mapping = {1: ("Álgebra",)}
    nombre, url = resolver_metadata_tema(1, mapping)
    assert nombre == "Álgebra"
    assert url == ""


def test_resolver_metadata_tema_con_tupla_nombre_y_url():
    """Devuelve (nombre, url) desde una tupla (nombre, url)."""
    mapping = {1: ("Álgebra", "http://wikipedia.org/x")}
    nombre, url = resolver_metadata_tema(1, mapping)
    assert nombre == "Álgebra"
    assert url == "http://wikipedia.org/x"


def test_resolver_metadata_tema_con_dict():
    """Devuelve (nombre, url) desde un diccionario."""
    mapping = {1: {"name": "Álgebra", "url": "http://wikipedia.org/x"}}
    nombre, url = resolver_metadata_tema(1, mapping)
    assert nombre == "Álgebra"
    assert url == "http://wikipedia.org/x"


def test_resolver_metadata_tema_desconocido():
    """Devuelve ``ID_<n>`` y url vacía cuando el tema no está en el mapeo."""
    nombre, url = resolver_metadata_tema(999, {})
    assert nombre == "ID_999"
    assert url == ""
