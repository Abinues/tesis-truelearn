"""Utilidades compartidas sobre el INKClassifier de TrueLearn y el mapeo de temas de PEEK."""
from __future__ import annotations

from typing import Any


def extraer_pesos_reales(classifier: Any) -> dict[str, dict[str, float]]:
    """
    Extrae los pesos aprendidos de un INKClassifier entrenado.

    ``get_params()`` solo devuelve la configuración inicial, por lo que los
    pesos reales se obtienen del tercer elemento de ``get_learner_model()``
    (un ``LearnerMetaWeights``), que es la vía pública de la librería.

    Args:
        classifier (Any): Instancia de ``INKClassifier`` ya entrenada. Se
            tipa como ``Any`` porque ``truelearn`` no publica stubs.

    Returns:
        dict[str, dict[str, float]]: Diccionario con las claves ``"novelty"``,
        ``"interest"`` y ``"bias"``. Cada una contiene ``"mean"`` y
        ``"variance"`` del peso correspondiente.
    """
    lmw = classifier.get_learner_model()[2]
    return {
        "novelty": {
            "mean":     lmw.novelty_weights.mean,
            "variance": lmw.novelty_weights.variance,
        },
        "interest": {
            "mean":     lmw.interest_weights.mean,
            "variance": lmw.interest_weights.variance,
        },
        "bias": {
            "mean":     lmw.bias_weights.mean,
            "variance": lmw.bias_weights.variance,
        },
    }


def resolver_nombre_tema(topic_id: Any, mapping: dict[Any, Any]) -> str:
    """
    Obtiene el nombre de un tema sin importar el formato del mapeo.

    Acepta tuplas ``(nombre, url)`` o ``(url, nombre)`` y diccionarios con la
    clave ``"name"``.

    Args:
        topic_id (Any): Identificador del tema en el dataset.
        mapping (dict[Any, Any]): Mapeo ``topic_id`` -> metadatos del tema.

    Returns:
        str: Nombre del tema, o ``"ID_{topic_id}"`` si no se encuentra.
    """
    meta = mapping.get(topic_id)
    if isinstance(meta, tuple):
        if len(meta) > 1 and "http" in str(meta[0]):
            return str(meta[1])
        return str(meta[0])
    elif isinstance(meta, dict):
        return str(meta.get("name", f"ID_{topic_id}"))
    return f"ID_{topic_id}"


def resolver_metadata_tema(topic_id: Any, mapping: dict[Any, Any]) -> tuple[str, str]:
    """
    Obtiene el nombre y la URL de un tema sin importar el formato del mapeo.

    Acepta tuplas ``(nombre, url)`` o ``(url, nombre)`` y diccionarios con las
    claves ``"name"`` y ``"url"``.

    Args:
        topic_id (Any): Identificador del tema en el dataset.
        mapping (dict[Any, Any]): Mapeo ``topic_id`` -> metadatos del tema.

    Returns:
        tuple[str, str]: Par ``(nombre, url)``. Si el tema no existe devuelve
        ``("ID_{topic_id}", "")``.
    """
    meta = mapping.get(topic_id)

    if isinstance(meta, tuple):
        if len(meta) > 1 and "http" in str(meta[0]):
            return str(meta[1]), str(meta[0])
        return str(meta[0]), (str(meta[1]) if len(meta) > 1 else "")
    elif isinstance(meta, dict):
        return str(meta.get("name", f"ID_{topic_id}")), str(meta.get("url", ""))
    return f"ID_{topic_id}", ""
