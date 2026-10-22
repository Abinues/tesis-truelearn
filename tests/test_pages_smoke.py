"""
Tests de humo de las páginas de Streamlit con ``AppTest``.

Cada página se ejecuta con el script real y solo se verifica que no lance una
excepción no manejada. Nacen de un bug real: en ``pages/prediccion.py``,
``df_fb`` se definía solo dentro de la rama ``else`` de ``if not feedbacks:``
y se usaba fuera de ella, lo que provocaba ``NameError`` con un perfil recién
entrenado (``explicit_feedback == []``).

El ``session_state`` se inyecta directamente (``perfil_actual``,
``uid_seleccionado``), de modo que las páginas nunca llaman a ``get_perfil()``
y no hace falta un backend Flask ni red.

``login.py`` y ``dashboard.py`` no se cubren aquí: dependen de ``requests`` y
de ``st.navigation``, que requieren un montaje de prueba más elaborado.
"""
import os
import warnings

import pytest
from streamlit.testing.v1 import AppTest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAGINAS = [
    os.path.join(_ROOT, "pages", "resumen.py"),
    os.path.join(_ROOT, "pages", "temas.py"),
    os.path.join(_ROOT, "pages", "evolucion.py"),
    os.path.join(_ROOT, "pages", "pesos_ink.py"),
    os.path.join(_ROOT, "pages", "prediccion.py"),
    os.path.join(_ROOT, "pages", "historial.py"),
]


def _perfil(con_kcs=True, con_historial=False, con_feedback=False):
    """Construye un perfil de aprendiz sintético para inyectar en la sesión.

    Args:
        con_kcs (bool): Si es True, incluye un Knowledge Component de ejemplo.
        con_historial (bool): Si es True, incluye una entrada de
            ``session_history``.
        con_feedback (bool): Si es True, incluye una corrección en
            ``explicit_feedback``.

    Returns:
        dict: Perfil con la misma estructura que devuelve ``/perfil/<uid>``.
    """
    kcs = [{
        "topic_id": 1, "topic_name": "Álgebra Lineal", "topic_url": "",
        "mean": 0.2, "variance": 0.4,
        "interest_mean": 0.1, "interest_var": 0.4, "timestamp": 0.0,
    }] if con_kcs else []

    hist = [{
        "session_id": 1, "timestamp": 100.0,
        "means": {"Álgebra Lineal": 0.2}, "variances": {"Álgebra Lineal": 0.4},
    }] if con_historial else []

    fb = [{
        "timestamp": 100.0, "topic_id": 1, "topic_name": "Álgebra Lineal",
        "tipo": "sé más", "valor": None,
        "mean_resultante": 0.4, "interest_mean_resultante": 0.3,
    }] if con_feedback else []

    return {
        "user_id": 7,
        "number_of_engagements": 3,
        "number_of_non_engagements": 1,
        "weights": {
            "novelty":  {"mean": 0.5, "variance": 0.5},
            "interest": {"mean": 0.5, "variance": 0.5},
            "bias":     {"mean": 0.0, "variance": 0.5},
        },
        "session_history": hist,
        "explicit_feedback": fb,
        "knowledge_components": kcs,
    }


ESCENARIOS = {
    "recien_entrenado_sin_feedback": _perfil(con_kcs=True, con_historial=False, con_feedback=False),
    "sin_knowledge_components":      _perfil(con_kcs=False),
    "con_historial_y_feedback":      _perfil(con_kcs=True, con_historial=True, con_feedback=True),
}


@pytest.mark.parametrize("pagina", PAGINAS, ids=[os.path.basename(p) for p in PAGINAS])
@pytest.mark.parametrize("nombre_escenario", ESCENARIOS.keys())
def test_pagina_no_explota(pagina, nombre_escenario):
    """Verifica que cada página se ejecuta sin excepciones en cada escenario.

    Args:
        pagina (str): Ruta absoluta de la página a ejecutar.
        nombre_escenario (str): Clave de ``ESCENARIOS`` con el perfil a inyectar.
    """
    at = AppTest.from_file(pagina, default_timeout=15)
    at.session_state["uid_seleccionado"] = 7
    at.session_state["perfil_actual"]    = ESCENARIOS[nombre_escenario]
    at.session_state["popup_open"]       = False
    at.session_state["popup_topic"]      = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()

    assert not at.exception, (
        f"{pagina} lanzó una excepción no manejada con el escenario "
        f"'{nombre_escenario}': {at.exception[0].message if at.exception else ''}"
    )


def test_sin_perfil_seleccionado_muestra_mensaje_no_crashea():
    """Verifica que ninguna página falla si aún no hay un aprendiz elegido.

    Con el ``session_state`` vacío, cada página debe mostrar un
    ``st.info(...)`` y detenerse con ``st.stop()``.
    """
    for pagina in PAGINAS:
        at = AppTest.from_file(pagina, default_timeout=15)
        at.run()
        assert not at.exception, f"{pagina} explotó sin perfil seleccionado: {at.exception}"
