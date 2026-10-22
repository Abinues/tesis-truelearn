"""
Tests de regresión de tres defectos de la interfaz.

1. ``pages/temas.py``: el buscador interpretaba el texto como expresión
   regular, así que ``(ML`` lanzaba un error y ``C++`` filtraba mal.
2. ``pages/evolucion.py``: ``"5" in sel`` también era verdadero para
   "Últimas 25", que terminaba mostrando solo 5 sesiones.
3. Las tarjetas de métricas dibujaban una curva que no correspondía a lo que
   decía la tarjeta (por ejemplo, la de interés mostraba la de conocimiento).
   Se quitaron las curvas de todas las tarjetas del dashboard.
"""
import json
import os
import warnings

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_pages_smoke import _perfil

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEMAS = os.path.join(_ROOT, "pages", "temas.py")
_RESUMEN = os.path.join(_ROOT, "pages", "resumen.py")
_EVOLUCION = os.path.join(_ROOT, "pages", "evolucion.py")
_PESOS = os.path.join(_ROOT, "pages", "pesos_ink.py")


def _kc(tid, nombre):
    """Construye un Knowledge Component mínimo para el perfil de prueba."""
    return {
        "topic_id": tid, "topic_name": nombre, "topic_url": "",
        "mean": 0.2, "variance": 0.4,
        "interest_mean": 0.1, "interest_var": 0.4, "timestamp": 0.0,
    }


def _app(ruta, perfil):
    """Crea un ``AppTest`` de una página con el perfil inyectado en la sesión."""
    at = AppTest.from_file(ruta, default_timeout=15)
    at.session_state["uid_seleccionado"] = 7
    at.session_state["perfil_actual"] = perfil
    at.session_state["popup_open"] = False
    at.session_state["popup_topic"] = None
    return at


@pytest.mark.parametrize("busqueda", ["(ML", "C++", "[abc", "*", "\\"])
def test_buscador_trata_el_texto_como_literal(busqueda):
    """Caracteres especiales de regex no deben romper el buscador de temas."""
    perfil = _perfil()
    perfil["knowledge_components"] = [_kc(1, "C++ Básico"), _kc(2, "Cálculo"), _kc(3, "Intro (ML)")]
    at = _app(_TEMAS, perfil)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
        at.text_input(key="mapa_busq").set_value(busqueda).run()
    assert not at.exception, at.exception[0].message if at.exception else ""


def test_buscador_cpp_solo_encuentra_cpp():
    """Buscar ``C++`` debe dejar únicamente el tema que lo contiene."""
    perfil = _perfil()
    perfil["knowledge_components"] = [_kc(1, "C++ Básico"), _kc(2, "Cálculo"), _kc(3, "C Sharp")]
    at = _app(_TEMAS, perfil)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
        at.text_input(key="mapa_busq").set_value("c++").run()
    assert not at.exception
    textos = " ".join(m.value for m in at.markdown) + " ".join(b.label for b in at.button)
    assert "C++ Básico" in textos
    assert "Cálculo" not in textos and "C Sharp" not in textos


def _puntos_de_la_linea_de_evolucion(at):
    """Devuelve la cantidad de sesiones que dibuja la línea de evolución.

    Busca entre los componentes de ECharts de la página el gráfico de línea
    (el único con eje ``xAxis``) y cuenta los puntos de su serie. Lee tanto el
    formato de las versiones nuevas de Streamlit (``bidi_component``) como el
    de las anteriores (``component_instance``).

    Args:
        at (AppTest): Aplicación ya ejecutada.

    Returns:
        int: Cantidad de puntos de la serie de la línea.
    """
    for tipo in ("bidi_component", "component_instance"):
        for el in at.get(tipo):
            crudo = getattr(el.proto, "json", None) or getattr(el.proto, "json_args", None)
            if not crudo:
                continue
            opciones = json.loads(crudo).get("options", {})
            series = opciones.get("series", [])
            if "xAxis" in opciones and series and series[0].get("type") == "line":
                return len(series[0]["data"])
    raise AssertionError("No se encontró el gráfico de línea de evolución")


@pytest.mark.parametrize("opcion,esperado", [
    ("Todas las sesiones", 30),
    ("Últimas 5", 5),
    ("Últimas 10", 10),
    ("Últimas 25", 25),
])
def test_filtro_de_sesiones_de_evolucion(opcion, esperado):
    """"Últimas 25" debe mostrar 25 sesiones y no confundirse con "Últimas 5"."""
    perfil = _perfil(con_historial=True)
    perfil["session_history"] = [
        {"session_id": i, "timestamp": float(i),
         "means": {"Álgebra Lineal": 0.01 * i}, "variances": {"Álgebra Lineal": 0.4}}
        for i in range(1, 31)
    ]
    at = _app(_EVOLUCION, perfil)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
        at.radio(key="ses_lim_ev").set_value(opcion).run()
    assert not at.exception
    assert _puntos_de_la_linea_de_evolucion(at) == esperado


@pytest.mark.parametrize("ruta", [_RESUMEN, _EVOLUCION, _PESOS])
def test_ninguna_tarjeta_dibuja_curvas(ruta):
    """Ninguna tarjeta de métrica lleva curva (sparkline) en ninguna página."""
    perfil = _perfil(con_historial=True)
    perfil["session_history"].append({
        "session_id": 2, "timestamp": 200.0,
        "means": {"Álgebra Lineal": 0.5}, "variances": {"Álgebra Lineal": 0.3},
    })
    at = _app(ruta, perfil)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
    assert not at.exception
    assert len(at.metric) >= 2
    for m in at.metric:
        assert len(m.proto.chart_data) == 0, m.label