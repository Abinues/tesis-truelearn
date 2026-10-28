"""
Tests de la página Evolución: el conocimiento de cada sesión es un promedio.

El valor de una sesión es el nivel de conocimiento promedio de sus temas
activos. Debe ser el mismo en las tarjetas AL INICIO / AL FINAL, en la línea
del gráfico y en la tabla de datos detallados, y debe calcularse igual que el
«Nivel de conocimiento» de la página Resumen.
"""
import json
import os
import warnings

from streamlit.testing.v1 import AppTest

from logic.dominio import pct_conocimiento
from tests.test_correcciones_ui import _EVOLUCION, _RESUMEN, _app
from tests.test_pages_smoke import _perfil

_SESIONES = [
    {"session_id": 1, "timestamp": 1.0,
     "means": {"A": 0.20}, "variances": {"A": 0.4}},
    {"session_id": 2, "timestamp": 2.0,
     "means": {"A": 0.20, "B": -0.10}, "variances": {"A": 0.4, "B": 0.5}},
    {"session_id": 3, "timestamp": 3.0,
     "means": {"A": 0.30, "B": 0.00, "C": 0.10}, "variances": {"A": 0.3, "B": 0.4, "C": 0.5}},
]


def _esperados():
    """Promedio esperado por sesión: porcentaje del promedio de los ``mean``."""
    return [pct_conocimiento(sum(s["means"].values()) / len(s["means"])) for s in _SESIONES]


def _kc(nombre, media):
    """Knowledge Component mínimo con la media indicada."""
    return {"topic_id": hash(nombre) % 1000, "topic_name": nombre, "topic_url": "",
            "mean": media, "variance": 0.4,
            "interest_mean": 0.1, "interest_var": 0.4, "timestamp": 0.0}


def _perfil_con_sesiones():
    """Perfil cuyos conocimientos actuales coinciden con la última sesión."""
    perfil = _perfil(con_historial=True)
    perfil["session_history"] = [dict(s) for s in _SESIONES]
    perfil["knowledge_components"] = [_kc(t, m) for t, m in _SESIONES[-1]["means"].items()]
    return perfil


V_DETALLE = "DETALLE DE SESIONES"
V_EVOL = "EVOLUCIÓN DEL CONOCIMIENTO PROMEDIO DEL ESTUDIANTE"


def _correr(ruta, perfil, vista=None):
    """Ejecuta una página con el perfil dado y devuelve el ``AppTest``.

    Para la página de sesiones, ``vista`` elige la pestaña del selector
    (en la página de sesiones, por defecto se pide DETALLE DE SESIONES, que
    es donde están las tarjetas y la tabla).
    """
    at = _app(ruta, perfil)
    if ruta == _EVOLUCION and vista is None:
        vista = V_DETALLE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
        if vista:
            at.button_group(key="sesiones_vista").set_value(vista).run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    return at


def _serie_de_la_linea(at):
    """Devuelve los datos de la serie del gráfico de línea de la página."""
    for tipo in ("bidi_component", "component_instance"):
        for el in at.get(tipo):
            crudo = getattr(el.proto, "json", None) or getattr(el.proto, "json_args", None)
            if not crudo:
                continue
            opciones = json.loads(crudo).get("options", {})
            series = opciones.get("series", [])
            if "xAxis" in opciones and series and series[0].get("type") == "line":
                return series[0]["data"], opciones
    raise AssertionError("No se encontró el gráfico de línea de evolución")


def _metrica(at, etiqueta):
    """Valor mostrado por la tarjeta con esa etiqueta."""
    return next(m.value for m in at.metric if m.label == etiqueta)


def test_la_linea_dibuja_el_promedio_de_cada_sesion():
    at = _correr(_EVOLUCION, _perfil_con_sesiones(), V_EVOL)
    datos, opciones = _serie_de_la_linea(at)
    assert datos == _esperados()
    assert "promedio" in opciones["yAxis"]["name"].lower()


def test_las_tarjetas_muestran_el_promedio_al_inicio_y_al_final():
    """AL INICIO y AL FINAL son el promedio de la primera y de la última sesión."""
    at = _correr(_EVOLUCION, _perfil_con_sesiones())
    esperado = _esperados()
    assert _metrica(at, "AL INICIO") == f"{esperado[0]:.1f}%"
    assert _metrica(at, "AL FINAL") == f"{esperado[-1]:.1f}%"
    delta = next(m.delta for m in at.metric if m.label == "AL FINAL")
    assert delta == f"{esperado[-1] - esperado[0]:+.1f}%"


def test_al_inicio_coincide_con_la_primera_fila_de_la_tabla():
    """La tarjeta y la tabla usan el mismo valor para la primera sesión."""
    at = _correr(_EVOLUCION, _perfil_con_sesiones())
    assert f"{at.dataframe[0].value['Conocimiento promedio'].iloc[0]:.1f}%" == _metrica(at, "AL INICIO")


def test_la_tabla_muestra_conocimiento_promedio_y_temas_activos():
    at = _correr(_EVOLUCION, _perfil_con_sesiones())
    tabla = at.dataframe[0].value
    assert list(tabla.columns) == ["Sesión", "Conocimiento promedio", "Temas activos"]
    assert tabla["Sesión"].tolist() == [1, 2, 3]
    assert tabla["Conocimiento promedio"].round(1).tolist() == [round(v, 1) for v in _esperados()]
    assert tabla["Temas activos"].tolist() == [1, 2, 3]


def test_no_queda_el_conocimiento_acumulado_ni_los_puntos():
    at = _correr(_EVOLUCION, _perfil_con_sesiones())
    tabla = at.dataframe[0].value
    assert "Conocimiento acumulado" not in tabla.columns
    assert not any(str(v).endswith("pts") for v in tabla["Conocimiento promedio"])
    with open(_EVOLUCION, encoding="utf-8") as f:
        assert "máximo posible" not in f.read()


def test_al_final_coincide_con_el_nivel_de_conocimiento_de_resumen():
    perfil = _perfil_con_sesiones()
    evolucion = _correr(_EVOLUCION, perfil)
    resumen = _correr(_RESUMEN, perfil)
    assert _metrica(evolucion, "AL FINAL") == _metrica(resumen, "NIVEL DE CONOCIMIENTO")


def test_una_sesion_sin_temas_no_rompe_la_pagina():
    perfil = _perfil_con_sesiones()
    perfil["session_history"].insert(0, {"session_id": 0, "timestamp": 0.0, "means": {}, "variances": {}})
    at = _correr(_EVOLUCION, perfil, V_EVOL)
    datos, _ = _serie_de_la_linea(at)
    assert datos[0] == 0.0 and len(datos) == 4


def test_resumen_incluye_las_correcciones_y_evolucion_no():
    """Una corrección cambia el perfil actual (Resumen) pero no el historial de sesiones."""
    perfil = _perfil_con_sesiones()
    perfil["knowledge_components"][0]["mean"] = 0.39   # corrección aplicada sobre el tema A
    evolucion = _correr(_EVOLUCION, perfil)
    resumen = _correr(_RESUMEN, perfil)
    assert _metrica(evolucion, "AL FINAL") == f"{_esperados()[-1]:.1f}%"
    assert _metrica(resumen, "NIVEL DE CONOCIMIENTO") != _metrica(evolucion, "AL FINAL")
