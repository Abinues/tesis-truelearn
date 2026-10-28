"""
Tests de regresión de tres defectos de la interfaz.

1. ``pages/temas.py``: el buscador interpretaba el texto como expresión
   regular, así que ``(ML`` lanzaba un error y ``C++`` filtraba mal.
2. ``pages/evolucion.py``: el rango de sesiones elegido en el control
   deslizante debe limitar exactamente las sesiones que dibuja la línea.
3. Las tarjetas de métricas dibujaban una curva que no correspondía a lo que
   decía la tarjeta (por ejemplo, la de interés mostraba la de conocimiento).
   Se quitaron las curvas de todas las tarjetas del dashboard.
"""
import json
import os
import warnings

import pytest
from streamlit.testing.v1 import AppTest

from logic import dominio as D
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
        at.button_group(key="temas_vista").set_value("TODOS LOS TEMAS").run()
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
        at.button_group(key="temas_vista").set_value("TODOS LOS TEMAS").run()
        at.text_input(key="mapa_busq").set_value("c++").run()
    assert not at.exception
    textos = " ".join(m.value for m in at.markdown) + " ".join(b.label for b in at.button)
    assert "C++ Básico" in textos
    assert "Cálculo" not in textos and "C Sharp" not in textos


@pytest.mark.parametrize("vista", [
    "TOP 5 TEMAS POR TIPO DE CONOCIMIENTO",
    "CONOCIMIENTO POR INTERÉS",
    "RED DE TEMAS DEL ESTUDIANTE",
    "TODOS LOS TEMAS",
])
def test_cada_vista_de_analisis_de_temas_se_dibuja(vista):
    """Ninguna de las cuatro vistas de «Análisis de temas» debe romperse."""
    perfil = _perfil()
    perfil["knowledge_components"] = [_kc(1, "C++ Básico"), _kc(2, "Cálculo"), _kc(3, "Intro (ML)")]
    at = _app(_TEMAS, perfil)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
        at.button_group(key="temas_vista").set_value(vista).run()
    assert not at.exception, at.exception[0].message if at.exception else ""


def test_tarjetas_de_temas_muestran_la_confianza_de_cada_variable():
    """Cada tarjeta muestra conocimiento e interés, cada uno con su «conf»."""
    perfil = _perfil()
    perfil["knowledge_components"] = [_kc(1, "Cálculo")]
    at = _app(_TEMAS, perfil)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
        at.button_group(key="temas_vista").set_value("TODOS LOS TEMAS").run()
    html = " ".join(m.value for m in at.markdown if not m.value.lstrip().startswith("<style>"))
    assert html.count("conf:") == 2
    assert "Seguro" not in html and "Incierto" not in html


def _perfil_con_etiquetas():
    """Cuatro temas con combinaciones distintas de etiqueta de conocimiento e interés."""
    def kc(tid, nombre, mean, interest_mean):
        return {**_kc(tid, nombre), "mean": mean, "interest_mean": interest_mean}
    perfil = _perfil()
    perfil["knowledge_components"] = [
        kc(1, "TemaAltoAlto",   D.UMBRAL_ALTO + 0.01,  D.UMBRAL_INTERES_ALTO + 0.1),
        kc(2, "TemaAltoBajo",   D.UMBRAL_ALTO + 0.01,  D.UMBRAL_INTERES_MEDIO - 0.5),
        kc(3, "TemaMedioMedio", D.UMBRAL_MEDIO + 0.001, (D.UMBRAL_INTERES_ALTO + D.UMBRAL_INTERES_MEDIO) / 2),
        kc(4, "TemaMuyBajoAlto", D.UMBRAL_BASICO - 0.1, D.UMBRAL_INTERES_ALTO + 0.1),
    ]
    return perfil


def _temas_visibles(at):
    """Nombres de los temas que se dibujan como tarjeta en «Todos los temas»."""
    texto = " ".join(m.value for m in at.markdown)
    return {n for n in ("TemaAltoAlto", "TemaAltoBajo", "TemaMedioMedio", "TemaMuyBajoAlto")
            if n in texto}


def _abrir_todos_los_temas():
    at = _app(_TEMAS, _perfil_con_etiquetas())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
        at.button_group(key="temas_vista").set_value("TODOS LOS TEMAS").run()
    assert not at.exception
    return at


def test_los_filtros_arrancan_en_todos_y_muestran_todos_los_temas():
    at = _abrir_todos_los_temas()
    assert at.multiselect(key="mapa_filtro_conocimiento").value == ["Todos"]
    assert at.multiselect(key="mapa_filtro_interes").value == ["Todos"]
    assert _temas_visibles(at) == {"TemaAltoAlto", "TemaAltoBajo", "TemaMedioMedio", "TemaMuyBajoAlto"}


def test_los_filtros_ofrecen_todos_mas_las_etiquetas_de_cada_variable():
    at = _abrir_todos_los_temas()
    assert at.multiselect(key="mapa_filtro_conocimiento").options == ["Todos", "Alto", "Medio", "Básico", "Muy bajo"]
    assert at.multiselect(key="mapa_filtro_interes").options == ["Todos", "Alto", "Medio", "Bajo"]


def test_filtrar_por_etiqueta_de_conocimiento():
    at = _abrir_todos_los_temas()
    at.multiselect(key="mapa_filtro_conocimiento").set_value(["Todos", "Alto"]).run()
    assert not at.exception
    assert at.multiselect(key="mapa_filtro_conocimiento").value == ["Alto"]   # «Todos» se quita solo
    assert _temas_visibles(at) == {"TemaAltoAlto", "TemaAltoBajo"}


def test_filtrar_por_etiqueta_de_interes():
    at = _abrir_todos_los_temas()
    at.multiselect(key="mapa_filtro_interes").set_value(["Todos", "Alto"]).run()
    assert _temas_visibles(at) == {"TemaAltoAlto", "TemaMuyBajoAlto"}


def test_los_dos_filtros_se_combinan_y_con_la_busqueda():
    at = _abrir_todos_los_temas()
    at.multiselect(key="mapa_filtro_conocimiento").set_value(["Todos", "Alto", "Muy bajo"]).run()
    at.multiselect(key="mapa_filtro_interes").set_value(["Todos", "Alto"]).run()
    assert _temas_visibles(at) == {"TemaAltoAlto", "TemaMuyBajoAlto"}
    at.text_input(key="mapa_busq").set_value("muybajo").run()
    assert _temas_visibles(at) == {"TemaMuyBajoAlto"}


def test_elegir_varias_etiquetas_incluye_cualquiera_de_ellas():
    at = _abrir_todos_los_temas()
    at.multiselect(key="mapa_filtro_conocimiento").set_value(["Todos", "Alto", "Medio"]).run()
    assert _temas_visibles(at) == {"TemaAltoAlto", "TemaAltoBajo", "TemaMedioMedio"}


def test_elegir_todos_de_nuevo_limpia_el_filtro():
    at = _abrir_todos_los_temas()
    at.multiselect(key="mapa_filtro_conocimiento").set_value(["Todos", "Alto"]).run()
    at.multiselect(key="mapa_filtro_conocimiento").set_value(["Alto", "Todos"]).run()
    assert at.multiselect(key="mapa_filtro_conocimiento").value == ["Todos"]
    assert len(_temas_visibles(at)) == 4


def test_vaciar_un_filtro_vuelve_a_todos():
    at = _abrir_todos_los_temas()
    at.multiselect(key="mapa_filtro_conocimiento").set_value(["Todos", "Alto"]).run()
    at.multiselect(key="mapa_filtro_conocimiento").set_value([]).run()
    assert at.multiselect(key="mapa_filtro_conocimiento").value == ["Todos"]
    assert len(_temas_visibles(at)) == 4


def test_filtros_sin_resultados_muestran_el_aviso():
    at = _abrir_todos_los_temas()
    at.multiselect(key="mapa_filtro_conocimiento").set_value(["Todos", "Medio"]).run()
    at.multiselect(key="mapa_filtro_interes").set_value(["Todos", "Bajo"]).run()
    assert not at.exception
    assert _temas_visibles(at) == set()
    assert any("No se encontraron temas" in i.value for i in at.info)


NOMBRE_LARGO = "Statistical Natural Language Processing and Computational Linguistics"


def _opciones_de_graficos(at):
    """Opciones de todos los gráficos ECharts que dibujó la página."""
    salida = []
    for tipo in ("bidi_component", "component_instance"):
        for el in at.get(tipo):
            crudo = getattr(el.proto, "json", None) or getattr(el.proto, "json_args", None)
            if crudo:
                opciones = json.loads(crudo).get("options")
                if opciones:
                    salida.append(opciones)
    return salida


def _nombres_en(obj):
    """Todos los valores de ``name`` (y de ``topic_name``) que aparecen en unas opciones."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "name" and isinstance(v, str):
                yield v
            yield from _nombres_en(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _nombres_en(v)


def _perfil_con_nombre_largo():
    perfil = _perfil()
    perfil["knowledge_components"] = [_kc(1, NOMBRE_LARGO), _kc(2, "Cálculo")]
    return perfil


@pytest.mark.parametrize("vista", [
    "TOP 5 TEMAS POR TIPO DE CONOCIMIENTO",
    "CONOCIMIENTO POR INTERÉS",
    "RED DE TEMAS DEL ESTUDIANTE",
])
def test_los_graficos_de_temas_guardan_el_nombre_completo(vista):
    """Los tooltips leen el nombre del dato: no debe llegar recortado."""
    at = _app(_TEMAS, _perfil_con_nombre_largo())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
        at.button_group(key="temas_vista").set_value(vista).run()
    assert not at.exception
    nombres = {n for o in _opciones_de_graficos(at) for n in _nombres_en(o)}
    assert NOMBRE_LARGO in nombres


def test_el_titulo_de_cada_tarjeta_muestra_el_nombre_completo():
    at = _app(_TEMAS, _perfil_con_nombre_largo())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
        at.button_group(key="temas_vista").set_value("TODOS LOS TEMAS").run()
    titulos = [m.value for m in at.markdown if NOMBRE_LARGO in m.value]
    assert titulos, "no se encontró el título completo"
    assert all("ellipsis" not in v and "nowrap" not in v for v in titulos)


def test_el_top_de_resumen_guarda_el_nombre_completo():
    at = _app(_RESUMEN, _perfil_con_nombre_largo())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
    assert not at.exception
    nombres = []
    for o in _opciones_de_graficos(at):
        y = o.get("yAxis")
        if isinstance(y, dict) and isinstance(y.get("data"), list):
            nombres += y["data"]
    assert NOMBRE_LARGO in nombres


def test_la_burbuja_seleccionada_se_busca_por_id_y_no_por_nombre():
    """Dos temas con el mismo nombre largo no deben mezclarse (se identifica por topic_id)."""
    at = _app(_TEMAS, _perfil_con_nombre_largo())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
        at.button_group(key="temas_vista").set_value("CONOCIMIENTO POR INTERÉS").run()
    serie = next(o["series"][0] for o in _opciones_de_graficos(at)
                 if o.get("series") and o["series"][0].get("type") == "scatter")
    assert {d["topic_id"] for d in serie["data"]} == {1, 2}


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


@pytest.mark.parametrize("ini,fin", [(1, 30), (26, 30), (21, 30), (6, 30), (10, 14)])
def test_filtro_de_sesiones_de_evolucion(ini, fin):
    """El control deslizante limita la línea de evolución al rango elegido."""
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
        at.button_group(key="sesiones_vista").set_value(
            "EVOLUCIÓN DEL CONOCIMIENTO PROMEDIO DEL ESTUDIANTE").run()
        at.select_slider(key="ses_rango_7").set_value((ini, fin)).run()
    assert not at.exception
    assert _puntos_de_la_linea_de_evolucion(at) == fin - ini + 1


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
        if ruta == _EVOLUCION:   # las tarjetas están en la vista DETALLE DE SESIONES
            at.button_group(key="sesiones_vista").set_value("DETALLE DE SESIONES").run()
    assert not at.exception
    assert len(at.metric) >= 2
    for m in at.metric:
        assert len(m.proto.chart_data) == 0, m.label