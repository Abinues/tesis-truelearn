"""
tests/test_popup_editar.py
─────────────────────────────────────────────────────────────────────────────
Tests de shared.popup_editar(): el popup de corrección explícita, el
mecanismo central de todo el proyecto (es literalmente el "feedback en
tiempo real" del título de la tesis). Antes de este archivo tenía 0% de
cobertura — ninguna página lo ejercitaba en los smoke tests porque todas
usan popup_open=False.

Se usa AppTest sobre un fixture mínimo (tests/fixtures/_app_popup_editar.py)
que llama a popup_editar directo. post_feedback y get_prediccion se
mockean (son funciones de red, ya cubiertas aparte en test_shared_http.py)
para poder verificar el flujo de UI sin depender de un backend real.
"""
import os

from streamlit.testing.v1 import AppTest

import shared

_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
_APP = os.path.join(_FIXTURES, "_app_popup_editar.py")

KC_BASE = {
    "topic_id": 1, "topic_name": "Álgebra Lineal", "topic_url": "",
    "mean": 0.1, "variance": 0.3, "interest_mean": 0.4,
}


def _run(kc_data=None, uid=7):
    at = AppTest.from_file(_APP, default_timeout=15)
    at.session_state["kc_data"] = kc_data or dict(KC_BASE)
    at.session_state["uid"] = uid
    at.run()
    return at


def _boton(at, texto):
    return next(b for b in at.button if texto in b.label)


# ── Renderizado básico ────────────────────────────────────────────────

def test_popup_muestra_las_metricas_correctas():
    at = _run()
    assert not at.exception
    valores = {m.label: m.value for m in at.metric}
    assert valores["Conocimiento"] == f"{shared.pct_conocimiento(0.1):.1f}%"
    assert valores["Confianza del sistema"] == f"{shared.pct_certeza(0.3):.1f}%"
    assert valores["Interés"] == f"{shared.pct_interes(0.4):.1f}%"


def test_popup_ordena_los_cuadros_conocimiento_interes_confianza():
    at = _run()
    assert [m.label for m in at.metric] == ["Conocimiento", "Interés", "Confianza del sistema"]


def test_popup_muestra_las_4_opciones_de_correccion():
    at = _run()
    opciones = at.radio[0].options
    assert opciones == ["sé más", "sé menos", "me interesa", "no me interesa"]


def test_popup_muestra_link_wikipedia_solo_si_hay_url():
    at_con_url = _run({**KC_BASE, "topic_url": "http://wikipedia.org/x"})
    assert any("wikipedia" in (m.value or "").lower() for m in at_con_url.markdown)

    at_sin_url = _run({**KC_BASE, "topic_url": ""})
    assert not any("wikipedia" in (m.value or "").lower() for m in at_sin_url.markdown)


# ── Cierre del popup (solo con la X de arriba a la derecha) ───────────

def test_no_hay_boton_cerrar():
    at = _run()
    assert not at.exception
    assert not any("Cerrar" in b.label for b in at.button)


def test_cerrar_con_la_x_limpia_el_estado_del_popup(monkeypatch):
    estado = {"popup_topic": KC_BASE, "popup_open": True}
    monkeypatch.setattr(shared.st, "session_state", estado)

    shared._cerrar_popup()

    assert estado["popup_topic"] is None
    assert estado["popup_open"] is False


# ── Botón "Aplicar": camino feliz ────────────────────────────────────

def test_aplicar_exito_llama_al_backend_y_resetea_popup(monkeypatch):
    """
    NOTA sobre st.rerun(): en el camino exitoso, popup_editar llama a
    st.rerun() al final, que AppTest procesa como una re-ejecución
    completa del script. Eso significa que st.success(...) y las
    métricas "nuevas" mostradas ANTES del rerun son transitorias y no
    sobreviven en el estado final de `at` — lo único verificable después
    de un rerun es lo que persiste en session_state, y que las funciones
    de red se llamaron con los argumentos correctos.
    """
    capturado = {}

    def fake_post_feedback(uid, topic_id, topic_name, tipo):
        capturado["post_feedback_args"] = (uid, topic_id, topic_name, tipo)
        return {
            "ok": True, "mensaje": "Corrección aplicada.",
            "perfil": {"knowledge_components": [
                {**KC_BASE, "mean": 0.35, "variance": 0.2},
            ]},
        }

    llamadas_prediccion = []

    def fake_prediccion(uid, topic_id):
        llamadas_prediccion.append((uid, topic_id))
        return 0.40 if len(llamadas_prediccion) == 1 else 0.55

    monkeypatch.setattr(shared, "post_feedback", fake_post_feedback)
    monkeypatch.setattr(shared, "get_prediccion", fake_prediccion)

    at = _run()
    at.session_state["popup_topic"] = KC_BASE
    at.session_state["popup_open"] = True
    _boton(at, "Aplicar").click().run()

    assert not at.exception
    # Popup reseteado: persiste después del rerun.
    assert at.session_state["popup_topic"] is None
    assert at.session_state["popup_open"] is False
    # El backend se llamó con el tema y el tipo de corrección correctos.
    assert capturado["post_feedback_args"] == (7, 1, "Álgebra Lineal", "sé más")
    # get_prediccion se llamó DOS veces (antes y después de aplicar).
    assert llamadas_prediccion == [(7, 1), (7, 1)]


def test_aplicar_error_muestra_st_error_y_no_resetea_popup(monkeypatch):
    monkeypatch.setattr(shared, "post_feedback",
                         lambda uid, topic_id, topic_name, tipo: {"error": "Backend desconectado."})
    monkeypatch.setattr(shared, "get_prediccion", lambda uid, topic_id: None)

    at = _run()
    at.session_state["popup_topic"] = KC_BASE
    at.session_state["popup_open"] = True
    _boton(at, "Aplicar").click().run()

    assert not at.exception
    assert len(at.error) == 1
    assert "backend desconectado" in at.error[0].value.lower()
    # No se resetea el popup si la corrección falló: el usuario sigue
    # viendo el mismo tema para poder reintentar.
    assert at.session_state["popup_topic"] == KC_BASE
    assert at.session_state["popup_open"] is True


def test_aplicar_envia_el_tipo_seleccionado_en_el_radio(monkeypatch):
    capturado = {}

    def fake_post_feedback(uid, topic_id, topic_name, tipo):
        capturado["tipo"] = tipo
        return {"ok": True, "mensaje": "listo", "perfil": {"knowledge_components": []}}

    monkeypatch.setattr(shared, "post_feedback", fake_post_feedback)
    monkeypatch.setattr(shared, "get_prediccion", lambda uid, topic_id: None)

    at = _run()
    at.radio[0].set_value("no me interesa").run()
    _boton(at, "Aplicar").click().run()

    assert capturado["tipo"] == "no me interesa"