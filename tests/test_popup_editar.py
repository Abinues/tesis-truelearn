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
    assert valores["Interés"] == f"{shared.pct_interes(0.4):.1f}%"
    assert "Confianza del sistema" not in valores


def test_popup_muestra_la_confianza_de_cada_variable_debajo_de_su_cuadro():
    """La confianza va en el caption de cada cuadro: la del conocimiento sale de
    ``variance`` y la del interés de ``interest_var`` (50% si el tema no la trae)."""
    at = _run()
    captions = [c.value for c in at.caption]
    assert any(f"{shared.pct_certeza(0.3):.1f}%" in c for c in captions)
    assert any(f"{shared.pct_certeza(0.5):.1f}%" in c for c in captions)

    at = _run({**KC_BASE, "interest_var": 0.2})
    captions = [c.value for c in at.caption]
    assert any(f"{shared.pct_certeza(0.2):.1f}%" in c for c in captions)


def test_popup_ordena_los_cuadros_conocimiento_interes():
    at = _run()
    assert [m.label for m in at.metric] == ["Conocimiento", "Interés"]


def test_popup_muestra_las_4_opciones_de_correccion():
    at = _run()
    opciones = at.radio[0].options
    assert opciones == ["sé más", "sé menos", "me interesa", "no me interesa"]


def test_popup_muestra_link_wikipedia_solo_si_hay_url():
    at_con_url = _run({**KC_BASE, "topic_url": "http://wikipedia.org/x"})
    assert any("wikipedia" in (m.value or "").lower() for m in at_con_url.markdown)

    at_sin_url = _run({**KC_BASE, "topic_url": ""})
    assert not any("wikipedia" in (m.value or "").lower() for m in at_sin_url.markdown)


# ── Cierre del popup (la X, o «Cerrar» una vez aplicada una corrección) ──

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

def _fakes(monkeypatch, capturado=None, perfil_nuevo=None):
    """Reemplaza las llamadas de red del popup por funciones falsas."""
    capturado = {} if capturado is None else capturado
    llamadas_prediccion = []

    def fake_post_feedback(uid, topic_id, topic_name, tipo):
        capturado["post_feedback_args"] = (uid, topic_id, topic_name, tipo)
        return {
            "ok": True, "mensaje": "Corrección aplicada.",
            "perfil": {"knowledge_components": [
                perfil_nuevo or {**KC_BASE, "mean": 0.35, "variance": 0.2,
                                 "interest_var": 0.3},
            ]},
        }

    def fake_prediccion(uid, topic_id):
        llamadas_prediccion.append((uid, topic_id))
        return 0.40 if len(llamadas_prediccion) == 1 else 0.55

    monkeypatch.setattr(shared, "post_feedback", fake_post_feedback)
    monkeypatch.setattr(shared, "get_prediccion", fake_prediccion)
    return capturado, llamadas_prediccion


def test_aplicar_exito_muestra_el_resultado_y_no_cierra_el_popup(monkeypatch):
    """
    Después de aplicar, el popup NO se cierra: reemplaza el selector por lo que
    cambió (métricas nuevas con su confianza, impacto en el modelo) y ofrece el
    botón «Cerrar». El estado del popup solo se limpia al cerrar.
    """
    capturado, llamadas_prediccion = _fakes(monkeypatch)

    at = _run()
    at.session_state["popup_topic"] = KC_BASE
    at.session_state["popup_open"] = True
    _boton(at, "Aplicar").click().run()

    assert not at.exception
    # El popup sigue abierto.
    assert at.session_state["popup_topic"] == KC_BASE
    assert at.session_state["popup_open"] is True
    # El backend se llamó con el tema y el tipo de corrección correctos.
    assert capturado["post_feedback_args"] == (7, 1, "Álgebra Lineal", "sé más")
    # El popup ya no consulta la probabilidad de enganche.
    assert llamadas_prediccion == []
    # Se ve el resultado: métricas nuevas, confianza nueva y botón Cerrar.
    valores = {m.label: m.value for m in at.metric}
    assert valores["Conocimiento ahora"] == f"{shared.pct_conocimiento(0.35):.1f}%"
    assert valores["Interés ahora"] == f"{shared.pct_interes(0.4):.1f}%"
    captions = [c.value for c in at.caption]
    assert any(f"{shared.pct_certeza(0.2):.1f}%" in c for c in captions)   # confianza nueva del conocimiento
    assert any("Cerrar" in b.label for b in at.button)
    # Ya no se ofrece corregir de nuevo en el mismo popup.
    assert not any("Aplicar" in b.label for b in at.button)
    assert not any("Impacto en el modelo" in m.value for m in at.markdown)
    assert not at.get("progress")   # sin la barra azul debajo de las etiquetas


def test_el_resultado_sobrevive_a_una_re_ejecucion_del_popup(monkeypatch):
    """Si el popup se vuelve a dibujar con el mismo tema, sigue el resultado."""
    _fakes(monkeypatch)
    at = _run()
    _boton(at, "Aplicar").click().run()
    at.run()
    assert not at.exception
    assert any("Cerrar" in b.label for b in at.button)
    assert not any("Aplicar" in b.label for b in at.button)


def test_cerrar_despues_de_aplicar_limpia_el_estado_del_popup(monkeypatch):
    _fakes(monkeypatch)
    at = _run()
    at.session_state["popup_topic"] = KC_BASE
    at.session_state["popup_open"] = True
    _boton(at, "Aplicar").click().run()
    _boton(at, "Cerrar").click().run()

    assert not at.exception
    assert at.session_state["popup_topic"] is None
    assert at.session_state["popup_open"] is False
    assert at.session_state["popup_resultado"] is None


def test_un_resultado_viejo_no_reaparece_al_reabrir_el_tema(monkeypatch):
    """Con otros valores de partida (tema ya corregido) vuelve el selector."""
    _fakes(monkeypatch)
    at = _run()
    _boton(at, "Aplicar").click().run()

    at2 = _run({**KC_BASE, "mean": 0.35, "variance": 0.2})
    at2.session_state["popup_resultado"] = at.session_state["popup_resultado"]
    at2.run()
    assert any("Aplicar" in b.label for b in at2.button)


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

def test_una_diferencia_en_cero_se_muestra_en_gris_y_sin_flecha(monkeypatch):
    """Si la corrección no cambia el valor, el delta va en gris y sin icono."""
    _fakes(monkeypatch, perfil_nuevo={**KC_BASE})   # el tema queda exactamente igual
    at = _run()
    _boton(at, "Aplicar").click().run()
    assert not at.exception
    ahora = [m for m in at.metric if m.label.endswith("ahora")]
    assert len(ahora) == 2
    for m in ahora:
        assert m.delta == "0.0%"
        assert m.proto.color == m.proto.GRAY          # en gris
        assert m.proto.direction == m.proto.NONE      # sin flecha de subir ni bajar
    captions = " ".join(c.value for c in at.caption)
    assert "pts" not in captions    # sin «(+0.0 pts)» cuando la confianza no cambió


def test_el_helper_de_delta_distingue_cero_de_cambio():
    assert shared._delta(0.04) == {"delta": "0.0%", "delta_color": "gray", "delta_arrow": "off"}
    assert shared._delta(-0.04)["delta_arrow"] == "off"
    assert shared._delta(2.34) == {"delta": "+2.3%"}
    assert shared._delta(-2.34) == {"delta": "-2.3%"}


def test_una_diferencia_distinta_de_cero_conserva_flecha_y_color(monkeypatch):
    _fakes(monkeypatch)   # el tema pasa a mean 0.35: conocimiento sube
    at = _run()
    _boton(at, "Aplicar").click().run()
    m = next(m for m in at.metric if m.label == "Conocimiento ahora")
    assert m.proto.direction == m.proto.UP
    assert m.proto.color == m.proto.GREEN
