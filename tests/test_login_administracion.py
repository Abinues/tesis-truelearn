"""
Tests de la sección «Administración del modelo» de la página de inicio.

La página habla con el backend por ``requests``; acá se reemplaza por un
backend falso que registra los pedidos y devuelve lo que cada test necesita.
"""
import os
import warnings

import pytest
import requests
from streamlit.testing.v1 import AppTest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGIN = os.path.join(_ROOT, "pages", "login.py")

_USUARIOS = [{"user_id": 101, "tipo": "muchas", "interacciones": 50,
              "engagements": 30, "sesiones": 12}]

_LOG_TERMINAL = [
    "--- INICIANDO MÓDULO DE ENTRENAMIENTO (multi-usuario) ---",
    "Cargando dataset PEEK...",
    "Usuarios con muchas interacciones (≥40): 6",
    "Usuarios con pocas  interacciones (≤10):  4",
    "Entrenando usuario 1000 [muchas] — 50 interacciones...",
    "   Modelo guardado: models/model_u_1000.pkl",
    "Entrenando usuario 1001 [muchas] — 50 interacciones...",
]


class _Respuesta:
    """Respuesta HTTP mínima."""

    def __init__(self, datos, ok=True, estado=200):
        self._datos, self.ok, self.status_code = datos, ok, estado
        self.content = b"x"

    def json(self):
        return self._datos


class BackendFalso:
    """Backend de mentira: guarda los pedidos y responde con lo configurado."""

    def __init__(self):
        self.pedidos = []
        self.log = {"log": [], "done": False, "active": False}
        self.estado = {"aprendices": 3, "feedbacks": 0, "ficticios": 0,
                       "con_feedbacks": 0, "sin_copia_limpia": 0, "entrenando": False}
        self.ficticios = {"ok": True, "aprendices": 3, "feedbacks": 15}
        self.restablecer = {"ok": True, "restaurados": 3, "feedbacks_eliminados": 15, "sin_copia": []}
        self.falla = None            # (ruta, estado, cuerpo) para simular un error

    def __call__(self, metodo, url, **kwargs):
        ruta = url.replace("http://localhost:5000", "")
        self.pedidos.append((metodo, ruta, kwargs.get("json")))
        if self.falla and ruta == self.falla[0]:
            return _Respuesta(self.falla[2], ok=False, estado=self.falla[1])
        if ruta == "/usuarios":
            return _Respuesta(_USUARIOS)
        if ruta == "/log_entrenamiento":
            return _Respuesta(self.log)
        if ruta == "/administracion/estado":
            return _Respuesta(self.estado)
        if ruta == "/administracion/feedbacks_ficticios":
            return _Respuesta(self.ficticios)
        if ruta == "/administracion/restablecer":
            return _Respuesta(self.restablecer)
        if ruta == "/entrenar":
            return _Respuesta({"ok": True})
        raise AssertionError(f"Pedido inesperado: {metodo} {ruta}")

    def hubo(self, metodo, ruta):
        return [p for p in self.pedidos if p[0] == metodo and p[1] == ruta]


@pytest.fixture
def backend(monkeypatch):
    falso = BackendFalso()
    monkeypatch.setattr(requests, "request", falso)
    monkeypatch.setattr(requests, "get", lambda url, **kw: falso("GET", url, **kw))
    return falso


def _pagina():
    at = AppTest.from_file(_LOGIN, default_timeout=15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at.run()
    assert not at.exception, at.exception[0].message if at.exception else ""
    return at


def _boton(at, nombre):
    return next(b for b in at.button if b.label == nombre)


def _textos(at):
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption]
                    + [s.value for s in at.success] + [e.value for e in at.error]
                    + [w.value for w in at.warning])


# ── Qué se muestra ──────────────────────────────────────────────────────────

def test_hay_tres_acciones_y_un_resumen(backend):
    backend.estado.update(aprendices=3, feedbacks=15, ficticios=5)
    at = _pagina()
    for nombre in ("Entrenar modelos desde PEEK", "Agregar feedbacks ficticios", "Restablecer todo"):
        assert not _boton(at, nombre).disabled
    assert "3 estudiantes · 15 feedbacks (5 ficticios)" in _textos(at)


def test_el_resumen_no_menciona_ficticios_si_no_hay(backend):
    at = _pagina()
    assert "3 estudiantes · 0 feedbacks" in _textos(at)
    assert "ficticios)" not in _textos(at)


def test_nunca_se_muestra_el_log_de_la_terminal(backend):
    backend.log = {"log": _LOG_TERMINAL, "done": True, "active": False}
    at = _pagina()
    assert len(at.get("code")) == 0
    assert "Entrenando usuario" not in _textos(at) and "Modelo guardado" not in _textos(at)


def test_mientras_se_entrena_se_ve_el_progreso_y_los_botones_se_bloquean(backend):
    backend.log = {"log": _LOG_TERMINAL, "done": False, "active": True}
    at = _pagina()
    barras = at.get("progress")
    assert len(barras) == 1 and barras[0].proto.text == "Entrenando estudiantes: 2 de 10"
    for nombre in ("Entrenar modelos desde PEEK", "Agregar feedbacks ficticios", "Restablecer todo"):
        assert _boton(at, nombre).disabled


def test_mientras_se_entrena_no_se_pide_el_estado_de_los_feedbacks(backend):
    backend.log = {"log": _LOG_TERMINAL, "done": False, "active": True}
    _pagina()
    assert backend.hubo("GET", "/administracion/estado") == []


def test_antes_de_empezar_a_entrenar_avisa_que_carga_el_dataset(backend):
    backend.log = {"log": ["Cargando dataset PEEK..."], "done": False, "active": True}
    at = _pagina()
    assert "cargando el dataset PEEK" in at.get("progress")[0].proto.text


def test_al_terminar_dice_solo_que_se_completo(backend):
    backend.log = {"log": _LOG_TERMINAL + ["--- PROCESO FINALIZADO CON ÉXITO ---"],
                   "done": True, "active": False}
    at = _pagina()
    assert "Último entrenamiento: completado." in _textos(at)
    assert len(at.get("progress")) == 0


def test_si_algunos_aprendices_fallan_lo_dice_en_una_linea(backend):
    backend.log = {"log": _LOG_TERMINAL + ["   Usuario 1001 FALLÓ, se lo saltea: x",
                                           "   Usuario 1002 FALLÓ, se lo saltea: y"],
                   "done": True, "active": False}
    assert "2 estudiantes fallaron" in _textos(_pagina())


def test_un_error_se_muestra_corto_y_sin_traceback(backend):
    backend.log = {"log": _LOG_TERMINAL + ["\n❌ ERROR CRÍTICO: no hay dataset",
                                           "Traceback (most recent call last):",
                                           '  File "x.py", line 1, in <module>'],
                   "done": True, "active": False}
    at = _pagina()
    assert [e.value for e in at.error] == ["El entrenamiento falló: ERROR CRÍTICO: no hay dataset"]
    assert "Traceback" not in _textos(at)


# ── Acciones ────────────────────────────────────────────────────────────────

def test_entrenar_lanza_el_entrenamiento(backend):
    at = _pagina()
    _boton(at, "Entrenar modelos desde PEEK").click().run()
    assert len(backend.hubo("POST", "/entrenar")) == 1


def test_agregar_ficticios_usa_la_cantidad_elegida_y_confirma(backend):
    at = _pagina()
    assert at.number_input(key="admin_cantidad").value == 5
    at.number_input(key="admin_cantidad").set_value(12).run()
    _boton(at, "Agregar feedbacks ficticios").click().run()
    assert backend.hubo("POST", "/administracion/feedbacks_ficticios")[0][2] == {"por_aprendiz": 12, "dias": 30}
    assert "Se agregaron 15 feedbacks ficticios a 3 estudiantes." in _textos(at)


def test_agregar_ficticios_usa_la_ventana_de_dias_elegida(backend):
    at = _pagina()
    assert at.number_input(key="admin_dias").value == 30
    at.number_input(key="admin_dias").set_value(90).run()
    _boton(at, "Agregar feedbacks ficticios").click().run()
    assert backend.hubo("POST", "/administracion/feedbacks_ficticios")[0][2] == {"por_aprendiz": 5, "dias": 90}


def test_agregar_ficticios_muestra_el_error_del_backend(backend):
    backend.falla = ("/administracion/feedbacks_ficticios", 409, {"error": "Hay un entrenamiento en curso."})
    at = _pagina()
    _boton(at, "Agregar feedbacks ficticios").click().run()
    assert "Hay un entrenamiento en curso." in [e.value for e in at.error]


def test_restablecer_no_borra_nada_hasta_confirmar(backend):
    at = _pagina()
    _boton(at, "Restablecer todo").click().run()
    assert backend.hubo("POST", "/administracion/restablecer") == []


def test_restablecer_confirmado_informa_lo_que_borro(backend):
    at = _pagina()
    _boton(at, "Restablecer todo").click().run()
    _boton(at, "Sí, restablecer").click().run()
    assert len(backend.hubo("POST", "/administracion/restablecer")) == 1
    assert ("Listo: se eliminaron 15 feedbacks y 3 estudiantes volvieron a su estado entrenado."
            in _textos(at))


def test_restablecer_avisa_de_quienes_no_tienen_copia_limpia(backend):
    backend.restablecer = {"ok": True, "restaurados": 1, "feedbacks_eliminados": 2, "sin_copia": [101, 102]}
    at = _pagina()
    _boton(at, "Restablecer todo").click().run()
    _boton(at, "Sí, restablecer").click().run()
    assert any("2 estudiantes tenían feedbacks y no tienen copia limpia" in w.value for w in at.warning)


def test_cancelar_no_restablece(backend):
    at = _pagina()
    _boton(at, "Restablecer todo").click().run()
    _boton(at, "Cancelar").click().run()
    assert backend.hubo("POST", "/administracion/restablecer") == []


def test_no_hay_emojis_en_la_pagina():
    with open(_LOGIN, encoding="utf-8") as f:
        texto = f.read()
    emojis = [c for c in texto if ord(c) >= 0x1F000 or 0x2600 <= ord(c) <= 0x27BF or c == "️"]
    assert emojis == []
