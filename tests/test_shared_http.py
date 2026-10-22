"""
Tests de los helpers HTTP de ``shared.py`` que no dependen de Streamlit.

``post_feedback``, ``get_prediccion`` y ``fmt_ts`` no llaman a funciones de
Streamlit ni en sus ramas de error, así que se prueban directamente con
``requests.get`` y ``requests.post`` reemplazados mediante ``monkeypatch``.
``get_usuarios`` y ``get_perfil`` se prueban en ``test_shared_apptest.py``.
"""
import requests

import shared


class _FakeResponse:
    """Respuesta HTTP simulada con la interfaz mínima de ``requests.Response``."""

    def __init__(self, status_code=200, json_data=None, text=""):
        """Crea la respuesta simulada.

        Args:
            status_code (int): Código de estado HTTP.
            json_data (Any | None): Cuerpo JSON; si es None se usa ``{}``.
            text (str): Cuerpo en texto plano.
        """
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._json_data = json_data if json_data is not None else {}
        self.text = text

    def json(self):
        """Devuelve el cuerpo JSON simulado.

        Returns:
            Any: Los datos recibidos en el constructor.
        """
        return self._json_data


def test_post_feedback_exito(monkeypatch):
    """Verifica que ``post_feedback`` devuelve el JSON de una respuesta exitosa."""
    monkeypatch.setattr(shared.requests, "post",
                         lambda *a, **k: _FakeResponse(200, {"ok": True, "mensaje": "listo"}))
    res = shared.post_feedback(7, 3, "Álgebra", "sé más")
    assert res == {"ok": True, "mensaje": "listo"}


def test_post_feedback_castea_topic_id_a_int(monkeypatch):
    """Verifica que un ``numpy.int64`` llega al cuerpo como ``int`` nativo.

    Si no se convierte, ``requests.post(json=...)`` no puede serializarlo.
    """
    capturado = {}

    def fake_post(url, json, timeout):
        """Captura el cuerpo enviado y devuelve una respuesta exitosa."""
        capturado["body"] = json
        return _FakeResponse(200, {"ok": True})

    monkeypatch.setattr(shared.requests, "post", fake_post)

    import numpy as np
    shared.post_feedback(7, np.int64(3), "Álgebra", "sé más")

    assert type(capturado["body"]["topic_id"]) is int
    assert capturado["body"]["topic_id"] == 3


def test_post_feedback_http_error(monkeypatch):
    """Verifica que un código HTTP de error se devuelve en la clave ``error``."""
    monkeypatch.setattr(shared.requests, "post",
                         lambda *a, **k: _FakeResponse(404, text="no encontrado"))
    res = shared.post_feedback(7, 3, "Álgebra", "sé más")
    assert "error" in res
    assert "404" in res["error"]


def test_post_feedback_backend_desconectado(monkeypatch):
    """Verifica el mensaje devuelto cuando no hay conexión con el backend."""
    def fake_post(*a, **k):
        """Simula un backend inalcanzable."""
        raise requests.exceptions.ConnectionError("no hay conexión")
    monkeypatch.setattr(shared.requests, "post", fake_post)
    res = shared.post_feedback(7, 3, "Álgebra", "sé más")
    assert res == {"error": "Backend desconectado."}


def test_post_feedback_excepcion_generica(monkeypatch):
    """Verifica que una excepción genérica se devuelve como mensaje de error."""
    def fake_post(*a, **k):
        """Simula una excepción inesperada."""
        raise ValueError("algo raro")
    monkeypatch.setattr(shared.requests, "post", fake_post)
    res = shared.post_feedback(7, 3, "Álgebra", "sé más")
    assert res == {"error": "algo raro"}


def test_get_prediccion_exito(monkeypatch):
    """Verifica que ``get_prediccion`` devuelve la probabilidad de enganche."""
    monkeypatch.setattr(shared.requests, "get",
                         lambda *a, **k: _FakeResponse(200, {"probabilidad_enganche": 0.73}))
    assert shared.get_prediccion(7, 3) == 0.73


def test_get_prediccion_falla_devuelve_none(monkeypatch):
    """Verifica que un fallo de red devuelve None."""
    def fake_get(*a, **k):
        """Simula un tiempo de espera agotado."""
        raise requests.exceptions.Timeout("tardó demasiado")
    monkeypatch.setattr(shared.requests, "get", fake_get)
    assert shared.get_prediccion(7, 3) is None


def test_get_prediccion_http_no_ok_devuelve_none(monkeypatch):
    """Verifica que una respuesta HTTP no exitosa devuelve None."""
    monkeypatch.setattr(shared.requests, "get", lambda *a, **k: _FakeResponse(500))
    assert shared.get_prediccion(7, 3) is None


def test_fmt_ts_valido():
    """Verifica el formato de un timestamp válido.

    ``fmt_ts`` usa ``datetime.fromtimestamp()``, que devuelve hora local, por
    lo que se compara contra el mismo cálculo local y no contra un texto fijo
    que dependería del huso horario de la máquina.
    """
    from datetime import datetime
    esperado = datetime.fromtimestamp(0).strftime("%d/%m/%Y %H:%M")
    assert shared.fmt_ts(0) == esperado


def test_fmt_ts_valido_con_timestamp_no_cero():
    """Verifica el formato de un timestamp distinto de cero."""
    from datetime import datetime
    ts = 1_700_000_000
    esperado = datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M")
    assert shared.fmt_ts(ts) == esperado


def test_fmt_ts_invalido_devuelve_guion():
    """Verifica que un valor no convertible devuelve un guion."""
    assert shared.fmt_ts("no-es-una-fecha") == "—"


def test_fmt_ts_none_devuelve_guion():
    """Verifica que None devuelve un guion."""
    assert shared.fmt_ts(None) == "—"
