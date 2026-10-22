"""
Tests de ``shared.get_usuarios()`` y ``shared.get_perfil(uid)`` con ``AppTest``.

A diferencia de ``post_feedback`` y ``get_prediccion`` (ver
``test_shared_http.py``), estas dos funciones llaman a ``st.error()`` y
``st.stop()`` en sus ramas de error, por lo que requieren un contexto real de
Streamlit. Se ejecutan scripts mínimos de ``tests/fixtures/`` con
``requests.get`` reemplazado.

``st.stop()`` no lanza excepción: corta el script en ese punto, de modo que la
línea que guarda el resultado en ``session_state`` no se alcanza en las ramas
de error, y eso es lo que se verifica.
"""
import os

import pytest
import requests
from streamlit.testing.v1 import AppTest

import shared

_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


class _FakeResponse:
    """Respuesta HTTP simulada con la interfaz mínima de ``requests.Response``."""

    def __init__(self, status_code=200, json_data=None):
        """Crea la respuesta simulada.

        Args:
            status_code (int): Código de estado HTTP.
            json_data (Any | None): Cuerpo JSON; si es None se usa ``{}``.
        """
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._json_data = json_data if json_data is not None else {}

    def json(self):
        """Devuelve el cuerpo JSON simulado.

        Returns:
            Any: Los datos recibidos en el constructor.
        """
        return self._json_data


@pytest.fixture
def mock_get(monkeypatch):
    """Permite definir qué devuelve o lanza ``shared.requests.get``.

    Solo afecta al módulo ``shared``, no al ``requests.get`` global.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture para reemplazar atributos.

    Returns:
        Callable: Función que recibe el reemplazo de ``requests.get``.
    """
    def _aplicar(fn):
        """Reemplaza ``shared.requests.get`` por ``fn``."""
        monkeypatch.setattr(shared.requests, "get", fn)
    return _aplicar


def test_get_usuarios_exito(mock_get):
    """Verifica que ``get_usuarios`` devuelve la lista y no muestra errores."""
    mock_get(lambda *a, **k: _FakeResponse(200, [{"user_id": 1}]))
    at = AppTest.from_file(os.path.join(_FIXTURES, "_app_get_usuarios.py"))
    at.run()
    assert not at.exception
    assert at.session_state["resultado"] == [{"user_id": 1}]
    assert len(at.error) == 0


def test_get_usuarios_backend_desconectado_muestra_error_y_corta(mock_get):
    """Verifica el mensaje de conexión y que el script se detiene."""
    def fake_get(*a, **k):
        """Simula un backend inalcanzable."""
        raise requests.exceptions.ConnectionError("sin conexión")
    mock_get(fake_get)

    at = AppTest.from_file(os.path.join(_FIXTURES, "_app_get_usuarios.py"))
    at.run()

    assert not at.exception
    assert len(at.error) == 1
    assert "no se puede conectar" in at.error[0].value.lower()
    with pytest.raises(KeyError):
        _ = at.session_state["resultado"]


def test_get_usuarios_error_generico_muestra_mensaje_y_corta(mock_get):
    """Verifica el mensaje de error inesperado ante una excepción genérica."""
    def fake_get(*a, **k):
        """Simula una excepción inesperada."""
        raise ValueError("algo salió mal")
    mock_get(fake_get)

    at = AppTest.from_file(os.path.join(_FIXTURES, "_app_get_usuarios.py"))
    at.run()

    assert not at.exception
    assert len(at.error) == 1
    assert "error inesperado" in at.error[0].value.lower()


def test_get_perfil_exito(mock_get):
    """Verifica que ``get_perfil`` devuelve el perfil del aprendiz pedido."""
    mock_get(lambda *a, **k: _FakeResponse(200, {"user_id": 7, "knowledge_components": []}))
    at = AppTest.from_file(os.path.join(_FIXTURES, "_app_get_perfil.py"))
    at.session_state["uid_a_pedir"] = 7
    at.run()
    assert not at.exception
    assert at.session_state["resultado"]["user_id"] == 7


def test_get_perfil_404_muestra_error_y_corta(mock_get):
    """Verifica el mensaje cuando el perfil no existe (404) y que el script se detiene."""
    mock_get(lambda *a, **k: _FakeResponse(404))
    at = AppTest.from_file(os.path.join(_FIXTURES, "_app_get_perfil.py"))
    at.session_state["uid_a_pedir"] = 999
    at.run()

    assert not at.exception
    assert len(at.error) == 1
    assert "no existe perfil" in at.error[0].value.lower()
    with pytest.raises(KeyError):
        _ = at.session_state["resultado"]


def test_get_perfil_backend_desconectado(mock_get):
    """Verifica el mensaje de backend desconectado en ``get_perfil``."""
    def fake_get(*a, **k):
        """Simula un backend inalcanzable."""
        raise requests.exceptions.ConnectionError("sin conexión")
    mock_get(fake_get)

    at = AppTest.from_file(os.path.join(_FIXTURES, "_app_get_perfil.py"))
    at.session_state["uid_a_pedir"] = 7
    at.run()

    assert not at.exception
    assert len(at.error) == 1
    assert "desconectado" in at.error[0].value.lower()


def test_get_perfil_error_generico(mock_get):
    """Verifica que una excepción genérica se muestra como error."""
    def fake_get(*a, **k):
        """Simula una respuesta corrupta."""
        raise ValueError("json corrupto")
    mock_get(fake_get)

    at = AppTest.from_file(os.path.join(_FIXTURES, "_app_get_perfil.py"))
    at.session_state["uid_a_pedir"] = 7
    at.run()

    assert not at.exception
    assert len(at.error) == 1
    assert "error" in at.error[0].value.lower()
