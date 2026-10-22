"""
Tests de integración de los endpoints Flask de ``logic/api_backend.py``.

Usan ``app.test_client()`` y reemplazan ``cargar_modelo``, ``guardar_modelo``,
``cargar_perfil`` y ``guardar_perfil`` con ``monkeypatch``, para no tocar disco
ni requerir un modelo ``.pkl`` real (que a su vez necesitaría ``truelearn`` y
el dataset PEEK).
"""
import copy
import os

import pytest

import logic.api_backend as backend
from tests.test_correccion import TOPIC, FakeClassifier, FakeKC


@pytest.fixture
def client():
    """Cliente de pruebas de Flask con ``TESTING`` activado.

    Returns:
        flask.testing.FlaskClient: Cliente para llamar a los endpoints.
    """
    backend.app.config["TESTING"] = True
    return backend.app.test_client()


def _perfil_base():
    """Construye un perfil de aprendiz con un único tema (``TOPIC``).

    Returns:
        dict: Perfil con la misma estructura que persiste el backend.
    """
    return {
        "user_id": 7,
        "number_of_engagements": 3,
        "number_of_non_engagements": 1,
        "weights": {
            "novelty":  {"mean": 0.5, "variance": 0.5},
            "interest": {"mean": 0.5, "variance": 0.5},
            "bias":     {"mean": 0.0, "variance": 0.5},
        },
        "session_history": [],
        "explicit_feedback": [],
        "knowledge_components": [
            {
                "topic_id": TOPIC, "topic_name": "Álgebra Lineal", "topic_url": "",
                "mean": 0.0, "variance": 0.5,
                "interest_mean": 0.0, "interest_var": 0.5,
                "timestamp": 0.0,
            }
        ],
    }


def test_perfil_404_si_no_existe(client, monkeypatch):
    """``GET /perfil/<uid>`` responde 404 si el perfil no existe."""
    monkeypatch.setattr(backend, "cargar_perfil", lambda uid: None)
    r = client.get("/perfil/7")
    assert r.status_code == 404


def test_perfil_200_si_existe(client, monkeypatch):
    """``GET /perfil/<uid>`` responde 200 con el perfil si existe."""
    monkeypatch.setattr(backend, "cargar_perfil", lambda uid: _perfil_base())
    r = client.get("/perfil/7")
    assert r.status_code == 200
    assert r.get_json()["user_id"] == 7


def test_feedback_sin_body_400(client):
    """``POST /feedback/<uid>`` responde 400 si no hay cuerpo."""
    r = client.post("/feedback/7")
    assert r.status_code == 400


def test_feedback_tipo_invalido_400(client, monkeypatch):
    """``POST /feedback/<uid>`` responde 400 si el tipo de corrección es inválido."""
    monkeypatch.setattr(backend, "cargar_modelo", lambda uid: FakeClassifier())
    monkeypatch.setattr(backend, "cargar_perfil", lambda uid: _perfil_base())
    r = client.post("/feedback/7", json={"topic_id": TOPIC, "tipo": "no existe"})
    assert r.status_code == 400


def test_feedback_modelo_no_encontrado_404(client, monkeypatch):
    """``POST /feedback/<uid>`` responde 404 si no existe el modelo del aprendiz."""
    monkeypatch.setattr(backend, "cargar_modelo", lambda uid: None)
    r = client.post("/feedback/7", json={"topic_id": TOPIC, "tipo": "sé más"})
    assert r.status_code == 404


def test_feedback_topic_inexistente_404(client, monkeypatch):
    """``POST /feedback/<uid>`` responde 404 si el tema no existe en el modelo.

    Regresión: antes respondía 200 sin haber cambiado nada.
    """
    clf = FakeClassifier(know_pairs=[], int_pairs=[])
    monkeypatch.setattr(backend, "cargar_modelo", lambda uid: clf)
    monkeypatch.setattr(backend, "cargar_perfil", lambda uid: _perfil_base())
    r = client.post("/feedback/7", json={"topic_id": 12345, "tipo": "sé más"})
    assert r.status_code == 404
    assert "error" in r.get_json()


def test_feedback_flujo_completo_aplica_correccion_y_guarda(client, monkeypatch):
    """Verifica el flujo completo: corrige el modelo y guarda el perfil.

    El registro de feedback debe incluir el valor resultante real
    (``mean_resultante``). Antes ``valor`` quedaba en None y cualquier vista
    que lo usara mostraba el mismo número para todas las correcciones.
    """
    kc_know = FakeKC(0.0, 0.5)
    kc_int  = FakeKC(0.0, 0.5)
    clf = FakeClassifier(know_pairs=[(TOPIC, kc_know)], int_pairs=[(TOPIC, kc_int)])

    guardados = {}

    monkeypatch.setattr(backend, "cargar_modelo", lambda uid: clf)
    monkeypatch.setattr(backend, "cargar_perfil", lambda uid: _perfil_base())
    monkeypatch.setattr(backend, "guardar_modelo", lambda uid, c: guardados.setdefault("modelo", c))
    monkeypatch.setattr(backend, "guardar_perfil", lambda uid, p: guardados.setdefault("perfil", copy.deepcopy(p)))

    r = client.post("/feedback/7", json={
        "topic_id": TOPIC, "topic_name": "Álgebra Lineal", "tipo": "sé más",
    })

    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True

    assert kc_know.mean > 0.0

    perfil_guardado = guardados["perfil"]
    assert len(perfil_guardado["explicit_feedback"]) == 1
    registro = perfil_guardado["explicit_feedback"][0]
    assert registro["tipo"] == "sé más"

    assert registro["mean_resultante"] == pytest.approx(kc_know.mean)
    assert "interest_mean_resultante" in registro


def test_feedback_topic_id_como_string_numerico_se_normaliza(client, monkeypatch):
    """Un ``topic_id`` enviado como texto numérico se normaliza a ``int``."""
    kc_know = FakeKC(0.0, 0.5)
    clf = FakeClassifier(know_pairs=[(TOPIC, kc_know)], int_pairs=[])
    monkeypatch.setattr(backend, "cargar_modelo", lambda uid: clf)
    monkeypatch.setattr(backend, "cargar_perfil", lambda uid: _perfil_base())
    monkeypatch.setattr(backend, "guardar_modelo", lambda uid, c: None)
    monkeypatch.setattr(backend, "guardar_perfil", lambda uid, p: None)

    r = client.post("/feedback/7", json={"topic_id": str(TOPIC), "tipo": "sé más"})
    assert r.status_code == 200
    assert kc_know.mean > 0.0


def test_entrenar_rechaza_si_ya_hay_uno_activo(client, monkeypatch):
    """``POST /entrenar`` rechaza iniciar otro entrenamiento si hay uno activo."""
    monkeypatch.setattr(backend, "_training_active", True)
    r = client.post("/entrenar")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is False
    assert "en curso" in body["mensaje"]


def test_log_entrenamiento_devuelve_estado(client):
    """``GET /log_entrenamiento`` devuelve las claves ``log``, ``done`` y ``active``."""
    r = client.get("/log_entrenamiento")
    assert r.status_code == 200
    body = r.get_json()
    assert "log" in body and "done" in body and "active" in body


def test_reemplazar_reintenta_si_windows_deniega_el_acceso(tmp_path, monkeypatch):
    """``_reemplazar_con_reintentos`` reintenta ante ``PermissionError`` y termina bien.

    En Windows, un antivirus o indexador puede tener abierto el destino unos
    milisegundos y ``os.replace`` falla con "Acceso denegado".

    Args:
        tmp_path (pathlib.Path): Directorio temporal con los archivos de prueba.
        monkeypatch (pytest.MonkeyPatch): Fixture para simular el bloqueo.
    """
    tmp = tmp_path / "modelo.pkl.tmp"
    destino = tmp_path / "modelo.pkl"
    tmp.write_text("nuevo")
    destino.write_text("viejo")
    real = os.replace
    llamadas = {"n": 0}

    def replace_inestable(origen, fin):
        """Falla las dos primeras veces y luego reemplaza de verdad."""
        llamadas["n"] += 1
        if llamadas["n"] < 3:
            raise PermissionError(5, "Acceso denegado")
        return real(origen, fin)

    monkeypatch.setattr(backend.os, "replace", replace_inestable)
    monkeypatch.setattr(backend.time, "sleep", lambda segundos: None)

    backend._reemplazar_con_reintentos(str(tmp), str(destino))

    assert llamadas["n"] == 3
    assert destino.read_text() == "nuevo"
    assert not tmp.exists()


def test_reemplazar_se_rinde_y_limpia_el_temporal(tmp_path, monkeypatch):
    """Si el bloqueo no cede, relanza ``PermissionError`` y borra el temporal.

    El archivo definitivo conserva su contenido anterior.

    Args:
        tmp_path (pathlib.Path): Directorio temporal con los archivos de prueba.
        monkeypatch (pytest.MonkeyPatch): Fixture para simular el bloqueo.
    """
    tmp = tmp_path / "perfil.json.tmp"
    destino = tmp_path / "perfil.json"
    tmp.write_text("nuevo")
    destino.write_text("viejo")

    def replace_bloqueado(origen, fin):
        """Simula un destino bloqueado de forma permanente."""
        raise PermissionError(5, "Acceso denegado")

    monkeypatch.setattr(backend.os, "replace", replace_bloqueado)
    monkeypatch.setattr(backend.time, "sleep", lambda segundos: None)

    with pytest.raises(PermissionError):
        backend._reemplazar_con_reintentos(str(tmp), str(destino), intentos=3)

    assert destino.read_text() == "viejo"
    assert not tmp.exists()



def test_feedback_valor_manual_ya_no_existe_400(client, monkeypatch):
    """«valor manual» se quitó: el backend lo rechaza como tipo inválido."""
    monkeypatch.setattr(backend, "cargar_modelo", lambda uid: FakeClassifier())
    monkeypatch.setattr(backend, "cargar_perfil", lambda uid: _perfil_base())
    r = client.post("/feedback/7", json={"topic_id": TOPIC, "tipo": "valor manual"})
    assert r.status_code == 400
    assert "tipo inválido" in r.get_json()["error"]
