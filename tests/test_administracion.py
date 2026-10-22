"""
Tests de la administración del modelo (logic/administracion.py).

Usan aprendices entrenados de verdad con TrueLearn (pocos eventos sintéticos)
en una carpeta temporal, así que no tocan los datos del proyecto.
"""
import importlib.util
import json
import os
import pickle
import random
from datetime import datetime

import pytest

import logic.api_backend as backend
import logic.entrenador as entrenador
from logic import administracion

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_UIDS = (101, 102)


def _entrenar_aprendiz(uid, n_eventos, carpeta_datos, carpeta_modelos):
    """Entrena un aprendiz sintético y escribe su modelo y su perfil."""
    from truelearn.models import EventModel, Knowledge, KnowledgeComponent

    mapping = {i: (f"Tema {i}", "") for i in range(1, 9)}
    azar = random.Random(uid)
    eventos = []
    for t in range(n_eventos):
        tid = azar.randint(1, 8)
        kc = KnowledgeComponent(mean=0.5, variance=0.5)
        eventos.append((EventModel(Knowledge({tid: kc}), event_time=float(t)), azar.random() < 0.6))
    clasificador, historial, n_eng, n_non = entrenador.entrenar_usuario(uid, eventos, mapping, num_epochs=1)
    with open(os.path.join(carpeta_modelos, f"model_u_{uid}.pkl"), "wb") as f:
        pickle.dump(clasificador, f)
    perfil, _ = entrenador.generar_perfil_json(uid, clasificador, mapping, n_eng, n_non, historial)
    return {"user_id": uid, "tipo": "muchas", "interacciones": n_eventos,
            "engagements": n_eng, "sesiones": len(historial), "weights": perfil["weights"]}


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Carpeta temporal con dos aprendices entrenados y el backend apuntando a ella."""
    datos, modelos = tmp_path / "data", tmp_path / "models"
    datos.mkdir()
    modelos.mkdir()
    monkeypatch.setattr(entrenador, "DATA_DIR", str(datos))
    monkeypatch.setattr(backend, "DATA_DIR", str(datos))
    monkeypatch.setattr(backend, "MODEL_DIR", str(modelos))
    monkeypatch.setattr(backend, "_training_active", False)
    indice = [_entrenar_aprendiz(uid, 12, str(datos), str(modelos)) for uid in _UIDS]
    with open(datos / "usuarios_entrenados.json", "w", encoding="utf-8") as f:
        json.dump(indice, f)
    backend.app.config["TESTING"] = True
    return {"datos": datos, "modelos": modelos, "client": backend.app.test_client()}


def _perfil(entorno, uid):
    with open(entorno["datos"] / f"visual_profile_{uid}.json", encoding="utf-8") as f:
        return json.load(f)


def _medias(entorno, uid):
    """Medias de conocimiento e interés de cada tema del modelo guardado."""
    with open(entorno["modelos"] / f"model_u_{uid}.pkl", "rb") as f:
        modelo = pickle.load(f)
    conocimiento, interes = modelo.get_learner_model()[:2]
    return (sorted((t, round(kc.mean, 9)) for t, kc in conocimiento.knowledge.topic_kc_pairs()),
            sorted((t, round(kc.mean, 9)) for t, kc in interes.knowledge.topic_kc_pairs()))


def _feedback_real(entorno, uid):
    """Aplica un feedback real por el endpoint de siempre."""
    kc = _perfil(entorno, uid)["knowledge_components"][0]
    r = entorno["client"].post(f"/feedback/{uid}", json={
        "topic_id": kc["topic_id"], "topic_name": kc["topic_name"], "tipo": "sé más"})
    assert r.status_code == 200


# ── Estado ──────────────────────────────────────────────────────────────────

def test_estado_cuenta_aprendices_y_crea_la_copia_limpia(entorno):
    r = entorno["client"].get("/administracion/estado")
    datos = r.get_json()
    assert datos["aprendices"] == 2 and datos["feedbacks"] == 0 and datos["ficticios"] == 0
    assert datos["sin_copia_limpia"] == 0 and datos["entrenando"] is False
    for uid in _UIDS:
        assert (entorno["datos"] / "base" / f"visual_profile_{uid}.json").exists()
        assert (entorno["modelos"] / "base" / f"model_u_{uid}.pkl").exists()


def test_estado_no_toma_como_limpio_a_quien_ya_tiene_feedbacks(entorno):
    _feedback_real(entorno, 101)
    datos = entorno["client"].get("/administracion/estado").get_json()
    assert datos["con_feedbacks"] == 1 and datos["sin_copia_limpia"] == 1
    assert not (entorno["datos"] / "base" / "visual_profile_101.json").exists()


# ── Restablecer ─────────────────────────────────────────────────────────────

def test_restablecer_borra_feedbacks_y_su_efecto_en_el_modelo(entorno):
    cliente = entorno["client"]
    cliente.get("/administracion/estado")           # crea las copias limpias
    perfil_inicial = _perfil(entorno, 101)
    medias_iniciales = _medias(entorno, 101)

    _feedback_real(entorno, 101)
    assert len(_perfil(entorno, 101)["explicit_feedback"]) == 1
    assert _medias(entorno, 101) != medias_iniciales

    r = cliente.post("/administracion/restablecer")
    assert r.status_code == 200
    cuerpo = r.get_json()
    assert cuerpo["ok"] and cuerpo["restaurados"] == 2
    assert cuerpo["feedbacks_eliminados"] == 1 and cuerpo["sin_copia"] == []
    assert _perfil(entorno, 101) == perfil_inicial
    assert _medias(entorno, 101) == medias_iniciales


def test_restablecer_no_toca_a_quien_no_tiene_copia_limpia_y_lo_informa(entorno):
    _feedback_real(entorno, 101)                    # sin copia limpia previa
    perfil_con_feedback = _perfil(entorno, 101)
    cuerpo = entorno["client"].post("/administracion/restablecer").get_json()
    assert cuerpo["sin_copia"] == [101]
    assert _perfil(entorno, 101) == perfil_con_feedback


def test_restablecer_con_el_perfil_sin_cambios_no_rompe_nada(entorno):
    perfil = _perfil(entorno, 102)
    cuerpo = entorno["client"].post("/administracion/restablecer").get_json()
    assert cuerpo["ok"] and cuerpo["feedbacks_eliminados"] == 0
    assert _perfil(entorno, 102) == perfil


# ── Feedbacks ficticios ─────────────────────────────────────────────────────

def test_agrega_la_cantidad_pedida_por_aprendiz_y_los_marca_como_ficticios(entorno):
    r = entorno["client"].post("/administracion/feedbacks_ficticios", json={"por_aprendiz": 4, "semilla": 7})
    cuerpo = r.get_json()
    assert r.status_code == 200 and cuerpo["aprendices"] == 2 and cuerpo["feedbacks"] == 8
    for uid in _UIDS:
        feedbacks = _perfil(entorno, uid)["explicit_feedback"]
        assert len(feedbacks) == 4
        assert all(f["ficticio"] is True for f in feedbacks)
        assert all(f["tipo"] in administracion.TIPOS_FICTICIOS for f in feedbacks)
        assert all(f["mean_resultante"] is not None for f in feedbacks)


def test_los_ficticios_modifican_el_modelo_como_un_feedback_real(entorno):
    antes = _medias(entorno, 101)
    entorno["client"].post("/administracion/feedbacks_ficticios", json={"por_aprendiz": 6, "semilla": 1})
    assert _medias(entorno, 101) != antes


def test_con_la_misma_semilla_salen_los_mismos_feedbacks(entorno):
    cliente = entorno["client"]
    cliente.get("/administracion/estado")

    def corrida():
        cliente.post("/administracion/restablecer")
        cliente.post("/administracion/feedbacks_ficticios", json={"por_aprendiz": 5, "semilla": 3})
        return [(f["topic_id"], f["tipo"]) for f in _perfil(entorno, 101)["explicit_feedback"]]

    assert corrida() == corrida()


def test_los_ficticios_se_pueden_restablecer(entorno):
    perfil_inicial = _perfil(entorno, 101)
    entorno["client"].post("/administracion/feedbacks_ficticios", json={"por_aprendiz": 5})
    assert len(_perfil(entorno, 101)["explicit_feedback"]) == 5
    datos = entorno["client"].get("/administracion/estado").get_json()
    assert datos["feedbacks"] == 10 and datos["ficticios"] == 10
    entorno["client"].post("/administracion/restablecer")
    assert _perfil(entorno, 101) == perfil_inicial


def _fechas(entorno, uid):
    """Fechas (día calendario) de los feedbacks de un aprendiz, en el orden guardado."""
    return [datetime.fromtimestamp(f["timestamp"]) for f in _perfil(entorno, uid)["explicit_feedback"]]


def test_cada_feedback_ficticio_cae_en_un_dia_distinto(entorno):
    entorno["client"].post("/administracion/feedbacks_ficticios", json={"por_aprendiz": 10, "semilla": 2})
    for uid in _UIDS:
        fechas = _fechas(entorno, uid)
        assert len({f.date() for f in fechas}) == 10


def test_las_fechas_estan_en_los_ultimos_30_dias_y_ninguna_es_futura(entorno):
    entorno["client"].post("/administracion/feedbacks_ficticios", json={"por_aprendiz": 15})
    hoy = datetime.now().date()
    for f in _fechas(entorno, 101):
        assert 1 <= (hoy - f.date()).days <= 30
        assert 8 <= f.hour <= 22


def test_las_fechas_van_de_la_mas_vieja_a_la_mas_nueva(entorno):
    entorno["client"].post("/administracion/feedbacks_ficticios", json={"por_aprendiz": 12, "semilla": 5})
    fechas = _fechas(entorno, 101)
    assert fechas == sorted(fechas)


def test_se_puede_elegir_la_ventana_de_dias(entorno):
    entorno["client"].post("/administracion/feedbacks_ficticios", json={"por_aprendiz": 5, "dias": 5})
    hoy = datetime.now().date()
    fechas = _fechas(entorno, 102)
    assert len({f.date() for f in fechas}) == 5
    assert all(1 <= (hoy - f.date()).days <= 5 for f in fechas)


def test_con_mas_feedbacks_que_dias_se_repiten_dias_pero_no_hay_fechas_futuras(entorno):
    entorno["client"].post("/administracion/feedbacks_ficticios", json={"por_aprendiz": 8, "dias": 3})
    ahora = datetime.now()
    fechas = _fechas(entorno, 101)
    assert len(fechas) == 8 and all(f < ahora for f in fechas)
    assert len({f.date() for f in fechas}) <= 3


@pytest.mark.parametrize("dias", [0, 366, "x"])
def test_ventana_de_dias_invalida_se_rechaza(entorno, dias):
    r = entorno["client"].post("/administracion/feedbacks_ficticios", json={"por_aprendiz": 3, "dias": dias})
    assert r.status_code == 400
    assert _perfil(entorno, 101)["explicit_feedback"] == []


@pytest.mark.parametrize("valor", [0, -1, 51, "x", None])
def test_cantidad_invalida_se_rechaza(entorno, valor):
    r = entorno["client"].post("/administracion/feedbacks_ficticios", json={"por_aprendiz": valor})
    assert r.status_code == 400
    assert _perfil(entorno, 101)["explicit_feedback"] == []


# ── Entrenamiento en curso y copia limpia tras entrenar ─────────────────────

def test_no_se_puede_restablecer_ni_agregar_mientras_se_entrena(entorno, monkeypatch):
    monkeypatch.setattr(backend, "_training_active", True)
    cliente = entorno["client"]
    assert cliente.post("/administracion/restablecer").status_code == 409
    assert cliente.post("/administracion/feedbacks_ficticios", json={"por_aprendiz": 2}).status_code == 409
    assert cliente.get("/administracion/estado").get_json()["entrenando"] is True


def test_durante_el_entrenamiento_el_estado_no_crea_copias_limpias(entorno, monkeypatch):
    """Los archivos están cambiando: una copia tomada ahora podría quedar a medias."""
    monkeypatch.setattr(backend, "_training_active", True)
    datos = entorno["client"].get("/administracion/estado").get_json()
    assert datos["aprendices"] == 2
    assert not (entorno["datos"] / "base").exists()


def test_el_estado_no_vuelve_a_leer_perfiles_que_no_cambiaron(entorno, monkeypatch):
    cliente = entorno["client"]
    cliente.get("/administracion/estado")
    lecturas = []
    original = backend.cargar_perfil
    monkeypatch.setattr(backend, "cargar_perfil", lambda uid: lecturas.append(uid) or original(uid))
    cliente.get("/administracion/estado")
    assert lecturas == []
    _feedback_real(entorno, 101)
    lecturas.clear()
    assert cliente.get("/administracion/estado").get_json()["feedbacks"] == 1
    assert lecturas == [101]


def test_guardar_base_toma_el_estado_actual_aunque_ya_hubiera_copia(entorno):
    cliente = entorno["client"]
    cliente.get("/administracion/estado")
    _feedback_real(entorno, 101)
    con_feedback = _perfil(entorno, 101)
    assert administracion.guardar_base(backend) == 2          # equivale a terminar de entrenar
    cliente.post("/administracion/restablecer")
    assert _perfil(entorno, 101) == con_feedback


def test_guardar_base_no_lanza_excepciones(entorno, monkeypatch):
    monkeypatch.setattr(administracion, "_guardar_base_de", lambda *a: (_ for _ in ()).throw(OSError("disco lleno")))
    assert administracion.guardar_base(backend) == 0


def test_sin_aprendices_entrenados_todo_responde_en_cero(entorno):
    os.remove(entorno["datos"] / "usuarios_entrenados.json")
    cliente = entorno["client"]
    assert cliente.get("/administracion/estado").get_json()["aprendices"] == 0
    assert cliente.post("/administracion/restablecer").get_json()["restaurados"] == 0
    assert cliente.post("/administracion/feedbacks_ficticios", json={}).get_json()["feedbacks"] == 0


# ── Script que conecta el backend ───────────────────────────────────────────

_SCRIPT = os.path.join(_ROOT, "ajustar_backend.py")


@pytest.mark.skipif(not os.path.exists(_SCRIPT), reason="ajustar_backend.py ya no está en el proyecto")
class TestAjustarBackend:
    """El script de conexión debe ser seguro de correr más de una vez."""

    @staticmethod
    def _cargar():
        spec = importlib.util.spec_from_file_location("ajustar_backend", _SCRIPT)
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
        return modulo

    _ORIGINAL = (
        "def run():\n"
        "    try:\n"
        "        with contextlib.redirect_stdout(capture):\n"
        "            entrenar_modelo_usuario()\n"
        "\n"
        "        with _training_lock:\n"
        "            pass\n"
        "    finally:\n"
        "        pass\n"
        "\n"
        "\n"
        "# ── Entry point ───────────────\n"
        "if __name__ == '__main__':\n"
        "    pass\n"
    )

    def test_agrega_los_dos_fragmentos_una_sola_vez(self):
        modulo = self._cargar()
        nuevo, _ = modulo.ajustar(self._ORIGINAL)
        assert nuevo.count("administracion.registrar(") == 1
        assert nuevo.count("administracion.guardar_base(") == 1
        otra_vez, _ = modulo.ajustar(nuevo)
        assert otra_vez == nuevo

    def test_la_copia_limpia_queda_fuera_del_with_y_dentro_del_try(self):
        modulo = self._cargar()
        nuevo, _ = modulo.ajustar(self._ORIGINAL)
        assert "\n        administracion.guardar_base(sys.modules[__name__])\n" in nuevo
        assert nuevo.index("guardar_base(") < nuevo.index("with _training_lock")

    def test_sin_linea_entry_point_usa_el_if_name_main(self):
        modulo = self._cargar()
        sin_marca = self._ORIGINAL.replace("# ── Entry point ───────────────\n", "")
        nuevo, _ = modulo.ajustar(sin_marca)
        assert nuevo.index("administracion.registrar(") < nuevo.index("if __name__")
        compile(nuevo, "api_backend.py", "exec")

    def test_avisa_si_no_encuentra_donde_agregar(self):
        modulo = self._cargar()
        with pytest.raises(ValueError):
            modulo.ajustar("def run():\n    pass\n")

    def test_el_backend_del_proyecto_ya_queda_conectado(self):
        ruta = os.path.join(_ROOT, "logic", "api_backend.py")
        with open(ruta, encoding="utf-8") as f:
            texto = f.read()
        nuevo, _ = self._cargar().ajustar(texto)
        compile(nuevo, ruta, "exec")
