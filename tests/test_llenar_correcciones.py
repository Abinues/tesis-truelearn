"""
Tests de ``llenar_correcciones.py``, el script que llena de correcciones a los
aprendices entrenados.

Se reemplazan ``cargar_modelo``, ``cargar_perfil``, ``guardar_modelo`` y
``guardar_perfil`` del backend (y, en un caso, sus directorios) para no tocar
datos reales ni requerir ``truelearn``.
"""
import json
import random
import re
import time

import pytest

import llenar_correcciones as lc
import logic.api_backend as backend
from tests.test_correccion import FakeClassifier, FakeKC

TIPOS_VALIDOS = {"sé más", "sé menos", "me interesa", "no me interesa"}


def _kc(topic_id, mean=0.0, interest_mean=0.0):
    """Crea un componente de conocimiento como aparece en el perfil JSON.

    Args:
        topic_id (int): Identificador del tema.
        mean (float): Media de conocimiento.
        interest_mean (float): Media de interés.

    Returns:
        dict: Componente con la estructura de ``knowledge_components``.
    """
    return {
        "topic_id": topic_id, "topic_name": f"Tema {topic_id}", "topic_url": "",
        "mean": mean, "variance": 0.5,
        "interest_mean": interest_mean, "interest_var": 0.5, "timestamp": 0.0,
    }


def _perfil(topic_ids):
    """Crea un perfil de aprendiz con los temas indicados.

    Args:
        topic_ids (Iterable[int]): Identificadores de los temas.

    Returns:
        dict: Perfil sin correcciones previas.
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
        "knowledge_components": [_kc(t) for t in topic_ids],
    }


def test_generar_plan_respeta_cantidad_fechas_y_tipos():
    """El plan tiene entre mínimo y máximo correcciones, ordenadas y en rango."""
    perfil = _perfil(range(1, 9))
    ahora = 1_800_000_000.0
    plan = lc.generar_plan(perfil, random.Random(3), minimo=4, maximo=9, dias=10, ahora=ahora)

    assert 4 <= len(plan) <= 9
    fechas = [ts for ts, _ in plan]
    assert fechas == sorted(fechas)
    assert all(ahora - 10 * 86400 <= ts <= ahora for ts in fechas)
    assert all(kc in perfil["knowledge_components"] for _, kc in plan)


def test_generar_plan_con_pocos_temas_repite_temas():
    """Con menos temas que correcciones, algunos temas se corrigen más de una vez."""
    perfil = _perfil([1, 2])
    plan = lc.generar_plan(perfil, random.Random(1), minimo=6, maximo=6, dias=5, ahora=time.time())
    assert len(plan) == 6
    assert {kc["topic_id"] for _, kc in plan} <= {1, 2}


def test_generar_plan_sin_temas_devuelve_lista_vacia():
    """Un perfil sin Knowledge Components no genera correcciones."""
    plan = lc.generar_plan(_perfil([]), random.Random(1), 3, 5, 5, time.time())
    assert plan == []


def test_elegir_tipo_prefiere_se_menos_si_el_conocimiento_es_alto():
    """Con conocimiento muy alto, "sé menos" aparece más que "sé más"."""
    kc = _kc(1, mean=10.0, interest_mean=0.0)
    rng = random.Random(0)
    tipos = [lc.elegir_tipo(kc, rng) for _ in range(400)]
    assert tipos.count("sé menos") > tipos.count("sé más")


def test_aplicar_plan_usa_la_fecha_simulada_y_restaura_el_reloj(monkeypatch):
    """Cada corrección queda con la fecha del plan y el reloj real se restaura."""
    topic = 42
    clf = FakeClassifier(know_pairs=[(topic, FakeKC(0.0, 0.5))],
                         int_pairs=[(topic, FakeKC(0.0, 0.5))])
    guardados = {}
    monkeypatch.setattr(backend, "cargar_modelo", lambda uid: clf)
    monkeypatch.setattr(backend, "cargar_perfil", lambda uid: guardados.get("perfil", _perfil([topic])))
    monkeypatch.setattr(backend, "guardar_modelo", lambda uid, c: None)
    monkeypatch.setattr(backend, "guardar_perfil", lambda uid, p: guardados.__setitem__("perfil", p))

    reloj_real = backend.time
    plan = [(1000.0, _kc(topic)), (2000.0, _kc(topic))]
    backend.app.config["TESTING"] = True
    aplicadas, errores = lc.aplicar_plan(backend.app.test_client(), 7, plan, random.Random(0))

    assert (aplicadas, errores) == (2, [])
    feedback = guardados["perfil"]["explicit_feedback"]
    assert [f["timestamp"] for f in feedback] == [1000.0, 2000.0]
    assert all(f["tipo"] in TIPOS_VALIDOS for f in feedback)
    assert backend.time is reloj_real


def test_aplicar_plan_informa_errores_de_tema_inexistente(monkeypatch):
    """Una corrección sobre un tema ausente del modelo se informa como error."""
    monkeypatch.setattr(backend, "cargar_modelo", lambda uid: FakeClassifier())
    monkeypatch.setattr(backend, "cargar_perfil", lambda uid: _perfil([5]))
    backend.app.config["TESTING"] = True

    aplicadas, errores = lc.aplicar_plan(
        backend.app.test_client(), 7, [(1000.0, _kc(5))], random.Random(0))

    assert aplicadas == 0
    assert len(errores) == 1


def test_aplicar_plan_continua_si_una_correccion_lanza_excepcion(monkeypatch):
    """Una excepción del backend se registra como error y las demás siguen.

    Reproduce el ``PermissionError`` que Windows lanza al guardar un archivo
    bloqueado: la corrección afectada falla y las siguientes se aplican.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture para reemplazar la persistencia del backend.
    """
    topic = 42
    clf = FakeClassifier(know_pairs=[(topic, FakeKC(0.0, 0.5))],
                         int_pairs=[(topic, FakeKC(0.0, 0.5))])
    llamadas = {"guardar": 0}

    def guardar_modelo_inestable(uid, c):
        """Falla en la primera llamada, como un archivo bloqueado."""
        llamadas["guardar"] += 1
        if llamadas["guardar"] == 1:
            raise PermissionError(5, "Acceso denegado")

    monkeypatch.setattr(backend, "cargar_modelo", lambda uid: clf)
    monkeypatch.setattr(backend, "cargar_perfil", lambda uid: _perfil([topic]))
    monkeypatch.setattr(backend, "guardar_modelo", guardar_modelo_inestable)
    monkeypatch.setattr(backend, "guardar_perfil", lambda uid, p: None)
    backend.app.config["TESTING"] = True

    plan = [(1000.0 + i, _kc(topic)) for i in range(3)]
    aplicadas, errores = lc.aplicar_plan(backend.app.test_client(), 7, plan, random.Random(0))

    assert aplicadas == 2
    assert len(errores) == 1
    assert "PermissionError" in errores[0]


def test_muchas_correcciones_sobre_un_tema_no_lo_clavan_en_un_extremo(monkeypatch):
    """Tras 100 correcciones sobre un mismo tema, el conocimiento no queda en un límite.

    El tipo se elige según el estado actual del tema, así que los valores
    oscilan en vez de empujar siempre en la misma dirección hasta saturar.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture para reemplazar la persistencia del backend.
    """
    from logic import dominio as D

    topic = 42
    kc_know = FakeKC(0.0, 0.5)
    clf = FakeClassifier(know_pairs=[(topic, kc_know)], int_pairs=[(topic, FakeKC(0.0, 0.5))])
    guardados = {}
    monkeypatch.setattr(backend, "cargar_modelo", lambda uid: clf)
    monkeypatch.setattr(backend, "cargar_perfil", lambda uid: guardados.get("perfil", _perfil([topic])))
    monkeypatch.setattr(backend, "guardar_modelo", lambda uid, c: None)
    monkeypatch.setattr(backend, "guardar_perfil", lambda uid, p: guardados.__setitem__("perfil", p))

    plan = [(1000.0 + i, _kc(topic)) for i in range(100)]
    backend.app.config["TESTING"] = True
    aplicadas, errores = lc.aplicar_plan(backend.app.test_client(), 7, plan, random.Random(5))

    assert (aplicadas, errores) == (100, [])
    assert D.KNOW_MIN < kc_know.mean < D.KNOW_MAX


def test_main_simular_no_modifica_nada(tmp_path, monkeypatch, capsys):
    """Con ``--simular`` el perfil queda intacto y no se crea ningún respaldo.

    Args:
        tmp_path (pathlib.Path): Directorio temporal usado como raíz de datos.
        monkeypatch (pytest.MonkeyPatch): Fixture para reemplazar los directorios del backend.
        capsys (pytest.CaptureFixture): Fixture para capturar la salida estándar.
    """
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    models.mkdir()
    monkeypatch.setattr(backend, "DATA_DIR", str(data))
    monkeypatch.setattr(backend, "MODEL_DIR", str(models))

    perfil = _perfil([1, 2, 3])
    (data / "visual_profile_7.json").write_text(json.dumps(perfil), encoding="utf-8")
    (data / "usuarios_entrenados.json").write_text(
        json.dumps([{"user_id": 7, "tipo": "muchas"}]), encoding="utf-8")
    (models / "model_u_7.pkl").write_bytes(b"no-se-lee-en-simulacion")

    codigo = lc.main(["--simular", "--semilla", "1"])

    assert codigo == 0
    assert json.loads((data / "visual_profile_7.json").read_text(encoding="utf-8")) == perfil
    assert not list(tmp_path.glob("backup_correcciones_*"))
    assert "aplicaría" in capsys.readouterr().out


def test_main_sin_indice_de_usuarios_devuelve_error(tmp_path, monkeypatch):
    """Sin ``usuarios_entrenados.json`` el script termina con código 1."""
    monkeypatch.setattr(backend, "DATA_DIR", str(tmp_path / "vacio"))
    assert lc.main(["--simular"]) == 1


@pytest.mark.parametrize("argv", [["--minimo", "0"], ["--minimo", "5", "--maximo", "2"], ["--dias", "0"]])
def test_argumentos_invalidos_terminan_con_error(argv):
    """Rangos inválidos de ``--minimo``, ``--maximo`` o ``--dias`` abortan el script.

    Args:
        argv (list[str]): Argumentos inválidos a analizar.
    """
    with pytest.raises(SystemExit):
        lc.parsear_argumentos(argv)


def _preparar_datos(tmp_path, monkeypatch, previas):
    """Crea un aprendiz con ``previas`` correcciones en un directorio temporal.

    Args:
        tmp_path (pathlib.Path): Directorio temporal usado como raíz de datos.
        monkeypatch (pytest.MonkeyPatch): Fixture para reemplazar los directorios del backend.
        previas (int): Cantidad de correcciones que ya tiene el aprendiz.
    """
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    models.mkdir()
    monkeypatch.setattr(backend, "DATA_DIR", str(data))
    monkeypatch.setattr(backend, "MODEL_DIR", str(models))
    perfil = _perfil([1, 2, 3])
    perfil["explicit_feedback"] = [{"timestamp": float(i)} for i in range(previas)]
    (data / "visual_profile_7.json").write_text(json.dumps(perfil), encoding="utf-8")
    (data / "usuarios_entrenados.json").write_text(
        json.dumps([{"user_id": 7, "tipo": "muchas"}]), encoding="utf-8")
    (models / "model_u_7.pkl").write_bytes(b"no-se-lee-en-simulacion")


def test_completar_omite_a_quien_ya_llego_al_minimo(tmp_path, monkeypatch, capsys):
    """Con ``--completar``, un aprendiz con al menos ``--minimo`` correcciones se omite.

    Args:
        tmp_path (pathlib.Path): Directorio temporal usado como raíz de datos.
        monkeypatch (pytest.MonkeyPatch): Fixture para reemplazar los directorios del backend.
        capsys (pytest.CaptureFixture): Fixture para capturar la salida estándar.
    """
    _preparar_datos(tmp_path, monkeypatch, previas=60)
    lc.main(["--simular", "--completar", "--minimo", "50", "--maximo", "100"])
    salida = capsys.readouterr().out
    assert "ya tiene 60" in salida
    assert "#7: aplicaría" not in salida


def test_completar_agrega_solo_las_que_faltan(tmp_path, monkeypatch, capsys):
    """Con ``--completar``, a quien tiene menos del mínimo se le suman las que faltan.

    Con 40 previas y un rango de 50 a 100, deben agregarse entre 10 y 60.

    Args:
        tmp_path (pathlib.Path): Directorio temporal usado como raíz de datos.
        monkeypatch (pytest.MonkeyPatch): Fixture para reemplazar los directorios del backend.
        capsys (pytest.CaptureFixture): Fixture para capturar la salida estándar.
    """
    _preparar_datos(tmp_path, monkeypatch, previas=40)
    lc.main(["--simular", "--completar", "--minimo", "50", "--maximo", "100", "--semilla", "3"])
    agregadas = int(re.search(r"aplicaría (\d+)", capsys.readouterr().out).group(1))
    assert 10 <= agregadas <= 60
