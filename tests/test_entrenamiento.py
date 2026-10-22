"""
Tests de ``logic/entrenador.py``, ``logic/peek_loader.py`` y de las
herramientas exploratorias ``extractor_peek.py`` y ``preparar_datos.py``.

No se descarga el dataset PEEK real. Los tests de ``entrenar_usuario`` y
``generar_perfil_json`` usan una secuencia diminuta y controlada de
``EventModel`` contra el ``INKClassifier`` real; verifican, entre otras cosas,
que el último snapshot de ``session_history`` coincide con el modelo final
persistido. Los tests de ``cargar_dataset_peek`` reemplazan
``truelearn.datasets.load_peek_dataset`` y solo comprueban el cambio de
directorio y el bloqueo del wrapper.

Todo el módulo se omite si ``truelearn`` no está instalado. El
``importorskip`` va antes de cualquier import del proyecto para que un fallo de
importación no interrumpa la recolección de toda la suite. Los tests de
``seleccionar_usuarios`` están en ``test_entrenador_puro.py`` porque no
dependen de ``truelearn``.

``extractor_peek.py`` y ``preparar_datos.py`` no son llamados por
``dashboard.py``, ``api_backend.py`` ni ``entrenador.py``; son herramientas
manuales de exploración y se prueban de forma liviana.
"""
import os

import pytest

truelearn = pytest.importorskip("truelearn", reason="truelearn no está instalado")

from logic.peek_loader import cargar_dataset_peek  # noqa: E402


def _eventos_sinteticos():
    """Crea una secuencia diminuta y determinística de eventos.

    Son cinco eventos sobre dos temas, con marcas de tiempo crecientes y una
    mezcla de enganche y desenganche (3 enganches y 2 desenganches).

    Returns:
        list[tuple[EventModel, bool]]: Pares (evento, enganchado).
    """
    from truelearn.models import EventModel, Knowledge, KnowledgeComponent

    eventos = []
    for t, (topic_id, engaged) in enumerate([
        (1, True), (2, False), (1, True), (2, True), (1, False),
    ]):
        ev = EventModel(Knowledge({topic_id: KnowledgeComponent(mean=0.5, variance=0.5)}),
                         event_time=float(t))
        eventos.append((ev, engaged))
    return eventos


def test_entrenar_usuario_devuelve_session_history_no_vacio():
    """``entrenar_usuario`` devuelve un historial con una entrada por evento."""
    from logic.entrenador import entrenar_usuario

    eventos = _eventos_sinteticos()
    mapping = {1: ("Álgebra", "http://x"), 2: ("Cálculo", "http://y")}

    classifier, session_history, num_eng, num_non_eng = entrenar_usuario(
        99, eventos, mapping, num_epochs=1
    )

    assert num_eng == 3
    assert num_non_eng == 2
    assert len(session_history) == 5
    assert all(s["means"] for s in session_history)


def test_entrenar_usuario_ultima_epoca_coincide_con_modelo_final():
    """El último snapshot de ``session_history`` coincide con el modelo final.

    Debe cumplirse sin importar cuántas épocas se usen; se prueba con
    ``num_epochs=3`` para verificar que la corrección generaliza y no es
    casualidad de una sola pasada.
    """
    from logic.entrenador import entrenar_usuario

    eventos = _eventos_sinteticos()
    mapping = {1: ("Álgebra", "http://x"), 2: ("Cálculo", "http://y")}

    classifier, session_history, _, _ = entrenar_usuario(99, eventos, mapping, num_epochs=3)

    ultimo_snapshot = session_history[-1]
    knowledge_model = classifier.get_learner_model()[0]
    medias_finales_reales = {}
    for tid, kc in knowledge_model.knowledge.topic_kc_pairs():
        nombre = mapping[tid][0]
        medias_finales_reales[nombre] = kc.mean

    assert ultimo_snapshot["means"] == pytest.approx(medias_finales_reales)


def test_generar_perfil_json_incluye_pesos_y_kcs(tmp_path, monkeypatch):
    """``generar_perfil_json`` incluye los tres pesos y un KC por tema.

    Args:
        tmp_path (pathlib.Path): Directorio temporal para ``DATA_DIR``.
        monkeypatch (pytest.MonkeyPatch): Fixture para reemplazar ``DATA_DIR``.
    """
    from logic import entrenador

    monkeypatch.setattr(entrenador, "DATA_DIR", str(tmp_path))

    eventos = _eventos_sinteticos()
    mapping = {1: ("Álgebra", "http://x"), 2: ("Cálculo", "http://y")}
    classifier, session_history, num_eng, num_non_eng = entrenador.entrenar_usuario(
        99, eventos, mapping, num_epochs=1
    )

    perfil, ruta_json = entrenador.generar_perfil_json(
        99, classifier, mapping, num_eng, num_non_eng, session_history
    )

    assert perfil["user_id"] == 99
    assert set(perfil["weights"].keys()) == {"novelty", "interest", "bias"}
    assert len(perfil["knowledge_components"]) == 2
    assert os.path.exists(ruta_json)


def test_generar_perfil_json_metadata_tupla_url_primero(tmp_path, monkeypatch):
    """Cubre los metadatos como tupla (url, nombre)."""
    from logic import entrenador

    monkeypatch.setattr(entrenador, "DATA_DIR", str(tmp_path))
    eventos = _eventos_sinteticos()
    mapping = {1: ("http://x", "Álgebra"), 2: ("http://y", "Cálculo")}
    classifier, session_history, num_eng, num_non_eng = entrenador.entrenar_usuario(
        99, eventos, mapping, num_epochs=1
    )
    perfil, _ = entrenador.generar_perfil_json(99, classifier, mapping, num_eng, num_non_eng, session_history)

    nombres = {kc["topic_name"] for kc in perfil["knowledge_components"]}
    assert nombres == {"Álgebra", "Cálculo"}
    urls = {kc["topic_url"] for kc in perfil["knowledge_components"]}
    assert urls == {"http://x", "http://y"}


def test_generar_perfil_json_metadata_dict(tmp_path, monkeypatch):
    """Cubre los metadatos como diccionario."""
    from logic import entrenador

    monkeypatch.setattr(entrenador, "DATA_DIR", str(tmp_path))
    eventos = _eventos_sinteticos()
    mapping = {
        1: {"name": "Álgebra", "url": "http://x"},
        2: {"name": "Cálculo", "url": "http://y"},
    }
    classifier, session_history, num_eng, num_non_eng = entrenador.entrenar_usuario(
        99, eventos, mapping, num_epochs=1
    )
    perfil, _ = entrenador.generar_perfil_json(99, classifier, mapping, num_eng, num_non_eng, session_history)

    nombres = {kc["topic_name"] for kc in perfil["knowledge_components"]}
    assert nombres == {"Álgebra", "Cálculo"}


def test_generar_perfil_json_metadata_desconocida(tmp_path, monkeypatch):
    """Cubre el respaldo ``ID_<n>`` cuando el mapeo no trae metadatos del tema."""
    from logic import entrenador

    monkeypatch.setattr(entrenador, "DATA_DIR", str(tmp_path))
    eventos = _eventos_sinteticos()
    mapping = {}
    classifier, session_history, num_eng, num_non_eng = entrenador.entrenar_usuario(
        99, eventos, mapping, num_epochs=1
    )
    perfil, _ = entrenador.generar_perfil_json(99, classifier, mapping, num_eng, num_non_eng, session_history)

    nombres = {kc["topic_name"] for kc in perfil["knowledge_components"]}
    assert nombres == {"ID_1", "ID_2"}


def test_entrenar_modelo_usuario_flujo_completo(tmp_path, monkeypatch):
    """Ejecuta ``entrenar_modelo_usuario()`` de punta a punta con datos sintéticos.

    Cubre la selección de usuarios, el entrenamiento, el guardado de los
    ``.pkl`` y ``.json`` y el índice global. ``cargar_dataset_peek`` se
    reemplaza para devolver un dataset diminuto.

    Args:
        tmp_path (pathlib.Path): Directorio temporal para ``data/`` y ``models/``.
        monkeypatch (pytest.MonkeyPatch): Fixture para reemplazar constantes y funciones.
    """
    from logic import entrenador

    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    monkeypatch.setattr(entrenador, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(entrenador, "MODEL_DIR", str(model_dir))
    monkeypatch.setattr(entrenador, "N_USUARIOS_MUCHAS", 1)
    monkeypatch.setattr(entrenador, "N_USUARIOS_POCAS", 1)
    monkeypatch.setattr(entrenador, "UMBRAL_MUCHAS", 3)
    monkeypatch.setattr(entrenador, "UMBRAL_POCAS_MAX", 10)

    mapping = {1: ("Álgebra", "http://x"), 2: ("Cálculo", "http://y")}
    train_sintetico = [
        (10, _eventos_sinteticos() * 2),
        (20, _eventos_sinteticos()[:1]),
    ]

    monkeypatch.setattr(entrenador, "cargar_dataset_peek",
                         lambda data_dir: (train_sintetico, None, mapping))

    entrenador.entrenar_modelo_usuario()

    assert (model_dir / "model_u_10.pkl").exists()
    assert (model_dir / "model_u_20.pkl").exists()
    assert (data_dir / "visual_profile_10.json").exists()
    assert (data_dir / "visual_profile_20.json").exists()
    assert (data_dir / "usuarios_entrenados.json").exists()

    import json
    indice = json.loads((data_dir / "usuarios_entrenados.json").read_text(encoding="utf-8"))
    assert {u["user_id"] for u in indice} == {10, 20}
    assert {u["tipo"] for u in indice} == {"muchas", "pocas"}


def test_entrenar_modelo_usuario_maneja_error_sin_reventar(tmp_path, monkeypatch, capsys):
    """``entrenar_modelo_usuario`` atrapa las excepciones y las registra.

    De lo contrario, el hilo de ``/entrenar`` en ``api_backend.py`` moriría en
    silencio sin marcar ``_training_done``.

    Args:
        tmp_path (pathlib.Path): Directorio temporal.
        monkeypatch (pytest.MonkeyPatch): Fixture para reemplazar la carga del dataset.
        capsys (pytest.CaptureFixture): Fixture para capturar la salida estándar.
    """
    from logic import entrenador

    def _falla(data_dir):
        """Simula que el dataset PEEK no está disponible."""
        raise RuntimeError("PEEK no disponible")

    monkeypatch.setattr(entrenador, "cargar_dataset_peek", _falla)

    entrenador.entrenar_modelo_usuario()

    salida = capsys.readouterr().out
    assert "ERROR CRÍTICO" in salida


def test_cargar_dataset_peek_llama_a_load_peek_dataset(tmp_path, monkeypatch):
    """``cargar_dataset_peek`` llama a ``load_peek_dataset`` desde el directorio dado.

    Verifica además que el directorio de trabajo se restaura al terminar.

    Args:
        tmp_path (pathlib.Path): Directorio de datos temporal.
        monkeypatch (pytest.MonkeyPatch): Fixture para reemplazar ``load_peek_dataset``.
    """
    llamadas = []

    def fake_load_peek_dataset():
        """Registra el directorio de trabajo y devuelve un dataset ficticio."""
        llamadas.append(os.getcwd())
        return ("train", "test", "mapping")

    monkeypatch.setattr("truelearn.datasets.load_peek_dataset", fake_load_peek_dataset)

    cwd_antes = os.getcwd()
    resultado = cargar_dataset_peek(str(tmp_path))

    assert resultado == ("train", "test", "mapping")
    assert len(llamadas) == 1
    assert os.path.abspath(llamadas[0]) == os.path.abspath(str(tmp_path))
    assert os.getcwd() == cwd_antes


def test_cargar_dataset_peek_restaura_cwd_incluso_si_falla(tmp_path, monkeypatch):
    """Restaura el directorio de trabajo aunque ``load_peek_dataset`` falle.

    Regresión: en la versión original, una excepción dejaba el directorio de
    trabajo del proceso fijado en ``data/`` para siempre.

    Args:
        tmp_path (pathlib.Path): Directorio de datos temporal.
        monkeypatch (pytest.MonkeyPatch): Fixture para reemplazar ``load_peek_dataset``.
    """
    def fake_load_peek_dataset_que_falla():
        """Simula una falla de red al descargar PEEK."""
        raise RuntimeError("simulando una falla de red al descargar PEEK")

    monkeypatch.setattr("truelearn.datasets.load_peek_dataset", fake_load_peek_dataset_que_falla)

    cwd_antes = os.getcwd()
    with pytest.raises(RuntimeError):
        cargar_dataset_peek(str(tmp_path))

    assert os.getcwd() == cwd_antes


def test_cargar_dataset_peek_crea_el_directorio_si_no_existe(tmp_path, monkeypatch):
    """``cargar_dataset_peek`` crea el directorio de datos si no existe."""
    destino = tmp_path / "no_existe_todavia"
    monkeypatch.setattr("truelearn.datasets.load_peek_dataset", lambda: (1, 2, 3))

    assert not destino.exists()
    cargar_dataset_peek(str(destino))
    assert destino.exists()


def test_extractor_peek_obtener_datos_usuario_por_id(monkeypatch):
    """``obtener_datos_usuario`` devuelve los eventos del usuario pedido."""
    from logic import extractor_peek

    train_sintetico = [(10, ["eventosA"]), (20, ["eventosB"])]
    monkeypatch.setattr(extractor_peek, "cargar_dataset_peek",
                         lambda data_dir: (train_sintetico, None, {"mapa": True}))

    uid, eventos, mapping = extractor_peek.obtener_datos_usuario(usuario_objetivo_id=20)
    assert uid == 20
    assert eventos == ["eventosB"]
    assert mapping == {"mapa": True}


def test_extractor_peek_obtener_datos_usuario_default_mas_de_20_eventos(monkeypatch):
    """Sin id, devuelve el primer usuario con más de 20 eventos."""
    from logic import extractor_peek

    train_sintetico = [(1, list(range(5))), (2, list(range(25)))]
    monkeypatch.setattr(extractor_peek, "cargar_dataset_peek",
                         lambda data_dir: (train_sintetico, None, {}))

    uid, eventos, _ = extractor_peek.obtener_datos_usuario()
    assert uid == 2
    assert len(eventos) == 25


def test_extractor_peek_usuario_no_encontrado(monkeypatch):
    """Devuelve ``(None, None, None)`` si el usuario pedido no existe."""
    from logic import extractor_peek

    monkeypatch.setattr(extractor_peek, "cargar_dataset_peek",
                         lambda data_dir: ([(1, [0])], None, {}))
    uid, eventos, mapping = extractor_peek.obtener_datos_usuario(usuario_objetivo_id=999)
    assert (uid, eventos, mapping) == (None, None, None)


def test_preparar_datos_extraer_muestra_genera_json(tmp_path, monkeypatch, capsys):
    """``extraer_muestra`` genera ``datos_usuario_prueba.json`` con los eventos.

    Args:
        tmp_path (pathlib.Path): Directorio temporal para ``DATA_DIR``.
        monkeypatch (pytest.MonkeyPatch): Fixture para reemplazar el dataset.
        capsys (pytest.CaptureFixture): Fixture para capturar la salida estándar.
    """
    from truelearn.models import EventModel, Knowledge, KnowledgeComponent

    from logic import preparar_datos

    monkeypatch.setattr(preparar_datos, "DATA_DIR", str(tmp_path))

    eventos_usuario = []
    for t in range(16):
        ev = EventModel(Knowledge({1: KnowledgeComponent(mean=0.3, variance=0.4)}), event_time=float(t))
        eventos_usuario.append((ev, t % 2 == 0))

    train_sintetico = [(42, eventos_usuario)]
    mapping = {1: ("Álgebra", "http://x")}
    monkeypatch.setattr(preparar_datos, "cargar_dataset_peek",
                         lambda data_dir: (train_sintetico, None, mapping))

    preparar_datos.extraer_muestra()

    ruta = tmp_path / "datos_usuario_prueba.json"
    assert ruta.exists()
    import json
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    assert len(datos) == 16
    assert datos[0]["temas_video"][0]["nombre"] == "Álgebra"


def test_preparar_datos_sin_usuario_con_historial_no_rompe(tmp_path, monkeypatch, capsys):
    """Si nadie tiene más de 15 eventos, no se genera archivo y se informa.

    Args:
        tmp_path (pathlib.Path): Directorio temporal para ``DATA_DIR``.
        monkeypatch (pytest.MonkeyPatch): Fixture para reemplazar el dataset.
        capsys (pytest.CaptureFixture): Fixture para capturar la salida estándar.
    """
    from logic import preparar_datos

    monkeypatch.setattr(preparar_datos, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(preparar_datos, "cargar_dataset_peek",
                         lambda data_dir: ([(1, [0] * 5)], None, {}))

    preparar_datos.extraer_muestra()

    assert not (tmp_path / "datos_usuario_prueba.json").exists()
    assert "No se encontró un usuario" in capsys.readouterr().out
