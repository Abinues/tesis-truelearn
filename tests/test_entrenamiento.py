"""
tests/test_entrenamiento.py
─────────────────────────────────────────────────────────────────────────────
Tests de logic/entrenador.py y logic/peek_loader.py. Antes de este archivo,
ambos módulos tenían 0% de cobertura de tests — justamente la parte más
importante metodológicamente del proyecto (acá vive el fix del punto 2:
el desfase entre session_history y el modelo final persistido).

No se descarga el dataset PEEK real (no es alcanzable desde este
entorno de todos modos): `test_entrenar_usuario_*` y
`test_generar_perfil_json_*` usan una secuencia de EventModel
diminuta y CONTROLADA (no el dataset completo) contra el INKClassifier
REAL de truelearn — se instaló para esta refactorización y ya se usa en
tests/test_truelearn_integration.py. Todos los tests de este archivo se
saltan solos (`importorskip`) si truelearn no está disponible.

Los tests de seleccionar_usuarios() (una función pura, sin relación con
truelearn) viven aparte, en tests/test_entrenador_puro.py, para que
corran siempre sin depender de si la librería está instalada.

`test_cargar_dataset_peek_*` sí puede probarse sin truelearn en absoluto,
mockeando `truelearn.datasets.load_peek_dataset` — lo único que se
verifica es el comportamiento de chdir/Lock del propio wrapper, no la
descarga real.
"""
import os

import pytest

# El importorskip va ANTES de cualquier import propio del proyecto, no
# después: si algún módulo importado más abajo llegara a depender de
# truelearn a nivel de módulo (aunque ahora mismo logic/entrenador.py y
# logic/peek_loader.py NO lo hacen), un import
# fallido interrumpiría la RECOLECCIÓN de toda la suite (no solo de este
# archivo), y pytest.importorskip no llega a ejecutarse a tiempo para
# evitarlo.
truelearn = pytest.importorskip("truelearn", reason="truelearn no está instalado")

from logic.peek_loader import cargar_dataset_peek  # noqa: E402

# ── seleccionar_usuarios ──────────────────────────────────────────────
# NOTA: los tests de seleccionar_usuarios() viven en
# tests/test_entrenador_puro.py, no acá — esa función es pura y no
# necesita truelearn instalado para nada, así que no tiene sentido que
# dependa del importorskip de este archivo.


# ── entrenar_usuario / generar_perfil_json contra el INKClassifier real ──

def _eventos_sinteticos():
    """Una secuencia diminuta y determinística de eventos: 2 temas,
    con timestamps crecientes, mezclando enganche y desenganche."""
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
    from logic.entrenador import entrenar_usuario

    eventos = _eventos_sinteticos()
    mapping = {1: ("Álgebra", "http://x"), 2: ("Cálculo", "http://y")}

    classifier, session_history, num_eng, num_non_eng = entrenar_usuario(
        99, eventos, mapping, num_epochs=1
    )

    assert num_eng == 3       # True, True, True -> 3 engaged en la lista de arriba... ver nota
    assert num_non_eng == 2
    assert len(session_history) == 5  # una entrada por evento (timestamps únicos)
    # Cada entrada trae sus propias medias/varianzas, no listas vacías
    assert all(s["means"] for s in session_history)


def test_entrenar_usuario_ultima_epoca_coincide_con_modelo_final():
    """
    Regresión directa: el snapshot de
    session_history debe coincidir con el estado final del classifier
    persistido, sin importar cuántas épocas se usen. Se prueba
    explícitamente con num_epochs=3 (no solo el default 1) para
    verificar que el fix generaliza y no es casualidad de una sola
    pasada.
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
    assert len(perfil["knowledge_components"]) == 2  # temas 1 y 2
    assert os.path.exists(ruta_json)


def test_generar_perfil_json_metadata_tupla_url_primero(tmp_path, monkeypatch):
    """Cubre la rama de meta como tupla (url, nombre) — antes sin testear."""
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
    """Cubre la rama de meta como dict — antes sin testear."""
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
    """Cubre la rama de fallback cuando el mapping no trae nada para ese topic_id."""
    from logic import entrenador

    monkeypatch.setattr(entrenador, "DATA_DIR", str(tmp_path))
    eventos = _eventos_sinteticos()
    mapping = {}  # sin metadata para ningún tema
    classifier, session_history, num_eng, num_non_eng = entrenador.entrenar_usuario(
        99, eventos, mapping, num_epochs=1
    )
    perfil, _ = entrenador.generar_perfil_json(99, classifier, mapping, num_eng, num_non_eng, session_history)

    nombres = {kc["topic_name"] for kc in perfil["knowledge_components"]}
    assert nombres == {"ID_1", "ID_2"}


# ── entrenar_modelo_usuario(): la orquestación completa, con el dataset mockeado ──

def test_entrenar_modelo_usuario_flujo_completo(tmp_path, monkeypatch):
    """
    Ejercita entrenar_modelo_usuario() de punta a punta —selección de
    usuarios, entrenamiento, guardado de .pkl y .json, índice global—
    sin descargar el dataset PEEK real: se mockea cargar_dataset_peek
    para devolver un dataset sintético diminuto.
    """
    from logic import entrenador

    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    monkeypatch.setattr(entrenador, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(entrenador, "MODEL_DIR", str(model_dir))
    monkeypatch.setattr(entrenador, "N_USUARIOS_POR_ESTRATO", 1)

    mapping = {1: ("Álgebra", "http://x"), 2: ("Cálculo", "http://y")}
    # 12 usuarios: 4 con 1 evento, 4 con 3, 3 con 5 y 1 con 10 -> los cuatro estratos
    ev = _eventos_sinteticos()
    train_sintetico = (
        [(u, ev[:1]) for u in range(1, 5)] +
        [(u, ev[:3]) for u in range(5, 9)] +
        [(u, ev) for u in range(9, 12)] +
        [(12, ev * 2)]
    )

    monkeypatch.setattr(entrenador, "cargar_dataset_peek",
                         lambda data_dir: (train_sintetico, None, mapping))

    entrenador.entrenar_modelo_usuario()

    import json
    indice = json.loads((data_dir / "usuarios_entrenados.json").read_text(encoding="utf-8"))
    assert len(indice) == 4
    assert {u["tipo"] for u in indice} == {"pocas", "medias", "muchas", "muy_activas"}
    for u in indice:
        assert (model_dir / f"model_u_{u['user_id']}.pkl").exists()
        assert (data_dir / f"visual_profile_{u['user_id']}.json").exists()


def test_entrenar_modelo_usuario_maneja_error_sin_reventar(tmp_path, monkeypatch, capsys):
    """entrenar_modelo_usuario() atrapa cualquier excepción y la loguea,
    en vez de propagarla (así el hilo de /entrenar en api_backend.py no
    muere en silencio sin marcar _training_done)."""
    from logic import entrenador

    def _falla(data_dir):
        raise RuntimeError("PEEK no disponible")

    monkeypatch.setattr(entrenador, "cargar_dataset_peek", _falla)

    entrenador.entrenar_modelo_usuario()  # no debe lanzar

    salida = capsys.readouterr().out
    assert "ERROR CRÍTICO" in salida


# ── cargar_dataset_peek: chdir/Lock, sin necesitar el dataset real ──────

def test_cargar_dataset_peek_llama_a_load_peek_dataset(tmp_path, monkeypatch):
    llamadas = []

    def fake_load_peek_dataset():
        llamadas.append(os.getcwd())
        return ("train", "test", "mapping")

    monkeypatch.setattr("truelearn.datasets.load_peek_dataset", fake_load_peek_dataset)

    cwd_antes = os.getcwd()
    resultado = cargar_dataset_peek(str(tmp_path))

    assert resultado == ("train", "test", "mapping")
    assert len(llamadas) == 1
    assert os.path.abspath(llamadas[0]) == os.path.abspath(str(tmp_path))
    assert os.getcwd() == cwd_antes  # el cwd se restauró


def test_cargar_dataset_peek_restaura_cwd_incluso_si_falla(tmp_path, monkeypatch):
    """
    Regresión directa: en la versión original,
    si load_peek_dataset() lanzaba una excepción, el cwd del proceso
    quedaba "pegado" en data/ para siempre. Acá se fuerza la excepción y
    se verifica que el cwd se restaura de todos modos.
    """
    def fake_load_peek_dataset_que_falla():
        raise RuntimeError("simulando una falla de red al descargar PEEK")

    monkeypatch.setattr("truelearn.datasets.load_peek_dataset", fake_load_peek_dataset_que_falla)

    cwd_antes = os.getcwd()
    with pytest.raises(RuntimeError):
        cargar_dataset_peek(str(tmp_path))

    assert os.getcwd() == cwd_antes


def test_cargar_dataset_peek_crea_el_directorio_si_no_existe(tmp_path, monkeypatch):
    destino = tmp_path / "no_existe_todavia"
    monkeypatch.setattr("truelearn.datasets.load_peek_dataset", lambda: (1, 2, 3))

    assert not destino.exists()
    cargar_dataset_peek(str(destino))
    assert destino.exists()


# ── extractor_peek.py y preparar_datos.py ────────────────────────────────
#
# NOTA: ninguno de los dos módulos es llamado
# por dashboard.py, api_backend.py ni entrenador.py — son herramientas
# exploratorias standalone (preparar_datos.py sí tiene un bloque
# __main__, pensado para correrse a mano en la línea de comandos durante
# la Fase 2 de la propuesta; extractor_peek.py no tiene ni eso). Se
# testean igual, livianamente, para que no se rompan en silencio si
# alguien los sigue usando de forma interactiva/manual.

def test_extractor_peek_obtener_datos_usuario_por_id(monkeypatch):
    from logic import extractor_peek

    train_sintetico = [(10, ["eventosA"]), (20, ["eventosB"])]
    monkeypatch.setattr(extractor_peek, "cargar_dataset_peek",
                         lambda data_dir: (train_sintetico, None, {"mapa": True}))

    uid, eventos, mapping = extractor_peek.obtener_datos_usuario(usuario_objetivo_id=20)
    assert uid == 20
    assert eventos == ["eventosB"]
    assert mapping == {"mapa": True}


def test_extractor_peek_obtener_datos_usuario_default_mas_de_20_eventos(monkeypatch):
    from logic import extractor_peek

    train_sintetico = [(1, list(range(5))), (2, list(range(25)))]
    monkeypatch.setattr(extractor_peek, "cargar_dataset_peek",
                         lambda data_dir: (train_sintetico, None, {}))

    uid, eventos, _ = extractor_peek.obtener_datos_usuario()  # sin id -> primero con >20 eventos
    assert uid == 2
    assert len(eventos) == 25


def test_extractor_peek_usuario_no_encontrado(monkeypatch):
    from logic import extractor_peek

    monkeypatch.setattr(extractor_peek, "cargar_dataset_peek",
                         lambda data_dir: ([(1, [0])], None, {}))
    uid, eventos, mapping = extractor_peek.obtener_datos_usuario(usuario_objetivo_id=999)
    assert (uid, eventos, mapping) == (None, None, None)


def test_preparar_datos_extraer_muestra_genera_json(tmp_path, monkeypatch, capsys):
    from truelearn.models import EventModel, Knowledge, KnowledgeComponent

    from logic import preparar_datos

    monkeypatch.setattr(preparar_datos, "DATA_DIR", str(tmp_path))

    eventos_usuario = []
    for t in range(16):  # > 15, para pasar el filtro "usuario con historial"
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
    from logic import preparar_datos

    monkeypatch.setattr(preparar_datos, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(preparar_datos, "cargar_dataset_peek",
                         lambda data_dir: ([(1, [0] * 5)], None, {}))  # nadie tiene > 15 eventos

    preparar_datos.extraer_muestra()  # no debe lanzar

    assert not (tmp_path / "datos_usuario_prueba.json").exists()
    assert "No se encontró un usuario" in capsys.readouterr().out
