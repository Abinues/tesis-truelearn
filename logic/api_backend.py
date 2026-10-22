"""
Backend Flask del dashboard TrueLearn.

Endpoints:
    GET  /usuarios                       Lista de usuarios entrenados.
    GET  /perfil/<uid>                   Perfil JSON de un usuario.
    POST /feedback/<uid>                 Aplica feedback explícito y recalcula el perfil.
    GET  /historial_feedback/<uid>       Historial de feedbacks aplicados.
    GET  /prediccion/<uid>/<topic_id>    Probabilidad de enganche para un tema.
    POST /entrenar                       Lanza el entrenamiento en segundo plano.
    GET  /log_entrenamiento              Log y estado del entrenamiento en curso.

Ejecución:
    python -m logic.api_backend
    (escucha en http://localhost:5000)

Variables de entorno:
    TRUELEARN_DEBUG=1  Habilita el modo debug de Flask. No usarlo si el puerto
        queda expuesto más allá de localhost: el depurador de Werkzeug permite
        ejecutar código arbitrario.
"""

import json
import os
import pickle
import sys
import threading
import time

from flask import Flask, jsonify, request

from logic.dominio import clamp_conocimiento, clamp_interes, clamp_variance
from logic.modelo_utils import extraer_pesos_reales

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR  = os.path.join(BASE_DIR, "data")

app = Flask(__name__)


def _reemplazar_con_reintentos(tmp, ruta, intentos=10, espera=0.05):
    """
    Reemplaza ``ruta`` por ``tmp`` reintentando si Windows deniega el acceso.

    En Windows, ``os.replace`` lanza ``PermissionError`` si otro proceso (el
    antivirus, el indexador de búsqueda, un visor de archivos) tiene abierto el
    destino en ese instante. Suele durar milisegundos, así que se reintenta con
    una espera creciente antes de darse por vencido.

    Args:
        tmp (str): Ruta del archivo temporal ya escrito.
        ruta (str): Ruta definitiva que debe quedar reemplazada.
        intentos (int): Cantidad máxima de intentos.
        espera (float): Segundos de espera tras el primer fallo; se duplica en
            cada reintento hasta un máximo de un segundo.

    Returns:
        None

    Raises:
        PermissionError: Si después de todos los intentos el acceso sigue
            denegado. El archivo temporal se elimina antes de relanzarla.
    """
    for intento in range(intentos):
        try:
            os.replace(tmp, ruta)
            return
        except PermissionError:
            if intento == intentos - 1:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
            time.sleep(min(espera * (2 ** intento), 1.0))


def cargar_modelo(uid):
    """
    Carga el modelo entrenado de un usuario desde ``models/``.

    Un archivo corrupto o incompatible se registra en el log y se trata como
    modelo no disponible.

    Args:
        uid (int): Identificador del usuario.

    Returns:
        INKClassifier | None: El clasificador deserializado, o ``None`` si el
        archivo no existe o no se pudo deserializar.
    """
    ruta = os.path.join(MODEL_DIR, f"model_u_{uid}.pkl")
    if not os.path.exists(ruta):
        return None
    try:
        with open(ruta, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as e:
        app.logger.error(f"No se pudo deserializar el modelo de {uid}: {e}")
        return None


def guardar_modelo(uid, classifier):
    """
    Guarda el modelo de un usuario en ``models/`` de forma atómica.

    Escribe en un archivo temporal y lo renombra, para no dejar un ``.pkl``
    truncado si el proceso se interrumpe durante la escritura.

    Args:
        uid (int): Identificador del usuario.
        classifier (INKClassifier): Clasificador a persistir.

    Returns:
        None
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    ruta = os.path.join(MODEL_DIR, f"model_u_{uid}.pkl")
    tmp  = ruta + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(classifier, f)
    _reemplazar_con_reintentos(tmp, ruta)


def cargar_perfil(uid):
    """
    Carga el perfil JSON de un usuario desde ``data/``.

    Args:
        uid (int): Identificador del usuario.

    Returns:
        dict | None: El perfil, o ``None`` si no existe el archivo.
    """
    ruta = os.path.join(DATA_DIR, f"visual_profile_{uid}.json")
    if not os.path.exists(ruta):
        return None
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def guardar_perfil(uid, perfil):
    """
    Guarda el perfil JSON de un usuario en ``data/`` de forma atómica.

    Args:
        uid (int): Identificador del usuario.
        perfil (dict): Perfil a persistir.

    Returns:
        None
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    ruta = os.path.join(DATA_DIR, f"visual_profile_{uid}.json")
    tmp  = ruta + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(perfil, f, indent=4, ensure_ascii=False)
    _reemplazar_con_reintentos(tmp, ruta)


def reconstruir_perfil_desde_modelo(uid, classifier, perfil_viejo):
    """
    Regenera los componentes de conocimiento y los pesos desde el modelo.

    Conserva el resto del perfil (``session_history``, ``explicit_feedback``,
    etc.) y reutiliza el nombre y la URL de cada tema del perfil anterior.

    Args:
        uid (int): Identificador del usuario.
        classifier (INKClassifier): Clasificador con el estado actualizado.
        perfil_viejo (dict): Perfil previo del usuario.

    Returns:
        dict: Copia del perfil con ``knowledge_components`` y ``weights``
        recalculados.
    """
    models          = classifier.get_learner_model()
    knowledge_model = models[0]
    interest_model  = models[1]

    interest_dict = {tid: kc for tid, kc in interest_model.knowledge.topic_kc_pairs()}

    meta_map = {
        kc["topic_id"]: {"name": kc["topic_name"], "url": kc["topic_url"]}
        for kc in perfil_viejo.get("knowledge_components", [])
    }

    nuevos_kcs = []
    for topic_id, kc in knowledge_model.knowledge.topic_kc_pairs():
        meta   = meta_map.get(topic_id, {})
        int_kc = interest_dict.get(topic_id)

        nuevos_kcs.append({
            "topic_id":      topic_id,
            "topic_name":    meta.get("name", f"ID_{topic_id}"),
            "topic_url":     meta.get("url", ""),
            "mean":          kc.mean,
            "variance":      kc.variance,
            "interest_mean": int_kc.mean     if int_kc else 0.0,
            "interest_var":  int_kc.variance if int_kc else 0.5,
            "timestamp":     kc.timestamp,
        })

    perfil_actualizado = {**perfil_viejo}
    perfil_actualizado["knowledge_components"] = nuevos_kcs
    perfil_actualizado["weights"]              = extraer_pesos_reales(classifier)

    return perfil_actualizado


class TopicNoEncontradoError(Exception):
    """El tema indicado no existe en el modelo de conocimiento o de interés del usuario."""


def aplicar_correccion_directa(classifier, topic_id, tipo):
    """
    Corrige los componentes de conocimiento del modelo directamente con ``.update()``.

    No se usa ``fit()`` porque el ``InterestClassifier`` ignora la etiqueta y
    siempre sube el interés; la única forma de bajarlo o de reducir la
    varianza correctamente es modificar el componente de conocimiento (KC)
    directamente. Las medias y varianzas resultantes se limitan con
    ``logic.dominio`` para que backend y frontend usen siempre el mismo rango.

    Efecto por tipo de feedback::

        Tipo            Knowledge KC                  Interest KC
        sé más          mean sube, variance baja      sin cambio
        sé menos        mean baja, variance baja      sin cambio
        me interesa     sin cambio                    mean sube, variance baja
        no me interesa  sin cambio                    mean baja, variance sube

    Las magnitudes son ``delta_know`` (0.25), ``delta_int`` (0.75) y
    ``reduccion_var`` (0.15), calibradas contra el impacto medio de un
    evento real de ``fit()``, y ``aumento_var`` (0.30), una decisión de
    diseño sin equivalente empírico: la varianza sube cuando el usuario
    contradice al modelo.

    Args:
        classifier (INKClassifier): Clasificador a corregir. Se modifica
            en el lugar.
        topic_id (int): Identificador del tema a corregir.
        tipo (str): Uno de ``"sé más"``, ``"sé menos"``, ``"me interesa"`` o
            ``"no me interesa"``.

    Returns:
        INKClassifier: El mismo clasificador, ya corregido.

    Raises:
        TopicNoEncontradoError: Si el tema no existe en el modelo de
            conocimiento o de interés requerido por el tipo de corrección.
    """
    models          = classifier.get_learner_model()
    knowledge_model = models[0]
    interest_model  = models[1]

    ts_ahora = time.time()

    kc_know = None
    for tid, kc in knowledge_model.knowledge.topic_kc_pairs():
        if tid == topic_id:
            kc_know = kc
            break

    kc_int = None
    for tid, kc in interest_model.knowledge.topic_kc_pairs():
        if tid == topic_id:
            kc_int = kc
            break

    delta_know    = 0.25
    delta_int     = 0.75
    reduccion_var = 0.15
    aumento_var   = 0.30

    if tipo == "sé más":
        if kc_know is None:
            raise TopicNoEncontradoError(topic_id)
        base = kc_know.mean + delta_know
        kc_know.update(mean=clamp_conocimiento(base),
                        variance=clamp_variance(kc_know.variance - reduccion_var),
                        timestamp=ts_ahora)

    elif tipo == "sé menos":
        if kc_know is None:
            raise TopicNoEncontradoError(topic_id)
        base = kc_know.mean - delta_know
        kc_know.update(mean=clamp_conocimiento(base),
                        variance=clamp_variance(kc_know.variance - reduccion_var),
                        timestamp=ts_ahora)

    elif tipo == "me interesa":
        if kc_int is None:
            raise TopicNoEncontradoError(topic_id)
        kc_int.update(mean=clamp_interes(kc_int.mean + delta_int),
                       variance=clamp_variance(kc_int.variance - reduccion_var),
                       timestamp=ts_ahora)

    elif tipo == "no me interesa":
        if kc_int is None:
            raise TopicNoEncontradoError(topic_id)
        kc_int.update(mean=clamp_interes(kc_int.mean - delta_int),
                       variance=clamp_variance(kc_int.variance + aumento_var),
                       timestamp=ts_ahora)

    return classifier


@app.route("/usuarios", methods=["GET"])
def listar_usuarios():
    """
    Devuelve la lista de usuarios entrenados con su resumen.

    Returns:
        Response: El contenido de ``data/usuarios_entrenados.json`` como JSON,
        o un error con código 404 si todavía no hay usuarios entrenados.
    """
    ruta_indice = os.path.join(DATA_DIR, "usuarios_entrenados.json")
    if not os.path.exists(ruta_indice):
        return jsonify({"error": "No hay usuarios entrenados aún."}), 404
    with open(ruta_indice, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/perfil/<int:uid>", methods=["GET"])
def obtener_perfil(uid):
    """
    Devuelve el perfil completo de un usuario.

    Args:
        uid (int): Identificador del usuario, tomado de la URL.

    Returns:
        Response: El perfil como JSON, o un error con código 404 si no existe.
    """
    perfil = cargar_perfil(uid)
    if perfil is None:
        return jsonify({"error": f"No se encontró perfil para usuario {uid}"}), 404
    return jsonify(perfil)


def _coerce_topic_id(raw):
    """
    Normaliza un ``topic_id`` a ``int`` cuando es posible.

    Protege frente a tipos numéricos no nativos que lleguen desde el frontend.

    Args:
        raw (Any): Valor recibido en el cuerpo de la petición.

    Returns:
        int | Any: El valor convertido a ``int``, o el valor original si no
        se puede convertir.
    """
    try:
        return int(raw)
    except (TypeError, ValueError):
        return raw


@app.route("/feedback/<int:uid>", methods=["POST"])
def aplicar_feedback(uid):
    """
    Recibe feedback explícito del usuario y actualiza el modelo.

    Cuerpo JSON esperado::

        {
            "topic_id":   123,
            "topic_name": "Machine Learning",
            "tipo":       "sé más" | "sé menos" | "me interesa" | "no me interesa"
        }

    Aplica la corrección, guarda el modelo, registra el feedback en el
    historial junto con la media resultante y guarda el perfil actualizado.

    Args:
        uid (int): Identificador del usuario, tomado de la URL.

    Returns:
        Response: JSON con ``ok``, ``mensaje`` y el ``perfil`` actualizado.
        Responde 400 si el cuerpo, el ``topic_id`` o el
        ``tipo`` no son válidos, y 404 si no existe el modelo, el perfil o
        el tema.
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Body JSON requerido"}), 400

    topic_id   = _coerce_topic_id(body.get("topic_id"))
    topic_name = body.get("topic_name", f"ID_{topic_id}")
    tipo       = body.get("tipo", "sé más")

    if topic_id is None:
        return jsonify({"error": "topic_id es requerido"}), 400

    TIPOS_VALIDOS = ["sé más", "sé menos", "me interesa", "no me interesa"]
    if tipo not in TIPOS_VALIDOS:
        return jsonify({"error": f"tipo inválido. Opciones: {TIPOS_VALIDOS}"}), 400

    classifier = cargar_modelo(uid)
    if classifier is None:
        return jsonify({"error": f"No se encontró modelo para usuario {uid}"}), 404

    perfil = cargar_perfil(uid)
    if perfil is None:
        return jsonify({"error": f"No se encontró perfil para usuario {uid}"}), 404

    try:
        classifier = aplicar_correccion_directa(classifier, topic_id, tipo)
    except TopicNoEncontradoError:
        return jsonify({
            "error": f"El tema {topic_id} no existe en el modelo del usuario {uid}"
        }), 404

    guardar_modelo(uid, classifier)

    registro_feedback = {
        "timestamp":  time.time(),
        "topic_id":   topic_id,
        "topic_name": topic_name,
        "tipo":       tipo,
    }
    perfil.setdefault("explicit_feedback", []).append(registro_feedback)

    perfil_actualizado = reconstruir_perfil_desde_modelo(uid, classifier, perfil)

    nkc = next((k for k in perfil_actualizado["knowledge_components"]
                if k["topic_id"] == topic_id), None)
    if nkc:
        registro_feedback["mean_resultante"]          = nkc["mean"]
        registro_feedback["interest_mean_resultante"] = nkc["interest_mean"]

    guardar_perfil(uid, perfil_actualizado)

    return jsonify({
        "ok":      True,
        "mensaje": f"Feedback '{tipo}' aplicado sobre '{topic_name}'",
        "perfil":  perfil_actualizado,
    })


@app.route("/historial_feedback/<int:uid>", methods=["GET"])
def historial_feedback(uid):
    """
    Devuelve solo el historial de feedbacks explícitos de un usuario.

    Args:
        uid (int): Identificador del usuario, tomado de la URL.

    Returns:
        Response: Lista JSON de feedbacks aplicados, o un error con código
        404 si el usuario no tiene perfil.
    """
    perfil = cargar_perfil(uid)
    if perfil is None:
        return jsonify({"error": f"No se encontró perfil para usuario {uid}"}), 404
    return jsonify(perfil.get("explicit_feedback", []))


@app.route("/prediccion/<int:uid>/<int:topic_id>", methods=["GET"])
def obtener_prediccion(uid, topic_id):
    """
    Devuelve la probabilidad de enganche actual del modelo para un tema.

    Se consulta antes y después de aplicar feedback para medir su impacto.
    La predicción usa un evento sintético con ``mean=0.5`` y ``variance=0.3``
    que representa un recurso "promedio".

    Args:
        uid (int): Identificador del usuario, tomado de la URL.
        topic_id (int): Identificador del tema, tomado de la URL.

    Returns:
        Response: JSON con ``topic_id`` y ``probabilidad_enganche`` (redondeada
        a 4 decimales). Responde 404 si no existe el modelo y 500 si la
        predicción falla.
    """
    from truelearn.models import EventModel, Knowledge, KnowledgeComponent
    classifier = cargar_modelo(uid)
    if classifier is None:
        return jsonify({"error": f"No se encontró modelo para usuario {uid}"}), 404
    kc     = KnowledgeComponent(mean=0.5, variance=0.3)
    evento = EventModel(Knowledge({topic_id: kc}), event_time=time.time())
    try:
        prob = classifier.predict_proba(evento)
        return jsonify({"topic_id": topic_id,
                        "probabilidad_enganche": round(float(prob), 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


_training_log: list[str] = []
_training_done    = False
_training_active  = False
_training_lock    = threading.Lock()


@app.route("/entrenar", methods=["POST"])
def iniciar_entrenamiento():
    """
    Lanza el entrenador en un hilo separado.

    Captura la salida estándar del entrenador en ``_training_log`` para que el
    frontend la lea con ``/log_entrenamiento``. El estado compartido se
    protege con un candado, de modo que dos peticiones simultáneas no puedan
    iniciar dos entrenamientos en paralelo.

    Returns:
        Response: JSON con ``ok`` y ``mensaje``. ``ok`` es ``False`` si ya hay
        un entrenamiento en curso.
    """
    global _training_log, _training_done, _training_active

    with _training_lock:
        if _training_active:
            return jsonify({"ok": False, "mensaje": "Ya hay un entrenamiento en curso."})
        _training_active = True
        _training_log    = ["🚀 Iniciando entrenamiento..."]
        _training_done   = False

    def run():
        """
        Ejecuta el entrenamiento y actualiza el estado compartido.

        Redirige la salida estándar a un ``LogCapture``, registra el éxito o
        el error con su traceback, y al final marca el proceso como terminado
        y no activo.

        Returns:
            None
        """
        global _training_log, _training_done, _training_active
        import contextlib
        import io

        class LogCapture(io.TextIOBase):
            """Flujo de texto que acumula en ``_training_log`` lo que se imprime."""

            def write(self, msg):
                """
                Agrega un mensaje no vacío al log de entrenamiento.

                Args:
                    msg (str): Texto escrito en la salida estándar.

                Returns:
                    int: Cantidad de caracteres recibidos.
                """
                if msg.strip():
                    with _training_lock:
                        _training_log.append(msg.rstrip())
                return len(msg)

            def flush(self):
                """
                No hace nada, porque cada mensaje se guarda al escribirse.

                Returns:
                    None
                """
                pass

        try:
            from logic.entrenador import entrenar_modelo_usuario

            capture = LogCapture()
            with contextlib.redirect_stdout(capture):
                entrenar_modelo_usuario()

            administracion.guardar_base(sys.modules[__name__])

            with _training_lock:
                _training_log.append("✨ PROCESO FINALIZADO CON ÉXITO")
        except Exception as e:
            import traceback
            with _training_lock:
                _training_log.append(f"❌ ERROR: {e}")
                _training_log.append(traceback.format_exc())
        finally:
            with _training_lock:
                _training_done   = True
                _training_active = False

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return jsonify({"ok": True, "mensaje": "Entrenamiento iniciado."})


@app.route("/log_entrenamiento", methods=["GET"])
def log_entrenamiento():
    """
    Devuelve el log acumulado del entrenamiento en curso.

    El dashboard consulta este endpoint cada 2 segundos para mostrar el
    progreso.

    Returns:
        Response: JSON con ``log`` (lista de mensajes), ``done`` (si terminó)
        y ``active`` (si está en curso).
    """
    with _training_lock:
        return jsonify({
            "log":    list(_training_log),
            "done":   _training_done,
            "active": _training_active,
        })


# ── Administración del modelo (restablecer todo, feedbacks ficticios) ──────────
from logic import administracion  # noqa: E402

administracion.registrar(app, sys.modules[__name__])


if __name__ == "__main__":
    import io
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    debug_mode = os.environ.get("TRUELEARN_DEBUG", "0") == "1"
    if debug_mode:
        print("⚠️  Corriendo con debug=True: NO exponer este puerto fuera de localhost.")
    print("🚀 Backend TrueLearn corriendo en http://localhost:5000")
    app.run(debug=debug_mode, port=5000, threaded=True)
