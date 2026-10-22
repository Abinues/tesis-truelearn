"""
Administración del modelo: restablecer todo y feedbacks ficticios.

Agrega al backend tres endpoints que usa la sección «Administración del
modelo» de la página de inicio:

    GET  /administracion/estado                  Resumen: aprendices y feedbacks.
    POST /administracion/restablecer             Deja todo sin feedbacks ni correcciones.
    POST /administracion/feedbacks_ficticios     Agrega correcciones de prueba, con fechas distintas.

Cómo se restablece
------------------
Cada aprendiz tiene una *copia limpia* de su modelo y de su perfil, tal como
quedaron al terminar el entrenamiento:

    models/base/model_u_<uid>.pkl
    data/base/visual_profile_<uid>.json

«Restablecer todo» copia esos archivos de vuelta sobre los de trabajo. Así se
borran todos los feedbacks (reales y ficticios) y también su efecto sobre el
modelo, sin volver a entrenar.

La copia limpia se crea:

    - Al terminar «Entrenar modelos desde PEEK» (``guardar_base``).
    - La primera vez que se consulta el estado, para los aprendices que todavía
      no tienen feedbacks (``asegurar_bases``).

Un aprendiz que ya tenía feedbacks antes de que existiera su copia limpia no se
puede restablecer: el modelo ya fue modificado. Se informa en la respuesta y se
soluciona entrenando de nuevo, que además deja la copia limpia.

El módulo no importa ``api_backend``: recibe ese módulo como argumento
(``registrar(app, api)``) y usa sus funciones y rutas en el momento de cada
llamada. Así evita importaciones circulares y los tests pueden cambiar las
carpetas de datos.
"""
import json
import os
import random
import shutil
import threading
import time
from datetime import datetime, timedelta

from flask import Blueprint, jsonify, request

TIPOS_FICTICIOS = ("sé más", "sé menos", "me interesa", "no me interesa")
PESOS_TIPOS = (3, 2, 3, 1)
MAX_POR_APRENDIZ = 50
DIAS_FICTICIOS = 30          # los feedbacks ficticios se reparten en los últimos 30 días
MAX_DIAS_FICTICIOS = 365

_bloqueo = threading.Lock()

# Cantidad de feedbacks por perfil, guardada junto con la fecha y el tamaño del
# archivo. Evita leer todos los perfiles (pueden ser grandes) cada vez que la
# página de inicio pide el estado.
_cache_feedbacks = {}


# ── Rutas ───────────────────────────────────────────────────────────────────

def _ruta_perfil(api, uid, base=False):
    """Ruta del perfil JSON de un aprendiz (de trabajo o copia limpia).

    Args:
        api (module): Módulo ``api_backend``.
        uid (int): Identificador del aprendiz.
        base (bool): ``True`` para la copia limpia.

    Returns:
        str: Ruta del archivo.
    """
    carpeta = os.path.join(api.DATA_DIR, "base") if base else api.DATA_DIR
    return os.path.join(carpeta, f"visual_profile_{uid}.json")


def _ruta_modelo(api, uid, base=False):
    """Ruta del modelo ``.pkl`` de un aprendiz (de trabajo o copia limpia).

    Args:
        api (module): Módulo ``api_backend``.
        uid (int): Identificador del aprendiz.
        base (bool): ``True`` para la copia limpia.

    Returns:
        str: Ruta del archivo.
    """
    carpeta = os.path.join(api.MODEL_DIR, "base") if base else api.MODEL_DIR
    return os.path.join(carpeta, f"model_u_{uid}.pkl")


def _copiar(api, origen, destino):
    """Copia un archivo de forma atómica (archivo temporal y renombrado).

    Usa ``_reemplazar_con_reintentos`` del backend, si existe, porque en Windows
    el renombrado puede fallar un instante si otro programa tiene abierto el
    archivo.

    Args:
        api (module): Módulo ``api_backend``.
        origen (str): Archivo a copiar.
        destino (str): Archivo de destino; se crea la carpeta si no existe.
    """
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    tmp = destino + ".tmp"
    shutil.copyfile(origen, tmp)
    getattr(api, "_reemplazar_con_reintentos", os.replace)(tmp, destino)


def _ids(api):
    """Identificadores de los aprendices entrenados, según el índice global.

    Args:
        api (module): Módulo ``api_backend``.

    Returns:
        list[int]: Identificadores, o lista vacía si no hay índice.
    """
    ruta = os.path.join(api.DATA_DIR, "usuarios_entrenados.json")
    if not os.path.exists(ruta):
        return []
    with open(ruta, "r", encoding="utf-8") as f:
        return [u["user_id"] for u in json.load(f)]


def _tiene_base(api, uid):
    """Indica si el aprendiz tiene su copia limpia completa (perfil y modelo)."""
    return (os.path.exists(_ruta_perfil(api, uid, base=True))
            and os.path.exists(_ruta_modelo(api, uid, base=True)))


def _guardar_base_de(api, uid):
    """Guarda el estado actual de un aprendiz como su copia limpia.

    Args:
        api (module): Módulo ``api_backend``.
        uid (int): Identificador del aprendiz.

    Returns:
        bool: ``True`` si había perfil y modelo para copiar.
    """
    perfil, modelo = _ruta_perfil(api, uid), _ruta_modelo(api, uid)
    if not (os.path.exists(perfil) and os.path.exists(modelo)):
        return False
    _copiar(api, perfil, _ruta_perfil(api, uid, base=True))
    _copiar(api, modelo, _ruta_modelo(api, uid, base=True))
    return True


# ── Operaciones ─────────────────────────────────────────────────────────────

def guardar_base(api):
    """Guarda la copia limpia de todos los aprendices (se llama tras entrenar).

    Nunca lanza excepciones: un fallo al copiar no debe tumbar el entrenamiento.

    Args:
        api (module): Módulo ``api_backend``.

    Returns:
        int: Cantidad de aprendices con copia limpia guardada.
    """
    guardados = 0
    try:
        for uid in _ids(api):
            if _guardar_base_de(api, uid):
                guardados += 1
    except Exception as e:  # noqa: BLE001
        api.app.logger.error(f"No se pudo guardar la copia limpia: {e}")
    return guardados


def asegurar_bases(api):
    """Crea la copia limpia de los aprendices que no tienen feedbacks y no la tienen.

    Un aprendiz sin feedbacks no fue modificado, así que su estado actual es el
    del entrenamiento.

    Args:
        api (module): Módulo ``api_backend``.

    Returns:
        int: Cantidad de copias limpias creadas.
    """
    creadas = 0
    for uid in _ids(api):
        if _tiene_base(api, uid):
            continue
        cuenta = _contar_feedbacks(api, uid)
        if cuenta is None or cuenta[0]:
            continue
        if _guardar_base_de(api, uid):
            creadas += 1
    return creadas


def _contar_feedbacks(api, uid):
    """Cuenta los feedbacks de un aprendiz, leyendo el perfil solo si cambió.

    Args:
        api (module): Módulo ``api_backend``.
        uid (int): Identificador del aprendiz.

    Returns:
        tuple[int, int] | None: ``(feedbacks, ficticios)``, o ``None`` si el
        aprendiz no tiene perfil.
    """
    ruta = _ruta_perfil(api, uid)
    try:
        info = os.stat(ruta)
    except FileNotFoundError:
        return None
    version = (info.st_mtime_ns, info.st_size)
    guardado = _cache_feedbacks.get(ruta)
    if guardado and guardado[0] == version:
        return guardado[1]
    perfil = api.cargar_perfil(uid)
    if perfil is None:
        return None
    feedbacks = perfil.get("explicit_feedback", [])
    cuenta = (len(feedbacks), sum(1 for f in feedbacks if f.get("ficticio")))
    _cache_feedbacks[ruta] = (version, cuenta)
    return cuenta


def estado(api, crear_bases=True):
    """Resumen del estado de los aprendices y sus feedbacks.

    Crea antes las copias limpias que falten (ver ``asegurar_bases``), salvo que
    se indique lo contrario.

    Args:
        api (module): Módulo ``api_backend``.
        crear_bases (bool): Si es ``False`` no crea copias limpias (se usa
            mientras se entrena, cuando los archivos todavía están cambiando).

    Returns:
        dict: ``aprendices``, ``feedbacks``, ``ficticios``, ``con_feedbacks`` y
        ``sin_copia_limpia`` (aprendices con feedbacks que no se pueden
        restablecer sin entrenar de nuevo).
    """
    if crear_bases:
        asegurar_bases(api)
    resumen = {"aprendices": 0, "feedbacks": 0, "ficticios": 0,
               "con_feedbacks": 0, "sin_copia_limpia": 0}
    for uid in _ids(api):
        cuenta = _contar_feedbacks(api, uid)
        if cuenta is None:
            continue
        resumen["aprendices"] += 1
        resumen["feedbacks"] += cuenta[0]
        resumen["ficticios"] += cuenta[1]
        if cuenta[0]:
            resumen["con_feedbacks"] += 1
            if not _tiene_base(api, uid):
                resumen["sin_copia_limpia"] += 1
    return resumen


def restablecer(api):
    """Deja a todos los aprendices como quedaron al terminar el entrenamiento.

    Args:
        api (module): Módulo ``api_backend``.

    Returns:
        dict: ``restaurados`` (aprendices vueltos a su copia limpia),
        ``feedbacks_eliminados`` y ``sin_copia`` (identificadores de los que no
        tenían copia limpia y ya tenían feedbacks, que no se tocaron).
    """
    asegurar_bases(api)
    restaurados, eliminados, sin_copia = 0, 0, []
    for uid in _ids(api):
        perfil = api.cargar_perfil(uid)
        if perfil is None:
            continue
        if not _tiene_base(api, uid):
            if perfil.get("explicit_feedback"):
                sin_copia.append(uid)
            continue
        eliminados += len(perfil.get("explicit_feedback", []))
        _copiar(api, _ruta_modelo(api, uid, base=True), _ruta_modelo(api, uid))
        _copiar(api, _ruta_perfil(api, uid, base=True), _ruta_perfil(api, uid))
        restaurados += 1
    return {"restaurados": restaurados, "feedbacks_eliminados": eliminados,
            "sin_copia": sin_copia}


def _fechas_ficticias(azar, cantidad, dias):
    """Elige fechas distintas, dentro de los últimos ``dias`` días, ordenadas de la más vieja a la más nueva.

    Cada fecha cae en un día diferente (desde ayer hacia atrás) y a una hora
    al azar entre las 8:00 y las 22:00. Si se piden más feedbacks que días, los
    días se repiten, pero nunca hay una fecha en el futuro.

    Args:
        azar (random.Random): Generador de números aleatorios.
        cantidad (int): Cantidad de fechas.
        dias (int): Cantidad de días hacia atrás desde ayer.

    Returns:
        list[float]: Marcas de tiempo (segundos desde 1970), de menor a mayor.
    """
    posibles = range(1, dias + 1)
    if cantidad <= dias:
        elegidos = azar.sample(posibles, cantidad)
    else:
        elegidos = [azar.choice(posibles) for _ in range(cantidad)]
    hoy = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    fechas = [(hoy - timedelta(days=d)).timestamp() + azar.randint(8 * 3600, 22 * 3600)
              for d in elegidos]
    return sorted(fechas)


def _agregar_a_aprendiz(api, uid, cantidad, azar, dias=DIAS_FICTICIOS):
    """Aplica ``cantidad`` correcciones ficticias a un aprendiz y las registra.

    Cada corrección usa la misma función que los feedbacks reales
    (``aplicar_correccion_directa``) y queda en ``explicit_feedback`` marcada
    con ``"ficticio": True``. Cada uno lleva una fecha distinta de los últimos
    ``dias`` días (ver ``_fechas_ficticias``), en el mismo orden en que se
    aplican al modelo.

    Args:
        api (module): Módulo ``api_backend``.
        uid (int): Identificador del aprendiz.
        cantidad (int): Cantidad de feedbacks a agregar.
        azar (random.Random): Generador de números aleatorios.
        dias (int): Ventana de días en la que se reparten las fechas.

    Returns:
        int: Cantidad de feedbacks agregados.
    """
    classifier = api.cargar_modelo(uid)
    perfil = api.cargar_perfil(uid)
    temas = perfil.get("knowledge_components", []) if perfil else []
    if classifier is None or not temas:
        return 0

    if not _tiene_base(api, uid) and not perfil.get("explicit_feedback"):
        _guardar_base_de(api, uid)

    fechas = _fechas_ficticias(azar, cantidad, dias)
    registros = []
    for _ in range(cantidad):
        tema = azar.choice(temas)
        tipo = azar.choices(TIPOS_FICTICIOS, weights=PESOS_TIPOS)[0]
        try:
            api.aplicar_correccion_directa(classifier, tema["topic_id"], tipo)
        except api.TopicNoEncontradoError:
            continue
        conocimiento, interes = classifier.get_learner_model()[:2]
        mean = next((kc.mean for t, kc in conocimiento.knowledge.topic_kc_pairs()
                     if t == tema["topic_id"]), None)
        mean_int = next((kc.mean for t, kc in interes.knowledge.topic_kc_pairs()
                         if t == tema["topic_id"]), None)
        registros.append({
            "timestamp":  fechas[len(registros)],
            "topic_id":   tema["topic_id"],
            "topic_name": tema["topic_name"],
            "tipo":       tipo,
            "mean_resultante":          mean,
            "interest_mean_resultante": mean_int,
            "ficticio":   True,
        })

    if not registros:
        return 0
    api.guardar_modelo(uid, classifier)
    perfil.setdefault("explicit_feedback", []).extend(registros)
    api.guardar_perfil(uid, api.reconstruir_perfil_desde_modelo(uid, classifier, perfil))
    return len(registros)


def agregar_ficticios(api, por_aprendiz, semilla=None, dias=DIAS_FICTICIOS):
    """Agrega feedbacks ficticios a todos los aprendices entrenados.

    Args:
        api (module): Módulo ``api_backend``.
        por_aprendiz (int): Cantidad de feedbacks por aprendiz.
        semilla (int | None): Semilla para obtener siempre los mismos feedbacks
            (los días elegidos dependen además de la fecha de hoy).
        dias (int): Ventana de días en la que se reparten las fechas.

    Returns:
        dict: ``aprendices`` (a los que se les agregó alguno) y ``feedbacks``.
    """
    asegurar_bases(api)
    aprendices, total = 0, 0
    for uid in _ids(api):
        azar = random.Random(f"{semilla}-{uid}") if semilla is not None else random.Random()
        agregados = _agregar_a_aprendiz(api, uid, por_aprendiz, azar, dias)
        if agregados:
            aprendices += 1
            total += agregados
    return {"aprendices": aprendices, "feedbacks": total}


# ── Endpoints ───────────────────────────────────────────────────────────────

def registrar(app, api):
    """Registra los endpoints de administración en la aplicación Flask.

    Args:
        app (flask.Flask): Aplicación del backend.
        api (module): Módulo ``api_backend`` (se consulta en cada llamada).
    """
    if "administracion" in app.blueprints:
        return
    bp = Blueprint("administracion", __name__, url_prefix="/administracion")

    def _ocupado():
        """Respuesta de error si hay un entrenamiento o una acción en curso."""
        if getattr(api, "_training_active", False):
            return jsonify({"ok": False, "error": "Hay un entrenamiento en curso."}), 409
        return None

    @bp.route("/estado", methods=["GET"])
    def ruta_estado():
        """Devuelve cuántos aprendices y feedbacks hay."""
        entrenando = bool(getattr(api, "_training_active", False))
        datos = estado(api, crear_bases=not entrenando)
        datos["entrenando"] = entrenando
        return jsonify(datos)

    @bp.route("/restablecer", methods=["POST"])
    def ruta_restablecer():
        """Borra todos los feedbacks y vuelve los modelos a su estado entrenado."""
        if (ocupado := _ocupado()):
            return ocupado
        if not _bloqueo.acquire(blocking=False):
            return jsonify({"ok": False, "error": "Hay otra acción en curso."}), 409
        try:
            return jsonify({"ok": True, **restablecer(api)})
        finally:
            _bloqueo.release()

    @bp.route("/feedbacks_ficticios", methods=["POST"])
    def ruta_ficticios():
        """Agrega feedbacks ficticios con fechas distintas.

        Body: ``{"por_aprendiz": 5, "dias": 30, "semilla": 1}`` (``dias`` y
        ``semilla`` son opcionales).
        """
        if (ocupado := _ocupado()):
            return ocupado
        cuerpo = request.get_json(silent=True) or {}
        try:
            por_aprendiz = int(cuerpo.get("por_aprendiz", 5))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "por_aprendiz debe ser un número entero."}), 400
        if not 1 <= por_aprendiz <= MAX_POR_APRENDIZ:
            return jsonify({"ok": False,
                            "error": f"por_aprendiz debe estar entre 1 y {MAX_POR_APRENDIZ}."}), 400
        try:
            dias = int(cuerpo.get("dias", DIAS_FICTICIOS))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "dias debe ser un número entero."}), 400
        if not 1 <= dias <= MAX_DIAS_FICTICIOS:
            return jsonify({"ok": False,
                            "error": f"dias debe estar entre 1 y {MAX_DIAS_FICTICIOS}."}), 400
        if not _bloqueo.acquire(blocking=False):
            return jsonify({"ok": False, "error": "Hay otra acción en curso."}), 409
        try:
            return jsonify({"ok": True, **agregar_ficticios(
                api, por_aprendiz, cuerpo.get("semilla"), dias)})
        finally:
            _bloqueo.release()

    app.register_blueprint(bp)
