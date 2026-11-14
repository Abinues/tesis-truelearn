import json
import os
import pickle

from logic.modelo_utils import extraer_pesos_reales, resolver_nombre_tema
from logic.peek_loader import cargar_dataset_peek

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Muestreo estratificado para el desplegable de la app ────────────────
# Los usuarios de ejemplo del desplegable ("Aprendiz") se eligen por
# muestreo aleatorio estratificado según su actividad, no por el orden del
# archivo ni con cortes de eventos fijados a mano:
#   1. Se calculan los cortes de la cantidad de eventos por usuario con
#      todos los usuarios del split de entrenamiento: terciles (33 % y
#      67 %) y percentil 95. Los cortes salen de los datos.
#   2. Los usuarios quedan repartidos en cuatro estratos:
#         "pocas"        hasta el tercil 1
#         "medias"       entre el tercil 1 y el tercil 2
#         "muchas"       entre el tercil 2 y el percentil 95
#         "muy_activas"  por encima del percentil 95 (el 5 % más activo)
#      El estrato "muy_activas" existe porque el tercio más activo empieza
#      en pocos eventos y sus perfiles tienen poca historia: la
#      demostración necesita perfiles con evolución visible.
#   3. De cada estrato se sortean N_USUARIOS_POR_ESTRATO usuarios AL AZAR
#      con semilla fija (reproducible).
# Es muestreo aleatorio estratificado (Cochran, 1977) con asignación igual
# (no proporcional) a propósito: interesa mostrar cada perfil de
# actividad, no reproducir las proporciones de la población.
N_USUARIOS_POR_ESTRATO = 30
SEMILLA_MUESTREO       = 2026
PERCENTIL_MUY_ACTIVOS  = 0.95
ESTRATOS_ACTIVIDAD     = ("pocas", "medias", "muchas", "muy_activas")

# Nota sobre los PESOS del modelo (novelty/interest/bias, ver
# calibrar_dominio.py): a diferencia de los KCs, hay UNA sola muestra de
# pesos por usuario, no una por tema; sus extremos no se calibran con este
# conjunto de usuarios de ejemplo.

# NUM_EPOCHS
# ─────────────────────────────────────────────────────────────────────────
# FIX METODOLÓGICO IMPORTANTE:
#
# La versión original entrenaba NUM_EPOCHS=50 pasadas completas sobre la
# MISMA secuencia de eventos históricos de un usuario, pero solo
# guardaba el snapshot de session_history durante la época 0. El modelo
# que finalmente se persistía en el .pkl (y con el que se calculaban las
# predicciones y sobre el que se aplicaban las correcciones del usuario)
# era el de después de las 50 épocas. Es decir: la curva de "Evolución
# del Aprendizaje" que ve el estudiante NUNCA terminaba en el mismo
# estado que el resto del dashboard mostraba como "estado actual" — dos
# fuentes de verdad distintas y contradictorias, justo en un proyecto
# cuyo objetivo central es la TRANSPARENCIA del modelo.
#
# Un aprendiz real vive cada evento una sola vez, en orden cronológico:
# eso es lo que TrueLearn está diseñado para modelar de forma online.
# Repetir la misma secuencia muchas veces no corresponde a ningún
# escenario real y no está justificado en la propuesta ni documentado en
# el código original (los valores fueron ajustados por prueba y error).
#
# Por eso el valor por defecto pasa a ser 1 (una sola pasada = fiel al
# comportamiento online real). Si igualmente se necesita más de una
# época para que el modelo "converja" a estimaciones menos ruidosas para
# la demo, subir este valor es válido, PERO ahora el snapshot de
# session_history se toma en la ÚLTIMA época (ver entrenar_usuario más
# abajo), no en la primera, así que el gráfico de evolución y el estado
# final del modelo SIEMPRE van a coincidir, sin importar cuántas épocas
# se usen. Documentar en la memoria la elección final de este valor.
NUM_EPOCHS = 1


def seleccionar_usuarios(train, n_muchas, n_pocas, umbral_muchas, umbral_pocas_max):
    """
    Recorre el dataset y devuelve dos listas:
      - usuarios_muchas: hasta n_muchas usuarios con >= umbral_muchas eventos
      - usuarios_pocas:  hasta n_pocas  usuarios con <= umbral_pocas_max eventos
    """
    usuarios_muchas = []
    usuarios_pocas  = []

    for uid, events in train:
        n = len(events)
        if len(usuarios_muchas) < n_muchas and n >= umbral_muchas:
            usuarios_muchas.append((uid, events))
        elif len(usuarios_pocas) < n_pocas and 1 <= n <= umbral_pocas_max:
            usuarios_pocas.append((uid, events))

        if len(usuarios_muchas) >= n_muchas and len(usuarios_pocas) >= n_pocas:
            break

    return usuarios_muchas, usuarios_pocas


def muestrear_estratificado(train, n_por_estrato=N_USUARIOS_POR_ESTRATO,
                            semilla=SEMILLA_MUESTREO,
                            percentil_muy_activos=PERCENTIL_MUY_ACTIVOS):
    """
    Muestreo aleatorio estratificado por actividad (terciles y percentil 95
    de la cantidad de eventos por usuario).

    Args:
        train: iterable de ``(uid, eventos)``. Se ignoran usuarios sin eventos.
        n_por_estrato (int): usuarios a sortear en cada estrato.
        semilla (int): semilla del generador (misma semilla, misma muestra).
        percentil_muy_activos (float): cuantil (0-1) que separa "muy_activas".

    Returns:
        tuple[list, dict]: lista de ``(uid, eventos, tipo)`` con ``tipo`` en
        ``"pocas"``, ``"medias"``, ``"muchas"`` o ``"muy_activas"``, y un dict
        con los tres cortes (``corte_pocas``, ``corte_medias``,
        ``corte_muchas``) y el tamaño de cada estrato en la población.
    """
    import random

    usuarios = [(uid, evs) for uid, evs in train if len(evs) >= 1]
    if not usuarios:
        return [], {"corte_pocas": None, "corte_medias": None,
                    "corte_muchas": None, "poblacion": {}}

    conteos = sorted(len(evs) for _, evs in usuarios)

    def cuantil(q):
        pos = (len(conteos) - 1) * q
        i = int(pos)
        j = min(i + 1, len(conteos) - 1)
        return conteos[i] + (conteos[j] - conteos[i]) * (pos - i)

    c1, c2, c3 = cuantil(1 / 3), cuantil(2 / 3), cuantil(percentil_muy_activos)

    estratos = {t: [] for t in ESTRATOS_ACTIVIDAD}
    for uid, evs in usuarios:
        n = len(evs)
        if n <= c1:
            tipo = "pocas"
        elif n <= c2:
            tipo = "medias"
        elif n <= c3:
            tipo = "muchas"
        else:
            tipo = "muy_activas"
        estratos[tipo].append((uid, evs))

    rng = random.Random(semilla)
    muestra = []
    for tipo in ESTRATOS_ACTIVIDAD:
        grupo = estratos[tipo]
        elegidos = rng.sample(grupo, min(n_por_estrato, len(grupo)))
        muestra.extend((uid, evs, tipo) for uid, evs in sorted(elegidos, key=lambda t: t[0]))

    info = {
        "corte_pocas": c1, "corte_medias": c2, "corte_muchas": c3,
        "poblacion": {t: len(g) for t, g in estratos.items()},
    }
    return muestra, info


def entrenar_usuario(uid, eventos, mapping, num_epochs=NUM_EPOCHS):
    """
    Entrena el INKClassifier para un usuario y devuelve el clasificador
    junto con el historial de sesiones y las estadísticas de engagement.

    FIX: el snapshot de cada "sesión" ahora se
    toma en la ÚLTIMA época (epoch == num_epochs - 1), no en la época 0.
    Así, sin importar cuántas épocas se configuren, el último punto de
    session_history SIEMPRE coincide con el estado final del classifier
    que se persiste y que se usa para predicciones/correcciones. Con
    NUM_EPOCHS=1 (el valor por defecto) esto es exactamente equivalente
    al comportamiento de "una sola pasada cronológica real".

    FIX: INKClassifier se importa acá adentro
    (no al principio del archivo) para que el resto de logic/entrenador.py
    —en particular seleccionar_usuarios(), una función pura sin ninguna
    relación con truelearn— se pueda importar y testear sin necesitar la
    librería instalada. Mismo patrón que ya usan logic/peek_loader.py y
    logic/api_backend.py.
    """
    from truelearn.learning import INKClassifier

    classifier = INKClassifier()

    num_engagements     = sum(1 for _, label in eventos if label)
    num_non_engagements = len(eventos) - num_engagements

    historial_sesiones_dict = {}
    ultima_epoch = num_epochs - 1

    for epoch in range(num_epochs):
        for event, label in eventos:
            # FIX: se encontraron eventos
            # reales en el dataset PEEK con topic_kc_pairs() vacío (0.01%
            # del total, ~14 de 14050 usuarios en train). TrueLearn no
            # los admite: classifier.fit() lanza
            # "ValueError: Each group must contain multiple ratings"
            # porque el InterestClassifier necesita al menos un tema para
            # armar la comparación. Nunca se manifestó en los tests
            # (todos los eventos sintéticos usados tenían al menos un
            # tema) ni en la corrida original de 25 usuarios (ninguno de
            # los 25 seleccionados tenía la mala suerte de incluir uno),
            # pero es prácticamente inevitable al entrenar miles de
            # usuarios reales. Se saltea el evento (no aporta ninguna
            # señal de todos modos, al no tener tema que actualizar); NO
            # se excluye de num_engagements/num_non_engagements más
            # arriba, porque esos conteos reflejan el comportamiento de
            # visualización real del estudiante, tenga o no el video
            # temas wikificados.
            if not any(True for _ in event.knowledge.topic_kc_pairs()):
                continue

            classifier.fit(event, label)

            # Foto del estado SOLO en la última época: es el pase cuyo
            # estado final es el que efectivamente se persiste.
            if epoch == ultima_epoch:
                t = event.event_time
                snapshot_means    = {}
                snapshot_variance = {}

                knowledge_model = classifier.get_learner_model()[0]
                for tid, kc in knowledge_model.knowledge.topic_kc_pairs():
                    nombre = resolver_nombre_tema(tid, mapping)
                    snapshot_means[nombre]    = kc.mean
                    snapshot_variance[nombre] = kc.variance

                # Si hay varios eventos en el mismo instante, se guarda el
                # estado final de esa "sesión" (last-write-wins por diseño).
                historial_sesiones_dict[t] = {
                    "means":     snapshot_means,
                    "variances": snapshot_variance,
                }

    # Convertir dict ordenado → lista con session_id
    session_history = []
    for session_id, t in enumerate(sorted(historial_sesiones_dict.keys()), start=1):
        snap = historial_sesiones_dict[t]
        session_history.append({
            "session_id": session_id,
            "timestamp":  t,
            "means":      snap["means"],
            "variances":  snap["variances"],
        })

    return classifier, session_history, num_engagements, num_non_engagements


def generar_perfil_json(uid, classifier, mapping, num_eng, num_non_eng, session_history):
    """
    Construye el diccionario del perfil completo y lo guarda en data/.
    Incluye los pesos REALES aprendidos (no los de configuración inicial).
    """
    models         = classifier.get_learner_model()
    knowledge_model = models[0]
    interest_model  = models[1]

    interest_dict = {tid: kc for tid, kc in interest_model.knowledge.topic_kc_pairs()}

    # ── Pesos reales ────────────────────────────────────────────────────
    pesos = extraer_pesos_reales(classifier)

    perfil = {
        "user_id":                  uid,
        "number_of_engagements":    num_eng,
        "number_of_non_engagements": num_non_eng,
        "weights": pesos,           # novelty/interest/bias con mean y variance
        "session_history":          session_history,
        "explicit_feedback":        [],
        "knowledge_components":     [],
    }

    for topic_id, kc in knowledge_model.knowledge.topic_kc_pairs():
        meta = mapping.get(topic_id)

        topic_name = "Unknown"
        topic_url  = ""

        if isinstance(meta, tuple):
            if len(meta) > 1 and "http" in str(meta[0]):
                topic_url  = meta[0]
                topic_name = meta[1]
            else:
                topic_name = meta[0]
                topic_url  = meta[1] if len(meta) > 1 else ""
        elif isinstance(meta, dict):
            topic_name = meta.get("name", "Unknown")
            topic_url  = meta.get("url",  "")
        else:
            topic_name = f"ID_{topic_id}"

        int_kc = interest_dict.get(topic_id)

        perfil["knowledge_components"].append({
            "topic_id":       topic_id,
            "topic_name":     topic_name,
            "topic_url":      topic_url,
            "mean":           kc.mean,
            "variance":       kc.variance,
            "interest_mean":  int_kc.mean     if int_kc else 0.0,
            "interest_var":   int_kc.variance if int_kc else 0.5,
            "timestamp":      kc.timestamp,
        })

    # Guardar JSON individual por usuario
    os.makedirs(DATA_DIR, exist_ok=True)
    ruta_json = os.path.join(DATA_DIR, f"visual_profile_{uid}.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(perfil, f, indent=4, ensure_ascii=False)

    return perfil, ruta_json


def entrenar_modelo_usuario():
    print("\n--- 🚀 INICIANDO MÓDULO DE ENTRENAMIENTO (multi-usuario) ---")

    try:
        # ── Cargar dataset ────────────────────────────────────────────────
        # FIX: se usa el loader centralizado, que
        # protege el os.chdir() con try/finally y con un Lock (thread-safe
        # frente al hilo de entrenamiento que lanza api_backend.py).
        print("📡 Cargando dataset PEEK...")
        train, _, mapping = cargar_dataset_peek(DATA_DIR)

        # ── Seleccionar usuarios (muestreo aleatorio estratificado) ──────
        todos_los_usuarios, info = muestrear_estratificado(
            train, N_USUARIOS_POR_ESTRATO, SEMILLA_MUESTREO)
        print(f"✅ Cortes de actividad (eventos por usuario): pocas ≤ {info['corte_pocas']:.1f} < "
              f"medias ≤ {info['corte_medias']:.1f} < muchas ≤ {info['corte_muchas']:.1f} < muy_activas")
        for tipo in ESTRATOS_ACTIVIDAD:
            elegidos = sum(1 for _, _, t in todos_los_usuarios if t == tipo)
            print(f"   {tipo:<11}: {elegidos} sorteados de {info['poblacion'][tipo]} en la población")

        # ── Entrenar y guardar ───────────────────────────────────────────
        os.makedirs(MODEL_DIR, exist_ok=True)
        resumen_todos = []
        usuarios_con_error = []

        for uid, eventos, tipo in todos_los_usuarios:
            print(f"\n🧠 Entrenando usuario {uid} [{tipo}] — {len(eventos)} interacciones...")

            # FIX: con miles de usuarios reales
            # de PEEK es prácticamente inevitable toparse con algún caso
            # inesperado en los datos (el bug de eventos sin temas del
            # mismo punto 27 fue el primero encontrado así). Antes, un
            # error en CUALQUIER usuario mataba el proceso entero,
            # descartando el trabajo ya hecho con todos los usuarios
            # anteriores (sus .pkl/.json individuales sí quedan guardados
            # en disco, pero el índice global nunca se escribía). Ahora
            # se seguÍa con el resto y se reporta al final quién falló.
            try:
                classifier, session_history, num_eng, num_non_eng = entrenar_usuario(
                    uid, eventos, mapping
                )

                # Guardar modelo .pkl
                ruta_pkl = os.path.join(MODEL_DIR, f"model_u_{uid}.pkl")
                with open(ruta_pkl, "wb") as f:
                    pickle.dump(classifier, f)
                print(f"   💾 Modelo guardado: {ruta_pkl}")

                # Generar y guardar JSON de perfil
                perfil, ruta_json = generar_perfil_json(
                    uid, classifier, mapping, num_eng, num_non_eng, session_history
                )
                print(f"   📝 JSON guardado:   {ruta_json}")
                print(f"   📊 Engagements: {num_eng} | Non: {num_non_eng} | Sesiones: {len(session_history)}")

                pesos = perfil["weights"]
                print(f"   ⚖️  Pesos → novelty: {pesos['novelty']['mean']:.4f} | "
                      f"interest: {pesos['interest']['mean']:.4f} | "
                      f"bias: {pesos['bias']['mean']:.4f}")

                resumen_todos.append({
                    "user_id":       uid,
                    "tipo":          tipo,
                    "interacciones": len(eventos),
                    "engagements":   num_eng,
                    "sesiones":      len(session_history),
                    "weights":       pesos,
                })
            except Exception as e:
                print(f"   ⚠️  Usuario {uid} FALLÓ, se lo saltea: {e}")
                usuarios_con_error.append({"user_id": uid, "tipo": tipo, "error": str(e)})

        if usuarios_con_error:
            print(f"\n⚠️  {len(usuarios_con_error)} usuario(s) fallaron y se saltearon: "
                  f"{[u['user_id'] for u in usuarios_con_error]}")

        # ── Guardar índice global ────────────────────────────────────────
        ruta_indice = os.path.join(DATA_DIR, "usuarios_entrenados.json")
        with open(ruta_indice, "w", encoding="utf-8") as f:
            json.dump(resumen_todos, f, indent=4, ensure_ascii=False)
        print(f"\n📋 Índice de usuarios guardado en: {ruta_indice}")
        print("\n--- ✨ PROCESO FINALIZADO CON ÉXITO ---")

    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # FIX: forzar UTF-8 en stdout/stderr, igual
    # que en logic/api_backend.py -- entrenador.py imprime bastantes
    # emojis (📡✅🧠💾...), y correrlo con la salida redirigida en Windows
    # (ej. un futuro pipeline de CI) tira UnicodeEncodeError con el
    # códec cp1252 por defecto. Se pone acá, no a nivel de módulo, para
    # no interferir con contextlib.redirect_stdout() que usa el endpoint
    # /entrenar de api_backend.py (ese redirige a un objeto propio sin
    # .reconfigure(), y nunca ejecuta este bloque).
    import io
    import sys
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    entrenar_modelo_usuario()