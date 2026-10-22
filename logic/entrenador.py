"""
Entrenamiento de modelos INKClassifier de TrueLearn sobre el dataset PEEK.

Selecciona usuarios con muchas y con pocas interacciones, entrena un modelo
por usuario y guarda el modelo (``models/*.pkl``), su perfil visual
(``data/visual_profile_*.json``) y un índice global
(``data/usuarios_entrenados.json``).

Constantes de configuración:
    N_USUARIOS_MUCHAS, N_USUARIOS_POCAS: cuántos usuarios de cada tipo se
        entrenan.
    UMBRAL_MUCHAS: mínimo de eventos para contar como usuario con "muchas"
        interacciones.
    UMBRAL_POCAS_MAX: máximo de eventos para contar como usuario con "pocas"
        interacciones.
    NUM_EPOCHS: pasadas sobre la secuencia de eventos de cada usuario. El
        valor 1 reproduce el comportamiento online real: cada evento se
        vive una sola vez, en orden cronológico.
"""
import json
import os
import pickle

from logic.modelo_utils import extraer_pesos_reales, resolver_nombre_tema
from logic.peek_loader import cargar_dataset_peek

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

N_USUARIOS_MUCHAS    = 60
N_USUARIOS_POCAS     = 40
UMBRAL_MUCHAS        = 40
UMBRAL_POCAS_MAX     = 10

NUM_EPOCHS = 1


def seleccionar_usuarios(train, n_muchas, n_pocas, umbral_muchas, umbral_pocas_max):
    """
    Selecciona usuarios con muchas y con pocas interacciones.

    Recorre el dataset en orden y se detiene en cuanto ambas listas se
    completan.

    Args:
        train (Iterable[tuple]): Pares ``(uid, eventos)`` del conjunto de
            entrenamiento de PEEK.
        n_muchas (int): Cantidad máxima de usuarios con muchas interacciones.
        n_pocas (int): Cantidad máxima de usuarios con pocas interacciones.
        umbral_muchas (int): Mínimo de eventos para entrar en el grupo de
            "muchas".
        umbral_pocas_max (int): Máximo de eventos para entrar en el grupo de
            "pocas". Se exige al menos un evento.

    Returns:
        tuple[list, list]: ``(usuarios_muchas, usuarios_pocas)``, cada una
        como lista de pares ``(uid, eventos)``.
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


def entrenar_usuario(uid, eventos, mapping, num_epochs=NUM_EPOCHS):
    """
    Entrena un INKClassifier con los eventos de un usuario.

    Los eventos sin temas se omiten porque ``fit()`` no los admite, pero sí
    cuentan en las estadísticas de enganche, que reflejan el comportamiento
    real del estudiante. La foto de cada sesión se toma en la última época,
    por lo que el último punto del historial coincide siempre con el estado
    final del clasificador.

    Args:
        uid (int): Identificador del usuario.
        eventos (list[tuple]): Pares ``(evento, etiqueta)`` del usuario, en
            orden cronológico.
        mapping (dict): Mapeo de temas del dataset PEEK.
        num_epochs (int): Número de pasadas sobre los eventos. Por defecto
            ``NUM_EPOCHS``.

    Returns:
        tuple: ``(classifier, session_history, num_engagements,
        num_non_engagements)``, donde ``session_history`` es una lista de
        dicts con ``session_id``, ``timestamp``, ``means`` y ``variances``.
    """
    from truelearn.learning import INKClassifier

    classifier = INKClassifier()

    num_engagements     = sum(1 for _, label in eventos if label)
    num_non_engagements = len(eventos) - num_engagements

    historial_sesiones_dict = {}
    ultima_epoch = num_epochs - 1

    for epoch in range(num_epochs):
        for event, label in eventos:
            if not any(True for _ in event.knowledge.topic_kc_pairs()):
                continue

            classifier.fit(event, label)

            if epoch == ultima_epoch:
                t = event.event_time
                snapshot_means    = {}
                snapshot_variance = {}

                knowledge_model = classifier.get_learner_model()[0]
                for tid, kc in knowledge_model.knowledge.topic_kc_pairs():
                    nombre = resolver_nombre_tema(tid, mapping)
                    snapshot_means[nombre]    = kc.mean
                    snapshot_variance[nombre] = kc.variance

                historial_sesiones_dict[t] = {
                    "means":     snapshot_means,
                    "variances": snapshot_variance,
                }

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
    Construye el perfil visual completo de un usuario y lo guarda en ``data/``.

    El perfil incluye los pesos aprendidos (no los de configuración inicial),
    el historial de sesiones y, por cada tema, la media y varianza de
    conocimiento junto con la media y varianza de interés.

    Args:
        uid (int): Identificador del usuario.
        classifier (INKClassifier): Clasificador ya entrenado.
        mapping (dict): Mapeo de temas del dataset PEEK.
        num_eng (int): Cantidad de eventos con enganche.
        num_non_eng (int): Cantidad de eventos sin enganche.
        session_history (list[dict]): Historial devuelto por
            ``entrenar_usuario``.

    Returns:
        tuple[dict, str]: El perfil como diccionario y la ruta del archivo
        ``visual_profile_{uid}.json`` donde se guardó.
    """
    models         = classifier.get_learner_model()
    knowledge_model = models[0]
    interest_model  = models[1]

    interest_dict = {tid: kc for tid, kc in interest_model.knowledge.topic_kc_pairs()}

    pesos = extraer_pesos_reales(classifier)

    perfil = {
        "user_id":                  uid,
        "number_of_engagements":    num_eng,
        "number_of_non_engagements": num_non_eng,
        "weights": pesos,
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

    os.makedirs(DATA_DIR, exist_ok=True)
    ruta_json = os.path.join(DATA_DIR, f"visual_profile_{uid}.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(perfil, f, indent=4, ensure_ascii=False)

    return perfil, ruta_json


def entrenar_modelo_usuario():
    """
    Ejecuta el entrenamiento completo multiusuario sobre PEEK.

    Carga el dataset, selecciona los usuarios con muchas y pocas
    interacciones, entrena uno a uno guardando su modelo y su perfil, y al
    final escribe el índice global ``data/usuarios_entrenados.json``. Si un
    usuario falla, se omite y se informa al final, sin interrumpir al resto.
    Cualquier error general se imprime con su traceback en lugar de
    propagarse.

    Returns:
        None: El progreso se informa por salida estándar.
    """
    print("\n--- 🚀 INICIANDO MÓDULO DE ENTRENAMIENTO (multi-usuario) ---")

    try:
        print("📡 Cargando dataset PEEK...")
        train, _, mapping = cargar_dataset_peek(DATA_DIR)

        usuarios_muchas, usuarios_pocas = seleccionar_usuarios(
            train,
            N_USUARIOS_MUCHAS, N_USUARIOS_POCAS,
            UMBRAL_MUCHAS, UMBRAL_POCAS_MAX,
        )

        print(f"✅ Usuarios con muchas interacciones (≥{UMBRAL_MUCHAS}): {len(usuarios_muchas)}")
        print(f"✅ Usuarios con pocas  interacciones (≤{UMBRAL_POCAS_MAX}):  {len(usuarios_pocas)}")

        todos_los_usuarios = (
            [(uid, evs, "muchas") for uid, evs in usuarios_muchas] +
            [(uid, evs, "pocas")  for uid, evs in usuarios_pocas]
        )

        os.makedirs(MODEL_DIR, exist_ok=True)
        resumen_todos = []
        usuarios_con_error = []

        for uid, eventos, tipo in todos_los_usuarios:
            print(f"\n🧠 Entrenando usuario {uid} [{tipo}] — {len(eventos)} interacciones...")

            try:
                classifier, session_history, num_eng, num_non_eng = entrenar_usuario(
                    uid, eventos, mapping
                )

                ruta_pkl = os.path.join(MODEL_DIR, f"model_u_{uid}.pkl")
                with open(ruta_pkl, "wb") as f:
                    pickle.dump(classifier, f)
                print(f"   💾 Modelo guardado: {ruta_pkl}")

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
    import io
    import sys
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    entrenar_modelo_usuario()
