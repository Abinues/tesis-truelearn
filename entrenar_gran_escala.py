"""
Entrenamiento a gran escala sobre PEEK, para calibrar ``logic/dominio.py``.

Entrena a todos los usuarios del conjunto de entrenamiento de PEEK que
califican como "muchas" (``UMBRAL_MUCHAS`` eventos o más) o "pocas"
(``UMBRAL_POCAS_MAX`` eventos o menos), en lugar de la submuestra reducida de
``logic/entrenador.py``. Los perfiles y modelos se guardan en
``data_calibracion/``, una carpeta aparte que no se mezcla con ``data/`` (la
que lee el dashboard).

Uso:
    python entrenar_gran_escala.py

El proceso tarda varios minutos (unos 10 a 12 para unos 8150 usuarios) y
requiere poder descargar el dataset PEEK. Al terminar, se analiza el
resultado con:

    python calibrar_dominio.py data_calibracion

Constantes:
    UMBRAL_MUCHAS: Mínimo de eventos para contar como usuario con "muchas"
        interacciones.
    UMBRAL_POCAS_MAX: Máximo de eventos para contar como usuario con "pocas"
        interacciones.
    LIMITE_ALTO: Tope de usuarios por grupo, lo bastante alto para no limitar
        la selección en la práctica.
"""
import json
import os
import pickle
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CALIBRACION_DIR = os.path.join(BASE_DIR, "data_calibracion")

UMBRAL_MUCHAS = 40
UMBRAL_POCAS_MAX = 10
LIMITE_ALTO = 10_000_000


def main():
    """
    Entrena a todos los usuarios que califican y guarda sus modelos y perfiles.

    Carga PEEK, selecciona todos los usuarios con muchas y con pocas
    interacciones, y entrena un modelo por usuario. Cada modelo se guarda como
    ``model_u_{uid}.pkl`` y cada perfil como ``visual_profile_{uid}.json`` en
    ``data_calibracion/``. Imprime el progreso cada 500 usuarios. Los usuarios
    que fallan se omiten y se listan en
    ``data_calibracion/errores_entrenamiento.json``.

    Returns:
        None: El resultado se informa por salida estándar.
    """
    from logic.entrenador import entrenar_usuario, generar_perfil_json, seleccionar_usuarios
    from logic.peek_loader import cargar_dataset_peek

    os.makedirs(DATA_CALIBRACION_DIR, exist_ok=True)

    print("📡 Cargando dataset PEEK (puede tardar si es la primera vez)...")
    train, _, mapping = cargar_dataset_peek(DATA_CALIBRACION_DIR)

    print("🔍 Seleccionando TODOS los usuarios reales que califican...")
    usuarios_muchas, usuarios_pocas = seleccionar_usuarios(
        train, n_muchas=LIMITE_ALTO, n_pocas=LIMITE_ALTO,
        umbral_muchas=UMBRAL_MUCHAS, umbral_pocas_max=UMBRAL_POCAS_MAX,
    )
    print(f"   'muchas' (>={UMBRAL_MUCHAS} eventos): {len(usuarios_muchas)} usuarios reales")
    print(f"   'pocas'  (<={UMBRAL_POCAS_MAX} eventos): {len(usuarios_pocas)} usuarios reales")

    todos = [(uid, evs) for uid, evs in usuarios_muchas] + \
            [(uid, evs) for uid, evs in usuarios_pocas]
    total = len(todos)
    print(f"📋 Total a entrenar: {total}")
    print("   (a este ritmo, esperá varios minutos -- se imprime progreso cada 500 usuarios)\n")

    ok = 0
    errores = []
    t0 = time.time()

    import logic.entrenador as _entrenador_mod
    _entrenador_mod.DATA_DIR = DATA_CALIBRACION_DIR

    for i, (uid, eventos) in enumerate(todos):
        try:
            classifier, session_history, num_eng, num_non_eng = entrenar_usuario(uid, eventos, mapping)

            ruta_pkl = os.path.join(DATA_CALIBRACION_DIR, f"model_u_{uid}.pkl")
            with open(ruta_pkl, "wb") as f:
                pickle.dump(classifier, f)

            generar_perfil_json(uid, classifier, mapping, num_eng, num_non_eng, session_history)
            ok += 1
        except Exception as e:
            errores.append({"user_id": uid, "error": str(e)})

        if (i + 1) % 500 == 0:
            transcurrido = time.time() - t0
            restante = transcurrido / (i + 1) * (total - i - 1)
            print(f"   ... {i + 1}/{total} procesados "
                  f"({transcurrido:.0f}s transcurridos, ~{restante:.0f}s restantes)")

    print(f"\n✅ Terminado en {time.time() - t0:.1f}s")
    print(f"   Usuarios entrenados OK: {ok}")
    print(f"   Usuarios con error (saltados): {len(errores)}")

    if errores:
        ruta_errores = os.path.join(DATA_CALIBRACION_DIR, "errores_entrenamiento.json")
        with open(ruta_errores, "w", encoding="utf-8") as f:
            json.dump(errores, f, indent=2, ensure_ascii=False)
        print(f"   Detalle de errores guardado en: {ruta_errores}")

    print(f"\n📂 Perfiles guardados en: {DATA_CALIBRACION_DIR}")
    print("\nSiguiente paso:")
    print("   python calibrar_dominio.py data_calibracion")


if __name__ == "__main__":
    import io
    import sys
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
