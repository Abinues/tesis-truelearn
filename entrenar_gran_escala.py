"""
entrenar_gran_escala.py  (versión CENSO)
─────────────────────────────────────────────────────────────────────────────
Entrena a TODOS los usuarios del split de entrenamiento de PEEK que tengan
al menos un evento, sin ningún corte por cantidad de eventos.

Por qué cambió respecto de la versión anterior
----------------------------------------------
La versión anterior solo entrenaba a quienes tenían >= 40 eventos
("muchas") o <= 10 ("pocas"). Esos cortes eran un criterio de diseño sin
fuente que los respalde, y dejaban afuera a los usuarios intermedios
(11 a 39 eventos), que eran una parte importante del dataset. Calibrar
la escala con una submuestra que excluye a una parte de la población
sesga la calibración. Un censo (todos los usuarios) no necesita
justificar ningún corte: la calibración describe a la población completa.

Qué hace
--------
  1. Carga PEEK reutilizando la descarga que ya está en data_calibracion/
     (no vuelve a bajar el dataset).
  2. Entrena a todos los usuarios con >= 1 evento.
  3. Escribe los perfiles en data_calibracion_todos/ (carpeta nueva, no
     toca data_calibracion/ ni data/).
  4. Guarda eventos_por_usuario.json ({user_id: nº de eventos}), que usan
     calibracion_por_estrato.py y el muestreo estratificado de la app.

No guarda los modelos .pkl (no hacen falta para calibrar y ocuparían
varios GB); si los querés, poné GUARDAR_MODELOS = True.

Uso:
    python entrenar_gran_escala.py
    python calibrar_dominio.py data_calibracion_todos
    python calibracion_por_estrato.py data_calibracion_todos

Tarda más que la versión anterior (más usuarios). Imprime progreso cada
500 usuarios.
"""
import json
import os
import pickle
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data_calibracion")         # descarga ya hecha
SALIDA_DIR = os.path.join(BASE_DIR, "data_calibracion_todos")    # perfiles del censo
GUARDAR_MODELOS = False

UMBRAL_MINIMO = 1            # todos los usuarios con al menos 1 evento
LIMITE_ALTO = 10_000_000     # "sin límite práctico"


def main():
    from logic.entrenador import entrenar_usuario, generar_perfil_json, seleccionar_usuarios
    from logic.peek_loader import cargar_dataset_peek

    os.makedirs(SALIDA_DIR, exist_ok=True)

    print("📡 Cargando dataset PEEK (se reutiliza la descarga de data_calibracion/)...")
    train, _, mapping = cargar_dataset_peek(DATASET_DIR)

    print("🔍 Seleccionando TODOS los usuarios con al menos 1 evento...")
    # Con umbral_muchas = 1 todos los usuarios caen en el primer grupo.
    todos_los_usuarios, _ = seleccionar_usuarios(
        train, n_muchas=LIMITE_ALTO, n_pocas=0,
        umbral_muchas=UMBRAL_MINIMO, umbral_pocas_max=0,
    )
    total = len(todos_los_usuarios)
    print(f"📋 Total a entrenar: {total} usuarios")

    eventos_por_usuario = {str(uid): len(evs) for uid, evs in todos_los_usuarios}
    with open(os.path.join(SALIDA_DIR, "eventos_por_usuario.json"), "w", encoding="utf-8") as f:
        json.dump(eventos_por_usuario, f)

    import logic.entrenador as _entrenador_mod
    _entrenador_mod.DATA_DIR = SALIDA_DIR

    ok, errores, t0 = 0, [], time.time()
    for i, (uid, eventos) in enumerate(todos_los_usuarios):
        try:
            classifier, session_history, num_eng, num_non_eng = entrenar_usuario(uid, eventos, mapping)
            if GUARDAR_MODELOS:
                with open(os.path.join(SALIDA_DIR, f"model_u_{uid}.pkl"), "wb") as f:
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
        ruta = os.path.join(SALIDA_DIR, "errores_entrenamiento.json")
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(errores, f, indent=2, ensure_ascii=False)
        print(f"   Detalle de errores: {ruta}")

    print(f"\n📂 Perfiles guardados en: {SALIDA_DIR}")
    print("\nSiguiente paso:")
    print("   python calibrar_dominio.py data_calibracion_todos")
    print("   python calibracion_por_estrato.py data_calibracion_todos")


if __name__ == "__main__":
    import io
    import sys
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
