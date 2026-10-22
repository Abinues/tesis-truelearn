"""Script exploratorio que vuelca a JSON el historial de un usuario de PEEK."""
import json
import os

from logic.modelo_utils import resolver_metadata_tema
from logic.peek_loader import cargar_dataset_peek

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def extraer_muestra():
    """
    Extrae el historial completo de un usuario de PEEK y lo guarda como JSON.

    Toma el primer usuario del conjunto de entrenamiento con más de 15
    eventos. Por cada evento registra la marca de tiempo, si el usuario se
    enganchó y los temas del video con su media y varianza. El resultado se
    escribe en ``data/datos_usuario_prueba.json`` para inspección manual.

    Returns:
        None: No devuelve valor. Si no hay un usuario con historial
        suficiente, informa por consola y termina sin generar el archivo.
    """
    print(f"📥 Cargando PEEK Dataset (destino: {DATA_DIR})...")
    train, _test, mapping = cargar_dataset_peek(DATA_DIR)

    usuario_id = None
    lista_eventos = None

    for uid, evs in train:
        if len(evs) > 15:
            usuario_id = uid
            lista_eventos = evs
            break

    if not usuario_id:
        print("❌ No se encontró un usuario con suficiente historial.")
        return

    print(f"✅ Usuario encontrado: {usuario_id} con {len(lista_eventos)} interacciones.")

    datos_finales = []
    for event, label in lista_eventos:
        temas = []
        for topic_id, kc in event.knowledge.topic_kc_pairs():
            nombre, _url = resolver_metadata_tema(topic_id, mapping)
            temas.append({
                "topic_id": topic_id,
                "nombre": nombre,
                "mean": kc.mean,
                "variance": kc.variance
            })

        datos_finales.append({
            "timestamp": event.event_time,
            "se_engancho": bool(label),
            "temas_video": temas
        })

    ruta_json = os.path.join(DATA_DIR, "datos_usuario_prueba.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(datos_finales, f, indent=4, ensure_ascii=False)

    print(f"🚀 ¡Éxito! Archivo generado en: {os.path.abspath(ruta_json)}")


if __name__ == "__main__":
    import io
    import sys
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    extraer_muestra()
