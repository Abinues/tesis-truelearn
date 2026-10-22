"""Acceso puntual a los eventos de un usuario del dataset PEEK."""
import os

from logic.peek_loader import cargar_dataset_peek

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def obtener_datos_usuario(usuario_objetivo_id=None):
    """
    Extrae los eventos crudos de un usuario del conjunto de entrenamiento de PEEK.

    Args:
        usuario_objetivo_id (int | None): Identificador del usuario buscado.
            Si es ``None`` se toma el primer usuario con más de 20 eventos.

    Returns:
        tuple: ``(uid, eventos, mapping)`` del usuario encontrado, donde
        ``eventos`` es la lista de pares ``(evento, etiqueta)`` y ``mapping``
        es el mapeo de temas del dataset. Si no se encuentra ningún usuario
        devuelve ``(None, None, None)``.
    """
    train, _, mapping = cargar_dataset_peek(DATA_DIR)

    for uid, events in train:
        if usuario_objetivo_id is None and len(events) > 20:
            return uid, events, mapping
        if uid == usuario_objetivo_id:
            return uid, events, mapping

    return None, None, None
