"""Carga de modelos TrueLearn entrenados y persistidos en disco."""
import os
import pickle


def cargar_modelo_entrenado(uid):
    """
    Carga el clasificador entrenado y persistido de un usuario.

    Busca el archivo ``models/model_u_{uid}.pkl`` en la raíz del proyecto,
    con el mismo nombre que usan ``entrenador.py`` y ``api_backend.py``.

    Args:
        uid (int): Identificador del usuario cuyo modelo se quiere cargar.

    Returns:
        INKClassifier | None: El clasificador deserializado, o ``None`` si
        no existe el archivo del modelo.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = os.path.join(base_dir, "models", f"model_u_{uid}.pkl")

    if os.path.exists(ruta):
        with open(ruta, "rb") as f:
            return pickle.load(f)
    return None
