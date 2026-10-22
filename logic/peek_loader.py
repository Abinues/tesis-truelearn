"""Punto único de carga del dataset PEEK, seguro frente a hilos y a errores."""
from __future__ import annotations

import os
import threading

_chdir_lock = threading.Lock()


def cargar_dataset_peek(data_dir: str):
    """
    Carga el dataset PEEK descargándolo en ``data_dir``.

    La librería ``truelearn`` descarga el dataset en el directorio de trabajo,
    por lo que se cambia temporalmente el directorio actual del proceso. El
    cambio se protege con un candado, de modo que solo una carga puede estar
    en curso a la vez, y con ``try/finally``, de modo que el directorio
    original se restaura incluso si la carga falla.

    Args:
        data_dir (str): Carpeta donde se descarga y se lee el dataset. Se
            crea si no existe.

    Returns:
        tuple: Resultado de ``truelearn.datasets.load_peek_dataset()``, es
        decir ``(train, test, mapping)``.

    Raises:
        Exception: Cualquier error de ``load_peek_dataset()`` se propaga
            después de restaurar el directorio de trabajo.
    """
    from truelearn.datasets import load_peek_dataset

    os.makedirs(data_dir, exist_ok=True)

    with _chdir_lock:
        old_cwd = os.getcwd()
        try:
            os.chdir(data_dir)
            return load_peek_dataset()
        finally:
            os.chdir(old_cwd)
