"""
Conecta ``logic/administracion.py`` con ``logic/api_backend.py``.

Hace dos agregados, y nada más:

    1. Antes del bloque final ``if __name__ == "__main__":`` (o de la línea
       «Entry point», si la tiene), registra los endpoints de administración
       (restablecer todo y feedbacks ficticios).
    2. En el hilo de entrenamiento, justo después de ``entrenar_modelo_usuario()``,
       guarda la copia limpia de cada aprendiz (lo que usa «Restablecer todo»).

Hace una copia ``api_backend.py.bak`` antes de escribir. Si no encuentra el lugar
donde agregar algo, no modifica el archivo y lo avisa. Se puede correr más de una
vez: lo que ya está agregado no se repite.

Uso, desde la carpeta del proyecto (con ``logic/administracion.py`` ya copiado)::

    python ajustar_backend.py
    python ajustar_backend.py ruta/a/api_backend.py
"""
import ast
import re
import shutil
import sys

BLOQUE_REGISTRO = (
    "# ── Administración del modelo (restablecer todo, feedbacks ficticios) ──────────\n"
    "from logic import administracion  # noqa: E402\n"
    "\n"
    "administracion.registrar(app, sys.modules[__name__])\n"
    "\n"
    "\n"
)


def ajustar(texto):
    """Agrega los dos fragmentos al texto de ``api_backend.py``.

    Args:
        texto (str): Contenido del archivo con saltos de línea ``\\n``.

    Returns:
        tuple[str, list[str]]: El texto nuevo y el informe de lo realizado.

    Raises:
        ValueError: Si no se encuentra dónde agregar alguno de los fragmentos.
    """
    informe = []

    if "administracion.registrar(" in texto:
        informe.append("registro de los endpoints (ya estaba)")
    else:
        m = (re.search(r"^# ── Entry point[^\n]*\n", texto, re.MULTILINE)
             or re.search(r"^if __name__ == [\"']__main__[\"']:", texto, re.MULTILINE))
        if not m:
            raise ValueError("No encontré 'if __name__ == \"__main__\":' donde registrar los endpoints.")
        texto = texto[:m.start()] + BLOQUE_REGISTRO + texto[m.start():]
        informe.append("endpoints de administración registrados")

    if "administracion.guardar_base(" in texto:
        informe.append("copia limpia tras entrenar (ya estaba)")
    else:
        m = re.search(r"^(?P<sangria>[ \t]+)entrenar_modelo_usuario\(\)[ \t]*\n", texto, re.MULTILINE)
        if not m:
            raise ValueError("No encontré 'entrenar_modelo_usuario()' dentro del hilo de entrenamiento.")
        sangria_with = m.group("sangria")
        if len(sangria_with) < 4:
            raise ValueError("La llamada a entrenar_modelo_usuario() tiene una sangría inesperada.")
        sangria = sangria_with[:-4]
        linea = f"{sangria}administracion.guardar_base(sys.modules[__name__])\n"
        texto = texto[:m.end()] + "\n" + linea + texto[m.end():]
        informe.append("copia limpia de cada aprendiz al terminar el entrenamiento")

    return texto, informe


def procesar(ruta):
    """Lee, ajusta y escribe un ``api_backend.py``, con copia de seguridad.

    Args:
        ruta (str): Ruta del archivo.
    """
    with open(ruta, encoding="utf-8", newline="") as f:
        original = f.read()
    crlf = "\r\n" in original
    try:
        nuevo, informe = ajustar(original.replace("\r\n", "\n"))
        ast.parse(nuevo)
    except (ValueError, SyntaxError) as error:
        print(f"{ruta}: NO se modificó. {error}")
        print("Mandame este api_backend.py y lo ajusto sobre tu archivo exacto.")
        return
    if crlf:
        nuevo = nuevo.replace("\n", "\r\n")
    if nuevo == original:
        print(f"{ruta}: ya estaba conectado, sin cambios.")
        return
    shutil.copyfile(ruta, ruta + ".bak")
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        f.write(nuevo)
    print(f"{ruta}: listo (copia en {ruta}.bak)")
    for linea in informe:
        print("  -", linea)


if __name__ == "__main__":
    for archivo in sys.argv[1:] or ["logic/api_backend.py"]:
        try:
            procesar(archivo)
        except FileNotFoundError:
            print(f"{archivo}: no existe, revisá la ruta.")
