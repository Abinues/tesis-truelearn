"""versiones.py: imprime las versiones usadas en el proyecto, listas para pegar en la tesis.

Uso (raíz del proyecto, con el entorno virtual activado):
    python versiones.py
Pegame la salida. Si querés la lista completa de lo instalado: python versiones.py --todo
"""
import platform
import sys
from importlib import metadata

PAQUETES = [
    "truelearn", "streamlit", "streamlit-echarts", "flask", "pandas", "numpy",
    "requests", "playwright", "pytest", "pytest-playwright", "openpyxl",
    "matplotlib", "trueskill", "mpmath", "scipy", "scikit-learn", "ruff", "mypy",
]

print(f"Python {platform.python_version()} ({platform.python_implementation()})")
print(f"Sistema: {platform.system()} {platform.release()} ({platform.version()})")
print(f"Arquitectura: {platform.machine()}\n")
print("| Paquete | Versión |")
print("|---|---|")
for nombre in PAQUETES:
    try:
        print(f"| {nombre} | {metadata.version(nombre)} |")
    except metadata.PackageNotFoundError:
        print(f"| {nombre} | no instalado |")

# Opcional: python versiones.py --todo  imprime además todo lo instalado
if "--todo" in sys.argv:
    todos = sorted({(d.metadata["Name"], d.version) for d in metadata.distributions() if d.metadata["Name"]})
    print("\nTodos los paquetes instalados:")
    print(", ".join(f"{n}=={v}" for n, v in todos))