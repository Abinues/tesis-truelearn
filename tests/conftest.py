"""
Configuración común de pytest para la suite de tests.

Streamlit agrega al ``sys.path`` el directorio de ``dashboard.py`` (la raíz del
proyecto) al ejecutar ``streamlit run``, por eso ``from shared import ...`` y
``from logic.x import ...`` funcionan desde ``pages/*.py``. pytest no lo hace,
así que aquí se replica para que los tests importen ``logic.*`` igual que la
aplicación real.

Además define un hook que guarda una captura de pantalla en
``tests/_e2e_failures/`` cuando falla un test E2E que recibe el fixture
``pagina`` (tupla ``(page, url)``, ver ``test_e2e.py``).
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Guarda una captura de pantalla de la página cuando falla un test E2E.

    Solo actúa si el test falló en la fase ``call`` y recibió el fixture
    ``pagina``. Los tests con AppTest no usan navegador, por lo que no se
    ven afectados. Si la captura falla, solo imprime el motivo.

    Args:
        item (pytest.Item): Test que se está ejecutando.
        call (pytest.CallInfo): Información de la fase de ejecución actual.

    Yields:
        None: Cede el control para que pytest construya el reporte.
    """
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" or not report.failed:
        return
    pagina_fixture = item.funcargs.get("pagina")
    if not pagina_fixture:
        return
    try:
        pg, _url = pagina_fixture
        carpeta = os.path.join(os.path.dirname(__file__), "_e2e_failures")
        os.makedirs(carpeta, exist_ok=True)
        nombre = item.name.replace("/", "_").replace("\\", "_").replace(":", "_")
        ruta = os.path.join(carpeta, f"{nombre}.png")
        pg.screenshot(path=ruta, full_page=True)
        print(f"\n📸 Captura de pantalla del fallo guardada en: {ruta}")
    except Exception as e:
        print(f"\n(no se pudo capturar pantalla del fallo: {e})")
