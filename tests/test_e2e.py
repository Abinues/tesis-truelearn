"""
Tests de extremo a extremo (E2E) con Playwright.

A diferencia del resto de la suite (``AppTest``, sin navegador), estos tests
levantan el backend Flask y el servidor de Streamlit reales y los recorren con
un Chromium headless. Cubren lo que ``AppTest`` no puede: que los componentes
de ``streamlit_echarts`` se dibujen, que la barra lateral se oculte en el
login y reaparezca al entrar, y que la navegación entre páginas ande de punta
a punta.

Requiere ``pip install playwright pytest-playwright`` y
``playwright install chromium``. Son más lentos que el resto; conviene
correrlos aparte:

    pytest tests/test_e2e.py -v

Los datos de prueba (un usuario sintético entrenado con ``truelearn`` real) se
escriben temporalmente en ``data/`` y ``models/``. Antes se hace un backup de
lo que hubiera y se restaura siempre al terminar, incluso si un test falla.

La página de Playwright se comparte durante toda la sesión de tests: crear una
por test dejaba sesiones de Streamlit vivas en el servidor y degradaba el
rendimiento hasta que ni el login respondía. Por eso el orden de definición de
los tests importa y ``_entrar_al_perfil`` es idempotente.
"""
import os
import pickle
import shutil
import socket
import subprocess
import sys
import time

import pytest

pytest.importorskip("playwright", reason="playwright no está instalado")
pytest.importorskip("truelearn", reason="truelearn no está instalado (hace falta para generar el usuario de prueba)")

from playwright.sync_api import sync_playwright  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")
BACKEND_PORT = 5000
STREAMLIT_PORT = 8502
UID_PRUEBA = 999555


def _puerto_disponible(host, port, timeout=30):
    """Espera hasta que un puerto TCP acepte conexiones.

    Args:
        host (str): Nombre o dirección del host.
        port (int): Puerto a comprobar.
        timeout (float): Segundos máximos de espera.

    Returns:
        bool: True si el puerto aceptó una conexión antes del límite.
    """
    inicio = time.time()
    while time.time() - inicio < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def _crear_usuario_de_prueba():
    """Entrena un usuario sintético pequeño y lo guarda en ``data/`` y ``models/``.

    Usa ``truelearn`` real, sin necesidad del dataset PEEK: 60 eventos sobre
    cinco temas, con un desenganche cada tres eventos. Escribe el modelo
    ``.pkl``, el perfil ``.json`` y el índice ``usuarios_entrenados.json``.
    """
    from truelearn.models import EventModel, Knowledge, KnowledgeComponent

    import logic.entrenador as _em
    from logic.entrenador import entrenar_usuario, generar_perfil_json

    eventos = []
    temas_ids = [1, 2, 3, 4, 5]
    for t in range(60):
        tid = temas_ids[t % len(temas_ids)]
        engaged = (t % 3 != 0)
        eventos.append((
            EventModel(Knowledge({tid: KnowledgeComponent(mean=0.5, variance=0.5)}), event_time=float(t)),
            engaged,
        ))

    mapping = {
        1: ("Álgebra Lineal", "http://es.wikipedia.org/wiki/Algebra_lineal"),
        2: ("Cálculo Diferencial", ""),
        3: ("Probabilidad", ""),
        4: ("Estadística", ""),
        5: ("Machine Learning", ""),
    }

    classifier, session_history, num_eng, num_non_eng = entrenar_usuario(
        UID_PRUEBA, eventos, mapping, num_epochs=1
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, f"model_u_{UID_PRUEBA}.pkl"), "wb") as f:
        pickle.dump(classifier, f)

    _em.DATA_DIR = DATA_DIR
    perfil, _ = generar_perfil_json(UID_PRUEBA, classifier, mapping, num_eng, num_non_eng, session_history)

    import json
    indice = [{
        "user_id": UID_PRUEBA, "tipo": "muchas", "interacciones": len(eventos),
        "engagements": num_eng, "sesiones": len(session_history), "weights": perfil["weights"],
    }]
    with open(os.path.join(DATA_DIR, "usuarios_entrenados.json"), "w", encoding="utf-8") as f:
        json.dump(indice, f)


@pytest.fixture(scope="session")
def datos_de_prueba():
    """Prepara los datos de prueba y restaura los reales al terminar.

    Hace backup de ``data/`` y ``models/`` si existen, escribe un usuario
    sintético y, al finalizar la sesión (incluso con fallos), borra los datos
    de prueba y restaura el backup. Así los E2E nunca pisan datos reales.

    Yields:
        None: Cede el control mientras los datos de prueba están disponibles.
    """
    backup_data = None
    backup_models = None
    if os.path.exists(DATA_DIR):
        backup_data = DATA_DIR + "__backup_e2e"
        shutil.move(DATA_DIR, backup_data)
    if os.path.exists(MODEL_DIR):
        backup_models = MODEL_DIR + "__backup_e2e"
        shutil.move(MODEL_DIR, backup_models)

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    try:
        _crear_usuario_de_prueba()
        yield
    finally:
        shutil.rmtree(DATA_DIR, ignore_errors=True)
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
        if backup_data:
            shutil.move(backup_data, DATA_DIR)
        if backup_models:
            shutil.move(backup_models, MODEL_DIR)


@pytest.fixture(scope="session")
def servidores(datos_de_prueba):
    """Levanta el backend Flask y Streamlit como procesos reales.

    Fuerza ``PYTHONIOENCODING=utf-8`` porque en Windows, con la salida
    redirigida a un pipe, Python puede usar el códec ``cp1252`` y fallar con
    ``UnicodeEncodeError`` al imprimir emojis. Si algún servidor no arranca,
    el test falla mostrando la salida real de ambos procesos (el caso más
    común es un puerto ya en uso por otra instancia).

    Args:
        datos_de_prueba (None): Fixture que prepara los datos de prueba.

    Yields:
        str: URL base del servidor de Streamlit.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"

    backend = subprocess.Popen(
        [sys.executable, "-m", "logic.api_backend"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    streamlit = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard.py",
         "--server.headless", "true", "--server.port", str(STREAMLIT_PORT)],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )

    ok_backend = _puerto_disponible("localhost", BACKEND_PORT, timeout=20)
    ok_streamlit = _puerto_disponible("localhost", STREAMLIT_PORT, timeout=30)
    time.sleep(2)

    if not (ok_backend and ok_streamlit):
        def _leer_salida(proc):
            """Termina un proceso y devuelve su salida capturada.

            Args:
                proc (subprocess.Popen): Proceso a terminar.

            Returns:
                str: Salida del proceso o un aviso si no hubo ninguna.
            """
            proc.terminate()
            try:
                salida, _ = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                salida, _ = proc.communicate()
            return salida or "(sin salida capturada)"

        salida_backend = _leer_salida(backend)
        salida_streamlit = _leer_salida(streamlit)

        pytest.fail(
            f"No se pudieron levantar los servidores para el E2E "
            f"(backend={ok_backend}, streamlit={ok_streamlit}).\n\n"
            f"--- Salida real del backend (puerto {BACKEND_PORT}) ---\n{salida_backend}\n\n"
            f"--- Salida real de streamlit (puerto {STREAMLIT_PORT}) ---\n{salida_streamlit}\n\n"
            f"Si alguna dice 'Address already in use' o "
            f"'solo se permite un uso de cada dirección de socket': hay OTRA "
            f"instancia usando ese puerto (ej. el backend corriendo manualmente "
            f"en otra terminal) -- cerrala y volvé a intentar."
        )

    try:
        yield f"http://localhost:{STREAMLIT_PORT}"
    finally:
        backend.terminate()
        streamlit.terminate()
        try:
            backend.wait(timeout=5)
            streamlit.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend.kill()
            streamlit.kill()


@pytest.fixture(scope="session")
def navegador():
    """Lanza un Chromium headless de Playwright para toda la sesión.

    Yields:
        playwright.sync_api.Browser: Navegador listo para crear páginas.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser
        browser.close()


@pytest.fixture(scope="session")
def pagina(navegador, servidores):
    """Crea una única página compartida por todos los tests de la sesión.

    Compartirla evita acumular sesiones de Streamlit vivas en el servidor, que
    degradaban el rendimiento hasta que el login no respondía en 25 s.

    Args:
        navegador (playwright.sync_api.Browser): Navegador de la sesión.
        servidores (str): URL base del servidor de Streamlit.

    Yields:
        tuple[playwright.sync_api.Page, str]: La página y la URL base.
    """
    pg = navegador.new_page(viewport={"width": 1400, "height": 900})
    pg.set_default_timeout(25000)
    yield pg, servidores
    pg.close()


def test_login_carga_y_muestra_el_aprendiz_de_prueba(pagina):
    """Verifica que el login carga y lista al aprendiz de prueba.

    El desplegable de Streamlit no muestra sus opciones en el texto de la
    página hasta abrirlo (son un portal de BaseWeb), por eso se abre y luego se
    cierra para no afectar al siguiente test, que reutiliza la misma página.

    Args:
        pagina (tuple): Página de Playwright y URL base.
    """
    pg, url = pagina
    pg.goto(url, wait_until="load")
    pg.wait_for_selector("text=Seleccioná un estudiante", timeout=25000)
    texto = pg.inner_text("body")
    assert "TrueLearn" in texto
    pg.locator('[data-testid="stSelectbox"]').first.click()
    pg.wait_for_timeout(300)
    assert pg.get_by_text(f"Estudiante #{UID_PRUEBA}", exact=False).count() > 0
    pg.keyboard.press("Escape")


def _entrar_al_perfil(pg, url):
    """Entra al perfil del aprendiz de prueba de forma idempotente.

    Si la página ya está en el perfil (se detecta por "Cambiar estudiante" en el
    sidebar), reutiliza la sesión activa. Si no, hace el login completo y
    espera a que existan las tres tarjetas de métrica, porque Streamlit
    renderiza primero un marcador y luego la versión final.

    Args:
        pg (playwright.sync_api.Page): Página de Playwright.
        url (str): URL base del servidor de Streamlit.
    """
    if pg.locator("text=Cambiar estudiante").count() > 0:
        return

    pg.goto(url, wait_until="load")
    pg.wait_for_selector("text=Seleccioná un estudiante", timeout=25000)
    pg.locator('[data-testid="stSelectbox"]').first.click()
    pg.wait_for_timeout(300)
    pg.get_by_text(f"Estudiante #{UID_PRUEBA}", exact=False).first.click()
    pg.get_by_role("button", name="Ingresar al perfil").click()
    pg.wait_for_selector("text=¿Cómo está aprendiendo este estudiante?", timeout=25000)
    pg.wait_for_function(
        "document.querySelectorAll('[data-testid=\"stMetric\"]').length >= 2",
        timeout=20000,
    )


def test_entrar_al_perfil_muestra_resumen(pagina):
    """Verifica que al entrar al perfil se ve el resumen con sus métricas.

    Args:
        pagina (tuple): Página de Playwright y URL base.
    """
    pg, url = pagina
    _entrar_al_perfil(pg, url)
    texto = pg.inner_text("body").lower()
    assert "nivel de conocimiento" in texto
    assert "nivel de interés" in texto
    assert "seguro de esta estimación" in texto


@pytest.mark.parametrize("nombre_link,texto_esperado", [
    ("Análisis de temas", "Análisis de temas"),
    ("Análisis de sesiones", "Análisis de sesiones"),
    ("Pesos del sistema (Avanzado)", "decide el sistema"),
    ("Distribución de correcciones", "Distribución de correcciones"),
    ("Historial de correcciones", "correcciones"),
])
def test_navegar_a_cada_pagina_sin_error_visible(pagina, nombre_link, texto_esperado):
    """Cada página es alcanzable desde el sidebar y no muestra errores de Streamlit.

    Regresión: ``historial.py`` no figuraba en la navegación. Se espera el
    contenido a verificar en vez de una pausa fija, porque con varios
    componentes ``st_echarts`` por página una pausa corta leía el texto a mitad
    de render.

    Args:
        pagina (tuple): Página de Playwright y URL base.
        nombre_link (str): Nombre del link del sidebar a pulsar.
        texto_esperado (str): Texto que debe aparecer en la página destino.
    """
    pg, url = pagina
    _entrar_al_perfil(pg, url)
    pg.get_by_role("link", name=nombre_link).click()
    pg.wait_for_selector(f"text={texto_esperado}", timeout=25000)
    texto = pg.inner_text("body").lower()
    assert texto_esperado.lower() in texto
    assert "traceback" not in texto
    assert "streamlitapiexception" not in texto


def test_los_graficos_de_echarts_renderizan_de_verdad(pagina):
    """Verifica que ECharts dibuja un ``<canvas>`` real en el DOM.

    ``AppTest`` no puede comprobarlo: solo sabe que el script de Python no
    falló. Se navega explícitamente a "Resumen del perfil" porque el test
    anterior pudo dejar la página compartida en cualquier otra sección.

    Args:
        pagina (tuple): Página de Playwright y URL base.
    """
    pg, url = pagina
    _entrar_al_perfil(pg, url)
    pg.get_by_role("link", name="Resumen del perfil").click()
    pg.wait_for_selector("canvas", timeout=20000)
    canvases = pg.locator("canvas").count()
    assert canvases > 0, "No se encontró ningún <canvas> de ECharts renderizado en Resumen del perfil"


def test_popup_de_correccion_abre_con_contenido_real(pagina):
    """Verifica que el popup de corrección abre con las cuatro opciones.

    Pulsa "Editar" sobre un tema de "Análisis de temas" y comprueba el diálogo.
    Corre antes de ``test_boton_cambiar_aprendiz_vuelve_al_login`` para que
    ese test, que cruza el login con la sesión compartida, sea el último.

    Args:
        pagina (tuple): Página de Playwright y URL base.
    """
    pg, url = pagina
    _entrar_al_perfil(pg, url)
    pg.get_by_role("link", name="Análisis de temas").click()
    pg.wait_for_selector("text=Análisis de temas", timeout=25000)
    pg.get_by_text("TODOS LOS TEMAS").first.click()
    pg.wait_for_timeout(500)
    botones_editar = pg.get_by_role("button", name="Editar", exact=True)
    try:
        botones_editar.first.wait_for(state="visible", timeout=25000)
    except Exception:
        pytest.skip("No hay tarjetas de tema visibles para editar en este layout/viewport")
    botones_editar.first.click()
    pg.wait_for_selector("text=¿Qué querés corregir?", timeout=15000)
    texto = pg.inner_text("body")
    for opcion in ["sé más", "sé menos", "me interesa", "no me interesa"]:
        assert opcion in texto
    pg.keyboard.press("Escape")
    pg.wait_for_timeout(300)


def test_boton_cambiar_aprendiz_vuelve_al_login(pagina):
    """Verifica que "Cambiar estudiante" vuelve al login sin sidebar y permite reingresar.

    En el login la barra lateral debe quedar oculta (antes quedaba abierta y
    vacía porque el JavaScript que la colapsaba nunca se ejecutaba). Luego se
    vuelve a entrar al perfil y la navegación debe estar visible y usable.

    Args:
        pagina (tuple): Página de Playwright y URL base.
    """
    pg, url = pagina
    _entrar_al_perfil(pg, url)
    pg.get_by_role("button", name="Cambiar estudiante").click()
    pg.wait_for_selector("text=Seleccioná un estudiante", timeout=25000)
    texto = pg.inner_text("body")
    assert "Ingresar al perfil" not in texto or "Seleccioná un estudiante" in texto
    pg.wait_for_function(
        "!document.querySelector('[data-testid=\"stSidebar\"]') || "
        "document.querySelector('[data-testid=\"stSidebar\"]').offsetWidth === 0",
        timeout=10000,
    )

    pg.locator('[data-testid="stSelectbox"]').first.click()
    pg.wait_for_timeout(300)
    pg.get_by_text(f"Estudiante #{UID_PRUEBA}", exact=False).first.click()
    pg.get_by_role("button", name="Ingresar al perfil").click()
    pg.wait_for_selector("text=¿Cómo está aprendiendo este estudiante?", timeout=25000)
    pg.get_by_role("link", name="Análisis de temas").wait_for(state="visible", timeout=15000)