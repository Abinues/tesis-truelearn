"""
Punto de entrada del dashboard de TrueLearn.

Ejecución, con el backend (``python -m logic.api_backend``) corriendo en el
puerto 5000:

    streamlit run dashboard.py

Configura la página, inicializa el estado de sesión y decide qué mostrar:

    - Sin aprendiz seleccionado: solo la página de inicio (``pages/login.py``),
      sin barra de navegación y con la barra lateral oculta mediante CSS (se
      vuelve a mostrar al elegir un aprendiz).
    - Con aprendiz seleccionado: carga su perfil, lo deja disponible para las
      páginas en ``perfil_actual``, muestra un resumen del aprendiz en la
      barra lateral con el botón para cambiarlo, y la navegación entre las
      seis páginas del perfil.

Estado de sesión:
    uid_seleccionado: Aprendiz elegido, o ``None`` si no hay ninguno.
    perfil_actual: Perfil del aprendiz elegido, compartido con las páginas.
    popup_topic, popup_open: Tema abierto en el popup de corrección y si el
        popup está abierto. Se reinician al cambiar de página.
    last_busq, last_orden: Última búsqueda y orden del mapa de temas.
    last_page: Última página visitada.
"""
import streamlit as st

from shared import get_perfil, inject_css

st.set_page_config(
    page_title="TrueLearn · Perfil del Aprendiz",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

for key, default in {
    "uid_seleccionado":  None,
    "popup_topic":       None,
    "popup_open":        False,
    "last_busq":         "",
    "last_orden":        0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state["uid_seleccionado"] is None:
    st.markdown(
        """<style>
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"] { display: none !important; }
        </style>""",
        unsafe_allow_html=True,
    )
    pg = st.navigation(
        [st.Page("pages/login.py", title="Inicio", icon=":material/home:", default=True)],
        position="hidden",
    )
    pg.run()
    st.stop()

uid    = st.session_state["uid_seleccionado"]
perfil = get_perfil(uid)

if not perfil:
    st.error(f"No se encontró perfil para el aprendiz {uid}.")
    if st.button("← Volver"):
        st.session_state["uid_seleccionado"] = None
        st.rerun()
    st.stop()

st.session_state["perfil_actual"] = perfil

kcs     = perfil.get("knowledge_components", [])
eng     = perfil.get("number_of_engagements", 0)
non_eng = perfil.get("number_of_non_engagements", 0)
total   = eng + non_eng
tasa    = round(eng / total * 100, 1) if total > 0 else 0

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-user">
        <div class="sidebar-uid">Aprendiz #{uid}</div>
        <div style="font-size:.73rem;opacity:.85">
            📊 {total} eventos · ✅ {tasa}% enganche · 🗂 {len(kcs)} temas
        </div>
    </div>
    <div style="height:.4rem"></div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='btn-cambiar'>", unsafe_allow_html=True)
    if st.button("← Cambiar aprendiz", use_container_width=True, key="btn_cambiar"):
        st.session_state["uid_seleccionado"]  = None
        st.session_state["perfil_actual"]     = None
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='margin-top:2rem;font-size:.65rem;opacity:.5;text-align:center'>"
        "🎓 TrueLearn · UNA · 2026</div>",
        unsafe_allow_html=True)

pg = st.navigation([
    st.Page("pages/resumen.py",    title="Resumen del perfil",         icon=":material/dashboard:"),
    st.Page("pages/temas.py",      title="Mapa de temas",              icon=":material/bubble_chart:"),
    st.Page("pages/evolucion.py",  title="Evolución temporal",         icon=":material/show_chart:"),
    st.Page("pages/pesos_ink.py",  title="Pesos del sistema (Avanzado)",       icon=":material/tune:"),
    st.Page("pages/prediccion.py", title="Predicción y correcciones",  icon=":material/model_training:"),
    st.Page("pages/historial.py",  title="Historial de correcciones",  icon=":material/history:"),
])
if "last_page" not in st.session_state:
    st.session_state["last_page"] = None
current_page = pg.url_path if hasattr(pg, "url_path") else str(pg)
if current_page != st.session_state.get("last_page"):
    st.session_state["popup_open"]  = False
    st.session_state["popup_topic"] = None
    st.session_state["last_page"]   = current_page

pg.run()
