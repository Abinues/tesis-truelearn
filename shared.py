"""
Constantes, CSS, funciones auxiliares y popup compartidos por las páginas.

Reúne lo que usan varias páginas del dashboard:

    - Constantes: dirección del backend y colores.
    - ``inject_css``: estilos globales de la interfaz.
    - Funciones de comunicación con el backend Flask: ``get_usuarios``,
      ``get_perfil``, ``post_feedback`` y ``get_prediccion``.
    - ``fmt_ts``: formato de fechas.
    - ``popup_editar``: popup de corrección explícita del modelo.

Las etiquetas de clasificación y las fórmulas de porcentaje viven en
``logic.dominio`` y se importan aquí para que las páginas las tomen de este
módulo.
"""
from datetime import datetime

import requests
import streamlit as st

from logic.dominio import (
    PCT_INICIO_CONOCIMIENTO,
    PCT_INICIO_INTERES,
    interes_label,
    nivel_label,
    pct_certeza,
    pct_conocimiento,
    pct_interes,
    var_label,
)

BACKEND  = "http://localhost:5000"
TEAL     = "#3b598d"
GRID_COL = "#808080"


def inject_css():
    """
    Inyecta los estilos globales de la interfaz.

    Define el fondo blanco, la barra lateral azul con texto blanco, el estilo
    del botón de cambiar aprendiz y de los datos del aprendiz, las etiquetas
    (badges) de conocimiento, confianza e interés, el scroll del mapa de
    conocimiento y la ocultación del menú y del pie nativos.

    Returns:
        None
    """
    import streamlit as st
    st.markdown("""
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: #ffffff !important;
    }
    .block-container {
        background: #ffffff !important;
        padding-top: 1.5rem !important;
    }

    [data-testid="stSidebar"] {
        background: #3b598d !important;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.12) !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: background .15s !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.22) !important;
    }
    [data-testid="stSidebar"] .btn-cambiar .stButton > button {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        font-size: .83rem !important;
    }
    [data-testid="stSidebar"] .btn-cambiar .stButton > button:hover {
        background: rgba(255,90,90,0.25) !important;
    }
    .sidebar-user {
        background: rgba(255,255,255,0.12);
        border-radius: 10px;
        padding: .6rem .85rem;
        margin-bottom: .6rem;
    }
    .sidebar-uid {
        font-size: .95rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: .15rem;
    }

    .badge {
        font-size: .68rem; padding: .18rem .5rem;
        border-radius: 999px; font-weight: 600;
        display: inline-block; white-space: nowrap;
    }
    .b-hi      { background: #8d3b82; color: #FFFFFF; }
    .b-mid     { background: #3b8d46; color: #FFFFFF; }
    .b-lo      { background: #8d6f3b; color: #FFFFFF; }
    .b-vlo     { background: #8d3b3b; color: #FFFFFF; }
    .b-var-hi  { background: #3b8d8d; color: #FFFFFF; }
    .b-var-mid { background: #8d8d3b; color: #FFFFFF; }
    .b-var-lo  { background: #8d3b59; color: #FFFFFF; }
    .b-int-hi  { background: #5f8fe2; color: #FFFFFF; }
    .b-int-md  { background: #3b598d; color: #FFFFFF; }
    [data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child {
        background-color: #3b598d !important;
    }
    .st-key-popup_apply button, .st-key-btn_ingresar button {
        background: #3b598d !important; border: 1px solid #3b598d !important; color: #FFFFFF !important;
    }
    .st-key-popup_apply button p, .st-key-btn_ingresar button p { color: #FFFFFF !important; }
    .st-key-popup_apply button:hover, .st-key-btn_ingresar button:hover {
        background: #2f4771 !important; border-color: #2f4771 !important;
    }
    [class*="st-key-edit_"] button, .st-key-corr_edit_btn button {
        background: #3b598d !important; border: 1px solid #3b598d !important; color: #FFFFFF !important;
    }
    [class*="st-key-edit_"] button p, .st-key-corr_edit_btn button p { color: #FFFFFF !important; }
    [class*="st-key-edit_"] button:hover, .st-key-corr_edit_btn button:hover {
        background: #2f4771 !important; border-color: #2f4771 !important;
    }
    .b-int-lo  { background: #172338; color: #FFFFFF; }

    #mapa-cards-wrap {
        max-height: calc(100vh - 260px);
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: #cbd5e1 transparent;
        padding-right: 4px;
    }

    #MainMenu { visibility: hidden !important; }
    footer     { visibility: hidden !important; }
    </style>
    """, unsafe_allow_html=True)


def get_usuarios():
    """
    Pide al backend la lista de usuarios entrenados.

    Si no se puede conectar al backend o ocurre otro error, lo muestra en la
    página y detiene su ejecución.

    Returns:
        list[dict]: Resumen de cada usuario entrenado, o una lista vacía si
        el backend respondió con error.
    """
    try:
        r = requests.get(f"{BACKEND}/usuarios", timeout=5)
        return r.json() if r.ok else []
    except requests.exceptions.ConnectionError:
        st.error("No se puede conectar al backend. "
                 "Asegurate de correr: `python -m logic.api_backend`")
        st.stop()
    except Exception as e:
        st.error(f"Error inesperado: {e}")
        st.stop()


def get_perfil(uid):
    """
    Pide al backend el perfil completo de un usuario.

    Si el perfil no existe, el backend no responde o ocurre otro error, lo
    muestra en la página y detiene su ejecución.

    Args:
        uid (int): Identificador del usuario.

    Returns:
        dict | None: El perfil del usuario, o ``None`` si el backend
        respondió con un error distinto de 404.
    """
    try:
        r = requests.get(f"{BACKEND}/perfil/{uid}", timeout=5)
        if r.status_code == 404:
            st.error(f"No existe perfil para el usuario {uid}. Corré el entrenador primero.")
            st.stop()
        return r.json() if r.ok else None
    except requests.exceptions.ConnectionError:
        st.error("Backend desconectado al cargar el perfil.")
        st.stop()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()


def post_feedback(uid, topic_id, topic_name, tipo):
    """
    Envía al backend una corrección explícita sobre un tema.

    El ``topic_id`` se convierte a ``int`` nativo, porque los tipos de numpy
    (por ejemplo los que salen de un ``DataFrame``) no se pueden serializar a
    JSON.

    Args:
        uid (int): Identificador del usuario.
        topic_id (int | None): Identificador del tema a corregir.
        topic_name (str): Nombre del tema.
        tipo (str): Tipo de corrección: ``"sé más"``, ``"sé menos"``,
            ``"me interesa"`` o ``"no me interesa"``.

    Returns:
        dict: La respuesta JSON del backend (con ``ok``, ``mensaje`` y el
        ``perfil`` actualizado), o un diccionario con la clave ``"error"`` si
        la petición falló.
    """
    body = {
        "topic_id":   int(topic_id) if topic_id is not None else None,
        "topic_name": str(topic_name),
        "tipo":       tipo,
    }
    try:
        r = requests.post(f"{BACKEND}/feedback/{uid}", json=body, timeout=8)
        return r.json() if r.ok else {"error": f"HTTP {r.status_code}: {r.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Backend desconectado."}
    except Exception as e:
        return {"error": str(e)}


def get_prediccion(uid, topic_id):
    """
    Pide al backend la probabilidad de enganche del modelo para un tema.

    Args:
        uid (int): Identificador del usuario.
        topic_id (int): Identificador del tema.

    Returns:
        float | None: Probabilidad de enganche entre 0 y 1, o ``None`` si la
        consulta falló.
    """
    try:
        r = requests.get(f"{BACKEND}/prediccion/{uid}/{topic_id}", timeout=5)
        return r.json().get("probabilidad_enganche", None) if r.ok else None
    except Exception:
        return None


def fmt_ts(ts):
    """
    Da formato de fecha y hora a una marca de tiempo.

    Args:
        ts (float | str): Marca de tiempo en segundos desde la época Unix.

    Returns:
        str: Fecha con el formato ``"dd/mm/aaaa hh:mm"`` en hora local, o
        ``"—"`` si el valor no se puede convertir.
    """
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%d/%m/%Y %H:%M")
    except Exception:
        return "—"


def _cerrar_popup():
    """
    Limpia el estado del popup cuando el usuario lo cierra con la X.

    Returns:
        None
    """
    st.session_state["popup_topic"] = None
    st.session_state["popup_open"]  = False


@st.dialog("Editar estimación del modelo", width="large", on_dismiss=_cerrar_popup)
def popup_editar(kc_data, uid):
    """
    Muestra el popup de corrección explícita de un tema.

    Presenta las métricas actuales del tema (conocimiento, confianza e
    interés), sus etiquetas y un selector con las cuatro correcciones
    posibles. Al aplicar una corrección, la envía al backend, muestra las
    métricas nuevas y el cambio en la probabilidad de enganche (consultada
    antes y después), cierra el popup y recarga la página para que refleje el
    cambio. Se cierra con la X de la esquina superior derecha, sin aplicar nada, y al cerrarse limpia ``popup_open`` y ``popup_topic``. Si el
    backend devuelve un error, lo muestra y deja el popup abierto.

    Args:
        kc_data (dict): Datos del tema, con las claves ``"topic_id"``,
            ``"topic_name"``, ``"mean"``, ``"variance"``, ``"interest_mean"``
            y, opcionalmente, ``"topic_url"``.
        uid (int): Identificador del usuario al que se aplica la corrección.

    Returns:
        None
    """
    lbl, cls   = nivel_label(kc_data["mean"])
    vlbl, vcls = var_label(kc_data["variance"])
    ilbl, icls = interes_label(kc_data["interest_mean"])

    st.markdown(
        f"<div style='font-size:1.1rem;font-weight:700;color:#0f172a;"
        f"letter-spacing:-.02em;margin-bottom:.7rem'>{kc_data['topic_name']}</div>",
        unsafe_allow_html=True,
    )
    know_pop = pct_conocimiento(kc_data["mean"])
    cert_pop = pct_certeza(kc_data["variance"])
    int_pop  = pct_interes(kc_data["interest_mean"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Conocimiento", f"{know_pop:.1f}%",
              delta=f"{know_pop - PCT_INICIO_CONOCIMIENTO:+.1f}%",
              help=f"Valor técnico: {kc_data['mean']:.3f} · Inicio: {PCT_INICIO_CONOCIMIENTO:.1f}%")
    c2.metric("Interés", f"{int_pop:.1f}%",
              delta=f"{int_pop - PCT_INICIO_INTERES:+.1f}%",
              help=f"Valor técnico: {kc_data['interest_mean']:.3f} · Inicio: {PCT_INICIO_INTERES:.1f}%")
    c3.metric("Confianza del sistema", f"{cert_pop:.1f}%",
              delta=f"{cert_pop - 50.0:+.1f}%",
              help=f"1 - varianza={kc_data['variance']:.3f} · Inicio: 50%")
    st.markdown(
        f"<div style='margin:.4rem 0 .7rem'>"
        f"<span class='badge {cls}'>{lbl}</span> "
        f"<span class='badge {icls}'>Interés {ilbl}</span> "
        f"<span class='badge {vcls}'>Certeza: {vlbl}</span></div>",
        unsafe_allow_html=True,
    )
    st.progress(min(max(know_pop / 100, 0.0), 1.0))
    if kc_data.get("topic_url"):
        st.markdown(
            f"<a href='{kc_data['topic_url']}' target='_blank' "
            f"style='font-size:.8rem;color:{TEAL};font-weight:600'>"
            f"Ver en Wikipedia ↗</a>", unsafe_allow_html=True)
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    tipo = st.radio(
        "¿Qué querés corregir?",
        ["sé más", "sé menos", "me interesa", "no me interesa"],
        horizontal=True, key="popup_tipo",
    )
    descs = {
        "sé más":         "↑ Aumenta el conocimiento estimado y reduce la incertidumbre.",
        "sé menos":       "↓ Reduce el conocimiento estimado y reduce la incertidumbre.",
        "me interesa":    "Aumenta el interés estimado en este tema.",
        "no me interesa": "Reduce el interés. La incertidumbre de interés sube.",
    }
    st.caption(descs[tipo])
    st.markdown("")

    ca = st.container()
    with ca:
        if st.button("Aplicar", type="primary",
                     use_container_width=True, key="popup_apply"):
            prob_antes = get_prediccion(uid, kc_data["topic_id"])

            with st.spinner("Actualizando modelo..."):
                res = post_feedback(uid, kc_data["topic_id"],
                                    kc_data["topic_name"], tipo)
            if "error" in res:
                st.error(f"Error: {res['error']}")
            else:
                st.success(res.get("mensaje", "Corrección aplicada."))
                nuevos = res.get("perfil", {}).get("knowledge_components", [])
                nkc = next((k for k in nuevos
                            if k["topic_id"] == kc_data["topic_id"]), None)
                if nkc:
                    know_n = pct_conocimiento(nkc["mean"])
                    cert_n = pct_certeza(nkc["variance"])
                    int_n  = pct_interes(nkc["interest_mean"])
                    n1, n2, n3 = st.columns(3)
                    n1.metric("Conocimiento",  f"{know_n:.1f}%",
                              delta=f"{know_n - know_pop:+.1f}%")
                    n2.metric("Interés",       f"{int_n:.1f}%",
                              delta=f"{int_n - int_pop:+.1f}%")
                    n3.metric("Confianza",     f"{cert_n:.1f}%",
                              delta=f"{cert_n - cert_pop:+.1f}%")

                prob_despues = get_prediccion(uid, kc_data["topic_id"])
                if prob_antes is not None and prob_despues is not None:
                    delta_pp = round((prob_despues - prob_antes) * 100, 1)
                    icono    = "↑" if delta_pp > 0 else ("↓" if delta_pp < 0 else "→")
                    color    = "#5f8fe2" if delta_pp > 0 else ("#172338" if delta_pp < 0 else "#64748b")
                    st.markdown(
                        f"<div style='background:#f8fafc;border:1px solid #e2e8f0;"
                        f"border-radius:10px;padding:.7rem 1rem;margin-top:.6rem;"
                        f"font-size:.82rem;color:#374151'>"
                        f"<b>Impacto en el modelo:</b> la probabilidad de enganche "
                        f"pasó de <b>{prob_antes*100:.1f}%</b> a "
                        f"<b style='color:{color}'>{prob_despues*100:.1f}%</b> "
                        f"<span style='color:{color};font-weight:700'>"
                        f"({icono} {abs(delta_pp)}pp)</span></div>",
                        unsafe_allow_html=True)

                st.session_state["popup_topic"] = None
                st.session_state["popup_open"]  = False
                st.rerun()
