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
    .st-key-popup_apply button, .st-key-popup_cerrar button, .st-key-btn_ingresar button {
        background: #3b598d !important; border: 1px solid #3b598d !important; color: #FFFFFF !important;
    }
    .st-key-popup_apply button p, .st-key-popup_cerrar button p, .st-key-btn_ingresar button p { color: #FFFFFF !important; }
    .st-key-popup_apply button:hover, .st-key-popup_cerrar button:hover, .st-key-btn_ingresar button:hover {
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


def _delta(diferencia):
    """
    Arma los argumentos de ``st.metric`` para mostrar una diferencia.

    Si la diferencia es 0 (a un decimal) se muestra en gris y sin flecha; en
    otro caso, con el signo y la flecha habituales de Streamlit.

    Args:
        diferencia (float): Diferencia en puntos porcentuales.

    Returns:
        dict: Argumentos ``delta``, ``delta_color`` y ``delta_arrow`` para
        ``st.metric``.
    """
    if round(diferencia, 1) == 0:
        return {"delta": "0.0%", "delta_color": "gray", "delta_arrow": "off"}
    return {"delta": f"{diferencia:+.1f}%"}


def _pts(diferencia):
    """
    Texto con el cambio de confianza, o vacío si no cambió.

    Args:
        diferencia (float): Diferencia en puntos porcentuales.

    Returns:
        str: Por ejemplo ``" (+4.0 pts)"``, o ``""`` si la diferencia es 0.
    """
    return "" if round(diferencia, 1) == 0 else f" ({diferencia:+.1f} pts)"


def _cerrar_popup():
    """
    Limpia el estado del popup cuando el usuario lo cierra.

    Se usa tanto con la X de la esquina como con el botón «Cerrar» que
    aparece después de aplicar una corrección.

    Returns:
        None
    """
    st.session_state["popup_topic"]     = None
    st.session_state["popup_open"]      = False
    st.session_state["popup_resultado"] = None


def _firma_tema(kc_data):
    """
    Resume en una tupla el estado de un tema, para reconocer su popup.

    El resultado de una corrección solo se muestra mientras el popup siga
    abierto sobre el mismo tema y con los mismos valores de antes de corregir.
    Si el tema se vuelve a abrir más tarde (con valores ya corregidos), la
    firma cambia y el resultado viejo no reaparece.

    Args:
        kc_data (dict): Datos del tema.

    Returns:
        tuple: Identificador del tema y sus medias y varianzas.
    """
    return (kc_data["topic_id"], kc_data["mean"], kc_data["variance"],
            kc_data["interest_mean"], kc_data.get("interest_var", 0.5))


def _mostrar_resultado(res, kc_data):
    """
    Muestra lo que cambió después de aplicar una corrección.

    Presenta el mensaje del backend, las métricas nuevas del tema (con la
    diferencia respecto de antes de corregir y la confianza del sistema) y el
    botón «Cerrar», que cierra el popup y recarga la página para que refleje
    el cambio. Las diferencias en 0 se muestran en gris, sin flecha.

    Args:
        res (dict): Resultado guardado en ``popup_resultado``, con las claves
            ``"mensaje"`` y ``"nkc"`` (tema ya corregido o ``None``).
        kc_data (dict): Datos del tema tal como estaban antes de corregir.

    Returns:
        None
    """
    know_pop      = pct_conocimiento(kc_data["mean"])
    cert_know_pop = pct_certeza(kc_data["variance"])
    int_pop       = pct_interes(kc_data["interest_mean"])
    cert_int_pop  = pct_certeza(kc_data.get("interest_var", 0.5))

    st.success(res.get("mensaje", "Corrección aplicada."))
    nkc = res.get("nkc")
    if nkc:
        know_n      = pct_conocimiento(nkc["mean"])
        cert_know_n = pct_certeza(nkc["variance"])
        int_n       = pct_interes(nkc["interest_mean"])
        cert_int_n  = pct_certeza(nkc.get("interest_var", 0.5))
        n1, n2 = st.columns(2)
        with n1:
            with st.container(border=True):
                st.metric("Conocimiento ahora", f"{know_n:.1f}%",
                          **_delta(know_n - know_pop))
                st.caption(f"El sistema está **{cert_know_n:.1f}%** seguro de esta estimación"
                           + _pts(cert_know_n - cert_know_pop))
        with n2:
            with st.container(border=True):
                st.metric("Interés ahora", f"{int_n:.1f}%",
                          **_delta(int_n - int_pop))
                st.caption(f"El sistema está **{cert_int_n:.1f}%** seguro de esta estimación"
                           + _pts(cert_int_n - cert_int_pop))

    st.markdown("")
    if st.button("Cerrar", type="primary", use_container_width=True, key="popup_cerrar"):
        _cerrar_popup()
        st.rerun()


@st.dialog("Editar estimación del modelo", width="medium", on_dismiss=_cerrar_popup)
def popup_editar(kc_data, uid):
    """
    Muestra el popup de corrección explícita de un tema.

    Presenta las métricas actuales del tema (conocimiento e interés, cada una
    con la confianza del sistema en esa estimación), sus etiquetas y un
    selector con las cuatro correcciones posibles. Al aplicar una corrección,
    la envía al backend y, sin cerrarse, reemplaza el selector por lo que
    cambió: las métricas nuevas, con su confianza, y el botón «Cerrar». Ese
    botón, igual que la
    X de la esquina superior derecha, limpia ``popup_open`` y ``popup_topic``;
    el botón además recarga la página para que refleje el cambio. Si el
    backend devuelve un error, lo muestra y deja el selector disponible para
    reintentar.

    Args:
        kc_data (dict): Datos del tema, con las claves ``"topic_id"``,
            ``"topic_name"``, ``"mean"``, ``"variance"``, ``"interest_mean"``
            y, opcionalmente, ``"interest_var"`` (si falta se toma 0.5) y
            ``"topic_url"``.
        uid (int): Identificador del usuario al que se aplica la corrección.

    Returns:
        None
    """
    lbl, cls   = nivel_label(kc_data["mean"])
    ilbl, icls = interes_label(kc_data["interest_mean"])

    st.markdown(
        f"<div style='font-size:1.1rem;font-weight:700;color:#0f172a;"
        f"letter-spacing:-.02em;margin-bottom:.7rem'>{kc_data['topic_name']}</div>",
        unsafe_allow_html=True,
    )
    know_pop      = pct_conocimiento(kc_data["mean"])
    cert_know_pop = pct_certeza(kc_data["variance"])
    int_pop       = pct_interes(kc_data["interest_mean"])
    cert_int_pop  = pct_certeza(kc_data.get("interest_var", 0.5))
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.metric("Conocimiento", f"{know_pop:.1f}%",
                      **_delta(know_pop - PCT_INICIO_CONOCIMIENTO),
                      help=f"Valor técnico: {kc_data['mean']:.3f} · Inicio: {PCT_INICIO_CONOCIMIENTO:.1f}%")
            st.caption(f"El sistema está **{cert_know_pop:.1f}%** seguro de esta estimación")
    with c2:
        with st.container(border=True):
            st.metric("Interés", f"{int_pop:.1f}%",
                      **_delta(int_pop - PCT_INICIO_INTERES),
                      help=f"Valor técnico: {kc_data['interest_mean']:.3f} · Inicio: {PCT_INICIO_INTERES:.1f}%")
            st.caption(f"El sistema está **{cert_int_pop:.1f}%** seguro de esta estimación")
    st.markdown(
        f"<div style='margin:.4rem 0 .7rem'>"
        f"<span class='badge {cls}'>{lbl}</span> "
        f"<span class='badge {icls}'>Interés {ilbl}</span></div>",
        unsafe_allow_html=True,
    )
    if kc_data.get("topic_url"):
        st.markdown(
            f"<a href='{kc_data['topic_url']}' target='_blank' "
            f"style='font-size:.8rem;color:{TEAL};font-weight:600'>"
            f"Ver en Wikipedia ↗</a>", unsafe_allow_html=True)
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    # Si ya se aplicó una corrección sobre este tema, se muestra lo que cambió
    # (y el botón Cerrar) en lugar del selector.
    guardado = st.session_state.get("popup_resultado")
    if guardado and guardado.get("firma") == _firma_tema(kc_data):
        _mostrar_resultado(guardado, kc_data)
        return

    formulario = st.empty()
    aplicado   = None
    with formulario.container():
        tipo = st.radio(
            "¿Qué querés corregir?",
            ["sé más", "sé menos", "me interesa", "no me interesa"],
            horizontal=True, key="popup_tipo",
        )
        descs = {
            "sé más":         "↑ Aumenta el conocimiento estimado y la confianza del sistema en él.",
            "sé menos":       "↓ Reduce el conocimiento estimado y aumenta la confianza del sistema en él.",
            "me interesa":    "↑ Aumenta el interés estimado y la confianza del sistema en él.",
            "no me interesa": "↓ Reduce el interés estimado y aumenta la confianza del sistema en él.",
        }
        st.caption(descs[tipo])
        st.markdown("")

        if st.button("Aplicar", type="primary",
                     use_container_width=True, key="popup_apply"):
            with st.spinner("Actualizando modelo..."):
                res = post_feedback(uid, kc_data["topic_id"],
                                    kc_data["topic_name"], tipo)
            if "error" in res:
                st.error(f"Error: {res['error']}")
            else:
                nuevos = res.get("perfil", {}).get("knowledge_components", [])
                nkc = next((k for k in nuevos
                            if k["topic_id"] == kc_data["topic_id"]), None)
                aplicado = {
                    "firma":        _firma_tema(kc_data),
                    "mensaje":      res.get("mensaje", "Corrección aplicada."),
                    "nkc":          nkc,
                }

    if aplicado:
        # El selector se reemplaza por el resultado, sin cerrar el popup.
        st.session_state["popup_resultado"] = aplicado
        formulario.empty()
        _mostrar_resultado(aplicado, kc_data)
