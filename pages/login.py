"""
Página de selección del estudiante (inicio).

Permite elegir, de un desplegable, uno de los estudiantes entrenados y entrar
a su perfil. También incluye:

    - Un panel de presentación del proyecto.
    - Un aviso sobre el sesgo temático del dataset PEEK.
    - Una sección «Administración del modelo» para entrenar desde PEEK (con una
      barra de progreso, sin el log de la terminal), agregar feedbacks
      ficticios y restablecer todo.

Si el backend no responde, muestra el comando para levantarlo y detiene la
página.

Estado de sesión que usa:
    uid_seleccionado: Se escribe al presionar «Ingresar al perfil».
"""
import re

import requests
import streamlit as st

from ajuste_letras import aplicar_letras_grandes
from shared import BACKEND, get_usuarios, inject_css

inject_css()
aplicar_letras_grandes()


def _pedir(metodo, ruta, **kwargs):
    """
    Llama a un endpoint del backend.

    Args:
        metodo (str): ``"GET"`` o ``"POST"``.
        ruta (str): Ruta del endpoint, por ejemplo ``"/administracion/estado"``.
        **kwargs: Argumentos extra de ``requests.request`` (``json``, ``timeout``).

    Returns:
        tuple[dict | None, str | None]: La respuesta JSON y ``None`` si salió bien,
        o ``None`` y un mensaje de error corto si falló.
    """
    kwargs.setdefault("timeout", 120)
    try:
        r = requests.request(metodo, f"{BACKEND}{ruta}", **kwargs)
        try:
            datos = r.json()
        except ValueError:
            datos = {}
        if not r.ok:
            return None, datos.get("error") or datos.get("mensaje") or f"Error {r.status_code}"
        return datos, None
    except requests.exceptions.RequestException:
        return None, "No se pudo conectar con el backend."


def _resumir_log(lineas):
    """
    Resume el log del entrenamiento en unas pocas cifras, sin mostrarlo.

    Args:
        lineas (list[str]): Líneas del log que devuelve el backend.

    Returns:
        dict: ``hechos`` (estudiantes entrenados), ``total`` (estudiantes a
        entrenar, 0 si todavía no se sabe), ``fallaron`` y ``error`` (primera
        línea de error, o ``None``).
    """
    total = 0
    for linea in lineas:
        m = re.search(r"Usuarios con .*:\s*(\d+)\s*$", linea)
        if m:
            total += int(m.group(1))
    error = next((re.sub(r"^\W+", "", l.strip()) for l in lineas if "ERROR" in l), None)
    return {
        "hechos":   sum(1 for l in lineas if "Entrenando usuario" in l),
        "total":    total,
        "fallaron": sum(1 for l in lineas if "FALLÓ" in l),
        "error":    error,
    }


def _cerrar_confirmacion():
    """Marca como cerrada la confirmación de «Restablecer todo» (X, Esc o botones)."""
    st.session_state["admin_confirmando"] = False


@st.dialog("Restablecer todo", on_dismiss=_cerrar_confirmacion)
def _confirmar_restablecer():
    """
    Pide confirmación antes de borrar todos los feedbacks.

    Al confirmar, llama al backend y deja el resultado en
    ``st.session_state["admin_mensaje"]`` para mostrarlo en el panel.

    Estado de sesión que usa:
        admin_confirmando: ``True`` mientras la confirmación está abierta.
    """
    st.markdown(
        "<style>.st-key-admin_confirmar button {background:#3b598d !important;"
        "border:1px solid #3b598d !important;color:#FFFFFF !important}"
        ".st-key-admin_confirmar button p {color:#FFFFFF !important}"
        ".st-key-admin_confirmar button:hover {background:#2f4771 !important;"
        "border-color:#2f4771 !important}</style>",
        unsafe_allow_html=True,
    )
    st.write("Se van a borrar todos los feedbacks de todos los estudiantes, "
             "reales y ficticios, y los modelos vuelven a su estado al terminar "
             "el entrenamiento. No se puede deshacer.")
    c_no, c_si = st.columns(2)
    if c_no.button("Cancelar", use_container_width=True, key="admin_cancelar"):
        _cerrar_confirmacion()
        st.rerun()
    if c_si.button("Sí, restablecer", type="primary", use_container_width=True,
                   key="admin_confirmar"):
        _cerrar_confirmacion()
        datos, error = _pedir("POST", "/administracion/restablecer")
        if error:
            st.session_state["admin_mensaje"] = ("error", error)
        else:
            texto = (f"Listo: se eliminaron {datos['feedbacks_eliminados']} feedbacks y "
                     f"{datos['restaurados']} estudiantes volvieron a su estado entrenado.")
            st.session_state["admin_mensaje"] = ("success", texto)
            if datos["sin_copia"]:
                st.session_state["admin_aviso"] = (
                    f"{len(datos['sin_copia'])} estudiantes tenían feedbacks y no tienen copia "
                    "limpia, así que no se tocaron. Entrená el modelo de nuevo para dejarlos limpios.")
        st.rerun()

usuarios = get_usuarios()

col_brand, col_main = st.columns([1, 2], gap="large")

with col_brand:
    st.markdown("""
    <div style="position:sticky;top:2rem;padding:2rem 1rem;
                min-height:92vh;display:flex;flex-direction:column;
                align-items:center;justify-content:center;text-align:center;
                background:#3b598d;border-radius:16px">
        <div style="font-size:1.432rem;font-weight:800;color:#ffffff;
                    letter-spacing:-.03em;line-height:1.2;margin-bottom:.6rem">
            TrueLearn
        </div>
        <div style="font-size:1.05rem;color:#ffffff;line-height:1.9;
                    margin-bottom:2rem">
            Visualización del perfil<br>del estudiante y corrección<br>
            del modelo en tiempo real
        </div>
        <div style="width:40px;height:2px;background:#ffffff;
                    border-radius:2px;margin:0 auto 1.8rem"></div>
        <div style="font-size:.95rem;color:#ffffff;line-height:2.1;
                    text-align:center">
            Dataset PEEK<br>
            VideoLectures.net<br>
            TrueLearn INK<br>
            UNA · Fac. Politécnica<br>
            2026
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_main:
    st.markdown("""
    <div style="padding:2rem 0 1rem">
        <div style="font-size:1.178rem;font-weight:700;color:#0f172a;
                    letter-spacing:-.02em;margin-bottom:.3rem">
            Seleccioná un estudiante
        </div>
        <div style="font-size:1.05rem;color:#64748b;margin-bottom:1.5rem">
            Elegí un perfil del menú para visualizar su conocimiento estimado.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not usuarios:
        st.error("Backend no disponible. Corré: `python -m logic.api_backend`")
        st.stop()

    muy_activos = [u for u in usuarios if u.get("tipo") == "muy_activas"]
    muchos = [u for u in usuarios if u.get("tipo") == "muchas"]
    medios = [u for u in usuarios if u.get("tipo") == "medias"]
    pocos  = [u for u in usuarios if u.get("tipo") == "pocas"]
    todos  = muy_activos + muchos + medios + pocos

    def fmt_opcion(u):
        """
        Da formato al texto de un estudiante para el desplegable.

        Args:
            u (dict): Resumen del estudiante, con las claves ``"user_id"``,
                ``"tipo"``, ``"interacciones"``, ``"engagements"`` y,
                opcionalmente, ``"sesiones"``.

        Returns:
            str: Texto con el identificador, el tipo de estudiante, la cantidad de
            eventos, el porcentaje de enganche y la cantidad de sesiones.
        """
        tasa_e = round(u["engagements"] / u["interacciones"] * 100) if u["interacciones"] > 0 else 0
        tipo_txt = {"muy_activas": "Actividad muy alta",
                    "muchas": "Muchas interacciones",
                    "medias": "Interacciones medias"}.get(u.get("tipo"), "Pocas interacciones")
        return (f"Estudiante #{u['user_id']} ({tipo_txt}) — "
                f"{u['interacciones']} eventos · {tasa_e}% enganche · "
                f"{u.get('sesiones',0)} sesiones")

    opciones_fmt = [fmt_opcion(u) for u in todos]
    opciones_fmt.insert(0, "— Seleccioná un estudiante —")

    sel_idx = st.selectbox(
        "Estudiante",
        range(len(opciones_fmt)),
        format_func=lambda i: opciones_fmt[i],
        label_visibility="collapsed",
        key="login_dropdown",
    )

    if sel_idx > 0:
        u_sel = todos[sel_idx - 1]
        tasa_sel = round(u_sel["engagements"] / u_sel["interacciones"] * 100) if u_sel["interacciones"] > 0 else 0
        rotulo_sel = {"muy_activas": "Muy activo",
                      "muchas": "Activo",
                      "medias": "Interacciones medias"}.get(u_sel.get("tipo"), "Pocas interacciones")

        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;
                    border-radius:14px;padding:1.2rem 1.4rem;margin:.8rem 0 1rem">
            <div style="font-size:1.25rem;font-weight:700;color:#0f172a;
                        margin-bottom:.6rem">
                Estudiante #{u_sel['user_id']}
                <span style="font-size:.95rem;font-weight:500;color:#64748b;
                             margin-left:.5rem">
                    {rotulo_sel}
                </span>
            </div>
            <div style="display:flex;gap:1.5rem;font-size:1.02rem;color:#475569">
                <span><b>{u_sel['interacciones']}</b> eventos</span>
                <span><b>{tasa_sel}%</b> enganche</span>
                <span><b>{u_sel.get('sesiones',0)}</b> sesiones</span>
                <span>Pesos aprendidos</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Ingresar al perfil →", type="primary",
                     use_container_width=True, key="btn_ingresar"):
            st.session_state["uid_seleccionado"] = u_sel["user_id"]
            st.rerun()

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#fff8ed;border:1px solid #fed7aa;border-radius:10px;
                padding:.7rem 1rem;font-size:1rem;color:#92400e;margin-bottom:1rem">
        <b>Nota sobre el dataset:</b> PEEK proviene de VideoLectures.net y está
        dominado por Ciencias de la Computación e IA. Los perfiles reflejan ese
        sesgo temático y no representan estudiantes de otras disciplinas.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Administración del modelo"):
        st.caption("Entrená el modelo desde PEEK, agregá feedbacks de prueba o dejá todo limpio.")

        log, _ = _pedir("GET", "/log_entrenamiento", timeout=3)
        log = log or {}
        entrenando = bool(log.get("active"))

        # Mientras se entrena, el panel se actualiza solo cada 2 segundos.
        @st.fragment(run_every=2 if entrenando else None)
        def _panel_administracion():
            """Dibuja los botones, el estado del entrenamiento y los mensajes."""
            log_actual, _ = _pedir("GET", "/log_entrenamiento", timeout=3)
            log_actual = log_actual or {}
            activo = bool(log_actual.get("active"))

            if st.session_state.get("admin_entrenando") and not activo:
                st.session_state["admin_entrenando"] = False
                st.rerun()                      # recarga la lista de estudiantes
            st.session_state["admin_entrenando"] = activo

            c_entrenar, c_ficticios, c_restablecer = st.columns(3)
            with c_entrenar:
                entrenar = st.button(
                    "Entrenar modelos desde PEEK", use_container_width=True,
                    key="btn_entrenar", disabled=activo,
                    help="Entrena a todos los estudiantes desde cero con el dataset PEEK. "
                         "Tarda unos minutos y reemplaza sus feedbacks.")
            with c_ficticios:
                agregar = st.button(
                    "Agregar feedbacks ficticios", use_container_width=True,
                    key="btn_ficticios", disabled=activo,
                    help="Agrega correcciones de prueba a cada estudiante y las aplica al modelo. "
                         "Quedan marcadas como ficticias.")
                cantidad = st.number_input(
                    "Feedbacks por estudiante", min_value=1, max_value=50, value=5,
                    step=1, key="admin_cantidad", disabled=activo)
                dias = st.number_input(
                    "Repartidos en los últimos (días)", min_value=1, max_value=365,
                    value=30, step=1, key="admin_dias", disabled=activo,
                    help="Cada feedback ficticio cae en un día distinto de esta ventana, "
                         "desde ayer hacia atrás.")
            with c_restablecer:
                restablecer = st.button(
                    "Restablecer todo", use_container_width=True, key="btn_restablecer",
                    disabled=activo,
                    help="Borra todos los feedbacks y deja los modelos como quedaron al entrenar.")

            if entrenar:
                _, error = _pedir("POST", "/entrenar", timeout=10)
                if error:
                    st.session_state["admin_mensaje"] = ("error", error)
                st.rerun()
            if agregar:
                datos, error = _pedir("POST", "/administracion/feedbacks_ficticios",
                                      json={"por_aprendiz": int(cantidad), "dias": int(dias)})
                if error:
                    st.session_state["admin_mensaje"] = ("error", error)
                else:
                    st.session_state["admin_mensaje"] = (
                        "success",
                        f"Se agregaron {datos['feedbacks']} feedbacks ficticios a "
                        f"{datos['aprendices']} estudiantes.")
                st.rerun()
            if restablecer:
                st.session_state["admin_confirmando"] = True
            if st.session_state.get("admin_confirmando"):
                _confirmar_restablecer()

            if activo:
                r = _resumir_log(log_actual.get("log", []))
                if r["hechos"] and r["total"]:
                    st.progress(min(r["hechos"] / r["total"], 1.0),
                                text=f"Entrenando estudiantes: {r['hechos']} de {r['total']}")
                else:
                    st.progress(0.0, text="Preparando el entrenamiento (cargando el dataset PEEK)...")
            elif log_actual.get("done"):
                r = _resumir_log(log_actual.get("log", []))
                if r["error"]:
                    st.error(f"El entrenamiento falló: {r['error'][:200]}")
                else:
                    st.caption("Último entrenamiento: completado."
                               + (f" {r['fallaron']} estudiantes fallaron y se saltearon."
                                  if r["fallaron"] else ""))

            tipo, texto = st.session_state.pop("admin_mensaje", (None, None))
            if tipo == "success":
                st.success(texto)
            elif tipo == "error":
                st.error(texto)
            aviso = st.session_state.pop("admin_aviso", None)
            if aviso:
                st.warning(aviso)

            est = None if activo else _pedir("GET", "/administracion/estado", timeout=10)[0]
            if est:
                st.caption(
                    f"{est['aprendices']} estudiantes · {est['feedbacks']} feedbacks"
                    + (f" ({est['ficticios']} ficticios)" if est["ficticios"] else ""))

        _panel_administracion()

st.stop()
