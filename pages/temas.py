"""
Página «¿Dónde cae cada tema?» (Mapa de temas).

Exploración completa de los temas del aprendiz:

    - Sunburst de temas agrupados por nivel de conocimiento (Alto, Medio y
      Básico).
    - Gráfico de burbujas de conocimiento por interés, donde el tamaño
      indica la confianza del sistema. Al hacer clic en una burbuja se
      muestra el detalle del tema.
    - Tarjetas con todos los temas, con búsqueda, orden y botón para abrir
      el popup de corrección.

Estado de sesión que usa:
    uid_seleccionado: Identificador del aprendiz elegido.
    perfil_actual: Perfil del aprendiz (si falta, se pide al backend).
    popup_open, popup_topic: Control del popup de corrección.
    last_busq, last_orden: Última búsqueda y orden, para cerrar el popup
        cuando cambian.
"""
import pandas as pd
import streamlit as st
from streamlit_echarts import JsCode, st_echarts

from logic.dominio import (
    PCT_INICIO_CONOCIMIENTO,
    PCT_INICIO_INTERES,
    UMBRAL_ALTO,
    UMBRAL_BASICO,
    UMBRAL_MEDIO,
    pct_certeza,
    pct_conocimiento,
    pct_interes,
)
from shared import TEAL, get_perfil, inject_css, interes_label, nivel_label, popup_editar, var_label

inject_css()

uid    = st.session_state.get("uid_seleccionado")
perfil = st.session_state.get("perfil_actual") or get_perfil(uid)
if not perfil or not uid:
    st.info("Seleccioná un aprendiz.")
    st.stop()

kcs    = perfil.get("knowledge_components", [])
df_kcs = pd.DataFrame(kcs) if kcs else pd.DataFrame()

if st.session_state.get("popup_open") and st.session_state.get("popup_topic"):
    popup_editar(st.session_state["popup_topic"], uid)

st.title(":material/bubble_chart: ¿Dónde cae cada tema?")
st.markdown(
    "Exploración completa de los temas del aprendiz: "
    "dónde se ubica cada uno, cuáles le cuestan y cómo se distribuyen por nivel."
)

if df_kcs.empty:
    st.warning("No hay temas disponibles para este aprendiz.")
    st.stop()


row1_1, row1_2 = st.columns([1, 2.3], gap="small")

with row1_1:
    with st.container(border=True):
        st.markdown("**TOP 5 TEMAS POR TIPO DE CONOCIMIENTO**")

        altos   = df_kcs[df_kcs["mean"] >= UMBRAL_ALTO]
        medios  = df_kcs[(df_kcs["mean"] >= UMBRAL_MEDIO) & (df_kcs["mean"] < UMBRAL_ALTO)]
        basicos = df_kcs[(df_kcs["mean"] >= UMBRAL_BASICO) & (df_kcs["mean"] < UMBRAL_MEDIO)]

        def make_ch(df_s, color_temas, n=5):
            """
            Construye los nodos hijos del sunburst para un grupo de temas.

            Toma los ``n`` temas de mayor conocimiento del grupo.

            Args:
                df_s (pd.DataFrame): Temas del grupo, con las columnas
                    ``"topic_name"`` y ``"mean"``.
                color_temas (str): Color hexadecimal de los nodos.
                n (int): Cantidad máxima de temas. Por defecto 5.

            Returns:
                list[dict]: Nodos con ``name`` (recortado a 18 caracteres),
                ``value`` (porcentaje de conocimiento, con mínimo visual de
                2.5) e ``itemStyle``.
            """
            return [
                {
                    "name": str(r["topic_name"])[:18],
                    "value": max(pct_conocimiento(r["mean"]), 2.5),
                    "itemStyle": {
                        "color": color_temas
                    }
                }
                for _, r in df_s.nlargest(n, "mean").iterrows()
            ]


        sb_data = []

        for label, filt, color_categoria, color_temas in [
            (f"ALTO ({len(altos)})",     altos,   "#8d3b82", "#984f8f"),
            (f"MEDIO ({len(medios)})",   medios,  "#3b8d46", "#4f9859"),
            (f"BÁSICO ({len(basicos)})", basicos, "#8d6f3b", "#987d4f"),
        ]:
            if len(filt) > 0:
                sb_data.append({
                    "name": label,
                    "itemStyle": {
                        "color": color_categoria
                    },
                    "children": make_ch(filt, color_temas)
                })

        if sb_data:
            sb_opt = {
                "tooltip": {"trigger": "item", "formatter": "{b}: {c}%"},
                "series": [{"type":"sunburst","data":sb_data,"center": ["50%", "50%"],"radius":["15%","98%"],
                            "sort":None,"emphasis":{"focus":"ancestor"},
                            "levels":[
                                        {},
                                        {
                                            "r0":"12%", "r":"42%",
                                            "itemStyle":{"borderWidth":2},
                                            "label":{
                                                "rotate":"tangential",
                                                "fontSize":11,
                                                "color":"#fff",
                                                "fontWeight":"bold"
                                            }
                                        },
                                        {
                                            "r0":"42%", "r":"75%",
                                            "label":{
                                                "align":"right",
                                                "fontSize":9,
                                                "color":"#fff"
                                            }
                                        },
                                        {
                                            "r0":"75%", "r":"78%",
                                            "itemStyle":{"borderWidth":3},
                                            "label":{
                                                "position":"outside",
                                                "padding":3,
                                                "fontSize":8,
                                                "color":"#fff"
                                            }
                                        }
                                    ]}],
            }
            st_echarts(sb_opt, height="628px", width="100%", key="sunburst_temas", theme="streamlit")
        else:
            st.info("No hay temas con conocimiento ≥ 10% para mostrar en este gráfico.")


with row1_2:
        df_s = df_kcs.copy()
        df_s["certeza"] = 1 / df_s["variance"].clip(lower=0.01)
        min_c = df_s["certeza"].min()
        max_c = df_s["certeza"].max()
        rng_c = max_c - min_c if max_c != min_c else 1.0
        df_s["bsize"] = 10 + (df_s["certeza"] - min_c) / rng_c * 50
        ts_vals = df_s["timestamp"].fillna(0)
        min_ts = ts_vals.min()
        max_ts = ts_vals.max()
        rng_ts = max_ts - min_ts if max_ts != min_ts else 1.0

        def ts_color(ts):
            """
            Asigna un color según qué tan reciente es la marca de tiempo.

            Interpola linealmente entre un gris (el tema más antiguo) y el
            azul ``#3b598d`` (el más reciente).

            Args:
                ts (float): Marca de tiempo del tema.

            Returns:
                str: Color hexadecimal con el formato ``"#rrggbb"``.
            """
            t = (float(ts) - min_ts) / rng_ts if rng_ts > 0 else 0.5

            r = int(148 + (59 - 148) * t)
            g = int(163 + (89 - 163) * t)
            b = int(184 + (141 - 184) * t)

            return f"#{r:02x}{g:02x}{b:02x}"

        scatter_data = [{
            "value":      [pct_interes(r["interest_mean"]),
                        pct_conocimiento(r["mean"]),
                        round(r["bsize"],1)],
            "name":       str(r["topic_name"])[:30],
            "symbolSize": round(r["bsize"],1),
            "itemStyle":  {"color": ts_color(r["timestamp"] or 0)},
        } for _, r in df_s.iterrows()]
        rev_max = float(df_s["bsize"].max()) or 1

        @st.fragment
        def bubble_section():
            """
            Dibuja el gráfico de burbujas y el detalle del tema seleccionado.

            Cada burbuja es un tema: su posición horizontal es el interés, la
            vertical el conocimiento, el tamaño la confianza del sistema y el
            color qué tan reciente es la estimación. Al hacer clic en una
            burbuja se muestran las métricas y las etiquetas del tema. Se
            ejecuta como fragmento para que la selección no recargue toda la
            página.

            Returns:
                None
            """
            col_b, col_d = st.columns(2)
            with col_b:
                st.markdown("**CONOCIMIENTO POR INTERÉS SEGUN CONFIANZA DEL SISTEMA**")
                scatter_opts = {
                    "animation": False,
                    "tooltip":   {"trigger":"item",
                                "formatter": JsCode(
                                    "function(p){ var v=p.value;"
                                    "return '<b>'+p.data.name+'</b><br/>"
                                    "Conocimiento: '+v[1].toFixed(1)+'%<br/>"
                                    "Interés: '+v[0].toFixed(1)+'%'; }"
                                )},
                    "grid": {"top":"0%","bottom":"0%"},
                    "xAxis": {"name":"Interés (%)","type":"value","min":0,"max":100,
                            "nameLocation":"middle","nameGap":28,
                            "nameTextStyle":{"color":"#64748b","fontSize":11},
                            "splitLine":{"show": False},
                            "axisLabel":{"color":"#64748b","formatter":"{value}%"}},
                    "yAxis": {"name":"Conocimiento (%)","type":"value","min":0,"max":100,
                            "nameLocation":"middle","nameGap":40,
                            "nameTextStyle":{"color":"#64748b","fontSize":11},
                            "splitLine":{"show": False},
                            "axisLabel":{"color":"#64748b","formatter":"{value}%"}},
                    "visualMap": {"show":False,"dimension":2,"min":0,"max":float(rev_max),
                                "inRange":{"symbolSize":[10,60]}},
                    "series": [{"type":"scatter","data":scatter_data,
                                "itemStyle":{"opacity":0.75},
                                "emphasis":{"itemStyle":{"borderColor":"#333","borderWidth":2}},
                                "markLine":{"silent":True,
                                            "lineStyle":{"type":"dashed","color":"#cbd5e1"},
                                            "data":[
                                                {"xAxis":pct_interes(df_s["interest_mean"].median())},
                                                {"yAxis":pct_conocimiento(df_s["mean"].median())},
                                            ]}}],
                }
                result = st_echarts(options=scatter_opts, height="230px", key="bubble_scatter",
                                    theme="streamlit", on_select="rerun", selection_mode="points")

            with col_d:
                indices = result["selection"].get("point_indices",[]) if result else []
                if indices:
                    clicked = scatter_data[indices[0]]
                    match   = df_kcs[df_kcs["topic_name"].str[:30] == clicked["name"]]
                    if not match.empty:
                        row = match.iloc[0]
                        lbl,cls  = nivel_label(row["mean"])
                        vlbl,vcls= var_label(row["variance"])
                        ilbl,icls= interes_label(row["interest_mean"])
                        know_b   = pct_conocimiento(row["mean"])
                        cert_b   = pct_certeza(row["variance"])
                        int_b    = pct_interes(row["interest_mean"])

                        st.markdown(f"**{row['topic_name']}**")
                        m1,m2,m3 = st.columns(3)
                        m1.metric("CONOCIMIENTO",f"{know_b:.1f}%",delta=f"{know_b-PCT_INICIO_CONOCIMIENTO:+.1f}%",border=True,delta_color="blue",
                                  help="Qué tan bien maneja el estudiante este tema puntual, según sus interacciones con los videos relacionados.")
                        m2.metric("INTERÉS",     f"{int_b:.1f}%", delta=f"{int_b-PCT_INICIO_INTERES:+.1f}%", border=True,delta_color="blue",
                                  help="Qué tanto interés mostró el estudiante por este tema puntual, según cómo interactuó con los videos relacionados.")
                        m3.metric("CONFIANZA",   f"{cert_b:.1f}%",delta=f"{cert_b-50.0:+.1f}%",border=True,delta_color="blue",
                                  help="Qué tan segura está esta estimación puntual. Con pocas interacciones en este tema, es poco firme; con más, se vuelve más confiable.")
                        st.markdown(
                            f"<span class='badge {cls}'>{lbl}</span>&nbsp;"
                            f"<span class='badge {icls}'>Int. {ilbl}</span>&nbsp;"
                            f"<span class='badge {vcls}'>{vlbl}</span>",
                            unsafe_allow_html=True)
                        if row.get("topic_url"):
                            st.markdown(f"[Wikipedia]({row['topic_url']})")

                else:
                    st.info("Clic en una burbuja para ver el detalle.", icon=":material/touch_app:")

        with st.container( border=True):
            bubble_section()


        with st.container(border=True):

            st.markdown("**TODOS LOS TEMAS**")

            # Buscador (sin selector de ordenamiento)
            busq = st.text_input(
                "Buscar tema",
                placeholder="Buscar tema...",
                label_visibility="collapsed",
                key="mapa_busq"
            )

            if busq != st.session_state.get("last_busq", ""):
                st.session_state.update({
                    "popup_open": False,
                    "popup_topic": None,
                    "last_busq": busq
                })

            # Filtrar temas
            df_filt = df_kcs.copy()

            if busq:
                df_filt = df_filt[
                    df_filt["topic_name"].str.contains(
                        busq, case=False, regex=False, na=False
                    )
                ]

            # Mantener el orden original
            df_filt = df_filt.reset_index(drop=True)

            # Tarjetas de temas
            with st.container(height=250, border=True):

                if df_filt.empty:
                    st.info("No se encontraron temas.")

                else:
                    st.markdown(
                        '<div id="mapa-cards-wrap">',
                        unsafe_allow_html=True
                    )

                    N = 3

                    for i in range(0, len(df_filt), N):
                        cols = st.columns(N, gap="small")

                        for j, (_, row) in enumerate(
                            df_filt.iloc[i:i + N].iterrows()
                        ):
                            lbl, cls = nivel_label(row["mean"])
                            ilbl, icls = interes_label(row["interest_mean"])
                            vlbl, vcls = var_label(row["variance"])

                            url_part = (
                                f'<a href="{row["topic_url"]}" target="_blank" '
                                f'style="font-size:.68rem;color:{TEAL};'
                                f'font-weight:600">Wikipedia ↗</a>'
                                if row.get("topic_url")
                                else ""
                            )

                            with cols[j]:
                                with st.container(border=True):

                                    # Nombre del tema
                                    st.markdown(
                                        f"<div style='font-weight:700;"
                                        f"font-size:.82rem;color:#0f172a;"
                                        f"margin-bottom:.4rem;"
                                        f"white-space:nowrap;overflow:hidden;"
                                        f"text-overflow:ellipsis'>"
                                        f"{str(row['topic_name'])[:30]}</div>",
                                        unsafe_allow_html=True
                                    )

                                    # Métricas
                                    v1, v2, v3 = st.columns(3)

                                    for col, lbl_, val in [
                                        (
                                            v1,
                                            "Nivel",
                                            pct_conocimiento(row["mean"])
                                        ),
                                        (
                                            v2,
                                            "Interés",
                                            pct_interes(row["interest_mean"])
                                        ),
                                        (
                                            v3,
                                            "Conf.",
                                            pct_certeza(row["variance"])
                                        ),
                                    ]:
                                        col.markdown(
                                            f'<div style="text-align:center">'
                                            f'<div style="font-size:.55rem;'
                                            f'color:#94a3b8;text-transform:uppercase;'
                                            f'letter-spacing:.06em">{lbl_}</div>'
                                            f'<div style="font-size:.9rem;'
                                            f'font-weight:700;color:#0f172a">'
                                            f'{val}%</div></div>',
                                            unsafe_allow_html=True
                                        )

                                    # Etiquetas
                                    bh = (
                                        f'<span class="badge {cls}">'
                                        f'{lbl}</span>&nbsp;'
                                    )
                                    bh += (
                                        f'<span class="badge {icls}">'
                                        f'Int.&nbsp;{ilbl}</span>&nbsp;'
                                    )
                                    bh += (
                                        f'<span class="badge {vcls}">'
                                        f'{vlbl}</span>'
                                    )

                                    if url_part:
                                        bh += (
                                            f'<br><small style="display:block;'
                                            f'margin-top:.2rem">'
                                            f'{url_part}</small>'
                                        )

                                    st.markdown(
                                        bh,
                                        unsafe_allow_html=True
                                    )

                                    # Botón Editar
                                    if st.button(
                                        "Editar",
                                        key=f"edit_{row['topic_id']}_{i}_{j}",
                                        use_container_width=True
                                    ):
                                        popup_data = row.to_dict()

                                        popup_data["topic_id"] = int(
                                            popup_data["topic_id"]
                                        )
                                        popup_data["mean"] = float(
                                            popup_data["mean"]
                                        )
                                        popup_data["variance"] = float(
                                            popup_data["variance"]
                                        )
                                        popup_data["interest_mean"] = float(
                                            popup_data["interest_mean"]
                                        )

                                        st.session_state["popup_topic"] = popup_data
                                        st.session_state["popup_open"] = True

                                        st.rerun()

                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )

