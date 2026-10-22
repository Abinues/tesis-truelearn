"""
Página «Correcciones al modelo».

Muestra el registro de correcciones explícitas que el estudiante aplicó sobre
su perfil y permite aplicar una nueva. Contiene cuatro bloques:

    1. Embudo de correcciones: temas explorados, temas corregidos, total de
       correcciones y correcciones al alza.
    2. Barras apiladas con las correcciones por día y por tipo.
    3. Selector de tema y botón para abrir el popup de corrección.
    4. Tabla con el historial completo de correcciones.

Estado de sesión que usa:
    uid_seleccionado: Identificador del aprendiz elegido.
    perfil_actual: Perfil del aprendiz (si falta, se pide al backend).
    popup_open, popup_topic: Control del popup de corrección.
"""
import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

from shared import fmt_ts, get_perfil, inject_css, popup_editar


inject_css()

uid    = st.session_state.get("uid_seleccionado")
perfil = st.session_state.get("perfil_actual") or get_perfil(uid)
if not perfil or not uid:
    st.info("Seleccioná un aprendiz para comenzar.")
    st.stop()

kcs       = perfil.get("knowledge_components", [])
feedbacks = perfil.get("explicit_feedback", [])
df_kcs    = pd.DataFrame(kcs) if kcs else pd.DataFrame()

if st.session_state.get("popup_open") and st.session_state.get("popup_topic"):
    popup_editar(st.session_state["popup_topic"], uid)

st.title(":material/edit_note: Correcciones al modelo")
st.markdown(
    "Cada vez que el estudiante (o el evaluador) corrigió la estimación del sistema, "
    "queda registrado acá. El sistema ajusta su modelo con cada corrección."
)
label_tipo = {
    "sé más":         "Sé más",
    "sé menos":       "Sé menos",
    "me interesa":    "Me interesa",
    "no me interesa": "No me interesa",
}

df_fb = pd.DataFrame(feedbacks)
if not df_fb.empty:
    df_fb["Fecha"] = df_fb["timestamp"].apply(fmt_ts)
    df_fb["idx"]   = range(len(df_fb))

if not feedbacks:
    st.info("Todavía no se aplicaron correcciones sobre este perfil.")

col1, col2 = st.columns(2, gap="small")

with col1:
        with st.container(border=True):
            st.markdown("**EMBUDO DE CORRECCIONES**")
            n_temas_    = len(df_kcs)
            n_feedback  = len(set(fb.get("topic_id") for fb in feedbacks))
            n_total_fb  = len(feedbacks)
            n_positivos = sum(1 for fb in feedbacks if fb.get("tipo") in ["sé más","me interesa"])
            funnel_opt = {
                "tooltip": {"trigger":"item","formatter":"{b}: {c}"},
                "series":  [{"name":"Correcciones","type":"funnel",
                             "left":"15%","width":"70%","sort":"descending",
                             "label":{"position":"inside","formatter":"{b}\n{c}",
                                      "color":"#fff","fontSize":11,"fontWeight":"bold"},
                             "itemStyle":{"borderColor":"#fff","borderWidth":2},
                             "data":[
                                 {"value":n_temas_,    "name":"Temas explorados","itemStyle":{"color":"#293e63"}},
                                 {"value":n_feedback,  "name":"Temas corregidos","itemStyle":{"color":"#2f4771"}},
                                 {"value":n_total_fb,  "name":"Total correcciones","itemStyle":{"color":"#35507f"}},
                                 {"value":n_positivos, "name":"Correcciones positivas","itemStyle":{"color":"#3b598d"}},
                             ]}],
            }
            st_echarts(funnel_opt, height="320px", key="funnel_calib", theme="streamlit")

with col2:
        with st.container(border=True):
            st.markdown("**CORRECCIONES EN EL TIEMPO**")
            from collections import defaultdict
            from datetime import datetime

            TIPO_COLOR = {"sé más":"#8d3b82","me interesa":"#5f8fe2",
                          "sé menos":"#8d6f3b","no me interesa":"#172338"}

            por_fecha_tipo = defaultdict(lambda: defaultdict(int))
            for _, row in df_fb.iterrows():
                try:
                    fecha = datetime.fromtimestamp(float(row["timestamp"])).strftime("%Y-%m-%d")
                except Exception:
                    fecha = "Sin fecha"
                por_fecha_tipo[fecha][row.get("tipo","")] += 1

            from datetime import datetime as dt2
            fechas_iso     = sorted(f for f in por_fecha_tipo if f != "Sin fecha")
            fechas_display = [dt2.strptime(f, "%Y-%m-%d").strftime("%d/%m/%Y") for f in fechas_iso]

            bar_series = []
            for tipo, color in TIPO_COLOR.items():
                lbl      = label_tipo.get(tipo, tipo)
                data_pts = []
                for f in fechas_iso:
                    cnt = por_fecha_tipo[f].get(tipo, 0)
                    data_pts.append({
                        "value": cnt,
                        "itemStyle": {"color": color},
                        "label": {
                            "show":       cnt > 0,
                            "position":   "top",
                            "formatter":  str(cnt) if cnt > 0 else "",
                            "color":      color,
                            "fontWeight": "bold",
                            "fontSize":   11,
                        },
                    })
                bar_series.append({
                    "name":        lbl,
                    "type":        "bar",
                    "stack":       "total",
                    "data":        data_pts,
                    "barMaxWidth": 48,
                    "itemStyle":   {"color": color},
                    "emphasis":    {"focus": "series"},
                })

            if bar_series and fechas_iso:
                mom_opt = {
                    "tooltip": {"trigger": "axis"},
                    "legend":  {"data": [s["name"] for s in bar_series],
                                "bottom": 0, "type": "scroll",
                                "textStyle": {"fontSize": 10}},
                    "grid":    {"bottom": "20%", "top": "8%","left":"10%", "containLabel": True},
                    "xAxis":   {"type": "category", "data": fechas_display,
                                "axisLabel": {"rotate": 30, "color": "#64748b", "fontSize": 10},
                                "axisTick": {"show": False}},
                    "yAxis":   {"type": "value", "name": "Correcciones",
                                "nameLocation":"middle","nameGap":40,
                                "nameTextStyle": {"color": "#64748b", "fontSize": 10},
                                "axisLabel": {"color": "#64748b", "fontSize": 10},
                                "splitLine": {"show": False},
                                "axisLine": {"show": False}},
                    "series":  bar_series,
                }
                st_echarts(mom_opt, height="320px", key="acum_corrections", theme="streamlit")
            else:
                st.info("Sin correcciones registradas aún.")


col3, col4 = st.columns(2, gap="small")

with col3:
 with st.container(border=True):
        st.subheader(":material/edit: Aplicar una corrección")
        st.caption(
            "Seleccioná un tema y presioná **Editar** para abrir el panel de corrección. "
            "Funciona igual que en el Mapa de Temas."
        )

        if not df_kcs.empty:
            from shared import popup_editar
            nombres  = {kc["topic_name"]: kc for kc in kcs}
            col_sel, col_btn = st.columns([3, 1], gap="small")
            with col_sel:
                tema_corr = st.selectbox("Seleccioná un tema:", list(nombres.keys()), key="corr_tema")
            with col_btn:
                st.markdown("<div style='margin-top:1.85rem'></div>", unsafe_allow_html=True)
                if st.button("Editar", type="primary", use_container_width=True, key="corr_edit_btn"):
                    st.session_state["popup_topic"] = nombres[tema_corr]
                    st.session_state["popup_open"]  = True
                    st.rerun()


with col4:
        with st.container(border=True):
            st.markdown("**HISTORIAL DE CORRECCIONES**")
            if df_fb.empty:
                st.info("Sin correcciones registradas aún.")
            else:
                df_show = df_fb.copy()
                df_show["Tipo"] = df_show["tipo"].map(label_tipo).fillna(df_show["tipo"])
                st.dataframe(
                    df_show[["Fecha","topic_name","Tipo"]].rename(
                        columns={"topic_name":"Tema"}),
                    use_container_width=True, hide_index=True, height=200)
