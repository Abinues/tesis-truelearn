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

from ajuste_letras import aplicar_letras_grandes
from shared import fmt_ts, get_perfil, inject_css, popup_editar


inject_css()
aplicar_letras_grandes()

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

st.title(":material/show_chart: Distribución de correcciones")
st.markdown(
    "Cada vez que el estudiante corrigió la estimación del sistema, "
    "se muestra su distribución acá. El sistema ajusta su modelo con cada corrección."
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
                             "left":"6%","width":"88%","sort":"descending","minSize":"40%",
                             "label":{"position":"inside","formatter":"{b}\n{c}",
                                      "color":"#fff","fontSize":15,"fontWeight":"bold"},
                             "itemStyle":{"borderColor":"#fff","borderWidth":2},
                             "data":[
                                 {"value":n_temas_,    "name":"Temas explorados","itemStyle":{"color":"#293e63"}},
                                 {"value":n_feedback,  "name":"Temas corregidos","itemStyle":{"color":"#2f4771"}},
                                 {"value":n_total_fb,  "name":"Total correcciones","itemStyle":{"color":"#35507f"}},
                                 {"value":n_positivos, "name":"Correcciones positivas","itemStyle":{"color":"#3b598d"}},
                             ]}],
            }
            st_echarts(funnel_opt, height="400px", key="funnel_calib", theme="streamlit")

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
                            "fontSize":   15,
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
                                "textStyle": {"fontSize": 15}},
                    "grid":    {"bottom": "28%", "top": "8%","left":"14%", "containLabel": True},
                    "xAxis":   {"type": "category", "data": fechas_display,
                                "axisLabel": {"rotate": 30, "color": "#64748b", "fontSize": 15},
                                "axisTick": {"show": False}},
                    "yAxis":   {"type": "value", "name": "Correcciones",
                                "nameLocation":"middle","nameGap":46,
                                "nameTextStyle": {"color": "#64748b", "fontSize": 15},
                                "axisLabel": {"color": "#64748b", "fontSize": 15},
                                "splitLine": {"show": False},
                                "axisLine": {"show": False}},
                    "series":  bar_series,
                }
                st_echarts(mom_opt, height="400px", key="acum_corrections", theme="streamlit")
            else:
                st.info("Sin correcciones registradas aún.")





