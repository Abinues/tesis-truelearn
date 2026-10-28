"""
Página «¿Cómo está aprendiendo este estudiante?» (Resumen del aprendiz).

Resume el perfil estimado por TrueLearn para el aprendiz elegido:

    - Dos tarjetas: nivel de conocimiento y nivel de interés, cada una con su
      diferencia respecto del estado inicial y, debajo, qué tan seguro está
      el sistema de esa estimación.
    - Anillo con el porcentaje de participación (videos completados sobre el
      total).
    - Comparación entre videos completados y no completados.
    - Los 10 temas con mayor nivel de conocimiento.

Estado de sesión que usa:
    uid_seleccionado: Identificador del aprendiz elegido.
    perfil_actual: Perfil del aprendiz (si falta, se pide al backend).
    popup_open, popup_topic: Control del popup de corrección.
"""
import pandas as pd
import streamlit as st
from streamlit_echarts import JsCode, st_echarts

from logic.dominio import (
    PCT_INICIO_CONOCIMIENTO,
    PCT_INICIO_INTERES,
    pct_certeza,
    pct_conocimiento,
    pct_interes,
)
from ajuste_letras import aplicar_letras_grandes
from shared import get_perfil, inject_css, popup_editar

# Los tooltips con nombres de tema largos se mantienen dentro del gráfico y
# pasan a varias líneas, en vez de salirse de la pantalla.
TOOLTIP_AJUSTE = {
    "confine": True,
    "extraCssText": "max-width:340px;white-space:normal;word-break:break-word;",
}

inject_css()
aplicar_letras_grandes()

uid    = st.session_state.get("uid_seleccionado")
perfil = st.session_state.get("perfil_actual") or get_perfil(uid)
if not perfil or not uid:
    st.info("Seleccioná un aprendiz para comenzar.")
    st.stop()

kcs       = perfil.get("knowledge_components", [])
historial = perfil.get("session_history", [])
eng       = perfil.get("number_of_engagements", 0)
non_eng   = perfil.get("number_of_non_engagements", 0)
total     = eng + non_eng
df_kcs    = pd.DataFrame(kcs) if kcs else pd.DataFrame()

tasa_eng  = round(eng / total * 100, 1) if total > 0 else 0.0
mean_know = df_kcs["mean"].mean()          if not df_kcs.empty else 0.0
mean_int  = df_kcs["interest_mean"].mean() if not df_kcs.empty else 0.0
var_know  = df_kcs["variance"].mean()      if not df_kcs.empty else 0.5
var_int   = df_kcs["interest_var"].mean()  if not df_kcs.empty else 0.5

know_pct  = pct_conocimiento(mean_know)
int_pct   = pct_interes(mean_int)
cert_know = pct_certeza(var_know)   # confianza del sistema en el conocimiento
cert_int  = pct_certeza(var_int)    # confianza del sistema en el interés

_n_temas_spark = len(df_kcs) if not df_kcs.empty else 1
_max_posible   = _n_temas_spark * 100
spark_know = [
    round(min(100, sum(pct_conocimiento(v) for v in s.get("means",{}).values())
              / _max_posible * 100), 1)
    for s in historial
] or None

spark_cert = None
if historial and any(s.get("variances") for s in historial):
    spark_cert = [
        pct_certeza(sum(s.get("variances",{}).values()) /
                    max(len(s.get("variances",{})), 1))
        for s in historial
    ] or None

spark = spark_know

if st.session_state.get("popup_open") and st.session_state.get("popup_topic"):
    popup_editar(st.session_state["popup_topic"], uid)

st.title(":material/person: ¿Cómo está aprendiendo este estudiante?")
st.markdown(
    f"Perfil de **Aprendiz #{uid}** generado a partir de sus interacciones con videos educativos. "
    "Cada indicador refleja lo que el sistema estima sobre su conocimiento e interés."
)

col_izq, col_der = st.columns([1.4, 1], gap="small")

with col_izq:
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True, height=225):
            st.metric("NIVEL DE CONOCIMIENTO", f"{know_pct:.1f}%",
                    delta=f"{know_pct - PCT_INICIO_CONOCIMIENTO:+.1f}%",
                    delta_color="blue",
                    help=f"Qué tan bien maneja el estudiante, en promedio, los temas con los que interactuó — estimado a partir de qué videos completó y cuáles no. Antes de cualquier interacción, el sistema no favorece 'sabe' ni 'no sabe': ese punto neutral equivale a {PCT_INICIO_CONOCIMIENTO:.1f}%.")
            st.caption(f"El sistema está **{cert_know:.1f}%** seguro de esta estimación")
    with col2:
        with st.container(border=True, height=225):
            st.metric("NIVEL DE INTERÉS", f"{int_pct:.1f}%",
                    delta=f"{int_pct - PCT_INICIO_INTERES:+.1f}%",
                    delta_color="blue",
                    help=f"Qué tanto interés mostró el estudiante por los temas, en promedio, según cómo interactuó con los videos. El punto neutral (sin ninguna interacción) equivale a {PCT_INICIO_INTERES:.1f}%.")
            st.caption(f"El sistema está **{cert_int:.1f}%** seguro de esta estimación")

    row2_1, row2_2 = st.columns([1, 1.5], gap="small")

    with row2_1:
        with st.container(border=True):
            st.markdown("**% DE PARTICIPACIÓN**")
            ring_opt = {
                "series": [{
                    "type": "gauge", "startAngle": 90, "endAngle": -270,
                    "pointer": {"show": False},
                    "progress": {"show": True, "overlap": False, "roundCap": True, "clip": False,
                                "itemStyle": {"borderWidth": 1, "borderColor": "#e2e8f0"}},
                    "radius": "100%",
                    "axisLine":  {"lineStyle": {"width": 30}},
                    "splitLine": {"show": False}, "axisTick": {"show": False}, "axisLabel": {"show": False},
                    "data": [{"value": tasa_eng, "name": "Participación",
                            "title":  {"offsetCenter": ["0%", "-10%"], "fontSize": 18, "color": "#3b598d"},
                            "detail": {"offsetCenter": ["0%", "18%"]},
                            "itemStyle": {"color": "#3b598d"}}],
                    "title":  {"fontSize": 18},
                    "detail": {"width": 80, "height": 26, "fontSize": 25, "color": "#3b598d",
                            "borderColor": "#3b598d", "borderRadius": 20, "borderWidth": 1,
                            "formatter": "{value}%"},
                }],
            }
            st_echarts(ring_opt, height="250px", key="ring_eng", theme="streamlit")
    with row2_2:
        with st.container(border=True):
            st.markdown("**VIDEOS COMPLETADOS VS NO COMPLETADOS**")
            video_icon = (
                "path://M8 5v14l11-7z"
            )
            x_icon = (
                "path://M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 "
                "17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"
            )
            pic_opt = {
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "legend": {"data": ["Completados", "No completados"], "top": 5, "textStyle": {"fontSize": 16}},
                "grid":   {"containLabel": True, "left": 20, "right": 45, "top": 48, "bottom": 15},
                "yAxis":  {"data": ["Videos"], "inverse": True,
                        "axisLine": {"show": False}, "axisTick": {"show": False},
                        "axisLabel": {"fontSize": 1}},
                "xAxis":  {"splitLine": {"show": False}, "axisLabel": {"show": False},
                        "axisTick": {"show": False}, "axisLine": {"show": False}},
                "series": [
                    {"name": "Completados", "type": "pictorialBar",
                    "label": {"show": True, "position": "right", "offset": [10,0],
                            "textStyle": {"fontSize": 19, "fontWeight": "bold"}},
                    "symbolRepeat": True, "symbolSize": ["80%","60%"],
                    "barCategoryGap": "40%", "itemStyle": {"color": "#5f8fe2"},
                    "data": [{"value": eng,     "symbol": video_icon}]},
                    {"name": "No completados", "type": "pictorialBar",
                    "barGap": "10%",
                    "label": {"show": True, "position": "right", "offset": [10,0],
                            "textStyle": {"fontSize": 19, "fontWeight": "bold"}},
                    "symbolRepeat": True, "symbolSize": ["80%","60%"],
                    "itemStyle": {"color": "#172338"},
                    "data": [{"value": non_eng, "symbol": x_icon}]},
                ],
            }
            st_echarts(pic_opt, height="250px", key="pictorial_bar", theme="streamlit")

with col_der:
    with st.container(border=True):
        st.markdown("**TOP 10 TEMAS POR NIVEL DE CONOCIMIENTO**")
        if not df_kcs.empty:
            top10 = df_kcs.nlargest(10, "mean").sort_values("mean").reset_index(drop=True)
            bar_opt = {
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"},
                            "formatter": "{b}: {c}%", **TOOLTIP_AJUSTE},
                "grid": {"top":"0%","left":"3%","right":"18%","bottom":"8%","containLabel":True},
                "xAxis": {"interval":25,"type":"value","name":"Conocimiento (%)","nameLocation":"middle","nameGap":34,"min":0,"max":100,
                          "nameTextStyle":{"color":"#64748b","fontSize":16},
                          "axisLabel":{"color":"#64748b","fontSize":15,
                                       "formatter":"{value}%"},
                          "splitLine":{"show": False}},
                "yAxis": {"type":"category",
                          "data":[str(n) for n in top10["topic_name"].tolist()],
                          "axisLabel":{"color":"#475569","fontSize":16,
                                       "formatter":JsCode("function(v){return v.length>24?v.slice(0,23)+'\u2026':v;}")},
                          "axisLine":{"show":False},"axisTick":{"show":False}},
                "series": [{"type":"bar",
                            "data":[{"value": pct_conocimiento(r["mean"]),
                                     "itemStyle":{"color": "#3b598d",
                                                  "borderRadius":[0,4,4,0]}}
                                    for i,(_, r) in enumerate(top10.iterrows())],
                            "label":{"show":True,"position":"right",
                                     "formatter":"{c}%","color":"#475569","fontSize":16},
                            "barMaxWidth":22}],
            }
            st_echarts(bar_opt, height="500px", key="bar_top10", theme="streamlit")
        else:
            st.info("No hay temas disponibles.")
