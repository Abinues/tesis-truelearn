"""
Página «¿Cómo avanzó el aprendizaje con el tiempo?» (Evolución del aprendizaje).

Muestra cómo cambió el conocimiento estimado sesión a sesión:

    - Conocimiento promedio en la primera y en la última sesión.
    - Gráfico de ríos con los 5 temas cuyo conocimiento más cambió.
    - Línea de evolución del conocimiento, con filtro por cantidad de
      sesiones, y marcas del mejor momento, del punto más bajo y del promedio.
    - Tabla desplegable con el conocimiento promedio y los temas activos de
      cada sesión.

El conocimiento de cada sesión es el promedio del nivel de conocimiento de los
temas activos en esa sesión (los que el estudiante ya había visto hasta ese
momento), expresado en porcentaje. Se calcula igual que el «Nivel de
conocimiento» de la página Resumen: porcentaje del promedio de los ``mean``.

Estado de sesión que usa:
    uid_seleccionado: Identificador del aprendiz elegido.
    perfil_actual: Perfil del aprendiz (si falta, se pide al backend).
    popup_open, popup_topic: Control del popup de corrección.
"""
import pandas as pd
import streamlit as st
from streamlit_echarts import JsCode, st_echarts

from logic.dominio import pct_certeza, pct_conocimiento
from shared import GRID_COL, TEAL, get_perfil, inject_css, popup_editar

inject_css()

uid    = st.session_state.get("uid_seleccionado")
perfil = st.session_state.get("perfil_actual") or get_perfil(uid)
if not perfil or not uid:
    st.info("Seleccioná un aprendiz para comenzar.")
    st.stop()

kcs       = perfil.get("knowledge_components", [])
historial = perfil.get("session_history", [])
df_kcs    = pd.DataFrame(kcs) if kcs else pd.DataFrame()

if st.session_state.get("popup_open") and st.session_state.get("popup_topic"):
    popup_editar(st.session_state["popup_topic"], uid)

st.title(":material/show_chart: ¿Cómo avanzó el aprendizaje con el tiempo?")
st.markdown(
    "Acá podés ver cómo fue creciendo el conocimiento del estudiante sesión a sesión, "
    "qué temas ganaron o perdieron protagonismo, y cómo el sistema fue ganando confianza "
    "en sus estimaciones a medida que el estudiante interactuó más."
)

if not historial:
    st.warning("Todavía no hay historial de sesiones para este estudiante.")
    st.stop()

rows = []
for s in historial:
    means = s.get("means", {})
    vars_ = s.get("variances", {})
    promedio_sesion = pct_conocimiento(sum(means.values()) / len(means)) if means else 0.0
    rows.append({
        "Sesión":     s["session_id"],
        "timestamp":  s.get("timestamp", 0),
        "promedio":   promedio_sesion,
        "n_temas":    len(means),
        "mean_var":   (sum(vars_.values()) / len(vars_)) if vars_ else 0.5,
        "_means":     means,
    })
df_hist = pd.DataFrame(rows).sort_values("Sesión")

first_pct  = float(df_hist["promedio"].iloc[0])
last_pct   = float(df_hist["promedio"].iloc[-1])
mean_var_f = df_kcs["variance"].mean() if not df_kcs.empty else 0.5
cert_pct   = pct_certeza(mean_var_f)

col_sup_izq, col_sup_der = st.columns([1, 8], gap="small")

with col_sup_izq:
    st.metric("AL INICIO", f"{first_pct:.1f}%",border=True,width=190,height=155,
                help="Nivel de conocimiento promedio del estudiante en su primera sesión registrada, calculado sobre los temas activos en ese momento.")
    
    st.metric("AL FINAL", f"{last_pct:.1f}%",
                delta=f"{last_pct - first_pct:+.1f}%",border=True,width=190,height=155,
                delta_color="blue",
                help="Nivel de conocimiento promedio del estudiante en su sesión más reciente, calculado sobre los temas activos en ese momento. La flecha muestra cuánto cambió desde la primera sesión.")

with col_sup_der:
    with st.container(border=True):
        st.markdown("**¿QUÉ 5 TEMAS CAMBIARON MÁS CON EL TIEMPO?**")

        temas_var = {}
        for s in historial:
            for tema, val in s.get("means", {}).items():
                temas_var.setdefault(tema, []).append(val)
        variaciones = {t: max(v)-min(v) for t,v in temas_var.items() if len(v)>1}
        top5 = sorted(variaciones, key=variaciones.get, reverse=True)[:5]

        river_data = []
        for s in historial:
            tid = s["session_id"]
            for tema in top5:
                raw = s.get("means",{}).get(tema, 0)
                pct = max(pct_conocimiento(raw), 0.1)
                river_data.append([tid, pct, tema[:22]])

        if river_data:
            river_opt = {
                "tooltip": {"trigger":"axis","axisPointer":{"type":"line","lineStyle":{"color":"rgba(0,0,0,0.2)","width":1}},
                            "formatter": JsCode(
                                "function(params){"
                                "  var out = 'Sesión ' + params[0].value[0] + '<br/>';"
                                "  params.forEach(function(p){"
                                "    out += p.value[2] + ': ' + p.value[1].toFixed(1) + '%<br/>';"
                                "  });"
                                "  return out;"
                                "}"
                            )},
                "legend":  {"data":[t[:22] for t in top5],"bottom":0,"type":"scroll","textStyle":{"fontSize":10}},
                "singleAxis": {"top":40,"bottom":60,"axisTick":{},"axisLabel":{},
                            "type":"value","name":"Sesión",
                            "axisPointer":{"animation":True,"label":{"show":True}},
                            "splitLine":{"show":True,"lineStyle":{"type":"dashed","opacity":0.2}}},
                "series": [{
                    "type": "themeRiver",
                    "color": [
                        "#3b598d",
                        "#8d3b59",
                        "#598d3b",
                        "#7f3b8d",
                        "#8d6f3b"
                    ],
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 20,
                            "shadowColor": "rgba(0,0,0,0.8)"
                        }
                    },
                    "data": river_data
                }],
            }
            st_echarts(river_opt, height="250px", key="themeriver", theme="streamlit")
        else:
            st.info("No hay suficientes datos para mostrar este gráfico.")


col_inf_izq, col_inf_der = st.columns([2.5, 1], gap="small")

with col_inf_izq:
    with st.container(border=True):
        st.markdown("**EVOLUCIÓN DEL CONOCIMIENTO PROMEDIO DEL ESTUDIANTE**")

        opts_s = ["Todas las sesiones"]
        if len(df_hist) >= 5:
            opts_s.append("Últimas 5")
        if len(df_hist) >= 10:
            opts_s.append("Últimas 10")
        if len(df_hist) >= 25:
            opts_s.append("Últimas 25")

        sel = st.radio(
                "Mostrar",
                opts_s,
                horizontal=True,
                key="ses_lim_ev",
                label_visibility="collapsed"
            )

        df_p = df_hist.copy()
        limite = {"Últimas 5": 5, "Últimas 10": 10, "Últimas 25": 25}.get(sel)
        if limite:
            df_p = df_p.tail(limite)

        vals_norm = df_p["promedio"].tolist()

        line_opt = {
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"},
                        "backgroundColor": "#0f172a", "borderColor": "#0f172a",
                        "textStyle": {"color": "#f8fafc", "fontSize": 12},
                        "formatter": "Sesión {b0}<br/>Conocimiento promedio: {c0}%"},
            "grid": {"left":"3%","right":"3%","bottom":"12%","top":"10%","containLabel":True},
            "xAxis": {"type":"category","data":df_p["Sesión"].tolist(),
                    "name":"Número de sesión","nameLocation":"middle","nameGap":28,
                    "nameTextStyle":{"color":"#64748b","fontSize":11},
                    "axisLine":{"lineStyle":{"color":GRID_COL}},
                    "axisLabel":{"color":"#64748b","fontSize":11},
                    "splitLine":{"show":False}},
            "yAxis": {"type":"value","name":"Conocimiento promedio (%)","min":0,"max":100,
                    "nameLocation":"middle","nameGap":40,
                    "nameTextStyle":{"color":"#64748b","fontSize":11},
                    "axisLine":{"show":False},"axisTick":{"show":False},
                    "axisLabel":{"color":"#64748b","fontSize":11,"formatter":"{value}%"},
                    "splitLine":{"show":False}},
            "series": [{"name":"Conocimiento promedio (%)","data":vals_norm,
                        "type":"line","smooth":True,"symbol":"circle","symbolSize":7,
                        "lineStyle":{"color":TEAL,"width":2.5},
                        "itemStyle":{"color":TEAL,"borderColor":"#fff","borderWidth":2},
                        "areaStyle":{"color":{"type":"linear","x":0,"y":0,"x2":0,"y2":1,
                                            "colorStops":[{"offset":0,"color":"rgba(14,165,233,0.25)"},
                                                          {"offset":1,"color":"rgba(14,165,233,0.02)"}]}},
                        "markPoint":{"data":[{"type":"max","name":"Mejor momento","itemStyle":{"color":"#5f8fe2"}},
                                            {"type":"min","name":"Punto bajo","itemStyle":{"color":"#172338"}}]},
                        "markLine":{"data":[{"type":"average","name":"Promedio"}],
                                    "lineStyle":{"color":"#808080","type":"dashed"}}}],
        }
        st_echarts(line_opt, height="200px", key="evol_line", theme="streamlit", renderer="svg")

with col_inf_der:
    with st.expander("VER DATOS DETALLADOS DE SESIONES", icon=":material/table_view:"):
        df_show = df_hist[["Sesión","promedio","n_temas"]].copy()
        df_show["Conocimiento Promedio"] = df_show["promedio"].round(1).astype(str) + "%"
        df_show["Temas activos"] = df_show["n_temas"]
        st.dataframe(df_show[["Sesión","Conocimiento Promedio","Temas activos"]],
                    use_container_width=True, hide_index=True,height=255)
