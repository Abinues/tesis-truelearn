"""
Página «Análisis de sesiones».

Muestra cómo cambió el conocimiento estimado sesión a sesión. Debajo del
título hay un selector de vistas y un control deslizante para elegir el rango
de sesiones; la página muestra solo la vista elegida:

    - DETALLE DE SESIONES: conocimiento promedio en la primera y en la última
      sesión del rango (tarjetas AL INICIO y AL FINAL) y una tabla con el
      conocimiento promedio y los temas activos de cada sesión.
    - ¿QUÉ 5 TEMAS CAMBIARON MÁS CON EL TIEMPO?: gráfico de ríos con los 5
      temas cuyo conocimiento más cambió dentro del rango.
    - EVOLUCIÓN DEL CONOCIMIENTO PROMEDIO DEL ESTUDIANTE: línea de evolución
      del conocimiento, con marcas del mejor momento, del punto más bajo y del
      promedio.

El control deslizante es único para las tres vistas: al cambiar de vista se
conserva el rango elegido.

El conocimiento de cada sesión es el promedio del nivel de conocimiento de los
temas activos en esa sesión (los que el estudiante ya había visto hasta ese
momento), expresado en porcentaje. Se calcula igual que el «Nivel de
conocimiento» de la página Resumen: porcentaje del promedio de los ``mean``.

Estado de sesión que usa:
    uid_seleccionado: Identificador del aprendiz elegido.
    perfil_actual: Perfil del aprendiz (si falta, se pide al backend).
    sesiones_vista: Vista elegida en el selector.
    ses_rango_<uid>: Rango de sesiones elegido en el control deslizante.
    popup_open, popup_topic: Control del popup de corrección.
"""
import json

import pandas as pd
import streamlit as st
from streamlit_echarts import JsCode, st_echarts

from logic.dominio import pct_conocimiento
from ajuste_letras import aplicar_letras_grandes
from shared import GRID_COL, TEAL, get_perfil, inject_css, popup_editar

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

historial = perfil.get("session_history", [])

if st.session_state.get("popup_open") and st.session_state.get("popup_topic"):
    popup_editar(st.session_state["popup_topic"], uid)

st.title(":material/show_chart: Análisis de sesiones")

if not historial:
    st.warning("Todavía no hay historial de sesiones para este estudiante.")
    st.stop()

rows = []
for s in historial:
    means = s.get("means", {})
    promedio_sesion = pct_conocimiento(sum(means.values()) / len(means)) if means else 0.0
    rows.append({
        "Sesión":   s["session_id"],
        "promedio": promedio_sesion,
        "n_temas":  len(means),
    })
df_hist = pd.DataFrame(rows).sort_values("Sesión")

V_DETALLE, V_TEMAS, V_EVOL = (
    "DETALLE DE SESIONES",
    "¿QUÉ 5 TEMAS CAMBIARON MÁS CON EL TIEMPO?",
    "EVOLUCIÓN DEL CONOCIMIENTO PROMEDIO DEL ESTUDIANTE",
)
vista = st.pills(
    "Vista",
    [V_EVOL, V_TEMAS, V_DETALLE],
    default=V_EVOL,
    key="sesiones_vista",
    label_visibility="collapsed",
) or V_EVOL

# ── Control deslizante: qué sesiones se muestran (vale para las tres vistas) ──
ids = df_hist["Sesión"].tolist()
if len(ids) >= 2:
    ses_ini, ses_fin = st.select_slider(
        "Sesiones a mostrar:",
        options=ids,
        value=(ids[0], ids[-1]),
        key=f"ses_rango_{uid}",
    )
else:
    ses_ini = ses_fin = ids[0]

df_p     = df_hist[(df_hist["Sesión"] >= ses_ini) & (df_hist["Sesión"] <= ses_fin)]
hist_p   = [s for s in historial if ses_ini <= s["session_id"] <= ses_fin]
st.caption(f"Mostrando {len(df_p)} de {len(df_hist)} sesiones "
           f"(de la sesión {ses_ini} a la {ses_fin}).")

first_pct = float(df_p["promedio"].iloc[0])
last_pct  = float(df_p["promedio"].iloc[-1])


# ── DETALLE DE SESIONES ─────────────────────────────────────────────────────
if vista == V_DETALLE:
    col_tarjetas, col_tabla = st.columns([1, 2.4], gap="small")

    with col_tarjetas:
        st.metric("AL INICIO", f"{first_pct:.1f}%", border=True, height=185,
                  help="Nivel de conocimiento promedio del estudiante en la primera sesión del rango elegido, calculado sobre los temas activos en ese momento.")
        st.metric("AL FINAL", f"{last_pct:.1f}%",
                  delta=f"{last_pct - first_pct:+.1f}%", border=True, height=185,
                  delta_color="blue",
                  help="Nivel de conocimiento promedio del estudiante en la última sesión del rango elegido, calculado sobre los temas activos en ese momento. La flecha muestra cuánto cambió desde la primera sesión del rango.")

    with col_tabla:
        df_show = df_p[["Sesión", "promedio", "n_temas"]].rename(
            columns={"promedio": "Conocimiento promedio", "n_temas": "Temas activos"})
        st.dataframe(
            df_show,
            width="stretch", hide_index=True, height=396,
            column_config={
                "Conocimiento promedio": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )


# ── ¿QUÉ 5 TEMAS CAMBIARON MÁS CON EL TIEMPO? ───────────────────────────────
elif vista == V_TEMAS:
    with st.container(border=True):
        temas_var = {}
        for s in hist_p:
            for tema, val in s.get("means", {}).items():
                temas_var.setdefault(tema, []).append(val)
        variaciones = {t: max(v)-min(v) for t,v in temas_var.items() if len(v)>1}
        top5 = sorted(variaciones, key=variaciones.get, reverse=True)[:5]

        # Nombre corto para dibujar en el gráfico (único aunque dos temas
        # empiecen igual) y nombre completo para el tooltip.
        corto = {}
        for tema in top5:
            nombre, n = tema[:22], 2
            while nombre in corto.values():
                nombre, n = f"{tema[:19]} ({n})", n + 1
            corto[tema] = nombre
        completo = {v: k for k, v in corto.items()}

        river_data = []
        for s in hist_p:
            tid = s["session_id"]
            for tema in top5:
                raw = s.get("means",{}).get(tema, 0)
                pct = max(pct_conocimiento(raw), 0.1)
                river_data.append([tid, pct, corto[tema]])

        if river_data:
            river_opt = {
                "tooltip": {"trigger":"axis","axisPointer":{"type":"line","lineStyle":{"color":"rgba(0,0,0,0.2)","width":1}},
                            **TOOLTIP_AJUSTE,
                            "formatter": JsCode(
                                "function(params){"
                                f"  var completo = {json.dumps(completo, ensure_ascii=False)};"
                                "  var out = 'Sesión ' + params[0].value[0] + '<br/>';"
                                "  params.forEach(function(p){"
                                "    out += (completo[p.value[2]] || p.value[2]) + ': ' + p.value[1].toFixed(1) + '%<br/>';"
                                "  });"
                                "  return out;"
                                "}"
                            )},
                "legend":  {"data":[corto[t] for t in top5],"bottom":0,"type":"scroll","textStyle":{"fontSize":15}},
                "singleAxis": {"top":40,"right":80,"bottom":80,"axisTick":{},"axisLabel":{"fontSize":15},"nameTextStyle":{"fontSize":15},
                            "type":"value","name":"Sesión","min":int(ses_ini),"max":int(ses_fin),
                            "axisPointer":{"animation":True,"label":{"show":True}},
                            "splitLine":{"show":True,"lineStyle":{"type":"dashed","opacity":0.2}}},
                "series": [{
                    "type": "themeRiver",
                    "label": {"fontSize": 15},
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
            st_echarts(river_opt, height="380px", key="themeriver", theme="streamlit")
        else:
            st.info("Elegí un rango de al menos 2 sesiones para ver qué temas cambiaron más.")


# ── EVOLUCIÓN DEL CONOCIMIENTO PROMEDIO DEL ESTUDIANTE ──────────────────────
else:
    with st.container(border=True):
        vals_norm = df_p["promedio"].tolist()

        line_opt = {
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"},
                        "backgroundColor": "#0f172a", "borderColor": "#0f172a",
                        "textStyle": {"color": "#f8fafc", "fontSize": 16},
                        "formatter": "Sesión {b0}<br/>Conocimiento promedio: {c0}%"},
            "grid": {"left":"3%","right":"8%","bottom":"22%","top":"12%","containLabel":True},
            "xAxis": {"type":"category","data":df_p["Sesión"].tolist(),
                    "name":"Número de sesión","nameLocation":"middle","nameGap":38,
                    "nameTextStyle":{"color":"#64748b","fontSize":15},
                    "axisLine":{"lineStyle":{"color":GRID_COL}},
                    "axisLabel":{"color":"#64748b","fontSize":15},
                    "splitLine":{"show":False}},
            "yAxis": {"type":"value","name":"Conocimiento promedio (%)","min":0,"max":100,
                    "nameLocation":"middle","nameGap":54,
                    "nameTextStyle":{"color":"#64748b","fontSize":15},
                    "axisLine":{"show":False},"axisTick":{"show":False},
                    "axisLabel":{"color":"#64748b","fontSize":15,"formatter":"{value}%"},
                    "splitLine":{"show":False}},
            "series": [{"name":"Conocimiento promedio (%)","data":vals_norm,
                        "type":"line","smooth":True,"symbol":"circle","symbolSize":7,
                        "lineStyle":{"color":TEAL,"width":2.5},
                        "itemStyle":{"color":TEAL,"borderColor":"#fff","borderWidth":2},
                        "areaStyle":{"color":{"type":"linear","x":0,"y":0,"x2":0,"y2":1,
                                            "colorStops":[{"offset":0,"color":"rgba(14,165,233,0.25)"},
                                                          {"offset":1,"color":"rgba(14,165,233,0.02)"}]}},
                        "markPoint":{"symbolSize":64,"label":{"fontSize":15},"data":[{"type":"max","name":"Mejor momento","itemStyle":{"color":"#5f8fe2"}},
                                            {"type":"min","name":"Punto bajo","itemStyle":{"color":"#172338"}}]},
                        "markLine":{"label":{"fontSize":15},"data":[{"type":"average","name":"Promedio"}],
                                    "lineStyle":{"color":"#808080","type":"dashed"}}}],
        }
        st_echarts(line_opt, height="480px", key="evol_line", theme="streamlit", renderer="svg")
