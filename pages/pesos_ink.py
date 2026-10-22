"""
Página «¿Cómo decide el sistema qué recomendar?» (Pesos del INKClassifier).

Muestra los tres pesos que el INKClassifier aprendió para el aprendiz y que
combina para predecir si completará un video:

    - Nivel del video (novedad), interés previo y actividad general (sesgo),
      cada uno con su nivel de seguridad.
    - Gráfico de anillo con la proporción en que pesa cada factor.
    - Diagrama de cajas con la confianza del sistema por cuartil de
      conocimiento de los temas.
    - Red de temas del estudiante, coloreada por nivel de dominio.

Estado de sesión que usa:
    uid_seleccionado: Identificador del aprendiz elegido.
    perfil_actual: Perfil del aprendiz (si falta, se pide al backend).
    popup_open, popup_topic: Control del popup de corrección.
"""
import pandas as pd
import streamlit as st
from streamlit_echarts import JsCode, st_echarts

from logic.dominio import (
    PCT_INICIO_PESO_ACTIVIDAD,
    PCT_INICIO_PESO_INTERES,
    PCT_INICIO_PESO_NOVELTY,
    categoria_dominio,
    pct_conocimiento,
    pct_peso_actividad,
    pct_peso_interes,
    pct_peso_novelty,
)
from shared import TEAL, get_perfil, inject_css, popup_editar

inject_css()

uid    = st.session_state.get("uid_seleccionado")
perfil = st.session_state.get("perfil_actual") or get_perfil(uid)
if not perfil or not uid:
    st.info("Seleccioná un aprendiz para comenzar.")
    st.stop()

if st.session_state.get("popup_open") and st.session_state.get("popup_topic"):
    popup_editar(st.session_state["popup_topic"], uid)

pesos  = perfil.get("weights", {})
nov_m  = pesos.get("novelty",  {}).get("mean",     0.5)
nov_v  = pesos.get("novelty",  {}).get("variance",  0.5)
int_m  = pesos.get("interest", {}).get("mean",     0.5)
int_v  = pesos.get("interest", {}).get("variance",  0.5)
bias_m = pesos.get("bias",     {}).get("mean",     0.0)
bias_v = pesos.get("bias",     {}).get("variance",  0.5)

nov_pct   = pct_peso_novelty(nov_m)
int_pct   = pct_peso_interes(int_m)
bias_pct  = pct_peso_actividad(bias_m)
nov_cert  = round(max(0,(1-nov_v*2))*100,1)
int_cert  = round(max(0,(1-int_v*2))*100,1)
bias_cert = round(max(0,(1-bias_v*2))*100,1)

kcs_pk  = pd.DataFrame(perfil.get("knowledge_components",[]))
hist_pk = perfil.get("session_history",[])
n_temas = len(kcs_pk)
max_pos = n_temas * 100 if n_temas else 1
spark   = [round(sum(pct_conocimiento(v) for v in s.get("means",{}).values())/max_pos*100,1)
           for s in hist_pk] or None

st.title(":material/tune: ¿Cómo decide el sistema qué recomendar?")
with st.expander("¿Qué significan estos factores?", expanded=False):
    st.markdown(f"""
Estos tres factores son los que el sistema aprendió a combinar para predecir si este estudiante va a completar un video o no. Un valor alto significa que ese factor pesa mucho en la predicción para este estudiante en particular; un valor bajo significa que pesa poco.

**Nivel del video:** qué tan importante es, para este estudiante, que el video esté al nivel de lo que ya sabe.
- **Más alto:** aprende mejor cuando el video está calibrado a su nivel.
- **Más bajo:** el nivel del video no predice bien si lo va a completar.
- **Punto de partida, antes de cualquier interacción:** {PCT_INICIO_PESO_NOVELTY:.1f}%.

**Interés previo:** qué tan importante es que el video sea de un tema que ya le interesaba de antes.
- **Más alto:** completa más videos cuando el tema ya le gusta.
- **Más bajo:** el interés previo no predice bien si va a participar.
- **Punto de partida, antes de cualquier interacción:** {PCT_INICIO_PESO_INTERES:.1f}%.

**Actividad general:** su tendencia general a completar videos, más allá del tema puntual.
- **Más alto:** estudiante que suele completar casi todo lo que empieza.
- **Más bajo:** estudiante selectivo, que abandona más seguido.
- **Punto de partida, antes de cualquier interacción:** {PCT_INICIO_PESO_ACTIVIDAD:.1f}%.

Los "Seguridad" de al lado de cada factor indican qué tan firme está el sistema de ese número — con pocas interacciones, es apenas una suposición; con más, se vuelve más confiable.
""")

col_izq, col_der = st.columns([1, 2.8], gap="small")

with col_izq:
    c1, c2 = st.columns(2, gap="small")
    with c1:
        c1.metric("NIVEL DEL VIDEO",   f"{nov_pct:.1f}%",  delta=f"{nov_pct-PCT_INICIO_PESO_NOVELTY:+.1f}%",  border=True,height=212,delta_color="yellow",
                help="Qué tanto pesa, para este estudiante, que el video coincida con su nivel a la hora de predecir si lo va a completar. Ver '¿Qué significan estos factores?' arriba.")
    with c2:
        c2.metric("SEGURIDAD",   f"{nov_cert:.1f}%", delta=f"{nov_cert:+.1f}%",    border=True,height=212,delta_color="yellow",
                help="Qué tan firme está el sistema de la estimación de al lado ('Nivel del video'). Con pocas interacciones es una suposición; con más, se vuelve más confiable.")

    c3, c4 = st.columns(2, gap="small")
    with c3:
        c3.metric("INTERÉS PREVIO",   f"{int_pct:.1f}%",  delta=f"{int_pct-PCT_INICIO_PESO_INTERES:+.1f}%",  border=True,height=212, delta_color="orange",
                help="Qué tanto pesa, para este estudiante, el interés que ya tenía en el tema antes de ver el video. Ver '¿Qué significan estos factores?' arriba.")
    with c4:
        c4.metric("SEGURIDAD", f"{int_cert:.1f}%", delta=f"{int_cert:+.1f}%",    border=True,height=212, delta_color="orange",
                help="Qué tan firme está el sistema de la estimación de al lado ('Interés previo'). Con pocas interacciones es una suposición; con más, se vuelve más confiable.")

    c5, c6 = st.columns(2, gap="small")
    with c5:
        c5.metric("ACTIVIDAD",         f"{bias_pct:.1f}%", delta=f"{bias_pct-PCT_INICIO_PESO_ACTIVIDAD:+.1f}%", border=True,height=212,delta_color="violet",
                help="Qué tanto pesa la tendencia general de este estudiante a completar videos, más allá del tema puntual. Ver '¿Qué significan estos factores?' arriba.")
    with c6:
        c6.metric("SEGURIDAD",  f"{bias_cert:.1f}%",delta=f"{bias_cert:+.1f}%",   border=True,height=212,delta_color="violet",
                help="Qué tan firme está el sistema de la estimación de al lado ('Actividad'). Con pocas interacciones es una suposición; con más, se vuelve más confiable.")


with col_der:
    c7, c8 = st.columns([1, 1], gap="small")
    with c7:
        with st.container(border=True):
            st.markdown("**¿QUÉ FACTOR DOMINA LAS RECOMENDACIONES?**")
            abs_nov  = abs(nov_m)
            abs_int  = abs(int_m)
            abs_bias = abs(bias_m+0.5)
            total_w  = abs_nov + abs_int + abs_bias or 1
            pie_opt = {
                "tooltip": {"trigger":"item",
                            "formatter": "{b}<br/>Explica el {d}% del comportamiento"},
                "legend":  {"orient":"vertical","left":"left","textStyle":{"color":"#475569","fontSize":10}},
                "series":  [{"name":"Factor dominante","type":"pie","radius":["35%","70%"],"center":["62%","50%"],
                            "avoidLabelOverlap":True,
                            "itemStyle":{"borderRadius":8,"borderColor":"#fff","borderWidth":2},
                            "label":{"show":True,"formatter":"{b}\n{d}%","fontSize":10,"color":"#374151"},
                            "emphasis":{"label":{"show":True,"fontSize":12,"fontWeight":"bold"},
                                        "itemStyle":{"shadowBlur":10,"shadowColor":"rgba(0,0,0,.2)"}},
                            "data":[
                                {"name":"Nivel del video","value":round(abs_nov/total_w*100,1),"itemStyle":{"color":"#faca2b"}},
                                {"name":"Interés previo", "value":round(abs_int/total_w*100,1),"itemStyle":{"color":"#ffa421"}},
                                {"name":"Actividad",       "value":round(abs_bias/total_w*100,1),"itemStyle":{"color":"#803df5"}},
                            ]}],
            }
            st_echarts(pie_opt, height="240px", key="pie_pesos", theme="streamlit")
    with c8:
        with st.container(border=True):
            st.markdown("**CONFIANZA DEL SISTEMA SEGÚN CUÁNTO SABE DE CADA TEMA**")
            if not kcs_pk.empty and len(kcs_pk) >= 4:
                kcs_pk["cuartil"] = pd.qcut(kcs_pk["mean"], q=4,
                                            labels=["Q1 (menos\nconocido)","Q2","Q3","Q4 (más\nconocido)"],
                                            duplicates="drop")
                grupos = ["Q1 (menos\nconocido)","Q2","Q3","Q4 (más\nconocido)"]
                box_data, labels_ok = [], []
                for g in grupos:
                    vals = sorted(((1-kcs_pk[kcs_pk["cuartil"]==g]["variance"])*100).dropna().tolist())
                    if len(vals) >= 2:
                        n = len(vals)
                        box_data.append([round(vals[0],1),round(vals[n//4],1),round(vals[n//2],1),round(vals[3*n//4],1),round(vals[-1],1)])
                        labels_ok.append(g)
                if box_data:
                    box_opt = {
                        "tooltip": {
                            "trigger":"axis",
                            "formatter": JsCode(
                                "function(params){"
                                "  var d = params[0].data;"
                                "  return params[0].name + '<br/>' +"
                                "    'Máximo: ' + d[4].toFixed(1) + '%<br/>' +"
                                "    'Q3: ' + d[3].toFixed(1) + '%<br/>' +"
                                "    'Mediana: ' + d[2].toFixed(1) + '%<br/>' +"
                                "    'Q1: ' + d[1].toFixed(1) + '%<br/>' +"
                                "    'Mínimo: ' + d[0].toFixed(1) + '%';"
                                "}"
                            ),
                        },
                        "grid":    {"left":"5%","right":"5%","bottom":"12%","top":"3%","containLabel":True},
                        "xAxis":   {
                            "type":"category","data":labels_ok,
                            "name":"Grupo de temas (de menor a mayor conocimiento)",
                            "nameLocation":"middle","nameGap":32,
                            "nameTextStyle":{"color":"#64748b","fontSize":10},
                            "axisLabel":{"color":"#64748b","fontSize":10}
                        },
                        "yAxis":   {
                            "type":"value","name":"Confianza del sistema (%)","min":0,"max":100,
                            "nameLocation":"middle","nameGap":40,
                            "nameTextStyle":{"color":"#64748b","fontSize":11},
                            "splitLine":{"lineStyle":{"color":"#f1f5f9"}},
                            "axisLabel":{"color":"#64748b","formatter":"{value}%"}
                        },
                        "series":  [{
                            "name":"Confianza (%)",
                            "type":"boxplot",
                            "data":box_data,
                            "itemStyle":{"color":"rgba(14,165,233,0.25)","borderColor":TEAL,"borderWidth":2},
                            "emphasis":{"itemStyle":{"color":"rgba(14,165,233,0.45)"}},
                        }],
                    }
                    st_echarts(box_opt, height="240px", key="boxplot_conf", theme="streamlit")
                else:
                    st.info("No hay suficientes datos para agrupar.")
            else:
                st.info("Se necesitan al menos 4 temas para mostrar este gráfico.")

    with st.container(border=True):
                st.markdown("**RED DE TEMAS DEL ESTUDIANTE**")
                if not kcs_pk.empty:
                    top_g = kcs_pk.sort_values("mean", ascending=False).reset_index(drop=True)
                    min_m = top_g["mean"].min()
                    max_m = top_g["mean"].max()
                    rng_m = max_m - min_m if max_m != min_m else 1.0
                    categories = [
                        {"name":"ALTO",     "itemStyle":{"color":"#8d3b82"}},
                        {"name":"MEDIO",    "itemStyle":{"color":"#3b8d46"}},
                        {"name":"BÁSICO",   "itemStyle":{"color":"#8d6f3b"}},
                        {"name":"MUY BAJO", "itemStyle":{"color":"#8d3b3b"}},
                    ]
                    cat_colors = {0:"#8d3b82",1:"#3b8d46",2:"#8d6f3b",3:"#8d3b3b"}
                    n_g = len(top_g)
                    nodes_g = []
                    for _, row in top_g.iterrows():
                        sz   = 12 + ((row["mean"]-min_m)/rng_m)*40
                        cidx = categoria_dominio(row["mean"])
                        nodes_g.append({
                            "id":         str(int(row["topic_id"])),
                            "name":       str(row["topic_name"])[:18],
                            "symbolSize": int(round(sz)),
                            "value":      pct_conocimiento(row["mean"]),
                            "category":   int(cidx),
                            "itemStyle":  {"color": str(cat_colors[cidx])},
                            "label":      {"show": True,"fontSize":9,"color":"black","fontWeight":"bold"},
                        })
                    links_g = []
                    for i in range(n_g-1):
                        links_g.append({"source":nodes_g[i]["id"],"target":nodes_g[i+1]["id"]})
                    for i in range(0,n_g-2,3):
                        links_g.append({"source":nodes_g[i]["id"],"target":nodes_g[min(i+2,n_g-1)]["id"]})
                    graph_opt = {
                        "tooltip": {"trigger":"item"},
                        "legend":  [{"data":["ALTO","MEDIO","BÁSICO","MUY BAJO"],"bottom":5,
                                    "textStyle":{"fontSize":10}}],
                        "animationDuration":1500,"animationEasingUpdate":"quinticInOut",
                        "series": [{"name":"Temas","type":"graph","layout":"force",
                                    "data":nodes_g,"links":links_g,"categories":categories,
                                    "roam":True,"draggable":True,
                                    "force":{"repulsion":150,"gravity":0.08,"edgeLength":[60,140]},
                                    "label":{"position":"right","formatter":"{b}","fontSize":9},
                                    "lineStyle":{"color":"source","curveness":0.3,"opacity":0.3},
                                    "emphasis":{"focus":"adjacency","lineStyle":{"width":5}}}],
                    }
                    st_echarts(graph_opt, height="270px", key="graph_pesos", theme="streamlit")
                else:
                    st.info("No hay temas disponibles.")
