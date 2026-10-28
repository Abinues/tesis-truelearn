"""
Página «¿Cómo decide el sistema qué recomendar?» (Pesos del INKClassifier).

Muestra los tres pesos que el INKClassifier aprendió para el aprendiz y que
combina para predecir si completará un video:

    - Nivel del video (novedad), interés previo y actividad general (sesgo),
      cada uno con su nivel de seguridad.
    - Gráfico de anillo con la proporción en que pesa cada factor.
    - Diagrama de cajas con la confianza del sistema por cuartil de
      conocimiento de los temas.

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
    pct_conocimiento,
    pct_peso_actividad,
    pct_peso_interes,
    pct_peso_novelty,
)
from ajuste_letras import aplicar_letras_grandes
from shared import TEAL, get_perfil, inject_css, popup_editar

inject_css()
aplicar_letras_grandes()

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

col_izq, col_der = st.columns([1.3, 2.8], gap="small")

with col_izq:
    c1, c2 = st.columns(2, gap="small")
    with c1:
        c1.metric("NIVEL DEL VIDEO",   f"{nov_pct:.1f}%",  delta=f"{nov_pct-PCT_INICIO_PESO_NOVELTY:+.1f}%",  border=True,height=160,delta_color="yellow",
                help="Qué tanto pesa, para este estudiante, que el video coincida con su nivel a la hora de predecir si lo va a completar. Ver '¿Qué significan estos factores?' arriba.")
    with c2:
        c2.metric("SEGURIDAD",   f"{nov_cert:.1f}%", delta=f"{nov_cert:+.1f}%",    border=True,height=160,delta_color="yellow",
                help="Qué tan firme está el sistema de la estimación de al lado ('Nivel del video'). Con pocas interacciones es una suposición; con más, se vuelve más confiable.")

    c3, c4 = st.columns(2, gap="small")
    with c3:
        c3.metric("INTERÉS PREVIO",   f"{int_pct:.1f}%",  delta=f"{int_pct-PCT_INICIO_PESO_INTERES:+.1f}%",  border=True,height=160, delta_color="orange",
                help="Qué tanto pesa, para este estudiante, el interés que ya tenía en el tema antes de ver el video. Ver '¿Qué significan estos factores?' arriba.")
    with c4:
        c4.metric("SEGURIDAD", f"{int_cert:.1f}%", delta=f"{int_cert:+.1f}%",    border=True,height=160, delta_color="orange",
                help="Qué tan firme está el sistema de la estimación de al lado ('Interés previo'). Con pocas interacciones es una suposición; con más, se vuelve más confiable.")

    c5, c6 = st.columns(2, gap="small")
    with c5:
        c5.metric("ACTIVIDAD",         f"{bias_pct:.1f}%", delta=f"{bias_pct-PCT_INICIO_PESO_ACTIVIDAD:+.1f}%", border=True,height=160,delta_color="violet",
                help="Qué tanto pesa la tendencia general de este estudiante a completar videos, más allá del tema puntual. Ver '¿Qué significan estos factores?' arriba.")
    with c6:
        c6.metric("SEGURIDAD",  f"{bias_cert:.1f}%",delta=f"{bias_cert:+.1f}%",   border=True,height=160,delta_color="violet",
                help="Qué tan firme está el sistema de la estimación de al lado ('Actividad'). Con pocas interacciones es una suposición; con más, se vuelve más confiable.")


with col_der:
        with st.container(border=True):
            st.markdown("**¿QUÉ FACTOR DOMINA LAS RECOMENDACIONES?**")
            abs_nov  = abs(nov_m)
            abs_int  = abs(int_m)
            abs_bias = abs(bias_m+0.5)
            total_w  = abs_nov + abs_int + abs_bias or 1
            pie_opt = {
                "tooltip": {"trigger":"item",
                            "formatter": "{b}<br/>Explica el {d}% del comportamiento"},
                "legend":  {"orient":"horizontal","bottom":0,"textStyle":{"color":"#475569","fontSize":15}},
                "series":  [{"name":"Factor dominante","type":"pie","radius":["28%","46%"],"center":["46%","42%"],
                            "avoidLabelOverlap":True,
                            "itemStyle":{"borderRadius":8,"borderColor":"#fff","borderWidth":2},
                            "label":{"show":True,"formatter":"{d}%","fontSize":16,"overflow":"none","color":"#374151"},
                            "emphasis":{"label":{"show":True,"fontSize":18,"fontWeight":"bold"},
                                        "itemStyle":{"shadowBlur":10,"shadowColor":"rgba(0,0,0,.2)"}},
                            "data":[
                                {"name":"Nivel del video","value":round(abs_nov/total_w*100,1),"itemStyle":{"color":"#faca2b"}},
                                {"name":"Interés previo", "value":round(abs_int/total_w*100,1),"itemStyle":{"color":"#ffa421"}},
                                {"name":"Actividad",       "value":round(abs_bias/total_w*100,1),"itemStyle":{"color":"#803df5"}},
                            ]}],
            }
            st_echarts(pie_opt, height="430px", key="pie_pesos", theme="streamlit")
   
       