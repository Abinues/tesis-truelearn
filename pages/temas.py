"""
Página «Análisis de temas».

Exploración completa de los temas del aprendiz: dónde se ubica cada uno,
cuáles le cuestan y cómo se distribuyen por nivel. Debajo del título hay un
selector de vistas y la página muestra solo la elegida:

    - TOP 5 TEMAS POR TIPO DE CONOCIMIENTO: sunburst de temas agrupados por
      nivel de conocimiento (Alto, Medio y Básico).
    - CONOCIMIENTO POR INTERÉS: gráfico de burbujas con el conocimiento y el
      interés de cada tema. Al hacer clic en una burbuja se muestra el
      detalle del tema, con la confianza del sistema en cada estimación.
    - RED DE TEMAS DEL ESTUDIANTE: red de temas coloreada por nivel de
      dominio.
    - TODOS LOS TEMAS: tarjetas con todos los temas (conocimiento e interés,
      cada uno con su confianza), con búsqueda por nombre, filtros por la
      etiqueta de conocimiento y por la de interés, y botón para abrir el
      popup de corrección.

Estado de sesión que usa:
    uid_seleccionado: Identificador del aprendiz elegido.
    perfil_actual: Perfil del aprendiz (si falta, se pide al backend).
    temas_vista: Vista elegida en el selector.
    popup_open, popup_topic: Control del popup de corrección.
    last_busq: Última búsqueda y filtros, para cerrar el popup cuando cambian.
    mapa_filtro_conocimiento, mapa_filtro_interes: Etiquetas elegidas en los
        filtros de «Todos los temas».
"""
import html
import inspect

import pandas as pd
import streamlit as st
from streamlit_echarts import JsCode, st_echarts

from logic.dominio import (
    PCT_INICIO_CONOCIMIENTO,
    PCT_INICIO_INTERES,
    UMBRAL_ALTO,
    UMBRAL_BASICO,
    UMBRAL_BASICO_PCT,
    UMBRAL_MEDIO,
    categoria_dominio,
    pct_certeza,
    pct_conocimiento,
    pct_interes,
)
from ajuste_letras import aplicar_letras_grandes
from shared import TEAL, get_perfil, inject_css, interes_label, nivel_label, popup_editar

inject_css()
aplicar_letras_grandes()

uid    = st.session_state.get("uid_seleccionado")
perfil = st.session_state.get("perfil_actual") or get_perfil(uid)
if not perfil or not uid:
    st.info("Seleccioná un aprendiz.")
    st.stop()

kcs    = perfil.get("knowledge_components", [])
df_kcs = pd.DataFrame(kcs) if kcs else pd.DataFrame()

if st.session_state.get("popup_open") and st.session_state.get("popup_topic"):
    popup_editar(st.session_state["popup_topic"], uid)

st.title(":material/bubble_chart: Análisis de temas")

if df_kcs.empty:
    st.warning("No hay temas disponibles para este aprendiz.")
    st.stop()

# Los tooltips con nombres de tema largos se mantienen dentro del gráfico y
# pasan a varias líneas, en vez de salirse de la pantalla.
TOOLTIP_AJUSTE = {
    "confine": True,
    "extraCssText": "max-width:340px;white-space:normal;word-break:break-word;",
}

# Los nombres de tema se guardan completos en los gráficos (así el tooltip los
# muestra enteros); solo la etiqueta dibujada dentro del gráfico se acorta.
def _etiqueta_corta(n):
    """JsCode que acorta a ``n`` caracteres la etiqueta dibujada, sin tocar el nombre."""
    return JsCode(
        "function(p){var s=p.name;"
        f"return s.length>{n}?s.slice(0,{n - 1})+'\u2026':s;}}"
    )


TODOS = "Todos"
# «Select all» de Streamlit sobra: la opción «Todos» ya cumple esa función.
# (El parámetro existe solo en las versiones recientes de Streamlit.)
SIN_SELECT_ALL = (
    {"select_all": False}
    if "select_all" in inspect.signature(st.multiselect).parameters else {}
)
ETIQUETAS_CONOCIMIENTO = ["Alto", "Medio", "Básico", "Muy bajo"]
ETIQUETAS_INTERES      = ["Alto", "Medio", "Bajo"]


def _normalizar_filtro(key):
    """
    Mantiene coherente la opción «Todos» de un filtro múltiple.

    Se ejecuta cuando el usuario cambia el filtro. Si agregó una etiqueta
    mientras «Todos» estaba elegido, quita «Todos»; si agregó «Todos» junto
    con etiquetas, deja solo «Todos»; y si vació el filtro, vuelve a «Todos».

    Args:
        key (str): Clave del ``st.multiselect`` en ``st.session_state``.

    Returns:
        None
    """
    sel  = list(st.session_state.get(key, [TODOS]))
    prev = st.session_state.get(f"{key}_prev", [TODOS])
    if TODOS in sel and len(sel) > 1:
        sel = [s for s in sel if s != TODOS] if TODOS in prev else [TODOS]
    if not sel:
        sel = [TODOS]
    st.session_state[key] = sel
    st.session_state[f"{key}_prev"] = sel


def _aplicar_filtro(serie_etiquetas, elegidas):
    """
    Indica qué filas pasan un filtro por etiqueta.

    Args:
        serie_etiquetas (pd.Series): Etiqueta de cada tema.
        elegidas (list[str]): Etiquetas elegidas en el filtro; si incluye
            «Todos» no se filtra.

    Returns:
        pd.Series: Máscara booleana con las filas que se conservan.
    """
    if TODOS in elegidas:
        return pd.Series(True, index=serie_etiquetas.index)
    return serie_etiquetas.isin(elegidas)


V_TOP5, V_BURBUJAS, V_RED, V_TODOS = (
    "TOP 5 TEMAS POR TIPO DE CONOCIMIENTO",
    "CONOCIMIENTO POR INTERÉS",
    "RED DE TEMAS DEL ESTUDIANTE",
    "TODOS LOS TEMAS",
)
vista = st.pills(
    "Vista",
    [V_TOP5, V_BURBUJAS, V_RED, V_TODOS],
    default=V_TOP5,
    key="temas_vista",
    label_visibility="collapsed",
) or V_TOP5


# ── TOP 5 TEMAS POR TIPO DE CONOCIMIENTO ────────────────────────────────────
if vista == V_TOP5:
    with st.container(border=True):
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
                    "name": str(r["topic_name"]),
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
                "tooltip": {"trigger": "item", "formatter": "{b}: {c}%", **TOOLTIP_AJUSTE},
                "series": [{"type":"sunburst","data":sb_data,"center": ["50%", "50%"],"radius":["15%","98%"],
                            "sort":None,"emphasis":{"focus":"ancestor"},
                            "label":{"formatter":_etiqueta_corta(18)},
                            "levels":[
                                        {},
                                        {
                                            "r0":"10%", "r":"45%",
                                            "itemStyle":{"borderWidth":2},
                                            "label":{
                                                "rotate":"tangential",
                                                "fontSize":16,
                                                "color":"#fff",
                                                "fontWeight":"bold"
                                            }
                                        },
                                        {
                                            "r0":"45%", "r":"95%",
                                            "label":{
                                                "align":"right",
                                                "fontSize":14,
                                                "color":"#fff"
                                            }
                                        },
                                        {
                                            "r0":"95%", "r":"98%",
                                            "itemStyle":{"borderWidth":3},
                                            "label":{
                                                "position":"outside",
                                                "padding":3,
                                                "fontSize":13,
                                                "color":"#fff"
                                            }
                                        }
                                    ]}],
            }
            st_echarts(sb_opt, height="500px", width="100%", key="sunburst_temas", theme="streamlit")
        else:
            st.info(f"No hay temas con conocimiento ≥ {UMBRAL_BASICO_PCT:.0f}% para mostrar en este gráfico.")


# ── CONOCIMIENTO POR INTERÉS ────────────────────
elif vista == V_BURBUJAS:
    scatter_data = [{
        "value": [pct_interes(r["interest_mean"]), pct_conocimiento(r["mean"])],
        "name":  str(r["topic_name"]),
        "topic_id": int(r["topic_id"]),
    } for _, r in df_kcs.iterrows()]

    @st.fragment
    def bubble_section():
        """
        Dibuja el gráfico de burbujas y el detalle del tema seleccionado.

        Cada burbuja es un tema: su posición horizontal es el interés y la
        vertical el conocimiento. Al hacer clic en una burbuja se muestran
        las métricas del tema, con la confianza del sistema en cada una, y
        sus etiquetas. Se
        ejecuta como fragmento para que la selección no recargue toda la
        página.

        Returns:
            None
        """
        col_b, col_d = st.columns([3, 2], gap="small")
        with col_b:
            scatter_opts = {
                "animation": False,
                "tooltip":   {"trigger":"item", **TOOLTIP_AJUSTE,
                            "formatter": JsCode(
                                "function(p){ var v=p.value;"
                                "return '<b>'+p.data.name+'</b><br/>"
                                "Conocimiento: '+v[1].toFixed(1)+'%<br/>"
                                "Interés: '+v[0].toFixed(1)+'%'; }"
                            )},
                "grid": {"top":44,"bottom":"14%","left":"12%","right":44},
                "xAxis": {"name":"Interés (%)","type":"value","min":0,"max":100,
                        "nameLocation":"middle","nameGap":40,
                        "nameTextStyle":{"color":"#64748b","fontSize":15},
                        "splitLine":{"show": False},
                        "axisLabel":{"color":"#64748b","fontSize":14,"formatter":"{value}%"}},
                "yAxis": {"name":"Conocimiento (%)","type":"value","min":0,"max":100,
                        "nameLocation":"middle","nameGap":56,
                        "nameTextStyle":{"color":"#64748b","fontSize":15},
                        "splitLine":{"show": False},
                        "axisLabel":{"color":"#64748b","fontSize":14,"formatter":"{value}%"}},
                "series": [{"type":"scatter","data":scatter_data,"symbolSize":22,
                            "itemStyle":{"color":"#3b598d","opacity":0.75},
                            "emphasis":{"itemStyle":{"borderColor":"#333","borderWidth":2}},
                            "markLine":{"silent":True,
                                        "lineStyle":{"type":"dashed","color":"#cbd5e1"},
                                        "data":[
                                            {"xAxis":pct_interes(df_kcs["interest_mean"].median())},
                                            {"yAxis":pct_conocimiento(df_kcs["mean"].median())},
                                        ]}}],
            }
            result = st_echarts(options=scatter_opts, height="500px", key="bubble_scatter",
                                theme="streamlit", on_select="rerun", selection_mode="points")

        with col_d:
            indices = result["selection"].get("point_indices",[]) if result else []
            if indices:
                clicked = scatter_data[indices[0]]
                match   = df_kcs[df_kcs["topic_id"] == clicked["topic_id"]]
                if not match.empty:
                    row = match.iloc[0]
                    lbl,cls  = nivel_label(row["mean"])
                    ilbl,icls= interes_label(row["interest_mean"])
                    know_b   = pct_conocimiento(row["mean"])
                    int_b    = pct_interes(row["interest_mean"])
                    cert_k   = pct_certeza(row["variance"])
                    cert_i   = pct_certeza(row.get("interest_var", 0.5))

                    st.markdown(f"**{row['topic_name']}**")
                    card_k, card_i = st.columns(2, gap="small")
                    with card_k:
                        with st.container(border=True):
                            st.metric("CONOCIMIENTO",f"{know_b:.1f}%",delta=f"{know_b-PCT_INICIO_CONOCIMIENTO:+.1f}%",delta_color="blue",
                                      help="Qué tan bien maneja el estudiante este tema puntual, según sus interacciones con los videos relacionados.")
                            st.caption(f"El sistema está **{cert_k:.1f}%** seguro de esta estimación")
                            st.markdown(f"<span class='badge {cls}'>{lbl}</span>", unsafe_allow_html=True)
                    with card_i:
                        with st.container(border=True):
                            st.metric("INTERÉS",     f"{int_b:.1f}%", delta=f"{int_b-PCT_INICIO_INTERES:+.1f}%",delta_color="blue",
                                      help="Qué tanto interés mostró el estudiante por este tema puntual, según cómo interactuó con los videos relacionados.")
                            st.caption(f"El sistema está **{cert_i:.1f}%** seguro de esta estimación")
                            st.markdown(f"<span class='badge {icls}'>{ilbl}</span>", unsafe_allow_html=True)

            else:
                st.info("Clic en una burbuja para ver el detalle.", icon=":material/touch_app:")

    with st.container(border=True):
        bubble_section()


# ── RED DE TEMAS DEL ESTUDIANTE ─────────────────────────────────────────────
elif vista == V_RED:
    with st.container(border=True):
        top_g = df_kcs.sort_values("mean", ascending=False).reset_index(drop=True)
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
                "name":       str(row["topic_name"]),
                "symbolSize": int(round(sz)),
                "value":      pct_conocimiento(row["mean"]),
                "category":   int(cidx),
                "itemStyle":  {"color": str(cat_colors[cidx])},
                "label":      {"show": True,"fontSize":14,"color":"black","fontWeight":"bold"},
            })
        links_g = []
        for i in range(n_g-1):
            links_g.append({"source":nodes_g[i]["id"],"target":nodes_g[i+1]["id"]})
        for i in range(0,n_g-2,3):
            links_g.append({"source":nodes_g[i]["id"],"target":nodes_g[min(i+2,n_g-1)]["id"]})
        graph_opt = {
            "tooltip": {"trigger":"item", **TOOLTIP_AJUSTE},
            "legend":  [{"data":["ALTO","MEDIO","BÁSICO","MUY BAJO"],"bottom":5,
                        "textStyle":{"fontSize":15}}],
            "animationDuration":1500,"animationEasingUpdate":"quinticInOut",
            "series": [{"name":"Temas","type":"graph","layout":"force",
                        "data":nodes_g,"links":links_g,"categories":categories,
                        "roam":True,"draggable":True,
                        "force":{"repulsion":420,"gravity":0.06,"edgeLength":[90,220]},
                        "label":{"position":"right","formatter":_etiqueta_corta(18),"fontSize":14},
                        "lineStyle":{"color":"source","curveness":0.3,"opacity":0.3},
                        "emphasis":{"focus":"adjacency","lineStyle":{"width":5}}}],
        }
        st_echarts(graph_opt, height="500px", key="graph_temas", theme="streamlit")


# ── TODOS LOS TEMAS ─────────────────────────────────────────────────────────
else:
    with st.container(border=True):
        # Buscador y filtros por etiqueta, en la misma fila
        col_busq, col_fk, col_fi = st.columns([2, 1, 1], gap="small")
        with col_busq:
            busq = st.text_input(
                "Buscar tema",
                placeholder="Buscar tema...",
                key="mapa_busq"
            )
        with col_fk:
            fil_k = st.multiselect(
                "Conocimiento",
                [TODOS] + ETIQUETAS_CONOCIMIENTO,
                default=[TODOS],
                key="mapa_filtro_conocimiento",
                on_change=_normalizar_filtro,
                args=("mapa_filtro_conocimiento",),
                **SIN_SELECT_ALL,
            )
        with col_fi:
            fil_i = st.multiselect(
                "Interés",
                [TODOS] + ETIQUETAS_INTERES,
                default=[TODOS],
                key="mapa_filtro_interes",
                on_change=_normalizar_filtro,
                args=("mapa_filtro_interes",),
                **SIN_SELECT_ALL,
            )

        # Si cambia la búsqueda o un filtro, se cierra el popup
        estado_filtros = (busq, tuple(fil_k), tuple(fil_i))
        if estado_filtros != st.session_state.get("last_busq", ("", (TODOS,), (TODOS,))):
            st.session_state.update({
                "popup_open": False,
                "popup_topic": None,
                "last_busq": estado_filtros
            })

        # Filtrar temas
        df_filt = df_kcs.copy()

        if busq:
            df_filt = df_filt[
                df_filt["topic_name"].str.contains(
                    busq, case=False, regex=False, na=False
                )
            ]

        df_filt = df_filt[
            _aplicar_filtro(df_filt["mean"].apply(lambda m: nivel_label(m)[0]), fil_k)
            & _aplicar_filtro(df_filt["interest_mean"].apply(lambda m: interes_label(m)[0]), fil_i)
        ]

        # Mantener el orden original
        df_filt = df_filt.reset_index(drop=True)

        # Tarjetas de temas
        with st.container(height=400, border=True):

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

                        url_part = (
                            f'<a href="{row["topic_url"]}" target="_blank" '
                            f'style="font-size:.9rem;color:{TEAL};'
                            f'font-weight:600">Wikipedia ↗</a>'
                            if row.get("topic_url")
                            else ""
                        )

                        with cols[j]:
                            with st.container(border=True):

                                # Nombre del tema
                                st.markdown(
                                    f"<div style='font-weight:700;"
                                    f"font-size:1.05rem;color:#0f172a;"
                                    f"margin-bottom:.4rem;"
                                    f"overflow-wrap:anywhere'>"
                                    f"{html.escape(str(row['topic_name']))}</div>",
                                    unsafe_allow_html=True
                                )

                                # Dos columnas: conocimiento | interés, cada una con
                                # su valor, su confianza y su etiqueta
                                col_k, col_i = st.columns(2, gap="small")
                                for col_, lbl_, val, conf, etq, cls_ in [
                                    (
                                        col_k,
                                        "Conocimiento",
                                        pct_conocimiento(row["mean"]),
                                        pct_certeza(row["variance"]),
                                        lbl, cls,
                                    ),
                                    (
                                        col_i,
                                        "Interés",
                                        pct_interes(row["interest_mean"]),
                                        pct_certeza(row.get("interest_var", 0.5)),
                                        ilbl, icls,
                                    ),
                                ]:
                                    with col_:
                                        st.markdown(
                                            f'<div style="font-size:.74rem;color:#94a3b8;'
                                            f'text-transform:uppercase;letter-spacing:.06em">'
                                            f'{lbl_}</div>'
                                            f'<div style="font-size:1.35rem;font-weight:700;'
                                            f'color:#0f172a;line-height:1.25">{val}%</div>'
                                            f'<div style="font-size:.8rem;color:#64748b;'
                                            f'margin-bottom:.35rem">conf: {conf}%</div>'
                                            f'<div style="margin-bottom:.6rem">'
                                            f'<span class="badge {cls_}">{etq}</span></div>',
                                            unsafe_allow_html=True
                                        )

                                if url_part:
                                    st.markdown(
                                        f'<small style="display:block;'
                                        f'margin-top:.4rem">{url_part}</small>',
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
                                    popup_data["interest_var"] = float(
                                        popup_data.get("interest_var", 0.5)
                                    )

                                    st.session_state["popup_topic"] = popup_data
                                    st.session_state["popup_open"] = True

                                    st.rerun()

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )
