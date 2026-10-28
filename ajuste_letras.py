"""
Ajustes de tipografía compartidos por todas las páginas.

El tamaño base de las letras se sube desde ``.streamlit/config.toml``
(``baseFontSize``): eso agranda todo lo que Streamlit dibuja solo (párrafos,
métricas, tablas, botones, menú lateral, ventana de corrección). Este módulo
cubre lo que esa opción no alcanza:

    - Las etiquetas de las métricas («NIVEL DE CONOCIMIENTO») se parten en dos
      líneas en lugar de cortarse con «…».
    - Las etiquetas de colores (Alto, Seguro…) y los textos chicos del menú
      lateral, que tienen un tamaño fijo en el código.

Uso: llamar ``aplicar_letras_grandes()`` justo después de ``inject_css()``.
"""
import streamlit as st

_CSS = """
<style>
/* Etiquetas de las métricas: se parten en líneas en vez de cortarse con «…» */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    line-height: 1.25 !important;
}
[data-testid="stMetricLabel"] p {
    font-size: .86rem !important;
    word-break: normal !important;
    overflow-wrap: normal !important;
}
/* Etiquetas de colores (Alto, Medio, Seguro…) */
.badge { font-size: .86rem !important; padding: .2rem .6rem !important; }
/* Menú lateral: textos que traen un tamaño fijo en el código */
[data-testid="stSidebar"] .sidebar-uid { font-size: 1.2rem !important; }
[data-testid="stSidebar"] [style*="font-size:.73rem"] { font-size: .95rem !important; }
[data-testid="stSidebar"] [style*="font-size:.65rem"] { font-size: .85rem !important; opacity: .75 !important; }
[data-testid="stSidebar"] .btn-cambiar .stButton > button { font-size: 1rem !important; }
</style>
"""


def aplicar_letras_grandes():
    """Inyecta el CSS de tipografía. Se llama una vez por página."""
    st.markdown(_CSS, unsafe_allow_html=True)
