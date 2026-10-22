"""
Script mínimo de Streamlit que abre el popup de corrección, para probarlo con AppTest.

Toma el tema (``kc_data``) y el usuario (``uid``) de ``st.session_state``; si
faltan, usa un tema de ejemplo y el usuario 7.
"""
import streamlit as st

from shared import popup_editar

kc_data = st.session_state.get("kc_data", {
    "topic_id": 1, "topic_name": "Álgebra Lineal", "topic_url": "",
    "mean": 0.1, "variance": 0.3, "interest_mean": 0.4,
})
uid = st.session_state.get("uid", 7)
popup_editar(kc_data, uid)
