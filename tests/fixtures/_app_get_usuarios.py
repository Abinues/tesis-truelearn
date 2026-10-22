"""
Script mínimo de Streamlit que llama a ``shared.get_usuarios``, para probarlo con AppTest.

Guarda el resultado en ``st.session_state["resultado"]``. En las ramas de
error, ``st.stop()`` corta la ejecución antes de guardarlo.
"""
import streamlit as st

import shared

resultado = shared.get_usuarios()
st.session_state["resultado"] = resultado
