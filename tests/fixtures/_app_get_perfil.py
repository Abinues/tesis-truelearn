"""
Script mínimo de Streamlit que llama a ``shared.get_perfil``, para probarlo con AppTest.

Pide el perfil del usuario ``uid_a_pedir`` (1 por defecto) y guarda el
resultado en ``st.session_state["resultado"]``. En las ramas de error,
``st.stop()`` corta la ejecución antes de guardarlo.
"""
import streamlit as st

import shared

uid = st.session_state.get("uid_a_pedir", 1)
resultado = shared.get_perfil(uid)
st.session_state["resultado"] = resultado
