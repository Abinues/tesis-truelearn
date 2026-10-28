"""
Página «Historial de correcciones».

Lista en una tabla, con fecha, tema y tipo, los feedbacks
explícitos que el estudiante aplicó sobre el modelo. Si todavía no hay
ninguno, muestra un mensaje que invita a aplicar la primera corrección.

Estado de sesión que usa:
    uid_seleccionado: Identificador del aprendiz elegido.
    perfil_actual: Perfil del aprendiz.
"""
import pandas as pd
import streamlit as st

from ajuste_letras import aplicar_letras_grandes
from shared import fmt_ts, inject_css

inject_css()
aplicar_letras_grandes()

uid       = st.session_state.get("uid_seleccionado")
perfil    = st.session_state.get("perfil_actual")
if not perfil or not uid:
    st.info("Seleccioná un aprendiz en la pantalla de inicio.")
    st.stop()
feedbacks = perfil.get("explicit_feedback", [])

st.title(":material/edit_note: Historial de correcciones")
st.markdown(
    "Registro de feedbacks explícitos aplicados sobre el modelo."
)

if feedbacks:
    df_fb = pd.DataFrame(feedbacks)
    df_fb["Fecha"]     = df_fb["timestamp"].apply(fmt_ts)
    df_fb = df_fb.rename(columns={"topic_name": "Tema", "tipo": "Tipo"})
    st.dataframe(
        df_fb[["Fecha", "Tema", "Tipo"]].reset_index(drop=True),
        use_container_width=True, hide_index=True, height=500)
else:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;height:60vh;color:#94a3b8;text-align:center">
        <div style="font-size:1.3rem;font-weight:600;color:#64748b;margin-bottom:.4rem">
            Sin correcciones aún
        </div>
        <div style="font-size:1.05rem;color:#94a3b8">
            Usá el botón Editar en Análisis de temas (Todos los temas)<br>
            para aplicar tu primera corrección.
        </div>
    </div>
    """, unsafe_allow_html=True)
