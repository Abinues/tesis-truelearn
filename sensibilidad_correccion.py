"""
sensibilidad_correccion.py
─────────────────────────────────────────────────────────────────────────────
Muestra cuánto cambia el efecto de una corrección explícita según el único
parámetro de diseño, EVENTOS_EQUIVALENTES (cuántos eventos reales de
precisión pesa la palabra directa del estudiante), para un tema típico:

  conocimiento: media 0.0, varianza 0.433 (mediana real tras 1 evento)
  interés:      media 1.0, varianza 0.405

Usa la misma regla que logic/api_backend.py (observación bayesiana con el
ancla en el extremo de la escala calibrada). No lee datos ni modifica nada.

Uso (raíz del proyecto, venv activado):
    python sensibilidad_correccion.py [analisis_sensibilidad]
"""
import csv
import os
import sys

from logic.api_backend import PRECISION_POR_EVENTO
from logic.dominio import (INT_MAX, INT_MIN, KNOW_MAX, KNOW_MIN,
                           pct_certeza, pct_conocimiento, pct_interes)

CASOS = [
    # nombre, variable, media inicial, varianza inicial, ancla
    ("sé más",         "conocimiento", 0.0, 0.433, KNOW_MAX),
    ("sé menos",       "conocimiento", 0.0, 0.433, KNOW_MIN),
    ("me interesa",    "interes",      1.0, 0.405, INT_MAX),
    ("no me interesa", "interes",      1.0, 0.405, INT_MIN),
]


def efecto(media, var, ancla, variable, k):
    p = k * PRECISION_POR_EVENTO[variable]
    var_n = 1.0 / (1.0 / var + p)
    media_n = media + p * var_n * (ancla - media)
    return media_n, var_n


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "analisis_sensibilidad"
    os.makedirs(out_dir, exist_ok=True)
    filas = []
    for nombre, variable, m0, v0, ancla in CASOS:
        a_pct = pct_conocimiento if variable == "conocimiento" else pct_interes
        for k in (1, 2, 4, 8):
            m1, v1 = efecto(m0, v0, ancla, variable, k)
            filas.append({
                "corrección": nombre, "eventos equivalentes": k,
                "Δ media": round(m1 - m0, 3), "Δ puntos de escala (%)": round(a_pct(m1) - a_pct(m0), 1),
                "varianza antes": v0, "varianza después": round(v1, 3),
                "confianza antes (%)": pct_certeza(v0), "confianza después (%)": pct_certeza(v1),
            })
    for f in filas:
        marca = "  <- valor adoptado" if f["eventos equivalentes"] == 4 else ""
        print(f"{f['corrección']:<15} k={f['eventos equivalentes']}  Δmedia {f['Δ media']:+.3f} "
              f"({f['Δ puntos de escala (%)']:+.1f} pts)  conf {f['confianza antes (%)']}→{f['confianza después (%)']}{marca}")
    ruta = os.path.join(out_dir, "sensibilidad_correccion.csv")
    with open(ruta, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"\nGuardado en {ruta}")


if __name__ == "__main__":
    main()
