"""
calibrar_correcciones.py
─────────────────────────────────────────────────────────────────────────────
Respalda con datos reales la MAGNITUD de las correcciones de
aplicar_correccion_directa(): la corrección es una observación bayesiana de
peso EVENTOS_EQUIVALENTES eventos (1/var' = 1/var + p, p = k × precisión por
evento; ver logic/api_backend.py).

Principio que se quiere poder defender: una corrección explícita del
estudiante pesa como evidencia, y por eso su efecto debe ser comparable
al de los eventos reales que el modelo ya procesa. Este script lo mide
con los perfiles entrenados (data_calibracion_todos/), usando session_history:

  1. Cuánto cambian la media y la varianza de un tema en cada
     actualización real (por orden de actualización: 1.ª, 2.ª, 3.ª-4.ª...).
  2. Cuántas veces la media y la varianza de un tema real se mueven tanto
     como una corrección típica («sé más» sobre un tema de varianza 0,433:
     |Δmedia| ≈ 0,145; Δvarianza ≈ 0,151; se calculan abajo con la fórmula).
  3. A cuántas actualizaciones reales "equivale" una corrección, según la
     curva mediana de varianza por número de actualizaciones: partiendo de
     la varianza mediana de un tema con n actualizaciones, se aplica la
     corrección bayesiana y se busca con qué n tiene la curva ese valor.

Solo conocimiento: session_history no guarda la media del interés.

Solo lectura. Escribe CSV y JSON en analisis_sensibilidad/.

Uso:  python calibrar_correcciones.py [data_calibracion] [analisis_sensibilidad]
"""
from __future__ import annotations

import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd

from logic.api_backend import EVENTOS_EQUIVALENTES, PRECISION_POR_EVENTO
from logic.dominio import KNOW_MAX

VAR_TIPICA = 0.433   # varianza mediana de un tema tras su 1.ª actualización (censo)


def corregir_varianza(var):
    """Varianza tras una corrección de conocimiento (misma regla que la app)."""
    return 1.0 / (1.0 / var + EVENTOS_EQUIVALENTES * PRECISION_POR_EVENTO["conocimiento"])


_P = EVENTOS_EQUIVALENTES * PRECISION_POR_EVENTO["conocimiento"]
# «sé más» sobre un tema típico (media 0, varianza VAR_TIPICA):
DELTA_KNOW = round(_P * corregir_varianza(VAR_TIPICA) * KNOW_MAX, 3)      # ≈ 0,145
REDUCCION_VAR = round(VAR_TIPICA - corregir_varianza(VAR_TIPICA), 3)      # ≈ 0,151
PRIOR_MEAN, PRIOR_VAR = 0.0, 0.5   # valores iniciales de TrueLearn (init_skill, def_var)
TRAMOS = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 10**9)]


def actualizaciones(perfil):
    """Lista de (orden, Δmedia, Δvarianza, varianza_antes) por cada actualización real."""
    previo, orden, filas = {}, {}, []
    for s in sorted(perfil.get("session_history") or [], key=lambda x: x.get("session_id", 0)):
        medias, varis = s.get("means") or {}, s.get("variances") or {}
        for tema, v in varis.items():
            m = medias.get(tema)
            if m is None:
                continue
            if tema not in previo:
                orden[tema] = 1
                filas.append((1, m - PRIOR_MEAN, PRIOR_VAR - v, PRIOR_VAR))
            elif v != previo[tema][1] or m != previo[tema][0]:
                orden[tema] += 1
                filas.append((orden[tema], m - previo[tema][0], previo[tema][1] - v, previo[tema][1]))
            previo[tema] = (m, v)
    return filas


def cargar(data_dir):
    rutas = sorted(glob.glob(os.path.join(data_dir, "visual_profile_*.json")))
    if not rutas:
        sys.exit(f"No hay visual_profile_*.json en {data_dir}")
    filas = []
    for ruta in rutas:
        with open(ruta, "r", encoding="utf-8") as f:
            filas.extend(actualizaciones(json.load(f)))
    return pd.DataFrame(filas, columns=["orden", "d_media", "d_var", "var_antes"])


def tramo(n):
    for lo, hi in TRAMOS:
        if lo <= n <= hi:
            return f"{lo}" if lo == hi else (f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}")
    return "?"


def analizar(df):
    df = df.copy()
    df["tramo"] = df["orden"].map(tramo)
    df["abs_media"] = df["d_media"].abs()
    orden_tramos = [tramo(lo) for lo, _ in TRAMOS]
    g = df.groupby("tramo")
    tabla = pd.DataFrame({
        "actualizaciones (n-ésima del tema)": orden_tramos,
    })
    agg = g.agg(n=("orden", "size"),
                abs_media_p25=("abs_media", lambda s: s.quantile(.25)),
                abs_media_mediana=("abs_media", "median"),
                abs_media_p75=("abs_media", lambda s: s.quantile(.75)),
                d_var_p25=("d_var", lambda s: s.quantile(.25)),
                d_var_mediana=("d_var", "median"),
                d_var_p75=("d_var", lambda s: s.quantile(.75)),
                pct_mueve_como_correccion_media=("abs_media", lambda s: float((s >= DELTA_KNOW).mean() * 100)),
                pct_mueve_como_correccion_var=("d_var", lambda s: float((s >= REDUCCION_VAR).mean() * 100)))
    tabla = tabla.merge(agg, left_on="actualizaciones (n-ésima del tema)", right_index=True, how="left")

    # varianza mediana DESPUÉS de n actualizaciones (curva de referencia).
    # Solo órdenes con muestra suficiente y envolvente monótona: en las colas
    # (pocos temas con 80+ actualizaciones) la mediana es ruidosa y no
    # decreciente, y daba equivalencias absurdas.
    df["var_despues"] = df["var_antes"] - df["d_var"]
    g_orden = df.groupby("orden")["var_despues"]
    curva = g_orden.median()[g_orden.size() >= 300]
    curva = curva[curva.index <= 200]
    curva = pd.Series(np.minimum.accumulate(curva.to_numpy()), index=curva.index)

    def n_equivalente(var):
        """Primera actualización en que la curva mediana llega a `var` (o supera el rango)."""
        ok = curva[curva <= var + 1e-12]
        return float(ok.index[0]) if len(ok) else float(curva.index.max())

    equiv = []
    for n in (1, 2, 4, 7, 15):
        if n not in curva.index:
            continue
        v0 = float(curva.loc[n])
        v1 = corregir_varianza(v0)
        equiv.append({
            "temas con n actualizaciones": n,
            "varianza mediana": v0,
            "varianza tras la corrección": v1,
            "equivale a n' actualizaciones": n_equivalente(v1),
            "actualizaciones extra equivalentes": n_equivalente(v1) - n,
        })
    resumen = {
        "actualizaciones analizadas": int(len(df)),
        "1.ª actualización: |Δmedia| mediana": float(df[df.orden == 1]["abs_media"].median()),
        "1.ª actualización: Δvarianza mediana": float(df[df.orden == 1]["d_var"].median()),
        "todas: |Δmedia| mediana": float(df["abs_media"].median()),
        "todas: Δvarianza mediana": float(df["d_var"].median()),
        "constantes de la corrección": {"delta_know": DELTA_KNOW, "reduccion_var": REDUCCION_VAR},
    }
    return tabla, pd.DataFrame(equiv), resumen


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data_calibracion"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "analisis_sensibilidad"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Leyendo {data_dir} ...")
    df = cargar(data_dir)
    tabla, equiv, resumen = analizar(df)
    with pd.option_context("display.width", 220, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:,.3f}"):
        print("\nCambio real por actualización (conocimiento)\n", tabla.to_string(index=False))
        print("\nEquivalencia de una corrección en actualizaciones reales\n", equiv.to_string(index=False))
    print()
    for k, v in resumen.items():
        print(f"  {k}: {v}")
    tabla.to_csv(os.path.join(out_dir, "correcciones_cambio_real.csv"), index=False, encoding="utf-8-sig")
    equiv.to_csv(os.path.join(out_dir, "correcciones_equivalencia.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "correcciones.json"), "w", encoding="utf-8") as f:
        json.dump({"resumen": resumen, "cambio_real": tabla.to_dict("records"),
                   "equivalencia": equiv.to_dict("records")}, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado en {out_dir}/")


if __name__ == "__main__":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
