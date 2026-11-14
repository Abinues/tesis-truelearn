"""
calibracion_por_estrato.py
─────────────────────────────────────────────────────────────────────────────
Respalda la calibración de la escala con TODOS los usuarios (censo) y
muestra que no depende del nivel de actividad.

Qué calcula (solo lectura sobre la carpeta de perfiles):

  1. Estratos por actividad: los usuarios se dividen en tres grupos por
     TERCILES de su cantidad de eventos (pocos / intermedios / muchos).
     Los cortes salen de los datos, no se eligen a mano.
  2. Por estrato y para el total: nº de usuarios, nº de temas (KC), y los
     percentiles 1 y 99 de la media de conocimiento y de interés, y la
     mediana de la varianza (cuánta incertidumbre queda).
  3. Comparación con la calibración anterior de logic/dominio.py
     (KNOW −0.379/0.398, INT 2.720 de máximo): ¿cambia el rango al incluir
     a todos? ¿cambia entre estratos?
  4. Cortes de etiquetas por CUANTILES de los datos reales (criterio
     normativo, ver logic/dominio.py): conocimiento en cuartiles
     (Muy bajo | Básico | Medio | Alto), interés en terciles
     (Bajo | Medio | Alto). Los valores empatados reciben el RANGO PROMEDIO
     (convención estándar), así que un bloque de valores idénticos que
     cruza un cuantil queda en una sola etiqueta, la de su rango promedio;
     el corte se fija a mitad de camino entre dos valores distintos
     vecinos. Se expresan en el porcentaje 0–100 de la escala con los p1/p99
     del total (los cuatro extremos de la escala son percentiles 1 y 99).
  5. Impacto por estrato: qué % de los temas de cada estrato queda en cada
     etiqueta con esos cortes (si un estrato quedara casi todo en «Alto» o
     en «Bajo», habría que discutirlo en la tesis).

Uso:
    python calibracion_por_estrato.py [data_calibracion_todos] [analisis_sensibilidad]

Escribe calibracion_por_estrato.csv / _etiquetas.csv / .json.
"""
from __future__ import annotations

import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd

# Calibración vigente antes del censo (para comparar).
ANTES = {"KNOW_MIN": -0.379, "KNOW_MAX": 0.398, "INT_MIN": -1.0, "INT_MAX": 2.720}   # corrida con 8.152 usuarios
ESTRATOS = ["Pocos eventos", "Intermedios", "Muchos eventos"]


def eventos_de(perfil, tabla_eventos):
    """Nº de eventos del usuario: json del entrenamiento → perfil → sesiones."""
    uid = str(perfil.get("user_id"))
    if uid in tabla_eventos:
        return int(tabla_eventos[uid])
    if "number_of_engagements" in perfil:
        return int(perfil["number_of_engagements"]) + int(perfil.get("number_of_non_engagements", 0))
    return len(perfil.get("session_history") or [])


def cargar(data_dir):
    rutas = sorted(glob.glob(os.path.join(data_dir, "visual_profile_*.json")))
    if not rutas:
        sys.exit(f"No hay visual_profile_*.json en {data_dir}")
    ruta_ev = os.path.join(data_dir, "eventos_por_usuario.json")
    tabla = {}
    if os.path.exists(ruta_ev):
        with open(ruta_ev, "r", encoding="utf-8") as f:
            tabla = json.load(f)
    filas = []
    for ruta in rutas:
        with open(ruta, "r", encoding="utf-8") as f:
            p = json.load(f)
        n_ev = eventos_de(p, tabla)
        for kc in p.get("knowledge_components") or []:
            if kc.get("mean") is None:
                continue
            filas.append((p.get("user_id"), n_ev, float(kc["mean"]), float(kc.get("variance", np.nan)),
                          float(kc.get("interest_mean", np.nan)), float(kc.get("interest_var", np.nan))))
    return pd.DataFrame(filas, columns=["user_id", "eventos", "k_mean", "k_var", "i_mean", "i_var"])


def asignar_estratos(df):
    """Terciles de eventos POR USUARIO (no por tema): cada usuario pesa una vez."""
    por_usuario = df.groupby("user_id")["eventos"].first()
    c1, c2 = por_usuario.quantile([1 / 3, 2 / 3]).to_numpy()
    def est(n):
        return ESTRATOS[0] if n <= c1 else (ESTRATOS[1] if n <= c2 else ESTRATOS[2])
    df = df.copy()
    df["estrato"] = df["eventos"].map(est)
    return df, (float(c1), float(c2))


def pct(x, lo, hi):
    return np.clip((x - lo) / (hi - lo) * 100.0, 0.0, 100.0)


def resumen_estrato(d, nombre):
    return {
        "estrato": nombre,
        "usuarios": int(d["user_id"].nunique()),
        "temas (KC)": int(len(d)),
        "eventos por usuario (mediana)": float(d.groupby("user_id")["eventos"].first().median()),
        "conocimiento p1": float(d["k_mean"].quantile(.01)),
        "conocimiento p99": float(d["k_mean"].quantile(.99)),
        "interés p1": float(d["i_mean"].quantile(.01)),
        "interés p99": float(d["i_mean"].quantile(.99)),
        "varianza conocimiento (mediana)": float(d["k_var"].median()),
        "varianza interés (mediana)": float(d["i_var"].median()),
    }


def cortes_rango_promedio(valores, cuantiles):
    """Cortes en valor crudo para clasificar por cuantiles con RANGO PROMEDIO.

    Cada valor distinto recibe como posición el promedio de las posiciones
    que ocupan sus repeticiones: (menores + 0,5·iguales) / N. El corte de
    cada cuantil es el punto medio entre el último valor distinto por
    debajo y el primero que lo alcanza. Devuelve una lista en el orden de
    `cuantiles`.
    """
    v, cnt = np.unique(np.asarray(valores, dtype=float), return_counts=True)
    posicion = (np.cumsum(cnt) - cnt + 0.5 * cnt) / cnt.sum()
    cortes = []
    for q in cuantiles:
        i = int(np.searchsorted(posicion, q, side="left"))
        i = min(max(i, 1), len(v) - 1)
        cortes.append(float((v[i - 1] + v[i]) / 2.0))
    return cortes


def cortes_cuantiles(df, know_min, know_max, int_min, int_max):
    """Cuartiles del conocimiento y terciles del interés (rango promedio),
    en valor crudo y en % de la escala."""
    kb, km, ka = cortes_rango_promedio(df["k_mean"], [.25, .50, .75])
    im, ia = cortes_rango_promedio(df["i_mean"], [1 / 3, 2 / 3])
    crudos = {"BASICO": kb, "MEDIO": km, "ALTO": ka, "INTERES_MEDIO": im, "INTERES_ALTO": ia}
    pcts = {
        "UMBRAL_BASICO_PCT": float(pct(kb, know_min, know_max)),
        "UMBRAL_MEDIO_PCT": float(pct(km, know_min, know_max)),
        "UMBRAL_ALTO_PCT": float(pct(ka, know_min, know_max)),
        "UMBRAL_INTERES_MEDIO_PCT": float(pct(im, int_min, int_max)),
        "UMBRAL_INTERES_ALTO_PCT": float(pct(ia, int_min, int_max)),
    }
    return pcts, crudos


def etiquetas_por_estrato(df, crudos):
    k, i = df["k_mean"].to_numpy(), df["i_mean"].to_numpy()
    lk = np.select([k >= crudos["ALTO"], k >= crudos["MEDIO"], k >= crudos["BASICO"]],
                   ["Alto", "Medio", "Básico"], default="Muy bajo")
    li = np.select([i >= crudos["INTERES_ALTO"], i >= crudos["INTERES_MEDIO"]],
                   ["Alto", "Medio"], default="Bajo")
    d = df.assign(lk=lk, li=li)
    filas = []
    for est in ESTRATOS + ["TOTAL"]:
        s = d if est == "TOTAL" else d[d["estrato"] == est]
        fila = {"estrato": est, "temas": int(len(s))}
        for et in ("Muy bajo", "Básico", "Medio", "Alto"):
            fila[f"conocimiento: {et} (%)"] = float((s["lk"] == et).mean() * 100)
        for et in ("Bajo", "Medio", "Alto"):
            fila[f"interés: {et} (%)"] = float((s["li"] == et).mean() * 100)
        filas.append(fila)
    return pd.DataFrame(filas)


def analizar(df):
    df, (c1, c2) = asignar_estratos(df)
    filas = [resumen_estrato(df[df["estrato"] == e], e) for e in ESTRATOS] + [resumen_estrato(df, "TOTAL")]
    tabla = pd.DataFrame(filas)
    tot = filas[-1]
    know_min, know_max = tot["conocimiento p1"], tot["conocimiento p99"]
    int_min, int_max = tot["interés p1"], tot["interés p99"]
    cortes, crudos = cortes_cuantiles(df, know_min, know_max, int_min, int_max)
    etiquetas = etiquetas_por_estrato(df, crudos)
    resumen = {
        "usuarios": int(df["user_id"].nunique()),
        "temas (KC)": int(len(df)),
        "cortes de tercil de eventos por usuario": {"pocos ≤": c1, "intermedios ≤": c2},
        "calibración anterior": ANTES,
        "calibración con todos los usuarios": {
            "KNOW_MIN": know_min, "KNOW_MAX": know_max, "INT_MIN": int_min, "INT_MAX": int_max},
        "cortes de etiquetas por cuantiles": cortes,
        "cortes de etiquetas en valor crudo": crudos,
    }
    return tabla, etiquetas, resumen


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data_calibracion_todos"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "analisis_sensibilidad"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Leyendo {data_dir} ...")
    df = cargar(data_dir)
    tabla, etiquetas, resumen = analizar(df)
    with pd.option_context("display.width", 240, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:,.3f}"):
        print("\nCalibración por estrato de actividad\n", tabla.to_string(index=False))
        print("\nReparto de etiquetas por estrato (con los cortes por cuantiles del total)\n",
              etiquetas.to_string(index=False))
    print()
    for k, v in resumen.items():
        print(f"  {k}: {v}")
    print("\nPara logic/dominio.py (cortes por cuantiles):")
    for k, v in resumen["cortes de etiquetas por cuantiles"].items():
        print(f"  {k} = {v:.1f}")
    cal = resumen["calibración con todos los usuarios"]
    print(f"  KNOW_MIN, KNOW_MAX = {cal['KNOW_MIN']:.3f}, {cal['KNOW_MAX']:.3f}")
    print(f"  INT_MIN,  INT_MAX  = {cal['INT_MIN']:.3f}, {cal['INT_MAX']:.3f}")
    tabla.to_csv(os.path.join(out_dir, "calibracion_por_estrato.csv"), index=False, encoding="utf-8-sig")
    etiquetas.to_csv(os.path.join(out_dir, "calibracion_por_estrato_etiquetas.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "calibracion_por_estrato.json"), "w", encoding="utf-8") as f:
        json.dump({"resumen": resumen, "estratos": tabla.to_dict("records"),
                   "etiquetas": etiquetas.to_dict("records")}, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado en {out_dir}/")


if __name__ == "__main__":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
