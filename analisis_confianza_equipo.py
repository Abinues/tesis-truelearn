"""
analisis_confianza_equipo.py
─────────────────────────────────────────────────────────────────────────────
Segunda parte del respaldo de la confianza (1 − varianza) × 100.

analisis_sensibilidad.py mostró que la confianza sube con la cantidad de
actualizaciones de cada tema, pero con mucha dispersión dentro de cada
tramo y, en el interés, una meseta a partir de ~20 actualizaciones. La
hipótesis es mecánica: TrueLearn actualiza todos los temas de un video
como un EQUIPO de TrueSkill, así que cuantos más temas tiene el video,
menos se reduce la varianza de cada uno (la reducción se reparte). Este
script lo mide con los datos reales de PEEK:

  - cuenta, por (usuario, tema), los eventos totales, los enganchados y
    el tamaño medio del equipo (temas por video);
  - cruza esos conteos con la confianza de cada tema en los perfiles;
  - muestra la confianza por tramo de eventos Y por tamaño de equipo;
  - compara cuánta variación de la confianza explica el nº de eventos
    solo (R²) frente a nº de eventos + tamaño de equipo.

Es de SOLO LECTURA sobre data_calibracion/. No entrena nada: usa las
mismas funciones de carga que entrenar_gran_escala.py.

Uso (raíz de tesis/, venv activado), después de entrenar_gran_escala.py:

    python analisis_confianza_equipo.py                 # data_calibracion/
    python analisis_confianza_equipo.py data_calibracion analisis_sensibilidad
"""
from __future__ import annotations

import collections
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd

UMBRAL_MUCHAS = 1        # censo: todos los usuarios con al menos 1 evento (igual que entrenar_gran_escala.py)
UMBRAL_POCAS_MAX = 0
LIMITE_ALTO = 10_000_000

TRAMOS_N = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 10**9)]
TRAMOS_EQ = [(1.0, 2.0, "1-2 temas"), (2.01, 5.0, "3-5 temas"),
             (5.01, 10.0, "6-10 temas"), (10.01, 1e9, "11+ temas")]


def contar_eventos(eventos):
    """
    eventos: lista de (evento, engaged) de UN usuario, como la devuelve
    seleccionar_usuarios(). Devuelve {topic_id: [n_total, n_enganchados, suma_tamaño_equipo]}.
    Se saltan los eventos sin temas, igual que entrenar_usuario().
    """
    conteo = collections.defaultdict(lambda: [0, 0, 0])
    for evento, engaged in eventos:
        ids = [tid for tid, _ in evento.knowledge.topic_kc_pairs()]
        if not ids:
            continue
        for tid in ids:
            c = conteo[tid]
            c[0] += 1
            c[1] += 1 if engaged else 0
            c[2] += len(ids)
    return conteo


def contar_todos():
    from logic.entrenador import seleccionar_usuarios
    from logic.peek_loader import cargar_dataset_peek

    base = os.path.dirname(os.path.abspath(__file__))
    train, _, _ = cargar_dataset_peek(os.path.join(base, "data_calibracion"))
    muchas, pocas = seleccionar_usuarios(
        train, n_muchas=LIMITE_ALTO, n_pocas=LIMITE_ALTO,
        umbral_muchas=UMBRAL_MUCHAS, umbral_pocas_max=UMBRAL_POCAS_MAX)
    filas = []
    for uid, evs in list(muchas) + list(pocas):
        for tid, (n, ne, suma_eq) in contar_eventos(evs).items():
            filas.append((uid, tid, n, ne, suma_eq / n))
    return pd.DataFrame(filas, columns=["user_id", "topic_id", "n_total", "n_eng", "eq_medio"])


def cargar_confianza(data_dir):
    filas = []
    for ruta in sorted(glob.glob(os.path.join(data_dir, "visual_profile_*.json"))):
        with open(ruta, "r", encoding="utf-8") as f:
            p = json.load(f)
        for kc in p.get("knowledge_components") or []:
            if kc.get("variance") is None:
                continue
            filas.append((p.get("user_id"), kc.get("topic_id"),
                          float(kc["variance"]), float(kc.get("interest_var", 0.5))))
    d = pd.DataFrame(filas, columns=["user_id", "topic_id", "k_var", "i_var"])
    # confianza = (1 - varianza / varianza inicial) x 100 (transformación afín de la
    # anterior: los R² y los Spearman no cambian)
    d["conf_k"] = np.clip((1 - d["k_var"] / 0.5) * 100, 0, 100)
    d["conf_i"] = np.clip((1 - d["i_var"] / 0.5) * 100, 0, 100)
    return d


def r2(y, columnas):
    X = np.column_stack([np.ones(len(y))] + columnas)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return float(1 - res.var() / y.var())


def analizar(conteos, conf):
    d = conf.merge(conteos, on=["user_id", "topic_id"], how="inner")
    if d.empty:
        sys.exit("No hubo coincidencias usuario/tema entre perfiles y eventos.")
    res = {"temas cruzados": int(len(d)), "temas en perfiles": int(len(conf))}

    filas = []
    for lo, hi in TRAMOS_N:
        for a, b, nombre in TRAMOS_EQ:
            s = d[(d["n_total"] >= lo) & (d["n_total"] <= hi) & (d["eq_medio"] >= a) & (d["eq_medio"] <= b)]
            if len(s) < 20:
                continue
            tramo = f"{lo}" if lo == hi else (f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}")
            filas.append({
                "eventos del tema": tramo, "tamaño de equipo": nombre, "n temas": int(len(s)),
                "conf. conocimiento mediana": float(s["conf_k"].median()),
                "conf. interés mediana": float(s["conf_i"].median()),
            })
    tabla = pd.DataFrame(filas)

    ln = np.log(d["n_total"].to_numpy(float))
    leq = np.log(d["eq_medio"].to_numpy(float))
    for etiqueta, col in (("conocimiento", "conf_k"), ("interés", "conf_i")):
        y = d[col].to_numpy(float)
        res[f"R² {etiqueta}: solo log(eventos)"] = r2(y, [ln])
        res[f"R² {etiqueta}: log(eventos) + log(tamaño de equipo)"] = r2(y, [ln, leq])
        res[f"Spearman {etiqueta} vs eventos"] = float(pd.Series(y).rank().corr(pd.Series(ln).rank()))
        res[f"Spearman {etiqueta} vs tamaño de equipo"] = float(pd.Series(y).rank().corr(pd.Series(leq).rank()))
    res["Tamaño de equipo medio: mediana / p90"] = [
        float(d["eq_medio"].median()), float(d["eq_medio"].quantile(.9))]
    return tabla, res


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data_calibracion_todos"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "analisis_sensibilidad"
    os.makedirs(out_dir, exist_ok=True)
    print("Contando eventos por usuario y tema en PEEK ...")
    conteos = contar_todos()
    print(f"  {len(conteos)} pares (usuario, tema)")
    print(f"Leyendo perfiles de {data_dir} ...")
    conf = cargar_confianza(data_dir)
    tabla, res = analizar(conteos, conf)

    print("\nConfianza por eventos del tema y tamaño de equipo (temas por video)")
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:,.2f}"):
        print(tabla.to_string(index=False))
    print()
    for k, v in res.items():
        print(f"  {k}: {v}")
    tabla.to_csv(os.path.join(out_dir, "confianza_por_equipo.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(out_dir, "confianza_por_equipo.json"), "w", encoding="utf-8") as f:
        json.dump({"resumen": res, "tabla": tabla.to_dict("records")}, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado en {out_dir}/")


if __name__ == "__main__":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
