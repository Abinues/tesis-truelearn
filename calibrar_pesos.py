"""
calibrar_pesos.py
─────────────────────────────────────────────────────────────────────────────
Calibra los rangos de los PESOS del clasificador (novelty / interest / bias)
con el censo completo, leyendo los visual_profile_*.json de
data_calibracion_todos/. Cada perfil ya trae "weights" con la media y la
varianza de cada peso (los guarda logic/entrenador.py), así que no hace falta
data/usuarios_entrenados.json ni volver a entrenar.

Es la misma regla que KNOW_MIN/MAX: percentiles 1 y 99. Hay UNA muestra por
usuario, así que n = nº de usuarios (14.050 con el censo).

Solo lectura: no modifica ningún archivo del proyecto.

Uso (raíz del proyecto, venv activado):
    python calibrar_pesos.py [data_calibracion_todos] [analisis_sensibilidad]

Escribe analisis_sensibilidad/calibracion_pesos.csv y .json.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

FACTORES = [("novelty", "WEIGHT_NOVELTY"), ("interest", "WEIGHT_INTERES"), ("bias", "WEIGHT_BIAS")]


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data_calibracion_todos"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "analisis_sensibilidad"
    rutas = sorted(glob.glob(os.path.join(data_dir, "visual_profile_*.json")))
    if not rutas:
        print(f"No se encontró ningún visual_profile_*.json en {data_dir}")
        sys.exit(1)

    medias = {f: [] for f, _ in FACTORES}
    sin_pesos = 0
    for ruta in rutas:
        with open(ruta, "r", encoding="utf-8") as fh:
            perfil = json.load(fh)
        pesos = perfil.get("weights") or {}
        if not all(f in pesos and "mean" in pesos[f] for f, _ in FACTORES):
            sin_pesos += 1
            continue
        for f, _ in FACTORES:
            medias[f].append(float(pesos[f]["mean"]))

    n = len(medias["novelty"])
    print(f"Perfiles leídos: {len(rutas)}  |  con pesos: {n}  |  sin pesos: {sin_pesos}")

    try:
        from logic import dominio as D
        actuales = {
            "novelty": (D.WEIGHT_NOVELTY_MIN, D.WEIGHT_NOVELTY_MAX),
            "interest": (D.WEIGHT_INTERES_MIN, D.WEIGHT_INTERES_MAX),
            "bias": (D.WEIGHT_BIAS_MIN, D.WEIGHT_BIAS_MAX),
        }
    except Exception:
        actuales = {}

    filas, reporte = [], {"n_usuarios": n}
    for f, nombre in FACTORES:
        v = np.array(medias[f])
        p1, p99 = np.percentile(v, [1, 99])
        fila = {
            "peso": f, "n": n, "mínimo": v.min(), "p1": p1, "mediana": np.median(v),
            "promedio": v.mean(), "p99": p99, "máximo": v.max(),
            "p1 en dominio.py": actuales.get(f, (None, None))[0],
            "p99 en dominio.py": actuales.get(f, (None, None))[1],
        }
        filas.append(fila)
        reporte[f] = {k: (None if x is None else float(x)) for k, x in fila.items() if k not in ("peso", "n")}
        print(f"\n{f}: n={n}")
        for k in ("mínimo", "p1", "mediana", "promedio", "p99", "máximo"):
            print(f"   {k:9s} {fila[k]:+.4f}")

    print("\nPara logic/dominio.py (percentiles 1 y 99):")
    for f, nombre in FACTORES:
        v = np.array(medias[f])
        p1, p99 = np.percentile(v, [1, 99])
        print(f"  {nombre}_MIN, {nombre}_MAX = {p1:.4f}, {p99:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "calibracion_pesos.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    with open(os.path.join(out_dir, "calibracion_pesos.json"), "w", encoding="utf-8") as fh:
        json.dump(reporte, fh, indent=2, ensure_ascii=False)
    print(f"\nGuardado en {out_dir}/calibracion_pesos.csv y .json")


if __name__ == "__main__":
    main()