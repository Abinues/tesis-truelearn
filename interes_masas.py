"""
interes_masas.py
─────────────────────────────────────────────────────────────────────────────
Diagnóstico: ¿por qué los terciles de interés no reparten 33/33/33?

calibracion_por_estrato.py mostró, con el censo y la regla «mayor o igual»,
33,3 % Bajo / 18,2 % Medio / 48,4 % Alto (con el rango promedio en los empates
el reparto es 33,3 / 39,0 / 27,7 %). Los cortes por cuantiles solo reparten parejo si los valores
son todos distintos. Si muchos temas tienen EXACTAMENTE el mismo valor de
interés (por ejemplo, los temas vistos una sola vez en videos con 5 temas
reciben la misma actualización), el corte cae sobre ese valor y todos
ellos van a la misma etiqueta.

Este script lista los valores de interés más repetidos, qué porcentaje de
todos los temas representa cada uno y en qué etiqueta cae con los cortes
vigentes de logic/dominio.py. Solo lectura.

Uso (raíz del proyecto, venv activado):
    python interes_masas.py [data_calibracion_todos] [analisis_sensibilidad]
"""
import collections
import csv
import glob
import json
import os
import sys

from logic.dominio import interes_label, pct_interes


def etiqueta(valor):
    # Misma función que usa la aplicación: compara el valor crudo con el corte.
    # (No comparar el porcentaje ya redondeado: el bloque 1,086 se muestra como
    # 36,8 % y caería en «Alto» por redondeo, cuando el corte lo deja en «Medio».)
    return interes_label(valor)[0]


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data_calibracion_todos"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "analisis_sensibilidad"
    os.makedirs(out_dir, exist_ok=True)
    cont, total = collections.Counter(), 0
    for ruta in glob.glob(os.path.join(data_dir, "visual_profile_*.json")):
        with open(ruta, "r", encoding="utf-8") as f:
            for kc in json.load(f).get("knowledge_components") or []:
                if kc.get("interest_mean") is not None:
                    cont[round(float(kc["interest_mean"]), 4)] += 1
                    total += 1
    if not total:
        sys.exit(f"No hay perfiles en {data_dir}")
    print(f"{total} temas, {len(cont)} valores distintos (redondeo a 4 decimales)\n")
    print("valor      % de los temas   % acumulado   % de escala   etiqueta")
    filas, acum = [], 0.0
    for valor, n in cont.most_common(12):
        pct, acum = n / total * 100, acum + n / total * 100
        filas.append({"valor": valor, "% de los temas": round(pct, 2),
                      "% acumulado": round(acum, 2),
                      "% de escala": round(pct_interes(valor), 1), "etiqueta": etiqueta(valor)})
        print(f"{valor:<10} {pct:>10.2f} {acum:>14.2f} {pct_interes(valor):>13.1f}   {etiqueta(valor)}")
    with open(os.path.join(out_dir, "interes_masas.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"\nGuardado en {out_dir}/interes_masas.csv")


if __name__ == "__main__":
    main()
