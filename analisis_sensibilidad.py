"""
analisis_sensibilidad.py
─────────────────────────────────────────────────────────────────────────────
Respalda con datos dos decisiones de diseño de logic/dominio.py que no
tienen una fuente en la literatura:

  A. El recorte en los percentiles 1/99 antes de normalizar a 0-100 %.
     Se recalcula con otros cortes (0,5/99,5, 2/98, 5/95 y mín/máx
     absolutos) y se mide cuánto cambian los porcentajes y las etiquetas.

  B. Los umbrales de etiqueta (conocimiento 70/40/10 %, interés 50/30 %)
     y la fórmula de confianza (1 − varianza) × 100.
       - Umbrales: a qué percentil de los datos equivale cada uno, qué
         proporción de temas cae en cada etiqueta y cuántos cambian de
         etiqueta si el umbral se mueve ±5 puntos.
       - Confianza: si sube con la cantidad de veces que el modelo
         actualizó cada tema (si el indicador se comporta como debe).

Es de SOLO LECTURA sobre los perfiles. No toca logic/dominio.py ni data/.
Escribe en una carpeta nueva (por defecto analisis_sensibilidad/):
tablas CSV, un JSON con todo y, si hay matplotlib, dos figuras PNG.

Uso (parado en la raíz de tesis/, con el venv activado), DESPUÉS de
python entrenar_gran_escala.py:

    python analisis_sensibilidad.py                      # data_calibracion/
    python analisis_sensibilidad.py data_calibracion salida_dir

Requisitos: numpy y pandas (ya están en el entorno del dashboard).
matplotlib es opcional: sin él se omiten las figuras.

Cómo se cuenta "cuántas veces se actualizó un tema"
-----------------------------------------------------
Los perfiles no guardan ese contador, pero session_history sí guarda la
varianza de cada tema en cada paso. Un tema se actualiza cuando su
varianza cambia respecto del paso anterior (o cuando aparece por primera
vez). Limitaciones, para decirlas en la tesis:
  - Si dos eventos comparten timestamp, session_history guarda solo el
    último estado, así que el conteo puede quedar un poco por debajo.
  - session_history solo guarda conocimiento. Se usa el mismo conteo para
    el interés, porque cada evento actualiza el conocimiento y el
    interés de los temas que contiene.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd

# ── Parámetros que reproducen logic/dominio.py ──────────────────────────────
# Si cambiás los umbrales en dominio.py, cambialos también acá.
# Cortes vigentes (cuantiles del censo): conocimiento = cuartiles 75/50/25,
# interés = terciles 67/33 de la escala 0-100.
UMBRAL_K = (64.5, 55.3, 31.6)   # Alto / Medio / Básico (resto: Muy bajo)
UMBRAL_I = (36.8, 27.9)         # Alto / Medio (resto: Bajo); Alto queda por encima del bloque de empates
VAR_INICIAL = 0.5               # varianza inicial de TrueLearn; confianza = (1 - var/VAR_INICIAL) x 100
CORTE_ACTUAL = "1 / 99 (actual)"

# (percentil inferior, percentil superior, nombre)
ESCENARIOS = [
    (0.0, 100.0, "mín / máx absolutos"),
    (0.5, 99.5, "0,5 / 99,5"),
    (1.0, 99.0, CORTE_ACTUAL),
    (2.0, 98.0, "2 / 98"),
    (5.0, 95.0, "5 / 95"),
]

# Tramos de "cantidad de actualizaciones" para la curva de confianza.
TRAMOS = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 10**9)]

ETIQ_K = ["Alto", "Medio", "Básico", "Muy bajo"]
ETIQ_I = ["Alto", "Medio", "Bajo"]


# ── Carga ───────────────────────────────────────────────────────────────────
def contar_actualizaciones(sesiones):
    """Devuelve ({tema: n_actualizaciones}, n_transiciones, n_que_suben)."""
    previo, conteo = {}, {}
    trans = sube = 0
    for s in sorted(sesiones, key=lambda x: x.get("session_id", 0)):
        for nombre, v in (s.get("variances") or {}).items():
            if nombre not in previo:
                conteo[nombre] = 1
            elif v != previo[nombre]:
                conteo[nombre] += 1
                trans += 1
                if v > previo[nombre]:
                    sube += 1
            previo[nombre] = v
    return conteo, trans, sube


def cargar(data_dir):
    rutas = sorted(glob.glob(os.path.join(data_dir, "visual_profile_*.json")))
    if not rutas:
        sys.exit(f"No hay visual_profile_*.json en {data_dir}. "
                 "Corré antes: python entrenar_gran_escala.py")
    filas = []
    trans_total = sube_total = 0
    for i, ruta in enumerate(rutas, 1):
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                perfil = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  no se pudo leer {os.path.basename(ruta)}: {e}")
            continue
        kcs = perfil.get("knowledge_components") or []
        if not kcs:
            continue
        conteo, trans, sube = contar_actualizaciones(perfil.get("session_history") or [])
        trans_total += trans
        sube_total += sube
        uid = perfil.get("user_id")
        for kc in kcs:
            if kc.get("mean") is None or kc.get("variance") is None:
                continue
            filas.append((
                uid, kc.get("topic_name"),
                float(kc["mean"]), float(kc["variance"]),
                float(kc.get("interest_mean", 0.0)), float(kc.get("interest_var", 0.5)),
                conteo.get(kc.get("topic_name"), np.nan),
            ))
        if i % 1000 == 0:
            print(f"  ... {i}/{len(rutas)} perfiles leídos")
    df = pd.DataFrame(filas, columns=[
        "user_id", "topic_name", "k_mean", "k_var", "i_mean", "i_var", "n_act"])
    return df, len(rutas), trans_total, sube_total


# ── Normalización y etiquetas (misma lógica que logic/dominio.py) ───────────
def a_pct(x, lo, hi):
    return np.clip((x - lo) / (hi - lo) * 100.0, 0.0, 100.0)


def etiqueta_k(p, umbrales=UMBRAL_K):
    a, m, b = umbrales
    return np.select([p >= a, p >= m, p >= b], ETIQ_K[:3], ETIQ_K[3])


def etiqueta_i(p, umbrales=UMBRAL_I):
    a, m = umbrales
    return np.select([p >= a, p >= m], ETIQ_I[:2], ETIQ_I[2])


def cortes_rango_promedio(valores, cuantiles):
    """Cortes por cuantiles con RANGO PROMEDIO para los empates: el corte es el
    punto medio entre el último valor distinto por debajo del cuantil y el
    primero que lo alcanza (posición = (menores + 0,5·iguales) / N)."""
    v, cnt = np.unique(np.asarray(valores, dtype=float), return_counts=True)
    pos = (np.cumsum(cnt) - cnt + 0.5 * cnt) / cnt.sum()
    out = []
    for q in cuantiles:
        i = min(max(int(np.searchsorted(pos, q, side="left")), 1), len(v) - 1)
        out.append(float((v[i - 1] + v[i]) / 2.0))
    return out


def etiquetas_cuantiles_k(p):
    """Conocimiento con cortes RECALCULADOS como cuartiles de `p` (criterio normativo)."""
    return etiqueta_k(p, tuple(cortes_rango_promedio(p, [.75, .50, .25])))


def etiquetas_cuantiles_i(p):
    """Interés con cortes RECALCULADOS como terciles de `p` (empates: rango promedio)."""
    return etiqueta_i(p, tuple(cortes_rango_promedio(p, [2 / 3, 1 / 3])))


def reparto(etiquetas, orden):
    s = pd.Series(etiquetas).value_counts(normalize=True) * 100
    return {e: float(s.get(e, 0.0)) for e in orden}


def rangos(df, p_lo, p_hi):
    k_lo, k_hi = np.percentile(df["k_mean"], [p_lo, p_hi])
    i_lo, i_hi = np.percentile(df["i_mean"], [p_lo, p_hi])
    return float(k_lo), float(k_hi), float(i_lo), float(i_hi)


# ── A. Sensibilidad del corte ───────────────────────────────────────────────
def sensibilidad_corte(df):
    ref = next(e for e in ESCENARIOS if e[2] == CORTE_ACTUAL)
    k_lo, k_hi, i_lo, i_hi = rangos(df, ref[0], ref[1])
    ref_k = etiquetas_cuantiles_k(a_pct(df["k_mean"], k_lo, k_hi))
    ref_i = etiquetas_cuantiles_i(a_pct(df["i_mean"], i_lo, i_hi))
    ref_kf = etiqueta_k(a_pct(df["k_mean"], k_lo, k_hi))
    ref_if = etiqueta_i(a_pct(df["i_mean"], i_lo, i_hi))

    filas_k, filas_i = [], []
    for p_lo, p_hi, nombre in ESCENARIOS:
        k_lo, k_hi, i_lo, i_hi = rangos(df, p_lo, p_hi)

        pk = a_pct(df["k_mean"], k_lo, k_hi)
        lk = etiquetas_cuantiles_k(pk)
        lkf = etiqueta_k(pk)
        fila = {
            "corte": nombre, "MIN": k_lo, "MAX": k_hi,
            "% por debajo del mínimo": float((df["k_mean"] < k_lo).mean() * 100),
            "% por encima del máximo": float((df["k_mean"] > k_hi).mean() * 100),
            "mediana %": float(np.median(pk)),
            "rango intercuartil %": float(np.percentile(pk, 75) - np.percentile(pk, 25)),
            "% que cambia de etiqueta vs actual (cortes por cuantiles)": float((lk != ref_k).mean() * 100),
            "% que cambia con cortes fijos vigentes": float((lkf != ref_kf).mean() * 100),
        }
        fila.update({f"% {e}": v for e, v in reparto(lk, ETIQ_K).items()})
        filas_k.append(fila)

        pi = a_pct(df["i_mean"], i_lo, i_hi)
        li = etiquetas_cuantiles_i(pi)
        lif = etiqueta_i(pi)
        fila = {
            "corte": nombre, "MIN": i_lo, "MAX": i_hi,
            "% por encima del máximo": float((df["i_mean"] > i_hi).mean() * 100),
            "mediana %": float(np.median(pi)),
            "rango intercuartil %": float(np.percentile(pi, 75) - np.percentile(pi, 25)),
            "% que cambia de etiqueta vs actual (cortes por cuantiles)": float((li != ref_i).mean() * 100),
            "% que cambia con cortes fijos vigentes": float((lif != ref_if).mean() * 100),
        }
        fila.update({f"% {e}": v for e, v in reparto(li, ETIQ_I).items()})
        filas_i.append(fila)
    return pd.DataFrame(filas_k), pd.DataFrame(filas_i)


# ── B1. Umbrales de etiqueta ────────────────────────────────────────────────
def analisis_umbrales(df):
    ref = next(e for e in ESCENARIOS if e[2] == CORTE_ACTUAL)
    k_lo, k_hi, i_lo, i_hi = rangos(df, ref[0], ref[1])
    pk = a_pct(df["k_mean"], k_lo, k_hi)
    pi = a_pct(df["i_mean"], i_lo, i_hi)
    base_k, base_i = etiqueta_k(pk), etiqueta_i(pi)

    # a qué percentil de los datos equivale cada umbral
    equiv = []
    for nombre, umbrales, serie in (("Conocimiento", UMBRAL_K, pk),
                                    ("Interés", UMBRAL_I, pi)):
        for u in umbrales:
            equiv.append({
                "variable": nombre, "umbral %": u,
                "% de temas por debajo": float((serie < u).mean() * 100),
                "% de temas en o por encima": float((serie >= u).mean() * 100),
            })

    # robustez: mover todos los umbrales ±5 puntos
    robustez = []
    for delta in (-10, -5, 5, 10):
        uk = tuple(max(0.0, min(100.0, u + delta)) for u in UMBRAL_K)
        ui = tuple(max(0.0, min(100.0, u + delta)) for u in UMBRAL_I)
        robustez.append({
            "desplazamiento (puntos)": delta,
            "Conocimiento: % que cambia de etiqueta": float((etiqueta_k(pk, uk) != base_k).mean() * 100),
            "Interés: % que cambia de etiqueta": float((etiqueta_i(pi, ui) != base_i).mean() * 100),
        })

    # alternativa relativa: cuartiles de la propia distribución
    q = np.percentile(pk, [25, 50, 75])
    alt_k = np.select([pk >= q[2], pk >= q[1], pk >= q[0]], ETIQ_K[:3], ETIQ_K[3])
    t = np.percentile(pi, [100 / 3, 200 / 3])
    alt_i = np.select([pi >= t[1], pi >= t[0]], ETIQ_I[:2], ETIQ_I[2])
    alternativa = {
        "Conocimiento: cuartiles (umbrales equivalentes %)": [float(x) for x in q],
        "Conocimiento: % de acuerdo con umbrales fijos": float((alt_k == base_k).mean() * 100),
        "Interés: terciles (umbrales equivalentes %)": [float(x) for x in t],
        "Interés: % de acuerdo con umbrales fijos": float((alt_i == base_i).mean() * 100),
    }
    reparto_base = {
        "Conocimiento": reparto(base_k, ETIQ_K),
        "Interés": reparto(base_i, ETIQ_I),
    }
    return pd.DataFrame(equiv), pd.DataFrame(robustez), alternativa, reparto_base


# ── B2. Confianza vs cantidad de actualizaciones ────────────────────────────
def analisis_confianza(df, trans, sube):
    d = df.dropna(subset=["n_act"]).copy()
    d["conf_k"] = np.clip((1 - d["k_var"] / VAR_INICIAL) * 100, 0, 100)
    d["conf_i"] = np.clip((1 - d["i_var"] / VAR_INICIAL) * 100, 0, 100)

    def spearman(a, b):
        return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

    resumen = {
        "KC con conteo de actualizaciones": int(len(d)),
        "KC sin conteo (tema no encontrado en session_history)": int(len(df) - len(d)),
        "Spearman (n_act, confianza conocimiento)": spearman(d["n_act"], d["conf_k"]),
        "Spearman (n_act, confianza interés)": spearman(d["n_act"], d["conf_i"]),
        "Confianza conocimiento: mín / mediana / máx": [
            float(d["conf_k"].min()), float(d["conf_k"].median()), float(d["conf_k"].max())],
        "Confianza interés: mín / mediana / máx": [
            float(d["conf_i"].min()), float(d["conf_i"].median()), float(d["conf_i"].max())],
        "% de transiciones en que la varianza SUBE": (
            float(sube / trans * 100) if trans else float("nan")),
        "Transiciones de varianza observadas": int(trans),
    }

    filas = []
    for lo, hi in TRAMOS:
        s = d[(d["n_act"] >= lo) & (d["n_act"] <= hi)]
        if s.empty:
            continue
        etiqueta = f"{lo}" if lo == hi else (f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}")
        filas.append({
            "actualizaciones": etiqueta, "n temas": int(len(s)),
            "conf. conocimiento p25": float(s["conf_k"].quantile(.25)),
            "conf. conocimiento mediana": float(s["conf_k"].median()),
            "conf. conocimiento p75": float(s["conf_k"].quantile(.75)),
            "conf. interés p25": float(s["conf_i"].quantile(.25)),
            "conf. interés mediana": float(s["conf_i"].median()),
            "conf. interés p75": float(s["conf_i"].quantile(.75)),
        })
    tabla = pd.DataFrame(filas)
    med_k = tabla["conf. conocimiento mediana"].to_numpy()
    med_i = tabla["conf. interés mediana"].to_numpy()
    resumen["Medianas de conocimiento no decrecientes entre tramos"] = bool(np.all(np.diff(med_k) >= -1e-9))
    resumen["Medianas de interés no decrecientes entre tramos"] = bool(np.all(np.diff(med_i) >= -1e-9))
    return tabla, resumen


# ── Figuras (opcionales) ────────────────────────────────────────────────────
def figuras(sk, si, conf, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib no está instalado: se omiten las figuras)")
        return
    azul, gris = "#3b598d", "#7a8599"
    colores = ["#3b598d", "#7f9bc8", "#c3cde0", "#e5e8ef"]

    fig, ejes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, tabla, etiquetas, titulo in (
        (ejes[0], sk, ETIQ_K, "Conocimiento"), (ejes[1], si, ETIQ_I, "Interés")):
        base = np.zeros(len(tabla))
        for e, c in zip(etiquetas, colores):
            v = tabla[f"% {e}"].to_numpy()
            ax.bar(tabla["corte"], v, bottom=base, label=e, color=c, edgecolor="white")
            base += v
        ax.set_title(titulo)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4, frameon=False)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    ejes[0].set_ylabel("% de temas por etiqueta")
    fig.suptitle("Reparto de etiquetas según el corte de percentiles", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_sensibilidad_cortes.png"), dpi=200)
    plt.close(fig)

    x = np.arange(len(conf))
    fig, ax = plt.subplots(figsize=(7, 4))
    for pref, color, nombre in (("conocimiento", azul, "Conocimiento"), ("interés", gris, "Interés")):
        ax.plot(x, conf[f"conf. {pref} mediana"], marker="o", color=color, label=nombre)
        ax.fill_between(x, conf[f"conf. {pref} p25"], conf[f"conf. {pref} p75"],
                        color=color, alpha=.15)
    ax.set_xticks(x)
    ax.set_xticklabels(conf["actualizaciones"])
    ax.set_xlabel("Veces que el modelo actualizó el tema")
    ax.set_ylabel("Confianza (%)")
    ax.set_title("Confianza según las veces que se actualizó el tema (mediana y rango p25–p75)", fontsize=10)
    ax.legend()
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_confianza_vs_actualizaciones.png"), dpi=200)
    plt.close(fig)


# ── Principal ───────────────────────────────────────────────────────────────
def mostrar(titulo, tabla):
    print("\n" + "=" * 78 + f"\n{titulo}\n" + "=" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:,.3f}"):
        print(tabla.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default="data_calibracion")
    ap.add_argument("out_dir", nargs="?", default="analisis_sensibilidad")
    a = ap.parse_args()

    print(f"Leyendo perfiles de {a.data_dir} ...")
    df, n_perfiles, trans, sube = cargar(a.data_dir)
    print(f"{n_perfiles} perfiles, {len(df)} Knowledge Components")
    os.makedirs(a.out_dir, exist_ok=True)

    sk, si = sensibilidad_corte(df)
    mostrar("A1. Sensibilidad del corte - CONOCIMIENTO", sk)
    mostrar("A2. Sensibilidad del corte - INTERÉS (MIN = percentil inferior del escenario)", si)

    equiv, robustez, alternativa, reparto_base = analisis_umbrales(df)
    mostrar("B1a. A qué percentil de los datos equivale cada umbral (corte 1/99)", equiv)
    mostrar("B1b. Robustez: mover los umbrales", robustez)
    print("\nB1c. Alternativa relativa (cuantiles de la propia distribución)")
    for k, v in alternativa.items():
        print(f"  {k}: {v}")
    print("\nReparto de etiquetas con los umbrales actuales:")
    for var, rep in reparto_base.items():
        print(f"  {var}: " + ", ".join(f"{e} {v:.1f}%" for e, v in rep.items()))

    conf, resumen = analisis_confianza(df, trans, sube)
    mostrar("B2. Confianza según veces que se actualizó el tema", conf)
    print()
    for k, v in resumen.items():
        print(f"  {k}: {v}")

    sk.to_csv(os.path.join(a.out_dir, "sensibilidad_conocimiento.csv"), index=False, encoding="utf-8-sig")
    si.to_csv(os.path.join(a.out_dir, "sensibilidad_interes.csv"), index=False, encoding="utf-8-sig")
    equiv.to_csv(os.path.join(a.out_dir, "umbrales_equivalencia.csv"), index=False, encoding="utf-8-sig")
    robustez.to_csv(os.path.join(a.out_dir, "umbrales_robustez.csv"), index=False, encoding="utf-8-sig")
    conf.to_csv(os.path.join(a.out_dir, "confianza_vs_actualizaciones.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(a.out_dir, "analisis_sensibilidad.json"), "w", encoding="utf-8") as f:
        json.dump({
            "n_perfiles": n_perfiles, "n_kc": int(len(df)),
            "sensibilidad_conocimiento": sk.to_dict("records"),
            "sensibilidad_interes": si.to_dict("records"),
            "umbrales_equivalencia": equiv.to_dict("records"),
            "umbrales_robustez": robustez.to_dict("records"),
            "umbrales_alternativa": alternativa,
            "reparto_etiquetas": reparto_base,
            "confianza_tramos": conf.to_dict("records"),
            "confianza_resumen": resumen,
        }, f, indent=2, ensure_ascii=False)
    figuras(sk, si, conf, a.out_dir)
    print(f"\nResultados guardados en {a.out_dir}/")


if __name__ == "__main__":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
