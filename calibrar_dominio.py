"""
calibrar_dominio.py
─────────────────────────────────────────────────────────────────────────────
Calcula estadísticas empíricas (mínimo, máximo, percentiles) del `mean` y
la `variance` de los Knowledge Components de TODOS los usuarios ya
entrenados, leyendo directamente los data/visual_profile_*.json que
genera logic/entrenador.py.

No modifica ningún archivo del proyecto: es de solo lectura sobre data/,
y solo ESCRIBE un reporte nuevo en data/calibracion_dominio.json.

Uso (parado en la raíz de tesis/, con el venv activado):

    python calibrar_dominio.py

Requisito: correrlo DESPUÉS de reentrenar con datos reales de PEEK
(python -m logic.entrenador), no con perfiles viejos.
"""
import glob
import json
import os
import statistics
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Por defecto analiza data/ (la carpeta que usa el dashboard/backend en
# vivo). Se puede apuntar a otra carpeta pasándola como argumento, por
# ejemplo para calibrar contra una carpeta de entrenamiento a gran
# escala separada, sin mezclarla con los usuarios "de demo":
#   python calibrar_dominio.py data_calibracion
DATA_DIR = os.path.join(BASE_DIR, sys.argv[1] if len(sys.argv) > 1 else "data")


def cargar_perfiles():
    rutas = sorted(glob.glob(os.path.join(DATA_DIR, "visual_profile_*.json")))
    if not rutas:
        print(f"❌ No se encontró ningún data/visual_profile_*.json en {DATA_DIR}")
        print("   Corré primero: python -m logic.entrenador")
        sys.exit(1)

    perfiles = []
    for ruta in rutas:
        try:
            with open(ruta, "r", encoding="utf-8") as f:
                perfiles.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️  No se pudo leer {os.path.basename(ruta)}: {e}")
    return perfiles


def percentil(valores_ordenados, p):
    """Percentil por interpolación lineal, sin depender de numpy/pandas."""
    n = len(valores_ordenados)
    if n == 0:
        return None
    if n == 1:
        return valores_ordenados[0]
    k = (n - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return valores_ordenados[f]
    return valores_ordenados[f] * (c - k) + valores_ordenados[c] * (k - f)


def resumen(valores, nombre):
    if not valores:
        print(f"  {nombre}: sin datos")
        return None
    vs = sorted(valores)
    r = {
        "n": len(vs),
        "min": vs[0],
        "p1": percentil(vs, 1),
        "p5": percentil(vs, 5),
        "p25": percentil(vs, 25),
        "mediana": percentil(vs, 50),
        "p75": percentil(vs, 75),
        "p95": percentil(vs, 95),
        "p99": percentil(vs, 99),
        "max": vs[-1],
        "promedio": statistics.fmean(vs),
    }
    print(f"  {nombre}  (n={r['n']})")
    print(f"    mínimo:    {r['min']:+.4f}")
    print(f"    p1:        {r['p1']:+.4f}")
    print(f"    p5:        {r['p5']:+.4f}")
    print(f"    p25:       {r['p25']:+.4f}")
    print(f"    mediana:   {r['mediana']:+.4f}")
    print(f"    promedio:  {r['promedio']:+.4f}")
    print(f"    p75:       {r['p75']:+.4f}")
    print(f"    p95:       {r['p95']:+.4f}")
    print(f"    p99:       {r['p99']:+.4f}")
    print(f"    máximo:    {r['max']:+.4f}")
    return r


def cargar_indice_usuarios():
    """
    Lee data/usuarios_entrenados.json (el índice global que genera
    entrenador.py), que trae el triple de weights (novelty/interest/bias)
    por usuario. A diferencia de knowledge_components (miles de muestras,
    varias por usuario), acá hay UNA sola muestra por usuario -> con 25
    usuarios entrenados son apenas 25 puntos. Suficiente para una
    primera mirada, no para una calibración tan robusta como la de
    conocimiento/interés.
    """
    ruta = os.path.join(DATA_DIR, "usuarios_entrenados.json")
    if not os.path.exists(ruta):
        return None
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️  No se pudo leer usuarios_entrenados.json: {e}")
        return None


def analizar_pesos(indice):
    if not indice:
        print("(no se encontró data/usuarios_entrenados.json — se omite el análisis de pesos)")
        return None

    datos = {"novelty_mean": [], "novelty_var": [],
             "interest_mean": [], "interest_var": [],
             "bias_mean": [], "bias_var": []}

    for u in indice:
        pesos = u.get("weights", {})
        for factor, clave in [("novelty", "novelty"), ("interest", "interest"), ("bias", "bias")]:
            f = pesos.get(factor, {})
            if f.get("mean") is not None:
                datos[f"{clave}_mean"].append(float(f["mean"]))
            if f.get("variance") is not None:
                datos[f"{clave}_var"].append(float(f["variance"]))

    print(f"\n⚠️  ADVERTENCIA: solo {len(indice)} muestras (una por usuario), no {len(indice)*100}+ "
          f"como con conocimiento/interés. Tratar estos números como orientativos, no como una "
          f"calibración tan sólida como KNOW_MIN/MAX.\n")

    resultados = {}
    for nombre_bonito, clave in [
        ("novelty_weights.mean",  "novelty_mean"),
        ("novelty_weights.variance", "novelty_var"),
        ("interest_weights.mean", "interest_mean"),
        ("interest_weights.variance", "interest_var"),
        ("bias_weights.mean",     "bias_mean"),
        ("bias_weights.variance", "bias_var"),
    ]:
        print("-" * 70)
        resultados[clave] = resumen(datos[clave], nombre_bonito)
    return resultados


def main():
    perfiles = cargar_perfiles()
    print(f"📂 Leyendo {len(perfiles)} perfiles desde {DATA_DIR}\n")

    know_means, know_vars = [], []
    int_means, int_vars = [], []
    n_kcs_por_usuario = []

    for perfil in perfiles:
        kcs = perfil.get("knowledge_components", [])
        n_kcs_por_usuario.append(len(kcs))
        for kc in kcs:
            if kc.get("mean") is not None:
                know_means.append(float(kc["mean"]))
            if kc.get("variance") is not None:
                know_vars.append(float(kc["variance"]))
            if kc.get("interest_mean") is not None:
                int_means.append(float(kc["interest_mean"]))
            if kc.get("interest_var") is not None:
                int_vars.append(float(kc["interest_var"]))

    print(f"👥 Usuarios: {len(perfiles)}")
    print(f"📊 Knowledge Components totales: {sum(n_kcs_por_usuario)}  "
          f"(mín {min(n_kcs_por_usuario) if n_kcs_por_usuario else 0}, "
          f"máx {max(n_kcs_por_usuario) if n_kcs_por_usuario else 0} por usuario)\n")

    print("=" * 70)
    print("CONOCIMIENTO — mean")
    print("=" * 70)
    r_know_mean = resumen(know_means, "knowledge.mean")

    print()
    print("=" * 70)
    print("CONOCIMIENTO — variance")
    print("=" * 70)
    r_know_var = resumen(know_vars, "knowledge.variance")

    print()
    print("=" * 70)
    print("INTERÉS — mean")
    print("=" * 70)
    r_int_mean = resumen(int_means, "interest_mean")

    print()
    print("=" * 70)
    print("INTERÉS — variance")
    print("=" * 70)
    r_int_var = resumen(int_vars, "interest_var")

    # ── Sugerencia de rangos para logic/dominio.py ──────────────────────
    # Se usan los percentiles 1 / 99 en vez del mínimo/máximo absoluto: el
    # extremo absoluto suele estar dominado por 1 o 2 usuarios con
    # muchísimas interacciones en una sola dirección (todo enganche o
    # todo desenganche), y calibrar el 0%/100% de la escala visual contra
    # ese único caso deja a todo el resto amontonado en una franja angosta
    # — exactamente el problema que se está corrigiendo.
    print()
    print("=" * 70)
    print("SUGERENCIA para logic/dominio.py (basada en percentiles 1/99)")
    print("=" * 70)
    if r_know_mean:
        print(f"  KNOW_MIN, KNOW_MAX = {r_know_mean['p1']:.3f}, {r_know_mean['p99']:.3f}")
    if r_int_mean:
        print(f"  INT_MIN,  INT_MAX  = {r_int_mean['p1']:.3f}, {r_int_mean['p99']:.3f}")
    if r_know_var:
        print(f"  VAR_MIN,  VAR_MAX  = {r_know_var['min']:.4f}, {r_know_var['max']:.4f}")
    # ── Cortes de las etiquetas por CUANTILES de los datos reales ───────
    # Las etiquetas (Alto/Medio/...) son criterio NORMATIVO: ubican al
    # tema respecto de la población real. Conocimiento: cuartiles (4
    # etiquetas). Interés: terciles (3 etiquetas). Los valores empatados
    # reciben el RANGO PROMEDIO (convención estándar): un bloque de valores
    # idénticos que cruza un cuantil queda entero en una etiqueta, y el
    # corte es el punto medio entre dos valores distintos vecinos. Se
    # expresan en el % de la escala (p1/p99 de cada variable).
    cortes = None
    if r_know_mean and r_int_mean and len(know_means) > 4 and len(int_means) > 3:
        k_lo, k_hi = r_know_mean["p1"], r_know_mean["p99"]
        i_lo, i_hi = r_int_mean["p1"], r_int_mean["p99"]

        def cortes_rango_promedio(valores, cuantiles):
            from collections import Counter
            n = len(valores)
            cont = sorted(Counter(valores).items())
            menores, pos = 0, []
            for v, c in cont:
                pos.append((v, (menores + 0.5 * c) / n))
                menores += c
            salida = []
            for q in cuantiles:
                i = next((j for j, (_, p) in enumerate(pos) if p >= q), len(pos) - 1)
                i = min(max(i, 1), len(pos) - 1)
                salida.append((pos[i - 1][0] + pos[i][0]) / 2.0)
            return salida

        def a_pct(v, lo, hi):
            return min(100.0, max(0.0, (v - lo) / (hi - lo) * 100.0))

        kb, km, ka = cortes_rango_promedio(know_means, [0.25, 0.50, 0.75])
        im, ia = cortes_rango_promedio(int_means, [1 / 3, 2 / 3])
        cortes = {
            "UMBRAL_BASICO_PCT": a_pct(kb, k_lo, k_hi),
            "UMBRAL_MEDIO_PCT": a_pct(km, k_lo, k_hi),
            "UMBRAL_ALTO_PCT": a_pct(ka, k_lo, k_hi),
            "UMBRAL_INTERES_MEDIO_PCT": a_pct(im, i_lo, i_hi),
            "UMBRAL_INTERES_ALTO_PCT": a_pct(ia, i_lo, i_hi),
        }
        print()
        print("=" * 70)
        print("CORTES DE ETIQUETAS por cuantiles (conocimiento: cuartiles; interés: terciles)")
        print("=" * 70)
        print(f"  INT_MIN, INT_MAX  = {i_lo:.3f}, {i_hi:.3f}   (percentiles 1 y 99)")
        for nombre, valor in cortes.items():
            print(f"  {nombre} = {valor:.1f}")
        print("  (Con valores empatados el reparto no es exactamente 25 % o 33 % por")
        print("   etiqueta: ver calibracion_por_estrato.py e interes_masas.py.)")

    print()
    print("  Esto NO modifica ningún archivo del proyecto. Pasame el bloque")
    print("  de arriba (o el data/calibracion_dominio.json que se generó) y")
    print("  ajusto logic/dominio.py con estos valores reales.")

    reporte = {
        "n_usuarios": len(perfiles),
        "n_knowledge_components_total": sum(n_kcs_por_usuario),
        "knowledge_mean": r_know_mean,
        "knowledge_variance": r_know_var,
        "interest_mean": r_int_mean,
        "interest_variance": r_int_var,
        "cortes_etiquetas_por_cuantiles": cortes,
    }

    print()
    print("=" * 70)
    print("PESOS DEL MODELO (novelty / interest / bias) — orientativo, n = nº de usuarios")
    print("=" * 70)
    indice = cargar_indice_usuarios()
    r_pesos = analizar_pesos(indice)
    if r_pesos:
        reporte["pesos"] = r_pesos

    ruta_salida = os.path.join(DATA_DIR, "calibracion_dominio.json")
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Reporte completo guardado en: {ruta_salida}")


if __name__ == "__main__":
    # FIX: forzar UTF-8 en stdout/stderr,
    # mismo motivo que logic/api_backend.py y logic/entrenador.py.
    import io
    import sys
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
