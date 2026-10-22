"""
Estadísticas empíricas para calibrar ``logic/dominio.py``.

Calcula mínimo, máximo, percentiles y promedio de la media y la varianza de
los Knowledge Components (conocimiento e interés) de todos los usuarios ya
entrenados, leyendo los ``visual_profile_*.json`` que genera
``logic/entrenador.py``. También resume los pesos del modelo (novedad,
interés y sesgo) del índice ``usuarios_entrenados.json``.

Solo lee la carpeta analizada. Lo único que escribe es el reporte
``calibracion_dominio.json`` dentro de esa misma carpeta; no modifica
ningún archivo del proyecto.

Uso, desde la raíz del proyecto y con el entorno virtual activo:

    python calibrar_dominio.py                    # analiza data/
    python calibrar_dominio.py data_calibracion   # analiza otra carpeta

Debe ejecutarse después de entrenar con datos reales de PEEK
(``python -m logic.entrenador`` o ``python entrenar_gran_escala.py``).
"""
import glob
import json
import os
import statistics
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, sys.argv[1] if len(sys.argv) > 1 else "data")


def cargar_perfiles():
    """
    Carga todos los perfiles ``visual_profile_*.json`` de la carpeta analizada.

    Los archivos que no se pueden leer se informan y se omiten. Si no hay
    ningún perfil, informa cómo generarlos y termina el programa.

    Returns:
        list[dict]: Perfiles leídos, uno por usuario.
    """
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
    """
    Calcula un percentil por interpolación lineal, sin numpy ni pandas.

    Args:
        valores_ordenados (list[float]): Valores ya ordenados de menor a mayor.
        p (float): Percentil buscado, entre 0 y 100.

    Returns:
        float | None: El valor del percentil, o ``None`` si la lista está
        vacía.
    """
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
    """
    Calcula e imprime las estadísticas descriptivas de una serie de valores.

    Args:
        valores (list[float]): Valores a resumir, sin ordenar.
        nombre (str): Nombre de la variable, usado al imprimir.

    Returns:
        dict | None: Diccionario con ``n``, ``min``, ``p1``, ``p5``, ``p25``,
        ``mediana``, ``p75``, ``p95``, ``p99``, ``max`` y ``promedio``, o
        ``None`` si no hay valores.
    """
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
    Lee el índice global ``usuarios_entrenados.json``.

    Ese índice trae los pesos del modelo (novedad, interés y sesgo) de cada
    usuario. A diferencia de los Knowledge Components, hay una sola muestra
    por usuario.

    Returns:
        list[dict] | None: El índice, o ``None`` si el archivo no existe o no
        se puede leer.
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
    """
    Resume la media y la varianza de los tres pesos del modelo.

    Imprime una advertencia sobre el tamaño de la muestra (una por usuario) y
    las estadísticas de cada peso.

    Args:
        indice (list[dict] | None): Índice devuelto por
            ``cargar_indice_usuarios``.

    Returns:
        dict | None: Resumen de cada serie (``novelty_mean``, ``novelty_var``,
        ``interest_mean``, ``interest_var``, ``bias_mean`` y ``bias_var``), o
        ``None`` si no hay índice.
    """
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
    """
    Ejecuta la calibración completa e imprime y guarda el reporte.

    Resume la media y la varianza de conocimiento e interés de todos los
    Knowledge Components, sugiere los valores de ``KNOW_MIN``/``KNOW_MAX``,
    ``INT_MIN``/``INT_MAX`` y ``VAR_MIN``/``VAR_MAX`` para ``logic/dominio.py``
    (con los percentiles 1 y 99 en lugar del mínimo y el máximo absolutos,
    para que pocos casos extremos no definan la escala) y analiza los pesos
    del modelo. El reporte completo se guarda en ``calibracion_dominio.json``
    dentro de la carpeta analizada.

    Returns:
        None: El resultado se informa por salida estándar y en el archivo.
    """
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
    import io
    import sys
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
