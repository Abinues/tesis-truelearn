"""
Llena con correcciones explícitas a los aprendices ya entrenados.

Sirve para que el dashboard se vea cargado en una demostración: las páginas
«Predicción y correcciones», «Historial de correcciones» y «Evolución temporal»
quedan con datos en lugar de mostrarse vacías.

Cada corrección pasa por el mismo endpoint que usa la interfaz
(``POST /feedback/<uid>``, vía el cliente de pruebas de Flask), así que el
modelo ``.pkl`` y el perfil ``.json`` quedan igual que si un usuario hubiera
corregido a mano. No hace falta tener el backend levantado, pero conviene no
ejecutarlo mientras se entrena un modelo.

Las fechas se reparten entre ahora y los últimos ``--dias`` días, en orden
creciente por aprendiz. El tipo de cada corrección se elige en el momento de
aplicarla, según el estado que el modelo tiene entonces para ese tema (por
ejemplo, es más probable «sé menos» si ya estima un conocimiento alto). Así,
aunque un tema se corrija muchas veces, sus valores oscilan en lugar de quedar
clavados en un extremo.

Uso, desde la raíz del proyecto::

    python llenar_correcciones.py
    python llenar_correcciones.py --minimo 20 --maximo 40 --dias 30
    python llenar_correcciones.py --uid 12345 --uid 67890
    python llenar_correcciones.py --solo-sin-correcciones --semilla 42
    python llenar_correcciones.py --completar --sin-backup
    python llenar_correcciones.py --simular

Antes de modificar nada se copian ``data/`` y ``models/`` a una carpeta
``backup_correcciones_<fecha>/`` (se desactiva con ``--sin-backup``). Para
deshacer, basta con restaurar esas dos carpetas. Si una ejecución se
interrumpe, ``--completar`` la reanuda sin repetir a quienes ya están listos;
conviene sumarle ``--sin-backup`` para conservar el respaldo original.
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from types import SimpleNamespace

from logic import api_backend as backend
from logic import dominio as D

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEGUNDOS_POR_DIA = 86400
MARGEN_SEGUNDOS = 600


def parsear_argumentos(argv=None):
    """
    Lee los argumentos de la línea de comandos.

    Args:
        argv (list[str] | None): Argumentos a analizar. Si es ``None`` se usan
            los de ``sys.argv``.

    Returns:
        argparse.Namespace: Opciones ya validadas: ``minimo``, ``maximo``,
        ``dias``, ``uid``, ``max_usuarios``, ``solo_sin_correcciones``,
        ``semilla``, ``sin_backup`` y ``simular``.
    """
    p = argparse.ArgumentParser(
        description="Llena con correcciones explícitas a los aprendices entrenados.")
    p.add_argument("--minimo", type=int, default=50,
                   help="Mínimo de correcciones por aprendiz (default: 50).")
    p.add_argument("--maximo", type=int, default=100,
                   help="Máximo de correcciones por aprendiz (default: 100).")
    p.add_argument("--dias", type=int, default=21,
                   help="Las fechas se reparten en los últimos N días (default: 21).")
    p.add_argument("--uid", type=int, action="append", default=None,
                   help="Aprendiz a corregir; se puede repetir. Por defecto, todos.")
    p.add_argument("--max-usuarios", type=int, default=None,
                   help="Procesa solo los primeros N aprendices.")
    p.add_argument("--solo-sin-correcciones", action="store_true",
                   help="Omite a los aprendices que ya tienen correcciones.")
    p.add_argument("--completar", action="store_true",
                   help="Omite a quienes ya tienen --minimo correcciones o más y, a "
                        "los demás, les agrega las que faltan. Sirve para reanudar "
                        "una ejecución interrumpida.")
    p.add_argument("--semilla", type=int, default=None,
                   help="Semilla aleatoria, para repetir exactamente el resultado.")
    p.add_argument("--sin-backup", action="store_true",
                   help="No copiar data/ y models/ antes de modificar.")
    p.add_argument("--simular", action="store_true",
                   help="Muestra qué haría, sin escribir nada.")
    args = p.parse_args(argv)
    if args.minimo < 1 or args.maximo < args.minimo:
        p.error("--minimo debe ser >= 1 y --maximo >= --minimo.")
    if args.dias < 1:
        p.error("--dias debe ser >= 1.")
    return args


def listar_uids(uids_pedidos=None, max_usuarios=None):
    """
    Obtiene los aprendices entrenados que tienen modelo y perfil en disco.

    Args:
        uids_pedidos (list[int] | None): Si se indica, solo se consideran
            estos aprendices.
        max_usuarios (int | None): Si se indica, se devuelven a lo sumo los
            primeros N.

    Returns:
        list[int]: Identificadores de los aprendices a procesar.

    Raises:
        FileNotFoundError: Si no existe ``data/usuarios_entrenados.json``.
    """
    ruta = os.path.join(backend.DATA_DIR, "usuarios_entrenados.json")
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No existe {ruta}. Entrená primero los modelos (entrenar_gran_escala.py "
            "o el botón de administración del dashboard).")
    with open(ruta, "r", encoding="utf-8") as f:
        indice = json.load(f)

    uids = [int(u["user_id"]) for u in indice]
    if uids_pedidos:
        pedidos = set(uids_pedidos)
        faltan = pedidos - set(uids)
        if faltan:
            print(f"⚠️  No están en el índice y se omiten: {sorted(faltan)}")
        uids = [u for u in uids if u in pedidos]

    disponibles = []
    for uid in uids:
        ruta_modelo = os.path.join(backend.MODEL_DIR, f"model_u_{uid}.pkl")
        ruta_perfil = os.path.join(backend.DATA_DIR, f"visual_profile_{uid}.json")
        if os.path.exists(ruta_modelo) and os.path.exists(ruta_perfil):
            disponibles.append(uid)
        else:
            print(f"⚠️  Aprendiz #{uid}: falta el modelo o el perfil; se omite.")

    if max_usuarios is not None:
        disponibles = disponibles[:max_usuarios]
    return disponibles


def elegir_tipo(kc, rng):
    """
    Elige un tipo de corrección verosímil para un tema.

    Las probabilidades dependen de lo que el sistema estima hoy: si el
    conocimiento ya es alto, «sé menos» es más probable que «sé más», y lo
    mismo con el interés. Cerca de los extremos (por encima de 85 % o por
    debajo de 15 %) la corrección que empujaría aún más hacia ese extremo
    casi no se elige, para que pocos temas queden en 0 % o 100 %.

    Args:
        kc (dict): Componente de conocimiento del perfil, con las claves
            ``"mean"`` e ``"interest_mean"``.
        rng (random.Random): Generador aleatorio a usar.

    Returns:
        str: Uno de ``"sé más"``, ``"sé menos"``, ``"me interesa"`` o
        ``"no me interesa"``.
    """
    pc = D.pct_conocimiento(kc.get("mean", 0.0))
    pi = D.pct_interes(kc.get("interest_mean", 0.0))
    pesos = {
        "sé más":         3 if pc < 60 else (1 if pc < 85 else 0.2),
        "sé menos":       3 if pc > 40 else (1 if pc > 15 else 0.2),
        "me interesa":    3 if pi < 50 else (1 if pi < 85 else 0.2),
        "no me interesa": 3 if pi > 30 else (1 if pi > 15 else 0.2),
    }
    return rng.choices(list(pesos), weights=list(pesos.values()), k=1)[0]


def generar_plan(perfil, rng, minimo, maximo, dias, ahora):
    """
    Arma la lista de correcciones a aplicar a un aprendiz.

    Como máximo un 75 % de las correcciones recae en temas distintos y el
    resto repite temas ya corregidos, para que se vea evolución sobre un mismo
    tema. Las fechas se reparten al azar entre ``ahora - dias`` y unos minutos
    antes de ``ahora``, y quedan ordenadas de la más antigua a la más reciente.
    El tipo de corrección no se fija aquí: lo elige ``aplicar_plan`` en cada
    paso.

    Args:
        perfil (dict): Perfil del aprendiz, con ``"knowledge_components"``.
        rng (random.Random): Generador aleatorio a usar.
        minimo (int): Mínimo de correcciones.
        maximo (int): Máximo de correcciones.
        dias (int): Amplitud, en días, del rango de fechas.
        ahora (float): Marca de tiempo actual, en segundos.

    Returns:
        list[tuple[float, dict]]: Tuplas ``(timestamp, kc)`` en orden
        cronológico. Lista vacía si el perfil no tiene temas.
    """
    kcs = perfil.get("knowledge_components", [])
    if not kcs:
        return []

    total = rng.randint(minimo, maximo)
    distintos = min(len(kcs), max(1, round(total * 0.75)))
    temas = rng.sample(kcs, distintos)
    temas += [rng.choice(temas) for _ in range(total - distintos)]
    rng.shuffle(temas)

    inicio = ahora - dias * SEGUNDOS_POR_DIA
    fin = ahora - MARGEN_SEGUNDOS
    fechas = sorted(rng.uniform(inicio, fin) for _ in temas)
    return list(zip(fechas, temas, strict=True))


def aplicar_plan(client, uid, plan, rng):
    """
    Aplica las correcciones de un aprendiz a través del endpoint de feedback.

    Para cada corrección reemplaza temporalmente el reloj del backend, de modo
    que el registro y los Knowledge Components queden con la fecha simulada.
    El reloj original se restaura siempre al terminar. El tipo se elige con
    ``elegir_tipo`` a partir del estado más reciente del tema, que se toma de
    la respuesta del backend a la corrección anterior.

    Args:
        client (flask.testing.FlaskClient): Cliente de pruebas del backend.
        uid (int): Identificador del aprendiz.
        plan (list[tuple[float, dict]]): Salida de ``generar_plan``.
        rng (random.Random): Generador aleatorio para elegir los tipos.

    Returns:
        tuple[int, list[str]]: Cantidad de correcciones aplicadas y lista de
        mensajes de error de las que fallaron. Una excepción del backend (por
        ejemplo, un archivo bloqueado en Windows) se registra como error y el
        resto de las correcciones continúa.
    """
    reloj_original = backend.time
    estado = {kc["topic_id"]: kc for _, kc in plan}
    aplicadas = 0
    errores = []
    try:
        for ts, kc in plan:
            tipo = elegir_tipo(estado[kc["topic_id"]], rng)
            backend.time = SimpleNamespace(time=lambda ts=ts: ts)
            try:
                r = client.post(f"/feedback/{uid}", json={
                    "topic_id":   kc["topic_id"],
                    "topic_name": kc.get("topic_name", f"ID_{kc['topic_id']}"),
                    "tipo":       tipo,
                })
            except Exception as e:
                errores.append(f"{tipo} sobre tema {kc['topic_id']}: {type(e).__name__}: {e}")
                continue
            if r.status_code == 200:
                aplicadas += 1
                for nuevo in r.get_json()["perfil"]["knowledge_components"]:
                    if nuevo["topic_id"] == kc["topic_id"]:
                        estado[kc["topic_id"]] = nuevo
                        break
            else:
                detalle = (r.get_json(silent=True) or {}).get("error", r.status_code)
                errores.append(f"{tipo} sobre tema {kc['topic_id']}: {detalle}")
    finally:
        backend.time = reloj_original
    return aplicadas, errores


def hacer_backup():
    """
    Copia ``data/`` y ``models/`` a una carpeta con la fecha actual.

    Returns:
        str: Ruta de la carpeta de respaldo creada.
    """
    base = os.path.dirname(os.path.abspath(backend.DATA_DIR))
    destino = os.path.join(base, "backup_correcciones_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(destino)
    for origen in (backend.DATA_DIR, backend.MODEL_DIR):
        if os.path.isdir(origen):
            shutil.copytree(origen, os.path.join(destino, os.path.basename(origen)))
    return destino


def main(argv=None):
    """
    Ejecuta el llenado de correcciones.

    Args:
        argv (list[str] | None): Argumentos de línea de comandos; si es
            ``None`` se usan los de ``sys.argv``.

    Returns:
        int: Código de salida: 0 si todo salió bien, 1 si no había
        aprendices que procesar o alguna corrección falló.
    """
    args = parsear_argumentos(argv)
    rng = random.Random(args.semilla)

    try:
        uids = listar_uids(args.uid, args.max_usuarios)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    if not uids:
        print("❌ No hay aprendices con modelo y perfil para procesar.")
        return 1

    if not args.simular and not args.sin_backup:
        print(f"💾 Respaldo creado en: {hacer_backup()}")

    backend.app.config["TESTING"] = True
    client = backend.app.test_client()
    ahora = time.time()

    total_aplicadas = 0
    total_errores = 0
    procesados = 0
    inicio = time.time()

    for i, uid in enumerate(uids, start=1):
        perfil = backend.cargar_perfil(uid)
        previas = len(perfil.get("explicit_feedback", []))
        if args.solo_sin_correcciones and previas > 0:
            print(f"[{i}/{len(uids)}] #{uid}: ya tiene {previas} correcciones; se omite.")
            continue
        minimo, maximo = args.minimo, args.maximo
        if args.completar:
            if previas >= args.minimo:
                print(f"[{i}/{len(uids)}] #{uid}: ya tiene {previas} correcciones; se omite.")
                continue
            minimo = args.minimo - previas
            maximo = max(minimo, args.maximo - previas)

        plan = generar_plan(perfil, rng, minimo, maximo, args.dias, ahora)
        if not plan:
            print(f"[{i}/{len(uids)}] #{uid}: sin temas; se omite.")
            continue

        if args.simular:
            print(f"[{i}/{len(uids)}] #{uid}: aplicaría {len(plan)} correcciones "
                  f"(ya tiene {previas}).")
            total_aplicadas += len(plan)
            procesados += 1
            continue

        aplicadas, errores = aplicar_plan(client, uid, plan, rng)
        total_aplicadas += aplicadas
        total_errores += len(errores)
        procesados += 1
        estado = "✅" if not errores else "⚠️"
        print(f"[{i}/{len(uids)}] {estado} #{uid}: {aplicadas}/{len(plan)} correcciones "
              f"(total ahora: {previas + aplicadas}).")
        for e in errores:
            print(f"      ✗ {e}")

    verbo = "Se aplicarían" if args.simular else "Se aplicaron"
    print(f"\n{verbo} {total_aplicadas} correcciones en {procesados} aprendices "
          f"({time.time() - inicio:.1f} s).")
    if total_errores:
        print(f"⚠️  {total_errores} correcciones fallaron; revisá los mensajes de arriba.")
    if not args.simular and total_aplicadas:
        print("Recargá el dashboard (F5) para ver los cambios.")
    return 1 if total_errores else 0


if __name__ == "__main__":
    sys.exit(main())
