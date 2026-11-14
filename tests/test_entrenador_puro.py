"""
tests/test_entrenador_puro.py
─────────────────────────────────────────────────────────────────────────────
Tests de seleccionar_usuarios(): una función pura de logic/entrenador.py
que no toca truelearn para nada (solo filtra tuplas (uid, eventos) por
cantidad). Se separó de tests/test_entrenamiento.py a propósito: ese
archivo tiene un importorskip("truelearn") a nivel de módulo porque el
resto de sus tests SÍ construyen un INKClassifier real, así que se
saltaría entero si truelearn no está instalado. Estos tests, en cambio,
no tienen ninguna razón para depender de eso — deben poder correr
siempre, con o sin la librería instalada (el import de truelearn en logic/entrenador.py está dentro de
entrenar_usuario(), no a nivel de módulo, justamente para que esto sea posible).
"""
from logic.entrenador import muestrear_estratificado, seleccionar_usuarios


def test_seleccionar_usuarios_separa_por_umbral():
    train = [
        (1, [0] * 50),   # "muchas" (>= 40)
        (2, [0] * 5),    # "pocas" (<= 10)
        (3, [0] * 15),   # ni una cosa ni la otra -> se descarta
        (4, [0] * 40),   # exactamente en el umbral de "muchas"
        (5, [0] * 10),   # exactamente en el umbral de "pocas"
    ]
    muchas, pocas = seleccionar_usuarios(train, n_muchas=10, n_pocas=10,
                                          umbral_muchas=40, umbral_pocas_max=10)
    uids_muchas = {uid for uid, _ in muchas}
    uids_pocas  = {uid for uid, _ in pocas}
    assert uids_muchas == {1, 4}
    assert uids_pocas == {2, 5}
    assert 3 not in uids_muchas and 3 not in uids_pocas


def test_seleccionar_usuarios_respeta_el_limite_n():
    train = [(uid, [0] * 50) for uid in range(20)]  # 20 usuarios, todos "muchas"
    muchas, pocas = seleccionar_usuarios(train, n_muchas=3, n_pocas=3,
                                          umbral_muchas=40, umbral_pocas_max=10)
    assert len(muchas) == 3
    assert len(pocas) == 0


def test_seleccionar_usuarios_sin_usuarios_que_califiquen():
    train = [(1, [0] * 20), (2, [0] * 25)]  # ninguno cae en "muchas" ni "pocas"
    muchas, pocas = seleccionar_usuarios(train, n_muchas=5, n_pocas=5,
                                          umbral_muchas=40, umbral_pocas_max=10)
    assert muchas == []
    assert pocas == []


def test_seleccionar_usuarios_corta_apenas_se_llenan_ambas_cuotas():
    """Cubre la rama de corte temprano (break): en cuanto se juntan
    suficientes 'muchas' y 'pocas', el resto del dataset ni se recorre."""
    train = [
        (1, [0] * 50),   # muchas #1
        (2, [0] * 5),    # pocas #1
        (3, [0] * 999),  # nunca debería procesarse: el break ya cortó antes
    ]
    muchas, pocas = seleccionar_usuarios(train, n_muchas=1, n_pocas=1,
                                          umbral_muchas=40, umbral_pocas_max=10)
    assert len(muchas) == 1 and len(pocas) == 1
    assert 3 not in {uid for uid, _ in muchas} | {uid for uid, _ in pocas}




# ── Muestreo aleatorio estratificado (terciles y percentil 95 de eventos) ──

def _poblacion():
    # 100 usuarios: 30 con 1-5 eventos, 30 con 20-25, 34 con 40-45 y 6 con 500+
    return ([(u, [0] * (1 + u % 5)) for u in range(30)] +
            [(100 + u, [0] * (20 + u % 6)) for u in range(30)] +
            [(200 + u, [0] * (40 + u % 6)) for u in range(34)] +
            [(300 + u, [0] * (500 + u)) for u in range(6)])


def test_muestreo_estratificado_cubre_los_cuatro_estratos():
    muestra, info = muestrear_estratificado(_poblacion(), n_por_estrato=5, semilla=1)
    por_tipo = {}
    for uid, evs, tipo in muestra:
        por_tipo.setdefault(tipo, []).append(len(evs))
    assert set(por_tipo) == {"pocas", "medias", "muchas", "muy_activas"}
    assert all(len(v) == 5 for v in por_tipo.values())
    c1, c2, c3 = info["corte_pocas"], info["corte_medias"], info["corte_muchas"]
    assert c1 < c2 < c3
    assert max(por_tipo["pocas"]) <= c1
    assert c1 < min(por_tipo["medias"]) and max(por_tipo["medias"]) <= c2
    assert c2 < min(por_tipo["muchas"]) and max(por_tipo["muchas"]) <= c3
    assert min(por_tipo["muy_activas"]) > c3
    assert sum(info["poblacion"].values()) == 100


def test_muy_activas_es_el_cinco_por_ciento_superior():
    _, info = muestrear_estratificado(_poblacion(), n_por_estrato=5, semilla=1)
    # percentil 95 de 100 usuarios: quedan 5 por encima
    assert info["poblacion"]["muy_activas"] == 5


def test_muestreo_estratificado_es_reproducible_y_depende_de_la_semilla():
    a, _ = muestrear_estratificado(_poblacion(), n_por_estrato=5, semilla=7)
    b, _ = muestrear_estratificado(_poblacion(), n_por_estrato=5, semilla=7)
    c, _ = muestrear_estratificado(_poblacion(), n_por_estrato=5, semilla=8)
    assert [u for u, _, _ in a] == [u for u, _, _ in b]
    assert [u for u, _, _ in a] != [u for u, _, _ in c]


def test_muestreo_estratificado_no_depende_del_orden_del_archivo():
    """Cada estrato se sortea al azar: no se queda con los primeros del archivo."""
    pob = _poblacion()
    muestra, _ = muestrear_estratificado(pob, n_por_estrato=5, semilla=3)
    primeros = {u for u, _ in pob[:5]}
    assert {u for u, _, t in muestra if t == "pocas"} != primeros


def test_muestreo_estratificado_ignora_usuarios_sin_eventos_y_acepta_poblacion_chica():
    muestra, info = muestrear_estratificado([(1, []), (2, [0]), (3, [0] * 4)], n_por_estrato=10)
    assert {u for u, _, _ in muestra} == {2, 3}
    vacio, info_vacio = muestrear_estratificado([], n_por_estrato=3)
    assert vacio == [] and info_vacio["corte_pocas"] is None
