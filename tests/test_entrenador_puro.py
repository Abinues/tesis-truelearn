"""
Tests de ``seleccionar_usuarios()``, función pura de ``logic/entrenador.py``.

La función solo filtra tuplas ``(uid, eventos)`` por cantidad y no usa
``truelearn``. Se separó de ``test_entrenamiento.py`` porque ese módulo tiene
un ``importorskip("truelearn")`` global y se omitiría entero sin la librería;
estos tests deben correr siempre. Es posible porque el import de ``truelearn``
en ``logic/entrenador.py`` está dentro de ``entrenar_usuario()``.
"""
from logic.entrenador import seleccionar_usuarios


def test_seleccionar_usuarios_separa_por_umbral():
    """Separa usuarios con "muchas" y "pocas" interacciones y descarta el resto.

    Los umbrales son inclusivos: 40 eventos cuentan como "muchas" y 10 como
    "pocas"; 15 no califica en ninguno.
    """
    train = [
        (1, [0] * 50),
        (2, [0] * 5),
        (3, [0] * 15),
        (4, [0] * 40),
        (5, [0] * 10),
    ]
    muchas, pocas = seleccionar_usuarios(train, n_muchas=10, n_pocas=10,
                                          umbral_muchas=40, umbral_pocas_max=10)
    uids_muchas = {uid for uid, _ in muchas}
    uids_pocas  = {uid for uid, _ in pocas}
    assert uids_muchas == {1, 4}
    assert uids_pocas == {2, 5}
    assert 3 not in uids_muchas and 3 not in uids_pocas


def test_seleccionar_usuarios_respeta_el_limite_n():
    """No devuelve más usuarios que el límite ``n`` de cada grupo."""
    train = [(uid, [0] * 50) for uid in range(20)]
    muchas, pocas = seleccionar_usuarios(train, n_muchas=3, n_pocas=3,
                                          umbral_muchas=40, umbral_pocas_max=10)
    assert len(muchas) == 3
    assert len(pocas) == 0


def test_seleccionar_usuarios_sin_usuarios_que_califiquen():
    """Devuelve listas vacías si ningún usuario cumple los umbrales."""
    train = [(1, [0] * 20), (2, [0] * 25)]
    muchas, pocas = seleccionar_usuarios(train, n_muchas=5, n_pocas=5,
                                          umbral_muchas=40, umbral_pocas_max=10)
    assert muchas == []
    assert pocas == []


def test_seleccionar_usuarios_corta_apenas_se_llenan_ambas_cuotas():
    """Cubre el corte temprano: al llenar ambas cuotas no se recorre el resto."""
    train = [
        (1, [0] * 50),
        (2, [0] * 5),
        (3, [0] * 999),
    ]
    muchas, pocas = seleccionar_usuarios(train, n_muchas=1, n_pocas=1,
                                          umbral_muchas=40, umbral_pocas_max=10)
    assert len(muchas) == 1 and len(pocas) == 1
    assert 3 not in {uid for uid, _ in muchas} | {uid for uid, _ in pocas}
