"""Grupo 5: reproducibilidad del ruido y conservación de las señales originales."""
import numpy as np

import modelo
import diagnostico


def test_ruido_reproducible_con_la_misma_semilla():
    r1 = modelo.generar_ruido(441, semilla=7)
    r2 = modelo.generar_ruido(441, semilla=7)
    np.testing.assert_array_equal(r1, r2)
    assert r1.shape == (2, 441)


def test_ruido_cambia_con_otra_semilla():
    r1 = modelo.generar_ruido(441, semilla=7)
    r2 = modelo.generar_ruido(441, semilla=8)
    assert not np.array_equal(r1, r2)


def test_orden_de_la_malla():
    X = modelo.generar_senales(21)
    assert X.shape == (2, 441)
    np.testing.assert_allclose(X[:, 0], [-1.0, -1.0])
    np.testing.assert_allclose(X[:, 1], [-0.9, -1.0])
    np.testing.assert_allclose(X[:, 21], [-1.0, -0.9])
    np.testing.assert_allclose(X[:, -1], [1.0, 1.0])


def test_senales_y_ruido_originales_no_se_modifican():
    X = modelo.generar_senales(21)
    ruido = modelo.generar_ruido(X.shape[1])
    X_copia, ruido_copia = X.copy(), ruido.copy()
    for c in (0.2, 0.99):
        A = modelo.construir_matriz(c)
        B = modelo.calcular_lecturas(A, X, ruido)
        X_hat = modelo.reconstruir(A, B)
        diagnostico.estadisticas_error(X_hat, X, A, B)
        assert not np.shares_memory(B, X)
    np.testing.assert_array_equal(X, X_copia)
    np.testing.assert_array_equal(ruido, ruido_copia)