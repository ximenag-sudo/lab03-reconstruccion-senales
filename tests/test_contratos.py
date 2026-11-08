"""Grupo 2: contratos de entrada y caso singular."""
import numpy as np
import pytest

import modelo


def test_rechaza_formas_incompatibles():
    A = modelo.construir_matriz(0.2)
    X = np.ones((2, 5))
    with pytest.raises(ValueError):
        modelo.calcular_lecturas(A, np.ones((3, 5)), np.zeros((3, 5)))
    with pytest.raises(ValueError):
        modelo.calcular_lecturas(A, X, np.zeros((2, 4)))
    with pytest.raises(ValueError):
        modelo.calcular_lecturas(np.eye(3), X, np.zeros_like(X))
    with pytest.raises(ValueError):
        modelo.reconstruir(A, np.ones((3, 5)))


@pytest.mark.parametrize("valor", [np.nan, np.inf, -np.inf])
def test_rechaza_valores_no_finitos(valor):
    A = modelo.construir_matriz(0.2)
    X = np.ones((2, 3))
    X_malo = X.copy()
    X_malo[0, 1] = valor
    with pytest.raises(ValueError):
        modelo.calcular_lecturas(A, X_malo, np.zeros_like(X))
    with pytest.raises(ValueError):
        modelo.reconstruir(A, X_malo)
    with pytest.raises(ValueError):
        modelo.construir_matriz(valor)


def test_c_igual_1_es_singular_y_se_comunica():
    A = modelo.construir_matriz(1.0)
    assert np.linalg.matrix_rank(A) == 1
    with pytest.raises(ValueError, match="singular"):
        modelo.reconstruir(A, np.ones((2, 3)))


def test_c_099_no_es_singular_aunque_sea_sensible():
    A = modelo.construir_matriz(0.99)
    assert np.linalg.matrix_rank(A) == 2
    X = np.array([[1.0], [2.0]])
    X_hat = modelo.reconstruir(A, A @ X)
    np.testing.assert_allclose(X_hat, X, atol=1e-10)