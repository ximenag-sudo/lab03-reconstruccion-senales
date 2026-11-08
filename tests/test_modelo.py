import numpy as np
import pytest

import modelo

TOL = 1e-12


@pytest.mark.parametrize("c, b_esperado", [
    (0.2, [[1.4], [2.2]]),
    (0.99, [[2.98], [2.99]]),
])
def test_producto_matricial_contra_calculo_manual(c, b_esperado):
    A = modelo.construir_matriz(c)
    X = np.array([[1.0], [2.0]])
    B = A @ X
    np.testing.assert_allclose(B, b_esperado, atol=TOL)


@pytest.mark.parametrize("c", [0.2, 0.99])
def test_recuperacion_sin_ruido(c):
    A = modelo.construir_matriz(c)
    X = np.array([[1.0], [2.0]])
    B = modelo.calcular_lecturas(A, X, np.zeros_like(X))
    X_hat = modelo.reconstruir(A, B)
    np.testing.assert_allclose(X_hat, X, atol=TOL)