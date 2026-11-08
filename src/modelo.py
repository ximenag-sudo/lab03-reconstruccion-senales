"""Modelo matricial de dos sensores: b = A(c) x + ruido.

Convención: cada señal es una COLUMNA. Con N señales:
    X: (2, N)   componentes x1 (fila 0) y x2 (fila 1)
    A: (2, 2)
    B: (2, N)   lecturas de los sensores 1 y 2
"""
import numpy as np


def construir_matriz(c):
    """Devuelve A(c) = [[1, c], [c, 1]]."""
    if not np.isfinite(c):
        raise ValueError("c debe ser un número finito")
    return np.array([[1.0, c], [c, 1.0]])


def generar_senales(n=21):
    """Malla n x n de componentes entre -1 y 1. Devuelve X de forma (2, n*n).

    meshgrid devuelve dos matrices (n, n): en x1 varía la columna y en x2
    varía la fila. ravel() las aplana y vstack las apila: fila 0 = x1, fila 1 = x2.
    """
    valores = np.linspace(-1.0, 1.0, n)
    x1, x2 = np.meshgrid(valores, valores)
    return np.vstack([x1.ravel(), x2.ravel()])


def generar_ruido(n_senales, sigma=0.01, semilla=7):
    """Un ruido normal por lectura y por señal: forma (2, n_senales)."""
    rng = np.random.default_rng(semilla)
    return rng.normal(0.0, sigma, size=(2, n_senales))


def _validar_finitos(*arreglos):
    for arr in arreglos:
        if not np.all(np.isfinite(arr)):
            raise ValueError("Hay valores no finitos (NaN o inf)")


def calcular_lecturas(A, X, ruido):
    """B = A @ X + ruido. Exige formas compatibles."""
    if A.shape != (2, 2):
        raise ValueError(f"A debe ser (2, 2), no {A.shape}")
    if X.ndim != 2 or X.shape[0] != 2:
        raise ValueError(f"X debe ser (2, N), no {X.shape}")
    if ruido.shape != X.shape:
        raise ValueError(f"ruido {ruido.shape} debe tener la forma de X {X.shape}")
    _validar_finitos(A, X, ruido)
    return A @ X + ruido


def reconstruir(A, B):
    """Resuelve A x = b para todas las columnas de B en una sola llamada."""
    if A.shape != (2, 2):
        raise ValueError(f"A debe ser (2, 2), no {A.shape}")
    if B.ndim != 2 or B.shape[0] != 2:
        raise ValueError(f"B debe ser (2, N), no {B.shape}")
    _validar_finitos(A, B)
    if np.linalg.matrix_rank(A) < 2:
        raise ValueError("A es singular (c = 1): no se puede reconstruir")
    return np.linalg.solve(A, B)