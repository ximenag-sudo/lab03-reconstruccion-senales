"""Grupo 4: estadísticas, máscara y clasificación contra un arreglo pequeño calculado a mano."""
import numpy as np
import pytest

import modelo
import diagnostico

ERROR = np.array([[0.03, 0.0, 0.3, -0.6],
                  [0.04, 0.0, 0.4,  0.8]])


@pytest.fixture
def est():
    A = modelo.construir_matriz(0.0)
    X = np.zeros((2, 4))
    B = np.zeros((2, 4))
    return diagnostico.estadisticas_error(ERROR, X, A, B, umbral=0.1)


def test_media_por_componente(est):
    np.testing.assert_allclose(est["media"], [-0.0675, 0.31], atol=1e-12)
    assert est["media"].shape == (2,)


def test_desviacion_estandar_ddof0(est):
    esperado = np.sqrt([0.10816875, 0.1043])
    np.testing.assert_allclose(est["std"], esperado, atol=1e-12)
    np.testing.assert_allclose(est["std"], est["std_numpy"], atol=1e-12)


def test_norma_por_senal(est):
    np.testing.assert_allclose(est["norma"], [0.05, 0.0, 0.5, 1.0], atol=1e-12)
    assert est["norma"].shape == (4,)


def test_mascara_y_clasificacion(est):
    np.testing.assert_array_equal(est["supera"], [False, False, True, True])
    np.testing.assert_array_equal(
        est["clase"], ["no supera", "no supera", "supera", "supera"])
    assert est["proporcion"] == pytest.approx(0.5)


def test_error_maximo_y_residuo(est):
    assert est["i_max"] == 3
    assert est["norma_max"] == pytest.approx(1.0)
    assert est["residuo_max"] == pytest.approx(1.0)