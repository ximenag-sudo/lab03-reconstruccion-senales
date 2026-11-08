"""Grupo 3: identidades de autovectores, QR y SVD. Se comprueban relaciones, no signos."""
import numpy as np
import pytest

import modelo
import diagnostico

TOL = 1e-12


@pytest.fixture(params=[0.2, 0.99])
def caso(request):
    c = request.param
    A = modelo.construir_matriz(c)
    return c, A, diagnostico.diagnostico_matricial(A)


def test_autovectores(caso):
    c, A, d = caso
    for lam, v in zip(d["autovalores"], d["autovectores"].T):
        np.testing.assert_allclose(A @ v, lam * v, atol=TOL)
    np.testing.assert_allclose(np.sort(d["autovalores"]), [1 - c, 1 + c], atol=TOL)


def test_direccion_dificil_es_la_diferencia(caso):
    c, A, d = caso
    v = d["autovectores"][:, np.argmin(d["autovalores"])]
    assert abs(v @ np.array([1.0, -1.0]) / np.sqrt(2)) == pytest.approx(1.0, abs=TOL)


def test_qr(caso):
    _, A, d = caso
    Q, R = d["Q"], d["R"]
    np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=TOL)
    np.testing.assert_allclose(Q @ R, A, atol=TOL)
    np.testing.assert_allclose(R, np.triu(R), atol=TOL)


def test_svd(caso):
    c, A, d = caso
    U, s, Vt = d["U"], d["s"], d["Vt"]
    np.testing.assert_allclose(U @ np.diag(s) @ Vt, A, atol=TOL)
    np.testing.assert_allclose(s, [1 + c, 1 - c], atol=TOL)
    assert d["kappa2"] == pytest.approx((1 + c) / (1 - c), rel=1e-10)