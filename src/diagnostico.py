"""Diagnóstico matricial y estadístico del experimento."""
import numpy as np


def diagnostico_matricial(A):
    """Autovalores/autovectores, QR, SVD, kappa_2 y discrepancias de verificación."""
    lam, V = np.linalg.eigh(A)
    Q, R = np.linalg.qr(A)
    U, s, Vt = np.linalg.svd(A)

    kappa = s.max() / s.min()
    return {
        "autovalores": lam,
        "autovectores": V,
        "Q": Q, "R": R,
        "U": U, "s": s, "Vt": Vt,
        "kappa2": kappa,
        "err_autovec": np.max(np.abs(A @ V - V * lam)),
        "err_Q_ortog": np.max(np.abs(Q.T @ Q - np.eye(2))),
        "err_QR": np.max(np.abs(Q @ R - A)),
        "err_SVD": np.max(np.abs(U @ np.diag(s) @ Vt - A)),
    }


def estadisticas_error(X_hat, X, A, B, umbral=0.1):
    """Error, residuo y estadísticas. Señales en columnas: forma (2, N)."""
    error = X_hat - X
    residuo = A @ X_hat - B

    media = error.mean(axis=1)
    centrado = error - media[:, np.newaxis]
    std = np.sqrt((centrado ** 2).mean(axis=1))

    norma = np.linalg.norm(error, axis=0)
    supera = norma > umbral
    clase = np.where(supera, "supera", "no supera")
    i_max = int(np.argmax(norma))

    return {
        "error": error, "residuo": residuo,
        "media": media, "std": std, "std_numpy": error.std(axis=1, ddof=0),
        "norma": norma, "supera": supera, "clase": clase,
        "proporcion": float(supera.mean()),
        "i_max": i_max, "norma_max": float(norma[i_max]),
        "residuo_max": float(np.max(np.linalg.norm(residuo, axis=0))),
    }