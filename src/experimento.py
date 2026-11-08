"""Experimento completo: genera tablas y la figura."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import modelo
import diagnostico

VALORES_C = (0.2, 0.99)
N = 21
UMBRAL = 0.1


def ejecutar():
    X = modelo.generar_senales(N)
    ruido = modelo.generar_ruido(X.shape[1])
    resultados = {}
    for c in VALORES_C:
        A = modelo.construir_matriz(c)
        B = modelo.calcular_lecturas(A, X, ruido)
        X_hat = modelo.reconstruir(A, B)
        resultados[c] = {
            "A": A,
            "mat": diagnostico.diagnostico_matricial(A),
            "rec": diagnostico.estadisticas_error(X_hat, X, A, B, UMBRAL),
        }
    return X, resultados


def tabla_matricial(resultados):
    lineas = ["| c | autovalores | valores singulares | kappa2 | err Av=lv | err QtQ=I | err QR=A | err USVt=A |",
              "|---|---|---|---|---|---|---|---|"]
    for c, r in resultados.items():
        m = r["mat"]
        lineas.append(
            f"| {c} | {np.round(m['autovalores'], 4).tolist()} | {np.round(m['s'], 4).tolist()} "
            f"| {m['kappa2']:.2f} | {m['err_autovec']:.1e} | {m['err_Q_ortog']:.1e} "
            f"| {m['err_QR']:.1e} | {m['err_SVD']:.1e} |")
    return "\n".join(lineas)


def tabla_reconstruccion(resultados):
    lineas = ["| c | media err (x1, x2) | desv. est. (x1, x2) | error euclid. máx | proporción > umbral | residuo máx |",
              "|---|---|---|---|---|---|"]
    for c, r in resultados.items():
        e = r["rec"]
        lineas.append(
            f"| {c} | {np.round(e['media'], 5).tolist()} | {np.round(e['std'], 5).tolist()} "
            f"| {e['norma_max']:.4f} | {e['proporcion']:.3f} | {e['residuo_max']:.1e} |")
    return "\n".join(lineas)


def figura(resultados, ruta):
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    vmax = max(r["rec"]["norma"].max() for r in resultados.values())
    for eje, (c, r) in zip(ejes, resultados.items()):
        mapa = r["rec"]["norma"].reshape(N, N)
        im = eje.imshow(mapa, extent=(-1, 1, -1, 1), origin="lower",
                        vmin=0, vmax=vmax, cmap="viridis")
        eje.set_title(f"c = {c}")
        eje.set_xlabel("x1 (normalizado)")
    ejes[0].set_ylabel("x2 (normalizado)")
    fig.colorbar(im, ax=ejes, label="||x_hat - x|| (normalizado)")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("../resultados", exist_ok=True)
    X, res = ejecutar()
    t1, t2 = tabla_matricial(res), tabla_reconstruccion(res)
    with open("../resultados/tablas.md", "w", encoding="utf-8") as f:
        f.write("## Diagnóstico matricial\n\n" + t1 + "\n\n## Reconstrucción\n\n" + t2 + "\n")
    figura(res, "../resultados/error_reconstruccion.png")
    print(t1, "\n")
    print(t2)