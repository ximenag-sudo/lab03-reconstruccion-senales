# Laboratorio 3: Reconstrucción de señales y sensibilidad con NumPy

Autora: Ximena Criollo

Se reconstruyen señales a partir de lecturas de dos sensores con el modelo `b = A(c) x + ε`, donde `A(c) = [[1, c], [c, 1]]`. Se comparan `c = 0.2` y `c = 0.99` con las mismas señales y el mismo ruido, y se verifican los resultados con pruebas automatizadas.

## Parámetros del experimento

- Malla de señales: 21 × 21 = 441 señales con componentes entre −1 y 1.
- Convención: cada señal es una **columna**. `X` tiene forma (2, 441) y las lecturas `B` también.
- Ruido: normal, media 0, desviación estándar 0.01, `numpy.random.default_rng(7)`. Una sola realización, reutilizada para los dos valores de `c`.
- Umbral didáctico de error: 0.1 (norma euclidiana).

## Versiones usadas

Python 3.13.17, NumPy 2.5.4, Matplotlib 3.11.2, pytest 9.2.0 (Windows 11).

## Cómo reproducirlo desde cero (PowerShell)

```
git clone https://github.com/ximenag-sudo/lab03-reconstruccion-senales.git lab03
cd lab03
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
cd src
python experimento.py
cd ..
```

Si PowerShell bloquea la activación del entorno, ejecutar una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. En Linux o macOS: `python3 -m venv .venv` y `source .venv/bin/activate`.

## Dónde quedan los resultados

- `resultados/tablas.md`: tabla de diagnóstico matricial y tabla de reconstrucción.
- `resultados/error_reconstruccion.png`: figura de dos paneles con el error sobre la malla (misma escala de color).

## Estructura

- `src/modelo.py`: construcción de A, señales, ruido, lecturas y reconstrucción.
- `src/diagnostico.py`: autovalores, QR, SVD, κ₂ y estadísticas del error.
- `src/experimento.py`: ejecuta el experimento y genera tablas y figura.
- `tests/`: 27 pruebas en cinco grupos (caso manual, contratos, identidades, estadísticas, reproducibilidad).
- `informe.md`: informe con resultados y discusión.

## Predicción (escrita antes de ver los resultados)

Creo que el caso c = 0.99 será más sensible al ruido que c = 0.2, porque con c cercano a 1 los dos sensores miden casi lo mismo, así que las lecturas se parecen mucho aunque las señales reales sean distintas. Al separarlas, el ruido pequeño se confunde con esa diferencia mínima y se amplifica. Espero que el error de reconstrucción sea mayor en c = 0.99 que en c = 0.2. Además, creo que el residuo seguirá siendo pequeño en ambos casos.