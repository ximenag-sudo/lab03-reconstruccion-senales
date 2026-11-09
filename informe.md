# Informe: Reconstrucción de señales y sensibilidad (Laboratorio 3)

Autora: Ximena Criollo

## 1. Modelo y decisiones

El modelo es `b = A(c) x + ε` con `A(c) = [[1, c], [c, 1]]`. Se comparan `c = 0.2` y `c = 0.99` con las mismas 441 señales (malla de 21 × 21 entre −1 y 1) y el mismo ruido normal (media 0, σ = 0.01, `default_rng(7)`).

- **Convención:** cada señal es una columna. `X` y `B` tienen forma (2, 441), `A` es (2, 2) y el ruido es (2, 441). Así `A @ X` produce todas las lecturas sin ciclos y `numpy.linalg.solve(A, B)` resuelve las 441 señales en una sola llamada.
- **Organización:** `modelo.py` (cálculo), `diagnostico.py` (verificación y estadísticas) y `experimento.py` (tablas y figura) están separados. Las pruebas están en `tests/`.
- **Contratos:** las funciones rechazan formas incompatibles y valores no finitos. `c = 1` se comunica como matriz singular (rango 1), mientras que `c = 0.99` no se trata como singular.
- **Tolerancias:** 1e-12 en las identidades matriciales (el redondeo observado es del orden de 1e-16) y 1e-10 al reconstruir sin ruido con `c = 0.99`.

**Predicción previa** (escrita antes de ejecutar el experimento): c = 0.99 sería más sensible al ruido porque los dos sensores miden casi lo mismo, y el residuo seguiría siendo pequeño en ambos casos.

## 2. Resultados

### Diagnóstico matricial

| c | autovalores | valores singulares | κ₂ | err Av=λv | err QᵀQ=I | err QR=A | err UΣVᵀ=A |
|---|---|---|---|---|---|---|---|
| 0.2 | [0.8, 1.2] | [1.2, 0.8] | 1.50 | 0.0e+00 | 4.4e-16 | 2.2e-16 | 2.5e-16 |
| 0.99 | [0.01, 1.99] | [1.99, 0.01] | 199.00 | 6.4e-17 | 4.4e-16 | 6.7e-16 | 7.8e-16 |

Las cuatro identidades se cumplen hasta el redondeo del computador (≈1e-16).

### Reconstrucción

| c | media del error (x1, x2) | desv. est. (x1, x2) | error euclidiano máx. | proporción > 0.1 | residuo máx. |
|---|---|---|---|---|---|
| 0.2 | [−0.00093, −0.00067] | [0.00987, 0.01023] | 0.0380 | 0.000 | 2.5e-16 |
| 0.99 | [−0.01075, 0.00979] | [0.66943, 0.66969] | 2.7945 | 0.902 | 5.0e-16 |

El umbral de 0.1 es un criterio didáctico fijo, no una tolerancia instrumental.

### Figura

![Error de reconstrucción sobre la malla](resultados/error_reconstruccion.png)

Los dos paneles usan la misma escala de color. Con c = 0.2 el error es casi uniforme y pequeño. Con c = 0.99 aparece un mosaico de errores grandes: el error depende del ruido de cada señal, que es aleatorio, y no de su posición en la malla.

## 3. Discusión

**¿Qué diferencia hay entre `*` y `@`, y qué errores de forma podrían pasar inadvertidos?**
`@` es el producto matricial (cada lectura mezcla las dos componentes) y `*` multiplica elemento a elemento. Con 441 señales, `A * X` falla por forma, pero con una sola señal (`X` de forma (2, 1)) o con N = 2, `A * X` se ejecuta por broadcasting y da un resultado equivocado sin avisar. Algo parecido pasa si se organizan las señales en filas y N = 2: `A @ X` y `solve` funcionan, pero con otra interpretación. Por eso las funciones validan formas y hay pruebas de contrato.

**¿Qué aportan autovalores, QR y SVD, y por qué no basta con imprimirlos?**
Los autovalores de A son 1 + c (dirección [1, 1]) y 1 − c (dirección [1, −1]): indican qué combinación de señales se amplifica o se atenúa. QR da un factor Q ortogonal y uno triangular R. La SVD da los valores singulares, que son factores de amplificación, y de ahí κ₂ = σmáx/σmín (1.5 y 199). Imprimir las salidas no demuestra nada: hay que comprobar numéricamente `Av ≈ λv`, `QᵀQ ≈ I`, `QR ≈ A` y `UΣVᵀ ≈ A`, con tolerancia y sin depender de signos, porque distintas convenciones son válidas.

**¿Cómo se relaciona la dirección difícil con κ₂ y los errores? ¿Por qué κ₂ no predice cada error?**
La dirección difícil es [1, −1]/√2, con autovalor 1 − c: 0.8 para c = 0.2 y 0.01 para c = 0.99. Reconstruir exige dividir esa componente entre 0.01, así que amplifica el ruido unas 100 veces. Con ruido de σ = 0.01 se espera una desviación de unos 0.71 por componente (0.01·100/√2) y se observó 0.67. Para c = 0.2 se espera ≈ 0.011 y se observó ≈ 0.010. Pero κ₂ describe el peor caso de amplificación relativa. Cada error absoluto concreto depende de cuánto de ese ruido cae sobre la dirección difícil y del tamaño de la señal, por eso κ₂ no predice error por error.

**¿Por qué el residuo puede ser pequeño aunque la reconstrucción sea mala? ¿Qué cambia en c = 1?**
`solve` hace que `A x̂ ≈ b` se cumpla casi exactamente (residuo ≈ 1e-16 en ambos casos), pero `b` ya contiene el ruido: el error real es `x̂ − x = A⁻¹ε`, que puede ser grande cuando A⁻¹ amplifica. Con c = 0.99 el residuo fue 5.0e-16 y el error euclidiano máximo 2.79. En c = 1 las dos filas de A son iguales, el rango es 1 y σmín = 0: κ₂ es infinito, el sistema no tiene solución única y no se puede reconstruir. El código lo detecta y lo comunica, y las pruebas lo verifican sin simularlo como tercera configuración.

**¿Qué eje se reduce en cada estadística, cómo se usó broadcasting y cómo se protegió el arreglo original?**
Como cada señal es una columna, la media y la desviación por componente reducen el eje 1 (entre señales) y dan forma (2,). La norma euclidiana por señal reduce el eje 0 (entre componentes) y da forma (441,). Para centrar, `error - media[:, np.newaxis]` resta una columna (2, 1) a todo el arreglo (2, 441) por broadcasting (esa vista no copia datos), y la desviación se comprobó con ddof=0. Para proteger el original: `np.vstack` copia los datos de la malla, `A @ X + ruido` y `solve` crean arrays nuevos, y una prueba compara `X` y el ruido con copias hechas antes del procesamiento.

## 4. Limitaciones

- Una sola realización de ruido y una sola semilla: los números exactos cambian con otra semilla, aunque la diferencia entre c = 0.2 y c = 0.99 es de dos órdenes de magnitud.
- El umbral de 0.1 es didáctico y el análisis usa errores absolutos.
- Solo se comparan dos valores de c y no se estudian otros niveles de ruido.

## 5. Ejecución limpia

Se clonó el repositorio en una carpeta nueva (`lab03_limpio`) y se siguió únicamente el README en PowerShell (Windows 11).

Comandos:

```
git clone https://github.com/ximenag-sudo/lab03-reconstruccion-senales.git lab03_limpio
cd lab03_limpio
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
cd src
python experimento.py
cd ..
```

Versiones: Python 3.13.17, NumPy 2.5.4, Matplotlib 3.11.2, pytest 9.2.0.

Resultados:

- `pytest`: 27 passed en 2.15 s.
- `experimento.py`: se obtuvieron las mismas dos tablas de la sección 2 (κ₂ = 1.50 y 199.00; error euclidiano máximo 0.0380 y 2.7945; proporción sobre el umbral 0.000 y 0.902; residuo máximo 2.5e-16 y 5.0e-16). La ejecución es reproducible con la semilla 7.

## 6. Nota individual

**Predicción.** Antes de ejecutar el experimento predije que c = 0.99 sería más sensible al ruido que c = 0.2, porque con c cercano a 1 los dos sensores miden casi lo mismo y el ruido se confunde con la pequeña diferencia entre las dos lecturas. También predije que el residuo seguiría siendo pequeño en ambos casos. Los resultados coincidieron: la desviación estándar del error fue ≈ 0.01 con c = 0.2 y ≈ 0.67 con c = 0.99 (κ₂ = 1.50 frente a 199), y el residuo máximo fue del orden de 1e-16 en los dos. Lo que más me sorprendió fue ver un residuo casi cero junto a un error de reconstrucción de hasta 2.79: confirma que un residuo pequeño no garantiza haber recuperado la señal real.

**Prueba.** Expliqué `test_producto_matricial_contra_calculo_manual`. Comprueba que `A @ X` da [1.4, 2.2] para c = 0.2 y x = [1, 2], y [2.98, 2.99] para c = 0.99. Esos valores los calculé a mano (por ejemplo, 1·1 + 0.2·2 = 1.4) y los escribí directamente en la prueba. Si el valor esperado se calculara con la misma función que se está probando, la prueba pasaría siempre, aunque la función estuviera mal. Por eso el cálculo manual es una referencia independiente.

**Decisión de implementación.** Organicé las señales como columnas: `X` tiene forma (2, 441), igual que las lecturas `B`. Así `A @ X` calcula todas las lecturas sin ciclos y `np.linalg.solve(A, B)` resuelve las 441 señales en una sola llamada. Elegí `solve` y no la inversa porque es más estable numéricamente y porque el enunciado lo pedía. Además, las funciones validan las formas de entrada para que un error de orientación (filas contra columnas) no pase inadvertido.