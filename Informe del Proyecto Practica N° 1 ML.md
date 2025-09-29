# Linear Models, Regularization, and Model Selection - Informe del Proyecto

> **Curso:** 2025-G1-910040-3-PEUCD-MACHINE LEARNING I  
> **Deadline:** Lunes 29 de septiembre de 2025, 23:59  
> **Entorno:** Python, NumPy, pandas, Matplotlib, scikit-learn  
> **Repositorio:** *(este README reemplaza el contenido anterior)*

---

## Integrantes del grupo

- **Buleje Ticse, Jean Carlos** 
- **Sebastian Rios, Wilder Teddy**   

---

## Descripción general del trabajo

Implementamos y comparamos **modelos lineales** (OLS, Ridge, Lasso) sobre dos conjuntos de datos:

1. **California Housing** (objetivo: `MedHouseVal`).  
2. **Bike Sharing** (agregación de `hour.csv` a **nivel diario** para `cnt` y análisis de **estacionalidad**).

Se realizaron cuatro partes:
- **A. OLS desde cero y Gradiente Descendente (GD)**
- **B. Baseline con `LinearRegression` (scikit-learn)**
- **C. Regularización (Ridge/Lasso), selección de α por CV, y polinomios (grado 2)**
- **D. Bike Rentals: repetición del pipeline con enfoque estacional y (cuando corresponde) CV temporal**

Se incluyeron:
- Gráficos de **costo vs. iteraciones** (GD),  
- **Predicho vs. real** en *test*,  
- **Paths de coeficientes** (Ridge/Lasso) en espacio base y (para bicicletas) en el **espacio transformado**.

---

## Diferencias observadas entre OLS, Ridge y Lasso

### Contexto experimental y preprocesamiento
- **Split**: separamos **train/test** antes de cualquier transformación.  
- **Estandarización**: `StandardScaler` ajustado **solo en train** y aplicado a test.  
- **Modelado**:  
  - OLS *desde cero* con NumPy (y verificación con `LinearRegression`).  
  - **Ridge/Lasso** con búsqueda de **α** en escala log: $[10^{-3}, 10^{2}]$.  
  - **Paths**: coeficientes vs α, eje log.  
- **Bike Sharing**: agregamos `hour.csv` → diario (suma de `cnt`; estadísticas climáticas promedio o máximas/mínimas razonadas). Creamos variables **estacionales** (mes, weekday, workingday, holiday; y dummies correspondientes).  
- **Polinomios (grado 2)**: aplicados a numéricas; dummies para categóricas; todo integrado con `ColumnTransformer` + `Pipeline`.

### OLS (mínimos cuadrados ordinarios)
- **Fortalezas**:  
  - Solución cerrada $\hat{\beta} = (X^\top X)^{-1}X^\top y$ (o mejor, por **pseudo-inversa** / `lstsq` para estabilidad).  
  - **Interpretabilidad** directa de coeficientes.  
  - Buen desempeño en espacio base cuando no hay alta multicolinealidad.
- **Debilidades observadas**:  
  - **Multicolinealidad** y **alta dimensionalidad** (p. ej., polinomios) → coeficientes grandes, **varianza elevada**, sensibilidad al ruido.  
  - Riesgo de **sobreajuste** sin control de complejidad (especialmente en el caso polinomial).

### Ridge (penalización L2)
- **Efecto principal**: **contracción** de coeficientes (no los lleva exactamente a cero).  
- **Cuándo brilla**: cuando hay **multicolinealidad** y muchas variables correlacionadas; reduce varianza y **estabiliza** OLS.  
- **Paths**: trayectorias **suaves** al incrementar α; ningún coeficiente cae a cero.  
- **Empírico**: en el espacio polinomial (california y bicicletas), **mejor generalización** que OLS con α seleccionado por CV.

### Lasso (penalización L1)
- **Efecto principal**: induce **esparsidad** (lleva coeficientes a **cero**), lo que conlleva **selección de variables**.  
- **Cuándo brilla**: con **muchas** features (polinomios + dummies) y cuando interpretabilidad/parsimonia es clave.  
- **Paths**: varios coeficientes se vuelven **exactamente cero** a medida que sube α.  
- **Empírico**: en el espacio transformado de bicicletas, Lasso **filtró** interacciones y términos con baja señal manteniendo rendimiento competitivo; modelo más simple y legible.

### Conclusión comparativa
- **Sesgo–Varianza**:  
  - OLS → **bajo sesgo** / **alta varianza** en escenarios complejos.  
  - Ridge/Lasso → **aumentan el sesgo** pero **reducen varianza**, ganando en **generalización**.  
- **Multicolinealidad**: Ridge **>** OLS; Lasso además **selecciona** variables.  
- **Polinomios**: Regularización (especialmente con α elegido por CV) **supera a OLS** en estabilidad y error de test.

---

## Efecto de la tasa de aprendizaje en el Descenso del Gradiente (GD)

### Configuración
- Implementamos GD para OLS sobre datos **estandarizados**.  
- Probamos **≥ 2 learning rates** (p. ej., 0.01 y 0.1).  
- Reportamos **costo vs. iteraciones** y comparamos parámetros/errores con la solución OLS.

### Hallazgos clave (observados en las curvas)
- **LR pequeña (≈ 0.01)**:  
  - Convergencia **estable**, pero **lenta**.  
  - Se requieren más iteraciones para acercarnos a la solución óptima.
- **LR mayor (≈ 0.1)**:  
  - **Acelera** la caída del costo al inicio.  
  - Puede exhibir **oscilaciones** o desaceleración si roza el límite de estabilidad.  
- Con una **LR adecuada** y suficientes iteraciones, los **parámetros de GD** aproximan bien a los de OLS y el **error de test** es similar.

### Buenas prácticas aplicadas
- **Estandarización** previa (mejora la condición numérica y la estabilidad de GD).  
- **Criterios de parada**: máximo de épocas + monitoreo de reducción de costo.  
- **Nota de claridad**: explicitamos la forma del costo (MSE o ½·MSE) para interpretar bien la escala.

---

## Cómo la validación cruzada (k-fold) influyó en la elección de la fuerza de regularización

### Metodología aplicada
- Búsqueda de **α** en escala logarítmica $[10^{-3}, 10^{2}]$ para Ridge/Lasso.  
- **KFold (K=5)** con `cross_val_score` o uso de **`RidgeCV` / `LassoCV`**.  
- **Prevención de data leakage**:  
  - Escalado y generación de features dentro de un **`Pipeline`** (y `ColumnTransformer` cuando corresponde) para que cada *fold* ajuste transformaciones **solo con su train**.  
  - En datos temporales (bikes), consideramos **`TimeSeriesSplit`** (cuando aplicable) para respetar el orden temporal (opcional si el enunciado exige “mismos pasos”, pero metodológicamente recomendable).

### Impacto observado
- La elección de **α por CV**:  
  - **Mejoró** el error de generalización vs. α elegidos “a ojo”.  
  - En el espacio polinomial, CV tendió a escoger **α más altos** en Ridge (mayor contracción) y **α que inducen esparsidad útil** en Lasso (manteniendo un subconjunto informativo).  
- **Curvas MSE (CV) vs α** (cuando trazadas) mostraron mínimo claro alrededor del α\* finalmente seleccionado.

### Conclusión operativa
- La **CV** fue determinante para equilibrar **sesgo–varianza** y seleccionar la **fuerza de regularización** que **maximizó el desempeño en test**, especialmente con polinomios y estacionalidad.

---

## Notas específicas de la Parte D — Bike Rentals (estacionalidad)

- **Fuente**: `hour.csv` (no `day.csv`).  
- **Agregación**:  
  - `cnt` **diario** = suma por fecha.  
  - Variables climáticas (temp, atemp, hum, windspeed): promedios diarios (y/o estadísticas adicionales si se justifican).  
- **Estacionalidad**:  
  - Dummies de **mes**, **weekday**, **workingday**, **holiday**; opcionalmente seno/coseno para capturar ciclos anuales.  
- **Modelado**: repetimos pipeline (OLS, Ridge, Lasso) con y sin **polinomios grado 2** en numéricas (manteniendo dummies para categóricas).  
- **Resultados cualitativos**:  
  - Señal estacional marcada; **Ridge** estabiliza; **Lasso** filtra términos redundantes/interacciones débiles.  
  - **Paths** en el espacio transformado evidencian qué grupos de features (meses, términos polinomiales) ganan/pierden peso al variar α.

---

## Gráficos incluidos en el notebook

1. **Costo vs iteraciones (GD)** con ≥ 2 learning rates (anotar LR en leyenda).  
2. **Predicho vs real (test)** con línea **y = x** (incluye R²/MSE en el subtítulo o leyenda).  
3. **Regularization Paths — Ridge** (coeficientes vs α, escala log).  
4. **Regularization Paths — Lasso** (coeficientes vs α, escala log; resaltar esparsidad).  
5. *(Parte D)* **Predicho vs real** (diario) y **paths en espacio transformado** (polinomios + dummies; mostrar top-k por claridad).  
6. *(Opcional recomendado)* **MSE (CV) vs α** para Ridge y Lasso.

---

## Reproducibilidad y prácticas de ingeniería

- **Semillas** (`random_state=42`) para replicabilidad.  
- **Split antes del escalado** (evita *data leakage*).  
- **`Pipeline` y `ColumnTransformer`** en CV para garantizar que estandarización/polinomios/dummies se ajusten **solo** con datos de entrenamiento en cada *fold*.  
- **TimeSeriesSplit** (cuando procede) para respetar temporalidad en bikes.  
- **Estabilidad numérica en OLS**: preferir `np.linalg.pinv` o `lstsq` sobre invertir explícitamente $X^\top X$.

---

## Lecciones aprendidas

- Regularización (Ridge/Lasso) es crucial cuando ampliamos el espacio de características (polinomios, dummies), mitigando multicolinealidad y mejorando generalización.
- Lasso agrega valor en interpretabilidad (esparsidad), mientras Ridge estabiliza parámetros cuando hay alta correlación.
- GD requiere escalado y una LR bien elegida para converger eficientemente hacia la solución de OLS.
- CV dentro de Pipeline evita fugas de información y es el mecanismo central para elegir α de forma robusta.
- En series con estacionalidad, respetar el orden temporal en la validación mejora la credibilidad de los resultados.

---

## Apéndice: Fórmulas y detalles técnicos

- **OLS (cerrado):**  
  $$\hat{\beta} = (X^\top X)^{-1}X^\top y \quad \text{(o, mejor, } \hat{\beta} = \text{pinv}(X)y\text{)}$$
- **Ridge:**  
  $$\hat{\beta}_{ridge} = (X^\top X + \alpha I)^{-1}X^\top y$$
- **Lasso:**  
  Resolución por métodos de optimización convexa (coordinate descent en *scikit-learn*).
- **Costo MSE:**  
  $$J(\beta) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 
  \quad \Big(\text{o } \frac{1}{2n}\sum(\cdot)^2 \text{ para GD}\Big)$$

----
