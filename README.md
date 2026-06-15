# Linear Models, Regularization, and Model Selection — Informe de Proyecto

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.22%2B-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.4%2B-150458?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-F7931E?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5%2B-ffffff?logo=matplotlib&logoColor=black)

Este repositorio contiene la implementación, análisis y comparación de modelos lineales fundamentales y técnicas de regularización avanzadas aplicadas sobre dos conjuntos de datos de referencia: **California Housing** y **Bike Sharing**. El objetivo primordial es evaluar el balance sesgo-varianza, el impacto de la multicolinealidad y la efectividad de la validación cruzada en la selección de hiperparámetros.

---

## 📌 Información del Curso y Entrega

| Atributo | Detalle |
| :--- | :--- |
| **Curso** | Machine Learning I (2025-G1-910040-3-PEUCD) |
| **Fecha Límite** | Lunes 29 de septiembre de 2025, 23:59 |
| **Entorno Tecnológico** | Python, NumPy, pandas, Matplotlib, scikit-learn |
| **Estado del Repositorio** | Versión Final (Reemplaza el contenido base anterior) |

### 👥 Integrantes del Grupo
* **Buleje Ticse, Jean Carlos**
* **Sebastian Rios, Wilder Teddy**

---

## 🗺️ Estructura y Descripción del Proyecto

El desarrollo experimental se estructuró de manera rigurosa en cuatro fases secuenciales:

* **Fase A: OLS desde Cero y Descenso del Gradiente (GD)** Implementación matricial nativa de Mínimos Cuadrados Ordinarios (Solución Cerrada) y optimización iterativa mediante Descenso del Gradiente sobre datos estandarizados.
* **Fase B: Baseline con Scikit-Learn** Construcción de un modelo de referencia utilizando la clase `LinearRegression` de scikit-learn para validar las métricas e implementaciones de la Fase A.
* **Fase C: Regularización y Complejidad Polinomial** Extensión del espacio de características a polinomios de grado 2. Implementación de penalizaciones **L1 (Lasso)** y **L2 (Ridge)**, automatizando la búsqueda del hiperparámetro de regularización óptimo $\\alpha$ mediante validación cruzada.
* **Fase D: Enfoque Estacional y Temporal (Bike Rentals)** Agregación y procesamiento del set de datos `hour.csv` a nivel diario. Construcción de variables categóricas estacionales y análisis del comportamiento de los coeficientes a través del espacio transformado empleando validación cruzada adaptada.

---

## 📊 Análisis Comparativo: OLS vs. Ridge vs. Lasso

A continuación, se resumen las propiedades mecánicas y los hallazgos empíricos de los tres enfoques de modelado:

| Característica | OLS (Mínimos Cuadrados Ordinarios) | Ridge (Penalización $L_2$) | Lasso (Penalización $L_1$) |
| :--- | :--- | :--- | :--- |
| **Mecanismo Matemático** | Minimiza la suma de errores al cuadrado de forma directa. | Añade una penalización cuadrática: $\\alpha \\Vert \\beta \\Vert_2^2$. | Añade una penalización absoluta: $\\alpha \\Vert \\beta \\Vert_1$. |
| **Efecto en Coeficientes** | No altera los coeficientes originales. | **Contracción (Shrinkage)** asintótica hacia cero. | **Esparsidad (Sparsity)**; fuerza coeficientes exactamente a cero. |
| **Selección de Features** | No realiza (mantiene todas las variables). | No realiza (mantiene todas, reduce magnitud). | **Sí realiza** (funciona como selector intrínseco). |
| **Resistencia a Multicolinealidad** | Muy baja; alta varianza e inestabilidad numérica. | **Alta**; estabiliza la matriz e invierte con seguridad. | **Media/Alta**; selecciona una variable y descarta correlacionadas. |
| **Comportamiento Polinomial** | Tiende severamente al **sobreajuste** (Overfitting). | Excelente generalización mitigando la varianza de interacciones. | Destacable parsimonia; filtra interacciones de baja señal. |

### Hallazgos Empíricos Detallados

1.  **OLS (Fortalezas y Debilidades):** Su solución cerrada proporciona una interpretabilidad directa e inmediata cuando los supuestos de Gauss-Márkov se cumplen. No obstante, al introducir características polinomiales (Grado 2) y variables *dummy*, sufre de **explosión de varianza** y coeficientes artificialmente gigantescos debido a la multicolinealidad.
2.  **Ridge (Estabilización):** Al trazar los *Regularization Paths* (coeficientes vs. $\\alpha$), se observaron curvas suaves donde ningún coeficiente llega a anularse por completo. En escenarios de alta dimensionalidad (polinomios), Ridge superó consistentemente a OLS en el conjunto de prueba (*test error*).
3.  **Lasso (Simplificación):** En el dataset de Bike Rentals (espacio transformado), Lasso eliminó con éxito múltiples términos polinomiales redundantes e interacciones débiles. Esto derivó en un modelo con rendimiento competitivo, pero sustancialmente más parsimonioso y fácil de auditar.

---

## 📉 Optimización por Descenso del Gradiente (GD)

Se evaluó la convergencia del Descenso del Gradiente para OLS bajo diferentes tasas de aprendizaje ($\\eta$ o *Learning Rate*), obteniendo las siguientes conclusiones clave:

* **Tasa de Aprendizaje Pequeña ($\\eta \\approx 0.01$):** Garantiza una trayectoria monótona decreciente en la función de costo, libre de oscilaciones dañinas. Sin embargo, la convergencia es **lenta**, requiriendo un presupuesto de iteraciones sustancialmente alto para alcanzar la vecindad del óptimo analítico.
* **Tasa de Aprendizaje Mayor ($\\eta \\approx 0.1$):** Acelera drásticamente la caída del costo en las primeras épocas. No obstante, si se aproxima al límite teórico de estabilidad, puede manifestar oscilaciones amortiguadas o ralentizar su tasa de mejora final.
* **Alineación Analítica:** Con un esquema de parada óptimo y previa **estandarización estricta** de las variables, los coeficientes hallados por GD convergen con un error despreciable hacia los obtenidos mediante la solución analítica de OLS.

> [!IMPORTANT]  
> La **estandarización previa** de los datos no es opcional: previene que la función de costo posea contornos elípticos elongados (mal condicionamiento numérico), asegurando que el vector de gradientes apunte eficientemente hacia el mínimo global.

---

## 🔄 Validación Cruzada (K-Fold) y Control de Fuga de Datos

Para encontrar la fuerza de regularización óptima ($\\alpha$) en una escala logarítmica de $[10^{-3}, 10^{2}]$, se adoptó un enfoque riguroso de ingeniería de características:

1.  **Prevención Extrema de Data Leakage:** Todo el preprocesamiento (`StandardScaler`, `PolynomialFeatures`, e imputación/codificación de variables categóricas) se encapsuló en un **`Pipeline`** gobernado por un **`ColumnTransformer`**.
2.  **Ajuste por Fold:** Este diseño garantiza que las transformaciones se calculen e introduzcan **únicamente con los datos de entrenamiento de cada fold específico** en la validación cruzada ($K=5$), manteniendo el set de prueba complementario completamente aislado.
3.  **Validación Temporal:** Para el conjunto de datos de *Bike Rentals*, dada su naturaleza secuencial, se incorporó un esquema fundamentado en **`TimeSeriesSplit`**, salvaguardando la flecha del tiempo y evitando predecir el pasado con información del futuro.

**Resultado:** Las curvas de Error Cuadrático Medio (MSE) en Validación vs. $\\alpha$ revelaron mínimos globales nítidos, demostrando que la optimización vía CV previene de forma objetiva el sub e introduce el nivel exacto de sesgo para maximizar la generalización.

---

## 🚴 Tratamiento de Estacionalidad en Bike Rentals (Parte D)

El tratamiento de la serie temporal agregada a nivel diario a partir de `hour.csv` requirió ingeniería de variables especializada:

* **Agregación Semántica:** La variable objetivo `cnt` fue consolidada como la sumatoria diaria completa. Las métricas atmosféricas (`temp`, `atemp`, `hum`, `windspeed`) se resumieron mediante promedios y extremos térmicos diarios justificados.
* **Modelado del Ciclo:** Se construyeron matrices de variables indicadoras (*dummies*) para capturar patrones de comportamiento institucional y climático (`mes`, `día de la semana`, `día laborable`, `feriado`).
* **Resultados Cualitativos:** El pipeline polinomial con regularización expuso de forma transparente la interacción estacional. Los gráficos de trayectorias permitieron visualizar cómo los coeficientes asociados a meses invernales o combinaciones de lluvia/viento se contraen aceleradamente bajo Lasso, preservando la robustez de las variables de alta certidumbre predictiva.

---

## 🎨 Visualizaciones Clave Incluidas

El notebook asociado genera un conjunto de gráficos analíticos esenciales para la auditoría de los modelos:

1.  **Evolución del Costo:** Gráfico de curvas de aprendizaje ($J(\cdot)$ vs. Iteraciones) comparando simultáneamente las diferentes tasas de aprendizaje en el Descenso del Gradiente.
2.  **Calibración del Modelo (Predicho vs. Real):** Gráficos de dispersión en el conjunto de *test* contrastados con la línea de identidad $y = x$. Cada visualización reporta explícitamente métricas de rendimiento $R^2$ y Error Cuadrático Medio (MSE).
3.  **Regularization Paths (Ridge y Lasso):** Trazado del valor de los coeficientes en función de $\\alpha$ (eje x en escala logarítmica), permitiendo diagnosticar visualmente la inducción de esparsidad y la velocidad de contracción de características individuales y polinomiales.

---

## ⚙️ Prácticas de Ingeniería y Reproducibilidad

* **Determinismo:** Fijación de semillas globales estricta (`random_state=42`) en todas las divisiones de datos y algoritmos estocásticos.
* **Estabilidad Numérica:** Reemplazo de la inversión directa de la matriz de diseño $(X^\top X)^{-1}$ por métodos numéricamente estables como la pseudo-inversa (`np.linalg.pinv`) o mínimos cuadrados directos (`np.linalg.lstsq`).

---

## 📖 Apéndice: Formulación Matemática

### Mínimos Cuadrados Ordinarios (OLS)
$$\\hat{\\beta}_{OLS} = (X^\top X)^{-1} X^\top y$$

### Regresión Ridge (Penalización $L_2$)
$$\\hat{\\beta}_{Ridge} = (X^\top X + \\alpha I)^{-1} X^\top y$$

### Regresión Lasso (Penalización $L_1$)
$$\\min_{\\beta} \\left\\{ \\frac{1}{n} \\Vert y - X\\beta \\Vert_2^2 + \\alpha \\Vert \\beta \\Vert_1 \\right\\}$$

### Función de Costo MSE (Para Descenso del Gradiente)
$$J(\\beta) = \\frac{1}{2n} \\sum_{i=1}^{n} \\left( y_i - X_i\\beta \\right)^2$$
