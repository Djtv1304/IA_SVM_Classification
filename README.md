# Informe de Support Vector Machine (SVM): Predicción de Bajo Peso al Nacer

## 1. Introducción y Objetivo

El presente informe tiene como finalidad describir de manera detallada y estructurada el desarrollo, entrenamiento y evaluación de un modelo de **Support Vector Machine (SVM)** con kernel lineal aplicado a datos obstétricos para predecir si un recién nacido presenta o no bajo peso al nacer (is_low_weight). El objetivo concreto es:

* Utilizar variables demográficas y biométricas (edad materna y peso al nacer) para clasificar correctamente a un bebé en la categoría de bajo peso (1) o peso normal (0).
* Documentar el procedimiento completo de preprocesamiento, entrenamiento, evaluación y visualización del modelo SVM, resaltando las métricas clave, especialmente la precisión (accuracy).
* Comparar implícitamente el rendimiento del SVM con otros algoritmos de clasificación aplicados al mismo problema.

## 2. Descripción del Conjunto de Datos

El dataset emplea dos variables predictoras y una variable objetivo, con los siguientes atributos:

| Variable            | Descripción                                                                | Tipo                 |
| ------------------- | -------------------------------------------------------------------------- | -------------------- |
| age                 | Edad de la madre (en años).                                                | Numérica             |
| bwt                 | Peso de nacimiento (Birthday Weight) del bebé (en gramos).                 | Numérica             |
| is_low_weight       | Indicador binario: 1 si el peso < 2500 g (bajo peso), 0 en caso contrario. | Categórica (Binaria) |

### Ejemplos de Datos

| age | bwt  | is_low_weight |
| --- | ---- | --------------- |
| 19  | 2523 | 0               |
| 33  | 2551 | 1               |
| …   | …    | …               |

## 3. Preprocesamiento de Datos

1. **Carga del CSV**
   ```python
   dataset = pd.read_csv('birthwt_data_modified.csv')
   ```
   El archivo contiene múltiples registros de nacimientos con las columnas age, bwt y is_low_weight.

2. **Separación de características (X) y variable objetivo (y)**
   ```python
   X = dataset.iloc[:, :-1].values  # [age, bwt]
   y = dataset.iloc[:, -1].values   # [is_low_weight]
   ```

3. **División en Conjunto de Entrenamiento y Prueba**
   * Proporción: 75% entrenamiento / 25% prueba.
   * Aleatoriedad fija (random_state = 0) para reproducibilidad.
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.25, random_state=0
   )
   ```

4. **Escalado de Características**
   * Se utiliza **StandardScaler** para normalizar age y bwt.
   * **Crítico para SVM**: Los algoritmos SVM son extremadamente sensibles a la escala de las características, ya que buscan el hiperplano óptimo basándose en distancias euclidianas.
   * Garantiza que ambas variables tengan media 0 y desviación estándar 1.
   ```python
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test  = sc.transform(X_test)
   ```

## 4. Entrenamiento del Modelo SVM

1. **Creación y ajuste del clasificador**
   ```python
   classifier = SVC(kernel='linear', random_state=0)
   classifier.fit(X_train, y_train)
   ```
   * **Kernel Lineal**: Se utiliza un kernel lineal que busca el hiperplano óptimo que separe las clases con el mayor margen posible.
   * **Principio SVM**: Maximiza la distancia entre el hiperplano de decisión y los puntos más cercanos de cada clase (vectores de soporte).

2. **Predicción de Ejemplo Puntual**
   * Se proyecta un caso de prueba: edad = 24 años, peso al nacer = 3200 g.
   * **Resultado obtenido**: [0] - Peso normal
   ```python
   resultado = classifier.predict(sc.transform([[24, 3200]]))
   print(f"Predicción para edad 24 y peso 3200 g: {resultado}")  # Resultado: [0]
   ```

## 5. Evaluación y Métricas

### 5.1 Predicciones sobre el Conjunto de Prueba

Se obtienen las predicciones del modelo (y_pred) y se comparan con las etiquetas reales (y_test):

```
[[0 0]
 [0 0]
 [1 1]
 [0 0]
 [0 0]
 [0 0]
 [0 0]
 [0 0]
 [1 0]
 [1 0]
 [0 0]
 [1 1]
 [1 1]
 [1 1]
 [0 0]
 [0 0]
 [0 1]
 [0 0]
 [1 1]
 [1 1]
 [0 0]
 [0 0]
 [1 1]
 [1 1]
 [1 1]
 [1 0]
 [0 0]
 [0 0]
 [0 0]
 [0 1]
 [1 0]
 [0 0]
 [0 0]
 [0 1]
 [0 0]
 [0 0]
 [0 1]
 [1 0]
 [0 0]
 [0 0]
 [0 0]
 [0 0]
 [0 0]
 [1 0]
 [1 1]
 [0 0]
 [0 0]
 [0 0]]
```

Cada fila muestra [predicción, valor_real]. Por ejemplo, [1 0] indica un **falso positivo** (predijo bajo peso cuando el bebé no lo es).

### 5.2 Matriz de Confusión

Se construye la matriz de confusión a partir de y_test y y_pred:

```
[[28  6]
 [ 4 10]]
```

|                           | Predicción = 0 (No bajo peso) | Predicción = 1 (Bajo peso) |
| ------------------------- | ----------------------------: | -------------------------: |
| **Real = 0 (No bajo peso)** |                        28 |                          6 |
| **Real = 1 (Bajo peso)**    |                         4 |                         10 |

* **Verdaderos Negativos (TN)** = 28
* **Falsos Positivos (FP)** = 6  
* **Falsos Negativos (FN)** = 4
* **Verdaderos Positivos (TP)** = 10

### 5.3 Métrica de Precisión (Accuracy)

La precisión global se calcula como:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \;=\; \frac{10 + 28}{10 + 28 + 6 + 4} \;=\; \frac{38}{48} = 0{,}7917
$$

* **Valor: 0.7917 ⇒ 79,17%**
* **Interpretación**: En términos prácticos, **79 de cada 100 predicciones** sobre el conjunto de prueba resultan correctas.
* **Rendimiento SVM**: Este accuracy del 79,17% indica que el modelo SVM con kernel lineal logra un rendimiento sólido y comparable a otros algoritmos de clasificación en este problema específico.

## 6. Visualización de los Resultados

### 6.1 Conjunto de Entrenamiento

![image](https://github.com/user-attachments/assets/11c4fc55-c1d7-4ee8-ae68-c8e57d2dde3a)

**Descripción de la Gráfica:**
* El plano se graficó usando las características originales (Age o Edad ↔ age, Birthday Weight (BWT) ↔ bwt).
* La región en **verde** corresponde a la clase 1 (bajo peso), y la región en **rojo** a la clase 0 (peso normal).
* Los puntos rojos representan observaciones de la clase 0, y los verdes las de la clase 1.
* **Línea de decisión SVM**: La frontera entre regiones representa el hiperplano óptimo encontrado por el algoritmo SVM.

**Observaciones:**
* Se aprecia una separación **lineal clara** entre las regiones de decisión.
* La mayoría de puntos de clase 0 (rojos) quedan correctamente ubicados en la región roja, y viceversa.
* **Vectores de soporte**: Los puntos más cercanos a la línea de decisión son los que determinan el hiperplano óptimo.
* Algunos ejemplos están cerca del límite, indicando casos límite donde la decisión es más incierta.

### 6.2 Conjunto de Prueba

![image](https://github.com/user-attachments/assets/a972cdbf-14f8-42a8-8690-a80b368ae937)

**Descripción de la Gráfica:**
* Misma lógica gráfica que en el conjunto de entrenamiento.
* Se observan puntos que caen en zonas opuestas a su color, los cuales corresponden a errores de clasificación (FP o FN).

**Observaciones:**
* La **frontera de decisión se mantiene consistente** entre entrenamiento y prueba, indicando buena generalización.
* La densidad de puntos mal clasificados es relativamente baja, lo que justifica el accuracy del 79,17%.
* Se observan algunos puntos rojos dentro del área verde (Falsos Negativos) y puntos verdes en el área roja (Falsos Positivos), principalmente en la zona de transición.
* **Robustez del modelo**: La consistencia visual entre ambos conjuntos sugiere que el modelo no presenta sobreajuste significativo.

## 7. Análisis Detallado del Rendimiento SVM

### 7.1 Interpretación de la Precisión (79,17%)

* **Rendimiento sólido**: Un accuracy del 79,17% es **considerablemente bueno** para un problema médico/obstétrico, donde la complejidad de los factores que influyen en el peso al nacer es alta.
* **Comparación implícita**: Este rendimiento es **idéntico** al obtenido con Regresión Logística en el mismo dataset, sugiriendo que ambos algoritmos capturan efectivamente los patrones lineales subyacentes.
* **Ventaja del SVM**: A diferencia de la regresión logística, SVM busca maximizar el margen de separación, lo que puede proporcionar mayor robustez ante nuevos datos.

### 7.2 Análisis de Métricas Complementarias

1. **Tasa de Verdaderos Positivos (Sensibilidad)**

  ![image](https://github.com/user-attachments/assets/fd2fe1e7-c5bb-4d79-b55a-0a3b41f3bcb0)

   * **Mejora significativa**: 71,43% vs 57,14% de la Regresión Logística.
   * De los 14 casos reales de bajo peso en prueba, el modelo SVM clasificó correctamente **10** (vs 8 en Regresión Logística).
   * **Impacto clínico**: Mayor detección de casos de bajo peso, reduciendo riesgos médicos.

2. **Tasa de Verdaderos Negativos (Especificidad)**

  ![image](https://github.com/user-attachments/assets/445289f5-a4c3-4369-813c-30b12a0aaa7d)

   * Ligeramente inferior a la Regresión Logística (88,24%), pero aún robusta.
   * De 34 casos reales de peso normal, el modelo clasificó correctamente 28.

4. **Balance Sensibilidad-Especificidad**
   * **SVM** logra un mejor balance: 71,43% sensibilidad vs 82,35% especificidad.
   * **Ventaja clínica**: Reduce significativamente los falsos negativos (casos críticos no detectados).

### 7.3 Distribución de Errores - **MEJORA CLAVE**

* **Falsos Negativos (FN = 4)**: **Reducción del 33%** comparado con Regresión Logística (6 → 4).
  * **Impacto crítico**: Menos bebés con bajo peso quedan sin detectar.
  * **Beneficio médico**: Mayor cobertura de atención prenatal/postnatal especializada.

* **Falsos Positivos (FP = 6)**: Aumento moderado comparado con Regresión Logística (4 → 6).
  * **Trade-off aceptable**: El incremento en intervenciones innecesarias es compensado por la mejor detección de casos críticos.

## 8. Ventajas Específicas del SVM

### 8.1 Características Técnicas

1. **Maximización del Margen**
   * SVM no solo encuentra una línea que separe las clases, sino la que maximiza la distancia a los puntos más cercanos.
   * **Resultado**: Mayor robustez ante variaciones en los datos de entrada.

2. **Vectores de Soporte**
   * El modelo se basa únicamente en los puntos críticos (vectores de soporte) para definir la frontera.
   * **Ventaja**: Menos sensible a outliers alejados de la frontera de decisión.

3. **Flexibilidad de Kernels**
   * Aunque se usó kernel lineal, SVM permite kernels no lineales (RBF, polinomial) para problemas más complejos.
   * **Potencial**: Capacidad de adaptarse a relaciones no lineales futuras.

### 8.2 Rendimiento Comparativo

| Métrica | SVM (Kernel Lineal) | Regresión Logística | Mejora |
|---------|--------------------|--------------------|--------|
| **Accuracy** | **79,17%** | 79,17% | ≈ |
| **Sensibilidad** | **71,43%** | 57,14% | **+25%** |
| **Especificidad** | 82,35% | 88,24% | -7% |
| **Falsos Negativos** | **4** | 6 | **-33%** |
| **Falsos Positivos** | 6 | 4 | +50% |

**Conclusión**: SVM ofrece un **perfil de rendimiento superior** para aplicaciones médicas donde minimizar falsos negativos es prioritario.

## 9. Recomendaciones y Mejoras Futuras

### 9.1 Optimización del Modelo SVM

1. **Exploración de Kernels No Lineales**
   * Probar kernel RBF (`kernel='rbf'`) para capturar relaciones no lineales.
   * Ajustar parámetros `C` (regularización) y `gamma` (ancho del kernel RBF).

2. **Optimización de Hiperparámetros**
   * Usar GridSearchCV para encontrar la combinación óptima de parámetros.
   * Evaluar diferentes valores de `C` para balancear sesgo-varianza.

3. **Ajuste del Umbral de Decisión**
   * Aunque SVM usa margen máximo, se puede ajustar el umbral post-entrenamiento.
   * Objetivo: Maximizar sensibilidad manteniendo especificidad aceptable.

### 9.2 Enriquecimiento del Dataset

1. **Variables Adicionales**
   * Factores maternos: IMC, antecedentes obstétricos, consumo de tabaco/alcohol.
   * Variables socioeconómicas: educación, acceso a atención prenatal.
   * **Expectativa**: Mejora significativa en accuracy y reducción de falsos negativos.

2. **Ingeniería de Características**
   * Crear variables derivadas: ratio edad/peso, índices de riesgo compuestos.
   * Transformaciones no lineales de variables existentes.

### 9.3 Validación Robusta

1. **Cross-Validation Estratificada**
   * Implementar k-fold cross-validation manteniendo proporción de clases.
   * **Objetivo**: Estimación más robusta del rendimiento real.

2. **Validación Temporal**
   * Si hay datos temporales, validar con datos de periodos posteriores.
   * **Importancia**: Verificar estabilidad del modelo en el tiempo.

## 10. Conclusiones

* **Rendimiento Destacado**: El modelo SVM logró una **precisión (accuracy) del 79,17%**, idéntica a la Regresión Logística, pero con un perfil de rendimiento superior en métricas críticas.

* **Ventaja Clave - Sensibilidad Mejorada**: Con **71,43% de sensibilidad** (vs 57,14% de Regresión Logística), SVM detecta significativamente más casos de bajo peso, reduciendo el riesgo clínico.

* **Balance Optimizado**: La reducción del 33% en falsos negativos (6 → 4) compensa el incremento moderado en falsos positivos, resultando en un modelo más seguro para aplicaciones médicas.

* **Robustez del Modelo**: Las visualizaciones confirman una separación clara y consistente entre conjuntos de entrenamiento y prueba, indicando buena capacidad de generalización.

* **Recomendación Principal**: Para el contexto obstétrico analizado, **SVM con kernel lineal es preferible** a la Regresión Logística debido a su superior capacidad de detección de casos críticos (bajo peso al nacer).

* **Perspectiva Futura**: La incorporación de variables adicionales y la exploración de kernels no lineales tienen potencial para elevar significativamente el rendimiento por encima del 79,17% actual, especialmente en la reducción de falsos negativos restantes.

**Valor Práctico**: El modelo SVM desarrollado representa una herramienta valiosa para la detección temprana de bajo peso al nacer, contribuyendo a mejores resultados de salud materno-infantil a través de intervenciones oportunas y focalizadas.

**Autor:** Diego Toscano <br>
**Contacto:** [diego.toscano@udla.edu.ec](mailto:diego.toscano@udla.edu.ec)
