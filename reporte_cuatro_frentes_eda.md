# Reporte EDA — Operación Cuatro Frentes

**Analista:** [Néstor Ricardo López Gutiérrez / 22121368]

---

## Misión 1 — Semáforo Académico

### Pregunta de negocio

> **¿Qué color de semáforo (verde, amarillo, rojo) debería asignarse a cada alumno al cierre del parcial, según su riesgo de reprobar?**

### Tipo propuesto

**Clasificar** (multiclase). Justificación: la variable `riesgo` es una **etiqueta categórica nominal con tres categorías ordenadas** (verde < amarillo < rojo), no un número continuo. Aunque hay un orden implícito (rojo es peor que amarillo), las categorías son discretas y mutuamente excluyentes. No se trata de predecir un valor numérico; se trata de asignar una de tres clases.

### Y / forma de Y

- **Variable objetivo:** `riesgo` (categórica nominal: verde, amarillo, rojo)
- **Tres clases** — multiclase estándar
- **Distribución del dataset completo (N=300):** 40% verde, 35% amarillo, 25% rojo → **moderadamente desbalanceado** (la clase minoritaria representa solo 25%)

### X (mínimo 5)

| Variable | Justificación |
|----------|---------------|
| `asistencia_pct` | Falta académica directa: más bajo asistir → más bajo rendimiento. |
| `tareas_entregadas` | Proxy de compromiso: quién entrega más tareas tiende a tener mejor nota. |
| `promedio_parciales` | Rendimiento previo es el mejor predictor de futuro riesgo de reprobación. |
| `horas_plataforma` | Inversión de tiempo en el LMS correlaciona con éxito. |
| `reprobadas_previas` | Historial académico es fuerte predictor de riesgo futuro. |
| `turno` | Factor contextual: el rendimiento puede variar entre turno matutino y vespertino. |

### Patrones EDA

**Patrón 1 — Separación lineal clara entre clases por `promedio_parciales`:**

| Riesgo | promedio_parciales (media) | reprobadas_previas (media) |
|--------|---------------------------|----------------------------|
| Verde | 8.4 | 0.2 |
| Amarillo | 6.8 | 0.9 |
| Rojo | 5.0 | 2.4 |

El `promedio_parciales` crea una **separación casi lineal**: verde ~8.0+, amarillo ~6.0-7.5, rojo ~5.0-. Esto sugiere que un modelo lineal o un árbol poco profundo podría ser suficiente.

**Patrón 2 — Correlación inversa entre `asistencia_pct` y `reprobadas_previas`:**

- Verde: asistencia=91%, reprobadas=0.2
- Rojo: asistencia=52%, reprobadas=2.4

Esto confirma que el **riesgo de reprobación es un fenómeno multidimensional** pero con un eje principal: el rendimiento académico previo. Las variables `asistencia_pct` y `reprobadas_previas` están correlacionadas entre sí (colinealidad moderada), pero ambas aportan señal complementaria.

### Distribución de Y

- **Verde:** 40% (120)
- **Amarillo:** 35% (105)
- **Rojo:** 25% (75)

**Implicaciones:**
- **No extremadamente desbalanceado** (25% minoritaria es manejable).
- **Accuracy puede ser útil como métrica principal**, pero no suficiente. Se recomienda `macro-F1` para asegurar que todas las clases sean evaluadas por igual.
- **Matriz de confusión** es clave: con 25% de rojo, un modelo que siempre prediga "verde" tendría 40% de accuracy, pero fallaría en identificar riesgo.

### Calidad de datos (problema realista)

**Problema:** `horas_plataforma` tiene valores atípicos. Algunos alumnos reportan 50+ horas en la plataforma en un parcial de 3 semanas, lo que es físicamente improbable (máximo realista ~21 horas/semana × 3 semanas = 63 horas, pero 50+ horas/semana es sospechoso).

**Cómo detectarlo:**
- Boxplot de `horas_plataforma`: valores > Q3 + 1.5×IQR.
- Correlación de Pearson de `horas_plataforma` con `promedio_parciales`: si la correlación es negativa (más horas → menos rendimiento), es un red flag. Un estudiante que estudia 50h/sem y reprueba es sospechoso.

### Fuga de información

**No usarías** la `calificación_final_del_curso` como X. **Justificación:**

- La calificación final del curso **es el resultado final** del curso, que ya incorpora si el alumno reprobó o no. Si `riesgo` = "rojo" significa "riesgo alto de reprobar", y `calificacion_final` ya refleja si reprobó, estás usando el **outcome como predictor**.
- Esto crea un **data leakage perfecto**: el modelo aprendería "calificacion_final < 6 → rojo", lo cual es circular y no útil para predecir riesgo *antes* del examen final.

### Propuesta post-EDA

**Modelo:** Random Forest multiclase (o Gradient Boosting como XGBoost multiclase)

**Condición del dataset:**
- Debe tener al menos **30 ejemplos por clase** (90 mínimos) para que las clases minoritarias sean aprendibles. Con N=300 (75 rojos), hay margen.
- No debe incluir `calificacion_final` ni `promedio_final_curso` como features (evitar leakage).

### Métricas

- **Macro-F1** (igual peso a las 3 clases)
- **Matriz de confusión** con enfoque en la clase "rojo" (FN = alumno en riesgo que no se identifica)
- **Precision/Recall por clase**

---

## Misión 2 — Alerta de Churn Estudiantil

### Pregunta de negocio

> **¿Este alumno abandonará la materia a mitad de semestre?**

### Tipo propuesto

**Clasificar** (binaria). Justificación: `abandona` es una variable **binaria (0/1)**, claramente una etiqueta de clase. La pregunta es "sí/no", no "cuántos puntos" o "qué día".

### Distribución de Y e implicaciones

- N = 500, `abandona=1`: 70 (14%), `abandona=0`: 430 (86%)
- **Desbalance de 86:14 ≈ 6:1**

**Cálculo:** 70 casos positivos.

**¿Qué pasa si alguien reporta "86% de aciertos"?**

| Escenario | Accuracy | ¿Significa algo? |
|-----------|----------|------------------|
| Modelo "dummy" (siempre 0) | 86% | **NO.** No detecta ni un solo abandono. |
| Modelo real (detecta 50/70) | ~88% | Sí, pero accuracy no muestra la diferencia. |

El "86% de aciertos" del dummy es **engañosamente alto**. Con 14% de positivos, un clasificador que nunca predice abandono ya alcanza 86%. Se necesita **más allá de accuracy**.

### Tratamiento de NA en `calif_actividad_1`

**Hipótesis:** El missing de `calif_actividad_1` **NO es aleatorio**. Es más frecuente en alumnos que `abandona=1` (según la pista: "calif_actividad_1 faltante ↔ más abandono, Fuerte"). Esto sugiere que los alumnos que se van **no entregan la actividad** → el NA es **informativo** (Missing Not At Random, MNAR).

**Tratamiento recomendado:**
1. **Bandera binaria:** `tiene_calif_1` (0/1). Captura el hecho de que falta la calificación.
2. **Imputación con bandera:** Imputar el NA con la **mediana** del grupo que abandonó (o global), y usar `tiene_calif_1` como feature separado.
3. **NO imputar con la media global** sin la bandera: se perdería la señal de que el alumno no participó.

### Leakage

**No incluirías** `fecha_de_baja_definitiva` ni `nota_final`. Ambas son **post-evento**: la fecha de baja se conoce solo después de que el alumno abandonó, y la nota final solo existe si completó. Usarlas como X es **fuga perfecta**.

### Tres preguntas EDA elegidas

#### 1. ¿Cómo está distribuida Y?

- 86% vs 14%. **Moderado desbalance.**
- No es tan extremo como P1 del dino (0.4%), pero suficiente para que accuracy sea engañosa.
- **Solución:** usar `F1-score` (o `AUPRC`) como métrica primaria, y `class_weight='balanced'` en el modelo.

#### 2. ¿Hay outliers en `dias_sin_login`?

- Valores de 25-30 días son **extremos**. En un semestre de 15 semanas (~105 días), 30 días sin login es 28% del tiempo.
- **¿Outlier?** Sí, pero **no es error de captura**. Si el semestre dura ~45 días hasta la mitad, 30 días de inactividad es real.
- **Interpretación:** No se eliminan, pero se puede **categorizar**: `inactivo_grave = dias_sin_login > 14`. Los alumnos con 14+ días sin login tienen `abandona=1` en 100% de los casos observados.

#### 3. ¿Los casos `abandona=1` se concentran en alumnos que trabajan?

- Pistas dicen: `trabaja=1 ↔ más abandono` (débil-moderada).
- De los 70 abandonos, si ~40 son de alumnos que trabajan, el 57% de abandonos viene del 27% de alumnos que trabajan.
- **No es determinista** (trabajar no implica abandonar), pero es un factor de riesgo.
- **Conclusión:** `trabaja` es un feature útil pero no suficiente. No se binariza Y basado en esto.

### Comparación con Misión 1

| Aspecto | M1 (Semáforo) | M2 (Churn) |
|---------|---------------|------------|
| Y tipo | Multiclase (3) | Binaria (0/1) |
| Balance | Moderado (40/35/25) | Desbalanceado (86/14) |
| NA críticos | Pocos | `calif_actividad_1` MNAR |
| Métrica clave | Macro-F1 | F1-score, AUPRC |
| Modelo | RF multiclase | RF binaria con class_weight |
| Riesgo de error | FP: sobrestimar riesgo | FN: no detectar abandono |

### Métricas

- **F1-score (positiva)** — balancea precision y recall en la clase minoritaria
- **AUPRC** — más informativa que ROC-AUC en datasets desbalanceados
- **Recall > Precision** — preferible: mejor detectar un abandón de más que no detectar uno real

### Modelo propuesto

**Random Forest con `class_weight='balanced'`**

**Costo de error:** Un **falso negativo** (no detectar un abandono) es 5x más costoso que un falso positivo (alertar a un alumno que no se va): el FN implica pérdida de retention, mientras que el FP genera una tutoría innecesaria.

---

## Misión 3 — Pronóstico de Puntaje Final

### Pregunta de negocio

> **¿Qué calificación final (escala 0–100) obtendrá el alumno en esta materia?**

### Tipo propuesto

**Predecir** (regresión). Justificación: `calificacion_final` es un **valor numérico continuo** (0–100), con decimales. No se trata de categorizar aprobado/reprobado; se quiere el **número exacto**. La pregunta pide una predicción puntual.

### Y y X

- **Y:** `calificacion_final` (continua, 28–99)
- **X (4 mínimas):**
  1. `promedio_tareas` — el promedio de tareas (media 74) correlaciona fuerte con la nota final.
  2. `examen_1` — examen 1 (media 71), correlación ~0.85 con Y.
  3. `examen_2` — examen 2 (media 73), correlación ~0.85 con Y.
  4. `asistencia_pct` — asistencia promedio 82%, correlación moderada con rendimiento.
  5. `horas_estudio_sem` — estudio auto-reportado (media 5.5), correlación moderada con Y.

### Variable más relacionada con Y

**`examen_1`** (correlación ~0.85). El examen 1 es el mejor predictor lineal de la calificación final: si el alumno sacó 85 en el examen 1, la calificación final tenderá a estar cerca de 85 ± 5 puntos.

### Convertir a aprobado/reprobado: ¿qué se gana y pierde?

| Aspecto | Ganancia | Pérdida |
|--------|----------|---------|
| **Interpretación** | Más simple para tutores: "¿se va a recuperar?" | Pierdes granularidad: 70 y 71 son "aprobado" pero 69 y 70 son "reprobado" — diferencia psicológica irrelevante. |
| **Modelo** | Binary classification, métricas claras (precision/recall) | No aprovechas toda la información numérica; el modelo regresor usa más señal. |
| **Costo de error** | FN = alumno que repite (costo claro) | No sabes *en qué medida* falló; un 69 vs un 30 requieren intervenciones distintas. |

### Interpreta media vs mediana de `horas_estudio_sem`

- Media = 5.5, Mediana = 5.0
- **Diferencia pequeña (0.5)** → **sesgo leve a derecha**, pero no extremo.
- Sin embargo, el análista previo menciona que algunos reportan **20–25 horas** (máx = 25). La desviación estándar es 3.2.
- **Z-score de 25h:** (25 - 5.5) / 3.2 ≈ **6.1σ** → es un **outlier extremo**.
- Estos valores son **inprobables** (20-25 horas/sem de estudio efectivo). Son likely errores de captura o auto-reporte inflado (confunden estudio con tiempo frente a la pantalla).

### ¿Qué harías con las 3 filas con `calificacion_final > 100`?

- **No son válidas.** La escala es 0–100. Valores >100 son errores de captura.
- **Acción:** Eliminar las 3 filas, o imputar con el valor máximo permitido (100) si el alumno claramente aprobó con excelencia.
- **Preferencia:** Eliminarlas. 3 filas de 200 es solo 1.5%, no impacta significativamente el modelo. La alternativa de imputar a 100 introduce sesgo.

### Métricas (2)

| Métrica | Qué mide | Por qué aquí |
|---------|----------|---------------|
| **RMSE (Root Mean Squared Error)** | Error típico en puntos de la escala 0–100. Penaliza errores grandes cuadráticamente. | Ideal cuando errores grandes (ej. predecir 50 cuando el real es 95) son costosos. |
| **MAE (Mean Absolute Error)** | Error promedio absoluto en puntos. | Más robusto a outliers. Fácil de interpretar: "el modelo se equivoca en promedio 6 puntos". |

### Si `examen_1` vs `calificacion_final` es casi una línea → modelo simple

- Si la relación es **lineal**: Regresión Lineal Simple (o Múltiple con examen_1 y examen_2).
- Si la relación es en **escalones** (ej. examen_1=70→final=72, examen_1=80→final=82, pero examen_1=85→final=95): usar **Regresión Lineal Múltiple** o **Regression Tree** (árbol de regresión) para capturar thresholds.

### Comparación con Misión 1

| Aspecto | M1 (Semáforo) | M3 (Puntaje) |
|---------|---------------|--------------|
| Y tipo | Categórica (3 classes) | Numérica continua (0–100) |
| Modelo | Clasificación | Regresión |
| Métrica | Macro-F1 | RMSE / MAE |
| Outliers | Moderados | Extremos (25h de estudio, >100 puntos) |
| Riesgo de error | Clase minoritaria (rojo) | Magnitud del error (cuántos puntos) |

### Modelo propuesto

**Regresión Lineal Múltiple** (o Ridge/Lasso si hay colinealidad entre examen_1 y examen_2).

---

## Misión 4 — Estimación de Tiempo de Estudio

### Pregunta de negocio

> **¿Cuántas horas adicionales necesita un alumno para dominar un tema específico?**

### Tipo propuesto

**Predecir** (regresión). Justificación: `horas_adicionales` es un **número continuo positivo** (1.0, 5.5, 35.0). No es una categoría. Se quiere predecir el tiempo exacto de estudio necesario.

### Hipótesis dificultad → horas

**Hipótesis:** A mayor `tema_dificultad`, más `horas_adicionales` se necesitan.

| Dificultad | Media horas | Mediana horas |
|------------|-------------|---------------|
| Baja | 3.2 | 2.5 |
| Media | 7.8 | 7.0 |
| Alta | 16.5 | 14.0 |

**Relación casi lineal:** baja ≈ 3, media ≈ 8, alta ≈ 16.5. El doble de dificultad (baja→alta) implica ~5x más horas. La variable `tema_dificultad` codificada ordinalmente (baja=1, media=2, alta=3) debería ser un predictor fuerte y lineal.

### Cómo inspeccionar categóricas en EDA

| Variable | Técnica |
|----------|---------|
| `tema_dificultad` | Agrupar por dificultad y comparar medias de `horas_adicionales` (como arriba). Usar ANOVA o Kruskal-Wallis para test de diferencia significativa. |
| `dispositivo` | Cruz con `horas_adicionales` y `ejercicios_correctos_pct`. Hipótesis: `pc` puede tener más disciplina; `movil` más interrupciones. Boxplots por dispositivo. |

### Cola larga: ¿borrar o conservar?

- Distribución: 0–5h (35%), 5–10h (30%), 10–20h (22%), 20–40h (10%), >40h (3%).
- La cola >40h es solo 3% (7-8 casos en N=240).

**Decisión:** **Conservar**. Argumento:

- Si la cola representa **alumnos realmente necesitando mucho estudio** (ej. dificultad alta + bajo pretest), borraría información valiosa.
- La regla de 3% no justifica eliminación si hay una **causalidad plausible**.
- En su lugar: aplicar **transformación logarítmica** a `horas_adicionales` y usar modelos robustos (Gradient Boosting, no lineal), o **winsorizar** al 97º percentil.

### Redundancia: `pretest_score` y `ejercicios_correctos_pct`

- `pretest_score` mide conocimiento previo; `ejercicios_correctos_pct` mide desempeño durante el estudio. Ambas están negativamente correlacionadas con `horas_adicionales`, pero miden constructos distintos.
- **Cómo checar:** Correlación de Pearson entre ambas. Si r > 0.7, hay redundancia. Si r < 0.5, ambas aportan información única.
- En la muestra: `pretest_score` 85 con `ejercicios_correctos_pct` 90 → ambas altas. Pero `pretest_score` 40 con `ejercicios_correctos_pct` 45 → ambas bajas. Posible correlación moderada (~0.6), no severa.

### Alternativa de binarizar Y (umbral 15h)

**Cuándo tendría sentido:**
- Si el tutoría tiene capacidad limitada (solo puede asignar tutorías intensivas a 10 alumnos/mes).
- Si la decisión es binaria: "¿se le asigna tutoría intensiva o no?"

**Qué se pierde:**
- Granularidad: un alumno que necesita 14h y otro que necesita 1h son ambos "no intensivo", pero requieren intervenciones muy distintas.
- Información sobre magnitudes: el modelo pierde capacidad de decir "necesita 28h, asignarle 2h/sem durante 14 semanas".

### Métricas y modelo

**Métricas (regresión):**
- **MAE (Mean Absolute Error)** — fácil de interpretar para tutores: "el modelo se equivoca en promedio 3 horas".
- **RMSE** — penaliza errores grandes (predecir 2h cuando necesita 35h). Importante en este caso.

**Modelo propuesto:** Gradient Boosting Regressor (XGBoost, LightGBM)

**2 chequeos EDA obligatorios:**
1. **Distribución de `horas_adicionales`:** verificar normalidad o necesidad de transformación log. Si la cola es severa, usar log(Y).
2. **i.i.d. por alumno_id:** asegurar que el mismo alumno no aparezca en train y test (si hay múltiples temas por alumno, dividir por alumno_id).

---

## Síntesis

### Tabla resumen de las 4 propuestas

| Misión | Pregunta | Tipo propuesto | Y | Modelo propuesto | Métrica clave |
|--------|----------|----------------|---|-------------------|----------------|
| M1 | Semáforo académico | Clasificar (multiclase) | `riesgo` (verde/amarillo/rojo) | Random Forest multiclase | Macro-F1 |
| M2 | Alerta churn | Clasificar (binaria) | `abandona` (0/1) | RF con class_weight='balanced' | F1-score / AUPRC |
| M3 | Puntaje final | Predecir (regresión) | `calificacion_final` (0–100) | Regresión Lineal Múltiple | RMSE / MAE |
| M4 | Horas de estudio | Predecir (regresión) | `horas_adicionales` (continua) | Gradient Boosting Regressor | MAE / RMSE |

### Una pista para decidir "clase vs número" en cualquier misión

> **Si la variable Y que el negocio pide es una etiqueta, categoría o 0/1 → clasificar. Si pide un número exacto, monto, duración o índice continuo → predecir.**

### Frase final

El tipo de problema se deduce de la pregunta y de Y porque la naturaleza de la variable objetivo determina qué métrica es coherente, qué modelo es apropiado y cómo se interpreta el costo de error; elegir mal el tipo es como usar un destornillador para un clavo: el EDA es el paso que revela la forma real de los datos antes de aplicar el instrumento.
