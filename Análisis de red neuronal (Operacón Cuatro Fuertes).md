# Conclusión: ¿Red Neuronal para las 4 Misiones de Cuatro Frentes?

**Respuesta general: NO sirven para hacer redes neuronales de perceptrón simple o perceptrón multi capa.**  
El EDA de `reporte_cuatro_frentes_eda.md` **desaconseja** redes neuronales para los 4 frentes. Sin embargo, hay matices: el **perceptrón simple** (≈ regresión logística/lineal) sí funciona en **M2 y M3** como baseline aceptable.

---

## Resumen misión a misión

| Misión | N | Tipo de Y | Perceptrón simple | MLP pequeña | Modelo recomendado por EDA |
|--------|---|-----------|-------------------|-------------|----------------------------|
| **M1** Semáforo Académico | 300 | Multiclase (3 clases) | Lineal no basta | 100 ejemplos/clase → overfitting | **Random Forest multiclase** |
| **M2** Alerta de Churn | 500 | Binaria (0/1, 14%) | Baseline (reglog) | 70 positivos → overfitting | **RF con `class_weight='balanced'`** |
| **M3** Pronóstico de Puntaje | 200 | Continua (0–100) | ≈ Regresión Lineal | Overkill (relación lineal) | **Regresión Lineal Múltiple** |
| **M4** Tiempo de Estudio | 240 | Continua (0–40h) | Lineal no basta | Riesgoso (outliers) | **Gradient Boosting Regressor** |

---

## Justificación detallada por misión

### Misión 1 — Semáforo Académico

| Aspecto | Detalle |
|---------|---------|
| **N** | 300 filas, 100 por clase |
| **Problema** | Clasificación multiclase |
| **¿Perceptrón simple funciona?** | **No.** Fronteras entre verde/amarillo/rojo no son lineales (ej. asistencia=85%, reprobadas_previas=0 puede ser verde o amarillo según promedio). |
| **¿MLP funciona?** | **No.** ~3,000 parámetros vs ~100 ejemplos/clase → ratio **30:1**. Overfitting garantizado. |
| **Hallazgo EDA clave** | Separación casi lineal por `promedio_parciales`, pero interacciones no lineales con `turno` y `asistencia_pct` |
| **Modelo del EDA** | Random Forest + Macro-F1 |

---

### Misión 2 — Alerta de Churn Estudiantil

| Aspecto | Detalle |
|---------|---------|
| **N** | 500 filas, 70 positivos (14%) |
| **Problema** | Clasificación binaria desbalanceada |
| **¿Perceptrón simple funciona?** | **Sí, como baseline.** Equivalente exacto a regresión logística con `class_weight='balanced'`. ~7 pesos, bajo riesgo de overfitting. Captura relación lineal entre `dias_sin_login` y `abandona`. |
| **¿MLP funciona?** | **No.** 70 positivos vs 3,000+ parámetros → overfitting. NA en `calif_actividad_1` complica aún más el preprocesamiento de NN. |
| **Hallazgo EDA clave** | `dias_sin_login > 14` → `abandona=1` en 100% de casos. Relación fuerte pero no capturada por un perceptrón lineal solo. |
| **Modelo del EDA** | RF con `class_weight='balanced'`, métrica: F1-score/AUPRC |

---

### Misión 3 — Pronóstico de Puntaje Final

| Aspecto | Detalle |
|---------|---------|
| **N** | 200 filas |
| **Problema** | Regresión (0–100) |
| **¿Perceptrón simple funciona?** | **Sí, funciona perfectamente.** Correlaciones ~0.85 entre `examen_1`, `examen_2` y `calificacion_final` → relación **casi lineal**. Un perceptrón simple (sin hidden layers, función identidad) es **equivalente a regresión lineal múltiple**. ~6 pesos, ~200 samples → ratio 33:1, aceptable. |
| **¿MLP funciona?** | **Overkill.** No hay no-linealidad compleja que capturar. Agregar capas ocultas no mejora el rendimiento y aumenta riesgo de overfitting con N=200. |
| **Hallazgo EDA clave** | `examen_1` vs `calificacion_final` → "casi una línea" |
| **Modelo del EDA** | Regresión Lineal Múltiple (o Ridge/Lasso) |

---

### Misión 4 — Estimación de Tiempo de Estudio

| Aspecto | Detalle |
|---------|---------|
| **N** | 240 filas |
| **Problema** | Regresión (horas continuas, 0–40) |
| **¿Perceptrón simple funciona?** | **Insuficiente.** La relación entre `tema_dificultad` y `horas_adicionales` es no lineal (3.2 → 7.8 → 16.5). La cola larga (>40h en 3%) no es lineal. Un perceptrón lineal subestima los extremos. |
| **¿MLP funciona?** | **Tentador pero arriesgado.** 240 samples, outliers (Z≈6σ en `horas_estudio_sem`). MLP con función de pérdida cuadrática es **sensible a outliers**. No hay suficientes datos para regularizar. |
| **Hallazgo EDA clave** | `tema_dificultad` → medias: 3.2/7.8/16.5h. Relación casi lineal ordinal, pero no lineal continua. |
| **Modelo del EDA** | Gradient Boosting Regressor + MAE/RMSE |

---

## Regla general de las 4 misiones

> **Una red neuronal tiene sentido solo cuando se cumple AL MENOS UNA de estas condiciones:**
> 1. **N > 1,000 × número de parámetros** del modelo
> 2. **Y de alta dimensionalidad** (imágenes, secuencias temporales largas, texto)
> 3. **Relaciones no lineales complejas** sin alternativa interpretable (ej. patrones espaciales en imágenes)
>
> **Las 4 misiones comparten:**
> - N = 200–500
> - Features tabulares (5–8)
> - Relaciones predominantemente lineales o simples interacciones arbóreas
> - **→ Los modelos de árboles y regresión lineal son superiores en rendimiento, interpretabilidad y robustez.**

---

## ¿Cuándo SÍ justificaría una NN? (según el EDA)

| Misión | Cambio necesario | Arquitectura |
|--------|-------------------|--------------|
| **M1** | Convertir `asistencia_pct`, `promedio_parciales` en imagen de heatmap por alumno | CNN 1D + clasificador |
| **M2** | Añadir secuencia temporal de logins (últimos 30 días como serie) | LSTM con embeddings de `alumno_id` |
| **M3** | Escribir el texto de la actividad y usar NLP para predecir nota | Transformer + regresión sobre embeddings |
| **M4** | Imágenes de la pantalla de tutoría + video del progreso | CNN + LSTM multi-modal |

**Con los datos actuales: ninguna se cumple.**

---

## Veredicto final

| Modelo | M1 | M2 | M3 | M4 |
|--------|----|----|----|-----|
| Perceptrón simple | No | baseline | ≈ RegLin | No |
| MLP (1-2 capas) | No | No | overkill | arriesgado |
| Random Forest | Sis | Si | posible | Si |
| Gradient Boosting | Si | Si | posible | Si |
| Regresión Lineal | No | No | Si | No |
| **Elegido por EDA** | **RF** | **RF** | **RegLin** | **XGBoost** |
