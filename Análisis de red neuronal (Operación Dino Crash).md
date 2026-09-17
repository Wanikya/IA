# Conclusión: ¿Red Neuronal para P1 (Dino Crash)?

**Respuesta: NO.**  
El EDA realizado en `reporte_dino_crash_eda.md` **desaconseja explícitamente** el uso de cualquier arquitectura de red neuronal (perceptrón simple, MLP, CNN, LSTM) para el escenario P1 con el dataset actual.

---

## Justificación estructurada por hallazgos del EDA

| # | Hallazgo del EDA | Evidencia numérica | Por qué invalida a una NN |
|---|------------------|-------------------|---------------------------|
| 1 | **Tamaño efectivo de clase positiva minúsculo** | ~100–150 casos `will_die_next_frame=1` en 50,000 frames | Una MLP mínima (2 capas, 64→32 neuronas) ≈ 5,000–10,000 parámetros. Ratio **parámetros / positivos ≈ 50:1–100:1** → overfitting severo. |
| 2 | **Datos tabulares de baja dimensionalidad** | 7–8 features numéricas/categóricas | Las NN no tienen *inductive bias* ventajoso frente a árboles en datos tabulares estructurados. No hay imágenes, audio, texto ni secuencias largas. |
| 3 | **Predictor dominante con umbral físico claro** | `dist_obstacle < 20 px` concentra ~60% de muertes en 3% de frames | Un árbol de profundidad 3–4 aprende este umbral **exactamente** (partición del espacio). Una NN lo aproxima con miles de parámetros y sin interpretabilidad. |
| 4 | **Interacción no lineal simple: `speed × dist_obstacle`** | A 13.0 speed, umbral crítico ≈ 35 px; a 6.0 speed, umbral ≈ 20 px | Un Random Forest captura esta interacción nativamente. Un perceptrón simple **no puede** (lineal en features). Un MLP la aprende pero con overfitting (ver #1). |
| 5 | **Desbalance extremo** | Positivos = 0.3% (ratio 240:1) | NN requieren *class weighting*, *focal loss*, *oversampling* cuidadoso, validación estratificada por sesión. Árboles: `class_weight='balanced'` nativo, estable. |
| 6 | **Violación i.i.d.: autocorrelación temporal por sesión** | Frames 80, 81, 82 de sesión 7 comparten speed=6.8, score=16, obstacle=cactus_small | Si se barajan frames en batches, la NN **memoriza sesiones** (leakage temporal). Requiere GroupKFold + arquitectura recurrente → más parámetros, más datos. |
| 7 | **Baseline heurístico fuerte** | Regla `dist_obstacle < 20 AND jump=0` → recall/precision altos sin entrenar | Una NN debe batir este baseline con 100 positivos. Probabilidad de éxito: baja. |

---

## Lo que el reporte original dice textualmente

> **Misión 3, Pregunta 1:**  
> *"N es suficientemente grande para Random Forest y regresión logística, pero **no** para redes neuronales profundas. Un modelo con 50,000 parámetros sería un riesgo de overfitting."*

> **Tabla guía (Misión 5), fila 1:**  
> | Si tu EDA encuentra… | Modelos poco razonables (y por qué) |  
> |---------------------|------------------------------------|  
> | Y binaria, tabular, N mediano | **Red profunda sin más datos (overfitting)** |

> **Misión 6, contraejemplo NN:**  
> *"Escenario donde una red neuronal tendría sentido: **1,000,000+ frames de 2,000+ partidas, con imágenes de pantalla**... Condiciones: N muy grande, alta dimensionalidad, no linealidad compleja."*

---

## El único caso "neuronal" que sí encaja: Perceptrón simple = Regresión Logística

| Modelo | Parámetros | Captura interacción `speed × dist` | Recomendación |
|--------|------------|-----------------------------------|---------------|
| **Perceptrón simple (1 capa, sin hidden layers)** | ~8 pesos | **No** (lineal) | Úsalo como **baseline lineal** (equivalente a Regresión Logística con `class_weight='balanced'`). |
| **MLP (1+ hidden layers)** | 1,000–50,000+ | Sí | **No** — overfitting con 100 positivos. |
| **Random Forest (100 árboles, max_depth=5)** | ~500 nodos efectivos | Sí (particiones condicionales) | **Sí — modelo propuesto por el EDA**. |

---

## ¿Qué tendría que cambiar para que una NN tenga sentido?

| Condición necesaria | Qué implica en la práctica |
|---------------------|----------------------------|
| **Input = imágenes crudas (80×80 px)** | CNN > árboles para patrones espaciales (posición obstáculo, altura dino). |
| **Secuencia temporal (10–20 frames)** | LSTM/GRU captura dinámica de "caída anticipada". |
| **N > 500,000 frames con >5,000 positivos** | Ratio parámetros/casos aceptable. |
| **Multi-tarea (muerte + tipo obstáculo + posición)** | *Shared representation* justifica capacidad extra. |

**Con el dataset actual: ninguna se cumple.**

---

## Decisión final (post-EDA)

| Paso | Acción |
|------|--------|
| 1. **Baseline** | Regla fija `dist_obstacle < 20 AND jump=0` (sanity check, 0 código ML). |
| 2. **Modelo principal** | **Random Forest** `class_weight='balanced'` + *threshold tuning* (maximizar F1/PR-AUC). |
| 3. **Validación** | **GroupKFold por `session_id`** (nunca mezclar frames de la misma partida en train/test). |
| 4. **Interpretabilidad** | *Feature importance* + *partial dependence plots* de `dist_obstacle` × `speed`. |
| 5. **NN** | **Descartada** — no resuelve ningún problema que el RF no resuelva mejor con menos datos. |