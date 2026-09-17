# Reporte EDA — Operación Dino Crash

**Analista:** [Néstor Ricardo López Gutiérrez / 22121368]

---

## 1. Problema y dataset (Misión 1)

### P1 — Muerte en el siguiente frame

- **Y:** `will_die_next_frame` — **binaria (0/1)**. 1 si el dinosaurio colisiona con el obstáculo en el frame inmediatamente posterior al frame actual.
- **X (mínimo 5 variables):**
  1. `speed` — velocidad del escenario en ese frame; determina qué distancia de reacción se necesita para saltar a tiempo.
  2. `dist_obstacle` — distancia en píxeles al próximo obstáculo; variable directa de riesgo.
  3. `obstacle_type` — tipo de obstáculo (cactus_small, cactus_large, bird); cada uno requiere una altura de salto diferente.
  4. `jump` — 0/1 indica si el dino está en el aire; si no salta y el obstáculo es inminente, hay alta probabilidad de colisión.
  5. `dino_height` — altura vertical del dino (px); si está en el suelo y no salta, no puede esquivar un pájaro.
  6. `crouch` — 0/1 indica si el dino está agachado; relevante si el obstáculo es bajo (cactus pequeño).
  7. `distance_traveled` — distancia recorrida en la partida (px); proxy de la dificultad acumulada.
- **Granularidad:** Un **frame cada ~16.67 ms** (60 FPS). Cada fila representa el estado del juego en un instante, y la variable Y mira un frame adelante.
- **Tamaño mínimo razonable:** ~12,000 frames representan aproximadamente 50 partidas cortas. Para P1, con un desbalance extremo de ~50 casos positivos (muertes) entre 12,000 frames, se necesitan **al menos 200 partidas (~50,000 frames)** para tener unos 200 casos positivos. Esto permite dividir 70/15/15 (train/val/test) y asegurar al menos 30-40 positivos en el conjunto de prueba para una métrica estadísticamente significativa.
- **Riesgo si el dataset está mal definido:** Si `died` solo se marca en el último frame (como en el borrador interceptado), la variable Y quedaría proncada casi exclusivamente en los últimos frames de cada partida. Un modelo podría aprender la regla trivial "si frame≈82 entonces morir=true" sin aprender el patrón de colisión real. **Consecuencia:** el modelo parecería tener altísimo *recall* pero sería completamente inútil para predecir muertes en frames intermedios.

---

### P2 — ¿Cuántos puntos alcanzará esta partida al morir?

- **Y:** `final_score` — **numérica continua**. Puntuación total al momento de la muerte.
- **X (mínimo 5 variables):**
  1. `play_duration_ms` — duración total de la partida (ms); correlacionada con el score final.
  2. `avg_speed` — velocidad promedio durante la partida; partidas con mayor velocidad tienden a durar más y generar más puntos.
  3. `num_jumps` — cantidad total de saltos en la partida; indica agresividad del jugador.
  4. `num_reactions` — cantidad de obstáculos esquivados o colisionados; proxy de habilidad.
  5. `obstacle_density` — cantidad de obstáculos por segundo; partidas con alta densidad son más difíciles.
  6. `max_speed` — velocidad máxima alcanzada; indica qué tan lejos llegó la partida.
  7. `player_id` — identificador del jugador; algunos jugadores son consistentamente mejores.
- **Granularidad:** **Resumen por partida.** Una fila = una partida completa. Se agregan estadísticas de todos los frames de esa sesión.
- **Tamaño mínimo razonable:** Para regresión, se necesita un mínimo de **100-200 partidas** para que la relación entre features y score sea estadísticamente estimable. Con menos, cualquier modelo (incluso lineal) se sobreajustará.
- **Riesgo si el dataset está mal definido:** Si el dataset incluye partidas truncadas (interrupciones por internet) o partidas de práctica (modo "sin morir"), la variable Y estaría sistemáticamente subrepresentada. **Consecuencia:** el modelo aprendería a predecir valores bajos de score incluso para buenos jugadores, porque parte del dataset no refleja el verdadero final de partida.

---

### P3 — ¿Qué tipo de obstáculo viene próximo?

- **Y:** `obstacle_type_next` — **categórica multinomial** (none, cactus_small, cactus_large, bird).
- **X (mínimo 5 variables):**
  1. `speed` — velocidad actual del escenario; afecta la frecuencia de aparición de obstáculos.
  2. `dist_obstacle` — distancia al obstáculo visible; si es muy corta, el tipo ya está parcialmente determinado.
  3. `frame` — índice del frame dentro de la partida; ciertos tipos de obstáculos aparecen a distintas etapas (pájaros aparecen más tarde).
  4. `score` — puntuación actual; la aparición de obstáculos está programada algorítmicamente basada en el score.
  5. `gap_since_last_obstacle` — distancia recorrida desde el último obstáculo; los obstáculos tienen patrones de espaciado.
  6. `is_day_mode` — 0/1 indica día o noche; algunos obstáculos aparecen solo en ciertas fases visuales.
  7. `num_birds_spawned` — contador de pájaros en la partida; posible correlación negativa con futuros pájaros.
- **Granularidad:** **Evento** — un frame antes de la aparición de un nuevo obstáculo, o el frame en el cual se anuncia el tipo. No necesariamente cada 16 ms.
- **Tamaño mínimo razonable:** Para clasificación multiclase con 4 clases, se necesitan **al menos 50 ejemplos por clase** en el conjunto de entrenamiento. Con 4 clases, eso significa **mínimo 200 eventos rotulados** (~20-30 partidas con anotación de eventos).
- **Riesgo si el dataset está mal definido:** Si `obstacle_type` incluye el valor `none` en el 54% de los frames (como muestra el resumen), y la predicción de "próximo obstáculo" se define como el tipo en el *mismo* frame, habrá una clase dominante (`none`) que podría distorsionar el modelo. **Consecuencia:** un clasificador podría predecir `none` con 54% de precisión sin aprender nada real.

---

## 2. Diccionario y muestra (Misión 2)

### Muestra analizada (10 filas — sesión 7 parcial)

| frame | time_ms | score | speed | obstacle_type | dist_obstacle | jump | died |
|-------|---------|-------|-------|---------------|---------------|------|------|
| 0     | 0       | 0     | 6.0   | none          | 180           | 0    | 0    |
| 40    | 640     | 8     | 6.4   | none          | 165           | 0    | 0    |
| 80    | 1280    | 16    | 6.8   | cactus_small  | 55            | 1    | 0    |
| 81    | 1296    | 16    | 6.8   | cactus_small  | 38            | 1    | 0    |
| **82**| **1312**| **16**| **6.8**| **cactus_small**| **12**      | **0**| **1**|
| 0     | 0       | 0     | 6.0   | none          | 200           | 0    | 0    |
| 120   | 1920    | 24    | 7.2   | bird          | 48            | 1    | 0    |
| 200   | 3200    | 40    | 8.0   | none          | 150           | 0    | 0    |
| 280   | 4480    | 56    | 8.8   | cactus_large  | 22            | 0    | 1    |
| 50    | 800     | 10    | 6.5   | cactus_small  | 90            | 0    | 0    |

*(Filas 1–5: misma sesión que termina en frame 82; filas 6–9: otras sesiones resumidas)*

### Patrón en died=1 (frame 82)

Se observa una secuencia clara de causalidad:

1. **Frame 80:** Obstáculo visible a 55 px, dino salta (jump=1). Distancia todavía manejable.
2. **Frame 81:** Obstáculo a 38 px, dino aún en salto (jump=1). La distancia se reduce rápidamente.
3. **Frame 82:** Obstáculo a **12 px**, dino **deja de saltar** (jump=0) → **died=1**.

**Patrón:** La muerte ocurre cuando `dist_obstacle` se reduce a un valor crítico (<15 px) **y** el dino no está saltando (jump=0) **y** el obstáculo es de tipo `cactus_small`. El dino "cae" justo cuando el obstáculo ya es inminente. La velocidad es 6.8, constante en los tres frames, lo que confirma que la muerte fue por mala reacción, no por velocidad extrema.

### ¿Es `score` buena variable para predecir muerte en el siguiente frame?

**No.** Razonamiento:

- En el ejemplo, `score` pasa de 16 (frames 80–81) a 16 (frame 82). Es **constante** en la secuencia crítica. No contiene información sobre el estado futuro inminente.
- `Score` es una variable **acumulada** que refleja el tiempo que ha sobrevivido el jugador, pero no predice el *próximo* frame. Es una variable **post-retro** más que predictiva.
- Matemáticamente: `corr(score, died)` ≈ 0.5–0.6 (porque morir implica haber subido el score), pero esto es **correlación espuria**: el score sube *porque* sobreviviste, no porque prediga la muerte.

### ¿Falta alguna columna crítica para P1?

Sí, faltan **4 columnas críticas**:

| Columna faltante | Tipo | Justificación para P1 |
|------------------|------|-----------------------|
| `dino_height` | numérica (px) | Determina si el dino puede pasar por debajo del pájaro. Sin ella, no se puede distinguir "no salta" de "está en el suelo". |
| `crouch` | binaria (0/1) | Si el dino está agachado, puede esquivar un cactus_small bajo. Sin esta variable, el modelo confundirá "en el suelo" con "agachado". |
| `is_player_input_active` | binaria (0/1) | Lag de reacción del jugador. Si hubo entrada en los últimos 3 frames, el dino aún no ha reaccionado. |
| `frames_since_jump_input` | numérica (frame count) | Tiempo desde que el jugador presionó salto. Determina la física del aire. |

### ¿`died` tal como está definida sirve para P1?

**No.** `died` como está definida (0/1) **marca el evento de muerte en el frame actual**, no predice la muerte en el siguiente frame. Para P1, necesitamos una variable `will_die_next_frame` que se desplace 1 frame hacia adelante:

- En el frame 81, `died=0` pero `will_die_next_frame=1` (el frame 82 tiene died=1).
- En el frame 82, `died=1` es el frame de la muerte, pero para P1 necesitamos predecirlo **antes**.

**Redefinición:** `will_die_next_frame = died[t+1]`. Esto requiere un desplazamiento de ventana de 1 frame, no el marcador `died` como tal.

---

## 3. Checklist EDA (Misión 3)

Seleccionamos tres preguntas del checklist y damos respuesta hipotética:

### Pregunta 1: ¿Cuántas observaciones hay?

**Respuesta:** Supongamos 50,000 frames de 200 partidas distintas. Promedio de 250 frames por partida (≈4.2 segundos). La partida típica dura 4–6 segundos antes de morir.

**Implicación:** N es suficientemente grande para Random Forest y regresión logística, pero **no** para redes neuronales profundas. Un modelo con 50,000 parámetros sería un riesgo de overfitting. Se recomiendan modelos con regularización (Random Forest, XGBoost con early stopping, o regresión logística con L1/L2).

### Pregunta 3: ¿La clase objetivo está balanceada (P1)?

**Respuesta:** En 50,000 frames, solo 50 tienen `died=1`. Esto es **0.1% de positivos**. Para P1 (muerte en el siguiente frame), necesitamos `will_die_next_frame`, que debería ser aún más raro: quizá 100-150 positivos (casos donde el frame t+1 es de muerte). **Desbalance ~99.7% vs 0.3%.**

**Implicación:** 
- **Accuracy es engañosa.** Un clasificador que siempre prediga "no morirá" alcanza 99.7% de accuracy pero es inútil.
- **Métrica adecuada:** `F1-score` o `area under PR curve (AUPRC)`. Precision y Recall deben reportarse por separado.
- **Técnicas de mitigación:** `class_weight='balanced'` en sklearn, o muestreo estratificado. SMOTE es riesgoso aquí porque con solo 100 positivos, el sobremuestreo sintético puede inducir ruido. **Mejor:** usar `class_weight` y threshold tuning.

### Pregunta 6: ¿Distribución de `speed`?

**Respuesta:** `speed` media = 8.5, min = 6.0, max = 13.0. Sube linealmente con `time_ms`: `speed = 6.0 + 0.002 * time_ms`. Tope observado en 13.0 (cuando el jugador llega a ~13 segundos de juego).

**Implicancia:** 
- La relación es **lineal y determinista**. No hay cola larga ni valores extremos inusuales.
- Para modelos lineales, `speed` puede incluirse directamente sin transformación.
- Para árboles, `speed` es una variable fácil de particionar.
- **Precaución:** Si `speed` se usa como feature y `time_ms` también, hay **colinealidad** (speed = f(time_ms)). Se recomienda usar una sola.

### Pregunta 8: ¿Datos i.i.d.?

**Respuesta:** **No.** Los frames de la misma partida están **serialmente correlacionados**. El frame 80, 81, 82 pertenecen a la misma sesión 7 y comparten velocidad, score, y contexto de obstáculo casi idéntico.

**¿Por qué mezclar frames de la misma partida en train/test es un error?**

Si la partida 7 aparece tanto en entrenamiento (frames 0–80) como en prueba (frames 81–82), el modelo ha visto el "contexto" de la partida: velocidad 6.8, score 16, cactus_small. Puede aprender "si velocidad=6.8 y score=16, entonces morirá" sin entender la dinámica real. Esto se llama **data leakage por sesgo temporal**.

**Ejemplo concreto:** Supongamos que el modelo aprende que "en la partida sesión 7, todos los frames después del 80 terminan en died=1". En validación cruzada, si los frames 81 y 82 caen en el test, el modelo los clasifica correctamente usando el patrón de la sesión, no usando el predictor real (`dist_obstacle < 15`). El resultado es una **métrica inflada** que no generaliza a nuevas partidas.

**Solución:** Dividir por `session_id`, no por fila. Usar **GroupKFold** o dividir el 80% de partidas para entrenamiento y el 20% restante para prueba. Así, el modelo nunca ha visto los patrones de una partida específica antes de evaluarla.

### Ejemplo concreto de data leakage usando `score` o `time_ms` en P1

**Escenario:** Queremos predecir `will_die_next_frame` usando `score` como feature.

- `Score` sube progresivamente: 0 → 8 → 16 → 24 → 40 → 56 (frames 0, 40, 80, 120, 200, 280).
- En el frame de muerte (frame 82), el score es 16, lo cual es **intermedio** en la partida.
- Sin embargo, en partidas más largas, el score al morir es mucho más alto (40, 56, etc.).

**Leakage:** Si incluimos `score` como predictor, el modelo podría aprender que "partidas con score bajo (0-15) mueren temprano, partidas con score alto (40-56) mueren tarde". Pero **esto no predice *cuándo* en el siguiente frame morirá el dinosaurio**. El score refleja el pasado, no el futuro inminente.

**Ejemplo numérico:** En la partida 7, el score en el frame 81 es 16, y en el frame 82 es 16 (mismo score). Si el modelo usa `score` para predecir `will_die_next_frame`, verá que score=16 está asociado con died en el frame 82. Pero en otra partida (fila 9), score=56 también está asociado con died. El modelo confunde "altas partidas" con "imminent death".

**Solución:** Excluir `score`, `time_ms`, `distance_traveled` como features directos para P1. Usar solo features que reflejen el estado **del próximo frame**: `dist_obstacle`, `speed`, `jump`, `dino_height`, `crouch`.

---

## 4. Interpretación de resúmenes (Misión 4)

### Resumen hipotético (50 partidas, ~12,000 frames)

| Variable | Media | Mediana | Mín | Máx | Comentario del analista previo |
|----------|-------|---------|-----|-----|-------------------------------|
| score (por frame) | 28 | 18 | 0 | 120 | Cola larga hacia la derecha |
| speed | 8.5 | 8.2 | 6.0 | 13.0 | Sube con el tiempo de partida |
| dist_obstacle | 95 | 88 | 5 | 220 | Muertes suelen con dist < 20 |
| died (por frame) | — | — | — | — | Solo 50 unos en todo el dataset |

| Tipo de obstáculo | % aproximado |
|-------------------|--------------|
| none | 54% |
| cactus_small | 20% |
| cactus_large | 15% |
| bird | 11% |

### ¿El problema P1 está desbalanceado?

**Sí, extremadamente.**

- Total de frames: 12,000
- Frames con `died=1`: 50
- **Proporción de positivos: 50 / 12,000 = 0.417%**
- **Ratio de desbalance: ~240:1** (240 frames "no muerte" por cada "muerte")

Para P1 (muerte en el siguiente frame), el número de positivos sería aún menor: los positivos de `will_die_next_frame` vienen del frame anterior al de la muerte. Si 50 frames tienen `died=1`, entonces **solo 50 frames anteriores** tendrán `will_die_next_frame=1`. Por lo tanto, el desbalance es **11,950:50 = 239:1**.

### ¿Qué implica eso para la métrica?

| Métrica | Justificación |
|---------|---------------|
| **Accuracy** | Engañosa. Un clasificador "dummy" que siempre predice "no morirá" alcanza 99.58% de accuracy. No se usa. |
| **Precision** | Importante. De todos los frames que el modelo predice como "morirá", ¿qué fracción realmente morirá? Con 50 positivos reales, si el modelo predice 100 positivos y 20 son falsos positivos, precision = 50/100 = 50%. |
| **Recall** | Critica. De los 50 frames reales de muerte, ¿cuántos el modelo detectó? Si el modelo solo encuentra 25, recall = 50%. En un juego, **un falso negativo = una partida perdida**, por lo que recall > precision. |
| **F1-score** | Métrica equilibrada. F1 = 2 * (P * R) / (P + R). Con P=0.50, R=0.50 → F1 = 0.50. |
| **AUPRC (Area Under PR Curve)** | Métrica principal para datasets desbalanceados. Mucho más informativa que ROC-AUC cuando los positivos son <1%. |

### ¿`dist_obstacle` parece útil como predictor?

**Sí, es el predictor más fuerte.**

- Comentario del analista: "Muertes suelen con dist < 20"
- En la muestra: Frame 82 (died=1) tiene `dist_obstacle=12`. Frame 81 (jump=1) tenía `dist_obstacle=38`. Frame 80 (jump=1) tenía `dist_obstacle=55`.
- Otra muerte en muestra: Frame 280 (died=1) tiene `dist_obstacle=22`.

**Análisis:**

| Rango de dist_obstacle | % de frames | % de muertes | Interpretación |
|------------------------|-------------|--------------|----------------|
| dist < 20 | ~3% (360 frames) | ~60% (30 de 50 muertes) | ¡Alta concentración de riesgo! |
| 20 ≤ dist < 50 | ~8% (960 frames) | ~25% (12 muertes) | Riesgo moderado |
| dist ≥ 50 | ~89% (10,680 frames) | ~15% (8 muertes) | Bajo riesgo |

**Conclusión:** `dist_obstacle` es altamente predictivo. Un umbral de ~20 px separa claramente "riesgo crítico" de "riesgo bajo". Un árbol de decisión puede aprender fácilmente "si dist_obstacle < 20 y jump=0 → morir".

### ¿La distribución de `score` sugiere regresión simple?

**No.** La distribución de `score` tiene:

- **Media = 28, Mediana = 18, Máx = 120**
- Diferencia media-mediana = 10 px, lo que indica **cola derecha significativa**
- **Coeficiente de asimetría estimado:** (3*(28-18)/120) ≈ 0.25, pero con max=120 y median=18, el coeficiente real es ~1.5 (asimetría moderada a alta)

**Implicaciones:**

| Aspecto | Análisis |
|---------|----------|
| **Normalidad** | No es normal. Cola derecha. No sirve regresión lineal simple sin transformación. |
| **Transformación** | Log(score) o raíz cuadrada(score) reducirían la asimetría. |
| **Modelo alternativo** | Árbol regresor (no asume normalidad). O regresión lineal con transformación logarítmica de Y. |
| **Alternativa robusta** | Gamma regression (para datos positivos asimétricos) o quantile regression. |

---

## 5. Elección de modelo (Misiones 5–6)

### Tabla guía del árbol de decisión "dataset → modelo"

| Si tu EDA encuentra… | Tipo de problema | Modelos razonables | Modelos poco razonables (y por qué) |
|---------------------|------------------|--------------------|------------------------------------|
| Y binaria, tabular, N mediano (12K frames) | Clasificación | Regresión logística, árbol, Random Forest | Red profunda sin más datos (overfitting) |
| Y binaria muy desbalanceada (0.4% positivos) | Clasificación | Mismo + class_weight='balanced', threshold tuning | Accuracy como única métrica |
| Y numérica (score final, cola derecha) | Regresión | Regresión lineal transformada (log Y), árbol regresor | Clasificador binario (ignora la magnitud del score) |
| Y categórica multiclase (4 tipos de obstáculo) | Clasificación multiclase | Logística multinomial con regularización, Random Forest multiclase | Regresión lineal sobre códigos 1,2,3,4 (asume orden falso) |
| Secuencia de frames por sesión | Serie/Secuencia | Features + clasificador (baseline); LSTM/GRU si >200 sesiones | Ignorar orden temporal (los frames están serialmente correlacionados) |
| Relación lineal clara (speed vs time_ms) | Interpretable | Regresión lineal simple | Ensemble opaco (Random Forest) sin necesidad de no-linealidad |
| Muchas variables categóricas (obstacle_type, session_id) | Tabular | Árboles, one-hot encoding | Distancia euclidiana cruda en texto (no es métrica válida para categóricas) |

### Tabla escenario → modelo (P1, P2, P3)

| Escenario | Fila de la guía que aplica | Modelo propuesto | 2 condiciones del dataset que deben cumplirse |
|-----------|----------------------------|------------------|------------------------------------------------|
| **P1** (muerte en siguiente frame) | Y binaria, tabular, N mediano + desbalanceada | **Random Forest con class_weight='balanced'** + threshold tuning | 1. División por sesión (no por frame) para evitar leakage temporal. 2. features `dist_obstacle` y `jump` disponibles con granularidad de 16 ms. |
| **P2** (score final) | Y numérica, cola derecha | **Árbol regresor** (o regresión lineal con log(score) como Y) | 1. Dataset con al menos 200 partidas completas. 2. features de agregación por partida (avg_speed, num_jumps, etc.). |
| **P3** (tipo de obstáculo próximo) | Y categórica multiclase | **Random Forest multiclase** con one-hot encoding de `obstacle_type` | 1. Dataset con al menos 50 ejemplos por tipo de obstáculo. 2. features con `time_ms` o `distance_traveled` para detectar patrones de aparición. |

### Misión 6: Contraejemplos

#### Escenario donde un árbol profundo parecería buena idea pero el EDA lo desaconseja

**Escenario:** P1 (predecir muerte en siguiente frame) con un dataset de solo 500 frames de 5 partidas.

**Por qué se tentaría:** Un árbol profundo puede "memorizar" patrones como "si speed=6.8, dist_obstacle=12, jump=0 → muerte". Parece perfecto para capturar interacciones complejas.

**Por qué el EDA lo desaconseja:**
- **N=500 es muy pequeño** para un árbol profundo. Un árbol con depth=10 puede tener hasta 1024 hojas, más nodos que observaciones → **memoriza, no generaliza**.
- Solo 3-5 casos positivos (muertes) → imposible aprender patrones estadísticamente significativos.
- Los 5 partidas comparten pocos patrones de obstáculos → alto riesgo de overfit a los obstáculos específicos que aparecieron.

**Alternativa del EDA:** Usar **regresión logística simple** (menos parámetros, más robusta) o **no modelar hasta tener más datos**. O usar un árbol **max_depth=3** con poda.

---

#### Escenario donde una red neuronal tendría sentido

**Escenario:** P1 (predecir muerte en siguiente frame) con un dataset de **1,000,000+ frames** de **2,000+ partidas**, con imágenes de pantaneo de alta resolución.

**Qué justificaría el EDA:**
- **N muy grande (1M+):** suficiente para una red neuronal sin sobreajuste.
- **Datos de alta dimensionalidad:** imágenes de 80×80 px (6,400 features) + features tabulares. Una red densa o CNN puede extraer patrones espaciales (posición del obstáculo, altura del dino) que un árbol tabular no capta.
- **No linealidad compleja:** interacciones entre speed, dist_obstacle, altura del dino, y tipo de obstáculo crean regiones de decisión no lineales.
- **Secuencia temporal:** si se usan LSTM/GRU sobre ventanas de 10 frames, la red puede aprender el patrón de "caída anticipada" antes de la colisión.

**Condiciones que el EDA debe cumplir:**
- Confirmar que los frames de la misma partida están divididos por sesión (no por fila).
- Verificar que `dist_obstacle` sigue siendo el predictor dominante (para validar que la red aprende lo mismo).
- Confirmar ausencia de leakage: el modelo no debe tener acceso a `died` del frame actual como feature.

---

#### ¿Se podría resolver P1 con reglas fijas? ¿Ventajas y límites?

**Regla fija propuesta:**
```
SI dist_obstacle < 20 AND jump = 0 THEN will_die_next_frame = 1
SI dist_obstacle < 20 AND jump = 1 THEN will_die_next_frame = 0 (probablemente salvas)
SINO will_die_next_frame = 0
```

**Ventajas:**

| Ventaja | Detalle |
|---------|---------|
| **Interpretabilidad total** | Cualquier desarrollador puede leer y validar la regla. |
| **Computación mínima** | No necesita entrenamiento, inferencia en microsegundos. |
| **No depende de datos** | Funciona aunque el dataset sea pequeño o no exista. |
| **Control de falsos positivos** | Se puede ajustar el umbral (20 px) manualmente. |

**Límites y por qué un modelo aprendido es mejor:**

| Límite de regla fija | Cómo lo soluciona un modelo aprendido |
|---------------------|--------------------------------------|
| **No considera `speed`** | A 13.0 speed, `dist_obstacle=25` es inminente. A 6.0 speed, `dist_obstacle=20` es salirable. Un árbol aprende la interacción `speed × dist_obstacle`. |
| **No considera `obstacle_type`** | Un bird a dist=20 es mortal si no sales. Un cactus_small a dist=20 es evitable si estás en el suelo y agachas. Un Random Forest aprende distintos umbrales por tipo. |
| **No considera `dino_height`** | Si el dino está en el aire (post-salto), `dist_obstacle=12, jump=0` no significa muerte. Un modelo aprende que el estado vertical importa. |
| **Umbral fijo no generaliza** | En partidas de alta velocidad, el umbral debe ser 35, no 20. Un modelo aprende el umbral óptimo per segmento. |
| **No captura retrasos de input** | Si el jugador pulsó salto 2 frames antes, el dino aún está en el suelo. Un modelo puede usar `frames_since_jump_input` como feature. |

**Conclusión:** Las reglas fijas son un excelente **baseline** y **sanity check**, pero un modelo aprendido (Random Forest con threshold tuning) capta interacciones no lineales entre 6-7 variables que las reglas simples no modelan. El EDA confirma esto: `dist_obstacle` es potente, pero `speed` y `obstacle_type` modulan su efecto, lo cual requiere un modelo no lineal.

---

## Síntesis

El dataset prioritario es un **dataset por-frame (16 ms) con 50,000+ filas de 200+ partidas**, con features físicas (`dist_obstacle`, `speed`, `dino_height`, `jump`, `crouch`) y variable objetivo `will_die_next_frame` desplazada 1 frame. El EDA revela un desbalance extremo (0.4% positivos), leakage temporal por sesión, y `dist_obstacle` como predictor dominante. **Solo después** de este análisis, se propone un **Random Forest con class_weight='balanced' y división por sesión**, no una red neuronal. El orden correcto es: entender → dividir → predecir.
