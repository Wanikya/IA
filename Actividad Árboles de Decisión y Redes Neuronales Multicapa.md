# Actividad: Árboles de Decisión y Redes Neuronales Multicapa

---

## Parte I: Conceptos y Definiciones

---

### Pregunta 1
> **¿Qué es un árbol de decisión y cuál es su objetivo principal dentro de un problema de clasificación?**

Un **árbol de decisión** es un modelo predictivo de aprendizaje supervisado que utiliza una estructura de árbol invertido para la toma de decisiones mediante nodos.

**Objetivo principal:**
Aprender reglas de decisión simples para dividir de manera recursiva un conjunto de datos heterogéneo en subconjuntos cada vez más homogéneos y puros.

---

### Pregunta 2
> **Explique con sus propias palabras los siguientes elementos de un árbol de decisión:**

* **Nodo raíz:** Punto inicial donde se ubica la totalidad del conjunto inicial de datos.
* **Nodo interno:** Nodo intermedio que recibe un subconjunto de datos de nodos superiores y aplica reglas de decisión específicas.
* **Rama:** Conexión entre nodos que representa el resultado lógico de una decisión tomada en un nodo superior.
* **Hoja:** Nodo terminal que no se puede dividir más, representando la predicción final o la clase asignada.

---

### Pregunta 3
> **¿Qué es una red neuronal multicapa y qué función cumplen sus capas?**

Una **Red Neuronal Multicapa** (también llamada *Perceptrón Multicapa* o **MLP**, por sus siglas en inglés) es un modelo predictivo compuesto por un conjunto de capas de neuronas artificiales (nodos) interconectadas que transforman la información mediante el uso de operaciones matemáticas no lineales.

* **Capa de entrada:** Recibe los datos crudos y los distribuye hacia las neuronas de la siguiente capa sin realizar transformaciones complejas.
* **Capa oculta:** Conjunto de capas intermedias encargadas del procesamiento mediante transformaciones lineales (pesos y sesgos) seguidas de funciones de activación no lineales.
* **Capa de salida:** Entrega el resultado o predicción final generado a partir de las representaciones procesadas por las capas ocultas.

---

### Pregunta 4
> **¿Qué representan los pesos y los sesgos dentro de una red neuronal? Explique también por qué sus valores cambian durante el entrenamiento.**

* **El peso ($w$):** Valor matemático asignado a la conexión entre neuronas. Representa la fuerza de la conexión y el nivel de influencia que tendrá la neurona de entrada sobre la neurona receptora.
* **El sesgo ($b$):** Término constante añadido a la suma ponderada antes de la activación. Permite desplazar la función de activación horizontalmente, otorgando flexibilidad a la neurona para activarse incluso cuando las entradas son cero.

**¿Por qué cambian durante el entrenamiento?**  
Los pesos y sesgos se inicializan con valores aleatorios. Tras cada iteración:
1. Se calcula la diferencia entre la salida predicha y el valor real mediante una **función de pérdida**.
2. El algoritmo de **retropropagación** (*backpropagation*) utiliza la regla de la cadena para calcular la contribución exacta de cada peso y sesgo a ese error.
3. Un algoritmo de optimización (como el descenso del gradiente) ajusta progresivamente los valores de $w$ y $b$ para **minimizar la pérdida global**.

---

### Pregunta 5
> **¿Cuál es la principal diferencia entre la forma en que aprende un árbol de decisión y una red neuronal multicapa? Explique qué elementos aprende cada modelo.**

| Criterio | Árbol de Decisión | Red Neuronal Multicapa (MLP) |
| :--- | :--- | :--- |
| **Mecanismo de Aprendizaje** | **Discreto y Combinatorio (*Greedy*):** Evalúa estadísticamente cada variable y selecciona divisiones ortogonales óptimas mediante métricas de pureza (Entropía, Ganancia de Información o Índice Gini). | **Continuo y Diferenciable:** Utiliza cálculo diferencial (propagación del error y descenso del gradiente) para ajustar iterativamente parámetros continuos. |
| **Elementos que Aprende** | Una **estructura explícita de reglas condicionales** (*If / Then*) organizadas jerárquicamente. | Una **matriz distribuida de pesos y sesgos** que define hiper-superficies continuas de decisión. |

---

## Parte II: Análisis y Aplicación

---

### Pregunta 6
> **Una institución bancaria desea desarrollar un sistema que detecte posibles compras fraudulentas considerando:**
> * Monto de la compra, Hora de la operación, Ciudad, Tipo de establecimiento, Compras del día e Historial del cliente.
> 
> **Analice las ventajas y desventajas de ambos modelos. ¿Cuál utilizaría y por qué?**

#### Análisis Comparativo

* **Árbol de Decisión:**
  * **Ventajas:** Alta interpretabilidad (permite auditar por qué la transacción de un cliente fue bloqueada) y rapidez en entrenamiento.
  * **Desventajas:** Rigidez ante límites continuos (monto, hora), baja capacidad para detectar combinaciones no lineales complejas entre variables y propensión al sobreajuste si el dataset de fraude está muy desbalanceado.

* **Red Neuronal Multicapa (MLP):**
  * **Ventajas:** Excelente capacidad para detectar correlaciones sutiles, no lineales y multidimensionales entre historial, ubicación, horario y montos atípicos.
  * **Desventajas:** Poca transparencia ("caja negra"), requiere normalización previa estricta y mayor volumen de transacciones históricas.

#### Selección de Modelo
* **Modelo elegido:** **Red Neuronal Multicapa (MLP)**.
* **Justificación:** Los patrones de fraude financiero evolucionan constantemente y suelen involucrar interacciones no lineales sutiles entre múltiples variables (por ejemplo, un monto moderado a una hora inusual en una ciudad distinta a la habitual). La MLP es muy superior identificando este tipo de anomalías complejas. Para cumplir con las regulaciones bancarias de auditoría, se pueden emplear algoritmos de explicabilidad *post-hoc* (como SHAP o LIME).

---

### Pregunta 7
> **Una escuela quiere detectar estudiantes en riesgo de reprobar considerando:**
> * Asistencia, Calificaciones, Tareas entregadas, Participación y Materias reprobadas anteriormente.
> 
> **Si ambos modelos obtienen prácticamente la misma precisión, ¿qué otros factores tomaría en cuenta para elegir uno de los dos modelos?**

#### Factores Determinantes
1. **Interpretabilidad y Explicabilidad Pedagógica:** La capacidad de explicar al docente, al estudiante y al tutor cuáles son las causas exactas del riesgo de reprobación.
2. **Acción Operativa:** Identificar qué variables modificables (asistencia, tareas entregadas) se pueden intervenir pedagógicamente.
3. **Simplicidad de Despliegue y Mantenimiento:** Facilidad de integrar el modelo en el sistema de gestión escolar existente con mínimos recursos computacionales.

> **Conclusión:** **Elegir Árbol de Decisión.**  
> En educación, explicar *por qué* un alumno está en riesgo es tan importante como detectarlo, pues esto permite diseñar un plan de regularización y tutoría específico para el estudiante.

---

### Pregunta 8
> **Un hospital desarrolla un sistema para determinar qué pacientes necesitan atención prioritaria (triaje) utilizando datos clínicos.**
> 
> **Una red neuronal obtiene mejores resultados que un árbol de decisión, pero resulta difícil explicar su respuesta. ¿Considera que la mayor precisión es suficiente para elegir la red neuronal? Analice las consecuencias.**

#### Análisis de Consecuencias
1. **Falsos Negativos y Riesgo de Vida:** Si la red neuronal comete un error clasificando a un paciente en estado crítico como "Prioridad Baja", los médicos no podrán auditar la lógica del modelo para detectar la falla a tiempo, lo que puede resultar letal.
2. **Responsabilidad Ética y Legal:** En caso de negligencia o fallo en el triaje, los profesionales de la salud deben responder legalmente por las decisiones clínicas tomadas. No se puede justificar una decisión médica inapropiada basándose en un algoritmo de "caja negra".
3. **Resistencia de la Comunidad Médica:** Los médicos tienden a rechazar herramientas cuyos criterios no pueden validar clínicamente con sus conocimientos en fisiología y sintomatología.

> **Conclusión:**  
> En sistemas médicos críticos, la precisión sola **no es suficiente**. Se debe optar por modelos interpretables o exigir el uso de sistemas híbridos donde un panel médico valide las predicciones antes de ejecutar la atención.

---

### Pregunta 9
> **Una empresa de reparto quiere predecir si un pedido llegará tarde. Ante un caso donde el Árbol indica "Llegará a tiempo" y la Red Neuronal indica "Probablemente llegará tarde":**
> 
> **¿Cómo determinaría cuál realiza una mejor predicción? Explique qué información adicional debería analizar.**

#### Criterios de Evaluación del Desempeño
* **Evaluación de Métricas en el Conjunto de Prueba (*Test Set*):** Comparar el desempeño histórico de ambos modelos específicamente en la métrica más crítica para la empresa (por ejemplo, el *Recall* o el *F1-Score* en la clase "Retrasado").
* **Nivel de Incertidumbre Probabilística:** Analizar la salida probabilística de la red neuronal. Si asigna un $51\%$ de probabilidad a "llegará tarde", la predicción tiene alta incertidumbre; si asigna un $98\%$, la predicción es altamente confiable.
* **Análisis de Atribución de Características (*Feature Attribution*):** Utilizar herramientas como SHAP para desglosar la contribución exacta de cada variable en la decisión de ambos modelos.

#### Información Adicional a Analizar
* **Datos Históricos en Condiciones Similares:** Buscar casos pasados con valores equivalentes de tráfico, clima y experiencia del repartidor para verificar qué sucedió realmente.
* **Análisis del Entorno en Tiempo Real:** Verificar si existen variables externas no contempladas correctamente por el árbol (por ejemplo, si el clima actual está en un valor límite donde el árbol aplica un corte rígido e incorrecto).

---

### Pregunta 10
> **Una empresa analiza dos sistemas para aprobación de crédito: un Árbol de Decisión (explicable) vs. una Red Neuronal (más precisa pero "caja negra").**
> 
> **Analice cuál utilizaría, sus ventajas, riesgos y si consideraría posible utilizar ambos dentro del mismo sistema.**

#### Decisión
Implementar un **Sistema Híbrido** con el **Árbol de Decisión** (o ensambles interpretables) como modelo principal y la **Red Neuronal** como modelo *challenger* y validador de patrones.

#### Matriz de Justificación

| Aspecto | Justificación / Estrategia |
| :--- | :--- |
| **Modelo Principal (Árbol / Reglas)** | • **Regulatorio:** Leyes de crédito justo (ECOA, GDPR Art. 22) exigen el derecho a una explicación explícita (*"¿Por qué se denegó el crédito?"*).<br>• **Auditoría:** Permite detectar y prevenir sesgos discriminatorios por raza, género o edad.<br>• **Confianza:** Reglas transparentes para el cliente y los auditores. |
| **Ventajas** | Cumplimiento legal, ética, fácil depuración (*debuggable*) y accionable (indica al cliente qué mejorar para calificar). |
| **Riesgos** | Menor precisión potencial $\rightarrow$ mayor tasa de falsos negativos (buenos clientes rechazados) o falsos positivos (riesgo de impago). |
| **Rol de la Red Neuronal** | • **Modelo Challenger:** Se entrena en paralelo; si supera consistentemente al árbol y aprueba pruebas de sesgo, se evalúa su integración.<br>• **Generador de Features:** Descubre interacciones complejas que luego se destilan como nuevas reglas explicables hacia el árbol (*Knowledge Distillation*).<br>• **Explicabilidad Post-Hoc:** Vía valores SHAP para validar o retar las decisiones del árbol. |

#### ¿Ambos modelos en el mismo sistema? **¡Sí!**
Mediante una arquitectura **Human-in-the-Loop + Model Cards**:
1. El **Árbol de Decisión** toma la decisión principal y emite la razón lógica.
2. La **Red Neuronal** genera un puntaje de riesgo (*scoring*) en segundo plano.
3. Si ambos modelos discrepan significativamente, la solicitud se envía a **revisión humana**.
4. Se mantienen registros completos para auditorías y reentrenamiento periódico.

---

### Síntesis Final

> **"No existe un algoritmo de Inteligencia Artificial que sea el mejor para todos los problemas."**

Esta afirmación refleja formalmente el **Teorema No Free Lunch** (Wolpert & Macready): promediado sobre el conjunto de todos los problemas posibles, todos los algoritmos tienen el mismo rendimiento promedio.

La elección del modelo óptimo depende de la interacción entre los siguientes factores:

* **Precisión vs. Interpretabilidad:** Existe un compromiso (*trade-off*) estructural. Los modelos con mayor capacidad predictiva para patrones complejos (redes neuronales) actúan como "cajas negras", mientras que los modelos altamente interpretables (árboles de decisión) presentan límites en problemas no lineales.
* **Cantidad de Datos:** Las redes neuronales requieren grandes volúmenes de datos etiquetados para entrenar sus parámetros. Con conjuntos de datos pequeños o tabulares, los árboles de decisión o sus ensambles suelen ofrecer un rendimiento superior.
* **Complejidad del Problema:** Los datos tabulares estructurados con relaciones condicionales son ideales para árboles de decisión. Los datos no estructurados (imágenes, audio, texto) o con interacciones multidimensionales exigen la capacidad de abstracción de las redes profundas.
* **Consecuencias de una Decisión Incorrecta:** En entornos de alto riesgo (medicina, finanzas reguladas, aviación), la interpretabilidad y el control del error son prioritarios sobre pequeños incrementos de precisión. En entornos de bajo riesgo (sistemas de recomendación), la precisión pura es el objetivo principal.