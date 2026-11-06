# 📝 Examen Teórico — Perceptrón, MLP y CNN

**Profesor:** Eduardo Alcaraz | **Grupo:** A | **Fecha:** 03-11-2026  
**Alumno:** Néstor Ricardo López Gutiérrez

---

## 📋 1. Datos del Alumno

| Campo | Información |
| :--- | :--- |
| **Nombre** | Néstor Ricardo López Gutiérrez |
| **Grupo** | A |
| **Fecha** | 03-11-2026 |

---

## ❓ Pregunta 1 — Idea del Perceptrón

1. Explica qué recibe un perceptrón, qué hace con los pesos y el bias, y qué produce.
2. ¿Qué quiere decir que dos clases sean *linealmente separables*?
3. Señala *una* limitación fuerte del perceptrón de una sola capa.

> ✏️ **Respuesta:**
> 1. Un perceptrón recibe un conjunto de datos o valores numéricos (entradas), los multiplica por sus respectivos pesos (que definen el nivel de importancia de cada característica), suma todos los productos junto con un sesgo (*bias*) y finalmente aplica una función para generar una salida binaria de decisión.
> 2. Dos clases son linealmente separables si en un plano o espacio vectorial se pueden dividir completamente mediante una recta (o hiperplano), sin que sus valores se solapen o mezclen.
> 3. No puede resolver problemas con clases que no sean linealmente separables (como la función lógica XOR).

---

## ❓ Pregunta 2 — Por qué aparece el MLP

1. ¿En qué sentido un MLP es “más de una capa de neuronas”?
2. ¿Para qué sirve la capa oculta?
3. Si quitas todas las activaciones no lineales, ¿qué tipo de comportamiento te queda y por qué importa?

> ✏️ **Respuesta:**
> 1. Un MLP (Perceptrón Multicapa) posee dos o más capas de procesamiento (capa de entrada, una o más capas ocultas y capa de salida), permitiéndole procesar datos más complejos que un perceptrón simple.
> 2. La capa oculta permite transformar y extraer características intermedias del dataset para aprender patrones no lineales complejos.
> 3. Queda un modelo equivalente a una combinación lineal (regresión o clasificación lineal simple), ya que la suma de funciones lineales sigue siendo lineal; las activaciones no lineales son las que permiten "curvar" las fronteras de decisión para aprender relaciones complejas.

---

## ❓ Pregunta 3 — XOR como lección histórica

1. ¿Qué enseña el ejemplo XOR sobre el perceptrón simple?
2. ¿Qué cambia al pasar a una red con capas ocultas y no linealidad?
3. ¿Eso significa que el perceptrón nunca sirve en problemas reales? Argumenta.

> ✏️ **Respuesta:**
> 1. Enseña que el perceptrón simple es incapaz de clasificar o resolver un conjunto de datos que no se pueda separar con una sola línea recta en un plano.
> 2. Al agregar capas ocultas y funciones de activación no lineales, la red puede generar fronteras de decisión curvas y complejas para separar clases entrelazadas.
> 3. **Falso.** El perceptrón simple sigue siendo útil por su bajo costo computacional y alta velocidad de procesamiento en problemas donde las características son o pueden transformarse en linealmente separables mediante preprocesamiento.

---

## ❓ Pregunta 4 — Cómo “aprende” la red

1. Describe con palabras el ciclo: predicción $\rightarrow$ medir error $\rightarrow$ ajustar pesos.
2. ¿Qué controla el *learning rate* (tasa de aprendizaje) a nivel conceptual?
3. Diferencia *subajuste* y *sobreajuste* con un ejemplo escolar breve.

> ✏️ **Respuesta:**
> 1. **Ciclo de aprendizaje:** La red realiza una **predicción** inicial con sus pesos actuales; luego **mide el error** comparando la predicción con el valor real esperado; finalmente, mediante **propagación hacia atrás (*backpropagation*)**, calcula los ajustes necesarios y **actualiza los pesos** antes de la siguiente iteración.
> 2. El *learning rate* controla la velocidad y la magnitud de los cambios (el tamaño del paso) que se aplican a los pesos en cada actualización.
> 3. 
>    * **Subajuste (*Underfitting*):** Ocurre cuando el modelo aprende tan poco que no detecta los patrones principales. *Ejemplo:* Un estudiante que solo repasa 5 minutos antes del examen y reprueba por falta de conocimiento básico.
>    * **Sobreajuste (*Overfitting*):** Ocurre cuando el modelo memoriza el conjunto de entrenamiento en exceso pero falla ante datos nuevos. *Ejemplo:* Un estudiante que memorizó de memoria las respuestas de exámenes pasados, pero al cambiarle las preguntas en el examen real reprueba.

---

## ❓ Pregunta 5 — Salidas: sigmoide y softmax

1. ¿Cuándo usarías sigmoide y cuándo softmax?
2. ¿Qué significa el vector que sale de un softmax en un problema de varias clases?
3. Pon un ejemplo del curso (juego, riesgo, imágenes, etc.) y di qué salida encajaría.

> ✏️ **Respuesta:**
> 1. **Sigmoide:** Se usa para clasificación binaria (resultado continuo entre 0 y 1) o clasificaciones multietiqueta independientes.  
>    **Softmax:** Se usa para clasificación multiclase excluyente, asegurando que la suma de todas las probabilidades de salida sea igual a 1.
> 2. Representa una distribución de probabilidad, donde cada valor numérico indica la probabilidad (entre 0% y 100%) de que la muestra pertenezca a esa clase específica.
> 3. *Ejemplo:* En el juego de esquivar objetos (donde el personaje debe elegir una entre 9 acciones posibles), se usa **Softmax** para obtener la probabilidad de cada dirección. En cambio, para decidir si el jugador colisionó o sigue vivo (resultado binario), encajaría una **Sigmoide**.

---

## ❓ Pregunta 6 — Motivación de la CNN

1. ¿Qué inconveniente conceptual tiene aplastar una imagen y meterla a un MLP denso?
2. ¿Qué aporta la idea de *filtro local* y *pesos compartidos*?
3. Menciona *dos* dominios (además de fotos cotidianas) donde una CNN tiene sentido.

> ✏️ **Respuesta:**
> 1. Al aplanar una imagen 2D a un vector 1D se destruye la relación espacial entre píxeles vecinos y se genera una cantidad masiva de parámetros, aumentando drásticamente el riesgo de sobreajuste.
> 2. **Filtro local:** Permite a la red enfocar su atención en pequeñas regiones vecinas de la imagen para detectar patrones básicos (bordes, esquinas, texturas).  
>    **Pesos compartidos:** Permite reutilizar el mismo filtro a lo largo de toda la imagen, reduciendo drásticamente la cantidad de parámetros necesarios.
> 3. *Ejemplos:* 
>    * **Imágenes médicas:** Detección de lesiones en radiografías, tomografías o resonancias magnéticas.
>    * **Procesamiento de audio:** Clasificación de espectrogramas (imágenes de frecuencias) para reconocimiento de voz o diagnóstico de sonidos de maquinaria.

---

## ❓ Pregunta 7 — Piezas de una CNN

Explica el *para qué* de:
1. Convolución
2. Pooling
3. Flatten + Dense al final
4. Dropout durante el entrenamiento

> ✏️ **Respuesta:**
> 1. **Convolución:** Extrae mapas de características de la imagen mediante la aplicación de filtros que resaltan patrones visuales (líneas, bordes horizontales/verticales, texturas).
> 2. **Pooling:** Reduce las dimensiones espaciales (alto y ancho) de los mapas de características para reducir el costo computacional y dar cierta invariancia a traslaciones o pequeñas deformaciones.
> 3. **Flatten + Dense:** *Flatten* convierte los mapas de características multidimensionales en un vector 1D, y las capas *Dense* (totalmente conectadas) toman esas características procesadas para realizar la clasificación final.
> 4. **Dropout:** Desactiva aleatoriamente un porcentaje de neuronas durante el entrenamiento para evitar que dependan excesivamente unas de otras, previniendo el sobreajuste.

---

## ❓ Pregunta 8 — Capas bajas vs capas altas

1. ¿Qué suelen captar las primeras capas de una CNN frente a las más profundas?
2. Usa una analogía (lupas, detectores, “de simple a complejo”).
3. ¿Por qué “poner más capas” no garantiza mejor modelo?

> ✏️ **Respuesta:**
> 1. Las capas iniciales (bajas) captan patrones simples y generales (bordes, curvas, cambios de color). Las capas profundas (altas) combinan esos patrones básicos para reconocer formas complejas y específicas (ojos, ruedas, rostros completos).
> 2. *Analogía:* Las primeras capas son como usar una lupa para identificar trazos y líneas individuales en un lienzo; las capas intermedias identifican si esos trazos forman un círculo; y las capas finales ensamblan todos los círculos y líneas para distinguir si la pintura es una bicicleta o un automóvil.
> 3. Agregar capas indiscriminadamente no garantiza mejoras porque puede causar problemas como el desvanecimiento del gradiente (*vanishing gradient*), sobreajuste severo o un costo computacional innecesario sin aportar información relevante.

---

## ❓ Pregunta 9 — Análisis: elige arquitectura

Elige *Perceptrón (o lineal simple)* / *MLP* / *CNN* y justifica en *una* frase:

1. Tabla de alumnos: promedio, faltas, horas de estudio $\rightarrow$ ¿aprueba?
2. Clasificar el color de un semáforo en una fotografía.
3. Clases en el plano que *no* se separan con una recta.
4. ¿Usarías CNN para autocompletar código o una secuencia larga de texto? ¿Sí/no y qué alternativa del curso mencionarías?

> ✏️ **Respuesta:**
> 1. **MLP:** Es la opción adecuada porque maneja datos tabulares numéricos y aprende relaciones no lineales complejas entre el promedio, las faltas y las horas de estudio.
> 2. **CNN:** Es el modelo ideal ya que preserva la estructura bidimensional y las características espaciales del semáforo dentro de la imagen.
> 3. **MLP:** Está diseñado específicamente para construir fronteras de decisión curvas y no lineales en el plano.
> 4. **No**, porque las CNN están diseñadas para datos espaciales 2D/3D (imágenes); para secuencias temporales o cadenas de texto se deben utilizar **RNN (Redes Neuronales Recurrentes)** o **Transformers**.

---

## ❓ Pregunta 10 — Análisis: mirada de proyecto

1. Train accuracy altísima y validation baja: ¿qué sospechas? ¿Qué dos preguntas le harías al equipo en la defensa?
2. ¿Por qué más neuronas o más capas no sustituyen un buen planteamiento del problema y buenos datos?
3. Escribe *tres* preguntas orales (tuyas) para verificar que distinguen perceptrón, MLP y CNN.

> ✏️ **Respuesta:**
> 1. **Diagnóstico:** El modelo sufre de un claro **sobreajuste (*overfitting*)**.  
>    *Preguntas para el equipo:*  
>    * a) ¿Qué técnicas de regularización (como Dropout, weight decay o Data Augmentation) emplearon durante el entrenamiento?  
>    * b) ¿Cómo realizaron la división, muestreo y balanceo entre los conjuntos de entrenamiento, validación y prueba?
> 2. Porque si los datos de entrada son deficientes, ruidosos o están mal etiquetados, aumentar la cantidad de neuronas solo provocará que el modelo memorice el ruido con mayor precisión, sin resolver el problema real.
> 3. *Preguntas orales propuestas:*  
>    * a) ¿Bajo qué condición específica un problema requiere pasar de un Perceptrón simple a un Perceptrón Multicapa (MLP)?  
>    * b) ¿Qué ventaja ofrece aplicar una operación de Convolución frente al aplanamiento directo de una imagen para un MLP?  
>    * c) ¿Qué es una capa oculta y cuál es la función de activar sus salidas con una función no lineal?