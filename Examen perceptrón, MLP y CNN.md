           ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            EXAMEN TEÓRICO — PERCEPTRÓN, MLP Y CNN (GRUPO A)

                            Eduardo Alcaraz
           ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


1 Datos del alumno
══════════════════

  ━━━━━━━━━━━
   Nombre  Néstor Ricardo López Gutiérrez
   Grupo   A 
   Fecha   03-11-2026  
  ━━━━━━━━━━━


2 Pregunta 1 — Idea del perceptrón
══════════════════════════════════

  1. Explica qué recibe un perceptrón, qué hace con los pesos y el bias,
     y qué produce.
  2. ¿Qué quiere decir que dos clases sean *linealmente separables*?
  3. Señala *una* limitación fuerte del perceptrón de una sola capa.


2.0.1 Respuesta
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

  ┌────
   1. Un perceptrón recibe un conjunto de datos o valores numéricos,
   generalmente desde un dataset, luego lo multiplica por los pesos que
   definen sus niveles de importancia, luego se suman todas las entradas
   multiplicadas por sus pesos, añadiendo a veces un sesgo, generando un
   valor binario de decisión.
   2. Dos clases linealmente separables son aquellas que, en un diagrama,
   se pueden separar mediante la función de una recta, sin que sus valores
   se combinen.
   3. No puede resolver problemas con clases que no sean linealmente
   separables.
  └────


3 Pregunta 2 — Por qué aparece el MLP
═════════════════════════════════════

  1. ¿En qué sentido un MLP es “más de una capa de neuronas”?
  2. ¿Para qué sirve la capa oculta ?
  3. Si quitas todas las activaciones no lineales, ¿qué tipo de
     comportamiento te queda y por qué importa?


3.0.1 Respuesta
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

  ┌────
   1. Un MLP es un perceptrón que posee 2 o más capas de procesamiento o capas
   de neuronas, lo que le permite procesar datasets más completos que un
   perceptrón simple.
   2. Para que el perecptrón pueda entender, procesar y aprender de un dataset
   linealmente no separable.
   3. Queda un modelo con regresión lineal simple, puesto que las activaciones
   no lineales son lo que permiten a la recta curvarse y permitir que aprenda
   patrones y relaciones no lineales.
  └────


4 Pregunta 3 — XOR como lección histórica
═════════════════════════════════════════

  1. ¿Qué enseña el ejemplo XOR sobre el perceptrón simple?
  2. ¿Qué cambia al pasar a una red con capas ocultas y no linealidad?
  3. ¿Eso significa que el perceptrón nunca sirve en problemas reales?
     Argumenta.


4.0.1 Respuesta
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

  ┌────
   1. Que un perceptrón simple no puede manejar un conjunto de datos que no
   se puedan separar con una recta en un plano.
   2. El perceptrón pasa a poder curvar su recta, lo que le permite el analizar
   valores que anteriormente le eran imposibles.
   3. Falso, el perceptrón simple sigue siendo útil, tanto por su bajo consumo
   como por su velocidad de procesamiento comparado con un MLP, puesto que al
   alterar o filtrar un dataset a los valores deseados se puede pasar a un
   dataset linealmente separable.
  └────


5 Pregunta 4 — Cómo “aprende” la red
════════════════════════════════════

  1. Describe con palabras el ciclo: predicción → medir error → ajustar
     pesos.
  2. ¿Qué controla el *learning rate* (tasa de aprendizaje) a nivel
     conceptual?
  3. Diferencia *subajuste* y *sobreajuste* con un ejemplo escolar
     breve.


5.0.1 Respuesta
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

  ┌────
   1. El ciclo de aprendizaje es la herramienta principal usada por los sistemas
   de aprendizaje automático (Machine Learning), basado en el análisis de datos
   con pesos iniciales, medición del error entre el resultado del análisis y el
   valor real esperado, y la propagación y ajuste de los pesos iniciales antes de
   iniciar un nuevo ciclo.
   2. La tasa de aprendizaje controla la velocidad y el tamaño de pasos que se dan
   en cada iteración del ciclo de aprendizaje.
   3. Un subajuste representa un bajo nivel de aprendizaje, como cuando un alumno
   estudia 5 minutos antes de un exámen, mientras que un sobreajuste representa un
   alto nivel de aprendizaje, lo que puede llevar a fallos cuando el dataset de
   pruebas cambia, como un alumno que estudió todos los examenes anteriores de una
   materia, pero el real cambia completamente el enfoque, provocando un muy posible
   reprobamiento.
  └────


6 Pregunta 5 — Salidas: sigmoide y softmax
══════════════════════════════════════════

  1. ¿Cuándo usarías sigmoide y cuándo softmax?
  2. ¿Qué significa el vector que sale de un softmax en un problema de
     varias clases?
  3. Pon un ejemplo del curso (juego, riesgo, imágenes, etc.) y di qué
     salida encajaría.


6.0.1 Respuesta
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

  ┌────
   1. Una salida sigmoide se usa cuando se espera un resultado binario, mientras
   que una salida softmax se usa para cuando se tienen dos o más clases excluyentes
   entre si.
   2. El vector significa la probabilidad de cada clase de ser la correcta.
   3. 
  └────


7 Pregunta 6 — Motivación de la CNN
═══════════════════════════════════

  1. ¿Qué inconveniente conceptual tiene aplastar una imagen y meterla a
     un MLP denso?
  2. ¿Qué aporta la idea de *filtro local* y *pesos compartidos*?
  3. Menciona *dos* dominios (además de fotos cotidianas) donde una CNN
     tiene sentido.


7.0.1 Respuesta
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

  ┌────
   1. Al aplastar una imagen existe la probabilidad de perder información
   o datos importantes.
   2. 
  └────


8 Pregunta 7 — Piezas de una CNN
════════════════════════════════

  Explica el *para qué* de:
  1. Convolución
  2. Pooling
  3. Flatten + Dense al final
  4. Dropout durante el entrenamiento


8.0.1 Respuesta
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

  ┌────
   1. Sirve para el análisis inicial de una imagen al aplicar filtros que
   destacan patrones básicos de lo que se busca.
   2. Sirve para el reducir el tamaño o dimensiones de la imagen resultante
   del paso anterior, destacando aún más los patrones a buscar para dar estabilidad
   a la imagen. 
   3. Sirve para tomar la decisión final basado en los datos extraídos del paso anteior.
   4. Sirve para evitar que el modelo sobreaprenda y genere problemas en el análisis
   de los datos.
  └────


9 Pregunta 8 — Capas bajas vs capas altas
═════════════════════════════════════════

  1. ¿Qué suelen captar las primeras capas de una CNN frente a las más
     profundas?
  2. Usa una analogía (lupas, detectores, “de simple a complejo”).
  3. ¿Por qué “poner más capas” no garantiza mejor modelo?


9.0.1 Respuesta
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

  ┌────
   1. Las capas iniciales captan patrones básicos dentro de una imagen, mientras que
   las capas finales buscan patrones complejos o específicos dentro de los datos
   recabados por las capas iniciales.
   2. Al analizar una imágen se puede iniciar buscando patrones o colores básicos, pero
   conforme se vaya avanzando se puede usar una lupa para ver buscar detalles específicos
   que permitan el concluir qué se está viendo.
   3. Por que el poner más capas no simboliza que el modelo aprenda mejor, pudiendo llegar
   a un sobreajuste o subajuste dependiendo de como se planteé el sistema de aprendizaje
   del modelo. 
  └────


10 Pregunta 9 — Análisis: elige arquitectura
════════════════════════════════════════════

  Elige *Perceptrón (o lineal simple)* / *MLP* / *CNN* y justifica en
  *una* frase:

  1. Tabla de alumnos: promedio, faltas, horas de estudio → ¿aprueba?
  2. Clasificar el color de un semáforo en una fotografía.
  3. Clases en el plano que *no* se separan con una recta.
  4. ¿Usarías CNN para autocompletar código o una secuencia larga de
     texto? ¿Sí/no y qué alternativa del curso mencionarías?


10.0.1 Respuesta
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

  ┌────
   1. Se usaría un MLP puesto que los datos no son linealmente separables.
   2. Se usaría un CNN puesto que es el modelo que procesa imágenes con mayor
   eficiencia.
   3. Se usaría un MLP puesto que fue diseñado para analizar datos linealmente
   no separables.
   4. No, un CNN tiene un mejor análisis de datos por pixel que de una cadena
   de texto.
  └────


11 Pregunta 10 — Análisis: mirada de proyecto
═════════════════════════════════════════════

  1. Train accuracy altísima y validation baja: ¿qué sospechas? ¿Qué dos
     preguntas le harías al equipo en la defensa?
  2. ¿Por qué más neuronas o más capas no sustituyen un buen
     planteamiento del problema y buenos datos?
  3. Escribe *tres* preguntas orales (tuyas) para verificar que
     distinguen perceptrón, MLP y CNN.


11.0.1 Respuesta
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

  ┌────
   1. Que el modelo sufre de un sobreajuste que le impide validar de manera
   correcta los datos.
   a) ¿Qué metódos de regularización se usaron en el proceso de aprendizaje?
   b) ¿Cuál es el ratio de distribución de sus datos?
   2. Por que no importa la cantidad de neuronas o capas que se agreguen si
   no se asegura que analicen de manera correcta los datos que se les entrega.
   3.
   a) ¿Si se tiene un conjunto de datos que pueden separarse linealmente, que
   modelo es el más óptimo a usar?
   b) ¿Por qué el usar una red convolucional es más óptimo que aplanar una
   imagen para que sea procesada por una red multi capa?
   c) ¿Qué es una capa oculta y cuál es su importancia?
  └────
