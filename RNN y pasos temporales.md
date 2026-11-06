# Explicación Detallada de la Red Neuronal Recurrente (RNN)

## 1. La Ecuación Matemática Fundamental y su Correspondencia

Las ecuaciones fundamentales que definen una RNN son:

$$h_t = f(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$
$$y_t = g(W_{hy} \cdot h_t + b_y)$$

### Mapeo de los Pasos del Ejemplo a la Fórmula

1. **Paso 1: Multiplicación de Entrada ($W_{xh} \cdot x_t$)**
   * Corresponde al primer término dentro de la función de activación $f(\dots)$ en la ecuación del estado oculto.
   * Proyecta el vector de entrada actual $x_t$ al espacio de dimensión oculta mediante la matriz de pesos $W_{xh}$.

2. **Paso 2: Multiplicación de la Memoria ($W_{hh} \cdot h_{t-1}$)**
   * Corresponde al segundo término dentro de $f(\dots)$.
   * Transforma el estado oculto anterior $h_{t-1}$ (la "memoria" pasada del sistema) a través de la matriz de pesos recurrentes $W_{hh}$. En el tiempo inicial ($t=1$), $h_0$ se inicializa con un vector de ceros.

3. **Paso 3: Suma y Sesgo ($W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h$)**
   * Realiza la combinación lineal de la información presente ($W_{xh} \cdot x_t$), la información pasada ($W_{hh} \cdot h_{t-1}$) y el vector de sesgo $b_h$.

4. **Paso 4: Activación ($h_t = f(\dots)$)**
   * Evalúa la función de activación no lineal $f$ (en este caso $\tanh$) sobre la suma obtenida para acotar los valores y generar el nuevo estado oculto $h_t$.

5. **Cálculo de la Salida ($y_t = g(W_{hy} \cdot h_t + b_y)$)**
   * Corresponde a la segunda ecuación fundamental. Proyecta el estado oculto actual $h_t$ al espacio de salida mediante $W_{hy}$, le suma el sesgo $b_y$ y aplica la función de activación $g$ (aquí tomada como función lineal/identidad).

---

## 2. Configuración Inicial y Resumen ($t=0, 1, 2$)

### Parámetros Iniciales
* **Dimensión de entrada:** 2
* **Dimensión oculta:** 3
* **Dimensión de salida:** 1

$$W_{xh} = \begin{bmatrix} 0.1 & 0.3 \\ 0.2 & 0.4 \\ 0.3 & 0.5 \end{bmatrix}, \quad W_{hh} = \begin{bmatrix} 0.6 & 0.1 & 0.2 \\ 0.1 & 0.7 & 0.3 \\ 0.2 & 0.3 & 0.8 \end{bmatrix}, \quad b_h = \begin{bmatrix} 0.1 \\ 0.2 \\ 0.3 \end{bmatrix}$$
$$W_{hy} = \begin{bmatrix} 0.5 & 0.3 & 0.2 \end{bmatrix}, \quad b_y = 0.1$$

* **Inicialización ($t=0$):** $h_0 = \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}$
* **Estado en $t=1$ ($x_1 = [1, 2]^T$):** $h_1 \approx \begin{bmatrix} 0.664 \\ 0.834 \\ 0.922 \end{bmatrix}, \quad y_1 = 0.866$
* **Estado en $t=2$ ($x_2 = [3, 4]^T$):** $h_2 \approx \begin{bmatrix} 0.978 \\ 0.997 \\ 0.999 \end{bmatrix}$

---

## 3. Extensión de Pasos Temporales ($t=3$ y $t=4$)

### Paso 4: Tercer Paso Temporal ($t=3$)

**Entrada:** $x_3 = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$

1. **Término de entrada ($W_{xh} \cdot x_3$):**
   $$\begin{bmatrix} 0.1 & 0.3 \\ 0.2 & 0.4 \\ 0.3 & 0.5 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} 0.1(5) + 0.3(6) \\ 0.2(5) + 0.4(6) \\ 0.3(5) + 0.5(6) \end{bmatrix} = \begin{bmatrix} 2.3 \\ 3.4 \\ 4.5 \end{bmatrix}$$

2. **Término recurrente ($W_{hh} \cdot h_2$):**
   $$\begin{bmatrix} 0.6 & 0.1 & 0.2 \\ 0.1 & 0.7 & 0.3 \\ 0.2 & 0.3 & 0.8 \end{bmatrix} \begin{bmatrix} 0.978 \\ 0.997 \\ 0.999 \end{bmatrix} = \begin{bmatrix} 0.6(0.978) + 0.1(0.997) + 0.2(0.999) \\ 0.1(0.978) + 0.7(0.997) + 0.3(0.999) \\ 0.2(0.978) + 0.3(0.997) + 0.8(0.999) \end{bmatrix} = \begin{bmatrix} 0.886 \\ 1.095 \\ 1.294 \end{bmatrix}$$

3. **Suma con bias:**
   $$\begin{bmatrix} 2.3 \\ 3.4 \\ 4.5 \end{bmatrix} + \begin{bmatrix} 0.886 \\ 1.095 \\ 1.294 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.2 \\ 0.3 \end{bmatrix} = \begin{bmatrix} 3.286 \\ 4.695 \\ 6.094 \end{bmatrix}$$

4. **Aplicar $\tanh$:**
   $$h_3 = \tanh \begin{bmatrix} 3.286 \\ 4.695 \\ 6.094 \end{bmatrix} \approx \begin{bmatrix} 0.997 \\ 1.000 \\ 1.000 \end{bmatrix}$$

5. **Cálculo de la salida $y_3$:**
   $$y_3 = W_{hy} \cdot h_3 + b_y = \begin{bmatrix} 0.5 & 0.3 & 0.2 \end{bmatrix} \begin{bmatrix} 0.997 \\ 1.000 \\ 1.000 \end{bmatrix} + 0.1 = 0.5(0.997) + 0.3(1.000) + 0.2(1.000) + 0.1 = 1.099$$

---

![Captura del paso t=3](Paso%204.jpeg)
*Figura 1: Imagen del cálculo paso a paso para t=3.*

---

### Paso 5: Cuarto Paso Temporal ($t=4$)

**Entrada:** $x_4 = \begin{bmatrix} 7 \\ 8 \end{bmatrix}$

1. **Término de entrada ($W_{xh} \cdot x_4$):**
   $$\begin{bmatrix} 0.1 & 0.3 \\ 0.2 & 0.4 \\ 0.3 & 0.5 \end{bmatrix} \begin{bmatrix} 7 \\ 8 \end{bmatrix} = \begin{bmatrix} 0.1(7) + 0.3(8) \\ 0.2(7) + 0.4(8) \\ 0.3(7) + 0.5(8) \end{bmatrix} = \begin{bmatrix} 3.1 \\ 4.6 \\ 6.1 \end{bmatrix}$$

2. **Término recurrente ($W_{hh} \cdot h_3$):**
   $$\begin{bmatrix} 0.6 & 0.1 & 0.2 \\ 0.1 & 0.7 & 0.3 \\ 0.2 & 0.3 & 0.8 \end{bmatrix} \begin{bmatrix} 0.997 \\ 1.000 \\ 1.000 \end{bmatrix} = \begin{bmatrix} 0.6(0.997) + 0.1(1.000) + 0.2(1.000) \\ 0.1(0.997) + 0.7(1.000) + 0.3(1.000) \\ 0.2(0.997) + 0.3(1.000) + 0.8(1.000) \end{bmatrix} = \begin{bmatrix} 0.898 \\ 1.100 \\ 1.299 \end{bmatrix}$$

3. **Suma con bias:**
   $$\begin{bmatrix} 3.1 \\ 4.6 \\ 6.1 \end{bmatrix} + \begin{bmatrix} 0.898 \\ 1.100 \\ 1.299 \end{bmatrix} + \begin{bmatrix} 0.1 \\ 0.2 \\ 0.3 \end{bmatrix} = \begin{bmatrix} 4.098 \\ 5.900 \\ 7.699 \end{bmatrix}$$

4. **Aplicar $\tanh$:**
   $$h_4 = \tanh \begin{bmatrix} 4.098 \\ 5.900 \\ 7.699 \end{bmatrix} \approx \begin{bmatrix} 0.999 \\ 1.000 \\ 1.000 \end{bmatrix}$$

5. **Cálculo de la salida $y_4$:**
   $$y_4 = W_{hy} \cdot h_4 + b_y = \begin{bmatrix} 0.5 & 0.3 & 0.2 \end{bmatrix} \begin{bmatrix} 0.999 \\ 1.000 \\ 1.000 \end{bmatrix} + 0.1 = 0.5(0.999) + 0.3(1.000) + 0.2(1.000) + 0.1 = 1.100$$

---

![Captura del paso t=4](Paso%205.jpeg)
*Figura 2: Imagen del cálculo paso a paso para t=4.*
