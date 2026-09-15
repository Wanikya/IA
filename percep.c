#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define EPOCAS 10000
#define K 0.1f

float entrenar(float x0, float x1, float target);
float predecir(float x0, float x1);
float sigmoide(float s);
void inicializar_pesos(void);

float Pesos[2];
float bias = 0.0f;
float Error;

float entrenar(float x0, float x1, float target) {
    float net = Pesos[0]*x0 + Pesos[1]*x1 + bias;
    float out = sigmoide(net);
    
    Error = target - out;
    
    float delta = K * Error * out * (1 - out);
    
    Pesos[0] += delta * x0;
    Pesos[1] += delta * x1;
    bias += delta;
    
    return out;
}

float predecir(float x0, float x1) {
    float net = Pesos[0]*x0 + Pesos[1]*x1 + bias;
    return sigmoide(net);
}

float sigmoide(float s) {
    return 1.0f / (1.0f + expf(-s));
}

void inicializar_pesos(void) {
    float limit = sqrtf(6.0f / 2.0f);
    for (int i = 0; i < 2; i++) {
        Pesos[i] = ((float)rand() / RAND_MAX) * 2 * limit - limit;
    }
    bias = ((float)rand() / RAND_MAX) * 2 * limit - limit;
}

int main(void) {
    srand((unsigned)time(NULL));
    inicializar_pesos();
    
    printf("Entrenando PERCEPTRÓN para problema AND\n");
    printf("========================================\n\n");
    
    // Training loop
    for (int epoca = 0; epoca < EPOCAS; epoca++) {
        float error_total = 0.0f;
        
        error_total += fabsf(entrenar(0, 0, 0));
        error_total += fabsf(entrenar(0, 1, 0));
        error_total += fabsf(entrenar(1, 0, 0));
        error_total += fabsf(entrenar(1, 1, 1));
        
        if (epoca % 1000 == 0) {
            printf("Época %5d | Error total: %.6f | Pesos: [%.4f, %.4f] | Bias: %.4f\n",
                   epoca, error_total, Pesos[0], Pesos[1], bias);
        }
        
        if (error_total < 0.001f) {
            printf("\nConvergió en época %d\n", epoca);
            break;
        }
    }
    
    printf("\n--- RESULTADOS FINALES ---\n");
    printf("Pesos finales: w0=%.6f, w1=%.6f, bias=%.6f\n\n", Pesos[0], Pesos[1], bias);
    
    printf("Pruebas:\n");
    printf("(0,0) -> %.4f (esperado: 0.0)\n", predecir(0, 0));
    printf("(0,1) -> %.4f (esperado: 0.0)\n", predecir(0, 1));
    printf("(1,0) -> %.4f (esperado: 0.0)\n", predecir(1, 0));
    printf("(1,1) -> %.4f (esperado: 1.0)\n", predecir(1, 1));
    
    return 0;
}