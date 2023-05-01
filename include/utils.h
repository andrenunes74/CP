#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

typedef struct cluster
{
    float x;
    float y;
    float size; //tamanho do cluster
}CLUSTER;

void inicializa(float *x, float *y, CLUSTER *c, int n, int k);
void atribui(float *x, float *y, CLUSTER *c, int n, int k);
void trocas(float *x, float *y, CLUSTER *c, int n, int k);
void trocasOpenMP(float *x, float *y, CLUSTER *c, int n,  int k, int t);
void trocasMPI(float *x, float *y, CLUSTER *c, int n,  int k);
void trocasMPIandOpenMP(float *x, float *y, CLUSTER *c, int n,  int k, int t);