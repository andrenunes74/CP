/*
    ---Algoritmo K-means simples, baseado no algoritmo de Lloyd---
O algoritmo classifica um conjunto de N amostras em K grupos (clusters)
*/
#include "../include/utils.h"

int main(int argc, char *argv[])
{
    /*--------------------variaveis globais------------------*/
    int i=0; 
    int t = atoi(argv[1]); 
    int n = atoi(argv[2]); 
    int k = atoi(argv[3]);
    float *x = malloc(n * sizeof (float)); 
    float *y = malloc(n * sizeof (float));
    CLUSTER *clusters = malloc(k * sizeof (struct cluster)); 
    /*-------------------------------------------------------*/

    inicializa(x,y,clusters,n,k); //Iniciar um vetor com valores aleatórios (N amostras no espaço (x,y)) e K clusters/
    atribui(x,y,clusters,n,k); //Atribuir cada amostra ao cluster mais pr óximo usando a distância euclidiana/

    //Versao sequencial
    if(t==0){
        while (i!=20)
        {
            trocas(x,y,clusters,n,k); //Atribuir cada amostra ao “cluster” mais próximo usando a distância euclidiana e devolve o número de trocas efetuadas/
            i++;
        }

        printf("N = %d, K = %d\n", n,k);
        for (int i = 0; i < k; i++)
        {
            printf("Center: (%.3f, %.3f) : Size: %d\n", clusters[i].x, clusters[i].y, (int)clusters[i].size);
        }
        printf("ITERAÇÕES: %d\n", i);
    }

    //Versao paralela OpenMP
    if(t==1){
        int t = atoi(argv[4]);
        while (i!=20)
        {
            trocasOpenMP(x,y,clusters,n,k,t); //Atribuir cada amostra ao “cluster” mais próximo usando a distância euclidiana e devolve o número de trocas efetuadas/
            i++;
        }

        printf("N = %d, K = %d\n", n,k);
        for (int i = 0; i < k; i++)
        {
            printf("Center: (%.3f, %.3f) : Size: %d\n", clusters[i].x, clusters[i].y, (int)clusters[i].size);
        }
        printf("ITERAÇÕES: %d\n", i);
    }

    //Versao paralela MPI
    if(t==2){
        MPI_Init(&argc, &argv);
        while (i!=20)
        {
            trocasMPI(x,y,clusters,n,k); //Atribuir cada amostra ao “cluster” mais próximo usando a distância euclidiana/
            i++;
        }

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank == 0){
            printf("N = %d, K = %d\n", n,k);
            for (int i = 0; i < k; i++)
            {
                printf("Center: (%.3f, %.3f) : Size: %d\n", clusters[i].x, clusters[i].y, (int)clusters[i].size);
            }
            printf("ITERAÇÕES: %d\n", i);
        }
        MPI_Finalize();
    }

    //Versao paralela MPI + OpenMP
    if(t==3){
        int t = atoi(argv[4]);
        MPI_Init(&argc, &argv);
        while (i!=20)
        {
            trocasMPIandOpenMP(x,y,clusters,n,k,t); //Atribuir cada amostra ao “cluster” mais próximo usando a distância euclidiana/
            i++;
        }

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank == 0){
            printf("N = %d, K = %d\n", n,k);
            for (int i = 0; i < k; i++)
            {
                printf("Center: (%.3f, %.3f) : Size: %d\n", clusters[i].x, clusters[i].y, (int)clusters[i].size);
            }
            printf("ITERAÇÕES: %d\n", i);
        }
        MPI_Finalize();
    }
    
    return 0;
}