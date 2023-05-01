#include "../include/utils.h"

/*Iniciar um vetor com valores aleatórios (N amostras no espaco (x,y))
Iniciar os K clusters com as coordenadas das primeiras K amostras*/
void inicializa(float *x, float *y, CLUSTER *c, int n, int k){
    CLUSTER aux;
    srand(10);

    //inicializa pontos
    //nao paralelizavel porque a funcao nao e thread safe
    for (int i = 0; i < n; i++)
    {
        x[i] = (float) rand() / RAND_MAX;
        y[i] = (float) rand() / RAND_MAX;
    }
    //inicializa clusters
    for (int i = 0; i < k; i++)
    {
        aux.x = x[i];
        aux.y = y[i];
        aux.size = 0;
        c[i] = aux;
    }
}

/*Atribuir cada amostra ao cluster mais próximo usando a distância euclidiana*/
void atribui(float *x, float *y, CLUSTER *c, int n,  int k){
    int cmenor; //indice do cluster a menor distancia 
    float dm;   //menor distancia
    float d;    //distancia atual

    for (int i = 0; i < n; i++)
    {
        cmenor=0; //indice do cluster a menor distancia
        dm = (((x[i] - c[0].x)*(x[i] - c[0].x)) + ((y[i] - c[0].y)*(y[i] - c[0].y))); //menor distância
        //verifica qual a menor distancia
        for(int j = 1; j < k; j++){
            d = (((x[i] - c[j].x)*(x[i] - c[j].x)) + ((y[i] - c[j].y)*(y[i] - c[j].y)));
            if(d<dm){
                dm=d;
                cmenor=j;
            }
        }
        c[cmenor].size++;
    }
}

/*Atribuir cada amostra ao “cluster” mais próximo usando a distância euclidiana 
e devolve o número de trocas efetuadas*/

//Versao sequencial----------------------------------------------------------------------------------------------------
void trocas(float *x, float *y, CLUSTER *c, int n,  int k){

    float centroidesX[k];   //somas dos x dos pontos de k clusters
    float centroidesY[k];   //somas dos y dos pontos de k clusters
    int novosC[k];          //tamanho dos k clusters

    //inicializa os arrays de floats
    for (size_t i = 0; i < k; i++)
    {
        centroidesX[i]=0;
        centroidesY[i]=0;
        novosC[i]=0;
    }

    int cmenor; //indice do cluster a menor distancia 
    float dm;   //menor distancia
    float d;    //distancia atual

    for (int i = 0; i < n; i++)
    {
        int cmenor=0; //indice do cluster a menor distancia
        float dm = (((x[i] - c[0].x)*(x[i] - c[0].x)) + ((y[i] - c[0].y)*(y[i] - c[0].y))); //menor distância
        
        //verifica qual a menor distancia
        for(int j = 1; j < k; j++){
            d = (((x[i] - c[j].x)*(x[i] - c[j].x)) + ((y[i] - c[j].y)*(y[i] - c[j].y)));
            if(d<dm){
                dm=d;
                cmenor=j;
            }
        }
        //atualiza o cluster novo
        novosC[cmenor]++;
        centroidesX[cmenor]+=x[i];
        centroidesY[cmenor]+=y[i];
    }

    //Recalcula os centroides com as somas e sizes calculados do ciclo acima
    for (size_t i = 0; i < k; i++)
    {
        CLUSTER p;
        p.x = (centroidesX[i]/novosC[i]);
        p.y = (centroidesY[i]/novosC[i]);
        p.size = novosC[i];
        c[i] = p;
    }
}

//Versao OpenMP----------------------------------------------------------------------------------------------------
void trocasOpenMP(float *x, float *y, CLUSTER *c, int n,  int k, int t){

    omp_set_num_threads(t); //define o numero de threads a usar

    float centroidesX[k];   //somas dos x dos pontos de k clusters
    float centroidesY[k];   //somas dos y dos pontos de k clusters
    int novosC[k];          //tamanho dos k clusters

    //inicializa os arrays de floats
    for (size_t i = 0; i < k; i++)
    {
        centroidesX[i]=0;
        centroidesY[i]=0;
        novosC[i]=0;
    }

    int cmenor; //indice do cluster a menor distancia 
    float dm;   //menor distancia
    float d;    //distancia atual

    #pragma omp parallel for private(cmenor,dm,d) reduction(+:centroidesX,centroidesY,novosC) 
    for (int i = 0; i < n; i++)
    {
        int cmenor=0;
        float dm = (((x[i] - c[0].x)*(x[i] - c[0].x)) + ((y[i] - c[0].y)*(y[i] - c[0].y))); //menor distância
        
        //verifica qual a menor distancia
        for(int j = 1; j < k; j++){
            d = (((x[i] - c[j].x)*(x[i] - c[j].x)) + ((y[i] - c[j].y)*(y[i] - c[j].y)));
            if(d<dm){
                dm=d;
                cmenor=j;
            }
        }
        //atualiza o cluster novo
        novosC[cmenor]++;
        centroidesX[cmenor]+=x[i];
        centroidesY[cmenor]+=y[i];
    }

    //Recalcula os centroides com as somas e sizes calculados do ciclo acima
    for (size_t i = 0; i < k; i++)
    {
        CLUSTER p;
        p.x = (centroidesX[i]/novosC[i]);
        p.y = (centroidesY[i]/novosC[i]);
        p.size = novosC[i];
        c[i] = p;
    }
}

//Versao MPI----------------------------------------------------------------------------------------------------
void trocasMPI(float *x, float *y, CLUSTER *c, int n,  int k){

    int rank,size;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);

    int elems = n/size;         //elementos a processar pelo core
    int ind = rank*elems;       //indice inicial do array de pontos a processar pelo processo
    int lim = (rank+1) * elems; //indice final do array de pontos a processar pelo processo

    float centroidesX[k];   //somas dos x dos pontos de k clusters
    float centroidesY[k];   //somas dos y dos pontos de k clusters
    float novosC[k];        //tamanho dos k clusters

    //inicializa os arrays de floats
    for (size_t i = 0; i < k; i++)
    {
        centroidesX[i]=0;
        centroidesY[i]=0;
        novosC[i]=0;
    }

    int cmenor; //indice do cluster a menor distancia 
    float dm;   //menor distancia
    float d;    //distancia atual

    for (int i = ind; i < lim; i++)
    {
        int cmenor=0; //indice do cluster a menor distancia
        float dm = (((x[i] - c[0].x)*(x[i] - c[0].x)) + ((y[i] - c[0].y)*(y[i] - c[0].y))); //menor distância
        
        //verifica qual a menor distancia
        for(int j = 1; j < k; j++){
            d = (((x[i] - c[j].x)*(x[i] - c[j].x)) + ((y[i] - c[j].y)*(y[i] - c[j].y)));
            if(d<dm){
                dm=d;
                cmenor=j;
            }
        }
        //atualiza o cluster novo
        novosC[cmenor]++;
        centroidesX[cmenor]+=x[i];
        centroidesY[cmenor]+=y[i];
    }

    float centroidesXa[k];   //array auxiliar para o reduce das somas dos pontos x dos clusters
    float centroidesYb[k];   //array auxiliar para o reduce das somas dos pontos y dos clusters
    float novosCc[k];        //array auxiliar para o reduce dos tamanhos dos clusters

    MPI_Reduce(novosC, novosCc, k, MPI_FLOAT, MPI_SUM, 0, comm);
    MPI_Reduce(centroidesX, centroidesXa, k, MPI_FLOAT, MPI_SUM, 0, comm);
    MPI_Reduce(centroidesY, centroidesYb, k, MPI_FLOAT, MPI_SUM, 0, comm);

    if(rank == 0){
        //Recalcula os centroides com as somas e sizes calculados do ciclo acima
        for (size_t i = 0; i < k; i++)
        {
            CLUSTER p;
            p.x = (centroidesXa[i]/novosCc[i]);
            p.y = (centroidesYb[i]/novosCc[i]);
            p.size = novosCc[i];
            c[i] = p;
        }
    }
    MPI_Bcast(c, k*3, MPI_FLOAT, 0, comm);
}

//Versao MPI e OpenMP----------------------------------------------------------------------------------------------------
void trocasMPIandOpenMP(float *x, float *y, CLUSTER *c, int n, int k, int t){

    omp_set_num_threads(t); //define o numero de threads a usar

    int rank,size;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);

    int elems = n/size;         //elementos a processar pelo core
    int ind = rank*elems;       //indice inicial do array de pontos a processar pelo processo
    int lim = (rank+1) * elems; //indice final do array de pontos a processar pelo processo

    float centroidesX[k];   //somas dos x dos pontos de k clusters
    float centroidesY[k];   //somas dos y dos pontos de k clusters
    float novosC[k];        //tamanho dos k clusters

    //inicializa os arrays de floats
    for (size_t i = 0; i < k; i++)
    {
        centroidesX[i]=0;
        centroidesY[i]=0;
        novosC[i]=0;
    }

    int cmenor; //indice do cluster a menor distancia 
    float dm;   //menor distancia
    float d;    //distancia atual

    #pragma omp parallel for private(cmenor,dm,d) reduction(+:centroidesX,centroidesY,novosC) 
    for (int i = ind; i < lim; i++)
    {
        int cmenor=0; //indice do cluster a menor distancia
        float dm = (((x[i] - c[0].x)*(x[i] - c[0].x)) + ((y[i] - c[0].y)*(y[i] - c[0].y))); //menor distância
        
        //verifica qual a menor distancia
        for(int j = 1; j < k; j++){
            d = (((x[i] - c[j].x)*(x[i] - c[j].x)) + ((y[i] - c[j].y)*(y[i] - c[j].y)));
            if(d<dm){
                dm=d;
                cmenor=j;
            }
        }
        //atualiza o cluster novo
        novosC[cmenor]++;
        centroidesX[cmenor]+=x[i];
        centroidesY[cmenor]+=y[i];
    }

    float centroidesXa[k];   //array auxiliar para o reduce das somas dos pontos x dos clusters
    float centroidesYb[k];   //array auxiliar para o reduce das somas dos pontos y dos clusters
    float novosCc[k];        //array auxiliar para o reduce dos tamanhos dos clusters

    MPI_Reduce(novosC, novosCc, k, MPI_FLOAT, MPI_SUM, 0, comm);
    MPI_Reduce(centroidesX, centroidesXa, k, MPI_FLOAT, MPI_SUM, 0, comm);
    MPI_Reduce(centroidesY, centroidesYb, k, MPI_FLOAT, MPI_SUM, 0, comm);

    if(rank == 0){
        //Recalcula os centroides com as somas e sizes calculados do ciclo acima
        for (size_t i = 0; i < k; i++)
        {
            CLUSTER p;
            p.x = (centroidesXa[i]/novosCc[i]);
            p.y = (centroidesYb[i]/novosCc[i]);
            p.size = novosCc[i];
            c[i] = p;
        }
    }
    MPI_Bcast(c, k*3, MPI_FLOAT, 0, comm);
}