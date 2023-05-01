CC = gcc 
BIN = bin/
SRC = src/
INCLUDES = include/
EXEC = k_means
MPICC = mpicc
MPIL = -I/usr/lib/x86_64-linux-gnu/openmpi/include

#DEFINIR NUMERO DE CLUSTERS
CP_CLUSTERS = 4

#DEFINIR NUMERO DE THREADS
THREADS = 8

#DEFINIR NUMERO DE CORES
NP = 4

CFLAGS = -w -O2 -std=c99 -g -fno-omit-frame-pointer -fopenmp

.DEFAULT_GOAL = k_means

k_means: $(SRC)k_means.c $(BIN)utils.o
	$(MPICC) $(MPIL) $(CFLAGS) $(SRC)k_means.c $(BIN)utils.o -o $(BIN)$(EXEC)

$(BIN)utils.o: $(SRC)utils.c $(INCLUDES)utils.h
	$(MPICC) $(MPIL) $(CFLAGS) -c $(SRC)utils.c -o $(BIN)utils.o 

runSEQ:
	./$(BIN)$(EXEC) 0 10000000 $(CP_CLUSTERS)

runOpenMP:
	./$(BIN)$(EXEC) 1 10000000 $(CP_CLUSTERS) $(THREADS)

runMPI:
	mpirun -np $(NP) ./$(BIN)$(EXEC) 2 10000000 $(CP_CLUSTERS)

runMPIandOMP:
	mpirun -np $(NP) ./$(BIN)$(EXEC) 3 10000000 $(CP_CLUSTERS) $(THREADS)

clean:
	rm -r bin/*

