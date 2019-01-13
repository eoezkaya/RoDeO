VERSION = 0.1
CC      = g++
AD_PATH = /home/emre/CoDiPack-master/include
CFLAGS  = -Wall -O2 -std=c++11 -fopenmp -I $(AD_PATH)
LDFLAGS = -lm -larmadillo
SOURCE_DIR = ./src


OBJ = Rodeo.o test_functions.o trust_region_gek.o kriging_training.o su2_optim.o \
auxilliary_functions.o linear_regression.o rbf.o kd_tree.o \
binary_search_tree.o kmeans_clustering.o kernel_regression.o
make: $(OBJ)
	$(CC) $(CFLAGS) -o rodeo $(OBJ) $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm *.o rodeo
