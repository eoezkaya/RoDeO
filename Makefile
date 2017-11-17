VERSION = 0.1
CC      = g++
CFLAGS  = -Wall -O2 -std=c++11 -fopenmp
LDFLAGS = -lm -larmadillo

OBJ = Rodeo.o test_functions.o trust_region_gek.o kriging_training.o su2_optim.o \
auxilliary_functions.o linear_regression.o rbf.o svm.o kd_tree.o interpolation.o \
binary_search_tree.o kmeans_clustering.o bitmap_test.o mnist.o 
make: $(OBJ)
	$(CC) $(CFLAGS) -o rodeo $(OBJ) $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

clean:
	rm *.o rodeo
