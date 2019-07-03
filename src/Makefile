VERSION = 0.1
#CC      = nvcc -arch compute_61 
CC      = nvcc -arch=sm_52
AD_PATH = /home/hari/Work/CoDiPack/include 
CUDA_PATH = /home/hari/Work/Common
CFLAGS  = -std=c++11 -Xcompiler -fopenmp -I $(AD_PATH) -I $(CUDA_PATH)
LDFLAGS = -lm -larmadillo -rdc=true -lcudadevrt -lgomp



OBJ = Rodeo.o test_functions.o trust_region_gek.o kriging_training.o su2_optim.o \
auxilliary_functions.o linear_regression.o rbf.o kd_tree.o \
binary_search_tree.o kmeans_clustering.o kernel_regression_cuda.o
make: $(OBJ)
	$(CC) $(CFLAGS) -o rodeo $(OBJ) $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $<

%.o: %.cu
	$(CC) $(CFLAGS) -c $<

clean:
	rm *.o rodeo

# For an independent file
# nvcc -arch compute_61 -std=c++11 New-version-double.cu -Xcompiler -fopenmp 
# -I /home/hari/Work/CoDiPack/include -I /home/hari/Work/Common -lm -larmadillo -rdc=true -lcudadevrt -lgomp -o double.out

# For an independent file (alternate, works with atomicAdd having arguments of type double, surprisingly)
# nvcc -arch=sm_52 -std=c++11 New-version-double.cu -Xcompiler -fopenmp 
# -I /home/hari/Work/CoDiPack/include -I /home/hari/Work/Common -lm -larmadillo -rdc=true -lcudadevrt -lgomp -o double.out

# Old Version 
# nvcc -arch compute_52 -std=c++11 New-version.cu 
# -I /home/hari/Work/CoDiPack/include -I /home/hari/Work/Common -lm -larmadillo -rdc=true -lcudadevrt -o double.out