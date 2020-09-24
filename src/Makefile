VERSION = 0.2
#CC      = nvcc -arch compute_61 
#NVCC      = nvcc -arch=sm_52
CPP     = g++
#AD_PATH = /home/emre/CoDiPack-master/include
#CUDA_PATH = ~/NVIDIA_CUDA-10.0_Samples/common/inc/ 
#CFLAGS_GPU  = -std=c++11 -Xcompiler -fopenmp -I $(AD_PATH) -I $(CUDA_PATH)
#CFLAGS  = -std=c++11 -fopenmp -I $(AD_PATH) 
CPPFLAGS  = -Wall -g -fopenmp  
#LDFLAGS_GPU = -lm -larmadillo -rdc=true -lcudadevrt -lgomp
LDFLAGS = -lm -larmadillo -lgomp

OBJ = auxilliary_functions.o aggregation_test.o kernel_regression_cpu.o kernel_regression_test.o kriging_training.o linear_regression.o  \
matrix_vector_operations.o metric.o optimization.o polynomials.o polynomials_test.o random_functions.o Rodeo.o surrogate_model.o test_functions.o trust_region_gek.o  
 
#OBJ_GPU = Rodeo.o test_functions.o trust_region_gek.o kriging_training.o optimization.o \
auxilliary_functions.o linear_regression.o kernel_regression_cuda.o 
make: $(OBJ)
	$(CPP) $(CPPFLAGS) -o rodeo $(OBJ) $(LDFLAGS)
	
#gpu: $(OBJ_GPU)
#	$(CC) $(CFLAGS_GPU) -o rodeo $(OBJ) $(LDFLAGS_GPU)	

%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c $<

#%.o: %.cu
#	$(NVCC) $(CFLAGS_GPU) -c $<

clean:
	rm *.o rodeo

