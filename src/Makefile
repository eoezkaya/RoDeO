VERSION = 0.2
CPP     = g++
#CPPFLAGS  = -Wall -O2 -fopenmp  
CPPFLAGS  = -Wall -g -fopenmp
#LDFLAGS = -lm -larmadillo -lgomp -lquadmath 
LDFLAGS = -lm -larmadillo -lgomp -lquadmath -lgtest -lpthread

OBJ = auxiliary_functions.o auxiliary_functions_test.o aggregation_test.o constraint_functions.o design.o drivers.o \
gek.o gek_test.o kriging_test.o kriging_training.o lhs.o linear_regression.o  \
matrix_vector_operations.o matrix_vector_operations_test.o metric.o objective_function.o \
optimization.o polynomials.o polynomials_test.o random_functions.o Rodeo.o surrogate_model.o test_functions.o trust_region_gek.o 
 
all : $(OBJ)
	$(CPP) $(CPPFLAGS) -o rodeo $(OBJ) $(LDFLAGS)

%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c $<

clean:
	rm *.o rodeo

