VERSION = 0.2
CPP     = g++
CPPFLAGS  = -Wall -g -fopenmp  
LDFLAGS = -lm -larmadillo -lgomp -lquadmath

OBJ = auxiliary_functions.o aggregation_test.o constraint_functions.o design.o drivers.o gek.o gek_test.o kriging_training.o lhs.o linear_regression.o  \
matrix_vector_operations.o metric.o objective_function.o optimization.o polynomials.o polynomials_test.o random_functions.o Rodeo.o surrogate_model.o test_functions.o trust_region_gek.o 
 
make: $(OBJ)
	$(CPP) $(CPPFLAGS) -o rodeo $(OBJ) $(LDFLAGS)
	
%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c $<

clean:
	rm *.o rodeo

