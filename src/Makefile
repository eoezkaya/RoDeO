VERSION = 0.2
CPP     = g++
#CPPFLAGS  = -Wall -O2 -fopenmp  
CPPFLAGS  = -Wall -g -fopenmp -DUNIT_TESTS
LDFLAGS = -lm -larmadillo -lgomp -lquadmath 
LDFLAGS_UNIT_TESTS = -lm -larmadillo -lgomp -lquadmath -lgtest -lpthread

OBJ = auxiliary_functions.o aggregation_model.o constraint_functions.o design.o drivers.o \
gek.o gek_test.o kriging_training.o lhs.o linear_regression.o  \
matrix_vector_operations.o metric.o objective_function.o \
optimization.o polynomials.o  random_functions.o Rodeo.o surrogate_model.o test_functions.o 

OBJ_UNIT_TESTS =  aggregation_test.o auxiliary_functions_test.o kriging_test.o matrix_vector_operations_test.o polynomials_test.o 
all : $(OBJ)
	$(CPP) $(CPPFLAGS) -o rodeo $(OBJ) $(LDFLAGS)
test: $(OBJ) $(OBJ_UNIT_TESTS)
	$(CPP) $(CPPFLAGS) -o rodeo $(OBJ) $(OBJ_UNIT_TESTS) $(LDFLAGS_UNIT_TESTS)
%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c $<

clean:
	rm *.o rodeo

