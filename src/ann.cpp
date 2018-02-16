#include "ann.hpp"
#include "Rodeo_macros.hpp"
#include "auxilliary_functions.hpp"
#include <armadillo>

using namespace arma;

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

namespace ann{




void test_ann1(void){

	int number_of_nodes_in_layers[2] ={4,4};
	feedForwardNetwork neuralNet1;
	neuralNet1.initialize(10,2,number_of_nodes_in_layers);


}



}
