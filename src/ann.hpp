#ifndef ANN_HPP
#define ANN_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"


using namespace arma;


namespace ann{


class feedForwardNetwork {
public:
	double *input;
	int number_of_inputs;
	double **w;
	int number_of_layers;
	int* number_of_nodes_in_layers;


	void initialize(int dim_x, int dim_layer,int *nNodes)

	{

		number_of_inputs = dim_x;
		number_of_layers = dim_layer;
		number_of_nodes_in_layers = nNodes;








	}


};




}




#endif
