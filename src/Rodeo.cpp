


#include<stdio.h>
#include<iostream>


#include "su2_optim.hpp"
#include "test_functions.hpp"
#include "rbf.hpp"
#include "kd_tree.hpp"
#include "binary_search_tree.hpp"
#include "linear_regression.hpp"
#include "kmeans_clustering.hpp"
#include "Rodeo_macros.hpp"
#include "auxilliary_functions.hpp"
#include "read_settings.hpp"
#include "kernel_regression.hpp"

int main(void){


	printf("\n\n\n");


	printf("	  ____       ____        ___   \n");
	printf("	 |  _ \\ ___ |  _ \\  ___ / _ \\  \n");
	printf("	 | |_) / _ \\| | | |/ _ \\ | | | \n");
	printf("	 |  _ < (_) | |_| |  __/ |_| | \n");
	printf("	 |_| \\_\\___/|____/ \\___|\\___/  \n");

	printf("\n");
	printf("    RObust DEsign Optimization Package      ");
	printf("\n\n\n");



	/* initialize random seed*/
	srand (time(NULL));
	arma_rng::set_seed_random();

	Rodeo_settings settings;
	settings.read();

	int ret = chdir (settings.cwd.c_str());



	trainMahalanobisDistance();




//	double parameter_bounds[4];
//	parameter_bounds[0]=0.0;
//	parameter_bounds[1]=200.0;
//	parameter_bounds[2]=0.0;
//	parameter_bounds[3]=200.0;
//
//	perform_kernel_regression_test(Eggholder,
//			Eggholder_adj,
//			parameter_bounds,
//			"Eggholder",
//			0,
//			100,
//			RANDOM_SAMPLING,
//			2,
//			settings.python_dir);


//	su2_optimize(settings.python_dir);

	//	initial_data_acquisitionGEK(settings.python_dir, 200);




}
