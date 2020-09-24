#include<stdio.h>
#include<iostream>


#include "optimization.hpp"
#include "test_functions.hpp"
#include "linear_regression.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "auxilliary_functions.hpp"
#include "read_settings.hpp"

#include "kernel_regression.hpp"
#include "kernel_regression_test.hpp"
#include "aggregation_test.hpp"
#include "surrogate_model.hpp"

#ifdef GPU_VERSION
#include "kernel_regression_cuda.h"
#endif

#include "trust_region_gek.hpp"
#include "kriging_training.hpp"
#include "polynomials.hpp"
#include "polynomials_test.hpp"
Rodeo_settings settings;

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



	settings.read();


	int ret = chdir (settings.cwd.c_str());

	if (ret != 0){

		fprintf(stderr, "Error: cannot change directory! at %s, line %d.\n",__FILE__, __LINE__);
		abort();
	}


	testAllKernelRegression();

	TestFunction testFunHimmelblau("Himmelblau",2);
	testFunHimmelblau.adj_ptr = HimmelblauAdj;
	testFunHimmelblau.func_ptr = Himmelblau;
	testFunHimmelblau.setVisualizationOn();
	testFunHimmelblau.setBoxConstraints(0.0,5.0);
	testFunHimmelblau.testSurrogateModel(GRADIENT_ENHANCED_KERNEL_REGRESSION,100);





}
