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

#ifdef GPU_VERSION
#include "kernel_regression_cuda.h"
#endif

#include "trust_region_gek.hpp"
#include "kriging_training.hpp"

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
		exit(-1);
	}




//	KrigingModel EggholderModel("Eggholder",2);
//	EggholderModel.test(Eggholder);

//
//
//
	TestFunction EggholderFunc("Eggholder",2);
	EggholderFunc.func_ptr = Eggholder;
////	EggholderFunc.adj_ptr = Eggholder_adj;
////	EggholderFunc.ifAdjointFunctionExist = true;;
	EggholderFunc.lower_bound.fill(0.0);
	EggholderFunc.upper_bound.fill(200.0);
	EggholderFunc.print();
	EggholderFunc.testEGO(400,100);

//	EggholderFunc.plot();
//	EggholderFunc.testKrigingModel(500);

//	EggholderFunc.generateRandomSamples(500,"Eggholder500.csv");




	return 0;





	OptimizerWithGradients OptimizationStudy("Eggholder400",2);

	OptimizationStudy.adj_fun = Eggholder_adj;
	OptimizationStudy.max_number_of_samples = 100;
	OptimizationStudy.lower_bound_dv.fill(0);
	OptimizationStudy.upper_bound_dv.fill(200.0);
	OptimizationStudy.doesValidationFileExist = true;

	OptimizationStudy.EfficientGlobalOptimization();




}
