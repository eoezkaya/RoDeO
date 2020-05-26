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
#include "surrogate_model.hpp"

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


//	TestFunction TestFunc1D("TestFunction1D",1);
//	TestFunc1D.func_ptr = TestFunction1D;
////	////	EggholderFunc.adj_ptr = Eggholder_adj;
////	////	EggholderFunc.ifAdjointFunctionExist = true;;
//	TestFunc1D.setBoxConstraints(0.0,10.0);
//	TestFunc1D.print();
////	TestFunc1D.plot();
////	TestFunc1D.testEGO(100,100);
//
////	TestFunc1D.plot();
//	TestFunc1D.testKrigingModel(20,true);
//
//	exit(1);


	TestFunction HimmelblauFunc("Himmelblau",2);
	HimmelblauFunc.func_ptr = Himmelblau;
	////	EggholderFunc.adj_ptr = Eggholder_adj;
	////	EggholderFunc.ifAdjointFunctionExist = true;;
	HimmelblauFunc.setBoxConstraints(-5.0,5.0);
	HimmelblauFunc.print();
	HimmelblauFunc.plot();
	HimmelblauFunc.testSurrogateModel(LINEAR_REGRESSION,100,true);

//	HimmelblauFunc.testEfficientGlobalOptimization(20,20, true);
	exit(1);


	TestFunction EggholderFunc("Eggholder",2);
	EggholderFunc.func_ptr = Eggholder;
////	EggholderFunc.adj_ptr = Eggholder_adj;
////	EggholderFunc.ifAdjointFunctionExist = true;;
	EggholderFunc.setBoxConstraints(-512.0,512.0);
	EggholderFunc.print();
	EggholderFunc.plot();

//	EggholderFunc.testEfficientGlobalOptimization(50,100, true);

//	EggholderFunc.plot();
//	EggholderFunc.testKrigingModel(50);

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
