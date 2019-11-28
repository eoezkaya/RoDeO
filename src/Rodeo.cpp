#include<stdio.h>
#include<iostream>


#include "su2_optim.hpp"
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


	AggregationModel settings_CD("CL",38);


	settings_CD.validationset_input_filename = "CL_Validation.csv";
	settings_CD.max_number_of_kriging_iterations = 10000;
	settings_CD.number_of_cv_iterations = 20;
	settings_CD.visualizeKrigingValidation = "yes";
	settings_CD.visualizeKernelRegressionValidation = "yes";
	settings_CD.visualizeAggModelValidation = "yes";
	train_aggregation_model(settings_CD);


	exit(1);

	double parameter_bounds[12];
	parameter_bounds[0]=-5.12; parameter_bounds[1]=5.12;
	parameter_bounds[2]=-5.12; parameter_bounds[3]=5.12;
	parameter_bounds[4]=-5.12; parameter_bounds[5]=5.12;
	parameter_bounds[6]=-5.12; parameter_bounds[7]=5.12;
	parameter_bounds[8]=-5.12; parameter_bounds[9]=5.12;
	parameter_bounds[10]=-5.12; parameter_bounds[11]=5.12;

	perform_aggregation_model_test_highdim(Rastrigin6D,
			Rastrigin6D_adj,
			parameter_bounds,
			"Rastrigin6D",
			400,
			RANDOM_SAMPLING,
			6);



	//	AggregationModel settings_CD("CD",13);
	//
	//
	//	settings_CD.validationset_input_filename = "CD_val.csv";
	//	settings_CD.max_number_of_kriging_iterations = 10000;
	//	settings_CD.visualizeKrigingValidation = "yes";
	//	settings_CD.visualizeKernelRegressionValidation = "yes";
	//	settings_CD.number_of_cv_iterations = 50;
	//
	//	settings_CD.visualizeAggModelValidation = "yes";
	//	train_aggregation_model(settings_CD);
	//
	//	exit(1);




	//		double parameter_bounds[4];
	//		parameter_bounds[0]=0.0; parameter_bounds[1]=200.0;
	//		parameter_bounds[2]=0.0; parameter_bounds[3]=200.0;
	//
	//
	//
	//	perform_NNregression_test(Eggholder,
	//						parameter_bounds,
	//						"Eggholder" ,
	//						1250,
	//						RANDOM_SAMPLING,
	//						2,
	//						100);


	//	double *parameter_bounds = new double[20];
	//	parameter_bounds[0]=150.0; parameter_bounds[1]=200.0;
	//	parameter_bounds[2]=220.0; parameter_bounds[3]=300.0;
	//	parameter_bounds[4]=6.0; parameter_bounds[5]=10.0;
	//	parameter_bounds[6]=-10.0; parameter_bounds[7]=10.0;
	//	parameter_bounds[8]=16.0; parameter_bounds[9]=45.0;
	//	parameter_bounds[10]=0.5; parameter_bounds[11]=1.0;
	//	parameter_bounds[12]=0.08; parameter_bounds[13]=0.018;
	//	parameter_bounds[14]=2.5; parameter_bounds[15]=6.0;
	//	parameter_bounds[16]=1700.0; parameter_bounds[17]=2500.0;
	//	parameter_bounds[18]=0.025; parameter_bounds[19]=0.08;
	//
	//
	//		perform_NNregression_test(Wingweight,
	//					parameter_bounds,
	//					"Wingweight" ,
	//					500,
	//					RANDOM_SAMPLING,
	//					10,
	//					500);
	//
	//	exit(1);
	//
	//
	//	perform_aggregation_model_test_highdim(Wingweight,
	//			WingweightAdj,
	//			parameter_bounds,
	//			"Wingweight",
	//			400,
	//			RANDOM_SAMPLING,
	//			10);



	//
	//
	//	perform_kernel_regression_test_highdim(Eggholder,
	//			EggholderAdj,
	//			parameter_bounds,
	//			"Eggholder",
	//			0,
	//			3000,
	//			RANDOM_SAMPLING,
	//			2);



	//	double parameter_bounds[4];
	//	parameter_bounds[0]=0.0; parameter_bounds[1]=200.0;
	//	parameter_bounds[2]=0.0; parameter_bounds[3]=200.0;
	//
	//
	//	generate_highdim_test_function_data_GEK(Eggholder,
	//					EggholderAdj,
	//					"Eggholder_validation.csv",
	//					parameter_bounds,
	//					2,
	//					1000,
	//					0,
	//					RANDOM_SAMPLING);
	//
	//
	//
	//		generate_highdim_test_function_data_GEK(Eggholder,
	//				EggholderAdj,
	//				"Eggholder400.csv",
	//				parameter_bounds,
	//				2,
	//				0,
	//				400,
	//				RANDOM_SAMPLING);
	//
	//
	//
	//	AggregationModel settings_Eggholder("Eggholder400",2);
	//
	//
	//	settings_Eggholder.validationset_input_filename = "Eggholder_validation.csv";
	//	settings_Eggholder.max_number_of_kriging_iterations = 10000;
	//	settings_Eggholder.visualizeKrigingValidation = "yes";
	//	settings_Eggholder.visualizeKernelRegressionValidation = "yes";
	//	settings_Eggholder.number_of_cv_iterations = 20;
	//
	//	settings_Eggholder.visualizeAggModelValidation = "yes";
	//	train_aggregation_model(settings_Eggholder);
	//
	//	exit(1);
	//
	//
	//
	//	KrigingModel model_area("Area",38);
	//
	//	train_kriging_response_surface(model_area);
	//
	//
	//	exit(1);
	//
	//
	//	AggregationModel settings_CD("CL",38);
	//
	//
	//	settings_CD.validationset_input_filename = "CL_Validation.csv";
	//	settings_CD.max_number_of_kriging_iterations = 1000;
	//	settings_CD.number_of_cv_iterations = 0;
	//	settings_CD.visualizeKrigingValidation = "yes";
	//	settings_CD.visualizeKernelRegressionValidation = "yes";
	//	settings_CD.visualizeAggModelValidation = "yes";
	//	train_aggregation_model(settings_CD);
	//
	//	exit(1);
	//
	//
	//
	//	initial_data_acquisition(200);



	/*
	 *  Sw: Wing Area (ft^2) (150,200)
	 *  Wfw: Weight of fuel in the wing (lb) (220,300)
	 *  A: Aspect ratio (6,10)
	 *  Lambda: quarter chord sweep (deg) (-10,10)
	 *  q: dynamic pressure at cruise (lb/ft^2)  (16,45)
	 *  lambda: taper ratio (0.5,1)
	 *  tc: aerofoil thickness to chord ratio (0.08,0.18)
	 *  Nz: ultimate load factor (2.5,6)
	 *  Wdg: flight design gross weight (lb)  (1700,2500)
	 *  Wp: paint weight (lb/ft^2) (0.025, 0.08)
	 *
	 */


	//	double parameter_bounds[20];
	//	parameter_bounds[0]=150.0; parameter_bounds[1]=200.0;
	//	parameter_bounds[2]=220.0; parameter_bounds[3]=300.0;
	//	parameter_bounds[4]=6.0; parameter_bounds[5]=10.0;
	//	parameter_bounds[6]=-10.0; parameter_bounds[7]=10.0;
	//	parameter_bounds[8]=16.0; parameter_bounds[9]=45.0;
	//	parameter_bounds[10]=0.5; parameter_bounds[11]=1.0;
	//	parameter_bounds[12]=0.08; parameter_bounds[13]=0.018;
	//	parameter_bounds[14]=2.5; parameter_bounds[15]=6.0;
	//	parameter_bounds[16]=1700.0; parameter_bounds[17]=2500.0;
	//	parameter_bounds[18]=0.025; parameter_bounds[19]=0.08;


	//	perform_kriging_test(Wingweight,
	//			parameter_bounds,
	//			"Wingweight" ,
	//			400,
	//			RANDOM_SAMPLING,
	//			10,
	//			LINEAR_REGRESSION_OFF);




	//
	//	perform_kernel_regression_test_highdim(Wingweight,
	//			WingweightAdj,
	//			parameter_bounds,
	//			"Wingweight",
	//			0,
	//			200,
	//			RANDOM_SAMPLING,
	//			10);



	// double parameter_bounds[10];
	// 	parameter_bounds[0]=-1.0; parameter_bounds[1]=2.0;
	// 	parameter_bounds[2]=0.0; parameter_bounds[3]=3.0;
	// 	parameter_bounds[4]=6.0; parameter_bounds[5]=10.0;
	// 	parameter_bounds[6]=-10.0; parameter_bounds[7]=10.0;
	// 	parameter_bounds[8]=16.0; parameter_bounds[9]=45.0;

	// perform_kernel_regression_test_highdim(test_function2KernelReg,
	// 		test_function2KernelRegAdj,
	// 			parameter_bounds,
	// 			"test_function2KernelReg",
	// 			0,
	// 			200,
	// 			RANDOM_SAMPLING,
	// 			5);

	//		double *parameter_bounds = new double[20];
	//		parameter_bounds[0]=150.0; parameter_bounds[1]=200.0;
	//		parameter_bounds[2]=220.0; parameter_bounds[3]=300.0;
	//		parameter_bounds[4]=6.0; parameter_bounds[5]=10.0;
	//		parameter_bounds[6]=-10.0; parameter_bounds[7]=10.0;
	//		parameter_bounds[8]=16.0; parameter_bounds[9]=45.0;
	//		parameter_bounds[10]=0.5; parameter_bounds[11]=1.0;
	//		parameter_bounds[12]=0.08; parameter_bounds[13]=0.018;
	//		parameter_bounds[14]=2.5; parameter_bounds[15]=6.0;
	//		parameter_bounds[16]=1700.0; parameter_bounds[17]=2500.0;
	//		parameter_bounds[18]=0.025; parameter_bounds[19]=0.08;
	//
	//	perform_kernel_regression_test_highdim_cuda(Wingweight,
	//			WingweightAdj,
	//			parameter_bounds,
	//			"Wingweight",
	//			0,
	//			400,
	//			RANDOM_SAMPLING,
	//			10);


	//	double parameter_bounds[4];
	//	parameter_bounds[0]=-5.0; parameter_bounds[1]=5.0;
	//	parameter_bounds[2]=-5.0; parameter_bounds[3]=5.0;
	//
	//	perform_kernel_regression_test_highdim_cuda(Waves2D,
	//			Waves2D_adj,
	//			parameter_bounds,
	//			"Waves",
	//			0,
	//			200,
	//			RANDOM_SAMPLING,
	//			2);


	//
	//
	//	delete[] parameter_bounds;


}
