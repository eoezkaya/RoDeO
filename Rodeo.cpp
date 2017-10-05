#include<stdio.h>
#include<iostream>


#include "su2_optim.hpp"
#include "test_functions.hpp"
#include "rbf.hpp"
#include "Rodeo_macros.hpp"



int main(void){

	/* initialize random seed*/
	srand (time(NULL));


	printf("\n**************************************************************\n");
	printf("*** RObust DEsign Optimization Package - TU Kaiserslautern ***");
	printf("\n**************************************************************\n");




	Classifier_Function_param test_function(2);

	test_function.function_name="classification_test1";
	test_function.func_ptr = classification_test1;
	test_function.bounds[0]=-4.0;
	test_function.bounds[1]= 4.0;
	test_function.bounds[2]=-6.0;
	test_function.bounds[3]= 6.0;
	test_function.noise_level = 0.1;



	perform_svn_test(test_function,
	        100,
	        0,
	        SVM_LINEAR);


//	perform_rbf_test(Eggholder,
//	        parameter_bounds,
//	        "Eggholder" ,
//	        20,
//	        LHS_CENTER,
//	        2,
//	        GAUSSIAN);




//	double parameter_bounds[2];
//	parameter_bounds[0]=-2.0;
//	parameter_bounds[1]=2.0;



//	perform_GEK_test1D(test_function1D,
//			test_function1D_adj,
//			parameter_bounds,
//			"test_function1D" ,
//			50,
//			50,
//			RANDOM_SAMPLING,
//			SVD,
//			LINEAR_REGRESSION_OFF);





//	perform_GEK_test(Eggholder,
//			Eggholder_adj,
//			parameter_bounds,
//			"Eggholder" ,
//			0,
//			20,
//			RANDOM_SAMPLING,
//			CHOLESKY,
//			2,
//			LINEAR_REGRESSION_OFF);
//
//
//
//
//
//
//	perform_kriging_test(Eggholder,
//			parameter_bounds,
//			"Eggholder" ,
//			100,
//			RANDOM_SAMPLING,
//			2,
//			CHOLESKY,
//			MATRIX_INVERSION,
//			LINEAR_REGRESSION_OFF,
//			MAXIMUM_LIKELIHOOD);
//
//
//	exit(1);


//	perform_trust_region_GEK_test(Eggholder,
//				Eggholder_adj,
//				parameter_bounds,
//				"Eggholder" ,
//				0,
//				50,
//				EXISTING_FILE,
//				CHOLESKY,
//				2,
//				LINEAR_REGRESSION_OFF);






}
