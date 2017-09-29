#include<stdio.h>
#include<iostream>


#include "su2_optim.hpp"
#include "test_functions.hpp"
#include "Rodeo_macros.hpp"



int main(void){

	/* initialize random seed*/
	srand (time(NULL));


	printf("\n**************************************************************\n");
	printf("*** RObust DEsign Optimization Package - TU Kaiserslautern ***");
	printf("\n**************************************************************\n");


	double parameter_bounds[4];
	parameter_bounds[0]=0.0;
	parameter_bounds[1]=100.0;
	parameter_bounds[2]=0.0;
	parameter_bounds[3]=100.0;


	perform_rbf_test(Eggholder,
	        parameter_bounds,
	        "Eggholder" ,
	        50,
	        RANDOM_SAMPLING,
	        2,
	        GAUSSIAN,
	        CHOLESKY);




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
