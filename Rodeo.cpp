#include<stdio.h>
#include<iostream>


#include "su2_optim.hpp"
#include "test_functions.hpp"
#include "rbf.hpp"
#include "svm.hpp"
#include "kd_tree.hpp"
#include "binary_search_tree.hpp"
#include "linear_regression.hpp"
#include "kmeans_clustering.hpp"
#include "interpolation.hpp"
#include "Rodeo_macros.hpp"
#include "auxilliary_functions.hpp"
#include "bitmap_image.hpp"
#include "mnist.hpp"


int main(void){

	/* initialize random seed*/
	srand (time(NULL));


	printf("\n**************************************************************\n");
	printf("*** RObust DEsign Optimization Package - TU Kaiserslautern ***");
	printf("\n**************************************************************\n");




	mnist();
	exit(1);


	Classifier_Function_param test_function(2);

	test_function.function_name="classification_test1";
	test_function.func_ptr = classification_test1;
	test_function.bounds[0]=-4.0;
	test_function.bounds[1]= 4.0;
	test_function.bounds[2]=-6.0;
	test_function.bounds[3]= 6.0;
	test_function.noise_level = 0.0;



	perform_svn_test(test_function,
	        200,
	        5,
	        SVM_RBF);

	exit(1);

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
