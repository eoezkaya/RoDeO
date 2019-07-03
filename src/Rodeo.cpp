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
#include "Rodeo_globals.hpp"
#include "auxilliary_functions.hpp"
#include "read_settings.hpp"
#include "kernel_regression.hpp"
#include "kernel_regression_cuda.h"

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
	arma_rng::set_seed_random();


	settings.read();


	int ret = chdir (settings.cwd.c_str());

	if (ret != 0){

		fprintf(stderr, "Error: cannot change directory! at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);
	}




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



//	perform_NNregression_test(Wingweight,
//			parameter_bounds,
//			"Wingweight" ,
//			400,
//			RANDOM_SAMPLING,
//			10,
//			500);

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

	double *parameter_bounds;
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

	perform_kernel_regression_test_highdim_cuda(empty,
			empty,
			parameter_bounds,
			"Housing",
			400,
			0,
			EXISTING_FILE,
			13);






}
