


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
#include "read_settings.hpp"


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

	Rodeo_settings settings;
	settings.read();

	int ret = chdir (settings.cwd.c_str());

//	double parameter_bounds[4];
//	parameter_bounds[0]=0.0;
//	parameter_bounds[1]=100.0;
//	parameter_bounds[2]=0.0;
//	parameter_bounds[3]=100.0;
//
//	perform_trust_region_GEK_test(Eggholder,
//			Eggholder_adj,
//			parameter_bounds,
//			"Eggholder" ,
//			0,
//			50,
//			RANDOM_SAMPLING,
//			CHOLESKY,
//			2,
//			LINEAR_REGRESSION_OFF,
//			settings.python_dir);
//
//
//	exit(1);


	//	double parameter_bounds[16];
	//	parameter_bounds[0]=0.05;    // [0.05, 0.15]
	//	parameter_bounds[1]=0.15;
	//
	//	parameter_bounds[2]=100.0;  // [100, 50000]
	//	parameter_bounds[3]=50000.0;
	//
	//	parameter_bounds[4]=63500;  // [63500, 115600]
	//	parameter_bounds[5]=115600;
	//
	//	parameter_bounds[6]=990;
	//	parameter_bounds[7]=1110; // [990, 1110]
	//
	//	parameter_bounds[8]=63.1;
	//	parameter_bounds[9]=116; // [[63.1, 116]]
	//
	//	parameter_bounds[10]=700;
	//	parameter_bounds[11]=820; // [700, 820]
	//
	//	parameter_bounds[12]=1120;
	//	parameter_bounds[13]=1680; // [1120, 1680]
	//
	//	parameter_bounds[14]=9855;
	//	parameter_bounds[15]=12045; // [9855, 12045]
	//
	//	perform_trust_region_GEK_test(Borehole,
	//			Borehole_adj,
	//			parameter_bounds,
	//			"Borehole" ,
	//			0,
	//			200,
	//			RANDOM_SAMPLING,
	//			CHOLESKY,
	//			8,
	//			LINEAR_REGRESSION_OFF);

	initial_data_acquisitionGEK(settings.python_dir, 200);

//	su2_optimize(settings.python_dir);


}
