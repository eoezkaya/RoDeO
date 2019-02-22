


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


	su2_statistics_around_solution(500, "su2optimal_stat.dat","su2optimal.csv");

	exit(1);

	OptimizationData optimization_naca0012;

	optimization_naca0012.name = "naca0012";

	optimization_naca0012.size_of_dv=38;

	uvec activity(38);
	vec lower_bound(38,fill::zeros);
	lower_bound.fill(-0.003);
	vec upper_bound(38,fill::zeros)	;
	upper_bound.fill(0.003);
	optimization_naca0012.variable_activity = activity;
	optimization_naca0012.variable_activity.fill(1);
	optimization_naca0012.lower_bound_dv = lower_bound;
	optimization_naca0012.upper_bound_dv = upper_bound;

	optimization_naca0012.max_number_of_samples=300;
	optimization_naca0012.number_of_outputs = 2;
	optimization_naca0012.include_initial_design=true;

	std::string outputs[2]={"Lift","Drag"};

	optimization_naca0012.output_names = outputs;

	uvec gradient_indices(3);
	gradient_indices(0)=1;
	gradient_indices(1)=1;
	gradient_indices(2)=0;

	optimization_naca0012.outputs_with_gradients = gradient_indices;

	vec geometric_constraints(1);
	geometric_constraints(0)= 0.081;

	optimization_naca0012.geometric_constraints = geometric_constraints;

	std::string geo_constraint_names[1]={"Area"};
	optimization_naca0012.geometric_constraints_names = geo_constraint_names;

	optimization_naca0012.number_outputs_with_gradients=2;

	optimization_naca0012.base_config_file_name="config_DEF.cfg";
	optimization_naca0012.history_file_name="naca0012_optimization_history.csv";

	std::string output_file_names[3]={"CL_Kriging.csv","CD_Kriging.csv","Area_Kriging.csv"};

	std::string config_file_names[2]={"inv_NACA0012_adj_lift.cfg","inv_NACA0012_adj_drag.cfg"};

	optimization_naca0012.output_file_names = output_file_names;
	optimization_naca0012.config_file_names = config_file_names;
	optimization_naca0012.number_of_geometric_features = 9;




	su2_robustoptimize_naca0012(optimization_naca0012);

	exit(1);

//	double parameter_bounds[10];
//		parameter_bounds[0]=0.0; parameter_bounds[1]=10.0;
//		parameter_bounds[2]=0.0; parameter_bounds[3]=10.0;
//		parameter_bounds[4]=0.0; parameter_bounds[5]=10.0;
//		parameter_bounds[6]=0.0; parameter_bounds[7]=10.0;
//		parameter_bounds[8]=0.0; parameter_bounds[9]=10.0;
//
//		perform_kernel_regression_test(test_function2KernelReg,
//				test_function2KernelRegAdj,
//				parameter_bounds,
//				"test_function2KernelReg",
//				0,
//				100,
//				RANDOM_SAMPLING,
//				5,
//				settings.python_dir);

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


	double parameter_bounds[20];
	parameter_bounds[0]=150.0; parameter_bounds[1]=200.0;
	parameter_bounds[2]=220.0; parameter_bounds[3]=300.0;
	parameter_bounds[4]=6.0; parameter_bounds[5]=10.0;
	parameter_bounds[6]=-10.0; parameter_bounds[7]=10.0;
	parameter_bounds[8]=16.0; parameter_bounds[9]=45.0;
	parameter_bounds[10]=0.5; parameter_bounds[11]=1.0;
	parameter_bounds[12]=0.08; parameter_bounds[13]=0.018;
	parameter_bounds[14]=2.5; parameter_bounds[15]=6.0;
	parameter_bounds[16]=1700.0; parameter_bounds[17]=2500.0;
	parameter_bounds[18]=0.025; parameter_bounds[19]=0.08;


	perform_NNregression_test(Wingweight,
			parameter_bounds,
			"Wingweight",
			1000,
			RANDOM_SAMPLING,
			10,
			100);


//	perform_kernel_regression_test(Wingweight,
//			WingweightAdj,
//			parameter_bounds,
//			"Wingweight",
//			0,
//			200,
//			RANDOM_SAMPLING,
//			10,
//			settings.python_dir);


//	su2_optimize(settings.python_dir);

	//	initial_data_acquisitionGEK(settings.python_dir, 200);




}
