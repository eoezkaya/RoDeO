#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include "auxilliary_functions.hpp"
#include "kriging_training.hpp"
#include "trust_region_gek.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "su2_optim.hpp"
#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>



using namespace arma;




void plot_airfoil(std::string initial_airfoil_file, std::string deformed_mesh_file, int nelems){


	std::ifstream myfile (deformed_mesh_file.c_str());
	std::string line;

	struct coordinates{
		int number_of_points;
		double *xcor;
		double *ycor;
	};

	double tempf1;
	double tempf2;

	FILE *inp;
	int i;
	int NDIME;
	int NELEM;
	int NPOIN;
	int NMARK;
	int MARKER_ELEMS;
	char temps[500];
	int tempd1;
	int tempd2;



	getline (myfile,line);


	//	cout<<line<<std::endl;

	getline (myfile,line);


	//	cout<<line<<std::endl;

	getline (myfile,line);


	//	cout<<line<<std::endl;







	/*original grid */

	for(i=0;i<nelems;i++){
		myfile  >> tempf1;
		myfile  >> tempf2;

		//		printf("%10.7f %10.7f\n",tempf1,tempf2);

	}

	for(i=0;i<nelems;i++){
		myfile  >> tempd1;
		myfile  >> tempd2;
		//		printf("%d %1d\n",tempd1,tempd2);
	}

	getline (myfile,line);


	//	cout<<line<<std::endl;

	getline (myfile,line);


	//	cout<<line<<std::endl;
	getline (myfile,line);


	//	cout<<line<<std::endl;

	coordinates deformed_airfoil;

	deformed_airfoil.xcor=new double[nelems];
	deformed_airfoil.ycor=new double[nelems];

	// read grid coordinates
	for(i=0;i<nelems;i++){
		myfile  >> tempf1;
		myfile  >> tempf2;
		deformed_airfoil.xcor[i]=tempf1;


		deformed_airfoil.ycor[i]=tempf2;


	}

	FILE *outp=fopen("deformed_airfoil_geometry.dat","w");

	fprintf(outp,"%15.10f %15.10f\n",deformed_airfoil.xcor[nelems-1],deformed_airfoil.ycor[nelems-1]);
	for(i=0;i<nelems;i++){

		fprintf(outp,"%15.10f %15.10f\n",deformed_airfoil.xcor[i],deformed_airfoil.ycor[i]);


	}


	fclose(outp);

	std::string file_name_for_plot = "deformed_airfoil.png";

	std::string python_command = "python -W ignore "+settings.python_dir+"/plot_airfoil.py deformed_airfoil_geometry.dat "+initial_airfoil_file+ " " +
			file_name_for_plot;

#if 0
	printf("python_command = %s\n",python_command.c_str());
#endif


	FILE* in = popen(python_command.c_str(), "r");
	fprintf(in, "\n");


	delete[] deformed_airfoil.xcor;
	delete[] deformed_airfoil.ycor;

}

//void SU2_CFD(std::string input_filename){
//
//	int number_of_design_variables = 38;
//	mat data(1,number_of_design_variables);
//
//	data.load(input_filename,csv_ascii);
//
//#if 1
//	cout.precision(11);
//	cout.setf(ios::fixed);
//
//	data.raw_print(cout, "dv:");
//#endif
//
//	double area,CL,CD,type,area_constraint;
//
//	type=1;
//	area_constraint = 0.0778116;
//
//	vec dv(number_of_design_variables);
//	vec grad(number_of_design_variables);
//
//	for(int i=0; i<number_of_design_variables; i++){
//
//		dv(i) = data(0,i);
//
//	}
//
//
//	//	call_SU2_Adjoint_Solver(
//	//			dv,
//	//			grad,
//	//			CL,
//	//			CD,
//	//			area,
//	//			type,
//	//			area_constraint);
//
//
//}



int is_equal(rowvec r1, rowvec r2){

	int flag = -1;

	int dim1 = r1.size();
	int dim2 = r2.size();

	if(dim1 != dim2){

		printf("Error: dim1 and dim2 are not equal!\n");
		exit(-1);

	}

	for(int i=0; i<dim1;i++) {

		if( fabs(r1(i)- r2(i)) < 10E-6 ) {

			flag = 1;
		}
		else{

			flag = 0;
			break;
		}
	}

	return flag;

}

void check_double_points_data(std::string filename, int nvars){
#if 1
	printf("checking double points for the file %s...\n",filename.c_str());
#endif
	mat data;


	bool ok = data.load(filename, csv_ascii);

	if(ok == false){

		printf("problem with loading: program will terminate\n");
		exit(1);
	}


#if 0
	data.print();
#endif

	int nrows = data.n_rows;
#if 0
	printf("data has initially %d rows with nvars = %d\n",nrows,nvars);
#endif


	mat dataX(nrows, nvars);

	for(int i=0; i<nrows; i++){

		for(int j=0; j<nvars; j++){

			dataX(i,j) = data(i,j);

		}

	}

#if 0
	dataX.print();
#endif

	for(int i=0; i<nrows; i++){

		rowvec ri = dataX.row(i);

		int eqflag = 0;
		int jindex = -1;

		/* check all the rows of the data */

		for(int j=0; j<nrows; j++){

			rowvec rj = dataX.row(j);
			if(i!=j){

				if(is_equal(ri,rj)){

					eqflag = 1;
					jindex = j;
				}

			}

		}

		if(eqflag == 1){

			printf("Error: data point %d and %d are the same!\n",i+1,jindex+1);

			printf("row %d:\n",i);
			dataX.row(i).print();
			printf("row %d:\n",jindex);
			dataX.row(jindex).print();


			exit(-1);

		}


	}



}



//void su2_statistics_around_solution(int number_of_samples, std::string output_file_name,std::string input_file_name){
//
//	FILE *outp = fopen(output_file_name.c_str(),"w");
//
//	int number_of_design_variables  = 38;
//
//
//	/* standart deviation for the perturbations */
//	const double gaussian_noise_level = 0.0001;
//	std::default_random_engine generator;
//	std::normal_distribution<double> distribution(0.0,1.0);
//
//	double CL,CD,area;
//
//	vec cd_vec(number_of_samples);
//	vec cl_vec(number_of_samples);
//	vec area_vec(number_of_samples);
//
//
//	rowvec dv_input= zeros<rowvec>(number_of_design_variables);      /* design vector */
//	vec dv         = zeros(number_of_design_variables);
//	vec dv_save         = zeros(number_of_design_variables);
//
//
//	mat data;
//	data.load(input_file_name.c_str(), csv_ascii);
//	data.print();
//
//	dv_input = data.row(0);
//
//	for(unsigned int i=0; i<dv.size();i++) dv(i)=dv_input(i);
//
//	dv.print();
//
//	dv_save = dv;
//
//
//	double cl_worst_case = LARGE;
//	double area_worst_case = LARGE;
//
//	for(unsigned int i=0;i<number_of_samples;i++){
//
//
//
//		/* generate a random design*/
//		for(int j= 0; j< number_of_design_variables; j++){
//			double number = distribution(generator);
//			if(number > 5 ) number = 5.0;
//			if(number < -5 ) number = -5.0;
//			double perturbation  = gaussian_noise_level* number;
//#if 0
//			printf("generated random number = %10.7f\n",number);
//			printf("perturbation            = %10.7f\n",perturbation);
//#endif
//			if(i!=0) dv(j)= dv(j)+perturbation ;
//
//
//		}
//
//		call_SU2_CFD_Solver(dv,CL,CD,area);
//
//
//		cd_vec(i) = CD;
//		cl_vec(i) = CL;
//		area_vec(i)=area;
//
//
//
//
//		if(CL < cl_worst_case) cl_worst_case = CL;
//		if(area < area_worst_case) area_worst_case = area;
//
//
//		printf("%10.7f %10.7f %10.7f %10.7f %10.7f\n",CD,CL,area,cl_worst_case,area_worst_case);
//		fprintf(outp,"%10.7f %10.7f %10.7f %10.7f %10.7f\n",CD,CL,area,cl_worst_case,area_worst_case);
//
//		dv = dv_save;
//
//
//	}
//
//	fclose(outp);
//
//	double cdmean = mean(cd_vec);
//	double clmean = mean(cl_vec);
//	double areamean = mean(area_vec);
//
//	double cdstd = stddev(cd_vec);
//	double clstd = stddev(cl_vec);
//	double areastd = stddev(area_vec);
//
//	printf("Statistics = \n");
//	printf("CD:  mean = %10.7f  std = %10.7f\n",cdmean,cdstd);
//	printf("CL:  mean = %10.7f  std = %10.7f\n",clmean,clstd);
//	printf("S :  mean = %10.7f  std = %10.7f\n",areamean,areastd);
//
//
//
//}
//
//
//void su2_optimize(std::string python_dir){
//
//	const int max_number_of_function_evaluations = 300;
//	int max_number_of_function_calculations_training = 10000;
//	int number_of_design_variables  = 24;
//	const int number_of_function_evals_inner_iter = 5;
//
//	vec best_EI_values(number_of_function_evals_inner_iter);
//	best_EI_values.fill(-LARGE);
//	mat dv_best_EI(number_of_function_evals_inner_iter,number_of_design_variables);
//	dv_best_EI.fill(0.0);
//
//	int worst_EI_array_indx = 0;
//
//
//	/* box constraints for the design variables */
//	double upper_bound_dv =  0.1;
//	double lower_bound_dv = -0.1;
//
//	const double lift_penalty_param = LARGE;
//	const double volume_penalty_param = LARGE;
//
//
//	/* constraints for cl*/
//	double CL_constraint = 0.285;
//	double Volume_constraint = 0.027;
//
//
//	/* geometric constraints */
//	double max_thickness_sec1= 0.06;
//	double max_thickness_sec2= 0.06;
//	double max_thickness_sec3= 0.05;
//	double max_thickness_sec4= 0.05;
//	double max_thickness_sec5= 0.04;
//
//
//
//
//
//
//	vec geometric_constraints(6);
//	geometric_constraints(0) = Volume_constraint;
//	geometric_constraints(1) = max_thickness_sec1;
//	geometric_constraints(2) = max_thickness_sec2;
//	geometric_constraints(3) = max_thickness_sec3;
//	geometric_constraints(4) = max_thickness_sec4;
//	geometric_constraints(5) = max_thickness_sec5;
//
//
//	vec objectives(8);
//
//	/* regularization parameter for Kriging */
//	double reg_param = 10E-7;
//
//
//	int number_of_EI_iter;
//	int number_of_EI_iter_local = 100000;
//	int number_of_EI_iter_global = 1000000;
//
//	int number_of_function_evals = 0;
//
//	/* copy training data from the samples folder */
//	system("cp ./samples/optimization_history.csv ./");
//	system("cp ./samples/CL_Kriging.csv ./");
//	system("cp ./samples/CD_Kriging.csv ./");
//	system("cp ./samples/Volume_Kriging.csv ./");
//
//
//
//	/* filenames for the Kriging input data */
//	std::string cl_kriging_input_file = "CL_Kriging.csv";
//	std::string cd_kriging_input_file = "CD_Kriging.csv";
//	std::string volume_kriging_input_file = "Volume_Kriging.csv";
//
//
//	/* file name for the optimization history */
//	std::string all_data_file = "optimization_history.csv";
//	check_double_points_data(all_data_file,number_of_design_variables);
//
//
//	/* check double points in the data */
//	check_double_points_data(volume_kriging_input_file,number_of_design_variables);
//	check_double_points_data(cd_kriging_input_file,number_of_design_variables);
//	check_double_points_data(cl_kriging_input_file,number_of_design_variables);
//
//
//
//
//
//	/* file names for the Kriging hyperparameters (for CD, CL and constraint) */
//	std::string cl_kriging_hyperparameters_file = "CL_Kriging_Hyperparameters.csv";
//	std::string cd_kriging_hyperparameters_file = "CD_Kriging_Hyperparameters.csv";
//	std::string volume_kriging_hyperparameters_file = "Volume_Kriging_Hyperparameters.csv";
//
//
//	/* obtain statistics from the existing data */
//	double EI_dist_tol = 0.0;
//
//	int max_iter_stat = 10000;
//
//	/* load samples from file*/
//	mat stat_data;
//	stat_data.load(all_data_file.c_str(), csv_ascii);
//
//
//
//	/* normalized input variables */
//	mat X_stat(stat_data.n_rows,number_of_design_variables);
//
//
//	/* set the input data matrix X */
//	for(int i=0; i<number_of_design_variables;i++){
//
//		X_stat.col(i) = stat_data.col(i);
//
//	}
//
//#if 0
//	printf("X_stat:\n");
//	X_stat.print();
//#endif
//
//	int avg_distance_stat = 0.0;
//	for(int i=0; i<max_iter_stat; i++){
//
//		rowvec dv(number_of_design_variables);
//
//		/* Generate a random design vector */
//		for(int k=0; k<number_of_design_variables; k++){
//
//			dv(k)= RandomDouble(lower_bound_dv, upper_bound_dv);
//
//		}
//
//
//		int indx = -1;
//		double min_dist = LARGE;
//		findKNeighbours(X_stat, dv, 1, &min_dist,&indx, 1);
//
//		rowvec nearest = X_stat.row(indx);
//
//#if 0
//		printf("dv:\n");
//		dv.print();
//		printf("the nearest point is:\n");
//		nearest.print();
//		printf("minimum L1 distance = %10.7e\n",min_dist);
//
//#endif
//
//		avg_distance_stat+= min_dist;
//	}
//
//	avg_distance_stat = avg_distance_stat/max_iter_stat;
//
//	/* set the distance tolerance */
//	EI_dist_tol = avg_distance_stat / 10.0;
//
//
//
//	int it_count_outer_loop = 0;
//	int number_of_initial_samples = 0;
//	/* start optimization loop */
//
//	while(1){
//
//		/* load samples from file*/
//		mat optimization_data;
//		mat cl_data;
//		mat cd_data;
//		mat volume_data;
//
//
//		optimization_data.load(all_data_file.c_str(), csv_ascii);
//		cl_data.load(cl_kriging_input_file.c_str(),csv_ascii );
//		cd_data.load(cd_kriging_input_file.c_str(),csv_ascii );
//		volume_data.load(volume_kriging_input_file.c_str(),csv_ascii );
//
//#if 0
//		printf("optimization data:\n");
//		optimization_data.print();
//
//#endif
//
//
//
//
//		/*dimension of R */
//		int number_of_data_points = optimization_data.n_rows;
//
//
//		int number_of_data_points_CL = cl_data.n_rows;
//		int number_of_data_points_CD = cd_data.n_rows;
//		int number_of_data_points_volume = volume_data.n_rows;
//
//		/* normalized input variables */
//		mat X(number_of_data_points,number_of_design_variables);
//		mat X_CL(number_of_data_points_CL,number_of_design_variables);
//		mat X_CD(number_of_data_points_CD,number_of_design_variables);
//		mat X_volume(number_of_data_points_volume,number_of_design_variables);
//
//		if(it_count_outer_loop == 0) {
//
//			number_of_initial_samples = number_of_data_points;
//		}
//
//
//		/* set the input data matrix X */
//		for(int i=0; i<number_of_design_variables;i++){
//
//			X.col(i) = optimization_data.col(i);
//			X_CL.col(i) =   cl_data.col(i);
//			X_CD.col(i) =   cd_data.col(i);
//			X_volume.col(i) = volume_data.col(i);
//
//		}
//
//#if 0
//		printf("X:\n");
//		X.print();
//#endif
//
//
//		/* find minimum and maximum of the columns of X */
//
//		vec x_max(number_of_design_variables);
//		x_max.fill(0.0);
//
//		vec x_min(number_of_design_variables);
//		x_min.fill(0.0);
//
//		for (int i = 0; i < number_of_design_variables; i++) {
//			x_max(i) = X.col(i).max();
//			x_min(i) = X.col(i).min();
//
//		}
//#if 0
//		printf("maximum = \n");
//		x_max.print();
//
//		printf("minimum = \n");
//		x_min.print();
//#endif
//		/* normalize data matrix */
//
//		for (unsigned int i = 0; i < X.n_rows; i++) {
//
//			for (int j = 0; j < number_of_design_variables; j++) {
//
//				X(i, j) = (1.0/number_of_design_variables)*(X(i, j) - x_min(j)) / (x_max(j) - x_min(j));
//
//			}
//		}
//		for (unsigned int i = 0; i < X_CL.n_rows; i++) {
//
//			for (int j = 0; j < number_of_design_variables; j++) {
//
//				X_CL(i, j) = (1.0/number_of_design_variables)*(X_CL(i, j) - x_min(j)) / (x_max(j) - x_min(j));
//
//			}
//		}
//		for (unsigned int i = 0; i < X_CD.n_rows; i++) {
//
//			for (int j = 0; j < number_of_design_variables; j++) {
//
//				X_CD(i, j) = (1.0/number_of_design_variables)*(X_CD(i, j) - x_min(j)) / (x_max(j) - x_min(j));
//
//			}
//		}
//
//		for (unsigned int i = 0; i < X_volume.n_rows; i++) {
//
//			for (int j = 0; j < number_of_design_variables; j++) {
//
//				X_volume(i, j) = (1.0/number_of_design_variables)*(X_volume(i, j) - x_min(j)) / (x_max(j) - x_min(j));
//
//			}
//		}
//
//
//		/* assign ys vectors for CL,CD and area */
//		vec ys_CL   =   optimization_data.col(number_of_design_variables);
//		vec ys_CD   =   optimization_data.col(number_of_design_variables+1);
//		vec ys_volume =   optimization_data.col(number_of_design_variables+2);
//
//#if 1
//		printf("ys_CL:\n");
//		trans(ys_CL).print();
//		printf("ys_CD:\n");
//		trans(ys_CD).print();
//		printf("ys_volume:\n");
//		trans(ys_volume).print();
//#endif
//
//
//		/* now find the best sample point */
//
//		double sample_min = LARGE;
//		double best_sample_CD = 0.0;
//		double best_sample_CL= 0.0;
//		double best_sample_volume = 0.0;
//		int sample_min_indx = -1;
//
//		for(unsigned int k=0; k<ys_CD.size(); k++){
//
//			double obj_fun = ys_CD(k);
//
//			/* add penalties for lift and area */
//			if(ys_CL(k) < CL_constraint){
//
//				obj_fun += lift_penalty_param*(CL_constraint-ys_CL(k));
//			}
//
//			if(ys_volume(k) < Volume_constraint){
//
//				obj_fun += volume_penalty_param*(Volume_constraint-ys_volume(k));
//			}
//
//			if(obj_fun < sample_min){
//
//				sample_min = obj_fun;
//				sample_min_indx = k;
//				best_sample_CD = ys_CD(k);
//				best_sample_CL = ys_CL(k);
//				best_sample_volume = ys_volume(k);
//
//			}
//
//		}
//#if 1
//		printf("Best sample in the data set is the %dth entry :\n",sample_min_indx);
//		optimization_data.row(sample_min_indx).print();
//		printf("objective function value = %10.7e\n",sample_min);
//		printf("CD   = %10.7e\n",best_sample_CD);
//		printf("CL   = %10.7e\n",best_sample_CL);
//		printf("Volume = %10.7e\n",best_sample_volume);
//
//
//#endif
//
//
//
//		/* correlation matrices for cd,cl and area */
//
//		mat R_CL(number_of_data_points_CL,number_of_data_points_CL);
//		mat R_CD(number_of_data_points_CD,number_of_data_points_CD);
//		mat R_volume(number_of_data_points_volume,number_of_data_points_volume);
//
//		/* lower and upper diagonal matrices for Cholesky decomposition */
//
//		mat U_CL(number_of_data_points_CL,number_of_data_points_CL);
//		mat U_CD(number_of_data_points_CD,number_of_data_points_CD);
//		mat U_volume(number_of_data_points_volume,number_of_data_points_volume);
//
//
//		mat L_CL(number_of_data_points_CL,number_of_data_points_CL);
//		mat L_CD(number_of_data_points_CD,number_of_data_points_CD);
//		mat L_volume(number_of_data_points_volume,number_of_data_points_volume);
//
//		/* vector of ones */
//		vec I_CL = ones(number_of_data_points_CL);
//		vec I_CD = ones(number_of_data_points_CD);
//		vec I_volume = ones(number_of_data_points_volume);
//
//
//		/* for the Kriging, update ys vectors */
//		ys_CL   =   cl_data.col(number_of_design_variables);
//		ys_CD   =   cd_data.col(number_of_design_variables);
//		ys_volume =   volume_data.col(number_of_design_variables);
//
//#if 0
//		cl_kriging_data.print();
//#endif
//
//
//#if 1
//		printf("number of samples (all data)  = %d\n",number_of_data_points);
//		printf("number of samples (CL data)   = %d\n",number_of_data_points_CL);
//		printf("number of samples (CD data)   = %d\n",number_of_data_points_CD);
//		printf("number of samples (volume data) = %d\n",number_of_data_points_volume);
//#endif
//
//		/* visualize the CL&CD of the samples */
//
//		std::string file_name_for_plot = "samples.png";
//
//		std::string python_command = "python -W ignore "+python_dir+"/plot_cdcl.py "+ all_data_file+ " "+
//				std::to_string(number_of_initial_samples)+ " " +
//				file_name_for_plot;
//
//		FILE* in = popen(python_command.c_str(), "r");
//		fprintf(in, "\n");
//
//
//
//
//		/* visualize the CL&CD of the samples */
//
//		mat kriging_weights_CD;
//		mat regression_weights_CD;
//		mat R_inv_ys_min_beta_CD_tr;
//		double r_CD;
//
//		vec beta0_CD_tr(number_of_design_variables+1);
//
//
//		mat kriging_weights_CL;
//		mat regression_weights_CL;
//		mat R_inv_ys_min_beta_CL_tr;
//		double r_CL;
//
//		vec beta0_CL_tr(number_of_design_variables+1);
//
//
//		vec kriging_weights_volume;
//		vec regression_weights_volume;
//
//
//
//		/* train response surface for CL */
//
//		r_CL = 4.395886;
//		train_TRGEK_response_surface(cl_kriging_input_file,
//				cl_kriging_hyperparameters_file,
//				LINEAR_REGRESSION_ON,
//				regression_weights_CL,
//				kriging_weights_CL,
//				R_inv_ys_min_beta_CL_tr,
//				r_CL,
//				beta0_CL_tr,
//				max_number_of_function_calculations_training,
//				number_of_design_variables,0);
//#if 1
//		printf("kriging weights (CL):\n");
//		trans(kriging_weights_CL).print();
//		printf("regression weights (CL):\n");
//		trans(regression_weights_CL).print();
//		printf("r (CL) = %10.7e\n", r_CL);
//#endif
//
//		/* train response surface for CD */
//
//		r_CD =  34.0;
//		train_TRGEK_response_surface(cd_kriging_input_file,
//				cd_kriging_hyperparameters_file,
//				LINEAR_REGRESSION_ON,
//				regression_weights_CD,
//				kriging_weights_CD,
//				R_inv_ys_min_beta_CD_tr,
//				r_CD,
//				beta0_CD_tr,
//				max_number_of_function_calculations_training,
//				number_of_design_variables,0);
//#if 1
//		printf("kriging weights (CD):\n");
//		trans(kriging_weights_CD).print();
//		printf("regression weights (CD):\n");
//		trans(regression_weights_CD).print();
//		printf("r (CD) = %10.7e\n", r_CD);
//#endif
//		/* train response surface for area */
//
//
//		train_kriging_response_surface(volume_kriging_input_file,
//				volume_kriging_hyperparameters_file,
//				LINEAR_REGRESSION_ON,
//				regression_weights_volume,
//				kriging_weights_volume,
//				reg_param,
//				max_number_of_function_calculations_training,
//				CSV_ASCII);
//
//#if 1
//		printf("kriging weights (volume):\n");
//		trans(kriging_weights_volume).print();
//		printf("regression weights (volume):\n");
//		trans(regression_weights_volume).print();
//#endif
//
//
//
//
//
//
//		/*update y vectors according to linear regression*/
//
//		mat augmented_X_CL(number_of_data_points_CL, number_of_design_variables + 1);
//		mat augmented_X_CD(number_of_data_points_CD, number_of_design_variables + 1);
//		mat augmented_X_volume(number_of_data_points_volume, number_of_design_variables + 1);
//
//		for (int i = 0; i < number_of_data_points_CL; i++) {
//
//			for (int j = 0; j <= number_of_design_variables; j++) {
//
//				if (j == 0){
//
//					augmented_X_CL(i, j) = 1.0;
//
//				}
//				else{
//
//					augmented_X_CL(i, j) = X_CL(i, j - 1);
//
//
//				}
//			}
//		}
//
//		for (int i = 0; i < number_of_data_points_CD; i++) {
//
//			for (int j = 0; j <= number_of_design_variables; j++) {
//
//				if (j == 0){
//
//					augmented_X_CD(i, j) = 1.0;
//
//				}
//				else{
//
//					augmented_X_CD(i, j) = X_CD(i, j - 1);
//
//
//				}
//			}
//		}
//
//		for (int i = 0; i < number_of_data_points_volume; i++) {
//
//			for (int j = 0; j <= number_of_design_variables; j++) {
//
//				if (j == 0){
//
//					augmented_X_volume(i, j) = 1.0;
//
//				}
//				else{
//
//					augmented_X_volume(i, j) = X_volume(i, j - 1);
//
//
//				}
//			}
//		}
//
//		vec ys_reg_cl = augmented_X_CL * regression_weights_CL;
//		vec ys_reg_cd = augmented_X_CD * regression_weights_CD;
//		vec ys_reg_volume = augmented_X_volume * regression_weights_volume;
//
//
//		ys_CL = ys_CL - ys_reg_cl;
//		ys_CD= ys_CD - ys_reg_cd;
//		ys_volume = ys_volume - ys_reg_volume;
//
//#if 0
//		printf("ys_CL:\n");
//		trans(ys_CL).print();
//		printf("ys_CD:\n");
//		trans(ys_CD).print();
//		printf("ys_volume:\n");
//		trans(ys_volume).print();
//#endif
//
//
//		vec theta_CL = kriging_weights_CL.col(0).head(number_of_design_variables);
//		vec gamma_CL = kriging_weights_CL.col(0).tail(number_of_design_variables);
//
//		vec theta_CD = kriging_weights_CD.col(0).head(number_of_design_variables);
//		vec gamma_CD = kriging_weights_CD.col(0).tail(number_of_design_variables);
//
//
//		vec theta_volume = kriging_weights_volume.head(number_of_design_variables);
//		vec gamma_volume = kriging_weights_volume.tail(number_of_design_variables);
//
//
//#if 0
//		printf("theta CL:\n");
//		trans(theta_CL).print();
//
//		printf("gamma CL:\n");
//		trans(gamma_CL).print();
//
//		printf("theta CD:\n");
//		trans(theta_CD).print();
//
//		printf("gamma CD:\n");
//		trans(gamma_CD).print();
//
//		printf("theta volume:\n");
//		trans(theta_volume).print();
//
//		printf("gamma volume:\n");
//		trans(gamma_volume).print();
//#endif
//
//
//		compute_R_matrix(theta_CL,
//				gamma_CL,
//				reg_param,
//				R_CL,
//				X_CL);
//#if 0
//		printf("R_CL:\n");
//		R_CL.print();
//#endif
//
//
//		/* Cholesky decomposition R = LDL^T */
//
//		int cholesky_return_CL = chol(U_CL, R_CL);
//
//		if (cholesky_return_CL == 0) {
//			printf("Error: Ill conditioned correlation matrix for CL, Cholesky decomposition failed...\n");
//			exit(-1);
//		}
//
//		L_CL = trans(U_CL);
//
//		vec R_inv_ys_CL(number_of_data_points_CL);
//		vec R_inv_I_CL(number_of_data_points_CL);
//
//		solve_linear_system_by_Cholesky(U_CL, L_CL, R_inv_ys_CL, ys_CL); /* solve R x = ys */
//		solve_linear_system_by_Cholesky(U_CL, L_CL, R_inv_I_CL, I_CL);      /* solve R x = I */
//
//
//		double	beta0_CL = (1.0/dot(I_CL,R_inv_I_CL)) * (dot(I_CL,R_inv_ys_CL));
//
//
//		vec ys_min_betaI_CL = ys_CL-beta0_CL*I_CL;
//
//		vec R_inv_ys_min_beta_CL(number_of_data_points_CL);
//
//
//
//		/* solve R x = ys-beta0*I */
//		solve_linear_system_by_Cholesky(U_CL, L_CL, R_inv_ys_min_beta_CL, ys_min_betaI_CL);
//
//
//		compute_R_matrix(theta_CD,
//				gamma_CD,
//				reg_param,
//				R_CD,
//				X_CD);
//#if 0
//		printf("R_CD:\n");
//		R_CD.print();
//#endif
//
//		int cholesky_return_CD = chol(U_CD, R_CD);
//
//		if (cholesky_return_CD == 0) {
//			printf("Error: Ill conditioned correlation matrix, Cholesky decomposition failed...\n");
//			exit(-1);
//		}
//
//		L_CD = trans(U_CD);
//
//		vec R_inv_ys_CD(number_of_data_points_CD);
//		vec R_inv_I_CD(number_of_data_points_CD);
//
//
//
//		solve_linear_system_by_Cholesky(U_CD, L_CD, R_inv_ys_CD, ys_CD); /* solve R x = ys */
//		solve_linear_system_by_Cholesky(U_CD, L_CD, R_inv_I_CD, I_CD);   /* solve R x = I */
//
//
//
//
//		double beta0_CD = (1.0/dot(I_CD,R_inv_I_CD)) * (dot(I_CD,R_inv_ys_CD));
//
//
//		vec ys_min_betaI_CD = ys_CD-beta0_CD*I_CD;
//
//		vec R_inv_ys_min_beta_CD(number_of_data_points_CD);
//
//
//
//		/* solve R x = ys-beta0*I */
//		solve_linear_system_by_Cholesky(U_CD, L_CD, R_inv_ys_min_beta_CD, ys_min_betaI_CD);
//
//
//		double ssqr_CD = (1.0 / number_of_data_points_CD) * dot(ys_min_betaI_CD, R_inv_ys_min_beta_CD);
//
//#if 0
//		printf("computing R for volume\n");
//#endif
//		compute_R_matrix(theta_volume,
//				gamma_volume,
//				reg_param,
//				R_volume,
//				X_volume);
//
//
//
//		int cholesky_return_volume = chol(U_volume, R_volume);
//
//		if (cholesky_return_volume == 0) {
//			printf("Error: Ill conditioned correlation matrix for V, Cholesky decomposition failed...\n");
//			exit(-1);
//		}
//
//		L_volume = trans(U_volume);
//
//		vec R_inv_ys_volume(number_of_data_points_volume);
//		vec R_inv_I_volume(number_of_data_points_volume);
//
//
//
//		solve_linear_system_by_Cholesky(U_volume, L_volume, R_inv_ys_volume, ys_volume); /* solve R x = ys */
//		solve_linear_system_by_Cholesky(U_volume, L_volume, R_inv_I_volume, I_volume);     /* solve R x = I */
//
//		double beta0_volume = (1.0/dot(I_volume,R_inv_I_volume)) * (dot(I_volume,R_inv_ys_volume));
//
//
//		vec ys_min_betaI_volume = ys_volume-beta0_volume*I_volume;
//
//		vec R_inv_ys_min_beta_volume(number_of_data_points_volume);
//
//
//
//		/* solve R x = ys-beta0*I */
//		solve_linear_system_by_Cholesky(U_volume, L_volume, R_inv_ys_min_beta_volume, ys_min_betaI_volume);
//
//
//#if 0
//		printf("beta0 for CL : %15.10f\n",beta0_CL);
//		printf("beta0 for CD : %15.10f\n",beta0_CD);
//		printf("beta0 for Volume : %15.10f\n",beta0_volume);
//#endif
//
//
//		best_EI_values.fill(-LARGE);
//		dv_best_EI.fill(0.0);
//		worst_EI_array_indx = 0;
//
//
//		number_of_EI_iter = number_of_EI_iter_local+number_of_EI_iter_global;
//
//		rowvec dp_best = optimization_data.row(sample_min_indx);
//
//		for(int iter_EI=0; iter_EI<number_of_EI_iter; iter_EI++){
//
//
//			rowvec dv(number_of_design_variables);
//			rowvec dvnorm(number_of_design_variables);
//
//
//			if(iter_EI < number_of_EI_iter_local){
//
//
//				for(int k=0; k<number_of_design_variables; k++){
//
//
//					double perturbation = RandomDouble(lower_bound_dv/100, upper_bound_dv/100);
//
//					dv(k)= dp_best(k)+perturbation;
//					dvnorm(k) = (1.0/number_of_design_variables)*(dv(k)-x_min(k)) / (x_max(k)-x_min(k));
//				}
//
//#if 0
//				printf("data point entry:\n");
//				dp.print();
//				printf("perturbed design:\n");
//				dv.print();
//#endif
//
//			}
//
//			else{
//
//				/* Generate a random design vector */
//				for(int k=0; k<number_of_design_variables; k++){
//
//					dv(k)= RandomDouble(lower_bound_dv, upper_bound_dv);
//					dvnorm(k) = (1.0/number_of_design_variables)*(dv(k)-x_min(k)) / (x_max(k)-x_min(k));
//				}
//
//			}
//
//
//
//#if 0
//			printf("dv:\n");
//			dv.print();
//#endif
//
//			/* Kriging estimate of the area */
//			double volume_tilde = calculate_f_tilde(dvnorm,
//					X_volume,
//					beta0_volume,
//					regression_weights_volume,
//					R_inv_ys_min_beta_volume,
//					kriging_weights_volume);
//
//			/* Kriging estimate of the CL */
//			double CL_tilde = calculate_f_tilde(dvnorm,
//					X_CL,
//					beta0_CL,
//					regression_weights_CL,
//					R_inv_ys_min_beta_CL,
//					kriging_weights_CL);
//
//
//			double CD_tilde = 0.0;
//			double CD_tilde_ssqr = 0.0;
//
//			calculate_f_tilde_and_ssqr(
//					dvnorm,
//					X_CD,
//					beta0_CD,
//					ssqr_CD,
//					regression_weights_CD,
//					R_inv_ys_min_beta_CD,
//					R_inv_I_CD,
//					I_CD,
//					kriging_weights_CD,
//					U_CD,
//					L_CD,
//					&CD_tilde,
//					&CD_tilde_ssqr);
//
//
//			double	standart_error = sqrt(CD_tilde_ssqr);
//
//
//			double min_dist=0;
//			int indx = -1;
//
//			/* find the closest data point */
//
//			findKNeighbours(X_CD,
//					dvnorm,
//					1,
//					&min_dist,
//					&indx,
//					1);
//
//
//
//			rowvec sp =  X_CD.row(indx);
//			rowvec sp_not_normalized(number_of_design_variables);
//			sp_not_normalized.fill(0.0);
//
//			for(int j=0; j<number_of_design_variables;j++) {
//
//				sp_not_normalized(j) = number_of_design_variables*sp(j)* (x_max(j) - x_min(j))+x_min(j);
//			}
//#if 0
//			printf("closest point           point:\n");
//
//			for(int k=0; k<number_of_design_variables; k++){
//
//				printf("%10.7e  %10.7e\n",dv(k),sp_not_normalized(k));
//
//			}
//
//			printf("grad data entry (CD):\n");
//			cd_kriging_data.row(indx).print();
//			printf("grad data entry (CL):\n");
//			cd_kriging_data.row(indx).print();
//
//#endif
//			rowvec xdiff = dvnorm-sp;
//			double distance = L1norm(xdiff, number_of_design_variables);
//#if 0
//			printf("distance = %10.7e\n",distance);
//#endif
//
//			/* get the functional value from the data */
//			double func_val_CD = cd_data(indx,number_of_design_variables);
//			double func_val_CL = cl_data(indx,number_of_design_variables);
//#if 0
//			printf("CD value at the nearest point = %10.7e\n",func_val_CD);
//			printf("CL value at the nearest point = %10.7e\n",func_val_CL);
//#endif
//
//
//			vec grad_CD(number_of_design_variables);
//
//			for(int j=0; j<number_of_design_variables; j++) {
//
//				grad_CD(j)= cd_data(indx,j+number_of_design_variables+1);
//			}
//
//			vec grad_CL(number_of_design_variables);
//
//			for(int j=0; j<number_of_design_variables; j++) {
//
//				grad_CL(j)= cl_data(indx,j+number_of_design_variables+1);
//			}
//
//#if 0
//			printf("gradient vector (CD):\n");
//			trans(grad_CD).print();
//			printf("gradient vector (CL):\n");
//			trans(grad_CL).print();
//#endif
//
//
//
//
//
//			double normgrad_CD= L1norm(grad_CD, number_of_design_variables);
//			double normgrad_CL= L1norm(grad_CL, number_of_design_variables);
//
//#if 0
//			printf("norm of the gradient vector (CD) = %10.7e\n",normgrad_CD);
//			printf("norm of the gradient vector (CL) = %10.7e\n",normgrad_CL);
//#endif
//
//
//			double factor_CD = exp(-r_CD*distance*normgrad_CD);
//			double factor_CL = exp(-r_CL*distance*normgrad_CL);
//
//			double fval_linmodel_CD= func_val_CD + dot((dv-sp_not_normalized),grad_CD);
//			double fval_linmodel_CL= func_val_CL + dot((dv-sp_not_normalized),grad_CL);
//
//			double fval_CD = factor_CD*fval_linmodel_CD + (1.0-factor_CD)*CD_tilde;
//			double fval_CL = factor_CL*fval_linmodel_CL + (1.0-factor_CL)*CL_tilde;
//
//			double	standart_error_CD = sqrt(CD_tilde_ssqr)	;
//
//			double EI_CD = 0.0;
//
//			double obj_fun = fval_CD;
//
//			/* add penalties for lift and area */
//			if(fval_CL < CL_constraint){
//
//				obj_fun += lift_penalty_param*(CL_constraint-fval_CL);
//			}
//
//			if(volume_tilde < Volume_constraint){
//
//				obj_fun += volume_penalty_param*(Volume_constraint-volume_tilde);
//			}
//
//
//			if(standart_error_CD!=0.0){
//
//				double	EIfac = (sample_min - obj_fun)/standart_error_CD;
//
//				/* calculate the Expected Improvement value */
//				EI_CD = (sample_min - obj_fun)*cdf(EIfac,0.0,1.0)+standart_error*pdf(EIfac,0.0,1.0);
//			}
//			else{
//
//				EI_CD =0.0;
//
//			}
//
//
//			/* find the closest point to dv in the current best EI designs */
//
//			double min_dist_EI = LARGE;
//			int    min_EI_indx = -1;
//			findKNeighbours(dv_best_EI, dv, 1, &min_dist_EI, &min_EI_indx , 1);
//
//			if(EI_CD > best_EI_values(worst_EI_array_indx && min_dist_EI > EI_dist_tol)){
//
//
//				best_EI_values(worst_EI_array_indx) = EI_CD;
//
//				for(int k=0;k<number_of_design_variables;k++) {
//
//					dv_best_EI(worst_EI_array_indx,k) = dv(k);
//				}
//
//
//				double worst_EI_value = 0.0;
//				find_min_with_index(best_EI_values,
//						number_of_function_evals_inner_iter,
//						&worst_EI_value ,
//						&worst_EI_array_indx);
//#if 1
//				printf("found a better EI value with EI = %10.7e\n",EI_CD);
//				printf("best EI values:\n");
//				trans(best_EI_values).print();
//				printf("worst EI index is now = %d\n",worst_EI_array_indx);
//				printf("volume tilde = %10.7e:\n",volume_tilde);
//				printf("CL tilde = %10.7e:\n",CL_tilde);
//				printf("CD tilde = %10.7e:\n",CD_tilde);
//				printf("factor_CD = %10.7e\n",factor_CD);
//				printf("fval_linmodel_CD = %10.7e\n",fval_linmodel_CD);
//				printf("fval_CD = %10.7e\n",fval_CD);
//				printf("factor_CL = %10.7e\n",factor_CL);
//				printf("fval_linmodel_CL = %10.7e\n",fval_linmodel_CL);
//				printf("fval_CL = %10.7e\n",fval_CL);
//				printf("EI value = %10.7e\n",EI_CD);
//#endif
//
//			} /* end of if */
//
//
//
//
//
//
//		} /* end of the EI loop */
//
//
//#if 1
//		printf("the most promising designs have been found...\n");
//#endif
//
//		for(int k=0; k<number_of_function_evals_inner_iter; k++ ){
//
//			double CD_exact=0.0;
//			double CL_exact=0.0;
//			double volume_exact=0.0;
//
//			vec gradient_cd(number_of_design_variables);
//			vec gradient_cl(number_of_design_variables);
//
//			vec dv = trans(dv_best_EI.row(k));
//
//#if 1
//			printf("design vector:\n");
//			trans(dv).print();
//#endif
//
//			//			printf("calling adjoint solver for lift...\n");
//			//			call_SU2_Adjoint_Solver(dv,gradient_cl,objectives,2,geometric_constraints);
//			//
//			//			printf("calling adjoint solver for drag...\n");
//			//			call_SU2_Adjoint_Solver(dv,gradient_cd,objectives,1,geometric_constraints);
//
//
//
//			CL_exact = objectives(0);
//			CD_exact = objectives(1);
//			volume_exact = objectives(2);
//
//#if 0
//			printf("Simulation results:\n");
//			printf("volume = %10.7f\n",volume_exact);
//			printf("cl = %10.7f\n",CL_exact);
//			printf("cd = %10.7f\n",CD_exact);
//			printf("gradient (drag):\n");
//			trans(gradient_cd).print();
//			printf("gradient (lift):\n");
//			trans(gradient_cl).print();
//#endif
//
//
//
//			/* insert a row to the data matrix*/
//
//			if(CD_exact > 0  && CL_exact > 0){
//
//				number_of_function_evals++;
//
//				optimization_data.insert_rows( number_of_data_points, 1 );
//				for(int i=0;i<number_of_design_variables;i++){
//
//					optimization_data(number_of_data_points,i) = dv(i);
//				}
//				optimization_data(number_of_data_points,number_of_design_variables)   = CL_exact;
//				optimization_data(number_of_data_points,number_of_design_variables+1) = CD_exact;
//				optimization_data(number_of_data_points,number_of_design_variables+2) = volume_exact;
//
//
//				/* insert a row to the cl kriging data matrix*/
//				cl_data.insert_rows( number_of_data_points_CL, 1 );
//
//				for(int i=0;i<number_of_design_variables;i++){
//
//					cl_data(number_of_data_points_CL,i) = dv(i);
//				}
//				cl_data(number_of_data_points_CL,number_of_design_variables) = CL_exact;
//
//				for(int i=0;i<number_of_design_variables;i++){
//
//					cl_data(number_of_data_points_CL,number_of_design_variables+1+i) = gradient_cl(i);
//				}
//
//
//
//
//				/* insert a row to the cd kriging data matrix*/
//				cd_data.insert_rows( number_of_data_points_CD, 1 );
//
//				for(int i=0;i<number_of_design_variables;i++){
//
//					cd_data(number_of_data_points_CD,i) = dv(i);
//				}
//
//				cd_data(number_of_data_points_CD,number_of_design_variables) = CD_exact;
//
//				for(int i=0;i<number_of_design_variables;i++){
//
//					cd_data(number_of_data_points_CD,number_of_design_variables+1+i) = gradient_cd(i);
//				}
//
//
//
//
//				/* insert a row to the area kriging data matrix*/
//				volume_data.insert_rows( number_of_data_points_volume, 1 );
//
//				for(int i=0;i<number_of_design_variables;i++){
//
//					volume_data(number_of_data_points_volume,i) = dv(i);
//				}
//				volume_data(number_of_data_points_volume,number_of_design_variables) = volume_exact;
//
//
//
//			}
//
//
//		} /* end of the simulation loop */
//
//		/* save updated data */
//		optimization_data.save(all_data_file.c_str(), csv_ascii);
//		cl_data.save(cl_kriging_input_file.c_str(), csv_ascii);
//		cd_data.save(cd_kriging_input_file.c_str(), csv_ascii);
//		volume_data.save(volume_kriging_input_file.c_str(), csv_ascii);
//
//
//		if (number_of_function_evals > max_number_of_function_evaluations ){
//
//			vec ys_CL =   optimization_data.col(number_of_design_variables);
//			vec ys_CD =   optimization_data.col(number_of_design_variables+1);
//			vec ys_volume = optimization_data.col(number_of_design_variables+2);
//
//			/* now find the best sample point */
//
//			double best_sample_CD = LARGE;
//			double best_sample_CL = LARGE;
//			double best_sample_volume = LARGE;
//			int sample_min_indx = -1;
//
//			for(unsigned int k=0; k<ys_CD.size(); k++){
//
//				if(ys_CD(k) < sample_min && ys_CL(k) >= CL_constraint && ys_volume(k) >= Volume_constraint){
//
//					sample_min_indx = k;
//					best_sample_CD = ys_CD(k);
//					best_sample_CL = ys_CL(k);
//					best_sample_volume = ys_volume(k);
//
//				}
//
//			}
//#if 1
//			printf("Optimization is finished...\n");
//			printf("Best sample in the data set is the %dth entry :\n",sample_min_indx);
//			optimization_data.row(sample_min_indx).print();
//			printf("CD = %10.7e\n",best_sample_CD);
//			printf("CL = %10.7e\n",best_sample_CL);
//			printf("Vol = %10.7e\n",best_sample_volume);
//
//#endif
//
//			break;
//
//
//
//
//
//		}
//
//		number_of_EI_iter_global = number_of_EI_iter_global/2;
//
//		it_count_outer_loop++;
//
//
//	} /* end of while(1) */
//
//
//
//
//
//}


//int su2_robustOptimizeRae2822(void){
//
//
//	/* file name for the optimization history */
//	std::string all_data_file = "rae2822_optimization_history.csv";
//
//	const int number_of_design_variables = 38;
//	const double gaussian_noise_level = 0.0001;
//	std::default_random_engine generator;
//	std::normal_distribution<double> distribution(0.0,1.0);
//
//	const int number_of_function_evals_inner_iter = 5;
//	int number_of_MC_iter1 = 500;
//
//	vec best_EI_values(number_of_function_evals_inner_iter);
//	best_EI_values.fill(-LARGE);
//	mat dv_best_EI(number_of_function_evals_inner_iter,number_of_design_variables);
//	dv_best_EI.fill(0.0);
//
//	int worst_EI_array_indx = 0;
//
//
//	/* box constraints for the design variables */
//	const double upper_bound_dv =  0.003;
//	const double lower_bound_dv = -0.003;
//
//	const double lift_penalty_param = LARGE;
//	const double area_penalty_param = LARGE;
//
//	double CL_constraint = 0.723;
//	double Area_constraint = 0.077;
//
//	KrigingModel model_area("Area",number_of_design_variables);
//	model_area.max_number_of_kriging_iterations = 1000;
//
//
//	printf("here");
//	train_kriging_response_surface(model_area);
//	model_area.update();
//
//
//
//	AggregationModel model_CL("CL",number_of_design_variables);
//	model_CL.validationset_input_filename = "CL_Validation.csv";
//	model_CL.max_number_of_kriging_iterations = 1000;
//	model_CL.number_of_cv_iterations = 0;
//	model_CL.number_of_cv_iterations_rho = 0;
//	model_CL.visualizeKrigingValidation = "yes";
//	model_CL.visualizeKernelRegressionValidation = "yes";
//	model_CL.visualizeAggModelValidation = "yes";
//	model_CL.train();
//	model_CL.update();
//
//
//	AggregationModel model_CD("CD",number_of_design_variables);
//	model_CD.validationset_input_filename = "CD_Validation.csv";
//	model_CD.max_number_of_kriging_iterations = 1000;
//	model_CD.number_of_cv_iterations = 0;
//	model_CD.number_of_cv_iterations_rho = 0;
//	model_CD.visualizeKrigingValidation = "yes";
//	model_CD.visualizeKernelRegressionValidation = "yes";
//	model_CD.visualizeAggModelValidation = "yes";
//	model_CD.train();
//	model_CD.update();
//
//
//
//
//
//
//
//
//	while(1){
//
//		/* load samples from file*/
//		mat optimization_data;
//		optimization_data.load(all_data_file.c_str(), csv_ascii);
//
//#if 0
//		printf("optimization data:\n");
//		optimization_data.print();
//
//#endif
//
//
//
//
//		/* now find the best sample point */
//
//		double sample_min = LARGE;
//		double best_sample_CD = 0.0;
//		double best_sample_CL= 0.0;
//		double best_sample_S = 0.0;
//		int sample_min_indx = -1;
//
//
//		vec ys_CL = optimization_data.col(number_of_design_variables);
//		vec ys_CD = optimization_data.col(number_of_design_variables+1);
//		vec ys_area = optimization_data.col(number_of_design_variables+2);
//
//
//
//
//		for(unsigned int i=0; i<optimization_data.n_rows; i++){
//
//
//			double obj_fun = ys_CD(i);
//
//			/* add penalties for lift and area */
//			if(ys_CL(i) < CL_constraint){
//
//				obj_fun += lift_penalty_param*(CL_constraint-ys_CD(i));
//			}
//
//			if(ys_area(i)  < Area_constraint){
//
//				obj_fun += area_penalty_param*(Area_constraint-ys_area(i));
//			}
//
//			if(obj_fun < sample_min){
//
//				sample_min = obj_fun;
//				sample_min_indx = i;
//				best_sample_CD = ys_CD(i);
//				best_sample_CL = ys_CL(i);
//				best_sample_S = ys_area(i);
//
//			}
//
//		}
//#if 1
//		printf("Best sample in the data set is the %dth entry (without robustness criteria) :\n",sample_min_indx);
//		optimization_data.row(sample_min_indx).print();
//		printf("objective function value = %10.7e\n",sample_min);
//		printf("CD = %10.7e\n",best_sample_CD);
//		printf("CL = %10.7e\n",best_sample_CL);
//		printf("S = %10.7e\n",best_sample_S);
//
//
//#endif
//
//		double min_J = LARGE;
//		double best_sample_avg_cd=0.0;
//		double best_sample_worst_cl=0.0;
//		double best_sample_worst_area=0.0;
//		int best_sample_index = -1;
//
//
//		mat X_optimization = optimization_data.submat(0,0,optimization_data.n_rows-1, number_of_design_variables-1);
//
//		/* for each sample in the data */
//		for(int sample_index =0; sample_index<optimization_data.n_rows; sample_index++ ){
//#if 0
//
//			printf("sample index = %d\n",sample_index);
//#endif
//
//			/* get the sample from the data matrix */
//
//			rowvec dv_simulation= X_optimization.row(sample_index);
//
//
//			rowvec dv_simulation_normalized(number_of_design_variables);
//#if 0
//			printf("dv_simulation =\n");
//			dv_simulation.print();
//#endif
//
//
//			normalize_vector(dv_simulation, dv_simulation_normalized, model_CD.xmin, model_CD.xmax);
//#if 0
//			printf("dv_simulation_normalized =\n");
//			dv_simulation_normalized.print();
//#endif
//
//
//
//
//			double CD_tilde = ys_CD(sample_index);
//
//#if 0
//			printf("CD_tilde = %10.7f\n",CD_tilde);
//#endif
//
//			double worst_case_cl=LARGE;
//			double worst_case_area=LARGE;
//
//
//			/* inner MC loop */
//			for(int inner_mc_iter=0; inner_mc_iter<number_of_MC_iter1;inner_mc_iter++ ){
//
//#if 0
//				printf("mc inner loop iter1 = %d\n",inner_mc_iter);
//#endif
//				rowvec dv_with_noise(number_of_design_variables);
//				rowvec dv_normalized_with_noise(number_of_design_variables);
//
//
//				/* generate a random design and normalize it*/
//				for(int j= 0; j< number_of_design_variables; j++){
//
//					double number = distribution(generator);
//
//					if(number > 5 ) number = 5.0;
//					if(number < -5 ) number = -5.0;
//
//					double perturbation  = gaussian_noise_level* number;
//#if 0
//					printf("generated random number = %10.7f\n",number);
//					printf("perturbation            = %10.7f\n",perturbation);
//#endif
//
//
//
//					dv_with_noise(j)= dv_simulation(j)+perturbation ;
//#if 0
//					printf("design parameter = %15.10f (unperturbed) %15.10f (perturbed) \n",dv_simulation(j),dv_with_noise(j));
//#endif
//					dv_normalized_with_noise(j) = (1.0/number_of_design_variables)*(dv_with_noise(j) - model_CD.xmin(j)) / (model_CD.xmax(j) - model_CD.xmin(j));
//
//				}
//
//#if 0
//
//				printf("dv_with_noise = ");
//				dv_with_noise.print();
//
//				printf("dv_normalized_with_noise = ");
//				dv_normalized_with_noise.print();
//#endif
//
//
//				double S_tilde = model_area.ftildeNorm(dv_normalized_with_noise);
//#if 0
//			printf("S_tilde = %10.7f\n",S_tilde);
//#endif
//
//				double CL_tilde = model_CL.ftilde(dv_normalized_with_noise);
//
//#if 0
//			printf("CL_tilde = %10.7f\n",CL_tilde);
//#endif
//
//				if(CL_tilde < worst_case_cl){
//
//					worst_case_cl = CL_tilde;
//
//				}
//
//				if(S_tilde < worst_case_area){
//
//					worst_case_area = S_tilde;
//
//				}
//
//
//			} /* end of MC loop */
//
//
//
//
//			double objective_fun = CD_tilde;
//
//			if (worst_case_cl < CL_constraint) {
//
//				objective_fun += LARGE;
//			}
//
//			if (worst_case_area < Area_constraint) {
//
//				objective_fun += LARGE;
//			}
//
//#if 1
//			printf("%d %10.7f %10.7f %10.7f %10.7f\n",sample_index,objective_fun,CD_tilde,worst_case_cl,worst_case_area);
//#endif
//
//			if (objective_fun < min_J) {
//
//				min_J = objective_fun;
//				best_sample_avg_cd = CD_tilde;
//				best_sample_worst_cl = worst_case_cl;
//				best_sample_worst_area = worst_case_area;
//				best_sample_index = sample_index;
//
//			}
//
//
//
//		} /* end of loop through all samples in the data */
//
//#if 1
//		printf("best sample in the data (with robustness criteria) = \n");
//		printf("data point index = %d\n",best_sample_index);
//		rowvec best_design=optimization_data.row(best_sample_index);
//		printf("best design = \n");
//		best_design.print();
//		printf("mean CD = %10.7f\n", best_sample_avg_cd );
//		printf("worst CL = %10.7f\n", best_sample_worst_cl );
//		printf("worst Area = %10.7f\n", best_sample_worst_area);
//#endif
//
//
//
//		exit(1);
//
//
//
//
//
//
//	}
//
//
//
//
//
//
//}




int su2_EGO(OptimizationData &optimization_plan){

	const int d = optimization_plan.size_of_dv;
	const int max_N = optimization_plan.max_number_of_samples;

	const double lift_penalty_param = LARGE;
	const double area_penalty_param = LARGE;

	double CL_constraint = 0.32;
	double Area_constraint = 0.081;

	/* copy training data from the samples folder */
	system("cp ./Samples/naca0012_optimization_history.csv ./");
	system("cp ./Samples/CL.csv ./");
	system("cp ./Samples/CD.csv ./");
	system("cp ./Samples/Area.csv ./");

	AggregationModel("CL",size_of_dv);

	AggregationModel("CD",size_of_dv);


	return 0;

}


//
//
//int su2_robustoptimize_naca0012(OptimizationData &optimization_plan){
//
//#if 0
//	optimization_plan.print();
//#endif
//
//	const int max_number_of_function_evaluations = optimization_plan.max_number_of_samples;
//	int number_of_design_variables  = optimization_plan.size_of_dv;
//
//	int max_number_of_function_calculations_training = 10000;
//	const int number_of_function_evals_inner_iter = 5;
//
//	int number_of_EI_iter;
//	int number_of_EI_iter_local = 100000;
//	int number_of_EI_iter_global = 1000000;
//
//	int number_of_MC_iter1 = 500;
//	int number_of_MC_iter2 = 1;
//
//
//
//	const double gaussian_noise_level = 0.0001;
//	std::default_random_engine generator;
//	std::normal_distribution<double> distribution(0.0,1.0);
//
//	vec best_EI_values(number_of_function_evals_inner_iter);
//	best_EI_values.fill(-LARGE);
//	mat dv_best_EI(number_of_function_evals_inner_iter,number_of_design_variables);
//	dv_best_EI.fill(0.0);
//
//	int worst_EI_array_indx = 0;
//
//
//	/* box constraints for the design variables */
//	const double upper_bound_dv =  0.003;
//	const double lower_bound_dv = -0.003;
//
//	const double lift_penalty_param = LARGE;
//	const double area_penalty_param = LARGE;
//
//	double CL_constraint = 0.32;
//	double Area_constraint = 0.081;
//
//	double reg_param = 10E-7;
//
//
//
//	int number_of_function_evals = 0;
//
//	/* copy training data from the samples folder */
//	system("cp ./Samples/naca0012_optimization_history.csv ./");
//	system("cp ./Samples/CL_Kriging.csv ./");
//	system("cp ./Samples/CD_Kriging.csv ./");
//	system("cp ./Samples/Area_Kriging.csv ./");
//
//
//
//	/* filenames for the Kriging input data */
//	std::string cl_kriging_input_file = "CL_Kriging.csv";
//	std::string cd_kriging_input_file = "CD_Kriging.csv";
//	std::string area_kriging_input_file = "Area_Kriging.csv";
//
//	/* file names for the Kriging hyperparameters (for CD, CL and area) */
//	std::string cl_kriging_hyperparameters_file = "CL_Kriging_Hyperparameters.csv";
//	std::string cd_kriging_hyperparameters_file = "CD_Kriging_Hyperparameters.csv";
//	std::string area_kriging_hyperparameters_file = "Area_Kriging_Hyperparameters.csv";
//
//
//	/* file name for the optimization history */
//	std::string all_data_file = "naca0012_optimization_history.csv";
//
//
//	/* obtain statistics from the existing data */
//	double EI_dist_tol = 0.0;
//
//	int max_iter_stat = 10000;
//
//	/* load samples from file*/
//	mat stat_data;
//	stat_data.load(all_data_file.c_str(), csv_ascii);
//
//#if 0
//	printf("stat_data:\n");
//	stat_data.print();
//
//#endif
//
//
//	/* normalized input variables */
//	mat X_stat(stat_data.n_rows,number_of_design_variables);
//
//
//	/* set the input data matrix X */
//	for(int i=0; i<number_of_design_variables;i++){
//
//		X_stat.col(i) = stat_data.col(i);
//
//	}
//
//#if 0
//	printf("X_stat:\n");
//	X_stat.print();
//#endif
//
//	int avg_distance_stat = 0.0;
//	for(int i=0; i<max_iter_stat; i++){
//
//		rowvec dv(number_of_design_variables);
//
//		/* Generate a random design vector */
//		for(int k=0; k<number_of_design_variables; k++){
//
//			dv(k)= RandomDouble(lower_bound_dv, upper_bound_dv);
//
//		}
//
//
//		int indx = -1;
//		double min_dist = LARGE;
//		findKNeighbours(X_stat, dv, 1, &min_dist,&indx, 1);
//
//		rowvec nearest = X_stat.row(indx);
//
//#if 0
//		printf("dv:\n");
//		dv.print();
//		printf("the nearest point is:\n");
//		nearest.print();
//		printf("minimum L1 distance = %10.7e\n",min_dist);
//
//#endif
//
//		avg_distance_stat+= min_dist;
//	}
//
//	avg_distance_stat = avg_distance_stat/max_iter_stat;
//
//	/* set the distance tolerance */
//	EI_dist_tol = avg_distance_stat / 10.0;
//
//
//
//	int it_count_outer_loop = 0;
//	int number_of_initial_samples = 0;
//
//
//
//
//
//	//	/* start optimization loop */
//
//	while(1){
//
//		/* load samples from file*/
//		mat optimization_data;
//		optimization_data.load(all_data_file.c_str(), csv_ascii);
//
//#if 0
//		printf("optimization data:\n");
//		optimization_data.print();
//
//#endif
//
//
//		/* normalized input variables */
//		mat X(optimization_data.n_rows,number_of_design_variables);
//
//		/*dimension of R */
//		int number_of_data_points = optimization_data.n_rows;
//
//		if(it_count_outer_loop == 0) {
//
//			number_of_initial_samples = number_of_data_points;
//		}
//
//
//		/* set the input data matrix X */
//		for(int i=0; i<number_of_design_variables;i++){
//
//			X.col(i) = optimization_data.col(i);
//
//		}
//
//#if 0
//		printf("X:\n");
//		X.print();
//#endif
//
//
//		/* find minimum and maximum of the columns of X */
//
//		vec x_max(number_of_design_variables);
//		x_max.fill(0.0);
//
//		vec x_min(number_of_design_variables);
//		x_min.fill(0.0);
//
//		for (int i = 0; i < number_of_design_variables; i++) {
//			x_max(i) = X.col(i).max();
//			x_min(i) = X.col(i).min();
//
//		}
//#if 0
//		printf("maximum = \n");
//		x_max.print();
//
//		printf("minimum = \n");
//		x_min.print();
//#endif
//		/* normalize data matrix */
//
//		for (unsigned int i = 0; i < X.n_rows; i++) {
//
//			for (int j = 0; j < number_of_design_variables; j++) {
//
//				X(i, j) = (1.0/number_of_design_variables)*(X(i, j) - x_min(j)) / (x_max(j) - x_min(j));
//			}
//		}
//
//
//		/* data matrices for surrogate model training (cl,cd,S) */
//		mat cl_kriging_data;
//		mat cd_kriging_data;
//		mat area_kriging_data;
//
//		/* load data for Kriging training */
//		cl_kriging_data.load(cl_kriging_input_file.c_str(), csv_ascii);
//		cd_kriging_data.load(cd_kriging_input_file.c_str(), csv_ascii);
//		area_kriging_data.load(area_kriging_input_file.c_str(), csv_ascii);
//
//
//		/* assign ys vectors for CL,CD and area */
//		vec ys_CL =   optimization_data.col(number_of_design_variables);
//		vec ys_CD =   optimization_data.col(number_of_design_variables+1);
//		vec ys_S = optimization_data.col(number_of_design_variables+2);
//
//
//
//
//
//
//		/* now find the best sample point */
//
//		double sample_min = LARGE;
//		double best_sample_CD = 0.0;
//		double best_sample_CL= 0.0;
//		double best_sample_S = 0.0;
//		int sample_min_indx = -1;
//
//		for(unsigned int k=0; k<ys_CD.size(); k++){
//
//			double obj_fun = ys_CD(k);
//
//			/* add penalties for lift and area */
//			if(ys_CL(k) < CL_constraint){
//
//				obj_fun += lift_penalty_param*(CL_constraint-ys_CL(k));
//			}
//
//			if(ys_S(k) < Area_constraint){
//
//				obj_fun += area_penalty_param*(Area_constraint-ys_S(k));
//			}
//
//			if(obj_fun < sample_min){
//
//				sample_min = obj_fun;
//				sample_min_indx = k;
//				best_sample_CD = ys_CD(k);
//				best_sample_CL = ys_CL(k);
//				best_sample_S = ys_S(k);
//
//			}
//
//		}
//#if 1
//		printf("Best sample in the data set is the %dth entry (without robustness criteria) :\n",sample_min_indx);
//		optimization_data.row(sample_min_indx).print();
//		printf("objective function value = %10.7e\n",sample_min);
//		printf("CD = %10.7e\n",best_sample_CD);
//		printf("CL = %10.7e\n",best_sample_CL);
//		printf("S = %10.7e\n",best_sample_S);
//
//
//#endif
//
//
//
//		/* correlation matrices for cd,cl and area */
//
//		mat R_CL(number_of_data_points,number_of_data_points);
//		mat R_CD(number_of_data_points,number_of_data_points);
//		mat R_S(number_of_data_points,number_of_data_points);
//
//		/* lower and upper diagonal matrices for Cholesky decomposition */
//
//		mat U_CL(number_of_data_points,number_of_data_points);
//		mat U_CD(number_of_data_points,number_of_data_points);
//		mat U_S(number_of_data_points,number_of_data_points);
//
//
//		mat L_CL(number_of_data_points,number_of_data_points);
//		mat L_CD(number_of_data_points,number_of_data_points);
//		mat L_S(number_of_data_points,number_of_data_points);
//
//		/* vector of ones */
//		vec I = ones(number_of_data_points);
//
//#if 0
//		cl_kriging_data.print();
//#endif
//
//
//#if 1
//		printf("number of samples = %d\n",number_of_data_points);
//#endif
//
//		/* visualize the CL&CD of the samples */
//
//		std::string file_name_for_plot = "samples.png";
//
//		std::string python_command = "python -W ignore "+settings.python_dir+"/plot_cdcl.py "+ all_data_file+ " "+
//				std::to_string(number_of_initial_samples)+ " " +
//				file_name_for_plot;
//
//		FILE* in = popen(python_command.c_str(), "r");
//		fprintf(in, "\n");
//
//
//
//
//
//
//
//		mat kriging_weights_CD;
//		mat regression_weights_CD;
//		mat R_inv_ys_min_beta_CD_tr;
//		double r_CD;
//
//		vec beta0_CD_tr(number_of_design_variables+1);
//
//
//		mat kriging_weights_CL;
//		mat regression_weights_CL;
//		mat R_inv_ys_min_beta_CL_tr;
//		double r_CL;
//
//		vec beta0_CL_tr(number_of_design_variables+1);
//
//
//		vec kriging_weights_S;
//		vec regression_weights_S;
//
//
//
//		/* train response surface for CL */
//
//		r_CL =  8.5449607e-02;
//		train_TRGEK_response_surface(cl_kriging_input_file,
//				cl_kriging_hyperparameters_file,
//				LINEAR_REGRESSION_ON,
//				regression_weights_CL,
//				kriging_weights_CL,
//				R_inv_ys_min_beta_CL_tr,
//				r_CL,
//				beta0_CL_tr,
//				max_number_of_function_calculations_training,
//				number_of_design_variables,0);
//#if 1
//		printf("kriging weights (CL):\n");
//		trans(kriging_weights_CL).print();
//		printf("regression weights (CL):\n");
//		trans(regression_weights_CL).print();
//		printf("r (CL) = %10.7e\n", r_CL);
//#endif
//
//		/* train response surface for CD */
//
//		r_CD =  7.9758370e-01;
//		train_TRGEK_response_surface(cd_kriging_input_file,
//				cd_kriging_hyperparameters_file,
//				LINEAR_REGRESSION_ON,
//				regression_weights_CD,
//				kriging_weights_CD,
//				R_inv_ys_min_beta_CD_tr,
//				r_CD,
//				beta0_CD_tr,
//				max_number_of_function_calculations_training,
//				number_of_design_variables,0);
//#if 1
//		printf("kriging weights (CD):\n");
//		trans(kriging_weights_CD).print();
//		printf("regression weights (CD):\n");
//		trans(regression_weights_CD).print();
//		printf("r (CD) = %10.7e\n", r_CD);
//#endif
//
//		/* train response surface for area */
//
//		train_kriging_response_surface(area_kriging_input_file,
//				area_kriging_hyperparameters_file,
//				LINEAR_REGRESSION_ON,
//				regression_weights_S,
//				kriging_weights_S,
//				reg_param,
//				max_number_of_function_calculations_training,
//				CSV_ASCII);
//
//#if 1
//		printf("kriging weights (area):\n");
//		trans(kriging_weights_S).print();
//		printf("regression weights (area):\n");
//		trans(regression_weights_S).print();
//#endif
//
//
//
//		/*update y vectors according to linear regression*/
//
//		mat augmented_X(number_of_data_points, number_of_design_variables + 1);
//
//		for (int i = 0; i < number_of_data_points; i++) {
//
//			for (int j = 0; j <= number_of_design_variables; j++) {
//
//				if (j == 0){
//
//					augmented_X(i, j) = 1.0;
//				}
//				else{
//
//					augmented_X(i, j) = X(i, j - 1);
//
//				}
//			}
//		}
//
//		vec ys_reg_cl = augmented_X * regression_weights_CL;
//		vec ys_reg_cd = augmented_X * regression_weights_CD;
//		vec ys_reg_S = augmented_X * regression_weights_S;
//
//
//		ys_CL = ys_CL - ys_reg_cl;
//		ys_CD = ys_CD - ys_reg_cd;
//		ys_S = ys_S - ys_reg_S;
//
//
//
//
//		vec theta_CL = kriging_weights_CL.col(0).head(number_of_design_variables);
//		vec gamma_CL = kriging_weights_CL.col(0).tail(number_of_design_variables);
//
//		vec theta_CD = kriging_weights_CD.col(0).head(number_of_design_variables);
//		vec gamma_CD = kriging_weights_CD.col(0).tail(number_of_design_variables);
//
//
//		vec theta_S = kriging_weights_S.head(number_of_design_variables);
//		vec gamma_S = kriging_weights_S.tail(number_of_design_variables);
//
//
//#if 0
//		printf("theta CL:\n");
//		trans(theta_CL).print();
//
//		printf("gamma CL:\n");
//		trans(gamma_CL).print();
//
//		printf("theta CD:\n");
//		trans(theta_CD).print();
//
//		printf("gamma CD:\n");
//		trans(gamma_CD).print();
//
//		printf("theta area:\n");
//		trans(theta_area).print();
//
//		printf("gamma area:\n");
//		trans(gamma_area).print();
//#endif
//
//
//		compute_R_matrix(theta_CL,
//				gamma_CL,
//				reg_param,
//				R_CL,
//				X);
//#if 0
//		printf("R_CL:\n");
//		R_CL.print();
//#endif
//
//
//		/* Cholesky decomposition R = LDL^T */
//
//		int cholesky_return_CL = chol(U_CL, R_CL);
//
//		if (cholesky_return_CL == 0) {
//			printf("Error: Ill conditioned correlation matrix for CL, Cholesky decomposition failed...\n");
//			exit(-1);
//		}
//
//		L_CL = trans(U_CL);
//
//		vec R_inv_ys_CL(number_of_data_points);
//		vec R_inv_I_CL(number_of_data_points);
//
//		solve_linear_system_by_Cholesky(U_CL, L_CL, R_inv_ys_CL, ys_CL); /* solve R x = ys */
//		solve_linear_system_by_Cholesky(U_CL, L_CL, R_inv_I_CL, I);      /* solve R x = I */
//
//
//		double	beta0_CL = (1.0/dot(I,R_inv_I_CL)) * (dot(I,R_inv_ys_CL));
//#if 0
//		printf("beta0_CL= %20.15f\n",beta0);
//#endif
//
//		vec ys_min_betaI_CL = ys_CL-beta0_CL*I;
//
//		vec R_inv_ys_min_beta_CL(number_of_data_points);
//
//
//
//		/* solve R x = ys-beta0*I */
//		solve_linear_system_by_Cholesky(U_CL, L_CL, R_inv_ys_min_beta_CL, ys_min_betaI_CL);
//
//
//		compute_R_matrix(theta_CD,
//				gamma_CD,
//				reg_param,
//				R_CD,
//				X);
//
//
//		int cholesky_return_CD = chol(U_CD, R_CD);
//
//		if (cholesky_return_CD == 0) {
//			printf("Error: Ill conditioned correlation matrix, Cholesky decomposition failed...\n");
//			exit(-1);
//		}
//
//		L_CD = trans(U_CD);
//
//		vec R_inv_ys_CD(number_of_data_points);
//		vec R_inv_I_CD(number_of_data_points);
//
//
//
//		solve_linear_system_by_Cholesky(U_CD, L_CD, R_inv_ys_CD, ys_CD); /* solve R x = ys */
//		solve_linear_system_by_Cholesky(U_CD, L_CD, R_inv_I_CD, I);   /* solve R x = I */
//
//
//
//
//		double beta0_CD = (1.0/dot(I,R_inv_I_CD)) * (dot(I,R_inv_ys_CD));
//
//
//		vec ys_min_betaI_CD = ys_CD-beta0_CD*I;
//
//		vec R_inv_ys_min_beta_CD(number_of_data_points);
//
//
//
//		/* solve R x = ys-beta0*I */
//		solve_linear_system_by_Cholesky(U_CD, L_CD, R_inv_ys_min_beta_CD, ys_min_betaI_CD);
//
//
//		double ssqr_CD = (1.0 / number_of_data_points) * dot(ys_min_betaI_CD, R_inv_ys_min_beta_CD);
//
//#if 0
//		printf("computing R for area\n");
//#endif
//		compute_R_matrix(theta_S,
//				gamma_S,
//				reg_param,
//				R_S,
//				X);
//
//
//
//		int cholesky_return_S = chol(U_S, R_S);
//
//		if (cholesky_return_S == 0) {
//			printf("Error: Ill conditioned correlation matrix for S, Cholesky decomposition failed...\n");
//			exit(-1);
//		}
//
//		L_S = trans(U_S);
//
//		vec R_inv_ys_S(number_of_data_points);
//		vec R_inv_I_S(number_of_data_points);
//
//
//
//		solve_linear_system_by_Cholesky(U_S, L_S, R_inv_ys_S, ys_S); /* solve R x = ys */
//		solve_linear_system_by_Cholesky(U_S, L_S, R_inv_I_S, I);     /* solve R x = I */
//
//		double beta0_S = (1.0/dot(I,R_inv_I_S)) * (dot(I,R_inv_ys_S));
//
//
//		vec ys_min_betaI_S = ys_S-beta0_S*I;
//
//		vec R_inv_ys_min_beta_S(number_of_data_points);
//
//
//
//		/* solve R x = ys-beta0*I */
//		solve_linear_system_by_Cholesky(U_S, L_S, R_inv_ys_min_beta_S, ys_min_betaI_S);
//
//
//#if 1
//		printf("beta0 for CL : %15.10f\n",beta0_CL);
//		printf("beta0 for CD : %15.10f\n",beta0_CD);
//		printf("beta0 for Area : %15.10f\n",beta0_S);
//#endif
//
//		/* try all the designs in the data to find the most robust design*/
//
//		double min_J = LARGE;
//		double best_sample_avg_cd=0.0;
//		double best_sample_worst_cl=0.0;
//		double best_sample_worst_area=0.0;
//		int best_sample_index = -1;
//
//
//		/* for each sample in the data */
//		for(int sample_index =0; sample_index <optimization_data.n_rows;sample_index++ ){
//#if 0
//
//			printf("sample index = %d\n",sample_index);
//#endif
//
//			/* get the sample from the data matrix */
//			rowvec dv_simulation(number_of_design_variables);
//			dv_simulation= optimization_data.row(sample_index);
//
//
//#if 0
//			printf("dv_simulation =\n");
//			dv_simulation.print();
//#endif
//
//			double average_cd=0.0;
//			double worst_case_cl=LARGE;
//			double worst_case_area=LARGE;
//
//
//			/* inner MC loop */
//			for(int inner_mc_iter=0; inner_mc_iter<number_of_MC_iter1;inner_mc_iter++ ){
//
//#if 0
//				printf("mc inner loop iter1 = %d\n",inner_mc_iter1);
//#endif
//				rowvec dv_with_noise(number_of_design_variables, fill::zeros);
//				rowvec dv_normalized_with_noise(number_of_design_variables, fill::zeros);
//
//
//				/* generate a random design and normalize it*/
//				for(int j= 0; j< number_of_design_variables; j++){
//
//					double number = distribution(generator);
//
//					if(number > 5 ) number = 5.0;
//					if(number < -5 ) number = -5.0;
//
//					double perturbation  = gaussian_noise_level* number;
//#if 0
//					printf("generated random number = %10.7f\n",number);
//					printf("perturbation            = %10.7f\n",perturbation);
//#endif
//
//
//
//					dv_with_noise(j)= dv_simulation(j)+perturbation ;
//#if 0
//					printf("design parameter = %15.10f (unperturbed) %15.10f (perturbed) \n",dv_simulation(j),dv_with_noise(j));
//#endif
//					dv_normalized_with_noise(j) = (1.0/number_of_design_variables)*(dv_with_noise(j) - x_min(j)) / (x_max(j) - x_min(j));
//
//				}
//
//#if 0
//
//				printf("dv_with_noise = ");
//				dv_with_noise.print();
//
//				printf("dv_normalized_with_noise = ");
//				dv_normalized_with_noise.print();
//#endif
//
//
//				/* Kriging estimate of the area */
//				double S_tilde = calculate_f_tilde(dv_normalized_with_noise,
//						X,
//						beta0_S,
//						regression_weights_S,
//						R_inv_ys_min_beta_S,
//						kriging_weights_S);
//
//				/* Kriging estimate of the CL */
//				double CL_tilde = calculate_f_tilde(dv_normalized_with_noise,
//						X,
//						beta0_CL,
//						regression_weights_CL,
//						R_inv_ys_min_beta_CL,
//						kriging_weights_CL);
//
//				/* Kriging estimate of the CD */
//				double CD_tilde = calculate_f_tilde(dv_normalized_with_noise,
//						X,
//						beta0_CD,
//						regression_weights_CD,
//						R_inv_ys_min_beta_CD,
//						kriging_weights_CD);
//
//
//				double min_dist=0;
//				int indx = -1;
//
//				/* find the closest data point */
//
//				findKNeighbours(X,
//						dv_normalized_with_noise,
//						1,
//						&min_dist,
//						&indx,
//						1);
//
//
//				/* sp is the nearest neighbour */
//				rowvec sp =  X.row(indx);
//				rowvec sp_not_normalized(number_of_design_variables);
//				sp_not_normalized.fill(0.0);
//
//				for(int j=0; j<number_of_design_variables;j++) {
//
//					sp_not_normalized(j) = number_of_design_variables*sp(j)* (x_max(j) - x_min(j))+x_min(j);
//				}
//#if 0
//				printf("closest point = \n");
//
//				for(int k=0; k<number_of_design_variables; k++){
//
//					printf("%10.7e  %10.7e\n",dv_with_noise(k),sp_not_normalized(k));
//
//				}
//
//				printf("grad data entry (CD):\n");
//				cd_kriging_data.row(indx).print();
//				printf("grad data entry (CL):\n");
//				cd_kriging_data.row(indx).print();
//
//#endif
//				rowvec xdiff = dv_normalized_with_noise-sp;
//				double distance = L1norm(xdiff, number_of_design_variables);
//#if 0
//				printf("distance = %10.7e\n",distance);
//#endif
//
//				/* get the functional value from the data */
//				double func_val_CD = cd_kriging_data(indx,number_of_design_variables);
//				double func_val_CL = cl_kriging_data(indx,number_of_design_variables);
//#if 0
//				printf("CD value at the nearest point = %10.7e\n",func_val_CD);
//				printf("CL value at the nearest point = %10.7e\n",func_val_CL);
//#endif
//
//
//				vec grad_CD(number_of_design_variables);
//
//				for(int j=0; j<number_of_design_variables; j++) {
//
//					grad_CD(j)= cd_kriging_data(indx,j+number_of_design_variables+1);
//				}
//
//				vec grad_CL(number_of_design_variables);
//
//				for(int j=0; j<number_of_design_variables; j++) {
//
//					grad_CL(j)= cl_kriging_data(indx,j+number_of_design_variables+1);
//				}
//
//#if 0
//				printf("gradient vector (CD):\n");
//				trans(grad_CD).print();
//				printf("gradient vector (CL):\n");
//				trans(grad_CL).print();
//#endif
//
//
//
//
//
//				double normgrad_CD= L1norm(grad_CD, number_of_design_variables);
//				double normgrad_CL= L1norm(grad_CL, number_of_design_variables);
//
//#if 0
//				printf("norm of the gradient vector (CD) = %10.7e\n",normgrad_CD);
//				printf("norm of the gradient vector (CL) = %10.7e\n",normgrad_CL);
//#endif
//
//
//				double factor_CD = exp(-r_CD*distance*normgrad_CD);
//				double factor_CL = exp(-r_CL*distance*normgrad_CL);
//
//				double fval_linmodel_CD= func_val_CD + dot((dv_with_noise-sp_not_normalized),grad_CD);
//				double fval_linmodel_CL= func_val_CL + dot((dv_with_noise-sp_not_normalized),grad_CL);
//
//				double fval_CD = factor_CD*fval_linmodel_CD + (1.0-factor_CD)*CD_tilde;
//				double fval_CL = factor_CL*fval_linmodel_CL + (1.0-factor_CL)*CL_tilde;
//
//#if 0
//				printf("CL_tilde = %10.7f\n",CL_tilde);
//				printf("CL_aggre = %10.7f\n",fval_CL);
//				printf("CD_tilde = %10.7f\n",CD_tilde);
//				printf("CD_aggre = %10.7f\n",fval_CD);
//				printf("Area     = %10.7f\n",S_tilde);
//#endif
//
//				average_cd = average_cd + fval_CD;
//
//				if(fval_CL < worst_case_cl){
//
//					worst_case_cl = fval_CL;
//
//				}
//
//				if(S_tilde < worst_case_area){
//
//					worst_case_area = S_tilde;
//
//				}
//
//
//			} /* end of MC loop */
//
//
//			average_cd = average_cd/number_of_MC_iter1;
//
//			double objective_fun = average_cd;
//
//			if (worst_case_cl < CL_constraint) {
//
//				objective_fun += LARGE;
//			}
//
//			if (worst_case_area < Area_constraint) {
//
//				objective_fun += LARGE;
//			}
//
//#if 1
//			printf("%d %10.7f %10.7f %10.7f %10.7f\n",sample_index,objective_fun,average_cd,worst_case_cl,worst_case_area);
//#endif
//
//			if (objective_fun < min_J) {
//
//				min_J = objective_fun;
//				best_sample_avg_cd = average_cd;
//				best_sample_worst_cl = worst_case_cl;
//				best_sample_worst_area = worst_case_area;
//				best_sample_index = sample_index;
//
//			}
//
//
//
//		} /* end of loop through all samples in the data */
//
//#if 1
//		printf("best sample in the data (with robustness criteria) = \n");
//		printf("data point index = %d\n",best_sample_index);
//		rowvec best_design=optimization_data.row(best_sample_index);
//		printf("best design = \n");
//		best_design.print();
//		printf("mean CD = %10.7f\n", best_sample_avg_cd );
//		printf("worst CL = %10.7f\n", best_sample_worst_cl );
//		printf("worst Area = %10.7f\n", best_sample_worst_area);
//#endif
//
//
//		if (number_of_function_evals > max_number_of_function_evaluations ){
//
//			return 0;
//		}
//
//
//		sample_min = min_J;
//
//
//
//
//
//		best_EI_values.fill(-LARGE);
//		dv_best_EI.fill(0.0);
//		worst_EI_array_indx = 0;
//
//		/* number of iterations performed in the EI loop */
//		number_of_EI_iter = number_of_EI_iter_local+number_of_EI_iter_global;
//
//
//
//		for(int iter_EI=0; iter_EI<number_of_EI_iter; iter_EI++){
//#if 1
//
//
//			if(iter_EI % 1000 == 0){
//
//				printf("\rEI loop iter = %d\n",iter_EI);
//				fflush(stdout);
//			}
//#endif
//			rowvec dv(number_of_design_variables,fill::zeros);
//			rowvec dvnorm(number_of_design_variables,fill::zeros);
//
//
//			if(iter_EI < number_of_EI_iter_local){
//				/* search around best sample in the data set */
//				for(int k=0; k<number_of_design_variables; k++){
//
//					double perturbation = RandomDouble(lower_bound_dv/100, upper_bound_dv/100);
//
//					dv(k)= best_design(k)+perturbation;
//					dvnorm(k) = (1.0/number_of_design_variables)*(dv(k)-x_min(k)) / (x_max(k)-x_min(k));
//
//				}
//
//
//			}
//
//			else{
//				/* Generate a random design vector and add a random perturbation to it*/
//				for(int k=0; k<number_of_design_variables; k++){
//
//					dv(k)= RandomDouble(lower_bound_dv, upper_bound_dv);
//					dvnorm(k) = (1.0/number_of_design_variables)*(dv(k)-x_min(k)) / (x_max(k)-x_min(k));
//
//				}
//
//
//			} /* end of else */
//#if 0
//			printf("dv:\n");
//			dv.print();
//			printf("dvnorm:\n");
//			dvnorm.print();
//#endif
//
//			double average_cd=0.0;
//			double worst_case_cl=LARGE;
//			double worst_case_area=LARGE;
//
//
//
//
//			/* start monte carlo loop */
//			for(int iter_MC=0; iter_MC <number_of_MC_iter2; iter_MC ++){
//
//
//				rowvec dv_perturbed(number_of_design_variables, fill::zeros);
//				rowvec dvnorm_perturbed(number_of_design_variables, fill::zeros);
//
//
//
//				/* perturn design vector dv and normalize it */
//				for(int k=0; k<number_of_design_variables; k++){
//
//
//					double number = distribution(generator);
//					if(number > 5 ) number = 5.0;
//					if(number < -5 ) number = -5.0;
//					double perturbation  = gaussian_noise_level* number;
//
//					if (number_of_MC_iter2 == 1) perturbation =0.0;
//
//					dv_perturbed(k) = dv(k) + perturbation;
//
//					/* normalized design vector	*/
//					dvnorm_perturbed(k) = (1.0/number_of_design_variables)*(dv_perturbed(k)-x_min(k)) / (x_max(k)-x_min(k));
//				}
//
//
//				/* Kriging estimate of the area */
//				double S_tilde = calculate_f_tilde(dvnorm_perturbed,
//						X,
//						beta0_S,
//						regression_weights_S,
//						R_inv_ys_min_beta_S,
//						kriging_weights_S);
//
//				/* Kriging estimate of the CL */
//				double CL_tilde = calculate_f_tilde(dvnorm_perturbed,
//						X,
//						beta0_CL,
//						regression_weights_CL,
//						R_inv_ys_min_beta_CL,
//						kriging_weights_CL);
//
//
//				/* Kriging estimate of the CD */
//				double CD_tilde = calculate_f_tilde(dvnorm_perturbed,
//						X,
//						beta0_CD,
//						regression_weights_CD,
//						R_inv_ys_min_beta_CD,
//						kriging_weights_CD);
//
//
//
//				double min_dist=0;
//
//				int indx = -1;
//
//				/* find the closest data point */
//
//				findKNeighbours(X,
//						dvnorm,
//						1,
//						&min_dist,
//						&indx,
//						1);
//
//
//
//				rowvec sp =  X.row(indx);
//				rowvec sp_not_normalized(number_of_design_variables);
//				sp_not_normalized.fill(0.0);
//
//				for(int j=0; j<number_of_design_variables;j++) {
//
//					sp_not_normalized(j) = number_of_design_variables*sp(j)* (x_max(j) - x_min(j))+x_min(j);
//				}
//#if 0
//				printf("closest point           point:\n");
//
//				for(int k=0; k<number_of_design_variables; k++){
//
//					printf("%10.7e  %10.7e\n",dv(k),sp_not_normalized(k));
//
//				}
//
//				printf("grad data entry (CD):\n");
//				cd_kriging_data.row(indx).print();
//				printf("grad data entry (CL):\n");
//				cd_kriging_data.row(indx).print();
//
//#endif
//				rowvec xdiff = dvnorm-sp;
//				double distance = L1norm(xdiff, number_of_design_variables);
//#if 0
//				printf("distance = %10.7e\n",distance);
//#endif
//
//				/* get the functional value from the data */
//				double func_val_CD = cd_kriging_data(indx,number_of_design_variables);
//				double func_val_CL = cl_kriging_data(indx,number_of_design_variables);
//#if 0
//				printf("CD value at the nearest point = %10.7e\n",func_val_CD);
//				printf("CL value at the nearest point = %10.7e\n",func_val_CL);
//#endif
//
//
//				vec grad_CD(number_of_design_variables);
//
//				for(int j=0; j<number_of_design_variables; j++) {
//
//					grad_CD(j)= cd_kriging_data(indx,j+number_of_design_variables+1);
//				}
//
//				vec grad_CL(number_of_design_variables);
//
//				for(int j=0; j<number_of_design_variables; j++) {
//
//					grad_CL(j)= cl_kriging_data(indx,j+number_of_design_variables+1);
//				}
//
//#if 0
//				printf("gradient vector (CD):\n");
//				trans(grad_CD).print();
//				printf("gradient vector (CL):\n");
//				trans(grad_CL).print();
//#endif
//
//
//
//
//
//				double normgrad_CD= L1norm(grad_CD, number_of_design_variables);
//				double normgrad_CL= L1norm(grad_CL, number_of_design_variables);
//
//#if 0
//				printf("norm of the gradient vector (CD) = %10.7e\n",normgrad_CD);
//				printf("norm of the gradient vector (CL) = %10.7e\n",normgrad_CL);
//#endif
//
//
//				double factor_CD = exp(-r_CD*distance*normgrad_CD);
//				double factor_CL = exp(-r_CL*distance*normgrad_CL);
//
//				double fval_linmodel_CD= func_val_CD + dot((dv-sp_not_normalized),grad_CD);
//				double fval_linmodel_CL= func_val_CL + dot((dv-sp_not_normalized),grad_CL);
//
//				double fval_CD = factor_CD*fval_linmodel_CD + (1.0-factor_CD)*CD_tilde;
//				double fval_CL = factor_CL*fval_linmodel_CL + (1.0-factor_CL)*CL_tilde;
//
//
//				average_cd = average_cd + fval_CD;
//
//				if(fval_CL < worst_case_cl){
//
//					worst_case_cl = fval_CL;
//
//				}
//
//				if(S_tilde < worst_case_area){
//
//					worst_case_area = S_tilde;
//
//				}
//
//
//
//
//
//
//			} /* end of MC loop */
//
//			average_cd = average_cd/number_of_MC_iter2;
//
//
//			/* calculate the EI value */
//
//			double CD_tilde = 0.0;
//			double CD_tilde_ssqr = 0.0;
//
//			calculate_f_tilde_and_ssqr(
//					dvnorm,
//					X,
//					beta0_CD,
//					ssqr_CD,
//					regression_weights_CD,
//					R_inv_ys_min_beta_CD,
//					R_inv_I_CD,
//					I,
//					kriging_weights_CD,
//					U_CD,
//					L_CD,
//					&CD_tilde,
//					&CD_tilde_ssqr);
//
//
//			double	standart_error_CD = sqrt(CD_tilde_ssqr)	;
//
//			double EI_CD = 0.0;
//
//			double obj_fun = average_cd ;
//
//			/* add penalties for lift and area */
//			if(worst_case_cl < CL_constraint){
//
//				obj_fun += LARGE;
//			}
//
//			if(worst_case_area < Area_constraint){
//
//				obj_fun += LARGE;
//			}
//
//
//			if(standart_error_CD!=0.0){
//
//				double	EIfac = (sample_min - obj_fun)/standart_error_CD;
//
//				/* calculate the Expected Improvement value */
//				EI_CD = (sample_min - obj_fun)*cdf(EIfac,0.0,1.0)+standart_error_CD*pdf(EIfac,0.0,1.0);
//			}
//			else{
//
//				EI_CD =0.0;
//
//			}
//
//			/* find the closest point to dv in the current best EI designs */
//
//			double min_dist_EI = LARGE;
//			int    min_EI_indx = -1;
//			findKNeighbours(dv_best_EI, dv, 1, &min_dist_EI, &min_EI_indx , 1);
//
//			if(EI_CD > best_EI_values(worst_EI_array_indx && min_dist_EI > EI_dist_tol)){
//
//
//				best_EI_values(worst_EI_array_indx) = EI_CD;
//
//				for(int k=0;k<number_of_design_variables;k++) {
//
//					dv_best_EI(worst_EI_array_indx,k) = dv(k);
//				}
//
//
//				double worst_EI_value = 0.0;
//				find_min_with_index(best_EI_values,
//						number_of_function_evals_inner_iter,
//						&worst_EI_value ,
//						&worst_EI_array_indx);
//#if 1
//				printf("found a better EI value with EI = %10.7e\n",EI_CD);
//				printf("best EI values:\n");
//				trans(best_EI_values).print();
//				printf("worst EI index is now = %d\n",worst_EI_array_indx);
//				printf("worst area = %10.7e:\n",worst_case_area);
//				printf("worst CL = %10.7e:\n",worst_case_cl);
//				printf("mean CD = %10.7e:\n",average_cd);
//				printf("EI value = %10.7e\n",EI_CD);
//#endif
//
//			} /* end of if */
//
//		} /* end of the EI loop */
//
//
//
//#if 1
//		printf("The most promising designs have been found...\n");
//#endif
//
//		vec gradient_cd(number_of_design_variables,fill::zeros);
//		vec gradient_cl(number_of_design_variables,fill::zeros);
//
//
//
//
//
//		for(int k=0; k<number_of_function_evals_inner_iter; k++ ){
//
//			vec dv = trans(dv_best_EI.row(k));
//
//			/* state outputs plus geometric constraints */
//			vec objectives(optimization_plan.number_of_outputs+optimization_plan.geometric_constraints.size(), fill::zeros);
//			vec gradient_out(number_of_design_variables);
//
//			for(int i=0; i<optimization_plan.number_outputs_with_gradients;i++){
//
//
//				printf("calling adjoint solver for %s...\n",optimization_plan.output_names[i].c_str());
//				call_SU2_Adjoint_Solver(i,dv,gradient_out,objectives,optimization_plan);
//
//				if(i==0) gradient_cl = gradient_out;
//				if(i==1) gradient_cd = gradient_out;
//
//
//
//			}
//
//
//			/* insert a row to the data matrix*/
//
//			if(objectives(0) > 0  && objectives(1) > 0){
//
//				number_of_function_evals++;
//
//				optimization_data.insert_rows( number_of_data_points, 1 );
//				for(int i=0;i<number_of_design_variables;i++){
//
//					optimization_data(number_of_data_points,i) = dv(i);
//				}
//				optimization_data(number_of_data_points,number_of_design_variables)   = objectives(0);
//				optimization_data(number_of_data_points,number_of_design_variables+1) = objectives(1);
//				optimization_data(number_of_data_points,number_of_design_variables+2) = objectives(2);
//
//
//				/* insert a row to the cl kriging data matrix*/
//				cl_kriging_data.insert_rows( number_of_data_points, 1 );
//
//				for(int i=0;i<number_of_design_variables;i++){
//
//					cl_kriging_data(number_of_data_points,i) = dv(i);
//				}
//				cl_kriging_data(number_of_data_points,number_of_design_variables) = objectives(0);
//
//				for(int i=0;i<number_of_design_variables;i++){
//
//					cl_kriging_data(number_of_data_points,number_of_design_variables+1+i) = gradient_cl(i);
//				}
//
//
//
//
//				/* insert a row to the cd kriging data matrix*/
//				cd_kriging_data.insert_rows( number_of_data_points, 1 );
//
//				for(int i=0;i<number_of_design_variables;i++){
//
//					cd_kriging_data(number_of_data_points,i) = dv(i);
//				}
//
//				cd_kriging_data(number_of_data_points,number_of_design_variables) = objectives(1);
//
//				for(int i=0;i<number_of_design_variables;i++){
//
//					cd_kriging_data(number_of_data_points,number_of_design_variables+1+i) = gradient_cd(i);
//				}
//
//
//
//
//				/* insert a row to the area kriging data matrix*/
//				area_kriging_data.insert_rows( number_of_data_points, 1 );
//
//				for(int i=0;i<number_of_design_variables;i++){
//
//					area_kriging_data(number_of_data_points,i) = dv(i);
//				}
//				area_kriging_data(number_of_data_points,number_of_design_variables) = objectives(2);
//
//
//
//			}
//
//
//
//		}  /* end of the simulation loop */
//
//
//		/* save updated data */
//		optimization_data.save(all_data_file.c_str(), csv_ascii);
//		cl_kriging_data.save(cl_kriging_input_file.c_str(), csv_ascii);
//		cd_kriging_data.save(cd_kriging_input_file.c_str(), csv_ascii);
//		area_kriging_data.save(area_kriging_input_file.c_str(), csv_ascii);
//
//
//		number_of_EI_iter_global = number_of_EI_iter_global/2;
//
//		it_count_outer_loop++;
//
//
//	} /* end of while(1) */
//
//
//
//
//
//}

//void su2_optimize(void){
//
//
//
//	int number_of_initial_samples   = 261;
//	int number_of_design_variables  = 38;
//
//	/* max number of function evaluations for the EA algorithm in Kriging */
//	const int max_number_of_function_calculations_training = 10000;
//	const int max_number_of_mc_iterations = 500000; /* number of mc outer iterations per tread */
//	const int max_number_of_inner_mc_iterations = 0; /* number of inner iterations in the MC loop */
//
//	const int max_number_of_function_evaluations = 300;
//
//	/* box constraints for the design variables */
//	const double upper_bound_dv =  0.003;
//	const double lower_bound_dv = -0.003;
//
//	/* standart deviation for the perturbations */
//	const double gaussian_noise_level = 0.0001;
//
//
//	/* constraints for CL and area*/
//	double CL_constraint = 0.723;
//	double Area_constraint = 0.0778;
//
//
//	/* step size for the adjoint assisted sampling */
//	const double stepsize0 = 0.0002;
//	double stepsize;
//
//	/* maximum number of simulations in the adjoint assisted sampling */
//	const int max_number_of_linesearches = 5;
//
//	/* maximum number of simulations in the EI based sampling */
//	const int number_of_samples_MC = 20;
//
//	/*regularization parameter for Kriging */
//	double reg_param = 10E-10;
//
//	/* vectors for the model parameters */
//
//	vec kriging_param_CL(number_of_design_variables);
//	vec kriging_param_CD(number_of_design_variables);
//	vec kriging_param_area(number_of_design_variables);
//
//	vec lin_reg_param_CL(number_of_design_variables);
//	vec lin_reg_param_CD(number_of_design_variables);
//	vec lin_reg_param_area(number_of_design_variables);
//
//
//	std::string su2_cfd_config_file = "turb_SA_RAE2822.cfg";
//
//	/* filenames for the Kriging input data */
//	std::string cl_kriging_input_file = "CL_Kriging.csv";
//	std::string cd_kriging_input_file = "CD_Kriging.csv";
//	std::string area_kriging_input_file = "Area_Kriging.csv";
//
//	/* file names for the Kriging hyperparameters (for CD, CL and area) */
//	std::string cl_kriging_hyperparameters_file = "CL_Kriging_Hyperparameters.csv";
//	std::string cd_kriging_hyperparameters_file = "CD_Kriging_Hyperparameters.csv";
//	std::string area_kriging_hyperparameters_file = "Area_Kriging_Hyperparameters.csv";
//
//	/* file name for the optimization history */
//	std::string all_data_file = "rae2822_optimization_history.csv";
//
//
//
//	vec dv= zeros(number_of_design_variables);      /* design vector */
//	vec dv_save= zeros(number_of_design_variables); /* old design vector  */
//
//	vec dvnorm= zeros(number_of_design_variables);      /* design vector (normalized) */
//	vec dv_savenorm= zeros(number_of_design_variables); /* old design vector (normalized)*/
//
//
//	vec grad_dv= zeros(number_of_design_variables);     /* gradient vector */
//	double CL=0.0,CD=0.0,area=0.0;
//
//
//
//
//	int cholesky_return;
//
//	/* Kriging model parameters */
//	double beta0_CL,beta0_CD,beta0_area;
//	double ssqr_CD;
//
//	int index_best_design = 0;
//	double min_CD = LARGE;
//	/* Kriging hyperparameters (theta and gamma valus for CD,CL and area)*/
//
//	vec theta_CL(number_of_design_variables);
//	vec theta_CD(number_of_design_variables);
//	vec theta_area(number_of_design_variables);
//	vec gamma_CL(number_of_design_variables);
//	vec gamma_CD(number_of_design_variables);
//	vec gamma_area(number_of_design_variables);
//
//
//
//	/* copy training data from the samples folder */
//	system("cp ./samples/rae2822_optimization_history.csv ./");
//	system("cp ./samples/CL_Kriging.csv ./");
//	system("cp ./samples/CD_Kriging.csv ./");
//	system("cp ./samples/Area_Kriging.csv ./");
//
//
//
//	std::vector<int> indices_of_ls_designs;
//
//
//
//	/* Optimization part */
//
//	int number_of_function_evaluations = 0;
//	int number_of_gradient_evaluations = 0;
//
//
//	std::default_random_engine generator;
//	std::normal_distribution<double> distribution(0.0,1.0);
//
//
//
//	while(1){
//
//
//		/* load samples from file*/
//		mat optimization_data;
//		optimization_data.load(all_data_file.c_str(), csv_ascii);
//
//		//		optimization_data.print();
//
//
//		/* visualize the CL&CD of the samples */
//
//		std::string file_name_for_plot = "samples.png";
//
//		std::string python_command = "python -W ignore plot_cdcl.py "+ all_data_file+ " "+
//				std::to_string(number_of_initial_samples)+ " " +
//				file_name_for_plot;
//
//		FILE* in = popen(python_command.c_str(), "r");
//		fprintf(in, "\n");
//
//
//		mat cl_kriging_data;
//		mat cd_kriging_data;
//		mat area_kriging_data;
//
//		/* load data for Kriging training */
//		cl_kriging_data.load(cl_kriging_input_file.c_str(), csv_ascii);
//		cd_kriging_data.load(cd_kriging_input_file.c_str(), csv_ascii);
//		area_kriging_data.load(area_kriging_input_file.c_str(), csv_ascii);
//
//		//		cl_kriging_data.print();
//
//
//
//
//		/* normalized input variables */
//		mat X(optimization_data.n_rows,number_of_design_variables);
//
//		/*dimension of R */
//		int dimension_of_R = optimization_data.n_rows;
//
//
//		/* set the input data matrix X */
//		for(int i=0; i<number_of_design_variables;i++){
//
//			X.col(i) = optimization_data.col(i);
//
//		}
//
//
//		/* find minimum and maximum of the columns of X */
//
//		vec x_max(number_of_design_variables);
//		x_max.fill(0.0);
//
//		vec x_min(number_of_design_variables);
//		x_min.fill(0.0);
//
//		for (int i = 0; i < number_of_design_variables; i++) {
//			x_max(i) = X.col(i).max();
//			x_min(i) = X.col(i).min();
//
//		}
//
//		//	printf("maximum = \n");
//		//	x_max.print();
//
//		//	printf("minimum = \n");
//		//	x_min.print();
//
//		/* normalize data matrix */
//
//		for (unsigned int i = 0; i < X.n_rows; i++) {
//			for (int j = 0; j < number_of_design_variables; j++) {
//				X(i, j) = (1.0/number_of_design_variables)*(X(i, j) - x_min(j)) / (x_max(j) - x_min(j));
//			}
//		}
//
//
//		/* calculate average distance in p-norm */
//
//		double average_distance = 0;
//
//		int number_of_distances_computed = 0;
//		for(unsigned i=0;i<X.n_rows;i++){
//
//			for(unsigned j=i+1;j<X.n_rows;j++){
//				double distance = norm(( X.row(i) - X.row(j)), number_of_design_variables);
//				average_distance+= distance;
//				number_of_distances_computed++;
//
//			}
//
//
//
//
//		}
//
//		average_distance = average_distance/number_of_distances_computed;
//
//		printf("Average distance of the data points in p-norm: %10.7f\n",average_distance );
//
//
//		/* set tolerance */
//		double tolerance = average_distance/100.0;
//
//		printf("tolerance is set to: %10.7f\n",tolerance );
//
//
//
//
//
//
//
//		//		exit(1);
//
//
//
//
//
//		//		optimization_data.print();
//
//
//
//
//
//
//
//
//
//
//		/* correlation matrices for cd,cl and area */
//
//		mat R_CL(dimension_of_R,dimension_of_R);
//		mat R_CD(dimension_of_R,dimension_of_R);
//		mat R_area(dimension_of_R,dimension_of_R);
//
//		mat U_CL(dimension_of_R,dimension_of_R);
//		mat U_CD(dimension_of_R,dimension_of_R);
//		mat U_area(dimension_of_R,dimension_of_R);
//
//
//		mat L_CL(dimension_of_R,dimension_of_R);
//		mat L_CD(dimension_of_R,dimension_of_R);
//		mat L_area(dimension_of_R,dimension_of_R);
//
//
//		/* assign ys vectors for CL,CD and area */
//		vec ys_CL =   optimization_data.col(number_of_design_variables);
//		vec ys_CD =   optimization_data.col(number_of_design_variables+1);
//		vec ys_area = optimization_data.col(number_of_design_variables+2);
//
//		/* vector of ones */
//		vec I = ones(dimension_of_R);
//
//		/* train surrogate models for CL, CD and area*/
//
//		train_kriging_response_surface(cl_kriging_input_file,
//				cl_kriging_hyperparameters_file,
//				LINEAR_REGRESSION_ON,
//				lin_reg_param_CL,
//				kriging_param_CL,
//				reg_param,
//				max_number_of_function_calculations_training,
//				CSV_ASCII);
//
//		train_kriging_response_surface(cd_kriging_input_file,
//				cd_kriging_hyperparameters_file,
//				LINEAR_REGRESSION_ON,
//				lin_reg_param_CD,
//				kriging_param_CD,
//				reg_param,
//				max_number_of_function_calculations_training,
//				CSV_ASCII);
//
//		train_kriging_response_surface(area_kriging_input_file,
//				area_kriging_hyperparameters_file,
//				LINEAR_REGRESSION_ON,
//				lin_reg_param_area,
//				kriging_param_area,
//				reg_param,
//				max_number_of_function_calculations_training,
//				CSV_ASCII);
//
//
//
//		/*update y vectors according to linear regression*/
//
//		mat augmented_X(dimension_of_R, number_of_design_variables + 1);
//
//		for (int i = 0; i < dimension_of_R; i++) {
//			for (int j = 0; j <= number_of_design_variables; j++) {
//				if (j == 0)
//					augmented_X(i, j) = 1.0;
//				else
//					augmented_X(i, j) = X(i, j - 1);
//
//			}
//		}
//
//		vec ys_reg_cl = augmented_X * lin_reg_param_CL;
//		vec ys_reg_cd = augmented_X * lin_reg_param_CD;
//		vec ys_reg_area = augmented_X * lin_reg_param_area;
//
//
//		ys_CL = ys_CL - ys_reg_cl;
//		ys_CD = ys_CD - ys_reg_cd;
//		ys_area = ys_area - ys_reg_area;
//
//
//
//
//		theta_CL = kriging_param_CL.head(number_of_design_variables);
//		gamma_CL = kriging_param_CL.tail(number_of_design_variables);
//
//		theta_CD = kriging_param_CD.head(number_of_design_variables);
//		gamma_CD = kriging_param_CD.tail(number_of_design_variables);
//
//
//		theta_area = kriging_param_area.head(number_of_design_variables);
//		gamma_area = kriging_param_area.tail(number_of_design_variables);
//
//
//
//		//		printf("theta CL:\n");
//		//		theta_CL.print();
//		//
//		//		printf("gamma CL:\n");
//		//		gamma_CL.print();
//		//
//		//
//		//		printf("theta CD:\n");
//		//		theta_CD.print();
//		//
//		//
//		//		printf("gamma CD:\n");
//		//		gamma_CD.print();
//		//
//		//		printf("theta area:\n");
//		//		theta_area.print();
//		//
//		//
//		//		printf("gamma area:\n");
//		//		gamma_area.print();
//
//
//
//		//		X.print();
//		//		printf("computing R for CL\n");
//		compute_R_matrix(theta_CL,
//				gamma_CL,
//				reg_param,
//				R_CL,
//				X);
//
//		//		R_CL.print();
//
//
//
//
//
//
//		cholesky_return = chol(U_CL, R_CL);
//
//		if (cholesky_return == 0) {
//			printf("Ill conditioned correlation matrix\n");
//			exit(-1);
//		}
//
//		L_CL = trans(U_CL);
//
//		vec R_inv_ys_CL(dimension_of_R);
//		vec R_inv_I_CL(dimension_of_R);
//
//
//
//		solve_linear_system_by_Cholesky(U_CL, L_CL, R_inv_ys_CL, ys_CL); /* solve R x = ys */
//		solve_linear_system_by_Cholesky(U_CL, L_CL, R_inv_I_CL, I);   /* solve R x = I */
//
//
//
//
//		beta0_CL = (1.0/dot(I,R_inv_I_CL)) * (dot(I,R_inv_ys_CL));
//		//	printf("beta0= %20.15f\n",beta0);
//
//		vec ys_min_betaI_CL = ys_CL-beta0_CL*I;
//
//		vec R_inv_ys_min_beta_CL(dimension_of_R);
//
//
//
//		/* solve R x = ys-beta0*I */
//		solve_linear_system_by_Cholesky(U_CL, L_CL, R_inv_ys_min_beta_CL, ys_min_betaI_CL);
//
//
//		//		ssqr_CL = (1.0 / dimension_of_R) * dot(ys_min_betaI_CL, R_inv_ys_min_beta_CL);
//
//
//
//		//		printf("computing R for CD\n");
//
//		compute_R_matrix(theta_CD,
//				gamma_CD,
//				reg_param,
//				R_CD,
//				X);
//
//
//
//		cholesky_return = chol(U_CD, R_CD);
//
//		if (cholesky_return == 0) {
//			printf("Ill conditioned correlation matrix\n");
//			exit(-1);
//		}
//
//		L_CD = trans(U_CD);
//
//		vec R_inv_ys_CD(dimension_of_R);
//		vec R_inv_I_CD(dimension_of_R);
//
//
//
//		solve_linear_system_by_Cholesky(U_CD, L_CD, R_inv_ys_CD, ys_CD); /* solve R x = ys */
//		solve_linear_system_by_Cholesky(U_CD, L_CD, R_inv_I_CD, I);   /* solve R x = I */
//
//
//
//
//		beta0_CD = (1.0/dot(I,R_inv_I_CD)) * (dot(I,R_inv_ys_CD));
//		//	printf("beta0= %20.15f\n",beta0);
//
//		vec ys_min_betaI_CD = ys_CD-beta0_CD*I;
//
//		vec R_inv_ys_min_beta_CD(dimension_of_R);
//
//
//
//		/* solve R x = ys-beta0*I */
//		solve_linear_system_by_Cholesky(U_CD, L_CD, R_inv_ys_min_beta_CD, ys_min_betaI_CD);
//
//
//		ssqr_CD = (1.0 / dimension_of_R) * dot(ys_min_betaI_CD, R_inv_ys_min_beta_CD);
//
//
//		//		printf("computing R for area\n");
//
//		compute_R_matrix(theta_area,
//				gamma_area,
//				reg_param,
//				R_area,
//				X);
//
//
//
//		cholesky_return = chol(U_area, R_area);
//
//		if (cholesky_return == 0) {
//			printf("Ill conditioned correlation matrix\n");
//			exit(-1);
//		}
//
//		L_area = trans(U_area);
//
//		vec R_inv_ys_area(dimension_of_R);
//		vec R_inv_I_area(dimension_of_R);
//
//
//
//		solve_linear_system_by_Cholesky(U_area, L_area, R_inv_ys_area, ys_area); /* solve R x = ys */
//		solve_linear_system_by_Cholesky(U_area, L_area, R_inv_I_area, I);   /* solve R x = I */
//
//
//
//
//		beta0_area = (1.0/dot(I,R_inv_I_area)) * (dot(I,R_inv_ys_area));
//		//	printf("beta0= %20.15f\n",beta0);
//
//		vec ys_min_betaI_area = ys_area-beta0_area*I;
//
//		vec R_inv_ys_min_beta_area(dimension_of_R);
//
//
//
//		/* solve R x = ys-beta0*I */
//		solve_linear_system_by_Cholesky(U_area, L_area, R_inv_ys_min_beta_area, ys_min_betaI_area);
//
//
//		//		ssqr_area = (1.0 / dimension_of_R) * dot(ys_min_betaI_area, R_inv_ys_min_beta_area);
//
//
//		printf("beta0 for CL : %15.10f\n",beta0_CL);
//		printf("beta0 for CD : %15.10f\n",beta0_CD);
//		printf("beta0 for Area : %15.10f\n",beta0_area);
//
//
//
//
//
//
//
//		printf("*****************************************************\n");
//
//
//
//
//
//
//		/* try all the designs in the data */
//
//		double min_J = LARGE;
//		double best_sample_avg_cd=0.0;
//		double best_sample_worst_cl=0.0;
//		double best_sample_worst_area=0.0;
//		int best_sample_index = -1;
//
//
//		/* for each sample in the data */
//		for(int sample_index =0; sample_index <optimization_data.n_rows;sample_index++ ){
//
//			//			printf("sample index = %d\n",sample_index);
//			vec dv_simulation(number_of_design_variables);
//			vec dv_simulation0(number_of_design_variables);
//
//			for(unsigned i=0;i<number_of_design_variables;i++){
//				dv_simulation(i) = optimization_data(sample_index,i);
//			}
//			dv_simulation0 = dv_simulation;
//
//			//			dv_simulation.print();
//
//
//			double average_cd=0.0;
//			double worst_case_cl=LARGE;
//			double worst_case_area=LARGE;
//			double cl_tilde =0.0,cd_tilde=0.0, area_tilde=0.0;
//
//
//
//			/* inner MC loop */
//			for(int inner_mc_iter=0; inner_mc_iter<200;inner_mc_iter++ ){
//
//				//				printf("mc inner loop iter = %d\n",inner_mc_iter);
//
//				vec dv_with_noise(number_of_design_variables);
//				rowvec dv_tilde_with_noise(number_of_design_variables);
//				dv_simulation = dv_simulation0;
//
//
//				/* generate a random design and normalize it*/
//				for(int j= 0; j< number_of_design_variables; j++){
//					double number = distribution(generator);
//					double perturbation  = gaussian_noise_level* number;
//					//						printf("generated random number = %10.7f\n",number);
//					//						printf("perturbation            = %10.7f\n",perturbation);
//					dv_with_noise(j)= dv_simulation(j)+perturbation ;
//					//						printf("design parameter = %15.10f (unperturbed) %15.10f (perturbed) \n",dv_simulation(j),dv_with_noise(j));
//
//					dv_tilde_with_noise(j) = (1.0/number_of_design_variables)*(dv_with_noise(j) - x_min(j)) / (x_max(j) - x_min(j));
//
//				}
//
//				//#pragma omp critical
//				//					{
//				//					printf("dv_tilde = ");
//				//					dv_tilde.print();
//				//
//				//					printf("dv_tilde perturbed= ");
//				//					dv_tilde_with_noise.print();
//				//					}
//
//				/* estimate the area of the new design by the surrogate model */
//
//				//			printf("Approximating area...\n");
//				area_tilde = 0.0;
//				area_tilde = calculate_f_tilde(dv_tilde_with_noise,
//						X,
//						beta0_area,
//						lin_reg_param_area,
//						R_inv_ys_min_beta_area,
//						kriging_param_area);
//
//				if(area_tilde < worst_case_area) worst_case_area = area_tilde;
//
//
//				//				printf("Area tilde = %10.7f\n",area_tilde );
//
//				/* estimate the cl of the new design by the surrogate model */
//
//				//			printf("Approximating CL...\n");
//				cl_tilde = 0.0;
//				cl_tilde = calculate_f_tilde(dv_tilde_with_noise,
//						X,
//						beta0_CL,
//						lin_reg_param_CL,
//						R_inv_ys_min_beta_CL,
//						kriging_param_CL);
//
//				if(cl_tilde < worst_case_cl) worst_case_cl = cl_tilde;
//
//
//				//				printf("CL tilde = %10.7f\n",cl_tilde );
//
//				//			printf("Approximating CD...\n");
//				cd_tilde=0.0;
//
//
//				/* estimate cd with uncertainty by the surrogate model */
//
//
//				cd_tilde = calculate_f_tilde(dv_tilde_with_noise,
//						X,
//						beta0_CD,
//						lin_reg_param_CD,
//						R_inv_ys_min_beta_CD,
//						kriging_param_CD);
//
//				average_cd+= cd_tilde;
//
//				//				printf("CD tilde = %10.7f\n",cd_tilde );
//				//				printf("average CD tilde = %10.7f\n",average_cd );
//
//
//
//				//			printf("Comparison between simulation and surrogate model\n");
//				//			printf("%10.7f %10.7f %10.7f %10.7f %10.7f %10.7f %10.7f %10.7f %10.7f\n",
//				//					cl_tilde,cl_exact, fabs(cl_tilde-cl_exact)/cl_tilde,
//				//					cd_tilde,cd_exact, fabs(cd_tilde-cd_exact)/cd_tilde,
//				//					area_tilde,area_exact, fabs(area_tilde-area_exact)/area_tilde);
//
//
//			} /* end of inner MC loop */
//
//
//			average_cd += optimization_data(sample_index,number_of_design_variables+1);
//			average_cd = average_cd/(201);
//			//			printf("average CD tilde = %10.7f\n",average_cd );
//			double J = average_cd;
//			if (worst_case_cl  <   CL_constraint)       J= LARGE;
//			if (worst_case_area < Area_constraint)      J= LARGE;
//
//
//			printf("%d %10.7f %10.7f %10.7f %10.7f\n",sample_index, optimization_data(sample_index,number_of_design_variables+1),
//					average_cd, worst_case_cl, worst_case_area);
//
//
//			//			printf("J = %10.7f\n",J );
//			if(J < min_J){
//
//				min_J = J;
//				best_sample_index = sample_index;
//				best_sample_avg_cd=average_cd;
//				best_sample_worst_cl=worst_case_cl;
//				best_sample_worst_area=worst_case_area;
//
//			}
//
//
//		}
//
//
//		printf("Best sample in the data set = \n");
//		printf("Design No = %d\n",best_sample_index);
//		rowvec best_design = optimization_data.row(best_sample_index);
//
//		best_design(number_of_design_variables)   = best_sample_worst_cl ;
//		best_design(number_of_design_variables+1) = best_sample_avg_cd ;
//		best_design(number_of_design_variables+2) = best_sample_worst_area ;
//
//		best_design.print();
//		printf("Average CD = %10.7f\n",best_sample_avg_cd );
//		printf("Worst case CL = %10.7f\n",best_sample_worst_cl );
//		printf("Worst case Area = %10.7f\n",best_sample_worst_area);
//		printf("*****************************************************\n");
//
//
//
//		printf("\n");
//		printf("Number of function evaluations = %d\n", number_of_function_evaluations);
//		printf("Number of gradient evaluations = %d\n", number_of_gradient_evaluations);
//		printf("\n");
//
//		if(number_of_function_evaluations >= max_number_of_function_evaluations){
//			printf("Terminating RoDeO...\n");
//			exit(1);
//
//		}
//
//
//
//
//
//
//
//
//
//
//
//		/*find the best design after gradient sampling (without checking robust design criteria)*/
//
//		index_best_design = -1;
//		min_CD = LARGE;
//		for(unsigned int i=0;i<optimization_data.n_rows;i++){
//			if(optimization_data(i,number_of_design_variables+1) < min_CD ){
//				if(optimization_data(i,number_of_design_variables) >= CL_constraint &&
//						optimization_data(i,number_of_design_variables+2) >= Area_constraint){
//					min_CD = optimization_data(i,number_of_design_variables+1);
//					index_best_design=i;
//
//				}
//
//			}
//
//		}
//
//
//
//		best_design = optimization_data.row(index_best_design);
//		double sample_min = best_design(number_of_design_variables+1);
//
//		printf("Best design in the data set up to now (without robustness criteria):\n");
//		printf("Design No = %d\n",index_best_design);
//		best_design.print();
//		printf("Sample min = %10.7f\n",sample_min);
//
//
//
//
//
//
//
//
//
//
//		/* Monte Carlo Loop */
//
//
//		double worst_EI_sample = -LARGE;
//		int worst_EI_sample_index =0;
//		std::vector<MC_design> best_designs(number_of_samples_MC);
//
//
//		std::vector<MC_design>::iterator it;
//
//		/* init the best designs vector */
//		for (it = best_designs.begin() ; it != best_designs.end(); ++it){
//
//			it->EI = -LARGE;
//			it->dv.resize( number_of_design_variables );
//			it->dv_original.resize( number_of_design_variables );
//			it->CL = 0.0;
//			it->CD = 0.0;
//			it->Area = 0.0;
//			it->J = 0.0;
//
//		}
//
//
//
//
//#pragma omp parallel
//		{
//			for(int mc_iter=0; mc_iter< max_number_of_mc_iterations;mc_iter++){
//
//				if(mc_iter % 10000 == 0 && omp_get_thread_num() == 0){
//
//					printf("MC  outer iteration %d\n",mc_iter);
//
//				}
//
//
//				//			printf("MC iter = %d\n",mc_iter);
//				vec dv_simulation(number_of_design_variables);
//				vec dv_simulation0(number_of_design_variables);
//				rowvec dv_tilde(number_of_design_variables);
//
//
//
//				/* generate a random design and normalize it*/
//				for(int j= 0; j< number_of_design_variables; j++){
//					dv_simulation(j)= RandomDouble(lower_bound_dv,upper_bound_dv);
//					dv_tilde(j) = (1.0/number_of_design_variables)*(dv_simulation(j)-x_min(j)) / (x_max(j)-x_min(j));
//
//				}
//
//				dv_simulation0 = dv_simulation;
//
//				double average_cd=0.0;
//				double worst_case_cl=LARGE;
//				double worst_case_area=LARGE;
//				double cl_tilde =0.0,cd_tilde=0.0, area_tilde=0.0;
//
//
//
//				vec dv_with_noise(number_of_design_variables);
//				rowvec dv_tilde_with_noise(number_of_design_variables);
//
//				/* inner MC loop */
//				for(int inner_mc_iter=0; inner_mc_iter<max_number_of_inner_mc_iterations;inner_mc_iter++ ){
//
//					//					printf("mc inner loop iter = %d\n",inner_mc_iter);
//
//
//
//					/* retrieve the values of dv */
//					dv_simulation = dv_simulation0;
//					/* generate a random design and normalize it*/
//					for(int j= 0; j< number_of_design_variables; j++){
//						double number = distribution(generator);
//						double perturbation  = gaussian_noise_level* number;
//						//						printf("generated random number = %10.7f\n",number);
//						//						printf("perturbation            = %10.7f\n",perturbation);
//						dv_with_noise(j)= dv_simulation(j)+perturbation ;
//						//						printf("design parameter = %15.10f (unperturbed) %15.10f (perturbed) \n",dv_simulation(j),dv_with_noise(j));
//
//						dv_tilde_with_noise(j) = (1.0/number_of_design_variables)*(dv_with_noise(j) - x_min(j)) / (x_max(j) - x_min(j));
//
//					}
//
//					//#pragma omp critical
//					//					{
//					//					printf("dv_tilde = ");
//					//					dv_tilde.print();
//					//
//					//					printf("dv_tilde perturbed= ");
//					//					dv_tilde_with_noise.print();
//					//					}
//
//					/* estimate the area of the new design by the surrogate model */
//
//					//			printf("Approximating area...\n");
//					area_tilde = 0.0;
//					area_tilde = calculate_f_tilde(dv_tilde_with_noise,
//							X,
//							beta0_area,
//							lin_reg_param_area,
//							R_inv_ys_min_beta_area,
//							kriging_param_area);
//
//					if(area_tilde < worst_case_area) worst_case_area = area_tilde;
//
//
//					//					printf("Area tilde = %10.7f\n",area_tilde );
//
//					/* estimate the cl of the new design by the surrogate model */
//
//					//			printf("Approximating CL...\n");
//					cl_tilde = 0.0;
//					cl_tilde = calculate_f_tilde(dv_tilde_with_noise,
//							X,
//							beta0_CL,
//							lin_reg_param_CL,
//							R_inv_ys_min_beta_CL,
//							kriging_param_CL);
//
//					if(cl_tilde < worst_case_cl) worst_case_cl = cl_tilde;
//
//
//					//					printf("CL tilde = %10.7f\n",cl_tilde );
//
//					//			printf("Approximating CD...\n");
//					cd_tilde=0.0;
//
//
//					/* estimate cd with uncertainty by the surrogate model */
//
//
//					cd_tilde = calculate_f_tilde(dv_tilde_with_noise,
//							X,
//							beta0_CD,
//							lin_reg_param_CD,
//							R_inv_ys_min_beta_CD,
//							kriging_param_CD);
//
//					average_cd+= cd_tilde;
//
//					//					printf("CD tilde = %10.7f\n",cd_tilde );
//
//
//
//					//			printf("Comparison between simulation and surrogate model\n");
//					//			printf("%10.7f %10.7f %10.7f %10.7f %10.7f %10.7f %10.7f %10.7f %10.7f\n",
//					//					cl_tilde,cl_exact, fabs(cl_tilde-cl_exact)/cl_tilde,
//					//					cd_tilde,cd_exact, fabs(cd_tilde-cd_exact)/cd_tilde,
//					//					area_tilde,area_exact, fabs(area_tilde-area_exact)/area_tilde);
//
//
//
//
//
//
//
//
//
//
//
//				} /* end of inner MC loop */
//
//
//
//
//
//				double sigma_sqr_CD=0.0;
//				cd_tilde =0.0;
//
//				calculate_f_tilde_and_ssqr(
//						dv_tilde,
//						X,
//						beta0_CD,
//						ssqr_CD,
//						lin_reg_param_CD,
//						R_inv_ys_min_beta_CD,
//						R_inv_I_CD,
//						I,
//						kriging_param_CD,
//						U_CD,
//						L_CD,
//						&cd_tilde,
//						&sigma_sqr_CD);
//
//				average_cd+= cd_tilde;
//
//
//				cl_tilde = 0.0;
//				cl_tilde = calculate_f_tilde(dv_tilde,
//						X,
//						beta0_CL,
//						lin_reg_param_CL,
//						R_inv_ys_min_beta_CL,
//						kriging_param_CL);
//
//				if(cl_tilde < worst_case_cl) worst_case_cl = cl_tilde;
//
//
//				area_tilde = 0.0;
//				area_tilde = calculate_f_tilde(dv_tilde,
//						X,
//						beta0_area,
//						lin_reg_param_area,
//						R_inv_ys_min_beta_area,
//						kriging_param_area);
//
//				if(area_tilde < worst_case_area) worst_case_area = area_tilde;
//
//
//
//				//#pragma omp critical
//				//					{
//				//				printf("worst_case_cl = %10.7f\n",worst_case_cl);
//				//				printf("worst_case_area = %10.7f\n",worst_case_area);
//				//				printf("avg. cd = %10.7f\n", average_cd/(max_number_of_inner_mc_iterations+1));
//				//					}
//
//
//				average_cd  = average_cd/(max_number_of_inner_mc_iterations+1);
//
//
//
//
//
//				double	standart_error = sqrt(sigma_sqr_CD)	;
//
//				//						printf("standart error (CD) = %10.7f\n", standart_error );
//
//				double EI;
//
//				if(standart_error!=0.0){
//					double	EIfac = (sample_min - average_cd)/standart_error;
//
//					/* calculate the Expected Improvement value */
//					EI = (sample_min - average_cd)*cdf(EIfac,0.0,1.0)+standart_error*pdf(EIfac,0.0,1.0);
//				}
//				else{
//					EI =0.0;
//
//				}
//
//				if (worst_case_cl  <   CL_constraint-0.001)       EI= 0.0;
//				if (worst_case_area < Area_constraint)     		  EI= 0.0;
//
//				double J = average_cd;
//				if (worst_case_cl  <   CL_constraint-0.01)        J= LARGE;
//				if (worst_case_area < Area_constraint-0.001)      J= LARGE;
//
//				/* assign the properties of the current design */
//
//				MC_design current_design;
//				current_design.dv = dv_tilde;
//				current_design.dv_original = dv_simulation;
//				current_design.J = J;
//				current_design.CL = worst_case_cl;
//				current_design.CD = average_cd ;
//				current_design.Area = worst_case_area;
//				current_design.EI = EI;
//
//
//				if(EI > worst_EI_sample){
//#pragma omp critical
//					{
//						printf("tread %d found a possible better design with EI = %15.10f\n",omp_get_thread_num(), EI);
//
//						int flag_points_are_too_close = 0;
//
//						for( int i=0;i< number_of_samples_MC; i++){
//							//							printf("here\n");
//							double dist = norm((dv_tilde-best_designs[i].dv), number_of_design_variables );
//							//							printf("here2\n");
//							//							printf("dist = %10.7f\n", dist);
//							if ( dist <  10.0*tolerance){
//								printf("dv1 and dv2 are too close\n");
//								for(int k=0; k<number_of_design_variables;k++ ){
//									printf("%10.7f %10.7f\n",dv_tilde(k),best_designs[i].dv(k));
//								}
//								flag_points_are_too_close = i;
//								break;
//							}
//
//						} /* end of for */
//
//						if(flag_points_are_too_close == 0){ /* if the new sample is not too close to other */
//
//
//							best_designs[worst_EI_sample_index] = current_design;
//							std::vector<MC_design>::iterator it;
//
//							printf("EI values of current 20 best designs = \n");
//							for (it = best_designs.begin() ; it != best_designs.end(); ++it){
//								printf("%15.10f ", it->EI);
//
//
//
//							}
//							printf("\n");
//
//
//
//						}
//
//
//						else{
//
//							/* if the EI value is better exchange both */
//							if(current_design.EI > best_designs[flag_points_are_too_close].EI){
//								best_designs[flag_points_are_too_close] = current_design;
//
//
//							}
//
//
//
//						} /* else */
//
//
//
//					} /* critical */
//				} /* end of if EI > worst_EI_sample */
//
//
//
//
//
//
//
//
//
//#pragma omp master
//				{
//					/* update the minimum of new samples */
//
//					if (best_designs.size() > 1){
//						worst_EI_sample = LARGE;
//						worst_EI_sample_index = -1;
//						for ( int j =0; j<  number_of_samples_MC; j++){
//							if (best_designs[j].EI < worst_EI_sample){
//								worst_EI_sample = best_designs[j].EI;
//								worst_EI_sample_index = j;
//
//							}
//
//						}
//
//						//											printf("worst design EI = %10.7f\n",worst_EI_sample);
//						//											printf("worst design EI index = %d\n", worst_EI_sample_index);
//
//					}
//
//
//
//				} // end of master
//
//
//
//
//			} /* end of MC loop */
//
//		} /* end of parallel */
//
//
//		/* simulate the best designs */
//		for (it = best_designs.begin() ; it != best_designs.end(); ++it){
//			printf("\n\n");
//
//			printf("simulating the best designs\n");
//			//			it->dv_original.print();
//
//
//			double cl_exact,cd_exact,area_exact;
//
//			call_SU2_CFD_Solver(it->dv_original,
//					cl_exact,
//					cd_exact,
//					area_exact);
//			number_of_function_evaluations++;
//
//			printf("EI = %10.7f\n", it->EI);
//			printf("J = %10.7f\n", it->J);
//			printf("CL   (surrogate) = %10.7f  CL   (simulation) = %10.7f\n", it->CL, cl_exact);
//			printf("CD   (surrogate) = %10.7f  CD   (simulation) = %10.7f\n", it->CD, cd_exact);
//			printf("Area (surrogate) = %10.7f  Area (simulation) = %10.7f\n", it->Area, area_exact);
//
//
//			printf("\n\n");
//
//
//
//			/* insert a row to the data matrix*/
//			optimization_data.insert_rows( dimension_of_R, 1 );
//			for(int i=0;i<number_of_design_variables;i++){
//				optimization_data(dimension_of_R,i) = it->dv_original(i);
//			}
//			optimization_data(dimension_of_R,number_of_design_variables)   = cl_exact;
//			optimization_data(dimension_of_R,number_of_design_variables+1) = cd_exact;
//			optimization_data(dimension_of_R,number_of_design_variables+2) = area_exact;
//
//
//			/* insert a row to the data matrix X*/
//			X.insert_rows( dimension_of_R, 1 );
//			for(int i=0;i<number_of_design_variables;i++){
//				X(dimension_of_R,i) = it->dv(i);
//			}
//
//
//			/* insert a row to the cl kriging data matrix*/
//			cl_kriging_data.insert_rows( dimension_of_R, 1 );
//			for(int i=0;i<number_of_design_variables;i++){
//				cl_kriging_data(dimension_of_R,i) = it->dv_original(i);
//			}
//			cl_kriging_data(dimension_of_R,number_of_design_variables) = cl_exact;
//
//
//			/* insert a row to the cd kriging data matrix*/
//			cd_kriging_data.insert_rows( dimension_of_R, 1 );
//			for(int i=0;i<number_of_design_variables;i++){
//				cd_kriging_data(dimension_of_R,i) = it->dv_original(i);
//			}
//			cd_kriging_data(dimension_of_R,number_of_design_variables) = cd_exact;
//
//			/* insert a row to the area kriging data matrix*/
//			area_kriging_data.insert_rows( dimension_of_R, 1 );
//			for(int i=0;i<number_of_design_variables;i++){
//				area_kriging_data(dimension_of_R,i) = it->dv_original(i);
//			}
//			area_kriging_data(dimension_of_R,number_of_design_variables) = area_exact;
//
//			/* save updated data */
//
//			optimization_data.save(all_data_file.c_str(), csv_ascii);
//			cl_kriging_data.save(cl_kriging_input_file.c_str(), csv_ascii);
//			cd_kriging_data.save(cd_kriging_input_file.c_str(), csv_ascii);
//			area_kriging_data.save(area_kriging_input_file.c_str(), csv_ascii);
//
//
//			/* update dimension of R */
//			dimension_of_R = optimization_data.n_rows;
//
//
//
//		}
//
//
//
//
//		/* gradient assisted sampling */
//
//		if(max_number_of_linesearches >0){
//
//			/*find the best design */
//
//
//
//			index_best_design = -1;
//			min_CD = LARGE;
//
//			for(unsigned int i=0;i<optimization_data.n_rows;i++){
//				if(optimization_data(i,number_of_design_variables+1) < min_CD ){
//					if(optimization_data(i,number_of_design_variables) >= CL_constraint &&
//							optimization_data(i,number_of_design_variables+2) >= Area_constraint &&
//							is_in_the_list(i,indices_of_ls_designs) == -1){
//						min_CD = optimization_data(i,number_of_design_variables+1);
//						index_best_design=i;
//
//
//					}
//
//
//
//				}
//
//
//			}
//
//
//			printf("Starting point for the gradient assisted sampling:\n");
//			printf("Design No = %d\n",index_best_design);
//
//
//			best_design = optimization_data.row(index_best_design);
//			best_design.print();
//
//
//
//
//			for(int i=0;i<number_of_design_variables;i++){
//				dv(i) = best_design(i);
//
//
//
//			}
//
//			//		dv.print();
//
//
//
//			//		exit(1);
//
//
//
//			stepsize = stepsize0;
//			/* CD value of the best design */
//			double f0 = best_design(number_of_design_variables+1);
//
//
//
//			call_SU2_Adjoint_Solver(dv, grad_dv,CL,CD,area);
//			number_of_gradient_evaluations++;
//
//			printf("CL = %10.7f\n",CL);
//			printf("CD = %10.7f\n",CD);
//			printf("area = %10.7f\n",area);
//
//			printf("gradient vector:\n");
//			grad_dv.print();
//
//
//			dv_save     = dv;
//			dv_savenorm = dv;
//
//			/* normalize dv_savenorm */
//			for (int j = 0; j < number_of_design_variables; j++) {
//				dv_savenorm(j) = (1.0/number_of_design_variables)
//						    																																																																																																																				 *(dv_savenorm(j) - x_min(j)) / (x_max(j) - x_min(j));
//			}
//
//
//			for(int ls_index; ls_index< max_number_of_linesearches; ls_index++ ){
//
//				printf("ls = %d\n",ls_index);
//
//
//
//
//				/* modify the design vector */
//				dv = dv - stepsize*grad_dv;
//
//				printf("dv dv0 |dv-dv0|:\n");
//				for(unsigned int i=0; i<dv.size();i++){
//					printf("%10.7f %10.7f %10.7f\n",dv(i),dv_save(i),fabs(dv(i)- dv_save(i)   ) );
//				}
//
//
//
//				dvnorm = dv;
//				/* normalize dvnorm*/
//				for (int j = 0; j < number_of_design_variables; j++) {
//					dvnorm(j) = (1.0/number_of_design_variables)
//									        																																																																																																																				*(dvnorm(j) - x_min(j)) / (x_max(j) - x_min(j));
//				}
//
//
//				/* compute the distance between dv and dv_save in normalized coordinates */
//
//				double normdv = norm((dvnorm-dv_savenorm),number_of_design_variables);
//				/* if the distance is too small break */
//				if ( normdv < tolerance) {
//					printf("design change is too small...\n");
//					printf("norm = %15.10f\n",normdv);
//
//					break;
//				}
//
//
//				call_SU2_CFD_Solver(dv,CL,CD,area);
//				number_of_function_evaluations++;
//
//
//				printf("CL = %10.7f\n",CL);
//				printf("CD = %10.7f\n",CD);
//				printf("area = %10.7f\n",area);
//
//
//				/* insert a row to the data matrix*/
//				optimization_data.insert_rows( dimension_of_R, 1 );
//				for(int i=0;i<number_of_design_variables;i++){
//					optimization_data(dimension_of_R,i) = dv(i);
//				}
//				optimization_data(dimension_of_R,number_of_design_variables)   = CL;
//				optimization_data(dimension_of_R,number_of_design_variables+1) = CD;
//				optimization_data(dimension_of_R,number_of_design_variables+2) = area;
//
//
//				/* insert a row to the data matrix X*/
//				X.insert_rows( dimension_of_R, 1 );
//				for(int i=0;i<number_of_design_variables;i++){
//					X(dimension_of_R,i) = dvnorm(i);
//				}
//
//
//				/* insert a row to the cl kriging data matrix*/
//				cl_kriging_data.insert_rows( dimension_of_R, 1 );
//				for(int i=0;i<number_of_design_variables;i++){
//					cl_kriging_data(dimension_of_R,i) = dv(i);
//				}
//				cl_kriging_data(dimension_of_R,number_of_design_variables) = CL;
//
//
//				/* insert a row to the cd kriging data matrix*/
//				cd_kriging_data.insert_rows( dimension_of_R, 1 );
//				for(int i=0;i<number_of_design_variables;i++){
//					cd_kriging_data(dimension_of_R,i) = dv(i);
//				}
//				cd_kriging_data(dimension_of_R,number_of_design_variables) = CD;
//
//				/* insert a row to the area kriging data matrix*/
//				area_kriging_data.insert_rows( dimension_of_R, 1 );
//				for(int i=0;i<number_of_design_variables;i++){
//					area_kriging_data(dimension_of_R,i) = dv(i);
//				}
//				area_kriging_data(dimension_of_R,number_of_design_variables) = area;
//
//				/* save updated data */
//
//				optimization_data.save(all_data_file.c_str(), csv_ascii);
//				cl_kriging_data.save(cl_kriging_input_file.c_str(), csv_ascii);
//				cd_kriging_data.save(cd_kriging_input_file.c_str(), csv_ascii);
//				area_kriging_data.save(area_kriging_input_file.c_str(), csv_ascii);
//
//
//				/* update dimension of R */
//				dimension_of_R = optimization_data.n_rows;
//
//
//				/* if no improvement can be made reduce the stepsize */
//				if(CD < f0 && CL >=CL_constraint && area >=Area_constraint ) {
//					printf("A better design is found\n");
//					break;
//				}
//
//				else {
//					stepsize = stepsize/2.0;
//					dv = dv_save;
//				}
//
//
//
//			} /* end of gradient sampling loop*/
//
//
//			indices_of_ls_designs.push_back(index_best_design);
//
//
//		} /* end of if */
//
//
//
//
//		printf("number of function evaluations = %d\n",number_of_function_evaluations);
//
//
//
//
//
//
//
//
//
//	} /* end of while(1) */
//
//
//
//
//}


/** RInitial Data Acquisition using the adjoint solver.
 *
 * @param[in] number_of_initial_samples
 * @param[in] number_of_design_variables
 * @param[in] variable_activity
 *
 *
 */



//void initial_data_acquisitionGEK(GEKSamplingData &sampling_plan){
//
//
//	unsigned int number_of_design_variables = sampling_plan.size_of_dv;
//
//	unsigned int number_of_active_design_variables =0;
//
//#if 1
//	sampling_plan.print();
//#endif
//
//
//
//	//	FILE* CL_Kriging_data;
//	//	FILE* CD_Kriging_data;
//	//	FILE* Volume_Kriging_data;
//	//	FILE* Thickness_Sec1_Kriging_data;
//	//	FILE* Thickness_Sec2_Kriging_data;
//	//	FILE* Thickness_Sec3_Kriging_data;
//	//	FILE* Thickness_Sec4_Kriging_data;
//	//	FILE* Thickness_Sec5_Kriging_data;
//	//	FILE* AllData;
//
//	int number_of_function_evals = sampling_plan.number_of_samples;
//
//	/* state outputs plus geometric constraints */
//	vec objectives(sampling_plan.number_of_outputs+sampling_plan.geometric_constraints.size());
//
//
//	vec deltax(number_of_design_variables);
//
//	for(unsigned int i=0; i<number_of_design_variables; i++){
//
//		deltax(i) = sampling_plan.upper_bound_dv(i)-sampling_plan.lower_bound_dv(i);
//
//	}
//
//
//	vec dv(number_of_design_variables);
//	mat dv_lhs(5*number_of_function_evals,number_of_design_variables);
//
//	/* generate lhs designs */
//
//	std::string str_problem_dim = std::to_string(number_of_design_variables);
//	std::string lhs_filename = "lhs_points.dat";
//
//	std::string python_command = "python -W ignore " + settings.python_dir +
//			"/lhs.py "+ lhs_filename+ " "+ str_problem_dim + " "+ std::to_string(5*number_of_function_evals)+ " center" ;
//
//#if 0
//	printf("%s\n",python_command.c_str());
//#endif
//
//	system(python_command.c_str());
//
//	dv_lhs.load("lhs_points.dat", raw_ascii);
//
//	/* shuffle samples */
//	dv_lhs = shuffle(dv_lhs);
//
//	for(unsigned int i=0; i<dv_lhs.n_rows;i++){
//		for(unsigned int j=0; j<dv_lhs.n_cols;j++){
//
//			dv_lhs(i,j)*= deltax(j);
//			dv_lhs(i,j)+= sampling_plan.lower_bound_dv(j);
//		}
//
//	}
//
//	if(sampling_plan.include_initial_design == true){
//
//		for(unsigned int j=0; j<dv_lhs.n_cols;j++){
//
//			dv_lhs(0,j)= 0.0;
//		}
//	}
//
//
//
//#if 0
//	printf("sampling plan = \n");
//	dv_lhs.print();
//#endif
//
//
//
//	int doe_iter = 0;
//	int count=0;
//
//
//	while(1){
//
//
//
//#if 1
//		printf("iter = %d\n",doe_iter);
//#endif
//
//
//
//		dv= trans(dv_lhs.row(count));
//		count++;
//
//		mat gradient_information(number_of_design_variables,sampling_plan.number_outputs_with_gradients);
//
//		vec gradient_out(number_of_design_variables);
//
//
//#if 1
//		printf("design vector:\n");
//		trans(dv).print();
//#endif
//
//		for(int i=0; i<sampling_plan.number_outputs_with_gradients;i++){
//
//
//			printf("calling adjoint solver for %s...\n",sampling_plan.output_names[i].c_str());
//			call_SU2_Adjoint_Solver(i,dv,gradient_out,objectives,sampling_plan);
//
//
//		}
//
//#if 1
//		printf("Area = %10.7f\n",objectives(2));
//		printf("cl = %10.7f\n",objectives(0));
//		printf("cd = %10.7f\n",objectives(1));
//#endif
//
//
//
//
//#if 0 /* validate adjoint results by finite difference approximations */
//
//		call_SU2_CFD_Solver(dv,CL,CD,area);
//		double f0_cd = CD;
//		double f0_cl = CL;
//		//		printf("f0 = %10.7f\n",f0);
//		double epsilon = 0.0001;
//
//		for(int i=0; i<dv.size(); i++){
//
//			vec dvsave = dv;
//			dv(i)+= epsilon;
//			call_SU2_CFD_Solver(dv,CL,CD,area);
//
//			dv = dvsave;
//			double fp_cd = CD;
//			double fp_cl = CL;
//			//			printf("fp = %15.10f\n",fp);
//			double fd_cd = (fp_cd - f0_cd)/epsilon;
//			double fd_cl = (fp_cl - f0_cl)/epsilon;
//			printf("fd_cd[%d] = %15.10f ad_cd[%d] = %15.10f\n",i,fd_cd,i,gradient_cd(i));
//			printf("fd_cl[%d] = %15.10f ad_cl[%d] = %15.10f\n",i,fd_cl,i,gradient_cl(i));
//
//
//		}
//
//#endif
//
//		//		if(objectives(0) > 0 && objectives(1) > 0){
//		//
//		//			CL_Kriging_data = fopen("CL_Kriging.csv","a+");
//		//			for(int i=0;i<number_of_design_variables;i++){
//		//				fprintf(CL_Kriging_data, "%10.7f, ",dv[i]);
//		//
//		//			}
//		//			fprintf(CL_Kriging_data, "%10.7f, ",objectives(0));
//		//
//		//			for(int i=0;i<number_of_design_variables-1;i++){
//		//				fprintf(CL_Kriging_data, "%10.7f, ",gradient_cl(i));
//		//
//		//			}
//		//			fprintf(CL_Kriging_data, "%10.7f\n",gradient_cl(number_of_design_variables-1));
//		//
//		//
//		//
//		//			fclose(CL_Kriging_data);
//		//
//		//			CD_Kriging_data = fopen("CD_Kriging.csv","a+");
//		//			for(int i=0;i<number_of_design_variables;i++){
//		//
//		//				fprintf(CD_Kriging_data, "%10.7f, ",dv[i]);
//		//
//		//			}
//		//			fprintf(CD_Kriging_data, "%10.7f, ",objectives(1));
//		//
//		//			for(int i=0;i<number_of_design_variables-1;i++){
//		//
//		//				fprintf(CD_Kriging_data, "%10.7f, ",gradient_cd(i));
//		//
//		//			}
//		//			fprintf(CD_Kriging_data, "%10.7f\n",gradient_cd(number_of_design_variables-1));
//		//
//		//
//		//
//		//			fclose(CD_Kriging_data);
//		//
//		//
//		//
//		//			Volume_Kriging_data = fopen("Volume_Kriging.csv","a+");
//		//			for(int i=0;i<number_of_design_variables;i++){
//		//				fprintf(Volume_Kriging_data, "%10.7f, ",dv[i]);
//		//
//		//			}
//		//			fprintf(Volume_Kriging_data, "%10.7f\n",objectives(2));
//		//
//		//
//		//			fclose(Volume_Kriging_data);
//		//
//		//
//		//			Thickness_Sec1_Kriging_data = fopen("Thickness_Sec1_Kriging.csv","a+");
//		//			for(int i=0;i<number_of_design_variables;i++){
//		//				fprintf(Thickness_Sec1_Kriging_data, "%10.7f, ",dv[i]);
//		//
//		//			}
//		//			fprintf(Thickness_Sec1_Kriging_data, "%10.7f\n",objectives(3));
//		//
//		//
//		//			fclose(Thickness_Sec1_Kriging_data);
//		//
//		//
//		//			Thickness_Sec2_Kriging_data = fopen("Thickness_Sec2_Kriging.csv","a+");
//		//			for(int i=0;i<number_of_design_variables;i++){
//		//				fprintf(Thickness_Sec2_Kriging_data, "%10.7f, ",dv[i]);
//		//
//		//			}
//		//			fprintf(Thickness_Sec2_Kriging_data, "%10.7f\n",objectives(4));
//		//
//		//
//		//			fclose(Thickness_Sec2_Kriging_data);
//		//
//		//
//		//
//		//			Thickness_Sec3_Kriging_data = fopen("Thickness_Sec3_Kriging.csv","a+");
//		//			for(int i=0;i<number_of_design_variables;i++){
//		//				fprintf(Thickness_Sec3_Kriging_data, "%10.7f, ",dv[i]);
//		//
//		//			}
//		//			fprintf(Thickness_Sec3_Kriging_data, "%10.7f\n",objectives(5));
//		//
//		//
//		//			fclose(Thickness_Sec3_Kriging_data);
//		//
//		//
//		//
//		//			Thickness_Sec4_Kriging_data = fopen("Thickness_Sec4_Kriging.csv","a+");
//		//			for(int i=0;i<number_of_design_variables;i++){
//		//				fprintf(Thickness_Sec4_Kriging_data, "%10.7f, ",dv[i]);
//		//
//		//			}
//		//			fprintf(Thickness_Sec4_Kriging_data, "%10.7f\n",objectives(6));
//		//
//		//
//		//			fclose(Thickness_Sec4_Kriging_data);
//		//
//		//
//		//
//		//			Thickness_Sec5_Kriging_data = fopen("Thickness_Sec5_Kriging.csv","a+");
//		//			for(int i=0;i<number_of_design_variables;i++){
//		//				fprintf(Thickness_Sec5_Kriging_data, "%10.7f, ",dv[i]);
//		//
//		//			}
//		//			fprintf(Thickness_Sec5_Kriging_data, "%10.7f\n",objectives(7));
//		//
//		//
//		//			fclose(Thickness_Sec5_Kriging_data);
//		//
//		//
//		//
//		//
//		//
//		//
//		//			AllData = fopen("oneram6_optimization_history.csv","a+");
//		//			for(int i=0;i<number_of_design_variables;i++){
//		//
//		//				fprintf(AllData, "%10.7f, ",dv[i]);
//		//
//		//			}
//		//			fprintf(AllData, "%10.7f, ",objectives(0));
//		//			fprintf(AllData, "%10.7f, ",objectives(1));
//		//			fprintf(AllData, "%10.7f\n",objectives(2));
//		//
//		//
//		//			fclose(AllData);
//		//
//		//			doe_iter++;
//		//
//		//		}
//
//
//		if( doe_iter >= number_of_function_evals ) break;
//
//	} /* end of DoE iterations */
//
//}
//
void DoE(int number_of_initial_samples ){


	FILE* CL_Kriging_data;
	FILE* CD_Kriging_data;
	FILE* Area_Kriging_data;
	FILE* AllData;

	printf("Design of Experiment...\n");
	int number_of_design_variables = 38;
	int number_of_function_evals = number_of_initial_samples;

	double upper_bound_dv =  0.003;
	double lower_bound_dv = -0.003;

	double area_constraint= 0.0816;

	double deltax = upper_bound_dv-lower_bound_dv;


	vec dv(number_of_design_variables);
	mat dv_lhs(number_of_function_evals,number_of_design_variables);


	/* generate lhs designs */

	std::string str_problem_dim = std::to_string(number_of_design_variables);
	std::string lhs_filename = "lhs_points.dat";

	std::string python_command = "python -W ignore " + settings.python_dir +
			"/lhs.py "+ lhs_filename+ " "+ str_problem_dim + " "+ std::to_string(5*number_of_function_evals)+ " center" ;


	system(python_command.c_str());


	dv_lhs.load("lhs_points.dat", raw_ascii);


	dv_lhs = shuffle(dv_lhs);


	dv_lhs*= deltax;
	dv_lhs+= lower_bound_dv;

	dv_lhs.print();




	int nsample=1;
	unsigned int count_lhs = 0;

	while (nsample <= number_of_function_evals && count_lhs < dv_lhs.n_rows){


		std::ifstream ifs("config_DEF.cfg");
		std::string basic_text;
		getline (ifs, basic_text, (char) ifs.eof());

#if 1
		cout<<basic_text;
#endif

		if(nsample == 0){

			for(int i=0;i<number_of_design_variables;i++){
				dv(i)= 0.0;
#if 0
				printf("dv[%d]= %10.7f\n",i,dv[i]);
#endif
			}


		}

		else{

			for(int i=0;i<number_of_design_variables;i++){
				dv(i)= dv_lhs(count_lhs,i);
#if 0
				printf("dv[%d]= %10.7f\n",i,dv[i]);
#endif
			}

		}

		count_lhs++;



		std::string dv_text = "DV_VALUE=";


		for(int i=0;i<number_of_design_variables-1;i++){
			dv_text+= std::to_string(dv[i])+",";

		}
		dv_text+= std::to_string(dv[number_of_design_variables-1])+"\n";


		cout<<dv_text;


		std::ofstream su2_def_input_file;
		su2_def_input_file.open ("config_DEF_new.cfg");
		su2_def_input_file << basic_text+dv_text;
		su2_def_input_file.close();





		system("SU2_DEF config_DEF_new.cfg");


#if 1
		plot_airfoil("rae2822_geometry.dat", "surface_grid.dat",192);
#endif



		system("SU2_GEO config_GEO.cfg");



		double *geo_data = new double[6];

		std::ifstream geo_outstream("of_func.dat");
		std::string str;
		int count=0;
		while (std::getline(geo_outstream, str))
		{

			count++;

			if(count == 4){

				//			cout<<str<<endl;

				str.erase(std::remove(str.begin(), str.end(), ','), str.end());
				std::stringstream ss(str);
				for(int i=0;i<6;i++){
					ss >> geo_data[i];
					//	        	  printf("geo_data[%d] = %10.7f\n",i,geo_data[i]);
				}



			}

		}

		double area = geo_data[0];
		printf("Area of the airfoil = %10.7f\n", area);






		if(area < area_constraint) continue;

		std::string solver_command = "parallel_computation.py -f turb_SA_RAE2822.cfg -n 2";
		system(solver_command.c_str());




		double CL=0,CD=0;

		int cl_found=0;
		int cd_found=0;
		std::string for_cl = "CL:";
		std::string for_cd = "CD:";


		std::size_t found,found2;

		std::ifstream forces_outstream("history.dat");

		std::string str_previous;

		while (std::getline(forces_outstream, str))
		{
#if 0
			cout<<str<<endl;
#endif

			str_previous = str;

		}

#if 0
		cout<<str_previous<<endl;
#endif


		std::vector<double> vect;

		std::stringstream ss(str_previous);

		double ff;


		while(1){
			ss>>ff;
			vect.push_back(ff);
			if (ss.peek() == ',')
				ss.ignore();
			else break;

		}
#if 0
		for (std::size_t i = 0; i < vect.size(); i++) {

			std::cout << i<< vect[i] << std::endl;

		}
#endif

		CL = vect[1];
		CD = vect[2];

		cout<<"CL = "<<CL<<endl;
		cout<<"CD = "<<CD<<endl;



		CL_Kriging_data = fopen("CL_Kriging.csv","a+");
		for(int i=0;i<number_of_design_variables;i++){
			fprintf(CL_Kriging_data, "%10.7f, ",dv[i]);

		}
		fprintf(CL_Kriging_data, "%10.7f\n",CL);


		fclose(CL_Kriging_data);


		CD_Kriging_data = fopen("CD_Kriging.csv","a+");
		for(int i=0;i<number_of_design_variables;i++){
			fprintf(CD_Kriging_data, "%10.7f, ",dv[i]);

		}
		fprintf(CD_Kriging_data, "%10.7f\n",CD);


		fclose(CD_Kriging_data);


		Area_Kriging_data = fopen("Area_Kriging.csv","a+");
		for(int i=0;i<number_of_design_variables;i++){
			fprintf(Area_Kriging_data, "%10.7f, ",dv[i]);

		}
		fprintf(Area_Kriging_data, "%10.7f\n",area);


		fclose(Area_Kriging_data);

		AllData = fopen("rae2822_optimization_history.csv","a+");
		for(int i=0;i<number_of_design_variables;i++){
			fprintf(AllData, "%10.7f, ",dv[i]);

		}
		fprintf(AllData, "%10.7f, ",CL);
		fprintf(AllData, "%10.7f, ",CD);
		fprintf(AllData, "%10.7f\n",area);


		fclose(AllData);



		nsample++;
	} // end of while loop





}
//
///* calls the SU2_CFD for a given vector of dv */
//int call_SU2_CFD_Solver(vec &dv,
//		double &CL,
//		double &CD,
//		double &area){
//
//	int number_of_design_variables=dv.size();
//
//
//	std::ifstream ifs("config_DEF.cfg");
//	std::string basic_text;
//	getline (ifs, basic_text, (char) ifs.eof());
//
//	std::string dv_text = "DV_VALUE=";
//
//
//	for(int i=0;i<number_of_design_variables-1;i++){
//		dv_text+= std::to_string(dv[i])+",";
//
//	}
//	dv_text+= std::to_string(dv[number_of_design_variables-1])+"\n";
//
//
//	//	cout<<dv_text;
//
//
//	std::ofstream su2_def_input_file;
//	su2_def_input_file.open ("config_DEF_new.cfg");
//	su2_def_input_file << basic_text+dv_text;
//	su2_def_input_file.close();
//
//
//	system("SU2_DEF config_DEF_new.cfg > su2def_output");
//
//#if 0
//	plot_airfoil("naca0012_airfoil_geometry.dat", "mesh_out.su2");
//#endif
//
//
//	system("cp mesh_out.su2  mesh_airfoil.su2");
//
//	system("SU2_GEO inv_NACA0012_cfd.cfg > su2geo_output");
//
//
//
//	double *geo_data = new double[10];
//
//	std::ifstream geo_outstream("of_func.dat");
//	std::string str;
//	int count=0;
//	while (std::getline(geo_outstream, str))
//	{
//
//		count++;
//
//		if(count == 4){
//
//			str.erase(std::remove(str.begin(), str.end(), ','), str.end());
//			std::stringstream ss(str);
//			for(int i=0;i<10;i++){
//				ss >> geo_data[i];
//
//			}
//
//		}
//
//	}
//
//	area = geo_data[6];
//
//	std::string solver_command = "parallel_computation.py -f inv_NACA0012_cfd.cfg -n 2 > su2cfd_output";
//
//	system(solver_command.c_str());
//
//
//
//	int cl_found=0;
//	int cd_found=0;
//	std::string for_cl = "CL:";
//	std::string for_cd = "CD:";
//
//
//	std::size_t found,found2;
//
//	std::ifstream forces_outstream("forces_breakdown.dat");
//
//	while (std::getline(forces_outstream, str))
//	{
//
//
//		found = str.find(for_cl);
//		if (found!=std::string::npos && cl_found==0){
//
//			found2 = str.find('|');
//
//			std::string cl_value;
//			cl_value.assign(str,found+3,found2-found-3);
//
//
//			CL = std::stod (cl_value);
//
//			cl_found=1;
//
//
//		}
//
//		found = str.find(for_cd);
//		if (found!=std::string::npos && cd_found==0){
//
//			found2 = str.find('|');
//
//			//		            std::cout << "CL at: " << found << found2<<'\n';
//
//			std::string cd_value;
//			cd_value.assign(str,found+3,found2-found-3);
//
//
//			//		            std::cout << "CL val: " <<cl_value<<endl;
//
//			CD = std::stod (cd_value);
//
//			cd_found=1;
//
//			//		            cout<<"CL = "<<CL<<endl;
//
//
//		}
//
//	}
//
//	//	cout<<"CL = "<<CL<<endl;
//	//	cout<<"CD = "<<CD<<endl;
//
//	return 0;
//}
//
//
//
///* calls the SU2_CFD_AD for a given vector of dv */
//int call_SU2_Adjoint_Solver(
//		int i,
//		vec &dv,
//		vec &gradient,
//		vec &objectives,
//		OptimizationData &optimization_plan
//){
//
//#if 1
//	printf("calling SU2 adjoint solver...\n");
//	printf("base config file = %s\n",optimization_plan.base_config_file_name.c_str());
//#endif
//
//	/* initialize all variables to zero */
//
//	objectives.fill(0.0);
//	gradient.fill(0.0);
//
//	int number_of_design_variables=dv.size();
//
//	std::ifstream ifs(optimization_plan.base_config_file_name);
//	std::string basic_text;
//	getline (ifs, basic_text, (char) ifs.eof());
//
//#if 1
//	printf("base config =\n");
//	cout<<basic_text;
//#endif
//
//
//	std::string dv_text = "DV_VALUE=";
//
//
//	for(int i=0;i<dv.size()-1;i++){
//
//		dv_text+= std::to_string(dv(i))+",";
//
//	}
//
//	dv_text+= std::to_string(dv[dv.size()-1])+"\n";
//
//
//
//	std::string dv_text2 = "DV_VALUE_NEW=";
//
//
//	for(int i=0;i<dv.size()-1;i++){
//
//		dv_text2+= std::to_string(dv(i))+",";
//
//	}
//
//	dv_text2+= std::to_string(dv[dv.size()-1])+"\n";
//
//#if 0
//	cout<<dv_text;
//	cout<<dv_text2;
//#endif
//
//
//
//	std::ofstream su2_def_input_file;
//	su2_def_input_file.open ("config_DEF_new.cfg");
//	su2_def_input_file << basic_text+dv_text+dv_text2;
//	su2_def_input_file.close();
//
//	system("SU2_DEF config_DEF_new.cfg > su2def_output");
//
//	//	system("cp mesh_ONERAM6_inv_FFD_deform.su2  mesh_ONERAM6_inv_FFD.su2");
//
//	system("SU2_GEO config_DEF_new.cfg > su2geo_output");
//
//
//
//
//	vec geo_input(optimization_plan.number_of_geometric_features);
//
//	std::ifstream geo_outstream("of_func.dat");
//	std::string str;
//	int count=0;
//	while (std::getline(geo_outstream, str))
//	{
//#if 0
//		printf("str = %s\n", str.c_str());
//#endif
//		count++;
//
//		if(count == 4){
//
//#if 0
//			printf("str = %s\n", str.c_str());
//#endif
//			str.erase(std::remove(str.begin(), str.end(), ','), str.end());
//			std::stringstream ss(str);
//
//
//
//			for(int i=0;i<optimization_plan.number_of_geometric_features;i++){
//
//				ss >> geo_input(i);
//
//			}
//
//		}
//
//	}
//#if 1
//	for(int i=0;i<optimization_plan.number_of_geometric_features;i++){
//
//		printf("geo data[%d] = %10.7f\n",i,geo_input(i));
//
//
//	}
//#endif
//
//
//
//
//
//	objectives(2) =  geo_input[6];
//
//
//#if 1
//	printf("Area = %10.7f\n", objectives(2));
//#endif
//	//
//	//
//	int geo_constraint_flag = 1;
//	//
//	if(objectives(2) < optimization_plan.geometric_constraints(0)) geo_constraint_flag = 0;
//	//	if(objectives(3) < constraints(1)) geo_constraint_flag = 0;
//	//	if(objectives(4) < constraints(2)) geo_constraint_flag = 0;
//	//	if(objectives(5) < constraints(3)) geo_constraint_flag = 0;
//	//	if(objectives(6) < constraints(4)) geo_constraint_flag = 0;
//	//	if(objectives(7) < constraints(5)) geo_constraint_flag = 0;
//	//
//	//
//	if(geo_constraint_flag == 0){
//		//
//		printf("One or more of the geometric constraints are not satisfied...\n");
//		return 0;
//
//	}
//
//
//
//	std::string solver_command;
//
//	solver_command = "discrete_adjoint.py -f " + optimization_plan.config_file_names[i]
//	                                                                                 + " -n 2 discrete_adjoint_"+ optimization_plan.output_names[i]+"_output";
//
//#if 1
//	printf("solver_command = %s\n",solver_command.c_str());
//#endif
//
//	system(solver_command.c_str());
//
//
//	int cl_found=0;
//	int cd_found=0;
//	std::string for_cl = "CL:";
//	std::string for_cd = "CD:";
//
//
//	std::size_t found,found2,found3;
//
//	std::ifstream forces_outstream("forces_breakdown.dat");
//
//	while (std::getline(forces_outstream, str))
//	{
//
//
//		found = str.find(for_cl);
//		if (found!=std::string::npos && cl_found==0){
//
//			found2 = str.find('|');
//
//			std::string cl_value;
//			cl_value.assign(str,found+3,found2-found-3);
//
//			objectives(0) = std::stod (cl_value);
//
//			cl_found=1;
//
//
//		}
//
//		found = str.find(for_cd);
//		if (found!=std::string::npos && cd_found==0){
//
//			found2 = str.find('|');
//
//
//			std::string cd_value;
//			cd_value.assign(str,found+3,found2-found-3);
//
//
//			objectives(1) = std::stod (cd_value);
//
//			cd_found=1;
//
//
//		}
//
//
//	}
//#if 1
//	printf("cl = %10.7f\n",objectives(0));
//	printf("cd = %10.7f\n",objectives(1));
//#endif
//
//
//	std::string of_filename;
//
//	if(i == 0) of_filename = "of_grad_cl.dat";
//	if(i == 1) of_filename = "of_grad_cd.dat";
//
//	std::ifstream grad_outstream(of_filename);
//
//	count=0;
//	while (std::getline(grad_outstream, str))
//	{
//
//		found = str.find("VARIABLES");
//		if (found==std::string::npos){
//
//
//
//			found2 = str.find(',');
//			found3 = str.find(',',found2+1);
//
//
//			std::string grad_value;
//			grad_value.assign(str,found2+1,found3-found2-1);
//
//			if (std::stod(grad_value) != 0){
//
//				gradient(count) = std::stod(grad_value);
//				count++;
//			}
//
//		}
//
//
//
//	} /* end of while */
//
//
//#if 1
//	printf("gradient:\n");
//	trans(gradient).print();
//#endif
//
//
//
//	return 0;
//}
//
//
///* calls the SU2_CFD_AD for a given vector of dv */
//int call_SU2_Adjoint_Solver(
//		int i,
//		vec &dv,
//		vec &gradient,
//		vec &objectives,
//		GEKSamplingData &sampling_plan
//){
//
//#if 1
//	printf("calling SU2 adjoint solver...\n");
//	printf("base config file = %s\n",sampling_plan.base_config_file_name.c_str());
//#endif
//
//	/* initialize all variables to zero */
//
//	objectives.fill(0.0);
//	gradient.fill(0.0);
//
//	int number_of_design_variables=dv.size();
//
//	std::ifstream ifs(sampling_plan.base_config_file_name);
//	std::string basic_text;
//	getline (ifs, basic_text, (char) ifs.eof());
//
//#if 1
//	printf("base config =\n");
//	cout<<basic_text;
//#endif
//
//
//	std::string dv_text = "DV_VALUE=";
//
//
//	for(int i=0;i<dv.size()-1;i++){
//
//		dv_text+= std::to_string(dv(i))+",";
//
//	}
//
//	dv_text+= std::to_string(dv[dv.size()-1])+"\n";
//
//
//
//	std::string dv_text2 = "DV_VALUE_NEW=";
//
//
//	for(int i=0;i<dv.size()-1;i++){
//
//		dv_text2+= std::to_string(dv(i))+",";
//
//	}
//
//	dv_text2+= std::to_string(dv[dv.size()-1])+"\n";
//
//#if 0
//	cout<<dv_text;
//	cout<<dv_text2;
//#endif
//
//
//
//	std::ofstream su2_def_input_file;
//	su2_def_input_file.open ("config_DEF_new.cfg");
//	su2_def_input_file << basic_text+dv_text+dv_text2;
//	su2_def_input_file.close();
//
//	system("SU2_DEF config_DEF_new.cfg > su2def_output");
//
//	//	system("cp mesh_ONERAM6_inv_FFD_deform.su2  mesh_ONERAM6_inv_FFD.su2");
//
//	system("SU2_GEO config_DEF_new.cfg > su2geo_output");
//
//
//
//
//	vec geo_input(sampling_plan.number_of_geometric_features);
//
//	std::ifstream geo_outstream("of_func.dat");
//	std::string str;
//	int count=0;
//	while (std::getline(geo_outstream, str))
//	{
//#if 0
//		printf("str = %s\n", str.c_str());
//#endif
//		count++;
//
//		if(count == 4){
//
//#if 0
//			printf("str = %s\n", str.c_str());
//#endif
//			str.erase(std::remove(str.begin(), str.end(), ','), str.end());
//			std::stringstream ss(str);
//
//
//
//			for(int i=0;i<sampling_plan.number_of_geometric_features;i++){
//
//				ss >> geo_input(i);
//
//			}
//
//		}
//
//	}
//#if 1
//	for(int i=0;i<sampling_plan.number_of_geometric_features;i++){
//
//		printf("geo data[%d] = %10.7f\n",i,geo_input(i));
//
//
//	}
//#endif
//
//
//
//
//
//	objectives(2) =  geo_input[6];
//
//
//#if 1
//	printf("Area = %10.7f\n", objectives(2));
//#endif
//	//
//	//
//	int geo_constraint_flag = 1;
//	//
//	if(objectives(2) < sampling_plan.geometric_constraints(0)) geo_constraint_flag = 0;
//	//	if(objectives(3) < constraints(1)) geo_constraint_flag = 0;
//	//	if(objectives(4) < constraints(2)) geo_constraint_flag = 0;
//	//	if(objectives(5) < constraints(3)) geo_constraint_flag = 0;
//	//	if(objectives(6) < constraints(4)) geo_constraint_flag = 0;
//	//	if(objectives(7) < constraints(5)) geo_constraint_flag = 0;
//	//
//	//
//	if(geo_constraint_flag == 0){
//		//
//		printf("One or more of the geometric constraints are not satisfied...\n");
//		return 0;
//
//	}
//
//
//
//	std::string solver_command;
//
//	solver_command = "discrete_adjoint.py -f " + sampling_plan.config_file_names[i]
//	                                                                             + " -n 2 discrete_adjoint_"+ sampling_plan.output_names[i]+"_output";
//
//#if 1
//	printf("solver_command = %s\n",solver_command.c_str());
//#endif
//
//	system(solver_command.c_str());
//
//
//	int cl_found=0;
//	int cd_found=0;
//	std::string for_cl = "CL:";
//	std::string for_cd = "CD:";
//
//
//	std::size_t found,found2,found3;
//
//	std::ifstream forces_outstream("forces_breakdown.dat");
//
//	while (std::getline(forces_outstream, str))
//	{
//
//
//		found = str.find(for_cl);
//		if (found!=std::string::npos && cl_found==0){
//
//			found2 = str.find('|');
//
//			std::string cl_value;
//			cl_value.assign(str,found+3,found2-found-3);
//
//			objectives(0) = std::stod (cl_value);
//
//			cl_found=1;
//
//
//		}
//
//		found = str.find(for_cd);
//		if (found!=std::string::npos && cd_found==0){
//
//			found2 = str.find('|');
//
//
//			std::string cd_value;
//			cd_value.assign(str,found+3,found2-found-3);
//
//
//			objectives(1) = std::stod (cd_value);
//
//			cd_found=1;
//
//
//		}
//
//
//	}
//#if 1
//	printf("cl = %10.7f\n",objectives(0));
//	printf("cd = %10.7f\n",objectives(1));
//#endif
//
//
//	std::string of_filename;
//
//	if(i == 0) of_filename = "of_grad_cl.dat";
//	if(i == 1) of_filename = "of_grad_cd.dat";
//
//	std::ifstream grad_outstream(of_filename);
//
//	count=0;
//	while (std::getline(grad_outstream, str))
//	{
//
//		found = str.find("VARIABLES");
//		if (found==std::string::npos){
//
//
//
//			found2 = str.find(',');
//			found3 = str.find(',',found2+1);
//
//
//			std::string grad_value;
//			grad_value.assign(str,found2+1,found3-found2-1);
//
//			if (std::stod(grad_value) != 0){
//
//				gradient(count) = std::stod(grad_value);
//				count++;
//			}
//
//		}
//
//
//
//	} /* end of while */
//
//
//#if 1
//	printf("gradient:\n");
//	trans(gradient).print();
//#endif
//
//
//
//	return 0;
//}





