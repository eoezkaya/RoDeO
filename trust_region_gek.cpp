
#include <armadillo>
using namespace arma;


#include "trust_region_gek.hpp"
#include "Rodeo_macros.hpp"
#include "kriging_training.hpp"
#include "test_functions.hpp"
#include "auxilliary_functions.hpp"


void decide_nearest_seeding_point(rowvec x,vec Nx, rowvec &seeding_point_coordinates){


	int dim = x.size();



	for(int i=0;i<dim;i++){

		double dx = 1.0/(Nx(i)-1);

		seeding_point_coordinates(i) = floor(x(i)/dx)*dx;
	}




}


void estimate_gradient(mat &regression_weights,
		mat &kriging_params,
		vec &gradient,
		mat &R_inv_ys_min_beta,
		rowvec &x,
		vec &beta0,
		mat &X){


	//	printf("estimate gradient\n");
	int dim = x.size();


	for(int i=1;i<dim+1;i++){

		//		printf("i = %d\n",i);
		gradient(i-1) = calculate_f_tilde(x,
				X,
				beta0(i),
				regression_weights.col(i),
				R_inv_ys_min_beta.col(i-1),
				kriging_params.col(i));



	}




}


int train_TRGEK_response_surface(std::string input_file_name,
		int linear_regression,
		mat &regression_weights,
		mat &kriging_params,
		mat &R_inv_ys_min_beta,
		vec &beta0,
		int &max_number_of_function_calculations,
		int dim) {


	double reg_param=0.00000001;

	printf("training trust-region Kriging response surface for the data : %s\n",
			input_file_name.c_str());


	mat data_functional_values; // data matrix for only functional values
	mat data_gradients;         // data matrix for only functional values + gradient sensitivities


	vec R_inv_ys_min_beta_func;

	std::ifstream in(input_file_name);

	if(!in) {
		printf("Cannot open input file %s...\n",input_file_name.c_str() );
		return 1;
	}


	std::vector<double> temp;



	std::string str;
	int count_f=0;
	int count_g=0;

	while (std::getline(in, str)) {
		// output the line
		//				std::cout << "line = "<<str << std::endl;


		std::string delimiter = ",";

		size_t pos = 0;
		std::string token;
		while ((pos = str.find(delimiter)) != std::string::npos) {
			token = str.substr(0, pos);
			//			std::cout << "token = "<<token << std::endl;


			temp.push_back(atof(token.c_str()));


			str.erase(0, pos + delimiter.length());
		}
		//				std::cout << "str = "<<str << std::endl;
		temp.push_back(atof(str.c_str()));

		//		std::cout<<"temp= \n";
		for (std::vector<double>::iterator it = temp.begin() ; it != temp.end(); ++it){

			//			std::cout<<*it<<std::endl;


		}


		if(temp.size() == dim+1){ // function values

			rowvec newrow(dim+1);
			int count=0;
			for (std::vector<double>::iterator it = temp.begin() ; it != temp.end(); ++it){

				//				std::cout<<*it<<std::endl;
				newrow(count)=*it;
				count++;

			}
			//			newrow.print();
			count_f++;
			data_functional_values.resize(count_f, dim+1);

			data_functional_values.row(count_f-1)=newrow;



		}
		else{ // function+gradient information

			rowvec newrow(2*dim+1);
			int count=0;
			for (std::vector<double>::iterator it = temp.begin() ; it != temp.end(); ++it){

				//				std::cout<<*it<<std::endl;
				newrow(count)=*it;
				count++;

			}
			//			newrow.print();
			count_g++;
			data_gradients.resize(count_g, 2*dim+1);

			data_gradients.row(count_g-1)=newrow;


		}




		temp.clear();

		// now we loop back and get the next line in 'str'
	}


	/* number of points with only functional values */
	int n_f_evals = data_functional_values.n_rows;
	/* number of points with functional values + gradient sensitivities*/
	int n_g_evals = data_gradients.n_rows;

	printf("number of points with only functional values = %d\n",n_f_evals);
	printf("number of points with sensitivity gradients = %d\n",n_g_evals);







	printf("functional values data :\n");
	data_functional_values.print();
	printf("\n");






	printf("gradient data (raw) :\n");
	data_gradients.print();
	printf("\n");



	mat X_func,X_grad;


	if(n_g_evals > 0){


		X_grad = data_gradients.submat(0, 0, n_g_evals - 1, dim-1);

		mat func_values_grad = data_gradients.submat( 0, 0, n_g_evals-1, dim );

		//		func_values_grad.print();

		data_functional_values.insert_rows(n_f_evals,func_values_grad);

		printf("data_functional_values = \n");
		data_functional_values.print();



	}


	if(n_g_evals > 0){




		X_func = data_functional_values.submat(0, 0, n_g_evals+n_f_evals - 1, dim-1);


	}



	printf("X_func = \n");
	X_func.print();

	printf("X_grad = \n");
	X_grad.print();






	/* find minimum and maximum of the columns of data */

	vec x_max(dim);
	x_max.fill(0.0);

	vec x_min(dim);
	x_min.fill(0.0);

	for (int i = 0; i < dim; i++) {
		x_max(i) = X_func.col(i).max();
		x_min(i) = X_func.col(i).min();

	}

	printf("maximum = \n");
	x_max.print();

	printf("minimum = \n");
	x_min.print();

	/* normalize data matrix */

	for (int i = 0; i < X_func.n_rows; i++) {
		for (int j = 0; j < dim; j++) {
			X_func(i, j) = (1.0/dim)*(X_func(i, j) - x_min(j)) / (x_max(j) - x_min(j));
		}
	}


	for (int i = 0; i < X_grad.n_rows; i++) {
		for (int j = 0; j < dim; j++) {
			X_grad(i, j) = (1.0/dim)*(X_grad(i, j) - x_min(j)) / (x_max(j) - x_min(j));
		}
	}


	printf("X_func(normalized) = \n");
	X_func.print();

	printf("X_grad(normalized) = \n");
	X_grad.print();




	for(int i=0;i<dim+1;i++){

		std::string kriging_input_filename;

		kriging_input_filename="trust_region_gek_input"+std::to_string(i)+".csv";

		vec regression_weights_single_output=zeros(dim+1);
		vec kriging_weights_single_output=zeros(dim);

		mat kriging_input_data;

		if(i==0){


			data_functional_values.save(kriging_input_filename, csv_ascii);
			printf("kriging_input_data = \n");
			data_functional_values.print();
			printf("\n");





		}

		else{
			//			kriging_input_data.set_size(n_g_evals,dim+1);

			for(int j=0;j<dim;j++) kriging_input_data.insert_cols( j, data_gradients.col(j) );

			kriging_input_data.insert_cols( dim, data_gradients.col(dim+i) );

			printf("kriging_input_data = \n");

			kriging_input_data.print();

			kriging_input_data.save(kriging_input_filename, csv_ascii);


		}

		train_kriging_response_surface(kriging_input_filename,
				"None",
				linear_regression,
				regression_weights_single_output,
				kriging_weights_single_output,
				reg_param,
				max_number_of_function_calculations,
				CSV_ASCII);


		regression_weights.insert_cols(i,regression_weights_single_output);
		kriging_params.insert_cols(i,kriging_weights_single_output);

		printf("regression_weights = \n");
		regression_weights_single_output.print();
		printf("kriging_weights = \n");
		kriging_weights_single_output.print();





		vec theta = kriging_weights_single_output.head(dim);
		vec gamma = kriging_weights_single_output.tail(dim);

		if(i==0) {

			mat R(n_g_evals+n_f_evals,n_g_evals+n_f_evals);
			vec I = ones(n_g_evals+n_f_evals);

			compute_R_matrix(theta,
					gamma,
					reg_param,
					R,
					X_func);

			printf("R = \n");
			R.print();
			printf("\n");


			vec ys = data_functional_values.col(dim);

			printf("ys = \n");
			ys.print();
			printf("\n");


			mat Rinv = inv(R);


			beta0(i) = (1.0/dot(I,Rinv*I)) * (dot(I,Rinv*ys));

			R_inv_ys_min_beta_func = Rinv* (ys-beta0(i)* I);

			R_inv_ys_min_beta_func.print();


		}
		else {

			mat R(n_g_evals,n_g_evals);
			vec I = ones(n_g_evals);



			compute_R_matrix(theta,
					gamma,
					reg_param,
					R,
					X_grad);




			//			printf("R = \n");
			//			R.print();


			vec ys = kriging_input_data.col(dim);


			printf("ys = \n");
			ys.print();
			printf("\n");

			mat Rinv = inv(R);


			beta0(i) = (1.0/dot(I,Rinv*I)) * (dot(I,Rinv*ys));
			vec R_inv_ys_min_beta_temp = Rinv* (ys-beta0(i)* I);


			R_inv_ys_min_beta.insert_cols(i-1,R_inv_ys_min_beta_temp);




		}



	}


	double in_sample_error = 0.0;
	for(unsigned int i=0;i<X_func.n_rows;i++){

		rowvec x = X_func.row(i);

		rowvec xb(dim);





		double func_val = calculate_f_tilde(x,
				X_func,
				beta0(0),
				regression_weights.col(0),
				R_inv_ys_min_beta_func,
				kriging_params.col(0));

		vec grad(dim);
		estimate_gradient(regression_weights,
				kriging_params,
				grad,
				R_inv_ys_min_beta,
				x,
				beta0,
				X_grad);


		for(int j=0; j<dim;j++) x(j) = dim*x(j)* (x_max(j) - x_min(j))+x_min(j);

		x.print();

		xb.fill(0.0);
		double func_val_exact = Eggholder_adj(x.memptr(),xb.memptr());

		//					printf("\n");
		//					x.print();
		//					printf("\n");
		//					printf("ftilde = %10.7f fexact= %10.7f\n",func_val,func_val_exact );

		printf("grad exact[0] = %10.7f grad exact[1] = %10.7f\n",xb[0],xb[1]);
		printf("grad approx[0] = %10.7f grad approx[1] = %10.7f\n",grad(0),grad(1));
		in_sample_error+= (func_val_exact-func_val)*(func_val_exact-func_val);


	}

	in_sample_error = sqrt(in_sample_error/X_func.n_rows);

	printf("in sample error = %10.7f\n",in_sample_error);







	/* visualize with contour plot if the problem is 2D */
	if (dim == 2){
		int resolution =100;

		double bounds[4];
		bounds[0]=0.0;
		bounds[1]=100.0;
		bounds[2]=0.0;
		bounds[3]=100.0;

		std::string kriging_response_surface_file_name = "Eggholder_TRGEK_response_surface.dat";
		std::string kriging_response_surface_g1_file_name = "Eggholder_TRGEK_response_surface_g1.dat";
		std::string kriging_response_surface_g2_file_name = "Eggholder_TRGEK_response_surface_g2.dat";

		FILE *kriging_response_surface_file = fopen(kriging_response_surface_file_name.c_str(),"w");
		FILE *kriging_response_surface_g1_file = fopen(kriging_response_surface_g1_file_name.c_str(),"w");
		FILE *kriging_response_surface_g2_file = fopen(kriging_response_surface_g2_file_name.c_str(),"w");



		double dx,dy; /* step sizes in x and y directions */
		rowvec x(2);
		rowvec xb(2);
		rowvec xnorm(2);

		double func_val;
		dx = (bounds[1]-bounds[0])/(resolution-1);
		dy = (bounds[3]-bounds[2])/(resolution-1);

		double out_sample_error=0.0;

		x[0] = bounds[0];
		for(int i=0;i<resolution;i++){
			x[1] = bounds[2];
			for(int j=0;j<resolution;j++){

				/* normalize x */
				xnorm(0)= (1.0/dim)*(x(0)- x_min(0)) / (x_max(0) - x_min(0));
				xnorm(1)= (1.0/dim)*(x(1)- x_min(1)) / (x_max(1) - x_min(1));

				func_val = calculate_f_tilde(xnorm, X_func, beta0(0),
						regression_weights.col(0), R_inv_ys_min_beta_func, kriging_params.col(0));



				vec grad(dim);
				estimate_gradient(regression_weights,
						kriging_params,
						grad,
						R_inv_ys_min_beta,
						xnorm,
						beta0,
						X_grad);






				fprintf(kriging_response_surface_file,"%10.7f %10.7f %10.7f\n",x(0),x(1),func_val);
				fprintf(kriging_response_surface_g1_file,"%10.7f %10.7f %10.7f\n",x(0),x(1),grad(0));
				fprintf(kriging_response_surface_g2_file,"%10.7f %10.7f %10.7f\n",x(0),x(1),grad(1));

				xb.fill(0.0);
				double func_val_exact = Eggholder_adj(x.memptr(),xb.memptr());


//				printf("%10.7f %10.7f %10.7f %10.7f %10.7f %10.7f\n",x(0),x(1),grad(0),xb(0),grad(1),xb(1));

				out_sample_error+= (func_val_exact-func_val)*(func_val_exact-func_val);



				x[1]+=dy;
			}
			x[0]+= dx;

		}
		fclose(kriging_response_surface_file);
		fclose(kriging_response_surface_g1_file);
		fclose(kriging_response_surface_g2_file);

		out_sample_error = sqrt(out_sample_error/(resolution*resolution));

		printf("out of sample error = %10.7f\n",out_sample_error);



		/* plot the kriging response surface */

//		std::string file_name_for_plot = "Eggholder_TRGEK_response_surface_";
//		file_name_for_plot += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";
//
//		std::string python_command = "python -W ignore plot_2d_surface.py "+ kriging_response_surface_file_name+ " "+ file_name_for_plot ;
//
//
//
//		FILE* in = popen(python_command.c_str(), "r");
//
//
//		fprintf(in, "\n");
//
//
//		/* plot the kriging response surface for df/dx1*/
//
//		std::string file_name_for_plot_g1 = "Eggholder_TRGEK_response_surface_g1_";
//		file_name_for_plot_g1 += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";
//
//		std::string python_command_g1 = "python -W ignore plot_2d_surface.py "+ kriging_response_surface_g1_file_name+ " "+ file_name_for_plot ;
//
//
//
//		FILE* in_g1 = popen(python_command_g1.c_str(), "r");
//
//
//		fprintf(in_g1, "\n");



	}



	/* END */


	double error_kriging=0.0;
	double error_new_method=0.0;

	vec Nx(2);

	Nx(0)=30;
	Nx(1)=30;

	for(int trial=0;trial<50000;trial++){


		rowvec x(2);
		rowvec xnorm(2);
		rowvec seedx(2);
		rowvec seedxnotnormalized(2);
		rowvec seedxb(2);



		x(0)=RandomDouble(0.0, 100.0);
		x(1)=RandomDouble(0.0, 100.0);


		/* normalize x */
		xnorm(0)= (1.0/dim)*(x(0)- x_min(0)) / (x_max(0) - x_min(0));
		xnorm(1)= (1.0/dim)*(x(1)- x_min(1)) / (x_max(1) - x_min(1));


//		printf("xnorm = \n");
//		xnorm.print();
//		printf("\n");

		decide_nearest_seeding_point(xnorm,Nx,seedx);

		//	printf("next seeding point coordinates = \n");
		//	seedx.print();


		//	printf("kriging_params matrix =\n");
		//	kriging_params.print();



		//	printf("beta0=\n");
		//	beta0.print();
		//	printf("\n");

		double ftilde_seed= calculate_f_tilde(seedx,
				X_func,
				beta0(0),
				regression_weights.col(0),
				R_inv_ys_min_beta_func,
				kriging_params.col(0));


		for(int j=0; j<dim;j++) seedxnotnormalized(j) = dim*seedx(j)* (x_max(j) - x_min(j))+x_min(j);

		double func_val_exact_seed = Eggholder(seedxnotnormalized.memptr());

		//	printf("ftilde_seed = %10.7f\n",ftilde_seed);
		//	printf("fex_seed    = %10.7f\n",func_val_exact_seed);


		vec grad(dim);


		estimate_gradient(regression_weights,
				kriging_params,
				grad,
				R_inv_ys_min_beta,
				seedx,
				beta0,
				X_grad);

		//	printf("estimated gradient at the seeding point=\n");
		//	grad.print();
		//	printf("\n");


		seedxb(0)=0.0;
		seedxb(1)=0.0;
		Eggholder_adj(seedxnotnormalized.memptr(), seedxb.memptr());

		//	printf("exact gradient at the seeding point=\n");
		//	seedxb.print();
		//	printf("\n");


		/*estimate functional value with the linear model */


		double ftilde_linmodel = ftilde_seed + dot((x-seedxnotnormalized),grad);

//		printf("ftilde_linmodel    = %10.7f\n",ftilde_linmodel);


		double func_val_exact = Eggholder(x.memptr());

//		printf("func_val_exact    = %10.7f\n",func_val_exact);


		double ftilde_x= calculate_f_tilde(xnorm,
				X_func,
				beta0(0),
				regression_weights.col(0),
				R_inv_ys_min_beta_func,
				kriging_params.col(0));

//		printf("ftilde_x    = %10.7f\n",ftilde_x);


		error_kriging    += pow((ftilde_x-func_val_exact),2.0);
		error_new_method += pow((ftilde_linmodel-func_val_exact),2.0);

//		printf("error_kriging = %10.7f\n", error_kriging);
//		printf("error_new_method = %10.7f\n",error_new_method);


	}


	error_new_method = sqrt(error_new_method/50000);
	error_kriging    = sqrt(error_kriging/50000);

	printf("error_kriging = %10.7f\n", error_kriging);
	printf("error_new_method = %10.7f\n",error_new_method);
	printf("ratio = %10.7f\n", error_kriging/error_new_method);



	exit(1);



















	/*estimate gradient*/










	//	int number_of_validation_points_func= (n_g_evals+n_f_evals)*1/5;
	//
	//	/* list of indices for the validation points */
	//	int* validation_point_indices = new int[number_of_validation_points_func];
	//
	//
	//	for(int mc_iter=0;mc_iter< 1000000; mc_iter++){
	//
	//
	//
	//		/* generate a random dx vector */
	//
	//		for(int i=0;i<dim;i++) dx(i) =  rand() % 100;
	//
	//
	//
	//		for(int cv_iter=0;cv_iter<20;cv_iter++){
	//
	//
	//			generate_validation_set(validation_point_indices, n_g_evals+n_f_evals, number_of_validation_points_func);
	//
	//
	//
	//
	//		}

















	return 0;

}
