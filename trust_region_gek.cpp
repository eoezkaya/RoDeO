
#include <armadillo>
#include <random>
#include <map>
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


void train_optimal_radius(
		double &r,
		mat &data_functional_values,
		mat &data_gradients,
		int linear_regression,
		mat &regression_weights,
		mat &kriging_params,
		int max_iter){

	const double reg_param=0.00000001; /* regularization parameter for the kriging */
	int dim = data_functional_values.n_cols-1; /* dimension of the problem */

	int validation_set_size = (data_functional_values.n_rows) / 5.0;  /* validation set size (default 20 percent of the points) */
	int reduced_set_size = data_functional_values.n_rows - validation_set_size;
	int reduced_set_size_grad = data_gradients.n_rows - validation_set_size;

	mat R(reduced_set_size,reduced_set_size);
	R.fill(0.0);

	uvec val_set_indices(validation_set_size);
	uvec val_set_indices_grad(validation_set_size);

	/* number of points with functional values */
	int n_f_evals = data_functional_values.n_rows;
	/* number of points with functional values + gradient sensitivities*/
	int n_g_evals = data_gradients.n_rows;


	int number_of_outer_iterations = 2000;

#if 0
	printf("calling train_seeding_points...\n");
	printf("linear regression = %d\n",linear_regression);
	printf("regression weights: \n");
	regression_weights.print();
	printf("kriging parameters: \n");
	kriging_params.print();
	printf("data for functional values:\n");
	data_functional_values.print();
	printf("data for gradients :\n");
	data_gradients.print();
	printf("validation set size =  %d\n",validation_set_size);
	printf("dim = %d\n",dim);
	printf("number of points with functional values = %d\n",n_f_evals);
	printf("number of points with sensitivity gradients = %d\n",n_g_evals);
#endif


	mat X_func,X_grad;


	if(n_g_evals > 0){

		X_grad = data_gradients.submat(0, 0, n_g_evals - 1, dim-1);


	}


	if(n_g_evals > 0){

		X_func = data_functional_values.submat(0, 0, n_f_evals - 1, dim-1);

	}


#if 0
	printf("X_func = \n");
	X_func.print();

	printf("X_grad = \n");
	X_grad.print();
#endif



	/* find minimum and maximum of the columns of data */

	vec x_max(dim);
	x_max.fill(0.0);

	vec x_min(dim);
	x_min.fill(0.0);

	for (int i = 0; i < dim; i++) {
		x_max(i) = X_func.col(i).max();
		x_min(i) = X_func.col(i).min();

	}
#if 0
	printf("maximum = \n");
	x_max.print();

	printf("minimum = \n");
	x_min.print();
#endif


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

#if 0
	printf("X_func(normalized) = \n");
	X_func.print();
	printf("X_grad(normalized) = \n");
	X_grad.print();
	printf("data gradient = \n");
	data_gradients.print();
#endif



	mat Rfunc(reduced_set_size,reduced_set_size);
	mat X_actual(reduced_set_size,dim);
	mat X_actual_grad(reduced_set_size_grad,dim);
	vec ys_actual(reduced_set_size);
	vec ys_actual_grad(reduced_set_size_grad);

	vec I= ones(reduced_set_size);
	vec Igrad= ones(reduced_set_size_grad);


	vec ys = data_functional_values.col(dim);
	vec theta = kriging_params.col(0).head(dim);
	vec gamma = kriging_params.col(0).tail(dim);



	double avg_cv_error=0.0;
	double avg_cv_error_best = LARGE;


	double min_r = 0.0;
	double max_r = 0.0;
	double average_distance = 0.0;
	double best_r = 0.0;

	srand(time(NULL));

#if 0
	printf("X_func:\n");
	X_func.print();
#endif

	int n_probes = 10000;
	vec probe_distances(n_probes);


	for(int i=0;i<n_probes; i++){

		rowvec np(dim);
		np.fill(0.0);


		/* generate a random point in the design space */
		for(int i=0; i<dim; i++){

			np(i) = RandomDouble(0.0,1.0)/dim;


		}

		double min_dist=0.0;
		int indx;

		/* find the closest seeding point to the np in the data set */
		findKNeighbours(X_func,
				np,
				1,
				&min_dist,
				&indx,
				2);


#if 0
		printf("point:\n");
		np.print();
		printf("nearest neighbour is:\n");
		X_func.row(indx).print();
		printf("minimum distance (L2 norm)= %10.7f\n\n",min_dist);

#endif

		probe_distances(i)=min_dist;


	}

	average_distance = mean(probe_distances);
	double std_distance = stddev(probe_distances);
	double var_distance = var(probe_distances);




	vec probe_distances_sample(X_func.n_rows);

	for(int i=0;i<X_func.n_rows; i++){

		rowvec np = X_func.row(i);

		double min_dist[2]={0.0,0.0};
		int indx[2];

		/* find the closest seeding point to the np in the data set */
		findKNeighbours(X_func,
				np,
				2,
				min_dist,
				indx,
				2);

#if 0
		printf("point:\n");
		np.print();
		printf("nearest neighbour is:\n");
		X_func.row(indx[0]).print();
		printf("minimum distance (L2 norm)= %10.7f\n\n",min_dist[0]);
		printf("second nearest neighbour is:\n");
		X_func.row(indx[1]).print();
		printf("minimum distance (L2 norm)= %10.7f\n\n",min_dist[1]);

#endif

		if(min_dist[0] > min_dist[1]){

			probe_distances_sample(i)=min_dist[0];
		}
		else{

			probe_distances_sample(i)=min_dist[1];
		}

	}

	double average_distance_sample = mean(probe_distances_sample);


	max_r = -log(0.1)/ max(probe_distances_sample);
	min_r = 0.0;
	max_r = 2.5;

	double dr = (max_r - min_r)/max_iter;

#if 0
	printf("average distance = %10.7f\n",average_distance);
	printf("average distance for the samples= %10.7f\n",average_distance_sample);
	printf("standart deviation = %10.7f\n",std_distance);
	printf("variance           = %10.7f\n",var_distance);
	printf("min                = %10.7f\n",min(probe_distances));
	printf("max                = %10.7f\n",max(probe_distances));
	printf("min (sample)       = %10.7f\n",min(probe_distances_sample));
	printf("max (sample)       = %10.7f\n",max(probe_distances_sample));
	printf("max_r = %10.7f\n",max_r);
	printf("factor at maximum distance at max r = %10.7f\n",exp(-max_r*max(probe_distances_sample)));
#endif


	int iter_sp = 0;
	std::random_device rd;

	while(1){ /* in this loop new radius values are tried */



		r = min_r+iter_sp*dr;

#if 0
		printf("iter_sp = %d\n",iter_sp);
		printf("r = %10.7f\n",r);
#endif



		double cv_error = 0.0;

		/* outer iterations loop for cv */

		for(int outer_iter=0; outer_iter<number_of_outer_iterations; outer_iter++){

			/* generate the set of indices for the validation set */
			generate_validation_set(val_set_indices, X_func.n_rows);

#if 0
			trans(val_set_indices).print();
#endif

			for(int i=0; i<val_set_indices.size(); i++){

				val_set_indices_grad(i) = val_set_indices(i)+ (n_f_evals - n_g_evals);
			}

#if 0
			trans(val_set_indices_grad).print();
			trans(val_set_indices).print();
#endif


			/* remove validation set from data : X_actual= X_func-val_set_indices*/
			remove_validation_points_from_data(X_func, ys, val_set_indices, X_actual, ys_actual);


			compute_R_matrix(theta,gamma,reg_param,Rfunc,X_actual);
#if 0
			Rfunc.print();
#endif

			mat Rinv = inv(Rfunc);

			double beta0 = (1.0/dot(I,Rinv*I)) * (dot(I,Rinv*ys_actual));

			vec R_inv_ys_min_beta = Rinv* (ys_actual-beta0* I);

#if 0
			printf("outer_iter = %d\n",outer_iter);
			val_set_indices.print();
			printf("X_func =\n");
			X_func.print();
			printf("X_func (reduced)=\n");
			X_actual.print();
			printf("ys =\n");
			ys.print();
			printf("ys (reduced)=\n");
			ys_actual.print();
#endif

			vec beta0grad(dim);
			mat R_inv_ys_min_beta_grad;

			for(int i=0; i<dim;i++){

				vec ysgrad = data_gradients.col(dim+i+1);
#if 0
				printf("ysgrad:\n");
				trans(ysgrad).print();
#endif

				remove_validation_points_from_data(X_grad, ysgrad, val_set_indices_grad, X_actual_grad, ys_actual_grad);

#if 0
				printf("ys_actual_grad:\n");
				trans(ys_actual_grad).print();
#endif
				vec theta_grad = kriging_params.col(i+1).head(dim);
				vec gamma_grad = kriging_params.col(i+1).tail(dim);

				mat Rgrad(reduced_set_size_grad,reduced_set_size_grad);
				Rgrad.fill(0.0);
				compute_R_matrix(theta_grad,gamma_grad,reg_param,Rgrad,X_actual_grad);

				mat Rinvgrad = inv(Rgrad);

				beta0grad(i) = (1.0/dot(Igrad,Rinvgrad*Igrad)) * (dot(Igrad,Rinvgrad*ys_actual_grad));

				vec R_inv_ys_min_beta_temp = Rinvgrad* (ys_actual_grad-beta0grad(i)* Igrad);

				R_inv_ys_min_beta_grad.insert_cols(i,R_inv_ys_min_beta_temp);

			}

#if 0
			/* validate kriging models at reduced set */
			for(unsigned int i=0;i<X_actual_grad.n_rows;i++){

				rowvec x = X_actual_grad.row(i);
				rowvec xb(dim);
				vec grad(dim);

				for(int k=0;k<dim;k++){

					grad(k) = calculate_f_tilde(x,
							X_actual_grad,
							beta0grad(k),
							regression_weights.col(k+1),
							R_inv_ys_min_beta_grad.col(k),
							kriging_params.col(k+1));

				}

				for(int j=0; j<dim;j++) {

					x(j) = dim*x(j)* (x_max(j) - x_min(j))+x_min(j);
				}

				xb.fill(0.0);
				double func_val_exact = Eggholder_adj(x.memptr(),xb.memptr());

				printf("\n");
				x.print();
				printf("grad exact[0] = %10.7f grad exact[1] = %10.7f\n",xb[0],xb[1]);
				printf("grad approx[0] = %10.7f grad approx[1] = %10.7f\n",grad(0),grad(1));

			}
#endif


			for(int inner_it=0; inner_it< val_set_indices.size(); inner_it++){
#if 0
				printf("\n\ninner iteration %d\n",inner_it);
				printf("trying point %d\n",val_set_indices(inner_it));
#endif

				/* get the validation point */
				rowvec x = X_func.row(val_set_indices(inner_it));

				/* calculate x in original coordinates */
				rowvec x_not_normalized(dim);
				x_not_normalized.fill(0.0);

				for(int j=0; j<dim;j++) {

					x_not_normalized(j) = dim*x(j)* (x_max(j) - x_min(j))+x_min(j);
				}

#if 0
				printf("x:\n");
				x.print();
				printf("x in original coordinates:\n");
				x_not_normalized.print();
#endif
				double min_dist=0.0;
				int indx;

				/* find the closest point to the validation point in the actual data set*/
				findKNeighbours(X_actual,
						x,
						1,
						&min_dist,
						&indx,
						2);

				if(fabs(min_dist)< EPSILON ){

					printf("fabs(min_dist)< SMALL!\n");
					exit(-1);
				}

				/* sp is the nearest data point */
				rowvec sp =  X_actual.row(indx);
				rowvec sp_not_normalized(dim);
				sp_not_normalized.fill(0.0);

				for(int j=0; j<dim;j++) {

					sp_not_normalized(j) = dim*sp(j)* (x_max(j) - x_min(j))+x_min(j);
				}

#if 0
				printf("X_actual:\n");
				X_actual.print();

				printf("X_func:\n");
				X_func.print();

				printf("nearest point:\n");
				printf("index = %d\n",indx);
				sp.print();
				printf("nearest point in original coordinates:\n");
				sp_not_normalized.print();
#endif


				rowvec xdiff = x-sp;
				double distance = L2norm(xdiff, dim);



				double fval_kriging = 0.0;



				/* pure kriging value at x */
				fval_kriging = calculate_f_tilde(x,
						X_actual,
						beta0,
						regression_weights.col(0),
						R_inv_ys_min_beta,
						kriging_params.col(0));




				/* calculate Taylor approximation */
				double fval_linmodel = ys_actual(indx);
# if 0
				double fexact_sp= Eggholder(sp_not_normalized.memptr());
#endif

				vec grad(dim);

				for(int k=0;k<dim;k++){

					grad(k) = calculate_f_tilde(sp,
							X_actual_grad,
							beta0grad(k),
							regression_weights.col(k+1),
							R_inv_ys_min_beta_grad.col(k),
							kriging_params.col(k+1));

				}


				double normgrad = L2norm(grad, dim);


				fval_linmodel+= dot((x_not_normalized-sp_not_normalized),grad);

				double factor = exp(-r*distance*normgrad);
				double fval = factor*fval_linmodel + (1.0-factor)*fval_kriging;

				double fexact = ys(val_set_indices(inner_it));

				double  sqr_error = (fval-fexact)* (fval-fexact);
				cv_error+= sqr_error;


#if 0

				printf("gradient vector:\n");
				trans(grad).print();
				printf("ys_actual(indx) = %10.7f\n",ys_actual(indx));
				printf("fval_linmodel = %10.7f\n",fval_linmodel);
				printf("fexact_sp = %10.7f\n",fexact_sp);
				printf("fval_kriging = %10.7f\n",fval_kriging);
				printf("diff = ");
				(x_not_normalized-sp_not_normalized).print();
				printf("first order term = %10.7f\n",dot((x_not_normalized-sp_not_normalized),grad));

				printf("factor        = %10.7f\n",factor);
				printf("r        = %10.7f\n",r);
				printf("f approx        = %10.7f\n",fval);
				printf("f exact         = %10.7f\n",fexact);
				printf("sqr_error       = %10.7f\n",sqr_error);
#endif



#if 0
				printf("cross validation error = %10.7f\n",cv_error);
#endif


			} /* end of inner cv iteration */




		} /* end of outer cv iteration */

		avg_cv_error = cv_error/(number_of_outer_iterations*val_set_indices.size());


#if 0
		printf("average cross validation error = %10.7f\n",avg_cv_error);
#endif



		if (avg_cv_error < avg_cv_error_best){
#if 0
			printf("average error decreased, the new point is accepted\n");
			printf("new points now\n");
#endif
			avg_cv_error_best = avg_cv_error;
			best_r= r;

		}
		else{


#if 0
			printf("average error increased\n");

#endif
		}
#if 1
		printf("r= %10.7f avg_error = %10.7f avg_error_best = %10.7f best r = %10.7f\n",r,avg_cv_error,avg_cv_error_best,best_r);
#endif
		if(iter_sp > max_iter ) break;


		iter_sp ++;



	} /* end of while(1) */



}
int train_TRGEK_response_surface(std::string input_file_name,
		int linear_regression,
		mat &regression_weights,
		mat &kriging_params,
		mat &R_inv_ys_min_beta,
		double &radius,
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
		printf("Error: Cannot open input file %s...\n",input_file_name.c_str() );
		exit(-1);
	}


	std::vector<double> temp;

	std::string str;
	int count_f=0;
	int count_g=0;

	while (std::getline(in, str)) {

		std::string delimiter = ",";

		size_t pos = 0;
		std::string token;
		while ((pos = str.find(delimiter)) != std::string::npos) {
			token = str.substr(0, pos);

			temp.push_back(atof(token.c_str()));


			str.erase(0, pos + delimiter.length());
		}
		temp.push_back(atof(str.c_str()));



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


				newrow(count)=*it;
				count++;

			}

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

#if 0
	printf("number of points with only functional values = %d\n",n_f_evals);
	printf("number of points with sensitivity gradients = %d\n",n_g_evals);
#endif

#if 0
	printf("functional values data :\n");
	data_functional_values.print();
	printf("\n");


	printf("gradient data (raw) :\n");
	data_gradients.print();
	printf("\n");
#endif

	mat X_func,X_grad;


	if(n_g_evals > 0){


		X_grad = data_gradients.submat(0, 0, n_g_evals - 1, dim-1);

		mat func_values_grad = data_gradients.submat( 0, 0, n_g_evals-1, dim );

		data_functional_values.insert_rows(n_f_evals,func_values_grad);

#if 0
		printf("data_functional_values = \n");
		data_functional_values.print();
#endif
	}


	if(n_g_evals > 0){

		X_func = data_functional_values.submat(0, 0, n_g_evals+n_f_evals - 1, dim-1);

	}


#if 0
	printf("X_func = \n");
	X_func.print();

	printf("X_grad = \n");
	X_grad.print();
#endif

	/* find minimum and maximum of the columns of data */
	vec x_max(dim);
	x_max.fill(0.0);

	vec x_min(dim);
	x_min.fill(0.0);

	for (int i = 0; i < dim; i++) {
		x_max(i) = X_func.col(i).max();
		x_min(i) = X_func.col(i).min();

	}
#if 0
	printf("maximum = \n");
	x_max.print();

	printf("minimum = \n");
	x_min.print();
#endif
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

#if 0
	printf("X_func(normalized) = \n");
	X_func.print();

	printf("X_grad(normalized) = \n");
	X_grad.print();
#endif



	if(n_f_evals == 0){

#if 1
		printf("all samples have gradients: training only functional values\n");

#endif
		std::string kriging_input_filename      = "trust_region_gek_input.csv";
		std::string kriging_hyperparam_filename = "trust_region_gek_hyperparam.csv";
		vec regression_weights_single_output=zeros(dim+1);
		vec kriging_weights_single_output=zeros(dim);

		mat kriging_input_data;
		data_functional_values.save(kriging_input_filename, csv_ascii);

#if 1
		printf("kriging_input_data = \n");
		data_functional_values.print();
		printf("\n");
#endif

		train_kriging_response_surface(kriging_input_filename,
				kriging_hyperparam_filename,
				linear_regression,
				regression_weights_single_output,
				kriging_weights_single_output,
				reg_param,
				max_number_of_function_calculations,
				CSV_ASCII);

		regression_weights.insert_cols(0,regression_weights_single_output);
		kriging_params.insert_cols(0,kriging_weights_single_output);

#if 1
		printf("regression_weights = \n");
		regression_weights_single_output.print();
		printf("kriging_weights = \n");
		kriging_weights_single_output.print();
#endif


	}

	else{

		for(int i=0;i<dim+1;i++){

			printf("Training %dth variable\n",i);

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

		}

	} /* end of else */


	vec ysfunc = data_functional_values.col(dim);
	vec Ifunc  = ones(X_func.n_rows);

	mat Rfunc(X_func.n_rows,X_func.n_rows);
	vec theta = kriging_params.col(0).head(dim);
	vec gamma = kriging_params.col(0).tail(dim);
	compute_R_matrix(theta,gamma,reg_param,Rfunc,X_func);

	mat Rinvfunc = inv(Rfunc);

	beta0(0) = (1.0/dot(Ifunc,Rinvfunc*Ifunc)) * (dot(Ifunc,Rinvfunc*ysfunc));
	vec R_inv_ys_min_beta_temp = Rinvfunc* (ysfunc-beta0(0)* Ifunc);
	R_inv_ys_min_beta.insert_cols(0,R_inv_ys_min_beta_temp);

#if 0
	printf("R_inv_ys_min_beta:\n");
	R_inv_ys_min_beta.print();
#endif

#if 0 /* test surrogate model for function values */
	double in_sample_error = 0.0;
	for(unsigned int i=0;i<X_func.n_rows;i++){

		rowvec x = X_func.row(i);



		double func_val = calculate_f_tilde(x,
				X_func,
				beta0(0),
				regression_weights.col(0),
				R_inv_ys_min_beta.col(0),
				kriging_params.col(0));



		for(int j=0; j<dim;j++) x(j) = dim*x(j)* (x_max(j) - x_min(j))+x_min(j);


		double func_val_exact = Eggholder(x.memptr());
		in_sample_error+= (func_val_exact-func_val)*(func_val_exact-func_val);


		printf("\n");
		printf("x[%d] = ",i);
		x.print();
		printf("\n");
		printf("ftilde = %10.7f fexact= %10.7f\n",func_val,func_val_exact );
		printf("in sample error = %20.15f\n", in_sample_error);


	}
#endif


	if(n_f_evals != 0){

		for(int i=0; i<dim; i++){

#if 0
			printf("i = %d\n",i);
#endif

			vec ysgrad = data_gradients.col(i+1+dim);

#if 0
			printf("ysgrad: \n",i);
			ysgrad.print();
#endif


			vec Igrad = ones(X_grad.n_rows);

			mat Rgrad(X_grad.n_rows,X_grad.n_rows);
			vec theta = kriging_params.col(i+1).head(dim);
			vec gamma = kriging_params.col(i+1).tail(dim);
			compute_R_matrix(theta,gamma,reg_param,Rgrad,X_grad);

#if 0
			printf("Rgrad: \n",i);
			Rgrad.print();
#endif
			mat Rinvgrad = inv(Rgrad);

			beta0(i+1) = (1.0/dot(Igrad,Rinvgrad*Igrad)) * (dot(Igrad,Rinvgrad*ysgrad));
			vec R_inv_ys_min_beta_temp = Rinvgrad* (ysgrad-beta0(i+1)* Igrad);
			R_inv_ys_min_beta.insert_cols(i+1,R_inv_ys_min_beta_temp);

		}



	}
#if 0 /* test surrogate models for sensitivities */
	double in_sample_error = 0.0;
	for(unsigned int i=0;i<X_grad.n_rows;i++){

		vec gradient(dim);

		rowvec x = X_grad.row(i);

		for(int k=0;k<dim;k++){

			//		printf("i = %d\n",i);
			gradient(k) = calculate_f_tilde(x,
					X_grad,
					beta0(k+1),
					regression_weights.col(k+1),
					R_inv_ys_min_beta.col(k+1),
					kriging_params.col(k+1));

			in_sample_error+= pow((gradient(k)-data_gradients(i,dim+k+1)),2.0);

		}


		printf("gradient vector at x = ");
		x.print();
		trans(gradient).print();
		printf("exact gradient vector = ");
		printf("%10.7f %10.7f\n",data_gradients(i,dim+1),data_gradients(i,dim+2));
		printf("in sample error = %20.15f\n", in_sample_error);

	}
#endif


#if 1

	if (dim == 2){
		int resolution =100;

		double bounds[4];
		bounds[0]=0.0;
		bounds[1]=100.0;
		bounds[2]=0.0;
		bounds[3]=100.0;

		std::string kriging_response_surface_file_name = "Eggholder_TRGEK_response_surface.dat";

		FILE *kriging_response_surface_file = fopen(kriging_response_surface_file_name.c_str(),"w");


		double dx,dy; /* step sizes in x and y directions */
		rowvec x(2);
		rowvec xb(2);
		rowvec xnorm(2);

		double func_val;
		dx = (bounds[1]-bounds[0])/(resolution-1);
		dy = (bounds[3]-bounds[2])/(resolution-1);

		double out_sample_error=0.0;

		double max_value = -LARGE;
		double min_value =  LARGE;

		rowvec pmin(dim);
		rowvec pmax(dim);

		x[0] = bounds[0];
		for(int i=0;i<resolution;i++){
			x[1] = bounds[2];
			for(int j=0;j<resolution;j++){

				/* normalize x */
				xnorm(0)= (1.0/dim)*(x(0)- x_min(0)) / (x_max(0) - x_min(0));
				xnorm(1)= (1.0/dim)*(x(1)- x_min(1)) / (x_max(1) - x_min(1));





				double func_val = calculate_f_tilde(xnorm,
						X_func,
						beta0(0),
						regression_weights.col(0),
						R_inv_ys_min_beta.col(0),
						kriging_params.col(0));

				if(func_val < min_value){

					min_value = func_val;
					pmin = x;

				}
				if(func_val > max_value){

					max_value = func_val;
					pmax = x;

				}


				double func_val_exact = Eggholder(x.memptr());


				double sqr_error = (func_val_exact-func_val)*(func_val_exact-func_val);
				out_sample_error+= sqr_error;
				fprintf(kriging_response_surface_file,"%10.7f %10.7f %10.7f\n",x(0),x(1),func_val);

#if 0
				if(i%10 == 0) printf("%10.7f %10.7f %10.7f %10.7f\n",x(0),x(1),func_val,sqr_error);
#endif




				x[1]+=dy;
			}
			x[0]+= dx;

		}
		fclose(kriging_response_surface_file);

		out_sample_error = out_sample_error/(resolution*resolution);

		printf("out of sample error (pure Kriging) = %10.7f\n",out_sample_error);
		printf("min = %10.7f\n",min_value);
		pmin.print();
		printf("max = %10.7f\n",max_value);
		pmax.print();

		std::string file_name_for_plot = "Eggholder_purekriging_response_surface_";
		file_name_for_plot += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";

		std::string python_command = "python -W ignore plot_2d_surface.py "+ kriging_response_surface_file_name+ " "+ file_name_for_plot ;



		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");

	}
	else{
		/* high dimensional data is not visualized */

		int number_of_samples = 10000;

		rowvec x(dim);
		rowvec xb(dim);
		rowvec xnorm(dim);

		double out_sample_error=0.0;

		double max_value = -LARGE;
		double min_value =  LARGE;

		rowvec pmin(dim);
		rowvec pmax(dim);


		double parameter_bounds[16];
		parameter_bounds[0]=0.05;    // [0.05, 0.15]
		parameter_bounds[1]=0.15;

		parameter_bounds[2]=100.0;  // [100, 50000]
		parameter_bounds[3]=50000.0;

		parameter_bounds[4]=63500;  // [63500, 115600]
		parameter_bounds[5]=115600;

		parameter_bounds[6]=990;
		parameter_bounds[7]=1110; // [990, 1110]

		parameter_bounds[8]=63.1;
		parameter_bounds[9]=116; // [[63.1, 116]]

		parameter_bounds[10]=700;
		parameter_bounds[11]=820; // [700, 820]

		parameter_bounds[12]=1120;
		parameter_bounds[13]=1680; // [1120, 1680]

		parameter_bounds[14]=9855;
		parameter_bounds[15]=12045; // [9855, 12045]



		for(int i=0; i<number_of_samples; i++){

			for(int j=0; j<dim;j++){

				x(j) = RandomDouble(parameter_bounds[j*2], parameter_bounds[j*2+1]);
				xnorm(j)= (1.0/dim)*(x(j)- x_min(j)) / (x_max(j) - x_min(j));

			}

			double func_val = calculate_f_tilde(xnorm,
					X_func,
					beta0(0),
					regression_weights.col(0),
					R_inv_ys_min_beta.col(0),
					kriging_params.col(0));

			if(func_val < min_value){

				min_value = func_val;
				pmin = x;

			}
			if(func_val > max_value){

				max_value = func_val;
				pmax = x;

			}


			double func_val_exact = Borehole(x.memptr());

			double sqr_error = (func_val_exact-func_val)*(func_val_exact-func_val);
			out_sample_error+= sqr_error;


#if 1
			printf("at x: ");
			x.print();
			printf("func_val = %10.7f func_val_exact = %10.7f sqr_error = %10.7f\n",func_val,func_val_exact,sqr_error);
#endif



		}

		out_sample_error = out_sample_error/number_of_samples;

		printf("out of sample error (pure Kriging) = %10.7f\n",out_sample_error);
		printf("min = %10.7f\n",min_value);
		pmin.print();
		printf("max = %10.7f\n",max_value);
		pmax.print();



	} /* end of else */


#endif

#if 1

	if (dim == 2){ /* visualize only linear model response surface */
		mat seeding_points = X_func;
		int resolution =100;

		double bounds[4];
		bounds[0]=0.0;
		bounds[1]=100.0;
		bounds[2]=0.0;
		bounds[3]=100.0;

		std::string kriging_response_surface_file_name = "Eggholder_pure_linmodel_response_surface.dat";

		FILE *kriging_response_surface_file = fopen(kriging_response_surface_file_name.c_str(),"w");


		double dx,dy; /* step sizes in x and y directions */
		rowvec x(2);
		rowvec xnorm(2);


		dx = (bounds[1]-bounds[0])/(resolution-1);
		dy = (bounds[3]-bounds[2])/(resolution-1);

		double out_sample_error=0.0;

		double max_value = -LARGE;
		double min_value =  LARGE;
		rowvec pmin(dim);
		rowvec pmax(dim);


		x[0] = bounds[0];
		for(int i=0;i<resolution;i++){
			x[1] = bounds[2];
			for(int j=0;j<resolution;j++){

				/* normalize x */
				xnorm(0)= (1.0/dim)*(x(0)- x_min(0)) / (x_max(0) - x_min(0));
				xnorm(1)= (1.0/dim)*(x(1)- x_min(1)) / (x_max(1) - x_min(1));




				double min_dist=0.0;
				int indx;

				/* find the closest seeding point */
				findKNeighbours(seeding_points,
						xnorm,
						1,
						&min_dist,
						&indx,
						dim);


				rowvec sp =  seeding_points.row(indx);
				rowvec sp_not_normalized(dim);
				sp_not_normalized.fill(0.0);

				for(int j=0; j<dim;j++) sp_not_normalized(j) = dim*sp(j)* (x_max(j) - x_min(j))+x_min(j);

				double func_val = calculate_f_tilde(sp,
						X_func,
						beta0(0),
						regression_weights.col(0),
						R_inv_ys_min_beta.col(0),
						kriging_params.col(0));


				vec gradient(dim);

				for(int k=0;k<dim;k++){

					//		printf("i = %d\n",i);
					gradient(k) = calculate_f_tilde(sp,
							X_grad,
							beta0(k+1),
							regression_weights.col(k+1),
							R_inv_ys_min_beta.col(k+1),
							kriging_params.col(k+1));


				}


				double ftilde_linmodel = func_val + dot((x-sp_not_normalized),gradient);
				fprintf(kriging_response_surface_file,"%10.7f %10.7f %10.7f\n",x(0),x(1),ftilde_linmodel);

				double func_val_exact = Eggholder(x.memptr());

				double srq_error = (ftilde_linmodel-func_val_exact)*(ftilde_linmodel-func_val_exact);
				out_sample_error+=srq_error;
#if 0
				printf("x:\n");
				x.print();
				printf("xnorm:\n");
				xnorm.print();
				printf("nearest point:\n");
				sp.print();
				printf("sp in original coordinates:\n");
				sp_not_normalized.print();
				printf("gradient:\n");
				trans(gradient).print();
				printf("func_val = %10.7f\n",func_val);
				printf("ftilde_linmodel = %10.7f\n",ftilde_linmodel);
				printf("func_val_exact = %10.7f\n",func_val_exact);
#endif


				if(ftilde_linmodel < min_value){

					min_value = ftilde_linmodel;
					pmin = x;

				}
				if(ftilde_linmodel > max_value){

					max_value = ftilde_linmodel;
					pmax = x;

				}

				x[1]+=dy;
			}
			x[0]+= dx;

		}
		fclose(kriging_response_surface_file);

		out_sample_error = out_sample_error/(resolution*resolution);


		printf("out of sample error (Taylor approximation) = %10.7f\n",out_sample_error);
		printf("min = %10.7f\n",min_value);
		pmin.print();
		printf("max = %10.7f\n",max_value);
		pmax.print();

		std::string file_name_for_plot = "Eggholder_linmodel_response_surface_";
		file_name_for_plot += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";

		std::string python_command = "python -W ignore plot_2d_surface.py "+ kriging_response_surface_file_name+ " "+ file_name_for_plot ;



		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");

	}

	else{ /*for higher dimensions */


		int number_of_samples = 10000;

		rowvec x(dim);
		rowvec xb(dim);
		rowvec xnorm(dim);

		double out_sample_error=0.0;

		double max_value = -LARGE;
		double min_value =  LARGE;

		rowvec pmin(dim);
		rowvec pmax(dim);


		double parameter_bounds[16];
		parameter_bounds[0]=0.05;    // [0.05, 0.15]
		parameter_bounds[1]=0.15;

		parameter_bounds[2]=100.0;  // [100, 50000]
		parameter_bounds[3]=50000.0;

		parameter_bounds[4]=63500;  // [63500, 115600]
		parameter_bounds[5]=115600;

		parameter_bounds[6]=990;
		parameter_bounds[7]=1110; // [990, 1110]

		parameter_bounds[8]=63.1;
		parameter_bounds[9]=116; // [[63.1, 116]]

		parameter_bounds[10]=700;
		parameter_bounds[11]=820; // [700, 820]

		parameter_bounds[12]=1120;
		parameter_bounds[13]=1680; // [1120, 1680]

		parameter_bounds[14]=9855;
		parameter_bounds[15]=12045; // [9855, 12045]



		for(int i=0; i<number_of_samples; i++){

			/* generate a random input vector and normalize it */

			for(int j=0; j<dim;j++){

				x(j) = RandomDouble(parameter_bounds[j*2], parameter_bounds[j*2+1]);
				xnorm(j)= (1.0/dim)*(x(j)- x_min(j)) / (x_max(j) - x_min(j));

			}

			double min_dist=0;
			int indx = -1;

			/* find the closest seeding point */
			findKNeighbours(X_grad,
					xnorm,
					1,
					&min_dist,
					&indx,
					1);


			double func_val = data_gradients(indx,dim);

			vec gradient(dim);

			for(int j=0; j<dim; j++) gradient(j)= data_gradients(indx,j+dim+1);

			rowvec sp =  X_grad.row(indx);
			rowvec sp_not_normalized(dim);
			sp_not_normalized.fill(0.0);

			for(int j=0; j<dim;j++) sp_not_normalized(j) = dim*sp(j)* (x_max(j) - x_min(j))+x_min(j);

			double ftilde_linmodel = func_val + dot((x-sp_not_normalized),gradient);

			double func_val_exact = Borehole(x.memptr());

			double srq_error = (ftilde_linmodel-func_val_exact)*(ftilde_linmodel-func_val_exact);
			out_sample_error+=srq_error;

#if 1

			printf("x:\n");
			x.print();
			printf("xnorm:\n");
			xnorm.print();
			printf("closest neighbour:\n");
			X_grad.row(indx).print();
			data_gradients.row(indx).print();
			trans(gradient).print();
			printf("functional value = %10.7f\n", func_val);
			printf("ftilde_linmodel = %10.7f\n", ftilde_linmodel);
			printf("func_val_exact = %10.7f\n", func_val_exact);

			exit(1);

#endif






		}





	}
#endif


	train_optimal_radius(
			radius,
			data_functional_values,
			data_gradients,
			linear_regression,
			regression_weights,
			kriging_params,
			200);




#if 1

	if (dim == 2){ /* visualize only linear model response surface */

		int resolution =100;

		double bounds[4];
		bounds[0]=0.0;
		bounds[1]=100.0;
		bounds[2]=0.0;
		bounds[3]=100.0;

		std::string kriging_response_surface_file_name = "Eggholder_hybrid_response_surface.dat";

		FILE *kriging_response_surface_file = fopen(kriging_response_surface_file_name.c_str(),"w");


		double dx,dy; /* step sizes in x and y directions */
		rowvec x(2);
		rowvec xnorm(2);


		dx = (bounds[1]-bounds[0])/(resolution-1);
		dy = (bounds[3]-bounds[2])/(resolution-1);

		double out_sample_error=0.0;

		double max_value = -LARGE;
		double min_value =  LARGE;
		double max_exactvalue = -LARGE;
		double min_exactvalue =  LARGE;
		rowvec pmin(dim);
		rowvec pmax(dim);
		rowvec pminex(dim);
		rowvec pmaxex(dim);

		x[0] = bounds[0];
		for(int i=0;i<resolution;i++){
			x[1] = bounds[2];
			for(int j=0;j<resolution;j++){

				/* normalize x */
				xnorm(0)= (1.0/dim)*(x(0)- x_min(0)) / (x_max(0) - x_min(0));
				xnorm(1)= (1.0/dim)*(x(1)- x_min(1)) / (x_max(1) - x_min(1));



				double min_dist=0.0;
				int indx;

				/* find the closest point */
				findKNeighbours(X_func,
						xnorm,
						1,
						&min_dist,
						&indx,
						dim);


				rowvec sp =  X_func.row(indx);
				rowvec sp_not_normalized(dim);
				sp_not_normalized.fill(0.0);

				for(int j=0; j<dim;j++) {

					sp_not_normalized(j) = dim*sp(j)* (x_max(j) - x_min(j))+x_min(j);
				}

#if 0
				printf("x:\n");
				x.print();
				printf("xnorm:\n");
				xnorm.print();
				printf("nearest point:\n");
				sp.print();
				printf("nearest point in original coordinates:\n");
				sp_not_normalized.print();
#endif

				rowvec xdiff = xnorm-sp;
				double distance = L2norm(xdiff, dim);



				//				if(ridge_flag) factor = factor*2;

				double fval_kriging = 0.0;


				/* pure kriging value at x = xnorm*/
				fval_kriging = calculate_f_tilde(xnorm,
						X_func,
						beta0(0),
						regression_weights.col(0),
						R_inv_ys_min_beta.col(0),
						kriging_params.col(0));




				/* calculate Taylor approximation */
				double fval_linmodel = ysfunc(indx);


				vec grad(dim);

				for(int k=0;k<dim;k++){

					grad(k) = calculate_f_tilde(sp,
							X_grad,
							beta0(k+1),
							regression_weights.col(k+1),
							R_inv_ys_min_beta.col(k+1),
							kriging_params.col(k+1));


				}

				double normgrad= L2norm(grad, dim);

				double factor = exp(-radius*distance*normgrad);

				fval_linmodel= fval_linmodel + dot((x-sp_not_normalized),grad);


				double fval = factor*fval_linmodel + (1.0-factor)*fval_kriging;


				fprintf(kriging_response_surface_file,"%10.7f %10.7f %10.7f\n",x(0),x(1),fval);

				double func_val_exact = Eggholder(x.memptr());

				if(fval < min_value){

					min_value = fval;
					pmin = x;

				}
				if(fval > max_value){

					max_value = fval;
					pmax = x;

				}


				if(func_val_exact < min_exactvalue){

					min_exactvalue = func_val_exact;
					pminex = x;

				}
				if(func_val_exact > max_exactvalue){

					max_exactvalue = func_val_exact;
					pmaxex = x;

				}



#if 0
				printf("factor = %10.7f\n", factor);
				printf("fkriging      = %10.7f flinmodel       = %10.7f\n",fval_kriging,fval_linmodel);
				printf("ftilde       = %10.7f fexact       = %10.7f\n",fval,func_val_exact);

#endif

				double srq_error = (fval-func_val_exact)*(fval-func_val_exact);
				out_sample_error+=srq_error;



				x[1]+=dy;
			}
			x[0]+= dx;

		}
		fclose(kriging_response_surface_file);

		out_sample_error = out_sample_error/(resolution*resolution);

		printf("out of sample error (hybrid model) = %10.7f\n",out_sample_error);
		printf("min = %10.7f\n",min_value);
		pmin.print();
		printf("max = %10.7f\n",max_value);
		pmax.print();
		printf("min (exact) = %10.7f\n",min_exactvalue);
		pminex.print();
		printf("max (exact) = %10.7f\n",max_exactvalue);
		pmaxex.print();

		std::string file_name_for_plot = "Eggholder_hybrid_response_surface_";
		file_name_for_plot += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";

		std::string python_command = "python -W ignore plot_2d_surface.py "+ kriging_response_surface_file_name+ " "+ file_name_for_plot ;



		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");

	}
#endif




	return 0;

}




