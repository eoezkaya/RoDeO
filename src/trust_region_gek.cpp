
#include <armadillo>
#include <random>
#include <map>
using namespace arma;


#include "trust_region_gek.hpp"
#include "Rodeo_macros.hpp"
#include "kriging_training.hpp"
#include "test_functions.hpp"
#include "auxilliary_functions.hpp"
#include "kernel_regression_cuda.h"
#include "Rodeo_globals.hpp"




double ftildeAggModel(AggregationModel &model_settings,
		rowvec &xp,
		rowvec xpNotNormalized,
		mat &X,
		mat &XTrainingNotNormalized,
		vec &yTrainingNotNormalized,
		mat &gradTrainingNotNormalized,
		vec &x_min,
		vec &x_max){


	int d = xp.size();
#if 0
	printf("xp = \n");
	xp.print();
	printf("xp (not normalized)= \n");
	xpNotNormalized.print();
#endif
	double ftildeKriging = calculate_f_tilde(xp,
			X,
			model_settings.beta0,
			model_settings.regression_weights,
			model_settings.R_inv_ys_min_beta,
			model_settings.kriging_weights);

#if 0
	printf("ftilde (Kriging) = %10.7f\n",ftildeKriging);
#endif


	double fKernelRegression = 0.0;


	fKernelRegression = kernelRegressorNotNormalized(X,
			XTrainingNotNormalized,
			yTrainingNotNormalized,
			gradTrainingNotNormalized,
			xpNotNormalized,
			x_min,
			x_max,
			model_settings.M,
			model_settings.sigma);
#if 0
	printf("ftilde (Kernel Regression with gradient) = %10.7f\n",fKernelRegression);
#endif

	double min_dist=0.0;
	int indx;

	/* find the closest seeding point to the xp in the data set */
	findKNeighbours(X,
			xp,
			1,
			&min_dist,
			&indx,
			1);


	rowvec xg = XTrainingNotNormalized.row(indx);

#if 0
	printf("xg = \n");
	xg.print();
#endif

	rowvec xdiff = xpNotNormalized-xg;
	double distance = L1norm(xdiff, d);

#if 0
	printf("L1 distance between xp and xg = %10.7f\n",distance);
#endif


	rowvec grad = gradTrainingNotNormalized.row(indx);

#if 0
	printf("grad = \n");
	grad.print();
#endif

	double normgrad = L1norm(grad, d);

#if 0
	printf("L1 norm of grad = %10.7f\n",normgrad);
#endif


	double w1 = exp(-model_settings.rho *distance*normgrad);

#if 0
	printf("w1 = %10.7f\n",w1);
#endif


	double fval = w1*fKernelRegression + (1.0-w1)*ftildeKriging;

	return fval;
}


double calcGenErrorAggModel(AggregationModel &model_settings,
		mat Xvalidation,
		mat XvalidationNotNormalized,
		vec yvalidation,
		mat X,
		mat XTrainingNotNormalized,
		vec yTrainingNotNormalized,
		mat gradTrainingNotNormalized,
		vec x_min,
		vec x_max){


	int Nval = Xvalidation.n_rows;


	double genError = 0.0;

	for(int i=0; i<Nval;i++){

		rowvec xpNotNormalized = XvalidationNotNormalized.row(i);
		rowvec xp = Xvalidation.row(i);
#if 0
		printf("xp =\n");
		xp.print();
#endif

		double fAggModel = ftildeAggModel(model_settings,
				xp,
				xpNotNormalized,
				X,
				XTrainingNotNormalized,
				yTrainingNotNormalized,
				gradTrainingNotNormalized,
				x_min,
				x_max);

		double fexact = yvalidation(i);

#if 0
		printf("ftilde (Agg. Model) = %10.7f, fexact = %10.7f\n",fAggModel,fexact);
#endif

		genError += (fAggModel - fexact)*(fAggModel - fexact);

	}

	return genError/Nval;

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



	/* number of points with functional values */
	int n_f_evals = data_functional_values.n_rows;
	/* number of points with functional values + gradient sensitivities*/
	int n_g_evals = data_gradients.n_rows;


	int number_of_outer_iterations = 10000;

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

	for (unsigned int i = 0; i < X_func.n_rows; i++) {

		for (int j = 0; j < dim; j++) {

			X_func(i, j) = (1.0/dim)*(X_func(i, j) - x_min(j)) / (x_max(j) - x_min(j));
		}
	}


	for (unsigned int i = 0; i < X_grad.n_rows; i++) {

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



	/* obtain sample statistics */

	vec probe_distances_sample(X_func.n_rows);

	for(unsigned int i=0; i<X_func.n_rows; i++){



		rowvec np = X_func.row(i);


		double min_dist[2]={0.0,0.0};
		int indx[2];

		/* find the closest two points to the np in the data set */
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

		probe_distances_sample(i)=min_dist[1];


	}

	double average_distance_sample = mean(probe_distances_sample);

	/* obtain gradient statistics */
	double sumnormgrad = 0.0;
	for(unsigned int i=0; i<X_grad.n_rows; i++){

		vec grad(dim);

		for(int k=0;k<dim;k++){

			grad(k) = data_gradients(i,dim+1+k);


		}
		double normgrad= L1norm(grad, dim);
#if 0
		printf("entry at the data gradients:\n");
		data_gradients.row(i).print();
		printf("gradient:\n");
		trans(grad).print();
		printf("norm of the gradient = %10.7e\n",normgrad);
#endif
		sumnormgrad+= normgrad;


	}

	double avg_norm_grad= sumnormgrad/X_grad.n_rows;
#if 1
	printf("average norm grad = %10.7e\n", avg_norm_grad);
#endif
	/* define the search range for the hyperparameter r */


	min_r = 0.0;
	//	max_r = 1.0/dim;
	max_r = -2*log(0.00001)/ (max(probe_distances_sample) * avg_norm_grad);
	//	max_r = 0.01;

	double dr;

	dr = (max_r - min_r)/(max_iter);

#if 1
	printf("average distance = %10.7f\n",average_distance);
	printf("average distance for the samples= %10.7f\n",average_distance_sample);
	printf("standart deviation = %10.7f\n",std_distance);
	printf("variance           = %10.7f\n",var_distance);
	printf("min                = %10.7f\n",min(probe_distances));
	printf("max                = %10.7f\n",max(probe_distances));
	printf("min (sample)       = %10.7f\n",min(probe_distances_sample));
	printf("max (sample)       = %10.7f\n",max(probe_distances_sample));
	printf("max_r = %10.7f\n",max_r);
	printf("dr = %10.7f\n",dr);
	printf("factor at maximum distance at max r = %10.7f\n",exp(-max_r*max(probe_distances_sample)* avg_norm_grad));
#endif





	for(int iter_sp=0; iter_sp<max_iter; iter_sp++){


		/* value of the hyper parameter to be tried */
		double hyper_param = min_r+(iter_sp)*dr;


#if 0
		printf("tid = %d\n",tid);
		printf("iter_sp = %d\n",iter_sp);
		printf("hyper_param = %10.7f\n",hyper_param);
#endif


		double avg_cv_error=0.0;
		double cv_error = 0.0;

		/* outer iterations loop for cv */
#pragma omp parallel
		{

			int num_out_iter_per_core = number_of_outer_iterations/omp_get_num_threads();

			for(int outer_iter=0; outer_iter<num_out_iter_per_core; outer_iter++){



				/* shows mapping between the original data set and the reduced data set */
				uvec data_map(reduced_set_size_grad);
				data_map.fill(-1);

				uvec val_set_indices(validation_set_size);
				uvec val_set_indices_grad(validation_set_size);

				/* correlation matrix for the reduced data set */
				mat Rfunc(reduced_set_size,reduced_set_size);
				Rfunc.fill(0.0);

				mat X_actual(reduced_set_size,dim);
				mat X_actual_grad(reduced_set_size_grad,dim);

				vec ys_actual(reduced_set_size);
				vec ys_actual_grad(reduced_set_size_grad);

				vec I= ones(reduced_set_size);
				vec Igrad= ones(reduced_set_size_grad);


				vec ys = data_functional_values.col(dim);


				/* assign Kriging hyper parameters */
				vec theta = kriging_params.col(0).head(dim);
				vec gamma = kriging_params.col(0).tail(dim);


				/* generate the set of indices for the validation set */
				generate_validation_set(val_set_indices, X_func.n_rows);


				for(unsigned int i=0; i<val_set_indices.size(); i++){

					val_set_indices_grad(i) = val_set_indices(i)+ (n_f_evals - n_g_evals);
				}

#if 0
				printf("validation set grad:\n");
				trans(val_set_indices_grad).print();
				printf("validation set:\n");
				trans(val_set_indices).print();

#endif


				/* remove validation set from data : X_actual= X_func-val_set_indices*/
				remove_validation_points_from_data(X_func,
						ys,
						val_set_indices,
						X_actual,
						ys_actual,
						data_map);


				vec ys_actual_wo_linreg = ys_actual;

				/* update ys_actual if linear regression is on */

				if (linear_regression == LINEAR_REGRESSION_ON) {

					mat augmented_X(X_actual.n_rows, dim + 1);

					for (unsigned int i = 0; i < X_actual.n_rows; i++) {

						for (int j = 0; j <= dim; j++) {

							if (j == 0){

								augmented_X(i, j) = 1.0;
							}
							else {

								augmented_X(i, j) = X_actual(i, j - 1);

							}
						}
					}

					/* now update the ys vector */

					vec ys_reg = augmented_X * regression_weights;

#if 0
					for(unsigned int i=0; i<ys.size(); i++){

						printf("%10.7f %10.7f\n",ys_actual(i),ys_reg(i));
					}
#endif
					ys_actual = ys_actual - ys_reg;

#if 0
					printf("Updated ys_actual vector = \n");
#endif



				} /* end of linear regression part*/




				/* evaluate the correlation matrix R */
				compute_R_matrix(theta,gamma,reg_param,Rfunc,X_actual);
#if 0
				Rfunc.print();
#endif


				/* compute Cholesky decomposition */
				mat U(Rfunc.n_rows,Rfunc.n_rows);
				int flag = chol(U, Rfunc);

				if (flag == 0) {

					printf("Error: Ill conditioned correlation matrix in Cholesky decomposition\n");

					exit(-1);
				}

				mat L = trans(U);

				vec R_inv_ys(Rfunc.n_rows); /* R^-1* ys */
				vec R_inv_I (Rfunc.n_rows); /* R^-1* I */
				vec R_inv_ys_min_beta(Rfunc.n_rows); /* R^-1* (ys-beta0*F) */


				solve_linear_system_by_Cholesky(U, L, R_inv_ys, ys_actual);
				solve_linear_system_by_Cholesky(U, L, R_inv_I, I);

				/* compute the bias term beta0 */
				double beta0 = (1.0/dot(I,R_inv_I)) * (dot(I,R_inv_ys));

				vec ys_min_betaI = ys_actual-beta0*I;

				solve_linear_system_by_Cholesky(U, L, R_inv_ys_min_beta, ys_min_betaI );


#if 0
#pragma omp master
				{
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
					printf("n_g_evals = %d\n",n_g_evals);
					printf("n_f_evals = %d\n",n_f_evals);
				}
#endif

				vec beta0grad(dim);
				mat R_inv_ys_min_beta_grad;

				/* if some samples have only functional values, then interpolation for sensitivities is required */
				if ( n_g_evals != n_f_evals ) {

					printf("Error: This part is not implemented properly yet\n");
					exit(1);


					for(int i=0; i<dim;i++){ /* for each component of the gradient vector */

						/* assign ys vector */
						vec ysgrad = data_gradients.col(dim+i+1);
#if 0
						printf("ysgrad:\n");
						trans(ysgrad).print();
#endif

						remove_validation_points_from_data(X_grad, ysgrad, val_set_indices_grad, X_actual_grad, ys_actual_grad,data_map);

#if 0
						printf("ys_actual_grad:\n");
						trans(ys_actual_grad).print();
#endif


						/* assign hyperparameters theta and gamma */
						vec theta_grad = kriging_params.col(i+1).head(dim);
						vec gamma_grad = kriging_params.col(i+1).tail(dim);

						mat Rgrad(reduced_set_size_grad,reduced_set_size_grad);
						Rgrad.fill(0.0);
						compute_R_matrix(theta_grad,gamma_grad,reg_param,Rgrad,X_actual_grad);

						beta0grad(i) = (1.0/dot(Igrad,solve(Rgrad,Igrad))) * (dot(Igrad,solve(Rgrad,ys_actual_grad)));

						vec R_inv_ys_min_beta_temp = solve( Rgrad, (ys_actual_grad-beta0grad(i)* Igrad));

						R_inv_ys_min_beta_grad.insert_cols(i,R_inv_ys_min_beta_temp);



					}

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


				/* iterate through the validation set */
				for(unsigned int inner_it=0; inner_it< val_set_indices.size(); inner_it++){

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
					double distance = L1norm(xdiff, dim);

					double fval_kriging = 0.0;

					/* pure kriging value at x */
					fval_kriging = calculate_f_tilde(x,
							X_actual,
							beta0,
							regression_weights.col(0),
							R_inv_ys_min_beta,
							kriging_params.col(0));


					/* calculate Taylor approximation */
					double fval_linmodel = ys_actual_wo_linreg(indx);
#if 0
					double fexact_sp= Eggholder(sp_not_normalized.memptr());
#endif

					vec grad(dim);


					if ( n_g_evals == n_f_evals ){

						int map_indx = data_map(indx);
#if 0
						printf("index at the gradient data matrix = %d\n",map_indx);
						data_gradients.row(map_indx).print();
#endif

						for(int k=0;k<dim;k++){

							grad(k) = data_gradients(map_indx,dim+1+k);


						}

#if 0
						printf("grad:\n");
						trans(grad).print();
#endif

					}
					else{

						for(int k=0;k<dim;k++){

							grad(k) = calculate_f_tilde(sp,
									X_grad,
									beta0grad(k),
									regression_weights.col(k+1),
									R_inv_ys_min_beta_grad.col(k),
									kriging_params.col(k+1));

						}

					}

					double normgrad = L1norm(grad, dim);


					fval_linmodel+= dot((x_not_normalized-sp_not_normalized),grad);

					double factor = exp(-hyper_param*distance*normgrad);
					double fval = factor*fval_linmodel + (1.0-factor)*fval_kriging;

					double fexact = ys(val_set_indices(inner_it));

					double  sqr_error = (fval-fexact)* (fval-fexact);
					cv_error+= sqr_error;


#if 0
#pragma omp master
					{
						printf("\n");
						//					printf("gradient vector:\n");
						//					trans(grad).print();
						printf("ys_actual(indx) = %10.7f\n",ys_actual_wo_linreg(indx));
						printf("fval_linmodel = %10.7f\n",fval_linmodel);
						printf("fval_kriging = %10.7f\n",fval_kriging);
						//					printf("diff = ");
						//					(x_not_normalized-sp_not_normalized).print();
						//					printf("first order term = %10.7f\n",dot((x_not_normalized-sp_not_normalized),grad));
						printf("factor        = %10.7f\n",factor);
						//					printf("r        = %10.7f\n",r);
						printf("f approx        = %10.7f\n",fval);
						printf("f exact         = %10.7f\n",fexact);
						printf("sqr_error       = %10.7f\n",sqr_error);
					}
#endif



#if 0
					printf("cross validation error = %10.7f\n",cv_error);
#endif


				} /* end of inner cv iteration */




			} /* end of outer cv iteration */

		} /* end of parallel region */

		avg_cv_error = cv_error/(number_of_outer_iterations*validation_set_size);


#if 0
		printf("average cross validation error = %10.7e\n",avg_cv_error);
#endif



		if (avg_cv_error < avg_cv_error_best){
#if 0
			printf("average error decreased, the new point is accepted\n");
			printf("new points now\n");
#endif
			avg_cv_error_best = avg_cv_error;
			best_r= hyper_param;


		}
		else{

#if 0
			printf("average error increased\n");

#endif

		}
#if 1
		printf("it = %d r= %10.7e avg_cv_error = %10.7e avg_error_best = %10.7e best r = %10.7e\n",iter_sp,hyper_param,avg_cv_error,avg_cv_error_best,best_r);
#endif

		//		if(avg_cv_error > avg_cv_error_best*1.2 ) break;


	} /* end of for */




	r = best_r;

}



/*
 *
 * @param[in] input_file_name
 * @param[in] linear_regression
 *
 * */



int train_TRGEK_response_surface(std::string input_file_name,
		std::string hyper_parameter_file_name,
		int linear_regression,
		mat &regression_weights,
		mat &kriging_params,
		mat &R_inv_ys_min_beta,
		double &radius,
		vec &beta0,
		int &max_number_of_function_calculations,
		int dim,
		int train_hyper_param) {

	/* regularization parameter added to the diagonal of the correlation matrix */

	double reg_param=0.000001;

	printf("training trust-region Kriging response surface for the data : %s\n",
			input_file_name.c_str());


	mat data_functional_values; /* data matrix for only functional values */
	mat data_gradients;         /* data matrix for only functional values + gradient sensitivities */

	/* R^-1 * (ys-betaI) */
	vec R_inv_ys_min_beta_func;

	std::ifstream in(input_file_name);

	if(!in) {
		printf("Error: Cannot open input file %s...\n",input_file_name.c_str() );
		exit(-1);
	}

	/* read the input file */
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



		if( int(temp.size()) == dim+1){ /* function values */

			rowvec newrow(dim+1);
			int count=0;
			for (std::vector<double>::iterator it = temp.begin() ; it != temp.end(); ++it){


				newrow(count)=*it;
				count++;

			}

			count_f++;
			data_functional_values.resize(count_f, dim+1);

			data_functional_values.row(count_f-1)=newrow;



		}
		else{ /* function+gradient information */

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

		/* now we loop back and get the next line in 'str' */
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

	/* gradient data format (not normalized):
	 *
	 * x1 x2 ... xn f g1 g2 ..gn
	 *
	 *
	 * functional data format (not normalized):
	 *
	 * x1 x2 ... xn f
	 *
	 *
	 */


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


	/* normalize data matrices X_func and X_grad */

	for (unsigned int i = 0; i < X_func.n_rows; i++) {

		for (int j = 0; j < dim; j++) {

			X_func(i, j) = (1.0/dim)*(X_func(i, j) - x_min(j)) / (x_max(j) - x_min(j));
		}
	}


	for (unsigned int i = 0; i < X_grad.n_rows; i++) {

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
		printf("all samples have gradient information: training only functional values\n");

#endif

		std::string kriging_input_filename      = "trust_region_gek_input.csv";

		vec regression_weights_single_output=zeros(dim+1);
		vec kriging_weights_single_output=zeros(dim);

		mat kriging_input_data;
		data_functional_values.save(kriging_input_filename, csv_ascii);

#if 0
		printf("kriging_input_data = \n");
		data_functional_values.print();
		printf("\n");
#endif

		/* train the Kriging response surface for functional values */

		train_kriging_response_surface(kriging_input_filename,
				hyper_parameter_file_name,
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
		trans(regression_weights_single_output).print();
		printf("kriging_weights = \n");
		trans(kriging_weights_single_output).print();
#endif


	}

	else{ /* there are some sample points without gradient information */

		printf("Error: This part has not been implemented yet\n");
		exit(1);

		for(int i=0;i<dim+1;i++){

#if 1
			printf("Training %dth variable\n",i);
#endif
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

	/* update ys_func if linear regression is on */
	if (linear_regression == LINEAR_REGRESSION_ON) {

		mat augmented_X(X_func.n_rows, dim + 1);

		for (unsigned int i = 0; i < X_func.n_rows; i++) {

			for (int j = 0; j <= dim; j++) {

				if (j == 0){

					augmented_X(i, j) = 1.0;
				}
				else {

					augmented_X(i, j) = X_func(i, j - 1);

				}
			}
		}

		/* now update the ys vector */

		vec ys_reg = augmented_X * regression_weights;

#if 0
		for(unsigned int i=0; i<ysfunc.size(); i++){

			printf("%10.7f %10.7f\n",ysfunc(i),ys_reg(i));
		}
#endif
		ysfunc = ysfunc - ys_reg;

#if 0
		printf("Updated ys vector = \n");
#endif



	} /* end of linear regression part*/



	vec Ifunc  = ones(X_func.n_rows);

	mat Rfunc(X_func.n_rows,X_func.n_rows);
	vec theta = kriging_params.col(0).head(dim);
	vec gamma = kriging_params.col(0).tail(dim);

	/* correlation matrix for functional value interpolations */
	compute_R_matrix(theta,gamma,reg_param,Rfunc,X_func);



	beta0(0) = (1.0/dot(Ifunc,solve(Rfunc,Ifunc) )) * (dot(Ifunc,solve(Rfunc,ysfunc)));
	vec R_inv_ys_min_beta_temp = solve( Rfunc, (ysfunc-beta0(0)* Ifunc));
	R_inv_ys_min_beta.insert_cols(0,R_inv_ys_min_beta_temp);

#if 0
	printf("R_inv_ys_min_beta:\n");
	R_inv_ys_min_beta.print();
#endif

#if 0 /* test surrogate model for function values */
	double in_sample_error = 0.0;
	for(unsigned int i=0;i<X_func.n_rows;i++){

		/* get a sample point */
		rowvec x = X_func.row(i);



		double func_val = calculate_f_tilde(x,
				X_func,
				beta0(0),
				regression_weights.col(0),
				R_inv_ys_min_beta.col(0),
				kriging_params.col(0));



		for(int j=0; j<dim;j++) x(j) = dim*x(j)* (x_max(j) - x_min(j))+x_min(j);


		double func_val_exact = Rosenbrock8D(x.memptr());
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


#if 0

	/* for 1D problems */

	if(dim == 1){

		int resolution =10000;

		double bounds[2];
		bounds[0]=-2.0;
		bounds[1]=2.0;

		std::string kriging_response_surface_file_name = "1DTest_TRGEK_response_surface.dat";

		FILE *kriging_response_surface_file = fopen(kriging_response_surface_file_name.c_str(),"w");


		double dx,dy; /* step sizes in x and y directions */
		rowvec x(1);
		rowvec xb(1);
		rowvec xnorm(1);

		double out_sample_error=0.0;

		double max_value = -LARGE;
		double min_value =  LARGE;

		rowvec pmin(dim);
		rowvec pmax(dim);


		dx = (bounds[1]-bounds[0])/(resolution-1);

		x[0] = bounds[0];

		for(int i=0;i<resolution;i++){

			/* normalize x */
			xnorm(0)= (1.0/dim)*(x(0)- x_min(0)) / (x_max(0) - x_min(0));

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


			double func_val_exact = test_function1D(x.memptr());


			double sqr_error = (func_val_exact-func_val)*(func_val_exact-func_val);
			out_sample_error+= sqr_error;
			fprintf(kriging_response_surface_file,"%10.7f %10.7f\n",x(0),func_val);




			x[0]+= dx;
		}
		fclose(kriging_response_surface_file);

		out_sample_error = out_sample_error/(resolution);

		printf("out of sample error (pure Kriging) = %10.7f\n",out_sample_error);
		printf("sqrt( out of sample error) (pure Kriging) = %10.7f\n",sqrt(out_sample_error));
		printf("min = %10.7f\n",min_value);
		pmin.print();
		printf("max = %10.7f\n",max_value);
		pmax.print();
#if 1
		std::string file_name_for_plot = "1DTest_purekriging_response_surface_";
		file_name_for_plot += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";

		std::string title = "1Dtest";
		std::string python_command = "python -W ignore plot_1d_function.py "
				+ kriging_response_surface_file_name+ " "
				+ file_name_for_plot + " "+title;



		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");
#endif



	}





	if (dim == 2){
		int resolution =100;

		double bounds[4];
		bounds[0]=-2.0;
		bounds[1]=2.0;
		bounds[2]=-2.0;
		bounds[3]=2.0;

		std::string kriging_response_surface_file_name = "Herbie2D_TRGEK_response_surface.dat";

		FILE *kriging_response_surface_file = fopen(kriging_response_surface_file_name.c_str(),"w");


		double dx,dy; /* step sizes in x and y directions */
		rowvec x(2);
		rowvec xb(2);
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


				double func_val_exact = Rosenbrock(x.memptr());


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
		printf("sqrt( out of sample error) (pure Kriging) = %10.7f\n",sqrt(out_sample_error));
		/*
	printf("min = %10.7f\n",min_value);
	pmin.print();
	printf("max = %10.7f\n",max_value);
	pmax.print();
		 */
		double around_sample_error = 0.0;

		double eps_param1 = (bounds[1]-bounds[0])/100.0;
		double eps_param2 = (bounds[3]-bounds[2])/100.0;

		for(unsigned int i=0;i<X_func.n_rows;i++){

			for(int j=0; j<1000; j++){
				/* get a sample point */
				rowvec x = X_func.row(i);


				double pert1 = eps_param1*RandomDouble(-1.0,1.0);
				double pert2 = eps_param2*RandomDouble(-1.0,1.0);
#if 0


				printf("pert1 = %10.7f pert2= %10.7f\n",pert1,pert2 );

#endif
				pert1= (1.0/dim)*(pert1) / (x_max(0) - x_min(0));
				pert2= (1.0/dim)*(pert2) / (x_max(1) - x_min(1));

#if 0
				printf("\n");
				printf("x[%d] = ",i);
				x.print();
				printf("pert1 = %10.7f pert2= %10.7f\n",pert1,pert2 );
				printf("eps_param1 = %10.7f eps_param2= %10.7f\n",eps_param1,eps_param2);

#endif

				x(0) += pert1;
				x(1) += pert2;
#if 0

				printf("x[%d] = ",i);
				x.print();
#endif

				double func_val = calculate_f_tilde(x,
						X_func,
						beta0(0),
						regression_weights.col(0),
						R_inv_ys_min_beta.col(0),
						kriging_params.col(0));

				x(0) = dim*x(0)* (x_max(0) - x_min(0))+x_min(0);
				x(1) = dim*x(1)* (x_max(1) - x_min(1))+x_min(1);

				double func_val_exact = Rosenbrock(x.memptr());
				around_sample_error+= (func_val_exact-func_val)*(func_val_exact-func_val);

#if 0
				printf("\n");
				printf("x[%d] = ",i);
				x.print();
				printf("\n");
				printf("ftilde = %10.7f fexact= %10.7f\n",func_val,func_val_exact );
				printf("around sample error (perturbed = %20.15f\n", around_sample_error);

#endif



			}
		}

		around_sample_error = around_sample_error/(1000*X_func.n_rows);
		printf("around sample error (perturbed) = %20.15f\n", around_sample_error);
		printf("sqrt(around sample error (perturbed) = %20.15f\n", sqrt(around_sample_error));



#if 0
		std::string file_name_for_plot = "Eggholder_purekriging_response_surface_";
		file_name_for_plot += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";

		std::string title = "Kriging";
		std::string python_command = "python -W ignore plot_2d_surface.py "
				+ kriging_response_surface_file_name+ " "
				+ file_name_for_plot + " "+title;



		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");
#endif
	}
	else if(dim>2) {
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



		parameter_bounds[0]=-2.0;
		parameter_bounds[1]= 2.0;

		parameter_bounds[2]=-2.0;
		parameter_bounds[3]= 2.0;

		parameter_bounds[4]=-2.0;
		parameter_bounds[5]= 2.0;

		parameter_bounds[6]=-2.0;
		parameter_bounds[7]= 2.0;

		parameter_bounds[7]=-2.0;
		parameter_bounds[8]= 2.0;


		parameter_bounds[8]=-2.0;
		parameter_bounds[9]= 2.0;


		parameter_bounds[10]=-2.0;
		parameter_bounds[11]= 2.0;


		parameter_bounds[12]=-2.0;
		parameter_bounds[13]= 2.0;

		parameter_bounds[14]=-2.0;
		parameter_bounds[15]= 2.0;


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


			double func_val_exact = Rosenbrock8D(x.memptr());

			double sqr_error = (func_val_exact-func_val)*(func_val_exact-func_val);
			out_sample_error+= sqr_error;


#if 0
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




#if 0

	if (dim == 1){ /* visualize only linear model response surface for 1D problems */
		mat seeding_points = X_func;
		int resolution =10000;

		double bounds[2];
		bounds[0]=-2.0;
		bounds[1]=2.0;

		std::string kriging_response_surface_file_name = "1DTest_pure_linmodel_response_surface.dat";

		FILE *kriging_response_surface_file = fopen(kriging_response_surface_file_name.c_str(),"w");


		double dx; /* step sizes in x */
		rowvec x(1);
		rowvec xnorm(1);


		dx = (bounds[1]-bounds[0])/(resolution-1);


		double out_sample_error=0.0;

		double max_value = -LARGE;
		double min_value =  LARGE;
		rowvec pmin(dim);
		rowvec pmax(dim);


		x[0] = bounds[0];
		for(int i=0;i<resolution;i++){


			/* normalize x */
			xnorm(0)= (1.0/dim)*(x(0)- x_min(0)) / (x_max(0) - x_min(0));

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





			for(int j=0; j<dim;j++) {

				sp_not_normalized(j) = dim*sp(j)* (x_max(j) - x_min(j))+x_min(j);

			}

#if 0
			printf("x:\n");
			x.print();
			printf("closest point is:\n");
			sp.print();
			printf("in original coordinates:\n");
			sp_not_normalized.print();
			printf("original data entry:\n");
			data_gradients.row(indx).print();


#endif

			/* estimate the functional value at the nearest point from the Kriging estimator */

			double func_val = calculate_f_tilde(sp,
					X_func,
					beta0(0),
					regression_weights.col(0),
					R_inv_ys_min_beta.col(0),
					kriging_params.col(0));





			vec gradient(dim);


			if(n_f_evals == 0){

				for(int k=0;k<dim;k++){

					gradient(k) = data_gradients(indx,dim+1+k);

				}

#if 0
				printf("gradient:\n");
				trans(gradient).print();

#endif
			}



			else{

				for(int k=0;k<dim;k++){

					//		printf("i = %d\n",i);
					gradient(k) = calculate_f_tilde(sp,
							X_grad,
							beta0(k+1),
							regression_weights.col(k+1),
							R_inv_ys_min_beta.col(k+1),
							kriging_params.col(k+1));


				}

			}


			double ftilde_linmodel = func_val + dot((x-sp_not_normalized),gradient);
			fprintf(kriging_response_surface_file,"%10.7f %10.7f\n",x(0),ftilde_linmodel);

			double func_val_exact = test_function1D(x.memptr());

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


			x[0]+= dx;

		}
		fclose(kriging_response_surface_file);

		out_sample_error = out_sample_error/(resolution);


		printf("out of sample error (Taylor approximation) = %10.7f\n",out_sample_error);
		printf("min = %10.7f\n",min_value);
		pmin.print();
		printf("max = %10.7f\n",max_value);
		pmax.print();
#if 1
		std::string file_name_for_plot = "1DTest_linmodel_response_surface_";
		file_name_for_plot += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";

		std::string title = "Taylor_approximation";
		std::string python_command = "python -W ignore plot_1d_function.py "
				+ kriging_response_surface_file_name+ " "
				+ file_name_for_plot + " "+title;



		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");
#endif
	}





	if (dim == 2){ /* visualize only linear model response surface */
		mat seeding_points = X_func;
		int resolution =100;

		double bounds[4];
		bounds[0]=-2.0;
		bounds[1]=2.0;
		bounds[2]=-2.0;
		bounds[3]=2.0;

		std::string kriging_response_surface_file_name = "Herbie2D_pure_linmodel_response_surface.dat";

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

				for(int j=0; j<dim;j++) {

					sp_not_normalized(j) = dim*sp(j)* (x_max(j) - x_min(j))+x_min(j);
				}

#if 0
				printf("x:\n");
				x.print();
				printf("closest point is:\n");
				sp.print();
				printf("in original coordinates:\n");
				sp_not_normalized.print();
				printf("original data entry:\n");
				data_gradients.row(indx).print();


#endif

				/* estimate the functional value at the nearest point from the Kriging estimator */

				double func_val = calculate_f_tilde(sp,
						X_func,
						beta0(0),
						regression_weights.col(0),
						R_inv_ys_min_beta.col(0),
						kriging_params.col(0));


				vec gradient(dim);


				if(n_f_evals == 0){

					for(int k=0;k<dim;k++){

						gradient(k) = data_gradients(indx,dim+1+k);

					}

#if 0
					printf("gradient:\n");
					trans(gradient).print();

#endif
				}



				else{

					for(int k=0;k<dim;k++){

						//		printf("i = %d\n",i);
						gradient(k) = calculate_f_tilde(sp,
								X_grad,
								beta0(k+1),
								regression_weights.col(k+1),
								R_inv_ys_min_beta.col(k+1),
								kriging_params.col(k+1));


					}

				}


				double ftilde_linmodel = func_val + dot((x-sp_not_normalized),gradient);
				fprintf(kriging_response_surface_file,"%10.7f %10.7f %10.7f\n",x(0),x(1),ftilde_linmodel);

				double func_val_exact = Herbie2D(x.memptr());

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
#if 0
		std::string file_name_for_plot = "Eggholder_linmodel_response_surface_";
		file_name_for_plot += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";

		std::string title = "Taylor_approximation";
		std::string python_command = "python -W ignore plot_2d_surface.py "
				+ kriging_response_surface_file_name+ " "
				+ file_name_for_plot + " "+title;



		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");
#endif
	}

	else if(dim>2){ /*for higher dimensions */


		int number_of_samples = 50000;

		rowvec x(dim);
		rowvec xb(dim);
		rowvec xnorm(dim);

		double out_sample_error=0.0;

		double max_value = -LARGE;
		double min_value =  LARGE;

		rowvec pmin(dim);
		rowvec pmax(dim);


		double parameter_bounds[16];

		parameter_bounds[0]=-2.0;
		parameter_bounds[1]= 2.0;

		parameter_bounds[2]=-2.0;
		parameter_bounds[3]= 2.0;

		parameter_bounds[4]=-2.0;
		parameter_bounds[5]= 2.0;

		parameter_bounds[6]=-2.0;
		parameter_bounds[7]= 2.0;

		parameter_bounds[8]=-2.0;
		parameter_bounds[9]= 2.0;

		parameter_bounds[10]=-2.0;
		parameter_bounds[11]=2.0;

		parameter_bounds[12]=-2.0;
		parameter_bounds[13]=2.0;

		parameter_bounds[14]=-2.0;
		parameter_bounds[15]=2.0;


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

			for(int j=0; j<dim;j++) {

				sp_not_normalized(j) = dim*sp(j)* (x_max(j) - x_min(j))+x_min(j);
			}

			double ftilde_linmodel = func_val + dot((x-sp_not_normalized),gradient);

			double func_val_exact = Rosenbrock8D(x.memptr());

			double srq_error = (ftilde_linmodel-func_val_exact)*(ftilde_linmodel-func_val_exact);
			out_sample_error+=srq_error;

#if 0

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



#endif

			if(ftilde_linmodel< min_value){

				min_value = ftilde_linmodel;
				pmin = x;

			}
			if(ftilde_linmodel > max_value){

				max_value = ftilde_linmodel;
				pmax = x;

			}





		}

		out_sample_error = out_sample_error/(number_of_samples);


		printf("out of sample error (Taylor approximation) = %10.7f\n",out_sample_error);
		printf("min = %10.7f\n",min_value);
		pmin.print();
		printf("max = %10.7f\n",max_value);
		pmax.print();


	}
#endif


	/* train the hyperparameter of the hybrid model */

	if( train_hyper_param == 1){

		train_optimal_radius(
				radius,
				data_functional_values,
				data_gradients,
				linear_regression,
				regression_weights,
				kriging_params,
				200);

	}




#if 0

	if (dim == 1){

		int resolution =10000;

		double bounds[2];
		bounds[0]=-2.0;
		bounds[1]=2.0;


		std::string kriging_response_surface_file_name = "1DTest_hybrid_response_surface.dat";

		FILE *kriging_response_surface_file = fopen(kriging_response_surface_file_name.c_str(),"w");


		double dx; /* step sizes in x  */
		rowvec x(1);
		rowvec xnorm(1);


		dx = (bounds[1]-bounds[0])/(resolution-1);


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

			/* normalize x */
			xnorm(0)= (1.0/dim)*(x(0)- x_min(0)) / (x_max(0) - x_min(0));




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
			double distance = L1norm(xdiff, dim);





			double fval_kriging = 0.0;


			/* pure kriging value at x = xnorm*/
			fval_kriging = calculate_f_tilde(xnorm,
					X_func,
					beta0(0),
					regression_weights.col(0),
					R_inv_ys_min_beta.col(0),
					kriging_params.col(0));


			vec grad(dim);


			if(n_f_evals == 0){

#if 0
				printf("data_gradients at row = %d\n",indx);
				data_gradients.row(indx).print();
#endif

				for(int k=0;k<dim;k++){

					grad(k) = data_gradients(indx,dim+1+k);

				}
			}
			else{

				for(int k=0;k<dim;k++){

					grad(k) = calculate_f_tilde(sp,
							X_grad,
							beta0(k+1),
							regression_weights.col(k+1),
							R_inv_ys_min_beta.col(k+1),
							kriging_params.col(k+1));


				}

			}

			double normgrad= L1norm(grad, dim);

			double factor = exp(-radius*distance*normgrad);

			/* calculate Taylor approximation at x = xnorm */

			double fval_linmodel= ysfunc(indx) + dot((x-sp_not_normalized),grad);

			double fval = factor*fval_linmodel + (1.0-factor)*fval_kriging;


			fprintf(kriging_response_surface_file,"%10.7f %10.7f\n",x(0),fval);

			double func_val_exact = test_function1D(x.memptr());

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



			x[0]+= dx;

		}
		fclose(kriging_response_surface_file);

		out_sample_error = out_sample_error/(resolution);

		printf("out of sample error (hybrid model) = %10.7f\n",out_sample_error);
		printf("min = %10.7f\n",min_value);
		pmin.print();
		printf("max = %10.7f\n",max_value);
		pmax.print();
		printf("min (exact) = %10.7f\n",min_exactvalue);
		pminex.print();
		printf("max (exact) = %10.7f\n",max_exactvalue);
		pmaxex.print();
#if 1
		std::string file_name_for_plot = "1DTest_hybrid_response_surface_";
		file_name_for_plot += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";

		std::string title = "Hybrid_model";
		std::string python_command = "python -W ignore plot_1d_function.py "
				+ kriging_response_surface_file_name+ " "
				+ file_name_for_plot + title;



		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");
#endif




	}


	if (dim == 2){

		int resolution =100;

		double bounds[4];
		bounds[0]=-2.0;
		bounds[1]=2.0;
		bounds[2]=-2.0;
		bounds[3]=2.0;

		std::string kriging_response_surface_file_name = "Herbie2D_hybrid_response_surface.dat";

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
				double distance = L1norm(xdiff, dim);



				//				if(ridge_flag) factor = factor*2;

				double fval_kriging = 0.0;


				/* pure kriging value at x = xnorm*/
				fval_kriging = calculate_f_tilde(xnorm,
						X_func,
						beta0(0),
						regression_weights.col(0),
						R_inv_ys_min_beta.col(0),
						kriging_params.col(0));


				vec grad(dim);


				if(n_f_evals == 0){

#if 0
					printf("data_gradients at row = %d\n",indx);
					data_gradients.row(indx).print();
#endif

					for(int k=0;k<dim;k++){

						grad(k) = data_gradients(indx,dim+1+k);

					}
				}
				else{

					for(int k=0;k<dim;k++){

						grad(k) = calculate_f_tilde(sp,
								X_grad,
								beta0(k+1),
								regression_weights.col(k+1),
								R_inv_ys_min_beta.col(k+1),
								kriging_params.col(k+1));


					}

				}

				double normgrad= L1norm(grad, dim);

				double factor = exp(-radius*distance*normgrad);

				/* calculate Taylor approximation at x = xnorm */

				double fval_linmodel= ysfunc(indx) + dot((x-sp_not_normalized),grad);

				double fval = factor*fval_linmodel + (1.0-factor)*fval_kriging;


				fprintf(kriging_response_surface_file,"%10.7f %10.7f %10.7f\n",x(0),x(1),fval);

				double func_val_exact = Rosenbrock(x.memptr());

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
		printf("sqrt( out of sample error (hybrid model)) = %10.7f\n",sqrt(out_sample_error));
		/*
	printf("min = %10.7f\n",min_value);
	pmin.print();
	printf("max = %10.7f\n",max_value);
	pmax.print();
	printf("min (exact) = %10.7f\n",min_exactvalue);
	pminex.print();
	printf("max (exact) = %10.7f\n",max_exactvalue);
	pmaxex.print();

		 */

		double around_sample_error = 0.0;
		double eps_param1 = (bounds[1]-bounds[0])/100.0;
		double eps_param2 = (bounds[3]-bounds[2])/100.0;

		for(unsigned int i=0;i<X_func.n_rows;i++){

			for(int j=0; j<1000; j++){
				/* get a sample point */
				rowvec x = X_func.row(i);


				double pert1 = eps_param1*RandomDouble(-1.0,1.0);
				double pert2 = eps_param2*RandomDouble(-1.0,1.0);
#if 0


				printf("pert1 = %10.7f pert2= %10.7f\n",pert1,pert2 );

#endif
				pert1= (1.0/dim)*(pert1) / (x_max(0) - x_min(0));
				pert2= (1.0/dim)*(pert2) / (x_max(1) - x_min(1));

#if 0
				printf("\n");
				printf("x[%d] = ",i);
				x.print();
				printf("pert1 = %10.7f pert2= %10.7f\n",pert1,pert2 );
				printf("eps_param1 = %10.7f eps_param2= %10.7f\n",eps_param1,eps_param2);

#endif

				x(0) += pert1;
				x(1) += pert2;
#if 0

				printf("x[%d] = ",i);
				x.print();
#endif


				double min_dist=0.0;
				int indx;

				/* find the closest point */
				findKNeighbours(X_grad,
						x,
						1,
						&min_dist,
						&indx,
						dim);

				rowvec sp =  X_func.row(indx);
				rowvec sp_not_normalized(dim);
				sp_not_normalized.fill(0.0);



				sp_not_normalized(0) = dim*sp(0)* (x_max(0) - x_min(0))+x_min(0);
				sp_not_normalized(1) = dim*sp(1)* (x_max(1) - x_min(1))+x_min(1);


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

				rowvec xdiff = x-sp;
				double distance = L1norm(xdiff, dim);


				double fval_kriging = 0.0;


				/* pure kriging value at x = xnorm*/
				fval_kriging = calculate_f_tilde(x,
						X_func,
						beta0(0),
						regression_weights.col(0),
						R_inv_ys_min_beta.col(0),
						kriging_params.col(0));


				/* get the original coordinates again */
				x(0) = dim*x(0)* (x_max(0) - x_min(0))+x_min(0);
				x(1) = dim*x(1)* (x_max(1) - x_min(1))+x_min(1);


				vec grad(dim);

				for(int k=0;k<dim;k++){

					grad(k) = data_gradients(indx,dim+1+k);

				}

				double normgrad= L1norm(grad, dim);

				double factor = exp(-radius*distance*normgrad);

				/* calculate Taylor approximation at x = xnorm */

				double fval_linmodel= ysfunc(indx) + dot((x-sp_not_normalized),grad);

				double fval = factor*fval_linmodel + (1.0-factor)*fval_kriging;



				double func_val_exact = Rosenbrock(x.memptr());
				around_sample_error+= (func_val_exact-fval)*(func_val_exact-fval);

#if 0
				printf("\n");
				printf("x[%d] = ",i);
				x.print();
				printf("\n");
				printf("ftilde = %10.7f fexact= %10.7f\n",func_val,func_val_exact );
				printf("around sample error (perturbed = %20.15f\n", around_sample_error);

#endif



			}
		}

		around_sample_error = around_sample_error/(1000*X_func.n_rows);

		printf("around sample error, perturbed) = %20.15f\n", around_sample_error);
		printf("sqrt(around sample error, perturbed = %20.15f\n", sqrt(around_sample_error));



#if 0
		std::string file_name_for_plot = "Eggholder_hybrid_response_surface_";
		file_name_for_plot += "_"+std::to_string(resolution)+ "_"+std::to_string(resolution)+".png";

		std::string title = "Hybrid_model";
		std::string python_command = "python -W ignore plot_2d_surface.py "
				+ kriging_response_surface_file_name+ " "
				+ file_name_for_plot + title;



		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");
#endif




	}
	else if(dim>2){

		int number_of_samples = 50000;

		rowvec x(dim);
		rowvec xb(dim);
		rowvec xnorm(dim);

		double out_sample_error=0.0;

		double max_value = -LARGE;
		double min_value =  LARGE;

		double max_exactvalue = -LARGE;
		double min_exactvalue =  LARGE;

		rowvec pmin(dim);
		rowvec pmax(dim);
		rowvec pminex(dim);
		rowvec pmaxex(dim);


		double parameter_bounds[16];



		parameter_bounds[0]=-2.0;
		parameter_bounds[1]= 2.0;

		parameter_bounds[2]=-2.0;
		parameter_bounds[3]= 2.0;

		parameter_bounds[4]=-2.0;
		parameter_bounds[5]= 2.0;

		parameter_bounds[6]=-2.0;
		parameter_bounds[7]= 2.0;

		parameter_bounds[8]=-2.0;
		parameter_bounds[9]= 2.0;

		parameter_bounds[10]=-2.0;
		parameter_bounds[11]=2.0;

		parameter_bounds[12]=-2.0;
		parameter_bounds[13]=2.0;

		parameter_bounds[14]=-2.0;
		parameter_bounds[15]=2.0;


		for(int i=0; i<number_of_samples; i++){

			/* generate a random input vector and normalize it */

			for(int j=0; j<dim;j++){

				x(j) = RandomDouble(parameter_bounds[j*2], parameter_bounds[j*2+1]);
				xnorm(j)= (1.0/dim)*(x(j)- x_min(j)) / (x_max(j) - x_min(j));

			}


			double fval_kriging = 0.0;


			/* pure kriging value at x = xnorm*/
			fval_kriging = calculate_f_tilde(xnorm,
					X_func,
					beta0(0),
					regression_weights.col(0),
					R_inv_ys_min_beta.col(0),
					kriging_params.col(0));

			double min_dist=0;
			int indx = -1;

			/* find the closest seeding point */
			findKNeighbours(X_grad,
					xnorm,
					1,
					&min_dist,
					&indx,
					1);



			rowvec sp =  X_grad.row(indx);
			rowvec sp_not_normalized(dim);
			sp_not_normalized.fill(0.0);

			for(int j=0; j<dim;j++) {

				sp_not_normalized(j) = dim*sp(j)* (x_max(j) - x_min(j))+x_min(j);
			}

			rowvec xdiff = xnorm-sp;
			double distance = L1norm(xdiff, dim);

			/* get the functional value from the data */
			double func_val = data_gradients(indx,dim);


			vec grad(dim);

			for(int j=0; j<dim; j++) {

				grad(j)= data_gradients(indx,j+dim+1);
			}


			double normgrad= L1norm(grad, dim);

			double factor = exp(-radius*distance*normgrad);

			double fval_linmodel= func_val + dot((x-sp_not_normalized),grad);

			double fval = factor*fval_linmodel + (1.0-factor)*fval_kriging;

			double func_val_exact = Rosenbrock8D(x.memptr());

			double srq_error = (fval-func_val_exact)*(fval-func_val_exact);
			out_sample_error+=srq_error;

#if 0

			printf("x:\n");
			x.print();
			printf("xnorm:\n");
			xnorm.print();
			printf("closest neighbour:\n");
			X_grad.row(indx).print();
			data_gradients.row(indx).print();
			trans(grad).print();
			printf("factor = %10.7f\n", factor);
			printf("fval_kriging = %10.7f\n", fval_kriging);
			printf("ftilde_linmodel = %10.7f\n", fval_linmodel);
			printf("fval = %10.7f\n", fval);
			printf("func_val_exact = %10.7f\n", func_val_exact);



#endif

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


		} /* end of for */

		out_sample_error = out_sample_error/(number_of_samples);


		printf("out of sample error (hybrid model) = %10.7f\n",out_sample_error);
		printf("min = %10.7f\n",min_value);
		pmin.print();
		printf("max = %10.7f\n",max_value);
		pmax.print();


	} /* end of else */

#endif


	return 0;

}

int train_aggregation_model(AggregationModel &model_settings) {

	printf("Training aggregation model for the data: %s...\n",model_settings.input_filename.c_str());


	if (model_settings.visualizeKrigingValidation == "yes" || model_settings.visualizeKernelRegressionValidation == "yes" ){

		if(model_settings.validationset_input_filename == "None") {

			printf("File name for validation is not specified!\n");
			exit(-1);

		}

	}


	/* first generate temp data for Kriging training */

	mat inputData;



	bool load_ok = inputData.load(model_settings.input_filename.c_str());

	if(load_ok == false)
	{
		printf("problem with loading the file %s\n",model_settings.input_filename.c_str());
		exit(-1);
	}


	fmat inputdataSinglePrecision;
	load_ok = inputdataSinglePrecision.load(model_settings.input_filename.c_str());

	if(load_ok == false)
	{
		printf("problem with loading the file %s\n",model_settings.input_filename.c_str());
		exit(-1);
	}


	int N = inputData.n_rows;
	int Ncols = inputData.n_cols;
	int d = model_settings.dim;

	mat inputDataKriging = inputData.submat(0,0,N-1,d);

	/* check dimensions of the data */

	if(Ncols != 2*d +1 ) {

		printf("Input data dimension does not match!\n");
		exit(-1);

	}

#if 0
	printf("Input data:\n");
	inputData.print();
#endif


	mat X = inputData.submat(0,0,N-1,d-1);

	/* find minimum and maximum of the columns of X */

	vec x_max(d);
	x_max.fill(0.0);

	vec x_min(d);
	x_min.fill(0.0);



	for (int i = 0; i < d; i++) {
		x_max(i) = X.col(i).max();
		x_min(i) = X.col(i).min();

	}

#if 0
	printf("maximum = \n");
	x_max.print();

	printf("minimum = \n");
	x_min.print();
#endif



	/* normalize data X */

	for (int i = 0; i < N; i++) {

		for (int j = 0; j < d; j++) {

			X(i, j) = (1.0/d)*(X(i, j) - x_min(j)) / (x_max(j) - x_min(j));

		}
	}




	mat validationData;
	mat Xvalidation;
	mat XvalidationNotNormalized;
	int Nval = 0;

	if(model_settings.validationset_input_filename != "None"){


#if 1
		printf("Reading validation set %s...\n",model_settings.validationset_input_filename.c_str());
#endif

		load_ok = validationData.load(model_settings.validationset_input_filename.c_str());

		if(load_ok == false)
		{
			printf("problem with loading the file %s\n",model_settings.validationset_input_filename.c_str());
			exit(-1);
		}


#if 0
		printf("Validation data =\n");
		validationData.print();
#endif


		Nval = validationData.n_rows;
		Xvalidation = validationData.submat(0,0,Nval-1,d-1);
		XvalidationNotNormalized = Xvalidation;


		/* normalize data X for validation*/

		for (int i = 0; i < Nval; i++) {

			for (int j = 0; j < d; j++) {

				Xvalidation(i, j) = (1.0/d)*(Xvalidation(i, j) - x_min(j)) / (x_max(j) - x_min(j));
			}
		}

#if 0
		printf("X Validation (normalized) =\n");
		Xvalidation.print();
#endif
	}



#if 0
	printf("Kriging input data:\n");
	inputDataKriging.print();
	printf("X:\n");
	X.print();

#endif

	printf("Training the Kriging parameters...\n");

	train_kriging_response_surface(inputDataKriging,
			model_settings.kriging_hyperparameters_filename ,
			LINEAR_REGRESSION_ON,
			model_settings.regression_weights,
			model_settings.kriging_weights,
			model_settings.epsilon_kriging,
			model_settings.max_number_of_kriging_iterations);

	printf("Training of the Kriging parameters done...\n");


	/*update y vectors according to linear regression */

	mat augmented_X(N, d + 1);

	for (int i = 0; i < N; i++) {

		for (int j = 0; j <= d; j++) {

			if (j == 0){

				augmented_X(i, j) = 1.0;
			}
			else{

				augmented_X(i, j) = X(i, j - 1);

			}
		}
	}


#if 1
	printf("Regression weights:\n");
	model_settings.regression_weights.t().print();
	printf("Kriging weights:\n");
	model_settings.kriging_weights.t().print();

#endif

	vec ys = inputData.col(d);

	compute_R_inv_ys_min_beta(X,
			ys,
			model_settings.kriging_weights,
			model_settings.regression_weights,
			model_settings.R_inv_ys_min_beta,
			model_settings.beta0,
			model_settings.epsilon_kriging,
			LINEAR_REGRESSION_ON);

	mat visualizeKriging;

	if(model_settings.visualizeKrigingValidation == "yes"){


		double genErrorKriging = 0.0;

		visualizeKriging = zeros(Nval,2);

		for(int i=0; i<Nval; i++){

			rowvec xp = Xvalidation.row(i);
#if 0
			printf("xp =\n");
			xp.print();
#endif
			double ftildeKriging = calculate_f_tilde(xp,
					X,
					model_settings.beta0,
					model_settings.regression_weights,
					model_settings.R_inv_ys_min_beta,
					model_settings.kriging_weights);

			double fexact = validationData(i,d);

			genErrorKriging += (fexact-ftildeKriging)*(fexact-ftildeKriging);
#if 0
			printf("ftilde (Kriging) = %10.7f, fexact = %10.7f\n",ftildeKriging,fexact);
#endif
			if(model_settings.visualizeKrigingValidation == "yes"){
				visualizeKriging(i,0) = fexact;
				visualizeKriging(i,1) = ftildeKriging;
			}

		}

		genErrorKriging = genErrorKriging/Nval;

		printf("Generalization Error for Kriging (MSE) = %10.7f\n",genErrorKriging);

		visualizeKriging.save("visualizeKriging.dat",raw_ascii);

		std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_1d_function_scatter.py visualizeKriging.dat visualizeKriging.png" ;


		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");

	}





	/* parameters for the kernel regression training */
	float sigma = 0.01;
	float wSvd = 0.0;
	float w12 = 0.01;


	/* lower diagonal matrix for the kernel regression part */
	fmat LKernelRegression(model_settings.dim,model_settings.dim);
	mat L= zeros(d,d);

	fmat inputDataKernelRegression = inputdataSinglePrecision.submat(0,0,N-1,2*d);

#if 0
	inputDataKernelRegression.print();
#endif

	/* normalize functional values */

	float yTrainingMax = 1.0;



	mat XTrainingNotNormalized = inputData.submat(0,0,N-1,d-1);
	vec yTrainingNotNormalized = inputData.col(d);
	mat gradTrainingNotNormalized = inputData.submat(0,d+1,N-1,2*d);

	yTrainingMax = inputDataKernelRegression.col(d).max();

	for (int i = 0; i < N; i++) {

		inputDataKernelRegression(i, d) = inputDataKernelRegression(i, d)/yTrainingMax ;

	}


	/* normalize training data */
	for (int i = 0; i < N; i++) {

		for (int j = 0; j < d; j++) {

			inputDataKernelRegression(i, j) = (1.0/d)*(inputDataKernelRegression(i, j) - x_min(j)) / (x_max(j) - x_min(j));
		}


	}


	if(model_settings.number_of_cv_iterations > 0){

		printf("Training the Mahalanobis matrix and sigma...\n");
		/* now train the Mahalanobis matrix */
		trainMahalanobisDistance(LKernelRegression, inputDataKernelRegression, sigma, wSvd, w12, model_settings.number_of_cv_iterations,L2_LOSS_FUNCTION);

		fmat Mtemp = LKernelRegression*trans(LKernelRegression);

		for(int i=0; i<d; i++)
			for(int j=0; j<d; j++) {

				model_settings.M(i,j) = Mtemp(i,j);
			}


		model_settings.sigma = sigma;



	}
	else{
		printf("Reading model settings...\n");
		model_settings.load_state();

	}





	mat visualizeKernelRegression;

	if(model_settings.visualizeKernelRegressionValidation == "yes"){

		visualizeKernelRegression = zeros(Nval,2);


		double genErrorKernelReg = 0.0;

		for(int i=0; i<Nval; i++){

			rowvec xpNotNormalized = XvalidationNotNormalized.row(i);
#if 0
			printf("xp =\n");
			xp.print();
#endif


			double fKernelRegression = 0.0;


			fKernelRegression = kernelRegressorNotNormalized(X,
					XTrainingNotNormalized,
					yTrainingNotNormalized,
					gradTrainingNotNormalized,
					xpNotNormalized,
					x_min,
					x_max,
					model_settings.M,
					model_settings.sigma);



			double fexact = validationData(i,d);

			genErrorKernelReg += (fKernelRegression-fexact)*(fKernelRegression-fexact);
#if 1
			printf("ftilde (Kernel Regression) = %10.7f, fexact = %10.7f\n",fKernelRegression,fexact);
#endif
			if(model_settings.visualizeKernelRegressionValidation == "yes"){
				visualizeKernelRegression(i,0) = fexact;
				visualizeKernelRegression(i,1) = fKernelRegression;
			}



		}

		genErrorKernelReg = genErrorKernelReg/Nval;

		printf("Generalization Error for Kernel Regregression (MSE) = %10.7f\n",genErrorKernelReg);


		visualizeKernelRegression.save("visualizeKernelRegression.dat",raw_ascii);

		std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_1d_function_scatter.py visualizeKernelRegression.dat visualizeKernelRegression.png" ;


		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");




	} /* end of validation part */




	/* training of the rho parameter */

	/* obtain sample statistics */

	vec probe_distances_sample(N);

	for(int i=0; i<N; i++){

		rowvec np = X.row(i);


		double min_dist[2]={0.0,0.0};
		int indx[2]={-1,-1};

		/* find the closest two points to the np in the data set */
		findKNeighbours(X,
				np,
				2,
				min_dist,
				indx,
				1);


#if 0
		printf("point:\n");
		np.print();
		printf("nearest neighbour is:\n");
		X.row(indx[0]).print();
		printf("minimum distance (L1 norm)= %10.7f\n\n",min_dist[0]);
		printf("second nearest neighbour is:\n");
		X.row(indx[1]).print();
		printf("minimum distance (L1 norm)= %10.7f\n\n",min_dist[1]);
#endif

		probe_distances_sample(i)=min_dist[1];


	}

	double average_distance_sample = mean(probe_distances_sample);



	/* obtain gradient statistics */
	double sumnormgrad = 0.0;
	for(unsigned int i=0; i<N; i++){

		rowvec grad= gradTrainingNotNormalized.row(i);


		double normgrad= L1norm(grad, d);
#if 0
		printf("%dth entry at the data gradients:\n",i);
		grad.print();
		printf("norm of the gradient = %10.7e\n",normgrad);
#endif
		sumnormgrad+= normgrad;


	}

	double avg_norm_grad= sumnormgrad/N;
#if 1
	printf("average norm grad = %10.7e\n", avg_norm_grad);
#endif
	/* define the search range for the hyperparameter r */


	double min_r = 0.0;
	//	max_r = 1.0/dim;
	double max_r = -2*log(0.00001)/ (max(probe_distances_sample) * avg_norm_grad);
	//	max_r = 0.01;

	double dr;



	dr = (max_r - min_r)/(model_settings.number_of_cv_iterations_rho);

#if 0
	printf("average distance in the training data= %10.7f\n",average_distance_sample);
	printf("min (sample)       = %10.7f\n",min(probe_distances_sample));
	printf("max (sample)       = %10.7f\n",max(probe_distances_sample));
	printf("max_r = %10.7f\n",max_r);
	printf("dr = %10.7f\n",dr);
	printf("factor at maximum distance at max r = %15.10f\n",exp(-max_r*max(probe_distances_sample)* avg_norm_grad));
#endif





	/* number of cross validation points */
	int NCVVal = N*0.2;
	int NCVTra = N - NCVVal;

#if 0
	printf("number of validation points (rho training) = %d\n", NCVVal);
	printf("number of rest points in the training data (rho training) = %d\n", NCVTra);
#endif



	mat inputDataCVval = inputData.submat(0,0,NCVVal-1,2*d);
	mat inputDataCVtra = inputData.submat(NCVVal,0,N-1,2*d);
	mat XCVval = X.submat(0,0,NCVVal-1,d-1);
	mat XCVtra = X.submat(NCVVal,0,N-1,d-1);

	mat XCVvalNotNormalized = XTrainingNotNormalized.submat(0,0,NCVVal-1,d-1);
	mat XCVtraNotNormalized = XTrainingNotNormalized.submat(NCVVal,0,N-1,d-1);
	vec yvalidation = inputDataCVval.col(d);
	vec ytraining   = inputDataCVtra.col(d);


	mat gradCVTraNotNormalized = inputDataCVtra.submat(0,d+1,NCVTra-1,2*d);
#if 0
	printf("inputDataCVval:\n");
	inputDataCVval.print();
	printf("XCVval:\n");
	XCVval.print();
	printf("XCVtra:\n");
	XCVtra.print();
	printf("XCVvalNotNormalized:\n");
	XCVvalNotNormalized.print();
#endif


	double minGenErrorCVLoop = LARGE;
	double optimal_rho = -1.0;


	compute_R_inv_ys_min_beta(XCVtra,
			ytraining,
			model_settings.kriging_weights,
			model_settings.regression_weights,
			model_settings.R_inv_ys_min_beta,
			model_settings.beta0,
			model_settings.epsilon_kriging,
			LINEAR_REGRESSION_ON);


	for(int rho_iter=0; rho_iter< model_settings.number_of_cv_iterations_rho; rho_iter++){


		/* value of the hyper parameter to be tried */
		double rho_trial = min_r+(rho_iter)*dr;

		model_settings.rho = rho_trial;


		double genError = calcGenErrorAggModel(model_settings,
				XCVval,
				XCVvalNotNormalized,
				yvalidation,
				XCVtra,
				XCVtraNotNormalized,
				ytraining,
				gradCVTraNotNormalized,
				x_min,
				x_max);

#if 0
		printf("CV iteration = %d, rho = %10.7f, Gen. Error = %10.7f\n",rho_iter,rho_trial,genError);
#endif

		if(genError < minGenErrorCVLoop){

			minGenErrorCVLoop = genError;
			optimal_rho = rho_trial;

		}

	}

#if 0
	printf("Optimal value of rho = %10.7f, Gen. Error = %10.7f\n",optimal_rho,minGenErrorCVLoop);
#endif

	model_settings.rho = optimal_rho;


	compute_R_inv_ys_min_beta(X,
			ys,
			model_settings.kriging_weights,
			model_settings.regression_weights,
			model_settings.R_inv_ys_min_beta,
			model_settings.beta0,
			model_settings.epsilon_kriging,
			LINEAR_REGRESSION_ON);




	mat visualizeAggModel;

	if(model_settings.visualizeAggModelValidation == "yes"){

		visualizeAggModel = zeros(Nval,2);


		double genErrorAggModel = 0.0;

		for(int i=0; i<Nval; i++){

			rowvec xpNotNormalized = XvalidationNotNormalized.row(i);
			rowvec xp = Xvalidation.row(i);
#if 0
			printf("xp =\n");
			xp.print();
#endif

			double fAggModel = ftildeAggModel(model_settings,
					xp,
					xpNotNormalized,
					X,
					XTrainingNotNormalized,
					yTrainingNotNormalized,
					gradTrainingNotNormalized,
					x_min,
					x_max);

			double fexact = validationData(i,d);

			genErrorAggModel += (fAggModel-fexact)*(fAggModel-fexact);

#if 1
			printf("ftilde (Agg. Model) = %10.7f, fexact = %10.7f\n",fAggModel,fexact);
#endif


			visualizeAggModel(i,0) = fexact;
			visualizeAggModel(i,1) = fAggModel;



		}


		genErrorAggModel = genErrorAggModel/Nval;

		printf("Generalization Error for the Agg. Model (MSE) = %10.7f\n",genErrorAggModel);


		visualizeAggModel.save("visualizeAggModel.dat",raw_ascii);

		std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_1d_function_scatter.py visualizeAggModel.dat visualizeAggModel.png" ;


		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");





	}


	model_settings.save_state();



	return 0;
}



