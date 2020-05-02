#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include "kriging_training.hpp"
#include "linear_regression.hpp"
#include "auxilliary_functions.hpp"
#include "Rodeo_macros.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;

/* global variables */

int total_number_of_function_evals;

double population_overall_max = -10E14;
int population_overall_max_tread_id = -1;




KrigingModel::KrigingModel(std::string name, int dimension){


	label = name;
	printf("Initializing settings for training %s data...\n",name.c_str());
	input_filename = name +".csv";
	kriging_hyperparameters_filename = name + "_Kriging_Hyperparameters.csv";
	regression_weights.zeros(dimension);
	kriging_weights.zeros(2*dimension);
	epsilon_kriging = 10E-06;
	max_number_of_kriging_iterations = 10000;
	dim = dimension;
	linear_regression = true;

	bool status = data.load(input_filename.c_str(), csv_ascii);
	if(status == true)
	{
		printf("Data input is done\n");
	}
	else
	{
		printf("Problem with data the input (cvs ascii format)\n");
		exit(-1);
	}


	N = data.n_rows;

	I = ones(N);

	X = data.submat(0, 0, N - 1, dim - 1);

	XnotNormalized = X;


	xmax = zeros(dim);
	xmin = zeros(dim);

	for (unsigned int i = 0; i < dim; i++) {

		xmax(i) = data.col(i).max();
		xmin(i) = data.col(i).min();

	}

	/* normalize data matrix */

	for (unsigned int i = 0; i < N; i++) {

		for (unsigned int j = 0; j < dim; j++) {

			X(i, j) = (1.0/dim)*(X(i, j) - xmin(j)) / (xmax(j) - xmin(j));
		}
	}

	sigma_sqr = 0.0;
	beta0 = 0.0;

	R.zeros(N,N);
	L.zeros(N,N);
	U.zeros(N,N);
	R_inv_ys_min_beta.zeros(N);
	R_inv_I.zeros(N);
	I.ones(N);

	ymin = min(data.col(dim));
	ymax = max(data.col(dim));
	yave = mean(data.col(dim));


}

void KrigingModel::print(void){
	printf("\n");
	printf("Kriging model: %s\n",label.c_str());
	printf("kriging_hyperparameters_filename: %s\n",kriging_hyperparameters_filename.c_str());
	printf("input_filename: %s\n",input_filename.c_str());
	printf("max_number_of_kriging_iterations = %d\n",max_number_of_kriging_iterations);
	printf("epsilon_kriging = %15.10f\n",epsilon_kriging);
	printf("\n");



}
void KrigingModel::updateModelParams(void){

	vec ys = data.col(dim);



	/* train linear regression */
	if (linear_regression) { // if linear regression is on


		train_linear_regression(X, ys, regression_weights, 10E-6);


		mat augmented_X(N, dim + 1);

		for (unsigned int i = 0; i < N; i++) {

			for (unsigned int j = 0; j <= dim; j++) {

				if (j == 0){

					augmented_X(i, j) = 1.0;
				}
				else {

					augmented_X(i, j) = X(i, j - 1);

				}
			}
		}

		/* now update the ys vector */

		vec ys_reg = augmented_X * regression_weights;

#if 0
		for(unsigned int i=0; i<ys.size(); i++){

			printf("%10.7f %10.7f\n",ys(i),ys_reg(i));
		}
#endif
		ys = ys - ys_reg;

#if 0
		printf("Updated ys vector = \n");
#endif

		for(unsigned int i=0; i<regression_weights.size();i++ ){

			if(fabs(regression_weights(i)) > 10E5){

				printf("WARNING: Linear regression coefficients are too large! \n");
				printf("regression_weights(%d) = %10.7f\n",i,regression_weights(i));
			}

		}



	} /* end of linear regression */

	vec theta = kriging_weights.head(dim);
	vec gamma = kriging_weights.tail(dim);


	compute_R_matrix(theta,
			gamma,
			epsilon_kriging,
			R ,
			X);


	/* Cholesky decomposition R = LDL^T */


	int cholesky_return = chol(U, R);

	if (cholesky_return == 0) {
		printf("Error: Ill conditioned correlation matrix, Cholesky decomposition failed at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);
	}

	L = trans(U);

	vec R_inv_ys(N);


	solve_linear_system_by_Cholesky(U, L, R_inv_ys, ys);    /* solve R x = ys */
#if 0
	printf("R_inv_ys =\n");
	trans(R_inv_ys).print();
#endif

	R_inv_I = zeros(N);


	solve_linear_system_by_Cholesky(U, L, R_inv_I, I);      /* solve R x = I */
#if 0
	printf("R_inv_I =\n");
	trans(R_inv_I).print();
#endif

	beta0 = (1.0/dot(I,R_inv_I)) * (dot(I,R_inv_ys));

	vec ys_min_betaI = ys - beta0*I;

	/* solve R x = ys-beta0*I */
	solve_linear_system_by_Cholesky(U, L, R_inv_ys_min_beta , ys_min_betaI);


	sigma_sqr = (1.0 / N) * dot(ys_min_betaI, R_inv_ys_min_beta);


}

void KrigingModel::updateWithNewData(void){

	R.reset();
	U.reset();
	L.reset();
	R_inv_I.reset();
	R_inv_ys_min_beta.reset();
	X.reset();
	XnotNormalized.reset();
	data.reset();
	I.reset();


	beta0 = 0.0;
	sigma_sqr = 0.0;



	/* data matrix input */

	bool status = data.load(input_filename.c_str(), csv_ascii);
	if(status == true)
	{
		printf("Data input is done\n");
	}
	else
	{
		printf("Problem with data the input (cvs ascii format) at %s, line %d.\n", __FILE__, __LINE__);
		exit(-1);
	}

	if(dim != data.n_cols - 1){

		printf("Dimensions of the data matrix do not match!, dim = %d, data.n_cols = %d at %s, line %d.\n",dim, int(data.n_cols), __FILE__, __LINE__);
		exit(-1);

	}


	N = data.n_rows;

	X = data.submat(0, 0, N - 1, dim - 1);

	XnotNormalized = X;

	for (unsigned int i = 0; i < dim; i++) {

		xmax(i) = data.col(i).max();
		xmin(i) = data.col(i).min();

	}

	/* normalize data matrix */

	for (unsigned int i = 0; i < N; i++) {

		for (unsigned int j = 0; j < dim; j++) {

			X(i, j) = (1.0/dim)*(X(i, j) - xmin(j)) / (xmax(j) - xmin(j));
		}
	}

#if 0
	printf("Normalized data = \n");
	X.print();
#endif


	R.zeros(N,N);
	L.zeros(N,N);
	U.zeros(N,N);
	R_inv_ys_min_beta.zeros(N);
	R_inv_I.zeros(N);
	I.ones(N);

	ymin = min(data.col(dim));
	ymax = max(data.col(dim));
	yave = mean(data.col(dim));



	updateModelParams();

}

/*
 * Evaluates the surrogate model at x = xp
 * @param[in] xp (normalized)
 * @return y=ftilde(xp)
 *
 * */
double KrigingModel::ftilde(rowvec xp){

	vec r(N);

	vec theta = kriging_weights.head(dim);
	vec gamma = kriging_weights.tail(dim);


	double f_regression = 0.0;
	double f_kriging = 0.0;


	/* if linear regression is on */
	if(linear_regression){

		for(unsigned int i=0; i<dim; i++){

			f_regression += xp(i)*regression_weights(i+1);
		}

		f_regression += regression_weights(0);

	}
#if 0
	printf("f_regression = %10.7f\n",f_regression);
#endif

	for(unsigned int i=0;i<N;i++){

		r(i) = compute_R(xp, X.row(i), theta, gamma);

	}

#if 0
	printf("size of vector r = %d\n",r.size());
	printf("r = \n",f_regression);
	trans(r).print();
	printf("size of vector R_inv_ys_min_beta = %d\n",R_inv_ys_min_beta.size());
	printf("R_inv_ys_min_beta = \n",f_regression);
	trans(R_inv_ys_min_beta).print();
#endif

	f_kriging = beta0+ dot(r,R_inv_ys_min_beta);

	double ftilde = f_regression+f_kriging;

	return ftilde;

}

/*
 * Evaluates the surrogate model at x = xp
 * @param[in] xp (not normalized)
 * @return y=ftilde(xp)
 *
 * */


double KrigingModel::calculateEI(rowvec xp){


	double ftilde = 0.0;
	double ssqr   = 0.0;

	ftilde_and_ssqr(xp,&ftilde,&ssqr);

#if 0
	printf("ftilde = %15.10f, ssqr = %15.10f\n",ftilde,ssqr);
#endif

	double	standart_error = sqrt(ssqr)	;

	double EI = 0.0;

	if(standart_error!=0.0){

		double	EIfac = (ymin - ftilde)/standart_error;

		/* calculate the Expected Improvement value */
		EI = (ymin - ftilde)*cdf(EIfac,0.0,1.0)+ standart_error * pdf(EIfac,0.0,1.0);
	}
	else{

		EI = 0.0;

	}

	return EI;

}


double KrigingModel::ftildeNotNormalized(rowvec xp){

	/* first normalize xp */

	rowvec xpNormalized(N);

	for(unsigned int i=0; i<dim; i++){

		xpNormalized(i) = (1.0/dim)*(xp(i) - xmin(i)) / (xmax(i) - xmin(i));

	}


	double ftildeVal = ftilde(xpNormalized);

	return ftildeVal;

}



void KrigingModel::ftilde_and_ssqr(rowvec xp,double *ftildeVal,double *ssqr){

	vec r(N);

	vec theta = kriging_weights.head(dim);
	vec gamma = kriging_weights.tail(dim);


	for(unsigned int i=0; i<N; i++){

		r(i) = compute_R(xp, X.row(i), theta, gamma);

	}



	*ftildeVal =  ftilde(xp);;

	vec R_inv_r(N);


	/* solve the linear system R x = r by Cholesky matrices U and L*/
	solve_linear_system_by_Cholesky(U, L, R_inv_r, r);


	*ssqr = sigma_sqr*( 1.0 - dot(r,R_inv_r)+ ( pow( (dot(r,R_inv_I) -1.0 ),2.0)) / (dot(I,R_inv_I) ) );


}


void KrigingModel::train(void){

	printf("\nTraining Kriging response surface for the data : %s\n",input_filename.c_str());

	vec ys = data.col(dim);

	/* train linear regression */
	if (linear_regression) { // if linear regression is on

		printf("Performing linear regression...\n");

		train_linear_regression(X, ys, regression_weights, 10E-6);

		printf("regression weights:\n");
		trans(regression_weights).print();

		mat augmented_X(N, dim + 1);

		for (unsigned int i = 0; i < N; i++) {

			for (unsigned int j = 0; j <= dim; j++) {

				if (j == 0){

					augmented_X(i, j) = 1.0;
				}
				else {

					augmented_X(i, j) = X(i, j - 1);

				}
			}
		}

		/* now update the ys vector */

		vec ys_reg = augmented_X * regression_weights;

#if 0
		for(unsigned int i=0; i<ys.size(); i++){

			printf("%10.7f %10.7f\n",ys(i),ys_reg(i));
		}
#endif
		ys = ys - ys_reg;

#if 0
		printf("Updated ys vector = \n");
#endif

		for(unsigned int i=0; i<regression_weights.size();i++ ){

			if(fabs(regression_weights(i)) > 10E5){

				printf("WARNING: Linear regression coefficients are too large= \n");
				printf("regression_weights(%d) = %10.7f\n",i,regression_weights(i));
			}

		}



	} /* end of linear regression */


	int max_number_of_function_calculations = max_number_of_kriging_iterations;

	/* initialize random seed*/
	srand (time(NULL));

	/* get the number of treads */
	int number_of_treads = 1;
#pragma omp parallel
	{

		int tid = omp_get_thread_num();
		number_of_treads = omp_get_num_threads();


		if (tid == 0){
			printf("number of threads used : %d\n", number_of_treads);
			max_number_of_function_calculations = max_number_of_function_calculations/number_of_treads;
			printf("number of function evaluations per thread : %d\n", max_number_of_function_calculations);
		}

	}


	int number_of_initial_population = dim * 16 / number_of_treads;

	if(number_of_initial_population < 100) number_of_initial_population = 100;


#pragma omp parallel
	{

		int tid;

		std::vector<EAdesign>::iterator it;
		std::vector<EAdesign> population;
		int population_size = 0;

		double population_max = -LARGE;
		int population_max_index = -1;



		mat Rmatrix= zeros(N,N);
		mat Umatrix= zeros(N,N);
		mat Lmatrix= zeros(N,N);

		mat Xmatrix = X;


		vec Ivector = I;
		vec ys = data.col(dim);

		unsigned int d = dim;
		double epsilon = epsilon_kriging;


		double Jnormalization_factor = 0.0;




		for (int i = 0; i < max_number_of_function_calculations; i++) {


			if(i==0){

				EAdesign initial_design(d);

				if(file_exist(kriging_hyperparameters_filename.c_str())){
#pragma omp master
					{
						printf("Hyperparameter file: %s exists...\n",kriging_hyperparameters_filename.c_str() );
					}


					vec weights_read_from_file;

					bool status = weights_read_from_file.load(kriging_hyperparameters_filename.c_str(),csv_ascii);

					if(status == false)
					{
						fprintf(stderr, "Error: Some problem with the hyperparameter input! at %s, line %d.\n",__FILE__, __LINE__);
						exit(-1);

					}

#if 1
#pragma omp master
					{
						printf("hyperparameters read from the file (theta; gamma)^T:\n");
						trans(weights_read_from_file).print();
					}
#endif

					if(weights_read_from_file.size() != 2*d){
#if 1
#pragma omp master
						{
							printf("hyper parameters do not match the problem dimensions!\n");
						}
#endif
						for (unsigned int j = 0; j < d; j++) {

							initial_design.theta(j) = randomDouble(0, 10); //theta
							initial_design.gamma(j) = randomDouble(0, 2); //gamma

						}

					}

					else{

						for(unsigned i=0; i<d;i++) {

							if(weights_read_from_file(i) < 100.0){

								initial_design.theta(i) = weights_read_from_file(i);
							}
							else{

								initial_design.theta(i) = 1.0;
							}


						}
						for(unsigned i=dim; i<2*d;i++) {

							if( weights_read_from_file(i) < 2.0) {

								initial_design.gamma(i-dim) = weights_read_from_file(i);

							}
							else{

								initial_design.gamma(i-dim) = 1.0;
							}
						}

					}


				}
				else{ /* assign random parameters if there is no file */

					for (unsigned int j = 0; j < d; j++) {

						initial_design.theta(j) = randomDouble(0, 10); //theta
						initial_design.gamma(j) = randomDouble(0, 2); //gamma

					}


				}
#if 0
				initial_design.theta.print();
				initial_design.gamma.print();

#endif

				initial_design.calculate_fitness(epsilon,Xmatrix,ys);


				if (initial_design.objective_val != -LARGE) {

					Jnormalization_factor = initial_design.objective_val;
					initial_design.objective_val = 0.0;
					initial_design.id = population_size;
					population.push_back(initial_design);
					population_size++;

				}


				continue;


			} /* i=0 */




			/* update population properties after initital iterations */
			if (i >= number_of_initial_population) {

				update_population_properties (population);
			}


#if 0
			if (total_number_of_function_evals % 100 == 0){

				printf("\r# of function calculations =  %d\n",
						total_number_of_function_evals);
				fflush(stdout);

			}
#endif
			EAdesign new_born(d);


			if (i < number_of_initial_population) {

				for (unsigned int j = 0; j < d; j++) {

					new_born.theta(j) = randomDouble(0, 10); //theta
					new_born.gamma(j) = randomDouble(0, 2); //gamma

				}

			} else {

				int father_id = -1;
				int mother_id = -1;
				pickup_random_pair(population, mother_id, father_id);

				crossover_kriging(population[mother_id], population[father_id],new_born);
			}

#if 0
			new_born.theta.print();
			new_born.gamma.print();
#endif

			new_born.calculate_fitness(epsilon,X,ys);


			if (new_born.objective_val != -LARGE) {

				if(Jnormalization_factor != 0){

					new_born.objective_val = new_born.objective_val-Jnormalization_factor;

				}
				else{
					Jnormalization_factor = new_born.objective_val;
					new_born.objective_val = 0.0;


				}



				new_born.id = population_size;

				population.push_back(new_born);
				population_size++;


			}

			if (population_size % 10 == 0 && i > number_of_initial_population) { //in each 10 iteration find population max

				population_max = -LARGE;
				for (it = population.begin(); it != population.end(); ++it) {

					if (it->objective_val > population_max) {
						population_max = it->objective_val;
						population_max_index = it->id;
					}

				}


			}

		} // end of for loop


		population_max= -LARGE;
		for (it = population.begin() ; it != population.end(); ++it){

			if ( it->objective_val >  population_max){
				population_max = it->objective_val;
				population_max_index = it->id;
			}


		}


#pragma omp barrier

#if 0
		printf("kriging model training is over...\n");
		printf("tread %d best design log likelihood = %10.7f\n",omp_get_thread_num(),population_max );
#endif



#pragma omp critical
		{
			if (population_max > population_overall_max){
#if 0
				printf("tread %d, population_overall_max = %10.7f\n",omp_get_thread_num(),population_overall_max);
#endif
				population_overall_max = population_max;
				population_overall_max_tread_id = omp_get_thread_num();
			}
		}


#pragma omp barrier


		tid = omp_get_thread_num();
		if(tid == population_overall_max_tread_id){


#if 0
			printf("tread %d has the best design with the log likelihood = %10.7f\n",population_overall_max_tread_id,population_overall_max );

			population.at(population_max_index).print();
#endif

			kriging_weights.reset();
			kriging_weights = zeros(2*d);


			for(unsigned int i=0;i<d;i++) {

				kriging_weights(i) = population.at(population_max_index).theta(i);
			}
			for(unsigned int i=0;i<d;i++) {

				kriging_weights(i+d)= population.at(population_max_index).gamma(i);
			}


			kriging_weights.save(kriging_hyperparameters_filename.c_str(),csv_ascii);

		}

	} /* end of parallel section */

	printf("Kring training is done\n");
	printf("Kriging weights:\n");
	trans(kriging_weights).print();


	updateModelParams();

}


void KrigingModel::validate(std::string filename, bool ifVisualize){

	mat dataValidation;

	printf("\nValidating Kriging response surface for the data : %s\n",filename.c_str());

	bool status = dataValidation.load(input_filename.c_str(), csv_ascii);
	if(status == true)
	{
		printf("Data input is done\n");
	}
	else
	{
		printf("Problem with data the input (cvs ascii format)\n");
		exit(-1);
	}


	unsigned int NValidation = dataValidation.n_rows;

	mat XValidation = dataValidation.submat(0, 0, NValidation - 1, dim - 1);

	mat XnotNormalizedValidation = XValidation;


	/* normalize data matrix for the Validation */

	for (unsigned int i = 0; i < N; i++) {

		for (unsigned int j = 0; j < dim; j++) {

			XValidation(i, j) = (1.0/dim)*(XValidation(i, j) - xmin(j)) / (xmax(j) - xmin(j));
		}
	}


}



EAdesign::EAdesign(int dimension){

	theta = zeros(dimension);
	gamma = zeros(dimension);


}

void EAdesign::print(void){

	printf("id = %d,  J = %10.7f, fitness =  %10.7f cross-over p = %10.7f\n",id,objective_val,fitness,crossover_probability);
#if 0
	printf("theta = \n");
	theta.print();
	if(gamma.size() > 0){
		printf("gamma = \n");
		gamma.print();
	}
#endif
}


int EAdesign::calculate_fitness(double epsilon, mat &X,vec &ys){

	double beta0 = 0.0;
	double ssqr = 0.0;
	double logdetR = 0.0;
	double objVal = 0.0;

	int N = X.n_rows;  /* number of samples */


	mat R = zeros(N,N);
	mat L = zeros(N,N);
	mat U = zeros(N,N);

	vec I = ones(N);

	// compute the correlation matrix R

	//	double reg_param = pow(10.0,-1.0*new_born.log_regularization_parameter);

	compute_R_matrix(theta, gamma,epsilon,R,X);


	//		R.raw_print(cout, "R:");


	// compute Cholesky decomposition
	int flag = chol(U, R);

	if (flag == 0) {

		printf("Ill conditioned correlation matrix! \n");
		objective_val = -LARGE;
		total_number_of_function_evals++;
		return 0;
	}

	L = trans(U);

	//L.raw_print(cout, "L:");

	//U.raw_print(cout, "U:");

	vec U_diagonal = U.diag();

	//U_diagonal.raw_print(cout, "diag(U):");

	logdetR = 0.0;

	for (int i = 0; i < N; i++) {

		if (U_diagonal(i) < 0) {

			objective_val = -LARGE;
			total_number_of_function_evals++;
			return 0;
		}

		logdetR += log(U_diagonal(i));

		//		printf("%d %30.25f %30.25f\n",i,logdetR,log(U_diagonal(i)));
	}

	logdetR = 2.0 * logdetR;

	vec R_inv_ys(N);
	vec R_inv_I(N);



	solve_linear_system_by_Cholesky(U, L, R_inv_ys, ys); /* solve R x = ys */
	solve_linear_system_by_Cholesky(U, L, R_inv_I, I);   /* solve R x = I */




	beta0 = (1.0/dot(I,R_inv_I)) * (dot(I,R_inv_ys));
	//	printf("beta0= %20.15f\n",beta0);

	vec ys_min_betaI = ys-beta0*I;

	vec R_inv_ys_min_beta(N);


	/* solve R x = ys-beta0*I */
	solve_linear_system_by_Cholesky(U, L, R_inv_ys_min_beta, ys_min_betaI);


	double fac = dot(ys_min_betaI, R_inv_ys_min_beta);

	ssqr = (1.0 / N) * fac;


	if(ssqr > 0 ){
		objVal = (- N / 2.0) * log(ssqr);
		objVal -= 0.5 * logdetR;
	}
	else{

		objective_val = -LARGE;
		total_number_of_function_evals++;
		return 0;
	}

#ifdef calculate_fitness_CHECK
	printf("\n");
	printf("objective function value = %10.7f\n",obj_val);
	printf("s^2= %20.15f\n",ssqr);
	printf("(-M/2.0)* log(ssqr) = %10.7f\n",(-dimension_of_R/2.0)* log(ssqr));
	printf("-0.5*logdetR = %10.7f\n",-0.5*logdetR);
	printf("\n");
#endif


	total_number_of_function_evals++;

	objective_val = objVal;

	return 0;



}



/* print all the population (can be a lot of mess!)*/

void print_population(std::vector<EAdesign> population){

	std::vector<EAdesign>::iterator it;

	printf("\nPopulation:\n");

	for (it = population.begin() ; it != population.end(); ++it){

		it->print();

	}
	printf("\n");

}










/* implementation according to the Forrester book */
//void compute_R_matrix_GEK(vec theta,
//		double reg_param,
//		mat& R,
//		mat &X,
//		mat &grad) {
//
//
//
//
//	int k = X.n_cols;
//	int n = X.n_rows;
//
//
//	mat Psi=zeros(n,n);
//	mat PsiDot=zeros(n,n);
//
//
//	mat Rfull;
//
//	for(int row = -1; row < k; row++){
//
//		if(row == -1){ /* first row */
//
//			for(int i=0; i<n;i++){
//				for(int j=i+1;j<n;j++){
//					Psi(i,j)=compute_R_Gauss(X.row(i),X.row(j), theta);
//
//				}
//			}
//
//			Psi = Psi+ trans(Psi)+ eye(n,n);
//
//			Rfull=Psi;
//
//
//			PsiDot=zeros(n,n);
//			for(int l=0;l<k; l++){
//
//
//				for(int i=0; i<n;i++){
//					for(int j=0;j<n;j++){
//						PsiDot(i,j)=2.0*theta(l)* (X(i,l)-X(j,l))*Psi(i,j);
//
//					}
//				}
//				Rfull = join_rows(Rfull,PsiDot);
//
//			}
//
//		}
//
//		else{ /* other rows */
//
//			mat Rrow;
//
//			PsiDot=zeros(n,n);
//
//			for(int i=0; i<n;i++){
//				for(int j=0;j<n;j++){
//					PsiDot(i,j)=-2.0*theta(row)* (X(i,row)-X(j,row))*Psi(i,j);
//
//				}
//			}
//
//
//
//			Rrow = PsiDot;
//
//
//			for(int l=0; l<k;l++){
//				mat PsiDot2=zeros(n,n);
//
//				if(l == row){
//					for(int i=0; i<n;i++){
//						for(int j=0;j<n;j++){
//							PsiDot2(i,j)=
//									(2.0*theta(l)-4.0*theta(l)*theta(l)* pow((X(i,l)-X(j,l)),2.0))*Psi(i,j);
//
//						}
//					}
//
//				}
//
//				else{
//
//
//					for(int i=0; i<n;i++){
//						for(int j=0;j<n;j++){
//							PsiDot2(i,j)=
//									(-4.0*theta(row)*theta(l)*(X(i,row)-X(j,row))*(X(i,l)-X(j,l)))*Psi(i,j);
//
//						}
//					}
//				}
//
//
//				Rrow = join_rows(Rrow,PsiDot2);
//
//			}
//
//
//
//			Rfull = join_cols(Rfull,Rrow);
//		}
//
//
//
//
//	} /* end of for loop for rows */
//
//
//
//	R = Rfull + reg_param*eye(n*(k+1),n*(k+1));
//
//
//
//} /* end of compute_R_matrix_GEK */
//




/** compute the f_tilde for the Kriging model
 * @param[in] xp : point of evaluation
 * @param[in] X : normalized data matrix
 * @param[in] beta0 = [I^T*R^-1*I]^-1 [I^T*R^-1*ys]
 * @param[in] sigma_sqr
 * @param[in] regression weights
 * @param[in] R_inv_ys_min_beta = R^-1*(ys-beta0*I)
 * @param[out] f_tilde : surrogate prediction
 */
double calculate_f_tilde(rowvec xp,
		mat &X,
		double beta0,
		vec regression_weights,
		vec R_inv_ys_min_beta,
		vec kriging_weights){


	int dim = xp.size();
#if 0
	printf("dim = %d\n",dim);
#endif

	vec r(X.n_rows);

	vec theta = kriging_weights.head(dim);
	vec gamma = kriging_weights.tail(dim);


	double f_regression = 0.0;
	double f_kriging = 0.0;


	/* if there exists linear regression part */
	if(regression_weights.size() !=0){

		for(unsigned int i=0;i<xp.size();i++){

			f_regression += xp(i)*regression_weights(i+1);
		}

		f_regression += regression_weights(0);

	}
#if 0
	printf("f_regression = %10.7f\n",f_regression);
#endif

	for(unsigned int i=0;i<X.n_rows;i++){

		r(i) = compute_R(xp, X.row(i), theta, gamma);

	}
#if 0
	printf("size of vector r = %d\n",r.size());
	printf("r = \n",f_regression);
	trans(r).print();
	printf("size of vector R_inv_ys_min_beta = %d\n",R_inv_ys_min_beta.size());
	printf("R_inv_ys_min_beta = \n",f_regression);
	trans(R_inv_ys_min_beta).print();
#endif

	f_kriging = beta0+ dot(r,R_inv_ys_min_beta);


	return f_regression+f_kriging;

}


/** compute the f_tilde and ssqr for the Kriging model
 * @param[in] xp : point of evaluation
 * @param[in] X : normalized data matrix
 * @param[in] beta0 = [I^T*R^-1*I]^-1 [I^T*R^-1*ys]
 * @param[in] sigma_sqr
 * @param[in] I : identity vector
 * @param[in] R_inv_ys_min_beta = R^-1*(ys-beta0*I)
 * @param[in] R_inv_I = R^-1* I
 * @param[in] U (Cloesky decomposition of R)
 * @param[in] L (Cloesky decomposition of R)
 * @param[out] f_tilde : surrogate prediction
 * @param[out] ssqr: kriging variance
 */
void calculate_f_tilde_and_ssqr(
		rowvec xp,
		mat &X,
		double beta0,
		double sigma_sqr,
		vec regression_weights,
		vec R_inv_ys_min_beta,
		vec R_inv_I,
		vec I,
		vec kriging_weights,
		mat &U,
		mat &L,
		double *f_tilde,
		double *ssqr){


	int dim = xp.size();
	vec r(X.n_rows);

	vec theta = kriging_weights.head(dim);
	vec gamma = kriging_weights.tail(dim);



	double f_regression = 0.0;
	double f_kriging = 0.0;


	if(regression_weights.size() !=0){

		for(unsigned int i=0;i<xp.size();i++){

			f_regression += xp(i)*regression_weights(i+1);
		}

		f_regression += regression_weights(0);

	}


	for(unsigned int i=0;i<X.n_rows;i++){

		r(i) = compute_R(xp, X.row(i), theta, gamma);

	}

	f_kriging = beta0+ dot(r,R_inv_ys_min_beta);



	*f_tilde =  f_regression+f_kriging;

	vec R_inv_r(X.n_rows);


	/* solve the linear system R x = r by Cholesky matrices U and L*/
	solve_linear_system_by_Cholesky(U, L, R_inv_r, r);


	*ssqr = sigma_sqr*( 1.0 - dot(r,R_inv_r)
			+ ( pow( (dot(r,R_inv_I) -1.0 ),2.0)) / (dot(I,R_inv_I) ) );



}


// IMPORTANT : vector x should be normalized!!!
double calculate_f_tilde_GEK(rowvec &x,
		mat &X,
		double beta0,
		vec regression_weights,
		vec R_inv_ys_min_beta,
		vec kriging_weights,
		int Ngrad){


#if 0
	printf("calculate_f_tilde_GEK...\n");
#endif

	int dim = x.size();

	int Ntotal =X.n_rows;
	int Nfunc = Ntotal-Ngrad;

	int dimension_R = Nfunc+Ngrad+ dim*Ngrad;

	vec r(dimension_R);

	vec theta = kriging_weights.head(dim);


#if 0
	theta.raw_print(cout, "theta:");
	x.raw_print(cout, "x:");
	regression_weights.raw_print(cout, "regression_weights:");
#endif



	double f_regression = 0.0;
	double f_kriging = 0.0;

	/* if there exists some regression weights, than perform linear regression */
	if (regression_weights.size() > 0){

		for(unsigned int i=0; i<x.size(); i++){

			f_regression += x(i)*regression_weights(i+1);
		}

		f_regression += regression_weights(0);
	}


	/* first Nfunc+Ngrad elements are normal functional correlations */
	for(int k=0; k<Nfunc+Ngrad; k++) {

		r(k)= compute_R_Gauss(X.row(k),x,theta);
	}

	int count=Nfunc+Ngrad;



	for(int k=Nfunc;k<Nfunc+Ngrad;k++) {

		for(int l=0;l<dim;l++) {

			r(count+l*Ngrad+k)= compR_dxi(X.row(k), x, theta, l) ;
		}
	}

	f_kriging = beta0+ dot(r,R_inv_ys_min_beta);

	return f_regression+f_kriging;

}



///* calculate the fitness of a member with the given theta (only for GEK)*/
//int calculate_fitness_GEK(member &new_born,
//		double reg_param,
//		mat &R,
//		mat &X,
//		vec &ys,
//		vec &F,
//		mat &grad,
//		int eqn_sol_method ) {
//
//	int Mtotal = X.n_rows;    /* number of total data points */
//	int Mgrad  = grad.n_rows; /* number of data points with only function value */
//	int Mfunc = Mtotal-Mgrad; /* number of data points with gradient information */
//	int dim = X.n_cols;
//	int dimension_of_R = Mfunc+Mgrad+dim*Mgrad; /* dimension of the correlation matrix */
//
//
//	double logdetR=0.0;
//
//	mat U(dimension_of_R, dimension_of_R);
//	mat D(dimension_of_R, dimension_of_R);
//	mat L(dimension_of_R, dimension_of_R);
//	mat V(dimension_of_R, dimension_of_R);
//	mat Ut(dimension_of_R, dimension_of_R);
//
//	vec s(dimension_of_R);
//	vec sinv(dimension_of_R);
//
//
//	// compute the correlation matrix R
//
//	//	double reg_param = pow(10.0, -1.0* new_born.log_regularization_parameter );
//
//	compute_R_matrix_GEK(new_born.theta, reg_param,R,X,grad);
//
//	//	R.print();
//
//#ifdef calculate_fitness_CHECK
//	mat Rinv= inv(R);
//#endif
//	//	invR.print();
//
//	//	mat check =(R*invR);
//	//	check.print();
//
//	//	exit(1);
//
//	if(eqn_sol_method == CHOLESKY){
//		/* compute Cholesky decomposition of the correlation matrix R */
//		int flag = chol(U, R);
//
//		if (flag == 0) {
//			printf("Ill conditioned correlation matrix in Cholesky decomposition...\n");
//			new_born.objective_val = -LARGE;
//			total_number_of_function_evals++;
//			return 0;
//		}
//
//
//		/* calculate upper triangular from U */
//		L = trans(U);
//
//
//	}
//
//
//	if( eqn_sol_method == SVD){
//		int flag_svd = svd( U, s, V, R );
//
//		if (flag_svd == 0) {
//
//			printf("Error: SVD could not be performed\n");
//			exit(-1);
//		}
//
//		sinv = 1.0/s;
//#if 0
//		printf("inverse singular values = \n");
//		sinv.print();
//#endif
//		double threshold = 10E-8;
//		for(int i=0; i< dimension_of_R; i++){
//			if(s(i)  < threshold){
//				sinv(i) = 0.0;
//			}
//
//		}
//#if 0
//		printf("inverse singular values after thresholding= \n");
//		sinv.print();
//		printf("\n");
//#endif
//
//		Ut = trans(U);
//
//		D.fill(0.0);
//		for(int i=0; i< dimension_of_R; i++){
//
//			D(i,i) = sinv(i);
//		}
//
//	}
//
//
//	/* calculate the determinant of R after Cholesky decomposition*/
//	if(eqn_sol_method == CHOLESKY){
//		vec U_diagonal = U.diag();
//
//		//U_diagonal.raw_print(cout, "diag(U):");
//
//		logdetR = 0.0;
//		for (int i = 0; i < dimension_of_R; i++) {
//			if (U_diagonal(i) < 0) {
//
//				new_born.objective_val = -LARGE;
//				return 0;
//			}
//
//			logdetR += log(U_diagonal(i));
//
//			//		printf("%d %30.25f %30.25f\n",i,logdetR,log(U_diagonal(i)));
//		}
//
//		logdetR = 2.0 * logdetR;
//
//	}
//
//	if(eqn_sol_method == SVD){
//
//		logdetR = 0.0;
//		for (int i = 0; i < dimension_of_R; i++) {
//
//			logdetR += log(s(i));
//
//		}
//
//	}
//
//
//
//
//
//
//	vec R_inv_ys(dimension_of_R);
//	vec R_inv_F(dimension_of_R);
//
//	if( eqn_sol_method == CHOLESKY){
//
//		solve_linear_system_by_Cholesky(U, L, R_inv_ys, ys);
//		solve_linear_system_by_Cholesky(U, L, R_inv_F, F);
//	}
//
//
//	if( eqn_sol_method == SVD){
//		R_inv_ys =V*D*Ut*ys;
//		R_inv_F  =V*D*Ut*F;
//	}
//
//
//#ifdef calculate_fitness_CHECK
//
//	vec R_inv_ys_check=Rinv*ys;
//
//	double R_inv_ys_error= norm((R_inv_ys-R_inv_ys_check), 2);
//
//	if(R_inv_ys_error > TOL) {
//		printf("R_inv_ys is wrong\n");
//
//		for(unsigned int i=0; i<R_inv_ys.size();i++){
//			printf("%10.7f %10.7f\n",R_inv_ys(i),R_inv_ys_check(i));
//
//
//		}
//
//		exit(-1);
//	}
//
//#endif
//
//
//
//#ifdef calculate_fitness_CHECK
//
//	vec R_inv_F_check=Rinv*F;
//
//	//v2_check.raw_print(cout, "v2_check:");
//
//	double R_inv_F_error= norm((R_inv_F-R_inv_F_check), 2);
//
//	if(R_inv_F_error > TOL) {
//		printf("R_inv_F is wrong\n");
//		printf("R_inv_F error = %10.7f\n",R_inv_F_error);
//
//		for(unsigned int i=0; i<R_inv_F.size();i++){
//			printf("%10.7f %10.7f\n",R_inv_F(i),R_inv_F_check(i));
//
//
//		}
//
//
//		exit(-1);
//	}
//
//#endif
//
//	double beta0 = (1.0/dot(F,R_inv_F)) * (dot(F,R_inv_ys));
//	printf("beta0= %10.7f\n",beta0);
//
//	vec ys_min_betaF = ys-beta0*F;
//
//	vec R_inv_ys_min_beta(dimension_of_R);
//
//	if( eqn_sol_method == CHOLESKY){
//
//		solve_linear_system_by_Cholesky(U, L, R_inv_ys_min_beta, ys_min_betaF);
//
//	}
//
//	if( eqn_sol_method == SVD){
//		R_inv_ys_min_beta = V*D*Ut*ys_min_betaF;
//
//	}
//
//
//#ifdef calculate_fitness_CHECK
//
//	double R_inv_ys_min_beta_check_error;
//	vec R_inv_ys_min_beta_check=Rinv*(ys_min_betaF);
//
//	//v2_check.raw_print(cout, "v2_check:");
//
//	R_inv_ys_min_beta_check_error= norm((R_inv_ys_min_beta_check-R_inv_ys_min_beta_check), 2);
//
//	if(R_inv_ys_min_beta_check_error > TOL) {
//		printf("R_inv_ys_min_beta_check is wrong\n");
//
//
//		for(unsigned int i=0; i<R_inv_ys_min_beta.size();i++){
//			printf("%10.7f %10.7f\n",R_inv_ys_min_beta(i),R_inv_ys_min_beta_check(i));
//
//
//		}
//
//		exit(-1);
//	}
//
//#endif
//
//	double fac = dot(ys_min_betaF, R_inv_ys_min_beta);
//
//	double ssqr = (1.0 / dimension_of_R) * fac;
//
//	double obj_val;
//
//#if 0
//	printf("s^2= %10.7f\n",ssqr);
//#endif
//
//	if(ssqr > 0 ){
//		obj_val = (-dimension_of_R / 2.0) * log(ssqr);
//		obj_val -= 0.5 * logdetR;
//	}
//	else{
//		new_born.objective_val = -LARGE;
//		total_number_of_function_evals++;
//		return 0;
//	}
//
//#if 0
//	printf("\n");
//	printf("objective function value = %10.7f\n",obj_val);
//	printf("(-M/2.0)* log(ssqr) = %10.7f\n",(-dimension_of_R/2.0)* log(ssqr));
//	printf("-0.5*logdetR = %10.7f\n",-0.5*logdetR);
//	printf("\n");
//#endif
//
//
//	total_number_of_function_evals++;
//
//	new_born.objective_val = obj_val;
//
//	return 1;
//
//
//
//}

/* calculate the fitness of a member with the given theta and gamma  */
int calculate_fitness(EAdesign &new_born,
		double &reg_param,
		mat &R,
		mat &U,
		mat &L,
		mat &X,
		vec &ys,
		vec &I) {

	//#define calculate_fitness_CHECK

	//	printf("calling calculate_fitness function...\n");
	double beta0, ssqr;
	double logdetR;
	double obj_val;

	int M = X.n_rows;  /* number of samples */
	int dimension_of_R = R.n_rows;

	// compute the correlation matrix R

	//	double reg_param = pow(10.0,-1.0*new_born.log_regularization_parameter);

	compute_R_matrix(new_born.theta,
			new_born.gamma,
			reg_param,
			R,
			X);


	//		R.raw_print(cout, "R:");


	// compute Cholesky decomposition
	int flag = chol(U, R);

	if (flag == 0) {
		printf("Ill conditioned correlation matrix\n");
		new_born.objective_val = -LARGE;
		total_number_of_function_evals++;
		return 0;
	}

	L = trans(U);

	//L.raw_print(cout, "L:");

	//U.raw_print(cout, "U:");

	vec U_diagonal = U.diag();

	//U_diagonal.raw_print(cout, "diag(U):");

	logdetR = 0.0;
	for (int i = 0; i < M; i++) {
		if (U_diagonal(i) < 0) {

			new_born.objective_val = -LARGE;
			total_number_of_function_evals++;
			return 0;
		}

		logdetR += log(U_diagonal(i));

		//		printf("%d %30.25f %30.25f\n",i,logdetR,log(U_diagonal(i)));
	}

	logdetR = 2.0 * logdetR;

	vec R_inv_ys(dimension_of_R);
	vec R_inv_I(dimension_of_R);



	solve_linear_system_by_Cholesky(U, L, R_inv_ys, ys); /* solve R x = ys */
	solve_linear_system_by_Cholesky(U, L, R_inv_I, I);   /* solve R x = I */




	beta0 = (1.0/dot(I,R_inv_I)) * (dot(I,R_inv_ys));
	//	printf("beta0= %20.15f\n",beta0);

	vec ys_min_betaI = ys-beta0*I;

	vec R_inv_ys_min_beta(dimension_of_R);


	/* solve R x = ys-beta0*I */
	solve_linear_system_by_Cholesky(U, L, R_inv_ys_min_beta, ys_min_betaI);


	double fac = dot(ys_min_betaI, R_inv_ys_min_beta);

	ssqr = (1.0 / dimension_of_R) * fac;


	if(ssqr > 0 ){
		obj_val = (-dimension_of_R / 2.0) * log(ssqr);
		obj_val -= 0.5 * logdetR;
	}
	else{
		new_born.objective_val = -LARGE;
		total_number_of_function_evals++;
		return 0;
	}

#ifdef calculate_fitness_CHECK
	printf("\n");
	printf("objective function value = %10.7f\n",obj_val);
	printf("s^2= %20.15f\n",ssqr);
	printf("(-M/2.0)* log(ssqr) = %10.7f\n",(-dimension_of_R/2.0)* log(ssqr));
	printf("-0.5*logdetR = %10.7f\n",-0.5*logdetR);
	printf("\n");
#endif


	total_number_of_function_evals++;

	new_born.objective_val = obj_val;

	return 1;
}

/* crossover function of two designs */
void crossover_kriging(EAdesign &father, EAdesign &mother, EAdesign &child) {

	int dim = father.theta.size();
	for (int i = 0; i < dim; i++) {
		//		printf("theta (m) = %10.7f theta (f) = %10.7f\n",mother.theta[i],father.theta[i]);
		//		printf("gamma (m) = %10.7f gamma (f) = %10.7f\n",mother.gamma[i],father.gamma[i]);
		child.theta(i) = random_number(father.theta(i), mother.theta(i), 4.0);
		child.gamma(i) = random_number(father.gamma(i), mother.gamma(i), 4.0);

		if (child.theta(i) < 0)
			child.theta(i) = randomDouble(0.0, 10.0);
		if (child.gamma(i) < 0)
			child.gamma(i) = randomDouble(0.1, 2.0);
		if (child.gamma(i) > 2.0)
			child.gamma(i) = randomDouble(0.1, 2.0);
#if 0
		if (i==0) printf("theta = %10.7f gamma = %10.7f\n",child.theta[i],child.gamma[i]);
#endif
	}

	/*
	child.log_regularization_parameter =
			random_number(father.log_regularization_parameter,
					mother.log_regularization_parameter, 4.0);


	if(child.log_regularization_parameter < 0 ) child.log_regularization_parameter=0.0;
	if(child.log_regularization_parameter > 14 ) child.log_regularization_parameter=14.0;

	 */


}

/* crossover function of two designs for GEK (only for theta components)*/
//void crossover_GEK(member &father, member &mother, member &child) {
//
//	int dim = father.theta.size();
//	for (int i = 0; i < dim; i++) {
//
//		child.theta(i) = random_number(father.theta(i), mother.theta(i), 4.0);
//
//
//		if (child.theta(i) < 0)
//			child.theta(i) = RandomDouble(0.0, 20.0);
//
//	}
//	/*
//
//	child.log_regularization_parameter =
//			random_number(father.log_regularization_parameter,
//					mother.log_regularization_parameter, 4.0);
//
//
//	if(child.log_regularization_parameter < 0 ) child.log_regularization_parameter=0.0;
//	if(child.log_regularization_parameter > 14 ) child.log_regularization_parameter=14.0;
//
//	 */
//
//
//}




void pickup_random_pair(std::vector<EAdesign> population, int &mother,int &father) {

	double rand_num;
	std::vector<EAdesign>::iterator it;

	//usleep(1000000);

	rand_num = rand() % 1000000;
	rand_num = rand_num / 1000000;

	double pro_sum = 0.0;
	for (it = population.begin(); it != population.end(); ++it) {

		pro_sum = pro_sum + it->crossover_probability;

		if (rand_num < pro_sum) {

			mother = it->id;
			break;
		}
	}

	while (1) {

		rand_num = rand() % 1000000;
		rand_num = rand_num / 1000000;

		double pro_sum = 0.0;
		for (it = population.begin(); it != population.end(); ++it) {
			pro_sum = pro_sum + it->crossover_probability;

			if (rand_num < pro_sum) {

				father = it->id;
				break;
			}
		}

		if (mother != father)
			break;

	}

}

void update_population_properties(std::vector<EAdesign> &population) {
	std::vector<EAdesign>::iterator it;
	double sum_fitness;
	double max, min;
	double x_std;
	sum_fitness = 0.0;
	max = -LARGE;
	min = LARGE;

	for (it = population.begin(); it != population.end(); ++it) {

		if (it->objective_val < min)
			min = it->objective_val;
		if (it->objective_val > max)
			max = it->objective_val;

	}


	for (it = population.begin(); it != population.end(); ++it) {

		x_std = (it->objective_val - min) / (max - min);
		it->fitness = x_std * 100.0;

		sum_fitness = sum_fitness + it->fitness;

	}

	for (it = population.begin(); it != population.end(); ++it) {


		it->crossover_probability = it->fitness / sum_fitness;
	}

}








