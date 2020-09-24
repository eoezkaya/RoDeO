#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include <cassert>

#include "kriging_training.hpp"
#include "linear_regression.hpp"
#include "auxilliary_functions.hpp"
#include "random_functions.hpp"
#include "Rodeo_macros.hpp"

#include "Rodeo_globals.hpp"
#include "Rodeo_macros.hpp"


#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;

/* global variables */

int total_number_of_function_evals;

double population_overall_max = -10E14;
int population_overall_max_tread_id = -1;




KrigingModel::KrigingModel():SurrogateModel(){}


KrigingModel::KrigingModel(std::string name):SurrogateModel(name),linearModel(name){

	linear_regression = false;
	modelID = KRIGING;
	hyperparameters_filename = label + "_kriging_hyperparameters.csv";


}

void KrigingModel::initializeSurrogateModel(void){

	if(label != "None"){

		printf("Initializing settings for the Kriging model...\n");

		modelID = KRIGING;

		ReadDataAndNormalize();

		kriging_weights =zeros<vec>(2*dim);
		epsilonKriging = 10E-12;
		max_number_of_kriging_iterations = 20000;


		for(unsigned int i=0; i<N; i++){

			rowvec sample1 = X.row(i);

			for(unsigned int j=i+1; j<N; j++){

				rowvec sample2 = X.row(j);

				if(ifTooCLose(sample1, sample2)) {

					printf("ERROR: Two samples in the training data are too close to each other!\n");
					exit(-1);
				}
			}
		}

		sigmaSquared = 0.0;
		beta0 = 0.0;

		correlationMatrix = zeros<mat>(N,N);
		upperDiagonalMatrix= zeros<mat>(N,N);
		R_inv_ys_min_beta = zeros<vec>(N);
		R_inv_I= zeros<vec>(N);
		vectorOfOnes= ones<vec>(N);

		ifInitialized = true;

		std::cout << "Kriging model initialization is done...\n";

	}
}

void KrigingModel::printHyperParameters(void) const{

	printVector(kriging_weights,"kriging_weights");

}
void KrigingModel::saveHyperParameters(void) const{

	kriging_weights.save(hyperparameters_filename, csv_ascii);

}

void KrigingModel::loadHyperParameters(void){

	kriging_weights.load(hyperparameters_filename, csv_ascii);

}

double KrigingModel::getyMin(void) const{

	return ymin;

}






void KrigingModel::setNumberOfTrainingIterations(unsigned int iter){

	max_number_of_kriging_iterations = iter;

}


void KrigingModel::setEpsilon(double inp){

	assert(inp>0);
	epsilonKriging = inp;


}

void KrigingModel::setLinearRegressionOn(void){
	linear_regression = true;

}
void KrigingModel::setLinearRegressionOff(void){
	linear_regression = false;

}




/** Adds the rowvector newsample to the data of the Kriging model and updates model parameters
 * @param[in] newsample
 *
 */

int KrigingModel::addNewSampleToData(rowvec newsample){


	/* avoid points that are too close to each other */

	bool flagTooClose=false;
	for(unsigned int i=0; i<N; i++){

		rowvec sample = rawData.row(i);

		if(ifTooCLose(sample, newsample)) {

			flagTooClose = true;
		}

	}

	if(!flagTooClose){

		rawData.resize(N+1, dim+1);

		rawData.row(N) = newsample;
		rawData.save(input_filename,csv_ascii);

		updateWithNewData();
		return 0;
	}
	else{

		std::cout<<"Warning: the new sample is too close to a sample in the training data, it is discarded!\n";
		return -1;
	}

}


void KrigingModel::printSurrogateModel(void) const{
	cout << "\nKriging Surrogate model:\n";
	cout<< "Number of samples: "<<N<<endl;
	cout<< "Number of input parameters: "<<dim<<"\n";
	printMatrix(rawData,"rawData");
	printMatrix(X,"X");


	printf("hyperparameters_filename: %s\n",hyperparameters_filename.c_str());
	printf("input_filename: %s\n",input_filename.c_str());
	printf("max_number_of_kriging_iterations = %d\n",max_number_of_kriging_iterations);
	printf("epsilonKriging = %15.10e\n",epsilonKriging);
	printVector(kriging_weights,"kriging_weights");
	printf("\n");



}
void KrigingModel::updateModelParams(void){

	vec ys = y;

	if(linear_regression){

		vec ysLinearRegression = linearModel.interpolateAll(X);
		ys = ys - ysLinearRegression;

	}

	computeCorrelationMatrix();

	/* Cholesky decomposition R = LDL^T */


	int cholesky_return = chol(upperDiagonalMatrix, correlationMatrix);

	if (cholesky_return == 0) {
		printf("ERROR: Ill conditioned correlation matrix, Cholesky decomposition failed at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);
	}


	vec R_inv_ys(N); R_inv_ys.fill(0.0);


	solveLinearSystemCholesky(upperDiagonalMatrix, R_inv_ys, ys);    /* solve R x = ys */

	R_inv_I = zeros(N);

	solveLinearSystemCholesky(upperDiagonalMatrix, R_inv_I, vectorOfOnes);      /* solve R x = I */


	beta0 = (1.0/dot(vectorOfOnes,R_inv_I)) * (dot(vectorOfOnes,R_inv_ys));

	vec ys_min_betaI = ys - beta0*vectorOfOnes;



	/* solve R x = ys-beta0*I */
	solveLinearSystemCholesky(upperDiagonalMatrix, R_inv_ys_min_beta , ys_min_betaI);


	sigmaSquared = (1.0 / N) * dot(ys_min_betaI, R_inv_ys_min_beta);


}

void KrigingModel::updateWithNewData(void){

	correlationMatrix.reset();
	upperDiagonalMatrix.reset();
	R_inv_I.reset();
	R_inv_ys_min_beta.reset();
	X.reset();
	rawData.reset();
	vectorOfOnes.reset();


	beta0 = 0.0;
	sigmaSquared = 0.0;



	/* data matrix input */

	bool status = rawData.load(input_filename.c_str(), csv_ascii);
	if(status == true)
	{
		printf("Data input is done\n");
	}
	else
	{
		printf("Problem with data the input (cvs ascii format) at %s, line %d.\n", __FILE__, __LINE__);
		exit(-1);
	}

	assert(dim == rawData.n_cols - 1);

	N = rawData.n_rows;
#if 1
	printf("%s model has now %d training samples\n",label.c_str(),N);
#endif
	X = rawData.submat(0, 0, N - 1, dim - 1);

	for (unsigned int i = 0; i < dim; i++) {

		xmax(i) = rawData.col(i).max();
		xmin(i) = rawData.col(i).min();

	}

	/* normalize input matrix */

	for (unsigned int i = 0; i < N; i++) {

		for (unsigned int j = 0; j < dim; j++) {

			X(i, j) = (1.0/dim)*(X(i, j) - xmin(j)) / (xmax(j) - xmin(j));
		}
	}

#if 0
	printf("Normalized data = \n");
	X.print();
#endif


	correlationMatrix.set_size(N,N);
	correlationMatrix.fill(0.0);
	upperDiagonalMatrix.set_size(N,N);
	upperDiagonalMatrix.fill(0.0);
	R_inv_ys_min_beta.set_size(N);
	R_inv_ys_min_beta.fill(0.0);
	R_inv_I.set_size(N);
	R_inv_I.fill(0.0);
	vectorOfOnes.set_size(N);
	vectorOfOnes.fill(1.0);

	ymin = min(rawData.col(dim));
	ymax = max(rawData.col(dim));
	yave = mean(rawData.col(dim));

#if 1
	printf("ymin = %15.10f, ymax = %15.10f, yave = %15.10f\n",ymin,ymax,yave);
#endif
	updateModelParams();

}

vec KrigingModel::computeCorrelationVector(rowvec x) const{


	vec r(N);

	vec theta = kriging_weights.head(dim);
	vec gamma = kriging_weights.tail(dim);

	for(unsigned int i=0;i<N;i++){

		r(i) = computeCorrelation(x, X.row(i), theta, gamma);

	}
	return r;

}

/*
 * Evaluates the surrogate model at x = xp
 * @param[in] xp (normalized)
 * @return y=ftilde(xp)
 *
 * */



double KrigingModel::interpolate(rowvec xp) const{


	double fLinearRegression = 0.0;
	double fKriging = 0.0;


	if(linear_regression){

		fLinearRegression = linearModel.interpolate(xp);
	}


	vec r = computeCorrelationVector(xp);

	fKriging = beta0+ dot(r,R_inv_ys_min_beta);

	return fLinearRegression+fKriging;;

}

/*
 * Evaluates the expected improvement value surrogate model at x = xp
 * @param[in] xp (normalized)
 * @return y=EI(xp)
 *
 * */


double KrigingModel::calculateExpectedImprovement(rowvec xp){



	double ftilde = 0.0;
	double ssqr   = 0.0;

	interpolateWithVariance(xp,&ftilde,&ssqr);

#if 0
	printf("ftilde = %15.10f, ssqr = %15.10f\n",ftilde,ssqr);
#endif

	double	sigma = sqrt(ssqr)	;

#if 0
	printf("standart_ERROR = %15.10f\n",sigma);
#endif

	double EI = 0.0;


	if(sigma!=0.0){


		double	Z = (ymin - ftilde)/sigma;
#if 0
		printf("EIfac = %15.10f\n",EIfac);
		printf("ymin = %15.10f\n",ymin);
#endif


		/* calculate the Expected Improvement value */
		EI = (ymin - ftilde)*cdf(Z,0.0,1.0)+ sigma * pdf(Z,0.0,1.0);
	}
	else{

		EI = 0.0;

	}
#if 0
	printf("EI = %15.10f\n",EI);
#endif
	return EI;

}


double KrigingModel::interpolateWithGradients(rowvec xp) const{

	cout << "ERROR: interpolateWithGradients does not exist for KrigingModel!\n";
	abort();


}


void KrigingModel::interpolateWithVariance(rowvec xp,double *ftildeOutput,double *sSqrOutput) const{

	*ftildeOutput =  interpolate(xp);

	vec R_inv_r(N);

	vec r = computeCorrelationVector(xp);

	/* solve the linear system R x = r by Cholesky matrices U and L*/
	solveLinearSystemCholesky(upperDiagonalMatrix, R_inv_r, r);


	*sSqrOutput = sigmaSquared*( 1.0 - dot(r,R_inv_r)+ ( pow( (dot(r,R_inv_I) -1.0 ),2.0)) / (dot(vectorOfOnes,R_inv_I) ) );


}


double KrigingModel::computeCorrelation(rowvec x_i, rowvec x_j, vec theta, vec gamma) const {


	double sum = 0.0;
	for (unsigned int k = 0; k < dim; k++) {

		sum += theta(k) * pow(fabs(x_i(k) - x_j(k)), gamma(k));

	}

	return exp(-sum);
}


void KrigingModel::computeCorrelationMatrix(void)  {

	vec theta = kriging_weights.head(dim);
	vec gamma = kriging_weights.tail(dim);

	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = i + 1; j < N; j++) {

			double corrVal = computeCorrelation(X.row(i), X.row(j), theta, gamma);
			correlationMatrix(i, j) = corrVal;
			correlationMatrix(j, i) = corrVal;
		}

	}

	correlationMatrix = correlationMatrix + eye(N,N) + eye(N,N)*epsilonKriging;

} /* end of compute_R_matrix */



void KrigingModel::train(void){

	if(ifInitialized == false){

		printf("ERROR: Kriging model must be initialized before training!\n");
		abort();
	}

	printf("\nTraining Kriging response surface for the data : %s\n",input_filename.c_str());

	vec ysKriging = y;


	if(linear_regression){
		printf("Linear regression is active...\n");

		linearModel.train();

		vec ysLinearModel = linearModel.interpolateAll(X);

		ysKriging = ysKriging - ysLinearModel;
	}
	else{

		printf("Linear regression is not active...\n");
	}

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

	if(number_of_initial_population < 100) {

		number_of_initial_population = 100;
	}


#pragma omp parallel
	{

		int tid;

		std::vector<EAdesign>::iterator it;
		std::vector<EAdesign> population;
		int population_size = 0;

		double population_max = -LARGE;
		int population_max_index = -1;


		mat Xmatrix = X;


		vec Ivector = vectorOfOnes;
		vec ys = ysKriging;

		unsigned int d = dim;
		double epsilon = epsilonKriging;


		double Jnormalization_factor = 0.0;



		EAdesign initial_design(d);

		initial_design.theta = generateRandomVector(0.0, 10.0, d);
		initial_design.gamma = generateRandomVector(0.0, 2.0, d);

		if(file_exist(hyperparameters_filename.c_str())){
#pragma omp master
			{
				printf("Hyperparameter file: %s exists...\n",hyperparameters_filename.c_str() );
			}


			vec weights_read_from_file;

			bool status = weights_read_from_file.load(hyperparameters_filename.c_str(),csv_ascii);

			if(status == false)
			{
				fprintf(stderr, "ERROR: Some problem with the hyperparameter input! at %s, line %d.\n",__FILE__, __LINE__);
				exit(-1);

			}

#if 1
#pragma omp master
			{
				printf("hyperparameters read from the file (theta; gamma):\n");
				printVector(weights_read_from_file);
			}
#endif

			if(weights_read_from_file.size() != 2*d){
#if 1
#pragma omp master
				{
					printf("Warning: hyper parameters read from the file do not match the problem dimensions!\n");
				}
#endif

			}

			else{

				for(unsigned int i=0; i<d;i++) {

					if(weights_read_from_file(i) < 100.0){

						initial_design.theta(i) = weights_read_from_file(i);
					}

				}
				for(unsigned int i=d; i<2*d;i++) {

					if( weights_read_from_file(i) < 2.0) {

						initial_design.gamma(i-d) = weights_read_from_file(i);

					}
				}

			}


		}
#if 1
#pragma omp master
		{
			printf("Initial design:\n");
			printVector(initial_design.theta,"theta");
			printVector(initial_design.gamma,"gamma");
		}
#endif

		initial_design.calculate_fitness(epsilon,Xmatrix,ys);


		if (initial_design.objective_val != -LARGE) {

			Jnormalization_factor = initial_design.objective_val;
			initial_design.objective_val = 0.0;
			initial_design.id = population_size;
			population.push_back(initial_design);
			population_size++;

		}

		for (int i = 0; i < max_number_of_function_calculations; i++) {

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

					new_born.theta(j) = generateRandomDouble(0, 10); //theta
					new_born.gamma(j) = generateRandomDouble(0, 2); //gamma

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


			kriging_weights.save(hyperparameters_filename.c_str(),csv_ascii);

		}

	} /* end of parallel section */

	printf("Kring training is done\n");
	printf("Kriging weights:\n");
	printVector(kriging_weights);


	updateModelParams();

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

	unsigned int N = X.n_rows;  /* number of samples */


	mat R(N,N,fill::zeros);
	mat U(N,N,fill::zeros);

	vec I(N,fill::ones);

	// compute the correlation matrix R

	//	double reg_param = pow(10.0,-1.0*new_born.log_regularization_parameter);

	compute_R_matrix(theta, gamma,epsilon,R,X);

#if 0
	R.raw_print(cout, "R:");
#endif


	// compute Cholesky decomposition
	int flag = chol(U, R);

	if (flag == 0) {

		printf("Ill conditioned correlation matrix! \n");
		objective_val = -LARGE;
		total_number_of_function_evals++;
		return 0;
	}



	vec U_diagonal = U.diag();
#if 0
	U_diagonal.raw_print(cout, "diag(U):");
#endif
	logdetR = 0.0;

	for (unsigned int i = 0; i < N; i++) {

		if (U_diagonal(i) < 0) {

			objective_val = -LARGE;
			total_number_of_function_evals++;
			return 0;
		}

		logdetR += log(U_diagonal(i));

		//		printf("%d %30.25f %30.25f\n",i,logdetR,log(U_diagonal(i)));
	}

	logdetR = 2.0 * logdetR;

	vec R_inv_ys(N,fill::zeros);
	vec R_inv_I(N,fill::zeros);

	solveLinearSystemCholesky(U, R_inv_ys, ys); /* solve R x = ys */
	solveLinearSystemCholesky(U, R_inv_I, I);   /* solve R x = I */



	beta0 = (1.0/dot(I,R_inv_I)) * (dot(I,R_inv_ys));
#if 0
	printf("beta0= %20.15f\n",beta0);
#endif


	vec ys_min_betaI = ys-beta0*I;

	vec R_inv_ys_min_beta(N,fill::zeros);


	/* solve R x = ys-beta0*I */
	solveLinearSystemCholesky(U, R_inv_ys_min_beta, ys_min_betaI);


	double fac = dot(ys_min_betaI, R_inv_ys_min_beta);

	double oneOverN = 1.0/double(N);
	double NoverTwo = double(N)/2.0;

	ssqr = oneOverN * fac;


	if(ssqr > 0 ){
		objVal = (- NoverTwo) * log(ssqr);
		objVal -= 0.5 * logdetR;
	}
	else{

		objective_val = -LARGE;
		total_number_of_function_evals++;
		return 0;
	}

#if 0
	printf("\n");
	printf("objective function value = %10.7f\n",objective_val);
	printf("s^2= %20.15f\n",ssqr);
	printf("(-N/2.0)* log(ssqr) = %10.7f\n",(-NoverTwo)* log(ssqr));
	printf("log(ssqr) = %10.7f\n",log(ssqr));
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







/* crossover function of two designs */
void crossover_kriging(EAdesign &father, EAdesign &mother, EAdesign &child) {

	int dim = father.theta.size();
	for (int i = 0; i < dim; i++) {
		//		printf("theta (m) = %10.7f theta (f) = %10.7f\n",mother.theta[i],father.theta[i]);
		//		printf("gamma (m) = %10.7f gamma (f) = %10.7f\n",mother.gamma[i],father.gamma[i]);
		child.theta(i) = generateRandomDoubleFromNormalDist(father.theta(i), mother.theta(i), 4.0);
		child.gamma(i) = generateRandomDoubleFromNormalDist(father.gamma(i), mother.gamma(i), 4.0);

		if (child.theta(i) < 0){
			child.theta(i) = generateRandomDouble(0.0, 10.0);
		}

		if (child.gamma(i) < 0){
			child.gamma(i) = generateRandomDouble(0.1, 2.0);
		}

		if (child.gamma(i) > 2.0){
			child.gamma(i) = generateRandomDouble(0.1, 2.0);
		}

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








