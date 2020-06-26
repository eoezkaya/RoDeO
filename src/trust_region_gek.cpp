
#include <armadillo>
#include <random>
#include <map>
using namespace arma;


#include "trust_region_gek.hpp"
#include "Rodeo_macros.hpp"
#include "kriging_training.hpp"
#include "test_functions.hpp"
#include "auxilliary_functions.hpp"

#ifdef GPU_VERSION
#include "kernel_regression_cuda.h"
#endif
#include "kernel_regression.hpp"

#include "Rodeo_globals.hpp"
#include "linear_regression.hpp"
#include "Rodeo_macros.hpp"



AggregationModel::AggregationModel(std::string name,int dimension){


	label = name;
	printf("Initializing settings for training %s data...\n",name.c_str());
	input_filename = name +".csv";
	kriging_hyperparameters_filename = name + "_Kriging_Hyperparameters.csv";
	visualizeKrigingValidation = false;
	visualizeKernelRegressionValidation = false;
	visualizeAggModelValidation = false;
	regression_weights.zeros(dimension);
	kriging_weights.zeros(2*dimension);
	M.zeros(dimension,dimension);
	rho =  0.0;
	sigma= 0.0;
	genErrorKriging = LARGE;
	genErrorKernelRegression = LARGE;
	genErrorAggModel = LARGE;

	epsilon_kriging = 10E-06;
	max_number_of_kriging_iterations = 10000;
	number_of_cv_iterations_rho = 10000;
	dim = dimension;
	validationset_input_filename = "None";
	number_of_cv_iterations = 10;
	linear_regression = true;


	bool status = data.load(input_filename.c_str(), csv_ascii);
	if(status == true)
	{
		printf("Data input is done\n");
	}
	else
	{
		printf("Problem with data the input (cvs ascii format) (line number %d in file %s)\n",__LINE__, __FILE__);
		exit(-1);
	}

	N = data.n_rows;

	/* default is full batch size */
	minibatchsize = N;

	X = data.submat(0, 0, N - 1, dim - 1);

	XnotNormalized = X;


	/* matrix that holds the gradient data (nrows= N-1, ncols = dim)*/

	grad = data.submat(0,dim+1, N-1, 2*dim);

	xmax = zeros(dim);
	xmin = zeros(dim);


	/* get the maximum and minimum of each column of the data matrix */

	for (unsigned int i = 0; i < dim; i++) {

		xmax(i) = data.col(i).max();
		xmin(i) = data.col(i).min();

	}

	/* normalize the data matrix */

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

/** updates the Aggregation model when the data changes
 *
 */

void AggregationModel::update(void){


	X.reset();
	data.reset();
	grad.reset();
	XnotNormalized.reset();


	/* data matrix input */

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

#if 0
	printf("Data matrix has %d columns and %d rows:\n",N,data.n_cols );
	data.print();
#endif



	/* update number of samples */
	N = data.n_rows; /* update number of samples */

	/* update the input matrix */
	X = data.submat(0, 0, N-1, dim-1);
#if 0
	printf("X:\n");
	X.print();
#endif


	grad = data.submat(0,dim+1, N-1, 2*dim);

	/* find minimum and maximum of the columns of data */


	for (unsigned int i = 0; i < dim; i++) {
		xmax(i) = data.col(i).max();
		xmin(i) = data.col(i).min();

	}

#if 0
	printf("maximum = \n");
	xmax.print();

	printf("minimum = \n");
	xmin.print();
#endif

	XnotNormalized = X;


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


	ymin = min(data.col(dim));
	ymax = max(data.col(dim));
	yave = mean(data.col(dim));


	updateKrigingModel();



}

void AggregationModel::updateKrigingModel(void){

#if 1
	printf("\nUpdating the Kriging model...\n");
#endif

	R.reset();
	L.reset();
	U.reset();
	R_inv_ys_min_beta.reset();
	R_inv_I.reset();
	I.reset();

	R.zeros(N,N);
	L.zeros(N,N);
	U.zeros(N,N);
	R_inv_ys_min_beta.zeros(N);
	R_inv_I.zeros(N);
	I.ones(N);

	if(N != X.n_rows){

		fprintf(stderr, "Error: Dimension N does not match with the size of the input matrix X! at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);

	}

	/* load Kriging hyperparameters */
	vec theta = kriging_weights.head(dim);
	vec gamma = kriging_weights.tail(dim);

	/* compute the correlation matrix R */
	compute_R_matrix(theta,
			gamma,
			epsilon_kriging,
			R,
			X);
#if 0
	printf("R =\n");
	R.print();
#endif

	vec ys = data.col(dim);



	if(linear_regression){

		mat augmented_X(N, dim + 1);

		for (unsigned int i = 0; i < N; i++) {

			for (unsigned int j = 0; j <= dim; j++) {

				if (j == 0){

					augmented_X(i, j) = 1.0;
				}
				else{

					augmented_X(i, j) = X(i, j - 1);

				}
			}
		}

		ys = ys - augmented_X * regression_weights;
	}


#if 0
	printf("ys =\n");
	trans(ys).print();
#endif


	/* Cholesky decomposition R = LDL^T */

	int cholesky_return = chol(U, R);

	if (cholesky_return == 0) {
		printf("Error: Ill conditioned correlation matrix, Cholesky decomposition failed at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);
	}



	vec R_inv_ys(N);
	solveLinearSystemCholesky(U, R_inv_ys, ys);    /* solve R x = ys */
#if 0
	printf("R_inv_ys =\n");
	trans(R_inv_ys).print();
#endif

	solveLinearSystemCholesky(U, R_inv_I, I);      /* solve R x = I */
#if 0
	printf("R_inv_I =\n");
	trans(R_inv_I).print();
#endif

	beta0 = (1.0/dot(I,R_inv_I)) * (dot(I,R_inv_ys));


	vec ys_min_betaI = ys - beta0*I;

	/* solve R x = ys-beta0*I */
	solveLinearSystemCholesky(U, R_inv_ys_min_beta , ys_min_betaI);


	sigma_sqr = (1.0 / N) * dot(ys_min_betaI, R_inv_ys_min_beta);

#if 1
	printf("Kriging model has been updated...\n");
#endif


}






double AggregationModel::ftildeKriging(rowvec xp){

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


	return f_regression+f_kriging;

}


double AggregationModel::ftilde(rowvec xp){

	if(sigma <= 0){

		fprintf(stderr, "Error: sigma must be positive! at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);


	}
	vec ys = data.col(dim);

	rowvec xpNotNormalized(dim);

	for(unsigned int i=0; i<dim; i++){

		xpNotNormalized(i) = xp(i)*dim * (xmax(i) - xmin(i)) + xmin(i);
	}


#if 0
	printf("xp = \n");
	xp.print();
	printf("xp (not normalized)= \n");
	xpNotNormalized.print();

#endif

	double fKriging = ftildeKriging(xp);


#if 0
	printf("ftilde (Kriging) = %10.7f\n",fKriging);
#endif


	vec kernelVal(N);
	vec weight(N);

	double kernelSum = 0.0;


	/* first evaluate the kernel sum */
	for (unsigned int i = 0; i < N; i++) {

		rowvec xi = X.row(i);
#if 0
		printf("xi:\n");
		xi.print();
#endif

		kernelVal(i) = gaussianKernel(xi, xp, sigma, M);
#if 0
		printf("kernelVal(%d) = %10.7f\n",i,kernelVal(i));
#endif
		kernelSum += kernelVal(i);
	}



	double fKernelRegression = 0.0;

	for (unsigned int i = 0; i < N; i++) {


		rowvec xi = XnotNormalized.row(i);

		rowvec xdiff = xpNotNormalized - xi;

		double gradTerm = dot(xdiff,grad.row(i));

		weight(i) = kernelVal(i) / kernelSum;
		fKernelRegression += (ys(i) + gradTerm) * weight(i);
#if 0
		printf("y(%d) * weight(%d) = %10.7f * %10.7f\n",i,i,ys(i),weight(i) );
#endif
	}


#if 0
	printf("ftilde (Kernel Regression with gradient) = %10.7f\n",fKernelRegression);
#endif

	double min_dist=0.0;
	int indx;

	/* find the closest seeding point to the xp in the data set */
	findKNeighbours(X,xp,1,&min_dist,&indx,1);


	rowvec xg = XnotNormalized.row(indx);

#if 0
	printf("xg = \n");
	xg.print();
#endif

	rowvec xdiff = xpNotNormalized-xg;
	double distance = L1norm(xdiff, dim);

#if 0
	printf("L1 distance between xp and xg = %10.7f\n",distance);
#endif


	rowvec gradVec = grad.row(indx);

#if 0
	printf("grad = \n");
	gradVec.print();
#endif

	double normgrad = L1norm(gradVec, dim);

#if 0
	printf("L1 norm of grad = %10.7f\n",normgrad);
#endif


	double w1 = exp(-rho *distance*normgrad);

#if 0
	printf("w1 = %10.7f\n",w1);
#endif


	double fval = w1*fKernelRegression + (1.0-w1)*fKriging;

	return fval;
}




void AggregationModel::ftilde_and_ssqr(
		rowvec xp,
		double *f_tilde,
		double *ssqr){



	vec r(N);

	vec theta = kriging_weights.head(dim);
	vec gamma = kriging_weights.tail(dim);


	for(unsigned int i=0; i<N; i++){

		r(i) = compute_R(xp, X.row(i), theta, gamma);

	}


	*f_tilde =  ftilde(xp);

	vec R_inv_r(N);


	/* solve the linear system R x = r by Cholesky matrices U and L*/
	solveLinearSystemCholesky(U, R_inv_r, r);

	/* vector of ones */
	vec I = ones(N);

	*ssqr = sigma_sqr*( 1.0 - dot(r,R_inv_r)+ ( pow( (dot(r,R_inv_I) -1.0 ),2.0)) / (dot(I,R_inv_I) ) );



}











void AggregationModel::load_state(void){

	double temp;
	std::string filename = label+"_settings_save.dat";
	FILE *inp = fopen(filename.c_str(),"r");

	for(unsigned int i=0; i<dim; i++){

		fscanf(inp,"%lf",&temp);
		regression_weights(i) = temp;
	}
	printf("regression_weights =\n");
	regression_weights.print();

	for(unsigned int i=0; i<2*dim; i++){

		fscanf(inp,"%lf",&temp);
		kriging_weights(i) = temp;
	}
	printf("kriging_weights =\n");
	kriging_weights.print();



	for(unsigned int i=0; i<dim; i++)
		for(unsigned int j=0; j<dim; j++){

			fscanf(inp,"%lf",&temp);
			M(i,j) = temp;
		}

	printf("M =\n");
	M.print();



	fscanf(inp,"%lf",&temp);
	rho = temp;
	printf("rho = %15.10f\n",rho);

	fscanf(inp,"%lf",&temp);
	sigma = temp;
	printf("sigma = %15.10f\n",sigma);

	fscanf(inp,"%lf",&temp);
	beta0 = temp;
	printf("beta0 = %15.10f\n",beta0);

	fclose(inp);



}

void AggregationModel::save_state(void){


	std::string filename = label+"_settings_save.dat";
	FILE *outp = fopen(filename.c_str(),"w");

	for(unsigned int i=0; i<dim; i++){

		fprintf(outp,"%15.10f\n",regression_weights(i));
	}
	for(unsigned int i=0; i<2*dim; i++){

		fprintf(outp,"%15.10f\n",kriging_weights(i));
	}

	for(unsigned int i=0; i<dim; i++)
		for(unsigned int j=0; j<dim; j++){

			fprintf(outp,"%15.10f\n",M(i,j));

		}

	fprintf(outp,"%15.10f\n",rho);
	fprintf(outp,"%15.10f\n",sigma);
	fprintf(outp,"%15.10f\n",beta0);

	fclose(outp);

}

void AggregationModel::train(void) {

	printf("Training aggregation model for the data: %s...\n",input_filename.c_str());


	/* quit if no filename for the validation is specified */

	if (visualizeKrigingValidation || visualizeKernelRegressionValidation){

		if(validationset_input_filename == "None") {

			printf("File name for validation is not specified! (line number %d in file %s)\n", __LINE__, __FILE__);
			exit(-1);

		}

	}


	/* data matrix must have 2d+1 columns */
	if(data.n_cols != 2*dim +1 ) {

		printf("Input data dimension does not match! at line number %d in file %s\n", __LINE__, __FILE__);
		exit(-1);

	}


	/* This is the data matrix used for Kriging training */

	mat inputDataKriging = data.submat(0,0,N-1,dim);

	/* check dimensions of the data */


	mat validationData;
	mat Xvalidation;
	mat XvalidationNotNormalized;
	unsigned int Nval = 0;

	if(validationset_input_filename != "None"){


#if 1
		printf("Reading validation set %s...\n",validationset_input_filename.c_str());
#endif

		bool load_ok = validationData.load(validationset_input_filename.c_str());

		if(load_ok == false)
		{
			printf("problem with loading the file %s (line number %d in file %s)\n",validationset_input_filename.c_str(),__LINE__, __FILE__);
			exit(-1);
		}


#if 0
		printf("Validation data =\n");
		validationData.print();
#endif

		Nval = validationData.n_rows;
		Xvalidation = validationData.submat(0,0,Nval-1,dim-1);
		XvalidationNotNormalized = Xvalidation;


		/* normalize data X for validation*/

		for (unsigned int i = 0; i < Nval; i++) {

			for (unsigned int j = 0; j < dim; j++) {

				Xvalidation(i, j) = (1.0/dim)*(Xvalidation(i, j) - xmin(j)) / (xmax(j) - xmin(j));
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

	printf("Training the Kriging hyperparameters...\n");

//	KrigingModel  primalModel(std::string name, int dimension);
//
//
//	train_kriging_response_surface(inputDataKriging,
//			kriging_hyperparameters_filename ,
//			LINEAR_REGRESSION_ON,
//			regression_weights,
//			kriging_weights,
//			epsilon_kriging,
//			max_number_of_kriging_iterations);

	printf("Training of the Kriging hyperparameters is done...\n");

#if 1
	printf("Regression weights:\n");
	regression_weights.t().print();
	printf("Kriging weights:\n");
	kriging_weights.t().print();

#endif


	updateKrigingModel();

	mat visualizeKriging;

	if(visualizeKrigingValidation){


		genErrorKriging = 0.0;

		visualizeKriging = zeros(Nval,2);

		for(unsigned int i=0; i<Nval; i++){

			rowvec xp = Xvalidation.row(i);
#if 0
			printf("xp =\n");
			xp.print();
#endif
			double ftilde = ftildeKriging(xp);

			double fexact = validationData(i,dim);

			genErrorKriging += (fexact-ftilde)*(fexact-ftilde);
#if 0
			printf("ftilde (Kriging) = %10.7f, fexact = %10.7f\n",ftilde,fexact);
#endif
			if(visualizeKrigingValidation ){
				visualizeKriging(i,0) = fexact;
				visualizeKriging(i,1) = ftilde;
			}

		}

		genErrorKriging = genErrorKriging/Nval;


		printf("Generalization Error for Kriging (MSE) = %10.7f\n",genErrorKriging);
#if 1
		visualizeKriging.save("visualizeKriging.dat",raw_ascii);

		std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_1d_function_scatter.py visualizeKriging.dat visualizeKriging.png" ;
		python_command += " Kriging value prediction";

		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");
#endif
	}




	/* parameters for the kernel regression training */
	double wSvd = 0.0;
	double w12 = 0.01;


	/* lower diagonal matrix for the kernel regression part */
	mat LKernelRegression(dim,dim);
	mat L= zeros(dim,dim);

	mat inputDataKernelRegression = data.submat(0,0,N-1,2*dim);

#if 0
	inputDataKernelRegression.print();
#endif

	/* normalize functional values */

	float yTrainingMax = 1.0;



	mat XTrainingNotNormalized = data.submat(0,0,N-1,dim-1);
	vec yTrainingNotNormalized = data.col(dim);
	mat gradTrainingNotNormalized = data.submat(0,dim+1,N-1,2*dim);

	yTrainingMax = inputDataKernelRegression.col(dim).max();

	for (unsigned int i = 0; i < N; i++) {

		inputDataKernelRegression(i, dim) = inputDataKernelRegression(i, dim)/yTrainingMax ;

	}


	/* normalize training data */
	for (unsigned int i = 0; i < N; i++) {

		for (unsigned int j = 0; j < dim; j++) {

			inputDataKernelRegression(i, j) = (1.0/dim)*(inputDataKernelRegression(i, j) - xmin(j)) / (xmax(j) - xmin(j));
		}


	}


	if(number_of_cv_iterations > 0){

		printf("Training the Mahalanobis matrix and sigma with %d iterations...\n",number_of_cv_iterations);
		/* now train the Mahalanobis matrix */
//		trainMahalanobisDistance(LKernelRegression, inputDataKernelRegression,
//				sigma, wSvd, w12, number_of_cv_iterations,L2_LOSS_FUNCTION, minibatchsize,10000);
//



		mat Mtemp = LKernelRegression*trans(LKernelRegression);

		for(unsigned int i=0; i<dim; i++)
			for(unsigned int j=0; j<dim; j++) {

				M(i,j) = Mtemp(i,j);
			}

		printf("Saving Aggregation model settings...\n");
		save_state();



	}
	else{
		printf("Reading Aggregation model settings...\n");
		load_state();

	}





	mat visualizeKernelRegression;

	if(visualizeKernelRegressionValidation){

		visualizeKernelRegression = zeros(Nval,2);


		genErrorKernelRegression = 0.0;

		for(unsigned int i=0; i<Nval; i++){

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
					xmin,
					xmax,
					M,
					sigma);



			double fexact = validationData(i,dim);

			genErrorKernelRegression += (fKernelRegression-fexact)*(fKernelRegression-fexact);
#if 0
			printf("ftilde (Kernel Regression) = %10.7f, fexact = %10.7f\n",fKernelRegression,fexact);
#endif
			if(visualizeKernelRegressionValidation){
				visualizeKernelRegression(i,0) = fexact;
				visualizeKernelRegression(i,1) = fKernelRegression;
			}



		}

		genErrorKernelRegression = genErrorKernelRegression/Nval;

		printf("Generalization Error for Kernel Regregression (MSE) = %10.7f\n",genErrorKernelRegression);



#if 1
		visualizeKernelRegression.save("visualizeKernelRegression.dat",raw_ascii);

		std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_1d_function_scatter.py visualizeKernelRegression.dat visualizeKernelRegression.png" ;
		python_command += " Kernel_regression model value prediction";

		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");

#endif


	} /* end of validation part */



	/* training of the rho parameter */


	if(number_of_cv_iterations_rho != 0){

		/* obtain sample statistics */

		vec probe_distances_sample(N);

		for(unsigned int i=0; i<N; i++){

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

			rowvec gradVec= gradTrainingNotNormalized.row(i);


			double normgrad= L1norm(gradVec, dim);
#if 0
			printf("%dth entry at the data gradients:\n",i);
			grad.print();
			printf("norm of the gradient = %10.7e\n",normgrad);
#endif
			sumnormgrad+= normgrad;


		}

		double avg_norm_grad= sumnormgrad/N;
#if 0
		printf("average norm grad = %10.7e\n", avg_norm_grad);
#endif
		/* define the search range for the hyperparameter r */


		double min_r = 0.0;
		//	max_r = 1.0/dim;
		double max_r = -2*log(0.00001)/ (max(probe_distances_sample) * avg_norm_grad);
		//	max_r = 0.01;

		double dr;



		dr = (max_r - min_r)/(number_of_cv_iterations_rho);

#if 1
		printf("average distance in the training data= %10.7f\n",average_distance_sample);
		printf("min (sample)       = %10.7f\n",min(probe_distances_sample));
		printf("max (sample)       = %10.7f\n",max(probe_distances_sample));
		printf("max_r = %10.7f\n",max_r);
		printf("dr = %10.7f\n",dr);
		printf("factor at maximum distance at max r = %15.10f\n",exp(-max_r*max(probe_distances_sample)* avg_norm_grad));
#endif





		/* number of cross validation points */
		unsigned int NCVVal = N*0.2;
		unsigned int NCVTra = N - NCVVal;

#if 0
		printf("number of validation points (rho training) = %d\n", NCVVal);
		printf("number of rest points in the training data (rho training) = %d\n", NCVTra);
#endif



		mat inputDataCVval = data.submat(0,0,NCVVal-1,2*dim);
		mat inputDataCVtra = data.submat(NCVVal,0,N-1,2*dim);
		mat XCVval = X.submat(0,0,NCVVal-1,dim-1);
		mat XCVtra = X.submat(NCVVal,0,N-1,dim-1);

		mat XCVvalNotNormalized = XTrainingNotNormalized.submat(0,0,NCVVal-1,dim-1);
		mat XCVtraNotNormalized = XTrainingNotNormalized.submat(NCVVal,0,N-1,dim-1);
		vec yvalidation = inputDataCVval.col(dim);
		vec ytraining   = inputDataCVtra.col(dim);


		mat gradCVTraNotNormalized = inputDataCVtra.submat(0,dim+1,NCVTra-1,2*dim);
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

		/* save matrices for the complete data */
		vec R_inv_ys_min_beta_save = R_inv_ys_min_beta;
		mat Xsave = X;
		mat XnotNormalized_save = XnotNormalized;
		mat data_save = data;
		mat grad_save = grad;
		N = NCVTra;
		grad = gradCVTraNotNormalized;


		X = XCVtra;
		XnotNormalized = XCVtraNotNormalized;
		data = inputDataCVtra;

		updateKrigingModel();


		for(unsigned int rho_iter=0; rho_iter<number_of_cv_iterations_rho; rho_iter++){


			vec fAggVal(NCVVal);
			vec fExact(NCVVal);

			/* value of the hyper parameter to be tried */
			double rho_trial = min_r+(rho_iter)*dr;

			rho = rho_trial;



			double genError = 0.0;

			for(unsigned int i=0; i<NCVVal;i++){


				rowvec xp = XCVval.row(i);
#if 0
				printf("%d xp =\n",i);
				xp.print();
#endif

				fAggVal(i) = ftilde(xp);
				fExact(i) = yvalidation(i);

#if 0
				printf("ftilde (Agg. Model) = %10.7f, fexact = %10.7f\n",fAggModel,fexact);
#endif

				genError += (fAggVal(i) - fExact(i))*(fAggVal(i) - fExact(i));

			}

			genError = genError/NCVVal;




			if(genError < minGenErrorCVLoop){
#if 1
				printf("\nCV iteration = %d, rho = %10.7f, Gen. Error = %10.7f\n",rho_iter,rho_trial,genError);

				for(unsigned int j=0; j<NCVVal;j++){

					printf("Sample %d, fexact = %15.10f  fAgg = %15.10f\n",j,fExact(j),fAggVal(j));

				}

#endif
				minGenErrorCVLoop = genError;
				optimal_rho = rho_trial;

			}

		} /* end of rho loop */

#if 1
		printf("Optimal value of rho = %10.7f, Gen. Error = %10.7f\n",optimal_rho,minGenErrorCVLoop);
#endif

		rho = optimal_rho;

		X = Xsave;
		XnotNormalized = XnotNormalized_save;
		data = data_save;
		grad = grad_save;
		N = data.n_rows;

		updateKrigingModel();


	}

	mat visualizeAggModel;

	if(visualizeAggModelValidation){

		visualizeAggModel = zeros(Nval,2);


		genErrorAggModel = 0.0;

		for(unsigned int i=0; i<Nval; i++){


			rowvec xp = Xvalidation.row(i);
#if 0
			printf("xp =\n");
			xp.print();
#endif

			double fAggModel = ftilde(xp);

			double fexact = validationData(i,dim);

			genErrorAggModel += (fAggModel-fexact)*(fAggModel-fexact);

#if 1
			printf("Sample %d, ftilde (Agg. Model) = %10.7f, fexact = %10.7f\n",i,fAggModel,fexact);
#endif


			visualizeAggModel(i,0) = fexact;
			visualizeAggModel(i,1) = fAggModel;



		}


		genErrorAggModel = genErrorAggModel/Nval;

		printf("Generalization Error for the Agg. Model (MSE) = %10.7f\n",genErrorAggModel);



		visualizeAggModel.save("visualizeAggModel.dat",raw_ascii);

		std::string python_command = "python -W ignore "+ settings.python_dir + "/plot_1d_function_scatter.py visualizeAggModel.dat visualizeAggModel.png" ;
		python_command += " Aggregation_Model value prediction";

		FILE* in = popen(python_command.c_str(), "r");


		fprintf(in, "\n");





	}

	printf("Saving Aggregation model settings...\n");
	save_state();

}





