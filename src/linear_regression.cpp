#include "linear_regression.hpp"
#include "auxilliary_functions.hpp"
#include "Rodeo_macros.hpp"

#include <armadillo>

using namespace arma;


LinearModel::LinearModel(std::string name, unsigned int dimension):SurrogateModel(name,dimension){

	modelID = LINEAR_REGRESSION;
	weights = zeros(dim+1);
	regularizationParam = 10E-6;


}

LinearModel::LinearModel():SurrogateModel(){


}

void LinearModel::setRegularizationParam(double value){

	regularizationParam = value;
}

double LinearModel::getRegularizationParam(void) const{

	return regularizationParam;
}


void LinearModel::train(void){


	mat augmented_X(N, dim + 1);

	vec ys = data.col(dim);

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

#if 0
	printf("augmented_X:\n");
	augmented_X.print();
#endif



	if(fabs(regularizationParam) < EPSILON ){
#if 0
		printf("Taking pseudo-inverse of augmented data matrix...\n");
#endif
		mat psuedo_inverse_X_augmented = pinv(augmented_X);

		//		psuedo_inverse_X_augmented.print();

		weights = psuedo_inverse_X_augmented * ys;

	}

	else{
#if 0
		printf("Regularization...\n");
#endif
		mat XtX = trans(augmented_X)*augmented_X;

		XtX = XtX + regularizationParam*eye(XtX.n_rows,XtX.n_rows);

		weights = inv(XtX)*trans(augmented_X)*ys;

	}

	for(unsigned int i=0; i<dim+1;i++ ){

		if(fabs(weights(i)) > 10E5){

			printf("WARNING: Linear regression coefficients are too large= \n");
			printf("regression_weights(%d) = %10.7f\n",i,weights(i));
		}

	}
#if 1
	printf("Linear regression weights:\n");
	trans(weights).print();

#endif

}

double LinearModel::calculateInSampleError(void) const{

	printf("Calculating in-sample error for the linear regression...\n");
	vec ys = data.col(dim);

	double inSampleError = 0;
	for(unsigned int i=0;i<N;i++){

		double fTilde = interpolate(X.row(i));

		inSampleError += (ys(i) - fTilde)*(ys(i) - fTilde);
#if 1
		printf("Sample %d:\n", i);
		printf("fExact = %15.10f, fTilde = %15.10f\n",ys(i),fTilde);
#endif

	}

	inSampleError = inSampleError/N;

	return inSampleError;



}


double LinearModel::interpolate(rowvec x) const{

	double fRegression = 0.0;
	for(unsigned int i=0; i<dim; i++){

		fRegression += x(i)*weights(i+1);
	}

	/* add bias term */
	fRegression += weights(0);

	return fRegression;


}

vec LinearModel::interpolateAll(mat X) const{

	unsigned int numberOfSamples = X.n_rows;

	vec result(numberOfSamples);


	for(unsigned int i=0; i<numberOfSamples; i++){

		rowvec x = X.row(i);

		double fRegression = 0.0;
		for(unsigned int j=0; j<dim; j++){

			fRegression += x(j)*weights(j+1);
		}

		/* add bias term */
		fRegression += weights(0);

		result(i) = fRegression;
	}

	return result;


}




void LinearModel::print(void) const{

	cout << "Surrogate model:"<< endl;
	cout<< "Number of samples: "<<N<<endl;
	cout<<"Number of input parameters: "<<dim<<endl;
	cout<<"Raw Data:\n";
	data.print();
	cout<<"xmin =";
	trans(xmin).print();
	cout<<"xmax =";
	trans(xmax).print();
	cout<<"ymin = "<<ymin<<endl;
	cout<<"ymax = "<<ymax<<endl;
	cout<<"ymean = "<<yave<<endl;
	cout<<"Regression weights:\n";
	trans(weights).print();

}


/** compute the coefficients of the linear regression
 *
 * @param[in] X  data matrix
 * @param[in] ys output values
 * @param[out] w regression weights (w0,w1,...,Wd)
 * @param[in] lambda parameter of Tikhonov regularization
 */

void train_linear_regression(mat &X, vec &ys, vec &w, double lambda=0){

	unsigned int dim = X.n_cols;

	mat augmented_X(X.n_rows, dim + 1);

	for (unsigned int i = 0; i < X.n_rows; i++) {

		for (unsigned int j = 0; j <= dim; j++) {

			if (j == 0){

				augmented_X(i, j) = 1.0;
			}

			else{

				augmented_X(i, j) = X(i, j - 1);
			}


		}
	}

	if(fabs(lambda) < EPSILON ){
#if 0
		printf("Taking pseudo-inverse of augmented data matrix...\n");
#endif
		mat psuedo_inverse_X_augmented = pinv(augmented_X);

		//		psuedo_inverse_X_augmented.print();

		w = psuedo_inverse_X_augmented * ys;

	}

	else{
#if 0
		printf("Regularization...\n");
#endif
		mat XtX = trans(augmented_X)*augmented_X;

		XtX = XtX + lambda*eye(XtX.n_rows,XtX.n_rows);

		w = inv(XtX)*trans(augmented_X)*ys;

	}

	for(unsigned int i=0; i<w.size();i++ ){

		if(fabs(w(i)) > 10E5){

			printf("WARNING: Linear regression coefficients are too large= \n");
			printf("regression_weights(%d) = %10.7f\n",i,w(i));
		}

	}



} 












