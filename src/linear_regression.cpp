#include "linear_regression.hpp"
#include "auxilliary_functions.hpp"
#include "Rodeo_macros.hpp"

#include <armadillo>

using namespace arma;


LinearModel::LinearModel(std::string name):SurrogateModel(name){

	modelID = LINEAR_REGRESSION;
	regularizationParam = 10E-6;


}

LinearModel::LinearModel():SurrogateModel(){


}


void LinearModel::initializeSurrogateModel(void){

	if(label != "None"){

		ReadDataAndNormalize();
		weights = zeros<vec>(dim+1);

	}

}

void LinearModel::saveHyperParameters(void) const  {

	weights.save(hyperparameters_filename, csv_ascii);

}

void LinearModel::loadHyperParameters(void){

	weights.load(hyperparameters_filename, csv_ascii);

}

void LinearModel::printHyperParameters(void) const{

	printVector(weights,"linear regression weights");

}





void LinearModel::setRegularizationParam(double value){

	regularizationParam = value;
}

double LinearModel::getRegularizationParam(void) const{

	return regularizationParam;
}




void LinearModel::train(void){

	if(ifInitialized){

		initializeSurrogateModel();
	}

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

		weights = psuedo_inverse_X_augmented * y;

	}

	else{
#if 0
		printf("Regularization...\n");
#endif
		mat XtX = trans(augmented_X)*augmented_X;

		XtX = XtX + regularizationParam*eye(XtX.n_rows,XtX.n_rows);

		weights = inv(XtX)*trans(augmented_X)*y;

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

	double inSampleError = 0;
	for(unsigned int i=0;i<N;i++){

		double fTilde = interpolate(X.row(i));

		inSampleError += (y(i) - fTilde)*(y(i) - fTilde);
#if 1
		printf("Sample %d:\n", i);
		printf("fExact = %15.10f, fTilde = %15.10f\n",y(i),fTilde);
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



void LinearModel::interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const{

	cout << "ERROR: interpolateWithVariance does not exist for LinearModel\n";
	abort();


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




void LinearModel::printSurrogateModel(void) const{

	cout << "Surrogate model:"<< endl;
	cout<< "Number of samples: "<<N<<endl;
	cout<<"Number of input parameters: "<<dim<<endl;
	cout<<"Raw Data:\n";
	rawData.print();
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















