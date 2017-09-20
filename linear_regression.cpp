#include "linear_regression.hpp"


#include <armadillo>

using namespace arma;



/*
 * IN : vector of functional values ys
 * OUT: vector of linear regression weights with w(0) as the bias term
 *
 *
 */

int train_linear_regression(mat &X, vec &ys, vec &w, double lambda=0){


	unsigned int dim = X.n_cols;

	mat augmented_X(X.n_rows, dim + 1);

	for (unsigned int i = 0; i < X.n_rows; i++) {
		for (unsigned int j = 0; j <= dim; j++) {
			if (j == 0)
				augmented_X(i, j) = 1.0;
			else
				augmented_X(i, j) = X(i, j - 1);

		}
	}


	if(fabs(lambda) < 10E-14 ){
		printf("Taking pseudo-inverse of augmented data matrix...\n");
		mat psuedo_inverse_X_augmented = pinv(augmented_X);

		//		psuedo_inverse_X_augmented.print();

		w = psuedo_inverse_X_augmented * ys;

	}

	else{
		printf("Regularization...\n");
		mat XtX = trans(augmented_X)*augmented_X;

		XtX = XtX + lambda*eye(XtX.n_rows,XtX.n_rows);


		w = inv(XtX)*trans(augmented_X)*ys;




	}


//	printf("Linear regression coefficients = \n");
//	w.print();




	return 0;
} 














