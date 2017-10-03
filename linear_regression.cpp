#include "linear_regression.hpp"
#include "Rodeo_macros.hpp"

#include <armadillo>

using namespace arma;

/** compute the coefficients of the linear regression
 *
 * @param[in] X  data matrix
 * @param[in] ys output values
 * @param[out] w regression weights (w0,w1,...,Wd)
 * @param[in] lambda parameter of Tikhonov regularization
 * @return 0 if successful
 */

int train_linear_regression(mat &X, vec &ys, vec &w, double lambda=0){

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


	return 0;
} 














