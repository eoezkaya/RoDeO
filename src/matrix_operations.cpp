#include <cassert>

#include "matrix_operations.hpp"


#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


rowvec normalizeRowVector(rowvec x, vec xmin, vec xmax){

	unsigned int dim = x.size();
	rowvec xnorm(dim);

	for(unsigned int i=0; i<dim; i++){

		xnorm(i) = (1.0/dim)*(x(i) - xmin(i)) / (xmax(i) - xmin(i));

	}

	return xnorm;

}

rowvec normalizeRowVectorBack(rowvec xnorm, vec xmin, vec xmax){

	unsigned int dim = xnorm.size();
	rowvec xp(dim);

	for(unsigned int i=0; i<dim; i++){

		assert(xmax(i) > xmin(i));
		xp(i) = xnorm(i)*dim * (xmax(i) - xmin(i)) + xmin(i);

	}
	return xp;
}

mat normalizeMatrix(mat matrixIn, vec xmin, vec xmax){

	unsigned int dim = matrixIn.n_cols;
	unsigned int N=matrixIn.n_rows;

	mat matrixOut(N,dim,fill::zeros);

	vec deltax = xmax-xmin;

	for(unsigned int i=0; i<N;i++){

		for(unsigned int j=0; j<dim;j++){

			assert(deltax(j) > 0);
			matrixOut(i,j)  = ((1.0/dim)*(matrixIn(i,j)-xmin(j)))/(deltax(j));

		}

	}


	return matrixOut;

}

mat normalizeMatrix(mat matrixIn, double xmin, double xmax){

	unsigned int dim = matrixIn.n_cols;
	vec xminTemp(dim); xminTemp.fill(xmin);
	vec xmaxTemp(dim); xmaxTemp.fill(xmin);

	mat matrixOut = normalizeMatrix(matrixIn, xminTemp, xmaxTemp);
	return matrixOut;

}

