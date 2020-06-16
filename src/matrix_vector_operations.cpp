#include <cassert>

#include "matrix_vector_operations.hpp"
#include "auxilliary_functions.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


void printMatrix(mat M, std::string name){


	std::cout<< '\n';
	if(name != "None") std::cout<<name<<": "<<M.n_rows<<"x"<<M.n_cols<<"matrix\n";
	M.print();
	std::cout<< '\n';

}

void printVector(vec v, std::string name){


	std::cout<< '\n';
	if(name != "None") std::cout<<"vector "<<name<<" has "<<v.size()<<" elements:\n";
	trans(v).print();
	std::cout<< '\n';

}

void printVector(rowvec v, std::string name){


	std::cout<< '\n';
	if(name != "None") std::cout<<"vector "<<name<<" has "<<v.size()<<" elements:\n";
	v.print();
	std::cout<< '\n';

}

vec normalizeColumnVector(vec x, double xmin, double xmax){

	return ( (xmax-xmin)*x+xmin );


}

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

mat normalizeMatrix(mat matrixIn){

	unsigned int dim = matrixIn.n_cols;
	unsigned int N=matrixIn.n_rows;

	vec xmin(dim);
	vec xmax(dim);

	for (unsigned int i = 0; i < dim; i++) {

		xmax(i) = matrixIn.col(i).max();
		xmin(i) = matrixIn.col(i).min();

	}


	mat matrixOut(N,dim,fill::zeros);

	vec deltax = xmax-xmin;

	for(unsigned int i=0; i<N;i++){

		for(unsigned int j=0; j<dim;j++){

			assert(deltax(j) > 0);

			matrixOut(i,j)  = ((matrixIn(i,j)-xmin(j)))/(deltax(j));

		}

	}


	return matrixOut;

}

bool isEqual(const mat &A, const mat&B, double tolerance){

	unsigned int m = A.n_rows;
	unsigned int n = A.n_cols;

	assert(m==B.n_rows && n==B.n_cols);

	for(unsigned int i=0; i<m; i++){

		for(unsigned int j=0; j<m; j++){

			if(checkValue(A(i,j),B(i,j), tolerance) == false) return false;

		}

	}

	return true;
}


bool checkIfSymmetricPositiveDefinite(const mat &M){

	mat R = M;
	return chol(R,M);

}



bool checkIfSymmetric(const mat &M){

	mat MT = trans(M);

	if(isEqual(M, MT, 10E-10)) return true;
	else return false;


}
