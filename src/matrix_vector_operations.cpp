/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2020 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, TU Kaiserslautern)
 *
 * This file is part of RoDeO
 *
 * RoDeO is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * RoDeO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU
 * General Public License along with CoDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, TU Kaiserslautern)
 *
 *
 *
 */




#include "matrix_vector_operations.hpp"
#include "auxiliary_functions.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


HighPrecisionMatrix factorizeCholeskyHP(const HighPrecisionMatrix& input) {

	int n = input.n_rows();
	HighPrecisionMatrix result(n, n);
	for (int i = 0; i < n; ++i) {
		for (int k = 0; k < i; ++k) {
			__float128 value = input(i, k);
			for (int j = 0; j < k; ++j)
				value -= result(i, j) * result(k, j);
			result(i, k) = value/result(k, k);
		}
		__float128 value = input(i, i);
		for (int j = 0; j < i; ++j)
			value -= result(i, j) * result(i, j);

		if(value < 0.0){
			cout<<"ERROR: Cholesky decomposition failed!\n";
			abort();

		}
		result(i, i) = sqrtq(value);
	}
	return result;
}


HighPrecisionMatrix transposeHP(const HighPrecisionMatrix& input) {

	int n = input.n_rows();
	HighPrecisionMatrix result(n, n);

	for (int i = 0; i < n; ++i) {

		for (int j = i; j < n; ++j){

			result(i,j) = input(j,i);


		}


	}


	return result;
}

/** solve a linear system Ax=b, with a given Cholesky decomposition of A
 *
 * @param[in] U
 * @param[out] x
 * @param[in] b
 *
 */
//void solveLinearSystemCholesky(const HighPrecisionMatrix& L, vec &x, vec b){
//
//	int n = L.n_rows();
//	HighPrecisionMatrix U(n,n);
//
//	mat L = trans(U);
//
//	unsigned int dim = x.size();
//
//	if(dim != U.n_rows || dim != b.size()){
//
//		fprintf(stderr, "Error: dimensions does not match! at %s, line %d.\n",__FILE__, __LINE__);
//		abort();
//
//	}
//	/* initialize x */
//
//	x.fill(0.0);
//
//	vec y(dim,fill::zeros);
//
//	/* forward subst. L y = b */
//
//	for (unsigned int i = 0; i < dim; i++) {
//
//		double residual = 0.0;
//
//		for (unsigned int j = 0; j < i; j++) {
//
//			residual = residual + L(i, j) * y(j);
//
//		}
//
//		y(i) = (b(i) - residual) / L(i, i);
//	}
//
//
//	/* back subst. U x = y */
//
//	for (int i = dim - 1; i >= 0; i--) {
//
//		double residual = 0.0;
//
//		for (int j = dim - 1; j > i; j--){
//
//			residual += U(i, j) * x(j);
//		}
//
//
//		x(i) = (y(i) - residual) / U(i, i);
//	}
//
//}

void copyRowVector(rowvec &a,rowvec b){

	assert(a.size() >= b.size());

	for(unsigned int i=0; i<b.size(); i++){

		a(i) = b(i);
	}


}

void copyRowVector(rowvec &a,rowvec b, unsigned int indx){

	assert(a.size() >= b.size() + indx);

	for(unsigned int i=indx; i<b.size() + indx; i++){

		a(i) = b(i-indx);
	}


}

void appendRowVectorToCSVData(rowvec v, std::string fileName){


	std::ofstream outfile;

	outfile.open(fileName, std::ios_base::app); // append instead of overwrite

	outfile.precision(10);
	for(unsigned int i=0; i<v.size(); i++){

		outfile << v(i) <<",";
	}

	outfile << "\n";

	outfile.close();

}


void testHighPrecisionCholesky(void){

	int N = 10;
	HighPrecisionMatrix  A(N, N);

	mat L(N,N,fill::randu);

	mat M = L*trans(L);

	printMatrix(M,"M");

	A = M;

	HighPrecisionMatrix Lchol = factorizeCholeskyHP(A);
	Lchol.print();

	HighPrecisionMatrix Uchol = transposeHP(Lchol);
	Uchol.print();
}


void printMatrix(mat M, std::string name){


	std::cout<< '\n';
	if(name != "None") std::cout<<name<<": "<<M.n_rows<<"x"<<M.n_cols<<"matrix\n";
	M.print();
	std::cout<< '\n';

}

void printVector(vec v, std::string name){

	std::cout.precision(11);
	std::cout.setf(ios::fixed);

	std::cout<< '\n';
	if(name != "None") std::cout<<"vector "<<name<<" has "<<v.size()<<" elements:\n";
	trans(v).raw_print();
	std::cout<< '\n';

}

void printVector(rowvec v, std::string name){

	std::cout.precision(11);
	std::cout.setf(ios::fixed);

	std::cout<< '\n';
	if(name != "None") std::cout<<"vector "<<name<<" has "<<v.size()<<" elements:\n";
	v.raw_print();
	std::cout<< '\n';

}



void printVector(std::vector<std::string> v){

	for (std::vector<std::string>::const_iterator i = v.begin(); i != v.end(); ++i){
		std::cout << *i << ' ';
	}
	std::cout<<"\n";


}

void printVector(std::vector<int> v){

	for (std::vector<int>::const_iterator i = v.begin(); i != v.end(); ++i){
		std::cout << *i << ' ';
	}
	std::cout<<"\n";


}

void printVector(std::vector<bool> v){

	for (std::vector<bool>::const_iterator i = v.begin(); i != v.end(); ++i){
		std::cout << *i << ' ';
	}
	std::cout<<"\n";


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

mat normalizeMatrix(mat matrixIn, vec xmin, vec xmax){

	unsigned int dim = matrixIn.n_cols;
	unsigned int N=matrixIn.n_rows;

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




