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

void abortIfHasNan(rowvec &v){

	if(v.has_nan()){

		std::cout<<"ERROR: NaN in a rowvector!\n";
		abort();

	}


}

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
	for(unsigned int i=0; i<v.size()-1; i++){

		outfile << v(i) <<",";
	}
	outfile << v(v.size()-1)<<"\n";


	outfile.close();

}


mat readMatFromCVSFile(std::string fileName){


	assert(!fileName.empty());


	mat dataBuffer;
	bool isReadOk  = dataBuffer.load(fileName,csv_ascii);

	if(isReadOk == false)
	{

		std::cout << "ERROR: Problem with loading data from file: "<<fileName<< endl;
		abort();
	}


	return dataBuffer;


}


void saveMatToCVSFile(mat M, std::string fileName){


	std::ofstream samplesFile(fileName);
	samplesFile.precision(15);

	if (samplesFile.is_open())
	{

		for(unsigned int i=0; i<M.n_rows; i++){

			rowvec r = M.row(i);

			for(unsigned int j=0; j<r.size()-1; j++){

				samplesFile << r(j)<<",";


			}

			samplesFile << r(r.size()-1)<<"\n";


		}

		samplesFile.close();
	}
	else{

		cout << "ERROR: Unable to open file";
		abort();
	}



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


	for(std::size_t i = 0; i < v.size()-1; ++i) {
		std::cout << v[i] << ", ";
	}

	std::cout << v[v.size()-1] << "\n";




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

	unsigned int dim = xmin.size();

	assert(dim == xmax.size());
	assert(matrixIn.n_cols >= dim);


	unsigned int N=matrixIn.n_rows;

	mat matrixOut(N,matrixIn.n_cols,fill::zeros);

	matrixOut.col(matrixIn.n_cols-1) = matrixIn.col(matrixIn.n_cols-1);

	vec deltax = xmax-xmin;

	for(unsigned int i=0; i<N;i++){

		for(unsigned int j=0; j<dim;j++){

			assert(deltax(j) > 0);

			matrixOut(i,j)  = ((matrixIn(i,j)-xmin(j)))/(deltax(j));

		}

	}

	return matrixOut;

}


mat normalizeMatrix(mat matrixIn, Bounds &boxConstraints){

	assert(boxConstraints.areBoundsSet());

	unsigned int dimBoxConstraints = boxConstraints.getDimension();
	unsigned int nColsDataMatrix = matrixIn.n_cols;
	unsigned int nRowsDataMatrix = matrixIn.n_rows;

	vec lowerBounds = boxConstraints.getLowerBounds();
	vec upperBounds = boxConstraints.getUpperBounds();
	vec deltaX = upperBounds - lowerBounds;

	mat matrixOut = matrixIn;


	bool normalizeAlsoLastColumn = true;

	if(nColsDataMatrix == dimBoxConstraints) normalizeAlsoLastColumn = true;
	if(nColsDataMatrix == dimBoxConstraints+1) normalizeAlsoLastColumn = false;

	for(unsigned int i=0; i<nRowsDataMatrix;i++){

		for(unsigned int j=0; j<nColsDataMatrix-1;j++){

			matrixOut(i,j)  = ((matrixIn(i,j)-lowerBounds(j)))/(deltaX(j));

		}

		if(normalizeAlsoLastColumn){

			matrixOut(i,nColsDataMatrix-1)  = ((matrixIn(i,nColsDataMatrix-1)-lowerBounds(nColsDataMatrix-1)))/(deltaX(nColsDataMatrix-1));
		}

	}



	return matrixOut;

}



bool isEqual(const mat &A, const mat&B, double tolerance){

	unsigned int m = A.n_rows;
	unsigned int n = A.n_cols;

	assert(m==B.n_rows && n==B.n_cols);

	for(unsigned int i=0; i<m; i++){

		for(unsigned int j=0; j<n; j++){

			if(checkValue(A(i,j),B(i,j), tolerance) == false) return false;

		}

	}

	return true;
}


int findInterval(double value, vec discreteValues){


	for(unsigned int i=0; i<discreteValues.size()-1; i++) {

		double xs = discreteValues[i];
		double xe = discreteValues[i+1];

		assert(xe>xs);

		if(value>=xs && value <xe) return i;

	}

	if (value > discreteValues[discreteValues.size()-1]) {

		return discreteValues.size()-1;
	}

	return -1;
}

