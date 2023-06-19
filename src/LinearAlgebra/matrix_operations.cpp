/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), RPTU
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, RPTU)
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
 * General Public License along with RoDeO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */




#include "./INCLUDE/matrix_operations.hpp"

#include "./INCLUDE/vector_operations.hpp"
#include "../Auxiliary/INCLUDE/auxiliary_functions.hpp"
#include "../Random/INCLUDE/random_functions.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


void printScalarValueWithName(std::string name, int value) {

	std::cout<<name<<" = "<<value<<"\n";

}
void printScalarValueWithName(std::string name, double value) {

	std::cout<<name<<" = "<<value<<"\n";
}


void printScalarValueWithName(std::string name, unsigned int value) {

	std::cout<<name<<" = "<<value<<"\n";
}




void printTwoScalarValuesWithNames(std::string name1, double value1,std::string name2, double value2 ){

	std::cout<<name1<<" = "<<value1<<" "<<name2<<" = "<<value2<<"\n";

}









void joinMatricesByColumns(mat& A, const mat& B){

	assert(A.n_rows == B.n_rows);
	A.insert_cols(A.n_cols, B);

}

void joinMatricesByRows(mat& A, const mat& B){

	assert(A.n_cols == B.n_cols);
	A.insert_rows(A.n_rows, B);

}




void appendMatrixToCSVData(const mat& A, std::string fileName){

	unsigned int nRows = A.n_rows;

	for(unsigned int i=0; i<nRows; i++){

		appendRowVectorToCSVData(A.row(i),fileName);
	}


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





int findIndexOfRow(const rowvec &v, const mat &A, double tolerance = 10E-8){

	assert(v.size() == A.n_cols);
	for(unsigned int i=0; i<A.n_rows; i++){
		if(isEqual(v,A.row(i), tolerance)) return i;
	}
	return -1;
}


mat shuffleRows(mat A){

	mat result = A;

	unsigned int Nrows = result.n_rows;
	assert(Nrows>0);

	for(unsigned int i=0; i<10*Nrows; i++){

		int indx1 = generateRandomInt(0, Nrows-1);
		int indx2 = generateRandomInt(0, Nrows-1);

		if(indx1 != indx2) result.swap_rows( indx1, indx2  );
	}
	return result;
}






