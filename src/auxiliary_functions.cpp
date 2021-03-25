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

#include "auxiliary_functions.hpp"
#include <chrono>
#include <random>
#include <string>
#include <math.h>
#include <vector>


void executePythonScript(std::string command){

	FILE* in = popen(command.c_str(), "r");
	fprintf(in, "\n");

}


void changeDirectoryToRodeoHome(void){

	const char* env_p;
	if(env_p = std::getenv("RODEO_HOME")){
		std::cout << "RODEO_HOME: " << env_p << '\n';
	}
	else{
		std::cout<<"The environmental variable RODEO_HOME is undefined!\n";
		abort();

	}

	int ret = chdir (env_p);
	if(ret!=0){

		std::cout<<"ERROR: Cannot change directory to $RODEO_HOME\n";
		abort();
	}

}

void changeDirectoryToUnitTests(void){

	int ret =chdir ("./UnitTests");
	if(ret!=0){

		std::cout<<"ERROR: Cannot change directory to $RODEO_HOME/UnitTests\n";
		abort();
	}

}

bool checkValue(double value, double expected, double tolerance){

	assert(tolerance > 0.0);

	if(fabs(value-expected) > tolerance) {
#if 0
		printf("\nvalue = %10.7f, expected = %10.7f, error = %10.7f, tolerance = %10.7f\n",value, expected,fabs(value-expected),tolerance );
#endif
		return false;
	}
	else return true;


}

bool checkValue(double value, double expected){

	double tolerance = 0.0;
	if(fabs(value) < 10E-10){

		tolerance = EPSILON;

	}
	else{

		tolerance = fabs(value) * 0.001;
	}

	double error = fabs(value-expected);
	if(error > tolerance) {

#if 0
		printf("\nvalue = %10.7f, expected = %10.7f, error = %10.7f, tolerance = %10.7f\n",value, expected,error, tolerance);
#endif
		return false;
	}
	else return true;


}

bool checkMatrix(mat values, mat expected, double tolerance){
	assert(values.n_rows == expected.n_rows);
	assert(values.n_cols == expected.n_cols);
	bool result = true;

	for(unsigned int i=0; i<values.n_rows; i++){

		for(unsigned int j=0; j<values.n_cols; j++){

			if(!checkValue(values(i,j), expected(i,j), tolerance)){

				result = false;
			}

		}

	}

	return result;

}


bool checkMatrix(mat values, mat expected){
	assert(values.n_rows == expected.n_rows);
	assert(values.n_cols == expected.n_cols);
	bool result = true;

	for(unsigned int i=0; i<values.n_rows; i++){

		for(unsigned int j=0; j<values.n_cols; j++){

			if(!checkValue(values(i,j), expected(i,j))){

				result = false;
			}

		}

	}

	return result;

}


void abortIfFalse(bool flag, std::string file, int line){

	if(flag == false){

		printf("Test failed at: at %s, line %d.\n",file.c_str(),line);
		abort();
	}

}


void abortIfFalse(bool flag){

	if(flag == false){

		printf("Test failed at\n");
		abort();
	}

}



double calculatePolynomial(double x, const rowvec &coeffs){

	//	printVector(coeffs,"coeffs");
	double sum = 0.0;

	for(unsigned int i=0; i<coeffs.size(); i++){
#if 0
		printf("coeffs(%d)*pow(%10.7f,%d) = %10.7f\n",i,x,i,coeffs(i)*pow(x,i));
#endif

		sum += coeffs(i)*pow(x,i);
	}


	return sum;
}

double calculateTensorProduct(const rowvec &x, const mat &coeffs){

	assert(x.size() == coeffs.n_rows);

	double prod = 1.0;
	for(unsigned int i=0; i<coeffs.n_rows; i++){

		double sum = calculatePolynomial(x(i), coeffs.row(i));

		prod = prod*sum;
	}

	return prod;
}



void normalizeDataMatrix(mat matrixIn, mat &matrixOut){

	int dim = matrixIn.n_cols;

	vec xmin(dim);
	vec xmax(dim);

	for(int i=0; i<dim;i++){

		xmin(i) = min(matrixIn.col(i));
		xmax(i) = max(matrixIn.col(i));
	}

#if 0
	printf("xmin = \n");
	xmin.print();
	printf("xmax = \n");
	xmax.print();
#endif

	for(int j=0; j<dim;j++){

		double delta = xmax(j)-xmin(j);

		if( delta != 0 ){

			for(unsigned int i=0; i<matrixIn.n_rows;i++){

				matrixOut(i,j)  = ((1.0/dim)*(matrixIn(i,j)-xmin(j)))/(delta);
			}

		}

	}




}

int check_if_lists_are_equal(int *list1, int *list2, int dim){

	int flag=1;
	for(int i=0; i<dim; i++){

		int item_to_check = list1[i];

		if ( is_in_the_list(item_to_check, list2, dim) == -1){

			flag = 0;
			return flag;


		}
	}

	return flag;
}







/** Returns the pdf of x, given the distribution described by mu and sigma..
 *
 * @param[in] x
 * @param[in] mu
 * @param[in] sigma
 * @return normal_pdf(x) with mu and sigma
 *
 */


double pdf(double x, double m, double s)
{
	static const double inv_sqrt_2pi = 0.3989422804014327;
	double a = (x - m) / s;

	return inv_sqrt_2pi / s * std::exp(-0.5 * a * a);
}



/** Returns the cdf of x, given the distribution described by mu and sigma..
 *
 * @param[in] x
 * @param[in] mu
 * @param[in] sigma
 * @return normal_cdf(x) with mu and sigma
 *
 */
double cdf(double x, double mu, double sigma)
{
	return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2.0))));
}


/** checks whether an entry is in a list or not.
 *
 * @param[in] entry
 * @param[in] list
 * @param[in] list size
 * @return -1 if no, position in the list if yes
 *
 */
int is_in_the_list(int entry, int *list, int list_size){

	int flag=-1;

	for(int i=0;i< list_size;i++) {

		if(list[i]==entry) flag=i;
	}

	return flag;
}

/** checks whether an entry is in a list or not.
 *
 * @param[in] entry
 * @param[in] list (as a std::vector)
 * @param[in] list size
 * @return -1 if no, position in the list if yes
 *
 */
int is_in_the_list(int entry, std::vector<int> &list){

	int flag=-1;

	for (std::vector<int>::iterator it = list.begin() ; it != list.end(); ++it){

		if(*it == entry) {

			flag = *it;
		}

	}

	return flag;
}

/** checks whether an entry is in a list or not.
 *
 * @param[in] entry
 * @param[in] list (as a arma::vec)
 * @param[in] list size
 * @return -1 if no, position in the list if yes
 *
 */
int is_in_the_list(unsigned int entry, uvec &list){

	int flag=-1;

	for (unsigned int i = 0 ; i< list.size(); i++){

		if(list(i) == entry) {

			flag = i;
		}

	}

	return flag;
}



bool ifIsInTheList(const std::vector<std::string> &vec, std::string item){

	if ( std::find(vec.begin(), vec.end(), item) != vec.end() )
		return true;
	else
		return false;



}


/** solve a linear system Ax=b, with a given Cholesky decomposition of A
 *
 * @param[in] U
 * @param[out] x
 * @param[in] b
 *
 */
void solveLinearSystemCholesky(mat U, vec &x, vec b){

	mat L = trans(U);

	unsigned int dim = x.size();

	if(dim != U.n_rows || dim != b.size()){

		std::cout<<"ERROR: Dimensions does not match!\n";
		abort();

	}
	/* initialize x */

	x.fill(0.0);

	vec y(dim,fill::zeros);

	/* forward subst. L y = b */

	for (unsigned int i = 0; i < dim; i++) {

		double residual = 0.0;

		for (unsigned int j = 0; j < i; j++) {

			residual = residual + L(i, j) * y(j);

		}

		y(i) = (b(i) - residual) / L(i, i);
	}


	/* back subst. U x = y */

	for (int i = dim - 1; i >= 0; i--) {

		double residual = 0.0;

		for (int j = dim - 1; j > i; j--){

			residual = residual + U(i, j) * x(j);
		}


		x(i) = (y(i) - residual) / U(i, i);
	}

}


bool checkLinearSystem(mat A, vec x, vec b, double tol){

	vec r = A*x-b;
	double norm = L1norm(r, r.size());

	if(norm > tol) return false;
	else return true;

}


vec calculateResidual(mat A, vec x, vec b){

	vec r = A*x-b;

	return r;
}





/** randomly generates the indices of a validation set
 *
 * @param[in] size   dimension of the validation set
 * @param[in] N      dimension of the data set
 * @param[out] indices
 *
 */
void generate_validation_set(int *indices, int size, int N){

	int number_of_indices_generated=0;
	int random_int;
	int flag;

#if 1
	printf("size of the validation set = %d\n",size);
	printf("size of the data set = %d\n",N);
#endif

	if(size <0 || size>N){

		printf("Error: Size of the validation set is wrong");
		exit(-1);
	}

	for(int i=0; i<size;i++){

		indices[i]=-1;
	}

	/* initialize random seed: */
	srand (time(NULL));

	while(number_of_indices_generated < size){

		while(1){

			/* generate a random index */
			random_int = rand() % N;
#if 1
			printf("random_int = %d\n",random_int);
#endif
			/* check if it is already in the list */
			flag = is_in_the_list(random_int, indices, size);
#if 1
			printf("flag = %d\n",flag);
#endif

			if(flag == -1){

				indices[number_of_indices_generated]= random_int;
				number_of_indices_generated++;
				break;

			}


		}

	}


}

/** randomly generates the indices of a validation set
 *
 * @param[in]  N:  dimension of the data set
 * @param[out] indices (arma::uvec)
 *
 */
void generate_validation_set(uvec &indices, int N){

	int size = indices.size();
	int number_of_indices_generated=0;
	int random_int;
	int flag;

#if 0
	printf("size of the validation set = %d\n",size);
	printf("size of the data set = %d\n",N);
#endif

	if(size <=0 || size>N){

		printf("Error: Size of the validation set is wrong");
		exit(-1);
	}

	indices.fill(-1);


	while(number_of_indices_generated < size){

		while(1){

			/* generate a random index */
			random_int = rand() % N;
			/* check if it is already in the list */
			flag= is_in_the_list(random_int, indices);

			if(flag == -1){

				indices[number_of_indices_generated]= random_int;
				number_of_indices_generated++;
				break;

			}


		}

	}

	indices = sort(indices);


}


/** generates a modified data set by removing validation points
 *
 * @param[in] X
 * @param[in] y
 * @param[in] indices
 * @param[out] Xmod
 * @param[out] ymod
 * @param[out] map
 */
void remove_validation_points_from_data(mat &X, vec &y, uvec & indices, mat &Xmod, vec &ymod, uvec &map){


	int added_rows=0;
	for(unsigned int j=0; j<X.n_rows; j++){ /* for each row in the data matrix */

		/* if j is not a validation point */
		if ( is_in_the_list(int(j), indices) == -1){

#if 0
			printf("%dth point is not a validation point\n",j);
#endif
			Xmod.row(added_rows)=X.row(j);
			ymod(added_rows)    =y(j);
			map(added_rows) = j;
			added_rows++;


		}


	}
#if 0
	printf("data set map\n");
	map.print();
#endif




}

/** generates a modified data set by removing validation points
 *
 * @param[in] X
 * @param[in] y
 * @param[in] indices
 * @param[out] Xmod
 * @param[out] ymod
 */

void remove_validation_points_from_data(mat &X, vec &y, uvec & indices, mat &Xmod, vec &ymod){


	int added_rows=0;
	for(unsigned int j=0; j<X.n_rows; j++){ /* for each row in the data matrix */

		/* if j is not a validation point */
		if ( is_in_the_list(int(j), indices) == -1){

#if 0
			printf("%dth point is not a validation point\n",j);
#endif
			Xmod.row(added_rows)=X.row(j);
			ymod(added_rows)    =y(j);
			added_rows++;

		}


	}

	exit(1);

}

bool checkifTooCLose(const rowvec &v1, const rowvec &v2, double tol){

	rowvec diff = v1 - v2;

	double distance = L1norm(diff, v1.size());

	if(distance < tol) return true;
	else return false;

}

bool checkifTooCLose(const rowvec &v1, const mat &M, double tol){


	unsigned int nRows = M.n_rows;
	bool ifTooClose = false;

	for(unsigned int i=0; i<nRows; i++){

		rowvec r = M.row(i);
		ifTooClose = checkifTooCLose(v1,r, tol);

		if(ifTooClose) {
			break;
		}

	}

	return ifTooClose;
}




bool file_exist(std::string filename)
{

	return file_exist(filename.c_str());
}

bool file_exist(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}


std::string removeSpacesFromString(std::string inputString){

	inputString.erase(remove_if(inputString.begin(), inputString.end(), isspace), inputString.end());
	return inputString;
}


void getValuesFromString(std::string str, std::vector<std::string> &values,char delimiter){


	assert(values.size() == 0);

	if(str[0] == '{' || str[0] == '['){

		str.erase(0,1);
	}

	if(str[str.length()-1] == '}' || str[str.length()-1] == ']'){

		str.erase(str.length()-1,1);
	}

	while(1){

		std::size_t found = str.find(delimiter);
		if (found==std::string::npos) break;

		std::string buffer;
		buffer.assign(str,0,found);
		str.erase(0,found+1);

		values.push_back(buffer);

	}

	values.push_back(str);


}






/*
 * Correlation function R(x^i,x^j)
 *
 * R(x^i,x^j)=exp(-sum_{k=1}^p (  theta_k* ( abs(x^i_k-x^j_k)**gamma_k  ) )  )
 * @param[in] x_i
 * @param[in] X_j
 * @param[in] theta
 * @param[in] gamma
 * @return R
 *
 * */
double compute_R(rowvec x_i, rowvec x_j, vec theta, vec gamma) {

	int dim = theta.size();

	double sum = 0.0;
	for (int k = 0; k < dim; k++) {

		sum += theta(k) * pow(fabs(x_i(k) - x_j(k)), gamma(k));

	}

	return exp(-sum);
}



/*
 * Correlation function R(x^i,x^j) with gamma = 2.0
 *
 * R(x^i,x^j)=exp(-sum_{k=1}^p (  theta_k* ( abs(x^i_k-x^j_k)**2.0  ) )  )
 * @param[in] x_i
 * @param[in] X_j
 * @param[in] theta
 * @return R
 *
 * */
double compute_R_Gauss(rowvec x_i,
		rowvec x_j,
		vec theta) {

	int dim = theta.size();

	double sum = 0.0;
	for (int k = 0; k < dim; k++) {

		sum += theta(k) * pow(fabs(x_i(k) - x_j(k)), 2.0);

	}

	return exp(-sum);
}



/*
 *
 *
 * derivative of R(x^i,x^j) w.r.t. x^i_k (for GEK)
 *
 *
 * */

double compR_dxi(rowvec x_i, rowvec x_j, vec theta, int k) {

	int dim = theta.size();
	double sum = 0.0;
	double result;



	/* first compute R(x^i,x^j) */
	for(int m=0;m<dim;m++){
		sum+=theta(m)*pow( fabs(x_i(m)-x_j(m)),2.0 );
	}
	sum=exp(-sum);
	result= -2.0*theta(k)* (x_i(k)-x_j(k))* sum;
	return result;
}





/*
 *
 *
 * derivative of R(x^i,x^j) w.r.t. x^j_k (for GEK)
 *
 *
 *
 * */

double compR_dxj(rowvec x_i, rowvec x_j, vec theta,  int k) {

	int dim = theta.size();
	double sum = 0.0;
	double result;


	/* first compute R(x^i,x^j) */
	for(int m=0;m<dim;m++){
		sum+=theta(m)*pow( fabs(x_i(m)-x_j(m)),2.0 );
	}
	sum=exp(-sum);

	result= 2.0*theta(k)* (x_i(k)-x_j(k))* sum;

	return result;
}


/*
 *
 * second derivative of R(x^i,x^j) w.r.t. x^i_l and x^j_k (hand derivation)
 * (for GEK)
 *
 * */

double compute_dr_dxi_dxj(rowvec x_i, rowvec x_j, vec theta,int l,int k){

	int dim = theta.size();
	double corr = 0.0;
	double dx;

	for (int i = 0;i<dim;i++){

		corr = corr + theta(i) * pow(fabs(x_i(i)-x_j(i)),2.0);
	}

	corr = exp(-corr);

	if (k == l){

		dx = 2.0*theta(k)*(-2.0*theta(k)*pow((x_i(k)-x_j(k)),2.0)+1.0)*corr;
	}
	if (k != l) {

		dx = -4.0*theta(k)*theta(l)*(x_i(k)-x_j(k))*(x_i(l)-x_j(l))*corr;
	}

	return dx;
}


/*
 *
 * Computation of the correlation matrix using standart correlation function
 *
 *
 * */

void compute_R_matrix(vec theta,
		vec gamma,
		double reg_param,
		mat &R,
		mat &X) {
	double temp;
	int nrows = R.n_rows;

	R.fill(0.0);


	for (int i = 0; i < nrows; i++) {
		for (int j = i + 1; j < nrows; j++) {

			temp = compute_R(X.row(i), X.row(j), theta, gamma);
			R(i, j) = temp;
			R(j, i) = temp;
		}

	}

	R = R + eye(nrows,nrows) + eye(nrows,nrows)*reg_param;

} /* end of compute_R_matrix */


