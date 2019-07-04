#ifndef AUX_FUNCTIONS_HPP
#define AUX_FUNCTIONS_HPP
#include "Rodeo_macros.hpp"
#include <armadillo>
#include <vector>
#include <map>
#include <math.h>
using namespace arma;


void normalizeDataMatrix(mat matrixIn, mat &matrixOut);


template<class T>
void find_max_with_index(T vec, int size, double *max_val, int *indx){

	*max_val = -LARGE;

	for(int i=0; i<size; i++){

		if(vec[i] > *max_val){

			*max_val = vec[i];
			*indx = i;
		}

	}


}



template<class T>
void find_min_with_index(T vec, int size, double *min_val, int *indx){

	*min_val = LARGE;

	for(int i=0; i<size; i++){

		if(vec[i] < *min_val){

			*min_val = vec[i];
			*indx = i;
		}

	}


}



double pdf(double x, double mu, double sigma);

/* Returns the probability of [-inf,x] of a gaussian distribution */
double cdf(double x, double mu, double sigma);

double RandomDouble(double a, double b);
float RandomFloat(float a, float b);

int RandomInt(int a, int b);

void RandomVector(rowvec &x);

double random_number(double xs, double xe, double sigma_factor);

void solve_linear_system_by_Cholesky(mat &U, mat &L, vec &x, vec &b);

bool file_exist(const char *fileName);


int check_if_lists_are_equal(int *list1, int *list2, int dim);
int is_in_the_list(int entry, int *list, int list_size);
int is_in_the_list(int entry, std::vector<int> &list);
int is_in_the_list(unsigned int entry, uvec &list);

void compute_max_min_distance_data(mat &x, double &max_distance, double &min_distance);


void generate_validation_set(int *indices, int size, int N);
void generate_validation_set(uvec &indices, int size);

void remove_validation_points_from_data(mat &X, vec &y, uvec & indices, mat &Xmod, vec &ymod);
void remove_validation_points_from_data(mat &X, vec &y, uvec & indices, mat &Xmod, vec &ymod, uvec &map);


/* distance functions */

template<class T>
double L1norm(T x, int p, int* index=NULL){
	double sum=0.0;
	if(index == NULL){

		for(int i=0;i<p;i++){

			sum+=fabs(x(i));
		}
	}
	else{

		for(int i=0;i<p;i++){

			sum+=fabs(x(index[i]));
		}

	}

	return sum;
}


template<class T>
double L2norm(T x, int p, int* index=NULL){

	double sum;
	if(index == NULL){
		sum=0.0;

		for(int i=0;i<p;i++){

			sum+=x(i)*x(i);
		}

	}
	else{

		sum=0.0;

		for(int i=0;i<p;i++){

			sum+=x(index[i])*x(index[i]);
		}

	}

	return sqrt(sum);
}

template<class T>
double Lpnorm(T x, int p, int size,int *index=NULL){
	double sum=0.0;


	if(index == NULL){

		for(int i=0;i<size;i++){

			sum+=pow(fabs(x(i)),p);
		}

	}
	else{

		for(int i=0;i<size;i++){

			sum+=pow(fabs(x(index[i])),p);
		}


	}
	return pow(sum,1.0/p);
}

double calcMetric(rowvec &xi,rowvec &xj, mat M);
float calcMetric(frowvec &xi,frowvec &xj, fmat M);

void findKNeighbours(mat &data, rowvec &p, int K, double* min_dist,int *indices, unsigned int norm=2);


void findKNeighbours(mat &data,
		             rowvec &p,
					 int K,
					 vec &min_dist,
					 uvec &indices,
					 mat M);


void findKNeighbours(mat &data,
		rowvec &p,
		int K,
		int *input_indx ,
		double* min_dist,
		int *indices,
		int number_of_independent_variables);

int getPopularlabel(int* labels, int size);

void testLPnorm(void);

#endif
