#ifndef AUX_FUNCTIONS_HPP
#define AUX_FUNCTIONS_HPP
#include "Rodeo_macros.hpp"
#include <armadillo>
#include <vector>
using namespace arma;

  
void find_max_with_index(double *vec, int size, double *max_val, int *indx);

void find_min_with_index(double *vec, int size, double *min_val, int *indx);


double pdf(double x, double mu, double sigma);

/* Returns the probability of [-inf,x] of a gaussian distribution */
double cdf(double x, double mu, double sigma);

double RandomDouble(double a, double b);
int RandomInt(int a, int b);

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

double L1norm(vec & x);

double L1norm(rowvec & x);

double L2norm(vec & x);

double L2norm(rowvec & x);

double Lpnorm(vec & x, int p);

double Lpnorm(rowvec & x, int p);

void findKNeighbours(mat &data, rowvec &p, int K, double* min_dist,int *indices);
void findKNeighbours(mat &data,
                     rowvec &p,
                     int K,
                     int *input_indx ,
                     double* min_dist,
                     int *indices,
                     int number_of_independent_variables);
#endif
