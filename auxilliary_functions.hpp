#ifndef AUX_FUNCTIONS_HPP
#define AUX_FUNCTIONS_HPP
#include "Rodeo_macros.hpp"
#include <armadillo>
#include <vector>
using namespace arma;

  
double pdf(double x, double mu, double sigma);

/* Returns the probability of [-inf,x] of a gaussian distribution */
double cdf(double x, double mu, double sigma);

double RandomDouble(double a, double b);

double random_number(double xs, double xe, double sigma_factor);

void solve_linear_system_by_Cholesky(mat &U, mat &L, vec &x, vec &b);

bool file_exist(const char *fileName);

int is_in_the_list(int entry, int *list, int list_size);
int is_in_the_list(int entry, std::vector<int> &list);

void compute_max_min_distance_data(mat &x, double &max_distance, double &min_distance);


#endif
