#ifndef RANDOM_FUNCTIONS_HPP
#define RANDOM_FUNCTIONS_HPP
#include <armadillo>

using namespace arma;

int generateRandomInt(int a, int b);

double generateRandomDouble(double a, double b);
void generateRandomDoubleArray(double *xp,double a, double b, unsigned int dim);


rowvec generateRandomRowVector(vec lb, vec ub);
rowvec generateRandomRowVector(double lb, double ub, unsigned int dim);

vec generateRandomVector(vec lb, vec ub);
vec generateRandomVector(double lb, double ub, unsigned int dim);

double generateRandomDoubleFromNormalDist(double xs, double xe, double sigma_factor);


void generateKRandomIntegers(uvec &numbers, unsigned int N, unsigned int k);
mat generateRandomWeightMatrix(unsigned int dim);




#endif
