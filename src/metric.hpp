#ifndef METRIC
#define METRIC
#include <armadillo>
using namespace arma;

double calculateMetric(rowvec &xi,rowvec &xj, mat M);
double calculateMetricAdjoint(rowvec xi, rowvec xj, mat M, mat &Mb, double calculateMetricb);


#endif
