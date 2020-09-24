#ifndef METRIC
#define METRIC
#include <armadillo>
using namespace arma;

double calculateL1norm(const rowvec &x);

double calculateMetric(rowvec &xi,rowvec &xj, mat M);
double calculateMetricAdjoint(rowvec xi, rowvec xj, mat M, mat &Mb, double calculateMetricb);


#endif
