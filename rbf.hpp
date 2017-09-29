#ifndef TRAIN_RBF_HPP
#define TRAIN_RBF_HPP
#include <armadillo>
using namespace arma;


int train_rbf(mat &X, vec &ys, vec &w, double &sigma, int type);
double calc_ftilde_rbf(mat &X, rowvec &xp, vec &w, int type, double sigma);

#endif
