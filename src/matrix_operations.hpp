#ifndef MATRIX_OPERATIONS_HPP
#define MATRIX_OPERATIONS_HPP
#include <armadillo>

using namespace arma;

rowvec normalizeRowVector(rowvec x, vec xmin, vec xmax);
rowvec normalizeRowVectorBack(rowvec xnorm, vec xmin, vec xmax);

mat normalizeMatrix(mat matrixIn, double xmin, double xmax);
mat normalizeMatrix(mat matrixIn, vec xmin, vec xmax);

#endif
