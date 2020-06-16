#ifndef MATRIX_OPERATIONS_HPP
#define MATRIX_OPERATIONS_HPP
#include <armadillo>

using namespace arma;

bool isEqual(const mat &A, const mat&B, double tolerance);

void printMatrix(mat M, std::string name="None");
void printVector(vec v, std::string name="None");
void printVector(rowvec v, std::string name="None");

vec normalizeColumnVector(vec x, double xmin, double xmax);

rowvec normalizeRowVector(rowvec x, vec xmin, vec xmax);
rowvec normalizeRowVectorBack(rowvec xnorm, vec xmin, vec xmax);

mat normalizeMatrix(mat matrixIn);

bool checkIfSymmetricPositiveDefinite(const mat &M);
bool checkIfSymmetric(const mat &M);
#endif
