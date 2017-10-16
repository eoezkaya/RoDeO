#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP
#include <armadillo>
using namespace arma;


typedef struct {
    double** data;   /* data points */
    int rowsize;     /* number of data points */
    int columnsize;  /* number of variables */
    int Ndim[3];     /* dimensions for structured tables */
    int last_interpolation_points[4]; /* keeps the indices of the interpolation points from the last interpolation */
    double *xmin; /* minimum of each column */
    double *xmax; /* maximum of each column */

    /* required for clustering operations */
    int *cluster_sizes;
    int *cluster_max_sizes;
    int **cluster_indices;
    double **cluster_means;
    int number_of_clusters;


} IntLookupTable;


void test_2d_table(void);

#endif
