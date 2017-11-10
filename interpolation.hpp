#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP
#include <armadillo>
using namespace arma;

namespace tableInterpolation{

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

struct intkdNode{
    int indx;
    rowvec x;
    struct intkdNode *left, *right;
};




void test_2d_table(void);
void test_scatter_table(void);
void interpolateTable2DRectilinear(IntLookupTable* table,
        double x, double y,
        double *result,
        int* comp_index, int number_of_vars_to_interpolate);

void interpolateTableScatterdata(IntLookupTable* table,
        double *interpolation_variable,
        int *x_index,
        double *result,
        int* comp_index,
        int number_of_indep_vars,
        int number_of_vars_to_interpolate );

}
#endif
