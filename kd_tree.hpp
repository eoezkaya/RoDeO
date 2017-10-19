#ifndef KD_TREE_HPP
#define KD_TREE_HPP
#include <armadillo>
using namespace arma;


struct kdNode{
    int indx;
    rowvec x;
    struct kdNode *left, *right;
};



void nearest(kdNode *root, kdNode *nd, int i, int dim, kdNode **best, double *best_dist, int &visited);

void kd_tree_test(void);

#endif
