#ifndef KD_TREE_HPP
#define KD_TREE_HPP
#include <armadillo>
using namespace arma;


namespace kdtree{

struct kdNode{
    int indx;
    rowvec x;
    struct kdNode *left, *right;
};

void kd_tree_test(void);
void kd_tree_test2(void);

}

#endif
