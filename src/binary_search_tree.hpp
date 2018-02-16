#ifndef BINARY_TREE_HPP
#define BINARY_TREE_HPP
#include <armadillo>
using namespace arma;

struct binaryTreeNode{
    int indx;
    rowvec x;
    binaryTreeNode *left, *right;
};

void binary_tree_test(void);

#endif
