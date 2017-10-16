#ifndef KD_TREE_HPP
#define KD_TREE_HPP
#include <armadillo>
using namespace arma;


class KdTuple{
    double x;
    int indx;

};

class KdTreeNode {
public:

    KdTuple divider;
    int table index;

    KdTreeNode *left;
    KdTreeNode *right;

    void print(void){

    }
} ;








#endif
