#include "kd_tree.hpp"
#include "Rodeo_macros.hpp"

#include <armadillo>

using namespace arma;



KdTreeNode* make_tree(KdTreeNode *t, int len, int i, int dim)
{
    KdTreeNode *n;

    if (!len) return 0;

    if ((n = find_median(t, t + len, i))) {
        i = (i + 1) % dim;
        n->left  = make_tree(t, n - t, i, dim);
        n->right = make_tree(n + 1, t + len - (n + 1), i, dim);
    }
    return n;
}
