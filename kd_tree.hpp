#ifndef KD_TREE_HPP
#define KD_TREE_HPP
#include <armadillo>
using namespace arma;

#define N 1000000
#define rand1() (rand() / (double)RAND_MAX)
#define rand_pt(v) { v.x[0] = rand1(); v.x[1] = rand1(); v.x[2] = rand1(); }

#define MAX_DIM 3
struct kd_node_t{
    double x[MAX_DIM];
    struct kd_node_t *left, *right;
};


inline double
dist(struct kd_node_t *a, struct kd_node_t *b, int dim)
{
    double t, d = 0;
    while (dim--) {
        t = a->x[dim] - b->x[dim];
        d += t * t;
    }
    return d;
}
inline void swap(struct kd_node_t *x, struct kd_node_t *y) {
    double tmp[MAX_DIM];
    memcpy(tmp,  x->x, sizeof(tmp));
    memcpy(x->x, y->x, sizeof(tmp));
    memcpy(y->x, tmp,  sizeof(tmp));
}


struct kd_node_t*
find_median(struct kd_node_t *start, struct kd_node_t *end, int idx);

struct kd_node_t*
make_tree(struct kd_node_t *t, int len, int i, int dim);

void nearest(struct kd_node_t *root, struct kd_node_t *nd, int i, int dim,
        struct kd_node_t **best, double *best_dist, int &visited);


void kd_tree_test(void);


#endif
