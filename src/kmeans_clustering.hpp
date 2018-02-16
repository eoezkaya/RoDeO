#ifndef KMEANS_CLUSTERING_HPP
#define KMEANS_CLUSTERING_HPP
#include <armadillo>
using namespace arma;

namespace kmeans{

void test_kmeansClustering(void);
void visualize_clusters(mat &data, mat &cluster_means,
        std::vector<int> *cluster_indices, int K);
int kMeansClustering(mat &data,
        mat &cluster_means,
        std::vector<int> *cluster_indices,
        int K);


int find_which_cluster(mat cluster_means, rowvec x);

}

#endif
