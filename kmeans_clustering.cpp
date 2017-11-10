#include "kmeans_clustering.hpp"
#include "Rodeo_macros.hpp"
#include "auxilliary_functions.hpp"

#include <armadillo>

using namespace arma;



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <chrono>



int find_which_cluster(mat cluster_means, rowvec x){

    double min_distance=LARGE;


    int index=-1;


    for (unsigned int i=0; i<cluster_means.n_rows; i++){

#if 1
        printf("cluster %d mean = \n",i);
        cluster_means.row(i).print();
#endif

        rowvec xdiff = x - cluster_means.row(i);
        double distance = Lpnorm(xdiff, 2, x.size());
#if 1
        printf("xdiff = \n",i);
        xdiff.print();
        printf("distance = %10.7f\n",distance);
#endif


        if(distance < min_distance){

            min_distance = distance;
            index = i;
        }

    }

    return index;

}


void visualize_clusters(mat &data, mat &cluster_means, std::vector<int> *cluster_indices, int K){

    int number_of_data_points = data.n_rows;
    int dim = data.n_cols;

    if(dim != 2){

        printf("ERROR in visualize_clusters! (dim must be 2)\n");
        exit(-1);
    }



    std::string filename= "cluster_visualization.dat";
    std::string file_name_for_plot = "cluster_visualization.png";
    FILE * visualization_output = fopen(filename.c_str(),"w");

    fprintf(visualization_output,"%d\n",K);

    for(int i=0; i<K;i++){ /* for each cluster */

        fprintf(visualization_output,"%10.7f %10.7f\n",
                cluster_means(i,0), cluster_means(i,1));

    }


    for(int i=0; i<K;i++){ /* for each cluster */

        if(cluster_indices[i].size() != 0){

            fprintf(visualization_output,"%d\n",cluster_indices[i].size());

            for(int j=0; j<cluster_indices[i].size();j++){

                int indx = cluster_indices[i][j];

                fprintf(visualization_output,"%10.7f %10.7f\n",
                        data(indx,0), data(indx,1));
#if 0
                printf("%10.7f %10.7f",
                        data(indx,0),data(indx,1));
#endif

            }

        }
    }


    std::string python_command = "python -W ignore plot_clusters.py "+ filename+ " "+ file_name_for_plot ;

#if 0
    printf("%s\n",python_command.c_str());
#endif

    FILE* in = popen(python_command.c_str(), "r");



    fprintf(in, "\n");

    fclose(visualization_output);


}


void print_clusters(mat &data, mat &cluster_means, std::vector<int> *cluster_indices, int K){

    for(int i=0; i<K; i++){

        printf("Cluster %d mean = \n", i);
        cluster_means.row(i).print();


        int cluster_size = cluster_indices[i].size();
#if 1
        printf("Cluster %d has %d points\n", i,cluster_size );
#endif

        if (cluster_indices[i].size() > 0){

            for(int j=0; j<cluster_size; j++) {

                int indx = cluster_indices[i][j];

                rowvec Row = data.row(indx);

                printf("%d ",indx);
                Row.print();

            }


        }


    }

}

/** performs K-means clustering to partition the lookup table data.
 *
 * @param[in]  data the data matrix
 * @param[in]  K number of clusters
 * @param[out] cluster_means
 * @param[out] cluster_indices
 */
int kMeansClustering(mat &data,
        mat &cluster_means,
        std::vector<int> *cluster_indices,
        int K){


#if 1
    printf("kMeansClustering...\n");
#endif

    int number_of_data_points = data.n_rows;
    int dim = data.n_cols;

    int max_it=10000,it_count=0;

    srand (time(NULL));

    cluster_means.fill(0.0);

#if 0
    cluster_means.print();
#endif

    mat cluster_means_old(K,dim);
    cluster_means_old.fill(0.0);

    vec distance(K);
    distance.fill(0.0);

    vec xmin(dim);
    vec xmax(dim);


    for(int i=0; i<dim; i++){

        xmin(i) = min(data.col(i));
        xmax(i) = max(data.col(i));

    }


#if 0
    printf("xmin =\n");
    xmin.print();
    printf("xmax =\n");
    xmax.print();
#endif


    for(int i=0; i<K;i++){

        cluster_indices[i].clear();
    }


    for(int i=0; i<K; i++){

        for(int j=0; j<dim; j++){

            cluster_means(i,j) = RandomDouble(xmin(j),xmax(j));

        }

    }

#if 0
    printf("Cluster means initialization =\n");
    cluster_means.print();
#endif



    while(1){

        for(int i=0; i<K;i++){

            cluster_indices[i].clear();
        }

#if 0
        printf("Kmeans iteration = %d\n",it_count);
#endif

        cluster_means_old =  cluster_means;


        /* for each point in the table */
        for(int i=0;i<number_of_data_points;i++){


            rowvec x = data.row(i);

#if 0
            printf("x =\n");
            x.print();
#endif

            for(int j=0; j<K;j++){ /* for each cluster center */

                rowvec xdiff = x - cluster_means.row(j);

#if 0
                printf("xdiff =\n");
                xdiff.print();
#endif
                /* compute distance of the ith point to the jth cluster center */

                distance(j)= Lpnorm(xdiff, dim, xdiff.size());


            } /* end of j loop */

#if 0
            printf("distance = \n");
            distance.print();
#endif

            /* find the minimum distance */
            double min_distance=LARGE;
            int index_of_min_dist=-1;

            find_min_with_index(distance, distance.size(), &min_distance, &index_of_min_dist);

#if 0
            printf("closest cluster is %d\n",index_of_min_dist);
#endif
            /* point i belongs to cluster index_of_min_dist */

            cluster_indices[index_of_min_dist].push_back(i);

        } /* end of i loop */

        cluster_means.fill(0.0);

        /* for each cluster in the table */
        for(int i=0;i<K;i++){

            int cluster_size = cluster_indices[i].size();
#if 0
            printf("Cluster %d has %d points\n", i,cluster_size );
#endif

            if (cluster_indices[i].size() > 0){

                for(unsigned int j=0; j<cluster_size; j++) {

                    int indx = cluster_indices[i][j];

                    rowvec addRow = data.row(indx);

                    for(int k=0; k<dim; k++){

                        cluster_means(i,k) = cluster_means(i,k) + addRow(k);


                    }
                }

                for(int k=0; k<dim; k++){

                    cluster_means(i,k) = cluster_means(i,k) / cluster_size;


                }

            }
            else{ /* if a cluster is empty */


                for(int j=0; j<dim; j++){

                    cluster_means(i,j) = RandomDouble(xmin(j),xmax(j));

                }

            }

        }

#if 0
        printf("Cluster means updated =\n");
        cluster_means.print();
        printf("\n");

#endif

        if (it_count>max_it) {

            break;
        }



        /* check convergence of the cluster means */

        int convergence = 1;

        for(int i=0; i<K;i++){

            rowvec diff;

            diff = cluster_means_old.row(i) - cluster_means.row(i);

            double normdiff = Lpnorm(diff, diff.size(), diff.size());
#if 0
            cluster_means_old.row(i).print();
            cluster_means.row(i).print();
            printf("normdiff = %15.10f\n",normdiff );
#endif
            if (normdiff > EPSILON_SINGLE){

                convergence = 0;
                break;
            }

        }


        if (convergence == 1){
#if 1
            printf("convergence is achieved after %d iterations\n",it_count );
#endif
            break;
        }

        it_count++;


    } /* end of while(1) */



    return 0;
}

void test_kmeansClustering(void){


    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    int number_of_datapoints = RandomInt(1000,1000);

    number_of_datapoints = 1000;

    int dim = RandomInt(2,5);

    dim = 2;

    vec xs(dim);
    vec xe(dim);

    for(int i=0; i<dim; i++){
        xe(i)= RandomDouble(-10.0,10.0);
        xs(i)= RandomDouble(-10.0,10.0);
    }

    mat dataMatrix(number_of_datapoints,dim);
    mat normalizedDataMatrix(number_of_datapoints,dim);

    /* construct data matrix */
    for(int i=0; i<number_of_datapoints; i++){

        for(int j=0; j<dim; j++){

            dataMatrix(i,j)=RandomDouble(xs(j),xe(j));

        }

    }

#if 1
    dataMatrix.print();
#endif

    normalizeDataMatrix(dataMatrix, normalizedDataMatrix);


#if 1
    printf("Normalized data matrix = \n");
    normalizedDataMatrix.print();
#endif

    int K = number_of_datapoints/50;
#if 1
    printf("number of points = %d\n",number_of_datapoints);
    printf("number of clusters = %d\n",K);
    printf("dimension = %d\n",dim);
#endif

    mat cluster_means(K,dim);

    std::vector<int> *cluster_indices;

    cluster_indices = new std::vector<int>[K];

    kMeansClustering(normalizedDataMatrix,
            cluster_means,
            cluster_indices,
            K);


    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    delete[] cluster_indices;

}

void test_iwd_with_kmeans_method(void){

    auto start = std::chrono::high_resolution_clock::now();

    int number_of_datapoints = 1000;
    int dim = 2;

    double *min_dist = new double[4];
    int * indices = new int[4];

    vec xs(dim);
    vec xe(dim);

    for(int i=0; i<dim; i++){
        xe(i)= RandomDouble(-10.0,10.0);
        xs(i)= RandomDouble(-10.0,10.0);
    }

    mat dataMatrix(number_of_datapoints,dim);
    mat normalizedDataMatrix(number_of_datapoints,dim);

    /* construct data matrix */
    for(int i=0; i<number_of_datapoints; i++){

        for(int j=0; j<dim; j++){

            dataMatrix(i,j)=RandomDouble(xs(j),xe(j));

        }

    }

#if 1
    dataMatrix.print();
#endif

    normalizeDataMatrix(dataMatrix, normalizedDataMatrix);

    int K = number_of_datapoints/50;

#if 1
    printf("number of points = %d\n",number_of_datapoints);
    printf("number of clusters = %d\n",K);
    printf("dimension = %d\n",dim);
#endif

    mat cluster_means(K,dim);

    std::vector<int> *cluster_indices;

    cluster_indices = new std::vector<int>[K];

    kMeansClustering(normalizedDataMatrix,
            cluster_means,
            cluster_indices,
            K);

#if 0
    print_clusters(normalizedDataMatrix, cluster_means, cluster_indices, K);
    visualize_clusters(normalizedDataMatrix, cluster_means, cluster_indices, K);
#endif
    int number_of_trials =1000000;

    for(int i=0; i<number_of_trials; i++){

        rowvec x(dim);

        for(int j=0; j<dim; j++){

            x(j)=RandomDouble(0,1.0);

        }

        int cluster_no = find_which_cluster(cluster_means,x);

#if 1
        printf("x = \n");
        x.print();
        printf("the point is in the cluster %d\n",cluster_no);
#endif

        int cluster_size = cluster_indices[cluster_no].size();

#if 1
        printf("the cluster has %d points\n", cluster_size);

        for(unsigned int j=0; j<cluster_size; j++){
            int indx = cluster_indices[cluster_no].at(j);
            normalizedDataMatrix.row(indx).print();

        }

#endif

        mat cluster(cluster_size,dim);

        for(unsigned int j=0; j<cluster_size; j++){
                    int indx = cluster_indices[cluster_no].at(j);
                    rowvec point = normalizedDataMatrix.row(indx);

                    for(int k=0; k<dim; k++){

                        cluster(j,k) = point(k);

                    }

         }


        findKNeighbours(cluster, x, 4, min_dist,indices);
#if 1
        for(int k=0; k<4; k++){

            printf("point %d\n",indices[k]);
            cluster.row(indices[k]).print();

        }
#endif
        exit(1);

    }


    delete[] min_dist;
    delete[] indices;
}

