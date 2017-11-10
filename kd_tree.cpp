#include "kd_tree.hpp"
#include "Rodeo_macros.hpp"
#include "auxilliary_functions.hpp"

#include <armadillo>

using namespace arma;



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>



inline double dist(kdNode *a, kdNode *b) {

    int dim = a->x.size();
    double t, d = 0;
    while (dim--) {
        t = a->x(dim) - b->x(dim);
        d += t * t;
    }
    return d;
}



double find_median(kdNode *nodelist, int len, int idx)
{

    if(len==2){

        return nodelist[0].x(idx);
    }

    vec temp(len);

    for(int i=0; i<len; i++){

        temp(i) = nodelist[i].x(idx);
    }

    temp= sort(temp);

    double median_value = temp(temp.size()/2);

    return median_value;
}



kdNode* make_tree(kdNode *root, int len, int indx){

#if 0
    printf("make_tree with root = %d and length = %d\n", root->indx, len);
#endif


    int dim = root[0].x.size();
#if 0
    printf("dim= %d\n",dim);
#endif
    if (len==0) {

        return NULL;
    }


    kdNode *start = &root[0];


#if 0
    printf("start node index= %d\n",start->indx);
    printf("end node index  = %d\n",end->indx);
    start->x.print();
    end->x.print();
#endif


    double median_value = find_median(start, len, indx);



    std::vector<int> left_indices;
    std::vector<int> right_indices;


    int count=0;
    int median_index =0;
    for(int i=0; i<len; i++){

        if(root[i].x(indx) < median_value) {

            left_indices.push_back(i);

        }
        else if( root[i].x(indx) == median_value){

            median_index=i;
        }

        else{

            right_indices.push_back(i);
        }


    }

    int leftsize = left_indices.size();
    int rightsize = right_indices.size();

#if 0
    printf("left indices = \n");
    for (std::vector<int>::iterator it = left_indices.begin() ; it != left_indices.end(); ++it)
        std::cout << ' ' << *it;
    std::cout << '\n';


    printf("right indices = \n");
    for (std::vector<int>::iterator it = right_indices.begin() ; it != right_indices.end(); ++it)
        std::cout << ' ' << *it;
    std::cout << '\n';

    printf("median = %10.7f\n", median_value);
    printf("median index = %d\n", median_index);
    printf("leftsize = %d\n", leftsize);
    printf("rightsize = %d\n", rightsize);
#endif



    kdNode *kdNodeVectemp = new kdNode[len];

    count=0;
    for (std::vector<int>::iterator it = left_indices.begin() ; it != left_indices.end(); ++it){
        kdNodeVectemp[count].x = root[*it].x;
        kdNodeVectemp[count].indx = root[*it].indx;
        count++;

    }

    kdNodeVectemp[count].x = root[median_index].x;
    kdNodeVectemp[count].indx = root[median_index].indx;
    median_index = count;

    count++;

    for (std::vector<int>::iterator it = right_indices.begin() ; it != right_indices.end(); ++it){
        kdNodeVectemp[count].x = root[*it].x;
        kdNodeVectemp[count].indx = root[*it].indx;

        count++;

    }

    for(int i=0; i<len; i++){

        root[i].x    = kdNodeVectemp[i].x;
        root[i].indx = kdNodeVectemp[i].indx;
#if 0
        printf("Node index = %d\n",root[i].indx);
        root[i].x.print();
#endif
    }

    kdNode *n= &root[median_index];

    indx = (indx + 1) % dim;

    if (leftsize == 1){
#if 0
        printf("there is only one node in left = %d\n",root[0].indx);
#endif
        n->left = &root[0];


    }
    else if(leftsize == 0){

        n->left = NULL;
    }
    else{
#if 0
        printf("calling left tree with root = %d and len = %d\n",leftsize);
#endif
        n->left  = make_tree(&root[0], leftsize, indx);

    }

    if (rightsize == 1){
#if 0
        printf("there is only one node in right = %d\n",root[leftsize+1].indx);
#endif
        n->right = &root[leftsize+1];


    }
    else if(rightsize == 0){

        n->right = NULL;
    }
    else{
#if 0
        printf("calling right tree with root = %d and len = %d\n",leftsize+1,rightsize);
#endif
        n->right = make_tree(&root[leftsize+1], rightsize, indx );
    }

    delete[] kdNodeVectemp;


    return n;
}

/** kNN algorithm for the kd trees.
 *
 * @param[in] root
 * @param[in] p
 * @param[in] K
 * @param[in] i
 * @param[out] best_index
 * @param[out] best_dist
 * @param[in] dim
 *
 */

void nearest(kdNode *root, kdNode *p, int K, int i, int* best_index, double *best_dist, int dim){

    for(int i=0; i<K; i++){

        best_dist[i] = LARGE;
        best_index[i] = -1;
    }

    int max_dist_index = root->indx;
    /* default root is the best initially*/

    rowvec xdiff = root->x - p->x;
    double d = Lpnorm(xdiff, dim, xdiff.size());

    double max_dist = d;

#if 0
    printf("\n\nroot = %d\n",root->indx);
    root->x.print();
    printf("distance to root = %10.7f\n",d);
#endif


#if 0
    printf("%10.7f is better than %10.7f\n",d,*best_dist);
#endif
    /* initialize  */
    best_index[0] = root->indx;
    best_dist[0]  = d;


    /* there is only one  node in the tree so the best point is the root */
    if (root->left == NULL && root->right == NULL) {
#if 0
        printf("tree has no left and right branches\n");
        printf("best_index = %d\n",*best_index);
        printf("best_distance = %10.7f\n",*best_dist);
        printf("root %d is terminating\n",root->indx );

#endif

        return;

    }

    /* There is only left branch in the tree */
    if (root->left != NULL && root->right == NULL) {
#if 0
        printf("tree has only left branch\n");
#endif

        find_max_with_index(best_dist,K,&max_dist,&max_dist_index);

        kdNode *left = root->left;

        double *dbestleft= new double[K];
        int *indexbestleft= new int[K];


        int inext = (i + 1) % dim;
        nearest(left, p,K,inext, indexbestleft, dbestleft,dim);

#if 0
        printf("best %d points in the left branch\n",K);
        for(int i=0; i<K; i++){

            printf("indx = %d, dist = %10.7f\n",indexbestleft[i],dbestleft[i]);
        }
#endif


        for(int j=0; j<K; j++){ /* check each new point */

            if (dbestleft[j] < max_dist){ /* a better point is found on the left branch*/
#if 0
                printf("dbestleft[%d] = %10.7f is better than %10.7f\n", j, dbestleft[j],max_dist );
#endif

                best_dist[max_dist_index] = dbestleft[j];
                best_index[max_dist_index] = indexbestleft[j];

                /* update max_dist and max_dist_index*/
                find_max_with_index(best_dist,K,&max_dist,&max_dist_index);
#if 0
                printf("max_dist = %10.7f\n", max_dist);
#endif
            }

        }

        delete[] dbestleft;
        delete[] indexbestleft;
#if 0
        printf("root %d is terminating\n",root->indx );
#endif
        return;
    }


    /* There is only right branch in the tree */
    if (root->left == NULL && root->right != NULL) {
#if 0
        printf("tree has only right branch\n");
#endif

        find_max_with_index(best_dist,K,&max_dist,&max_dist_index);

        kdNode *right = root->right;
        double *dbestright= new double[K];
        int *indexbestright= new int[K];

        int inext = (i + 1) % dim;
        nearest(right, p, K,inext, indexbestright, dbestright,dim);

#if 0
        printf("best %d points in the right branch\n",K);
        for(int i=0; i<K; i++){

            printf("indx = %d, dist = %10.7f\n",indexbestright[i],dbestright[i]);
        }
#endif

        for(int j=0; j<K; j++){ /* check each new point */

            if (dbestright[j] < max_dist){ /* a better point is found on the right */

#if 0
                printf("dbestright[%d] = %10.7f is better than %10.7f\n", j, dbestright[j],max_dist );
#endif
                best_dist[max_dist_index] = dbestright[j];
                best_index[max_dist_index] = indexbestright[j];

                /* update max_dist and max_dist_index*/
                find_max_with_index(best_dist,K,&max_dist,&max_dist_index);
#if 0
                printf("max_dist = %10.7f\n", max_dist);
#endif
            }

        }

        delete[] dbestright;
        delete[] indexbestright;
#if 0
        printf("root %d is terminating\n",root->indx );
#endif
        return;

    }

    /* both branches exist */
    if (root->left != NULL && root->right != NULL) {

#if 0
        printf("tree has two branches\n");
#endif

        double division = root->x(i);

        /* if the point is on the left region*/
        if(p->x(i) < division){

#if 0
            printf("the point is on the left : p->x(%d) = %10.7f < %10.7f\n",i,p->x(i),division);
#endif

            /* first find the best in the left branch */
            kdNode *left = root->left;
            double *dbestleft= new double[K];
            int *indexbestleft= new int[K];


            find_max_with_index(best_dist,K,&max_dist,&max_dist_index);

            int inext = (i + 1) % dim;
            nearest(left, p, K,inext, indexbestleft, dbestleft,dim);

            for(int j=0; j<K; j++){ /* check each new point */

                if (dbestleft[j] < max_dist){ /* a better point is found on the left branch*/

                    best_dist[max_dist_index] = dbestleft[j];
                    best_index[max_dist_index] = indexbestleft[j];

                    /* update max_dist and max_dist_index*/
                    find_max_with_index(best_dist,K,&max_dist,&max_dist_index);

                }

            }

            delete[] dbestleft;
            delete[] indexbestleft;


            if (max_dist > (division-p->x(i)) ){ /* if it is possible that a closer point may exist in the right branch */

                kdNode *right = root->right;
                double *dbestright= new double[K];
                int *indexbestright= new int[K];

                int inext = (i + 1) % dim;
                nearest(right, p, K,inext, indexbestright, dbestright,dim);

#if 0
                printf("best %d points in the right branch\n",K);
                for(int i=0; i<K; i++){

                    printf("indx = %d, dist = %10.7f\n",indexbestright[i],dbestright[i]);
                }
#endif

                for(int j=0; j<K; j++){ /* check each new point */

                    if (dbestright[j] < max_dist){ /* a better point is found on the right */

                        best_dist[max_dist_index] = dbestright[j];
                        best_index[max_dist_index] = indexbestright[j];

                        /* update max_dist and max_dist_index*/
                        find_max_with_index(best_dist,K,&max_dist,&max_dist_index);

                    }

                }

                delete[] dbestright;
                delete[] indexbestright;


            }
#if 0
            printf("root %d is terminating\n",root->indx );
#endif
            return;

        }
        else{ /* the point is on the right */
#if 0
            printf("the point is on the right : p->x(%d) = %10.7f > %10.7f\n",i,p->x(i),division);
#endif

            /* first find the best in the right branch */
            kdNode *right = root->right;
            double *dbestright= new double[K];
            int *indexbestright= new int[K];

            find_max_with_index(best_dist,K,&max_dist,&max_dist_index);

#if 0
            printf("max_dist = %10.7f\n", max_dist);
#endif
            int inext = (i + 1) % dim;
            nearest(right, p, K,inext, indexbestright, dbestright,dim);

#if 0
            printf("best %d points in the right branch\n",K);
            for(int i=0; i<K; i++){

                printf("indx = %d, dist = %10.7f\n",indexbestright[i],dbestright[i]);
            }
#endif

            for(int j=0; j<K; j++){ /* check each new point */

                if (dbestright[j] < max_dist){ /* a better point is found on the right */

                    best_dist[max_dist_index] = dbestright[j];
                    best_index[max_dist_index] = indexbestright[j];

                    /* update max_dist and max_dist_index*/
                    find_max_with_index(best_dist,K,&max_dist,&max_dist_index);

                }

            }

            delete[] dbestright;
            delete[] indexbestright;


            if (max_dist > (p->x(i)-division) ){ /* if it is possible that a closer point may exist in the left branch */

                kdNode *left = root->left;
                double *dbestleft= new double[K];
                int *indexbestleft= new int[K];

                int inext = (i + 1) % dim;
                nearest(left, p,K,inext, indexbestleft, dbestleft,dim);

                for(int j=0; j<K; j++){ /* check each new point */

                    if (dbestleft[j] < max_dist){ /* a better point is found on the left branch*/


                        best_dist[max_dist_index] = dbestleft[j];
                        best_index[max_dist_index] = indexbestleft[j];

                        /* update max_dist and max_dist_index*/
                        find_max_with_index(best_dist,K,&max_dist,&max_dist_index);

                    }

                }

                delete[] dbestleft;
                delete[] indexbestleft;

            }
#if 0
            printf("root %d is terminating\n",root->indx );
#endif
            return;


        }


    }

}

void build_kdNodeList(mat data,kdNode *kdNodeVec ){

    int dim = data.n_rows;

    for(int i=0; i<dim; i++){

        kdNodeVec[i].x = data.row(i);
        kdNodeVec[i].indx = i;
        kdNodeVec[i].left = NULL;
        kdNodeVec[i].right = NULL;

    }



}



void kd_tree_test(void){

    int number_of_points= 20;
    mat X(number_of_points,2);


    for (int i=0; i<number_of_points; i++){
        X(i,0)=RandomDouble(0,1.0);
        X(i,1)=RandomDouble(0,1.0);
    }


    X.print();

    kdNode *kdNodeVec = new kdNode[number_of_points];

    build_kdNodeList(X,kdNodeVec );


    kdNode * root = make_tree(kdNodeVec, number_of_points, 0);

#if 1

    printf("Root Node = %d\n",root->indx);


    for(int i=0; i<number_of_points;i++){
        printf("Node = %d\n",kdNodeVec[i].indx);
        kdNodeVec[i].x.print();
        if(kdNodeVec[i].left !=NULL){

            printf("Left = %d \n",kdNodeVec[i].left->indx);
        }

        if(kdNodeVec[i].right !=NULL){

            printf("Right = %d \n",kdNodeVec[i].right->indx);
        }
        printf("\n");

    }
#endif

    kdNode * p = new kdNode;
    rowvec xinp(2);


    xinp(0)=0.34;
    xinp(1)=0.24;

    xinp.print();

    p->x = xinp;

    int best_index=-1;
    double best_dist = LARGE;

    nearest(root, p, 1,0, &best_index, &best_dist, 2);

    printf("nearest point search with the kd tree = \n");
    printf("index = %d\n",best_index);
    printf("distance = %10.7f\n",best_dist);
    X.row(best_index).print();

    int indices=-1;
    double min_distance=0.0;
    findKNeighbours(X, xinp, 1, &min_distance,&indices);

    printf("nearest point with brute force search = \n");
    printf("index = %d\n",indices);
    printf("distance = %10.7f\n",min_distance);
    X.row(indices).print();

}

void kd_tree_test2(void){


    int dim = RandomInt(1,4);
    int number_of_points= RandomInt(500,1000);

    mat X(number_of_points,dim);


    X.randu();

#if 1
    printf("X = \n");
    X.print();
#endif
    int K=4;

    int *best_index= new int[K];
    double *best_dist = new double[K];

    int *best_indexKNN= new int[K];
    double *best_distKNN = new double[K];


    kdNode *kdNodeVec = new kdNode[number_of_points];

    build_kdNodeList(X,kdNodeVec );

    kdNode * root = make_tree(kdNodeVec, number_of_points, 0);


    int number_of_trials = RandomInt(1,1000);

    for(int i=0; i<number_of_trials; i++){
        rowvec p(dim);

        for(int j=0; j<dim; j++){

            p(j) = RandomDouble(0,1.0);

        }
        printf("p = \n");
        p.print();
        kdNode *xin = new kdNode;

        xin->x = p;

        nearest(root, xin, K,0, best_index, best_dist, dim);

        for(int i=0; i<K; i++){

            printf("index = %d dist = %10.7f \n\n",best_index[i],best_dist[i]);
            X.row(best_index[i]).print();

        }

        findKNeighbours(X, p, K, best_distKNN,best_indexKNN);

        for(int i=0; i<K; i++){

            printf("index = %d dist = %10.7f \n",best_indexKNN[i],best_distKNN[i]);
            X.row(best_index[i]).print();

        }

        if (check_if_lists_are_equal(best_index, best_indexKNN, K) == 0){

            printf("There is a mismatch in lists\n");
            exit(1);
        }


    }
    delete[] best_index;
    delete[] best_dist;
    delete[] best_indexKNN;
    delete[] best_distKNN;

}


