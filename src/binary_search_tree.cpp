#include "binary_search_tree.hpp"
#include "Rodeo_macros.hpp"
#include "auxilliary_functions.hpp"

#include <armadillo>

using namespace arma;



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>



double find_median(binaryTreeNode *nodelist, int len, int idx)
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





binaryTreeNode* make_tree(binaryTreeNode*root, int len, int indx){

#if 0
    printf("make_tree with root = %d and length = %d\n", root->indx, len);
#endif

    if (len==0) {

        return NULL;
    }


    binaryTreeNode *start = &root[0];


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



    binaryTreeNode *binaryNodeVectemp = new binaryTreeNode[len];

    count=0;
    for (std::vector<int>::iterator it = left_indices.begin() ; it != left_indices.end(); ++it){
        binaryNodeVectemp[count].x = root[*it].x;
        binaryNodeVectemp[count].indx = root[*it].indx;
        count++;

    }

    binaryNodeVectemp[count].x = root[median_index].x;
    binaryNodeVectemp[count].indx = root[median_index].indx;
    median_index = count;

    count++;

    for (std::vector<int>::iterator it = right_indices.begin() ; it != right_indices.end(); ++it){
        binaryNodeVectemp[count].x = root[*it].x;
        binaryNodeVectemp[count].indx = root[*it].indx;

        count++;

    }

    for(int i=0; i<len; i++){

        root[i].x    = binaryNodeVectemp[i].x;
        root[i].indx = binaryNodeVectemp[i].indx;
#if 0
        printf("Node index = %d\n",root[i].indx);
        root[i].x.print();
#endif
    }

    binaryTreeNode *n= &root[median_index];


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

    delete[] binaryNodeVectemp;

    return n;
}


void build_binaryNodeList(mat data,binaryTreeNode *binaryNodeList ){

    int dim = data.n_rows;

    for(int i=0; i<dim; i++){

        binaryNodeList[i].x = data.row(i);
        binaryNodeList[i].indx = i;
        binaryNodeList[i].left = NULL;
        binaryNodeList[i].right = NULL;

    }

}

void nearest(binaryTreeNode *root,
        binaryTreeNode *p,
        int K,
        int i,
        int *best_index,
        double *best_dist,
        int dim,
        int *howmanycalls = NULL){

#if 1
    *howmanycalls = *howmanycalls+1;
#endif


    for(int i=0; i<K; i++){

        best_dist[i] = LARGE;
        best_index[i] = -1;
    }


    /* default root is the best initially*/

    double max_dist = root->indx;

    double d = fabs(root->x(i) - p->x(i));
    int max_dist_index = d;



#if 0
    printf("\n\nroot = %d\n",root->indx);
    root->x.print();
    printf("distance to root = %10.7f (| %10.7f - %10.7f|   ) \n",d,root->x(i),p->x(i)  );
#endif

    /* initialize  */
    best_index[0] = root->indx;
    best_dist[0]  = d;

        /* there is only one  node in the tree so the best point is the root */
    if (root->left == NULL && root->right == NULL) {
#if 0
        printf("tree has no left and right branches\n");
        printf("best_index = %d\n",best_index[0]);
        printf("best_distance = %10.7f\n",best_dist[0]);
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

#if 0
        printf("max_dist = %10.7f\n", max_dist);
#endif

        binaryTreeNode *left = root->left;
        double *dbestleft= new double[K];
        int *indexbestleft= new int[K];



        /* return the K best of the left branch */
        nearest(left, p,K,i, indexbestleft, dbestleft,dim,howmanycalls);
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
#if 0
        printf("max_dist = %10.7f\n", max_dist);
#endif
        binaryTreeNode *right = root->right;

        double *dbestright= new double[K];
        int *indexbestright= new int[K];

        nearest(right, p, K,i, indexbestright, dbestright,dim,howmanycalls);

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
            binaryTreeNode *left = root->left;

            double *dbestleft= new double[K];
            int *indexbestleft= new int[K];


            find_max_with_index(best_dist,K,&max_dist,&max_dist_index);

#if 0
            printf("max_dist = %10.7f\n", max_dist);
#endif

            nearest(left, p,K, i, indexbestleft, dbestleft,dim,howmanycalls);
#if 0
            printf("best %d points in the left branch\n",K);
            for(int i=0; i<K; i++){

                printf("indx = %d, dist = %10.7f\n",indexbestleft[i],dbestleft[i]);
            }
#endif
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

                binaryTreeNode *right = root->right;

                double *dbestright= new double[K];
                int *indexbestright= new int[K];


                nearest(right, p, K,i, indexbestright, dbestright,dim,howmanycalls);

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
            binaryTreeNode *right = root->right;

            double *dbestright= new double[K];
            int *indexbestright= new int[K];

            find_max_with_index(best_dist,K,&max_dist,&max_dist_index);

#if 0
            printf("max_dist = %10.7f\n", max_dist);
#endif
            nearest(right, p, K,i, indexbestright, dbestright,dim,howmanycalls);

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

            if ( max_dist > (p->x(i)-division) ){ /* if it is possible that a closer point may exist in the left branch */

                binaryTreeNode *left = root->left;
                double *dbestleft= new double[K];
                int *indexbestleft= new int[K];


                /* return the K best of the left branch */
                nearest(left, p,K,i, indexbestleft, dbestleft,dim,howmanycalls);
#if 0
                printf("best %d points in the left branch\n",K);
                for(int i=0; i<K; i++){

                    printf("indx = %d, dist = %10.7f\n",indexbestleft[i],dbestleft[i]);
                }
#endif

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

void binary_tree_test(void){


    int number_of_points= RandomInt(100,5000);

    int dim = RandomInt(2,5);

    int which_coordinate = RandomInt(0,dim-1);

    int K=4;

    double xs = -10.0;
    double xe = 100.0;

    int *best_index= new int[K];
    double *best_dist = new double[K];

    int *best_indexKNN= new int[K];
    double *best_distKNN = new double[K];


    mat X(number_of_points,dim);


    for (int i=0; i<number_of_points; i++){

        for(int j=0; j<dim;j++){

            X(i,j)=RandomDouble(xs,xe);

        }
    }


#if 0
    X.print();
#endif

    binaryTreeNode *binaryNodeVec = new binaryTreeNode[number_of_points];

    build_binaryNodeList(X,binaryNodeVec );


    binaryTreeNode * root = make_tree(binaryNodeVec, number_of_points, which_coordinate);

#if 0

    printf("Root Node = %d\n",root->indx);


    for(int i=0; i<number_of_points;i++){
        printf("Node = %d\n",binaryNodeVec[i].indx);
        binaryNodeVec[i].x.print();
        if(binaryNodeVec[i].left !=NULL){

            printf("Left = %d \n",binaryNodeVec[i].left->indx);
        }

        if(binaryNodeVec[i].right !=NULL){

            printf("Right = %d \n",binaryNodeVec[i].right->indx);
        }
        printf("\n");

    }
#endif

    int number_of_trials = RandomInt(1,1000);


    int sum_number_of_calls = 0;
    for(int i=0; i<number_of_trials; i++){
        rowvec p(dim);

        for(int j=0; j<dim; j++){

            p(j) = RandomDouble(0,1.0);

        }
#if 1
        printf("\np = \n");
        p.print();
#endif
        binaryTreeNode *xin = new binaryTreeNode;

        xin->x = p;


        int howmanycalls = 0;

        nearest(root, xin, K,which_coordinate, best_index, best_dist, dim, &howmanycalls);

        sum_number_of_calls+= howmanycalls;

        printf("nearest point search with the binary search tree (coordinate = %d) \n", which_coordinate);
        printf("Number of calls = %d\n", howmanycalls);
        printf("Number of calls (avg)= %d\n", sum_number_of_calls/(i+1));
        for(int i=0; i<K; i++){

            printf("index = %d dist = %10.7f \n\n",best_index[i],best_dist[i]);
            X.row(best_index[i]).print();

        }

        findKNeighbours(X, p, K,&which_coordinate , best_distKNN, best_indexKNN, 1);

        printf("nearest point search with the brute force search (coordinate = %d) \n",which_coordinate);

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

