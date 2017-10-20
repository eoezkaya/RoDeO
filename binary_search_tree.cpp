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
        int dim){

    /* default root is the best initially*/


    double d = fabs(root->x(i) - p->x(i));

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

        double max_dist = -LARGE;
        int max_dist_index = -1;


        for(int j=0; j<K; j++){

            if(best_dist[j] > max_dist){
                max_dist = best_dist[j];
                max_dist_index = j;

            }
        }

        kdNode *left = root->left;
        double *dbestleft= new double[K];
        int *indexbestleft= new int[K];



        /* return the K best of the left branch */
        nearest(left, p,i, &indexbestleft, &dbestleft,dim);


        for(int j=0; j<K; j++){ /* check each new point */

            if (dbestleft[j] < max_dist){ /* a better point is found on the left */


                best_dist[max_dist_index] = dbestleft[j];
                best_index[max_dist_index] = indexbestleft[j];


                max_dist = -LARGE;
                max_dist_index = -1;

                /* update max distance */
                for(int k=0; k<K; k++){
                    dbestleft[k]=LARGE;
                    indexbestleft[k]=-1;
                }


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


        double max_dist = -LARGE;
        int max_dist_index = -1;


        for(int j=0; j<K; j++){

            if(best_dist[j] > max_dist){
                max_dist = best_dist[j];
                max_dist_index = j;

            }
        }



        kdNode *right = root->right;


        double *dbestright= new double[K];
        int *indexbestright= new int[K];

        nearest(right, p, i, &indexbestright, &dbestright,dim);



        for(int j=0; j<K; j++){ /* check each new point */

            if (dbestright[j] < max_dist){ /* a better point is found on the left */


                max_dist = -LARGE;
                max_dist_index = -1;

                /* update max distance */
                for(int k=0; k<K; k++){
                    dbestright[k]=LARGE;
                    indexbestright[k]=-1;
                }


                best_dist[max_dist_index] = dbestright[j];
                best_index[max_dist_index] = indexbestright[j];

                max_dist = -LARGE;
                max_dist_index = -1;

                /* update max distance */
                for(int k=0; k<K; k++){
                    dbestleft[k]=LARGE;
                    indexbestleft[k]=-1;
                }


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
            double dbestleft=LARGE;
            int indexbestleft=-1;


            nearest(left, p, i, &indexbestleft, &dbestleft,dim);

            if(dbestleft<*best_dist){
#if 0
                printf("a better point is found on the left since %10.7f < %10.7f\n",dbestleft,*best_dist);
#endif
                *best_dist = dbestleft;
                *best_index = indexbestleft;
            }


            if (dbestleft > (division-p->x(i)) ){ /* if it is possible that a closer point may exist in the right branch */

                kdNode *right = root->right;
                double dbestright=LARGE;
                int indexbestright=-1;

                int inext = (i + 1) % dim;
                nearest(right, p, inext, &indexbestright, &dbestright,dim);

                if (dbestright<*best_dist){
#if 0
                    printf("a better point is found on the right since %10.7f < %10.7f\n",dbestright,*best_dist);
#endif
                    *best_dist = dbestright;
                    *best_index = indexbestright;


                }


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
            double dbestright = LARGE;
            int indexbestright = -1;

            int inext = (i + 1) % dim;
            nearest(right, p, inext, &indexbestright, &dbestright,dim);

            if(dbestright<*best_dist){
#if 0
                printf("a better point is found on the right since %10.7f < %10.7f\n",dbestright,*best_dist);
#endif
                *best_dist = dbestright;
                *best_index = indexbestright;

            }


            if (dbestright > (p->x(i)-division) ){ /* if it is possible that a closer point may exist in the left branch */

                kdNode *left = root->left;
                double dbestleft=LARGE;
                int indexbestleft=-1;

                int inext = (i + 1) % dim;
                nearest(left, p, inext, &indexbestleft, &dbestleft,dim);

                if (dbestleft<*best_dist){
#if 0
                    printf("a better point is found on the left since %10.7f < %10.7f\n",dbestleft,*best_dist);
#endif
                    *best_dist = dbestleft;
                    *best_index = indexbestleft;


                }

            }
#if 0
            printf("root %d is terminating\n",root->indx );
#endif
            return;


        }


    }

}







void binary_tree_test(void){

    int number_of_points= 20;
    mat X(number_of_points,4);


    for (int i=0; i<number_of_points; i++){
        X(i,0)=RandomDouble(0,1.0);
        X(i,1)=RandomDouble(0,1.0);
        X(i,2)=RandomDouble(0,1.0);
        X(i,3)=RandomDouble(0,1.0);
    }


    X.print();

    binaryTreeNode *binaryNodeVec = new binaryTreeNode[number_of_points];

    build_binaryNodeList(X,binaryNodeVec );


    binaryTreeNode * root = make_tree(binaryNodeVec, number_of_points, 0);

#if 1

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



}

