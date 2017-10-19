#include "kd_tree.hpp"
#include "Rodeo_macros.hpp"

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
    kdNode *end   = &root[len-1];

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

#if 1
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

void nearest(kdNode *root, kdNode *p, int i, int* best_index, double *best_dist, int dim){

    /* default root is the best */

    double d = dist(root, p);
    *best_index = root->indx;
    *best_dist =d;

#if 1
    printf("\n\nroot = %d\n",root->indx);
    printf("distance to root = %10.7f\n",d);
#endif


    /* there is only one  node in the tree so the best point is the root */
    if (root->left == NULL && root->right == NULL) {
#if 1
        printf("tree has no left and right branches\n");
        printf("best_index = %d\n",*best_index);
        printf("best_distance = %10.7f\n",*best_dist);
#endif

        return;

    }

    /* There is only left branch in the tree */
    if (root->left != NULL && root->right == NULL) {
#if 1
        printf("tree has only left branch\n");
#endif

        kdNode *left = root->left;
        double dbestleft;
        int indexbestleft;

        i = (i + 1) % dim;
        nearest(left, p, i, &indexbestleft, &dbestleft,dim);

#if 1
        printf("best_distance left= %10.7f\n",dbestleft);
#endif


        if (dbestleft<d){ /* a better point is found on the left */

            *best_dist = dbestleft;
            *best_index = indexbestleft;
            return;

        }

        return;
    }


    /* There is only right branch in the tree */
    if (root->left == NULL && root->right != NULL) {
#if 1
        printf("tree has only right branch\n");
#endif


        kdNode *right = root->right;
        double dbestright;
        int indexbestright;

        i = (i + 1) % dim;
        nearest(right, p, i, &indexbestright, &dbestright,dim);

#if 1
        printf("best_distance right= %10.7f\n",dbestright);
#endif

        if (dbestright<d){

            *best_dist = dbestright;
            *best_index = indexbestright;
            return;

        }

        return;

    }

    /* both branches exist */
    if (root->left != NULL && root->right != NULL) {

#if 1
        printf("tree has two branches\n");
#endif

        double division = root->x(i);

        /* if the point is on the left region*/
        if(p->x(i) < division){

#if 1
            printf("the point is on the left\n");
#endif

            /* first find the best in the left branch */
            kdNode *left = root->left;
            double dbestleft;
            int indexbestleft;

            i = (i + 1) % dim;
            nearest(left, p, i, &indexbestleft, &dbestleft,dim);

            if(dbestleft<d){
                *best_dist = dbestleft;
                *best_index = indexbestleft;
            }


            if (*best_dist> (division-p->x(i)) ){ /* if it is possible that a closer point may exist in the right branch */

                kdNode *right = root->right;
                double dbestright;
                int indexbestright;

                i = (i + 1) % dim;
                nearest(right, p, i, &indexbestright, &dbestright,dim);

                if (dbestright<*best_dist){

                    *best_dist = dbestright;
                    *best_index = indexbestright;
                    return;

                }


            }

            return;

        }
        else{ /* the point is on the right */
#if 1
            printf("the point is on the right\n");
#endif

            /* first find the best in the right branch */
            kdNode *right = root->right;
            double dbestright;
            int indexbestright;

            i = (i + 1) % dim;
            nearest(right, p, i, &indexbestright, &dbestright,dim);

            if(dbestright<d){
                *best_dist = dbestright;
                *best_index = indexbestright;

            }


            if (*best_dist> (division-p->x(i)) ){ /* if it is possible that a closer point may exist in the left branch */

                kdNode *left = root->left;
                double dbestleft;
                int indexbestleft;

                i = (i + 1) % dim;
                nearest(left, p, i, &indexbestleft, &dbestleft,dim);

                if (dbestleft<d){

                    *best_dist = dbestleft;
                    *best_index = indexbestleft;
                    return;

                }

            }

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

    int number_of_points= 10;
    mat X(number_of_points,2);

    X.randu();

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

    printf("here\n");

    int best_index=-1;
    double best_dist = 0.0;

    nearest(root, p, 0, &best_index, &best_dist, 2);

    printf("nearest point = \n");
    printf("index = %d\n",best_index);
    X.row(best_index).print();


}




