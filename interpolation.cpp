#include "interpolation.hpp"
#include "Rodeo_macros.hpp"
#include "auxilliary_functions.hpp"
#include <armadillo>

using namespace arma;



#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

namespace tableInterpolation{

/* print an element of the lookup table indexed by indx */
void printTableElement(IntLookupTable* table, int indx, int how) {
    int i;
    double x;

    if (how == 2){
        /*print using original values */
        fprintf(stdout,"%5d ",indx);
        for (i = 0; i < table->columnsize; i++) {
            x= table->data[i][indx]*(table->xmax[i]-table->xmin[i])+table->xmin[i];
            fprintf(stdout,"%10.7f ", x);
        }

        printf("\n");

        /*print using normalized values */
        for (i = 0; i < table->columnsize; i++) {

            fprintf(stdout,"%d %10.7f", indx, table->data[i][indx]);
        }

        fprintf(stdout,"\n");
    }


    if (how == 0){
        /*print using original values */
        fprintf(stdout,"%5d ",indx);
        for (i = 0; i < table->columnsize; i++) {
            x= table->data[i][indx]*(table->xmax[i]-table->xmin[i])+table->xmin[i];
            fprintf(stdout,"%10.7f ", x);
        }
        fprintf(stdout,"\n");

    }

    if (how == 1){


        /*print using normalized values */
        for (i = 0; i < table->columnsize; i++) {

            fprintf(stdout,"%d %10.7f", indx, table->data[i][indx]);
        }

        fprintf(stdout,"\n");
    }



}

void printTable(IntLookupTable* table, int how) {

    for(int i=0; i<table->rowsize; i++){

        printTableElement(table, i, how) ;

    }


}



/** brute force KNeighbours search with given index array for distance computation
 *
 *
 */

void findKNeighbours(IntLookupTable* table,
        double *xinp,
        int K,
        int *input_indx ,
        double* min_dist,
        int *indices,
        int number_of_independent_variables){

    int number_of_points= table->rowsize;


    for(int i=0; i<K; i++){

        min_dist[i]= LARGE;
        indices[i]= -1;
    }



    for(int i=0; i<number_of_points; i++){ /* for each data point */

        rowvec x(number_of_independent_variables);

        for(int j=0; j<number_of_independent_variables; j++){

            x(j) = table->data[input_indx[j]][i];

        }


        rowvec xdiff(number_of_independent_variables);

        for(int j=0; j<number_of_independent_variables; j++){

            xdiff(j) = x(j)- xinp[j];

        }



#if 0
        printf("xdiff = \n");
        xdiff.print();
#endif

        double distance = L2norm(xdiff, xdiff.size());
#if 0
        printf("distance = %10.7f\n", distance);
#endif

        double worst_distance = -LARGE;
        int worst_distance_index = -1;


        find_max_with_index(min_dist, K, &worst_distance, &worst_distance_index);


        /* a better point is found */
        if(distance < worst_distance){

            min_dist[worst_distance_index]= distance;
            indices[worst_distance_index] = i;

        }

    }

}




/** compute the distance of the point x w.r.t. to a cluster mean xmean .
 *
 * @param[in] xmean    mean center coordinates
 * @param[in] x        point coordinates
 * @param[in] variable_index  array of indices
 * @param[in] number_of_active_vars  number of active variables in the distance computation
 *
 */
double ComputeDistanceToXmeani(double *xmean,
        double *x,
        int *variable_index,
        int number_of_active_vars){

    double sum=0.0;

    for(int i=0;i<number_of_active_vars;i++) {

        int ind = variable_index[i];
        sum+=(xmean[ind]-x[ind])*(xmean[ind]-x[ind]);

    }


    return sqrt(sum);
}


/** enlarge the size of a cluster if necessary.
 *
 * @param[in] table  pointer to lookup table
 * @param[in] index of the cluster to be enlarged
 *
 */
void EnlargeCluster(IntLookupTable* table, int which_cluster){

    int ideal_cluster_size=50;

    /* first get the old size of the cluster */
    int old_size= table->cluster_sizes[which_cluster];


    /* allocate buffer array */
    int *buffer =  new int[old_size];

    /* save the indices in buffer */
    for(int i=0; i<old_size; i++){

        buffer[i]=table->cluster_indices[which_cluster][i];

    }

    delete[] table->cluster_indices[which_cluster];

    table->cluster_indices[which_cluster] = new int[old_size+ideal_cluster_size];

    /* increment the max_size */
    table->cluster_max_sizes[which_cluster]=old_size+ideal_cluster_size;

    /* copy the indices back */
    for(int i=0;i< old_size; i++) {

        table->cluster_indices[which_cluster][i]=buffer[i];

    }

    /* delete buffer */

    delete[] buffer;

}

/** performs K-means clustering to partition the lookup table data.
 *
 * @param[in] table  pointer to lookup table
 * @param[in] ideal_cluster_size
 * @param[in] variable_index   indices of variables that are considered in the clustering
 * @param[in] number_of_active_vars   dim(variable_index)
 */
int KmeansClustering(IntLookupTable *table,
        int ideal_cluster_size,
        int *variable_index,
        int number_of_active_vars){

    int max_it=2000,it_count=0;

    srand (time(NULL));

    int number_of_points= table->rowsize;
    int number_of_clusters=number_of_points/ideal_cluster_size;

    table->number_of_clusters=number_of_clusters;

    double tolerance = EPSILON_SINGLE;

    /* if number of points is too few do nothing */
    if(number_of_clusters <=0) {

        table->number_of_clusters = 1;

        table->cluster_sizes = new int[1];

        table->cluster_sizes[0]= table->rowsize;

        table->cluster_indices = new int*[1];

        table->cluster_indices[0] = new int[table->rowsize];



        for(int i=0;i<table->rowsize;i++) {

            table->cluster_indices[0][i]=i;

        }

        return 0;
    }

    /* allocate array for cluster means (must be standart type!) */


    table->cluster_means = new double*[number_of_clusters];

    for(int i=0; i<table->number_of_clusters; i++){

        table->cluster_means[i] = new double[table->columnsize];


    }

    for(int i=0; i<number_of_clusters; i++){

        for(int j=0; j<table->columnsize; j++){

            table->cluster_means[i][j]=RandomDouble(0.0, 1.0);

        }

    }

    /* difference in cluster mean coordinates */


    double *xmean_differences = new double[number_of_clusters];

    double *distance= new double[number_of_clusters];

    table->cluster_sizes = new int[number_of_clusters];


    /* initially all the clusters are empty */

    for (int i=0;i<number_of_clusters;i++) {

        table->cluster_sizes[i]=0;
    }

    table->cluster_max_sizes = new int[number_of_clusters];



    /* initialize maximum cluster capacities */

    for(int i=0;i<number_of_clusters;i++){

        table->cluster_max_sizes[i]=ideal_cluster_size;
    }

    table->cluster_indices = new int*[number_of_clusters];



    for(int i=0;i<number_of_clusters;i++) {

        table->cluster_indices[i] = new int[ideal_cluster_size];

    }

    /* temp x vector */

    double *x = new double[table->columnsize];


    /* temp vector for cluster means, must be standart data type! */

    double **xmean0 = new double*[number_of_clusters];


    for(int i=0;i<number_of_clusters;i++){

        xmean0[i] = new double[table->columnsize];


    }

    while(1){


        /* save cluster means in xmean*/
        for(int i=0;i<number_of_clusters;i++){

            for(int j=0;j<table->columnsize;j++){

                /* jth component of ith xmean */
                xmean0[i][j]=table->cluster_means[i][j];
            }

        }

        if(it_count > 10){  /* after 10 iterations  */

            for(int i=0;i<number_of_clusters;i++){ /* for each cluster */

                if(table->cluster_sizes[i] == 0){  /* if cluster is empty */


                    /* find the cluster with maximum number of elements */
                    int cluster_with_maximum_elements=-1;
                    int max_elements=-1;

                    for(int j=0;j<number_of_clusters;j++){

                        if( table->cluster_sizes[j] > max_elements){

                            max_elements=table->cluster_sizes[j];
                            cluster_with_maximum_elements = j;
                        }

                    }

                    int random_cluster_member1 = rand() % max_elements;
                    int random_cluster_member2;

                    while(1){

                        random_cluster_member2 = rand() % max_elements;

                        if(random_cluster_member1 != random_cluster_member2) {

                            break;
                        }
                    }

                    int table_index1 = table->cluster_indices[cluster_with_maximum_elements][random_cluster_member1];


                    int table_index2 = table->cluster_indices[cluster_with_maximum_elements][random_cluster_member2];


                    for(int j=0;j<table->columnsize;j++){

                        /* jth component of ith xmean */
                        table->cluster_means[i][j]=RandomDouble(table->data[j][table_index1], table->data[j][table_index2] );

                    }


                    break;
                }

            }
        }


        if(it_count > max_it) {

            break;

        }


        for(int k=0;k<number_of_clusters;k++) {

            table->cluster_sizes[k]=0;

        }


        /* for each point in the table */
        for(int i=0;i<table->rowsize;i++){

            for(int k=0;k<table->columnsize;k++) {

                x[k]=table->data[k][i];
            }

            for(int j=0; j<number_of_clusters;j++){ /* for each cluster center */


                /* compute distance of the ith point to the jth cluster center */

                distance[j]= ComputeDistanceToXmeani(table->cluster_means[j],
                        x,
                        variable_index,
                        number_of_active_vars);


            } /* end of j loop */


            /* find the minimum distance */
            double min_distance=LARGE;
            int index_of_min_dist=0;

            for(int j=0; j<number_of_clusters;j++){

                if(distance[j] < min_distance){

                    min_distance=distance[j];
                    index_of_min_dist=j;
                }


            }


            /* point i belongs to cluster index_of_min_dist */

            table->cluster_indices[index_of_min_dist][table->cluster_sizes[index_of_min_dist]]=i;
            table->cluster_sizes[index_of_min_dist]++;



            if(table->cluster_sizes[index_of_min_dist] ==  table->cluster_max_sizes[index_of_min_dist]-1  ){

                EnlargeCluster(table, index_of_min_dist);

            }


        } /* end of i loop */


        /* initialize xmean to zero */
        for(int k=0;k<number_of_clusters;k++) {

            if(table->cluster_sizes[k] !=0){

                for(int j=0;j<table->columnsize;j++) {

                    table->cluster_means[k][j]=0;

                }
            }

        }

        /* for each cluster in the table */
        for(int i=0;i<number_of_clusters;i++){

            for(int k=0;k<table->cluster_sizes[i];k++) {

                for(int j=0;j<table->columnsize;j++) {

                    table->cluster_means[i][j]+=table->data[j][table->cluster_indices[i][k]];

                }
            }



        }

        for(int k=0;k<number_of_clusters;k++) {

            if(table->cluster_sizes[k] !=0 ){

                for(int j=0;j<table->columnsize;j++) {

                    table->cluster_means[k][j]/=table->cluster_sizes[k];

                }
            }

        }

        int flag=0;

        for(int i=0;i<number_of_clusters;i++){


            xmean_differences[i]=ComputeDistanceToXmeani(table->cluster_means[i],
                    xmean0[i],
                    variable_index,
                    number_of_active_vars);

            if (xmean_differences[i] > tolerance) {

                flag=1;

            }
        }



        if(flag == 0) {

            break;

        }

        it_count++;
    } /* end of while loop*/

    /* if there is a too small cluster merge it to the closest one */



    for(int k=0;k<number_of_clusters;k++) {   /* for each cluster */

        if(table->cluster_sizes[k] < 10) { /* if cluster size is smaller than a certain number */


            for(int i=0;i<table->cluster_sizes[k];i++){  /* for each data point in the cluster */

                int table_index = table->cluster_indices[k][i];  /* get the table index */

                for(int j=0;j<table->columnsize;j++) {

                    x[j]=table->data[j][table_index];
                }

                double min_distance=LARGE;
                int index_of_min_dist=0;

                for(int j=0; j<number_of_clusters;j++){ /* for each cluster center */

                    if(k!=j ){
                        /* compute distance to the jth cluster center */

                        double dist= ComputeDistanceToXmeani(table->cluster_means[j],
                                x,
                                variable_index,
                                number_of_active_vars);


                        if(dist < min_distance){

                            min_distance=dist;
                            index_of_min_dist=j;
                        }
                    }
                } /* end of j loop */

                /* point i belongs to cluster index_of_min_dist */

                table->cluster_indices[index_of_min_dist][table->cluster_sizes[index_of_min_dist]]=table_index;
                table->cluster_sizes[index_of_min_dist]++;



                if(table->cluster_sizes[index_of_min_dist] ==  table->cluster_max_sizes[index_of_min_dist]-1  ){

                    EnlargeCluster(table, index_of_min_dist);

                }


            } /* end of i loop */




            table->cluster_sizes[k]=0;
        }

    }



    /* print each cluster to the screeen */



    for(int k=0;k<number_of_clusters;k++) {
        if(table->cluster_sizes[k] > 0){
            printf("\ncluster %d = \n",k);

            printf("mean = ");
            for(int i=0;i<table->columnsize;i++){
                printf("%10.7f ",table->cluster_means[k][i] );

            }
            printf("\n");




            for(int j=0;j<table->cluster_sizes[k];j++) {
                printf("%5d",table->cluster_indices[k][j]);
                for(int i=0;i<table->columnsize;i++){
                    printf("%10.7f ",table->data[i][table->cluster_indices[k][j]]);
                }

                printf("\n");
            }


        }
    }











    for(int i=0;i<number_of_clusters;i++)
    {
        delete[] xmean0[i];

    }
    delete[] xmean0;

    delete[] x;
    delete[] distance;
    delete[] xmean_differences;

    return 0;
}


/** check if x between a and b.
 *
 * @param[in] x
 * @param[in] a
 * @param[in] b
 * @return   1 if x is in [a,b] 0 else
 */
int CheckInterval(double x, double a, double b){

    if( (x >=a && x<=b) || ( x >=b && x<=a) ) {

        return 1;
    }
    else {

        return 0;
    }

}

/**
 * function for 2D interpolation using rectilinear data f=f(x,y)
 * @param[IN]    table
 * @param[IN]    x                  : independent variable x
 * @param[IN]    y                  : independent variable y
 * @param[OUT]   result             : result vector
 * @param[IN]    comp_index         : indices of elements of result vector
 * @param[IN]    number_of_vars_to_interpolate = length(comp_index)
 */
void interpolateTable2DRectilinear(IntLookupTable* table,
        double x, double y,
        double *result,
        int* comp_index, int number_of_vars_to_interpolate)
{

    /* normalize the interpolation variables */

    double var1= (x-table->xmin[0])/
            (table->xmax[0]-table->xmin[0]);

    double var2= (y-table->xmin[1])/
            (table->xmax[1]-table->xmin[1]);

    int indx1_up,indx1_down,indx1_half;
    int indx2_up,indx2_down,indx2_half;

    int found1=0;
    int found2=0;

    double xs,xe;

    double f11,f12,f21,f22;
    double x1=0.0,x2=0.0,y1=0.0,y2=0.0;

    int left;

    /* first check last interpolation points */

    if (table->last_interpolation_points[0] != -1)
    {
        x1= table->data[0][table->last_interpolation_points[0]];
        x2= table->data[0][table->last_interpolation_points[2]];

        y1= table->data[1][table->last_interpolation_points[0]];
        y2= table->data[1][table->last_interpolation_points[2]];


        if ( CheckInterval(var1, x1, x2)   &&  CheckInterval(var2, y1, y2) )
        {

            for(int i=0;i<number_of_vars_to_interpolate;i++){

                f11=table->data[comp_index[i]][table->last_interpolation_points[0]];
                f21=table->data[comp_index[i]][table->last_interpolation_points[1]];
                f22=table->data[comp_index[i]][table->last_interpolation_points[2]];
                f12=table->data[comp_index[i]][table->last_interpolation_points[3]];

                /* apply bilinear formula */

                result[i]= f11*(x2-var1)*(y2-var2)+ f21*(var1-x1)*(y2-var2)+ f12*(x2-var1)*(var2-y1)+ f22*(var1-x1)*(var2-y1);
                result[i]=result[i]* (1.0/( (x2-x1)*(y2-y1) ) );

                /*correct the result xold = xnew*(xmax-xmin)+xmin*/
                result[i]=result[i]*(table->xmax[comp_index[i]]-table->xmin[comp_index[i]])+table->xmin[comp_index[i]];


            } /* end of i loop */


        } /* end of if */

    } /* end of if */

    /* if the point is inside the box */
    if ( CheckInterval(var1, 0.0, 1.0)   &&  CheckInterval(var2, 0.0, 1.0) )

    {

        /* bounds of the table */

        indx1_up=table->Ndim[0]-1;
        indx1_down=0;

        indx2_up=table->Ndim[1]-1;
        indx2_down=0;



        if(indx1_up == 1) found1=1;
        if(indx2_up == 1) found2=1;

        /* contraction in first coordinate direction*/
        while(!found1){

            indx1_half=(indx1_up-indx1_down)/2+indx1_down;

            int index1 = indx1_down;
            int index2 = indx1_half;


            xs=table->data[0][index1];
            xe=table->data[0][index2];

            left=CheckInterval(var1, xs , xe );

            if(left==1) {

                indx1_up=indx1_half;

            }
            else{

                indx1_down=indx1_half;

            }


            if (indx1_up == indx1_down+1 ) {

                found1=1;
            }

        } /* end of while loop */



        /* contraction in second coordinate direction*/
        while(!found2){

            indx2_half=(indx2_up-indx2_down)/2+indx2_down;

            int index1 = indx1_down+table->Ndim[0]*indx2_down;
            int index2 = indx1_down+table->Ndim[0]*indx2_half;

            xs=table->data[1][index1];
            xe=table->data[1][index2];

            left=CheckInterval(var2,  xs , xe  );

            if(left==1) {

                indx2_up=indx2_half;

            }
            else{

                indx2_down=indx2_half;

            }


            if (indx2_up == indx2_down+1 ) {

                found2=1;
            }

        } /* end of while loop */



        int indx1,indx2,indx3,indx4;

        indx1=indx1_down  + table->Ndim[0]*indx2_down;
        indx2=indx1_up    + table->Ndim[0]*indx2_down;
        indx3=indx1_up    + table->Ndim[0]*indx2_up;
        indx4=indx1_down  + table->Ndim[0]*indx2_up;


        /*
         *            f12          f22
         *              ------------
         *              -          -
         *              -          -
         *              -          -
         *              -          -
         *              -          -
         *              ------------
         *            f11           f21
         */







        x1=table->data[0][indx1];
        x2=table->data[0][indx3];

        y1=table->data[1][indx1];
        y2=table->data[1][indx3];


        printTableElement(table,indx1,0);
        printTableElement(table,indx2,0);
        printTableElement(table,indx3,0);
        printTableElement(table,indx4,0);

        for(int i=0;i<number_of_vars_to_interpolate;i++){

            f11=table->data[comp_index[i]][indx1];
            f21=table->data[comp_index[i]][indx2];
            f22=table->data[comp_index[i]][indx3];
            f12=table->data[comp_index[i]][indx4];


            /* apply bilinear formula */

            result[i]= f11*(x2-var1)*(y2-var2)+ f21*(var1-x1)*(y2-var2)+ f12*(x2-var1)*(var2-y1)+ f22*(var1-x1)*(var2-y1);
            result[i]=result[i]* (1.0/( (x2-x1)*(y2-y1) ) );

            /*correct the result xold = xnew*(xmax-xmin)+xmin*/
            result[i]=result[i]*(table->xmax[comp_index[i]]-table->xmin[comp_index[i]])+table->xmin[comp_index[i]];




        } /* end of i loop */



        /* save the interpolation points for the next interpolation */
        table->last_interpolation_points[0]=indx1;
        table->last_interpolation_points[1]=indx2;
        table->last_interpolation_points[2]=indx3;
        table->last_interpolation_points[3]=indx4;



    } /* end of if */
    else{


        /* restrict the extrapolations */
        //        if (var1 > 1.2)  var1 =1.2;
        //        if (var2 > 1.2)  var2 =1.2;
        //        if (var1 < -0.2) var1 =-0.2;
        //        if (var2 < -0.2)  var2 =-0.2;


        /* if function comes here extrapolation */

        int indx1=0,indx2=0,indx3=0,indx4=0;
        int Iindx,Jindx;


        /*
         *
         *    x1,y2 ----------------------------- x2,y2
         *          -                           -
         *          -                           -
         *          -                           -
         *          -                           -
         *          -                           -
         *          -                           -
         *          -                           -
         *    x1,y1 ----------------------------- x2,y1
         *
         */


        if( var1 <= 0 && var2 <= 0){

            Iindx=0; Jindx=0;
            indx1=0;
            Iindx=1; Jindx=0;
            indx2=table->Ndim[0];
            Iindx=1; Jindx=1;
            indx3=table->Ndim[0]+1;
            Iindx=0; Jindx=1;
            indx4=1;



            x1=table->data[0][indx1];
            x2=table->data[0][indx3];
            y1=table->data[1][indx1];
            y2=table->data[1][indx3];


        }

        if( var1 >= 1 && var2 <= 0){

            /*
             *
             *    x1,y2 ----------------------------- x2,y2
             *          -                           -
             *          -                           -
             *          -                           -
             *          -                           -
             *          -                           -
             *          -                           -
             *          -                           -
             *    x1,y1 ----------------------------- x2,y1
             *
             *                                                  x
             *
             */

            Iindx=table->Ndim[0]-2; Jindx=0;
            indx1=table->Ndim[0]*Jindx+Iindx;
            Iindx=table->Ndim[0]-1; Jindx=0;
            indx2=table->Ndim[0]*Jindx+Iindx;
            Iindx=table->Ndim[0]-1; Jindx=1;
            indx3=table->Ndim[0]*Jindx+Iindx;
            Iindx=table->Ndim[0]-2; Jindx=1;
            indx4=table->Ndim[0]*Jindx+Iindx;

            x1=table->data[0][indx1];
            x2=table->data[0][indx3];
            y1=table->data[1][indx1];
            y2=table->data[1][indx3];


        }

        if( var1 >= 1 && var2 >= 1){


            Iindx=table->Ndim[0]-2; Jindx=table->Ndim[1]-2;
            indx1=table->Ndim[0]*Jindx+Iindx;
            Iindx=table->Ndim[0]-1; Jindx=table->Ndim[1]-2;
            indx2=table->Ndim[0]*Jindx+Iindx;
            Iindx=table->Ndim[0]-1; Jindx=table->Ndim[1]-1;
            indx3=table->Ndim[0]*Jindx+Iindx;
            Iindx=table->Ndim[0]-2; Jindx=table->Ndim[1]-1;
            indx4=table->Ndim[0]*Jindx+Iindx;

            x1=table->data[0][indx1];
            x2=table->data[0][indx3];
            y1=table->data[1][indx1];
            y2=table->data[1][indx3];


        }

        if( var1 <= 0 && var2 >= 1){


            Iindx=0; Jindx=table->Ndim[1]-2;
            indx1=table->Ndim[0]*Jindx+Iindx;
            Iindx=1; Jindx=table->Ndim[1]-2;
            indx2=table->Ndim[0]*Jindx+Iindx;
            Iindx=1; Jindx=table->Ndim[1]-1;
            indx3=table->Ndim[0]*Jindx+Iindx;
            Iindx=0; Jindx=table->Ndim[1]-1;
            indx4=table->Ndim[0]*Jindx+Iindx;

            x1=table->data[0][indx1];
            x2=table->data[0][indx3];
            y1=table->data[1][indx1];
            y2=table->data[1][indx3];


        }


        if( var1 > 0 && var1 < 1 && var2 < 0){ /* x is ok y is less than 0 */


            indx1_down= 0;
            indx1_up  = table->Ndim[0]-1;
            indx2_down= 0;
            indx2_up  = 1;


            /* contraction in first coordinate direction*/
            while(!found1){

                indx1_half=(indx1_up-indx1_down)/2+indx1_down;

                int index1 = indx1_down;
                int index2 = indx1_half;


                xs=table->data[0][index1];
                xe=table->data[0][index2];

                left=CheckInterval(var1, xs , xe );

                if(left==1) {

                    indx1_up=indx1_half;

                }
                else{
                    indx1_down=indx1_half;

                }

                if (indx1_up == indx1_down+1 ) {

                    found1=1;
                }

            } /* end of while loop */


            Iindx=indx1_down; Jindx=indx2_down;
            indx1=table->Ndim[0]*Jindx+Iindx;
            Iindx=indx1_up; Jindx=indx2_down;
            indx2=table->Ndim[0]*Jindx+Iindx;
            Iindx=indx1_up; Jindx=indx2_up;
            indx3=table->Ndim[0]*Jindx+Iindx;
            Iindx=indx1_down; Jindx=indx2_up;
            indx4=table->Ndim[0]*Jindx+Iindx;

            x1=table->data[0][indx1];
            x2=table->data[0][indx3];
            y1=table->data[1][indx1];
            y2=table->data[1][indx3];

        }

        if( var1 > 0 && var1 < 1 && var2 > 1){ /* x is ok y is greater than 1 */


            indx1_down= 0;
            indx1_up  = table->Ndim[0]-1;
            indx2_down= table->Ndim[1]-2;
            indx2_up  = table->Ndim[1]-1;;

            /* contraction in first coordinate direction*/
            while(!found1){

                indx1_half=(indx1_up-indx1_down)/2+indx1_down;

                int index1 = indx1_down;
                int index2 = indx1_half;


                xs=table->data[0][index1];
                xe=table->data[0][index2];

                left=CheckInterval(var1, xs , xe );

                if(left==1) {

                    indx1_up=indx1_half;

                }
                else{
                    indx1_down=indx1_half;

                }


                if (indx1_up == indx1_down+1 ) found1=1;

            } /* end of while loop */


            Iindx=indx1_down; Jindx=indx2_down;
            indx1=table->Ndim[0]*Jindx+Iindx;
            Iindx=indx1_up; Jindx=indx2_down;
            indx2=table->Ndim[0]*Jindx+Iindx;
            Iindx=indx1_up; Jindx=indx2_up;
            indx3=table->Ndim[0]*Jindx+Iindx;
            Iindx=indx1_down; Jindx=indx2_up;
            indx4=table->Ndim[0]*Jindx+Iindx;

            x1=table->data[0][indx1];
            x2=table->data[0][indx3];
            y1=table->data[1][indx1];
            y2=table->data[1][indx3];


        }

        if( var2 > 0 && var2 < 1 && var1 < 0){ /* y is ok x is less than 0 */



            indx1_down= 0;
            indx1_up  = 1;
            indx2_down= 0;
            indx2_up  = table->Ndim[1]-1;

            /* contraction in second coordinate direction*/
            while(!found2){

                indx2_half=(indx2_up-indx2_down)/2+indx2_down;

                int index1 = indx1_down+table->Ndim[0]*indx2_down;
                int index2 = indx1_down+table->Ndim[0]*indx2_half;



                xs=table->data[1][index1];
                xe=table->data[1][index2];


                left=CheckInterval(var2,  xs , xe  );

                if(left==1) {

                    indx2_up=indx2_half;

                }
                else{

                    indx2_down=indx2_half;

                }


                if (indx2_up == indx2_down+1 ) {

                    found2=1;
                }

            } /* end of while loop */



            Iindx=indx1_down; Jindx=indx2_down;
            indx1=table->Ndim[0]*Jindx+Iindx;
            Iindx=indx1_up; Jindx=indx2_down;
            indx2=table->Ndim[0]*Jindx+Iindx;
            Iindx=indx1_up; Jindx=indx2_up;
            indx3=table->Ndim[0]*Jindx+Iindx;
            Iindx=indx1_down; Jindx=indx2_up;
            indx4=table->Ndim[0]*Jindx+Iindx;

            x1=table->data[0][indx1];
            x2=table->data[0][indx3];
            y1=table->data[1][indx1];
            y2=table->data[1][indx3];



        }


        if( var2 > 0 && var2 < 1 && var1 > 1){ /* y is ok x is greater than 1 */

            indx1_down= table->Ndim[0]-2;
            indx1_up  = table->Ndim[0]-1;;
            indx2_down= 0;
            indx2_up  = table->Ndim[1]-1;

            /* contraction in second coordinate direction*/
            while(!found2){

                indx2_half=(indx2_up-indx2_down)/2+indx2_down;

                int index1 = indx1_down+table->Ndim[0]*indx2_down;
                int index2 = indx1_down+table->Ndim[0]*indx2_half;



                xs=table->data[1][index1];
                xe=table->data[1][index2];



                left=CheckInterval(var2,  xs , xe  );

                if(left==1) {

                    indx2_up=indx2_half;

                }
                else{
                    indx2_down=indx2_half;

                }

                if (indx2_up == indx2_down+1 ) {

                    found2=1;
                }

            } /* end of while loop */


            Iindx=indx1_down; Jindx=indx2_down;
            indx1=table->Ndim[0]*Jindx+Iindx;
            Iindx=indx1_up; Jindx=indx2_down;
            indx2=table->Ndim[0]*Jindx+Iindx;
            Iindx=indx1_up; Jindx=indx2_up;
            indx3=table->Ndim[0]*Jindx+Iindx;
            Iindx=indx1_down; Jindx=indx2_up;
            indx4=table->Ndim[0]*Jindx+Iindx;

            x1=table->data[0][indx1];
            x2=table->data[0][indx3];
            y1=table->data[1][indx1];
            y2=table->data[1][indx3];


        }

#if 1
        printTableElement(table,indx1,0);
        printTableElement(table,indx2,0);
        printTableElement(table,indx3,0);
        printTableElement(table,indx4,0);
#endif

        for(int i=0;i<number_of_vars_to_interpolate;i++){


            f11=table->data[comp_index[i]][indx1];
            f21=table->data[comp_index[i]][indx2];
            f22=table->data[comp_index[i]][indx3];
            f12=table->data[comp_index[i]][indx4];

            /* apply bilinear formula */

            result[i]= f11*(x2-var1)*(y2-var2)+ f21*(var1-x1)*(y2-var2)+ f12*(x2-var1)*(var2-y1)+ f22*(var1-x1)*(var2-y1);
            result[i]=result[i]* (1.0/( (x2-x1)*(y2-y1) ) );

            /*correct the result xold = xnew*(xmax-xmin)+xmin*/
            result[i]=result[i]*(table->xmax[comp_index[i]]-table->xmin[comp_index[i]])+table->xmin[comp_index[i]];


        } /* end of i loop */

        /* save the interpolation points for the next interpolation */
        table->last_interpolation_points[0]=indx1;
        table->last_interpolation_points[1]=indx2;
        table->last_interpolation_points[2]=indx3;
        table->last_interpolation_points[3]=indx4;

    } /* end of else */




} /* end of the function*/


/** function to find the cluster index of point x
 *
 * @param[in] table  pointer to lookup table
 * @param[in] x
 * @param[in] x_index
 * @param[in] number_of_indep_vars
 * @return  cluster index of point x
 */
int FindWhichCluster(IntLookupTable* table, double *x, int *x_index, int number_of_indep_vars){

    double min_distance=LARGE;
    double distance;

    int index=-1;


    for(int i=0;i<table->number_of_clusters;i++){

            distance=0.0;

            for(int k=0; k<number_of_indep_vars; k++){

                distance+=(x[k]-table->cluster_means[i][x_index[k]])*(x[k]-table->cluster_means[i][x_index[k]]);

            }

            if (distance < min_distance){

                min_distance=distance;
                index=i;
            }


    } /* end of for */



    return index;

}


/** function to calculate the distance between x and table point indexed by indx
 *
 * @param[in] table  pointer to lookup table
 * @param[in] x
 * @param[in] indx
 * @param[in] variable_index
 * @param[in] number_of_indep_vars
 * @return  distance
 */
double calcDistanceScatter(IntLookupTable *table, double *x, int indx, int * variable_index, int number_of_indep_vars ){

    double distance=0.0;

    for(int i=0; i<number_of_indep_vars; i++){

        distance+=(x[i]-table->data[variable_index[i]][indx])*(x[i]-table->data[variable_index[i]][indx]);

    }

    return distance;

}

/** finds K nearest neighbours of a point
 *
 * @param[in] table  pointer to lookup table
 * @param[out] point_indices
 * @param[out] point_distances
 * @param[in] cluster_no
 * @param[in] K
 * @param[in] x_index
 * @param[in] number_of_indep_vars
 * @return K if cluster has less points than K
 */
int findKneighbours(IntLookupTable* table,
        int *point_indices,
        double *point_distances,
        int cluster_no,
        int K,
        double *x,
        int *x_index,
        int number_of_indep_vars){

    double max_distance = LARGE;
    int max_distance_index = 0;

    /* if cluster has less points than K */
    if(K > table->cluster_sizes[cluster_no]) {

        K = table->cluster_sizes[cluster_no];
    }

    for(int i=0; i<K; i++){

        point_distances[i]= LARGE;
        point_indices[i] = -1;
    }

    for(int i=0; i<table->cluster_sizes[cluster_no]; i++){ /* for each point in the cluster */

        double dist=0.0;

        int index= table->cluster_indices[cluster_no][i];

        /* calculate the distance */

        dist=calcDistanceScatter(table, x, index, x_index, number_of_indep_vars );

        if (dist < max_distance){ /* if a point with smaller distance is found */


            point_indices[max_distance_index]=index;
            point_distances[max_distance_index]=dist;

            max_distance = -LARGE;

            for(int j=0; j<K; j++){

                if (point_distances[j] > max_distance){

                    max_distance = point_distances[j];
                    max_distance_index = j;
                }

            }

        }


    }

    return K;

}


/**
 * Implementation of scatter data interpolation
 * @param[IN]    interpolation_variable
 * @param[IN]    x_index
 * @param[OUT]   result
 * @param[IN]    comp_index
 * @param[IN]    number_of_indep_vars: dim(x_index)
 * @param[IN]    number_of_vars_to_interpolate
 */
void interpolateTableScatterdata(IntLookupTable* table,
        double *interpolation_variable,
        int *x_index,
        double *result,
        int* comp_index,
        int number_of_indep_vars,
        int number_of_vars_to_interpolate ){

    double w[4]={0.0,0.0,0.0,0.0};
    double f[4]={0.0,0.0,0.0,0.0};

    double wsum=0.0;

    /* check if the interpolation routine is called with NaN arguments */
    int NaNflag = 0;

    for(int i=0;i<number_of_indep_vars;i++){

        if(std::isnan(interpolation_variable[i])){

            NaNflag = 1;
        }

    }

    if(NaNflag == 0){

        int cluster_no;

        double *x = new double[number_of_indep_vars];

        /* normalize the input vector x */
        for(int i=0; i<number_of_indep_vars; i++){

            x[i]= (interpolation_variable[i]-table->xmin[x_index[i]])/
                    (table->xmax[x_index[i]]-table->xmin[x_index[i]]);

        }

        /* search the corresponding cluster */

        if(table->number_of_clusters > 1){

            cluster_no=FindWhichCluster(table,x,x_index,number_of_indep_vars);

            if(cluster_no == -1) {

                printf("error in lookup table clustering\n");
                exit(1);
            }

        }
        else{

            cluster_no=0;

        }

        int *nearest_point_indices = new int[4];



        double *nearest_point_distances= new double[4];



        /* find the 4 nearest neighbours */
        int K = findKneighbours(table,
                nearest_point_indices,
                nearest_point_distances,
                cluster_no,
                4,
                x,
                x_index,
                number_of_indep_vars);

#if 0
        fprintf(stdout,"nearest four points=\n");
        printTableElement(table,nearest_point_indices[0],0);
        printTableElement(table,nearest_point_indices[1],0);
        printTableElement(table,nearest_point_indices[2],0);
        printTableElement(table,nearest_point_indices[3],0);
#endif

        for (int i=0; i<K; i++){

            if(fabs(nearest_point_distances[i]) < EPSILON) {

                nearest_point_distances[i]=EPSILON;
            }

        }

        /* find inverse distance weights */

        for (int i=0; i<K; i++){

            w[i]=1.0/(nearest_point_distances[i]*nearest_point_distances[i]);
            wsum+=w[i];
        }

        /* interpolation step */

        for(int i=0;i<number_of_vars_to_interpolate;i++){

            result[i]=0.0;
            for(int j=0; j<K; j++){

                f[j]= table->data[comp_index[i]][nearest_point_indices[j]];
                result[i]+=w[j]*f[j];
            }


            result[i]=result[i]/wsum;
            result[i]=result[i]*(table->xmax[comp_index[i]]-table->xmin[comp_index[i]])+table->xmin[comp_index[i]];
        }

        delete[] x;
        delete[] nearest_point_distances;
        delete[] nearest_point_indices;

    }
    else{

        /* NaN in arguments */

        for(int i=0;i<number_of_vars_to_interpolate;i++) {

            result[i]=NAN;
        }



    }


}

void intbuildkdNodeList(mat data,intkdNode *kdNodeVec ){

    int dim = data.n_rows;

    for(int i=0; i<dim; i++){

        kdNodeVec[i].x = data.row(i);
        kdNodeVec[i].indx = i;
        kdNodeVec[i].left = NULL;
        kdNodeVec[i].right = NULL;

    }



}



double intKdtreedist(intkdNode *a, intkdNode *b) {

    int dim = a->x.size();
    double t, d = 0;
    while (dim--) {
        t = a->x(dim) - b->x(dim);
        d += t * t;
    }
    return d;
}



double intKdtreefindmedian(intkdNode *nodelist, int len, int idx)
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



intkdNode* intmaketree(intkdNode *root, int len, int indx){

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


    intkdNode *start = &root[0];


#if 0
    printf("start node index= %d\n",start->indx);
    printf("end node index  = %d\n",end->indx);
    start->x.print();
    end->x.print();
#endif


    double median_value = intKdtreefindmedian(start, len, indx);



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



    intkdNode *kdNodeVectemp = new intkdNode[len];

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

    intkdNode *n= &root[median_index];

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
        n->left  = intmaketree(&root[0], leftsize, indx);

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
        n->right = intmaketree(&root[leftsize+1], rightsize, indx );
    }

    delete[] kdNodeVectemp;


    return n;
}

void intKdtreeNearest(intkdNode *root, intkdNode *p, int K, int i, int* best_index, double *best_dist, int dim){

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

        intkdNode *left = root->left;

        double *dbestleft= new double[K];
        int *indexbestleft= new int[K];


        int inext = (i + 1) % dim;
        intKdtreeNearest(left, p,K,inext, indexbestleft, dbestleft,dim);

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

        intkdNode *right = root->right;
        double *dbestright= new double[K];
        int *indexbestright= new int[K];

        int inext = (i + 1) % dim;
        intKdtreeNearest(right, p, K,inext, indexbestright, dbestright,dim);

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
            intkdNode *left = root->left;
            double *dbestleft= new double[K];
            int *indexbestleft= new int[K];


            find_max_with_index(best_dist,K,&max_dist,&max_dist_index);

            int inext = (i + 1) % dim;
            intKdtreeNearest(left, p, K,inext, indexbestleft, dbestleft,dim);

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

                intkdNode *right = root->right;
                double *dbestright= new double[K];
                int *indexbestright= new int[K];

                int inext = (i + 1) % dim;
                intKdtreeNearest(right, p, K,inext, indexbestright, dbestright,dim);

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
            intkdNode *right = root->right;
            double *dbestright= new double[K];
            int *indexbestright= new int[K];

            find_max_with_index(best_dist,K,&max_dist,&max_dist_index);

#if 0
            printf("max_dist = %10.7f\n", max_dist);
#endif
            int inext = (i + 1) % dim;
            intKdtreeNearest(right, p, K,inext, indexbestright, dbestright,dim);

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

                intkdNode *left = root->left;
                double *dbestleft= new double[K];
                int *indexbestleft= new int[K];

                int inext = (i + 1) % dim;
                intKdtreeNearest(left, p,K,inext, indexbestleft, dbestleft,dim);

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

void test_2d_table(void){

    IntLookupTable table2D;

    table2D.Ndim[0]=200;
    table2D.Ndim[1]=10;
    table2D.Ndim[2]=1;


    table2D.rowsize    = table2D.Ndim[0]*table2D.Ndim[1];
    table2D.columnsize = 5;

    table2D.data = new double*[table2D.columnsize];

    for(int i=0; i<table2D.columnsize; i++ ){

        table2D.data[i] = new double[table2D.rowsize];

    }



    /* generate data */

    double xs = -15.0;
    double xe = 22.0;

    double ys = 0.1;
    double ye = 1.8;


    double dx = (xe-xs)/(table2D.Ndim[0]-1);
    double dy = (ye-ys)/(table2D.Ndim[1]-1);

    double x,y;

    double f1,f2,f3;


    x=xs; y=ys;
    int count=0;

    for(int i=0; i<table2D.Ndim[1]; i++ ){

        x=xs;
        for(int j=0; j<table2D.Ndim[0]; j++){
            f1 = x*x+sin(x*y)+y;
            f2 = x+y+x*y;
            f3 = (1-x)+ y*(1-x);

            //            printf("%d %10.7f %10.7f %10.7f %10.7f %10.7f\n",count,x,y,f1,f2,f3);

            table2D.data[0][count]=x;
            table2D.data[1][count]=y;
            table2D.data[2][count]=f1;
            table2D.data[3][count]=f2;
            table2D.data[4][count]=f3;
            count++;
            x+=dx;
        }


        y+=dy;
    }

    table2D.xmin = new double[table2D.columnsize];
    table2D.xmax = new double[table2D.columnsize];

    for(int i=0; i<table2D.columnsize; i++) { /* for each column of the table */

        table2D.xmin[i]= LARGE;
        table2D.xmax[i]=-LARGE;

        for(int j=0;j<table2D.rowsize;j++){

            if (table2D.data[i][j] < table2D.xmin[i]) table2D.xmin[i]=table2D.data[i][j];
            if (table2D.data[i][j] > table2D.xmax[i]) table2D.xmax[i]=table2D.data[i][j];
        }




    }




    /* normalize the T-p table data  xnew= (xold-xmin)/(xmax-xmin)  */


    for(int i=0; i<table2D.columnsize; i++) {

        for(int j=0; j<table2D.rowsize; j++){


            table2D.data[i][j]= (table2D.data[i][j]-table2D.xmin[i])/(table2D.xmax[i]-table2D.xmin[i]);

        }

    }

    /* initialize last interpolation point indices */
    for(int i=0; i<4; i++) {

        table2D.last_interpolation_points[i]=-1;
    }


    printTable(&table2D, 0);


    /* try to interpolate in sample points */

    double result[3];
    int comp_index[3]={2,3,4};
    for(int i=0; i<100; i++){

        int indx = RandomInt(0,table2D.rowsize);

        printTableElement(&table2D,indx, 0);
        x = table2D.data[0][indx];
        x = x*(table2D.xmax[0]-table2D.xmin[0])+table2D.xmin[0];
        y = table2D.data[1][indx];
        y = y*(table2D.xmax[1]-table2D.xmin[1])+table2D.xmin[1];


        interpolateTable2DRectilinear(&table2D,
                x, y,
                result,
                comp_index, 3);


        printf("%d %10.7f %10.7f %10.7f  %10.7f  %10.7f \n", indx,x,y, result[0],result[1],result[2]);


    }



    /* try out of sample points */
    for(int i=0; i<100; i++){

        x = RandomDouble(0,1.0);
        x = x*(table2D.xmax[0]-table2D.xmin[0])+table2D.xmin[0];
        y = RandomDouble(0,1.0);
        y = y*(table2D.xmax[1]-table2D.xmin[1])+table2D.xmin[1];

        double f1,f2,f3;
        f1 = x*x+sin(x*y)+y;
        f2 = x+y+x*y;
        f3 = (1-x)+ y*(1-x);

        interpolateTable2DRectilinear(&table2D,
                x, y,
                result,
                comp_index, 3);


        printf("%d %10.7f %10.7f %10.7f  %10.7f  %10.7f \n", i,x,y, result[0],result[1],result[2]);
        printf("%d %10.7f %10.7f %10.7f  %10.7f  %10.7f \n\n", i,x,y, f1,f2,f3);


    }

    /* try extrapolations*/

    /* try out of sample points */
    for(int i=0; i<100; i++){


        double random1 = RandomDouble(0.0,1.0);
        double random2 = RandomDouble(0.0,1.0);

        if(random1 < 0.333){

            x = RandomDouble(-0.1,0.0);
        }

        if(random1 > 0.333 && random1 < 0.666){

            x = RandomDouble(0.0,1.0);
        }

        if(random1 > 0.666){

            x = RandomDouble(1.0,1.1);
        }

        if(random2 < 0.333){

            y = RandomDouble(-0.1,0.0);
        }

        if(random2 > 0.333 && random2 < 0.666){

            y = RandomDouble(0.0,1.0);
        }

        if(random2 > 0.666){

            y = RandomDouble(1.0,1.1);
        }




#if 1
        printf("x = %10.7f, y = %10.7f\n",x,y);
#endif

        x = x*(table2D.xmax[0]-table2D.xmin[0])+table2D.xmin[0];
        y = y*(table2D.xmax[1]-table2D.xmin[1])+table2D.xmin[1];
#if 1
        printf("x = %10.7f, y = %10.7f\n",x,y);
#endif
        double f1,f2,f3;
        f1 = x*x+sin(x*y)+y;
        f2 = x+y+x*y;
        f3 = (1-x)+ y*(1-x);

        interpolateTable2DRectilinear(&table2D,
                x, y,
                result,
                comp_index, 3);


        printf("%d %10.7f %10.7f %10.7f  %10.7f  %10.7f \n", i,x,y, result[0],result[1],result[2]);
        printf("%d %10.7f %10.7f %10.7f  %10.7f  %10.7f \n\n", i,x,y, f1,f2,f3);


    }


}

void test_scatter_table(void){

    IntLookupTable tablescatter;

    tablescatter.rowsize    = 100000;
    tablescatter.columnsize = 5;

    tablescatter.data = new double*[tablescatter.columnsize];

    for(int i=0; i<tablescatter.columnsize; i++ ){

        tablescatter.data[i] = new double[tablescatter.rowsize];

    }


    /* required for the bruce force method */

    double *min_dist = new double[4];
    int *indices = new int[4];

    /* required for the bruce force method */

    double xs = -15.0;
    double xe = 22.0;

    double ys = 0.1;
    double ye = 1.8;

    double x,y;

    double f1,f2,f3;

    int count=0;

    for(int i=0; i<tablescatter.rowsize; i++ ){

        x = RandomDouble(xs,xe);
        y = RandomDouble(ys,ye);
        f1 = exp((-x*x)/100.0 -(y*y)/100.0);
        f2 = x+y+x*y;
        f3 = (1-x)+ y*(1-x);

#if 0
        printf("%d %10.7f %10.7f %10.7f %10.7f %10.7f\n",count,x,y,f1,f2,f3);
#endif

        tablescatter.data[0][count]=x;
        tablescatter.data[1][count]=y;
        tablescatter.data[2][count]=f1;
        tablescatter.data[3][count]=f2;
        tablescatter.data[4][count]=f3;

        count++;

    }

    tablescatter.xmin = new double[tablescatter.columnsize];
    tablescatter.xmax = new double[tablescatter.columnsize];


    for(int i=0; i<tablescatter.columnsize; i++) { /* for each column of the table */

        tablescatter.xmin[i]= LARGE;
        tablescatter.xmax[i]=-LARGE;

        for(int j=0;j<tablescatter.rowsize;j++){

            if (tablescatter.data[i][j] < tablescatter.xmin[i]) {

                tablescatter.xmin[i]=tablescatter.data[i][j];
            }

            if (tablescatter.data[i][j] > tablescatter.xmax[i]) {

                tablescatter.xmax[i]=tablescatter.data[i][j];
            }
        }


    }








    /* normalize the T-p table data  xnew= (xold-xmin)/(xmax-xmin)  */


    for(int i=0; i<tablescatter.columnsize; i++) {

        for(int j=0; j<tablescatter.rowsize; j++){


            tablescatter.data[i][j]= (tablescatter.data[i][j]-tablescatter.xmin[i])/(tablescatter.xmax[i]-tablescatter.xmin[i]);

        }

    }


#if 0
    printTable(&tablescatter, 0);
#endif

    mat kddata(tablescatter.rowsize,2);

    for(int i=0; i<2; i++) { /* for each column of the table */

        for(int j=0;j<tablescatter.rowsize;j++){

            kddata(j,i) = tablescatter.data[i][j];

        }

    }

#if 0
    kddata.print();
#endif




    intkdNode *kdNodeVec = new intkdNode[tablescatter.rowsize];
    intbuildkdNodeList(kddata,kdNodeVec );
    intkdNode * root = intmaketree(kdNodeVec, tablescatter.rowsize, 0);


    int variable_index[2];

    variable_index[0] = 0;
    variable_index[1] = 1;

    /* cluster the table using cluster_size = 50 */
    KmeansClustering(&tablescatter,300,variable_index,2);

    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    /* try out of sample points */
    for(int i=0; i<10000000; i++){


        double x = RandomDouble(-0.1,1.1);
        double y = RandomDouble(-0.1,1.1);

        double f1,f2,f3;
        f1 = exp((-x*x)/100.0 -(y*y)/100.0);
        f2 = x+y+x*y;
        f3 = (1-x)+ y*(1-x);

        double xin[2];
        xin[0]=x;
        xin[1]=y;

        int x_index[2] = {0,1};

        double result[3];
        int comp_index[3]={2,3,4};
#if 0
        printf("\n\nx = %10.7f, y = %10.7f\n",x,y);
#endif

#if 0
        intfindKNeighbours(&tablescatter,
                xin,
                4,
                x_index ,
                min_dist,
                indices,
                2);


        printf("brute force KNN search results = \n");
        for(int j=0; j<4;j++){

            //            printf("indices[%d] = %d\n",j,indices[j]);
            printTableElement(&tablescatter, indices[j], 0);

        }
#endif


#if 0
        intkdNode *xinkd = new intkdNode;
        rowvec temp(2);
        temp(0) = x;
        temp(1) = y;
        xinkd->x = temp;

        intKdtreeNearest(root, xinkd, 4 ,0, indices, min_dist, 2);

#if 0
        printf("KD Tree results = \n");
        for(int j=0; j<4; j++){

            print_table_element(&tablescatter, indices[j], 0);

        }
#endif

#endif
        x = x*(tablescatter.xmax[0]-tablescatter.xmin[0])+tablescatter.xmin[0];
        y = y*(tablescatter.xmax[1]-tablescatter.xmin[1])+tablescatter.xmin[1];
#if 0
        printf("x = %10.7f, y = %10.7f\n",x,y);
#endif


        xin[0]=x;
        xin[1]=y;
#if 0
        printf("KNN with clustering results = \n");
#endif

#if 1
        interpolateTableScatterdata(&tablescatter,
                xin,
                x_index,
                result,
                comp_index,
                2,
                3 );
#endif

#if 0
        printf("%d %10.7f %10.7f %10.7f  %10.7f  %10.7f \n", i,x,y, result[0],result[1],result[2]);
        printf("%d %10.7f %10.7f %10.7f  %10.7f  %10.7f \n\n", i,x,y, f1,f2,f3);
#endif


    }
    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";


    delete[] min_dist;
    delete[] indices;
}

}
