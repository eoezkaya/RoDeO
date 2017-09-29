#include "rbf.hpp"
#include "Rodeo_macros.hpp"
#include "auxilliary_functions.hpp"

#include <armadillo>

using namespace arma;


double rbf_psi_linear(double r) {

    return r;
}
double rbf_psi_cubic(double r) {

    return r*r*r;
}
double rbf_psi_tps(double r) {

    return r*r*log(r);
}

double rbf_psi_gaussian(double r, double sigma) {

    return exp(-(r*r)/(2*sigma*sigma));
}


/** compute the coefficients of the Gram matrix for the Gaussian rbf
 *
 * @param[out] PSI
 * @param[in] X
 * @param[in] sigma
 */
void compute_PSI_gauss(mat &PSI, mat &X, double sigma, double lambda){

    int dim = PSI.n_rows;

#if 0
    printf("dimension the PSI matrix = %d\n",dim);
    PSI.print();
#endif


#if 0
    PSI.print();
#endif

    for(unsigned i=0; i<dim; i++){ /* for each row of the PSI matrix other than the last row*/

        for(unsigned int j=0; j<dim; j++){

            rowvec x1 = X.row(i);
            rowvec x2 = X.row(j);

#if 0
            printf("x1 = \n");
            x1.print();
            printf("x2 = \n");
            x2.print();
#endif

            rowvec xdiff = x1 -x2;
            PSI(i,j) = rbf_psi_gaussian(L2norm(xdiff),sigma);

            /* add regularization term */
            if(i == j) PSI(i,j)+= lambda;


        }


    }

}


/** compute the coefficients of the rbf model
 *
 * @param[in] X  data matrix(normalized)
 * @param[in] ys output values
 * @param[out] w rbf parameters (w0,w1,...,Wd)
 * @param[in] lambda parameter of Tikhonov regularization
 * @param[in] type_of_basis_function
 * @return 0 if successful
 */

int train_rbf(mat &X, vec &ys, vec &w, int type){

    /* get the number of data points */
    unsigned int dim = X.n_rows;
    unsigned int d = X.n_cols;

    /* number of iterations for cross validation */
    unsigned int max_number_of_iter_for_cv = 100;

    const double upper_bound_sigma = 5.0;
    const double lower_bound_sigma = 0.0;

    const double upper_bound_lambda = 1.0;
    const double lower_bound_lambda = 8.0;


    /* number of iterations in cross validation */
    int number_of_iter_cv=0;

    if(type == GAUSSIAN){

        int number_of_data_points_for_cross_val = dim/5;

        int size_of_Gram_matrix = dim - number_of_data_points_for_cross_val;

#if 1
        printf("size of the Gram matrix = %d\n",size_of_Gram_matrix);
#endif



        /* Gram matrix */
        mat PSI (size_of_Gram_matrix,size_of_Gram_matrix );
        PSI.fill(0.0);

        /* rhs of the system PSI w = y */
        vec ysmod (size_of_Gram_matrix);

        uvec cv_indices(number_of_data_points_for_cross_val);


        mat Xmod(size_of_Gram_matrix, d);

        /* outer iterations for the cross validation */
        for(int iter=0; iter<max_number_of_iter_for_cv; iter++){



            /* generate a random value for sigma and lambda*/
            double sigma = RandomDouble(lower_bound_sigma,upper_bound_sigma);
            double lambda = RandomDouble(lower_bound_lambda,upper_bound_lambda);
#if 1
            printf("sigma = %10.7f\n",sigma);

            lambda = pow(10.0, -lambda);
            printf("lambda = %10.7f\n",lambda);

#endif

            double avg_L2_error = 0.0;
            for(int inner_iter =0; inner_iter<20; inner_iter++){


#if 1
                printf("X = \n");
                X.print();
                printf("ys = \n");
                ys.print();
#endif

                generate_validation_set(cv_indices, dim);
#if 1

                printf("cross validation indices = \n");
                cv_indices.print();
#endif
                remove_validation_points_from_data(X, ys, cv_indices, Xmod, ysmod);

#if 1
                printf("Xmod = \n");
                Xmod.print();

                printf("ysmod = \n");
                ysmod.print();
#endif



                compute_PSI_gauss(PSI,Xmod,sigma,lambda);

#if 1
                printf("PSI matrix = \n");
                PSI.print();
#endif


                w = solve(PSI, ysmod);


#if 1
                printf("w = \n");
                w.print();
#endif


                double ftilde=0.0;

                for(int point=0; point<number_of_data_points_for_cross_val; point++){

                    rowvec xp = X.row((cv_indices(point)));

                    for(int i=0; i<dim; i++){

                        rowvec xi = Xmod.row(i);
                        rowvec xdiff = xp-xi;

                        ftilde += w(i)*rbf_psi_gaussian(L2norm(xdiff),sigma);

                    }



                }




                exit(1);



            }


        }



    }







    return 0;
} 














