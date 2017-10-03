#include "svm.hpp"
#include "Rodeo_macros.hpp"
#include "auxilliary_functions.hpp"

#include <armadillo>

using namespace arma;

double svm_linear_kernel(rowvec x1, rowvec x2){

    return dot(x1,x2);

}

double svm_polynomial_kernel(rowvec x1, rowvec x2, double c, double p){

    return pow( dot(x1,x2)+c ,p );

}

double svm_rbf_kernel(rowvec x1, rowvec x2, double sigma){

    rowvec xdiff = x1-x2;
    double norm_squared = pow( L2norm(xdiff), 2.0);

    return exp(-norm_squared/(2.0*sigma*sigma));

}


void train_svm(mat &X, vec &y,SVM_param model_parameters){

    unsigned int dim = y.size();
    double sigma=1.0;
    double c=1.0;
    double p=2;

    if(dim != X.n_rows){
        printf("Error: Size of the label vector does not match with the number of rows in data matrix\n");
        exit(-1);

    }

    for(unsigned int i=0; i<y.size();i++){
        if( (y(i)!=1) || (y(i)!=-1)){ /* labels must be either 1 or -1 */
            printf("Error: some labels are not valid\n");
            exit(-1);


        }

    }

    /* Kernel matrix */
    mat K(dim,dim);


    /* evaluate Kernel matrix */
    for(unsigned int i=0; i<dim; i++){

        for(unsigned int j=0; j<dim; j++){

            if(model_parameters.kernel_type == SVM_LINEAR){

                K(i,j) = svm_linear_kernel(X.row(i),X.row(j));
            }

            if(model_parameters.kernel_type == SVM_RBF){

                K(i,j) = svm_rbf_kernel(X.row(i),X.row(j),sigma);
            }

            if(model_parameters.kernel_type == SVM_POLYNOMIAL){

                K(i,j) = svm_polynomial_kernel(X.row(i),X.row(j),c,p);
            }


        }
    }

    /* save K, y and C for QP input */

    FILE *qp_input_file = fopen("qp_input.dat","w");

    for(unsigned int i=0; i<dim; i++){

            for(unsigned int j=0; j<dim; j++){

                fprintf(qp_input_file,"%15.10f\n",K(i,j));
            }

    }

    for(unsigned int i=0; i<dim; i++){

        fprintf(qp_input_file,"%15.10f\n",y(i));

    }

    fclose(qp_input_file);




}
