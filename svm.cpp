#include "svm.hpp"
#include "Rodeo_macros.hpp"
#include "auxilliary_functions.hpp"

#include <armadillo>


using namespace arma;

double svm_linear_kernel(rowvec x1, rowvec x2){

    return dot(x1,x2);

}

double svm_polynomial_kernel(rowvec x1, rowvec x2, double c, double p){

    return pow( (dot(x1,x2)+c) ,p );

}

double svm_rbf_kernel(rowvec x1, rowvec x2, double sigma){

    rowvec xdiff = x1-x2;
    double norm_squared = pow( L2norm(xdiff), 2.0);

    return exp(-norm_squared/(2.0*sigma*sigma));

}


void train_svm(mat &X, vec &y,vec &alpha, double &b, SVM_param & model_parameters){

    unsigned int dim = y.size();


    const double Cminlog=0;
    const double Cmaxlog= 14;
    const double cmax=0.0;
    const double cmin=0.0;
    const double sigmamax=10.0;
    const double sigmamin=0.01;



    urowvec numberofsv(model_parameters.max_inner_iter);
    numberofsv.fill(0);

    vec Cparam(model_parameters.max_inner_iter,fill::zeros);

    vec alpha_best(alpha.size(),fill::zeros);

    double b_best = 0.0;
    double C_best = 0.0;
    double sigma_best=0.0; /* for Gaussian kernel */
    double c_best = 0.0; /* for poly kernel */

    int min_number_of_sv = alpha.size();

    /* generate lhs designs for SVM training*/


    mat svm_lhs(model_parameters.max_inner_iter,2);
    svm_lhs.fill(-1.0);

    std::string str_problem_dim = std::to_string(2);
    std::string lhs_filename = "svm_lhs_points.dat";
    std::string python_command = "python -W ignore lhs.py "+ lhs_filename+ " "+ str_problem_dim + " "+ std::to_string(model_parameters.max_inner_iter)+ " center" ;

    system(python_command.c_str());


    svm_lhs.load("lhs_points.dat", raw_ascii);

#if 1
    printf("DoE for the LHS training...\n");
    svm_lhs.print();
#endif




#if 0
    printf("Start train_svm...\n");
#endif

    if(dim != X.n_rows){
        printf("Error: Size of the label vector does not match with the number of rows in data matrix\n");
        exit(-1);

    }

    for(unsigned int i=0; i<y.size(); i++){
        if( (y(i)!=1.0) && (y(i)!=-1.0)){ /* labels must be either 1 or -1 */
            printf("Label = %10.7f\n", y(i));
            printf("Error: some labels are not valid\n");
            exit(-1);


        }

    }





    for(int svm_inner_iter=0; svm_inner_iter<model_parameters.max_inner_iter; svm_inner_iter++ ){


            model_parameters.Csoftmargin = svm_lhs(svm_inner_iter,0)*(Cmaxlog-Cminlog)+Cminlog;

            model_parameters.Csoftmargin = pow(10.0,model_parameters.Csoftmargin);




        Cparam(svm_inner_iter) = model_parameters.Csoftmargin;

        if(model_parameters.kernel_type == SVM_RBF){

            model_parameters.sigmaGausskernel = svm_lhs(svm_inner_iter,1)*(sigmamax-sigmamin)+sigmamin;

        }

        if(model_parameters.kernel_type == SVM_POLYNOMIAL){

            model_parameters.cpolykernel = svm_lhs(svm_inner_iter,1)*(cmax-cmin)+cmin;

        }

#if 1
printf("C = %15.10f\n",model_parameters.Csoftmargin);
#endif

/* Kernel matrix */
mat K(dim,dim);
K.fill(-1.0);


/* evaluate Kernel matrix */
for(unsigned int i=0; i<dim; i++){

    for(unsigned int j=0; j<dim; j++){

        if(model_parameters.kernel_type == SVM_LINEAR){

            K(i,j) = svm_linear_kernel(X.row(i),X.row(j));
        }

        if(model_parameters.kernel_type == SVM_RBF){

            K(i,j) = svm_rbf_kernel(X.row(i),X.row(j),model_parameters.sigmaGausskernel);
        }

        if(model_parameters.kernel_type == SVM_POLYNOMIAL){

            K(i,j) = svm_polynomial_kernel(X.row(i),X.row(j),model_parameters.cpolykernel,model_parameters.ppolykernel);
        }


    }
}

#if 0
K.print();
#endif


/* save K, y and C for QP input */

FILE *qp_input_file = fopen("qp_input.dat","w");

for(unsigned int i=0; i<dim; i++){

    for(unsigned int j=0; j<dim; j++){

        double val = y(i)*y(j)*K(i,j);
#if 0
        printf("y(%d) = %5.2f y(%d) = %5.2f K(%d,%d) = %10.7f val = %10.7f\n",i,y(i),j,y(j),i,j,K(i,j),val);
#endif

        fprintf(qp_input_file,"%15.10f\n",val);
    }

}

for(unsigned int i=0; i<dim; i++){

    fprintf(qp_input_file,"%15.10f\n",y(i));

}

fclose(qp_input_file);

/* call the python script to solve qp problem */

std::string python_command = "python solve_qp_for_svm.py "+
        std::to_string(model_parameters.Csoftmargin) +
        " "+std::to_string(dim) +
        " qp_input.dat";

system(python_command.c_str());


alpha.load("Lagrange_multipliers.txt", raw_ascii);

#if 0
alpha.print();
#endif


double largest_margin_alpha = -LARGE;
for(unsigned int i=0; i<alpha.size(); i++){

    if(alpha(i) > 0  && fabs(alpha(i)- model_parameters.Csoftmargin) > 10E-6 ){

        if(alpha(i)>largest_margin_alpha){

            largest_margin_alpha = alpha(i);
        }


    }

}

#if 0
printf("Largest margin Lagrange multipler = %10.7f\n",largest_margin_alpha);
#endif

model_parameters.sv_tolerance = largest_margin_alpha/10E6;


/* identify support vectors for a given tolerance*/

for(unsigned int i=0; i<alpha.size(); i++){

    if(alpha(i) < model_parameters.sv_tolerance) {

        alpha(i) = 0.0;
    }

}

#if 0
printf("Lagrange multipliers = \n");
alpha.print();
#endif

/* compute intercept b */

int sv_index=0;
for(unsigned int i=0; i<alpha.size(); i++){

    if (alpha(i) > 0.0){

        b = y(i);
        sv_index = i;
        break;
    }
}


double sum=0.0;
for(unsigned int i=0; i<alpha.size(); i++){

    sum+= y(i)*alpha(i)*K(i,sv_index);

}

b=b-sum;

printf("b = %10.7f\n",b);


numberofsv(svm_inner_iter) = 0;
for(unsigned int i=0; i<alpha.size(); i++){
    //        printf("alpha(i) = %d %10.7f\n",i,alpha(i));
    if(alpha(i) > 0 ){

        if( fabs(alpha(i)- model_parameters.Csoftmargin) < 10E-6){

            printf("non-margin support vector %d:\n",i);
            X.row(i).print();
            numberofsv(svm_inner_iter) =  numberofsv(svm_inner_iter)+1;
        }
        else{

            printf("margin support vector %d:\n",i);
            X.row(i).print();
            numberofsv(svm_inner_iter) =  numberofsv(svm_inner_iter)+1;

        }

    }

}

if(numberofsv(svm_inner_iter) < min_number_of_sv){

    min_number_of_sv = numberofsv(svm_inner_iter);
    alpha_best = alpha;
    b_best = b;
    C_best= model_parameters.Csoftmargin;
    sigma_best=model_parameters.sigmaGausskernel;
    c_best = model_parameters.cpolykernel;


}


#if 1
printf("Number of support vectors = %6d\n",int(numberofsv(svm_inner_iter)));

#endif


    } /* end of loop svm_inner_iter */



#if 1
    printf("Cparam            #sv\n");
    for(int svm_inner_iter=0; svm_inner_iter<model_parameters.max_inner_iter; svm_inner_iter++ ){

        printf("%15.10e        %6d\n", Cparam(svm_inner_iter),int(numberofsv(svm_inner_iter)));
    }

    alpha_best.print();
    printf("min_number_of_sv = %d\n",min_number_of_sv);


#endif

    alpha=alpha_best;
    b = b_best;
    model_parameters.Csoftmargin = C_best;
    model_parameters.sigmaGausskernel = sigma_best;
    model_parameters.cpolykernel = c_best;


}
