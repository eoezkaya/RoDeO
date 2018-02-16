#include "linear_regression.hpp"
#include "auxilliary_functions.hpp"
#include "Rodeo_macros.hpp"

#include <armadillo>

using namespace arma;

/** compute the coefficients of the linear regression
 *
 * @param[in] X  data matrix
 * @param[in] ys output values
 * @param[out] w regression weights (w0,w1,...,Wd)
 * @param[in] lambda parameter of Tikhonov regularization
 * @return 0 if successful
 */

int train_linear_regression(mat &X, vec &ys, vec &w, double lambda=0){

    unsigned int dim = X.n_cols;

    mat augmented_X(X.n_rows, dim + 1);

    for (unsigned int i = 0; i < X.n_rows; i++) {

        for (unsigned int j = 0; j <= dim; j++) {

            if (j == 0){

                augmented_X(i, j) = 1.0;
            }

            else{

                augmented_X(i, j) = X(i, j - 1);
            }


        }
    }

    if(fabs(lambda) < EPSILON ){
#if 0
        printf("Taking pseudo-inverse of augmented data matrix...\n");
#endif
        mat psuedo_inverse_X_augmented = pinv(augmented_X);

        //		psuedo_inverse_X_augmented.print();

        w = psuedo_inverse_X_augmented * ys;

    }

    else{
#if 0
        printf("Regularization...\n");
#endif
        mat XtX = trans(augmented_X)*augmented_X;

        XtX = XtX + lambda*eye(XtX.n_rows,XtX.n_rows);

        w = inv(XtX)*trans(augmented_X)*ys;

    }


    return 0;
} 



double H(int n,double x)
{
    if (n == 0) return 1.0;
    else if (n == 1) return (2*x);
    else return (2 * x * H(n-1,x) - 2 * (n-1) * H(n-2,x));
}

double T(int n,double x)
{
    if (n == 0) return 1.0;
    else if (n == 1) return (x);
    else return (2 * x * T(n-1,x) -  T(n-2,x));
}



double test_function1D3(double *x){

    return T(10,x[0])+x[0]*x[0];

}




double test_function1D2(double *x){

    double x4 = pow(x[0],4);
    double x3 = pow(x[0],3);
    double x2 = pow(x[0],2);

    double noise= 2*RandomDouble(-1,1);

    return 0.2*x4+0.1*x3-2*x2+2*x[0]-2+noise;
}

double test_function1D2_without_noise(double *x){

    double x4 = pow(x[0],4);
    double x3 = pow(x[0],3);
    double x2 = pow(x[0],2);

    return 0.2*x4+0.1*x3-2*x2+2*x[0]-2;
}




void test_linear_regression(void){
    int i;
    printf("testing linear regression...\n");

    int N=20; /* number of data points */
    int degree_of_polynomial = 15;

    double bounds[2];
    bounds[0]=-1;
    bounds[1]=1;

    vec ys(N);
    mat X(N,degree_of_polynomial);

    FILE *lin_reg_points=fopen("lin_reg_points.dat","w");

    for(i=0; i<N; i++){

        double x = RandomDouble(bounds[0], bounds[1]);

        X(i,0) = (x-bounds[0]) / (bounds[1]-bounds[0]);
#if 1
        printf("%10.7f ", X(i,0));
#endif
        for(int j=1; j< degree_of_polynomial; j++){

            X(i,j) = pow( X(i,0), j+1);
#if 1
            printf("%10.7f ", X(i,j));
#endif

        }
#if 1
        printf("\n");
#endif


        ys(i) = test_function1D3(&x);
        fprintf(lin_reg_points,"%10.7f %10.7f\n",x,ys(i) );
    }

    fclose(lin_reg_points);

#if 0
    X.print();
    ys.print();
#endif


    vec w(degree_of_polynomial+1);
    train_linear_regression(X,ys,w,0);


    std::string filename= "test_lin_reg.dat";

    std::string file_name_for_plot = "test_lin_reg.png";

    printf("opening file %s for output...\n",filename.c_str());
    FILE *test_function_data=fopen(filename.c_str(),"w");


    int resolution=1000;

    double dx; /* step size in x*/
    double x;
    double func_val,func_val_ex;
    dx = (bounds[1]-bounds[0])/(resolution-1);


    double Ein =0;
    for(i=0; i<N; i++){
        rowvec xi(degree_of_polynomial+1);
        xi(0)=1.0;

        for(int j=1; j<degree_of_polynomial+1; j++){

            xi(j)=X(i,j-1);

        }

        func_val = dot(xi,w);
        func_val_ex =ys(i);
        printf("%10.7f %10.7f\n",func_val, func_val_ex);
        Ein += (func_val-func_val_ex)*(func_val-func_val_ex);

    }

    Ein = (Ein/N);

    printf("Ein = %10.7f\n",Ein);


    double Eout=0.0;
    x= bounds[0];
    for(int i=0;i<resolution;i++){
        rowvec xi(degree_of_polynomial+1);
        xi(0)=1.0;
        xi(1)=(x-bounds[0])/(bounds[1]-bounds[0]) ;

        for(int j=2;j<degree_of_polynomial+1;j++){

            xi(j)=pow(xi(1),j) ;
        }

        func_val = dot(xi,w);
        func_val_ex = test_function1D3(&x);
        Eout += (func_val-func_val_ex)*(func_val-func_val_ex);
        fprintf(test_function_data,"%10.7f %10.7f %10.7f\n",x,func_val,func_val_ex);
        x+= dx;

    }


    Eout = (Eout/resolution);

    printf("Eout = %10.7f\n",Eout);

    fclose(test_function_data);

    std::string python_command = "python -W ignore plot_1d_function_linreg.py "+ filename+ " "+ file_name_for_plot ;

    FILE* in = popen(python_command.c_str(), "r");

    fprintf(in, "\n");


}











