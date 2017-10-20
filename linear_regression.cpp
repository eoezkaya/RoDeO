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

    int N=8;
    int number_of_features = 10;

    double bounds[2];
    bounds[0]=-1;
    bounds[1]=1;
    //  generate_plot_1D_function(test_function1D,bounds,"test_function1D" );

    vec ys(N);
    mat X(N,number_of_features);

    FILE *lin_reg_points=fopen("lin_reg_points.dat","w");
    for(i=0; i<N; i++){
        double x = RandomDouble(bounds[0], bounds[1]);


        X(i,0) = (x-bounds[0]) / (bounds[1]-bounds[0]);
        X(i,1) = X(i,0)*X(i,0);

        X(i,2) = pow(X(i,0),3);
        X(i,3) = pow(X(i,0),4);
        X(i,4) = pow(X(i,0),5);
        X(i,5) = pow(X(i,0),6);
        X(i,6) = pow(X(i,0),7);
        X(i,7) = pow(X(i,0),8);
        X(i,8) = pow(X(i,0),9);
        X(i,9) = pow(X(i,0),10);


        ys(i) = test_function1D3(&x);
        fprintf(lin_reg_points,"%10.7f %10.7f\n",x,ys(i) );
    }

    fclose(lin_reg_points);

    #if 0
      X.print();
      ys.print();
#endif


    vec w(number_of_features+1);
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
        rowvec xi(number_of_features+1);
        xi(0)=1.0;
        xi(1)=X(i,0);
        xi(2)=X(i,1);

        xi(3)=X(i,2);
        xi(4)=X(i,3);
        xi(5)=X(i,4);
        xi(6)=X(i,5);
        xi(7)=X(i,6);
        xi(8)=X(i,7);
        xi(9)=X(i,8);
        xi(10)=X(i,9);

        double x = X(i,0)* (bounds[1]-bounds[0]) + bounds[0];


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
        rowvec xi(number_of_features+1);
        xi(0)=1.0;
        xi(1)=(x-bounds[0])/(bounds[1]-bounds[0]) ;
        xi(2)=xi(1)*xi(1);

        xi(3)=pow(xi(1),3);
        xi(4)=pow(xi(1),4);
        xi(5)=pow(xi(1),5);
        xi(6)=pow(xi(1),6);
        xi(7)=pow(xi(1),7);
        xi(8)=pow(xi(1),8);
        xi(9)=pow(xi(1),9);
        xi(10)=pow(xi(1),10);

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











