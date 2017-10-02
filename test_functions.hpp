#ifndef TEST_FUNCTIONS_HPP
#define TEST_FUNCTIONS_HPP
#include "Rodeo_macros.hpp"
#include <string>

class Function_param {
public:
    int number_of_independents;

    std::string func_name;
    std::string adjoint_name;
    double (*func_ptr)(double *);
    double (*adj_ptr)(double *, double *);

    double *bounds;

    Function_param (){

        number_of_independents = 0;
        name = "Empty";
        adjoint_name = "Empty";
        ptr = NULL;
        bounds=NULL;
        adj_ptr=NULL;

    }

    void print(void){
        printf("function name = %s\n",name.c_str());
        printf("Number of independent variables = %d\n",number_of_independents);


    }
} ;




double test_function1D(double *x);
double test_function1D_adj(double *x, double *xb);

double simple_2D_linear_test_function1(double *x);
double simple_2D_linear_test_function1_adj(double *x, double *xb);


double Eggholder(double *x);
double Eggholder_adj(double *x, double *xb);

double McCormick(double *x);
double McCormick_adj(double *x, double *xb);

double Goldstein_Price(double *x);
double Goldstein_Price_adj(double *x, double *xb);

double Rosenbrock(double *x);


double Himmelblau(double *x);
double Himmelblau_adj(double *x, double *xb);



double Borehole(double *x);
double Borehole_adj(double *x, double *xb);



void perform_kriging_test(double (*test_function)(double *),
		double *bounds,
		std::string function_name ,
		int  number_of_samples,
		int sampling_method,
		int problem_dimension,
		int method_for_solving_lin_eq_for_training,
		int method_for_solving_lin_eq_for_evaluation,
		int linear_regression = 0,
		int training_method =0);





void perform_GEK_test(double (*test_function)(double *),
		double (*test_function_adj)(double *, double *),
		double *bounds, std::string function_name ,
		int  number_of_samples_with_only_f_eval,
		int number_of_samples_with_g_eval,
		int sampling_method,
		int method_for_solving_lin_eq,
		int dim,
		int linear_regression);


void perform_trust_region_GEK_test(double (*test_function)(double *),
		double (*test_function_adj)(double *, double *),
		double *bounds, std::string function_name ,
		int  number_of_samples_with_only_f_eval,
		int number_of_samples_with_g_eval,
		int sampling_method,
		int method_for_solving_lin_eq,
		int dim,
		int linear_regression);




void perform_GEK_test1D(double (*test_function)(double *),
		double (*test_function_adj)(double *, double *),
		double *bounds, std::string function_name ,
		int  number_of_samples_with_only_f_eval,
		int number_of_samples_with_g_eval,
		int sampling_method,
		int eqn_sol_method_for_evaluation = 1,
		int linear_regression = 0);



void generate_2D_test_function_data(double (*test_function)(double *), std::string filename, double bounds[4], int number_of_function_evals, int sampling_method);

void generate_2D_test_function_data_GEK(double (*test_function)(double *),
		double (*test_function_adj)(double *, double *),
		std::string filename,
		double bounds[4],
		int number_of_samples_with_only_f_eval,
		int number_of_samples_with_g_eval,
		int sampling_method);


void generate_1D_test_function_data_GEK(double (*test_function)(double *),
		double (*test_function_adj)(double *, double *),
		std::string filename,
		double *bounds,
		int number_of_samples_with_only_f_eval,
		int number_of_samples_with_g_eval,
		int sampling_method,
		double *locations_func= NULL,
		double *locations_grad= NULL);

void generate_test_function_data(double (*test_function)(double *),
		std::string filename,
		double * bounds,
		int number_of_function_evals,
		int sampling_method,
		int problem_dimension);


void perform_rbf_test(double (*test_function)(double *),
        double *bounds,
        std::string function_name ,
        int  number_of_samples,
        int sampling_method,
        int problem_dimension,
        RBF_TYPE rbf_type);




#endif
