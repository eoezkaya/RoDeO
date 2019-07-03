#ifndef TEST_FUNCTIONS_HPP
#define TEST_FUNCTIONS_HPP
#include "Rodeo_macros.hpp"
#include <string>

class Function_param {
public:
    unsigned int number_of_independents;

    std::string function_name;
    std::string adjoint_function_name;
    double (*func_ptr)(double *);
    double (*adj_ptr)(double *, double *);
    double noise_level;

    /* x is between bounds[0] and bounds[1]; y is between bounds[2] and bounds[3] and so on  */
    double *bounds;

    Function_param (){

        number_of_independents = 0;
        function_name = "None";
        adjoint_function_name = "None";
        func_ptr = NULL;
        bounds=NULL;
        adj_ptr=NULL;
        noise_level = 0.0;

    }


    Function_param (int num){

            number_of_independents = num;
            function_name = "None";
            adjoint_function_name = "None";
            func_ptr = NULL;
            printf("allocating array for bound\n");
            bounds= (double *) malloc(sizeof(double)*2*num);
            adj_ptr=NULL;
            noise_level = 0.0;

    }



    void print(void){
        printf("printing function information...\n");
        printf("function name = %s\n",function_name.c_str());
        printf("Number of independent variables = %d\n",number_of_independents);


    }
} ;


class Classifier_Function_param {
public:
    unsigned int number_of_independents;
    std::string function_name;
    void (*func_ptr)(double *, double *, double *);
    double noise_level;

    /* x is between bounds[0] and bounds[1]; y is between bounds[2] and bounds[3] and so on  */
    double *bounds;

    Classifier_Function_param (){

        number_of_independents = 0;
        function_name = "None";
        func_ptr = NULL;
        bounds=NULL;
        noise_level = 0.0;

    }


    Classifier_Function_param (int num){

            number_of_independents = num;
            function_name = "None";
            func_ptr = NULL;
            bounds= (double *) malloc(sizeof(double)*2*num);
            noise_level = 0.0;

    }



    void print(void){
        printf("printing function information...\n");
        printf("function name = %s\n",function_name.c_str());
        printf("Number of independent variables = %d\n",number_of_independents);


    }
} ;

/* classification test functions */
void classification_test1(double *, double *, double *);
void classification_test2(double *x, double *label, double *y);
/* classification test functions */




/* regression test functions */
double test_function1D(double *x);
double test_function1D_adj(double *x, double *xb);

double test_function1KernelReg(double *x);
double test_function1KernelRegAdj(double *xin, double *xb);

double test_function2KernelReg(double *x);
double test_function2KernelRegAdj(double *x,double *xb);


double simple_2D_linear_test_function1(double *x);
double simple_2D_linear_test_function1_adj(double *x, double *xb);


double Eggholder(double *x);
double EggholderAdj(double *x, double *xb);


double Waves2D(double *x);
double Waves2D_adj(double *x,double *xb);

double Herbie2D(double *x);
double Herbie2D_adj(double *x, double *xb);


double McCormick(double *x);
double McCormick_adj(double *x, double *xb);

double Goldstein_Price(double *x);
double Goldstein_Price_adj(double *x, double *xb);

double Rosenbrock(double *x);
double Rosenbrock_adj(double *x, double *xb);


double Rosenbrock3D(double *x);
double Rosenbrock3D_adj(double *x, double *xb);

double Rosenbrock4D(double *x);
double Rosenbrock4D_adj(double *x, double *xb);



double Rosenbrock8D(double *x);
double Rosenbrock8D_adj(double *x, double *xb) ;

double Shubert(double *x);
double Shubert_adj(double *x, double *xb);

double Himmelblau(double *x);
double Himmelblau_adj(double *x, double *xb);



double Borehole(double *x);
double Borehole_adj(double *x, double *xb);


double Wingweight(double *x);
double WingweightAdj(double *xin, double *xb);

/* regression test functions */


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


void perform_NNregression_test(double (*test_function)(double *),
		double *bounds,
		std::string function_name ,
		int  number_of_samples,
		int sampling_method,
		int problem_dimension,
		int number_of_trials);

void perform_GEK_test(double (*test_function)(double *),
		double (*test_function_adj)(double *, double *),
		double *bounds,
		std::string function_name ,
		int  number_of_samples_with_only_f_eval,
		int number_of_samples_with_g_eval,
		int sampling_method,
		int method_for_solving_lin_eq,
		int dim,
		int linear_regression,
		std::string python_dir);


void perform_trust_region_GEK_test(double (*test_function)(double *),
		double (*test_function_adj)(double *, double *),
		double *bounds, std::string function_name ,
		int  number_of_samples_with_only_f_eval,
		int number_of_samples_with_g_eval,
		int sampling_method,
		int method_for_solving_lin_eq,
		int dim,
		int linear_regression,
		std::string python_dir);




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
		int sampling_method,
		std::string python_dir);


void generate_highdim_test_function_data_GEK(double (*test_function)(double *),
		double (*test_function_adj)(double *, double *),
		std::string filename,
		double *bounds,
		int dim,
		int number_of_samples_with_only_f_eval,
		int number_of_samples_with_g_eval,
		int sampling_method);

void generate_highdim_test_function_data_cuda(double (*test_function)(double *),
		double (*test_function_adj)(double *, double *),
		std::string filename,
		double *bounds,
		int dim,
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

void perform_kernel_regression_test(double (*test_function)(double *),
		double (*test_function_adj)(double *, double *),
		double *bounds,
		std::string function_name ,
		int  number_of_samples_with_only_f_eval,
		int number_of_samples_with_g_eval,
		int sampling_method,
		int dim,
		std::string python_dir);

void perform_kernel_regression_test_highdim(double (*test_function)(double *),
		double (*test_function_adj)(double *, double *),
		double *bounds,
		std::string function_name ,
		int  number_of_samples_with_only_f_eval,
		int number_of_samples_with_g_eval,
		int sampling_method,
		int dim);

void perform_kernel_regression_test_highdim_cuda(double (*test_function)(double *),
		double (*test_function_adj)(double *, double *),
		double *bounds,
		std::string function_name ,
		int  number_of_samples_with_only_f_eval,
		int number_of_samples_with_g_eval,
		int sampling_method,
		int dim);

void perform_rbf_test(double (*test_function)(double *),
        double *bounds,
        std::string function_name ,
        int  number_of_samples,
        int sampling_method,
        int problem_dimension,
        RBF_TYPE rbf_type);

void test_norms(int dim);


#endif
