#ifndef TEST_FUNCTIONS_HPP
#define TEST_FUNCTIONS_HPP
#include "Rodeo_macros.hpp"
#include <string>

#include <armadillo>



using namespace arma;


class TestFunction {

public:
	bool ifNoisy;
	bool ifAdjointFunctionExist;
    unsigned int numberOfInputParams;
    std::string function_name;
    double (*func_ptr)(double *);
    double (*adj_ptr)(double *, double *);
    double noise_level;

    vec lower_bound;
    vec upper_bound;

    TestFunction(std::string name,int dimension);
    void plot(void);
    void print(void);
    void generateRandomSamples(unsigned int nsamples, std::string filename);
    void testKrigingModel(int nsamples, bool ifVisualize = false);
    void testEGO(int nsamplesTrainingData, int maxnsamples, bool ifVisualize = false, bool ifWarmStart = false);

} ;







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
double Eggholder_adj(double *x, double *xb);

double Griewank2D(double *x);
double Griewank2D_adj(double *xin, double *xb);


double Rastrigin6D(double *x);
double Rastrigin6D_adj(double *xin, double *xb);



double Waves2D(double *x);
double Waves2D_adj(double *x,double *xb);

double Waves2Dperturbed(double *x);
double Waves2Dperturbed_adj(double *x,double *xb);

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


double empty(double *x);
double emptyAdj(double *x, double *xb);

/* regression test functions */


void perform_kriging_test(double (*test_function)(double *),
		double *bounds,
		std::string function_name ,
		int  number_of_samples,
		int sampling_method,
		int problem_dimension,
		int linear_regression = 0);




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
		std::string filename,
		double *bounds,
		int number_of_samples,
		int dim);

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



void perform_kernel_regression_test_highdim(double (*test_function)(double *),
		double (*test_function_adj)(double *, double *),
		double *bounds,
		std::string function_name ,
		int  number_of_samples_with_only_f_eval,
		int number_of_samples_with_g_eval,
		int sampling_method,
		int dim);


void perform_kernel_regression_test_highdim(double (*test_function)(double *),
		double *bounds,
		std::string function_name,
		int number_of_samples,
		int sampling_method,
		int dim);


void perform_aggregation_model_test_highdim(double (*test_function)(double *),
		double (*test_function_adj)(double *, double *),
		double *bounds,
		std::string function_name,
		int number_of_training_samples,
		int sampling_method,
		int dim);






#endif
