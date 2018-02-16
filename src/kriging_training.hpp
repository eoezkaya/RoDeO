#ifndef TRAIN_KRIGING_HPP
#define TRAIN_KRIGING_HPP
#include <armadillo>
using namespace arma;


//#define calculate_fitness_CHECK

class member {
public:
	double fitness;             
	double objective_val;       
	vec theta;
	vec gamma;	
	double crossover_probability;
	double death_probability;
//	double log_regularization_parameter;
	int id;

	void print(void){
		printf("id = %d,  J = %10.7f, fitness =  %10.7f cross-over p = %10.7f\n",id,objective_val,fitness,crossover_probability);

//		printf("theta = \n");
//		theta.print();
//		if(gamma.size() > 0){
//			printf("gamma = \n");
//			gamma.print();
//		}

	}
} ;



int train_kriging_response_surface(std::string input_file_name,
		std::string file_name_hyperparameters,
		int linear_regression,
		vec &regression_weights,
		vec &kriging_params,
		double &reg_param,
		int max_number_of_function_calculations,
		int data_file_format);


int train_GEK_response_surface(std::string input_file_name, 
		int linear_regression,
		vec &regression_weights, 
		vec &kriging_params, 
		double &reg_param,
		int &max_number_of_function_calculations,
		int dim,
		int eqn_sol_method);


double calculate_f_tilde(rowvec x,
		                 mat &X,
		                 double beta0,
		                 vec regression_weights,
		                 vec R_inv_ys_min_beta,
		                 vec kriging_weights);



void calculate_f_tilde_and_ssqr(
		rowvec x,
		mat &X,
		double beta0,
		double sigma_sqr,
		vec regression_weights,
		vec R_inv_ys_min_beta,
		vec R_inv_I,
		vec I,
		vec kriging_weights,
		mat &U,
		mat &L,
		double *f_tilde,
		double *ssqr);



double calculate_f_tilde_GEK(rowvec &x,
		                     mat &X,
		                     double beta0,
		                     vec regression_weights,
		                     vec R_inv_ys_min_beta,
		                     vec kriging_weights,
		                     int Ngrad);





void compute_R_matrix(vec theta, vec gamma, double reg_param,mat& R, mat &X);

void compute_R_matrix_GEK(vec theta, double reg_param, mat& R, mat &X, mat &grad);


double compute_R_Gauss(rowvec x_i, rowvec x_j, vec theta);
double compR_dxi(rowvec x_i, rowvec x_j, vec theta, int k);


void generate_validation_set(int *indices, int size, int N);

#endif
