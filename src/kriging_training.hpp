#ifndef TRAIN_KRIGING_HPP
#define TRAIN_KRIGING_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"


using namespace arma;


//#define calculate_fitness_CHECK


class KrigingModel {

public:
	int dim;
	int N;
	int linear_regression;
	vec regression_weights;
	vec kriging_weights;
	vec R_inv_ys_min_beta;
	vec R_inv_I;
	mat R;
	mat U;
	mat L;
	mat X;
	vec I;
	mat data;
	vec xmin;
	vec xmax;
	double beta0;
	double sigma_sqr;
	double genError;
	std::string label;
	std::string kriging_hyperparameters_filename;
	std::string input_filename;
	double epsilon_kriging;
	int max_number_of_kriging_iterations;

	KrigingModel(std::string name, int dimension);
	void update(void);
	double ftilde(rowvec xp);
	double ftildeNorm(rowvec xp);
	void calculate_f_tilde_and_ssqr(rowvec xp,double *f_tilde,double *ssqr);

	void train_hyperparameters(void);

	void save_state(void){


		std::string filename = label+"_settings_save.dat";
		FILE *outp = fopen(filename.c_str(),"w");

		for(int i=0; i<dim; i++){

			fprintf(outp,"%10.7f\n",regression_weights(i));
		}
		for(int i=0; i<2*dim; i++){

			fprintf(outp,"%10.7f\n",kriging_weights(i));
		}

		fprintf(outp,"%10.7f\n",beta0);

		fclose(outp);

	}

	void load_state(void){

		double temp;
		std::string filename = label+"_settings_save.dat";
		FILE *inp = fopen(filename.c_str(),"r");

		for(int i=0; i<dim; i++){

			fscanf(inp,"%lf",&temp);
			regression_weights(i) = temp;
		}

		for(int i=0; i<2*dim; i++){

			fscanf(inp,"%lf",&temp);
			kriging_weights(i) = temp;
		}




		fscanf(inp,"%lf",&temp);
		beta0 = temp;

		fclose(inp);



	}

};





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

double compute_R(rowvec x_i, rowvec x_j, vec theta, vec gamma);

int train_kriging_response_surface(std::string input_file_name,
		std::string file_name_hyperparameters,
		int linear_regression,
		vec &regression_weights,
		vec &kriging_params,
		double &reg_param,
		int max_number_of_function_calculations,
		int data_file_format);

int train_kriging_response_surface(mat data,
		std::string file_name_hyperparameters,
		int linear_regression,
		vec &regression_weights,
		vec &kriging_params,
		double &reg_param,
		int max_number_of_function_calculations);


int train_kriging_response_surface(KrigingModel &model);


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

void compute_R_inv_ys_min_beta(mat X,
		vec ys,
		vec kriging_params,
		vec regression_weights,
		vec &res,
		double &beta0,
		double epsilon_kriging,
		int linear_regression);


#endif
