#ifndef TRAIN_KRIGING_HPP
#define TRAIN_KRIGING_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"



using namespace arma;


class KrigingModel {

public:

	unsigned int dim;
	unsigned int N;
	bool linear_regression;
	vec regression_weights;
	vec kriging_weights;
	vec R_inv_ys_min_beta;
	vec R_inv_I;
	vec I;
	mat R;
	mat U;
	mat L;
	mat X;
	mat XnotNormalized;
	mat data;

	vec xmin;
	vec xmax;
	double beta0;
	double sigma_sqr;
	double genErrorKriging;
	std::string label;
	std::string kriging_hyperparameters_filename;
	std::string input_filename;
	double epsilon_kriging;
	unsigned int max_number_of_kriging_iterations;


	double ymin,ymax,yave;




	KrigingModel(std::string name, int dimension);
	void updateWithNewData(void);
	void updateModelParams(void);
	double ftilde(rowvec xp);
	double ftildeNotNormalized(rowvec xp);
	void ftilde_and_ssqr(rowvec xp,double *f_tilde,double *ssqr);
	double calculateEI(rowvec xp);

	void train(void);
	void validate(std::string filename, bool ifVisualize);
	void print(void);

};





class EAdesign {
public:
	double fitness;             
	double objective_val;       
	vec theta;
	vec gamma;	
	double crossover_probability;
	double death_probability;
	//	double log_regularization_parameter;
	unsigned int id;

	void print(void);
	EAdesign(int dimension);
	int calculate_fitness(double epsilon, mat &X,vec &ys);

} ;



int calculate_fitness(EAdesign &new_born,
		double &reg_param,
		mat &R,
		mat &U,
		mat &L,
		mat &X,
		vec &ys,
		vec &I);



void pickup_random_pair(std::vector<EAdesign> population, int &mother,int &father);
void crossover_kriging(EAdesign &father, EAdesign &mother, EAdesign &child);
void update_population_properties(std::vector<EAdesign> &population);


//int train_kriging_response_surface(KrigingModel &model);



#endif
