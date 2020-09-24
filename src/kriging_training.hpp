#ifndef TRAIN_KRIGING_HPP
#define TRAIN_KRIGING_HPP
#include <armadillo>
#include "Rodeo_macros.hpp"
#include "surrogate_model.hpp"
#include "linear_regression.hpp"



using namespace arma;



class KrigingModel : public SurrogateModel{

private:
	vec kriging_weights;
	vec R_inv_ys_min_beta;
	vec R_inv_I;
	vec vectorOfOnes;
	mat correlationMatrix;
	mat upperDiagonalMatrix;

	double beta0;
	double sigmaSquared;

	bool linear_regression;
	double epsilonKriging;

	double genErrorKriging;

	LinearModel linearModel;


	void updateWithNewData(void);
	void updateModelParams(void);
	void computeCorrelationMatrix(void);
	vec computeCorrelationVector(rowvec x) const;
	double computeCorrelation(rowvec x_i, rowvec x_j, vec theta, vec gamma) const;

public:

	unsigned int max_number_of_kriging_iterations;

	KrigingModel();
	KrigingModel(std::string name);

	void initializeSurrogateModel(void);
	void printSurrogateModel(void) const;
	void printHyperParameters(void) const;
	void saveHyperParameters(void) const;
	void loadHyperParameters(void);
	void train(void);
	double interpolateWithGradients(rowvec x) const ;
	double interpolate(rowvec x) const ;
	void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const;


	double calculateExpectedImprovement(rowvec xp);
	double getyMin(void) const;
	void setEpsilon(double inp);
	void setLinearRegressionOn(void);
	void setLinearRegressionOff(void);
	void setNumberOfTrainingIterations(unsigned int);

	int addNewSampleToData(rowvec newsample);


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
