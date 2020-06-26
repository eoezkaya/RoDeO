#ifndef TRAIN_TR_GEK_HPP
#define TRAIN_TR_GEK_HPP

#include "Rodeo_macros.hpp"
#include "kernel_regression.hpp"

class GradientKernelRegressionModel: public KernelRegressionModel{

private:

	mat gradientData;


public:

	GradientKernelRegressionModel();
	GradientKernelRegressionModel(std::string name);


};

class AggregationModel {

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
	mat grad;
	vec xmin;
	vec xmax;
	double beta0;
	double sigma_sqr;
	double genErrorKriging;
	double genErrorKernelRegression;
	double genErrorAggModel;
	std::string label;
	std::string kriging_hyperparameters_filename;
	std::string input_filename;
	double epsilon_kriging;
	unsigned int max_number_of_kriging_iterations;
	unsigned int minibatchsize;

	mat M;
	unsigned int number_of_cv_iterations_rho;
	unsigned int number_of_cv_iterations;
	double rho;
	double sigma;

	std::string validationset_input_filename;
	bool visualizeKrigingValidation;
	bool visualizeKernelRegressionValidation;
	bool visualizeAggModelValidation;

	double ymin,ymax,yave;



	AggregationModel(std::string name,int dimension);
	void update(void);
	double ftilde(rowvec xp);
	double ftildeKriging(rowvec xp);
	void ftilde_and_ssqr(rowvec xp,double *f_tilde,double *ssqr);
	void updateKrigingModel(void);
	void train(void);
	void save_state(void);
	void load_state(void);

};



//int train_TRGEK_response_surface(std::string input_file_name,
//		std::string hyper_parameter_file_name,
//		int linear_regression,
//		mat &regression_weights,
//		mat &kriging_params,
//		mat &R_inv_ys_min_beta,
//		double &radius,
//		vec &beta0,
//		int &max_number_of_function_calculations,
//		int dim,
//		int train_hyper_param);

int train_aggregation_model(AggregationModel &model_settings);

double ftildeAggModel(AggregationModel &model_settings,
		rowvec &xp,
		rowvec xpNotNormalized,
		mat &X,
		mat &XTrainingNotNormalized,
		vec &yTrainingNotNormalized,
		mat &gradTrainingNotNormalized,
		vec &x_min,
		vec &x_max);

double calcGenErrorAggModel(AggregationModel &model_settings,
		mat Xvalidation,
		mat XvalidationNotNormalized,
		vec yvalidation,
		mat X,
		mat XTrainingNotNormalized,
		vec yTrainingNotNormalized,
		mat gradTrainingNotNormalized,
		vec x_min,
		vec x_max);

#endif
