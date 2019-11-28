#ifndef TRAIN_TR_GEK_HPP
#define TRAIN_TR_GEK_HPP

#include "Rodeo_macros.hpp"


class AggregationModel {

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
	mat XnotNormalized;
	vec I;
	mat data;
	fmat dataSP;
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
	int max_number_of_kriging_iterations;
	unsigned int minibatchsize;

	mat M;
	int number_of_cv_iterations_rho;
	int number_of_cv_iterations;
	double rho;
	double sigma;

	std::string validationset_input_filename;
	std::string visualizeKrigingValidation;
	std::string visualizeKernelRegressionValidation;
	std::string visualizeAggModelValidation;



	AggregationModel(std::string name,int dimension);
	void update(void);
	double ftilde(rowvec xp);
	double ftildeKriging(rowvec xp);
	void train(void);

	void save_state(void){


		std::string filename = label+"_settings_save.dat";
		FILE *outp = fopen(filename.c_str(),"w");

		for(int i=0; i<dim; i++){

			fprintf(outp,"%10.7f\n",regression_weights(i));
		}
		for(int i=0; i<2*dim; i++){

			fprintf(outp,"%10.7f\n",kriging_weights(i));
		}

		for(int i=0; i<dim; i++)
			for(int j=0; j<dim; j++){

				fprintf(outp,"%10.7f\n",M(i,j));

			}

		fprintf(outp,"%10.7f\n",rho);
		fprintf(outp,"%10.7f\n",sigma);
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



		for(int i=0; i<dim; i++)
			for(int j=0; j<dim; j++){

				fscanf(inp,"%lf",&temp);
				M(i,j) = temp;
			}

		fscanf(inp,"%lf",&temp);
		rho = temp;
		fscanf(inp,"%lf",&temp);
		sigma = float(temp);
		fscanf(inp,"%lf",&temp);
		beta0 = temp;

		fclose(inp);



	}

};



int train_TRGEK_response_surface(std::string input_file_name,
		std::string hyper_parameter_file_name,
		int linear_regression,
		mat &regression_weights,
		mat &kriging_params,
		mat &R_inv_ys_min_beta,
		double &radius,
		vec &beta0,
		int &max_number_of_function_calculations,
		int dim,
		int train_hyper_param);

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
