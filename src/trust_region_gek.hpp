#ifndef TRAIN_TR_GEK_HPP
#define TRAIN_TR_GEK_HPP

class AggregationModel {

public:
	vec regression_weights;
	vec kriging_weights;
	vec R_inv_ys_min_beta;
	vec R_inv_I;
	mat R;
	mat U;
	mat L;
	mat M;
	int dim;
	int number_of_cv_iterations_rho;

	double rho;
	double sigma;
	double beta0;
	std::string label;
	std::string input_filename;
	std::string validationset_input_filename;
	std::string kriging_hyperparameters_filename;
	std::string visualizeKrigingValidation;
	std::string visualizeKernelRegressionValidation;
	std::string visualizeAggModelValidation;
	double epsilon_kriging;
	int max_number_of_kriging_iterations;
	int number_of_cv_iterations;

	AggregationModel(std::string name,int dimension){


		label = name;
		printf("Initializing settings for training %s data...\n",name.c_str());
		input_filename = name +".csv";
		kriging_hyperparameters_filename = name + "_Kriging_Hyperparameters.csv";
		visualizeKrigingValidation = "no";
		visualizeKernelRegressionValidation = "no";
		visualizeAggModelValidation = "no";
		regression_weights.zeros(dimension);
		kriging_weights.zeros(2*dimension);
		M.zeros(dimension,dimension);
		rho =  0.0;
		sigma= 0.0;
		beta0= 0.0;
		epsilon_kriging = 10E-06;
		max_number_of_kriging_iterations = 10000;
		number_of_cv_iterations_rho = 100000;
		dim = dimension;
		validationset_input_filename = "None";
		number_of_cv_iterations = 0;


	}

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

#endif
