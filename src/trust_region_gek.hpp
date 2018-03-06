#ifndef TRAIN_TR_GEK_HPP
#define TRAIN_TR_GEK_HPP


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

#endif
