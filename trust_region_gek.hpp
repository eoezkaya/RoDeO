#ifndef TRAIN_TR_GEK_HPP
#define TRAIN_TR_GEK_HPP



int train_TRGEK_response_surface(std::string input_file_name,
		int linear_regression,
		mat &regression_weights,
		mat &kriging_params,
		mat &R_inv_ys_min_beta,
		mat newpoints,
		vec &beta0,
		int &max_number_of_function_calculations,
		int dim);

int train_TRGEK_response_surfacev2(std::string input_file_name,
		int linear_regression,
		mat &regression_weights,
		mat &kriging_params,
		mat &R_inv_ys_min_beta,
		double &radius,
		vec &beta0,
		int &max_number_of_function_calculations,
		int dim);

#endif
