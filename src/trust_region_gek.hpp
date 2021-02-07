/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2020 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, TU Kaiserslautern)
 *
 * This file is part of RoDeO
 *
 * RoDeO is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * RoDeO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU
 * General Public License along with CoDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, TU Kaiserslautern)
 *
 *
 *
 */
#ifndef TRAIN_TR_GEK_HPP
#define TRAIN_TR_GEK_HPP

#include "Rodeo_macros.hpp"
#include "kernel_regression.hpp"
#include "kriging_training.hpp"


class AggregationModel : public SurrogateModel {

private:
	KernelRegressionModel kernelRegressionModel;
	KrigingModel krigingModel;

	double rho;
	double dr;

	PartitionData testDataForRhoOptimizationLoop;
	PartitionData trainingData;

public:

	unsigned int numberOfIterForRhoOptimization;


	AggregationModel();
	AggregationModel(std::string name);

	void initializeSurrogateModel(void);
	void printSurrogateModel(void) const;
	void printHyperParameters(void) const;
	void saveHyperParameters(void) const;
	void loadHyperParameters(void);
	void updateAuxilliaryFields(void);
	void train(void);
	double interpolate(rowvec x) const ;
	double interpolateWithGradients(rowvec x) const ;
	void interpolateWithVariance(rowvec xp,double *f_tilde,double *ssqr) const;

	unsigned int findNearestNeighbor(rowvec xp) const;


};



//class AggregationModel {
//
//public:
//
//	unsigned int dim;
//	unsigned int N;
//	bool linear_regression;
//
//	vec regression_weights;
//	vec kriging_weights;
//	vec R_inv_ys_min_beta;
//	vec R_inv_I;
//	vec I;
//
//	mat R;
//	mat U;
//	mat L;
//	mat X;
//	mat XnotNormalized;
//	mat data;
//	mat grad;
//	vec xmin;
//	vec xmax;
//	double beta0;
//	double sigma_sqr;
//	double genErrorKriging;
//	double genErrorKernelRegression;
//	double genErrorAggModel;
//	std::string label;
//	std::string kriging_hyperparameters_filename;
//	std::string input_filename;
//	double epsilon_kriging;
//	unsigned int max_number_of_kriging_iterations;
//	unsigned int minibatchsize;
//
//	mat M;
//	unsigned int number_of_cv_iterations_rho;
//	unsigned int number_of_cv_iterations;
//	double rho;
//	double sigma;
//
//	std::string validationset_input_filename;
//	bool visualizeKrigingValidation;
//	bool visualizeKernelRegressionValidation;
//	bool visualizeAggModelValidation;
//
//	double ymin,ymax,yave;
//
//
//
//	AggregationModel(std::string name,int dimension);
//	void update(void);
//	double ftilde(rowvec xp);
//	double ftildeKriging(rowvec xp);
//	void ftilde_and_ssqr(rowvec xp,double *f_tilde,double *ssqr);
//	void updateKrigingModel(void);
//	void train(void);
//	void save_state(void);
//	void load_state(void);
//
//};



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

//int train_aggregation_model(AggregationModel &model_settings);
//
//double ftildeAggModel(AggregationModel &model_settings,
//		rowvec &xp,
//		rowvec xpNotNormalized,
//		mat &X,
//		mat &XTrainingNotNormalized,
//		vec &yTrainingNotNormalized,
//		mat &gradTrainingNotNormalized,
//		vec &x_min,
//		vec &x_max);
//
//double calcGenErrorAggModel(AggregationModel &model_settings,
//		mat Xvalidation,
//		mat XvalidationNotNormalized,
//		vec yvalidation,
//		mat X,
//		mat XTrainingNotNormalized,
//		vec yTrainingNotNormalized,
//		mat gradTrainingNotNormalized,
//		vec x_min,
//		vec x_max);

#endif
