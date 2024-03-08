/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, RPTU)
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
 * General Public License along with RoDeO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */


#include <cassert>
#include "./INCLUDE/design.hpp"
#include "../Auxiliary/INCLUDE/auxiliary_functions.hpp"
#include "../LinearAlgebra/INCLUDE/matrix_operations.hpp"
#include "../LinearAlgebra/INCLUDE/vector_operations.hpp"
#include "../INCLUDE/Rodeo_macros.hpp"




#include <armadillo>
using namespace arma;


Design::Design(rowvec dv){

	dimension = dv.size();
	designParameters = dv;
	gradient = zeros<rowvec>(dimension);
	gradientLowFidelity = zeros<rowvec>(dimension);

}

Design::Design(unsigned int dim){

	dimension = dim;
	designParameters = zeros<rowvec>(dimension);
	gradient = zeros<rowvec>(dimension);
	gradientLowFidelity = zeros<rowvec>(dimension);
}


void Design::setDimension(unsigned int dim){

	dimension = dim;
	designParameters = zeros<rowvec>(dimension);
	gradient = zeros<rowvec>(dimension);
	gradientLowFidelity = zeros<rowvec>(dimension);
}



void Design::setNumberOfConstraints(unsigned int howManyConstraints){

	assert(dimension>0);

	numberOfConstraints = howManyConstraints;
	constraintTrueValues = zeros<rowvec>(numberOfConstraints);
	constraintTrueValuesLowFidelity = zeros<rowvec>(numberOfConstraints);
	constraintTangent = zeros<rowvec>(numberOfConstraints);
	constraintTangentLowFidelity = zeros<rowvec>(numberOfConstraints);

	constraintGradientsMatrix = zeros<mat>(numberOfConstraints, dimension);
	constraintGradientsMatrixLowFi = zeros<mat>(numberOfConstraints, dimension);

	constraintDifferentiationDirectionsMatrix = zeros<mat>(numberOfConstraints, dimension);
	constraintDifferentiationDirectionsMatrixLowFi = zeros<mat>(numberOfConstraints, dimension);


}

void Design::generateRandomDesignVector(vec lb, vec ub){
	designParameters = generateRandomVector<rowvec>(lb,ub);
}

void Design::generateRandomDesignVector(double lb, double ub){

	designParameters = generateRandomVector<rowvec>(lb,ub,dimension);
}


void Design::generateRandomDifferentiationDirection(void) {

	rowvec direction =  generateRandomVector<rowvec>(-1.0,1.0,dimension);
	tangentDirection =  makeUnitVector(direction);
}

rowvec Design::constructSampleObjectiveFunction(void) const{

	rowvec sample(dimension+1);
	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension) = trueValue;
	return sample;
}


rowvec Design::constructSampleObjectiveFunctionLowFi(void) const{

	rowvec sample(dimension+1);
	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension) = trueValueLowFidelity;

	return sample;
}



rowvec Design::constructSampleObjectiveFunctionWithTangent(void) const{

	assert(tangentDirection.size() == dimension);
	rowvec sample(2*dimension+2, fill::zeros);

	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension)   = trueValue;
	sample(dimension+1) = tangentValue;

	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+2+i) = tangentDirection(i);
	}

	return sample;
}

rowvec Design::constructSampleObjectiveFunctionWithTangentLowFi(void) const{

	assert(tangentDirection.size() == dimension);
	rowvec sample(2*dimension+2, fill::zeros);

	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension)   = trueValueLowFidelity;
	sample(dimension+1) = tangentValueLowFidelity;

	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+2+i) = tangentDirection(i);
	}

	return sample;
}

rowvec Design::constructSampleObjectiveFunctionWithGradient(void) const{

	assert(gradient.size() == dimension);
	rowvec sample(2*dimension+1);

	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension) = trueValue;

	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+1+i) = gradient(i);
	}
	return sample;
}

rowvec Design::constructSampleObjectiveFunctionWithZeroGradient(void) const{

	assert(gradient.size() == dimension);
	rowvec sample(2*dimension+1);

	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension) = trueValue;

	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+1+i) = 0.0;
	}
	return sample;
}



rowvec Design::constructSampleObjectiveFunctionWithGradientLowFi(void) const{

	assert(gradientLowFidelity.size() == dimension);
	rowvec sample(2*dimension+1);

	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension) = trueValueLowFidelity;

	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+1+i) = gradientLowFidelity(i);
	}
	return sample;
}

rowvec Design::constructSampleConstraint(int constraintID) const{

	assert(constraintID < int(numberOfConstraints));
	assert(constraintTrueValues.size() == numberOfConstraints);

	rowvec sample(dimension+1);
	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension) = constraintTrueValues(constraintID);

	return sample;
}

rowvec Design::constructSampleConstraintLowFi(int constraintID) const{

	assert(constraintID < int(numberOfConstraints));
	assert(constraintTrueValuesLowFidelity.size() == numberOfConstraints);

	rowvec sample(dimension+1);
	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension) = constraintTrueValuesLowFidelity(constraintID);

	return sample;
}


rowvec Design::constructSampleConstraintWithTangent(int constraintID) const{

	assert(constraintID < int(numberOfConstraints));
	assert(constraintTrueValues.size() == numberOfConstraints);
	assert(constraintDifferentiationDirectionsMatrix.n_rows == numberOfConstraints);
	assert(constraintTangent.size() == numberOfConstraints);

	rowvec sample(2*dimension+2);
	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension) = constraintTrueValues(constraintID);
	sample(dimension+1) = constraintTangent(constraintID);

	rowvec direction = constraintDifferentiationDirectionsMatrix.row(constraintID);

	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+2+i) = direction(i);
	}

	return sample;
}

rowvec Design::constructSampleConstraintWithTangentLowFi(int constraintID) const{

	assert(constraintID < int(numberOfConstraints));
	assert(constraintTrueValuesLowFidelity.size() == numberOfConstraints);
	assert(constraintDifferentiationDirectionsMatrixLowFi.n_rows == numberOfConstraints);
	assert(constraintTangentLowFidelity.size() == numberOfConstraints);

	rowvec sample(2*dimension+2);
	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension)   = constraintTrueValuesLowFidelity(constraintID);
	sample(dimension+1) = constraintTangentLowFidelity(constraintID);

	rowvec direction = constraintDifferentiationDirectionsMatrixLowFi.row(constraintID);

	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+2+i) = direction(i);
	}

	return sample;
}

rowvec Design::constructSampleConstraintWithGradient(int constraintID) const{

	assert(constraintID < int(numberOfConstraints));
	assert(constraintTrueValues.size() == numberOfConstraints);

	rowvec sample(2*dimension+1);
	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension) = constraintTrueValues(constraintID);
	rowvec constraintGradient = constraintGradientsMatrix.row(constraintID);
	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+1+i) = constraintGradient(i);
	}

	return sample;
}

rowvec Design::constructSampleConstraintWithGradientLowFi(int constraintID) const{

	assert(constraintID < int(numberOfConstraints));
	assert(constraintTrueValuesLowFidelity.size() == numberOfConstraints);

	rowvec sample(2*dimension+1);
	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension) = constraintTrueValuesLowFidelity(constraintID);
	rowvec constraintGradient = constraintGradientsMatrixLowFi.row(constraintID);
	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+1+i) = constraintGradient(i);
	}

	return sample;
}





bool Design::checkIfHasNan(void) const{

	bool ifHasNan = false;
	if(std::isnan(trueValue )) {

		ifHasNan = true;
	}
	if(gradient.has_nan()){

		ifHasNan = true;
	}

	if(constraintTrueValues.has_nan()){

		ifHasNan = true;
	}

	for (auto it = constraintGradients.begin(); it != constraintGradients.end(); it++){

		if(it->has_nan()){

			ifHasNan = true;
		}

	}
	return ifHasNan;


}



Design::Design(void){}

void Design::reset(void){

	designParameters.fill(0.0);
	designParametersNormalized.fill(0.0);
	constraintTrueValues.fill(0.0);
	constraintEstimates.fill(0.0);
	gradient.fill(0.0);
	gradientLowFidelity.fill(0.0);
	tangentDirection.fill(0.0);

	trueValue = 0.0;
	estimatedValue = 0.0;
	trueValueLowFidelity = 0;
	tangentValue = 0.0;
	tangentValueLowFidelity = 0.0;
	improvementValue = 0.0;

	ExpectedImprovementvalue = 0.0;


}

void Design::print(void) const{


	std::cout<< "\n***************** " << tag << " *****************\n";
	std::cout<<"Design parameters = \n";
	designParameters.print();

	std::cout<<"Function value = "<<trueValue<<"\n";

	if(fabs(trueValueLowFidelity) > 0.0 ){
		std::cout<<"Function value Low Fidelity = "<<trueValueLowFidelity<<"\n";
	}

	if(!gradient.is_zero() && gradient.size() > 0){
		gradient.print("gradient vector");

	}

	if(!gradientLowFidelity.is_zero() && gradientLowFidelity.size() > 0){
		gradientLowFidelity.print("gradient vector (Low Fi)");
	}


	if(constraintTrueValues.size() > 0){
		constraintTrueValues.print("constraint values");
	}

	if(!constraintGradients.empty()){

		int count = 0;
		for (auto it = constraintGradients.begin(); it != constraintGradients.end(); it++){

			if(!it->is_zero()){
				std::cout<<"Constraint gradient "<<count<<"\n";
				it->print();
				count++;
			}
		}
	}

	if(isDesignFeasible){
		std::cout<<"Feasibility = " << "YES"<<"\n";
	}
	else{
		std::cout<<"Feasibility = " << "NO"<<"\n";
	}
	std::cout<<"Improvement = "<<improvementValue<<"\n";
	std::cout<< "*********************************************************\n\n";

}



void Design::saveToAFile(std::string filename) const{

	assert(!filename.empty());

	std::ofstream fileOut;
	fileOut.open (filename);
	fileOut << tag<<"\n";
	fileOut << "Design parameters vector:\n";
	fileOut << designParameters;

	fileOut << std::fixed << "Objective function = " << trueValue << "\n";

	if(numberOfConstraints>0){

		fileOut << "Constraint values vector:\n";
		fileOut << constraintTrueValues;

	}

	if(isDesignFeasible){
		fileOut << "Feasibility = YES\n";
	}
	else{
		fileOut << "Feasibility = NO\n";
	}

	fileOut.close();

}
//
//
//void Design::saveToXMLFile(std::string filename) const{
//
//	assert(isNotEmpty(filename));
//	std::ofstream file(filename);
//	if (!file.is_open()) {
//		std::cerr << "Error opening file: " << filename << std::endl;
//		return;
//	}
//	file << "<Design>" << std::endl;
//	writeXmlElement(file, "DesignID", ID);
//	writeXmlElement(file, "ObjectiveFunction", trueValue);
//	writeXmlElementVector(file, "DesignParameters", designParameters);
//	writeXmlElementVector(file, "ConstraintValues", constraintTrueValues);
//	if(isDesignFeasible){
//		writeXmlElement(file, "Feasibility", "YES");
//	}
//	else{
//		writeXmlElement(file, "Feasibility", "NO");
//	}
//
//	file << "</Design>" << std::endl;
//	file.close();
//}
//
//
//void Design::readFromXmlFile(const std::string& filename) {
//
//	assert(isNotEmpty(filename));
//	std::ifstream file(filename);
//	if (!file.is_open()) {
//		std::cerr << "Error opening file: " << filename << std::endl;
//		return;
//	}
//
//	std::string line;
//	std::string tag;
//	std::string content;
//
//	while (std::getline(file, line)) {
//		std::istringstream iss(line);
//		if (std::getline(iss, tag, '>') && std::getline(iss, content, '<')) {
//			trim(tag);
//			trim(content);
//
//			if (tag == "DesignID") {
//				ID = std::stoi(content);
//			} else if (tag == "ObjectiveFunction") {
//				surrogateEstimate = std::stod(content);
//			} else if (tag == "DesignParameters") {
//				readVectorFromXmlFile(iss, designParameters);
//			} else if (tag == "ConstraintValues") {
//				readVectorFromXmlFile(iss, constraintSurrogateEstimates);
//			} else if (tag == "Feasibility") {
//				isDesignFeasible = (content == "YES");
//			}
//		}
//	}
//
//	file.close();
//}
//
//
//
//void Design::trim(std::string& str) {
//	size_t first = str.find_first_not_of(' ');
//	size_t last = str.find_last_not_of(' ');
//	str = str.substr(first, (last - first + 1));
//}
//
//// Helper function to read a vector from XML
//template <typename T>
//void Design::readVectorFromXmlFile(std::istringstream& iss, T& vec) {
//	std::string line;
//	int cnt = 0;
//	while (std::getline(iss, line)) {
//		trim(line);
//		if (line == "</Item>") {
//			break;
//		}
//		if (!line.empty()) {
//			vec(cnt)= std::stod(line);
//			cnt++;
//		}
//	}
//}
//
//template void Design::readVectorFromXmlFile(std::istringstream& iss, rowvec& vec);
//
//
//
//template <typename T>
//void Design::writeXmlElement(std::ofstream& file, const std::string& elementName, const T& value) const {
//	file << "\t<" << elementName << ">" << value << "</" << elementName << ">" << std::endl;
//}
//
//template <typename T>
//void Design::writeXmlElementVector(std::ofstream& file, const std::string& elementName, const T& values) const {
//	file << "\t<" << elementName << ">" << std::endl;
//	for (unsigned int i=0; i<values.size(); i++) {
//		double val = values(i);
//		file << "\t\t<Item>" << val << "</Item>" << std::endl;
//	}
//	file << "\t</" << elementName << ">" << std::endl;
//}
//
//
//
//template void Design::writeXmlElementVector(std::ofstream& file, const std::string& elementName, const rowvec& values) const;
//template void Design::writeXmlElement(std::ofstream& file, const std::string& elementName, const unsigned int& value) const;
//template void Design::writeXmlElement(std::ofstream& file, const std::string& elementName, const double& value) const;
//template void Design::writeXmlElement(std::ofstream& file, const std::string& elementName, const string& value) const;

void Design::saveDesignVectorAsCSVFile(std::string fileName) const{

	std::ofstream designVectorFile (fileName);
	designVectorFile.precision(10);
	if (designVectorFile.is_open())
	{
		for(unsigned int i=0; i<designParameters.size()-1; i++){
			designVectorFile << designParameters(i)<<",";
		}

		designVectorFile << designParameters(designParameters.size()-1);

		designVectorFile.close();
	}
	else{
		abortWithErrorMessage("ERROR: Unable to open file");
	}

}

void Design::saveDesignVector(std::string fileName) const{

	assert(isNotEmpty(fileName));
	std::ofstream designVectorFile (fileName);
	designVectorFile.precision(10);
	if (designVectorFile.is_open())
	{
		for(unsigned int i=0; i<designParameters.size(); i++){
			designVectorFile << designParameters(i)<<"\n";
		}
		designVectorFile.close();
	}
	else{
		abortWithErrorMessage("ERROR: Unable to open file");
	}
}


/************************************************************************************/



DesignForBayesianOptimization::DesignForBayesianOptimization(){};

DesignForBayesianOptimization::DesignForBayesianOptimization(unsigned int dimension, unsigned int numberOfConstraints){

	dim = dimension;
	constraintValues = zeros<rowvec>(numberOfConstraints);
	constraintSigmas = zeros<rowvec>(numberOfConstraints);
}

DesignForBayesianOptimization::DesignForBayesianOptimization(unsigned int dimension){
	dim = dimension;
}

DesignForBayesianOptimization::DesignForBayesianOptimization(rowvec designVector, unsigned int numberOfConstraints){

	dv = designVector;
	dim = designVector.size();
	constraintValues = zeros<rowvec>(numberOfConstraints);
	constraintSigmas = zeros<rowvec>(numberOfConstraints);

}

DesignForBayesianOptimization::DesignForBayesianOptimization(rowvec designVector){

	dv = designVector;
	dim = designVector.size();

}

void DesignForBayesianOptimization::gradientUpdateDesignVector(const rowvec &gradient, const vec &lb, const vec &ub, double stepSize){


	/* we go in the direction of gradient since we maximize */
	dv = dv + stepSize*gradient;


	for(unsigned int k=0; k<dim; k++){

		/* if new design vector does not satisfy the box constraints */
		if(dv(k) < lb(k)) dv(k) = lb(k);
		if(dv(k) > ub(k)) dv(k) = ub(k);

	}

}


void DesignForBayesianOptimization::generateRandomDesignVector(void){

	double lowerBound = 0.0;
	double upperBound = 1.0/dim;
	dv = generateRandomVector<rowvec>(lowerBound, upperBound , dim);


}

void DesignForBayesianOptimization::generateRandomDesignVector(vec lb, vec ub){
	dv = generateRandomVector<rowvec>(lb, ub);
}

void DesignForBayesianOptimization::generateRandomDesignVectorAroundASample(const rowvec &sample, vec lb, vec ub, double factor){

	assert(sample.size() == dim);

	vec lowerBounds(dim, fill::zeros);
	vec upperBounds(dim, fill::zeros);


	double dx = factor/dim;

	for(unsigned int i=0; i<dim; i++){

		lowerBounds(i) = sample(i) - dx;
		upperBounds(i) = sample(i) + dx;
		if(lowerBounds(i) < lb(i))    lowerBounds(i) = lb(i);
		if(upperBounds(i) > ub(i))    upperBounds(i) = ub(i);

	}
	dv = generateRandomVector<rowvec>(lowerBounds, upperBounds);
}

void DesignForBayesianOptimization::print(void) const{
	std::cout.precision(15);
	std::cout<<tag<<"n";
	std::cout<<"Design vector = \n";
	dv.print();
	std::cout<<"Objective function value = "<<objectiveFunctionValue<<"\n";
	std::cout<<"Sigma = " << sigma << "\n";
	std::cout<<"Acquisition function value = "<< valueAcqusitionFunction <<"\n";

	if(constraintValues.size() > 0){

		std::cout<<"Constraint values = \n";
		constraintValues.print();
	}
	if(constraintValues.size() > 0){

		std::cout<<"Constraint feasibility probabilities = \n";
		constraintFeasibilityProbabilities.print();

	}


	std::cout<<"\n";
}


double DesignForBayesianOptimization::calculateProbalityThatTheEstimateIsLessThanAValue(double value){

	return cdf(value, objectiveFunctionValue, sigma) ;


}
double DesignForBayesianOptimization::calculateProbalityThatTheEstimateIsGreaterThanAValue(double value){


	return 1.0 - cdf(value, objectiveFunctionValue, sigma) ;

}


void DesignForBayesianOptimization::updateAcqusitionFunctionAccordingToConstraints(void){


	for(unsigned int i=0; i<constraintFeasibilityProbabilities.size(); i++){

		valueAcqusitionFunction *= constraintFeasibilityProbabilities(i);
	}


}


void DesignForBayesianOptimization::reset(void){

	constraintFeasibilityProbabilities.fill(0.0);
	constraintSigmas.fill(0.0);
	constraintValues.fill(0.0);
	dv.fill(0.0);
	objectiveFunctionValue  = 0.0;
	sigma  = 0.0;
	valueAcqusitionFunction = 0.0;

}

