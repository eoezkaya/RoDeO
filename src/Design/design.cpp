#include <iostream>
#include <fstream>
#include <cmath>
#include<string>
#include "./INCLUDE/design.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>


using namespace std;

namespace Rodop{

Design::Design(){}


Design::Design(vec dv){

	dimension = dv.getSize();
	designParameters = dv;
	gradient.resize(dimension);
	gradientLowFidelity.resize(dimension);

}



Design::Design(int dim){


	if (dim <= 0) {
	    throw std::invalid_argument("Dimension must be greater than 0.");
	}

	dimension = dim;
	designParameters.resize(dimension);
	gradient.resize(dimension);
	gradientLowFidelity.resize(dimension);
}


void Design::setDimension(unsigned int dim){

	if (dim <= 0) {
	    throw std::invalid_argument("Dimension must be greater than 0.");
	}


	dimension = dim;
	designParameters.resize(dimension);
	gradient.resize(dimension);
	gradientLowFidelity.resize(dimension);
}



void Design::setNumberOfConstraints(unsigned int howManyConstraints){

	if (dimension <= 0) {
	    throw std::invalid_argument("Dimension must be greater than 0.");
	}


	numberOfConstraints = howManyConstraints;
	constraintTrueValues.resize(numberOfConstraints);
	constraintTrueValuesLowFidelity.resize(numberOfConstraints);
	constraintTangent.resize(numberOfConstraints);
	constraintTangentLowFidelity.resize(numberOfConstraints);

	constraintGradientsMatrix.resize(numberOfConstraints, dimension);
	constraintGradientsMatrixLowFi.resize(numberOfConstraints, dimension);
	constraintDifferentiationDirectionsMatrix.resize(numberOfConstraints, dimension);
	constraintDifferentiationDirectionsMatrixLowFi.resize(numberOfConstraints, dimension);

}

void Design::generateRandomDesignVector(vec lb, vec ub){
	designParameters.fillRandom(lb,ub);
}

void Design::generateRandomDesignVector(double lb, double ub){

	designParameters.fillRandom(lb,ub);
}


void Design::generateRandomDifferentiationDirection(void) {

	vec direction(dimension);
	direction.fillRandom(-1.0,1.0);
	tangentDirection = direction.unitVector();
}

vec Design::constructSampleObjectiveFunction(void) const{

	vec sample(dimension+1);
	sample.copyVector(designParameters);
	sample(dimension) = trueValue;
	return sample;
}


vec Design::constructSampleObjectiveFunctionLowFi(void) const{

	vec sample(dimension+1);
	sample.copyVector(designParameters);
	sample(dimension) = trueValueLowFidelity;

	return sample;
}



vec Design::constructSampleObjectiveFunctionWithTangent(void) const{

	if (tangentDirection.getSize() != dimension) {
	    throw std::invalid_argument("Tangent direction size does not match the expected dimension.");
	}

	vec sample(2*dimension+2);

	sample.copyVector(designParameters);
	sample(dimension)   = trueValue;
	sample(dimension+1) = tangentValue;

	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+2+i) = tangentDirection(i);
	}

	return sample;
}

vec Design::constructSampleObjectiveFunctionWithTangentLowFi(void) const{

	if (tangentDirection.getSize() != dimension) {
	    throw std::invalid_argument("Tangent direction size does not match the expected dimension.");
	}

	vec sample(2*dimension+2);

	sample.copyVector(designParameters);
	sample(dimension)   = trueValueLowFidelity;
	sample(dimension+1) = tangentValueLowFidelity;

	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+2+i) = tangentDirection(i);
	}

	return sample;
}

vec Design::constructSampleObjectiveFunctionWithGradient(void) const{

	if (gradient.getSize() != dimension) {
	    throw std::invalid_argument("Gradient size does not match the expected dimension.");
	}

	vec sample(2*dimension+1);

	sample.copyVector(designParameters);
	sample(dimension) = trueValue;

	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+1+i) = gradient(i);
	}
	return sample;
}

vec Design::constructSampleObjectiveFunctionWithZeroGradient(void) const{
	if (gradient.getSize() != dimension) {
	    throw std::invalid_argument("Gradient size does not match the expected dimension.");
	}

	vec sample(2*dimension+1);

	sample.copyVector(designParameters);
	sample(dimension) = trueValue;

	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+1+i) = 0.0;
	}
	return sample;
}



vec Design::constructSampleObjectiveFunctionWithGradientLowFi(void) const{

	if (gradientLowFidelity.getSize() != dimension) {
	    throw std::invalid_argument("Size of gradientLowFidelity does not match the expected dimension.");
	}

	vec sample(2*dimension+1);

	sample.copyVector(designParameters);
	sample(dimension) = trueValueLowFidelity;

	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+1+i) = gradientLowFidelity(i);
	}
	return sample;
}

vec Design::constructSampleConstraint(unsigned int constraintID) const{

	if (constraintID >= numberOfConstraints) {
	    throw std::out_of_range("Constraint ID is out of the valid range.");
	}

	if (constraintTrueValues.getSize() != numberOfConstraints) {
	    throw std::invalid_argument("The size of constraintTrueValues does not match the expected number of constraints.");
	}


	vec sample(dimension+1);
	sample.copyVector(designParameters);
	sample(dimension) = constraintTrueValues(constraintID);

	return sample;
}

vec Design::constructSampleConstraintLowFi(unsigned int constraintID) const{
	if (constraintID >= numberOfConstraints) {
	    throw std::out_of_range("constraintID is out of the valid range.");
	}

	if (constraintTrueValuesLowFidelity.getSize() != numberOfConstraints) {
	    throw std::invalid_argument("Size of constraintTrueValuesLowFidelity does not match the expected number of constraints.");
	}

	vec sample(dimension+1);
	sample.copyVector(designParameters);

	//	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension) = constraintTrueValuesLowFidelity(constraintID);

	return sample;
}


vec Design::constructSampleConstraintWithTangent(unsigned int constraintID) const{

	if (constraintID >= numberOfConstraints) {
	    throw std::out_of_range("Constraint ID is out of the valid range.");
	}

	if (constraintTrueValues.getSize() != numberOfConstraints) {
	    throw std::invalid_argument("The size of constraintTrueValues does not match the expected number of constraints.");
	}

	if (constraintDifferentiationDirectionsMatrix.getNRows() != numberOfConstraints) {
	    throw std::invalid_argument("The number of rows in constraintDifferentiationDirectionsMatrix does not match the expected number of constraints.");
	}

	if (constraintTangent.getSize() != numberOfConstraints) {
	    throw std::invalid_argument("The size of constraintTangent does not match the expected number of constraints.");
	}


	vec sample(2*dimension+2);
	sample.copyVector(designParameters);

	//	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension) = constraintTrueValues(constraintID);
	sample(dimension+1) = constraintTangent(constraintID);
	vec direction = constraintDifferentiationDirectionsMatrix.getRow(ID);

	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+2+i) = direction(i);
	}

	return sample;
}

vec Design::constructSampleConstraintWithTangentLowFi(unsigned int constraintID) const{

	if (constraintID >= numberOfConstraints) {
	    throw std::out_of_range("Constraint ID is out of the valid range.");
	}

	if (constraintTrueValuesLowFidelity.getSize() != numberOfConstraints) {
	    throw std::invalid_argument("The size of constraintTrueValuesLowFidelity does not match the expected number of constraints.");
	}

	if (constraintDifferentiationDirectionsMatrixLowFi.getNRows() != numberOfConstraints) {
	    throw std::invalid_argument("The number of rows in constraintDifferentiationDirectionsMatrixLowFi does not match the expected number of constraints.");
	}

	if (constraintTangentLowFidelity.getSize() != numberOfConstraints) {
	    throw std::invalid_argument("The size of constraintTangentLowFidelity does not match the expected number of constraints.");
	}

	vec sample(2*dimension+2);
	sample.copyVector(designParameters);
	//	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension)   = constraintTrueValuesLowFidelity(constraintID);
	sample(dimension+1) = constraintTangentLowFidelity(constraintID);

	vec direction = constraintDifferentiationDirectionsMatrixLowFi.getRow(constraintID);

	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+2+i) = direction(i);
	}

	return sample;
}

vec Design::constructSampleConstraintWithGradient(unsigned int constraintID) const{

	if (constraintID >= numberOfConstraints) {
	    throw std::out_of_range("Constraint ID is out of the valid range.");
	}

	if (constraintTrueValues.getSize() != numberOfConstraints) {
	    throw std::invalid_argument("The size of constraintTrueValues does not match the expected number of constraints.");
	}


	vec sample(2*dimension+1);
	sample.copyVector(designParameters);
	//	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension) = constraintTrueValues(constraintID);
	vec constraintGradient = constraintGradientsMatrix.getRow(constraintID);
	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+1+i) = constraintGradient(i);
	}

	return sample;
}

vec Design::constructSampleConstraintWithGradientLowFi(unsigned int constraintID) const{

	if (constraintID >= numberOfConstraints) {
	    throw std::out_of_range("Constraint ID is out of the valid range.");
	}

	if (constraintTrueValuesLowFidelity.getSize() != numberOfConstraints) {
	    throw std::invalid_argument("The size of constraintTrueValuesLowFidelity does not match the expected number of constraints.");
	}

	vec sample(2*dimension+1);
	sample.copyVector(designParameters);
	//	copyVectorFirstKElements(sample,designParameters, dimension);
	sample(dimension) = constraintTrueValuesLowFidelity(constraintID);
	vec constraintGradient = constraintGradientsMatrixLowFi.getRow(constraintID);
	for(unsigned int i=0; i<dimension; i++){
		sample(dimension+1+i) = constraintGradient(i);
	}

	return sample;
}

bool Design::checkIfHasNan(void) const{

	bool ifHasNan = false;
	if(isnan(trueValue )) {

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


	cout<< "\n***************** " << tag << " *****************\n";
	cout<<"Design parameters = \n";
	designParameters.print();

	cout<<"Function value = "<<trueValue<<"\n";

	if(fabs(trueValueLowFidelity) > 0.0 ){
		cout<<"Function value Low Fidelity = "<<trueValueLowFidelity<<"\n";
	}

	if(!gradient.is_zero() && gradient.getSize() > 0){
		gradient.print("gradient vector");

	}

	if(!gradientLowFidelity.is_zero() && gradientLowFidelity.getSize() > 0){
		gradientLowFidelity.print("gradient vector (Low Fi)");
	}


	if(constraintTrueValues.getSize() > 0){
		constraintTrueValues.print("constraint values");
	}

	if(!constraintGradients.empty()){

		int count = 0;
		for (auto it = constraintGradients.begin(); it != constraintGradients.end(); it++){

			if(!it->is_zero()){
				cout<<"Constraint gradient "<<count<<"\n";
				it->print();
				count++;
			}
		}
	}

	if(isDesignFeasible){
		cout<<"Feasibility = " << "YES"<<"\n";
	}
	else{
		cout<<"Feasibility = " << "NO"<<"\n";
	}
	cout<<"Improvement = "<<improvementValue<<"\n";
	cout<< "*********************************************************\n\n";

}


std::string Design::toString() const {

    std::ostringstream oss;

    // Start with the formatted tag and design parameters
    oss << "\n***************** " << tag << " *****************\n";
    oss << "Design parameters = \n";
    for (unsigned int i = 0; i < designParameters.getSize(); i++) {
        oss << designParameters(i) << " ";
    }
    oss << "\n";

    // Add the function value
    oss << "Function value = " << trueValue << "\n";

    // Include low fidelity function value if applicable
    if (fabs(trueValueLowFidelity) > 0.0) {
        oss << "Function value Low Fidelity = " << trueValueLowFidelity << "\n";
    }

    // Include gradient if available and not zero
    if (!gradient.is_zero() && gradient.getSize() > 0) {
        oss << "Gradient vector = \n";
        for (unsigned int i = 0; i < gradient.getSize(); i++) {
            oss << gradient(i) << " ";
        }
        oss << "\n";
    }

    // Include low fidelity gradient if available and not zero
    if (!gradientLowFidelity.is_zero() && gradientLowFidelity.getSize() > 0) {
        oss << "Gradient vector (Low Fi) = \n";
        for (unsigned int i = 0; i < gradientLowFidelity.getSize(); i++) {
            oss << gradientLowFidelity(i) << " ";
        }
        oss << "\n";
    }

    // Add constraint values if present
    if (constraintTrueValues.getSize() > 0) {
        oss << "Constraint values = \n";
        for (unsigned int i = 0; i < constraintTrueValues.getSize(); i++) {
            oss << constraintTrueValues(i) << " ";
        }
        oss << "\n";
    }

    // Add constraint gradients if they exist and are non-zero
    if (!constraintGradients.empty()) {
        int count = 0;
        for (auto it = constraintGradients.begin(); it != constraintGradients.end(); it++) {
            if (!it->is_zero()) {
                oss << "Constraint gradient " << count << ":\n";
                for (unsigned int i = 0; i < it->getSize(); i++) {
                    oss << (*it)(i) << " ";
                }
                oss << "\n";
                count++;
            }
        }
    }

    // Check and add feasibility status
    if (isDesignFeasible) {
        oss << "Feasibility = YES\n";
    } else {
        oss << "Feasibility = NO\n";
    }

    // Add improvement value
    oss << "Improvement = " << improvementValue << "\n";
    oss << "*********************************************************\n\n";

    // Return the constructed string
    return oss.str();
}


std::string Design::generateFormattedString(std::string msg, char c, int totalLength) const{
    if (totalLength < 0) {
        throw std::invalid_argument("Number of characters must be non-negative.");
    }

    if(msg.length()%2 == 1){
    	msg+=" ";
    }

    int numEquals = static_cast<int>((totalLength - msg.length() - 2)/2);


    std::string border(numEquals, c);



    std::ostringstream oss;
    oss << border << " " << msg << " " << border;

    return oss.str();
}


string Design::generateOutputString(void) const{

	string msg = generateFormattedString(tag, '-', 100);
	msg += "\nDesign parameters = \n";

	for(unsigned int i=0; i<dimension; i++){

		msg+= to_string(designParameters(i)) + " ";
	}
	msg += "\n";

	msg += "Objective function value : " + to_string(trueValue) + "\n";


	if(constraintTrueValues.getSize() > 0){

		msg +="Constraint values = \n";



		for(unsigned int i=0; i<constraintTrueValues.getSize(); i++){

			msg +=to_string(constraintTrueValues(i)) + " ";
		}


		msg +="\n";
	}

	string border(100, '-');
	msg += border + "\n";

	return msg;

}




void Design::saveToAFile(string filename) const{

	if (filename.empty()) {
	    throw std::invalid_argument("Filename cannot be empty.");
	}


	ofstream fileOut;
	fileOut.open (filename);
	fileOut << tag<<"\n";
	fileOut << "Design parameters vector:\n";

	for(unsigned int i=0; i<designParameters.getSize(); i++){

		fileOut << designParameters(i);
	}



	fileOut << fixed << "Objective function = " << trueValue << "\n";

	if(numberOfConstraints>0){

		fileOut << "Constraint values vector:\n";

		for(unsigned int i=0; i<constraintTrueValues.getSize(); i++){

			fileOut << constraintTrueValues(i);
		}

	}

	if(isDesignFeasible){
		fileOut << "Feasibility = YES\n";
	}
	else{
		fileOut << "Feasibility = NO\n";
	}

	fileOut.close();

}

void Design::saveDesignVectorAsCSVFile(const std::string& fileName) const {
    if (fileName.empty()) {
        throw std::invalid_argument("ERROR: File name must not be empty.");
    }

    std::ofstream designVectorFile(fileName);
    designVectorFile.precision(10);

    if (designVectorFile.is_open()) {
        for (unsigned int i = 0; i < designParameters.getSize() - 1; ++i) {
            designVectorFile << designParameters(i) << ",";
        }

        designVectorFile << designParameters(designParameters.getSize() - 1);

        designVectorFile.close();
    } else {
        throw std::runtime_error("ERROR: Unable to open file: " + fileName);
    }
}


void Design::saveDesignVector(const std::string& fileName) const {
    if (fileName.empty()) {
        throw std::invalid_argument("ERROR: File name must not be empty.");
    }

    std::ofstream designVectorFile(fileName);
    designVectorFile.precision(10);

    if (designVectorFile.is_open()) {
        for (unsigned int i = 0; i < designParameters.getSize(); ++i) {
            designVectorFile << designParameters(i) << "\n";
        }
        designVectorFile.close();
    } else {
        throw std::runtime_error("ERROR: Unable to open file: " + fileName);
    }
}



} /* Namespace Rodop */
