#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cassert>


#include "auxiliary_functions.hpp"
#include "Rodeo_macros.hpp"
#include "Rodeo_globals.hpp"
#include "test_functions.hpp"
#include "optimization.hpp"
#include "objective_function.hpp"


#include "bounds.hpp"


#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>


using namespace arma;

ObjectiveFunctionDefinition::ObjectiveFunctionDefinition(void){}

bool ObjectiveFunctionDefinition::checkIfDefinitionIsOk(void) const{

	if(ifDefined == false) {

		//		std::cout<<"ifDefined is false!";
		return false;
	}

	if(name.empty()){

		std::cout<<"name is empty!";
		return false;
	}


	if(designVectorFilename.empty()){
		std::cout<<"designVectorFilename is empty!";
		return false;

	}

	if(outputFilename.empty()){

		std::cout<<"outputFilename is empty!";
		return false;
	}
	if(nameHighFidelityTrainingData.empty()){
		std::cout<<"nameHighFidelityTrainingData is empty!";
		return false;
	}

	if(ifMultiLevel){

		if(nameLowFidelityTrainingData.empty() ||
				outputFilenameLowFi.empty()){
			return false;
		}

		if(nameLowFidelityTrainingData == nameHighFidelityTrainingData){

			return false;
		}

	}
	return true;
}

string ObjectiveFunctionDefinition::getNameOfSurrogateModel(SURROGATE_MODEL modelType) const {

	string modelName;
	if (modelType == ORDINARY_KRIGING)
		modelName = "Uses only functional values";

	if (modelType == GRADIENT_ENHANCED)
		modelName = "Uses gradients";

	if (modelType == TANGENT_ENHANCED)
		modelName = "Uses directional derivatives";

	return modelName;
}

void ObjectiveFunctionDefinition::printHighFidelityModel() const {

	string modelName = getNameOfSurrogateModel(modelHiFi);
	std::cout << "Surrogate model = " << modelName << "\n";

}

void ObjectiveFunctionDefinition::printLowFidelityModel() const {
	string modelNameLowFi = getNameOfSurrogateModel(modelLowFi);
	std::cout << "\tSurrogate model = " << modelNameLowFi << "\n";
}

void ObjectiveFunctionDefinition::print(void) const{


	std::cout<< "Name = "<<name<<"\n";


	if(!doesUseUDF){

		std::cout<< "Design vector filename = "<<designVectorFilename<<"\n";
		std::cout<< "Training data = " << nameHighFidelityTrainingData << "\n";
		std::cout<< "Output filename = " << outputFilename << "\n";
		std::cout<< "Executable = " << executableName << "\n";

	}
	else{

		std::cout<< "UDF = YES\n";
	}



	if(isNotEmpty(executableNameGradient)){

		std::cout<< "Executable for gradient = " << executableNameGradient << "\n";
		std::cout<< "Output file name for gradient = " << outputGradientFilename << "\n";
	}

	printHighFidelityModel();

	string ifMultiFidelity;
	if(ifMultiLevel){

		ifMultiFidelity = "YES";
	}
	else{

		ifMultiFidelity = "NO";
	}


	std::cout<< "Multilevel = "<< ifMultiFidelity <<"\n";

	if(ifMultiLevel){

		std::cout<< "Low fidelity model = " << "\n";
		std::cout<< "\tTraining data = " << nameLowFidelityTrainingData << "\n";
		std::cout<< "\tOutput filename = " << outputFilenameLowFi << "\n";

		if(!outputFilenameLowFiGradient.empty()){

			std::cout<< "\tOutput filename for gradient = " << outputFilenameLowFiGradient << "\n";
		}

		std::cout<< "\tExecutable = " << executableNameLowFi << "\n";

		if(!executableNameLowFiGradient.empty()){

			std::cout<< "\tExecutable for gradient = " << executableNameLowFiGradient << "\n";
		}

		printLowFidelityModel();




	}


}



/**********************************************************************************************/


ObjectiveFunction::ObjectiveFunction(){}

void ObjectiveFunction::setEvaluationMode(std::string mode){
	assert(isNotEmpty(mode));
	evaluationMode = mode;
}


void ObjectiveFunction::setDataAddMode(std::string mode){

	assert(isNotEmpty(mode));
	addDataMode = mode;
}

void ObjectiveFunction::setDimension(unsigned int dimension){

	dim = dimension;
}


bool ObjectiveFunction::isMultiFidelityActive(void) const{
	return definition.ifMultiLevel;
}


SURROGATE_MODEL ObjectiveFunction::getSurrogateModelType(void) const{
	return definition.modelHiFi;
}
SURROGATE_MODEL ObjectiveFunction::getSurrogateModelTypeLowFi(void) const{
	return definition.modelLowFi;
}

void ObjectiveFunction::bindWithOrdinaryKrigingModel() {
	output.printMessage(
			"Binding the surrogate model with the ORDINARY_KRIGING modeĺ...");
	surrogate = &surrogateModel;
}

void ObjectiveFunction::bindWithUniversalKrigingModel() {
	output.printMessage(
			"Binding the surrogate model with the UNIVERSAL_KRIGING modeĺ...");
	surrogateModel.setLinearRegressionOn();
	surrogate = &surrogateModel;
}

void ObjectiveFunction::bindWithGradientEnhancedModel() {
	output.printMessage(
			"Binding the surrogate model with the GRADIENT_ENHANCED modeĺ...");
	surrogateModelGradient.setAdjustThetaFactorOff();
	surrogate = &surrogateModelGradient;

}

void ObjectiveFunction::bindWithTangentEnhancedModel() {

	output.printMessage(
			"Binding the surrogate model with the TANGENT_ENHANCED modeĺ...");
	surrogateModelGradient.setDirectionalDerivativesOn();
	surrogate = &surrogateModelGradient;
}

void ObjectiveFunction::bindWithMultiFidelityModel() {

	surrogateModelML.setIDHiFiModel(definition.modelHiFi);
	surrogateModelML.setIDLowFiModel(definition.modelLowFi);

	surrogateModelML.setinputFileNameHighFidelityData(definition.nameHighFidelityTrainingData);
	surrogateModelML.setinputFileNameLowFidelityData(definition.nameLowFidelityTrainingData);


	/* TODO modify this ugly code */
	surrogateModelML.bindModels();

	output.printMessage("Binding the surrogate model with the Multi-fidelity model...");

	surrogate = &surrogateModelML;

}

void ObjectiveFunction::bindSurrogateModelSingleFidelity() {
	if (definition.modelHiFi == ORDINARY_KRIGING) {
		bindWithOrdinaryKrigingModel();
	}
	if (definition.modelHiFi == UNIVERSAL_KRIGING) {
		bindWithUniversalKrigingModel();
	}
	if (definition.modelHiFi == GRADIENT_ENHANCED) {
		bindWithGradientEnhancedModel();
	}
	if (definition.modelHiFi == TANGENT_ENHANCED) {
		bindWithTangentEnhancedModel();
	}
}

void ObjectiveFunction::bindSurrogateModel(void){

	assert(ifDefinitionIsSet);


	if(definition.ifMultiLevel){
		bindWithMultiFidelityModel();
	}
	else{
		bindSurrogateModelSingleFidelity();
	}

	ifSurrogateModelIsDefined = true;

}


void ObjectiveFunction::setParametersByDefinition(ObjectiveFunctionDefinition def){

	definition = def;
	ifDefinitionIsSet = true;

}


void ObjectiveFunction::setDisplayOn(void){
	output.ifScreenDisplay = true;
}
void ObjectiveFunction::setDisplayOff(void){
	output.ifScreenDisplay = false;
}
void ObjectiveFunction::setNumberOfTrainingIterationsForSurrogateModel(unsigned int nIter){
	numberOfIterationsForSurrogateTraining = nIter;
}
void ObjectiveFunction::setFileNameReadInput(std::string fileName){

	assert(!fileName.empty());
	definition.outputFilename = fileName;
}
void ObjectiveFunction::setFileNameReadInputLowFidelity(std::string fileName){
	assert(!fileName.empty());
	definition.outputFilenameLowFi = fileName;
}

void ObjectiveFunction::setFileNameDesignVector(std::string fileName){
	assert(!fileName.empty());
	definition.designVectorFilename = fileName;
}

std::string ObjectiveFunction::getFileNameDesignVector(void) const{
	return definition.designVectorFilename;
}


void ObjectiveFunction::setFileNameTrainingData(std::string fileName){

	assert(!fileName.empty());
	definition.nameHighFidelityTrainingData = fileName;
}



std::string ObjectiveFunction::getFileNameTrainingData(void) const{
	return definition.nameHighFidelityTrainingData;
}

mat ObjectiveFunction::getTrainingData(void) const{

	assert(ifSurrogateModelIsDefined);
	return surrogate->getRawData();

}

std::string ObjectiveFunction::getName(void) const{
	assert(ifDefinitionIsSet);
	return definition.name;
}

void ObjectiveFunction::setExecutablePath(std::string path){

	assert(!path.empty());
	definition.path = path;
}
void ObjectiveFunction::setExecutableName(std::string exeName){

	assert(!exeName.empty());
	definition.executableName = exeName;

}


void ObjectiveFunction::setParameterBounds(Bounds bounds){

	if(dim > 0 ){
		assert(dim == bounds.getDimension());
	}


	assert(bounds.areBoundsSet());
	boxConstraints = bounds;

	ifParameterBoundsAreSet = true;
}

MultiLevelModel ObjectiveFunction::getSurrogateModelML(void) const{
	return surrogateModelML;
}



void ObjectiveFunction::initializeSurrogate(void){

	assert(ifParameterBoundsAreSet);
	assert(ifDefinitionIsSet);
	assert(dim>0);

	bindSurrogateModel();


	surrogate->setName(definition.name);
	surrogate->setDimension(dim);
	surrogate->setBoxConstraints(boxConstraints);
	surrogate->setNameOfInputFile(definition.nameHighFidelityTrainingData);


	surrogate->readData();
	surrogate->normalizeData();

	surrogate->initializeSurrogateModel();
	surrogate->setNumberOfTrainingIterations(numberOfIterationsForSurrogateTraining);

#if 0
	surrogate->printSurrogateModel();
#endif



	ifInitialized = true;
}

void ObjectiveFunction::setFeasibleMinimum(double value){

	sampleMinimum = value;

}


void ObjectiveFunction::trainSurrogate(void){


	assert(ifInitialized);


	if(ifWarmStart){
		surrogate->setReadWarmStartFileFlag(true);
	}

	surrogate->train();
}





void ObjectiveFunction::calculateExpectedImprovement(DesignForBayesianOptimization &designCalculated) const{

	double ftilde, ssqr;

	surrogate->interpolateWithVariance(designCalculated.dv, &ftilde, &ssqr);

	double sigma = sqrt(ssqr);

	designCalculated.sigma = sigma;

	sigma = sigmaFactor*sigma;

	/* larger sigma means model uncertainty is higher. In this case, more exploration will take place */

#if 0
	printf("standart_ERROR = %15.10f\n",sigma);
#endif

	double expectedImprovementValue = 0.0;

	if(fabs(sigma) > EPSILON){

		double improvement = 0.0;
		improvement = sampleMinimum   - ftilde;

		double	Z = (improvement)/sigma;
#if 0
		printf("Z = %15.10f\n",Z);
		printf("ymin = %15.10f\n",yMin);
#endif

		expectedImprovementValue = improvement*cdf(Z,0.0,1.0)+   sigma * pdf(Z,0.0,1.0);


	}
	else{

		expectedImprovementValue = 0.0;

	}


	designCalculated.valueAcqusitionFunction = expectedImprovementValue;
	designCalculated.objectiveFunctionValue = ftilde;

}


void ObjectiveFunction::calculateProbabilityOfImprovement(DesignForBayesianOptimization &designCalculated) const{

	double ftilde, ssqr;

	surrogate->interpolateWithVariance(designCalculated.dv, &ftilde, &ssqr);

	double	sigma = sqrt(ssqr)	;
	designCalculated.sigma = sigma;

	double PI = designCalculated.calculateProbalityThatTheEstimateIsLessThanAValue(sampleMinimum);

	designCalculated.valueAcqusitionFunction = PI;

	designCalculated.objectiveFunctionValue = ftilde;

}

void ObjectiveFunction::calculateSurrogateEstimate(DesignForBayesianOptimization &designCalculated) const{

	double ftilde;
	ftilde = surrogate->interpolate(designCalculated.dv);
	designCalculated.objectiveFunctionValue = ftilde;
}

void ObjectiveFunction::calculateSurrogateEstimateUsingDerivatives(DesignForBayesianOptimization &designCalculated) const{

	double ftilde;
	ftilde = surrogate->interpolateUsingDerivatives(designCalculated.dv);
	designCalculated.objectiveFunctionValue = ftilde;
}


std::string ObjectiveFunction::getExecutionCommand(const std::string& exename) const {
	std::string command;

	// Check if the input ends with ".py", indicating a Python script
	if (exename.size() >= 3 && exename.substr(exename.size() - 3) == ".py") {
		command = "python " + exename;  // Prepends 'python ' to the command
	} else {
		// Check if the exename already contains a path (i.e., contains '/')
		if (exename.find('/') != std::string::npos) {
			command = exename;  // Use the full path as the command
		} else {
			command = "./" + exename;  // Prepend './' for relative path executables
		}
	}

	return command;
}

void ObjectiveFunction::addDesignToData(Design &d){

	assert((isNotEmpty(definition.nameHighFidelityTrainingData)));
	assert(ifInitialized);
	assert(isNotEmpty(addDataMode));


	if(definition.ifMultiLevel == false){

		rowvec newsample;

		if(addDataMode.compare("primal") == 0 ){
			newsample = d.constructSampleObjectiveFunction();
		}
		if(addDataMode.compare("tangent") == 0 ){
			newsample = d.constructSampleObjectiveFunctionWithTangent();
		}
		if(addDataMode.compare("adjoint") == 0 ){
			newsample = d.constructSampleObjectiveFunctionWithGradient();
		}
		if(addDataMode.compare("adjointWithZeroGradient") == 0 ){
			newsample = d.constructSampleObjectiveFunctionWithZeroGradient();
		}


		assert(newsample.size()>0);
		surrogate->addNewSampleToData(newsample);

	}
	else{

		rowvec newsampleHiFi;
		rowvec newsampleLowFi;

		if(addDataMode.compare("primalBoth") == 0 ){

			newsampleHiFi = d.constructSampleObjectiveFunction();
			newsampleLowFi = d.constructSampleObjectiveFunctionLowFi();

			assert(newsampleLowFi.size() >0);
			assert(newsampleHiFi.size()  >0);

			surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
			surrogate->addNewSampleToData(newsampleHiFi);

		}

		if(addDataMode.compare("primalHiFiAdjointLowFi") == 0 ){

			newsampleHiFi  = d.constructSampleObjectiveFunction();
			newsampleLowFi = d.constructSampleObjectiveFunctionWithGradientLowFi();

			assert(newsampleLowFi.size() >0);
			assert(newsampleHiFi.size()  >0);

			surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
			surrogate->addNewSampleToData(newsampleHiFi);

		}




	}

}


void ObjectiveFunction::addDesignToData(Design &d, string how){

	assert((isNotEmpty(definition.nameHighFidelityTrainingData)));
	assert(ifInitialized);
	assert(isNotEmpty(how));


	if(definition.ifMultiLevel == false){

		rowvec newsample;

		if(how.compare("primal") == 0 ){
			newsample = d.constructSampleObjectiveFunction();
		}
		if(how.compare("tangent") == 0 ){
			newsample = d.constructSampleObjectiveFunctionWithTangent();
		}
		if(how.compare("adjoint") == 0 ){
			newsample = d.constructSampleObjectiveFunctionWithGradient();
		}
		if(how.compare("adjointWithZeroGradient") == 0 ){
			newsample = d.constructSampleObjectiveFunctionWithZeroGradient();
		}


		assert(newsample.size()>0);
		surrogate->addNewSampleToData(newsample);

	}
	else{

		rowvec newsampleHiFi;
		rowvec newsampleLowFi;

		if(how.compare("primalBoth") == 0 ){

			newsampleHiFi = d.constructSampleObjectiveFunction();
			newsampleLowFi = d.constructSampleObjectiveFunctionLowFi();

			assert(newsampleLowFi.size() >0);
			assert(newsampleHiFi.size()  >0);

			surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
			surrogate->addNewSampleToData(newsampleHiFi);

		}

		if(how.compare("primalHiFiAdjointLowFi") == 0 ){

			newsampleHiFi  = d.constructSampleObjectiveFunction();
			newsampleLowFi = d.constructSampleObjectiveFunctionWithGradientLowFi();

			assert(newsampleLowFi.size() >0);
			assert(newsampleHiFi.size()  >0);

			surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
			surrogate->addNewSampleToData(newsampleHiFi);

		}




	}

}



void ObjectiveFunction::addLowFidelityDesignToData(Design &d){

	assert((isNotEmpty(definition.nameLowFidelityTrainingData)));
	assert(ifInitialized);
	assert(isNotEmpty(addDataMode));
	assert(definition.ifMultiLevel == true);

	rowvec newsampleLowFi;

	if(addDataMode.compare("primalLowFidelity") == 0 ){
		newsampleLowFi = d.constructSampleObjectiveFunctionLowFi();
		assert(newsampleLowFi.size()>0);
		surrogate->addNewLowFidelitySampleToData(newsampleLowFi);

	}

	if(addDataMode.compare("adjointLowFidelity") == 0 ){

		newsampleLowFi = d.constructSampleObjectiveFunctionWithGradientLowFi();
		assert(newsampleLowFi.size()>0);
		surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
	}

	if(addDataMode.compare("tangentLowFidelity") == 0 ){

		newsampleLowFi = d.constructSampleObjectiveFunctionWithTangentLowFi();
		assert(newsampleLowFi.size()>0);
		surrogate->addNewLowFidelitySampleToData(newsampleLowFi);
	}





}


rowvec ObjectiveFunction::readOutput(string filename, unsigned int howMany) const{

	assert(isNotEmpty(filename));

	rowvec result(howMany,fill::zeros);

	std::ifstream inputFileStream(filename);

	if (!inputFileStream.is_open()) {
		string msg = "There was a problem opening the output file : " + filename;
		abortWithErrorMessage(msg);
	}

	for(unsigned int i=0; i<howMany;i++){
		inputFileStream >> result(i);
	}

	inputFileStream.close();

	return result;
}


void ObjectiveFunction::writeDesignVariablesToFile(Design &d) const{

	assert(d.designParameters.size() == dim);
	assert(isNotEmpty(definition.designVectorFilename));

	std::ofstream outputFileStream(definition.designVectorFilename);

	if (!outputFileStream.is_open()) {
		abortWithErrorMessage("There was a problem opening the output file!\n");
	}

	for(unsigned int i=0; i<dim; i++) {

		outputFileStream << d.designParameters(i) << std::endl;
	}

	if(evaluationMode.compare("tangent") == 0 || evaluationMode.compare("tangentLowFi") == 0){

		assert(d.tangentDirection.size() == dim);
		for(unsigned int i=0; i<dim; i++) {

			outputFileStream << d.tangentDirection(i) << std::endl;
		}

	}
	outputFileStream.close();

}

void ObjectiveFunction::evaluateDesign(Design &d){

	assert(d.designParameters.size() == dim);

	setEvaluationMode("primal");
	if(!doesObjectiveFunctionPtrExist){
		writeDesignVariablesToFile(d);
		evaluateObjectiveFunction();
		rowvec result = readOutput(definition.outputFilename, 1);
		d.trueValue = result(0);

	}
	else{
		double functionValue = evaluateObjectiveFunctionDirectly(d.designParameters);
		d.trueValue = functionValue;
	}





}


void ObjectiveFunction::evaluateDesignGradient(Design &d){

	assert(dim>0);
	assert(d.designParameters.size() == dim);

	setEvaluationMode("adjoint");

	writeDesignVariablesToFile(d);
	evaluateGradient();

	rowvec result = readOutput(definition.outputGradientFilename, dim);
	d.gradient = result;

}





bool ObjectiveFunction::isHiFiEvaluation(void) const{

	if(evaluationMode.compare("primal") == 0 ) return true;
	if(evaluationMode.compare("adjoint") == 0 ) return true;
	if(evaluationMode.compare("tangent") == 0 ) return true;
	return false;
}

bool ObjectiveFunction::isLowFiEvaluation(void) const{

	if(evaluationMode.compare("primalLowFi") == 0 ) return true;
	if(evaluationMode.compare("adjointLowFi") == 0 ) return true;
	if(evaluationMode.compare("tangentLowFi") == 0 ) return true;
	return false;
}



void ObjectiveFunction::evaluateGradient(void) const{



	std::string runCommand;
	runCommand.clear();

	if(isHiFiEvaluation()){

		assert(isNotEmpty(definition.executableNameGradient));
		runCommand = getExecutionCommand(definition.executableNameGradient);


	}
	if(isLowFiEvaluation()){

		assert(isNotEmpty(definition.executableNameLowFiGradient));
		runCommand = getExecutionCommand(definition.executableNameLowFi);

	}

	if(isNotEmpty(runCommand)){

		output.printMessage("Calling executable for the objective function:", definition.name);

		int systemReturn = system(runCommand.c_str()) ;

		if(systemReturn == -1){

			string msg = "A process for the objective function/constraint execution could not be created, or its status could not be retrieved";

			abortWithErrorMessage(msg);

		}

		else{


			printWaitStatusIfSystemCallFails(systemReturn);
		}

	}

}

double ObjectiveFunction::evaluateObjectiveFunctionDirectly(const rowvec &x){

	assert(objectiveFunctionPtr!=nullptr);
	return objectiveFunctionPtr(x.memptr());

}



void ObjectiveFunction::evaluateObjectiveFunction(void){

	assert(isNotEmpty(definition.designVectorFilename));

	std::string runCommand;
	runCommand.clear();

	if(isHiFiEvaluation()){


		if(isNotEmpty(definition.executableName)){
			runCommand = getExecutionCommand(definition.executableName);
		}


	}
	if(isLowFiEvaluation()){

		if(isNotEmpty(definition.executableNameLowFi)){
			runCommand = getExecutionCommand(definition.executableNameLowFi);
		}

	}

	if(isNotEmpty(runCommand)){

		output.printMessage("Calling executable for the objective function:", definition.name);

		int systemReturn = system(runCommand.c_str()) ;

		if(systemReturn == -1){

			string msg = "A process for the objective function/constraint execution could not be created, or its status could not be retrieved";

			abortWithErrorMessage(msg);

		}

		else{


			printWaitStatusIfSystemCallFails(systemReturn);
		}

	}

}

void ObjectiveFunction::printWaitStatusIfSystemCallFails(int status) const{

	if (WIFEXITED(status)) {

		int statusCode = WEXITSTATUS(status);
		if (statusCode == 0) {

			output.printMessage("Objective function/constraint execution is done!");
		} else {
			string msg  = "There has been some problem with the objective function/constraint execution!";
			abortWithErrorMessage(msg);

		}
	}
}



double ObjectiveFunction::interpolate(rowvec x) const{
	return surrogate->interpolate(x);
}

double ObjectiveFunction::interpolateUsingDerivatives(rowvec x) const{
	return surrogate->interpolateUsingDerivatives(x);
}

pair<double, double> ObjectiveFunction::interpolateWithVariance(rowvec x) const{

	double ftilde,sigmaSqr;
	surrogate->interpolateWithVariance(x, &ftilde, &sigmaSqr);

	pair<double, double> result;
	result.first = ftilde;
	result.second = sqrt(sigmaSqr);

	return result;
}


void ObjectiveFunction::print(void) const{
	std::cout<<"\n================ Objective function definition =========================\n";
	definition.print();
	std::cout<< "Number of training iterations for model training = " << numberOfIterationsForSurrogateTraining << "\n";
	std::cout<<"==========================================================================\n";
}


string ObjectiveFunction::generateOutputString(void) const{


	std::string outputMsg;
	string tag = "Objective function definition";
	outputMsg = generateFormattedString(tag,'=', 100) + "\n";
	outputMsg+= "Name : " + definition.name + "\n";

	if(!definition.executableName.empty()){
		outputMsg+= "Executable : " + definition.executableName + "\n";
	}

	if(doesObjectiveFunctionPtrExist){

		outputMsg+= "API Call : YES\n";
	}


	outputMsg+= "Training data : " + definition.nameHighFidelityTrainingData + "\n";

	if(!definition.outputFilename.empty()){

		outputMsg+= "Output file : " + definition.outputFilename + "\n";
	}
	if(!definition.designVectorFilename.empty()){

		outputMsg+= "Design parameters file : " + definition.designVectorFilename + "\n";
	}

	string modelName = definition.getNameOfSurrogateModel(definition.modelHiFi);
	outputMsg+= "Surrogate model : " + modelName + "\n";

	outputMsg+= "Number of iterations for model training : " +std::to_string(numberOfIterationsForSurrogateTraining) + "\n";


	std::string border(100, '=');
	outputMsg += border + "\n";


	return outputMsg;

}

void ObjectiveFunction::printSurrogate(void) const{
	surrogate->printSurrogateModel();
}


void ObjectiveFunction::removeVeryCloseSamples(const Design& globalOptimalDesign){


	surrogate->removeVeryCloseSamples(globalOptimalDesign);
	surrogate->updateModelWithNewData();

}

void ObjectiveFunction::removeVeryCloseSamples(const Design& globalOptimalDesign, std::vector<rowvec> samples){


	surrogate->removeVeryCloseSamples(globalOptimalDesign,samples);
	surrogate->updateModelWithNewData();

}


void ObjectiveFunction::setSigmaFactor(double factor){

	assert(factor>0.0);
	sigmaFactor = factor;

}

void ObjectiveFunction::setGlobalOptimalDesign(Design d){
	assert(ifSurrogateModelIsDefined);
	surrogate->setGlobalOptimalDesign(d);

}

void ObjectiveFunction::setFunctionPtr(ObjectiveFunctionPtr func) {
	if (func == nullptr) {
		throw std::invalid_argument("Function pointer is null");
	}

	objectiveFunctionPtr = func;
	doesObjectiveFunctionPtrExist = true;
}



