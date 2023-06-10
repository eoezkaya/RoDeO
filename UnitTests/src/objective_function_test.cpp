/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), RPTU
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
 * General Public License along with CoDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */

#include<gtest/gtest.h>
#include "objective_function.hpp"
#include "optimization.hpp"
#include "LinearAlgebra/INCLUDE/vector_operations.hpp"
#include "standard_test_functions.hpp"
#include "auxiliary_functions.hpp"
#include "test_defines.hpp"


#ifdef OBJECTIVE_FUNCTION_TEST


class ObjectiveFunctionTest : public ::testing::Test {
protected:
	void SetUp() override {

		filenameTrainingData = himmelblauFunction.function.filenameTrainingData;
		filenameTrainingDataLowFi = himmelblauFunction.function.filenameTrainingDataLowFidelity;
		filenameTrainingDataHiFi  = himmelblauFunction.function.filenameTrainingDataHighFidelity;

		objFunTest.setDimension(2);
		vec lb(2); lb.fill(-6.0);
		vec ub(2); ub.fill(6.0);

		Bounds boxConstraints(lb,ub);
		objFunTest.setParameterBounds(boxConstraints);

		definition.designVectorFilename = "dv.dat";
		definition.executableName = "himmelblau";
		definition.executableNameLowFi = "himmelblauLowFi";
		definition.outputFilename = "objFunVal.dat";
		definition.outputFilenameLowFi = "objFunVal.dat";
		definition.name= "himmelblau";
		definition.nameHighFidelityTrainingData = filenameTrainingData;
		definition.nameLowFidelityTrainingData = filenameTrainingDataLowFi;

	}

	void TearDown() override {



	}

	ObjectiveFunction objFunTest;
	HimmelblauFunction himmelblauFunction;
	ObjectiveFunctionDefinition definition;
	mat trainingData;
	mat trainingDataLowFi;

	std::string filenameTrainingData;
	std::string filenameTrainingDataLowFi;
	std::string filenameTrainingDataHiFi;



	void setDefinitionForCase1(void){}

	void setDefinitionForCase2(void){

		definition.modelHiFi = GRADIENT_ENHANCED;
	}
	void setDefinitionForCase3(void){

		definition.modelHiFi = TANGENT;
	}

	void setDefinitionForCase4(void){
		/* In this case both models are Kriging */
		definition.ifMultiLevel = true;
		definition.executableNameLowFi = "himmelblauLowFi";
		definition.outputFilenameLowFi = "objFunValLowFi.dat";
		definition.nameLowFidelityTrainingData = filenameTrainingDataLowFi;
	}

	void setDefinitionForCase5(void){
		/* In this case HiFi model is Kriging and LowFi model is GGEK*/

		definition.ifMultiLevel = true;
		definition.modelHiFi  = ORDINARY_KRIGING;
		definition.modelLowFi = GRADIENT_ENHANCED;
		definition.executableNameLowFi = "himmelblauLowFi";
		definition.outputFilenameLowFi = "objFun.dat";
		definition.nameLowFidelityTrainingData = filenameTrainingDataLowFi;
	}



};



TEST_F(ObjectiveFunctionTest, addLowFiDesignToDataGGEKModel){

	himmelblauFunction.function.numberOfTrainingSamplesLowFi = 50;
	himmelblauFunction.function.numberOfTrainingSamples = 20;

	himmelblauFunction.function.generateTrainingSamplesMultiFidelityWithLowFiAdjoint();
	trainingDataLowFi = himmelblauFunction.function.trainingSamplesLowFidelity;

	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;
	d.trueValueLowFidelity = 2.67;
	rowvec gradientLowFi(2);
	gradientLowFi(0) = -18.9;
	gradientLowFi(1) = -22.4;
	d.gradientLowFidelity = gradientLowFi;


	setDefinitionForCase5();
	objFunTest.setParametersByDefinition(definition);
	objFunTest.setDataAddMode("adjointLowFidelity");

//	objFunTest.setDisplayOn();
	objFunTest.initializeSurrogate();
	objFunTest.addLowFidelityDesignToData(d);

	mat newData;
	newData.load(filenameTrainingDataLowFi, csv_ascii);

//	newData.print("newData");

	ASSERT_TRUE(newData.n_rows == trainingDataLowFi.n_rows+1);

	rowvec lastRow = newData.row(newData.n_rows-1);
	ASSERT_EQ(lastRow(0),2.1);
	ASSERT_EQ(lastRow(1),-1.9);
	ASSERT_EQ(lastRow(2),2.67);
	ASSERT_EQ(lastRow(3),-18.9);
	ASSERT_EQ(lastRow(4),-22.4);

	remove(filenameTrainingData.c_str());
	remove(filenameTrainingDataLowFi.c_str());

}


TEST_F(ObjectiveFunctionTest, addLowFiDesignToDataGGEKModelOnlyPrimalSolution){

	himmelblauFunction.function.numberOfTrainingSamplesLowFi = 50;
	himmelblauFunction.function.numberOfTrainingSamples = 20;

	himmelblauFunction.function.generateTrainingSamplesMultiFidelityWithLowFiAdjoint();
	trainingDataLowFi = himmelblauFunction.function.trainingSamplesLowFidelity;

	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;
	d.trueValueLowFidelity = 2.67;


	setDefinitionForCase5();
	objFunTest.setParametersByDefinition(definition);
	objFunTest.setDataAddMode("primalLowFidelity");

//	objFunTest.setDisplayOn();
	objFunTest.initializeSurrogate();
	objFunTest.addLowFidelityDesignToData(d);

	mat newData;
	newData.load(filenameTrainingDataLowFi, csv_ascii);

//	newData.print("newData");

	ASSERT_TRUE(newData.n_rows == trainingDataLowFi.n_rows+1);

	rowvec lastRow = newData.row(newData.n_rows-1);
	ASSERT_EQ(lastRow(0),2.1);
	ASSERT_EQ(lastRow(1),-1.9);
	ASSERT_EQ(lastRow(2),2.67);
	ASSERT_EQ(lastRow(3),0.0);
	ASSERT_EQ(lastRow(4),0.0);


	remove(filenameTrainingData.c_str());
	remove(filenameTrainingDataLowFi.c_str());

}






TEST_F(ObjectiveFunctionTest, testConstructor) {


	ASSERT_FALSE(objFunTest.ifInitialized);
	ASSERT_FALSE(objFunTest.ifSurrogateModelIsDefined);
	ASSERT_TRUE(objFunTest.getDimension() == 2);
	ASSERT_TRUE(objFunTest.ifParameterBoundsAreSet);

}


TEST_F(ObjectiveFunctionTest, testObjectiveFunctionDefinition) {

	setDefinitionForCase1();
	bool ifOk = definition.checkIfDefinitionIsOk();

	ASSERT_TRUE(ifOk);

}


TEST_F(ObjectiveFunctionTest, getSurrogateModelType) {
	setDefinitionForCase1();
	objFunTest.setParametersByDefinition(definition);
	SURROGATE_MODEL type = objFunTest.getSurrogateModelType();

	ASSERT_TRUE(type == ORDINARY_KRIGING);

}


TEST_F(ObjectiveFunctionTest, bindSurrogateModelCase1) {

	setDefinitionForCase1();
	objFunTest.setParametersByDefinition(definition);
	//	objFunTest.setDisplayOn();
	objFunTest.bindSurrogateModel();
	ASSERT_TRUE(objFunTest.ifSurrogateModelIsDefined);
}

TEST_F(ObjectiveFunctionTest, bindSurrogateModelCase2) {

	setDefinitionForCase2();
	objFunTest.setParametersByDefinition(definition);
	//	objFunTest.setDisplayOn();
	objFunTest.bindSurrogateModel();
	ASSERT_TRUE(objFunTest.ifSurrogateModelIsDefined);
}

TEST_F(ObjectiveFunctionTest, bindSurrogateModelCase3) {

	setDefinitionForCase3();
	objFunTest.setParametersByDefinition(definition);
	//	objFunTest.setDisplayOn();
	objFunTest.bindSurrogateModel();
	ASSERT_TRUE(objFunTest.ifSurrogateModelIsDefined);
}

TEST_F(ObjectiveFunctionTest, initializeSurrogateCase1) {

	himmelblauFunction.function.generateTrainingSamples();
	trainingData = himmelblauFunction.function.trainingSamples;

	setDefinitionForCase1();
	objFunTest.setParametersByDefinition(definition);
	objFunTest.initializeSurrogate();
	KrigingModel testModel = objFunTest.getSurrogateModel();

	mat rawData = testModel.getRawData();

	bool ifDataIsConsistent = isEqual(rawData, trainingData, 10E-8);
	ASSERT_TRUE(ifDataIsConsistent);
	/* check dimension */
	ASSERT_EQ(testModel.getDimension(), 2);
	ASSERT_FALSE(testModel.areGradientsOn());
	ASSERT_TRUE(testModel.ifDataIsRead);

	remove(filenameTrainingData.c_str());
}


TEST_F(ObjectiveFunctionTest, initializeSurrogateCase2) {

	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();
	trainingData = himmelblauFunction.function.trainingSamples;

	setDefinitionForCase2();
	objFunTest.setParametersByDefinition(definition);
	objFunTest.initializeSurrogate();
	GGEKModel testModel = objFunTest.getSurrogateModelGradient();

	mat rawData = testModel.getRawData();

	bool ifDataIsConsistent = isEqual(rawData, trainingData, 10E-8);
	ASSERT_TRUE(ifDataIsConsistent);
	/* check dimension */
	ASSERT_EQ(testModel.getDimension(), 2);
	ASSERT_TRUE(testModel.ifDataIsRead);

	remove(filenameTrainingData.c_str());
}

TEST_F(ObjectiveFunctionTest, initializeSurrogateCase3) {

	himmelblauFunction.function.generateTrainingSamplesWithTangents();
	trainingData = himmelblauFunction.function.trainingSamples;

	setDefinitionForCase3();
	objFunTest.setParametersByDefinition(definition);

	objFunTest.initializeSurrogate();
	TGEKModel testModel = objFunTest.getSurrogateModelTangent();

	mat rawData = testModel.getRawData();

	bool ifDataIsConsistent = isEqual(rawData, trainingData, 10E-8);
	ASSERT_TRUE(ifDataIsConsistent);
	/* check dimension */
	ASSERT_EQ(testModel.getDimension(), 2);
	ASSERT_FALSE(testModel.areGradientsOn());
	ASSERT_TRUE(testModel.ifDataIsRead);

	remove(filenameTrainingData.c_str());
}

TEST_F(ObjectiveFunctionTest, initializeSurrogateCase4) {

	himmelblauFunction.function.numberOfTrainingSamplesLowFi = 100;
	himmelblauFunction.function.numberOfTrainingSamples = 50;
	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();
	trainingData = himmelblauFunction.function.trainingSamples;
	trainingDataLowFi = himmelblauFunction.function.trainingSamplesLowFidelity;


	setDefinitionForCase4();
	objFunTest.setParametersByDefinition(definition);
	objFunTest.initializeSurrogate();

	MultiLevelModel testModel = objFunTest.getSurrogateModelML();

	unsigned int NSamplesHiFi  = testModel.getNumberOfHiFiSamples();
	unsigned int NSamplesLowFi = testModel.getNumberOfLowFiSamples();



	mat errorData = testModel.getRawDataError();
	mat lowFiData = testModel.getRawDataLowFidelity();

	ASSERT_EQ(NSamplesHiFi, himmelblauFunction.function.numberOfTrainingSamples);
	ASSERT_EQ(NSamplesLowFi, himmelblauFunction.function.numberOfTrainingSamplesLowFi);


	bool ifDataIsConsistent = isEqual(lowFiData, trainingDataLowFi, 10E-8);


	remove(filenameTrainingDataHiFi.c_str());
	remove(filenameTrainingDataLowFi.c_str());


}


TEST_F(ObjectiveFunctionTest, readOutput){

	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "2.144 56.12 77 0\n";
	readOutputTestFile.close();

	objFunTest.setFileNameReadInput("readOutputTestFile.txt");
	rowvec result = objFunTest.readOutput(4);

	EXPECT_EQ(result(0), 2.144);
	EXPECT_EQ(result(3), 0);


	remove("readOutputTestFile.txt");

}

TEST_F(ObjectiveFunctionTest, readOutputDesign){

	Design d(2);

	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "2.144\n";
	readOutputTestFile.close();

	objFunTest.setFileNameReadInput("readOutputTestFile.txt");
	objFunTest.setEvaluationMode("primal");
	objFunTest.readOutputDesign(d);
	EXPECT_EQ(d.trueValue, 2.144);


	remove("readOutputTestFile.txt");

}

TEST_F(ObjectiveFunctionTest, readOutputDesignAdjoint){

	Design d(2);

	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "2.144 3.2 89.1\n";
	readOutputTestFile.close();

	objFunTest.setFileNameReadInput("readOutputTestFile.txt");
	objFunTest.setEvaluationMode("adjoint");
	objFunTest.readOutputDesign(d);
	EXPECT_EQ(d.trueValue, 2.144);

	rowvec gradient = d.gradient;
	EXPECT_EQ(gradient(0), 3.2);
	EXPECT_EQ(gradient(1), 89.1);

	remove("readOutputTestFile.txt");

}

TEST_F(ObjectiveFunctionTest, readOutputDesignTangent){

	Design d(2);

	std::ofstream readOutputTestFile;
	readOutputTestFile.open ("readOutputTestFile.txt");
	readOutputTestFile << "2.144 -12.11\n";
	readOutputTestFile.close();

	objFunTest.setFileNameReadInput("readOutputTestFile.txt");
	objFunTest.setEvaluationMode("tangent");
	objFunTest.readOutputDesign(d);
	EXPECT_EQ(d.trueValue, 2.144);

	EXPECT_EQ(d.tangentValue, -12.11);
	remove("readOutputTestFile.txt");

}

TEST_F(ObjectiveFunctionTest, writeDesignVariablesToFile){

	Design d(2);
	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;

	objFunTest.setParametersByDefinition(definition);
	objFunTest.setEvaluationMode("primal");
	objFunTest.writeDesignVariablesToFile(d);

	rowvec dv(2);
	std::ifstream inputFileStream(definition.designVectorFilename, ios::in);
	inputFileStream >> dv(0);
	inputFileStream >> dv(1);

	EXPECT_EQ(dv(0), 2.1);
	EXPECT_EQ(dv(1), -1.9);

	remove(definition.designVectorFilename.c_str());
}



TEST_F(ObjectiveFunctionTest, evaluateDesign){

	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;

	compileWithCpp("himmelblau.cpp", definition.executableName);

	objFunTest.setParametersByDefinition(definition);
	objFunTest.setEvaluationMode("primal");
	objFunTest.evaluateDesign(d);

	EXPECT_EQ(d.trueValue,  73.74420);

	remove(definition.designVectorFilename.c_str());
	remove(definition.outputFilename.c_str());
	remove(definition.executableName.c_str());



}

TEST_F(ObjectiveFunctionTest, evaluateDesignLowFi){

	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;

	compileWithCpp("himmelblauLowFidelity.cpp", definition.executableNameLowFi);

	definition.ifMultiLevel = true;
	objFunTest.setParametersByDefinition(definition);
	objFunTest.setEvaluationMode("primalLowFi");
	objFunTest.evaluateDesign(d);

	EXPECT_EQ(d.trueValueLowFidelity,   63.3025834400);

	remove(definition.designVectorFilename.c_str());
	remove(definition.outputFilename.c_str());
	remove(definition.executableNameLowFi.c_str());
}







TEST_F(ObjectiveFunctionTest, evaluateDesignAdjoint){


	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;

	compileWithCpp("himmelblauAdjoint.cpp", definition.executableName);

	objFunTest.setParametersByDefinition(definition);
	objFunTest.setEvaluationMode("adjoint");
	objFunTest.evaluateDesign(d);

	EXPECT_EQ(d.trueValue,  73.74420);
	EXPECT_EQ(d.gradient(0),  -73.896);
	EXPECT_EQ(d.gradient(1),  -7.176);

	remove(definition.designVectorFilename.c_str());
	remove(definition.outputFilename.c_str());
	remove(definition.executableName.c_str());



}


TEST_F(ObjectiveFunctionTest, evaluateDesignAdjointLowFi){


	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;

	compileWithCpp("himmelblauAdjointLowFi.cpp", definition.executableNameLowFi);



	objFunTest.setParametersByDefinition(definition);
	objFunTest.setEvaluationMode("adjointLowFi");
	objFunTest.evaluateDesign(d);

	EXPECT_EQ(d.trueValueLowFidelity,  63.3025834400);
	EXPECT_EQ(d.gradientLowFidelity(0),  -74.8981800000);
	EXPECT_EQ(d.gradientLowFidelity(1),  -7.1264304000);

	remove(definition.designVectorFilename.c_str());
	remove(definition.outputFilename.c_str());
	remove(definition.executableName.c_str());

}

TEST_F(ObjectiveFunctionTest, evaluateDesignTangent){


	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;
	rowvec diffDirection(2);
	diffDirection(0) = 1.0;
	diffDirection(1) = 0.0;
	d.tangentDirection = diffDirection;

	compileWithCpp("himmelblauTangent.cpp", definition.executableName);

	objFunTest.setParametersByDefinition(definition);
	objFunTest.setEvaluationMode("tangent");
	objFunTest.evaluateDesign(d);

	EXPECT_EQ(d.trueValue,  73.74420);
	EXPECT_EQ(d.tangentValue,  -73.896);


	remove(definition.designVectorFilename.c_str());
	remove(definition.outputFilename.c_str());
	remove(definition.executableName.c_str());



}


TEST_F(ObjectiveFunctionTest, evaluateDesignTangentLowFi){


	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;
	rowvec diffDirection(2);
	diffDirection(0) = 1.0;
	diffDirection(1) = 0.0;
	d.tangentDirection = diffDirection;

	compileWithCpp("himmelblauTangentLowFi.cpp", definition.executableNameLowFi);

	objFunTest.setParametersByDefinition(definition);
	objFunTest.setEvaluationMode("tangentLowFi");
	objFunTest.evaluateDesign(d);

	EXPECT_EQ(d.trueValueLowFidelity,  63.3025834400);
	EXPECT_EQ(d.tangentValueLowFidelity,  -74.8981800000);


	remove(definition.designVectorFilename.c_str());
	remove(definition.outputFilename.c_str());
	remove(definition.executableName.c_str());

}


TEST_F(ObjectiveFunctionTest, addDesignToData){


	himmelblauFunction.function.generateTrainingSamples();
	trainingData = himmelblauFunction.function.trainingSamples;

	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;
	d.trueValue = 2.4;

	objFunTest.setParametersByDefinition(definition);
	objFunTest.setDataAddMode("primal");
	objFunTest.initializeSurrogate();
	objFunTest.addDesignToData(d);

	mat newData;
	newData.load(filenameTrainingData, csv_ascii);

	ASSERT_TRUE(newData.n_rows == trainingData.n_rows+1);

	rowvec lastRow = newData.row(newData.n_rows-1);
	ASSERT_EQ(lastRow(0),2.1);
	ASSERT_EQ(lastRow(1),-1.9);
	ASSERT_EQ(lastRow(2),2.4);

	remove(filenameTrainingData.c_str());

}


TEST_F(ObjectiveFunctionTest, addDesignToDataGGEKModel){


	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();
	trainingData = himmelblauFunction.function.trainingSamples;

	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;
	d.trueValue = 2.4;

	setDefinitionForCase2();
	objFunTest.setParametersByDefinition(definition);
	objFunTest.setDataAddMode("primal");
	objFunTest.initializeSurrogate();
	objFunTest.addDesignToData(d);

	mat newData;
	newData.load(filenameTrainingData, csv_ascii);

	//	newData.print("newData");



	ASSERT_TRUE(newData.n_rows == trainingData.n_rows+1);

	rowvec lastRow = newData.row(newData.n_rows-1);
	ASSERT_EQ(lastRow(0),2.1);
	ASSERT_EQ(lastRow(1),-1.9);
	ASSERT_EQ(lastRow(2),2.4);
	ASSERT_EQ(lastRow(3),0.0);
	ASSERT_EQ(lastRow(4),0.0);

	remove(filenameTrainingData.c_str());


}











TEST_F(ObjectiveFunctionTest, addDesignToDataTangentModel){


	himmelblauFunction.function.generateTrainingSamplesWithTangents();
	trainingData = himmelblauFunction.function.trainingSamples;

	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;
	d.trueValue = 2.4;

	setDefinitionForCase3();
	objFunTest.setParametersByDefinition(definition);
	objFunTest.setDataAddMode("primal");
	objFunTest.initializeSurrogate();
	objFunTest.addDesignToData(d);

	mat newData;
	newData.load(filenameTrainingData, csv_ascii);

	//	newData.print("newData");



	ASSERT_TRUE(newData.n_rows == trainingData.n_rows+1);

	rowvec lastRow = newData.row(newData.n_rows-1);
	ASSERT_EQ(lastRow(0),2.1);
	ASSERT_EQ(lastRow(1),-1.9);
	ASSERT_EQ(lastRow(2),2.4);
	ASSERT_EQ(lastRow(3),0.0);
	ASSERT_EQ(lastRow(4),0.0);
	ASSERT_EQ(lastRow(5),0.0);

	remove(filenameTrainingData.c_str());


}





TEST_F(ObjectiveFunctionTest, addDesignToDataWithMultiFidelity){

	himmelblauFunction.function.numberOfTrainingSamples = 20;
	himmelblauFunction.function.numberOfTrainingSamplesLowFi = 50;
	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();
	trainingData = himmelblauFunction.function.trainingSamples;
	trainingDataLowFi = himmelblauFunction.function.trainingSamplesLowFidelity;

	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;
	d.trueValue = 2.4;
	d.trueValueLowFidelity = 2.67;

	definition.ifMultiLevel = true;

	//	objFunTest.setDisplayOn();
	objFunTest.setParametersByDefinition(definition);
	objFunTest.setDataAddMode("primalBoth");
	objFunTest.initializeSurrogate();

	objFunTest.addDesignToData(d);

	mat newData;
	newData.load(filenameTrainingData, csv_ascii);

	//	newData.print("HiFi data");

	mat newDataLowFi;
	newDataLowFi.load(filenameTrainingDataLowFi, csv_ascii);

	//	newDataLowFi.print("LowFi data");

	ASSERT_TRUE(newData.n_rows      == trainingData.n_rows+1);
	ASSERT_TRUE(newDataLowFi.n_rows == trainingDataLowFi.n_rows+1);

	rowvec lastRow = newData.row(newData.n_rows-1);
	ASSERT_EQ(lastRow(0),2.1);
	ASSERT_EQ(lastRow(1),-1.9);
	ASSERT_EQ(lastRow(2),2.4);

	lastRow = newDataLowFi.row(newDataLowFi.n_rows-1);
	ASSERT_EQ(lastRow(0),2.1);
	ASSERT_EQ(lastRow(1),-1.9);
	ASSERT_EQ(lastRow(2),2.67);

	remove(filenameTrainingData.c_str());
	remove(filenameTrainingDataLowFi.c_str());

}


TEST_F(ObjectiveFunctionTest, addDesignToDataWithMultiFidelityGGEKModelLowFi){

	himmelblauFunction.function.numberOfTrainingSamples = 20;
	himmelblauFunction.function.numberOfTrainingSamplesLowFi = 50;
	himmelblauFunction.function.generateTrainingSamplesMultiFidelityWithLowFiAdjoint();
	trainingData = himmelblauFunction.function.trainingSamples;
	trainingDataLowFi = himmelblauFunction.function.trainingSamplesLowFidelity;

	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;
	d.trueValue = 2.4;
	d.trueValueLowFidelity = 2.67;
	rowvec gradientLowFi(2);
	gradientLowFi(0) = -88.9;
	gradientLowFi(1) =  3.2;
	d.gradientLowFidelity = gradientLowFi;

	setDefinitionForCase5();

	//	objFunTest.setDisplayOn();
	objFunTest.setParametersByDefinition(definition);
	objFunTest.setDataAddMode("primalHiFiAdjointLowFi");
	objFunTest.initializeSurrogate();

	objFunTest.addDesignToData(d);

	mat newData;
	newData.load(filenameTrainingData, csv_ascii);

	//	newData.print("HiFi data");

	mat newDataLowFi;
	newDataLowFi.load(filenameTrainingDataLowFi, csv_ascii);

	//	newDataLowFi.print("LowFi data");

	ASSERT_TRUE(newData.n_rows      == trainingData.n_rows+1);
	ASSERT_TRUE(newDataLowFi.n_rows == trainingDataLowFi.n_rows+1);

	rowvec lastRow = newData.row(newData.n_rows-1);
	ASSERT_EQ(lastRow(0),2.1);
	ASSERT_EQ(lastRow(1),-1.9);
	ASSERT_EQ(lastRow(2),2.4);

	lastRow = newDataLowFi.row(newDataLowFi.n_rows-1);
	ASSERT_EQ(lastRow(0),2.1);
	ASSERT_EQ(lastRow(1),-1.9);
	ASSERT_EQ(lastRow(2),2.67);
	ASSERT_EQ(lastRow(3),-88.9);
	ASSERT_EQ(lastRow(4),3.2);

	remove(filenameTrainingData.c_str());
	remove(filenameTrainingDataLowFi.c_str());

}







TEST_F(ObjectiveFunctionTest, addLowFiDesignToDataWithMultiFidelity){

	himmelblauFunction.function.numberOfTrainingSamples = 20;
	himmelblauFunction.function.numberOfTrainingSamplesLowFi = 50;
	himmelblauFunction.function.generateTrainingSamplesMultiFidelity();
	trainingData = himmelblauFunction.function.trainingSamples;
	trainingDataLowFi = himmelblauFunction.function.trainingSamplesLowFidelity;

	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;
	d.trueValue = 2.4;
	d.trueValueLowFidelity = 2.67;

	definition.ifMultiLevel = true;

	//	objFunTest.setDisplayOn();
	objFunTest.setParametersByDefinition(definition);
	objFunTest.setDataAddMode("primalLowFidelity");
	objFunTest.initializeSurrogate();

	objFunTest.addLowFidelityDesignToData(d);

	mat newDataLowFi;
	newDataLowFi.load(filenameTrainingDataLowFi, csv_ascii);

	//	newDataLowFi.print("LowFi data");


	ASSERT_TRUE(newDataLowFi.n_rows == trainingDataLowFi.n_rows+1);

	rowvec lastRow = newDataLowFi.row(newDataLowFi.n_rows-1);
	ASSERT_EQ(lastRow(0),2.1);
	ASSERT_EQ(lastRow(1),-1.9);
	ASSERT_EQ(lastRow(2),2.67);

	remove(filenameTrainingData.c_str());
	remove(filenameTrainingDataLowFi.c_str());

}









TEST_F(ObjectiveFunctionTest, addDesignToDataTangent){


	himmelblauFunction.function.generateTrainingSamplesWithTangents();
	trainingData = himmelblauFunction.function.trainingSamples;

	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;
	rowvec diffDirection(2);
	diffDirection(0) = 1.0;
	diffDirection(1) = 0.0;
	d.tangentDirection = diffDirection;
	d.trueValue = 2.4;
	d.tangentValue = 0.67;

	setDefinitionForCase3();
	objFunTest.setParametersByDefinition(definition);
	objFunTest.setDataAddMode("tangent");
	objFunTest.initializeSurrogate();
	objFunTest.addDesignToData(d);

	mat newData;
	newData.load(filenameTrainingData, csv_ascii);

	ASSERT_TRUE(newData.n_rows == trainingData.n_rows+1);

	rowvec lastRow = newData.row(newData.n_rows-1);
	ASSERT_EQ(lastRow(0),2.1);
	ASSERT_EQ(lastRow(1),-1.9);
	ASSERT_EQ(lastRow(2),2.4);

	remove(filenameTrainingData.c_str());

}

TEST_F(ObjectiveFunctionTest, addDesignToDataAdjoint){


	himmelblauFunction.function.generateTrainingSamplesWithAdjoints();
	trainingData = himmelblauFunction.function.trainingSamples;

	Design d(2);

	rowvec dvInput(2);
	dvInput(0) = 2.1;
	dvInput(1) = -1.9;
	d.designParameters = dvInput;
	d.trueValue = 2.4;
	rowvec gradient(2);
	gradient(0) = 89.1;
	gradient(1) = -100.9;
	d.gradient = gradient;

	setDefinitionForCase2();
	objFunTest.setParametersByDefinition(definition);
	objFunTest.setDataAddMode("adjoint");
	objFunTest.initializeSurrogate();
	objFunTest.addDesignToData(d);

	mat newData;
	newData.load(filenameTrainingData, csv_ascii);

	ASSERT_TRUE(newData.n_rows == trainingData.n_rows+1);
	ASSERT_TRUE(newData.n_cols == 5);

	rowvec lastRow = newData.row(newData.n_rows-1);
	ASSERT_EQ(lastRow(0),2.1);
	ASSERT_EQ(lastRow(1),-1.9);
	ASSERT_EQ(lastRow(2),2.4);

	remove(filenameTrainingData.c_str());

}




TEST_F(ObjectiveFunctionTest, trainSurrogate){

	himmelblauFunction.function.generateTrainingSamples();
	trainingData = himmelblauFunction.function.trainingSamples;

	setDefinitionForCase1();
	objFunTest.setParametersByDefinition(definition);
	objFunTest.initializeSurrogate();
	objFunTest.setParametersByDefinition(definition);
	objFunTest.initializeSurrogate();
	objFunTest.setNumberOfTrainingIterationsForSurrogateModel(1000);

	objFunTest.trainSurrogate();

}




#endif


