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

#include "gek.hpp"
#include "gek_test.hpp"
#include "auxiliary_functions.hpp"
#include "test_functions.hpp"
#include "polynomials.hpp"
#include <vector>
#include <armadillo>




//void testGEKcalculateRDot(void){
//
//	cout<<__func__<<"\n";
//	int dim = 2;
//	int N =  generateRandomInt(5,10);
//
//	generateRandomTestAndValidationDataForGradientModels(dim,N);
//
//	GEKModel testModel("testData");
//	testModel.initializeSurrogateModel();
//	testModel.GEK_weights.fill(1.0);
//	testModel.printSurrogateModel();
//	testModel.computeCorrelationMatrixDot();
//
//	printMatrix(testModel.correlationMatrixDot,"correlationMatrixDot");
//
//
//
//}
//
//void testGEKTrain(void){
//
//	cout<<__func__<<":";
//	int dim = 6;
//	int N =  generateRandomInt(10,20);
//
//	generateRandomTestAndValidationDataForGradientModels(dim,N);
//
//	GEKModel testModel("testData");
//	testModel.train();
//
//
//
//}
//
//
//
//
//
//void testGEKcalculateRDotValidateWithWingweight(void){
//	cout<<__func__<<":";
//	GEKModel testModel("Null");
//	testModel.dim = 10;
//	testModel.N = 5;
//
//	mat X(5,10);
//
//	X(0.0) = 0.1; X(0,1) = 0.2; X(0,2) = 0.1; X(0,3) = 0.1; X(0,4) = 0.4; X(0,5) = 0.0; X(0,6) = 0.1; X(0,7) = 0.5; X(0,8) = 0.1; X(0,9) = 0.4;
//	X(1.0) = 0.2; X(1,1) = 0.3; X(1,2) = 0.5; X(1,3) = 0.7; X(1,4) = 0.8; X(1,5) = 0.2; X(1,6) = 0.5; X(1,7) = 0.6; X(1,8) = 0.2; X(1,9) = 0.7;
//	X(2.0) = 0.0; X(2,1) = 0.3; X(2,2) = 0.6; X(2,3) = 0.7; X(2,4) = 0.8; X(2,5) = 0.3; X(2,6) = 0.5; X(2,7) = 0.6; X(2,8) = 0.2; X(2,9) = 0.9;
//	X(3.0) = 0.9; X(3,1) = 0.3; X(3,2) = 0.4; X(3,3) = 0.1; X(3,4) = 0.8; X(3,5) = 0.2; X(3,6) = 0.8; X(3,7) = 0.9; X(3,8) = 0.2; X(3,9) = 0.2;
//	X(4.0) = 0.2; X(4,1) = 0.8; X(4,2) = 0.5; X(4,3) = 0.9; X(4,4) = 0.8; X(4,5) = 0.2; X(4,6) = 0.2; X(4,7) = 0.6; X(4,8) = 0.2; X(4,9) = 0.0;
//
//
//	testModel.X = X;
//
//	vec yGEK(55);
//
//	yGEK(0) = 2.178360857017802;
//	yGEK(1) = 2.453961344881372;
//	yGEK(2) = 2.425744516471332;
//	yGEK(3) = 2.842435443681024;
//	yGEK(4) = 2.603226608240350;
//	yGEK(5) = 0.538331359223321;
//	yGEK(6) = 0.588965590964108;
//	yGEK(7) = 0.621919275545224;
//	yGEK(8) = 0.556808834062394;
//	yGEK(9) = 0.619664302803358;
//	yGEK(10) =  0.002498063622625;
//	yGEK(11) =  0.002699434844544;
//	yGEK(12) =  0.002655403932295;
//	yGEK(13) =  0.003181253305229;
//	yGEK(14) =	0.002527131073293;
//	yGEK(15) =	0.789566324442603;
//	yGEK(16) =  0.705708227087598;
//	yGEK(17) = 0.661141133036836;
//	yGEK(18) = 0.875442540973381;
//	yGEK(19) = 0.768967790236275;
//	yGEK(20) = 	-0.092962887947579;
//	yGEK(21) = 	0.051677105830095;
//	yGEK(22) = 	0.050834255489463;
//	yGEK(23) = -0.122400228974584;
//	yGEK(24) =  0.113172180817855;
//	yGEK(25) =  0.013273870280273;
//	yGEK(26) = 0.010441602341871;
//	yGEK(27) = 0.010271301675857;
//	yGEK(28) = 0.012305328823459;
//	yGEK(29) = 0.011377585906303;
//	yGEK(30) = 0.084219625776427;
//	yGEK(31) = 0.078411417533175;
//	yGEK(32) = 0.071199305494929;
//	yGEK(33) = 0.092407108866155;
//	yGEK(34) = 0.085440203406506;
//	yGEK(35) = -0.701786268433579;
//	yGEK(36) = -0.542825477018027;
//	yGEK(37) = -0.533972036211594;
//	yGEK(38) =-0.519773030060833;
//	yGEK(39) =-0.768918003387853;
//	yGEK(40) = 0.849635047813990;
//	yGEK(41) =0.877021189111815;
//	yGEK(42) =0.862717042522831;
//	yGEK(43) =0.841483476051508;
//	yGEK(44) =0.955637216350169;
//	yGEK(45) =0.463685546492343;
//	yGEK(46) =0.495766448693757;
//	yGEK(47) =0.487680540572910;
//	yGEK(48) =0.584256076763268;
//	yGEK(49) =0.540206838195445;
//	yGEK(50) =0.085250000000485;
//	yGEK(51) =0.087999999999795;
//	yGEK(52) =0.082499999999612;
//	yGEK(53) =0.107250000001216;
//	yGEK(54) =0.087999999999795;
//	testModel.yGEK = yGEK;
//
//	vec theta(10);
//
//	theta(0) = 0.012279429688457;
//	theta(1) = 0.103476703621396;
//	theta(2) = 0.023450618868551;
//	theta(3) = 0.061365259817944;
//	theta(4) = 0.052738210130522;
//	theta(5) = 0.181358134391257;
//	theta(6) = 0.538333452068542;
//	theta(7) = 0.191868736397573;
//	theta(8) = 0.033220525588041;
//	theta(9) = 0.015215884822339;
//
//
//	testModel.correlationMatrixDot = zeros(55,55);
//
//
//	testModel.epsilonGEK = 0.0000000001;
//	testModel.GEK_weights = theta;
//
//	testModel.computeCorrelationMatrixDot();
//
//	bool passTest = checkValue(testModel.correlationMatrixDot(0,0),1.00000000010);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(0,4),0.899279420159646);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(0,53),0.004410241181619);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(1,5),0.002151231135528);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(1,6),0.0);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(11,0),-0.018128065576320);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(11,1),0.0);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(11,2),0.0);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(11,3),0.0);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(17,20),-0.001492192874528);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(17,21),0.0);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(17,22),0.0);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(17,23),-0.000620054989134);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(18,54),0.000021508257779);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(18,53),0.0);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(18,52),-0.000179370752777);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(18,51),-0.000064711169440);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(18,50),0.000062053731041);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(54,53),0.022901396041769);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(54,52),0.027082575854412);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(testModel.correlationMatrixDot(54,51),0.027556368599073);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//
//
//	cout<<"\t passed\n";
//
//
//
//}
//
//void testGEKcalculateCorrelationVectorDotWithWingweight(void){
//	cout<<__func__<<":";
//	GEKModel testModel("Null");
//	testModel.dim = 10;
//	testModel.N = 5;
//
//	mat X(5,10);
//
//	X(0.0) = 0.1; X(0,1) = 0.2; X(0,2) = 0.1; X(0,3) = 0.1; X(0,4) = 0.4; X(0,5) = 0.0; X(0,6) = 0.1; X(0,7) = 0.5; X(0,8) = 0.1; X(0,9) = 0.4;
//	X(1.0) = 0.2; X(1,1) = 0.3; X(1,2) = 0.5; X(1,3) = 0.7; X(1,4) = 0.8; X(1,5) = 0.2; X(1,6) = 0.5; X(1,7) = 0.6; X(1,8) = 0.2; X(1,9) = 0.7;
//	X(2.0) = 0.0; X(2,1) = 0.3; X(2,2) = 0.6; X(2,3) = 0.7; X(2,4) = 0.8; X(2,5) = 0.3; X(2,6) = 0.5; X(2,7) = 0.6; X(2,8) = 0.2; X(2,9) = 0.9;
//	X(3.0) = 0.9; X(3,1) = 0.3; X(3,2) = 0.4; X(3,3) = 0.1; X(3,4) = 0.8; X(3,5) = 0.2; X(3,6) = 0.8; X(3,7) = 0.9; X(3,8) = 0.2; X(3,9) = 0.2;
//	X(4.0) = 0.2; X(4,1) = 0.8; X(4,2) = 0.5; X(4,3) = 0.9; X(4,4) = 0.8; X(4,5) = 0.2; X(4,6) = 0.2; X(4,7) = 0.6; X(4,8) = 0.2; X(4,9) = 0.0;
//
//
//	testModel.X = X;
//
//	vec yGEK(55);
//
//	yGEK(0) = 2.178360857017802;
//	yGEK(1) = 2.453961344881372;
//	yGEK(2) = 2.425744516471332;
//	yGEK(3) = 2.842435443681024;
//	yGEK(4) = 2.603226608240350;
//	yGEK(5) = 0.538331359223321;
//	yGEK(6) = 0.588965590964108;
//	yGEK(7) = 0.621919275545224;
//	yGEK(8) = 0.556808834062394;
//	yGEK(9) = 0.619664302803358;
//	yGEK(10) =  0.002498063622625;
//	yGEK(11) =  0.002699434844544;
//	yGEK(12) =  0.002655403932295;
//	yGEK(13) =  0.003181253305229;
//	yGEK(14) =	0.002527131073293;
//	yGEK(15) =	0.789566324442603;
//	yGEK(16) =  0.705708227087598;
//	yGEK(17) = 0.661141133036836;
//	yGEK(18) = 0.875442540973381;
//	yGEK(19) = 0.768967790236275;
//	yGEK(20) = 	-0.092962887947579;
//	yGEK(21) = 	0.051677105830095;
//	yGEK(22) = 	0.050834255489463;
//	yGEK(23) = -0.122400228974584;
//	yGEK(24) =  0.113172180817855;
//	yGEK(25) =  0.013273870280273;
//	yGEK(26) = 0.010441602341871;
//	yGEK(27) = 0.010271301675857;
//	yGEK(28) = 0.012305328823459;
//	yGEK(29) = 0.011377585906303;
//	yGEK(30) = 0.084219625776427;
//	yGEK(31) = 0.078411417533175;
//	yGEK(32) = 0.071199305494929;
//	yGEK(33) = 0.092407108866155;
//	yGEK(34) = 0.085440203406506;
//	yGEK(35) = -0.701786268433579;
//	yGEK(36) = -0.542825477018027;
//	yGEK(37) = -0.533972036211594;
//	yGEK(38) =-0.519773030060833;
//	yGEK(39) =-0.768918003387853;
//	yGEK(40) = 0.849635047813990;
//	yGEK(41) =0.877021189111815;
//	yGEK(42) =0.862717042522831;
//	yGEK(43) =0.841483476051508;
//	yGEK(44) =0.955637216350169;
//	yGEK(45) =0.463685546492343;
//	yGEK(46) =0.495766448693757;
//	yGEK(47) =0.487680540572910;
//	yGEK(48) =0.584256076763268;
//	yGEK(49) =0.540206838195445;
//	yGEK(50) =0.085250000000485;
//	yGEK(51) =0.087999999999795;
//	yGEK(52) =0.082499999999612;
//	yGEK(53) =0.107250000001216;
//	yGEK(54) =0.087999999999795;
//
//	yGEK = yGEK*100.0;
//	testModel.yGEK = yGEK;
//
//	vec theta(10);
//
//	theta(0) = 0.012279429688457;
//	theta(1) = 0.103476703621396;
//	theta(2) = 0.023450618868551;
//	theta(3) = 0.061365259817944;
//	theta(4) = 0.052738210130522;
//	theta(5) = 0.181358134391257;
//	theta(6) = 0.538333452068542;
//	theta(7) = 0.191868736397573;
//	theta(8) = 0.033220525588041;
//	theta(9) = 0.015215884822339;
//
//
//	testModel.correlationMatrixDot = zeros(55,55);
//
//	testModel.R_inv_ys_min_beta = zeros<vec>(55);
//	testModel.R_inv_F= zeros<vec>(55);
//
//	vec vectorOfF= zeros<vec>(55);
//
//	for(unsigned int i=0; i<5; i++) {
//
//		vectorOfF(i)=1.0;
//	}
//
//	testModel.epsilonGEK = 0.0000000001;
//	testModel.R_inv_ys_min_beta = zeros<vec>(55);
//	testModel.upperDiagonalMatrixDot= zeros<mat>(55,55);
//	testModel.vectorOfF = vectorOfF;
//	testModel.GEK_weights = theta;
//
//
//	testModel.updateAuxilliaryFields();
//
//	rowvec xp(10);
//
//	xp(0) = 0.193431156405215;
//	xp(1) = 0.252689256144722;
//	xp(2) = 0.649238334197916;
//	xp(3) = 0.446982789756469;
//	xp(4) = 0.051430365565162;
//	xp(5) = 0.248341860867715;
//	xp(6) = 0.930620140532264;
//	xp(7) = 0.058557456070293;
//	xp(8) = 0.857424125174308;
//	xp(9) = 0.621933924936728;
//
//
//	vec r = testModel.computeCorrelationVectorDot(xp);
//#if 0
//	printVector(r,"r");
//#endif
//	bool passTest = checkValue(r(0),0.63061520214);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(r(6), -0.000131397525669);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(r(7),0.003864832995364);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(r(8),-0.014103955282093);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(r(9),-0.000104111682624);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//	passTest = checkValue(r(54),0.012214446925104);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//
//
//	cout<<"\t passed\n";
//
//
//}
//
//void testGEKValueOfMuWithWingweight(void){
//	cout<<__func__<<":";
//	GEKModel testModel("Null");
//	testModel.dim = 10;
//	testModel.N = 5;
//
//	mat X(5,10);
//
//	X(0.0) = 0.1; X(0,1) = 0.2; X(0,2) = 0.1; X(0,3) = 0.1; X(0,4) = 0.4; X(0,5) = 0.0; X(0,6) = 0.1; X(0,7) = 0.5; X(0,8) = 0.1; X(0,9) = 0.4;
//	X(1.0) = 0.2; X(1,1) = 0.3; X(1,2) = 0.5; X(1,3) = 0.7; X(1,4) = 0.8; X(1,5) = 0.2; X(1,6) = 0.5; X(1,7) = 0.6; X(1,8) = 0.2; X(1,9) = 0.7;
//	X(2.0) = 0.0; X(2,1) = 0.3; X(2,2) = 0.6; X(2,3) = 0.7; X(2,4) = 0.8; X(2,5) = 0.3; X(2,6) = 0.5; X(2,7) = 0.6; X(2,8) = 0.2; X(2,9) = 0.9;
//	X(3.0) = 0.9; X(3,1) = 0.3; X(3,2) = 0.4; X(3,3) = 0.1; X(3,4) = 0.8; X(3,5) = 0.2; X(3,6) = 0.8; X(3,7) = 0.9; X(3,8) = 0.2; X(3,9) = 0.2;
//	X(4.0) = 0.2; X(4,1) = 0.8; X(4,2) = 0.5; X(4,3) = 0.9; X(4,4) = 0.8; X(4,5) = 0.2; X(4,6) = 0.2; X(4,7) = 0.6; X(4,8) = 0.2; X(4,9) = 0.0;
//
//
//	testModel.X = X;
//
//	vec yGEK(55);
//
//	yGEK(0) = 2.178360857017802;
//	yGEK(1) = 2.453961344881372;
//	yGEK(2) = 2.425744516471332;
//	yGEK(3) = 2.842435443681024;
//	yGEK(4) = 2.603226608240350;
//	yGEK(5) = 0.538331359223321;
//	yGEK(6) = 0.588965590964108;
//	yGEK(7) = 0.621919275545224;
//	yGEK(8) = 0.556808834062394;
//	yGEK(9) = 0.619664302803358;
//	yGEK(10) =  0.002498063622625;
//	yGEK(11) =  0.002699434844544;
//	yGEK(12) =  0.002655403932295;
//	yGEK(13) =  0.003181253305229;
//	yGEK(14) =	0.002527131073293;
//	yGEK(15) =	0.789566324442603;
//	yGEK(16) =  0.705708227087598;
//	yGEK(17) = 0.661141133036836;
//	yGEK(18) = 0.875442540973381;
//	yGEK(19) = 0.768967790236275;
//	yGEK(20) = 	-0.092962887947579;
//	yGEK(21) = 	0.051677105830095;
//	yGEK(22) = 	0.050834255489463;
//	yGEK(23) = -0.122400228974584;
//	yGEK(24) =  0.113172180817855;
//	yGEK(25) =  0.013273870280273;
//	yGEK(26) = 0.010441602341871;
//	yGEK(27) = 0.010271301675857;
//	yGEK(28) = 0.012305328823459;
//	yGEK(29) = 0.011377585906303;
//	yGEK(30) = 0.084219625776427;
//	yGEK(31) = 0.078411417533175;
//	yGEK(32) = 0.071199305494929;
//	yGEK(33) = 0.092407108866155;
//	yGEK(34) = 0.085440203406506;
//	yGEK(35) = -0.701786268433579;
//	yGEK(36) = -0.542825477018027;
//	yGEK(37) = -0.533972036211594;
//	yGEK(38) =-0.519773030060833;
//	yGEK(39) =-0.768918003387853;
//	yGEK(40) = 0.849635047813990;
//	yGEK(41) =0.877021189111815;
//	yGEK(42) =0.862717042522831;
//	yGEK(43) =0.841483476051508;
//	yGEK(44) =0.955637216350169;
//	yGEK(45) =0.463685546492343;
//	yGEK(46) =0.495766448693757;
//	yGEK(47) =0.487680540572910;
//	yGEK(48) =0.584256076763268;
//	yGEK(49) =0.540206838195445;
//	yGEK(50) =0.085250000000485;
//	yGEK(51) =0.087999999999795;
//	yGEK(52) =0.082499999999612;
//	yGEK(53) =0.107250000001216;
//	yGEK(54) =0.087999999999795;
//
//	yGEK = yGEK*100.0;
//	testModel.yGEK = yGEK;
//
//	vec theta(10);
//
//	theta(0) = 0.012279429688457;
//	theta(1) = 0.103476703621396;
//	theta(2) = 0.023450618868551;
//	theta(3) = 0.061365259817944;
//	theta(4) = 0.052738210130522;
//	theta(5) = 0.181358134391257;
//	theta(6) = 0.538333452068542;
//	theta(7) = 0.191868736397573;
//	theta(8) = 0.033220525588041;
//	theta(9) = 0.015215884822339;
//
//
//	testModel.correlationMatrixDot = zeros(55,55);
//
//	testModel.R_inv_ys_min_beta = zeros<vec>(55);
//	testModel.R_inv_F= zeros<vec>(55);
//
//	vec vectorOfF= zeros<vec>(55);
//
//	for(unsigned int i=0; i<5; i++) {
//
//		vectorOfF(i)=1.0;
//	}
//
//	testModel.epsilonGEK = 0.0000000001;
//	testModel.R_inv_ys_min_beta = zeros<vec>(55);
//	testModel.upperDiagonalMatrixDot= zeros<mat>(55,55);
//	testModel.vectorOfF = vectorOfF;
//	testModel.GEK_weights = theta;
//
//
//	testModel.updateAuxilliaryFields();
//
//#if 0
//	printVector(testModel.R_inv_ys_min_beta,"R_inv_ys_min_beta");
//	cout<<"beta0 = "<<testModel.beta0<<"\n";
//#endif
//
//	bool passTest = checkValue(testModel.beta0,2.567361163154011e+02);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//
//
//
//
//
//
//	cout<<"\t passed\n";
//
//
//}
//
//void testGEKPredictionWithWingweight(void){
//	cout<<__func__<<":";
//	GEKModel testModel("Null");
//	testModel.dim = 10;
//	testModel.N = 5;
//
//	mat X(5,10);
//
//	X(0.0) = 0.1; X(0,1) = 0.2; X(0,2) = 0.1; X(0,3) = 0.1; X(0,4) = 0.4; X(0,5) = 0.0; X(0,6) = 0.1; X(0,7) = 0.5; X(0,8) = 0.1; X(0,9) = 0.4;
//	X(1.0) = 0.2; X(1,1) = 0.3; X(1,2) = 0.5; X(1,3) = 0.7; X(1,4) = 0.8; X(1,5) = 0.2; X(1,6) = 0.5; X(1,7) = 0.6; X(1,8) = 0.2; X(1,9) = 0.7;
//	X(2.0) = 0.0; X(2,1) = 0.3; X(2,2) = 0.6; X(2,3) = 0.7; X(2,4) = 0.8; X(2,5) = 0.3; X(2,6) = 0.5; X(2,7) = 0.6; X(2,8) = 0.2; X(2,9) = 0.9;
//	X(3.0) = 0.9; X(3,1) = 0.3; X(3,2) = 0.4; X(3,3) = 0.1; X(3,4) = 0.8; X(3,5) = 0.2; X(3,6) = 0.8; X(3,7) = 0.9; X(3,8) = 0.2; X(3,9) = 0.2;
//	X(4.0) = 0.2; X(4,1) = 0.8; X(4,2) = 0.5; X(4,3) = 0.9; X(4,4) = 0.8; X(4,5) = 0.2; X(4,6) = 0.2; X(4,7) = 0.6; X(4,8) = 0.2; X(4,9) = 0.0;
//
//
//	testModel.X = X;
//
//	vec yGEK(55);
//
//	yGEK(0) = 2.178360857017802;
//	yGEK(1) = 2.453961344881372;
//	yGEK(2) = 2.425744516471332;
//	yGEK(3) = 2.842435443681024;
//	yGEK(4) = 2.603226608240350;
//	yGEK(5) = 0.538331359223321;
//	yGEK(6) = 0.588965590964108;
//	yGEK(7) = 0.621919275545224;
//	yGEK(8) = 0.556808834062394;
//	yGEK(9) = 0.619664302803358;
//	yGEK(10) =  0.002498063622625;
//	yGEK(11) =  0.002699434844544;
//	yGEK(12) =  0.002655403932295;
//	yGEK(13) =  0.003181253305229;
//	yGEK(14) =	0.002527131073293;
//	yGEK(15) =	0.789566324442603;
//	yGEK(16) =  0.705708227087598;
//	yGEK(17) = 0.661141133036836;
//	yGEK(18) = 0.875442540973381;
//	yGEK(19) = 0.768967790236275;
//	yGEK(20) = 	-0.092962887947579;
//	yGEK(21) = 	0.051677105830095;
//	yGEK(22) = 	0.050834255489463;
//	yGEK(23) = -0.122400228974584;
//	yGEK(24) =  0.113172180817855;
//	yGEK(25) =  0.013273870280273;
//	yGEK(26) = 0.010441602341871;
//	yGEK(27) = 0.010271301675857;
//	yGEK(28) = 0.012305328823459;
//	yGEK(29) = 0.011377585906303;
//	yGEK(30) = 0.084219625776427;
//	yGEK(31) = 0.078411417533175;
//	yGEK(32) = 0.071199305494929;
//	yGEK(33) = 0.092407108866155;
//	yGEK(34) = 0.085440203406506;
//	yGEK(35) = -0.701786268433579;
//	yGEK(36) = -0.542825477018027;
//	yGEK(37) = -0.533972036211594;
//	yGEK(38) =-0.519773030060833;
//	yGEK(39) =-0.768918003387853;
//	yGEK(40) = 0.849635047813990;
//	yGEK(41) =0.877021189111815;
//	yGEK(42) =0.862717042522831;
//	yGEK(43) =0.841483476051508;
//	yGEK(44) =0.955637216350169;
//	yGEK(45) =0.463685546492343;
//	yGEK(46) =0.495766448693757;
//	yGEK(47) =0.487680540572910;
//	yGEK(48) =0.584256076763268;
//	yGEK(49) =0.540206838195445;
//	yGEK(50) =0.085250000000485;
//	yGEK(51) =0.087999999999795;
//	yGEK(52) =0.082499999999612;
//	yGEK(53) =0.107250000001216;
//	yGEK(54) =0.087999999999795;
//
//	yGEK = yGEK*100.0;
//	testModel.yGEK = yGEK;
//
//	vec theta(10);
//
//	theta(0) = 0.012279429688457;
//	theta(1) = 0.103476703621396;
//	theta(2) = 0.023450618868551;
//	theta(3) = 0.061365259817944;
//	theta(4) = 0.052738210130522;
//	theta(5) = 0.181358134391257;
//	theta(6) = 0.538333452068542;
//	theta(7) = 0.191868736397573;
//	theta(8) = 0.033220525588041;
//	theta(9) = 0.015215884822339;
//
//
//	testModel.correlationMatrixDot = zeros(55,55);
//
//	testModel.R_inv_ys_min_beta = zeros<vec>(55);
//	testModel.R_inv_F= zeros<vec>(55);
//
//	vec vectorOfF= zeros<vec>(55);
//
//	for(unsigned int i=0; i<5; i++) {
//
//		vectorOfF(i)=1.0;
//	}
//
//	testModel.epsilonGEK = 0.0000000001;
//	testModel.R_inv_ys_min_beta = zeros<vec>(55);
//	testModel.upperDiagonalMatrixDot= zeros<mat>(55,55);
//	testModel.vectorOfF = vectorOfF;
//	testModel.GEK_weights = theta;
//
//
//	testModel.updateAuxilliaryFields();
//
//
//	rowvec x(10);
//	x(0) = 0.193431156405215;
//	x(1) = 0.252689256144722;
//	x(2) = 0.649238334197916;
//	x(3) = 0.446982789756469;
//	x(4) = 0.051430365565162;
//	x(5) = 0.248341860867715;
//	x(6) = 0.930620140532264;
//	x(7) = 0.058557456070293;
//	x(8) = 0.857424125174308;
//	x(9) = 0.621933924936728;
//
//	double ftilde = testModel.interpolate(x);
//
//	bool passTest = checkValue(ftilde,2.281376040473580e+02);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//
//
//	cout<<"\t passed\n";
//
//
//}
//
//
//void testGEKPredictionWithWaves(void){
//	cout<<__func__<<":";
//	GEKModel testModel("Null");
//	testModel.dim = 2;
//	testModel.N = 10;
//
//	int dimR = (testModel.dim + 1)* testModel.N;
//
//	mat X(testModel.N, testModel.dim);
//
//	X(0.0) = 0.1; X(0,1) = 0.2;
//	X(1.0) = 0.2; X(1,1) = 0.1;
//	X(2.0) = 0.4; X(2,1) = 0.3;
//	X(3.0) = 0.5; X(3,1) = 0.04;
//	X(4.0) = 0.26; X(4,1) = 0.1;
//	X(5.0) = 0.3; X(5,1) = 0.2;
//	X(6.0) = 0.25; X(6,1) = 0.25;
//	X(7.0) = 0.33; X(7,1) = 0.02;
//	X(8.0) = 0.45; X(8,1) = 0.07;
//	X(9.0) = 0.05; X(9,1) = 0.05;
//
//	testModel.X = X;
//	vec xmax(testModel.dim); xmax(0) = 10.0; xmax(1) = 10.0;
//	testModel.xmax = xmax;
//	vec xmin(testModel.dim); xmin(0) = 0.0; xmin(1) = 0.0;
//	testModel.xmin = xmin;
//
//	vec yGEK(dimR );
//
//
//
//	yGEK(0) = 0.255653805962070;
//	yGEK(1) = -1.172949331855071;
//	yGEK(2) = 1.949528533273748;
//	yGEK(3) = 0.152685598457796;
//	yGEK(4) = -1.299601492267296;
//	yGEK(5) = -0.933059119062538;
//	yGEK(6) = -0.675262089199912;
//	yGEK(7) =  1.232602357516264;
//	yGEK(8) =  0.582085628141997;
//	yGEK(9) = 1.381773290676036;
//	yGEK(10) =  -8.322931182319081;
//	yGEK(11) =  -13.072837556306617;
//	yGEK(12) =   -2.909969636264165;
//	yGEK(13) =	-16.781150892417497;
//	yGEK(14) =	9.370291197095920;
//	yGEK(15) =  19.203290512781098;
//	yGEK(16) = 5.673220070778751;
//	yGEK(17) = 19.004513865697710;
//	yGEK(18) = -18.222359233519928;
//	yGEK(19) = 	10.806044316353791;
//	yGEK(20) = 	15.136009543390283;
//	yGEK(21) = 	-18.185936412549175;
//	yGEK(22) = 5.588276434179326;
//	yGEK(23) =  -14.347120287630521;
//	yGEK(24) =  -18.185936412549175;
//	yGEK(25) = 15.136009543390283;
//	yGEK(26) = 19.178405583005986;
//	yGEK(27) =  -7.788366638483701;
//	yGEK(28) = -19.708988161499477;
//	yGEK(29) = -16.829416891253057;
//
//	testModel.yGEK = yGEK;
//	vec y(testModel.N);
//	y(0) = 0.255653805962070;
//	y(1) = -1.172949331855071;
//	y(2) = 1.949528533273748;
//	y(3) = 0.152685598457796;
//	y(4) = -1.299601492267296;
//	y(5) = -0.933059119062538;
//	y(6) = -0.675262089199912;
//	y(7) =  1.232602357516264;
//	y(8) =  0.582085628141997;
//	y(9) = 1.381773290676036;
//
//	testModel.y = y;
//
//	vec theta(testModel.dim);
//
//
//	theta(0) = 31.849996699396986;
//	theta(1) = 27.539640852205078;
//
//
//
//	testModel.correlationMatrixDot = zeros(dimR ,dimR );
//
//	testModel.R_inv_ys_min_beta = zeros<vec>(dimR );
//	testModel.R_inv_F= zeros<vec>(dimR );
//
//	vec vectorOfF= zeros<vec>(dimR );
//
//	for(unsigned int i=0; i<testModel.N; i++) {
//
//		vectorOfF(i)=1.0;
//	}
//
//	testModel.epsilonGEK = 0.00000001;
//	testModel.R_inv_ys_min_beta = zeros<vec>(dimR);
//	testModel.upperDiagonalMatrixDot= zeros<mat>(dimR,dimR);
//	testModel.vectorOfF = vectorOfF;
//	testModel.GEK_weights = theta;
//
//
//	testModel.updateAuxilliaryFields();
//
//
//
//	rowvec x(testModel.dim);
//
//
//	x(0) = 0.1;
//	x(1) = 0.2;
//
//
//	double ftilde = testModel.interpolate(x);
//
//#if 1
//	cout<<"\nftilde = "<<ftilde<<"\n";
//#endif
//
//	bool passTest = checkValue(ftilde,0.255653930557824);
//	abortIfFalse(passTest,__FILE__, __LINE__);
//
//
//	testModel.calculateInSampleError();
//
//	cout<<"\t passed\n";
//
//
//}
//
//
//void testGEKPredictionWithWavesV2(void){
//
//	GEKModel testModel("Waves2DTrial");
//	testModel.train();
//	testModel.calculateInSampleError();
//
//
//
//}
//
//void testGEKWithWingweight(void){
//
//	cout<<__func__<<":";
//	int dim = 10;
//	int N= 100;
//	int NValidation = 100;
//
//	TestFunction funcWingweight("Wingweight",dim);
//	funcWingweight.adj_ptr = WingweightAdj;
//
//
//
//
//	vec ub(10);
//	ub(0) = 200.0;
//	ub(1) = 300.0;
//	ub(2) = 10.0;
//	ub(3) = 10.0;
//	ub(4) = 45.0;
//	ub(5) = 1.0;
//	ub(6) = 0.18;
//	ub(7) = 6.0;
//	ub(8) = 2500.0;
//	ub(9) = 0.08;
//
//	vec lb(10);
//	lb(0) = 150.0;
//	lb(1) = 220.0;
//	lb(2) = 6.0;
//	lb(3) = -10.0;
//	lb(4) = 16.0;
//	lb(5) = 0.5;
//	lb(6) = 0.08;
//	lb(7) = 2.5;
//	lb(8) = 1700.0;
//	lb(9) = 0.025;
//
//	funcWingweight.setBoxConstraints(lb,ub);
//
//	mat samples = funcWingweight.generateRandomSamplesWithGradients(N);
//	mat samplesForValidation = funcWingweight.generateRandomSamples(100);
//#if 1
//	printMatrix(samplesForValidation,"samplesForValidation");
//#endif
//
//	vec ysamplesValidation = samplesForValidation.col(dim);
//	vec ysamples           = samples.col(dim);
//
//	mat XsamplesValidation = samplesForValidation.submat(0,0,NValidation-1,dim-1);
//	mat Xsamples           = samples.submat(0,0,N-1,dim-1);
//
//
//
//#if 1
//	printMatrix(XsamplesValidation,"XsamplesValidation");
//#endif
//
//
//#if 1
//	printMatrix(samples,"samples");
//#endif
//
//	samples.save("WingweightTest.csv",csv_ascii);
//
//	GEKModel modelWingweight("WingweightTest");
//
//	modelWingweight.maxNumberOfTrainingIterations = 10000;
//	modelWingweight.train();
//
//
//	Xsamples = normalizeMatrix(Xsamples, modelWingweight.xmin, modelWingweight.xmax);
//	Xsamples = (1.0/dim) * Xsamples;
//
//	for(int i=0; i<N;i++){
//
//		rowvec x = Xsamples.row(i);
//
//		printVector(x,"x");
//
//		double ftilde = modelWingweight.interpolate(x);
//
//		cout<<"ftilde = "<<ftilde<<" fexact = "<<ysamples(i)<<"\n";
//
//
//	}
//
//
//	XsamplesValidation = normalizeMatrix(XsamplesValidation, modelWingweight.xmin, modelWingweight.xmax);
//	XsamplesValidation = (1.0/dim) * XsamplesValidation;
//
//
//
//	for(int i=0; i<NValidation;i++){
//
//		rowvec x = XsamplesValidation.row(i);
//
//		printVector(x,"x");
//
//		double ftilde = modelWingweight.interpolate(x);
//
//		cout<<"ftilde = "<<ftilde<<" fexact = "<<ysamplesValidation(i)<<"\n";
//
//
//	}
//
//
//	double inSampleError = modelWingweight.calculateInSampleError();
//#if 1
//	printf("inSampleError = %15.10f\n",inSampleError);
//#endif
//
//
//}
//
//
//
//
//
//
//
//void testGEKwithEggholder(void){
//
//	cout<<__func__<<":";
//	int dim = 2;
//	int N= 20;
//
//	TestFunction func("Eggholder",dim);
//	func.func_ptr = Eggholder;
//	func.adj_ptr = EggholderAdj;
//	func.setBoxConstraints(0,200.0);
//	func.validateAdjoints();
//	func.plot(100);
//	func.testSurrogateModel(GRADIENT_ENHANCED_KRIGING,N);
//
//
//}
//
//
//void testGEKwithWaves2D(void){
//
//	cout<<__func__<<":";
//	int dim = 2;
//	int N= 100;
//
//	TestFunction func("Waves2D",dim);
//	func.func_ptr = Waves2D;
//	func.adj_ptr = Waves2DAdj;
//	func.setBoxConstraints(0,10.0);
//	func.validateAdjoints();
//	func.plot(100);
//	func.testSurrogateModel(GRADIENT_ENHANCED_KRIGING,N);
//
//
//}


