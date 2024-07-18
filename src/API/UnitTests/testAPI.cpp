#include "../INCLUDE/rodeo.h"
#include <iostream>
#include <math.h>


double Himmelblau(const double *x){

	double c1 = 1.0;
	double c2 = 1.0;
	double c3 = 11.0;
	double c4 = 1.0;
	double c5 = 1.0;
	double c6 = 7.0;
	return pow( (c1*x[0]*x[0]+c2*x[1]- c3 ), 2.0 ) + pow( (c4 * x[0]+ c5 * x[1]*x[1]- c6), 2.0 );


}

double Rosenbrock(const double *x){

	return (1.0-x[0])* (1.0-x[0]) + 100.0 *(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);

}

double Constraint1(const double *x){

	return x[1] - x[0]*x[0];

}

double Constraint2(const double *x){

	return pow((x[0]-1),3) - x[1] + 0.7;

}



int main(void){

	std::cout<<"##################### ROSENBROCK CONSTRAINED OPTIMIZATION BEGIN #####################\n";

	RobustDesignOptimizer testOptimizer;

	double lowerBounds[2] = {-1.5,-0.5};
	double upperBounds[2] = { 1.5, 2.5};

	testOptimizer.setName("testMinimization");
	testOptimizer.setCurrentWorkingDirectory("./test_Rosenbrock");
	testOptimizer.setDoEStrategy("random");
	testOptimizer.setDimension(2);
	testOptimizer.setObjectiveFunction(Rosenbrock, "Rosenbrock", "rosenbrock.csv");
	testOptimizer.addConstraint(Constraint1,"Constraint1 > 0", "constraint1.csv");
	testOptimizer.addConstraint(Constraint1,"Constraint2 > 0", "constraint2.csv");


	testOptimizer.setBoxConstraints(lowerBounds,upperBounds);
	testOptimizer.setDoEOn(50);
	testOptimizer.setMaxNumberOfFunctionEvaluations(50);


	testOptimizer.run();
	std::cout<<"##################### ROSENBROCK CONSTRAINED OPTIMIZATION END #####################\n";
	return 0;
}
