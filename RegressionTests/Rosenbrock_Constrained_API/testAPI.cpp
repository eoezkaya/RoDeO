#include "../../src/API/INCLUDE/rodeo.h"
#include <iostream>
#include <math.h>

double Rosenbrock(const double *x){

        std::cout<<"Evaluating Rosenbrock function at x[0] = "<<x[0]<<" and x[1] = "<<x[1] <<"\n";
        double f = (1.0-x[0])* (1.0-x[0]) + 100.0 *(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]); 
        std::cout<<"f(x) = "<< f <<"\n\n";
	return f;

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

	testOptimizer.name      = "testMinimization";
	testOptimizer.setDimension(2);
	testOptimizer.setObjectiveFunction(Rosenbrock, "Rosenbrock", "rosenbrock.csv");
	testOptimizer.addConstraint(Constraint1,"Constraint1 > 0", "constraint1.csv");
	testOptimizer.addConstraint(Constraint2,"Constraint2 > 0", "constraint2.csv");


	testOptimizer.setBoxConstraints(lowerBounds,upperBounds);
	testOptimizer.setDoEOn(50);
	testOptimizer.setMaxNumberOfFunctionEvaluations(50);

	testOptimizer.run();
	std::cout<<"##################### ROSENBROCK CONSTRAINED OPTIMIZATION END #####################\n";
	return 0;
}
