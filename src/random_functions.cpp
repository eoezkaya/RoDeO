#include <cassert>

#include "random_functions.hpp"


#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;

double generateRandomDouble(double a, double b) {

	double random = ((double) rand()) / (double) RAND_MAX;
	double diff = b - a;
	double r = random * diff;
	return a + r;
}


void generateRandomDoubleArray(double *xp,double a, double b, unsigned int dim) {

	for(unsigned int i=0; i<dim; i++) {

		xp[i] = generateRandomDouble(a,b);

	}


}



rowvec generateRandomRowVector(vec lb, vec ub){
	unsigned int dim = lb.size();
	rowvec x(dim);
	for(unsigned int i=0; i<dim; i++) {
		assert(lb(i) <= ub(i));
		x(i) = generateRandomDouble(lb(i), ub(i));
	}
	return x;

}

rowvec generateRandomRowVector(double lb, double ub, unsigned int dim){

	assert(lb <= ub);
	rowvec x(dim,fill::zeros);

	for(unsigned int i=0; i<dim; i++) {

		x(i) = generateRandomDouble(lb, ub);
	}
	return x;

}

vec generateRandomVector(vec lb, vec ub){
	unsigned int dim = lb.size();
	vec x(dim);
	for(unsigned int i=0; i<dim; i++) {
		assert(lb(i) <= ub(i));
		x(i) = generateRandomDouble(lb(i), ub(i));
	}
	return x;

}

vec generateRandomVector(double lb, double ub, unsigned int dim){

	assert(lb <= ub);
	vec x(dim,fill::zeros);

	for(unsigned int i=0; i<dim; i++) {

		x(i) = generateRandomDouble(lb, ub);
	}
	return x;

}




/** generate a random number between xs and xe using the normal distribution
 *
 */
double generateRandomDoubleFromNormalDist(double xs, double xe, double sigma_factor){

	double sigma=fabs((xe-xs))/sigma_factor;
	double mu=(xe+xs)/2.0;

	if (sigma == 0.0) sigma=1.0;

	/* construct a trivial random generator engine from a time-based seed */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);
	std::normal_distribution<double> distribution (mu,sigma);
	return distribution(generator);
}






