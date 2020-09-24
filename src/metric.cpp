#include "metric.hpp"
#include "auxilliary_functions.hpp"

double calculateL1norm(const rowvec &x){

	double sum = 0.0;
	for(unsigned int i=0; i<x.size(); i++){

		sum += fabs(x(i));

	}

	return sum;
}

double calculateMetric(rowvec &xi,rowvec &xj, mat M){

	rowvec diff= xi-xj;
	colvec diffT= trans(diff);

	return dot(diff,M*diffT);

}


double calculateMetricAdjoint(rowvec xi, rowvec xj, mat M, mat &Mb, double calculateMetricb) {

	int dim = xi.size();
	rowvec diff(dim);


	rowvec tempb(dim, fill::zeros);
	double calculateMetric;

	diff = xi-xj;
	colvec diffT= trans(diff);

	calculateMetric = dot(diff,M*diffT);
	double sumb = 0.0;
	sumb = calculateMetricb;
	tempb = sumb*diff;

	for (int i = dim-1; i > -1; --i) {
		double sumb = 0.0;
		sumb = tempb[i];
		tempb[i] = 0.0;
		for (int j = dim-1; j > -1; --j){

			Mb(i,j) += diff[j]*sumb;
		}
	}

	return calculateMetric;
}
