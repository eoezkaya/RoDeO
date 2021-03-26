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

#include "metric.hpp"
#include "auxiliary_functions.hpp"

double calculateL1norm(const rowvec &x){

	double sum = 0.0;
	for(unsigned int i=0; i<x.size(); i++){

		sum += fabs(x(i));

	}

	return sum;
}

double calculateWeightedL1norm(const rowvec &x, vec w){

	double sum = 0.0;
	for(unsigned int i=0; i<x.size(); i++){

		sum += w(i)*fabs(x(i));

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

unsigned int findNearestNeighborL1(const rowvec &xp, const mat &X){

	assert(X.n_rows>0);

	unsigned int index = -1;
	double minL1Distance = LARGE;



	for(unsigned int i=0; i<X.n_rows; i++){

		rowvec x = X.row(i);

		rowvec xdiff = xp -x;

		double L1distance = calculateL1norm(xdiff);
		if(L1distance< minL1Distance){

			minL1Distance = L1distance;
			index = i;

		}

	}


	return index;


}


