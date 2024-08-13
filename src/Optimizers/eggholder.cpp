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
 * General Public License along with RoDEO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */

#include<stack>
#include<math.h>



double Eggholder(double *x){
	return -(x[1]+47.0)*sin(sqrt(fabs(x[1]+0.5*x[0]+47.0)))-x[0]*sin(sqrt(fabs(x[0]-(x[1]+47.0) )));

}


double EggholderAdj(double *x, double *xb) {
	double fabs0;
	double fabs0b;
	double fabs1;
	double fabs1b;
	double temp;
	double temp0;
	int branch;

	xb[0] = 0.0;
	xb[1] = 0.0;

	std::stack<int> intStack;


	if (x[1] + 0.5*x[0] + 47.0 >= 0.0) {
		fabs0 = x[1] + 0.5*x[0] + 47.0;
		intStack.push(1);
	} else {
		fabs0 = -(x[1]+0.5*x[0]+47.0);
		intStack.push(0);
	}
	if (x[0] - (x[1] + 47.0) >= 0.0) {
		fabs1 = x[0] - (x[1] + 47.0);
		intStack.push(0);
	} else {
		fabs1 = -(x[0]-(x[1]+47.0));
		intStack.push(1);
	}
	temp = sqrt(fabs0);
	temp0 = sqrt(fabs1);
	xb[1] = xb[1] - sin(temp);
	fabs0b = (fabs0 == 0.0 ? 0.0 : -(cos(temp)*(x[1]+47.0)/(2.0*
			temp)));
	xb[0] = xb[0] - sin(temp0);
	fabs1b = (fabs1 == 0.0 ? 0.0 : -(cos(temp0)*x[0]/(2.0*temp0)));

	branch = intStack.top();
	intStack.pop();

	if (branch == 0) {
		xb[0] = xb[0] + fabs1b;
		xb[1] = xb[1] - fabs1b;
	} else {
		xb[1] = xb[1] + fabs1b;
		xb[0] = xb[0] - fabs1b;
	}
	branch = intStack.top();
	intStack.pop();

	if (branch == 0) {
		xb[0] = xb[0] - 0.5*fabs0b;
		xb[1] = xb[1] - fabs0b;
	} else {
		xb[1] = xb[1] + fabs0b;
		xb[0] = xb[0] + 0.5*fabs0b;
	}

	return -(x[1]+47.0)*sin(sqrt(fabs(x[1]+0.5*x[0]+47.0)))-x[0]*sin(sqrt(fabs(x[0]-(x[1]+47.0) )));
}
