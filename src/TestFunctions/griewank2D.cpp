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

#include<math.h>
#include<stack>

double griewank2D(double *x){

	double sum = 0.0;
	double prod = 1.0;

	for(int i=0; i<2; i++){

		double xi = x[i];
		sum = sum + (xi*xi)/4000.0;
		prod = prod * cos(xi/sqrt(i+1));
	}

	return sum - prod + 1;

}

double griewank2DTangent(double *x, double *xd, double *fdot) {
	double sum = 0.0;
	double sumd = 0.0;
	double prod = 1.0;
	double prodd = 0.0;

	for (int i = 0; i < 2; ++i) {
		double xi = x[i];
		double xid = xd[i];

		double temp;
		sumd = sumd + 2*xi*xid/4000.0;
		sum = sum + xi*xi/4000.0;
		double sqrti = sqrt(i+1);
		temp = cos(xi/sqrti);
		prodd = temp*prodd - prod*sin(xi/sqrti)*xid/sqrti;
		prod = prod*temp;
	}
	*fdot = sumd - prodd;

	return sum - prod + 1;

}

double griewank2DAdjoint(double *x, double *xb) {

    double sumb = 0.0;
    double prod = 1.0;
    double prodb = 0.0;

    std::stack<double> tape;

    for (int i = 0; i < 2; ++i) {
        double xi = x[i];

        tape.push(prod);
        prod = prod*cos(xi/sqrt(i+1));
        tape.push(xi);
    }
    sumb = 1.0;
    prodb = -1.0;
    for (int i = 1; i > -1; --i) {
        double xi;
        double xib = 0.0;
        double temp;

        xi = tape.top();
        tape.pop();
        prod = tape.top();
        tape.pop();
        temp = sqrt(i+1);
        xib = 2*xi*sumb/4000.0 - sin(xi/temp)*prod*prodb/temp;
        prodb = cos(xi/temp)*prodb;
        xb[i] = xb[i] + xib;
    }

    return griewank2D(x);
}


