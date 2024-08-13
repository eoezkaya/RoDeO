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

/* Himmelblau test function
 *
 * f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2
 * four local minima :
 * f(3.0,2.0)=0.0
 * f(2.805118, 3.131312) = 0.0
 * f(-3.779310, -3.283186) = 0.0
 * f(3.584428, -1.848126)  = 0.0
 *
 *
 * */

double Himmelblau(double *x){

	double c1 = 1.0;
	double c2 = 1.0;
	double c3 = 11.0;
	double c4 = 1.0;
	double c5 = 1.0;
	double c6 = 7.0;
	return pow( (c1*x[0]*x[0]+c2*x[1]- c3 ), 2.0 ) + pow( (c4 * x[0]+ c5 * x[1]*x[1]- c6), 2.0 );


}


double HimmelblauTangent(double *x, double *xd, double *fdot) {
	double c1 = 1.0;
	double c2 = 1.0;
	double c3 = 11.0;
	double c4 = 1.0;
	double c5 = 1.0;
	double c6 = 7.0;
	double f;
	double fd;
	double arg1;
	double arg1d;
	double arg2;
	double arg2d;
	arg1d = c1*2*x[0]*xd[0] + c2*xd[1];
	arg1 = c1*x[0]*x[0] + c2*x[1] - c3;
	arg2d = c4*xd[0] + c5*2*x[1]*xd[1];
	arg2 = c4*x[0] + c5*x[1]*x[1] - c6;
	fd = 2.0*pow(arg1, 2.0-1)*arg1d + 2.0*pow(arg2, 2.0-1)*arg2d;
	f = pow(arg1, 2.0) + pow(arg2, 2.0);
	*fdot = fd;
	return f;
}


double HimmelblauAdj(double *x, double *xb) {
	double c1 = 1.0;
	double c2 = 1.0;
	double c3 = 11.0;
	double c4 = 1.0;
	double c5 = 1.0;
	double c6 = 7.0;


	double tempb;
	double tempb0;

	tempb = 2.0*pow(c1*(x[0]*x[0])-c3+c2*x[1], 2.0-1);
	tempb0 = 2.0*pow(c4*x[0]-c6+c5*(x[1]*x[1]), 2.0-1);
	xb[0] =  c4*tempb0 + 2*x[0]*c1*tempb;
	xb[1] =  2*x[1]*c5*tempb0 + c2*tempb;
	return Himmelblau(x);

}

double HimmelblauAdjLowFi(double *x, double *xb) {
	double c1 = 1.1;
	double c2 = 0.9;
	double c3 = 11.05;
	double c4 = 1.05;
	double c5 = 1.08;
	double c6 = 6.97;

	double fb = 0.0;
	double tempb;
	double tempb0;

	fb = 1.0;
	tempb = 2.0*pow(c1*(x[0]*x[0])-c3+c2*x[1], 2.0-1)*fb;
	tempb0 = 2.0*pow(c4*x[0]-c6+c5*(x[1]*x[1]), 2.0-1)*fb;
	xb[0] = xb[0] + c4*tempb0 + 2*x[0]*c1*tempb;
	xb[1] = xb[1] + 2*x[1]*c5*tempb0 + c2*tempb;
	return pow( (c1*x[0]*x[0]+c2*x[1]- c3 ), 2.0 ) + pow( (c4 * x[0]+ c5 * x[1]*x[1]- c6), 2.0 );

}


double HimmelblauLowFi(double *x){

	double c1 = 1.1;
	double c2 = 0.9;
	double c3 = 11.05;
	double c4 = 1.05;
	double c5 = 1.08;
	double c6 = 6.97;
	return pow( (c1*x[0]*x[0]+c2*x[1]- c3 ), 2.0 ) + pow( (c4 * x[0]+ c5 * x[1]*x[1]- c6), 2.0 );


}

double HimmelblauTangentLowFi(double *x, double *xd, double *fdot) {
	double c1 = 1.1;
	double c2 = 0.9;
	double c3 = 11.05;
	double c4 = 1.05;
	double c5 = 1.08;
	double c6 = 6.97;
	double f;
	double fd;
	double arg1;
	double arg1d;
	double arg2;
	double arg2d;
	arg1d = c1*2*x[0]*xd[0] + c2*xd[1];
	arg1 = c1*x[0]*x[0] + c2*x[1] - c3;
	arg2d = c4*xd[0] + c5*2*x[1]*xd[1];
	arg2 = c4*x[0] + c5*x[1]*x[1] - c6;
	fd = 2.0*pow(arg1, 2.0-1)*arg1d + 2.0*pow(arg2, 2.0-1)*arg2d;
	f = pow(arg1, 2.0) + pow(arg2, 2.0);
	*fdot = fd;
	return f;
}

/***********************************************************************************/
