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

/*
 *  Sw: Wing Area (ft^2) (150,200)
 *  Wfw: Weight of fuel in the wing (lb) (220,300)
 *  A: Aspect ratio (6,10)
 *  Lambda: quarter chord sweep (deg) (-10,10)
 *  q: dynamic pressure at cruise (lb/ft^2)  (16,45)
 *  lambda: taper ratio (0.5,1)
 *  tc: aerofoil thickness to chord ratio (0.08,0.18)
 *  Nz: ultimate load factor (2.5,6)
 *  Wdg: flight design gross weight (lb)  (1700,2500)
 *  Wp: paint weight (lb/ft^2) (0.025, 0.08)
 *
 */
double Wingweight(double *x){


	double Sw=x[0];
	double Wfw=x[1];
	double A=x[2];
	double Lambda=x[3];
	double q=x[4];
	double lambda=x[5];
	double tc=x[6];
	double Nz=x[7];
	double Wdg=x[8];
	double Wp=x[9];


	double deg = (Lambda*3.141592654)/180.0;

	double W = 0.036*pow(Sw,0.758)*pow(Wfw,0.0035)*pow((A/(cos(deg)*cos(deg))),0.6) *
			pow(q,0.006)*pow(lambda,0.04)*pow( (100.0*tc/cos(deg)), -0.3) *pow( (Nz*Wdg),0.49) + Sw*Wp;

	return(W);
}

double WingweightAdj(double *x, double *xb) {
	double Sw = x[0];
	double Swb = 0.0;
	double Wfw = x[1];
	double Wfwb = 0.0;
	double A = x[2];
	double Ab = 0.0;
	double Lambda = x[3];
	double Lambdab = 0.0;
	double q = x[4];
	double qb = 0.0;
	double lambda = x[5];
	double lambdab = 0.0;
	double tc = x[6];
	double tcb = 0.0;
	double Nz = x[7];
	double Nzb = 0.0;
	double Wdg = x[8];
	double Wdgb = 0.0;
	double Wp = x[9];
	double Wpb = 0.0;
	double deg = Lambda*3.141592654/180.0;
	double degb = 0.0;
	double W = 0.036*pow(Sw, 0.758)*pow(Wfw, 0.0035)*pow(A/(cos(deg)*cos(deg))
			, 0.6)*pow(q, 0.006)*pow(lambda, 0.04)*pow(100.0*tc/cos(deg), -0.3)*pow(Nz
					*Wdg, 0.49) + Sw*Wp;
	double Wb = 0.0;
	double temp;
	double temp0;
	double temp1;
	double temp2;
	double temp3;
	double temp4;
	double temp5;
	double tempb;
	double temp6;
	double temp7;
	double temp8;
	double tempb0;
	double temp9;
	double temp10;
	double temp11;
	double temp12;
	double tempb1;
	double tempb2;
	double tempb3;
	double tempb4;

	Wb = 1.0;
	temp = pow(lambda, 0.04);
	temp0 = pow(q, 0.006);
	temp1 = temp0*temp;
	temp2 = cos(deg);
	temp3 = temp2*temp2;
	temp4 = A/temp3;
	temp5 = pow(temp4, 0.6);
	temp6 = cos(deg);
	temp7 = tc/temp6;
	temp8 = pow(100.0*temp7, -0.3);
	temp9 = pow(Nz*Wdg, 0.49);
	temp10 = pow(Wfw, 0.0035);
	temp11 = pow(Sw, 0.758);
	temp12 = temp11*temp10;
	tempb = temp5*temp1*0.036*Wb;
	tempb3 = temp12*temp9*temp8*0.036*Wb;
	Wpb = Sw*Wb;
	tempb4 = 0.6*pow(temp4, 0.6-1)*temp1*tempb3/temp3;
	qb = 0.006*pow(q, 0.006-1)*temp*temp5*tempb3;
	lambdab = 0.04*pow(lambda, 0.04-1)*temp0*temp5*tempb3;
	Ab = tempb4;
	tempb0 = temp8*tempb;
	Swb = Wp*Wb + 0.758*pow(Sw, 0.758-1)*temp10*temp9*tempb0;
	tempb2 = -(100.0*0.3*pow(100.0*temp7, -0.3-1)*temp12*temp9*tempb/temp6);
	degb = sin(deg)*2*temp2*temp4*tempb4 + sin(deg)*temp7*tempb2;
	tcb = tempb2;
	Wfwb = 0.0035*pow(Wfw, 0.0035-1)*temp11*temp9*tempb0;
	tempb1 = 0.49*pow(Nz*Wdg, 0.49-1)*temp12*tempb0;
	Nzb = Wdg*tempb1;
	Wdgb = Nz*tempb1;
	Lambdab = 3.141592654*degb/180.0;
	xb[9] = Wpb;
	xb[8] = Wdgb;
	xb[7] = Nzb;
	xb[6] = tcb;
	xb[5] = lambdab;
	xb[4] = qb;
	xb[3] = Lambdab;
	xb[2] = Ab;
	xb[1] = Wfwb;
	xb[0] = Swb;
	return(W);
}
