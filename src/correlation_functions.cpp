/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2021 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, TU Kaiserslautern)
 *
 *  file is part of RoDeO
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

#include <stdio.h>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <cassert>

#include "correlation_functions.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;


Correlationfunction::Correlationfunction(){}


mat Correlationfunction::corrbiquadspline_gekriging(mat &X, vec theta){

	unsigned int dim = size(X,1);
	unsigned int m   = size(X,0);

	//data.normalizeSampleInputMatrix;

	// mat x = data.getInputMatrix();

	mat correlationMatrix = zeros(m*(dim+1),m*(dim+1));

	unsigned int mzmax = m*(m-1)/2;                /*  number of non-zero distances  */
	mat ij = zeros(mzmax, 2);                      /* initialize matrix with indices */
	mat d  = zeros(mzmax, dim);                    /* initialize matrix with distances */

	vec ll = zeros(m);
	vec ll1 = zeros(m);

	for(unsigned int k=1; k<m;k++){

		ll(k)= m-k;
		ij.submat(sum(ll1),0,sum(ll)-1,1)   = join_rows((k-1)*ones<vec>(m-k),linspace(k,m-1,m-k));
		d.submat(sum(ll1),0,sum(ll)-1,dim-1)  = repmat(X.row(k-1),m-k,1)-X.rows(k,m-1);                    /*  differences between points */
     	ll1(k)= m-k;

    }

  // Kriging correlation function

	unsigned int  mn = mzmax*dim;
	vec ss = zeros(mn);
	vec xi = reshape(abs(d) % repmat(theta.t(),mzmax,1),mn,1);

	uvec i1 = find(xi <= 0.4);
	uvec i2 = find(0.4 < xi && xi < 1);

	if  (!i1.is_empty())
	    ss(i1) = 1 - 15*pow(xi(i1),2)+ 35*pow(xi(i1),3) - 195.0/8*pow(xi(i1),4);


	if  (!i2.is_empty())
		ss(i2) = 5.0/3 - 20.0/3*xi(i2) + 10.0*pow(xi(i2),2)- 20.0/3*pow(xi(i2),3) + 5.0/3*pow(xi(i2),4);

	mat ss1 = reshape(ss,mzmax,dim);
    vec r = prod(ss1,1);

    uvec idx = find(r > 0);

    vec o = linspace(0,m-1,m);
    double epsilonKriging = 2.220446049250313e-16;
    double mu = (10+m)*epsilonKriging;

    mat location     = join_cols(ij.rows(idx),repmat(o,1,2));
	vec correlation  = join_cols(r(idx),ones<vec>(m)+mu);

	sp_mat R(conv_to<umat>::from(location.t()),correlation.t());

	correlationMatrix.submat(0,0,m-1,m-1) = R;


	// First order partial derivative

	vec u = reshape(sign(d) % repmat(theta.t(),mzmax,1), mn,1);

	vec dr = zeros(mn);

    if  (!i1.is_empty())
		dr(i1) = -u(i1)%(-30*xi(i1) + 105*pow(xi(i1),2) - 195.0/2*pow(xi(i1),3));

	if  (!i2.is_empty())
		dr(i2) = -u(i2)%(-20.0/3 + 20*xi(i2)- 20*pow(xi(i2),2) + 20.0/3*pow(xi(i2),3));

	uvec ii = conv_to<uvec>::from(linspace(0,mzmax-1,mzmax));
	vec dr1 = dr;

	for(unsigned int j=0; j<dim;j++){
	   vec sj = ss1.submat(0,j,size(ss1,0)-1,j);
	   ss1.submat(0,j,size(ss1,0)-1,j) = dr1(ii);
	   dr1(ii) = prod(ss1,1);
	   ss1.submat(0,j,size(ss1,0)-1,j) = sj;
	   ii = ii + mzmax;
   }

	 mat dr2 = reshape(dr1,mzmax,dim);
	 mat location1 = join_cols(ij,repmat(o,1,2));

	for (unsigned int i=0;i<dim; i++){
		vec correlation  = join_cols(dr2.submat(0,i,mzmax-1,i),zeros<vec>(m));
		sp_mat SR(conv_to<umat>::from(location1.t()),correlation.t());
		correlationMatrix.submat(0,(i+1)*m,m-1,(i+2)*m-1) = SR - SR.t();
	}

	// Second order partial derivative

	for (unsigned int i=0; i<dim; i++){

	     vec ddr = zeros(mn);

	     if  (!i1.is_empty())
	          ddr(i1) = -(-30 + 210*xi(i1) - 585.0/2*pow(xi(i1),2)) * pow(theta(i),2);

	     if  (!i2.is_empty())
	          ddr(i2) = -(20- 40*xi(i2) + 20.0*pow(xi(i2),2)) * pow(theta(i),2);

	     uvec ii = conv_to<uvec>::from(linspace(0,mzmax-1,mzmax));

	     for (unsigned int k=0; k<dim; k++){

	         vec sj = ss1.submat(0,k,size(ss1,0)-1,k);
	         ss1.submat(0,k,size(ss1,0)-1,k) = ddr(ii);
	         ddr(ii) = prod(ss1,1);
	         ss1.submat(0,k,size(ss1,0)-1,k) = sj;
	         ii = ii + mzmax;

	      }

	      mat ddr1 = reshape(ddr,mzmax,dim);
	      vec correlation  = join_cols(ddr1.submat(0,i,mzmax-1,i),30*pow(theta(i),2)*(ones<vec>(m)+mu));
	      sp_mat SR1(conv_to<umat>::from(location1.t()),correlation.t());
	      correlationMatrix.submat((i+1)*m,(i+1)*m,(i+2)*m-1,(i+2)*m-1) = SR1;

	     }

		for (unsigned int i=0; i<dim; i++){
			for (unsigned int j=i+1; j<dim; j++){

				vec sj = ss1.submat(0,j,size(ss1,0)-1,j);
				vec si = ss1.submat(0,i,size(ss1,0)-1,i);
				ss1.submat(0,j,size(ss1,0)-1,j) = dr.rows(j*mzmax,(j+1)*mzmax-1);
				ss1.submat(0,i,size(ss1,0)-1,i) = -dr.rows(i*mzmax,(i+1)*mzmax-1);
				vec dr = prod(ss1,1);
				ss1.submat(0,j,size(ss1,0)-1,j) = sj;
				ss1.submat(0,i,size(ss1,0)-1,i) = si;
				vec correlation = join_cols(dr,zeros<vec>(m));
				sp_mat SR(conv_to<umat>::from(location1.t()),correlation.t());
			    correlationMatrix.submat((i+1)*m,(j+1)*m,(i+2)*m-1,(j+2)*m-1) = SR + SR.t()-diagmat(SR);
			}
		}

     correlationMatrix = correlationMatrix + correlationMatrix.t()-diagmat(correlationMatrix);
     return correlationMatrix;
 }

void Correlationfunction::corrbiquadspline_gekriging_vec(mat &xtest, mat &X, vec theta){

	unsigned int dim = size(X,1);
	unsigned int m   = size(X,0);
	unsigned int m1  = size(xtest,0);

	// mat x = data.getInputMatrix();

	mat correlationVec = zeros(m1,m*(dim+1));

	unsigned int mzmax = m*m1;                     /*  number of non-zero distances  */
	mat d  = zeros(mzmax, dim);                    /* initialize matrix with distances */

	for(unsigned int k=1; k<m1;k++){
		d.submat((k-1)*m,0,m*k-1,dim-1)  = repmat(xtest.row(k-1),m,1)-X;                    /*  differences between points */
    }

  // Kriging correlation function

	unsigned int  mn = m1*dim;
	vec ss = zeros(mn);
	vec xi = reshape(abs(d) % repmat(theta.t(),mzmax,1),mn,1);

	uvec i1 = find(xi <= 0.4);
	uvec i2 = find(0.4 < xi && xi < 1);

	if  (!i1.is_empty())
	    ss(i1) = 1 - 15*pow(xi(i1),2)+ 35*pow(xi(i1),3) - 195.0/8*pow(xi(i1),4);


	if  (!i2.is_empty())
		ss(i2) = 5.0/3 - 20.0/3*xi(i2) + 10.0*pow(xi(i2),2)- 20.0/3*pow(xi(i2),3) + 5.0/3*pow(xi(i2),4);

	mat ss1 = reshape(ss,mzmax,dim);
    vec r = prod(ss1,1);
    mat r_g = reshape(r,m,m1);

    correlationVec.submat(0,0,m1-1,m-1) = r_g.t();


	// First order partial derivative

	vec u = reshape(sign(d) % repmat(theta.t(),mzmax,1), mn,1);

	vec dr = zeros(mn);

    if  (!i1.is_empty())
		dr(i1) = -u(i1)%(-30*xi(i1) + 105*pow(xi(i1),2) - 195.0/2*pow(xi(i1),3));

	if  (!i2.is_empty())
		dr(i2) = -u(i2)%(-20.0/3 + 20*xi(i2)- 20*pow(xi(i2),2) + 20.0/3*pow(xi(i2),3));

	uvec ii = conv_to<uvec>::from(linspace(0,mzmax-1,mzmax));
	vec dr1 = dr;

	for(unsigned int j=0; j<dim;j++){
	   vec sj = ss1.submat(0,j,size(ss1,0)-1,j);
	   ss1.submat(0,j,size(ss1,0)-1,j) = dr1(ii);
	   dr1(ii) = prod(ss1,1);
	   ss1.submat(0,j,size(ss1,0)-1,j) = sj;
	   ii = ii + mzmax;
   }

	 mat dr2 = reshape(dr1,mzmax,dim);
	 mat r_g1;


	for (unsigned int i=0;i<dim; i++){
		r_g1 = reshape(dr2.submat(0,i,mzmax-1,i),m,m1);
		correlationVec.submat(0,(i+1)*m,m1-1,(i+2)*m-1) = r_g1.t();
	}


 }


void Correlationfunction::corrgaussian_gekriging(mat &X, vec theta){


	unsigned int dim = size(X,1);
	unsigned int m   = size(X,0);

	mat correlationMatrix = zeros(m*(dim+1),m*(dim+1));

	unsigned int mzmax = m*(m-1)/2;                /*  number of non-zero distances  */
	mat ij = zeros(mzmax, 2);                      /* initialize matrix with indices */
	mat d  = zeros(mzmax, dim);                    /* initialize matrix with distances */

	vec ll = zeros(m);
	vec ll1 = zeros(m);

	for(unsigned int k=1; k<m;k++){

		ll(k)= m-k;
		ij.submat(sum(ll1),0,sum(ll)-1,1)   = join_rows((k-1)*ones<vec>(m-k),linspace(k,m-1,m-k));
		d.submat(sum(ll1),0,sum(ll)-1,dim-1)  = repmat(X.row(k-1),m-k,1)-X.rows(k,m-1);                    /*  differences between points */
		ll1(k)= m-k;

    }

	// Kriging correlation function

	unsigned int  mn = mzmax*dim;
	double epsilonKriging = 2.220446049250313e-16;
	double mu = (10+m)*epsilonKriging;

	mat td = d % d % repmat(-theta.t(),mzmax,1);
	vec r = exp(sum(td,1));

	uvec idx = find(r > 0);
	vec o = linspace(0,m-1,m);

	mat location     = join_cols(ij.rows(idx),repmat(o,1,2));
	vec correlation  = join_cols(r(idx),ones<vec>(m)+mu);

	sp_mat R(conv_to<umat>::from(location.t()),correlation.t());

	correlationMatrix.submat(0,0,m-1,m-1) = R;


	// First order partial derivative

    for (unsigned int i=0;i<dim; i++){
	     vec correlation  = join_cols(2*theta(i)*d.submat(0,i,mzmax-1,i) % r(idx),zeros<vec>(m));
		 sp_mat SR(conv_to<umat>::from(location.t()),correlation.t());
	     correlationMatrix.submat(0,(i+1)*m,m-1,(i+2)*m-1) = SR - SR.t();
    }


	// Second order partial derivative

    for (unsigned int i=0; i<dim; i++){

    	 vec correlation  = join_cols(2*theta(i)*(-2*theta(i)*d.submat(0,i,mzmax-1,i) % d.submat(0,i,mzmax-1,i)+1) % r(idx), 2*theta(i)*(ones<vec>(m)+mu));
    	 sp_mat SR1(conv_to<umat>::from(location.t()),correlation.t());
    	 correlationMatrix.submat((i+1)*m,(i+1)*m,(i+2)*m-1,(i+2)*m-1) = SR1;

        }

    for (unsigned int i=0; i<dim; i++){
    	for (unsigned int j=i+1; j<dim; j++){

    	    vec correlation  = join_cols(-4*theta(i)*theta(j) * d.submat(0,i,mzmax-1,i) % d.submat(0,j,mzmax-1,j) % r(idx), zeros<vec>(m));
    		sp_mat SR(conv_to<umat>::from(location.t()),correlation.t());
    		correlationMatrix.submat((i+1)*m,(j+1)*m,(i+2)*m-1,(j+2)*m-1) = SR + SR.t()-diagmat(SR);

    	}
    }

	 correlationMatrix = correlationMatrix + correlationMatrix.t()-diagmat(correlationMatrix);

 }


void Correlationfunction::corrgaussian_gekriging_vec(mat &xtest,mat &X, vec theta){


	unsigned int dim = size(X,1);
	unsigned int m   = size(X,0);
	unsigned int m1  = size(xtest,0);

	mat correlationVec = zeros(m1,m*(dim+1));

    unsigned int mzmax = m*m1;                     /*  number of non-zero distances  */
    mat d  = zeros(mzmax, dim);                    /* initialize matrix with distances */

	for(unsigned int k=1; k<m1;k++){
	   d.submat((k-1)*m,0,m*k-1,dim-1)  = repmat(xtest.row(k-1),m,1)-X;                    /*  differences between points */
	 }

	// Kriging correlation function

	unsigned int  mn = mzmax*dim;

	mat td = d % d % repmat(-theta.t(),mzmax,1);
	vec r = exp(sum(td,1));
    mat r_g = reshape(r,m,m1);
	correlationVec.submat(0,0,m1-1,m-1) = r_g.t();

  // First order partial derivative

	mat dist;

    for (unsigned int i=0;i<dim; i++){
    	 dist = reshape(d.submat(0,i,mzmax-1,i),m,m1);
	     correlationVec.submat(0,(i+1)*m,m1-1,(i+2)*m-1) = 2*theta(i)*dist.t() % r_g;
    }
 }

void Correlationfunction::corrgaussian_kriging(mat &X, vec theta){

	unsigned int dim = size(X,1);
	unsigned int m   = size(X,0);

	mat correlationMatrix = zeros(m,m);

	unsigned int mzmax = m*(m-1)/2;                /*  number of non-zero distances  */
	mat ij = zeros(mzmax, 2);                      /* initialize matrix with indices */
	mat d  = zeros(mzmax, dim);                    /* initialize matrix with distances */

	vec ll = zeros(m);
	vec ll1 = zeros(m);

	for(unsigned int k=1; k<m;k++){

		ll(k)= m-k;
	    ij.submat(sum(ll1),0,sum(ll)-1,1)   = join_rows((k-1)*ones<vec>(m-k),linspace(k,m-1,m-k));
	    d.submat(sum(ll1),0,sum(ll)-1,dim-1)  = repmat(X.row(k-1),m-k,1)-X.rows(k,m-1);                    /*  differences between points */
		ll1(k)= m-k;

    }

	// Kriging correlation function

	unsigned int  mn = mzmax*dim;
	double epsilonKriging = 2.220446049250313e-16;
	double mu = (10+m)*epsilonKriging;

	mat td = d % d % repmat(-theta.t(),mzmax,1);  //
	vec r = exp(sum(td,1));

	uvec idx = find(r > 0);
	vec o = linspace(0,m-1,m);

	mat location  = join_cols(ij.rows(idx),repmat(o,1,2));

	cout << size(location) << endl;
	vec correlation  = join_cols(r(idx),ones<vec>(m)+mu);
	cout << size(correlation) << endl;

	sp_mat R(conv_to<umat>::from(location.t()),correlation.t());

	correlationMatrix = R + R.t()-diagmat(R);

 }

void Correlationfunction::corrgaussian_kriging_vec(mat &xtest, mat &X, vec theta){

	unsigned int dim = size(X,1);
	unsigned int m   = size(X,0);
	unsigned int m1  = size(xtest,0);

	mat correlationVec = zeros(m1,m);

	unsigned int mzmax = m*m1;                     /*  number of non-zero distances  */
	mat d  = zeros(mzmax, dim);                    /* initialize matrix with distances */

	for(unsigned int k=1; k<m1;k++){
		  d.submat((k-1)*m,0,m*k-1,dim-1)  = repmat(xtest.row(k-1),m,1)-X;                    /*  differences between points */
	 }

	// Kriging correlation function

	unsigned int  mn = mzmax*dim;

	mat td = d % d % repmat(-theta.t(),mzmax,1);
	vec r = exp(sum(td,1));
	mat r_g = reshape(r,m,m1);
	correlationVec.submat(0,0,m1-1,m-1) = r_g.t();

 }

void Correlationfunction::corrbiquadspline_kriging(mat &X,vec theta){


	unsigned int dim = size(X,1);
	unsigned int m   = size(X,0);

	mat correlationMatrix = zeros(m,m);

	unsigned int mzmax = m*(m-1)/2;                /*  number of non-zero distances  */
	mat ij = zeros(mzmax, 2);                      /* initialize matrix with indices */
	mat d  = zeros(mzmax, dim);                    /* initialize matrix with distances */

	vec ll = zeros(m);
	vec ll1 = zeros(m);

	for(unsigned int k=1; k<m;k++){

		ll(k)= m-k;
		ij.submat(sum(ll1),0,sum(ll)-1,1)   = join_rows((k-1)*ones<vec>(m-k),linspace(k,m-1,m-k));
		d.submat(sum(ll1),0,sum(ll)-1,dim-1)  = repmat(X.row(k-1),m-k,1)-X.rows(k,m-1);                    /*  differences between points */
		ll1(k)= m-k;

    }

  // Kriging correlation function

	unsigned int  mn = mzmax*dim;
	vec ss = zeros(mn);
	vec xi = reshape(abs(d) % repmat(theta.t(),mzmax,1),mn,1);

	uvec i1 = find(xi <= 0.4);
	uvec i2 = find(0.4 < xi && xi < 1);

	if  (!i1.is_empty())
	    ss(i1) = 1 - 15*pow(xi(i1),2)+ 35*pow(xi(i1),3) - 195.0/8*pow(xi(i1),4);


	if  (!i2.is_empty())
		ss(i2) = 5.0/3 - 20.0/3*xi(i2) + 10.0*pow(xi(i2),2)- 20.0/3*pow(xi(i2),3) + 5.0/3*pow(xi(i2),4);

	mat ss1 = reshape(ss,mzmax,dim);
    vec r = prod(ss1,1);

    uvec idx = find(r > 0);

    vec o = linspace(0,m-1,m);
    double epsilonKriging = 2.220446049250313e-16;
    double mu = (10+m)*epsilonKriging;

    mat location     = join_cols(ij.rows(idx),repmat(o,1,2));
	vec correlation  = join_cols(r(idx),ones<vec>(m)+mu);

	sp_mat R(conv_to<umat>::from(location.t()),correlation.t());

	correlationMatrix = R + R.t()-diagmat(R);

 }

void Correlationfunction::corrbiquadspline_kriging_vec(mat &xtest, mat &X,vec theta){

	unsigned int dim = size(X,1);
	unsigned int m   = size(X,0);
	unsigned int m1  = size(xtest,0);

	// mat x = data.getInputMatrix();

	mat correlationVec = zeros(m1,m);

	unsigned int mzmax = m*m1;                     /*  number of non-zero distances  */
	mat d  = zeros(mzmax, dim);                    /* initialize matrix with distances */

	for(unsigned int k=1; k<m1;k++){
		d.submat((k-1)*m,0,m*k-1,dim-1)  = repmat(xtest.row(k-1),m,1)-X;                    /*  differences between points */
    }

  // Kriging correlation function

	unsigned int  mn = m1*dim;
	vec ss = zeros(mn);
	vec xi = reshape(abs(d) % repmat(theta.t(),mzmax,1),mn,1);

	uvec i1 = find(xi <= 0.4);
	uvec i2 = find(0.4 < xi && xi < 1);

	if  (!i1.is_empty())
	    ss(i1) = 1 - 15*pow(xi(i1),2)+ 35*pow(xi(i1),3) - 195.0/8*pow(xi(i1),4);


	if  (!i2.is_empty())
		ss(i2) = 5.0/3 - 20.0/3*xi(i2) + 10.0*pow(xi(i2),2)- 20.0/3*pow(xi(i2),3) + 5.0/3*pow(xi(i2),4);

	mat ss1 = reshape(ss,mzmax,dim);
    vec r = prod(ss1,1);
    mat r_g = reshape(r,m,m1);

    correlationVec.submat(0,0,m1-1,m-1) = r_g.t();

 }
