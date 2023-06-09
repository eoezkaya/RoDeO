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
#ifndef VECTOR_MANIP_HPP
#define VECTOR_MANIP_HPP

#include <armadillo>
#include <cassert>
using namespace arma;

template <typename T>
T normalizeVector(T x, vec xmin, vec xmax){

	unsigned int dim = x.size();

	assert(dim >0);
	assert(dim == xmin.size());
	assert(dim == xmax.size());

	T xnorm(dim);

	for(unsigned int i=0; i<dim; i++){

		assert(xmax(i) > xmin(i));
		xnorm(i) = (1.0/dim)*(x(i) - xmin(i)) / (xmax(i) - xmin(i));

	}

	return xnorm;

}

template <typename T>
T normalizeVectorBack(T xnorm, vec xmin, vec xmax){

	unsigned int dim = xnorm.size();
	assert(dim >0);
	assert(dim == xmin.size());
	assert(dim == xmax.size());

	T xp(dim);

	for(unsigned int i=0; i<dim; i++){

		assert(xmax(i) > xmin(i));
		xp(i) = xnorm(i)*dim * (xmax(i) - xmin(i)) + xmin(i);

	}
	return xp;
}



template <typename T>
T makeUnitVector(T x){

	assert(x.size() > 0);

	double normX = norm(x,2);
	T result = x/normX;
	return result;
}



template <typename T>
void addOneElement(T &v, double val){

	unsigned int size = v.size();
	assert(size>0);

	unsigned int newsize = size+1;

	v.resize(newsize);
	v(newsize-1) = val;

}


template <typename T>
void copyVector(T &a, T b){
	assert(a.size() >= b.size());
	for(unsigned int i=0; i<b.size(); i++){
		a(i) = b(i);
	}
}


template <typename T>
void copyVector(T &a, T b, unsigned int indx){
	assert(a.size() >= b.size() + indx);
	for(unsigned int i=indx; i<b.size() + indx; i++){
		a(i) = b(i-indx);
	}
}


template <typename T>
T joinVectors(const T& v1, const T& v2){

	assert(v1.size()>0);
	assert(v2.size()>0);
	unsigned int size = v1.size() + v2.size();

	T result(size);


	for(unsigned int i=0; i<size;i++){

		if(i<v1.size()){
			result(i) = v1(i);
		}
		else{
			result(i) = v2(i-v1.size());
		}

	}

	return result;

}


#endif
