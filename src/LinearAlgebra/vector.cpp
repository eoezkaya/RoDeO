/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2024 Chair for Scientific Computing (SciComp), RPTU
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
 * General Public License along with RoDeO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */

#include "./INCLUDE/vector.hpp"
#include <iostream>
#include <cstdlib> // For rand() function
#include <ctime>   // For seeding srand()
#include <cassert>
using namespace rodeo;

// Constructors and Destructor
vec::vec(int size) : size(size) {
    elements = new double[size];
}

vec::vec(const vec& other) : size(other.size) {
    elements = new double[size];
    for (int i = 0; i < size; ++i) {
        elements[i] = other.elements[i];
    }
}

vec::~vec() {
    delete[] elements;
}

// Assignment operator
vec& vec::operator=(const vec& other) {
    if (this != &other) {
        delete[] elements;
        size = other.size;
        elements = new double[size];
        for (int i = 0; i < size; ++i) {
            elements[i] = other.elements[i];
        }
    }
    return *this;
}

// Access operator
double& vec::operator()(int index) {
    return elements[index];
}

const double& vec::operator()(int index) const {
    return elements[index];
}

// vec addition
vec vec::operator+(const vec& other) const {
    vec result(size);
    for (int i = 0; i < size; ++i) {
        result(i) = elements[i] + other(i);
    }
    return result;
}

// vec subtraction
vec vec::operator-(const vec& other) const {
    vec result(size);
    for (int i = 0; i < size; ++i) {
        result(i) = elements[i] - other(i);
    }
    return result;
}

// vec scalar multiplication
vec vec::operator*(double scalar) const {
    vec result(size);
    for (int i = 0; i < size; ++i) {
        result(i) = elements[i] * scalar;
    }
    return result;
}


// Display vec
void vec::display() const {
    for (int i = 0; i < size; ++i) {
        std::cout << elements[i] << " ";
    }
    std::cout << std::endl;
}
void vec::print(void) const {

	display();
}

int vec::getSize(void) const{
	return size;

}

void vec::fill(double val){
	for (int i = 0; i < size; ++i) {
		elements[i] = val;
	}

}

// Resize method to change the size of the vector
void vec::resize(int newSize) {
    double* newElements = new double[newSize];

    // Copy elements from the existing vector to the new vector
    int minSize = std::min(size, newSize);
    for (int i = 0; i < minSize; ++i) {
        newElements[i] = elements[i];
    }

    // Deallocate memory for the existing vector
    delete[] elements;

    // Update the Vector object with the new elements and size
    elements = newElements;
    size = newSize;
}


