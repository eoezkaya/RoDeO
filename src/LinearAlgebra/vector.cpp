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
//#include "./INCLUDE/matrix.hpp"
#include <iostream>
#include <cstdlib> // For rand() function
#include <ctime>   // For seeding srand()
#include <cassert>
#include <math.h>
using namespace rodeo;

// Constructors and Destructor


vec::vec(int defaultSize) : size(defaultSize), elements(nullptr) {
    if (size > 0) {
        elements = new double[size];
        // Initialize elements to zero or any other default values
        for (int i = 0; i < size; ++i) {
            elements[i] = 0.0;
        }
    }
}

vec::vec(const vec& other) : size(other.size) {
	elements = new double[size];
	for (int i = 0; i < size; ++i) {
		elements[i] = other.elements[i];
	}
}

vec::~vec() {

	if(size>0) {
		delete[] elements;
	}
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


// FillRandom method to fill the vector with random values
void vec::fillRandom() {
	// Seed the random number generator
	std::srand(static_cast<unsigned>(std::time(0)));

	for (int i = 0; i < size; ++i) {
		// Generate a random double between 0.0 and 1.0
		elements[i] = static_cast<double>(std::rand()) / RAND_MAX;
	}
}


// Resize method to change the size of the vector
void vec::resize(int newSize) {
	double* newElements = new double[newSize];

	for(int i = 0; i<newSize; ++i) newElements[i] = 0.0;

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


// L1Norm method to calculate the L1 norm of the vector
double vec::calculateL1Norm() const {
	double norm = 0.0;

	for (int i = 0; i < size; ++i) {
		norm += std::abs(elements[i]);
	}

	return norm;
}


// L2Norm method to calculate the L2 norm of the vector
double vec::calculateL2Norm() const {
	double sum = 0.0;
	for (int i = 0; i < size; ++i) {
		sum += elements[i] * elements[i];
	}
	return sqrt(sum);
}


// Check if all elements are zero or very close to zero
bool vec::isZero(double tolerance) const {
	for (int i = 0; i < size; ++i) {
		if (std::abs(elements[i]) > tolerance) {
			return false;
		}
	}
	return true;
}


// Normalize method to make the vector a unit vector based on the specified norm
void vec::normalize(double normType) {
	// Calculate the norm of the vector

	assert(isZero() == false);

	double vectorNorm = 0.0;
	for (int i = 0; i < size; ++i) {
		vectorNorm += std::pow(std::abs(elements[i]), normType);
	}
	vectorNorm = std::pow(vectorNorm, 1.0 / normType);


	// Normalize the vector
	for (int i = 0; i < size; ++i) {
		elements[i] /= vectorNorm;
	}
}


// Add a double to the end of the vector
void vec::addElement(double value) {
	// Create a new array with increased size
	double* newElements = new double[size + 1];

	// Copy existing elements to the new array
	for (int i = 0; i < size; ++i) {
		newElements[i] = elements[i];
	}

	// Add the new element to the end
	newElements[size] = value;

	// Delete the old array and update the Vector object
	delete[] elements;
	elements = newElements;
	++size;
}


// Copy the first "n" elements to another vector
vec vec::head(int n) const {

	assert(n <= size);
	assert(n > 0);

	vec newVector(n);

	// Copy the first "n" elements to the new vector
	for (int i = 0; i < n; ++i) {
		newVector(i) = elements[i];
	}

	return newVector;
}


// Copy the last "n" elements to another vector
vec vec::tail(int n) const {
	// Check if n is valid
	assert(n <= size);
	assert(n > 0);

	// Create a new vector with size "n"
	vec result(n);

	// Copy the last "n" elements to the new vector
	for (int i = 0; i < n; ++i) {
		result(i) = elements[size - n + i];
	}

	return result;
}


// Calculate the mean of the vector
double vec::calculateMean() const {

	assert(size>0);

    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += elements[i];
    }

    return sum / static_cast<double>(size);
}


// Calculate the standard deviation of the vector
double vec::calculateStandardDeviation() const {

	assert(size>0);

    // Calculate the mean of the vector
    double mean = calculateMean();

    // Calculate the sum of squared differences from the mean
    double sumSquaredDifferences = 0.0;
    for (int i = 0; i < size; ++i) {
        double difference = elements[i] - mean;
        sumSquaredDifferences += difference * difference;
    }

    // Calculate the variance
    double variance = sumSquaredDifferences / static_cast<double>(size);

    // Calculate the standard deviation (square root of the variance)
    return sqrt(variance);
}

// Concatenate method to concatenate two vectors
vec vec::concatenate(const vec& other) const {
    // Create a new vector with combined size
    vec result(size + other.getSize());

    // Copy elements from the first vector to the result vector
    for (int i = 0; i < size; ++i) {
        result(i) = elements[i];
    }

    // Copy elements from the second vector to the result vector
    for (int i = 0; i < other.size; ++i) {
        result(i + size) = other(i);
    }

    return result;
}


// Static method for matrix-vector multiplication
//vec vec::matmul(const mat& matrix, const vec& vector) {
//    // Check if matrix and vector dimensions are compatible for multiplication
//	assert(matrix.getNCols() == vector.getSize());
//
//    // Create a result vector with the number of rows in the matrix
//    vec result(matrix.getNRows());
//
//    // Perform matrix-vector multiplication
//    for (int i = 0; i < matrix.getNRows(); ++i) {
//        double sum = 0.0;
//        for (int j = 0; j < matrix.getNCols(); ++j) {
//            sum += matrix(i, j) * vector(j);
//        }
//        result(i) = sum;
//    }
//
//    return result;
//}


