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
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random>
#include <iomanip>
#ifdef OPENBLAS
#include <cblas.h>
#endif

namespace Rodop {





vec::vec() : size(0), data(nullptr) {}

vec::vec(unsigned int size) : size(size), data(new double[size]) {
	std::fill(data, data + size, 0.0);
}

vec::vec(unsigned int size, double value) : size(size), data(new double[size]) {
	std::fill(data, data + size, value);
}

vec::vec(const vec& other) : size(other.size), data(new double[other.size]) {
	copyData(other.data, data, other.size);
}

vec::vec(std::initializer_list<double> init_list) : size(init_list.size()), data(new double[init_list.size()]) {
	std::copy(init_list.begin(), init_list.end(), data);
}

vec& vec::operator=(const vec& other) {
	if (this == &other) {
		return *this; // Handle self-assignment
	}

	delete[] data;

	size = other.size;
	data = new double[other.size];
	copyData(other.data, data, other.size);

	return *this;
}

vec::~vec() {
	delete[] data;
}




void vec::reset() {
	delete[] data;
	data = nullptr;
	size = 0;
}

void vec::resize(unsigned int newSize, double defaultValue) {

	// Allocate new memory for the new size
	double* newData = new double[newSize];

	// Determine the number of elements to copy
	int elementsToCopy = std::min(size, newSize);

	// Copy the existing elements to the new array
	std::copy(data, data + elementsToCopy, newData);

	// If newSize is greater, initialize the new elements with defaultValue
	if (newSize > size) {
		std::fill(newData + size, newData + newSize, defaultValue);
	}

	// Delete the old data and assign the new data to the vector
	delete[] data;
	data = newData;
	size = newSize;
}

// Access operators
double& vec::operator()(unsigned int index) {
	if (index >= size) {
		throw std::out_of_range("Index out of range.");
	}
	return data[index];
}

const double& vec::operator()(unsigned int index) const {
	if (index >= size) {
		throw std::out_of_range("Index out of range.");
	}
	return data[index];
}

vec vec::operator+(const vec& other) const {
	if (size != other.size) {
		throw std::invalid_argument("Vectors must be of the same size");
	}

	vec result(size);
	for (unsigned int i = 0; i < size; ++i) {
		result.data[i] = data[i] + other.data[i];
	}
	return result;
}

vec vec::operator-(const vec& other) const {
	if (size != other.size) {
		throw std::invalid_argument("Vectors must be of the same size");
	}

	vec result(size);
	for (unsigned int i = 0; i < size; ++i) {
		result.data[i] = data[i] - other.data[i];
	}
	return result;
}

bool vec::operator<(const vec& other) const {
	if (size != other.size) {
		throw std::invalid_argument("Vectors must be of the same size to compare.");
	}

	for (unsigned int i = 0; i < size; ++i) {
		if (data[i] >= other.data[i]) {
			return false;
		}
	}

	return true;
}

bool vec::operator>(const vec& other) const {
	if (size != other.size) {
		throw std::invalid_argument("Vectors must be of the same size to compare.");
	}

	for (unsigned int i = 0; i < size; ++i) {
		if (data[i] <= other.data[i]) {
			return false;
		}
	}

	return true;
}



std::vector<double> vec::toStdVector() const {
	return std::vector<double>(data, data + size);
}


void vec::fromStdVector(const std::vector<double>& stdVec) {
	// Resize the vec object to match the size of the input std::vector
	resize(static_cast<int>(stdVec.size()));

	// Copy elements from std::vector to the vec data array
	std::copy(stdVec.begin(), stdVec.end(), data);
}

vec vec::operator*(double a) const {

	vec result(size);
	for (unsigned int i = 0; i < size; ++i) {
		result.data[i] = data[i]*a;
	}
	return result;
}


vec vec::operator+(double a) const {

	vec result(size);
	for (unsigned int i = 0; i < size; ++i) {
		result.data[i] = data[i]+a;
	}
	return result;
}

vec vec::operator-(double a) const {

	vec result(size);
	for (unsigned int i = 0; i < size; ++i) {
		result.data[i] = data[i]-a;
	}
	return result;
}


double vec::sum() const {
	double total = 0.0;
	for (unsigned int i = 0; i < size; ++i) {
		total += data[i];
	}
	return total;
}

double vec::product() const {
	if (size == 0) {
		return 1.0; // By convention, the product of an empty set is 1
	}

	double totalProduct = 1.0;
	for (unsigned int i = 0; i < size; ++i) {
		totalProduct *= data[i];
	}
	return totalProduct;
}


vec vec::scaleAndAdd(const vec& v, double a, double b) {
	unsigned int size = v.getSize();
	vec result(size);

	for (unsigned int i = 0; i < size; ++i) {
		result.data[i] = a * v.data[i] + b;
	}

	return result;
}


vec vec::scaleAndAdd(const vec& x, double a, const vec& b) {
	if (x.size != b.size) {
		throw std::invalid_argument("Vectors must be of the same size");
	}


#ifdef OPENBLAS

	vec result(x.size);
	// Copy b to result
	cblas_dcopy(b.size, b.data, 1, result.data, 1);

	// Perform result = a * x + result
	cblas_daxpy(x.size, a, x.data, 1, result.data, 1);

	return result;

#else

	return x*a + b;
#endif


}

vec vec::scaleAndAddNoLib(const vec& x, double a, const vec& b) {
	if (x.size != b.size) {
		throw std::invalid_argument("Vectors must be of the same size");
	}
	return x*a + b;

}




double* vec::getPointer() const {
	return data;
}



unsigned int vec::getSize() const {
	return size;
}



void vec::print(const std::string& msg, unsigned int precision) const {
	// Print the message if it's not empty
	if (!msg.empty()) {
		std::cout << msg << "\n";
	}

	// Check if the vector is empty
	if (size == 0) {
		std::cout << "Vector is empty." << std::endl;
		return;
	}

	// Set the precision for floating-point output
	std::cout << std::fixed << std::setprecision(precision);

	// Print the vector elements
	for (unsigned int i = 0; i < size; ++i) {
		std::cout << data[i];
		if (i < size - 1) {
			std::cout << ", ";  // Add comma separator except for the last element
		}
	}
	std::cout << std::endl;  // End with a newline
}

std::string vec::toString(unsigned int precision) const {
	std::ostringstream oss;
	oss << std::fixed << std::setprecision(precision);

	// Check if the vector is empty
	if (size == 0) {
		oss << "Vector is empty.";
		return oss.str();
	}

	// Append vector elements
	for (unsigned int i = 0; i < size; ++i) {
		oss << data[i];
		if (i < size - 1) {
			oss << ", ";  // Add comma separator except for the last element
		}
	}

	return oss.str();
}

double vec::get(unsigned int index) const {
	if (index >= size) {
		throw std::out_of_range("Index out of range");
	}
	return data[index];
}

void vec::set(unsigned int index, double value) {
	if (index >= size) {
		throw std::out_of_range("Index out of range");
	}
	data[index] = value;
}

void vec::fill(double value) {
	std::fill(data, data + size, value);
}

void vec::fillWithIntegersAscendingOrder(){

	for(unsigned int i=0; i<size; i++){
		data[i] = static_cast<int>(i+1);
	}

}


void vec::fillRandom() {
	// Create a random number generator
	std::random_device rd;  // Seed generator
	std::mt19937 gen(rd()); // Mersenne Twister engine
	std::uniform_real_distribution<> dis(0.0, 1.0); // Range [0, 1)

	for (unsigned int i = 0; i < size; ++i) {
		data[i] = dis(gen); // Generate random number in the range [0, 1)
	}
}

void vec::fillRandom(double a, double b) {
	if (a > b) {
		throw std::invalid_argument("Lower bound 'a' must be less than or equal to upper bound 'b'.");
	}

	// Create a random number generator
	std::random_device rd;  // Seed generator
	std::mt19937 gen(rd()); // Mersenne Twister engine
	std::uniform_real_distribution<> dis(a, b); // Range [a, b]

	for (unsigned int i = 0; i < size; ++i) {
		data[i] = dis(gen); // Generate random number in the range [a, b]
	}
}


void vec::fillRandom(const vec& a, const vec& b) {
	if (a.getSize() != size || b.getSize() != size) {
		throw std::invalid_argument("Input vectors must have the same size as the target vector.");
	}

	// Create a random number generator
	std::random_device rd;  // Seed generator
	std::mt19937 gen(rd()); // Mersenne Twister engine

	for (unsigned int i = 0; i < size; ++i) {
		if (a(i) > b(i)) {
			throw std::invalid_argument("For each index, the element in vector 'a' must be less than or equal to the element in vector 'b'.");
		}

		std::uniform_real_distribution<> dis(a(i), b(i)); // Range [a(i), b(i)]
		data[i] = dis(gen); // Generate random number in the range [a(i), b(i)]
	}
}


vec vec::generateRandomVectorBetween(const vec& lower, const vec& upper) {
	if (lower.size != upper.size) {
		throw std::invalid_argument("Both vectors must be of the same size.");
	}

	vec result(lower.size);
	std::random_device rd;  // Seed generator
	std::mt19937 gen(rd()); // Mersenne Twister engine

	for (unsigned int i = 0; i < lower.size; ++i) {
		std::uniform_real_distribution<> dis(lower.data[i], upper.data[i]);
		result.data[i] = dis(gen);
	}

	return result;
}



void vec::appendVector(const vec& other, int position) {
	if (position < -1 || position > static_cast<int>(size)) {
		throw std::out_of_range("Position out of range.");
	}

	int newSize = size + other.size;
	double* newData = new double[newSize];

	if (position == -1) {
		position = size;
	}

	// Copy elements before the position
	std::copy(data, data + position, newData);
	// Copy the new vector elements
	std::copy(other.data, other.data + other.size, newData + position);
	// Copy the remaining elements
	std::copy(data + position, data + size, newData + position + other.size);

	delete[] data;
	data = newData;
	size = newSize;
}

void vec::copyVector(const vec& source, int position) {
	if (position < 0 || position >= static_cast<int>(size)) {
		throw std::out_of_range("Position out of range.");
	}

	int copySize = std::min(size - position, source.size);

	// Copy elements from source to this vector starting at the given position
	std::copy(source.data, source.data + copySize, data + position);
}

void vec::push_back(double value) {
	// Allocate new memory with one more element
	double* newData = new double[size + 1];

	// Copy the existing elements to the new array
	for (unsigned int i = 0; i < size; ++i) {
		newData[i] = data[i];
	}

	// Add the new element at the end
	newData[size] = value;

	// Delete the old data
	delete[] data;

	// Update the pointer and size
	data = newData;
	++size;
}

void vec::copyData(const double* source, double* destination, int size) {
	std::copy(source, source + size, destination);
}



double vec::dotProduct(const vec& v1, const vec& v2) {
	if (v1.size != v2.size) {
		throw std::invalid_argument("Vectors must be of the same size");
	}

	double result = 0.0;
	for (unsigned int i = 0; i < v1.size; ++i) {
		result += v1.data[i] * v2.data[i];
	}
	return result;
}

double vec::dot(const vec& v) const {
	if (size != v.size) {
		throw std::invalid_argument("Vectors must be of the same size");
	}

	double result = 0.0;

#ifndef OPENBLAS


	for (unsigned int i = 0; i < size; ++i) {
		result += data[i] * v.data[i];
	}

#else

	result = cblas_ddot(size, data, 1, v.data, 1);

#endif
	return result;
}

double vec::dotNoLib(const vec& v) const {
	if (size != v.size) {
		throw std::invalid_argument("Vectors must be of the same size");
	}

	double result = 0.0;

	for (unsigned int i = 0; i < size; ++i) {
		result += data[i] * v.data[i];
	}

	return result;
}

double vec::findMin() const {
	if (size == 0) {
		throw std::runtime_error("Cannot find the minimum of an empty vector.");
	}
	return *std::min_element(data, data + size);
}

int vec::findMinIndex() const {
	if (size == 0) {
		throw std::runtime_error("Cannot find the index of the minimum element of an empty vector.");
	}
	return std::distance(data, std::min_element(data, data + size));
}

double vec::findMax() const {
	if (size == 0) {
		throw std::runtime_error("Cannot find the minimum of an empty vector.");
	}
	return *std::max_element(data, data + size);
}

int vec::findMaxIndex() const {
	if (size == 0) {
		throw std::runtime_error("Cannot find the index of the maximum element of an empty vector.");
	}
	return std::distance(data, std::max_element(data, data + size));
}

vec vec::head(unsigned int n) const {
	if (n > size) {
		throw std::out_of_range("Head size is out of range.");
	}
	vec result(n);
	std::copy(data, data + n, result.data);
	return result;
}

vec vec::tail(unsigned int n) const {
	if (n > size) {
		throw std::out_of_range("Tail size is out of range.");
	}
	vec result(n);
	std::copy(data + size - n, data + size, result.data);
	return result;
}

double vec::norm(NormType type) const {
	double result = 0.0;

	switch (type) {
	case L2:
		for (unsigned int i = 0; i < size; ++i) {
			result += data[i] * data[i];
		}
		result = std::sqrt(result);
		break;

	case L1:
		for (unsigned int i = 0; i < size; ++i) {
			result += std::abs(data[i]);
		}
		break;

	case LINF:
		for (unsigned int i = 0; i < size; ++i) {
			result = std::max(result, std::abs(data[i]));
		}
		break;

	default:
		throw std::invalid_argument("Unsupported norm type.");
	}

	return result;
}

vec vec::unitVector(NormType type) const {
	double vectorNorm = norm(type);
	if (vectorNorm == 0.0) {
		throw std::runtime_error("Cannot compute the unit vector of a zero vector.");
	}

	vec result(size);
	for (unsigned int i = 0; i < size; ++i) {
		result.data[i] = data[i] / vectorNorm;
	}

	return result;
}


void vec::appendToCSV(const std::string& filename, int precision) const {
    // Check if the CSV file exists and has the expected number of columns
    std::ifstream inFile(filename);
    if (inFile.is_open()) {
        std::string line;
        if (std::getline(inFile, line)) {
            std::stringstream ss(line);
            std::string cell;
            unsigned int columnCount = 0;

            // Count the number of columns in the first line
            while (std::getline(ss, cell, ',')) {
                columnCount++;
            }

            // Validate column count
            if (columnCount != size) {
                throw std::runtime_error("Size of vector does not match the number of columns in the CSV file.");
            }
        }
        inFile.close();
    }

    // Open the CSV file in append mode
    std::ofstream outFile(filename, std::ios::app);
    if (!outFile.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Set precision for floating-point output
    outFile << std::fixed << std::setprecision(precision);

    // Append vector data as a new row
    for (unsigned int i = 0; i < size; ++i) {
        outFile << data[i];
        if (i < size - 1) {
            outFile << ",";
        }
    }
    outFile << "\n";
    outFile.close();
}


void vec::saveToCSV(const std::string& filename) const {
	std::ofstream outFile(filename);
	if (!outFile.is_open()) {
		throw std::runtime_error("Cannot open file: " + filename);
	}

	for (unsigned int i = 0; i < size; ++i) {
		outFile << data[i];
		if (i < size - 1) {
			outFile << "\n"; // Save each element on a new line
		}
	}
	outFile.close();
}

void vec::readFromCSV(const std::string& filename) {
	std::ifstream inFile(filename);
	if (!inFile.is_open()) {
		throw std::runtime_error("Cannot open file: " + filename);
	}

	std::vector<double> tempData;
	std::string line;
	while (std::getline(inFile, line)) {
		std::istringstream ss(line);
		double value;
		if (!(ss >> value)) {
			throw std::runtime_error("Invalid data in CSV file.");
		}
		tempData.push_back(value);
	}
	inFile.close();

	// Resize vector to fit new data and copy data
	resize(tempData.size());
	std::copy(tempData.begin(), tempData.end(), data);
}







bool vec::is_equal(const vec& other, double tolerance) const {
	// First, check if the sizes of the two vectors are the same
	if (size != other.size) {
		return false;
	}

	// Compare each element with the corresponding element in the other vector
	for (unsigned int i = 0; i < size; ++i) {
		if (std::abs(data[i] - other.data[i]) > tolerance) {
			return false; // If any element difference is greater than the tolerance, return false
		}
	}

	// If all elements are within the tolerance, the vectors are considered equal
	return true;
}



bool vec::is_zero(double tolerance) const {
	for (unsigned int i = 0; i < size; ++i) {
		if (std::abs(data[i]) > tolerance) {
			return false;
		}
	}
	return true;
}

bool vec::has_zeros(double tolerance) const {
	for (unsigned int i = 0; i < size; ++i) {
		if (std::abs(data[i]) <= tolerance) {
			return true;
		}
	}
	return false;
}


bool vec::is_between(const vec& lb, const vec& ub) const {
	if (size != lb.size || size != ub.size) {
		throw std::invalid_argument("All vectors must be of the same size");
	}

	for (unsigned int i = 0; i < size; ++i) {
		if (data[i] < lb(i) || data[i] > ub(i)) {
			return false;
		}
	}

	return true;
}


bool vec::has_nan() const {
	for (unsigned int i = 0; i < size; ++i) {
		if (std::isnan(data[i])) {
			return true;
		}
	}
	return false;
}

/* this version is around 20 percent faster */
vec vec::normalizeVectorFast(const vec &xmin, const vec &xdiff) const{

	vec xnorm(size);
	double fac = 1.0/size;

	for(unsigned int i=0; i<size; i++){

		xnorm(i) = fac*(data[i] - xmin(i)) / xdiff(i);
	}

	return xnorm;
}

vec vec::normalizeVector(const vec &xmin, const vec &xmax) const{

	vec xnorm(size);
	double fac = 1.0/size;

	for(unsigned int i=0; i<size; i++){

		xnorm(i) = fac*(data[i] - xmin(i)) / (xmax(i) - xmin(i));
	}

	return xnorm;
}

vec vec::normalizeVector(double xmin, double xmax) const{

	vec xnorm(size);
	double fac = 1.0/size;
	double dx = xmax-xmin;

	for(unsigned int i=0; i<size; i++){

		xnorm(i) = fac*(data[i] - xmin) / dx;
	}

	return xnorm;
}




vec vec::denormalizeVector(const vec &xmin, const vec &xmax) const {

	vec original(size);

	for(unsigned int i = 0; i < size; i++) {
		original(i) = (data[i] * size) * (xmax(i) - xmin(i)) + xmin(i);
	}

	return original;
}

vec vec::denormalizeVectorFast(const vec &xmin, const vec &xdiff) const {

	vec original(size);

	for(unsigned int i = 0; i < size; i++) {
		original(i) = (data[i] * size) * (xdiff(i)) + xmin(i);
	}

	return original;
}

void vec::swap(vec& other) {
	// Swap the size
	std::swap(size, other.size);

	// Swap the data pointers
	std::swap(data, other.data);
}

void vec::sort(bool ascending) {
	if (size <= 1) {
		return;  // No need to sort if the vector has 0 or 1 elements
	}
	if (ascending) {
		std::sort(data, data + size);  // Sorts in ascending order
	} else {
		std::sort(data, data + size, std::greater<double>());  // Sorts in descending order
	}
}

bool vec::is_sorted(bool ascending) const {
	if (size <= 1) {
		return true;  // A vector with 0 or 1 elements is trivially sorted
	}

	for (unsigned int i = 0; i < size - 1; ++i) {
		if (ascending) {
			if (data[i] > data[i + 1]) {
				return false;  // Not sorted in ascending order
			}
		} else {
			if (data[i] < data[i + 1]) {
				return false;  // Not sorted in descending order
			}
		}
	}

	return true;  // The vector is sorted in the specified order
}

int vec::findInterval(double value) const {

	if(size == 0) return -1;
	// Check if the vector is sorted in ascending order
	if (!is_sorted(true)) {
		throw std::logic_error("Vector is not sorted in ascending order.");
	}

	for (unsigned int i = 0; i < size - 1; ++i) {
		double xs = data[i];
		double xe = data[i + 1];

		if (value >= xs && value < xe) {
			return i;
		}
	}

	if (value > data[size - 1]) {
		return size - 1;
	}

	return -1;  // Value is not within any interval
}

double vec::computeGaussianCorrelation(const vec &v1, const vec &v2, const vec &theta, int n){

	return computeGaussianCorrelation(v1.getPointer(), v2.getPointer(), theta.getPointer(), n);
}


double vec::computeGaussianCorrelation(const double *v1, const double *v2, const double *theta, int n){

#ifdef OPENBLAS
	return computeGaussianCorrelationOpenBlas(v1,v2,theta,n);
#else
	return computeGaussianCorrelationNaive(v1,v2,theta,n);
#endif
}

double vec::computeGaussianCorrelationNaive(const double *v1, const double *v2, const double *theta, int n) {

	double sum = 0.0;
	for (int i = 0; i < n; ++i) {
		double diff = v1[i] - v2[i];
		sum += theta[i] * (diff * diff);
	}
	return std::exp(-sum);
}


double vec::computeGaussianCorrelationDot(const double *xi,
		const double *xj,
		const double *direction,
		const double *theta,
		int n)  {

	double sumd = 0.0;
	double sum  = 0.0;

	for (int k = 0; k < n; k++) {

		double diff = xi[k] - xj[k];
		sumd += -2.0*theta[k] * diff *direction[k];
		sum  += theta[k] * diff* diff;
	}

	double correlation = std::exp(-sum);

	double derivative = -1.0*sumd*correlation;
	return derivative;
}


double vec::computeGaussianCorrelationDotDot(const double *xi,
		const double *xj,
		const double *direction1,
		const double *direction2,
		const double *theta,
		int n){


	double td = 0.0;
	double t = 0.0;
	double td0 = 0.0;
	double tdd = 0.0;
	double temp;
	for (int i = 0; i < n; i++) {
		temp = 2.0*theta[i]*direction1[i];
		tdd = tdd + temp*direction2[i];
		td = td - temp*(xi[i]-xj[i]);
		td0 = td0 - theta[i]*2.0*(xi[i]-xj[i])*direction2[i];
		t += theta[i]*(xi[i]-xj[i])*(xi[i]-xj[i]);

	}
	temp = std::exp(-t);
	double resultdd = -(temp*tdd-td*exp(-t)*td0);
	return resultdd;

}


#ifdef OPENBLAS
double vec::computeGaussianCorrelationOpenBlas(const double *v1, const double *v2, const double *theta, int n) {
	std::vector<double> diff(n);

	for (int i = 0; i < n; ++i) {
		diff[i] = v1[i] - v2[i];
	}

	for (int i = 0; i < n; ++i) {
		diff[i] *= diff[i];
	}

	double sum = cblas_ddot(n, diff.data(), 1, theta, 1);

	return std::exp(-sum);
}
#endif


double vec::computeExponentialCorrelation(const double *v1, const double *v2, const double *theta, const double *gamma, int n) {

#ifdef OPENBLAS
	return computeExponentialCorrelationOpenBlas(v1,v2,theta,gamma,n);
#else
	return computeExponentialCorrelationNaive(v1,v2,theta,gamma,n);
#endif

}

double vec::computeExponentialCorrelationDot(const double *xi,
		const double *xj,
		const double *direction,
		const double *theta,
		const double *gamma,
		int n)  {

	double sum  = 0.0;
	double sumd = 0.0;
	double fabs0 = 0.0;
	double fabs0d = 0.0;

	//	vec diff = xi - xj;
	//	double normDiff = diff.norm();
	//	// We need to check this
	//	if(normDiff < std::numeric_limits<double>::epsilon()*10) {
	//		return 0.0;
	//	}


	for (int k = 0; k < n; ++k) {
		if (xi[k] - xj[k] >= 0.0) {
			fabs0d = -direction[k];
			fabs0 = xi[k] - xj[k];
		} else {
			fabs0d = direction[k];
			fabs0 = -(xi[k]-xj[k]);
		}
		double exponentialPart  = pow(fabs0, gamma[k]);
		double exponentialPartd = gamma[k]*std::pow(fabs0, (gamma[k]-1))*fabs0d;

		sumd = sumd + theta[k]*exponentialPartd;
		sum += theta[k]*exponentialPart;
	}
	double correlationd;
	correlationd = -(exp(-sum)*sumd);

	return correlationd;
}




double vec::computeExponentialCorrelationNaive(const double *v1, const double *v2, const double *theta, const double *gamma, int n) {
	double sum = 0.0;

	for (int k = 0; k < n; ++k) {
		double diff = std::abs(v1[k] - v2[k]);
//		double exponentialPart = diff*diff;
		double exponentialPart = std::pow(diff, gamma[k]);
		sum += theta[k] * exponentialPart;
	}

	double correlation = std::exp(-sum);
	return correlation;
}

double vec::computeExponentialCorrelationOpenBlas(const double *v1, const double *v2, const double *theta, const double *gamma, int n) {
	// Step 1: Compute the element-wise difference between v1 and v2 and raise it to the power of gamma
	double *diff = new double[n];
	for (int k = 0; k < n; ++k) {
		diff[k] = std::pow(std::abs(v1[k] - v2[k]), gamma[k]);
	}

	// Step 2: Compute the weighted sum using theta
	double sum = 0.0;
	sum = cblas_ddot(n, theta, 1, diff, 1);

	// Step 3: Clean up the dynamically allocated memory
	delete[] diff;

	// Step 4: Compute and return the correlation
	return std::exp(-sum);
}


/* The quantile method will calculate the value below which a given percentage of the data falls.
 * If p=0.9, it should give the value such that 90% of the samples are below this value.
 *
 */
double vec::quantile(double p) const {
	if (p < 0.0 || p > 1.0) {
		throw std::invalid_argument("Probability must be between 0 and 1.");
	}

	if (size == 0) {
		throw std::runtime_error("Cannot compute quantile of an empty vector.");
	}

	// Copy data to a temporary vector and sort it
	std::vector<double> sorted_data(data, data + size);
	std::sort(sorted_data.begin(), sorted_data.end());

	// Find the index for the quantile
	double idx = p * (size - 1);
	int lower_idx = static_cast<int>(std::floor(idx));
	int upper_idx = static_cast<int>(std::ceil(idx));

	if (lower_idx == upper_idx) {
		return sorted_data[lower_idx];
	} else {
		// Linear interpolation if the index is not an integer
		double weight = idx - lower_idx;
		return sorted_data[lower_idx] * (1 - weight) + sorted_data[upper_idx] * weight;
	}
}

double vec::mean() const {
	if (size == 0) {
		throw std::runtime_error("Cannot compute mean of an empty vector.");
	}
	double sum = 0.0;
	for (unsigned int i = 0; i < size; ++i) {
		sum += data[i];
	}
	return sum / size;
}

// Method to calculate the standard deviation of the vector elements
double vec::standardDeviation() const {
	if (size == 0) {
		throw std::runtime_error("Cannot compute standard deviation of an empty vector.");
	}
	double meanValue = mean();
	double sumSquares = 0.0;
	for (unsigned int i = 0; i < size; ++i) {
		double diff = data[i] - meanValue;
		sumSquares += diff * diff;
	}
	return std::sqrt(sumSquares / size);
}



} // namespace Rodop
