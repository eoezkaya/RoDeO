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
#ifndef VEC_H_
#define VEC_H_
#include<string>
#include<vector>

namespace Rodop {

enum NormType { L2, L1, LINF };

class vec {
public:
	vec();
	vec(unsigned int size);
	vec(unsigned int size, double value);
	vec(std::initializer_list<double> init_list);

	vec(const vec& other);       // Copy constructor
	vec& operator=(const vec& other); // Copy assignment operator
	~vec();                      // Destructor

	void resize(unsigned int newSize, double defaultValue = 0.0);
	void reset();

	bool isEmpty() const{
		if(size>0) return false;
		else return true;
	}

	double* getPointer() const;

	std::vector<double> toStdVector() const;
	void fromStdVector(const std::vector<double>& stdVec);


	double& operator()(unsigned int index);
	const double& operator()(unsigned int index) const;
	vec operator+(const vec& other) const;
	vec operator-(const vec& other) const;

	vec operator*(double a) const;
	vec operator+(double a) const;
	vec operator-(double a) const;


	bool operator<(const vec& other) const;
	bool operator>(const vec& other) const;


	double sum() const;
	double product() const;

	static vec scaleAndAdd(const vec& v, double a, double b);
	static vec scaleAndAdd(const vec& x, double a, const vec& b);
	static vec scaleAndAddNoLib(const vec& x, double a, const vec& b);

	std::string toString(unsigned int precision = 4) const;
	void print(const std::string& msg = {}, unsigned int precision = 4) const;

	unsigned int getSize() const;         // Get the size of the vector


	double get(unsigned int index) const; // Get an element at a specific index
	void set(unsigned int index, double value); // Set an element at a specific index

	void fill(double value);
	void fillWithIntegersAscendingOrder();
	void fillRandom();
	void fillRandom(double a, double b);
	void fillRandom(const vec& a, const vec& b);

	static vec generateRandomVectorBetween(const vec& lower, const vec& upper);

	void appendVector(const vec& other, int position = -1);
	void copyVector(const vec& source, int position = 0);
	void push_back(double value);

	void appendToCSV(const std::string& filename, int precision = 6) const;

	void saveToCSV(const std::string& filename) const;
	void readFromCSV(const std::string& filename);


	static double dotProduct(const vec& v1, const vec& v2);

	double dot(const vec& v) const;
	double dotNoLib(const vec& v) const;

	double findMin() const;
	int findMinIndex() const;

	double findMax() const;
	int findMaxIndex() const;

	vec head(unsigned int n) const;
	vec tail(unsigned int n) const;

	double norm(NormType type = L2) const;
	vec unitVector(NormType type = L2) const;

	bool is_equal(const vec& other, double tolerance = 1e-9) const;
	bool is_zero(double tolerance = 10E-9) const;
	bool has_zeros(double tolerance = 1e-9) const;
	bool is_between(const vec& lb, const vec& ub) const;
	bool has_nan() const;

	vec normalizeVector(double xmin, double xmax) const;
	vec normalizeVector(const vec &xmin, const vec &xmax) const;
	vec denormalizeVector(const vec &xmin, const vec &xmax) const;
	vec normalizeVectorFast(const vec &xmin, const vec &xdiff) const;
	vec denormalizeVectorFast(const vec &xmin, const vec &xdiff) const;

	void swap(vec& other);
	void sort(bool ascending = true);
	bool is_sorted(bool ascending = true) const;
	int findInterval(double value) const;

	static double computeGaussianCorrelation(const vec &v1, const vec &v2, const vec &theta, int n);
	static double computeGaussianCorrelation(const double *v1, const double *v2, const double *theta, int n);
	static double computeGaussianCorrelationNaive(const double *v1, const double *v2, const double *theta, int n);
#ifdef OPENBLAS
static double computeGaussianCorrelationOpenBlas(const double *v1, const double *v2, const double *theta, int n);
#endif

static double computeGaussianCorrelationDot(const double *xi,
		const double *xj,
		const double *direction,
		const double *theta,
		int n);


static double computeGaussianCorrelationDotDot(const double *xi,
		const double *xj,
		const double *direction1,
		const double *direction2,
		const double *theta,
		int n);




static double computeExponentialCorrelation(const double *v1, const double *v2, const double *theta, const double *gamma, int n);
static double computeExponentialCorrelationNaive(const double *v1, const double *v2, const double *theta, const double *gamma, int n);
#ifdef OPENBLAS
static double computeExponentialCorrelationOpenBlas(const double *v1, const double *v2, const double *theta, const double *gamma, int n);
#endif


static double computeExponentialCorrelationDot(const double *xi,
		const double *xj,
		const double *direction,
		const double *theta,
		const double *gamma,
		int n);

double quantile(double p) const;
double mean() const;
double standardDeviation() const;


private:
unsigned int size;
double* data;

void copyData(const double* source, double* destination, int size); // Helper function to copy data
};

} // namespace Rodop

#endif // VEC_H_
