

#ifndef VECTOR
#define VECTOR

#include <iostream>
//#include "matrix.hpp"
namespace rodeo{

class vec {
private:
	int size;
	double* elements;

public:
	// Constructors and Destructor

	vec(int size = 0);
	vec(const vec& other);
	~vec();

	// Assignment operator
	vec& operator=(const vec& other);

	// Access operator
	double& operator()(int index);
	const double& operator()(int index) const;

	// vec addition
	vec operator+(const vec& other) const;

	// vec subtraction
	vec operator-(const vec& other) const;

	// vec scalar multiplication
	vec operator*(double scalar) const;


	// Display vec
	void display() const;

	static int len(vec v){
		return v.size;
	}

	void print(void) const;

	int getSize(void) const;

	void fill(double);
	void fillRandom();

	void resize(int newSize);

	bool isZero(double tolerance = 1e-10) const;

	double calculateL1Norm() const;
	double calculateL2Norm() const;

	void normalize(double normType);

	vec head(int n) const;
	vec tail(int n) const;

	double calculateMean() const;
	double calculateStandardDeviation() const;

	void addElement(double value);

	vec concatenate(const vec& other) const;

//	static vec matmul(const mat& matrix, const vec& vector);
};

}

#endif // vec_H
