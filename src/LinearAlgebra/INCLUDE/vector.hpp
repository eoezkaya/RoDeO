

#ifndef VECTOR
#define VECTOR

#include <iostream>

namespace rodeo{

class vec {
private:
	int size;
	double* elements;

public:
	// Constructors and Destructor
	vec(int size);
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

	void resize(int newSize);
};

}

#endif // vec_H
