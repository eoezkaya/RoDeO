
#ifndef BOUNDS_HPP
#define BOUNDS_HPP
#include<vector>
#include "../../LinearAlgebra/INCLUDE/vector.hpp"

using namespace std;

namespace Rodop {

class Bounds{

private:

	unsigned int dimension=0;
	vec upperBounds;
	vec lowerBounds;
	bool ifBoundsAreSet = false;

public:

	Bounds();
	Bounds(unsigned int);

	Bounds(vec , vec);
	Bounds(vector<double> lb, vector<double> ub);

	void reset(void);

	void setDimension(unsigned int);
	unsigned int getDimension(void) const;

	void setBounds(vec lowerBoundsInput, vec upperBoundsInput);
	void setBounds(vector<double> lb, vector<double> ub);
	void setBounds(double, double);
	void setBounds(double* lb, double* upperBound);

	bool checkIfBoundsAreValid(void) const;
	bool checkBounds(void) const;

	bool areBoundsSet(void) const;

	bool isPointWithinBounds(const vec &inputVector) const;

	vec generateVectorWithinBounds() const;
	vector<double> generateStdVectorWithinBounds() const;

	void print(void) const;

	vec getLowerBounds(void) const;
	vec getUpperBounds(void) const;

};

}

#endif
