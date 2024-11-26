#include <stdexcept>
#include "./INCLUDE/bounds.hpp"



namespace Rodop {

Bounds::Bounds(){}

Bounds::Bounds(unsigned int dim){
	dimension = dim;
}

Bounds::Bounds(std::vector<double> lb, std::vector<double> ub) {
	if (lb.size() != ub.size()) {
		throw std::invalid_argument("Lower and upper bounds must have the same size.");
	}

	unsigned int size = lb.size();
	lowerBounds.resize(size);
	upperBounds.resize(size);

	for (unsigned int i = 0; i < size; ++i) {
		lowerBounds(i) = lb[i];
		upperBounds(i) = ub[i];
	}

	dimension = size;

	if (!checkBounds()) {
		throw std::invalid_argument("Invalid bounds: each lower bound must be less than the corresponding upper bound.");
	}

	ifBoundsAreSet = true;
}


Bounds::Bounds(vec lb, vec ub){

	lowerBounds = lb;
	upperBounds = ub;
	dimension = lb.getSize();

	if(!checkBounds()){
		throw std::runtime_error("checkBounds failed.");
	}
	ifBoundsAreSet = true;

}


void Bounds::reset(void){

	lowerBounds.reset();
	upperBounds.reset();
	dimension = 0;
	ifBoundsAreSet = false;

}

unsigned int Bounds::getDimension(void) const{
	return dimension;
}
void Bounds::setDimension(unsigned int value){
	dimension = value;
}

void Bounds::setBounds(double lowerBound, double upperBound) {

	if (lowerBound >= upperBound) {
		throw std::invalid_argument("Upper bound must be greater than lower bound.");
	}
	if (dimension<=0) {
		throw std::invalid_argument("Dimension is not set.");
	}

	lowerBounds.resize(dimension);
	lowerBounds.fill(lowerBound);

	upperBounds.resize(dimension);
	upperBounds.fill(upperBound);

	ifBoundsAreSet = true;

}

void Bounds::setBounds(std::vector<double> lb, std::vector<double> ub){

	unsigned int size = lb.size();
	lowerBounds.resize(size);
	upperBounds.resize(size);

	for (unsigned int i = 0; i < size; ++i) {
		lowerBounds(i) = lb[i];
		upperBounds(i) = ub[i];
	}

	dimension = size;

	if (!checkBounds()) {
		throw std::invalid_argument("Invalid bounds: each lower bound must be less than the corresponding upper bound.");
	}

	ifBoundsAreSet = true;
}


void Bounds::setBounds(vec lowerBoundsInput, vec upperBoundsInput) {
	if (lowerBoundsInput.getSize() <= 0 || upperBoundsInput.getSize() <= 0) {
		throw std::invalid_argument("Bounds size must be greater than 0.");
	}
	if (lowerBoundsInput.getSize() != upperBoundsInput.getSize()) {
		throw std::invalid_argument("Lower bounds and upper bounds must have the same size.");
	}

	lowerBounds = lowerBoundsInput;
	upperBounds = upperBoundsInput;

	dimension = lowerBounds.getSize();
	if (!checkBounds()) {
		throw std::invalid_argument("Invalid bounds: lower bound must be less than upper bound for all dimensions.");
	}
	ifBoundsAreSet = true;
}



bool Bounds::areBoundsSet(void) const{
	return ifBoundsAreSet;
}


vec Bounds::getLowerBounds(void) const{
	return lowerBounds;
}

vec Bounds::getUpperBounds(void) const{
	return upperBounds;
}


bool Bounds::checkIfBoundsAreValid(void) const{

	if (dimension == 0) {
		throw std::invalid_argument("Dimension must be greater than 0.");
	}

	for(unsigned int i=0; i<dimension; i++){
		if(lowerBounds(i) >= upperBounds(i) ) {
			lowerBounds.print("lowerBounds");
			upperBounds.print("upperBounds");
			return false;
		}
	}
	return true;
}

bool Bounds::checkBounds(void) const{

	if (dimension <= 0) {
		throw std::invalid_argument("Dimension must be greater than 0.");
	}

	for(unsigned int i=0; i<dimension; i++){
		if(lowerBounds(i) >= upperBounds(i) ) {
			return false;
		}
	}
	return true;
}

bool Bounds::isPointWithinBounds(const vec &inputVector) const {
	if (inputVector.getSize() != dimension) {
		throw std::invalid_argument("Input vector size must match the dimension of the bounds.");
	}

	for (unsigned int i = 0; i < dimension; ++i) {
		if (inputVector(i) < lowerBounds(i) || inputVector(i) > upperBounds(i)) {
			return false;
		}
	}

	return true;
}


void Bounds::print(void) const{

	lowerBounds.print("Lower bounds");
	upperBounds.print("Upper bounds");

}

vec Bounds::generateVectorWithinBounds() const {
	if (!areBoundsSet()) {
		throw std::logic_error("Bounds must be set before generating a vector within bounds.");
	}

	vec randomVector(dimension);
	randomVector.fillRandom(lowerBounds, upperBounds);

	return randomVector;
}


std::vector<double> Bounds::generateStdVectorWithinBounds() const {
	if (!areBoundsSet()) {
		throw std::logic_error("Bounds must be set before generating a vector within bounds.");
	}

	vec randomVector(dimension);
	randomVector.fillRandom(lowerBounds, upperBounds);

	vector<double> result;
	for(unsigned int i=0; i<dimension; i++){
		result.push_back(randomVector(i));
	}


	return result;
}

}


