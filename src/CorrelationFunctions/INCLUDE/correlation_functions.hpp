#ifndef CORRELATION_FUNCTIONS_HPP
#define CORRELATION_FUNCTIONS_HPP

#include "../../LinearAlgebra/INCLUDE/matrix.hpp"
#include "../../LinearAlgebra/INCLUDE/vector.hpp"

namespace Rodop {

class CorrelationFunctionBase {

protected:

    mat X;
    unsigned int N = 0;
    unsigned int dim = 0;
    mat correlationMatrix;

    vec correlationVec;

    double epsilon = 1e-12;

    bool ifInputSampleMatrixIsSet = false;

public:

    CorrelationFunctionBase();
    virtual ~CorrelationFunctionBase() = default;

    void setInputSampleMatrix(const mat&);
    void setEpsilon(double);
    void setDimension(unsigned int);

    bool isInputSampleMatrixSet(void) const;

    mat getCorrelationMatrix(void) const;

    virtual void computeCorrelationMatrix(void) = 0;
    virtual vec computeCorrelationVector(const vec &x) const = 0;
    virtual void setHyperParameters(const vec &) = 0;
    virtual double computeCorrelation(const vec &xi, const vec &xj) const = 0;
    virtual double computeCorrelationDot(const vec &xi, const vec &xj, const vec &dir) const = 0;
};

}  // namespace Rodop

#endif  // CORRELATION_FUNCTIONS_HPP
