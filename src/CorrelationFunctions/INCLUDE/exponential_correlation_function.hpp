#ifndef EXP_CORRELATION_FUNCTION_HPP
#define EXP_CORRELATION_FUNCTION_HPP

#include "correlation_functions.hpp"

namespace Rodop {

class ExponentialCorrelationFunction : public CorrelationFunctionBase {

private:

    vec thetaParameters;
    vec gammaParameters;

public:

    ExponentialCorrelationFunction() = default;

    void setTheta(const vec& theta);
    void setGamma(const vec& gamma);

    void print(void) const;

    void initialize(void);


    vec getHyperParameters(void) const;

    bool checkIfParametersAreSetProperly(void) const;


    double computeCorrelation(const vec &x_i, const vec &x_j) const override;
    double computeCorrelationDot(const vec &x_i, const vec &x_j, const vec &x_dot) const override;
    void computeCorrelationMatrix(void) override;
    vec computeCorrelationVector(const vec &x) const override;
    void setHyperParameters(const vec &hyperParams) override;
};

}  // namespace Rodop

#endif  // EXP_CORRELATION_FUNCTION_HPP
