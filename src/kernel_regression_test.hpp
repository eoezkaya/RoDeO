#ifndef KERNELREGTEST
#define KERNELREGTEST

void generateRandomTestDataForKernelRegression(int dim, int N);

void testAllKernelRegression(void);
void testcalculateMahalanobisMatrix(void);
void testcalculateLossFunctions(void);
void testcalculateLossFunctionsAdjoint(void);
void testcalculateGaussianKernel(void);
void testcalculateMetric(void);
void testcalculateMetricAdjoint(void);
void testcalculateKernelRegressionWeights(void);
void testcalculateGaussianKernelAdjoint(void);
void testcalculateKernelRegressionWeightsAdjoint(void);
void testcalculateLossFunctionAdjointL2(void);
void testcalculateLossFunctionAdjointL1(void);
void testKernelRegressionTrain(void);

#endif
