#include <math.h>
double constraintFunction3(double* x){
return pow(x[0],2) + sin(x[1]) - 2*x[3];
}
