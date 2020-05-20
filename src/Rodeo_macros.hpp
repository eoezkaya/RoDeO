
#ifndef RODEO_MACROS_HPP
#define RODEO_MACROS_HPP





enum LOSS_FUNCTION {
	L1_LOSS_FUNCTION,
	L2_LOSS_FUNCTION};


enum SURROGATE_MODEL {
	LINEAR_REGRESSION,
	KRIGING};


#define LARGE 10E14
#define EPSILON 10-14
#define EPSILON_SINGLE 10E-6


//#define MIN(x,y) ((x) <= (y) ? x : y)
//#define MAX(x,y) ((x) <= (y) ? y : x)

#endif
