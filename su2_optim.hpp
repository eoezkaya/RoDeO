#ifndef SU2_OPTIM_HPP
#define SU2_OPTIM_HPP



#include <armadillo>



using namespace arma;


typedef struct {
	rowvec dv;           /* design vector (normalized)     */
	vec dv_original;     /* design vector (not normalized) */
	double EI;
	double J;
	double CL;
	double CD;
	double Area;
} MC_design;





void su2_optimize(void);

int call_SU2_CFD_Solver(vec &dv,
		               double &CL,
		               double &CD,
		               double &area);

int call_SU2_Adjoint_Solver(vec &dv,
		vec &gradient,
		double &CL,
		double &CD,
		double &area
);

void initial_data_acquisition(int number_of_initial_samples );

void su2_try_NACA0012_classic_solution(void);

#endif
