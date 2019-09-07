#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>
#include "kriging_training.hpp"
#include "linear_regression.hpp"
#include "auxilliary_functions.hpp"
#include "Rodeo_macros.hpp"

#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>

using namespace arma;

/* global variables */

int total_number_of_function_evals;

double population_overall_max = -10E14;
int population_overall_max_tread_id = -1;


/* print all the population (can be a lot of mess!)*/

void print_population(std::vector<member> population){
	std::vector<member>::iterator it;

	printf("population = \n");

	for (it = population.begin() ; it != population.end(); ++it){
		it->print();

	}
	printf("\n");

}




/*
 * Correlation function R(x^i,x^j)
 *
 * R(x^i,x^j)=exp(-sum_{k=1}^p (  theta_k* ( abs(x^i_k-x^j_k)**gamma_k  ) )  )
 * @param[in] x_i
 * @param[in] X_j
 * @param[in] theta
 * @param[in] gamma
 * @return R
 *
 * */
double compute_R(rowvec x_i, rowvec x_j, vec theta, vec gamma) {

	int dim = theta.size();

	double sum = 0.0;
	for (int k = 0; k < dim; k++) {

		sum += theta(k) * pow(fabs(x_i(k) - x_j(k)), gamma(k));

	}

	return exp(-sum);
}



/*
 * Correlation function R(x^i,x^j) with gamma = 2.0
 *
 * R(x^i,x^j)=exp(-sum_{k=1}^p (  theta_k* ( abs(x^i_k-x^j_k)**2.0  ) )  )
 * @param[in] x_i
 * @param[in] X_j
 * @param[in] theta
 * @return R
 *
 * */
double compute_R_Gauss(rowvec x_i,
		rowvec x_j,
		vec theta) {

	int dim = theta.size();

	double sum = 0.0;
	for (int k = 0; k < dim; k++) {

		sum += theta(k) * pow(fabs(x_i(k) - x_j(k)), 2.0);

	}

	return exp(-sum);
}



/*
 *
 *
 * derivative of R(x^i,x^j) w.r.t. x^i_k (for GEK)
 *
 *
 * */

double compR_dxi(rowvec x_i, rowvec x_j, vec theta, int k) {

	int dim = theta.size();
	double sum = 0.0;
	double result;



	/* first compute R(x^i,x^j) */
	for(int m=0;m<dim;m++){
		sum+=theta(m)*pow( fabs(x_i(m)-x_j(m)),2.0 );
	}
	sum=exp(-sum);
	result= -2.0*theta(k)* (x_i(k)-x_j(k))* sum;
	return result;
}





/*
 *
 *
 * derivative of R(x^i,x^j) w.r.t. x^j_k (for GEK)
 *
 *
 *
 * */

double compR_dxj(rowvec x_i, rowvec x_j, vec theta,  int k) {

	int dim = theta.size();
	double sum = 0.0;
	double result;


	/* first compute R(x^i,x^j) */
	for(int m=0;m<dim;m++){
		sum+=theta(m)*pow( fabs(x_i(m)-x_j(m)),2.0 );
	}
	sum=exp(-sum);

	result= 2.0*theta(k)* (x_i(k)-x_j(k))* sum;

	return result;
}


/*
 *
 * second derivative of R(x^i,x^j) w.r.t. x^i_l and x^j_k (hand derivation)
 * (for GEK)
 *
 * */

double compute_dr_dxi_dxj(rowvec x_i, rowvec x_j, vec theta,int l,int k){

	int dim = theta.size();
	double corr = 0.0;
	double dx;

	for (int i = 0;i<dim;i++){

		corr = corr + theta(i) * pow(fabs(x_i(i)-x_j(i)),2.0);
	}

	corr = exp(-corr);

	if (k == l){

		dx = 2.0*theta(k)*(-2.0*theta(k)*pow((x_i(k)-x_j(k)),2.0)+1.0)*corr;
	}
	if (k != l) {

		dx = -4.0*theta(k)*theta(l)*(x_i(k)-x_j(k))*(x_i(l)-x_j(l))*corr;
	}

	return dx;
}


/*
 *
 * Computation of the correlation matrix using standart correlation function
 *
 *
 * */

void compute_R_matrix(vec theta,
		vec gamma,
		double reg_param,
		mat &R,
		mat &X) {
	double temp;
	int nrows = R.n_rows;

	R.fill(0.0);


	for (int i = 0; i < nrows; i++) {
		for (int j = i + 1; j < nrows; j++) {

			temp = compute_R(X.row(i), X.row(j), theta, gamma);
			R(i, j) = temp;
			R(j, i) = temp;
		}

	}

	R = R + eye(nrows,nrows) + eye(nrows,nrows)*reg_param;

} /* end of compute_R_matrix */





/* implementation according to the Forrester book */
void compute_R_matrix_GEK(vec theta,
		double reg_param,
		mat& R,
		mat &X,
		mat &grad) {




	int k = X.n_cols;
	int n = X.n_rows;


	mat Psi=zeros(n,n);
	mat PsiDot=zeros(n,n);


	mat Rfull;

	for(int row = -1; row < k; row++){

		if(row == -1){ /* first row */

			for(int i=0; i<n;i++){
				for(int j=i+1;j<n;j++){
					Psi(i,j)=compute_R_Gauss(X.row(i),X.row(j), theta);

				}
			}

			Psi = Psi+ trans(Psi)+ eye(n,n);

			Rfull=Psi;


			PsiDot=zeros(n,n);
			for(int l=0;l<k; l++){


				for(int i=0; i<n;i++){
					for(int j=0;j<n;j++){
						PsiDot(i,j)=2.0*theta(l)* (X(i,l)-X(j,l))*Psi(i,j);

					}
				}
				Rfull = join_rows(Rfull,PsiDot);

			}

		}

		else{ /* other rows */

			mat Rrow;

			PsiDot=zeros(n,n);

			for(int i=0; i<n;i++){
				for(int j=0;j<n;j++){
					PsiDot(i,j)=-2.0*theta(row)* (X(i,row)-X(j,row))*Psi(i,j);

				}
			}



			Rrow = PsiDot;


			for(int l=0; l<k;l++){
				mat PsiDot2=zeros(n,n);

				if(l == row){
					for(int i=0; i<n;i++){
						for(int j=0;j<n;j++){
							PsiDot2(i,j)=
									(2.0*theta(l)-4.0*theta(l)*theta(l)* pow((X(i,l)-X(j,l)),2.0))*Psi(i,j);

						}
					}

				}

				else{


					for(int i=0; i<n;i++){
						for(int j=0;j<n;j++){
							PsiDot2(i,j)=
									(-4.0*theta(row)*theta(l)*(X(i,row)-X(j,row))*(X(i,l)-X(j,l)))*Psi(i,j);

						}
					}
				}


				Rrow = join_rows(Rrow,PsiDot2);

			}



			Rfull = join_cols(Rfull,Rrow);
		}




	} /* end of for loop for rows */



	R = Rfull + reg_param*eye(n*(k+1),n*(k+1));



} /* end of compute_R_matrix_GEK */





/** compute the f_tilde for the Kriging model
 * @param[in] xp : point of evaluation
 * @param[in] X : normalized data matrix
 * @param[in] beta0 = [I^T*R^-1*I]^-1 [I^T*R^-1*ys]
 * @param[in] sigma_sqr
 * @param[in] regression weights
 * @param[in] R_inv_ys_min_beta = R^-1*(ys-beta0*I)
 * @param[out] f_tilde : surrogate prediction
 */
double calculate_f_tilde(rowvec xp,
		mat &X,
		double beta0,
		vec regression_weights,
		vec R_inv_ys_min_beta,
		vec kriging_weights){


	int dim = xp.size();
#if 0
	printf("dim = %d\n",dim);
#endif

	vec r(X.n_rows);

	vec theta = kriging_weights.head(dim);
	vec gamma = kriging_weights.tail(dim);


	double f_regression = 0.0;
	double f_kriging = 0.0;


	/* if there exists linear regression part */
	if(regression_weights.size() !=0){

		for(unsigned int i=0;i<xp.size();i++){

			f_regression += xp(i)*regression_weights(i+1);
		}

		f_regression += regression_weights(0);

	}
#if 0
	printf("f_regression = %10.7f\n",f_regression);
#endif

	for(unsigned int i=0;i<X.n_rows;i++){

		r(i) = compute_R(xp, X.row(i), theta, gamma);

	}
#if 0
	printf("size of vector r = %d\n",r.size());
	printf("r = \n",f_regression);
	trans(r).print();
	printf("size of vector R_inv_ys_min_beta = %d\n",R_inv_ys_min_beta.size());
	printf("R_inv_ys_min_beta = \n",f_regression);
	trans(R_inv_ys_min_beta).print();
#endif

	f_kriging = beta0+ dot(r,R_inv_ys_min_beta);


	return f_regression+f_kriging;

}


/** compute the f_tilde and ssqr for the Kriging model
 * @param[in] xp : point of evaluation
 * @param[in] X : normalized data matrix
 * @param[in] beta0 = [I^T*R^-1*I]^-1 [I^T*R^-1*ys]
 * @param[in] sigma_sqr
 * @param[in] I : identity vector
 * @param[in] R_inv_ys_min_beta = R^-1*(ys-beta0*I)
 * @param[in] R_inv_I = R^-1* I
 * @param[in] U (Cloesky decomposition of R)
 * @param[in] L (Cloesky decomposition of R)
 * @param[out] f_tilde : surrogate prediction
 * @param[out] ssqr: kriging variance
 */
void calculate_f_tilde_and_ssqr(
		rowvec xp,
		mat &X,
		double beta0,
		double sigma_sqr,
		vec regression_weights,
		vec R_inv_ys_min_beta,
		vec R_inv_I,
		vec I,
		vec kriging_weights,
		mat &U,
		mat &L,
		double *f_tilde,
		double *ssqr){


	int dim = xp.size();
	vec r(X.n_rows);

	vec theta = kriging_weights.head(dim);
	vec gamma = kriging_weights.tail(dim);



	double f_regression = 0.0;
	double f_kriging = 0.0;


	if(regression_weights.size() !=0){

		for(unsigned int i=0;i<xp.size();i++){

			f_regression += xp(i)*regression_weights(i+1);
		}

		f_regression += regression_weights(0);

	}


	for(unsigned int i=0;i<X.n_rows;i++){

		r(i) = compute_R(xp, X.row(i), theta, gamma);

	}

	f_kriging = beta0+ dot(r,R_inv_ys_min_beta);



	*f_tilde =  f_regression+f_kriging;

	vec R_inv_r(X.n_rows);


	/* solve the linear system R x = r by Cholesky matrices U and L*/
	solve_linear_system_by_Cholesky(U, L, R_inv_r, r);


	*ssqr = sigma_sqr*( 1.0 - dot(r,R_inv_r)
			+ ( pow( (dot(r,R_inv_I) -1.0 ),2.0)) / (dot(I,R_inv_I) ) );



}


// IMPORTANT : vector x should be normalized!!!
double calculate_f_tilde_GEK(rowvec &x,
		mat &X,
		double beta0,
		vec regression_weights,
		vec R_inv_ys_min_beta,
		vec kriging_weights,
		int Ngrad){


#if 0
	printf("calculate_f_tilde_GEK...\n");
#endif

	int dim = x.size();

	int Ntotal =X.n_rows;
	int Nfunc = Ntotal-Ngrad;

	int dimension_R = Nfunc+Ngrad+ dim*Ngrad;

	vec r(dimension_R);

	vec theta = kriging_weights.head(dim);


#if 0
	theta.raw_print(cout, "theta:");
	x.raw_print(cout, "x:");
	regression_weights.raw_print(cout, "regression_weights:");
#endif



	double f_regression = 0.0;
	double f_kriging = 0.0;

	/* if there exists some regression weights, than perform linear regression */
	if (regression_weights.size() > 0){

		for(unsigned int i=0; i<x.size(); i++){

			f_regression += x(i)*regression_weights(i+1);
		}

		f_regression += regression_weights(0);
	}


	/* first Nfunc+Ngrad elements are normal functional correlations */
	for(int k=0; k<Nfunc+Ngrad; k++) {

		r(k)= compute_R_Gauss(X.row(k),x,theta);
	}

	int count=Nfunc+Ngrad;



	for(int k=Nfunc;k<Nfunc+Ngrad;k++) {

		for(int l=0;l<dim;l++) {

			r(count+l*Ngrad+k)= compR_dxi(X.row(k), x, theta, l) ;
		}
	}

	f_kriging = beta0+ dot(r,R_inv_ys_min_beta);

	return f_regression+f_kriging;

}



/* calculate the fitness of a member with the given theta (only for GEK)*/
int calculate_fitness_GEK(member &new_born,
		double reg_param,
		mat &R,
		mat &X,
		vec &ys,
		vec &F,
		mat &grad,
		int eqn_sol_method ) {

	int Mtotal = X.n_rows;    /* number of total data points */
	int Mgrad  = grad.n_rows; /* number of data points with only function value */
	int Mfunc = Mtotal-Mgrad; /* number of data points with gradient information */
	int dim = X.n_cols;
	int dimension_of_R = Mfunc+Mgrad+dim*Mgrad; /* dimension of the correlation matrix */


	double logdetR=0.0;

	mat U(dimension_of_R, dimension_of_R);
	mat D(dimension_of_R, dimension_of_R);
	mat L(dimension_of_R, dimension_of_R);
	mat V(dimension_of_R, dimension_of_R);
	mat Ut(dimension_of_R, dimension_of_R);

	vec s(dimension_of_R);
	vec sinv(dimension_of_R);


	// compute the correlation matrix R

	//	double reg_param = pow(10.0, -1.0* new_born.log_regularization_parameter );

	compute_R_matrix_GEK(new_born.theta, reg_param,R,X,grad);

	//	R.print();

#ifdef calculate_fitness_CHECK
	mat Rinv= inv(R);
#endif
	//	invR.print();

	//	mat check =(R*invR);
	//	check.print();

	//	exit(1);

	if(eqn_sol_method == CHOLESKY){
		/* compute Cholesky decomposition of the correlation matrix R */
		int flag = chol(U, R);

		if (flag == 0) {
			printf("Ill conditioned correlation matrix in Cholesky decomposition...\n");
			new_born.objective_val = -LARGE;
			total_number_of_function_evals++;
			return 0;
		}


		/* calculate upper triangular from U */
		L = trans(U);


	}


	if( eqn_sol_method == SVD){
		int flag_svd = svd( U, s, V, R );

		if (flag_svd == 0) {

			printf("Error: SVD could not be performed\n");
			exit(-1);
		}

		sinv = 1.0/s;
#if 0
		printf("inverse singular values = \n");
		sinv.print();
#endif
		double threshold = 10E-8;
		for(int i=0; i< dimension_of_R; i++){
			if(s(i)  < threshold){
				sinv(i) = 0.0;
			}

		}
#if 0
		printf("inverse singular values after thresholding= \n");
		sinv.print();
		printf("\n");
#endif

		Ut = trans(U);

		D.fill(0.0);
		for(int i=0; i< dimension_of_R; i++){

			D(i,i) = sinv(i);
		}

	}


	/* calculate the determinant of R after Cholesky decomposition*/
	if(eqn_sol_method == CHOLESKY){
		vec U_diagonal = U.diag();

		//U_diagonal.raw_print(cout, "diag(U):");

		logdetR = 0.0;
		for (int i = 0; i < dimension_of_R; i++) {
			if (U_diagonal(i) < 0) {

				new_born.objective_val = -LARGE;
				return 0;
			}

			logdetR += log(U_diagonal(i));

			//		printf("%d %30.25f %30.25f\n",i,logdetR,log(U_diagonal(i)));
		}

		logdetR = 2.0 * logdetR;

	}

	if(eqn_sol_method == SVD){

		logdetR = 0.0;
		for (int i = 0; i < dimension_of_R; i++) {

			logdetR += log(s(i));

		}

	}






	vec R_inv_ys(dimension_of_R);
	vec R_inv_F(dimension_of_R);

	if( eqn_sol_method == CHOLESKY){

		solve_linear_system_by_Cholesky(U, L, R_inv_ys, ys);
		solve_linear_system_by_Cholesky(U, L, R_inv_F, F);
	}


	if( eqn_sol_method == SVD){
		R_inv_ys =V*D*Ut*ys;
		R_inv_F  =V*D*Ut*F;
	}


#ifdef calculate_fitness_CHECK

	vec R_inv_ys_check=Rinv*ys;

	double R_inv_ys_error= norm((R_inv_ys-R_inv_ys_check), 2);

	if(R_inv_ys_error > TOL) {
		printf("R_inv_ys is wrong\n");

		for(unsigned int i=0; i<R_inv_ys.size();i++){
			printf("%10.7f %10.7f\n",R_inv_ys(i),R_inv_ys_check(i));


		}

		exit(-1);
	}

#endif



#ifdef calculate_fitness_CHECK

	vec R_inv_F_check=Rinv*F;

	//v2_check.raw_print(cout, "v2_check:");

	double R_inv_F_error= norm((R_inv_F-R_inv_F_check), 2);

	if(R_inv_F_error > TOL) {
		printf("R_inv_F is wrong\n");
		printf("R_inv_F error = %10.7f\n",R_inv_F_error);

		for(unsigned int i=0; i<R_inv_F.size();i++){
			printf("%10.7f %10.7f\n",R_inv_F(i),R_inv_F_check(i));


		}


		exit(-1);
	}

#endif

	double beta0 = (1.0/dot(F,R_inv_F)) * (dot(F,R_inv_ys));
	printf("beta0= %10.7f\n",beta0);

	vec ys_min_betaF = ys-beta0*F;

	vec R_inv_ys_min_beta(dimension_of_R);

	if( eqn_sol_method == CHOLESKY){

		solve_linear_system_by_Cholesky(U, L, R_inv_ys_min_beta, ys_min_betaF);

	}

	if( eqn_sol_method == SVD){
		R_inv_ys_min_beta = V*D*Ut*ys_min_betaF;

	}


#ifdef calculate_fitness_CHECK

	double R_inv_ys_min_beta_check_error;
	vec R_inv_ys_min_beta_check=Rinv*(ys_min_betaF);

	//v2_check.raw_print(cout, "v2_check:");

	R_inv_ys_min_beta_check_error= norm((R_inv_ys_min_beta_check-R_inv_ys_min_beta_check), 2);

	if(R_inv_ys_min_beta_check_error > TOL) {
		printf("R_inv_ys_min_beta_check is wrong\n");


		for(unsigned int i=0; i<R_inv_ys_min_beta.size();i++){
			printf("%10.7f %10.7f\n",R_inv_ys_min_beta(i),R_inv_ys_min_beta_check(i));


		}

		exit(-1);
	}

#endif

	double fac = dot(ys_min_betaF, R_inv_ys_min_beta);

	double ssqr = (1.0 / dimension_of_R) * fac;

	double obj_val;

#if 0
	printf("s^2= %10.7f\n",ssqr);
#endif

	if(ssqr > 0 ){
		obj_val = (-dimension_of_R / 2.0) * log(ssqr);
		obj_val -= 0.5 * logdetR;
	}
	else{
		new_born.objective_val = -LARGE;
		total_number_of_function_evals++;
		return 0;
	}

#if 0
	printf("\n");
	printf("objective function value = %10.7f\n",obj_val);
	printf("(-M/2.0)* log(ssqr) = %10.7f\n",(-dimension_of_R/2.0)* log(ssqr));
	printf("-0.5*logdetR = %10.7f\n",-0.5*logdetR);
	printf("\n");
#endif


	total_number_of_function_evals++;

	new_born.objective_val = obj_val;

	return 1;



}

/* calculate the fitness of a member with the given theta and gamma  */
int calculate_fitness(member &new_born,
		double &reg_param,
		mat &R,
		mat &U,
		mat &L,
		mat &X,
		vec &ys,
		vec &I) {

	//#define calculate_fitness_CHECK

	//	printf("calling calculate_fitness function...\n");
	double beta0, ssqr;
	double logdetR;
	double obj_val;

	int M = X.n_rows;  /* number of samples */
	int dimension_of_R = R.n_rows;

	// compute the correlation matrix R

	//	double reg_param = pow(10.0,-1.0*new_born.log_regularization_parameter);

	compute_R_matrix(new_born.theta,
			new_born.gamma,
			reg_param,
			R,
			X);


	//		R.raw_print(cout, "R:");


	// compute Cholesky decomposition
	int flag = chol(U, R);

	if (flag == 0) {
		printf("Ill conditioned correlation matrix\n");
		new_born.objective_val = -LARGE;
		total_number_of_function_evals++;
		return 0;
	}

	L = trans(U);

	//L.raw_print(cout, "L:");

	//U.raw_print(cout, "U:");

	vec U_diagonal = U.diag();

	//U_diagonal.raw_print(cout, "diag(U):");

	logdetR = 0.0;
	for (int i = 0; i < M; i++) {
		if (U_diagonal(i) < 0) {

			new_born.objective_val = -LARGE;
			total_number_of_function_evals++;
			return 0;
		}

		logdetR += log(U_diagonal(i));

		//		printf("%d %30.25f %30.25f\n",i,logdetR,log(U_diagonal(i)));
	}

	logdetR = 2.0 * logdetR;

	vec R_inv_ys(dimension_of_R);
	vec R_inv_I(dimension_of_R);



	solve_linear_system_by_Cholesky(U, L, R_inv_ys, ys); /* solve R x = ys */
	solve_linear_system_by_Cholesky(U, L, R_inv_I, I);   /* solve R x = I */




	beta0 = (1.0/dot(I,R_inv_I)) * (dot(I,R_inv_ys));
	//	printf("beta0= %20.15f\n",beta0);

	vec ys_min_betaI = ys-beta0*I;

	vec R_inv_ys_min_beta(dimension_of_R);



	/* solve R x = ys-beta0*I */
	solve_linear_system_by_Cholesky(U, L, R_inv_ys_min_beta, ys_min_betaI);


	double fac = dot(ys_min_betaI, R_inv_ys_min_beta);

	ssqr = (1.0 / dimension_of_R) * fac;


	if(ssqr > 0 ){
		obj_val = (-dimension_of_R / 2.0) * log(ssqr);
		obj_val -= 0.5 * logdetR;
	}
	else{
		new_born.objective_val = -LARGE;
		total_number_of_function_evals++;
		return 0;
	}

#ifdef calculate_fitness_CHECK
	printf("\n");
	printf("objective function value = %10.7f\n",obj_val);
	printf("s^2= %20.15f\n",ssqr);
	printf("(-M/2.0)* log(ssqr) = %10.7f\n",(-dimension_of_R/2.0)* log(ssqr));
	printf("-0.5*logdetR = %10.7f\n",-0.5*logdetR);
	printf("\n");
#endif


	total_number_of_function_evals++;

	new_born.objective_val = obj_val;

	return 1;
}

/* crossover function of two designs */
void crossover_kriging(member &father, member &mother, member &child) {

	int dim = father.theta.size();
	for (int i = 0; i < dim; i++) {
		//		printf("theta (m) = %10.7f theta (f) = %10.7f\n",mother.theta[i],father.theta[i]);
		//		printf("gamma (m) = %10.7f gamma (f) = %10.7f\n",mother.gamma[i],father.gamma[i]);
		child.theta(i) = random_number(father.theta(i), mother.theta(i), 4.0);
		child.gamma(i) = random_number(father.gamma(i), mother.gamma(i), 4.0);

		if (child.theta(i) < 0)
			child.theta(i) = RandomDouble(0.0, 10.0);
		if (child.gamma(i) < 0)
			child.gamma(i) = RandomDouble(0.1, 2.0);
		if (child.gamma(i) > 2.0)
			child.gamma(i) = RandomDouble(0.1, 2.0);

		//		if (i==0) printf("theta = %10.7f gamma = %10.7f\n",child.theta[i],child.gamma[i]);
	}

	/*
	child.log_regularization_parameter =
			random_number(father.log_regularization_parameter,
					mother.log_regularization_parameter, 4.0);


	if(child.log_regularization_parameter < 0 ) child.log_regularization_parameter=0.0;
	if(child.log_regularization_parameter > 14 ) child.log_regularization_parameter=14.0;

	 */


}

/* crossover function of two designs for GEK (only for theta components)*/
void crossover_GEK(member &father, member &mother, member &child) {

	int dim = father.theta.size();
	for (int i = 0; i < dim; i++) {

		child.theta(i) = random_number(father.theta(i), mother.theta(i), 4.0);


		if (child.theta(i) < 0)
			child.theta(i) = RandomDouble(0.0, 20.0);

	}
	/*

	child.log_regularization_parameter =
			random_number(father.log_regularization_parameter,
					mother.log_regularization_parameter, 4.0);


	if(child.log_regularization_parameter < 0 ) child.log_regularization_parameter=0.0;
	if(child.log_regularization_parameter > 14 ) child.log_regularization_parameter=14.0;

	 */


}




void pickup_random_pair(std::vector<member> population, int &mother,
		int &father) {

	double rand_num;
	std::vector<member>::iterator it;

	//usleep(1000000);

	rand_num = rand() % 1000000;
	rand_num = rand_num / 1000000;

	double pro_sum = 0.0;
	for (it = population.begin(); it != population.end(); ++it) {
		pro_sum = pro_sum + it->crossover_probability;

		if (rand_num < pro_sum) {

			mother = it->id;
			break;
		}
	}

	while (1) {

		rand_num = rand() % 1000000;
		rand_num = rand_num / 1000000;

		double pro_sum = 0.0;
		for (it = population.begin(); it != population.end(); ++it) {
			pro_sum = pro_sum + it->crossover_probability;

			if (rand_num < pro_sum) {

				father = it->id;
				break;
			}
		}

		if (mother != father)
			break;

	}

}

void update_population_properties(std::vector<member> &population) {
	std::vector<member>::iterator it;
	double sum_fitness;
	double max, min;
	double x_std;
	sum_fitness = 0.0;
	max = -LARGE;
	min = LARGE;

	for (it = population.begin(); it != population.end(); ++it) {

		if (it->objective_val < min)
			min = it->objective_val;
		if (it->objective_val > max)
			max = it->objective_val;

	}


	for (it = population.begin(); it != population.end(); ++it) {

		x_std = (it->objective_val - min) / (max - min);
		it->fitness = x_std * 100.0;

		sum_fitness = sum_fitness + it->fitness;

	}

	for (it = population.begin(); it != population.end(); ++it) {


		it->crossover_probability = it->fitness / sum_fitness;
	}

}



int train_kriging_response_surface(KrigingModel &model){


	int linear_regression = model.linear_regression;
	double reg_param = model.epsilon_kriging;
	int max_number_of_function_calculations = model.max_number_of_kriging_iterations;

	std::string input_file_name = model.input_filename;
	std::string file_name_hyperparameters= model.kriging_hyperparameters_filename;


	total_number_of_function_evals = 0;

	printf("\n\ntraining Kriging response surface for the data : %s\n",
			input_file_name.c_str());

	/* initialize random seed*/
	srand (time(NULL));

	/* get the number of treads */
	int number_of_treads = 1;
#pragma omp parallel
	{

		int tid = omp_get_thread_num();
		number_of_treads = omp_get_num_threads();


		if (tid == 0){
			printf("number of threads used : %d\n", number_of_treads);
			max_number_of_function_calculations = max_number_of_function_calculations/number_of_treads;
			printf("number of function evaluations per thread : %d\n", max_number_of_function_calculations);
		}

	}

	mat data; /* data matrix */


	/* data matrix input */

	bool status = data.load(input_file_name.c_str(), csv_ascii);
	if(status == true)
	{
		cout << "Data input is done" << endl;
	}
	else
	{
		cout << "Problem with data the input (cvs ascii format)" << endl;
		exit(-1);
	}


	int nrows = data.n_rows; /* number of samples */
	int ncols = data.n_cols; /* number of design variables + 1 (output) */
	int dim = ncols - 1;     /* number of design variables */


	int number_of_initial_population = dim * 16 / number_of_treads;

	if(number_of_initial_population < 100) number_of_initial_population = 100;

	printf(
			"Data has %d rows (number of data points) and %d columns (number of variables+1)...\n",
			nrows, ncols);

	if(model.dim != ncols-1){

		printf("Error: number of independent variables does not match with the input data (dim != ncols-1)\n");
		exit(-1);

	}


	mat X = data.submat(0, 0, nrows - 1, ncols - 2);


	vec ys = data.col(dim);


#if 0
	printf("X:\n");
	X.print();
	printf("ys:\n");
	ys.print();
#endif

	/* find minimum and maximum of the columns of data */

	vec x_max(dim);
	x_max.fill(0.0);

	vec x_min(dim);
	x_min.fill(0.0);

	for (int i = 0; i < dim; i++) {
		x_max(i) = data.col(i).max();
		x_min(i) = data.col(i).min();

	}

#if 0
	printf("maximum = \n");
	x_max.print();

	printf("minimum = \n");
	x_min.print();
#endif


	/* normalize data matrix */

	for (int i = 0; i < nrows; i++) {

		for (int j = 0; j < dim; j++) {

			X(i, j) = (1.0/dim)*(X(i, j) - x_min(j)) / (x_max(j) - x_min(j));
		}
	}

#if 0
	printf("Normalized data = \n");
	X.print();
#endif

	/* train linear regression */
	if (linear_regression == LINEAR_REGRESSION_ON) { // if linear regression is on


		int lin_reg_return = train_linear_regression(X, ys, model.regression_weights, 10E-6);


		mat augmented_X(nrows, dim + 1);

		for (int i = 0; i < nrows; i++) {

			for (int j = 0; j <= dim; j++) {

				if (j == 0){

					augmented_X(i, j) = 1.0;
				}
				else {

					augmented_X(i, j) = X(i, j - 1);

				}
			}
		}

		/* now update the ys vector */

		vec ys_reg = augmented_X * regression_weights;

#if 0
		for(unsigned int i=0; i<ys.size(); i++){

			printf("%10.7f %10.7f\n",ys(i),ys_reg(i));
		}
#endif
		ys = ys - ys_reg;

#if 0
		printf("Updated ys vector = \n");
#endif

		for(unsigned int i=0; i<regression_weights.size();i++ ){

			if(fabs(regression_weights(i)) > 10E5){

				printf("WARNING: Linear regression coefficients are too large= \n");
				printf("regression_weights(%d) = %10.7f\n",i,regression_weights(i));
			}

		}



	} /* end of linear regression */

	/* Kriging part starts from here*/

	/* main training loop with EA*/

#pragma omp parallel shared(ys,X, kriging_params,number_of_initial_population,nrows,dim,total_number_of_function_evals)
	{

		int tid;

		std::vector<member>::iterator it;
		std::vector <member> population;
		int population_size = 0;

		double population_max = -LARGE;
		int population_max_index;


		/* allocate the correlation matrix */
		mat Rmatrix= zeros(nrows, nrows);


		/* allocate the matrices for Cholesky decomposition */
		mat Umatrix= zeros(nrows, nrows);
		mat Lmatrix= zeros(nrows, nrows);

		/* allocate the identity vector */
		vec I = ones(nrows);


		double Jnormalization_factor = 0.0;




		for (int i = 0; i < max_number_of_function_calculations; i++) {


			if(i==0){

				member new_born;
				new_born.theta.set_size(dim);
				new_born.gamma.set_size(dim);

				if(file_exist(file_name_hyperparameters.c_str())){
#pragma omp master
					{
						printf("Hyperparameter file %s exists...\n",file_name_hyperparameters.c_str() );
					}


					mat hyper_param(1,2*dim);

					bool status = hyper_param.load(file_name_hyperparameters.c_str(), csv_ascii);
					if(status == false)
					{
						cout << "Some problem with the hyperparameter input" << endl;
						exit(-1);

					}

#if 1
#pragma omp master
					{
						printf("hyper parameters read from the file (theta; gamma)^T:\n");
						hyper_param.print();
					}
#endif

					if(hyper_param.n_cols != 2*dim){
#if 1
#pragma omp master
						{
							printf("hyper parameters do not match the problem dimensions\n");
						}
#endif
						for (int j = 0; j < dim; j++) {

							new_born.theta(j) = RandomDouble(0, 10); //theta
							new_born.gamma(j) = RandomDouble(0, 2); //gamma

						}

					}

					else{

						for(unsigned i=0; i<dim;i++) {

							if(hyper_param(0,i) < 100.0){

								new_born.theta(i) = hyper_param(0,i);
							}
							else{

								new_born.theta(i) = 1.0;
							}


						}
						for(unsigned i=dim; i<2*dim;i++) {

							if( hyper_param(0,i) < 2.0) {

								new_born.gamma(i-dim) = hyper_param(0,i);

							}
							else{

								new_born.gamma(i-dim) = 1.0;
							}
						}

					}


				}
				else{ /* assign random parameters if there is no file */

					for (int j = 0; j < dim; j++) {

						new_born.theta(j) = RandomDouble(0, 10); //theta
						new_born.gamma(j) = RandomDouble(0, 2); //gamma

					}


				}
#if 0
				new_born.theta.print();
				new_born.gamma.print();

#endif

				calculate_fitness(new_born,
						reg_param,
						Rmatrix,
						Umatrix,
						Lmatrix,
						X,
						ys,
						I);


				if (new_born.objective_val != -LARGE) {
					Jnormalization_factor = new_born.objective_val;
					new_born.objective_val = 0.0;
					new_born.id = population_size;
					population.push_back(new_born);
					population_size++;




				}


				continue;


			} /* i=0 */




			/* update population properties after initital iterations */
			if (i >= number_of_initial_population) {

				update_population_properties (population);
			}



			if (total_number_of_function_evals % 100 == 0){

				printf("\r# of function calculations =  %d\n",
						total_number_of_function_evals);
				fflush(stdout);

			}

			member new_born;
			new_born.theta.set_size(dim);
			new_born.gamma.set_size(dim);
			new_born.theta.zeros(dim);
			new_born.gamma.zeros(dim);



			if (i < number_of_initial_population) {

				for (int j = 0; j < dim; j++) {
					new_born.theta(j) = RandomDouble(0, 10); //theta
					new_born.gamma(j) = RandomDouble(0, 2); //gamma

				}

			} else {

				int father_id, mother_id;
				pickup_random_pair(population, mother_id, father_id);
				//		printf("cross-over\n");
				crossover_kriging(population[mother_id], population[father_id],
						new_born);
			}


			//		new_born.theta.print();
			//		new_born.gamma.print();


			calculate_fitness(new_born,
					reg_param,
					Rmatrix,
					Umatrix,
					Lmatrix,
					X,
					ys,
					I);


			if (new_born.objective_val != -LARGE) {

				if(Jnormalization_factor != 0){
					new_born.objective_val = new_born.objective_val-Jnormalization_factor;

				}
				else{
					Jnormalization_factor = new_born.objective_val;
					new_born.objective_val = 0.0;


				}



				new_born.id = population_size;

				population.push_back(new_born);
				population_size++;


			}

			if (population_size % 10 == 0 && i > number_of_initial_population) { //in each 10 iteration find population max
				population_max = -LARGE;
				for (it = population.begin(); it != population.end(); ++it) {

					if (it->objective_val > population_max) {
						population_max = it->objective_val;
						population_max_index = it->id;
					}

				}


			}

		} // end of for loop


		population_max= -LARGE;
		for (it = population.begin() ; it != population.end(); ++it){

			if ( it->objective_val >  population_max){
				population_max = it->objective_val;
				population_max_index = it->id;
			}


		}


#pragma omp barrier

#if 0
		printf("kriging model training is over...\n");
		printf("tread %d best design log likelihood = %10.7f\n",omp_get_thread_num(),population_max );
#endif



#pragma omp critical
		{
			if (population_max > population_overall_max){
#if 0
				printf("tread %d, population_overall_max = %10.7f\n",omp_get_thread_num(),population_overall_max);
#endif
				population_overall_max = population_max;
				population_overall_max_tread_id = omp_get_thread_num();
			}
		}


#pragma omp barrier


		tid = omp_get_thread_num();
		if(tid == population_overall_max_tread_id){


#if 1
			printf("tread %d has the best design with the log likelihood = %10.7f\n",population_overall_max_tread_id,population_overall_max );
#endif
			population.at(population_max_index).print();

			kriging_params.set_size(2*dim);

			for(int i=0;i<dim;i++) {

			model.kriging_weights(i) = population.at(population_max_index).theta(i);
			}
			for(int i=0;i<dim;i++) {

				model.kriging_weights(i+dim)= population.at(population_max_index).gamma(i);
			}

			FILE *hyperparameters_output = fopen(file_name_hyperparameters.c_str(),"w");
			for(unsigned int i=0; i<kriging_params.size()-1;i++){

				fprintf(hyperparameters_output,"%10.7f, ",kriging_params(i) );

			}
			fprintf(hyperparameters_output,"%10.7f",kriging_params(2*dim-1));
			fclose(hyperparameters_output);

		}

	} /* end of parallel section */



	vec theta = model.kriging_weights.head(dim);
	vec gamma = model.kriging_weights.tail(dim);

	compute_R_matrix(theta,
			gamma,
			reg_param,
			model.R ,
			X);






}





/*
 *  IN input_file_name
 *  IN,OUT file_name_hyperparameters: file name for the kriging hyperparameters in csv format
 *  IN linear_regression: determines if linear regression is on or off
 *  OUT regression_weights: weight vector of the linear regression
 *  OUT kriging_params: vector of hyperparemeters
 *  IN reg_param: regularization parameter of the correlation matrix
 *  IN max_number_of_function_calculations: maximum number of function evaluations for the EA
 *  IN data_file_format
 */
int train_kriging_response_surface(std::string input_file_name,
		std::string file_name_hyperparameters,
		int linear_regression,
		vec &regression_weights,
		vec &kriging_params,
		double &reg_param,
		int max_number_of_function_calculations,
		int data_file_format) {

	total_number_of_function_evals = 0;



	printf("\n\ntraining Kriging response surface for the data : %s\n",
			input_file_name.c_str());



	/* initialize random seed*/
	srand (time(NULL));

	/* get the number of treads */
	int number_of_treads = 1;
#pragma omp parallel
	{

		int tid = omp_get_thread_num();
		number_of_treads = omp_get_num_threads();


		if (tid == 0){
			printf("number of threads used : %d\n", number_of_treads);
			max_number_of_function_calculations = max_number_of_function_calculations/number_of_treads;
			printf("number of function evaluations per thread : %d\n", max_number_of_function_calculations);
		}

	}

	mat data; /* data matrix */




	/* data matrix input */

	if(data_file_format == RAW_ASCII){
		bool status= data.load(input_file_name.c_str(), raw_ascii);
		if(status == true)
		{
			//			cout << "Data input is done" << endl;
		}
		else
		{
			cout << "Problem with the data input (raw ascii format)" << endl;
			exit(-1);
		}


	}

	if(data_file_format == CSV_ASCII){
		bool status = data.load(input_file_name.c_str(), csv_ascii);
		if(status == true)
		{
			cout << "Data input is done" << endl;
		}
		else
		{
			cout << "Problem with data the input (cvs ascii format)" << endl;
			exit(-1);
		}

	}


#if 0
	/* calculate average distance in p-norm */

	double average_distance = 0;

	int number_of_distances_computed = 0;
	for(unsigned i=0;i<data.n_rows;i++){

		for(unsigned j=i+1;j<data.n_rows;j++){
			double distance = norm(( data.row(i) - data.row(j)), data.n_cols-1);
			average_distance+= distance;
			number_of_distances_computed++;

		}




	}

	average_distance = average_distance/number_of_distances_computed;

	printf("Average distance in p-norm: %10.7f\n",average_distance );






	double tolerance = average_distance/100.0;
	int number_of_rows_removed=0;


	for(unsigned i=0;i<data.n_rows;i++){

		for(unsigned j=i+1;j<data.n_rows;j++){
			double distance = norm(( data.row(i) - data.row(j)), data.n_cols-1);
			//			printf("distance : %10.7f\n",distance);
			if( distance < tolerance ){
				data.shed_row(j);
				number_of_rows_removed++;
			}




		}


	}

	if(number_of_rows_removed > 0){
		printf("redundant data is found in the data\n");
		printf("number of rows removed from data: %d\n",number_of_rows_removed);
	}


#endif

	//	data.print();




	int nrows = data.n_rows; /* number of samples */
	int ncols = data.n_cols; /* number of design variables + 1 (output) */
	int dim = ncols - 1;     /* number of design variables */




	int number_of_initial_population = dim * 16 / number_of_treads;

	if(number_of_initial_population < 100) number_of_initial_population = 100;

	//	printf("number_of_initial_population = %d\n",number_of_initial_population);


	printf(
			"Data has %d rows (number of data points) and %d columns (number of variables+1)...\n",
			nrows, ncols);

	mat X = data.submat(0, 0, nrows - 1, ncols - 2);


	vec ys = data.col(dim);



#if 0
	printf("X:\n");
	X.print();
	printf("ys:\n");
	ys.print();
#endif

	/* find minimum and maximum of the columns of data */

	vec x_max(dim);
	x_max.fill(0.0);

	vec x_min(dim);
	x_min.fill(0.0);

	for (int i = 0; i < dim; i++) {
		x_max(i) = data.col(i).max();
		x_min(i) = data.col(i).min();

	}

#if 0
	printf("maximum = \n");
	x_max.print();

	printf("minimum = \n");
	x_min.print();
#endif


	/* normalize data matrix */

	for (int i = 0; i < nrows; i++) {

		for (int j = 0; j < dim; j++) {

			X(i, j) = (1.0/dim)*(X(i, j) - x_min(j)) / (x_max(j) - x_min(j));
		}
	}

#if 0
	printf("Normalized data = \n");
	X.print();
#endif

	/* train linear regression */
	if (linear_regression == LINEAR_REGRESSION_ON) { // if linear regression is on


		int lin_reg_return = train_linear_regression(X, ys, regression_weights, 10E-6);


		mat augmented_X(nrows, dim + 1);

		for (int i = 0; i < nrows; i++) {

			for (int j = 0; j <= dim; j++) {

				if (j == 0){

					augmented_X(i, j) = 1.0;
				}
				else {

					augmented_X(i, j) = X(i, j - 1);

				}
			}
		}

		/* now update the ys vector */

		vec ys_reg = augmented_X * regression_weights;

#if 0
		for(unsigned int i=0; i<ys.size(); i++){

			printf("%10.7f %10.7f\n",ys(i),ys_reg(i));
		}
#endif
		ys = ys - ys_reg;

#if 0
		printf("Updated ys vector = \n");
#endif

		for(unsigned int i=0; i<regression_weights.size();i++ ){

			if(fabs(regression_weights(i)) > 10E5){

				printf("WARNING: Linear regression coefficients are too large= \n");
				printf("regression_weights(%d) = %10.7f\n",i,regression_weights(i));
			}

		}



	} /* end of linear regression */

	/* Kriging part starts from here*/

	/* main training loop with EA*/

#pragma omp parallel shared(ys,X, kriging_params,number_of_initial_population,nrows,dim,total_number_of_function_evals)
	{

		int tid;

		std::vector<member>::iterator it;
		std::vector <member> population;
		int population_size = 0;

		double population_max = -LARGE;
		int population_max_index;


		/* allocate the correlation matrix */
		mat Rmatrix= zeros(nrows, nrows);


		/* allocate the matrices for Cholesky decomposition */
		mat Umatrix= zeros(nrows, nrows);
		mat Lmatrix= zeros(nrows, nrows);

		/* allocate the identity vector */
		vec I = ones(nrows);


		double Jnormalization_factor = 0.0;




		for (int i = 0; i < max_number_of_function_calculations; i++) {


			if(i==0){

				member new_born;
				new_born.theta.set_size(dim);
				new_born.gamma.set_size(dim);

				if(file_exist(file_name_hyperparameters.c_str())){
#pragma omp master
					{
						printf("Hyperparameter file %s exists...\n",file_name_hyperparameters.c_str() );
					}


					mat hyper_param(1,2*dim);

					bool status = hyper_param.load(file_name_hyperparameters.c_str(), csv_ascii);
					if(status == false)
					{
						cout << "Some problem with the hyperparameter input" << endl;
						exit(-1);

					}

#if 1
#pragma omp master
					{
						printf("hyper parameters read from the file (theta; gamma)^T:\n");
						hyper_param.print();
					}
#endif

					if(hyper_param.n_cols != 2*dim){
#if 1
#pragma omp master
						{
							printf("hyper parameters do not match the problem dimensions\n");
						}
#endif
						for (int j = 0; j < dim; j++) {

							new_born.theta(j) = RandomDouble(0, 10); //theta
							new_born.gamma(j) = RandomDouble(0, 2); //gamma

						}

					}

					else{

						for(unsigned i=0; i<dim;i++) {

							if(hyper_param(0,i) < 100.0){

								new_born.theta(i) = hyper_param(0,i);
							}
							else{

								new_born.theta(i) = 1.0;
							}


						}
						for(unsigned i=dim; i<2*dim;i++) {

							if( hyper_param(0,i) < 2.0) {

								new_born.gamma(i-dim) = hyper_param(0,i);

							}
							else{

								new_born.gamma(i-dim) = 1.0;
							}
						}

					}


				}
				else{ /* assign random parameters if there is no file */

					for (int j = 0; j < dim; j++) {

						new_born.theta(j) = RandomDouble(0, 10); //theta
						new_born.gamma(j) = RandomDouble(0, 2); //gamma

					}


				}
#if 0
				new_born.theta.print();
				new_born.gamma.print();

#endif

				calculate_fitness(new_born,
						reg_param,
						Rmatrix,
						Umatrix,
						Lmatrix,
						X,
						ys,
						I);


				if (new_born.objective_val != -LARGE) {
					Jnormalization_factor = new_born.objective_val;
					new_born.objective_val = 0.0;
					new_born.id = population_size;
					population.push_back(new_born);
					population_size++;




				}


				continue;


			} /* i=0 */




			/* update population properties after initital iterations */
			if (i >= number_of_initial_population) {

				update_population_properties (population);
			}



			if (total_number_of_function_evals % 100 == 0){

				printf("\r# of function calculations =  %d\n",
						total_number_of_function_evals);
				fflush(stdout);

			}

			member new_born;
			new_born.theta.set_size(dim);
			new_born.gamma.set_size(dim);
			new_born.theta.zeros(dim);
			new_born.gamma.zeros(dim);



			if (i < number_of_initial_population) {

				for (int j = 0; j < dim; j++) {
					new_born.theta(j) = RandomDouble(0, 10); //theta
					new_born.gamma(j) = RandomDouble(0, 2); //gamma

				}

			} else {

				int father_id, mother_id;
				pickup_random_pair(population, mother_id, father_id);
				//		printf("cross-over\n");
				crossover_kriging(population[mother_id], population[father_id],
						new_born);
			}


			//		new_born.theta.print();
			//		new_born.gamma.print();


			calculate_fitness(new_born,
					reg_param,
					Rmatrix,
					Umatrix,
					Lmatrix,
					X,
					ys,
					I);


			if (new_born.objective_val != -LARGE) {

				if(Jnormalization_factor != 0){
					new_born.objective_val = new_born.objective_val-Jnormalization_factor;

				}
				else{
					Jnormalization_factor = new_born.objective_val;
					new_born.objective_val = 0.0;


				}



				new_born.id = population_size;

				population.push_back(new_born);
				population_size++;


			}

			if (population_size % 10 == 0 && i > number_of_initial_population) { //in each 10 iteration find population max
				population_max = -LARGE;
				for (it = population.begin(); it != population.end(); ++it) {

					if (it->objective_val > population_max) {
						population_max = it->objective_val;
						population_max_index = it->id;
					}

				}


			}

		} // end of for loop


		population_max= -LARGE;
		for (it = population.begin() ; it != population.end(); ++it){

			if ( it->objective_val >  population_max){
				population_max = it->objective_val;
				population_max_index = it->id;
			}


		}


#pragma omp barrier		

#if 0
		printf("kriging model training is over...\n");
		printf("tread %d best design log likelihood = %10.7f\n",omp_get_thread_num(),population_max );
#endif



#pragma omp critical
		{
			if (population_max > population_overall_max){
#if 0
				printf("tread %d, population_overall_max = %10.7f\n",omp_get_thread_num(),population_overall_max);
#endif
				population_overall_max = population_max;
				population_overall_max_tread_id = omp_get_thread_num();
			}
		}


#pragma omp barrier			


		tid = omp_get_thread_num();
		if(tid == population_overall_max_tread_id){


#if 1
			printf("tread %d has the best design with the log likelihood = %10.7f\n",population_overall_max_tread_id,population_overall_max );
#endif
			population.at(population_max_index).print();

			kriging_params.set_size(2*dim);

			for(int i=0;i<dim;i++) kriging_params(i)     = population.at(population_max_index).theta(i);
			for(int i=0;i<dim;i++) kriging_params(i+dim)= population.at(population_max_index).gamma(i);

			FILE *hyperparameters_output = fopen(file_name_hyperparameters.c_str(),"w");
			for(unsigned int i=0; i<kriging_params.size()-1;i++){

				fprintf(hyperparameters_output,"%10.7f, ",kriging_params(i) );

			}
			fprintf(hyperparameters_output,"%10.7f",kriging_params(2*dim-1));
			fclose(hyperparameters_output);

		}

	} /* end of parallel section */

	return 0;

}


int train_kriging_response_surface(mat data,
		std::string file_name_hyperparameters,
		int linear_regression,
		vec &regression_weights,
		vec &kriging_params,
		double &reg_param,
		int max_number_of_function_calculations) {

	total_number_of_function_evals = 0;


	/* initialize random seed*/
	srand (time(NULL));

	/* get the number of treads */
	int number_of_treads = 1;
#pragma omp parallel
	{

		int tid = omp_get_thread_num();
		number_of_treads = omp_get_num_threads();


		if (tid == 0){
			printf("number of threads used : %d\n", number_of_treads);
			max_number_of_function_calculations = max_number_of_function_calculations/number_of_treads;
			printf("number of function evaluations per thread : %d\n", max_number_of_function_calculations);
		}

	}



	int nrows = data.n_rows; /* number of samples */
	int ncols = data.n_cols; /* number of design variables + 1 (output) */
	int dim = ncols - 1;     /* number of design variables */




	int number_of_initial_population = dim * 16 / number_of_treads;

	if(number_of_initial_population < 100) number_of_initial_population = 100;


	printf(
			"Data has %d rows (number of data points) and %d columns (number of variables+1)...\n",
			nrows, ncols);

	mat X = data.submat(0, 0, nrows - 1, ncols - 2);


	vec ys = data.col(dim);



#if 0
	printf("X:\n");
	X.print();
	printf("ys:\n");
	ys.print();
#endif

	/* find minimum and maximum of the columns of data */

	vec x_max(dim);
	x_max.fill(0.0);

	vec x_min(dim);
	x_min.fill(0.0);

	for (int i = 0; i < dim; i++) {
		x_max(i) = data.col(i).max();
		x_min(i) = data.col(i).min();

	}

#if 0
	printf("maximum = \n");
	x_max.print();

	printf("minimum = \n");
	x_min.print();
#endif


	/* normalize data matrix */

	for (int i = 0; i < nrows; i++) {

		for (int j = 0; j < dim; j++) {

			X(i, j) = (1.0/dim)*(X(i, j) - x_min(j)) / (x_max(j) - x_min(j));
		}
	}

#if 0
	printf("Normalized data = \n");
	X.print();
#endif

	/* train linear regression */
	if (linear_regression == LINEAR_REGRESSION_ON) { // if linear regression is on


		int lin_reg_return = train_linear_regression(X, ys, regression_weights, 10E-6);


		mat augmented_X(nrows, dim + 1);

		for (int i = 0; i < nrows; i++) {

			for (int j = 0; j <= dim; j++) {

				if (j == 0){

					augmented_X(i, j) = 1.0;
				}
				else {

					augmented_X(i, j) = X(i, j - 1);

				}
			}
		}

		/* now update the ys vector */

		vec ys_reg = augmented_X * regression_weights;

#if 0
		for(unsigned int i=0; i<ys.size(); i++){

			printf("%10.7f %10.7f\n",ys(i),ys_reg(i));
		}
#endif
		ys = ys - ys_reg;

#if 0
		printf("Updated ys vector = \n");
#endif

		for(unsigned int i=0; i<regression_weights.size();i++ ){

			if(fabs(regression_weights(i)) > 10E5){

				printf("WARNING: Linear regression coefficients are too large= \n");
				printf("regression_weights(%d) = %10.7f\n",i,regression_weights(i));
			}

		}



	} /* end of linear regression */

	/* Kriging part starts from here*/

	/* main training loop with EA*/

#pragma omp parallel shared(ys,X, kriging_params,number_of_initial_population,nrows,dim,total_number_of_function_evals)
	{

		int tid;

		std::vector<member>::iterator it;
		std::vector <member> population;
		int population_size = 0;

		double population_max = -LARGE;
		int population_max_index;


		/* allocate the correlation matrix */
		mat Rmatrix= zeros(nrows, nrows);


		/* allocate the matrices for Cholesky decomposition */
		mat Umatrix= zeros(nrows, nrows);
		mat Lmatrix= zeros(nrows, nrows);

		/* allocate the identity vector */
		vec I = ones(nrows);


		double Jnormalization_factor = 0.0;




		for (int i = 0; i < max_number_of_function_calculations; i++) {


			if(i==0){

				member new_born;
				new_born.theta.set_size(dim);
				new_born.gamma.set_size(dim);

				if(file_exist(file_name_hyperparameters.c_str())){
#pragma omp master
					{
						printf("Hyperparameter file %s exists...\n",file_name_hyperparameters.c_str() );
					}


					mat hyper_param(1,2*dim);

					bool status = hyper_param.load(file_name_hyperparameters.c_str(), csv_ascii);
					if(status == false)
					{
						cout << "Some problem with the hyperparameter input" << endl;
						exit(-1);

					}

#if 1
#pragma omp master
					{
						printf("hyper parameters read from the file (theta; gamma)^T:\n");
						hyper_param.print();
					}
#endif

					if(hyper_param.n_cols != 2*dim){
#if 1
#pragma omp master
						{
							printf("hyper parameters do not match the problem dimensions\n");
						}
#endif
						for (int j = 0; j < dim; j++) {

							new_born.theta(j) = RandomDouble(0, 10); //theta
							new_born.gamma(j) = RandomDouble(0, 2); //gamma

						}

					}

					else{

						for(unsigned int i=0; i<dim;i++) {

							if(hyper_param(0,i) < 100.0){

								new_born.theta(i) = hyper_param(0,i);
							}
							else{

								new_born.theta(i) = 1.0;
							}


						}
						for(unsigned int i=dim; i<2*dim;i++) {

							if( hyper_param(0,i) < 2.0) {

								new_born.gamma(i-dim) = hyper_param(0,i);

							}
							else{

								new_born.gamma(i-dim) = 1.0;
							}
						}

					}


				}
				else{ /* assign random parameters if there is no file */

					for (int j = 0; j < dim; j++) {

						new_born.theta(j) = RandomDouble(0, 10); //theta
						new_born.gamma(j) = RandomDouble(0, 2); //gamma

					}


				}
#if 0
				new_born.theta.print();
				new_born.gamma.print();

#endif

				calculate_fitness(new_born,
						reg_param,
						Rmatrix,
						Umatrix,
						Lmatrix,
						X,
						ys,
						I);


				if (new_born.objective_val != -LARGE) {
					Jnormalization_factor = new_born.objective_val;
					new_born.objective_val = 0.0;
					new_born.id = population_size;
					population.push_back(new_born);
					population_size++;




				}


				continue;


			} /* i=0 */




			/* update population properties after initital iterations */
			if (i >= number_of_initial_population) {

				update_population_properties (population);
			}



			if (total_number_of_function_evals % 100 == 0){

				printf("\r# of function calculations =  %d\n",
						total_number_of_function_evals);
				fflush(stdout);

			}

			member new_born;
			new_born.theta.set_size(dim);
			new_born.gamma.set_size(dim);
			new_born.theta.zeros(dim);
			new_born.gamma.zeros(dim);



			if (i < number_of_initial_population) {

				for (int j = 0; j < dim; j++) {
					new_born.theta(j) = RandomDouble(0, 10); //theta
					new_born.gamma(j) = RandomDouble(0, 2); //gamma

				}

			} else {

				int father_id, mother_id;
				pickup_random_pair(population, mother_id, father_id);
				//		printf("cross-over\n");
				crossover_kriging(population[mother_id], population[father_id],
						new_born);
			}


			//		new_born.theta.print();
			//		new_born.gamma.print();


			calculate_fitness(new_born,
					reg_param,
					Rmatrix,
					Umatrix,
					Lmatrix,
					X,
					ys,
					I);


			if (new_born.objective_val != -LARGE) {

				if(Jnormalization_factor != 0){
					new_born.objective_val = new_born.objective_val-Jnormalization_factor;

				}
				else{
					Jnormalization_factor = new_born.objective_val;
					new_born.objective_val = 0.0;


				}



				new_born.id = population_size;

				population.push_back(new_born);
				population_size++;


			}

			if (population_size % 10 == 0 && i > number_of_initial_population) { //in each 10 iteration find population max
				population_max = -LARGE;
				for (it = population.begin(); it != population.end(); ++it) {

					if (it->objective_val > population_max) {
						population_max = it->objective_val;
						population_max_index = it->id;
					}

				}


			}

		} // end of for loop


		population_max= -LARGE;
		for (it = population.begin() ; it != population.end(); ++it){

			if ( it->objective_val >  population_max){
				population_max = it->objective_val;
				population_max_index = it->id;
			}


		}


#pragma omp barrier

#if 0
		printf("kriging model training is over...\n");
		printf("tread %d best design log likelihood = %10.7f\n",omp_get_thread_num(),population_max );
#endif



#pragma omp critical
		{
			if (population_max > population_overall_max){
#if 0
				printf("tread %d, population_overall_max = %10.7f\n",omp_get_thread_num(),population_overall_max);
#endif
				population_overall_max = population_max;
				population_overall_max_tread_id = omp_get_thread_num();
			}
		}


#pragma omp barrier


		tid = omp_get_thread_num();
		if(tid == population_overall_max_tread_id){


#if 1
			printf("tread %d has the best design with the log likelihood = %10.7f\n",population_overall_max_tread_id,population_overall_max );
#endif
			population.at(population_max_index).print();

			kriging_params.set_size(2*dim);

			for(int i=0;i<dim;i++) kriging_params(i)     = population.at(population_max_index).theta(i);
			for(int i=0;i<dim;i++) kriging_params(i+dim)= population.at(population_max_index).gamma(i);

			FILE *hyperparameters_output = fopen(file_name_hyperparameters.c_str(),"w");
			for(unsigned int i=0; i<kriging_params.size()-1;i++){

				fprintf(hyperparameters_output,"%10.7f, ",kriging_params(i) );

			}
			fprintf(hyperparameters_output,"%10.7f",kriging_params(2*dim-1));
			fclose(hyperparameters_output);

		}

	} /* end of parallel section */

	return 0;

}





int train_GEK_response_surface(std::string input_file_name,
		int linear_regression,
		vec &regression_weights,
		vec &kriging_params,
		double &reg_param,
		int &max_number_of_function_calculations,
		int dim,
		int eqn_sol_method) {

	printf("training Kriging response surface for the data : %s\n",
			input_file_name.c_str());



	int number_of_treads = 1;
#pragma omp parallel
	{

		int tid = omp_get_thread_num();
		number_of_treads = omp_get_num_threads();


		if (tid == 0){
			printf("number of threads used : %d\n", number_of_treads);
			printf("number of function evaluations per thread : %d\n", max_number_of_function_calculations = max_number_of_function_calculations/number_of_treads);
		}

	}

	int number_of_initial_population = dim * 32 / number_of_treads;

	mat data_functional_values; // data matrix for only functional values
	mat data_gradients;         // data matrix for only functional values + gradient sensitivities


	std::ifstream in(input_file_name);

	if(!in) {
		printf("Cannot open input file %s...\n",input_file_name.c_str() );
		return 1;
	}


	std::vector<double> temp;



	std::string str;
	int count_f=0;
	int count_g=0;

	while (std::getline(in, str)) {
		// output the line
		//				std::cout << "line = "<<str << std::endl;


		std::string delimiter = ",";

		size_t pos = 0;
		std::string token;
		while ((pos = str.find(delimiter)) != std::string::npos) {
			token = str.substr(0, pos);
			//			std::cout << "token = "<<token << std::endl;


			temp.push_back(atof(token.c_str()));


			str.erase(0, pos + delimiter.length());
		}
		//				std::cout << "str = "<<str << std::endl;
		temp.push_back(atof(str.c_str()));

		//		std::cout<<"temp= \n";
		for (std::vector<double>::iterator it = temp.begin() ; it != temp.end(); ++it){

			//			std::cout<<*it<<std::endl;


		}


		if(temp.size() == dim+1){ // function values

			rowvec newrow(dim+1);
			int count=0;
			for (std::vector<double>::iterator it = temp.begin() ; it != temp.end(); ++it){

				//				std::cout<<*it<<std::endl;
				newrow(count)=*it;
				count++;

			}
			//			newrow.print();
			count_f++;
			data_functional_values.resize(count_f, dim+1);

			data_functional_values.row(count_f-1)=newrow;



		}
		else{ // function+gradient information

			rowvec newrow(2*dim+1);
			int count=0;
			for (std::vector<double>::iterator it = temp.begin() ; it != temp.end(); ++it){

				//				std::cout<<*it<<std::endl;
				newrow(count)=*it;
				count++;

			}
			//			newrow.print();
			count_g++;
			data_gradients.resize(count_g, 2*dim+1);

			data_gradients.row(count_g-1)=newrow;


		}




		temp.clear();

		// now we loop back and get the next line in 'str'
	}


	printf("functional values data :\n");
	data_functional_values.print();
	printf("\n");



	/* number of points with only functional values */
	int n_f_evals = data_functional_values.n_rows;
	/* number of points with functional values + gradient sensitivities*/
	int n_g_evals = data_gradients.n_rows;

	printf("number of points with only functional values = %d\n",n_f_evals);
	printf("number of points with sensitivity gradients = %d\n",n_g_evals);

	mat X; /* data matrix */
	if(n_f_evals > 0){

		X = data_functional_values.submat(0, 0, n_f_evals - 1, data_functional_values.n_cols - 2);
	}


	//		X.print();

	vec ys_func;

	if(n_f_evals > 0){

		ys_func = data_functional_values.col(dim);
	}


	if(n_g_evals > 0){

		X.insert_rows( n_f_evals , data_gradients.submat(0, 0, n_g_evals - 1, dim-1));
	}




#if 0
	printf("Original data matrix = \n");
	X.print();
	printf("\n");
#endif


	vec x_max(dim);
	x_max.fill(0.0);

	vec x_min(dim);
	x_min.fill(0.0);

	for (int i = 0; i < dim; i++) {
		x_max(i) = X.col(i).max();
		x_min(i) = X.col(i).min();

	}


#if 0
	printf("maximum = \n");
	x_max.print();
	printf("\n");
	printf("minimum = \n");
	x_min.print();
	printf("\n");


	printf("gradient data (raw) :\n");
	data_gradients.print();
	printf("\n");
#endif


	/* normalize gradient data */
	for (unsigned int i = 0; i < data_gradients.n_rows; i++) {
		for (int j = 0; j < dim; j++) {
			data_gradients(i, j) = (data_gradients(i, j) - x_min(j)) / (x_max(j) - x_min(j));
		}

		for (int j = dim+1; j < 2*dim+1; j++) {
			data_gradients(i, j) = data_gradients(i, j) * (x_max(j-dim-1) - x_min(j-dim-1));
		}
	}


	/* normalize data matrix */

	for (unsigned int i = 0; i < X.n_rows; i++) {
		for (int j = 0; j < dim; j++) {
			X(i, j) = (X(i, j) - x_min(j)) / (x_max(j) - x_min(j));
		}
	}


	printf("Normalized data = \n");
	X.print();
	printf("\n");



	printf("gradient data (normalized) :\n");
	data_gradients.print();
	printf("\n");

	//	exit(1);

	if(n_g_evals > 0){

		ys_func.insert_rows(n_f_evals, data_gradients.col(dim));
	}


	//	printf("ys_func (augmented) = \n");
	//	ys_func.print();
	//	printf("\n");




	/* matrix that holds the gradient sensitivities */
	mat grad(n_g_evals, dim);

	if(n_g_evals > 0){



		grad = data_gradients.submat(0, dim+1, n_g_evals - 1, 2*dim);

		//	grad.print();

	}


	//	printf("Normalized data = \n");
	//	X.print();

	if (linear_regression == 1) { /* if linear regression is on */

		mat augmented_X(X.n_rows, dim + 1);

		for (unsigned int i = 0; i < X.n_rows; i++) {
			for ( int j = 0; j <= dim; j++) {
				if (j == 0)
					augmented_X(i, j) = 1.0;
				else
					augmented_X(i, j) = X(i, j - 1);

			}
		}

		//		printf("Augmented data matrix = \n");
		//		augmented_X.print();

		/* add a small regularization term if there are too few samples */
		//		if (X.n_rows < dim * 10) {
		//
		//		}

		//		printf("Taking pseudo-inverse of augmented data matrix...\n");
		mat psuedo_inverse_X_augmented = pinv(augmented_X);

		//		psuedo_inverse_X_augmented.print();

		regression_weights = psuedo_inverse_X_augmented * ys_func;


		for(unsigned int i=0; i<regression_weights.size();i++ ){

			if(fabs(regression_weights(i)) > 10E5){

				printf("WARNING: Linear regression coefficients are too large= \n");
				printf("regression_weights(%d) = %10.7f\n",i,regression_weights(i));
			}

		}



		/* now update the ys vector */

		ys_func = ys_func - augmented_X * regression_weights;

		//		printf("Updated ys vector = \n");

		//		ys.print();

	}
	/* end of linear regression */


	int dimension_of_R = n_f_evals+n_g_evals+ n_g_evals*dim;

	printf("dimension of the correlation matrix R = %d\n",dimension_of_R);


	/* set ys vector for GEK */

	vec ys(dimension_of_R);

	for(int i=0; i<n_f_evals+n_g_evals; i++ ) ys(i)=ys_func(i);


	/* for each data point with gradient information*/
	int pos = n_f_evals+n_g_evals;

	for(int j=0; j<dim; j++){ /* for each design variable */
		for(int i=0; i<n_g_evals; i++ ) {


			ys(pos + n_g_evals*j+i) = grad(i,j);

		}

	}

	printf("ys (kriging) = \n");
	ys.print();
	printf("\n");




	/* Kriging part starts from here*/

	/* main training loop */

	//#pragma omp parallel shared(ys,X,grad,kriging_params,number_of_initial_population,dim,total_number_of_function_evals)
	//		{

	int tid;

	std::vector<member>::iterator it;
	std::vector < member > population;
	int population_size = 0;

	double population_max = -LARGE;
	int population_max_index;




	/* allocate the correlation matrix */
	mat Rmatrix = eye(dimension_of_R, dimension_of_R);


	/* allocate the identity vector */
	vec F(dimension_of_R);
	F.fill(0.0);

	/* set first n_f_evals+n_g_evals entries to 1 */
	for(int i=0; i<n_f_evals+n_g_evals ;i++) F(i)=1.0;


	for (int i = 0; i < max_number_of_function_calculations; i++) {

		/* update population properties after initital iterations */
		if (i >= number_of_initial_population) {
			update_population_properties (population);
			//			print_population(population);

		}

		if (total_number_of_function_evals % 50 == 0)
			printf("# of function calculations =  %d\n",
					total_number_of_function_evals);

		//			int tid = omp_get_thread_num();
		//			printf("Thread %d i= %d\n",tid,i);


		member new_born;
		new_born.theta.set_size(dim);
		new_born.theta.zeros(dim);

		if (i < number_of_initial_population) {

			for (int j = 0; j < dim; j++) {
				new_born.theta(j) = RandomDouble(0, 20); //theta
			}

			//			new_born.log_regularization_parameter = RandomDouble(0, 14);
		} else {

			int father_id, mother_id;
			pickup_random_pair(population, mother_id, father_id);

#if 0
			printf("father_id = %d\n",father_id);
			printf("mother_id = %d\n",mother_id);
			printf("cross-over\n");
#endif

			crossover_GEK(population[mother_id], population[father_id],new_born);
		}


		//		new_born.theta.print();

		//		printf("calculate_fitness_GEK...\n");
		calculate_fitness_GEK(new_born,
				reg_param,
				Rmatrix,
				X,
				ys,
				F,
				grad,
				eqn_sol_method);



		if (new_born.objective_val != -LARGE) {
			new_born.id = population_size;


			population.push_back(new_born);
			population_size++;


		}

		if (population_size % 10 == 0 && i > number_of_initial_population) { //in each 10 iteration find population max
			population_max = -LARGE;
			for (it = population.begin(); it != population.end(); ++it) {

				if (it->objective_val > population_max) {
					population_max = it->objective_val;
					population_max_index = it->id;
				}

			}

		}

	} /* end of i loop */
	population_max= -LARGE;
	for (it = population.begin() ; it != population.end(); ++it){

		if ( it->objective_val >  population_max){
			population_max = it->objective_val;
			population_max_index = it->id;
		}


	}


#pragma omp barrier		

#if 0
	printf("GEK model training is over...\n");
	printf("tread %d best design log likelihood = %10.7f\n",omp_get_thread_num(),population_max );
#endif



#pragma omp critical
	{
		if (population_max > population_overall_max){
#if 0
			printf("tread %d, population_overall_max = %10.7f\n",omp_get_thread_num(),population_overall_max);
#endif
			population_overall_max = population_max;
			population_overall_max_tread_id = omp_get_thread_num();
		}
	}


#pragma omp barrier			


	tid = omp_get_thread_num();
	if(tid == population_overall_max_tread_id){

#if 1
		printf("tread %d has the best design with log likelihood = %10.7f\n",population_overall_max_tread_id,population_overall_max );
#endif
		population.at(population_max_index).print();

		kriging_params.set_size(dim);

		for(int i=0;i<dim;i++) kriging_params(i)= population.at(population_max_index).theta(i);

	}


	return 0;

}


void compute_R_inv_ys_min_beta(mat X,
		vec ys,
		vec kriging_params,
		vec regression_weights,
		vec &res,
		double &beta0,
		double epsilon_kriging,
		int linear_regression){

	int N= ys.size();
	int d= X.n_cols;

	vec theta = kriging_params.head(d);
	vec gamma = kriging_params.tail(d);

	mat R=zeros(N,N);
	mat U=zeros(N,N);

	compute_R_matrix(theta,
			gamma,
			epsilon_kriging,
			R,
			X);
#if 0
	printf("R =\n");
	R.print();
#endif


	mat augmented_X(N, d + 1);

	if(linear_regression == LINEAR_REGRESSION_ON){



		for (int i = 0; i < N; i++) {

			for (int j = 0; j <= d; j++) {

				if (j == 0){

					augmented_X(i, j) = 1.0;
				}
				else{

					augmented_X(i, j) = X(i, j - 1);

				}
			}
		}


	}


	ys = ys - augmented_X * regression_weights;
#if 0
	printf("ys =\n");
	trans(ys).print();
#endif
	/* vector of ones */
	vec I = ones(N);


	/* Cholesky decomposition R = LDL^T */

	int cholesky_return = chol(U, R);

	if (cholesky_return == 0) {
		printf("Error: Ill conditioned correlation matrix, Cholesky decomposition failed...\n");
		exit(-1);
	}

	mat L = trans(U);

	vec R_inv_ys(N);
	vec R_inv_I(N);

	solve_linear_system_by_Cholesky(U, L, R_inv_ys, ys);    /* solve R x = ys */
#if 0
	printf("R_inv_ys =\n");
	trans(R_inv_ys).print();
#endif


	solve_linear_system_by_Cholesky(U, L, R_inv_I, I);      /* solve R x = I */
#if 0
	printf("R_inv_I =\n");
	trans(R_inv_I).print();
#endif

	beta0 = (1.0/dot(I,R_inv_I)) * (dot(I,R_inv_ys));

	vec ys_min_betaI = ys - beta0*I;

	vec R_inv_ys_min_beta = zeros(N);

	/* solve R x = ys-beta0*I */
	solve_linear_system_by_Cholesky(U, L, R_inv_ys_min_beta , ys_min_betaI);


	res = R_inv_ys_min_beta;


}





