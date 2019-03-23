#include "kernel_regression.hpp"
#include "auxilliary_functions.hpp"
#include "Rodeo_macros.hpp"
#include "test_functions.hpp"

#include <armadillo>
#include <codi.hpp>

#include<iostream>
#include<fstream>


using namespace arma;




/*
 * calculates the generalized Mahalanobis distance between two points
 *
 * @param[in] x_i : first vector
 * @param[in] X_j : second vector
 * @param[in] M : dim x dim matrix
 * @param[in] dim
 * @return distance
 *
 * */

double calcMetric(double *xi, double *xj, double ** M, int dim) {

#if 0
	printf("calling calcMetric (primal)...\n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", M[i][j]);

		}
		printf("\n");
	}

#endif



	double *diff = new double[dim];

	for (int i = 0; i < dim; i++) {

		diff[i] = xi[i] - xj[i];
	}
#if 0
	rowvec xi_val(dim);
	rowvec xj_val(dim);
	rowvec diff_val(dim);
	mat M_val(dim, dim);

	for (int i = 0; i < dim; i++) {
		xi_val(i) = xi[i];
		xj_val(i) = xj[i];
	}

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++)
			M_val(i, j) = M[i][j];

	diff_val = xi_val - xj_val;

	printf("diff_val=\n");
	diff_val.print();

	colvec diffT = trans(diff_val);

	vec matVecProd = M_val * diffT;
	printf("M * xdiff = \n");
	matVecProd.print();

	double metric_val = dot(diff_val, M_val * diffT);

	printf("metric_val = %10.7f\n", metric_val);
#endif

	double *tempVec = new double[dim];

	double sum = 0.0;

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			sum = sum + M[i][j] * diff[j];
		}

		tempVec[i] = sum;
		sum = 0.0;

	}
#if 0
	printf("tempVec = \n");
	for(int i=0; i<dim; i++) {
		printf("%10.7f \n",tempVec[i] );

	}
#endif

	sum = 0.0;

	for (int i = 0; i < dim; i++) {

		sum = sum + tempVec[i] * diff[i];
	}
#if 0
	printf("sum = %10.7f\n",sum);
#endif

	delete[] diff;
	delete[] tempVec;


	if (sum < 0.0) {

		fprintf(stderr, "Error: metric is negative! at FILE = %s, LINE = %d.\n",__FILE__, __LINE__);
		exit(-1);
	}

	return sum;

}

/*
 * calculates the generalized Mahalanobis distance between two points
 * (differentiated in reverse mode )
 * @param[in] x_i : first vector
 * @param[in] X_j : second vector
 * @param[in] M : dim x dim matrix
 * @param[in] dim
 * @return distance
 *
 * */

codi::RealReverse calcMetric(double *xi, double *xj, codi::RealReverse **M,
		int dim) {

#if 0
	printf("calling calcMetric (adjoint)...\n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", M[i][j].getValue());

		}
		printf("\n");
	}

#endif

	codi::RealReverse *diff = new codi::RealReverse[dim];

	for (int i = 0; i < dim; i++) {

		diff[i] = xi[i] - xj[i];
	}
#if 0
	rowvec xi_val(dim);
	rowvec xj_val(dim);
	rowvec diff_val(dim);
	mat M_val(dim, dim);

	for (int i = 0; i < dim; i++) {
		xi_val(i) = xi[i];
		xj_val(i) = xj[i];
	}

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) {

			M_val(i, j) = M[i][j].getValue();
		}

	diff_val = xi_val - xj_val;

	printf("diff_val=\n");
	diff_val.print();

	colvec diffT = trans(diff_val);

	vec matVecProd = M_val * diffT;
	printf("M * xdiff = \n");
	matVecProd.print();

	double metric_val = dot(diff_val, M_val * diffT);

	printf("metric_val = %10.7f\n", metric_val);
#endif

	codi::RealReverse *tempVec = new codi::RealReverse[dim];

	codi::RealReverse sum = 0.0;

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			sum = sum + M[i][j] * diff[j];
		}

		tempVec[i] = sum;
		sum = 0.0;

	}
#if 0
	printf("tempVec = \n");
	for (int i = 0; i < dim; i++) {
		printf("%10.7f \n", tempVec[i].getValue());

	}
#endif

	sum = 0.0;

	for (int i = 0; i < dim; i++) {

		sum = sum + tempVec[i] * diff[i];
	}
#if 0
	printf("sum = %10.7f\n", sum.getValue());
#endif

	delete[] diff;
	delete[] tempVec;


	if (sum < 0.0) {

		fprintf(stderr, "Error: metric is negative! at %s, line %d.\n",__FILE__, __LINE__);
		fprintf(stderr, "metric val = %10.7f\n",sum.getValue());


		printf("M = \n");

		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {

				printf("%10.7f ", M[i][j].getValue());

			}
			printf("\n");
		}


		exit(-1);
	}

	return sum;

}

/*
 * calculates the generalized Mahalanobis distance between two points
 * (differentiated in forward mode )
 * @param[in] x_i : first vector
 * @param[in] X_j : second vector
 * @param[in] M : dim x dim matrix
 * @param[in] dim
 * @return distance
 *
 * */
codi::RealForward calcMetric(double *xi, double *xj, codi::RealForward ** M,
		int dim) {
#if 0
	printf("calling calcMetric...\n");
#endif

	codi::RealForward *diff = new codi::RealForward[dim];

	for (int i = 0; i < dim; i++) {

		diff[i] = xi[i] - xj[i];
	}

	codi::RealForward *tempVec = new codi::RealForward[dim];

	codi::RealForward sum = 0.0;

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			sum = sum + M[i][j] * diff[j];
		}

		tempVec[i] = sum;
		sum = 0.0;

	}
	sum = 0.0;

	for (int i = 0; i < dim; i++) {

		sum = sum + tempVec[i] * diff[i];
	}
#if 0
	printf("sum = %10.7f\n",sum.getValue());
#endif

	delete[] diff;
	delete[] tempVec;



	if (sum < 0.0) {

		fprintf(stderr, "Error: metric is negative! at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);
	}

	return sum;

}

double gaussianKernel(rowvec &xi, rowvec &xj, double sigma, mat &M) {
#if 0
	printf("calling gaussianKernel...\n");
	xi.print();
	xj.print();
#endif

	/* calculate distance between xi and xj with the matrix M */
	double metricVal = calcMetric(xi, xj, M);
#if 0
	printf("metricVal = %10.7f\n",metricVal);
#endif

	double sqr_two_pi = sqrt(2.0 * datum::pi);

	double kernelVal = (1.0 / (sigma * sqr_two_pi))* exp(-metricVal / (2 * sigma * sigma));

	kernelVal += 10E-14;

#if 0
	printf("kernelVal = %10.7f\n",kernelVal);

#endif
	return kernelVal;

}

double gaussianKernel(double *xi, double *xj, double sigma, double **M,
		int dim) {

#if 0
	printf("calling gaussianKernel...\n");
#endif

	/* calculate distance between xi and xj with the matrix M */
	double metricVal = calcMetric(xi, xj, M, dim);
#if 0
	printf("metricVal = %10.7f\n",metricVal);
#endif

	double sqr_two_pi = sqrt(2.0 * datum::pi);

	double kernelVal = (1.0 / (sigma * sqr_two_pi))* exp(-metricVal / (2 * sigma * sigma));
#if 0
	printf("kernelVal = %10.7f\n",kernelVal);

#endif

	if(isnan(kernelVal)){

		fprintf(stderr, "Error: kernel value is NaN! at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);
	}

	if(kernelVal < 0.0){

		fprintf(stderr, "Error: kernel value is negative! at %s, line %d.\n",__FILE__, __LINE__);
		exit(-1);
	}

	kernelVal += 10E-14;


	return kernelVal;

}

codi::RealReverse gaussianKernel(double *xi, double *xj,
		codi::RealReverse sigma, codi::RealReverse **M, int dim) {

#if 0
	printf("calling gaussianKernel...\n");
#endif

	/* calculate distance between xi and xj with the matrix M */
	codi::RealReverse metricVal = calcMetric(xi, xj, M, dim);
#if 0
	printf("metricVal = %10.7f\n",metricVal.getValue());
#endif

	double sqr_two_pi = sqrt(2.0 * datum::pi);

	codi::RealReverse kernelVal = (1.0 / (sigma * sqr_two_pi))* exp(-metricVal / (2 * sigma * sigma));

	if(isnan(kernelVal.getValue())){

		fprintf(stderr, "Error: kernel value is NaN! at %s, line %d.\n",__FILE__, __LINE__);

		printf("sigma = %10.7f\n",sigma.getValue());

		printf("M = \n");

		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {

				printf("%10.7f ", M[i][j].getValue());

			}
			printf("\n");
		}


		exit(-1);
	}



	kernelVal += 10E-14;
	//	printf("EPSILON = %10.7f ", EPSILON);

	if(kernelVal.getValue() < 0.0){

		fprintf(stderr, "Error: kernel value is negative or zero! at %s, line %d.\n",__FILE__, __LINE__);
		fprintf(stderr, "kernelVal = %20.15f\n",kernelVal.getValue() );
		fprintf(stderr, "metric val = %20.15f\n",metricVal.getValue());
		fprintf(stderr, "sigma = %20.15f\n",sigma.getValue());
		fprintf(stderr, "exp(-metricVal / (2 * sigma * sigma)) = %20.15f\n",exp(-metricVal / (2 * sigma * sigma)).getValue());



		printf("M = \n");

		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {

				printf("%10.7f ", M[i][j].getValue());

			}
			printf("\n");
		}



		exit(-1);
	}


#if 0
	printf("kernelVal = %10.7f\n",kernelVal.getValue());
#endif
	return kernelVal;

}

codi::RealForward gaussianKernel(double *xi, double *xj,
		codi::RealForward sigma, codi::RealForward **M, int dim) {

#if 0
	printf("calling gaussianKernel...\n");
#endif

	/* calculate distance between xi and xj with the matrix M */
	codi::RealForward metricVal = calcMetric(xi, xj, M, dim);
#if 0
	printf("metricVal = %10.7f\n",metricVal);
#endif

	double sqr_two_pi = sqrt(2.0 * datum::pi);

	codi::RealForward kernelVal = (1.0 / (sigma * sqr_two_pi))
																																																																																																							* exp(-metricVal / (2 * sigma * sigma));
#if 0
	printf("kernelVal = %10.7f\n",kernelVal);
#endif
	return kernelVal;

}

double SIGN(double a, double b) {

	if (b >= 0.0) {
		return fabs(a);
	} else {
		return -fabs(a);
	}
}

codi::RealReverse SIGN(codi::RealReverse a, codi::RealReverse b) {

	if (b >= 0.0) {
		return fabs(a);
	} else {

		return -fabs(a);
	}
}

codi::RealForward SIGN(codi::RealForward a, codi::RealForward b) {

	if (b >= 0.0) {
		return fabs(a);
	} else {

		return -fabs(a);
	}
}

double PYTHAG(double a, double b) {
	double at = fabs(a), bt = fabs(b), ct, result;

	if (at > bt) {
		ct = bt / at;
		result = at * sqrt(1.0 + ct * ct);
	} else if (bt > 0.0) {
		ct = at / bt;
		result = bt * sqrt(1.0 + ct * ct);
	} else
		result = 0.0;
	return (result);
}

codi::RealReverse PYTHAG(codi::RealReverse a, codi::RealReverse b) {
	codi::RealReverse at = fabs(a), bt = fabs(b), ct, result;

	if (at > bt) {
		ct = bt / at;
		result = at * sqrt(1.0 + ct * ct);
	} else if (bt > 0.0) {
		ct = at / bt;
		result = bt * sqrt(1.0 + ct * ct);
	} else
		result = 0.0;
	return (result);
}
codi::RealForward PYTHAG(codi::RealForward a, codi::RealForward b) {
	codi::RealForward at = fabs(a), bt = fabs(b), ct, result;

	if (at > bt) {
		ct = bt / at;
		result = at * sqrt(1.0 + ct * ct);
	} else if (bt > 0.0) {
		ct = at / bt;
		result = bt * sqrt(1.0 + ct * ct);
	} else
		result = 0.0;
	return (result);
}





double dsvd(mat &Lin, mat& data, double &sigma, double wLoss,double wSvd, double w12) {
	int flag, i, its, j, jj, k, l = 0, nm;
	double c, f, h, s, x, y, z;
	double anorm = 0.0, g = 0.0, scale = 0.0;
#if 0
	printf("Lin = \n");
	Lin.print();
#endif

	int m = Lin.n_rows;
	int n = Lin.n_cols;

	int dim = n;
	int N = data.n_rows;

	double **a;
	a = new double*[n];

	for (i = 0; i < n; i++) {

		a[i] = new double[n];
	}


	double **L;
	L = new double*[dim];
	for (int i = 0; i < dim; i++) {
		L[i] = new double[dim];

	}
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) {
			L[i][j]=Lin(i,j);

		}

	double **LT;
	LT = new double*[dim];
	for (int i = 0; i < dim; i++) {
		LT[i] = new double[dim];

	}

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) LT[i][j]=0.0;




	for (int i = 0; i < dim; i++) {
		for (int j = 0; j <= i; j++){

			LT[j][i] = L[i][j];
		}


	}
#if 0
	printf("LT = \n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", LT[i][j]);

		}
		printf("\n");
	}

#endif

	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
		{
			a[i][j]=0;
		}

	/* Multiplying matrix L and LT and storing in M */
	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
			for(int k = 0; k < dim; ++k)
			{
				a[i][j] += L[i][k] * LT[k][j];

			}
#if 1
	printf("a = \n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", a[i][j]);

		}
		printf("\n");
	}

#endif

#if 0
	mat Lval(dim,dim);
	mat LTval(dim,dim);
	mat aval(dim,dim);
	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) {
			Lval(i,j) = Lin(i,j);
		}
	LTval = trans(Lval);
	aval = Lval*LTval;

	printf("aval = \n");
	aval.print();

#endif







#if 0
	printf("Data matrix = \n");
	data.print();
#endif

	mat X = data.submat(0, 0, N - 1, dim - 1);

#if 0
	printf("X = \n");
	X.print();
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

	printf("minimum = \n");
	x_min.print();
#endif
	/* normalize data */
	for (int i = 0; i < N; i++) {

		for (int j = 0; j < dim; j++) {

			X(i, j) = (1.0 / dim) * (X(i, j) - x_min(j))
																																																																																																									/ (x_max(j) - x_min(j));
		}
	}

#if 0
	printf("X(normalized) = \n");
	X.print();
#endif

	vec ys;

	ys = data.col(dim);

#if 0
	printf("ys = \n");
	ys.print();
#endif

	double maxys = max(ys);
	for (unsigned int i = 0; i < ys.size(); i++) {

		ys(i)=ys(i)/maxys;

	}


#if 0
	printf("ys (after normalization)= \n");
	ys.print();
#endif


#if 0
	printf("m = %d, n = %d, N= %d, dim= %d\n",m,n,N,dim);
#endif
#if 0
	printf("ys = \n");
	ys.print();
#endif

	if (m != n) {

		fprintf(stderr, "Error: The matrix M is not square!\n");
		exit(-1);

	}

	if (m < n) {
		fprintf(stderr, "#rows must be > #cols \n");
		exit(-1);
	}


	double *xp = new double[dim];
	double *xi = new double[dim];

	double *kernelVal = new double[N];

	double lossFunc = 0.0;
	for (i = 0; i < N; i++) {

#if 0
		printf("kernel regression for the sample number %d\n",i);

#endif
		for (k = 0; k < dim; k++) {

			xp[k] = X(i, k);
		}

		double kernelSum = 0.0;
		for (j = 0; j < N; j++) {

			if (i != j) {

				for (k = 0; k < dim; k++) {

					xi[k] = X(j, k);
				}
				kernelVal[j] = gaussianKernel(xi, xp, sigma, a, dim);
				kernelSum += kernelVal[j];
#if 0
				printf("kernelVal[%d]=%10.7f\n",j,kernelVal[j]);
#endif
			}
		}

		double fApprox = 0.0;
		for (int j = 0; j < N; j++) {
			if (i != j) {
				fApprox += kernelVal[j] * ys(j);
#if 0
				printf("kernelVal[%d] * ys(%d) = %10.7f\n",j,j,kernelVal[j]*ys(j));
#endif

			}
		}

		fApprox = fApprox / kernelSum;

#if 0
		printf("fApprox = %10.7f\n",fApprox);
		printf("fExact = %10.7f\n",ys(i));
#endif

		lossFunc += (fApprox - ys(i)) * (fApprox - ys(i));

	} // end of i loop

	lossFunc = lossFunc / N;

#if 0
	printf("lossFunc = %10.7f\n", lossFunc);
#endif

	/* SVD part */

	double **v;
	v = new double*[n];

	for (i = 0; i < n; i++) {

		v[i] = new double[n];
	}
	double *w = new double[n];

	double *rv1 = new double[n];

	/* Householder reduction to bidiagonal form */
	for (i = 0; i < n; i++) {
		/* left-hand reduction */
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m) {
			for (k = i; k < m; k++)
				scale += fabs(a[k][i]);
			if (scale) {
				for (k = i; k < m; k++) {
					a[k][i] = (a[k][i] / scale);
					s += (a[k][i] * a[k][i]);
				}
				f = a[i][i];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][i] = (f - g);
				if (i != n - 1) {
					for (j = l; j < n; j++) {
						for (s = 0.0, k = i; k < m; k++)
							s += (a[k][i] * a[k][j]);
						f = s / h;
						for (k = i; k < m; k++)
							a[k][j] += (f * a[k][i]);
					}
				}
				for (k = i; k < m; k++)
					a[k][i] = (a[k][i] * scale);
			}
		}
		w[i] = (scale * g);

		/* right-hand reduction */
		g = s = scale = 0.0;
		if (i < m && i != n - 1) {
			for (k = l; k < n; k++)
				scale += fabs(a[i][k]);
			if (scale) {
				for (k = l; k < n; k++) {
					a[i][k] = (a[i][k] / scale);
					s += (a[i][k] * a[i][k]);
				}
				f = a[i][l];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][l] = (f - g);
				for (k = l; k < n; k++)
					rv1[k] = a[i][k] / h;
				if (i != m - 1) {
					for (j = l; j < m; j++) {
						for (s = 0.0, k = l; k < n; k++)
							s += (a[j][k] * a[i][k]);
						for (k = l; k < n; k++)
							a[j][k] += (s * rv1[k]);
					}
				}
				for (k = l; k < n; k++)
					a[i][k] = (a[i][k] * scale);
			}
		}
		anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}

	/* accumulate the right-hand transformation */
	for (i = n - 1; i >= 0; i--) {
		if (i < n - 1) {
			if (g) {
				for (j = l; j < n; j++)
					v[j][i] = ((a[i][j] / a[i][l]) / g);
				/* double division to avoid underflow */
				for (j = l; j < n; j++) {
					for (s = 0.0, k = l; k < n; k++)
						s += (a[i][k] * v[k][j]);
					for (k = l; k < n; k++)
						v[k][j] += (s * v[k][i]);
				}
			}
			for (j = l; j < n; j++)
				v[i][j] = v[j][i] = 0.0;
		}
		v[i][i] = 1.0;
		g = rv1[i];
		l = i;
	}

	/* accumulate the left-hand transformation */
	for (i = n - 1; i >= 0; i--) {
		l = i + 1;
		g = w[i];
		if (i < n - 1)
			for (j = l; j < n; j++)
				a[i][j] = 0.0;
		if (g) {
			g = 1.0 / g;
			if (i != n - 1) {
				for (j = l; j < n; j++) {
					for (s = 0.0, k = l; k < m; k++)
						s += (a[k][i] * a[k][j]);
					f = (s / a[i][i]) * g;
					for (k = i; k < m; k++)
						a[k][j] += (f * a[k][i]);
				}
			}
			for (j = i; j < m; j++)
				a[j][i] = (a[j][i] * g);
		} else {
			for (j = i; j < m; j++)
				a[j][i] = 0.0;
		}
		++a[i][i];
	}

	/* diagonalize the bidiagonal form */
	for (k = n - 1; k >= 0; k--) { /* loop over singular values */
		for (its = 0; its < 30; its++) { /* loop over allowed iterations */
			flag = 1;
			for (l = k; l >= 0; l--) { /* test for splitting */
				nm = l - 1;
				if (fabs(rv1[l]) + anorm == anorm) {
					flag = 0;
					break;
				}
				if (fabs(w[nm]) + anorm == anorm)
					break;
			}
			if (flag) {
				c = 0.0;
				s = 1.0;
				for (i = l; i <= k; i++) {
					f = s * rv1[i];
					if (fabs(f) + anorm != anorm) {
						g = w[i];
						h = PYTHAG(f, g);
						w[i] = h;
						h = 1.0 / h;
						c = g * h;
						s = (-f * h);
						for (j = 0; j < m; j++) {
							y = a[j][nm];
							z = a[j][i];
							a[j][nm] = (y * c + z * s);
							a[j][i] = (z * c - y * s);
						}
					}
				}
			}
			z = w[k];
			if (l == k) { /* convergence */
				if (z < 0.0) { /* make singular value nonnegative */
					w[k] = (-z);
					for (j = 0; j < n; j++)
						v[j][k] = (-v[j][k]);
				}
				break;
			}
			if (its >= 30) {
				delete[] rv1;
				fprintf(stderr, "No convergence after 30,000! iterations \n");
				return (0);
			}

			/* shift from bottom 2 x 2 minor */
			x = w[l];
			nm = k - 1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = PYTHAG(f, 1.0);
			f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

			/* next QR transformation */
			c = s = 1.0;
			for (j = l; j <= nm; j++) {
				i = j + 1;
				g = rv1[i];
				y = w[i];
				h = s * g;
				g = c * g;
				z = PYTHAG(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y = y * c;
				for (jj = 0; jj < n; jj++) {
					x = v[jj][j];
					z = v[jj][i];
					v[jj][j] = (x * c + z * s);
					v[jj][i] = (z * c - x * s);
				}
				z = PYTHAG(f, h);
				w[j] = z;
				if (z) {
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = (c * g) + (s * y);
				x = (c * y) - (s * g);
				for (jj = 0; jj < m; jj++) {
					y = a[jj][j];
					z = a[jj][i];
					a[jj][j] = (y * c + z * s);
					a[jj][i] = (z * c - y * s);
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
	}
	delete[] rv1;

#if 0
	printf("singular values of a=\n");

	for (i = 0; i < n; i++) {

		printf("%10.7f\n",w[i]);
	}
#endif

	double temp;
	for (i = 0; i < n; ++i) {
		for (j = i + 1; j < n; ++j) {

			if (w[i] < w[j])

			{
				temp = w[i];
				w[i] = w[j];
				w[j] = temp;
			}
		}
	}

#if 1
	printf("singular values of a=\n");



	for (i = 0; i < n; i++) {

		printf("%10.7f\n",w[i]);
	}
#endif


	double wsum = 0.0;
	for (i = 0; i < n; i++) {

		wsum += w[i];

	}

	for (i = 0; i < n; i++) {

		w[i] = w[i]/wsum;

	}

#if 1
	printf("singular values of a (normalized) with wsum =%10.7f\n",wsum);


	for (i = 0; i < n; i++) {

		printf("%15.10f\n",w[i]);
	}
#endif


	double svd_multiplier = (1.0*n*(1.0*n+1))/2.0;

	svd_multiplier = 1.0/svd_multiplier;
#if 0
	printf("svd_multiplier = %10.7f\n",svd_multiplier);
#endif
	double reg_term_svd = 0.0;

	for (i = 0; i < n; i++) {
#if 1
		printf("%d * %10.7f = %10.7f\n",i+1,w[i],(i+1)*w[i]);
#endif
		reg_term_svd = reg_term_svd + (i + 1) * w[i];
	}
#if 0
	printf("reg_term_svd = %10.7f\n",reg_term_svd);
#endif
	double reg_term_L1 = 0.0;




	for (i = 0; i < n; i++)
		for (j = 0; j <= i; j++) {

			reg_term_L1 = reg_term_L1 + L[i][j] * L[i][j];
		}
#if 0
	printf("reg_term_L1 = %10.7f\n",reg_term_L1);
#endif



	for (i = 0; i < n; i++) {
		delete[] v[i];
		delete[] a[i];
		delete[] L[i];
		delete[] LT[i];
	}

	delete[] L;
	delete[] LT;

	delete[] a;
	delete[] v;
	delete[] w;

	delete[] kernelVal;
	delete[] xp;
	delete[] xi;

	double result = wLoss*lossFunc + wSvd * svd_multiplier *reg_term_svd + w12 * reg_term_L1;
#if 0
	printf("lossFunc = %10.7f, svd part = %10.7f, L1 reg. part = %10.7f\n",
			wLoss*lossFunc, wSvd *  svd_multiplier * reg_term_svd, w12 * reg_term_L1);
#endif

#if 0
	printf("result = %10.7f\n",result);
#endif

	return (result);

}

double dsvdTL(mat &Lin, mat &data, double sigma, int indx1, int indx2,
		double *sensVal, double wLoss,double wSvd, double w12) {
	int flag, i, its, j, jj, k, l = 0, nm;
	codi::RealForward c, f, h, s, x, y, z, sigmain = sigma;
	codi::RealForward anorm = 0.0, g = 0.0, scale = 0.0;
#if 0
	printf("Lin = \n");
	Lin.print();
#endif

	int m = Lin.n_rows;
	int n = Lin.n_cols;

	int dim = n;
	int N = data.n_rows;


	codi::RealForward **a;
	a = new codi::RealForward*[n];

	for (i = 0; i < n; i++) {

		a[i] = new codi::RealForward[n];
	}



	codi::RealForward **L;
	L = new codi::RealForward*[dim];
	for (int i = 0; i < dim; i++) {
		L[i] = new codi::RealForward[dim];

	}

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) {
			L[i][j]=Lin(i,j);

		}

	sigmain.setGradient(0.0);

	if (indx1 == -1 && indx2 == -1) {

		sigmain.setGradient(1.0);

	}

	else if (indx1 >= 0 && indx2 >= 0) {

		L[indx1][indx2].setGradient(1.0);

	} else {

		fprintf(stderr, "Error: invalid indexes indx1=%d, indx2=%d\n", indx1,
				indx2);

	}



	codi::RealForward **LT;
	LT = new codi::RealForward*[dim];
	for (int i = 0; i < dim; i++) {
		LT[i] = new codi::RealForward[dim];

	}

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) LT[i][j]=0.0;

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j <= i; j++){

			LT[j][i] = L[i][j];
		}


	}

	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
		{
			a[i][j]=0;
		}

	/* Multiplying matrix L and LT and storing in M */
	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
			for(int k = 0; k < dim; ++k)
			{
				a[i][j] += L[i][k] * LT[k][j];
			}


#if 0
	printf("Data matrix = \n");
	data.print();
#endif

	mat X = data.submat(0, 0, N - 1, dim - 1);

#if 0
	printf("X = \n");
	X.print();
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

	printf("minimum = \n");
	x_min.print();
#endif
	/* normalize data */
	for (int i = 0; i < N; i++) {

		for (int j = 0; j < dim; j++) {

			X(i, j) = (1.0 / dim) * (X(i, j) - x_min(j))
																																																																																																									/ (x_max(j) - x_min(j));
		}
	}

#if 0
	printf("X(normalized) = \n");
	X.print();
#endif

	vec ys;

	ys = data.col(dim);
#if 0
	printf("ys = \n");
	ys.print();
#endif

	double maxys = max(ys);
	for (unsigned int i = 0; i < ys.size(); i++) {
		ys(i)=ys(i)/maxys;

	}


#if 0
	printf("ys (after normalization)= \n");
	ys.print();
#endif
#if 0
	printf("m = %d, n = %d, N= %d, dim= %d\n",m,n,N,dim);
#endif

	if (m != n) {

		fprintf(stderr, "Error: The matrix M is not square!\n");
		exit(-1);

	}

	if (m < n) {
		fprintf(stderr, "#rows must be > #cols \n");
		exit(-1);
	}


	double *xp = new double[dim];
	double *xi = new double[dim];

	codi::RealForward *kernelVal = new codi::RealForward[N];

	codi::RealForward lossFunc = 0.0;
	for (i = 0; i < N; i++) {

#if 0
		printf("kernel regression for the sample number %d\n",i);

#endif
		for (k = 0; k < dim; k++) {

			xp[k] = X(i, k);
		}

		codi::RealForward kernelSum = 0.0;
		for (j = 0; j < N; j++) {

			if (i != j) {

				for (k = 0; k < dim; k++) {

					xi[k] = X(j, k);
				}
				kernelVal[j] = gaussianKernel(xi, xp, sigmain, a, dim);
				kernelSum += kernelVal[j];
#if 0
				printf("kernelVal[%d]=%10.7f\n",j,kernelVal[j]);
#endif
			}
		}

		codi::RealForward fApprox = 0.0;
		for (int j = 0; j < N; j++) {
			if (i != j) {
				fApprox += kernelVal[j] * ys(j);
#if 0
				printf("kernelVal[%d] * ys(%d) = %10.7f\n",j,j,kernelVal[j]*ys(j));
#endif

			}
		}

		fApprox = fApprox / kernelSum;

#if 0
		printf("fApprox = %10.7f\n",fApprox);
		printf("fExact = %10.7f\n",ys(i));
#endif

		lossFunc += (fApprox - ys(i)) * (fApprox - ys(i));

	} // end of i loop

	lossFunc = lossFunc / N;

#if 0
	printf("lossFunc = %10.7f\n",lossFunc.getValue());
#endif

	codi::RealForward **v;
	v = new codi::RealForward *[n];

	for (i = 0; i < n; i++) {

		v[i] = new codi::RealForward[n];
	}
	codi::RealForward *w = new codi::RealForward[n];

	//	codi::RealForward * rv1 = (codi::RealForward  *)malloc((unsigned int) n*sizeof(codi::RealForward ));

	codi::RealForward * rv1 = new codi::RealForward[m];
	/* Householder reduction to bidiagonal form */
	for (i = 0; i < n; i++) {
		/* left-hand reduction */
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m) {
			for (k = i; k < m; k++)
				scale += fabs(a[k][i]);
			if (scale != 0.0) {
				for (k = i; k < m; k++) {
					a[k][i] = (a[k][i] / scale);
					s += (a[k][i] * a[k][i]);
				}
				f = a[i][i];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][i] = (f - g);
				if (i != n - 1) {
					for (j = l; j < n; j++) {
						for (s = 0.0, k = i; k < m; k++)
							s += (a[k][i] * a[k][j]);
						f = s / h;
						for (k = i; k < m; k++)
							a[k][j] += (f * a[k][i]);
					}
				}
				for (k = i; k < m; k++)
					a[k][i] = (a[k][i] * scale);
			}
		}
		w[i] = (scale * g);

		/* right-hand reduction */
		g = s = scale = 0.0;
		if (i < m && i != n - 1) {
			for (k = l; k < n; k++)
				scale += fabs(a[i][k]);
			if (scale != 0.0) {
				for (k = l; k < n; k++) {
					a[i][k] = (a[i][k] / scale);
					s += (a[i][k] * a[i][k]);
				}
				f = a[i][l];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][l] = (f - g);
				for (k = l; k < n; k++)
					rv1[k] = a[i][k] / h;
				if (i != m - 1) {
					for (j = l; j < m; j++) {
						for (s = 0.0, k = l; k < n; k++)
							s += (a[j][k] * a[i][k]);
						for (k = l; k < n; k++)
							a[j][k] += (s * rv1[k]);
					}
				}
				for (k = l; k < n; k++)
					a[i][k] = (a[i][k] * scale);
			}
		}
		anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}

	/* accumulate the right-hand transformation */
	for (i = n - 1; i >= 0; i--) {
		if (i < n - 1) {
			if (g != 0.0) {
				for (j = l; j < n; j++)
					v[j][i] = ((a[i][j] / a[i][l]) / g);
				/* double division to avoid underflow */
				for (j = l; j < n; j++) {
					for (s = 0.0, k = l; k < n; k++)
						s += (a[i][k] * v[k][j]);
					for (k = l; k < n; k++)
						v[k][j] += (s * v[k][i]);
				}
			}
			for (j = l; j < n; j++)
				v[i][j] = v[j][i] = 0.0;
		}
		v[i][i] = 1.0;
		g = rv1[i];
		l = i;
	}

	/* accumulate the left-hand transformation */
	for (i = n - 1; i >= 0; i--) {
		l = i + 1;
		g = w[i];
		if (i < n - 1)
			for (j = l; j < n; j++)
				a[i][j] = 0.0;
		if (g != 0.0) {
			g = 1.0 / g;
			if (i != n - 1) {
				for (j = l; j < n; j++) {
					for (s = 0.0, k = l; k < m; k++)
						s += (a[k][i] * a[k][j]);
					f = (s / a[i][i]) * g;
					for (k = i; k < m; k++)
						a[k][j] += (f * a[k][i]);
				}
			}
			for (j = i; j < m; j++)
				a[j][i] = (a[j][i] * g);
		} else {
			for (j = i; j < m; j++)
				a[j][i] = 0.0;
		}
		++a[i][i];
	}

	/* diagonalize the bidiagonal form */
	for (k = n - 1; k >= 0; k--) { /* loop over singular values */
		for (its = 0; its < 30; its++) { /* loop over allowed iterations */
			flag = 1;
			for (l = k; l >= 0; l--) { /* test for splitting */
				nm = l - 1;
				if (fabs(rv1[l]) + anorm == anorm) {
					flag = 0;
					break;
				}
				if (fabs(w[nm]) + anorm == anorm)
					break;
			}
			if (flag) {
				c = 0.0;
				s = 1.0;
				for (i = l; i <= k; i++) {
					f = s * rv1[i];
					if (fabs(f) + anorm != anorm) {
						g = w[i];
						h = PYTHAG(f, g);
						w[i] = h;
						h = 1.0 / h;
						c = g * h;
						s = (-f * h);
						for (j = 0; j < m; j++) {
							y = a[j][nm];
							z = a[j][i];
							a[j][nm] = (y * c + z * s);
							a[j][i] = (z * c - y * s);
						}
					}
				}
			}
			z = w[k];
			if (l == k) { /* convergence */
				if (z < 0.0) { /* make singular value nonnegative */
					w[k] = (-z);
					for (j = 0; j < n; j++)
						v[j][k] = (-v[j][k]);
				}
				break;
			}
			if (its >= 30) {
				delete[] rv1;
				fprintf(stderr, "No convergence after 30,000! iterations \n");
				return (0);
			}

			/* shift from bottom 2 x 2 minor */
			x = w[l];
			nm = k - 1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = PYTHAG(f, 1.0);
			f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

			/* next QR transformation */
			c = s = 1.0;
			for (j = l; j <= nm; j++) {
				i = j + 1;
				g = rv1[i];
				y = w[i];
				h = s * g;
				g = c * g;
				z = PYTHAG(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y = y * c;
				for (jj = 0; jj < n; jj++) {
					x = v[jj][j];
					z = v[jj][i];
					v[jj][j] = (x * c + z * s);
					v[jj][i] = (z * c - x * s);
				}
				z = PYTHAG(f, h);
				w[j] = z;
				if (z != 0.0) {
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = (c * g) + (s * y);
				x = (c * y) - (s * g);
				for (jj = 0; jj < m; jj++) {
					y = a[jj][j];
					z = a[jj][i];
					a[jj][j] = (y * c + z * s);
					a[jj][i] = (z * c - y * s);
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
	}
	//	free((void*) rv1);
	delete[] rv1;

#if 0
	printf("singular values of M=\n");

	for (i = 0; i < n; i++) {

		printf("%10.7f %10.7f\n", w[i].getValue(), w[i].getGradient());
	}
#endif

	codi::RealForward temp;
	for (i = 0; i < n; ++i) {
		for (j = i + 1; j < n; ++j) {

			if (w[i] < w[j])

			{
				temp = w[i];
				w[i] = w[j];
				w[j] = temp;
			}
		}
	}

#if 0
	printf("singular values of a=\n");

	for (i = 0; i < n; i++) {

		printf("%10.7f %10.7f\n", w[i].getValue(), w[i].getGradient());
	}
#endif

	codi::RealForward wsum = 0.0;
	for (i = 0; i < n; i++) {

		wsum += w[i];

	}

	for (i = 0; i < n; i++) {

		w[i] = w[i]/wsum;

	}

	double svd_multiplier = (1.0*n*(1.0*n+1))/2.0;

	svd_multiplier = 1.0/svd_multiplier;

	codi::RealForward reg_term_svd = 0.0;

	for (i = 0; i < n; i++) {
#if 0
		printf("%d * %10.7f = %10.7f\n",i+1,w[i].getValue(),(i+1)*w[i].getValue());
#endif
		reg_term_svd = reg_term_svd + (i + 1) * w[i];
	}
#if 0
	printf("reg_term_svd = %10.7f\n",reg_term_svd.getValue());
#endif
	codi::RealForward reg_term_L1 = 0.0;

#if 0

	printf("L.getGradient() = \n");
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {

			printf("%10.7f ", L[i][j].getGradient());

		}
		printf("\n");
	}
#endif
	for (i = 0; i < n; i++) {
		for (j = 0; j <=i; j++) {

			reg_term_L1 = reg_term_L1 + L[i][j] * L[i][j];

#if 0
			printf(
					"i=%d j=%d L[%d][%d] = %10.7f Ld[%d][%d] = %10.7f reg_term_L1d = %10.7f\n",
					i, j, i, j, L[i][j].getValue(), i, j,
					L[i][j].getGradient(), reg_term_L1.getGradient());
#endif

		}
	}
#if 0
	printf("reg_term_L1 = %10.7f\n",reg_term_L1.getValue());
#endif

	for (i = 0; i < n; i++) {
		delete[] v[i];
		delete[] a[i];
		delete[] L[i];
		delete[] LT[i];

	}
	delete[] a;
	delete[] L;
	delete[] LT;
	delete[] v;
	delete[] w;

	delete[] kernelVal;
	delete[] xp;
	delete[] xi;


	codi::RealForward result = wLoss*lossFunc + wSvd * svd_multiplier *reg_term_svd
			+ w12 * reg_term_L1;

#if 1
	printf("result = %10.7f\n", result.getValue());
	printf("resultd = %10.7f\n", result.getGradient());
#endif

	*sensVal = result.getGradient();
	return (result.getValue());

}

double dsvdAdj(int iteration,mat &Lin, mat &data, double &sigma, mat &Mgradient, double &sigmab,
		double wLoss,double wSvd, double w12) {
	int flag, i, its, j, jj, k, l = 0, nm;
	codi::RealReverse c, f, h, s, x, y, z;
	codi::RealReverse anorm = 0.0, g = 0.0, scale = 0.0;

	codi::RealReverse sigmain = sigma;

#if 0
	printf("Lin = \n");
	Lin.print();
#endif



	int m = Lin.n_rows;
	int n = Lin.n_cols;

	int dim = n;
	int N = data.n_rows;
#if 0
	printf("m = %d, n = %d, N= %d, dim= %d\n",m,n,N,dim);
#endif

#if 0
	printf("Data matrix = \n");
	data.print();
#endif

	mat X = data.submat(0, 0, N - 1, dim - 1);

#if 0
	printf("X = \n");
	X.print();
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

	printf("minimum = \n");
	x_min.print();
#endif
	/* normalize data */
	for (int i = 0; i < N; i++) {

		for (int j = 0; j < dim; j++) {

			X(i, j) = (1.0 / dim) * (X(i, j) - x_min(j))
																																																																																																									/ (x_max(j) - x_min(j));
		}
	}

#if 0
	printf("X(normalized) = \n");
	X.print();
#endif

	vec ys;

	ys = data.col(dim);

#if 0
	printf("ys = \n");
	ys.print();
#endif

	/* normalize the entries of ys such that maximum value will be 1 */

	double maxys = max(ys);
	for (unsigned int i = 0; i < ys.size(); i++) {
		ys(i)=ys(i)/maxys;

	}


#if 0
	printf("ys (after normalization)= \n");
	ys.print();
#endif



	if (m != n) {

		fprintf(stderr, "Error: The matrix M is not square!\n");
		exit(-1);

	}

	if (m < n) {
		fprintf(stderr, "#rows must be > #cols \n");
		exit(-1);
	}

	codi::RealReverse **L;
	L = new codi::RealReverse*[dim];
	for (int i = 0; i < dim; i++) {
		L[i] = new codi::RealReverse[dim];

	}


	// activate tape and register input

	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
	tape.setActive();

	codi::RealReverse **a;
	a = new codi::RealReverse*[n];

	for (i = 0; i < n; i++) {

		a[i] = new codi::RealReverse[n];
	}


	for (int i = 0; i < dim; i++){
		for (int j = 0; j < dim; j++) {
			L[i][j]=Lin(i,j);
			tape.registerInput(L[i][j]);

		}
	}
#if 0
	printf("L = \n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", L[i][j].getValue());

		}
		printf("\n");
	}
#endif


	tape.registerInput(sigmain);


	codi::RealReverse **LT;
	LT = new codi::RealReverse*[dim];
	for (int i = 0; i < dim; i++) {
		LT[i] = new codi::RealReverse[dim];

	}

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) LT[i][j]=0.0;

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j <= i; j++){

			LT[j][i] = L[i][j];
		}


	}


#if 0
	printf("LT = \n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", LT[i][j].getValue());

		}
		printf("\n");
	}
#endif


	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
		{
			a[i][j]=0;
		}

	/* Multiplying matrix L and LT and storing in M */
	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
			for(int k = 0; k < dim; ++k)
			{
				a[i][j] += L[i][k] * LT[k][j];
			}

#if 0
	printf("a = \n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", a[i][j].getValue());

		}
		printf("\n");
	}
#endif


	double *xp = new double[dim];
	double *xi = new double[dim];

	codi::RealReverse *kernelVal = new codi::RealReverse[N];

	codi::RealReverse lossFunc = 0.0;
	for (i = 0; i < N; i++) {

#if 0
		printf("kernel regression for the sample number %d\n",i);

#endif
		for (k = 0; k < dim; k++) {

			xp[k] = X(i, k);
		}

		codi::RealReverse kernelSum = 0.0;
		for (j = 0; j < N; j++) {

			if (i != j) {

				for (k = 0; k < dim; k++) {

					xi[k] = X(j, k);
				}
				kernelVal[j] = gaussianKernel(xi, xp, sigmain, a, dim);
				kernelSum += kernelVal[j];
#if 0
				printf("kernelVal[%d]=%20.15f\n",j,kernelVal[j].getValue());
#endif
			}
		}


		if(kernelSum.getValue() == 0.0){

			fprintf(stderr, "Error: kernelSum is zero! at %s, line %d.\n",__FILE__, __LINE__);
			exit(-1);


		}

		codi::RealReverse fApprox = 0.0;
		for (int j = 0; j < N; j++) {
			if (i != j) {
				fApprox += kernelVal[j] * ys(j);
#if 0
				printf("kernelVal[%d] * ys(%d) = %20.15f\n",j,j,kernelVal[j].getValue()*ys(j));
				printf("fApprox = %10.7f\n",fApprox.getValue());
#endif

			}
		}

		fApprox = fApprox / kernelSum;

#if 0
		printf("fApprox = %10.7f\n",fApprox.getValue());
		printf("fExact = %10.7f\n",ys[i]);
#endif

		lossFunc += (fApprox - ys(i)) * (fApprox - ys(i));

	} // end of i loop

	lossFunc = lossFunc / N;

#if 0
	printf("lossFunc = %10.7f\n",lossFunc.getValue());
#endif




	codi::RealReverse **v;
	v = new codi::RealReverse*[n];

	for (i = 0; i < n; i++) {

		v[i] = new codi::RealReverse[n];
	}
	codi::RealReverse *w = new codi::RealReverse[n];

	codi::RealReverse* rv1 = (codi::RealReverse *) malloc(
			(unsigned int) n * sizeof(codi::RealReverse));

	/* Householder reduction to bidiagonal form */
	for (i = 0; i < n; i++) {
		/* left-hand reduction */
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m) {
			for (k = i; k < m; k++)
				scale += fabs(a[k][i]);
			if (scale != 0.0) {
				for (k = i; k < m; k++) {
					a[k][i] = (a[k][i] / scale);
					s += (a[k][i] * a[k][i]);
				}
				f = a[i][i];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][i] = (f - g);
				if (i != n - 1) {
					for (j = l; j < n; j++) {
						for (s = 0.0, k = i; k < m; k++)
							s += (a[k][i] * a[k][j]);
						f = s / h;
						for (k = i; k < m; k++)
							a[k][j] += (f * a[k][i]);
					}
				}
				for (k = i; k < m; k++)
					a[k][i] = (a[k][i] * scale);
			}
		}
		w[i] = (scale * g);

		/* right-hand reduction */
		g = s = scale = 0.0;
		if (i < m && i != n - 1) {
			for (k = l; k < n; k++)
				scale += fabs(a[i][k]);
			if (scale != 0.0) {
				for (k = l; k < n; k++) {
					a[i][k] = (a[i][k] / scale);
					s += (a[i][k] * a[i][k]);
				}
				f = a[i][l];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][l] = (f - g);
				for (k = l; k < n; k++)
					rv1[k] = a[i][k] / h;
				if (i != m - 1) {
					for (j = l; j < m; j++) {
						for (s = 0.0, k = l; k < n; k++)
							s += (a[j][k] * a[i][k]);
						for (k = l; k < n; k++)
							a[j][k] += (s * rv1[k]);
					}
				}
				for (k = l; k < n; k++)
					a[i][k] = (a[i][k] * scale);
			}
		}
		anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}

	/* accumulate the right-hand transformation */
	for (i = n - 1; i >= 0; i--) {
		if (i < n - 1) {
			if (g != 0.0) {
				for (j = l; j < n; j++)
					v[j][i] = ((a[i][j] / a[i][l]) / g);
				/* double division to avoid underflow */
				for (j = l; j < n; j++) {
					for (s = 0.0, k = l; k < n; k++)
						s += (a[i][k] * v[k][j]);
					for (k = l; k < n; k++)
						v[k][j] += (s * v[k][i]);
				}
			}
			for (j = l; j < n; j++)
				v[i][j] = v[j][i] = 0.0;
		}
		v[i][i] = 1.0;
		g = rv1[i];
		l = i;
	}

	/* accumulate the left-hand transformation */
	for (i = n - 1; i >= 0; i--) {
		l = i + 1;
		g = w[i];
		if (i < n - 1)
			for (j = l; j < n; j++)
				a[i][j] = 0.0;
		if (g != 0.0) {
			g = 1.0 / g;
			if (i != n - 1) {
				for (j = l; j < n; j++) {
					for (s = 0.0, k = l; k < m; k++)
						s += (a[k][i] * a[k][j]);
					f = (s / a[i][i]) * g;
					for (k = i; k < m; k++)
						a[k][j] += (f * a[k][i]);
				}
			}
			for (j = i; j < m; j++)
				a[j][i] = (a[j][i] * g);
		} else {
			for (j = i; j < m; j++)
				a[j][i] = 0.0;
		}
		++a[i][i];
	}

	/* diagonalize the bidiagonal form */
	for (k = n - 1; k >= 0; k--) { /* loop over singular values */
		for (its = 0; its < 30; its++) { /* loop over allowed iterations */
			flag = 1;
			for (l = k; l >= 0; l--) { /* test for splitting */
				nm = l - 1;
				if (fabs(rv1[l]) + anorm == anorm) {
					flag = 0;
					break;
				}
				if (fabs(w[nm]) + anorm == anorm)
					break;
			}
			if (flag) {
				c = 0.0;
				s = 1.0;
				for (i = l; i <= k; i++) {
					f = s * rv1[i];
					if (fabs(f) + anorm != anorm) {
						g = w[i];
						h = PYTHAG(f, g);
						w[i] = h;
						h = 1.0 / h;
						c = g * h;
						s = (-f * h);
						for (j = 0; j < m; j++) {
							y = a[j][nm];
							z = a[j][i];
							a[j][nm] = (y * c + z * s);
							a[j][i] = (z * c - y * s);
						}
					}
				}
			}
			z = w[k];
			if (l == k) { /* convergence */
				if (z < 0.0) { /* make singular value nonnegative */
					w[k] = (-z);
					for (j = 0; j < n; j++)
						v[j][k] = (-v[j][k]);
				}
				break;
			}
			if (its >= 30) {
				free((void*) rv1);
				fprintf(stderr, "No convergence after 30,000! iterations \n");
				return (0);
			}

			/* shift from bottom 2 x 2 minor */
			x = w[l];
			nm = k - 1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
			g = PYTHAG(f, 1.0);
			f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

			/* next QR transformation */
			c = s = 1.0;
			for (j = l; j <= nm; j++) {
				i = j + 1;
				g = rv1[i];
				y = w[i];
				h = s * g;
				g = c * g;
				z = PYTHAG(f, h);
				rv1[j] = z;
				c = f / z;
				s = h / z;
				f = x * c + g * s;
				g = g * c - x * s;
				h = y * s;
				y = y * c;
				for (jj = 0; jj < n; jj++) {
					x = v[jj][j];
					z = v[jj][i];
					v[jj][j] = (x * c + z * s);
					v[jj][i] = (z * c - x * s);
				}
				z = PYTHAG(f, h);
				w[j] = z;
				if (z != 0.0) {
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = (c * g) + (s * y);
				x = (c * y) - (s * g);
				for (jj = 0; jj < m; jj++) {
					y = a[jj][j];
					z = a[jj][i];
					a[jj][j] = (y * c + z * s);
					a[jj][i] = (z * c - y * s);
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
	}
	free((void*) rv1);

#if 0
	printf("singular values of M=\n");

	for (i = 0; i < n; i++) {

		printf("%10.7f\n",w[i].getValue());
	}
#endif

	codi::RealReverse temp;
	for (i = 0; i < n; ++i) {
		for (j = i + 1; j < n; ++j) {

			if (w[i] < w[j])

			{
				temp = w[i];
				w[i] = w[j];
				w[j] = temp;
			}
		}
	}

#if 0
	printf("singular values of a=\n");

	for (i = 0; i < n; i++) {

		printf("%10.7f\n",w[i].getValue());
	}
#endif

	codi::RealReverse wsum = 0.0;
	for (i = 0; i < n; i++) {

		wsum += w[i];

	}

	for (i = 0; i < n; i++) {

		w[i] = w[i]/wsum;

	}


	double svd_multiplier = (1.0*n*(1.0*n+1))/2.0;

	svd_multiplier = 1.0/svd_multiplier;

	codi::RealReverse reg_term_svd = 0.0;

	for (i = 0; i < n; i++) {
#if 0
		printf("%d * %10.7f = %10.7f\n",i+1,w[i].getValue(),(i+1)*w[i].getValue());
#endif
		reg_term_svd = reg_term_svd + (i + 1) * w[i];
	}
#if 0
	printf("reg_term_svd = %10.7f\n",reg_term_svd.getValue());
#endif
	codi::RealReverse reg_term_L1 = 0.0;

	for (i = 0; i < n; i++)
		for (j = 0; j <= i; j++) {

			reg_term_L1 = reg_term_L1 + L[i][j] * L[i][j];
		}
#if 0
	printf("reg_term_L1 = %10.7f\n",reg_term_L1.getValue());
#endif

	reg_term_L1 = reg_term_L1/(n*n);


	codi::RealReverse result = wLoss*lossFunc + wSvd * svd_multiplier*reg_term_svd
			+ w12 * reg_term_L1;

#if 1
	if(iteration %1000 == 0){
		printf("iteration = %d, objFun = %10.7f, lossFunc = %10.7f, svd part = %10.7f, L1 reg. part = %10.7f\n",
				iteration,result.getValue(),wLoss*lossFunc.getValue(), wSvd * svd_multiplier*reg_term_svd.getValue(), w12 * reg_term_L1.getValue());
	}
#endif


#if 0
	printf("result = %10.7f\n",result.getValue());
#endif

	tape.registerOutput(result);

	tape.setPassive();
	result.setGradient(1.0);
	tape.evaluate();



	sigmab = sigmain.getGradient();

	Mgradient.fill(0.0);

	for (i = 0; i < n; i++) {
		for (j = 0; j <= i; j++) {

			Mgradient(i, j) = L[i][j].getGradient();
		}
	}


	if(isnan(result.getValue())){

		fprintf(stderr, "Error: result value is NaN! at %s, line %d.\n",__FILE__, __LINE__);



		printf("L = \n");

		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {

				printf("%10.7f ", L[i][j].getValue());

			}
			printf("\n");
		}

		exit(-1);
	}




	for (i = 0; i < n; i++) {
		delete[] v[i];
		delete[] a[i];
		delete[] L[i];
		delete[] LT[i];

	}
	delete[] a;
	delete[] L;
	delete[] LT;
	delete[] v;
	delete[] w;

	delete[] kernelVal;
	delete[] xp;
	delete[] xi;

	tape.reset();


	if(iteration%1000 == 0)
		printf("Wanna see the # of times \n");

	return (result.getValue());

}

int trainMahalanobisDistance(mat &M, mat &data, double &sigma, double &wSvd, double &w12) {


	std::ofstream fout;
	fout.open("Regularization Weights.txt");

	double wLoss = 1.0;




	if(M.n_cols != M.n_rows){

		fprintf(stderr,"Error: The Mahalanobis matrix is not square!\n");
		exit(-1);
	}

	unsigned int n = M.n_cols;

	mat bestM(n,n);
	bestM.fill(0.0);


	/* divide the data set into training and validation sets */

	unsigned int N = data.n_rows;

	unsigned int NvalSet = N/5;
	unsigned int Ntra = N - NvalSet;


	uvec indicesvalset(NvalSet);
	generate_validation_set(indicesvalset, N);

#if 1
	printf("indices of the validation set =\n");
	indicesvalset.print();
#endif


	mat dataValidation(NvalSet,data.n_cols );
	mat dataTraining(N-NvalSet,data.n_cols );


	unsigned int added_rowsTraining=0.0;
	unsigned int added_rowsValidation=0.0;
	for(unsigned int j=0; j<N; j++){ /* for each row in the data matrix */

		/* if j is not a validation point */
		if ( is_in_the_list(int(j), indicesvalset) == -1){

#if 0
			printf("%dth point is not a validation point\n",j);
#endif
			dataTraining.row(added_rowsTraining)=data.row(j);
			added_rowsTraining++;

		}
		else{
#if 0
			printf("%dth point is a validation point\n",j);
#endif
			dataValidation.row(added_rowsValidation)=data.row(j);
			added_rowsValidation++;

		}


	}

#if 1

	printf("Training data set = \n");
	dataTraining.print();

	printf("Validation data set = \n");
	dataValidation.print();

#if 0
	printf("Data matrix = \n");
	data.print();
#endif

	mat XTraining = dataTraining.submat(0, 0, Ntra - 1, n - 1);
	mat XValidation = dataValidation.submat(0, 0, NvalSet - 1, n - 1);
#if 1
	printf("XTraining = \n");
	XTraining.print();

	printf("XValidation = \n");
	XValidation.print();
#endif
	vec x_max(n);
	x_max.fill(0.0);

	vec x_min(n);
	x_min.fill(n);

	for (int i = 0; i < n; i++) {

		x_max(i) = data.col(i).max();
		x_min(i) = data.col(i).min();

	}

#if 0
	printf("maximum = \n");
	x_max.print();

	printf("minimum = \n");
	x_min.print();
#endif
	/* normalize training and validation data */
	for (unsigned int i = 0; i < NvalSet; i++) {

		for (unsigned int j = 0; j < n; j++) {

			XValidation(i, j) = (1.0 / n) * (XValidation(i, j) - x_min(j))
																																																																																																									/ (x_max(j) - x_min(j));
		}
	}

	for (unsigned int i = 0; i < Ntra; i++) {

		for (unsigned int j = 0; j < n; j++) {

			XTraining (i, j) = (1.0 / n) * (XTraining (i, j) - x_min(j))
																																																																																																										/ (x_max(j) - x_min(j));
		}
	}



#if 1
	printf("XTraining  (normalized) = \n");
	XTraining.print();

	printf("XValidation (normalized) = \n");
	XValidation.print();

#endif

	vec ysTraining;

	ysTraining = dataTraining.col(n);
	vec ysValidation;

	ysValidation = dataTraining.col(n);



#if 0
	printf("ys (training data) = \n");
	ysTraining .print();
#endif



#endif


#if 0

	// consistency check

	double sigma_derivative;
	mat adjoint_res(n,n);
	adjoint_res.fill(0.0);

	L.print();

	double result = dsvd(L,data,sigma,wLoss,wSvd,w12);
	printf("output (primal) = %10.7f\n",result);

	double resultAdj = dsvdAdj(L, data,sigma, adjoint_res, sigma_derivative,wLoss,wSvd,w12);
	double resultTl = dsvdTL(L,data, sigma, 0, 0, &sigma_derivative,wLoss, wSvd,w12);


	printf("output (reverse) = %10.7f\n",resultAdj);
	printf("output (forward) = %10.7f\n",resultTl);

	double resultpert;
	mat fd(n,n);
	fd.fill(0.0);
	mat forward_res(n,n);
	forward_res.fill(0.0);
	mat Lcopy(n,n);

	double temp;
	const double perturbation_param = 0.000001;
	double adot_save;

	for(int i=0;i<n;i++)
		for(int j=0;j<=i;j++) {
			printf("i = %d, j= %d\n",i,j);
			resultTl = dsvdTL(L,data, sigma, i, j, &forward_res(i,j),wLoss,wSvd,w12);
			result = dsvd(L, data,sigma,wLoss,wSvd,w12);
			printf("result = %10.7f\n",resultTl);
			Lcopy=L;
			L(i,j)+=perturbation_param;
			cout.precision(12);
			printf("L perturbed = \n");
			L.print();
			resultpert = dsvd(L, data,sigma,wLoss,wSvd,w12);
			printf("resultpert = %10.7f\n",resultpert);

			fd(i,j)= (resultpert-result)/perturbation_param;
			L=Lcopy;



		}

	resultAdj = dsvdAdj(L, data,sigma, adjoint_res, sigma_derivative,wLoss,wSvd,w12);

	printf("fd results =\n");
	fd.print();
	printf("forward AD results =\n");
	forward_res.print();
	printf("reverse AD results =\n");
	adjoint_res.print();

	printf("error  between forward AD and fd=\n");
	mat errorForAD = fd - forward_res;
	errorForAD.print();
	printf("error  between forward AD and reverse AD=\n");
	mat errorRevAD = forward_res-adjoint_res;
	errorRevAD.print();

	printf("derivative of sigma (adjoint) = %10.7f\n",sigma_derivative);

	double sigma_derivative_forward_mode;

	resultTl = dsvdTL(L,data, sigma, -1, -1, &sigma_derivative_forward_mode,wLoss,wSvd,w12);

	printf("derivative of sigma (forward) = %10.7f\n",sigma_derivative_forward_mode);

	result = dsvd(L, data,sigma,wLoss,wSvd,w12);

	sigma+=perturbation_param;
	resultpert = dsvd(L, data,sigma,wLoss,wSvd,w12);

	double fd_sigma= (resultpert-result)/perturbation_param;

	printf("derivative of sigma (fd) = %10.7f\n",fd_sigma);
#endif

	int max_cv_iter = 3;

	vec wSvdtrial(max_cv_iter);
	wSvdtrial.randu();


	vec w12trial(max_cv_iter);
	w12trial.randu();


#if 1
	//Printing the weights into a text file
	for(int iter_cv;iter_cv< max_cv_iter; iter_cv++) 
		fout<<" "<<wSvdtrial(iter_cv)<<" \t "<< w12trial(iter_cv)<<"\n";
	fout<<"Now, iterating through the different combinations: \n";
#endif

	double bestValidationError = 10E14;
	double bestwSvd = 0.0;
	double bestw12 = 0.0;

	for(int iter_cv;iter_cv< max_cv_iter; iter_cv++){

		wSvd = wSvdtrial(iter_cv)*0.1;
		//w12 =  w12trial(iter_cv)*0.1;
		//fout<<" "<<wSvd<<" \t "<< w12<<"\n";
		
		for(int iter_cv_2=0;iter_cv_2< max_cv_iter; iter_cv_2++)
		{
			w12 =  w12trial(iter_cv_2)*0.1;
			fout<<" "<<wSvd<<" \t "<< w12<<"\n";

			int cv_curr_iteration = iter_cv*max_cv_iter + iter_cv_2;
#if 1
		printf("cv iteration = %d, wSvd = %10.7f, w12 = %10.7f\n",cv_curr_iteration,wSvd,w12);
#endif
		/* initialize the L matrix (lower diagonal) */

		mat L(n,n);


		L.fill(0.0);

		for(unsigned int i=0;i<n;i++)
			for(unsigned int j=0;j<=i;j++) {

				if (i==j) {

					L(i,j)=1.0;
				}
				else{

					L(i,j)=0.001;
				}

			}

#if 0
		L.print();
#endif
		/* optimization stage */

		int maxoOptIter = 1000;
		int optIter = 1;
		mat gradient(n, n);
		double stepSizeInit = 0.001;
		double stepSize = stepSizeInit;
		double fStep, fpreviousIter;
		double sigmab = 0.0;

		/* compute the initial gradient and objective function value */
		fpreviousIter = dsvdAdj(optIter,L, dataTraining, sigma, gradient, sigmab, wLoss,wSvd,w12);

#if 0
		printf("initial gradient = \n");
		gradient.print();
#endif

		mat Lsave(n, n);
		Lsave.fill(0.0);
		double sigmaSave;
		while (1) {

			Lsave = L;
			sigmaSave = sigma;

#if 0
			printf("gradient = \n");
			gradient.print();
			printf("L (before) = \n");
			L.print();
			printf("sigma (before) = %10.7f\n",sigma);
#endif

			/* update the lower triangular matrix L */
			for (unsigned int i = 0; i < n; i++)
				for (unsigned int j = 0; j <= i; j++) {

					L(i, j) = L(i, j) - stepSize * gradient(i, j);

				}

			/* update the bandwidth */

			sigma = sigma - stepSize*sigmab;

			if(sigma <= 0.0) sigma = 10E-6;

#if 0
			printf("L = \n");
			L.print();
			printf("sigma = %10.7f\n",sigma);
#endif
			//			sigma = sigma-stepSize*sigmab;


#if 0
			printf("stepSize = %10.7f\n",stepSize);
			printf("evaluating gradient vector...\n");
			M.print();
#endif


			fStep = dsvdAdj(optIter,L, dataTraining, sigma, gradient, sigmab, wLoss,wSvd,w12);


#if 0
			printf("fStep= %10.7f\n", fStep);

			//		printf("gradient=\n");
			//		gradient.print();
#endif

#if 0
			if (fStep > fpreviousIter) {

				L = Lsave;
				sigma = sigmaSave;

				stepSize = stepSize / 2.0;

			} else {

			}
#endif

			fpreviousIter = fStep;

			optIter++;
			if (optIter > maxoOptIter || stepSize < 10E-10) {
#if 0
				printf("optimization prcedure is terminating with iter =%d...\n",
						optIter);
#endif
				break;

			}
		}
#if 0

		printf("optimal sigma = %10.7f\n",sigma);
		printf("sigma derivative= %10.7f\n",sigmab);
#endif

#if 1
		printf("optimal M =\n");
		M = L*trans(L);
		M.print();
		printf("gradient w.r.t to M =\n");
		gradient.print();

#endif


#if 1
		mat U;
		vec s;
		mat V;




		svd(U, s, V, M);

		printf("singular values of M = \n");
		s.print();
#endif


		double ErrorValidation=0.0;

		for(int j=0; j<NvalSet;j++){

			rowvec xp = XValidation.row(j);
			double yApprox = kernelRegressor(XTraining, ysTraining, xp, M, sigma);
			double yExact = ysValidation(j);
			ErrorValidation += (yApprox-yExact)*(yApprox-yExact);

#if 0
			printf("xp = \n");
			xp.print();
			printf("yApprox = %10.7f\n",yApprox );
			printf("yExact  = %10.7f\n",yExact  );
			printf("Validation Error = %10.7f\n",ErrorValidation );

#endif
		}

		ErrorValidation = ErrorValidation/NvalSet;
		printf("Validation Error = %10.7f\n",ErrorValidation );

		if (ErrorValidation  < bestValidationError){

			bestM = M;
			bestwSvd = wSvd;
			bestw12 = w12;
			bestValidationError = ErrorValidation;
		}

	}

	}


#if 1
	printf("Best weights wSvd = %10.7f, w12 = %10.7f\n",bestwSvd,bestw12);
	printf("Best M = \n");
	bestM.print();
	fout<<"Best weights wSvd: "<<wSvd<<" \t w12: "<< w12<<"\n";
#endif

	M = bestM;

	return 0;

	fout.close();
}

double kernelRegressor(mat &X, vec &y, rowvec &xp, mat &M, double sigma) {

	int d = y.size();

	vec kernelVal(d);
	vec weight(d);
	double kernelSum = 0.0;
	double yhat = 0.0;



	for (int i = 0; i < d; i++) {

		rowvec xi = X.row(i);
		kernelVal(i) = gaussianKernel(xi, xp, sigma, M);
		kernelSum += kernelVal(i);
	}

	for (int i = 0; i < d; i++) {

		weight(i) = kernelVal(i) / kernelSum;
		yhat += y(i) * weight(i);
#if 0
		printf("y(%d) * weight(%d) = %10.7f * %10.7f\n",i,i,y(i),weight(i) );
#endif
	}

	return yhat;



}

	
