
#include "kernel_regression_cuda.h"
#include "auxilliary_functions.hpp"
#include "Rodeo_macros.hpp"
#include "test_functions.hpp"

#include<stdio.h>
#include<iostream>
#include<math.h>



#include <armadillo>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include <codi.hpp>

using namespace arma;


//This implementation using CAS incurs a non-trivial cost though.
//Had to use this because compute < 600 doesn't support atomic add with float and > 600 throws up some MemCpy - invalid code error
//__device__ float atomicDAdd(float* address, float val);
//
//__device__ float atomicDAdd(float* address, float val)
//{
//	unsigned long long int* address_as_ull =
//			(unsigned long long int*)address;
//	unsigned long long int old = *address_as_ull, assumed;
//	do {
//		assumed = old;
//		old = atomicCAS(address_as_ull, assumed,
//				__float_as_longlong(val +
//						__longlong_as_float(assumed)));
//	} while (assumed != old);
//	return __longlong_as_float(old);
//}


//__managed__ float MDevice[numVar*numVar+1];
__constant__ float MDevice[numVar*numVar+1];


float gaussianKernel(frowvec &xi, frowvec &xj, float sigma, fmat &M) {
#if 0
	printf("calling gaussianKernel...\n");
	xi.print();
	xj.print();
#endif

	/* calculate distance between xi and xj with the matrix M */
	float metricVal = calcMetric(xi, xj, M);
#if 0
	printf("metricVal = %10.7f\n",metricVal);
#endif

	float sqr_two_pi = sqrt(2.0 * datum::pi);

	float kernelVal = (1.0 / (sigma * sqr_two_pi))* exp(-metricVal / (2 * sigma * sigma));

	kernelVal += 10E-14;

#if 0
	printf("kernelVal = %10.7f\n",kernelVal);

#endif
	return kernelVal;

}




float SIGN(float a, float b) {

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

float PYTHAG(float a, float b) {
	float at = fabs(a), bt = fabs(b), ct, result;

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

/** calculate regularization terms for the given matrix L
 *
 * @param[in]  L:  lower diagonal matrix
 * @param[in]  wSvd: weight for the svd regularization part
 * @param[in]  w12: weight for the mixed 12 regularization part 
 * @param[out] regTerm 
 *
 */

int calcRegTerms(float *L, float *regTerm, float wSvd, float w12, int dim) {
	int flag, i, its, j, jj, k, l = 0, nm;
	float c, f, h, s, x, y, z;
	float anorm = 0.0, g = 0.0, scale = 0.0;


	int m = dim;
	int n = dim;


	float **a;
	a = new float*[dim];

	for (i = 0; i < dim; i++) {

		a[i] = new float[dim];
	}

	float **M;
	M= new float*[dim];

	for (i = 0; i < dim; i++) {

		M[i] = new float[dim];
	}


	float **LT;
	LT = new float*[dim];
	for (int i = 0; i < dim; i++) {
		LT[i] = new float[dim];

	}

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) {

			LT[i][j]=0.0;
		}




	for (int i = 0; i < dim; i++) {
		for (int j = 0; j <= i; j++){

			LT[j][i] = L[i*dim+j];
		}


	}

#if 0
	printf("L = \n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", L[i*dim+j]);

		}
		printf("\n");
	}	
#endif


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
			M[i][j]=0;
		}

	/* Multiplying matrix L and LT and storing in M */
	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
			for(int k = 0; k < dim; ++k)
			{
				M[i][j] += L[i*dim+k] * LT[k][j];

			}

	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
		{
			a[i][j]=M[i][j];

		}



#if 0
	printf("a = \n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", a[i][j]);

		}
		printf("\n");
	}

#endif

#if 0
	/* only for validation */
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


	/* SVD part */

	float **v;
	v = new float*[n];

	for (i = 0; i < n; i++) {

		v[i] = new float[n];
	}
	float *w = new float[n];

	float *rv1 = new float[n];

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
				/* float division to avoid underflow */
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
		for (its = 0; its < 30000; its++) { /* loop over allowed iterations */
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
			if (its >= 30000) {
				delete[] rv1;
				fprintf(stderr, "No convergence after 30,000! iterations \n");
				return 1;
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

	/* sort the singular values */

	float temp;
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

		printf("%10.7f\n",w[i]);
	}
#endif

	/* normalization */
	float wsum = 0.0;
	for (i = 0; i < n; i++) {

		wsum += w[i];

	}

	for (i = 0; i < n; i++) {

		w[i] = w[i]/wsum;

	}

#if 0
	printf("singular values of a (normalized) with wsum =%10.7f\n",wsum);


	for (i = 0; i < n; i++) {

		printf("%15.10f\n",w[i]);
	}
#endif


	float svd_multiplier = (1.0*n*(1.0*n+1))/2.0;

	svd_multiplier = 1.0/svd_multiplier;
#if 0
	printf("svd_multiplier = %10.7f\n",svd_multiplier);
#endif
	float reg_term_svd = 0.0;

	for (i = 0; i < n; i++) {
#if 0
		printf("%d * %10.7f = %10.7f\n",i+1,w[i],(i+1)*w[i]);
#endif
		reg_term_svd = reg_term_svd + (i + 1) * w[i];
	}
#if 0
	printf("reg_term_svd = %10.7f\n",reg_term_svd);
#endif


	float reg_term_L1 = 0.0;

	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++) {

			reg_term_L1 = reg_term_L1 + M[i][j]* M[i][j];
		}
#if 0
	printf("reg_term_L1 = %10.7f\n",reg_term_L1);
#endif



	for (i = 0; i < n; i++) {
		delete[] v[i];
		delete[] a[i];
		delete[] M[i];
		delete[] LT[i];
	}


	delete[] LT;
	delete[] M;
	delete[] a;
	delete[] v;
	delete[] w;



	*regTerm = wSvd * svd_multiplier *reg_term_svd + w12 * reg_term_L1;
#if 0
	printf("result = %10.7f\n",*regTerm);
#endif

	return 0;




}

/* forward mode */

int calcRegTerms(float *L, float *regTerm,float *regTermd, float wSvd, float w12, int dim, int derIndx) {
	int flag, i, its, j, jj, k, l = 0, nm;
	codi::RealForward c, f, h, s, x, y, z;
	codi::RealForward anorm = 0.0, g = 0.0, scale = 0.0;


	int m = dim;
	int n = dim;

	codi::RealForward *Lcodi = new codi::RealForward[dim*dim];
	for (int i = 0; i < dim*dim; i++) {

		Lcodi[i] = L[i];

	}

	Lcodi[derIndx].setGradient(1.0); 


	codi::RealForward **a;
	a = new codi::RealForward*[dim];

	for (i = 0; i < dim; i++) {

		a[i] = new codi::RealForward[dim];
	}

	codi::RealForward **M;
	M= new codi::RealForward*[dim];

	for (i = 0; i < dim; i++) {

		M[i] = new codi::RealForward[dim];
	}


	codi::RealForward **LT;
	LT = new codi::RealForward*[dim];
	for (int i = 0; i < dim; i++) {
		LT[i] = new codi::RealForward[dim];

	}

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) {

			LT[i][j]=0.0;
		}




	for (int i = 0; i < dim; i++) {
		for (int j = 0; j <= i; j++){

			LT[j][i] = Lcodi[i*dim+j];
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
			M[i][j]=0;
		}

	/* Multiplying matrix L and LT and storing in M */
	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
			for(int k = 0; k < dim; ++k)
			{
				M[i][j] += Lcodi[i*dim+k] * LT[k][j];

			}

	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
		{
			a[i][j]=M[i][j];

		}



#if 0
	printf("a = \n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", a[i][j]);

		}
		printf("\n");
	}

#endif




	/* SVD part */

	codi::RealForward **v;
	v = new codi::RealForward*[n];

	for (i = 0; i < n; i++) {

		v[i] = new codi::RealForward[n];
	}
	codi::RealForward *w = new codi::RealForward[n];

	codi::RealForward *rv1 = new codi::RealForward[n];

	/* Householder reduction to bidiagonal form */
	for (i = 0; i < n; i++) {
		/* left-hand reduction */
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m) {
			for (k = i; k < m; k++)
				scale += fabs(a[k][i]);
			if (scale!= 0) {
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
			if (scale!=0) {
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
			if (g!=0) {
				for (j = l; j < n; j++)
					v[j][i] = ((a[i][j] / a[i][l]) / g);
				/* float division to avoid underflow */
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
		if (g!=0) {
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
		for (its = 0; its < 30000; its++) { /* loop over allowed iterations */
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
			if (its >= 30000) {
				delete[] rv1;
				fprintf(stderr, "No convergence after 30,000! iterations \n");
				return 1;
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
				if (z!=0) {
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

	/* sort the singular values */

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

		printf("%10.7f\n",w[i].getValue());
	}
#endif

	/* normalization */
	codi::RealForward wsum = 0.0;
	for (i = 0; i < n; i++) {

		wsum += w[i];

	}

	for (i = 0; i < n; i++) {

		w[i] = w[i]/wsum;

	}

#if 0
	printf("singular values of a (normalized) with wsum =%10.7f\n",wsum.getValue());


	for (i = 0; i < n; i++) {

		printf("%15.10f\n",w[i].getValue());
	}
#endif


	float svd_multiplier = (1.0*n*(1.0*n+1))/2.0;

	svd_multiplier = 1.0/svd_multiplier;
#if 0
	printf("svd_multiplier = %10.7f\n",svd_multiplier);
#endif
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

	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++) {

			reg_term_L1 = reg_term_L1 + M[i][j]* M[i][j];
		}
#if 0
	printf("reg_term_L1 = %10.7f\n",reg_term_L1.getValue());
#endif



	for (i = 0; i < n; i++) {
		delete[] v[i];
		delete[] a[i];
		delete[] M[i];
		delete[] LT[i];
	}


	delete[] LT;
	delete[] M;
	delete[] a;
	delete[] v;
	delete[] w;


	codi::RealForward result = wSvd * svd_multiplier *reg_term_svd + w12 * reg_term_L1;
	*regTerm = result.getValue();

	*regTermd = result.getGradient();

	return 0;


}



int calcRegTerms(float *L, float *Lb,float *result , float wSvd, float w12, int dim) {
	int flag, i, its, j, jj, k, l = 0, nm;


	codi::RealReverse *Lcodi = new codi::RealReverse[dim*dim];
	for (int i = 0; i < dim*dim; i++) {

		Lcodi[i] = L[i];

	}

	/* activate tape and register input */

	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
	tape.setActive();

	codi::RealReverse regTerm=0.0;



	for (int i = 0; i < dim*dim; i++) {

		tape.registerInput(Lcodi[i]);

	}


	codi::RealReverse c, f, h, s, x, y, z;
	codi::RealReverse anorm = 0.0, g = 0.0, scale = 0.0;


	int m = dim;
	int n = dim;


	codi::RealReverse **a;
	a = new codi::RealReverse*[dim];

	for (i = 0; i < dim; i++) {

		a[i] = new codi::RealReverse[dim];
	}

	codi::RealReverse **M;
	M = new codi::RealReverse*[dim];

	for (i = 0; i < dim; i++) {

		M[i] = new codi::RealReverse[dim];
	}




	codi::RealReverse **LT;
	LT = new codi::RealReverse*[dim];
	for (int i = 0; i < dim; i++) {
		LT[i] = new codi::RealReverse[dim];

	}

	for (int i = 0; i < dim; i++)
		for (int j = 0; j < dim; j++) {

			LT[i][j]=0.0;
		}




	for (int i = 0; i < dim; i++) {
		for (int j = 0; j <= i; j++){

			LT[j][i] = Lcodi[i*dim+j];
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
			M[i][j]=0;
		}

	/* Multiplying matrix L and LT and storing in M */
	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
			for(int k = 0; k < dim; ++k)
			{
				M[i][j] += Lcodi[i*dim+k] * LT[k][j];

			}

	for(int i = 0; i < dim; ++i)
		for(int j = 0; j < dim; ++j)
		{
			a[i][j]=M[i][j];

		}

#if 0
	printf("M = \n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", M[i][j].getValue());

		}
		printf("\n");
	}

#endif

#if 0
	printf("a = \n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", a[i][j].getValue());

		}
		printf("\n");
	}

#endif


	/* SVD part */

	codi::RealReverse **v;
	v = new codi::RealReverse*[n];

	for (i = 0; i < n; i++) {

		v[i] = new codi::RealReverse[n];
	}
	codi::RealReverse *w = new codi::RealReverse[n];

	codi::RealReverse *rv1 = new codi::RealReverse[n];

	/* Householder reduction to bidiagonal form */
	for (i = 0; i < n; i++) {
		/* left-hand reduction */
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m) {
			for (k = i; k < m; k++)
				scale += fabs(a[k][i]);
			if (scale != 0) {
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
			if (scale !=0) {
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
			if (g !=0) {
				for (j = l; j < n; j++)
					v[j][i] = ((a[i][j] / a[i][l]) / g);
				/* float division to avoid underflow */
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
		if (g != 0) {
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
		for (its = 0; its < 30000; its++) { /* loop over allowed iterations */
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
			if (its >= 30000) {
				delete[] rv1;
				fprintf(stderr, "No convergence after 30,000! iterations \n");
				return 1;
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
				if (z != 0) {
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

#if 0
	printf("singular values of a (normalized) with wsum =%10.7f\n",wsum.getValue());


	for (i = 0; i < n; i++) {

		printf("%15.10f\n",w[i].getValue());
	}
#endif


	codi::RealReverse svd_multiplier = (1.0*n*(1.0*n+1))/2.0;

	svd_multiplier = 1.0/svd_multiplier;
#if 0
	printf("svd_multiplier = %10.7f\n",svd_multiplier);
#endif
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
		for (j = 0; j < n; j++) {

			reg_term_L1 = reg_term_L1 + M[i][j]* M[i][j];
		}
#if 0
	printf("reg_term_L1 = %10.7f\n",reg_term_L1.getValue());
#endif







	regTerm = wSvd * svd_multiplier *reg_term_svd + w12 * reg_term_L1;

#if 0
	printf("w12 * reg_term_L1 = %10.7f\n",w12 * reg_term_L1.getValue());
#endif


	tape.registerOutput(regTerm);

	tape.setPassive();
	regTerm.setGradient(1.0);
	tape.evaluate();

	for (int i = 0; i < numVar*numVar; i++) {


		Lb[i] = Lcodi[i].getGradient();

	}


	tape.reset();

	*result = regTerm.getValue(); 

	for (i = 0; i < n; i++) {

		delete[] v[i];
		delete[] a[i];
		delete[] M[i];
		delete[] LT[i];
	}


	delete[] LT;
	delete[] M;
	delete[] a;
	delete[] v;
	delete[] w;
	delete[] Lcodi;


	return 0;




}






float calcKernelValCPU(rowvec &xi, rowvec &xj, mat &M, float sigma){

	rowvec diff = xi - xj;
	colvec diffT = trans(diff);

	vec matVecProd = M * diffT;
	//	printf("M * xdiff = \n");
	//	matVecProd.print();

	float metricVal = dot(diff, M * diffT);

	float sqr_two_pi = sqrt(2.0 * 3.14159265359);

	float kernelVal = (1.0 / (sigma * sqr_two_pi))* exp(-metricVal / (2 * sigma * sigma));


	return (kernelVal);



}
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

float calcMetric(float *xi, float *xj, float *M, int dim) {

#if 0
	printf("calling calcMetric (primal)...\n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", M[i*dim+j]);

		}
		printf("\n");
	}

#endif



	float *diff = new float[dim];

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

	float metric_val = dot(diff_val, M_val * diffT);

	printf("metric_val = %10.7f\n", metric_val);
#endif

	float *tempVec = new float[dim];

	float sum = 0.0;

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			sum = sum + M[i*dim+j] * diff[j];
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
 * calculates the generalized Mahalanobis distance between two points, codiPack reverse mode
 * (differentiated in reverse mode )
 * @param[in] x_i : first vector
 * @param[in] X_j : second vector
 * @param[in] M : dim x dim matrix
 * @param[in] dim
 * @return distance
 *
 * */

codi::RealReverse calcMetric(float *xi, float *xj, codi::RealReverse *M,
		int dim) {

#if 0
	printf("calling calcMetric (adjoint)...\n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", M[i*dim+j].getValue());

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

			M_val(i, j) = M[i*dim+j].getValue();
		}

	diff_val = xi_val - xj_val;

	printf("diff_val=\n");
	diff_val.print();

	colvec diffT = trans(diff_val);

	vec matVecProd = M_val * diffT;
	printf("M * xdiff = \n");
	matVecProd.print();

	float metric_val = dot(diff_val, M_val * diffT);

	printf("metric_val = %10.7f\n", metric_val);
#endif

	codi::RealReverse *tempVec = new codi::RealReverse[dim];

	codi::RealReverse sum = 0.0;

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			sum = sum + M[i*dim+j] * diff[j];
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

				printf("%10.7f ", M[i*dim+j].getValue());

			}
			printf("\n");
		}


		exit(-1);
	}

	return sum;

}

codi::RealForward calcMetric(float *xi, float *xj, codi::RealForward *M,
		int dim) {

#if 0
	printf("calling calcMetric (adjoint)...\n");

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			printf("%10.7f ", M[i*dim+j].getValue());

		}
		printf("\n");
	}

#endif

	codi::RealForward *diff = new codi::RealForward[dim];

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

			M_val(i, j) = M[i*dim+j].getValue();
		}

	diff_val = xi_val - xj_val;

	printf("diff_val=\n");
	diff_val.print();

	colvec diffT = trans(diff_val);

	vec matVecProd = M_val * diffT;
	printf("M * xdiff = \n");
	matVecProd.print();

	float metric_val = dot(diff_val, M_val * diffT);

	printf("metric_val = %10.7f\n", metric_val);
#endif

	codi::RealForward *tempVec = new codi::RealForward[dim];

	codi::RealForward sum = 0.0;

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {

			sum = sum + M[i*dim+j] * diff[j];
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

				printf("%10.7f ", M[i*dim+j].getValue());

			}
			printf("\n");
		}


		exit(-1);
	}

	return sum;

}


float gaussianKernel(float *xi, float *xj, float sigma, float *M,
		int dim) {

#if 0
	printf("calling gaussianKernel...\n");
#endif

	/* calculate distance between xi and xj with the matrix M */
	float metricVal = calcMetric(xi, xj, M, dim);
#if 0
	printf("metricVal = %10.7f\n",metricVal);
#endif

	float sqr_two_pi = sqrt(2.0 * datum::pi);

	float kernelVal = (1.0 / (sigma * sqr_two_pi))* exp(-metricVal / (2 * sigma * sigma));
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

codi::RealReverse gaussianKernel(float *xi, float *xj,
		codi::RealReverse sigma, codi::RealReverse *M, int dim) {

#if 0
	printf("calling gaussianKernel...\n");
#endif

	/* calculate distance between xi and xj with the matrix M */
	codi::RealReverse metricVal = calcMetric(xi, xj, M, dim);
#if 0
	printf("metricVal = %10.7f\n",metricVal.getValue());
#endif

	float sqr_two_pi = sqrt(2.0 * datum::pi);

	codi::RealReverse kernelVal = (1.0 / (sigma * sqr_two_pi))* exp(-metricVal / (2 * sigma * sigma));

	if(isnan(kernelVal.getValue())){

		fprintf(stderr, "Error: kernel value is NaN! at %s, line %d.\n",__FILE__, __LINE__);

		printf("sigma = %10.7f\n",sigma.getValue());

		printf("M = \n");

		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {

				printf("%10.7f ", M[i*dim+j].getValue());

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

				printf("%10.7f ", M[i*dim+j].getValue());

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


codi::RealForward gaussianKernel(float *xi, float *xj,
		codi::RealForward sigma, codi::RealForward *M, int dim) {


	/* calculate distance between xi and xj with the matrix M */
	codi::RealForward metricVal = calcMetric(xi, xj, M, dim);

	float sqr_two_pi = sqrt(2.0 * datum::pi);

	codi::RealForward kernelVal = (1.0 / (sigma * sqr_two_pi))* exp(-metricVal / (2 * sigma * sigma));

	if(isnan(kernelVal.getValue())){

		fprintf(stderr, "Error: kernel value is NaN! at %s, line %d.\n",__FILE__, __LINE__);

		printf("sigma = %10.7f\n",sigma.getValue());

		printf("M = \n");

		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {

				printf("%10.7f ", M[i*dim+j].getValue());

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

				printf("%10.7f ", M[i*dim+j].getValue());

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

void calcLossFunCPU(float *result, float *input, float *data,int N){

	float LT[numVar][numVar];
	float L[numVar][numVar];
	float M[numVar*numVar+1];

	for (int i = 0; i < numVar; ++i)
		for (int j = 0; j < numVar; ++j)
			L[i][j] = input[i*numVar + j];




	for (int i = 0; i < numVar; ++i)
		for (int j = 0; j < numVar; ++j)
			LT[i][j] = 0.0;


	for (int i = 0; i < numVar; ++i) {
		for (int j = 0; j < i+1; ++j)
			LT[j][i] = L[i][j];
	}



	for (int i = 0; i < numVar; ++i)
		for (int j = 0; j < numVar; ++j)
			M[i*numVar + j] = 0;



	/* Multiplying matrix L and LT and storing in M */
	for (int i = 0; i < numVar; ++i)
		for (int j = 0; j < numVar; ++j)
			for (int k = 0; k < numVar; ++k)
				M[i*numVar + j] = M[i*numVar + j] + L[i][k]*LT[k][j];



	M[numVar*numVar] = input[numVar*numVar];

	float sigma = M[numVar*numVar];

	float *xp = new float[numVar];
	float *xi = new float[numVar];

	float *kernelVal = new float[N];

	float lossFunc = 0.0;

	for (int i = 0; i < N; i++) {

#if 0
		printf("kernel regression for the sample number %d\n",i);

#endif
		for (int k = 0; k < numVar; k++) {

			xp[k] = data[i*(numVar+1)+k];
		}

		float kernelSum = 0.0;
		for (int j = 0; j < N; j++) {

			if (i != j) {

				for (int k = 0; k < numVar; k++) {

					xi[k] = data[j*(numVar+1)+k];
				}
				kernelVal[j] = gaussianKernel(xi, xp, sigma, M, numVar);
				kernelSum += kernelVal[j];
#if 0
				printf("kernelVal[%d]=%10.7f\n",j,kernelVal[j]);
#endif
			}
		}

		float fApprox = 0.0;
		for (int j = 0; j < N; j++) {
			if (i != j) {
				fApprox += kernelVal[j] * data[j*(numVar+1)+numVar];

			}
		}

		fApprox = fApprox / kernelSum;

#if 0
		printf("fApprox = %10.7f\n",fApprox);
		printf("fExact = %10.7f\n",data[i*(numVar+1)+numVar]);
#endif

		lossFunc += (fApprox - data[i*(numVar+1)+numVar]) * (fApprox - data[i*(numVar+1)+numVar]);

	} // end of i loop

	lossFunc = lossFunc / N;

	*result = lossFunc; 

	delete[] xp;
	delete[] xi;
	delete[] kernelVal;


}



void calcLossFunCPU(codi::RealReverse *result, codi::RealReverse *input, float *inputb,float *data,int N){


	/* activate tape and register input */

	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
	tape.setActive();



	for (int i = 0; i < numVar*numVar+1; i++) {
		tape.registerInput(input[i]);

	}




	codi::RealReverse LT[numVar][numVar];
	codi::RealReverse L[numVar][numVar];
	codi::RealReverse M[numVar*numVar+1];

	for (int i = 0; i < numVar; ++i)
		for (int j = 0; j < numVar; ++j)
			M[i*numVar + j] = 0;





	for (int i = 0; i < numVar; ++i)
		for (int j = 0; j < numVar; ++j)
			L[i][j] = input[i*numVar + j];

#if 0
	printf("L = \n");
	for (int i = 0; i < numVar; ++i){
		for (int j = 0; j < numVar; ++j){

			printf("%10.7f ",L[i][j].getValue());

		}
		printf("\n");
	}
#endif	


	for (int i = 0; i < numVar; ++i)
		for (int j = 0; j < numVar; ++j) {

			LT[i][j] = 0.0;
		}





	for (int i = 0; i < numVar; ++i) {
		for (int j = 0; j < i+1; ++j){


			LT[j][i] = L[i][j];
		}
	}

#if 0
	printf("LT = \n");
	for (int i = 0; i < numVar; ++i){
		for (int j = 0; j < numVar; ++j){

			printf("%10.7f ",LT[i][j].getValue());

		}
		printf("\n");
	}
#endif	









	/* Multiplying matrix L and LT and storing in M */
	for (int i = 0; i < numVar; ++i)
		for (int j = 0; j < numVar; ++j)
			for (int k = 0; k < numVar; ++k) {

				M[i*numVar + j] = M[i*numVar + j] + L[i][k]*LT[k][j];
			}



#if 0
	printf("M = \n");
	for (int i = 0; i < numVar; ++i){
		for (int j = 0; j < numVar; ++j){

			printf("%10.7f ",M[i*numVar + j].getValue());

		}
		printf("\n");
	}
#endif	



	M[numVar*numVar] = input[numVar*numVar];



	codi::RealReverse sigma = M[numVar*numVar]; 

	float *xi = new float[numVar];
	float *xj = new float[numVar];

	codi::RealReverse **kernelValTable = new codi::RealReverse*[N];

	for(int i=0; i<N;i++) {

		kernelValTable[i] = new codi::RealReverse[N];

	}

	for (int i = 0; i < N; i++) 

		for (int j = 0; j < N; j++) kernelValTable[i][j] = 0.0;



	for (int i = 0; i < N; i++) {

		for (int j = i+1; j < N; j++) {



			for (int k = 0; k < numVar; k++) {

				xi[k] = data[i*(numVar+1)+k];
				xj[k] = data[j*(numVar+1)+k];
			}


			kernelValTable[i][j] = gaussianKernel(xi, xj, sigma, M, numVar);
			kernelValTable[j][i] = kernelValTable[i][j]; 
			//			printf("%d kernelValTable[%d][%d] = %10.7f\n",i*N+j,i,j,kernelValTable[i][j].getValue());





		}



	}

	codi::RealReverse lossFunc = 0.0;

	for (int i = 0; i < N; i++) {

#if 0
		printf("kernel regression for the sample number %d\n",i);

#endif


		codi::RealReverse kernelSum = 0.0;
		for (int j = 0; j < N; j++) {

			if (i != j) {


				kernelSum += kernelValTable[i][j];

			}
		}

		codi::RealReverse fApprox = 0.0;
		for (int j = 0; j < N; j++) {
			if (i != j) {
				fApprox += kernelValTable[i][j] * data[j*(numVar+1)+numVar];

			}
		}

		fApprox = fApprox / kernelSum;

#if 0
		printf("fApprox = %10.7f\n",fApprox.getValue());
		printf("fExact = %10.7f\n",data[i*(numVar+1)+numVar]);
#endif

		lossFunc += (fApprox - data[i*(numVar+1)+numVar]) * (fApprox - data[i*(numVar+1)+numVar]);

	} // end of i loop

	lossFunc = lossFunc / N;


#if 1

	printf("lossFunc (reverse mode CodiPack) = %10.7f\n",lossFunc.getValue());
#endif	

	*result = lossFunc; 


	tape.registerOutput(*result);

	tape.setPassive();
	result->setGradient(1.0);
	tape.evaluate();

#if 0
	printf("Mb = \n");
	for (int i = 0; i < numVar; ++i){
		for (int j = 0; j < numVar; ++j){

			printf("%10.7f ",M[i*numVar + j].getGradient());

		}
		printf("\n");
	}
#endif	




	for (int i = 0; i < numVar*numVar+1; i++) {


		inputb[i] = input[i].getGradient();

	}


	tape.reset();

	delete[] xi;
	delete[] xj;

	for(int i=0; i<N;i++) {

		delete[] kernelValTable[i];

	}

	delete[] kernelValTable;




}

void calcLossFunCPU(codi::RealForward *result, codi::RealForward *input,int tldIndx, float *data,int N){

	input[tldIndx].setGradient(1.0);



	codi::RealForward LT[numVar][numVar];
	codi::RealForward L[numVar][numVar];
	codi::RealForward M[numVar*numVar+1];

	for (int i = 0; i < numVar; ++i)
		for (int j = 0; j < numVar; ++j)
			L[i][j] = input[i*numVar + j];

#if 0
	printf("L = \n");
	for (int i = 0; i < numVar; ++i){
		for (int j = 0; j < numVar; ++j){

			printf("%10.7f ",L[i][j].getValue());

		}
		printf("\n");
	}
#endif	


	for (int i = 0; i < numVar; ++i)
		for (int j = 0; j < numVar; ++j) {

			LT[i][j] = 0.0;
		}





	for (int i = 0; i < numVar; ++i) {
		for (int j = 0; j < i+1; ++j){


			LT[j][i] = L[i][j];
		}
	}

#if 0
	printf("LT = \n");
	for (int i = 0; i < numVar; ++i){
		for (int j = 0; j < numVar; ++j){

			printf("%10.7f ",LT[i][j].getValue());

		}
		printf("\n");
	}
#endif	


	for (int i = 0; i < numVar; ++i)
		for (int j = 0; j < numVar; ++j)
			M[i*numVar + j] = 0;






	/* Multiplying matrix L and LT and storing in M */
	for (int i = 0; i < numVar; ++i)
		for (int j = 0; j < numVar; ++j)
			for (int k = 0; k < numVar; ++k) {

				M[i*numVar + j] = M[i*numVar + j] + L[i][k]*LT[k][j];
			}


#if 0
	printf("M = \n");
	for (int i = 0; i < numVar; ++i){
		for (int j = 0; j < numVar; ++j){

			printf("%10.7f ",M[i*numVar + j].getValue());

		}
		printf("\n");
	}
#endif	



	M[numVar*numVar] = input[numVar*numVar];

	codi::RealForward sigma = M[numVar*numVar]; 

	float *xi = new float[numVar];
	float *xj = new float[numVar];

	codi::RealForward **kernelValTable = new codi::RealForward*[N];

	for(int i=0; i<N;i++) {

		kernelValTable[i] = new codi::RealForward[N];

	}

	for (int i = 0; i < N; i++) 

		for (int j = 0; j < N; j++) kernelValTable[i][j] = 0.0;




	for (int i = 0; i < N; i++) {

		for (int j = 0; j < N; j++) {

			if(j>i){

				for (int k = 0; k < numVar; k++) {

					xi[k] = data[i*(numVar+1)+k];
					xj[k] = data[j*(numVar+1)+k];
				}


				kernelValTable[i][j] = gaussianKernel(xi, xj, sigma, M, numVar);

			}



		}



	}



	codi::RealForward lossFunc = 0.0;

	for (int i = 0; i < N; i++) {

#if 0
		printf("kernel regression for the sample number %d\n",i);

#endif


		codi::RealForward kernelSum = 0.0;
		for (int j = 0; j < N; j++) {

			if (i != j) {


				kernelSum += kernelValTable[i][j];

			}
		}

		codi::RealForward fApprox = 0.0;
		for (int j = 0; j < N; j++) {
			if (i != j) {
				fApprox += kernelValTable[i][j] * data[j*(numVar+1)+numVar];

			}
		}

		fApprox = fApprox / kernelSum;

#if 0
		printf("fApprox = %10.7f\n",fApprox.getValue());
		printf("fExact = %10.7f\n",data[i*(numVar+1)+numVar]);
#endif

		lossFunc += (fApprox - data[i*(numVar+1)+numVar]) * (fApprox - data[i*(numVar+1)+numVar]);

	} // end of i loop

	lossFunc = lossFunc / N;


#if 0

	printf("lossFunc = %10.7f\n",lossFunc.getValue());
#endif	

	*result = lossFunc; 
	delete[] xi;
	delete[] xj;

	for(int i=0; i<N;i++) {

		delete[] kernelValTable[i];

	}

	delete[] kernelValTable;


}




__global__ void calculateKernelValues_b(float *ab, float *X, float *kernelValTable, float *kernelValTableb, int N) {

	//__shared__ float MShared[numVar*numVar +1];

    /*Have to be careful not to exceed the size of the shared memory available
	Right now we are well within the maximum allowed size (64Kb) */

    int idx = threadIdx.x;  //Helpful for intra-block operations
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//	printf("tid = %d\n",tid)
	
	// if(idx < numVar*numVar+1){
	// 	MShared[idx] = MDevice[idx];
	// 	//fprintf(outMShared,"%10.7f\n",MShared[tid]);
    // }
    __syncthreads(); //Block size = 32 --> covered by a warp so no need to explicitly sync threads. However, this would be required if 
					 //block size is increased beyond 32
					 
    float sigma = MDevice[numVar*numVar];
    //float sigma = MShared[numVar*numVar];
	float sigmab = 0.0;
	/* calculate column index */
	int indx2 = tid%N;
	/* calculate row index */
	int indx1 = tid/N;



	if (indx2 > indx1) {
		int off1 = indx1*(numVar+1);
		int off2 = indx2*(numVar+1);

		float diff[numVar];

		float tempVec[numVar];
		float tempVecb[numVar];

		float sumb = 0.0;


		float kernelValb = 0.0;
		float temp;
		float temp0;
		float tempb;
		float tempb0;

		#pragma unroll
		for (int k = 0; k < numVar; ++k)
			diff[k] = X[off1 + k] - X[off2 + k];

		float sum = 0.0;
		#pragma unroll
		for (int i = 0; i < numVar; ++i) {
			#pragma unroll
			for (int j = 0; j < numVar; ++j)
				sum = sum + MDevice[i*numVar+j]*diff[j];
				//sum = sum + MShared[i*numVar+j]*diff[j];
			tempVec[i] = sum;
			sum = 0.0;
		}
		sum = 0.0;
		#pragma unroll
		for (int i = 0; i < numVar; ++i)
			sum = sum + tempVec[i]*diff[i];

		float sqr_two_pi;
		sqr_two_pi = sqrt(2.0*3.14159265359);
		float kernelVal = 1.0/(sigma*sqr_two_pi)*exp(-sum/(2*sigma*sigma))+10E-12;


		kernelValb = kernelValTableb[indx1*N + indx2];


		kernelValTableb[indx1*N + indx2] = 0.0;
		tempb = kernelValb/(sqr_two_pi*sigma);





		temp = 2*(sigma*sigma);
		temp0 = sum/temp; // temp0 = sum/2*(sigma*sigma)
		tempb0 = -(exp(-temp0)*tempb/temp); // -(exp(-sum/2*(sigma*sigma))*kernelValb/(sqr_two_pi*sigma)/temp) 
		sumb = tempb0;
		sigmab = -(exp(-temp0)*tempb/sigma) - 2*2*temp0*sigma*tempb0;



		#pragma unroll
		for (int i = 0; i < numVar; ++i){

			tempVecb[i] = 0.0;

		}

		#pragma unroll
		for (int i = numVar-1; i > -1; --i){

			tempVecb[i] = tempVecb[i] + diff[i]*sumb;
		}

		#pragma unroll
		for (int i = numVar-1; i > -1; --i) {
			sumb = tempVecb[i];
			tempVecb[i] = 0.0;
			#pragma unroll
			for (int j = numVar-1; j > -1; --j){

				float addTerm = diff[j]*sumb;


				atomicAdd( &ab[i*numVar + j],addTerm );

			}
			//				ab[i*numVar + j] = ab[i*numVar + j] + diff[j]*sumb;
		}
	} 
	atomicAdd( &ab[numVar*numVar],sigmab );
	//	ab[numVar*numVar] = ab[numVar*numVar] + sigmab;


}



__global__ void calculateKernelValues(float *X, float *kernelValTable, int N){


	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	float sigma = MDevice[numVar*numVar];

	/* calculate column index */
	int indx2 = tid%N;

	/* calculate row index */
	int indx1 = tid/N;

	if(indx2 > indx1){

		int off1 = indx1*(numVar+1);
		int off2 = indx2*(numVar+1);

		float diff[numVar];

		for (int k = 0; k < numVar; k++) {

			diff[k] = X[off1+k] - X[off2+k];

		}


		float tempVec[numVar];
		float sum = 0.0;

		for (int i = 0; i < numVar; i++) {
			for (int j = 0; j < numVar; j++) {

				sum = sum + MDevice[i*numVar+j] * diff[j];
			}

			tempVec[i] = sum;
			sum = 0.0;

		}


		sum = 0.0;

		for (int i = 0; i < numVar; i++) {

			sum = sum + tempVec[i] * diff[i];
		}




		float sqr_two_pi = sqrt(2.0 * 3.14159265359);

		float kernelVal = (1.0 / (sigma * sqr_two_pi))* exp(-sum / (2 * sigma * sigma)) + 10E-12;



		kernelValTable[indx1*N+indx2]= kernelVal;

	}



}

__global__  void calculateLossKernel(float *X,float *kernelValTable, float *sum, int N){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < N){


		float lossFunc = 0.0;

		float kernelSum = 0.0;

		for(int i=0; i<N; i++){

			if(tid != i){

				int indxKernelValTable;
				if(i<tid) {


					indxKernelValTable = i*N+tid;

				}
				else{

					indxKernelValTable = tid*N+i;

				}

				kernelSum += kernelValTable[indxKernelValTable];

			}



		}

		float fapprox=0.0;
		for(int i=0; i<N; i++){

			if(tid != i){
				int indxKernelValTable;

				if(i<tid) {

					indxKernelValTable = i*N+tid;

				}
				else{

					indxKernelValTable = tid*N+i;

				}

				fapprox += (kernelValTable[indxKernelValTable]/kernelSum)* X[i*(numVar+1)+numVar];

			}




		}


		//		lossFunc = (fapprox - X[tid*(numVar+1)+numVar]) * (fapprox - X[tid*(numVar+1)+numVar]);
		lossFunc = fabs(fapprox - X[tid*(numVar+1)+numVar]);
		sum[tid] = lossFunc;
	}

}


__global__  void calculateLossKernel_b(float *X, float *kernelValTable, float *
		kernelValTableb, float *sum, float *sumb, int N
) {


	int tid = threadIdx.x + blockIdx.x * blockDim.x;


	if (tid < N) {


		float lossFunc;
		float lossFuncb;
		float kernelSum=0.0;
		float kernelSumb;

		float fapproxb;


		for (int i = 0; i < N; ++i){
			if (tid != i) {
				int indxKernelValTable;
				if (i < tid)
					indxKernelValTable = i*N + tid;
				else
					indxKernelValTable = tid*N + i;
				kernelSum = kernelSum + kernelValTable[indxKernelValTable];


			} 
		}


		float fapprox = 0.0;
		for (int i = 0; i < N; ++i){
			if (tid != i) {
				int indxKernelValTable;
				if (i < tid)
					indxKernelValTable = i*N + tid;
				else
					indxKernelValTable = tid*N + i;
				fapprox = fapprox + kernelValTable[indxKernelValTable]/
						kernelSum*X[i*(numVar+1)+numVar];
#if 0
				if (isnan (fapprox ) || isinf (fapprox) ){

					printf("fapprox  is NaN or inf %10.7f\n",kernelSum);

					assert(0);

				}
#endif			

			} 
		}




		//		lossFunc = (fapprox - X[tid*(numVar+1)+numVar]) * (fapprox - X[tid*(numVar+1)+numVar]);
		lossFunc = fabs ( (fapprox - X[tid*(numVar+1)+numVar]) );


		sum[tid] = lossFunc;

		lossFuncb = sumb[tid];
#if 0
		if (isnan (sumb[tid] ) || isinf (sumb[tid]) ){

			printf("sumb[tid]  is NaN or inf!\n");



		}
#endif				


		sumb[tid] = 0.0;
		//		fapproxb = 2*(fapprox-X[tid*(numVar+1)+numVar])*lossFuncb;

		if((fapprox - X[tid*(numVar+1)+numVar]) >= 0){


			fapproxb = lossFuncb;

		}
		else{

			fapproxb = -lossFuncb;

		}



		kernelSumb = 0.0;
		for (int i = N-1; i > -1; --i) {

			if (tid != i)  {
				float tempb;
				int indxKernelValTable;
				if (i < tid)
					indxKernelValTable = i*N + tid;
				else
					indxKernelValTable = tid*N + i;



				tempb = X[i*(numVar+1)+numVar]*fapproxb/kernelSum;

				kernelValTableb[indxKernelValTable] = kernelValTableb[indxKernelValTable] + tempb;
				kernelSumb = kernelSumb - kernelValTable[indxKernelValTable]*
						tempb/kernelSum;
			}
		}
		for (int i = N-1; i > -1; --i) {

			if (tid != i)  {
				int indxKernelValTable;

				if (i < tid)
					indxKernelValTable = i*N + tid;
				else
					indxKernelValTable = tid*N + i;

				kernelValTableb[indxKernelValTable] = kernelValTableb[indxKernelValTable] + kernelSumb;



			}
		}


	} 

}

void calcLossFunGPU(float *result, float *input, float *data,int N){

	cudaEvent_t start, stop;
	cudaEventCreate( &start ) ;
	cudaEventCreate( &stop ) ;
	cudaEventRecord( start, 0 ) ;

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;


	float LT[numVar][numVar];
	float L[numVar][numVar];
	float M[numVar*numVar+1];



	for (int i = 0; i < numVar; i++)
		for (int j = 0; j < numVar; j++) {
			L[i][j]=input[i*numVar+j];

		}


#if 1
	printf("Data (host) = \n");

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < numVar+1; j++) {

			printf("%10.7f ", data[i*(numVar+1)+j]);

		}
		printf("\n");
	}

#endif	



#if 1
	printf("L = \n");

	for (int i = 0; i < numVar; i++) {
		for (int j = 0; j < numVar; j++) {

			printf("%10.7f ", L[i][j]);

		}
		printf("\n");
	}

#endif




	for (int i = 0; i < numVar; i++)
		for (int j = 0; j < numVar; j++) {

			LT[i][j]=0.0;
		}




	for (int i = 0; i < numVar; i++) {
		for (int j = 0; j <= i; j++){

			LT[j][i] = L[i][j];
		}


	}
#if 1
	printf("LT = \n");

	for (int i = 0; i < numVar; i++) {
		for (int j = 0; j < numVar; j++) {

			printf("%10.7f ", LT[i][j]);

		}
		printf("\n");
	}

#endif

	for(int i = 0; i < numVar; ++i)
		for(int j = 0; j < numVar; ++j)
		{
			M[i*numVar+j]=0;
		}

	/* Multiplying matrix L and LT and storing in M */
	for(int i = 0; i < numVar; ++i)
		for(int j = 0; j < numVar; ++j)
			for(int k = 0; k < numVar; ++k)
			{
				M[i*numVar+j] += L[i][k] * LT[k][j];

			}
#if 0
	printf("M = \n");

	for (int i = 0; i < numVar; i++) {
		for (int j = 0; j < numVar; j++) {

			printf("%10.7f ", M[i*numVar+j]);

		}
		printf("\n");
	}

#endif




	M[numVar*numVar] = input[numVar*numVar];



	/* copy the values of M to the constant memory */

	err= cudaMemcpyToSymbol(MDevice,M, (numVar*numVar+1)*sizeof(float));
	//for(int i=0; i<numVar*numVar+1; i++)MDevice[i] = M[i];

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy matrix M from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	float *dataDevice;


	// allocate the memory on the GPU for the data matrix
	err = cudaMalloc(&dataDevice, N *(numVar+1) * sizeof(float) ) ;

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	err = cudaMemcpy(dataDevice, data, N *(numVar+1) *sizeof(float), cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector data from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}



	float *kernelValuesDevice;

	// allocate the memory on the GPU for kernel Values
	err = cudaMalloc(&kernelValuesDevice, N*N* sizeof(float) ) ;

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector kernel values (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	int number_of_blocks = (N*N+number_of_threads_per_block-1)/number_of_threads_per_block;
	printf("Launching the first kernel with %d blocks...\n",number_of_blocks);




	calculateKernelValues<<<number_of_blocks,number_of_threads_per_block>>>(dataDevice, kernelValuesDevice, N);
	cudaDeviceSynchronize();



	printf("Kernel: calculateKernelValues is done ...\n");

#if 1

	/* this part is for validation */

	mat Mval(numVar,numVar);
	mat Xval(N,numVar);
	vec ys(N);

	float *kernelValuesHost = new float[N*N];


	err = cudaMemcpy(kernelValuesHost, kernelValuesDevice, N*N*sizeof(float), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector kernelValues from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}




	for(int i=0; i<numVar; i++){

		for(int j=0; j<numVar; j++){

			Mval(i,j) = M[i*numVar+j];

		}

	}

	printf("Mval = \n");
	Mval.print();

	for(int i=0; i<N; i++){

		for(int j=0; j<numVar; j++){

			Xval(i,j) = data[i*(numVar+1)+j];

		}

		ys(i) = data[i*(numVar+1)+(numVar)];

	}
	printf("Xval = \n");
	Xval.print();

	printf("ys = \n");
	ys.print();



	float sigma  = input[numVar*numVar];
	rowvec xi,xj;

	for(int i=0; i<N; i++){

		for(int j=i+1; j<N; j++){

			xi = Xval.row(i);
			xj = Xval.row(j);

			float kernelValCPU = calcKernelValCPU(xi, xj, Mval, sigma);
			float kernelValGPU = kernelValuesHost[i*N+j];
			printf("kernelValCPU = %19.7f, kernelValGPU = %19.7f, error = %15.12f\n",kernelValCPU,kernelValGPU,kernelValCPU-kernelValGPU);


		}



	}



	delete[] kernelValuesHost;

#endif	

	/* allocate the memory on the GPU for the kernelsum */
	float *lossSumDevice;

	err = cudaMalloc(&lossSumDevice, N * sizeof(float) ) ;

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector lossSumDevice (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	float *lossSumHost = new float[N];



	number_of_blocks = (N+number_of_threads_per_block-1)/number_of_threads_per_block;
	printf("Launching the second kernel with %d blocks...\n",number_of_blocks);


	calculateLossKernel<<<number_of_blocks,number_of_threads_per_block>>>(dataDevice,kernelValuesDevice, lossSumDevice, N);

	cudaDeviceSynchronize();



	printf("Kernel: calculateLossKernel is done ...\n");


	err = cudaMemcpy(lossSumHost, lossSumDevice, N*sizeof(float), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector lossSum from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}




#if 0

	vec lossValCPU(N);

	/* this part is for validation */
	for (int i=0; i<N; i++) {

		rowvec xi = Xval.row(i);

		float kernelSum=0.0;
		for(int j=0; j<N; j++){
			rowvec xj = Xval.row(j);

			if(i !=j){

				float kernelVal = calcKernelValCPU(xi, xj, Mval, sigma);
				kernelSum += kernelVal; 

			}


		}

		float sum = 0.0;
		for(int j=0; j<N; j++){
			rowvec xj = Xval.row(j);
			float kernelVal = calcKernelValCPU(xi, xj, Mval, sigma);
			if(i !=j){

				sum+=ys(j)*kernelVal;

			}






		}

		sum = sum/kernelSum;



		lossValCPU(i) = (ys(i)-sum)*(ys(i)-sum);

	}




	for (int i=0; i<N; i++) {
		printf( "lossGPU[%d] = %10.7f, lossGPU[%d] = %10.7f\n", i,lossSumHost[i],i, lossValCPU(i));
	}
#endif


	float totalLoss=0.0;
	for (int i=0; i<N; i++) {

		totalLoss+=lossSumHost[i];
	}


	*result = totalLoss/N;

	cudaEventRecord( stop, 0 ) ;
	cudaEventSynchronize( stop ) ;
	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime,start, stop ) ;
	printf( "Time to generate:%3.1f ms\n", elapsedTime );


	cudaEventDestroy( start ) ;
	cudaEventDestroy( stop ) ;

	delete[] lossSumHost;


	cudaFree(lossSumDevice);
	cudaFree(kernelValuesDevice);
	cudaFree(dataDevice);




}



void calcLossFunGPU_b(float *result, float *resultb, float *input,
		float *inputb, float *data, int N)
{


#if 0	
	printf("calling calcLossFunGPU_b...\n");
	printf("resultb = %10.7f\n",*resultb);
	printf("Data has %d points\n",N);
#endif	


#if 0	
	cudaEvent_t start, stop;
	cudaEventCreate( &start ) ;
	cudaEventCreate( &stop ) ;
	cudaEventRecord( start, 0 ) ;
#endif

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	float LT[numVar][numVar];
	float LTb[numVar][numVar];
	float L[numVar][numVar];
	float Lb[numVar][numVar];
	float M[numVar*numVar + 1];
	float Mb[numVar*numVar + 1];


	for (int i = 0; i < numVar; ++i){
		for (int j = 0; j < numVar; ++j){

			L[i][j] = input[i*numVar + j];
			Lb[i][j] = 0.0;
			LT[i][j] = 0.0;
			LTb[i][j] = 0.0;

		}
	}


	for (int i = 0; i < numVar; ++i) {
		for (int j = 0; j < i+1; ++j)

			LT[j][i] = L[i][j];
	}

#if 0
	printf("L = \n");
	for (int i = 0; i < numVar; ++i){
		for (int j = 0; j < numVar; ++j){

			printf("%10.7f ",L[i][j]);
		}
		printf("\n");
	}

	printf("LT = \n");
	for (int i = 0; i < numVar; ++i){
		for (int j = 0; j < numVar; ++j){

			printf("%10.7f ",LT[i][j]);
		}
		printf("\n");
	}

#endif	





	for (int i = 0; i < numVar; ++i)
		for (int j = 0; j < numVar; ++j) {

			M[i*numVar + j] = 0;
			Mb[i*numVar + j] = 0;
		}


	/* Multiplying matrix L and LT and storing in M */
	for (int i = 0; i < numVar; ++i)
		for (int j = 0; j < numVar; ++j)
			for (int k = 0; k < numVar; ++k)
				M[i*numVar + j] = M[i*numVar + j] + L[i][k]*LT[k][j];


#if 0
	printf("M = \n");
	for (int i = 0; i < numVar; ++i){
		for (int j = 0; j < numVar; ++j){

			printf("%10.7f ",M[i*numVar + j]);
		}
		printf("\n");
	}
#endif	

	M[numVar*numVar] = input[numVar*numVar];


#if 0
	printf("sigma = %10.7f\n", M[numVar*numVar]);
#endif


	/* copy the values of M to the constant memory "MDevice"*/

	err= cudaMemcpyToSymbol(MDevice,M, (numVar*numVar+1)*sizeof(float));
	//for(int i=0; i<numVar*numVar+1; i++)MDevice[i] = M[i];


	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy matrix M from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	float *dataDevice;


	/* allocate the memory on the GPU for the data matrix */
	err = cudaMalloc(&dataDevice, N *(numVar+1) * sizeof(float) ) ;

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector data (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	err = cudaMemcpy(dataDevice, data, N *(numVar+1) *sizeof(float), cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector data from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}



	float *kernelValuesDevice;

	// allocate the memory on the GPU for kernel Values
	err = cudaMalloc(&kernelValuesDevice, N*N* sizeof(float) ) ;

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector --kernelValuesDevice-- (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaMemset(kernelValuesDevice, 0, N*N* sizeof(float));

	float *kernelValuesDeviceb;

	// allocate the memory on the GPU for kernel Values
	err = cudaMalloc(&kernelValuesDeviceb, N*N* sizeof(float) ) ;

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector --kernelValuesDeviceb-- (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	//	cudaMemset(kernelValuesDeviceb, 0, N*N* sizeof(float));


	float *kernelValuesHostb = new float[N*N];

	for(int i=0; i<N*N; i++) {

		kernelValuesHostb[i] = 0.0;

	}

	err = cudaMemcpy(kernelValuesDeviceb, kernelValuesHostb, (N*N) *sizeof(float), cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector kernelValuesDeviceb from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}





	float *MDeviceb;

	// allocate the memory on the GPU for kernel Values
	err = cudaMalloc(&MDeviceb, (numVar*numVar + 1)* sizeof(float) ) ;

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector MDeviceb (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	float *MHostb = new float[numVar*numVar + 1];

	for(int i=0; i<numVar*numVar + 1; i++) {

		MHostb[i] = 0.0;

	}


	err = cudaMemcpy(MDeviceb, MHostb, (numVar*numVar + 1) *sizeof(float), cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector MHostb from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	delete[] MHostb;

	/* init adjoint of M to zero */
	//	cudaMemset(MDeviceb, 0, (numVar*numVar + 1)* sizeof(float));







	int number_of_blocks = (N*N+number_of_threads_per_block-1)/number_of_threads_per_block;

#if 0	
	printf("Launching the first primal kernel with %d blocks...\n",number_of_blocks);
#endif

	calculateKernelValues<<<number_of_blocks,number_of_threads_per_block>>>(dataDevice, kernelValuesDevice, N);
	cudaDeviceSynchronize();

#if 0
	printf("The primal kernel : calculateKernelValues is done...\n");
#endif



	float *lossSumDevice;

	// allocate the memory on the GPU for kernel Values
	err = cudaMalloc(&lossSumDevice, N*sizeof(float) ) ;

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector --lossSumDevice-- (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaMemset(lossSumDevice,0,N*sizeof(float));


	number_of_blocks = (N+number_of_threads_per_block-1)/number_of_threads_per_block;
#if 0
	printf("Launching the second primal kernel + adjoint with %d blocks...\n",number_of_blocks);
#endif


	float totalLossb = 0.0;
	totalLossb = *resultb/N;



	float *lossSumHostb = new float[N];

	for(int i=0; i<N;i++) lossSumHostb[i] = 0;; 


	float *lossSumDeviceb;

	// allocate the memory on the GPU for kernel Values
	err = cudaMalloc(&lossSumDeviceb, N*sizeof(float) ) ;

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector --lossSumDeviceb-- (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cudaMemset(lossSumDeviceb,0,N*sizeof(float));

	for (int i = N-1; i > -1; --i)
		lossSumHostb[i] = lossSumHostb[i] + totalLossb;
#if 0
	for (int i = N-1; i > -1; --i)
		printf("lossSumHostb[i] = %10.7f\n",i,lossSumHostb[i]);
#endif	

	err = cudaMemcpy(lossSumDeviceb, lossSumHostb, N *sizeof(float), cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector --lossSumDevice-- from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	//	cudaMemset(kernelValuesDeviceb, 0, (N*N)* sizeof(float));

	/* this subroutine evaluates the lossSumDevice and kernelValuesDeviceb */
	calculateLossKernel_b<<<number_of_blocks,number_of_threads_per_block>>>(dataDevice,kernelValuesDevice,kernelValuesDeviceb, lossSumDevice,lossSumDeviceb, N);

	cudaDeviceSynchronize();



	//cudaDeviceSynchronize();
#if 0
	printf("Kernel: calculateLossKernel_b is done ...\n");
#endif


	err = cudaMemcpy(kernelValuesHostb, kernelValuesDeviceb, N*N *sizeof(float), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector --kernelValues-- from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}




	float *lossSumHost = new float[N]();


	err = cudaMemcpy(lossSumHost, lossSumDevice, N*sizeof(float), cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector lossSum from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	float totalLoss=0.0;
	for (int i=0; i<N; i++) {

		totalLoss+=lossSumHost[i];
	}

	*result = totalLoss/N;
#if 0
	printf("result = %10.7f\n",*result);
#endif
	/* reverse sweep starts from here */



	cudaMemset(MDeviceb, 0, (numVar*numVar + 1)* sizeof(float));


	number_of_blocks = (N*N+number_of_threads_per_block-1)/number_of_threads_per_block;
#if 0
	printf("Launching the second adjoint kernel with %d blocks...\n",number_of_blocks);
#endif
	/* this subroutine evaluates MDeviceb */
	calculateKernelValues_b<<<number_of_blocks,number_of_threads_per_block>>>(MDeviceb, dataDevice, kernelValuesDevice, kernelValuesDeviceb, N);



	cudaDeviceSynchronize();


#if 0
	printf("Kernel: calculateKernelValues_b is done ...\n");
#endif
	for (int ii1 = 0; ii1 < numVar*numVar+1; ++ii1) {

		Mb[ii1] = 0.0;
	}

	err = cudaMemcpy(Mb, MDeviceb, (numVar*numVar+1)*sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector Mb from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


#if 0
	printf("Mb = \n");

	for (int i = 0; i < numVar; i++) {
		for (int j = 0; j < numVar; j++) {

			printf("%10.7f ", Mb[i*numVar+j]);

		}
		printf("\n");
	}



#endif	


	for (int i = numVar-1; i > -1; --i)
		for (int j = numVar-1; j > -1; --j)
			for (int k = numVar-1; k > -1; --k) {
				Lb[i][k] = Lb[i][k] + LT[k][j]*Mb[i*numVar+j];
				LTb[k][j] = LTb[k][j] + L[i][k]*Mb[i*numVar+j];
			}

	for (int i = numVar-1; i > -1; --i) {
		for (int j = i; j > -1; --j) {
			Lb[i][j] = Lb[i][j] + LTb[j][i];
			LTb[j][i] = 0.0;
		}
	}


	for (int i = numVar-1; i > -1; --i)
		for (int j = numVar-1; j > -1; --j) {
			inputb[i*numVar + j] = inputb[i*numVar + j] + Lb[i][j];
			Lb[i][j] = 0.0;
		}


	inputb[numVar*numVar] = Mb[numVar*numVar];

#if 0
	printf("inputb = \n");

	for (int i = 0; i < numVar; i++) {
		for (int j = 0; j < numVar; j++) {

			printf("%10.7f ", inputb[i*numVar+j]);

		}
		printf("\n");
	}

	printf("sigmab = %10.7f\n", inputb[numVar*numVar]);

#endif	

	cudaFree(dataDevice);
	cudaFree(kernelValuesDeviceb);
	cudaFree(kernelValuesDevice);
	cudaFree(lossSumDeviceb);
	cudaFree(lossSumDevice);
	cudaFree(MDeviceb);

	delete[] lossSumHost;
	delete[] lossSumHostb;
	delete[] kernelValuesHostb;

}


float kernelRegressor(fmat &X, fvec &y, frowvec &xp, fmat &M, float sigma) {

	int d = y.size();

	fvec kernelVal(d);
	fvec weight(d);
	float kernelSum = 0.0;
	float yhat = 0.0;



	for (int i = 0; i < d; i++) {

		frowvec xi = X.row(i);
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


/*
 * train the Mahalanobis matrix M and bandwidth parameter sigma
 * @param[in] data: sample data matrix (normalized values)
 * @param[in] max_cv_iter: number of iterations for cross validation loop
 * @param[out] wSvd: weight for svd regularization
 * @param[out] w12:  weight for mixed 12norm regularization
 * @param[out] M: Mahalanobis matrix
 * @param[out] sigma: bandwidth parameter for the Gaussian kernel
 *
 * */


int trainMahalanobisDistance(fmat &L, fmat &data, float &sigma, float &wSvd, float &w12,int max_cv_iter) {


	int max_opt_iter = 40000;

	unsigned int n = L.n_cols;
	unsigned int m = L.n_cols;
	float alpha = 0.9;

	if(m != n || m!=numVar || n!=numVar){

		fprintf(stderr,"Cols: %d and Rows: %d\n",n, m);
		fprintf(stderr,"Error: The Mahalanobis matrix is not square!\n");
		exit(-1);
	}

	int Ldim = numVar*numVar;

	/* lower diagonal matrix Lbest to keep the best L*/
	fmat bestL(numVar,numVar);
	bestL.fill(0.0);

	float bestsigma = 0.0;


	/* divide the data set into training and validation sets */

	unsigned int N = data.n_rows;


	/* size of the validation set, default to one fifth */
	unsigned int NvalidationSet = N/5;
	unsigned int Ntraining = N - NvalidationSet;


#if 1

	printf("number of training samples (core) = %d\n",Ntraining);
	printf("number of validation samples      = %d\n",NvalidationSet);

#endif




	fmat dataTraining      = data.submat( 0, 0, Ntraining-1, numVar );
	fmat dataValidation    = data.submat( Ntraining, 0, N-1, numVar );


	fmat XValidation = dataValidation.submat(0,0,NvalidationSet-1,numVar-1);
	fvec yValidation = dataValidation.col(numVar);
	fmat XTraining = dataTraining.submat(0,0,Ntraining-1,numVar-1);
	fvec yTraining = dataTraining.col(numVar);




#if 0

	printf("Training data set = \n");
	dataTraining.print();

	printf("Validation data set = \n");
	dataValidation.print();
#endif


#if 0
	printf("XTraining = \n");
	XTraining.print();
	printf("yTraining = \n");
	yTraining.print();
#endif

#if 0
	printf("XValidation = \n");
	XValidation.print();
	printf("yValidation = \n");
	yValidation.print();
#endif

	fvec wSvdtrial(max_cv_iter);
	fvec w12trial(max_cv_iter);


	if(max_cv_iter !=1){



		for(int i=0; i<max_cv_iter; i++){

			wSvdtrial(i) = pow(10.0,RandomFloat(-2,0.0));
			w12trial(i) = pow(10.0,RandomFloat(-2,0.0));
		}


#if 1
		printf("wSvdtrial = \n");
		wSvdtrial.print();
		printf("w12trial = \n");
		w12trial.print();
#endif


	}


	float *inputVec = new float[Ldim+1]();
	float *inputVecVel = new float[Ldim+1]();
	float *inputVecLocalBest = new float[Ldim+1]();
	float *inputVecb = new float[Ldim+1]();
	float *inputVecRegb = new float[Ldim]();
	float *gradientVec = new float[Ldim+1]();
	float *dataVecTraining = new float[Ntraining*(n+1)]();




#if 0
	printf("L = \n");
	for (int i = 0; i < numVar; i++){
		for (int j = 0; j < numVar; j++) {

			printf("%10.7f ",inputVec[i*numVar+j]);
		}
		printf("\n");
	}

	printf("sigma = %10.7f\n",inputVec[Ldim]);
#endif	


#if 1
	printf("copying training data...\n");
#endif	
	for (int i = 0; i < Ntraining; i++) {

		for (int j = 0; j < numVar+1; j++) {

			dataVecTraining[i*(n+1)+j ] = dataTraining(i, j);
		}
	}
#if 1
	printf("data copied = \n");

	for (int i = 0; i < Ntraining; i++) {

		for (int j = 0; j < numVar+1; j++) {

			printf("%10.7f ",dataVecTraining[i*(n+1)+j ]);
		}
		printf("\n");
	}

#endif	


	float optGenError = 10E14;

	/* cross validation loop to tune the weights for the regularization parameters */
	for(int iter_cv=0; iter_cv< max_cv_iter; iter_cv++){


		float learning_rateM = 0.0001;
		float learning_rateSigma = learning_rateM * 0.01;



		if(max_cv_iter !=1){

			wSvd = wSvdtrial(iter_cv);
			w12 =  w12trial(iter_cv);
		}
#if 1
		printf("Outer iteration = %d\n",iter_cv);
		printf("wSvd = %10.7f, w12 = %10.7f\n",wSvd,w12);
#endif		

		/* initialize the L matrix and sigma => everything is saved in the vector "inputVec" */

		for (int i = 0; i < numVar; i++)
			for (int j = 0; j < numVar; j++) {

				inputVec[i*numVar+j] = 0.0;
			}

		for (int i = 0; i < numVar; i++) {

			for (int j = 0; j <= i; j++) {

				if(i ==j) { /* main diagonal */

					inputVec[i*numVar+j] = 1.0+ RandomFloat(-0.1,0.1);
				}
				else {

					inputVec[i*numVar+j] = RandomFloat(0.0,0.1);
				}
			}
		}

		/* assign sigma */
		inputVec[Ldim] = RandomFloat(0.0,0.1);

		float lossVal,lossValb, regTerm;
		float objFunVal;
		lossVal = 0.0;
		lossValb = 1.0;

		for(int i=0;i<Ldim+1;i++) {

			inputVecb[i] = 0.0;
		}

		/* calculate the first gradient vector */

		printf("Evaluating the first gradient...\n");
		calcLossFunGPU_b(&lossVal, &lossValb, inputVec,inputVecb, dataVecTraining,Ntraining);
		printf("initial Loss (GPU Version)= %10.7f\n", lossVal);


#if 1	
		printf("gradient of the loss term = \n");

		for (int i = 0; i < numVar; i++) {
			for (int j = 0; j < numVar; j++) {

				printf("%10.7f ", inputVecb[i*numVar+j]);

			}
			printf("\n");
		}
		printf("sigma sensitivity = %10.7f\n", inputVecb[Ldim]);
#endif





		for(int i=0;i<Ldim+1;i++) {

			gradientVec[i]=inputVecb[i];
		}


#if 0

		/* call the CodiPack version for validation */

		codi::RealReverse *inputVecCodi = new codi::RealReverse[n*n+1];


		for(int i=0; i<n*n+1; i++){

			inputVecCodi[i] = inputVec[i];

		}

		codi::RealReverse lossValCodi = 0.0;
		float *inputVecbCodi = new float[n*n+1]();

		/* call the CodiPack version of "calcLossFunCPU" */ 

		printf("calling calcLossFunCPU (reverse AD)...\n");
		calcLossFunCPU(&lossValCodi,inputVecCodi, inputVecbCodi, dataVecTraining, Ntraining);

		printf("Lb (codipack result)= \n");

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {

				printf("%10.7f ", inputVecbCodi[i*n+j]);

			}
			printf("\n");
		}
		printf("sigmab = %10.7f\n", inputVecbCodi[n*n]);
		printf("lossValCodi = %10.7f\n", lossValCodi.getValue());	


#endif



#if 0
		printf("calculating regularization term...\n");
#endif
		for(int i=0;i<Ldim;i++) {

			inputVecRegb[i] = 0.0;
		}

		/* call the adjoint mode of the function to compute the regularization term */
		calcRegTerms(inputVec, inputVecRegb, &regTerm, wSvd, w12, n); 


#if 0	
		printf("gradient of the regularization term = \n");

		for (int i = 0; i < numVar; i++) {
			for (int j = 0; j < numVar; j++) {

				printf("%10.7f ", inputVecRegb[i*numVar+j]);

			}
			printf("\n");
		}
#endif

		objFunVal = lossVal + regTerm;

		printf("initial value of the objective function = %10.7f\n",objFunVal);

		/* add the regularization sensitivities to the gradient vector */

		for(int i=0;i<Ldim;i++) {

			gradientVec[i]+=inputVecRegb[i];
		}


#if 0	

		/* validation loop for the regularization term */


		float f0 = 0.0;
		float tempSave;
		calcRegTerms(inputVec, &f0, wSvd, w12, n);
		printf("f0 = %10.7f\n",f0);
		float epsValReg= 0.001;


		for (int i = 0; i < n; i++) {
			for (int j = 0; j <= i; j++) {

				printf("validating the (%d,%d) th element of M\n",i,j);
				tempSave = inputVec[i*n+j];
				inputVec[i*n+j]+=epsValReg;

				float f1 = 0.0;

				calcRegTerms(inputVec, &f1, wSvd, w12, n);
				printf("f1 = %10.7f, f0 = %10.7f\n",f1,f0);
				inputVec[i*n+j]= tempSave;

				float fdVal = (f1-f0)/epsValReg;

				printf("fd value = %10.7f, ad value = %10.7f\n",fdVal,inputVecRegb[i*n+j]);

				float f2,f2d;

				/* call forward mode */
				calcRegTerms(inputVec, &f2,&f2d, wSvd, w12, n, i*n+j);

				printf("primal value = %10.7f, forward ad value = %10.7f, ad value = %10.7f\n",f2,f2d,inputVecRegb[i*n+j]);


			}

		}




#endif

		/* optimization loop */


		/* check gradient */
		for(int i=0;i<Ldim;i++) {

			if( gradientVec[i] != gradientVec[i]){

				printf("gradientVec[%d] is NaN!\n",i);
				exit(1);

			}
		}


		float objectiveFunLocalBest = 10E14;



		for(int opt_iter=0 ; opt_iter < max_opt_iter; opt_iter++){



			/* update M */

			for (int i = 0; i < numVar; i++){
				for (int j = 0; j <= i; j++) {

					inputVec[i*numVar+j]= inputVec[i*numVar+j] + inputVecVel[i*numVar+j];


				}

			}


			for (int i = 0; i < numVar; i++){
				for (int j = 0; j <= i; j++) {

					if ( inputVec[i*numVar+j] < 0) {

						inputVec[i*numVar+j] = 10E-6;

					}


				}

			}

			/* update sigma */
			inputVec[Ldim]= inputVec[Ldim] + inputVecVel[Ldim];

			if(inputVec[Ldim] <= 0) {

				inputVec[Ldim] = 10E-06;
			}


			for(int i=0;i<Ldim+1;i++) {

				inputVecb[i] = 0.0;
			}

			/* calculate the gradient vector */
#if 0
			printf("evaluating gradient vector...\n");
#endif		
			calcLossFunGPU_b(&lossVal, &lossValb, inputVec,inputVecb, dataVecTraining,Ntraining);
#if 0
			printf("Loss (GPU Version)= %10.7f\n", lossVal);
#endif
			for(int i=0;i<Ldim+1;i++) {

				gradientVec[i]=inputVecb[i];
			}
#if 0
			printf("calculating the regularization term...\n");
#endif
			for(int i=0;i<Ldim;i++) {

				inputVecRegb[i] = 0.0;
			}

			/* call the adjoint mode of the function to compute the regularization term */
			calcRegTerms(inputVec, inputVecRegb, &regTerm, wSvd, w12, n); 

#if 0	
			printf("gradient of the regularization term = \n");

			for (int i = 0; i < numVar; i++) {
				for (int j = 0; j < numVar; j++) {

					printf("%10.7f ", inputVecRegb[i*numVar+j]);

				}
				printf("\n");
			}
#endif		


			/* add the regularization sensitivities to the gradient vector */

			for(int i=0;i<Ldim;i++) {

				gradientVec[i]+=inputVecRegb[i];
			}


			objFunVal = lossVal + regTerm;

			if(objFunVal < objectiveFunLocalBest){

				objectiveFunLocalBest = objFunVal;

				for(int i=0;i<Ldim+1;i++) {

					inputVecLocalBest[i]=inputVec[i];

				}




			}





			if(opt_iter % 100 == 0){


				printf("iter = %d, objective function = %10.7f, Leave One Out Error = %10.7f, Regularization term = %10.7f\n",opt_iter,objFunVal,lossVal, regTerm);

#if 0
				printf("L = \n");

				for (int i = 0; i < numVar; i++) {
					for (int j = 0; j < numVar; j++) {

						printf("%10.7f ", inputVec[i*numVar+j]);

					}
					printf("\n");
				}

				printf("sigma = %10.7f\n",inputVec[Ldim]);
#endif



			}


			/* update velocity vector */
			for(int i=0;i<Ldim;i++) {

				inputVecVel[i]=alpha* inputVecVel[i] - learning_rateM*gradientVec[i];

			}
			inputVecVel[Ldim]=alpha* inputVecVel[Ldim] - learning_rateSigma*gradientVec[Ldim];




		} /* end of local optimization loop */



		for (int i = 0; i < numVar; i++)
			for (int j = 0; j < numVar; j++) {

				L(i,j)= inputVecLocalBest[i*numVar+j];
			}

#if 1
		printf("local optimization result:\n");
		printf("L = \n");
		L.print();
		printf("sigma = %10.7f\n", inputVecLocalBest[Ldim]);

#endif
		sigma = inputVecLocalBest[Ldim];


		fmat M = L*trans(L);
#if 1
		printf("M = \n");
		M.print();
#endif
		float genError = 0.0;

		for(int i=0;i <NvalidationSet; i++){

			frowvec xp = XValidation.row(i);
			float ytilde = kernelRegressor(XTraining, yTraining, xp, M, sigma);
			float yexact = yValidation(i);

#if 0
			printf("x:\n");
			xp.print();
			printf("ytilde = %10.7f, yexact = %10.7f\n",ytilde,yexact);
#endif


			//			genError += (yexact-ytilde)*(yexact-ytilde);
			genError += fabs(yexact-ytilde);


		}

		genError = genError/NvalidationSet;

#if 1
		printf("Generalization error = %10.7f\n",genError);
#endif		
		if(genError < optGenError) {

#if 1
			printf("Better L has been found, updating L...\n");
#endif			
			bestL = L;
			bestsigma = sigma;
			optGenError = genError;


		}



	} /* end of cv loop */

	L = bestL;
	sigma = bestsigma;


	delete[] inputVec;
	delete[] inputVecb;
	delete[] inputVecRegb;
	delete[] dataVecTraining;
	delete[] gradientVec;
	return 0;

}


