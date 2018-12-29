#include "kernel_regression.hpp"
#include "auxilliary_functions.hpp"
#include "Rodeo_macros.hpp"

#include <armadillo>
#include <codi.hpp>



using namespace arma;


double calcMetric(double *xi,double *xj, double ** M, int dim){
#if 0
	printf("calling calcMetric...\n");
#endif

	double *diff=new double[dim];



	for(int i=0; i<dim; i++){

		diff[i]=xi[i]-xj[i];
	}




	double *tempVec=new double[dim];



	double sum=0.0;

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			for (int k = 0; k < dim; k++) {
				sum = sum + M[i][k]*diff[k];
			}

			tempVec[i] = sum;
			sum = 0;
		}
	}





#if 0
	printf("diff = \n");
	diff.print();
#endif



	sum=0.0;

	for (int i = 0; i < dim; i++) {

		sum = sum + tempVec[i]*diff[i];
	}



	delete [] diff;
	delete [] tempVec;
	return sum;

}


double gaussianKernel(rowvec &xi,
		rowvec &xj,
		double sigma,
		mat &M){
#if 0
	printf("calling gaussianKernel...\n");
	xi.print();
	xj.print();
#endif

	/* calculate distance between xi and xj with the matrix M */
	double metricVal = calcMetric(xi,xj,M);
#if 0
	printf("metricVal = %10.7f\n",metricVal);
#endif

	double sqr_two_pi = sqrt(2.0*datum::pi);

	double kernelVal = (1.0/(sigma*sqr_two_pi))*exp(-metricVal/(2*sigma*sigma));
#if 0
	printf("kernelVal = %10.7f\n",kernelVal);
#endif
	return kernelVal;


}


double gaussianKernel(double *xi,
		double *xj,
		double sigma,
		double **M,
		int dim){

	const double PI=3.14159265359;
#if 0
	printf("calling gaussianKernel...\n");
#endif

	/* calculate distance between xi and xj with the matrix M */
	double metricVal = calcMetric(xi,xj,M,dim);
#if 0
	printf("metricVal = %10.7f\n",metricVal);
#endif

	double sqr_two_pi = sqrt(2.0*PI);

	double kernelVal = (1.0/(sigma*sqr_two_pi))*exp(-metricVal/(2*sigma*sigma));
#if 0
	printf("kernelVal = %10.7f\n",kernelVal);
#endif
	return kernelVal;


}


double SIGN(double a, double b){

	if (b>=0.0) {
		return fabs(a);
	}
	else {
		return -fabs(a);
	}
}

codi::RealReverse SIGN(codi::RealReverse a, codi::RealReverse b){

	if (b>=0.0)
	{
		return fabs(a);
	}
	else {

		return -fabs(a);
	}
}

codi::RealForward SIGN(codi::RealForward a, codi::RealForward b){

	if (b>=0.0)
	{
		return fabs(a);
	}
	else {

		return -fabs(a);
	}
}

double PYTHAG(double a, double b)
{
	double at = fabs(a), bt = fabs(b), ct, result;

	if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
	else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
	else result = 0.0;
	return(result);
}

codi::RealReverse PYTHAG(codi::RealReverse a, codi::RealReverse b)
{
	codi::RealReverse at = fabs(a), bt = fabs(b), ct, result;

	if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
	else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
	else result = 0.0;
	return(result);
}
codi::RealForward PYTHAG(codi::RealForward a, codi::RealForward b)
{
	codi::RealForward at = fabs(a), bt = fabs(b), ct, result;

	if (at > bt)       { ct = bt / at; result = at * sqrt(1.0 + ct * ct); }
	else if (bt > 0.0) { ct = at / bt; result = bt * sqrt(1.0 + ct * ct); }
	else result = 0.0;
	return(result);
}

double dsvd(mat &M, mat&X, vec &ys, double sigma)
{
	int flag, i, its, j, jj, k, l=0, nm;
	double c, f, h, s, x, y, z;
	double anorm = 0.0, g = 0.0, scale = 0.0;



	int m = M.n_rows;
	int n = M.n_cols;

	if (m < n)
	{
		fprintf(stderr, "#rows must be > #cols \n");
		return(0);
	}


	int dim= X.n_cols;
	int N= X.n_cols;


	double **Xin;
	Xin=new double*[N];

	for(i=0;i<dim;i++){

		Xin[i]= new double[n];
	}

	for(i=0;i<N; i++){

		for(j=0;j<dim; j++){

			Xin[i][j]=X(i,j);

		}


	}

	double **a;
	a=new double*[n];

	for(i=0;i<n;i++){

		a[i]= new double[n];
	}

	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			a[i][j]=M(i,j);

	double *xp=new double[dim];
	double *xi=new double[dim];

	double *kernelVal=new double[N];

	double kernelSum=0.0;

	for(i=0; i<N; i++){
		for(k=0; k<dim; k++) xp[k]=Xin[i][k];

		for(j=0; j<N; j++){

			if(i!=j){
				for(k=0; k<dim; k++) xi[k]=Xin[j][k];
				kernelVal[i] = gaussianKernel(xi,xp,sigma,a,dim);
				kernelSum+=kernelVal[i];

			}
		}


	}



	double **v;
	v=new double*[n];

	for(i=0;i<n;i++){

		v[i]= new double[n];
	}
	double *w=new double[n];




	double* rv1 = (double *)malloc((unsigned int) n*sizeof(double));

	/* Householder reduction to bidiagonal form */
	for (i = 0; i < n; i++)
	{
		/* left-hand reduction */
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m)
		{
			for (k = i; k < m; k++)
				scale += fabs(a[k][i]);
			if (scale)
			{
				for (k = i; k < m; k++)
				{
					a[k][i] = (a[k][i]/scale);
					s += (a[k][i] * a[k][i]);
				}
				f = a[i][i];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][i] = (f - g);
				if (i != n - 1)
				{
					for (j = l; j < n; j++)
					{
						for (s = 0.0, k = i; k < m; k++)
							s += (a[k][i] * a[k][j]);
						f = s / h;
						for (k = i; k < m; k++)
							a[k][j] += (f * a[k][i]);
					}
				}
				for (k = i; k < m; k++)
					a[k][i] = (a[k][i]*scale);
			}
		}
		w[i] = (scale * g);

		/* right-hand reduction */
		g = s = scale = 0.0;
		if (i < m && i != n - 1)
		{
			for (k = l; k < n; k++)
				scale += fabs(a[i][k]);
			if (scale)
			{
				for (k = l; k < n; k++)
				{
					a[i][k] = (a[i][k]/scale);
					s += (a[i][k] * a[i][k]);
				}
				f = a[i][l];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][l] = (f - g);
				for (k = l; k < n; k++)
					rv1[k] = a[i][k] / h;
				if (i != m - 1)
				{
					for (j = l; j < m; j++)
					{
						for (s = 0.0, k = l; k < n; k++)
							s += (a[j][k] * a[i][k]);
						for (k = l; k < n; k++)
							a[j][k] += (s * rv1[k]);
					}
				}
				for (k = l; k < n; k++)
					a[i][k] = (a[i][k]*scale);
			}
		}
		anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}

	/* accumulate the right-hand transformation */
	for (i = n - 1; i >= 0; i--)
	{
		if (i < n - 1)
		{
			if (g)
			{
				for (j = l; j < n; j++)
					v[j][i] = ((a[i][j] / a[i][l]) / g);
				/* double division to avoid underflow */
				for (j = l; j < n; j++)
				{
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
	for (i = n - 1; i >= 0; i--)
	{
		l = i + 1;
		g = w[i];
		if (i < n - 1)
			for (j = l; j < n; j++)
				a[i][j] = 0.0;
		if (g)
		{
			g = 1.0 / g;
			if (i != n - 1)
			{
				for (j = l; j < n; j++)
				{
					for (s = 0.0, k = l; k < m; k++)
						s += (a[k][i] * a[k][j]);
					f = (s / a[i][i]) * g;
					for (k = i; k < m; k++)
						a[k][j] += (f * a[k][i]);
				}
			}
			for (j = i; j < m; j++)
				a[j][i] = (a[j][i]*g);
		}
		else
		{
			for (j = i; j < m; j++)
				a[j][i] = 0.0;
		}
		++a[i][i];
	}

	/* diagonalize the bidiagonal form */
	for (k = n - 1; k >= 0; k--)
	{                             /* loop over singular values */
		for (its = 0; its < 30; its++)
		{                         /* loop over allowed iterations */
			flag = 1;
			for (l = k; l >= 0; l--)
			{                     /* test for splitting */
				nm = l - 1;
				if (fabs(rv1[l]) + anorm == anorm)
				{
					flag = 0;
					break;
				}
				if (fabs(w[nm]) + anorm == anorm)
					break;
			}
			if (flag)
			{
				c = 0.0;
				s = 1.0;
				for (i = l; i <= k; i++)
				{
					f = s * rv1[i];
					if (fabs(f) + anorm != anorm)
					{
						g = w[i];
						h = PYTHAG(f, g);
						w[i] = h;
						h = 1.0 / h;
						c = g * h;
						s = (- f * h);
						for (j = 0; j < m; j++)
						{
							y = a[j][nm];
							z = a[j][i];
							a[j][nm] = (y * c + z * s);
							a[j][i] = (z * c - y * s);
						}
					}
				}
			}
			z = w[k];
			if (l == k)
			{                  /* convergence */
				if (z < 0.0)
				{              /* make singular value nonnegative */
					w[k] = (-z);
					for (j = 0; j < n; j++)
						v[j][k] = (-v[j][k]);
				}
				break;
			}
			if (its >= 30) {
				free((void*) rv1);
				fprintf(stderr, "No convergence after 30,000! iterations \n");
				return(0);
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
			for (j = l; j <= nm; j++)
			{
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
				for (jj = 0; jj < n; jj++)
				{
					x = v[jj][j];
					z = v[jj][i];
					v[jj][j] = (x * c + z * s);
					v[jj][i] = (z * c - x * s);
				}
				z = PYTHAG(f, h);
				w[j] = z;
				if (z)
				{
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = (c * g) + (s * y);
				x = (c * y) - (s * g);
				for (jj = 0; jj < m; jj++)
				{
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


#if 1
	printf("singular values of M=\n");

	for (i = 0; i < n; i++){

		printf("%10.7f\n",w[i]);
	}
#endif

	double temp,sum;
	for (i = 0; i < n; ++i)
	{
		for (j = i + 1; j < n; ++j)
		{

			if (w[i] < w[j])

			{
				temp =  w[i];
				w[i] = w[j];
				w[j] = temp;
			}
		}
	}

#if 1
	printf("singular values of M=\n");

	for (i = 0; i < n; i++){

		printf("%10.7f\n",w[i]);
	}
#endif
	sum=0.0;

	for (i = 0; i < n; i++)
	{
#if 1
		printf("%d * %10.7f = %10.7f\n",i+1,w[i],(i+1)*w[i]);
#endif
		sum = sum+(i+1)*w[i];
	}
#if 1
	printf("sum = %10.7f\n",sum);
#endif
	for(i=0;i<n;i++){
		delete[] v[i];
		delete[] a[i];

	}
	delete[] a;
	delete[] v;
	delete[] w;



	return(sum);


}


double dsvdAdj(mat &M, mat &gradient)
{
	int flag, i, its, j, jj, k, l=0, nm;
	codi::RealReverse c, f, h, s, x, y, z;
	codi::RealReverse anorm = 0.0, g = 0.0, scale = 0.0;



	int m = M.n_rows;
	int n = M.n_cols;

	if (m < n)
	{
		fprintf(stderr, "#rows must be > #cols \n");
		return(0);
	}

	codi::RealReverse **Min;
	Min=new codi::RealReverse*[n];

	for(i=0;i<n;i++){

		Min[i]= new codi::RealReverse[n];
	}
	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			Min[i][j]=M(i,j);



	codi::RealReverse **a;
	a=new codi::RealReverse*[n];

	for(i=0;i<n;i++){

		a[i]= new codi::RealReverse[n];
	}

	codi::RealReverse **v;
	v=new codi::RealReverse*[n];

	for(i=0;i<n;i++){

		v[i]= new codi::RealReverse[n];
	}
	codi::RealReverse *w=new codi::RealReverse[n];



	codi::RealReverse::TapeType& tape = codi::RealReverse::getGlobalTape();
	tape.setActive();


	for(i=0; i<n; i++){
		for(j=0; j<n; j++){

			tape.registerInput(Min[i][j]);
		}
	}


	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			a[i][j]=Min[i][j];

	codi::RealReverse* rv1 = (codi::RealReverse *)malloc((unsigned int) n*sizeof(codi::RealReverse));

	/* Householder reduction to bidiagonal form */
	for (i = 0; i < n; i++)
	{
		/* left-hand reduction */
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m)
		{
			for (k = i; k < m; k++)
				scale += fabs(a[k][i]);
			if (scale!=0.0)
			{
				for (k = i; k < m; k++)
				{
					a[k][i] = (a[k][i]/scale);
					s += (a[k][i] * a[k][i]);
				}
				f = a[i][i];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][i] = (f - g);
				if (i != n - 1)
				{
					for (j = l; j < n; j++)
					{
						for (s = 0.0, k = i; k < m; k++)
							s += (a[k][i] * a[k][j]);
						f = s / h;
						for (k = i; k < m; k++)
							a[k][j] += (f * a[k][i]);
					}
				}
				for (k = i; k < m; k++)
					a[k][i] = (a[k][i]*scale);
			}
		}
		w[i] = (scale * g);

		/* right-hand reduction */
		g = s = scale = 0.0;
		if (i < m && i != n - 1)
		{
			for (k = l; k < n; k++)
				scale += fabs(a[i][k]);
			if (scale!=0.0)
			{
				for (k = l; k < n; k++)
				{
					a[i][k] = (a[i][k]/scale);
					s += (a[i][k] * a[i][k]);
				}
				f = a[i][l];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][l] = (f - g);
				for (k = l; k < n; k++)
					rv1[k] = a[i][k] / h;
				if (i != m - 1)
				{
					for (j = l; j < m; j++)
					{
						for (s = 0.0, k = l; k < n; k++)
							s += (a[j][k] * a[i][k]);
						for (k = l; k < n; k++)
							a[j][k] += (s * rv1[k]);
					}
				}
				for (k = l; k < n; k++)
					a[i][k] = (a[i][k]*scale);
			}
		}
		anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}

	/* accumulate the right-hand transformation */
	for (i = n - 1; i >= 0; i--)
	{
		if (i < n - 1)
		{
			if (g!=0.0)
			{
				for (j = l; j < n; j++)
					v[j][i] = ((a[i][j] / a[i][l]) / g);
				/* double division to avoid underflow */
				for (j = l; j < n; j++)
				{
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
	for (i = n - 1; i >= 0; i--)
	{
		l = i + 1;
		g = w[i];
		if (i < n - 1)
			for (j = l; j < n; j++)
				a[i][j] = 0.0;
		if (g!=0.0)
		{
			g = 1.0 / g;
			if (i != n - 1)
			{
				for (j = l; j < n; j++)
				{
					for (s = 0.0, k = l; k < m; k++)
						s += (a[k][i] * a[k][j]);
					f = (s / a[i][i]) * g;
					for (k = i; k < m; k++)
						a[k][j] += (f * a[k][i]);
				}
			}
			for (j = i; j < m; j++)
				a[j][i] = (a[j][i]*g);
		}
		else
		{
			for (j = i; j < m; j++)
				a[j][i] = 0.0;
		}
		++a[i][i];
	}

	/* diagonalize the bidiagonal form */
	for (k = n - 1; k >= 0; k--)
	{                             /* loop over singular values */
		for (its = 0; its < 30; its++)
		{                         /* loop over allowed iterations */
			flag = 1;
			for (l = k; l >= 0; l--)
			{                     /* test for splitting */
				nm = l - 1;
				if (fabs(rv1[l]) + anorm == anorm)
				{
					flag = 0;
					break;
				}
				if (fabs(w[nm]) + anorm == anorm)
					break;
			}
			if (flag)
			{
				c = 0.0;
				s = 1.0;
				for (i = l; i <= k; i++)
				{
					f = s * rv1[i];
					if (fabs(f) + anorm != anorm)
					{
						g = w[i];
						h = PYTHAG(f, g);
						w[i] = h;
						h = 1.0 / h;
						c = g * h;
						s = (- f * h);
						for (j = 0; j < m; j++)
						{
							y = a[j][nm];
							z = a[j][i];
							a[j][nm] = (y * c + z * s);
							a[j][i] = (z * c - y * s);
						}
					}
				}
			}
			z = w[k];
			if (l == k)
			{                  /* convergence */
				if (z < 0.0)
				{              /* make singular value nonnegative */
					w[k] = (-z);
					for (j = 0; j < n; j++)
						v[j][k] = (-v[j][k]);
				}
				break;
			}
			if (its >= 30) {
				free((void*) rv1);
				fprintf(stderr, "No convergence after 30,000! iterations \n");
				return(0);
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
			for (j = l; j <= nm; j++)
			{
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
				for (jj = 0; jj < n; jj++)
				{
					x = v[jj][j];
					z = v[jj][i];
					v[jj][j] = (x * c + z * s);
					v[jj][i] = (z * c - x * s);
				}
				z = PYTHAG(f, h);
				w[j] = z;
				if (z!=0.0)
				{
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = (c * g) + (s * y);
				x = (c * y) - (s * g);
				for (jj = 0; jj < m; jj++)
				{
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

	for (i = 0; i < n; i++){

		printf("%10.7f\n",w[i].getValue());
	}
#endif

	codi::RealReverse temp,sum;
	for (i = 0; i < n; ++i)
	{
		for (j = i + 1; j < n; ++j)
		{

			if (w[i] < w[j])

			{
				temp =  w[i];
				w[i] = w[j];
				w[j] = temp;
			}
		}
	}

#if 0
	printf("singular values of M=\n");

	for (i = 0; i < n; i++){

		printf("%10.7f\n",w[i].getValue());
	}
#endif
	sum=0.0;

	for (i = 0; i < n; i++)
	{
#if 0
		printf("%d * %10.7f = %10.7f\n",i+1,w[i],(i+1)*w[i]);
#endif
		sum = sum+(i+1)*w[i];
	}
#if 0
	printf("sum = %10.7f\n",sum.getValue());
#endif


	for(i=0; i<n; i++)
		for(j=0; j<n; j++) sum+=fabs(Min[i][j]);

	tape.registerOutput(sum);

	tape.setPassive();
	sum.setGradient(1.0);
	tape.evaluate();

#if 0
	printf("matb=\n");

	for(i=0; i<n; i++){
		for(j=0; j<n; j++){

			gradient(i,j)=Min[i][j].getGradient();

			printf("%15.7f ",Min[i][j].getGradient());
		}

		printf("\n");
	}
#endif
	for(i=0; i<n; i++){
		for(j=0; j<n; j++){

			gradient(i,j)=Min[i][j].getGradient();

		}

	}


	tape.reset();

	for(i=0;i<n;i++){
		delete[] v[i];
		delete[] a[i];
		delete[] Min[i];
	}
	delete[] a;
	delete[] Min;
	delete[] v;
	delete[] w;
	return(sum.getValue());



}



double dsvdTl(mat &M, int indx1, int indx2, double *sensVal)
{
	int flag, i, its, j, jj, k, l=0, nm;
	codi::RealForward c, f, h, s, x, y, z;
	codi::RealForward anorm = 0.0, g = 0.0, scale = 0.0;



	int m = M.n_rows;
	int n = M.n_cols;

	if (m < n)
	{
		fprintf(stderr, "#rows must be > #cols \n");
		return(0);
	}

	codi::RealForward **Min;
	Min=new codi::RealForward*[n];

	for(i=0;i<n;i++){

		Min[i]= new codi::RealForward[n];
	}
	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			Min[i][j]=M(i,j);

	Min[indx1][indx2].setGradient(1.0);



	codi::RealForward **a;
	a=new codi::RealForward*[n];

	for(i=0;i<n;i++){

		a[i]= new codi::RealForward[n];
	}

	codi::RealForward **v;
	v=new codi::RealForward*[n];

	for(i=0;i<n;i++){

		v[i]= new codi::RealForward[n];
	}
	codi::RealForward *w=new codi::RealForward[n];






	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			a[i][j]=Min[i][j];

	codi::RealForward* rv1 = (codi::RealForward *)malloc((unsigned int) n*sizeof(codi::RealForward));

	/* Householder reduction to bidiagonal form */
	for (i = 0; i < n; i++)
	{
		/* left-hand reduction */
		l = i + 1;
		rv1[i] = scale * g;
		g = s = scale = 0.0;
		if (i < m)
		{
			for (k = i; k < m; k++)
				scale += fabs(a[k][i]);
			if (scale!=0.0)
			{
				for (k = i; k < m; k++)
				{
					a[k][i] = (a[k][i]/scale);
					s += (a[k][i] * a[k][i]);
				}
				f = a[i][i];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][i] = (f - g);
				if (i != n - 1)
				{
					for (j = l; j < n; j++)
					{
						for (s = 0.0, k = i; k < m; k++)
							s += (a[k][i] * a[k][j]);
						f = s / h;
						for (k = i; k < m; k++)
							a[k][j] += (f * a[k][i]);
					}
				}
				for (k = i; k < m; k++)
					a[k][i] = (a[k][i]*scale);
			}
		}
		w[i] = (scale * g);

		/* right-hand reduction */
		g = s = scale = 0.0;
		if (i < m && i != n - 1)
		{
			for (k = l; k < n; k++)
				scale += fabs(a[i][k]);
			if (scale!=0.0)
			{
				for (k = l; k < n; k++)
				{
					a[i][k] = (a[i][k]/scale);
					s += (a[i][k] * a[i][k]);
				}
				f = a[i][l];
				g = -SIGN(sqrt(s), f);
				h = f * g - s;
				a[i][l] = (f - g);
				for (k = l; k < n; k++)
					rv1[k] = a[i][k] / h;
				if (i != m - 1)
				{
					for (j = l; j < m; j++)
					{
						for (s = 0.0, k = l; k < n; k++)
							s += (a[j][k] * a[i][k]);
						for (k = l; k < n; k++)
							a[j][k] += (s * rv1[k]);
					}
				}
				for (k = l; k < n; k++)
					a[i][k] = (a[i][k]*scale);
			}
		}
		anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
	}

	/* accumulate the right-hand transformation */
	for (i = n - 1; i >= 0; i--)
	{
		if (i < n - 1)
		{
			if (g!=0.0)
			{
				for (j = l; j < n; j++)
					v[j][i] = ((a[i][j] / a[i][l]) / g);
				/* double division to avoid underflow */
				for (j = l; j < n; j++)
				{
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
	for (i = n - 1; i >= 0; i--)
	{
		l = i + 1;
		g = w[i];
		if (i < n - 1)
			for (j = l; j < n; j++)
				a[i][j] = 0.0;
		if (g!=0.0)
		{
			g = 1.0 / g;
			if (i != n - 1)
			{
				for (j = l; j < n; j++)
				{
					for (s = 0.0, k = l; k < m; k++)
						s += (a[k][i] * a[k][j]);
					f = (s / a[i][i]) * g;
					for (k = i; k < m; k++)
						a[k][j] += (f * a[k][i]);
				}
			}
			for (j = i; j < m; j++)
				a[j][i] = (a[j][i]*g);
		}
		else
		{
			for (j = i; j < m; j++)
				a[j][i] = 0.0;
		}
		++a[i][i];
	}

	/* diagonalize the bidiagonal form */
	for (k = n - 1; k >= 0; k--)
	{                             /* loop over singular values */
		for (its = 0; its < 30; its++)
		{                         /* loop over allowed iterations */
			flag = 1;
			for (l = k; l >= 0; l--)
			{                     /* test for splitting */
				nm = l - 1;
				if (fabs(rv1[l]) + anorm == anorm)
				{
					flag = 0;
					break;
				}
				if (fabs(w[nm]) + anorm == anorm)
					break;
			}
			if (flag)
			{
				c = 0.0;
				s = 1.0;
				for (i = l; i <= k; i++)
				{
					f = s * rv1[i];
					if (fabs(f) + anorm != anorm)
					{
						g = w[i];
						h = PYTHAG(f, g);
						w[i] = h;
						h = 1.0 / h;
						c = g * h;
						s = (- f * h);
						for (j = 0; j < m; j++)
						{
							y = a[j][nm];
							z = a[j][i];
							a[j][nm] = (y * c + z * s);
							a[j][i] = (z * c - y * s);
						}
					}
				}
			}
			z = w[k];
			if (l == k)
			{                  /* convergence */
				if (z < 0.0)
				{              /* make singular value nonnegative */
					w[k] = (-z);
					for (j = 0; j < n; j++)
						v[j][k] = (-v[j][k]);
				}
				break;
			}
			if (its >= 30) {
				free((void*) rv1);
				fprintf(stderr, "No convergence after 30,000! iterations \n");
				return(0);
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
			for (j = l; j <= nm; j++)
			{
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
				for (jj = 0; jj < n; jj++)
				{
					x = v[jj][j];
					z = v[jj][i];
					v[jj][j] = (x * c + z * s);
					v[jj][i] = (z * c - x * s);
				}
				z = PYTHAG(f, h);
				w[j] = z;
				if (z!=0.0)
				{
					z = 1.0 / z;
					c = f * z;
					s = h * z;
				}
				f = (c * g) + (s * y);
				x = (c * y) - (s * g);
				for (jj = 0; jj < m; jj++)
				{
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


#if 1
	printf("singular values of M=\n");

	for (i = 0; i < n; i++){

		printf("%10.7f\n",w[i].getValue());
	}
#endif

	codi::RealForward temp,sum;
	for (i = 0; i < n; ++i)
	{
		for (j = i + 1; j < n; ++j)
		{

			if (w[i] < w[j])

			{
				temp =  w[i];
				w[i] = w[j];
				w[j] = temp;
			}
		}
	}

#if 1
	printf("singular values of M=\n");

	for (i = 0; i < n; i++){

		printf("%10.7f\n",w[i].getValue());
	}
#endif
	sum=0.0;

	for (i = 0; i < n; i++)
	{
#if 1
		printf("%d * %10.7f = %10.7f\n",i+1,w[i],(i+1)*w[i]);
#endif
		sum = sum+(i+1)*w[i];
	}
#if 1
	printf("sumd = %10.7f\n",sum.getGradient());
	printf("sum = %10.7f\n",sum.getValue());
#endif


	for(i=0;i<n;i++){
		delete[] v[i];
		delete[] a[i];
		delete[] Min[i];
	}
	delete[] a;
	delete[] Min;
	delete[] v;
	delete[] w;

	*sensVal = sum.getGradient();
	return(sum.getValue());



}












int trainMahalanobisDistance(void){

	int n = 10;
	mat M(n,n,fill::randu);


	mat X(50,n,fill::randu);

	vec ys(50,fill::randu);


	double sigma=0.01;
	//	//	M=M*0.01;
	//		M.fill(5.0);
	//
	//
	//	mat U,V;
	//
	//	vec s;
	//
	//	dsvd(M);
	//	svd(U,s,V,M);
	//	s.print();
	//
	//
	//    exit(1);



	codi::RealForward **adot;
	adot=new codi::RealForward*[n];

	for(int i=0;i<n;i++){

		adot[i]= new codi::RealForward[n];
	}
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			adot[i][j]=M(i,j);




	double resultAdj;




#if 0

	double resultTl,resultTlpert;
	mat fd(n,n);
	mat forward_res(n,n);
	mat adjoint_res(n,n);
	mat Mcopy(n,n);
	double temp;
	const double perturbation_param = 0.0001;
	double adot_save;

	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++){

			resultTl = dsvdTl(M,i,j,&forward_res(i,j));
			Mcopy=M;
			M(i,j)+=perturbation_param;
			resultTlpert = dsvdTl(M,i,j,&temp);

			fd(i,j)= (resultTlpert-resultTl)/perturbation_param;
			M=Mcopy;


		}

	resultAdj = dsvdAdj(M,adjoint_res);



	printf("fd results =\n");
	fd.print();
	printf("forward AD results =\n");
	forward_res.print();
	printf("error  between forward AD and fd=\n");
	mat errorForAD = fd - forward_res;
	errorForAD.print();
	printf("error  between forward AD and reverse AD=\n");
	mat errorRevAD = forward_res-adjoint_res;
	errorRevAD.print();

#endif



	/* optimization stage */

	int maxoOptIter = 10000;
	int optIter=0;
	mat gradient(n,n);
	double stepSizeInit= 0.0001;
	double stepSize= stepSizeInit;
	double fStep,fpreviousIter;

	fpreviousIter = dsvd(M,X,ys,sigma);

	fStep = dsvdAdj(M,gradient);



	mat Msave(n,n);

	while(1){



		Msave = M;
		M = M-stepSize*gradient;

		for(int i=0;i<n;i++)
			for(int j=0;j<n;j++){

				if(M(i,j) < 0.0) M(i,j)=0.0;
			}
#if 0
		printf("stepSize = %10.7f\n",stepSize);
		printf("evaluating gradient vector...\n");
		M.print();
#endif


		fStep = dsvdAdj(M,gradient);



#if 1
		printf("fStep= %10.7f\n",fStep);

		//		printf("gradient=\n");
		//		gradient.print();
#endif


		fpreviousIter = fStep;




		optIter++;
		if(optIter > maxoOptIter || stepSize < 10E-10) {
#if 1
			printf("optimization prcedure is terminating with iter =%d...\n",optIter);
#endif
			break;

		}
	}

	M.print();
	dsvd(M,X,ys,sigma);

	return 0;
}





double kernelRegressor(mat &X, vec &y, rowvec &xp, mat &M, double sigma){

	int d = y.size();

	vec kernelVal(d);
	vec weight(d);
	double kernelSum=0.0;
	double yhat=0.0;

	for(int i=0; i<d; i++){

		rowvec xi = X.row(i);
		kernelVal(i) = gaussianKernel(xi,xp,sigma,M);
		kernelSum+=kernelVal(i);
	}

	for(int i=0; i<d; i++){

		weight(i)=kernelVal(i)/kernelSum;
		yhat = y(i)*weight(i);
	}

	return yhat;


}

