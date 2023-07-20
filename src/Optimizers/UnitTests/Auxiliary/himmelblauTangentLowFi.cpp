#include<stdio.h>
#include<math.h>

double HimmelblauTangentLowFi(double *x, double *xd, double *fdot) {
	double c1 = 1.1;
	double c2 = 0.9;
	double c3 = 11.05;
	double c4 = 1.05;
	double c5 = 1.08;
	double c6 = 6.97;
	double f;
	double fd;
	double arg1;
	double arg1d;
	double arg2;
	double arg2d;
	arg1d = c1*2*x[0]*xd[0] + c2*xd[1];
	arg1 = c1*x[0]*x[0] + c2*x[1] - c3;
	arg2d = c4*xd[0] + c5*2*x[1]*xd[1];
	arg2 = c4*x[0] + c5*x[1]*x[1] - c6;
	fd = 2.0*pow(arg1, 2.0-1)*arg1d + 2.0*pow(arg2, 2.0-1)*arg2d;
	f = pow(arg1, 2.0) + pow(arg2, 2.0);
	*fdot = fd;
	return f;
}

int main(void){

double x[2];
double xd[2];

FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fscanf(inp,"%lf",&xd[0]);
fscanf(inp,"%lf",&xd[1]);

fclose(inp);

double fdot = 0.0;
double result = HimmelblauTangentLowFi(x, xd, &fdot);
FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"%15.10f %15.10f\n",result,fdot);
fclose(outp);

return 0;
}
