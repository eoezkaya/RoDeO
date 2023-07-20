#include<stdio.h>
#include<math.h>

double HimmelblauAdjLowFi(double *x, double *xb) {
	double c1 = 1.1;
	double c2 = 0.9;
	double c3 = 11.05;
	double c4 = 1.05;
	double c5 = 1.08;
	double c6 = 6.97;
	double f = 0.0;
	double fb = 0.0;
	double tempb;
	double tempb0;
	double Himmelblau;
	fb = 1.0;
	tempb = 2.0*pow(c1*(x[0]*x[0])-c3+c2*x[1], 2.0-1)*fb;
	tempb0 = 2.0*pow(c4*x[0]-c6+c5*(x[1]*x[1]), 2.0-1)*fb;
	xb[0] = xb[0] + c4*tempb0 + 2*x[0]*c1*tempb;
	xb[1] = xb[1] + 2*x[1]*c5*tempb0 + c2*tempb;
	return pow( (c1*x[0]*x[0]+c2*x[1]- c3 ), 2.0 ) + pow( (c4 * x[0]+ c5 * x[1]*x[1]- c6), 2.0 );

}

int main(void){

double x[2];
double xb[2];

FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = HimmelblauAdjLowFi(x, xb);
FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"%15.10f %15.10f %15.10f\n",result,xb[0],xb[1]);
fclose(outp);

return 0;
}
