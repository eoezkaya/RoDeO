#include<stdio.h>
#include<math.h>

double HimmelblauAdj(double *x, double *xb) {
	double tempb;
	double tempb0;
	tempb = 2.0*pow(x[0]*x[0]+x[1]-11.0, 2.0-1);
	tempb0 = 2.0*pow(x[0]+x[1]*x[1]-7.0, 2.0-1);
	xb[0] = tempb0 + 2*x[0]*tempb;
	xb[1] = 2*x[1]*tempb0 + tempb;

	return pow( (x[0]*x[0]+x[1]-11.0), 2.0 ) + pow( (x[0]+x[1]*x[1]-7.0), 2.0 );

}

double constraintAdj(double *x, double *xb) {
	xb[0] = 1.0;
	xb[1] = 1.0;

	return x[0]+x[1];

}


int main(void){

double x[2];
double xb[2];
double xbconstraint[2];
double constraintValue;
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = HimmelblauAdj(x, xb);
constraintValue = constraintAdj(x, xbconstraint);

FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"%15.10f %15.10f %15.10f\n",result,xb[0],xb[1]);
fprintf(outp,"%15.10f %15.10f %15.10f\n",constraintValue,xbconstraint[0],xbconstraint[1]);
fclose(outp);

return 0;
}
