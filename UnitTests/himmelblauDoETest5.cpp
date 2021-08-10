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


double Constraint1Adj(double *x, double *xb) {
	
	xb[0] = 2.0*x[0];
	xb[1] = 2.0*x[1];
	return x[0]*x[0] + x[1]*x[1];

}

double Constraint2(double *x) {
	
	return x[0] + x[1];

}



int main(void){

double x[2];
double xb[2];
double xbConst1[2];
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = HimmelblauAdj(x,xb);
double constraint1 = Constraint1Adj(x,xbConst1);
double constraint2 = Constraint2(x);

FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"objective_function = %15.10f\n",result);
fprintf(outp,"objective_function_gradient = %15.10f, %15.10f\n",xb[0],xb[1]);
fprintf(outp,"Constraint1 = %15.10f\n", constraint1);
fprintf(outp,"Constraint1_gradient = %15.10f, %15.10f\n",xbConst1[0],xbConst1[1]);
fprintf(outp,"Constraint2 = %15.10f\n", constraint2);
fclose(outp);

return 0;
}
