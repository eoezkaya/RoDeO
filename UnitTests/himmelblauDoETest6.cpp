#include<stdio.h>
#include<math.h>

double Himmelblau(double *x) {
	
	return pow( (x[0]*x[0]+x[1]-11.0), 2.0 ) + pow( (x[0]+x[1]*x[1]-7.0), 2.0 );

}


double Constraint1(double *x) {
	
	return x[0]*x[0] + x[1]*x[1];

}




int main(void){

double x[2];
double xb[2];
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = Himmelblau(x);
double constraint1 = Constraint1(x);

FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"objective_function = %15.10f\n",result);
fprintf(outp,"Constraint1 = %15.10f\n", constraint1);
fclose(outp);

return 0;
}
