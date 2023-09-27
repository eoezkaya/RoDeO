#include<stdio.h>
#include<math.h>


double Rosenbrock(double *x ){
	return pow((1-x[0]),2) + 100.0 * pow((x[1] - x[0]*x[0]),2);	
}

int main(void){

double x[2];
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = Rosenbrock(x);
FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"%15.10f\n",result);
fclose(outp);
return 0;
}
