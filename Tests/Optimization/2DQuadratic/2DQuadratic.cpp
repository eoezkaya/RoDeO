#include<stdio.h>
#include<math.h>
#include<armadillo>

double Quadratic(double *x ){
	return 0.5*x[0]*x[0] + x[1]*x[1] - x[0]*x[1] + 2*x[0] + 6*x[1];		
}

int main(void){

double x[2];
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = Quadratic(x);
FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"%15.10f\n",result);
fclose(outp);
return 0;
}
