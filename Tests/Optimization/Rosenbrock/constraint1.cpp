#include<stdio.h>
#include<math.h>

double evaluateConstraint(double *x) {
	return x[1] - x[0]*x[0] ;
}
int main(void){

double x[2];
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = evaluateConstraint(x);

FILE *outp = fopen("constraintFunction1.dat","w");
fprintf(outp,"%15.10f\n",result);
fclose(outp);

return 0;
}
