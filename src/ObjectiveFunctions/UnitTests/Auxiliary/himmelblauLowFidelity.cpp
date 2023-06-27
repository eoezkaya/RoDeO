#include<stdio.h>
#include<math.h>

double Himmelblau(double *x) {

	double c1 = 1.1;
	double c2 = 0.9;
	double c3 = 11.05;
	double c4 = 1.05;
	double c5 = 1.08;
	double c6 = 6.97;
	return pow( (c1*x[0]*x[0]+c2*x[1]- c3 ), 2.0 ) + pow( (c4 * x[0]+ c5 * x[1]*x[1]- c6), 2.0 );

}

int main(void){

double x[2];
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = Himmelblau(x);

FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"%15.10f\n",result);
fclose(outp);

return 0;
}
