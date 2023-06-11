#include<stdio.h>
#include<math.h>

double LinearTestFunction(double *x) {

	return 2.0*x[0] - x[1] + 1.5*x[2] - 5.1*x[3] + 0.1* x[4];

}


int main(void){

double x[5];
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fscanf(inp,"%lf",&x[2]);
fscanf(inp,"%lf",&x[3]);
fscanf(inp,"%lf",&x[4]);
fclose(inp);

double result = LinearTestFunction(x);

FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"%15.10f\n",result);
fclose(outp);

return 0;
}
