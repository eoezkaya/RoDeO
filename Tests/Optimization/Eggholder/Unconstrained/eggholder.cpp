#include<stdio.h>
#include<math.h>

double Eggholder(double *x){
	return -(x[1]+47.0)*sin(sqrt(fabs(x[1]+0.5*x[0]+47.0)))-x[0]*sin(sqrt(fabs(x[0]-(x[1]+47.0) )));

}

int main(void){

double x[2];
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double result = Eggholder(x);


FILE *outp = fopen("objFunVal.dat","w");
fprintf(outp,"%15.10f\n",result);
fclose(outp);

return 0;
}
