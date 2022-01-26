#include<stdio.h>
#include<math.h>



double Constraint2(double *x) {
	
	return x[0]+ x[1];

}




int main(void){

double x[2];
double xb[2];
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double constraint2 = Constraint2(x);

FILE *outp = fopen("constraint2.dat","w");
fprintf(outp,"Constraint2 = %15.10f\n", constraint2);
fclose(outp);

return 0;
}
