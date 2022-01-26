#include<stdio.h>
#include<math.h>


double constraint2(double *x) {

	return x[0] * x[1];

}



int main(void){

double x[2];
FILE *inp = fopen("dv.dat","r");
fscanf(inp,"%lf",&x[0]);
fscanf(inp,"%lf",&x[1]);
fclose(inp);

double constraintVal = constraint2(x);
FILE *outp = fopen("constraint2.dat","w");
fprintf(outp,"constraint_function2 = %15.10f\n",constraintVal);
fclose(outp);

return 0;
}
