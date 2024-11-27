#include<stdio.h>
#include<math.h>
// Define constants for the Hartmann 6D function
const double alpha[4] = {1.0, 1.2, 3.0, 3.2};

const double A[4][6] = {
    {10.0, 3.0, 17.0, 3.5, 1.7, 8.0},
    {0.05, 10.0, 17.0, 0.1, 8.0, 14.0},
    {3.0, 3.5, 1.7, 10.0, 17.0, 8.0},
    {17.0, 8.0, 0.05, 10.0, 0.1, 14.0}
};

const double P[4][6] = {
    {0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886},
    {0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991},
    {0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650},
    {0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381}
};

double hartmann6D(double *x){
	
double result = 0.0;
    // Loop over the 4 terms in the Hartmann 6D function
    for (int i = 0; i < 4; i++) {
        double inner_sum = 0.0;
        
        // Calculate the inner sum for each dimension
        for (int j = 0; j < 6; j++) {
            inner_sum += A[i][j] * pow(x[j] - P[i][j], 2);
        }

        // Accumulate the result using the alpha coefficient
        result -= alpha[i] * exp(-inner_sum);
    }

    return result;



}

int main(void){

int dim = 6;
double x[dim];
FILE *inp = fopen("dv.dat","r");
for(int i = 0; i<dim; i++){
  fscanf(inp,"%lf",&x[i]);
}

fclose(inp);

double result = hartmann6D(x);
FILE *outp = fopen("objective.dat","w");
fprintf(outp,"%15.10f\n",result);
fclose(outp);


return 0;
}


