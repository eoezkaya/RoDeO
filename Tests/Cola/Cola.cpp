#include <iomanip>
#include <iostream>
#include <vector>
#include<stdio.h>
#include<math.h>

double Cola(double *x){
	
	double dis[46] = {
			1.27,
			1.69,1.43,
			2.04,2.35,2.43,
			3.09,3.18,3.26,2.85,
			3.20,3.22,3.27,2.88,1.55,
			2.86,2.56,2.58,2.59,3.12,3.06,
			3.17,3.18,3.18,3.12,1.31,1.64,3.00,
			3.21,3.18,3.18,3.17,1.70,1.36,2.95,1.32,
			2.38,2.31,2.42,1.94,2.85,2.81,2.56,2.91,2.97
	};


	double sum = 0.0, temp;
	int i, j, t, k = 1;
	double mt[20] = {0, 0, 0, 0};
	for( i = 4; i < 20; i++)
		mt[i] = x[i-3];
	for( i = 1; i < 10; i++)
		for( j = 0; j < i; j++) {
			temp = 0.0;
			for( t = 0; t < 2; t++ )
				temp += ( mt[i*2+t]-mt[j*2+t] )*( mt[i*2+t]-mt[j*2+t] );
			sum += ( dis[k-1] - sqrt(temp) )* ( dis[k-1] - sqrt(temp) );
			k++;
		}
	return sum;

}

int main(void){

double x[17];
FILE *inp = fopen("dv.dat","r");
for(int i = 0; i<17; i++){
  fscanf(inp,"%lf",&x[i]);
}

fclose(inp);

double result = Cola(x);
FILE *outp = fopen("objective.dat","w");
fprintf(outp,"%15.10f\n",result);
fclose(outp);


return 0;
}


