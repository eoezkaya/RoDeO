#include "auxilliary_functions.hpp"
#include <chrono>
#include <random>
#include <string>
#include<math.h>
#include <vector>


/* Returns the probability of x, given the distribution described by mu and sigma. */
double pdf(double x, double mu, double sigma)
{
	//Constants
	static const double pi = 3.14159265359;
	return exp( (-1.0 * (x - mu) * (x - mu)) / (2 * sigma * sigma)) / (sigma * sqrt(2 * pi));
}

/* Returns the probability of [-inf,x] of a gaussian distribution */
double cdf(double x, double mu, double sigma)
{
	return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2.0))));
}


/* checks whether an entry is in a list or not returns the list index if yes returns -1 if no */

int is_in_the_list(int entry, int *list, int list_size){
	int flag=-1;
	int i;

	for(i=0;i< list_size;i++) if(list[i]==entry) flag=i;

	return flag;
}

int is_in_the_list(int entry, std::vector<int> &list){
	int flag=-1;

	for (std::vector<int>::iterator it = list.begin() ; it != list.end(); ++it){
		if(*it == entry) flag = *it;

	}

	return flag;
}




void solve_linear_system_by_Cholesky(mat &U, mat &L, vec &x, vec &b){

	int dim = x.size();

	x.fill(0.0);

	vec y(dim);



	/* forward subst. L y = b */

	y.fill(0.0);

	for (int i = 0; i < dim; i++) {

		double residual = 0.0;
		for (int j = 0; j < i; j++) {
			residual = residual + L(i, j) * y(j);
			//		printf("%d %d %15.12f %15.12f %15.12f %15.12f\n",i,j,L(i,j),y(j),L(i,j)*y(j), residual);
		}

		y(i) = (b(i) - residual) / L(i, i);
	}

	/* back subst. U x = y */



	for (int i = dim - 1; i >= 0; i--) {
		double residual = 0.0;
		for (int j = dim - 1; j > i; j--)
			residual += U(i, j) * x(j);

		x(i) = (y(i) - residual) / U(i, i);
	}





}

/* generate a random number between a and b */
double RandomDouble(double a, double b) {

	double random = ((double) rand()) / (double) RAND_MAX;
	double diff = b - a;
	double r = random * diff;
	return a + r;
}





/* generate a random number using the normal distribution between xs and xe*/

double random_number(double xs, double xe, double sigma_factor){

	double sigma=fabs((xe-xs))/sigma_factor;
	double mu=(xe+xs)/2.0;
	//std::cout<<xs<<" "<<xe<<std::endl;
	//std::cout<<"sigma= "<<sigma<<" mu= "<<mu<<std::endl;

	if (sigma == 0.0) sigma=1.0;

	// construct a trivial random generator engine from a time-based seed:
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);
	std::normal_distribution<double> distribution (mu,sigma);
	return distribution(generator);
}

void compute_max_min_distance_data(mat &x, double &max_distance, double &min_distance){


	min_distance = LARGE;
	max_distance = -LARGE;

	int min_index[2]={0,0};
	int max_index[2]={0,0};

	for(unsigned int i=0; i< x.n_rows; i++){
		for(unsigned int j=i+1; j< x.n_rows; j++){

			double dist = 0.0;

			for(unsigned int k=0; k< x.n_cols;k++) dist+= (x(i,k)-x(j,k))*(x(i,k)-x(j,k));
			dist  = sqrt(dist);

			if(dist > max_distance) {
				max_distance = dist;
				max_index[0]=i;
				max_index[1]=j;
			}
			if(dist < min_distance) {
				min_distance = dist;
				min_index[0]=i;
				min_index[1]=j;


			}

		}



	}

	printf("maximum distance = %10.7f\n",max_distance);
	printf("between point %d = \n", max_index[0]);
	x.row(max_index[0]).print();
	printf("and point %d = \n", max_index[1]);
	x.row(max_index[1]).print();	


	printf("minimum distance = %10.7f\n",min_distance);
	printf("between point %d = \n", min_index[0]);
	x.row(min_index[0]).print();
	printf("and point %d = \n", min_index[1]);
	x.row(min_index[1]).print();

	printf("the ratio is = %10.7f\n",max_distance/min_distance );



}





bool file_exist(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}
