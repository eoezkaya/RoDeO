#include "auxilliary_functions.hpp"
#include <chrono>
#include <random>
#include <string>
#include <math.h>
#include <vector>


void generateKRandomInt(uvec &numbers, unsigned int N, unsigned int k){

	unsigned int numbersGenerated = 0;

	numbers.fill(0);


	while (numbersGenerated != k){


		int r = rand()%N;
#if 0
		printf("random number = %d\n",r);
#endif
		if (is_in_the_list(r, numbers) == -1 ){

			numbers(numbersGenerated) = r;
			numbersGenerated++;
#if 0
			printf("numbers =\n");
			numbers.print();
#endif
		}



	}




}


void normalize_vector(rowvec &xp, rowvec &xpnorm, vec xmin, vec xmax){

	int dim = xp.size();

	for(int i=0; i<dim; i++){

		xpnorm(i) = (1.0/dim)*(xp(i) - xmin(i)) / (xmax(i) - xmin(i));

	}



}

void normalize_vector_back(rowvec &xp, rowvec &xpnorm, vec xmin, vec xmax){

	int dim = xp.size();

	for(int i=0; i<dim; i++){

		xp(i) = xpnorm(i)*dim * (xmax(i) - xmin(i)) + xmin(i);


	}

}



void perturbVectorUniform(frowvec &xp,float sigmaPert){


	int size = xp.size();

	for(int i=0; i<size; i++){

		float eps = sigmaPert* RandomFloat(-1.0, 1.0);

		xp(i) += eps;


	}







}

void normalizeDataMatrix(mat matrixIn, mat &matrixOut){

	int dim = matrixIn.n_cols;

	vec xmin(dim);
	vec xmax(dim);

	for(int i=0; i<dim;i++){

		xmin(i) = min(matrixIn.col(i));
		xmax(i) = max(matrixIn.col(i));
	}

#if 0
	printf("xmin = \n");
	xmin.print();
	printf("xmax = \n");
	xmax.print();
#endif

	for(int j=0; j<dim;j++){

		double delta = xmax(j)-xmin(j);

		if( delta != 0 ){

			for(unsigned int i=0; i<matrixIn.n_rows;i++){

				matrixOut(i,j)  = ((1.0/dim)*(matrixIn(i,j)-xmin(j)))/(delta);
			}

		}

	}




}

int check_if_lists_are_equal(int *list1, int *list2, int dim){

	int flag=1;
	for(int i=0; i<dim; i++){

		int item_to_check = list1[i];

		if ( is_in_the_list(item_to_check, list2, dim) == -1){

			flag = 0;
			return flag;


		}
	}

	return flag;
}







/** Returns the pdf of x, given the distribution described by mu and sigma..
 *
 * @param[in] x
 * @param[in] mu
 * @param[in] sigma
 * @return normal_pdf(x) with mu and sigma
 *
 */
double pdf(double x, double mu, double sigma)
{
	/* Constants */
	static const double pi = 3.14159265359;
	return exp( (-1.0 * (x - mu) * (x - mu)) / (2 * sigma * sigma)) / (sigma * sqrt(2 * pi));
}

/** Returns the cdf of x, given the distribution described by mu and sigma..
 *
 * @param[in] x
 * @param[in] mu
 * @param[in] sigma
 * @return normal_cdf(x) with mu and sigma
 *
 */
double cdf(double x, double mu, double sigma)
{
	return 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2.0))));
}


/** checks whether an entry is in a list or not.
 *
 * @param[in] entry
 * @param[in] list
 * @param[in] list size
 * @return -1 if no, position in the list if yes
 *
 */
int is_in_the_list(int entry, int *list, int list_size){

	int flag=-1;

	for(int i=0;i< list_size;i++) {

		if(list[i]==entry) flag=i;
	}

	return flag;
}

/** checks whether an entry is in a list or not.
 *
 * @param[in] entry
 * @param[in] list (as a std::vector)
 * @param[in] list size
 * @return -1 if no, position in the list if yes
 *
 */
int is_in_the_list(int entry, std::vector<int> &list){

	int flag=-1;

	for (std::vector<int>::iterator it = list.begin() ; it != list.end(); ++it){

		if(*it == entry) {

			flag = *it;
		}

	}

	return flag;
}

/** checks whether an entry is in a list or not.
 *
 * @param[in] entry
 * @param[in] list (as a arma::vec)
 * @param[in] list size
 * @return -1 if no, position in the list if yes
 *
 */
int is_in_the_list(unsigned int entry, uvec &list){

	int flag=-1;

	for (unsigned int i = 0 ; i< list.size(); i++){

		if(list(i) == entry) {

			flag = i;
		}

	}

	return flag;
}



/** solve a linear system Ax=b, with a given Cholesky decomposition of A: U and L
 *
 * @param[in] U
 * @param[in] L
 * @param[out] x
 * @param[in] b
 *
 */
void solve_linear_system_by_Cholesky(mat &U, mat &L, vec &x, vec &b){

	int dim = x.size();

	/* initialize x */

	x.fill(0.0);

	vec y(dim);

	/* forward subst. L y = b */

	y.fill(0.0);

	for (int i = 0; i < dim; i++) {

		double residual = 0.0;
		for (int j = 0; j < i; j++) {

			residual = residual + L(i, j) * y(j);

		}

		y(i) = (b(i) - residual) / L(i, i);
	}

	/* back subst. U x = y */

	for (int i = dim - 1; i >= 0; i--) {

		double residual = 0.0;
		for (int j = dim - 1; j > i; j--){

			residual += U(i, j) * x(j);
		}


		x(i) = (y(i) - residual) / U(i, i);
	}

}

/** generate a random number between a and b
 *
 * @param[in] a
 * @param[in] b
 * @return random number between a and b
 *
 */
double RandomDouble(double a, double b) {

	double random = ((double) rand()) / (double) RAND_MAX;
	double diff = b - a;
	double r = random * diff;
	return a + r;
}

/** generate a random number between a and b
 *
 * @param[in] a
 * @param[in] b
 * @return random number between a and b
 *
 */
float RandomFloat(float a, float b) {

	float random = ((float) rand()) / (float) RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}


/** generate a random number between a and b
 *
 * @param[in] a
 * @param[in] b
 * @return random number in the interval [a,b]
 *
 */
int RandomInt(int a, int b) {

	b++;
	int diff = b-a;
	int random = rand() % diff;
	return a + random;
}

void RandomVector(rowvec &x){

	for(unsigned int i=0; i<x.size(); i++) {

		x(i) = RandomDouble(0.0, 1.0);
	}


}


/** generate a random number between xs and xe using the normal distribution
 *
 * @param[in] a
 * @param[in] b
 * @return random number between a and b
 *
 */
double random_number(double xs, double xe, double sigma_factor){

	double sigma=fabs((xe-xs))/sigma_factor;
	double mu=(xe+xs)/2.0;

	if (sigma == 0.0) sigma=1.0;

	/* construct a trivial random generator engine from a time-based seed */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);
	std::normal_distribution<double> distribution (mu,sigma);
	return distribution(generator);
}


/** randomly generates the indices of a validation set
 *
 * @param[in] size   dimension of the validation set
 * @param[in] N      dimension of the data set
 * @param[out] indices
 *
 */
void generate_validation_set(int *indices, int size, int N){

	int number_of_indices_generated=0;
	int random_int;
	int flag;

#if 1
	printf("size of the validation set = %d\n",size);
	printf("size of the data set = %d\n",N);
#endif

	if(size <0 || size>N){

		printf("Error: Size of the validation set is wrong");
		exit(-1);
	}

	for(int i=0; i<size;i++){

		indices[i]=-1;
	}

	/* initialize random seed: */
	srand (time(NULL));

	while(number_of_indices_generated < size){

		while(1){

			/* generate a random index */
			random_int = rand() % N;
#if 1
			printf("random_int = %d\n",random_int);
#endif
			/* check if it is already in the list */
			flag = is_in_the_list(random_int, indices, size);
#if 1
			printf("flag = %d\n",flag);
#endif

			if(flag == -1){

				indices[number_of_indices_generated]= random_int;
				number_of_indices_generated++;
				break;

			}


		}

	}


}

/** randomly generates the indices of a validation set
 *
 * @param[in]  N:  dimension of the data set
 * @param[out] indices (arma::uvec)
 *
 */
void generate_validation_set(uvec &indices, int N){

	int size = indices.size();
	int number_of_indices_generated=0;
	int random_int;
	int flag;

#if 0
	printf("size of the validation set = %d\n",size);
	printf("size of the data set = %d\n",N);
#endif

	if(size <=0 || size>N){

		printf("Error: Size of the validation set is wrong");
		exit(-1);
	}

	indices.fill(-1);


	while(number_of_indices_generated < size){

		while(1){

			/* generate a random index */
			random_int = rand() % N;
			/* check if it is already in the list */
			flag= is_in_the_list(random_int, indices);

			if(flag == -1){

				indices[number_of_indices_generated]= random_int;
				number_of_indices_generated++;
				break;

			}


		}

	}

	indices = sort(indices);


}


/** generates a modified data set by removing validation points
 *
 * @param[in] X
 * @param[in] y
 * @param[in] indices
 * @param[out] Xmod
 * @param[out] ymod
 * @param[out] map
 */
void remove_validation_points_from_data(mat &X, vec &y, uvec & indices, mat &Xmod, vec &ymod, uvec &map){


	int added_rows=0;
	for(unsigned int j=0; j<X.n_rows; j++){ /* for each row in the data matrix */

		/* if j is not a validation point */
		if ( is_in_the_list(int(j), indices) == -1){

#if 0
			printf("%dth point is not a validation point\n",j);
#endif
			Xmod.row(added_rows)=X.row(j);
			ymod(added_rows)    =y(j);
			map(added_rows) = j;
			added_rows++;


		}


	}
#if 0
	printf("data set map\n");
	map.print();
#endif




}

/** generates a modified data set by removing validation points
 *
 * @param[in] X
 * @param[in] y
 * @param[in] indices
 * @param[out] Xmod
 * @param[out] ymod
 */

void remove_validation_points_from_data(mat &X, vec &y, uvec & indices, mat &Xmod, vec &ymod){


	int added_rows=0;
	for(unsigned int j=0; j<X.n_rows; j++){ /* for each row in the data matrix */

		/* if j is not a validation point */
		if ( is_in_the_list(int(j), indices) == -1){

#if 0
			printf("%dth point is not a validation point\n",j);
#endif
			Xmod.row(added_rows)=X.row(j);
			ymod(added_rows)    =y(j);
			added_rows++;

		}


	}

	exit(1);

}






bool file_exist(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}


/** brute force KNeighbours search
 *
 * @param[in] data
 * @param[in] p
 * @param[in] K
 * @param[out] min_dist
 * @param[out] indices
 */

void findKNeighbours(mat &data, rowvec &p, int K, double* min_dist,int *indices, unsigned int norm){

#if 0
	printf("findKNeighbours(mat &data, rowvec &p, int K, double* min_dist,int *indices, int norm)...\n");
#endif

	int number_of_points= data.n_rows;
	int dim= data.n_cols;



	for(int i=0; i<K; i++){

		min_dist[i]= LARGE;
		indices[i]= -1;
	}


	for(int i=0; i<number_of_points; i++){ /* for each data point */

		rowvec x = data.row(i);
		rowvec xdiff = x-p;

		double distance = 0.0;

		if(norm == xdiff.size()){

			distance = Lpnorm(xdiff, dim, xdiff.size());
		}
		if(norm == 2){

			distance = L2norm(xdiff, dim);

		}
		if(norm == 1){

			distance = L1norm(xdiff, dim);

		}
#if 0
		printf("distance = %10.7f\n", distance);
#endif
		double worst_distance = -LARGE;
		int worst_distance_index = -1;


		find_max_with_index(min_dist, K, &worst_distance, &worst_distance_index);

		/* a better point is found */
		if(distance < worst_distance){

			min_dist[worst_distance_index]= distance;
			indices[worst_distance_index] = i;

		}

	}

	/* sort the indices */

	for(int i=0; i<K; i++){

		for(int j=i+1; j<K; j++){

			if(min_dist[i] > min_dist[j]){

				double temp;
				int tempindx;
				temp = min_dist[j];
				tempindx = indices[j];

				min_dist[j] =  min_dist[i];
				indices[j] = indices[i];

				min_dist[i] =  temp;
				indices[i] = tempindx;

			}


		}

	}



}




double calcMetric(rowvec &xi,rowvec &xj, mat M){
#if 0
	printf("calling calcMetric...\n");
	printf("M = \n");
	M.print();
#endif
	rowvec diff= xi-xj;

#if 0
	printf("diff = \n");
	diff.print();
#endif

	colvec diffT= trans(diff);

	return dot(diff,M*diffT);

}

float calcMetric(frowvec &xi,frowvec &xj, fmat M){
#if 0
	printf("calling calcMetric...\n");
	printf("M = \n");
	M.print();
#endif
	frowvec diff= xi-xj;

#if 0
	printf("diff = \n");
	diff.print();
#endif

	fcolvec diffT= trans(diff);

	return dot(diff,M*diffT);

}



/** brute force KNeighbours search
 *
 * @param[in] data
 * @param[in] p
 * @param[in] K
 * @param[out] min_dist
 * @param[out] indices
 * @param[in] M : metric matrix
 */

void findKNeighbours(mat &data,
		rowvec &p,
		int K,
		vec &min_dist,
		uvec &indices,
		mat M){

#if 0
	printf("calling findKNeighbours...\n");
	printf("point = \n");
	p.print();
#endif

	int number_of_points= data.n_rows;

	min_dist.fill(LARGE);
	indices.fill(-1);



	for(int i=0; i<number_of_points; i++){ /* for each data point */

		rowvec x = data.row(i);
		rowvec xdiff = x-p;

#if 0
		printf("xdiff = \n");
		xdiff.print();
#endif


		double distance =  calcMetric(x,p, M);

#if 0
		printf("distance = %10.7f\n", distance);
#endif

		double worst_distance = -LARGE;
		int worst_distance_index = -1;


		find_max_with_index(min_dist, K, &worst_distance, &worst_distance_index);


		/* a better point is found */
		if(distance < worst_distance){

			min_dist[worst_distance_index]= distance;
			indices[worst_distance_index] = i;

		}

	}

#if 0
	printf("%d nearest neighbors...\n",K);
	for(int i=0;i<K;i++){

		printf("index = %d\n",indices[i]);
		data.row(indices[i]).print();
	}
#endif



}




/** brute force KNeighbours search with given index array for distance computation
 *
 * @param[in] data
 * @param[in] p
 * @param[in] K
 * @param[in] input_indx
 * @param[out] min_dist
 * @param[out] indices
 * @param[in] number_of_independent_variables
 */

void findKNeighbours(mat &data,
		rowvec &p,
		int K,
		int *input_indx ,
		double* min_dist,
		int *indices,
		int number_of_independent_variables){

	int number_of_points= data.n_rows;

	for(int i=0; i<K; i++){

		min_dist[i]= LARGE;
		indices[i]= -1;
	}



	for(int i=0; i<number_of_points; i++){ /* for each data point */

		rowvec x = data.row(i);
		rowvec xdiff = x-p;

#if 0
		printf("xdiff = \n");
		xdiff.print();
#endif

		double distance = Lpnorm(xdiff, number_of_independent_variables, xdiff.size(), input_indx);
#if 0
		printf("distance = %10.7f\n", distance);
#endif

		double worst_distance = -LARGE;
		int worst_distance_index = -1;


		find_max_with_index(min_dist, K, &worst_distance, &worst_distance_index);


		/* a better point is found */
		if(distance < worst_distance){

			min_dist[worst_distance_index]= distance;
			indices[worst_distance_index] = i;

		}

	}

}





