#include "auxilliary_functions.hpp"
#include <chrono>
#include <random>
#include <string>
#include <math.h>
#include <vector>



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
int is_in_the_list(int entry, uvec &list){

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
int RandomInt(int a, int b) {

    int diff = b-a;
    int random = rand() % diff;
    return a + random;
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
    while(number_of_indices_generated < N){

        while(1){

            /* generate a random index */
            random_int = rand() % size;
            /* check if it is already in the list */
            flag= is_in_the_list(random_int, indices, N);

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
 * @param[in] size   dimension of the validation set
 * @param[out] indices (arma::uvec)
 *
 */
void generate_validation_set(uvec &indices, int size){

    int N= indices.size();
    int number_of_indices_generated=0;
    int random_int;
    int flag;
    while(number_of_indices_generated < N){

        while(1){

            /* generate a random index */
            random_int = rand() % size;
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


}

/* distance functions */

double L1norm(vec & x){

    double sum=0.0;
    for(int i=0;i<x.size();i++){

        sum+=fabs(x(i));
    }

    return sum;
}

double L1norm(rowvec & x){

    double sum=0.0;
    for(int i=0;i<x.size();i++){

        sum+=fabs(x(i));
    }

    return sum;
}

double L2norm(vec & x){

    double sum=0.0;
    for(int i=0;i<x.size();i++){

        sum+=x(i)*x(i);
    }

    return sqrt(sum);
}

double L2norm(rowvec & x){

    double sum=0.0;

    for(int i=0;i<x.size();i++){

        sum+=x(i)*x(i);
    }

    return sqrt(sum);
}

double Lpnorm(vec & x, int p){

    double sum=0.0;
    for(int i=0;i<x.size();i++){

        sum+=pow(fabs(x(i)),p);
    }

    return pow(sum,1.0/p);
}

double Lpnorm(rowvec & x, int p){

    double sum=0.0;
    for(int i=0;i<x.size();i++){

        sum+=pow(fabs(x(i)),p);
    }

    return pow(sum,1.0/p);
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
 * @param[out] indices
 */

void findKNeighbours(mat &data, rowvec &p, int K, double* min_dist,int *indices){

    int number_of_points= data.n_rows;
    int dim= data.n_cols;
    vec minimum_distances(K);



    minimum_distances.fill(LARGE);

    for(int i=0; i<number_of_points; i++){ /* for each data point */

        rowvec x = data.row(i);
        rowvec xdiff = x-p;

        double distance = Lpnorm(xdiff, dim);

        double worst_distance = -LARGE;
        int worst_distance_index = -1;


        for(unsigned int j=0; j<K; j++){

            if(minimum_distances(j) > worst_distance ){

                worst_distance =minimum_distances(j);
                worst_distance_index = j;
            }
        }

        /* a better point is found */
        if(distance < worst_distance){

            minimum_distances(worst_distance_index)= distance;
            indices[worst_distance_index] = i;

        }



    }

    *min_dist = min(minimum_distances);


}


