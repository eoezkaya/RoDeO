#include "auxilliary_functions.hpp"
#include "Rodeo_macros.hpp"
#include "bitmap_image.hpp"
#include "kmeans_clustering.hpp"
#include "svm.hpp"

#include <armadillo>

using namespace arma;
using namespace bitmap;


#define number_of_training_samples 60000
#define number_of_validation_samples 10000
#define number_of_pixels  28

template<class T>
void drawDigit(T digit, std::string filename){

    bitmap_image image(number_of_pixels,number_of_pixels);


    for(int i=0; i<number_of_pixels; i++){

        for(int j=0; j<number_of_pixels; j++){

            double col =255-digit(i+number_of_pixels*j);
            image.set_pixel(i,j,col,col,col);

        }

    }

    image.save_image(filename);

}




void read_mnist_validation_data(mat &validation_data, vec &validation_labels){

    FILE *inp = fopen("t10k-labels.idx1-ubyte","rb");

    if(inp == NULL){

        printf("ERROR: File t10k-labels.idx1-ubyte cannot be opended\n");
        exit(-1);
    }

    int32_t magic_number=0;
    int32_t number_of_items=0;

    fread(&magic_number,sizeof(number_of_items),1,inp);
    fread(&number_of_items,sizeof(number_of_items),1,inp);

    /*
printf("magic_number = %d\n",magic_number);
printf("Validation set has %d data points\n",number_of_items);
printf("The size of integer is %u bytes\n", sizeof(number_of_items)*8);

     */

    unsigned char ch;

    for(int i=0; i<number_of_validation_samples; i++){
        fread(&ch,1,1,inp);

        validation_labels(i)=ch;

    }



    fclose(inp);



    inp = fopen("t10k-images.idx3-ubyte","rb");

    fread(&magic_number,sizeof(number_of_items),1,inp);
    fread(&number_of_items,sizeof(number_of_items),1,inp);

    int32_t nrows=0;
    int32_t ncols=0;

    fread(&nrows,sizeof(nrows),1,inp);
    fread(&nrows,sizeof(nrows),1,inp);


    for(int i=0; i<number_of_validation_samples; i++){

        for(int j=0; j<number_of_pixels*number_of_pixels; j++){
            fread(&ch,1,1,inp);
            validation_data(i, j)=ch;
        }



    }


    fclose(inp);


}

void read_mnist_training_data(mat &training_data, vec &training_labels){

    FILE *inp = fopen("train-labels.idx1-ubyte","rb");

    if(inp == NULL){

        printf("ERROR: train-labels.idx1-ubyte cannot be opended\n");
        exit(-1);
    }

    int32_t magic_number=0;
    int32_t number_of_items=0;

    fread(&magic_number,sizeof(number_of_items),1,inp);
    fread(&number_of_items,sizeof(number_of_items),1,inp);

    /*
printf("magic_number = %d\n",magic_number);
printf("Validation set has %d data points\n",number_of_items);
printf("The size of integer is %u bytes\n", sizeof(number_of_items)*8);

     */

    unsigned char ch;

    for(int i=0; i<number_of_training_samples; i++){
        fread(&ch,1,1,inp);



        training_labels(i)=ch;

    }



    fclose(inp);



    inp = fopen("train-images.idx3-ubyte","rb");

    if(inp == NULL){

        printf("ERROR: train-images.idx3-ubyte cannot be opended\n");
        exit(-1);
    }

    fread(&magic_number,sizeof(number_of_items),1,inp);
    fread(&number_of_items,sizeof(number_of_items),1,inp);

    int32_t nrows=0;
    int32_t ncols=0;

    fread(&nrows,sizeof(nrows),1,inp);
    fread(&nrows,sizeof(nrows),1,inp);


    for(int i=0; i<number_of_training_samples; i++){
        for(int j=0; j<number_of_pixels*number_of_pixels; j++){
            fread(&ch,1,1,inp);
            training_data(i, j)=ch;
        }



    }


    fclose(inp);


}



void mnist(void){

    int dim = number_of_pixels*number_of_pixels;

    mat validation_data(number_of_validation_samples,number_of_pixels*number_of_pixels);
    mat training_data(number_of_training_samples,number_of_pixels*number_of_pixels);
    vec validation_labels(number_of_validation_samples);
    vec training_labels(number_of_training_samples);

    read_mnist_validation_data(validation_data, validation_labels);
    read_mnist_training_data(training_data, training_labels);







    mat Z = training_data;

    vec meanVal(dim);

    for(int i=0; i<dim; i++){

        meanVal(i) = mean(Z.col(i));


    }




#if 1
    printf("mean values of each row =\n");
    meanVal.print();
#endif


    for(int i=0; i<number_of_training_samples; i++){

        for(int j=0; j<dim; j++){

            Z(i,j)=Z(i,j) - meanVal(j);

        }

    }

    mat sigma = cov(Z);



    mat U;
    vec s;
    mat V;

    svd(U,s,V,sigma);

#if 1
    printf("s =\n");
    s.print();

//    vec U0 =255*U.col(0);
//    drawDigit(U0,"1_mode.bmp");
//    vec U1 =255*U.col(1);
//    drawDigit(U1,"2_mode.bmp");
//    vec U2 =255*U.col(2);
//    drawDigit(U2,"3_mode.bmp");
//    vec U3 =255*U.col(3);
//    drawDigit(U3,"4_mode.bmp");
//    vec U4 =255*U.col(4);
//    drawDigit(U4,"5_mode.bmp");

#endif


    int r=500;

    mat reducedU(dim,r);
    reducedU.fill(0.0);

    for(int i=0; i<dim; i++){

        for(int j=0; j<r; j++){

            reducedU(i,j) = U(i,j);

        }


    }



    mat reducedData= training_data*reducedU;



#if 0
    reducedData.print();
#endif




#if 1
    mat origData = reducedData*trans(reducedU);
    mat diff = origData- training_data;

//    for(int i=0; i<diff.n_rows; i++){
//
//        for(int j=0; j<diff.n_cols; j++){
//
//            if (diff(i,j) > 10E-10){
//
//                printf("ERROR: There is a difference between!\n");
//                exit(-1);
//
//            }
//
//        }
//
//
//    }

    drawDigit(origData.row(0),"reduced.bmp");
    drawDigit(training_data.row(0),"original.bmp");

#endif






}
