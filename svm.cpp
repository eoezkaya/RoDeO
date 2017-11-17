#include "svm.hpp"
#include "Rodeo_macros.hpp"
#include "auxilliary_functions.hpp"


#include <armadillo>


using namespace arma;

/* some kernels for svm */
double svm_linear_kernel(rowvec x1, rowvec x2){

    return dot(x1,x2);

}

double svm_polynomial_kernel(rowvec x1, rowvec x2, double c, double p){

    return pow( (dot(x1,x2)+c) ,p );

}

double svm_rbf_kernel(rowvec x1, rowvec x2, double sigma){

    rowvec xdiff = x1-x2;
    double norm_squared = pow( L2norm(xdiff, xdiff.size()), 2.0);

    return exp(-norm_squared/(2.0*sigma*sigma));

}


void train_svm(mat &X, vec &y,vec &alpha, double &b, SVM_param & model_parameters){

    unsigned int dim = y.size();
    double *K = new double[dim*dim];


    const double Cminlog=0;
    const double Cmaxlog= 14;
    const double cmax=0.0;
    const double cmin=0.0;
    const double sigmamax=10.0;
    const double sigmamin=0.01;

    urowvec numberofsv(model_parameters.max_inner_iter);
    numberofsv.fill(0);

    vec Cparam(model_parameters.max_inner_iter,fill::zeros);

    vec alpha_best(alpha.size(),fill::zeros);

    double b_best = 0.0;
    double C_best = 0.0;
    double sigma_best=0.0; /* for Gaussian kernel */
    double c_best = 0.0; /* for poly kernel */

    unsigned int min_number_of_sv = alpha.size();

    /* generate lhs designs for SVM training*/


    mat svm_lhs(model_parameters.max_inner_iter,2);
    svm_lhs.fill(-1.0);

    std::string str_problem_dim = std::to_string(2);
    std::string lhs_filename = "svm_lhs_points.dat";
    std::string python_command = "python -W ignore lhs.py "+ lhs_filename+ " "+ str_problem_dim + " "+ std::to_string(model_parameters.max_inner_iter)+ " center" ;

    system(python_command.c_str());


    svm_lhs.load("lhs_points.dat", raw_ascii);

#if 1
    printf("DoE for the LHS training...\n");
    svm_lhs.print();
#endif




#if 0
    printf("Start train_svm...\n");
#endif

    if(dim != X.n_rows){
        printf("Error: Size of the label vector does not match with the number of rows in data matrix\n");
        exit(-1);

    }

    for(unsigned int i=0; i<y.size(); i++){
        if( (y(i)!=1.0) && (y(i)!=-1.0)){ /* labels must be either 1 or -1 */
            printf("Label = %10.7f\n", y(i));
            printf("Error: some labels are not valid\n");
            exit(-1);


        }

    }

    for(int svm_inner_iter=0; svm_inner_iter<model_parameters.max_inner_iter; svm_inner_iter++ ){


        model_parameters.Csoftmargin = svm_lhs(svm_inner_iter,0)*(Cmaxlog-Cminlog)+Cminlog;

        model_parameters.Csoftmargin = pow(10.0,model_parameters.Csoftmargin);


        Cparam(svm_inner_iter) = model_parameters.Csoftmargin;

        if(model_parameters.kernel_type == SVM_RBF){

            model_parameters.sigmaGausskernel = svm_lhs(svm_inner_iter,1)*(sigmamax-sigmamin)+sigmamin;

        }

        if(model_parameters.kernel_type == SVM_POLYNOMIAL){

            model_parameters.cpolykernel = svm_lhs(svm_inner_iter,1)*(cmax-cmin)+cmin;

        }

#if 1
        printf("C = %15.10f\n",model_parameters.Csoftmargin);
#endif


#if 1
        printf("Allocating Kernel matrix (%d x %d matrix)\n",dim,dim);
#endif
        /* Kernel matrix */







        /* evaluate Kernel matrix */
        for(unsigned int i=0; i<dim; i++){

            for(unsigned int j=0; j<dim; j++){

                if(model_parameters.kernel_type == SVM_LINEAR){


                    K[i*dim+j] = svm_linear_kernel(X.row(i),X.row(j));

                }

                if(model_parameters.kernel_type == SVM_RBF){

                    K[i*dim+j]  = svm_rbf_kernel(X.row(i),X.row(j),model_parameters.sigmaGausskernel);
                }

                if(model_parameters.kernel_type == SVM_POLYNOMIAL){

                    K[i*dim+j]  = svm_polynomial_kernel(X.row(i),X.row(j),model_parameters.cpolykernel,model_parameters.ppolykernel);
                }


            }
        }



        /* save K, y and C for QP input */

        FILE *qp_input_file = fopen("qp_input.dat","w");

        for(unsigned int i=0; i<dim; i++){

            for(unsigned int j=0; j<dim; j++){

                double val = y(i)*y(j)*K[i*dim+j] ;
#if 0
                printf("y(%d) = %5.2f y(%d) = %5.2f K(%d,%d) = %10.7f val = %10.7f\n",i,y(i),j,y(j),i,j,K(i,j),val);
#endif

                fprintf(qp_input_file,"%15.10f\n",val);
            }

        }

        for(unsigned int i=0; i<dim; i++){

            fprintf(qp_input_file,"%15.10f\n",y(i));

        }

        fclose(qp_input_file);

        /* call the python script to solve qp problem */

        std::string python_command = "python solve_qp_for_svm.py "+
                std::to_string(model_parameters.Csoftmargin) +
                " "+std::to_string(dim) +
                " qp_input.dat";

        system(python_command.c_str());


        alpha.load("Lagrange_multipliers.txt", raw_ascii);

#if 0
        alpha.print();
#endif


        double largest_margin_alpha = -LARGE;
        for(unsigned int i=0; i<alpha.size(); i++){

            if(alpha(i) > 0  && fabs(alpha(i)- model_parameters.Csoftmargin) > 10E-6 ){

                if(alpha(i)>largest_margin_alpha){

                    largest_margin_alpha = alpha(i);
                }


            }

        }

#if 0
        printf("Largest margin Lagrange multipler = %10.7f\n",largest_margin_alpha);
#endif

        model_parameters.sv_tolerance = largest_margin_alpha/10E6;


        /* identify support vectors for a given tolerance*/

        for(unsigned int i=0; i<alpha.size(); i++){

            if(alpha(i) < model_parameters.sv_tolerance) {

                alpha(i) = 0.0;
            }

        }

#if 0
        printf("Lagrange multipliers = \n");
        alpha.print();
#endif

        /* compute intercept b */

        int sv_index=0;
        for(unsigned int i=0; i<alpha.size(); i++){

            if (alpha(i) > 0.0){

                b = y(i);
                sv_index = i;
                break;
            }
        }


        double sum=0.0;
        for(unsigned int i=0; i<alpha.size(); i++){

            sum+= y(i)*alpha(i)*K[i*dim+sv_index];

        }

        b=b-sum;

        printf("b = %10.7f\n",b);


        numberofsv(svm_inner_iter) = 0;
        for(unsigned int i=0; i<alpha.size(); i++){
            //        printf("alpha(i) = %d %10.7f\n",i,alpha(i));
            if(alpha(i) > 0 ){

                if( fabs(alpha(i)- model_parameters.Csoftmargin) < 10E-6){

                    printf("non-margin support vector %d:\n",i);
                    X.row(i).print();
                    numberofsv(svm_inner_iter) =  numberofsv(svm_inner_iter)+1;
                }
                else{

                    printf("margin support vector %d:\n",i);
                    X.row(i).print();
                    numberofsv(svm_inner_iter) =  numberofsv(svm_inner_iter)+1;

                }

            }

        }

        if(numberofsv(svm_inner_iter) < min_number_of_sv){

            min_number_of_sv = numberofsv(svm_inner_iter);
            alpha_best = alpha;
            b_best = b;
            C_best= model_parameters.Csoftmargin;
            sigma_best=model_parameters.sigmaGausskernel;
            c_best = model_parameters.cpolykernel;


        }


#if 1
        printf("Number of support vectors = %6d\n",int(numberofsv(svm_inner_iter)));

#endif


    } /* end of loop svm_inner_iter */



#if 1
    printf("Cparam            #sv\n");
    for(int svm_inner_iter=0; svm_inner_iter<model_parameters.max_inner_iter; svm_inner_iter++ ){

        printf("%15.10e        %6d\n", Cparam(svm_inner_iter),int(numberofsv(svm_inner_iter)));
    }

    alpha_best.print();
    printf("min_number_of_sv = %d\n",min_number_of_sv);


#endif

    alpha=alpha_best;
    b = b_best;
    model_parameters.Csoftmargin = C_best;
    model_parameters.sigmaGausskernel = sigma_best;
    model_parameters.cpolykernel = c_best;


    delete[] K;

}


void perform_svn_test(Classifier_Function_param &test_function,
        int  number_of_samples,
        int number_of_outliers,
        SVM_KERNEL_TYPE kernel_type)
{

#if 1
    test_function.print();
#endif


    /* allocate data matrix */
    mat data(number_of_samples,test_function.number_of_independents+1);
    data.fill(0.0);

    /* file name for data points in csv (comma separated values) format */
    std::string input_file_name = test_function.function_name+"_"
            + std::to_string(number_of_samples )
    +".csv";

    printf("input file name : %s\n",input_file_name.c_str());

    /* generate random sample points and identify labels */
    for(int i=0; i<number_of_samples; i++){

        for(unsigned int j=0; j<test_function.number_of_independents; j++){

            double lowerbound = test_function.bounds[j*2];
            double upperbound  = test_function.bounds[j*2+1];
#if 0
            printf("lower bound = %10.7f\n",lowerbound);
            printf("upper bound = %10.7f\n",upperbound);
#endif
            double random_number = RandomDouble(lowerbound , upperbound );
#if 0
            printf("random number = %10.7f\n",random_number);
#endif
            data(i,j) = random_number;


        }

        rowvec x = data.row(i);
        double label=0.0;
        double func_val=0.0;
        test_function.func_ptr(x.memptr(), &label, &func_val);

        /* add some noise to classification data */

        double random_number = RandomDouble(0.0 , 1.0 );

        if(random_number >=0.5){

            func_val+=test_function.noise_level;
        }
        else{

            func_val-=test_function.noise_level;
        }
        if (func_val > x(1)) {

            label=1.0;
        }
        else{

            label=-1.0;
        }

        /* generate outliers in the data*/
        random_number = RandomDouble(0.0 , 1.0 );
        if(random_number < 0.0) {

            label = -label;
        }

        data(i,test_function.number_of_independents) = label;

    }

#if 0
    data.print();
#endif


    data.save(input_file_name.c_str(), csv_ascii ) ;







    mat X = data.submat( 0, 0, number_of_samples-1, test_function.number_of_independents-1 );
    vec y = data.col(test_function.number_of_independents);

#if 1
    X.print();
    y.print();
#endif


    vec x_max(test_function.number_of_independents);
    x_max.fill(0.0);

    vec x_min(test_function.number_of_independents);
    x_min.fill(0.0);

    for (unsigned int i = 0; i < test_function.number_of_independents; i++) {
        x_max(i) = data.col(i).max();
        x_min(i) = data.col(i).min();

    }

#if 0
    printf("maximum = \n");
    x_max.print();

    printf("minimum = \n");
    x_min.print();

#endif

    /* normalize data */
    for (int i = 0; i < number_of_samples; i++) {

        for (unsigned int j = 0; j < test_function.number_of_independents; j++) {
            X(i, j) = (X(i, j) - x_min(j)) / (x_max(j) - x_min(j));
        }
    }

#if 0
    X.print();
#endif


    SVM_param model_parameters;
    model_parameters.kernel_type = kernel_type;
    model_parameters.ppolykernel = 10.0;
    model_parameters.cpolykernel = 0.0;
    model_parameters.max_inner_iter = 100;

    double b=0.0; /* intercept */

    vec alpha(number_of_samples);  /* vector of Lagrange multipliers */
    alpha.fill(-1.0); /* initialization with inadmissable values */

    train_svm(X, y, alpha, b, model_parameters);

#if 1

    model_parameters.print();

    printf("b = %10.7f\n",b);
    printf("alpha = \n");
    alpha.print();

    for(unsigned int i=0; i<alpha.size(); i++){

        if(alpha(i) > 0 ){

            if( fabs(alpha(i)- model_parameters.Csoftmargin) < 10E-5){

                printf("non-margin support vector %d:\n",i);
                X.row(i).print();

            }
            else{

                printf("margin support vector %d:\n",i);
                X.row(i).print();


            }

        }

    }
#endif

    if(model_parameters.kernel_type == SVM_LINEAR){

        /* weight vector for the linear kernel */
        vec w(test_function.number_of_independents);
        w.fill(0.0);


        for(unsigned int i=0; i<test_function.number_of_independents; i++){

            for(int j=0; j<number_of_samples; j++){

                /* if it is a support vector */
                if(alpha(j) > 0){

                    w(i)+= alpha(j)*y(j)*X(j,i);
                }

            }

        }


#if 0
        printf("w = \n");
        printf("%10.7fx +%10.7fy + %10.7f = 0\n",w(0),w(1),b);
#endif

        /* visualization only for the 2d data */
        if(test_function.number_of_independents == 2){

            int resolution = 10000;

            std::string filename_for_curve= test_function.function_name;
            filename_for_curve += "_curve"+std::to_string(resolution)+".dat";


            std::string file_name_for_plot = test_function.function_name;
            file_name_for_plot += "_"+std::to_string(resolution)+".png";


            std::string filename_for_svm= test_function.function_name;
            filename_for_svm += "_svm"+std::to_string(resolution)+".dat";

            printf("opening file %s for output...\n",filename_for_curve.c_str());
            FILE *test_function_data=fopen(filename_for_curve.c_str(),"w");
            FILE *svm_function_data=fopen(filename_for_svm.c_str(),"w");


            double dx; /* step size in x*/
            double x;



            dx = (test_function.bounds[1]-test_function.bounds[0])/(resolution-1);
            x= test_function.bounds[0];

            for(int i=0; i<resolution; i++){

                double func_val=0.0;
                double svm_val=0.0;
                double label=0.0;
                double xnorm = (x - x_min(0)) / (x_max(0) - x_min(0));

                svm_val = (-b - w(0)*xnorm)/w(1);

                svm_val = svm_val*(x_max(1) - x_min(1))+x_min(1);

                test_function.func_ptr(&x, &label, &func_val);

                fprintf(test_function_data,"%10.7f %10.7f\n",x,func_val);
                fprintf(svm_function_data,"%10.7f %10.7f\n",x,svm_val);
                x+= dx;

            }

            fclose(test_function_data);
            fclose(svm_function_data);

            /* call python script to visualize */

            std::string python_command = "python -W ignore plot_2d_classification.py "+
                    filename_for_curve+ " "+ input_file_name+ " "+file_name_for_plot + " " + filename_for_svm;
            FILE* in = popen(python_command.c_str(), "r");
            fprintf(in, "\n");



        } /* end of visualization part */


    }
    /* we use a different visualization method for kernels other than linear kernel
     *
     * DOES NOT WORK ALL THE TIME!!
     * */
    else{

        int resolution = 1000;
        /* visualization only for the 2d data */
        if(test_function.number_of_independents == 2){

            std::string filename_for_curve= test_function.function_name;
            filename_for_curve += "_curve"+std::to_string(resolution)+".dat";


            std::string file_name_for_plot = test_function.function_name;
            file_name_for_plot += "_"+std::to_string(resolution)+".png";


            std::string filename_for_svm= test_function.function_name;
            filename_for_svm += "_svm"+std::to_string(resolution)+".dat";

            printf("opening file %s for output...\n",filename_for_curve.c_str());
            FILE *test_function_data=fopen(filename_for_curve.c_str(),"w");
            FILE *svm_function_data=fopen(filename_for_svm.c_str(),"w");



            double dx,dy;
            double xcor,ycor;




            dx = (test_function.bounds[1]-test_function.bounds[0])/(resolution-1);
            dy = (test_function.bounds[3]-test_function.bounds[2])/(resolution-1);

            xcor = test_function.bounds[0];
            /* sweep in x coordinate */

            int svm_label=0;
            int svm_label0=0;


            for(int i=0; i<resolution; i++){

                svm_label=0;
                svm_label0=0;



                ycor=test_function.bounds[2];

#if 0

                printf("starting x sweep at x = %10.7f\n",xcor);
#endif

                for(int j=0; j<resolution; j++){ /* sweep in y-coordinate */

#if 0

                    printf("x = %10.7f y = %10.7f\n",xcor,ycor);
#endif

                    rowvec xinp(2);
                    rowvec xnorm(2);
                    xinp(0)=xcor;
                    xinp(1)=ycor;

                    /* normalized input vector */
                    xnorm(0) = (xinp(0) - x_min(0)) / (x_max(0) - x_min(0));
                    xnorm(1) = (xinp(1) - x_min(1)) / (x_max(1) - x_min(1));

                    /* iterate through all sv */
                    double svm_val=0;
                    for(unsigned int itersv=0; itersv<alpha.size(); itersv++){

                        if(alpha(itersv)>0){

#if 0
                            X.row(itersv).print();
                            printf("alpha(%d) = %10.7f\n",itersv,alpha(itersv));
                            printf("y(%d) = %10.7f\n",itersv,y(itersv));

                            model_parameters.print();

#endif

                            if(model_parameters.kernel_type == SVM_RBF){

                                double Kval=0.0;
                                Kval = svm_rbf_kernel(X.row(itersv), xnorm, model_parameters.sigmaGausskernel);

                                svm_val+= alpha(itersv)*y(itersv)*Kval;

                            }

                            if(model_parameters.kernel_type == SVM_POLYNOMIAL){

                                double Kval=0.0;
                                Kval = svm_polynomial_kernel(X.row(itersv), xnorm, model_parameters.cpolykernel,model_parameters.ppolykernel );
                                svm_val+= alpha(itersv)*y(itersv)*Kval;

                            }

                        }

                    }/* iterate through all sv */
#if 0
                    printf("svm_val = %10.7f\n",svm_val);
#endif


                    svm_val+=b;
#if 0
                    printf("svm_val = %10.7f\n",svm_val);
#endif

                    svm_label0 = svm_label;
                    if(svm_val>=0){

                        svm_label=1;
                    }
                    else{

                        svm_label=-1;
                    }
#if 0

                    printf("svm_label = %d\n",svm_label);
                    printf("svm_label0 = %d\n",svm_label0);

                    if(svm_label0 == -1) exit(1);

#endif


                    /* label change is found */
                    if( (svm_label*svm_label0) < 0){
#if 0
                        printf("svm_label change\n");
                        printf("svm_label = %d\n",svm_label);
                        printf("svm_label0 = %d\n",svm_label0);
                        printf("x = %10.7f y = %10.7f\n",xcor,ycor);
#endif
                        double func_val=0.0;
                        double label=0.0;
                        test_function.func_ptr(&xcor, &label, &func_val);
                        fprintf(test_function_data,"%10.7f %10.7f\n",xcor,func_val);
                        fprintf(svm_function_data,"%10.7f %10.7f\n",xcor,ycor);
                        break;
                    }


                    ycor+=dy;

                }/* sweep in y-coordinate */

                xcor+=dx;
            } /* sweep in x-coordinate */


            fclose(test_function_data);
            fclose(svm_function_data);

            /* call python script to visualize */

            std::string python_command = "python -W ignore plot_2d_classification.py "+
                    filename_for_curve+ " "+ input_file_name+ " "+file_name_for_plot + " " + filename_for_svm;
            FILE* in = popen(python_command.c_str(), "r");
            fprintf(in, "\n");
        }


    } /* end of else */


    /* make an out of sample test */

    int out_of_samples = 10000;
    int count_mismatch=0;
    for(int iter_out_of_sample=0; iter_out_of_sample<out_of_samples; iter_out_of_sample++){

        rowvec x(test_function.number_of_independents);
        rowvec xnorm(test_function.number_of_independents);


        for(unsigned int j=0; j<test_function.number_of_independents; j++){

            double lowerbound = test_function.bounds[j*2];
            double upperbound  = test_function.bounds[j*2+1];

            double random_number = RandomDouble(lowerbound , upperbound );

            x(j) = random_number;


        }

        /* normalize data */
        for (int i = 0; i < number_of_samples; i++) {

            for (unsigned int j = 0; j < test_function.number_of_independents; j++) {
                xnorm(j) = (x(j) - x_min(j)) / (x_max(j) - x_min(j));
            }
        }


        double label=0.0;
        double func_val=0.0;
        test_function.func_ptr(x.memptr(), &label, &func_val);

        /* iterate through all sv */
        double svm_val=0;

        for(unsigned int itersv=0; itersv<alpha.size(); itersv++){

            if(alpha(itersv)>0){


                if(model_parameters.kernel_type == SVM_RBF){

                    double Kval=0.0;
                    Kval = svm_rbf_kernel(X.row(itersv), xnorm, model_parameters.sigmaGausskernel);

                    svm_val+= alpha(itersv)*y(itersv)*Kval;

                }

                if(model_parameters.kernel_type == SVM_POLYNOMIAL){

                    double Kval=0.0;
                    Kval = svm_polynomial_kernel(X.row(itersv), xnorm, model_parameters.cpolykernel,model_parameters.ppolykernel );
                    svm_val+= alpha(itersv)*y(itersv)*Kval;

                }

            }

        }/* iterate through all sv */

        svm_val+=b;

        int svm_label=0;
        if(svm_val>=0){

            svm_label=1;
        }
        else{

            svm_label=-1;
        }


        if((svm_label*int(label)) <0 ){
            count_mismatch++;
#if 1
            printf("x = \n");
            x.print();
            printf("svm_label = %d\n",svm_label);
            printf("label     = %d\n",int(label));
#endif
        }




    }

    printf("Number of mismatches = %d\n",count_mismatch);
    printf("Ratio = %5.3f\n",double(count_mismatch)/out_of_samples);



}







