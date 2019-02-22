#ifndef SU2_OPTIM_HPP
#define SU2_OPTIM_HPP


#include <armadillo>

#include <string>

using namespace arma;


class GEKSamplingData {
public:
	unsigned int size_of_dv;
	uvec variable_activity;
	unsigned int number_of_samples;
	vec lower_bound_dv;
	vec upper_bound_dv;
	std::string *output_file_names;
	std::string *config_file_names;
	std::string base_config_file_name;
	std::string history_file_name;
	unsigned int number_of_outputs;
	unsigned int number_outputs_with_gradients;
	std::string *output_names;
	uvec outputs_with_gradients;
	vec geometric_constraints;
	std::string *geometric_constraints_names;
	bool include_initial_design;
	unsigned int number_of_active_design_variables;
	unsigned int number_of_geometric_features;


	void print(void){

		printf("....... Initial Data Acquisition using %d samples .........\n",number_of_samples);

		for(unsigned int i=0; i<size_of_dv; i++){

			if (variable_activity(i) == 1){

				printf("Variable %d : ACTIVE, bounds = %10.7f %10.7f\n",i,lower_bound_dv(i),upper_bound_dv(i));
			}
			else if (variable_activity(i) == 0){

				printf("Variable %d : INACTIVE\n",i);
			}
			else {

				fprintf(stderr, "Error: Invalid activity flag! at %s, line %d.\n",__FILE__, __LINE__);
				exit(-1);
			}
		}

		printf("Outputs =\n");

		for(int i=0;i <number_of_outputs; i++){

			if(outputs_with_gradients(i) == 1){
				printf("Output #%d = %s (with gradient)\n",i,output_names[i].c_str());
			}
			else{

				printf("Output #%d = %s\n",i,output_names[i].c_str());
			}
		}

		printf("Geometric Constraints =\n");

		for(int i=0;i <geometric_constraints.size(); i++){


			printf("Geometric constraint #%d = %s (> %10.7f)\n",i,geometric_constraints_names[i].c_str(),geometric_constraints(i));
		}


		printf("Output file names =\n");

		for(int i=0;i <number_of_outputs+geometric_constraints.size(); i++){

			printf("Output file #%d = %s\n",i,output_file_names[i].c_str());
		}


		printf("Base config file name = %s\n",base_config_file_name.c_str());
		printf("History file name = %s\n",history_file_name.c_str());

		printf("Config file names =\n");


		for(int i=0;i <number_outputs_with_gradients; i++){

			printf("Output file #%d = %s\n",i,config_file_names[i].c_str());
		}



	}

};


class OptimizationData {
public:
	std::string name;
	unsigned int size_of_dv;
	uvec variable_activity;
	unsigned int max_number_of_samples;
	vec lower_bound_dv;
	vec upper_bound_dv;
	std::string *output_file_names;
	std::string *config_file_names;
	std::string base_config_file_name;
	std::string history_file_name;
	unsigned int number_of_outputs;
	unsigned int number_outputs_with_gradients;
	std::string *output_names;
	uvec outputs_with_gradients;
	vec geometric_constraints;
	std::string *geometric_constraints_names;
	bool include_initial_design;
	unsigned int number_of_active_design_variables;
	unsigned int number_of_geometric_features;


	void print(void){

		printf("....... %s optimization using max %d samples .........\n",name.c_str(),max_number_of_samples);

		for(unsigned int i=0; i<size_of_dv; i++){

			if (variable_activity(i) == 1){

				printf("Variable %d : ACTIVE, bounds = %10.7f %10.7f\n",i,lower_bound_dv(i),upper_bound_dv(i));
			}
			else if (variable_activity(i) == 0){

				printf("Variable %d : INACTIVE\n",i);
			}
			else {

				fprintf(stderr, "Error: Invalid activity flag! at %s, line %d.\n",__FILE__, __LINE__);
				exit(-1);
			}
		}

		printf("Outputs =\n");

		for(int i=0;i <number_of_outputs; i++){

			if(outputs_with_gradients(i) == 1){
				printf("Output #%d = %s (with gradient)\n",i,output_names[i].c_str());
			}
			else{

				printf("Output #%d = %s\n",i,output_names[i].c_str());
			}
		}

		printf("Geometric Constraints =\n");

		for(int i=0;i <geometric_constraints.size(); i++){


			printf("Geometric constraint #%d = %s (> %10.7f)\n",i,geometric_constraints_names[i].c_str(),geometric_constraints(i));
		}


		printf("Output file names =\n");

		for(int i=0;i <number_of_outputs+geometric_constraints.size(); i++){

			printf("Output file #%d = %s\n",i,output_file_names[i].c_str());
		}


		printf("Base config file name = %s\n",base_config_file_name.c_str());
		printf("History file name = %s\n",history_file_name.c_str());

		printf("Config file names =\n");


		for(int i=0;i <number_outputs_with_gradients; i++){

			printf("Output file #%d = %s\n",i,config_file_names[i].c_str());
		}



	}

};



typedef struct {
	rowvec dv;           /* design vector (normalized)     */
	vec dv_original;     /* design vector (not normalized) */
	double EI;
	double J;
	double CL;
	double CD;
	double Area;
} MC_design;

void plot_airfoil(std::string initial_airfoil_file, std::string deformed_mesh_file);
void SU2_CFD(std::string input_filename);

void check_double_points_data(std::string filename);

void su2_optimize(std::string python_dir);

int su2_robustoptimize_naca0012(OptimizationData &optimization_plan);


int call_SU2_CFD_Solver(vec &dv,
		double &CL,
		double &CD,
		double &area);


int call_SU2_Adjoint_Solver(
		int i,
		vec &dv,
		vec &gradient,
		vec &objectives,
		GEKSamplingData &sampling_plan);

int call_SU2_Adjoint_Solver(
		int i,
		vec &dv,
		vec &gradient,
		vec &objectives,
		OptimizationData &optimization_plan
);


void initial_data_acquisition(int number_of_initial_samples );

void initial_data_acquisitionGEK(GEKSamplingData &sampling_plan);

void su2_statistics_around_solution(int number_of_samples, std::string output_file_name,std::string input_file_name);

#endif
