/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2020 Chair for Scientific Computing (SciComp), TU Kaiserslautern
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, TU Kaiserslautern)
 *
 * This file is part of RoDeO
 *
 * RoDeO is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * RoDeO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU
 * General Public License along with CoDiPack.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, TU Kaiserslautern)
 *
 *
 *
 */


#ifndef READ_SETTINGS
#define READ_SETTINGS
#include<algorithm>
#include<iostream>
class Rodeo_settings {
public:
	unsigned int number_of_independents;

	std::string cwd;
	std::string python_dir;
	std::string config_file;

	std::string *keywords;
	const int number_of_keywords = 3;

	Rodeo_settings(){

		keywords = new std::string[number_of_keywords];
		keywords[0]="WORKING_DIRECTORY=";
		keywords[1]="PYTHON_DIRECTORY=";
		keywords[2]="CONFIGURATION_FILE=";

		cwd = "None";
		python_dir = "None";
		config_file = "None";

	}




	void print(void){
		printf("\nSettings:\n");
		printf("WORKING_DIRECTORY= %s\n", cwd.c_str());
		printf("PYTHON_DIRECTORY= %s\n", python_dir.c_str());
		printf("CONFIGURATION_FILE= %s\n", config_file.c_str());
		printf("\n\n");

	}

	void read(void){



#if 1
		printf("Reading settings...\n");
#endif
		size_t len = 0;
		ssize_t readlen;
		char * line = NULL;
		FILE *inp = fopen("settings.dat","r");

		if (inp == NULL){

			printf("Error: The file: settings.dat cannot be opened\n");
			exit(-1);
		}

		while ((readlen = getline(&line, &len, inp)) != -1) {


			if(line[0]!= '#'){
				std::string str(line);

#if 0
				printf("Retrieved line of length %zu :\n", readlen);
				printf("%s\n", str.c_str());
#endif

				for(int key=0; key<number_of_keywords; key++){
#if 0
					printf("searching the keyword: %s\n", keywords[key].c_str());
#endif
					std::size_t found = str.find(keywords[key]);

					if (found!=std::string::npos){
#if 0
						printf("found the keyword: %s\n", keywords[key].c_str());
						printf("found = %d\n",found);
#endif
						str.erase(std::remove_if(str.begin(), str.end(), isspace), str.end());
						std::string sub_str = str.substr(found+keywords[key].length());
#if 0
						printf("keyword:%s\n", sub_str.c_str());
#endif
						switch(key){
						case 0: {
							cwd = sub_str;
							break;
						}
						case 1: {
							python_dir = sub_str;
							break;
						}
						case 2: {
							config_file = sub_str;
							break;
						}


						}




					}



				}





			} /* end of if */

		} /* end of while */


		fclose(inp);

		print();

		if(config_file == "None"){

			std::cout<<"ERROR: No configuration file is specified in settings.dat!\n";
			abort();

		}

		if(cwd == "None"){

			std::cout<<"ERROR: No working directory is specified in settings.dat!\n";
			abort();

		}


	}
} ;



#endif
