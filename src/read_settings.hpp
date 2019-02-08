#ifndef READ_SETTINGS
#define READ_SETTINGS
#include <algorithm>
class Rodeo_settings {
public:
	unsigned int number_of_independents;

	std::string cwd;
	std::string python_dir;
	double lambda;
	std::string *keywords;
	const int number_of_keywords = 3;

	Rodeo_settings(){

		keywords = new std::string[number_of_keywords];
		keywords[0]="cwd:";
		keywords[1]="lambda:";
		keywords[2]="python_dir:";

	}




	void print(void){
		printf("\nSettings:\n");
		printf("cwd: %s\n", cwd.c_str());
		printf("python_dir: %s\n", python_dir.c_str());
		printf("alpha: %10.7e\n\n", lambda);


	}

	void read(void){

#if 1
		printf("reading settings...\n");
#endif
		size_t len = 0;
		ssize_t readlen;
		char * line = NULL;
		FILE *inp = fopen("settings.dat","r");

		if (inp == NULL){

			printf("Error: settings.dat cannot be opened\n");
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
							lambda =  std::stod(sub_str);
							break;
						}
						case 2: {
							python_dir = sub_str;
							break;
						}


						}




					}



				}





			} /* end of if */

		} /* end of while */


		fclose(inp);

		print();
	}
} ;


#endif
