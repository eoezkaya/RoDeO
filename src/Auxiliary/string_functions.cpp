/*
 * RoDeO, a Robust Design Optimization Package
 *
 * Copyright (C) 2015-2023 Chair for Scientific Computing (SciComp), RPTU
 * Homepage: http://www.scicomp.uni-kl.de
 * Contact:  Prof. Nicolas R. Gauger (nicolas.gauger@scicomp.uni-kl.de) or Dr. Emre Özkaya (emre.oezkaya@scicomp.uni-kl.de)
 *
 * Lead developer: Emre Özkaya (SciComp, RPTU)
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
 * General Public License along with RoDeO.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors: Emre Özkaya, (SciComp, RPTU)
 *
 *
 *
 */

#include "./INCLUDE/auxiliary_functions.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <cassert>

using std::string;


bool isEqual(string s1, string s2){

	assert(!s1.empty());
	assert(!s2.empty());

	if(s1.length()!= s2.length()) return false;

	if(s1.compare(s2) == 0 ) return true;
	else return false;

}


bool isEmpty(std::string inputStr){

	if(inputStr.empty()){

		return true;
	}
	else{

		return false;
	}

}


bool isNotEmpty(std::string inputStr){

	if(inputStr.empty()){

		return false;
	}
	else{

		return true;
	}

}


bool checkIfOn(std::string keyword){

	if(isEmpty(keyword)) return false;

	bool flag = false;

	if(keyword == "YES") flag = true;
	if(keyword == "Yes") flag = true;
	if(keyword == "yes") flag = true;
	if(keyword == "ON") flag = true;
	if(keyword == "On") flag = true;
	if(keyword == "on") flag = true;
	if(keyword == "y") flag = true;
	if(keyword == "Y") flag = true;

	return flag;

}

bool checkIfOff(std::string keyword){

	if(isEmpty(keyword)) return true;
	bool flag = false;

	if(keyword == "NO") flag = true;
	if(keyword == "No") flag = true;
	if(keyword == "no") flag = true;
	if(keyword == "Off") flag = true;
	if(keyword == "OFF") flag = true;
	if(keyword == "off") flag = true;
	if(keyword == "N") flag = true;
	if(keyword == "n") flag = true;

	return flag;

}


std::string removeSpacesFromString(std::string inputString){

	inputString.erase(std::remove_if(inputString.begin(), inputString.end(), ::isspace), inputString.end());
	return inputString;
}



std::string removeKeywordFromString(std::string inputStr,  std::string keyword){

	assert(!keyword.empty());
	assert(!inputStr.empty());

	std::size_t found = inputStr.find(keyword);

	if(found != std::string::npos){

		inputStr.erase(std::remove_if(inputStr.begin(), inputStr.end(), ::isspace), inputStr.end());
		std::string sub_str = inputStr.substr(found+keyword.length() + 1);


		return sub_str;


	}
	else{

		return inputStr;
	}



}


std::vector<std::string> getStringValuesFromString(std::string str,char delimiter){

	str = removeSpacesFromString(str);

	std::vector<std::string> values;


	if(str[0] == '{' || str[0] == '['){

		str.erase(0,1);
	}

	if(str[str.length()-1] == '}' || str[str.length()-1] == ']'){

		str.erase(str.length()-1,1);
	}

	while(1){

		std::size_t found = str.find(delimiter);
		if (found==std::string::npos) break;

		std::string buffer;
		buffer.assign(str,0,found);
		str.erase(0,found+1);


		values.push_back(buffer);

	}

	values.push_back(str);

	return values;
}



vec getDoubleValuesFromString(std::string str,char delimiter){

	str = removeSpacesFromString(str);

	size_t n = std::count(str.begin(), str.end(), ',');

	vec values(n+1);


	if(str[0] == '{' || str[0] == '['){

		str.erase(0,1);
	}

	if(str[str.length()-1] == '}' || str[str.length()-1] == ']'){

		str.erase(str.length()-1,1);
	}

	int count = 0;
	while(1){

		std::size_t found = str.find(delimiter);
		if (found==std::string::npos) break;

		std::string buffer;
		buffer.assign(str,0,found);
		str.erase(0,found+1);


		values(count) = std::stod(buffer);
		count ++;
	}
	values(count) = std::stod(str);

	return values;
}





