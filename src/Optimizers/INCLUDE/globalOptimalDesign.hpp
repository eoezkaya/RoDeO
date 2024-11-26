#ifndef GLOBALOPTIMALDESIGN_HPP
#define GLOBALOPTIMALDESIGN_HPP

#include "../../Design/INCLUDE/design.hpp"
#include "../../Bounds/INCLUDE/bounds.hpp"


#ifdef UNIT_TESTS
#include<gtest/gtest.h>
#endif

namespace Rodop{

class GlobalOptimalDesign: public Design {

#ifdef UNIT_TESTS
	friend class GlobalDesignTest;
	FRIEND_TEST(GlobalDesignTest, ValidHistoryFile);
	FRIEND_TEST(GlobalDesignTest, NoFeasibleHistoryFile);
	FRIEND_TEST(GlobalDesignTest, generateXmlString);

#endif

private:

	void validateInputs(const mat& historyFile) const;
	unsigned int findBestDesignIndex(const mat& historyFile, bool& isFeasibleDesignFound) const;
	void extractDesignData(const vec& bestSample);

public:

	std::string xmlFileName = "globalOptimalDesign.xml";
	Bounds boxConstraints;



	void setGlobalOptimalDesignFromHistoryFile(const mat& historyFile);
	void setBoxConstraints(const Bounds& input);
	void setGradientGlobalOptimumFromTrainingData(const std::string &nameOfTrainingData);
	void saveToXMLFile(void) const;
	std::string generateXmlString(void) const;
	bool checkIfGlobalOptimaHasGradientVector(void) const;


	string generateXml(const std::string& elementName, const int& value) const;
	string generateXml(const std::string& elementName, const double& value) const;
	string generateXml(const std::string& elementName, const string& value) const;
	std::string generateXmlVector(const std::string& name, const vec& data) const;

};


} /* Namespace Rodop */

#endif
