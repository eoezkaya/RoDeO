#include <fstream>
#include<vector>


using namespace std;

namespace Rodop{

template <typename T>
std::string generateXml(const std::string& elementName, const T& value);

template <typename T>
std::string generateXmlVector(const std::string& name, const T& data);



template <typename T>
void writeXmlElement(std::ofstream& file, const std::string& elementName, const T& value);

template <typename T>
void writeXmlElementVector(std::ofstream& file, const std::string& elementName, const T& value);


double getDoubleValueFromXML(const std::string& xmlString, const std::string& keyword);
int getIntegerValueFromXML(const std::string& xmlString, const std::string& keyword);
string getStringValueFromXML(const std::string& xmlString, const std::string& keyword);

std::vector<double> getDoubleVectorValuesFromXML(const std::string& xmlString, const std::string& keyword);
std::vector<std::string> getStringVectorValuesFromXML(const std::string& input, const std::string& keyword);

}
