import xml.etree.ElementTree as ET

class ConstraintFunction:
    def __init__(self, name = None):
        self.name = name
        self.constraint_type = None
        self.input_file_name = None
        self.output_file_name = None
        self.constraint_value = 0.0
        self.executable_name = None
        self.constraint_type = ">"
        self.surrogate_model_type = "ORDINARY_KRIGING"
        self.number_of_training_iterations = 10000
        self.training_data_name = None
        
    def print(self):
        print("Name: ",self.name)
        print("Value:", self.constraint_value)
        print("Type:", self.constraint_type)
        print("Executable :",self.executable_name)
        print("Input file name:", self.input_file_name)
        print("Output file name:",self.output_file_name)
        print("Training data file name:",self.training_data_name)
        print("Surrogate model type:", self.surrogate_model_type)
        print("Number of training iterations:", self.number_of_training_iterations)
    
    def generate_xml_string(self):
        # Create the root element
        root = ET.Element("ConstraintFunction")

        # Create sub-elements for each field and add them to the root
        ET.SubElement(root, "Name").text = self.name
        ET.SubElement(root, "TrainingDataName").text = self.training_data_name
        ET.SubElement(root, "SurrogateModelType").text = self.surrogate_model_type
        ET.SubElement(root, "Type", self.constraint_type)
        ET.SubElement(root, "Value", self.constraint_value)
        ET.SubElement(root, "ExecutableName").text = self.executable_name
        ET.SubElement(root, "InputFileName").text = self.input_file_name
        ET.SubElement(root, "OutputFileName").text = self.output_file_name
        ET.SubElement(root, "NumberOfTrainingIterations").text = self.number_of_training_iterations

        # Create an ElementTree object and convert it to a string
        xml_tree = ET.ElementTree(root)
        xml_string = ET.tostring(root).decode()

        return xml_string  
    
    def parse_xml_string(self,xml_string):
    # Create an ElementTree object from the XML string
        root = ET.fromstring(xml_string)
        print("root = ", root.tag)
        for element in root:
            print("element tag = ", element.tag)
            if element.tag == "ConstraintFunction":
        # Iterate through child elements and fill the fields
                for child in element:
                    print("child tag = ", child.tag)
                    if child.tag == "Name":
                        self.name = child.text
                    elif child.tag == "TrainingDataName":
                        self.training_data_name = child.text
                    elif child.tag == "SurrogateModelType":
                        self.surrogate_model_type = child.text
                    elif child.tag == "ExecutableName":
                        self.executable_name = child.text
                    elif child.tag == "InputFileName":
                        self.input_file_name = child.text
                    elif child.tag == "OutputFileName":
                        self.output_file_name = child.text
                    elif child.tag == "Value":
                        self.constraint_value = child.text    
                    elif child.tag == "Type":
                        self.constraint_type = child.text
                    elif child.tag == "NumberOfTrainingIterations":
                        self.number_of_training_iterations = child.text            
                return 1
       
        return 0      
