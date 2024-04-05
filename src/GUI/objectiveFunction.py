import xml.etree.ElementTree as ET


class ObjectiveFunction:
    def __init__(self):
        self.name = "Objective Function"
        self.number_of_training_iterations = 10000
        self.executable_name = None
        self.executable_name_gradient = None
        self.input_file_name = None
        self.output_file_name = None
        self.output_file_name_gradient = None
        self.training_data_name = None
        self.surrogate_model_type = "ORDINARY_KRIGING"
    def generate_xml_string(self):
        # Create the root element
        root = ET.Element("ObjectiveFunction")
        self.print()

        # Create sub-elements for each field and add them to the root
        if(self.name != None):
            ET.SubElement(root, "Name").text = str(self.name)
        if(self.training_data_name != None):    
            ET.SubElement(root, "TrainingDataName").text = str(self.training_data_name)
        
        ET.SubElement(root, "SurrogateModelType").text = str(self.surrogate_model_type)
        
        if(self.executable_name != None):
            ET.SubElement(root, "ExecutableName").text = str(self.executable_name)
        if(self.executable_name_gradient != None):        
            ET.SubElement(root, "ExecutableNameGradient").text = str(self.executable_name_gradient)
        if(self.input_file_name != None):            
            ET.SubElement(root, "InputFileName").text = str(self.input_file_name)
        if(self.output_file_name != None):            
            ET.SubElement(root, "OutputFileName").text = str(self.output_file_name)
        if(self.output_file_name_gradient != None):            
            ET.SubElement(root, "OutputFileNameGradient").text = str(self.output_file_name_gradient)
        ET.SubElement(root, "NumberOfTrainingIterations").text =  str(self.number_of_training_iterations)

        # Create an ElementTree object and convert it to a string
        xml_tree = ET.ElementTree(root)
        xml_string = ET.tostring(root).decode()

        return xml_string
    
    def print(self):
        print("-"*30)
        print("Name: ",self.name)
        print("Executable :",self.executable_name)
        print("Executable for gradient :",self.executable_name_gradient)
        print("Input file name:", self.input_file_name)
        print("Output file name:",self.output_file_name)
        print("Training data file name:",self.training_data_name)
        print("Surrogate model type:", self.surrogate_model_type)
        print("Number of training iterations:", self.number_of_training_iterations)
        print("-"*30)
        
        
    def parse_xml_string(self,xml_string):
    # Create an ElementTree object from the XML string
        root = ET.fromstring(xml_string)
        print("root = ", root.tag)
        for element in root:
            print("element tag = ", element.tag)
            if element.tag == "ObjectiveFunction":
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

                return True
       
        return False


