
from objectiveFunction import ObjectiveFunction
from constraintFunction import ConstraintFunction
from parameters import SingleParameter
import xml.etree.ElementTree as ET

class GUISettings():
    def __init__(self):
        self.name = "OptimizationStudy"
        self.dimension = 0
        self.lowerBounds = []
        self.upperBounds = []
        self.parameters = []
        self.objective = ObjectiveFunction()
        self.constraints = []
        self.numberOfConstraints = 0
        self.lastGivenConstraintID = 0
        
        
        
    def print(self):
        print("Name:", self.name)
        print("Dim:", self.dimension)    
        print("Lower bounds:", self.lowerBounds)
        print("Upper bounds:", self.upperBounds)
        
    def printParameters(self):
        for parameter in self.parameters:
            parameter.print()    
            
    def printConstraints(self):
        for constraint in self.constraints:
            constraint.print()
    
    def generate_xml_string(self):
        root = ET.Element("Settings")
        
        ET.SubElement(root, "Name").text = str(self.name)
        
        # Create an ElementTree object and convert it to a string
        xml_tree = ET.ElementTree(root)
        xml_string = ET.tostring(root).decode()

        return xml_string
    
        
               
    def deleteConstraint(self, id):
        
        indx = 0
        indexToRemove = 0
        for constraint in self.constraints:
            if(constraint.ID == id):
                indexToRemove = indx
                break
            else:
                indx +=1
                
        self.constraints.pop(indexToRemove)       
                
    def updateConstraintName(self, newName, ID):
        
        for constraint in self.constraints:
            
            if(constraint.ID == ID):
                constraint.name = newName
                break
     
    def updateConstraintExeName(self, exeName, ID):
        
        for constraint in self.constraints:
            
            if(constraint.ID == ID):
                constraint.executable_name = exeName
                break 
    def updateConstraintTrainingFileName(self, filename, ID):
        
        for constraint in self.constraints:
            
            if(constraint.ID == ID):
                constraint.training_data_name = filename
                break  
    def updateConstraintDesignVectorFileName(self, filename, ID):
        
        for constraint in self.constraints:
            
            if(constraint.ID == ID):
                constraint.input_file_name = filename
                break          
             
    def updateConstraintOutputFileName(self, filename, ID):
        
        for constraint in self.constraints:
            
            if(constraint.ID == ID):
                constraint.output_file_name = filename
                break  
            
    