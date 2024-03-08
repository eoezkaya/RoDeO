
from objectiveFunction import ObjectiveFunction
from constraintFunction import ConstraintFunction

class GUISettings():
    def __init__(self):
        self.name = None
        self.dimension = 0
        self.lowerBounds = []
        self.upperBounds = []
        self.objective = ObjectiveFunction()
        self.constraints = []
        self.numberOfConstraints = 0
        self.lastGivenConstraintID = 0
        
        
        
    def print(self):
        print("Name:", self.name)
        print("Dim:", self.dimension)    
        print("Lower bounds:", self.lowerBounds)
        print("Upper bounds:", self.upperBounds)
        
        
    def printConstraints(self):
        for constraint in self.constraints:
            constraint.print()
               
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
     