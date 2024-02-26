
from objectiveFunction import ObjectiveFunction
from constraintFunction import ConstraintFunction
from scipy.sparse.linalg._eigen.lobpcg.lobpcg import _get_indx

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
                
                