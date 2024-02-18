
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
