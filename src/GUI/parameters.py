class SingleParameter:
    def __init__(self,name, type, lb, ub, increment = 0):
        self.name = name
        self.lb =lb
        self.ub =ub
        self.type = type
        self.increment = increment
    def print(self):
        print(self.name, self.type, self.lb, self.ub, self.increment)    
