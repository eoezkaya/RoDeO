import tkinter as tk
import customtkinter as ctk
from font_and_color_settings import *

class BoxConstraints(ctk.CTkFrame):

    def __init__(self, parent, info, settings):
        super().__init__(master=parent, fg_color= WHITE)
        
        self.rowconfigure((0,1), weight = 1, uniform = 'a')
        self.columnconfigure((0,1,2), weight = 1, uniform = 'a')
        
        
        self.lb = ctk.StringVar()  
        self.ub = ctk.StringVar()  
        self.globalSettings = settings
        
        
        self.upperBoundsLabel = ctk.CTkLabel(self, text = "Upper bounds:",bg_color = "transparent", fg_color= WHITE)
        self.lowerBoundsLabel = ctk.CTkLabel(self, text = "Lower bounds:",bg_color = "transparent")
           
        self.upperBoundsEntry = ctk.CTkEntry(self, textvariable = self.ub)  
        self.lowerBoundsEntry = ctk.CTkEntry(self, textvariable = self.lb)
        
        self.lb.trace('w',self.readLowerBoxConstraintValues)
        self.ub.trace('w',self.readUpperBoxConstraintValues)
          
        self.upperBoundsEntry.bind('<FocusIn>',  lambda event: info.updateInfoText("enter the upper bounds for the parameters, e.g. 1.4,-2.9,11.9, 5.5")) 
        self.lowerBoundsEntry.bind('<FocusIn>',  lambda event: info.updateInfoText("enter the lower bounds for the parameters, e.g. 1.4,-2.9,11.9, 5.5")) 
 
 
        self.upperBoundsEntry.grid(row = 0, column = 1)
        self.lowerBoundsEntry.grid(row = 1, column = 1)
        
        self.upperBoundsLabel.grid(row = 0, column = 0)
        self.lowerBoundsLabel.grid(row = 1, column = 0)
        
        self.check_var1 = ctk.StringVar(value="off")
        checkbox1 = ctk.CTkCheckBox(self, text="Get box constraints from training data", command=self.checkbox1_event,
                                     variable=self.check_var1, onvalue="on", offvalue="off",width = 500)
        
        checkbox1.grid(row = 0, column = 2)
        
        
        self.check_var2 = ctk.StringVar(value="off")
        checkbox2 = ctk.CTkCheckBox(self, text="Use same values for all box constraints",
                                     variable=self.check_var2, onvalue="on", offvalue="off", width = 500)
        
        checkbox2.grid(row = 1, column = 2)
        
        
        
        self.lowerBounds = []
        self.upperBounds = []
        
        
    def checkbox1_event(self):
        print("checkbox toggled, current value:", self.check_var1.get())    
        if(self.check_var1.get() == "on"):
            self.upperBoundsEntry.configure(state="disabled")
            self.lowerBoundsEntry.configure(state="disabled")
        else:
            self.upperBoundsEntry.configure(state="normal")
            self.lowerBoundsEntry.configure(state="normal")
    
            
        
    def string_to_array(self,input_string):
    # Remove parentheses and split the string by commas
        values_str = input_string.strip('()')
        values_list = values_str.split(',')

    # Convert each value to an integer
        try:
            result_array = [float(value) for value in values_list]
        except ValueError:
            return None    
        return result_array
        
    def readLowerBoxConstraintValues(self,*args):
        lb = self.string_to_array(self.lb.get())
        
        self.globalSettings.lowerBounds = lb
                
        self.globalSettings.print()
        
        
    def readUpperBoxConstraintValues(self,*args):
        ub = self.string_to_array(self.ub.get())
        
        self.globalSettings.upperBounds = ub
                
        self.globalSettings.print()    
        
            