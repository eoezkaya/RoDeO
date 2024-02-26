import tkinter as tk
import customtkinter as ctk
from tkinter import simpledialog
from constraintFunction import ConstraintFunction
from font_and_color_settings import *

class SingleConstraintFrame(ctk.CTkFrame):
    def __init__(self, master, info, settings ):
        super().__init__(master, fg_color= "transparent")
        self.settings = settings
        self.info = info
        self.ID = self.settings.lastGivenConstraintID+1
        self.settings.lastGivenConstraintID = self.ID 
        self.label_text = ctk.StringVar()
        labelText = "Constraint" + str(self.ID)
        self.label_text.set(labelText)
        self.label = ctk.CTkLabel(self, textvariable = self.label_text, font = (FONT,LABELSIZE_NORMAL), width = 200)
        self.label.pack(side = "left", pady=3)
        
        self.edit_button = ctk.CTkButton(self, text="Edit")
        self.edit_button.pack(side = "left", pady = 3, padx = 3)
        
        self.delete_button = ctk.CTkButton(self, text="X", command=self.deleteConstraint)
        self.delete_button.pack(side = "left", pady=3, padx = 3)
        

        self.constraint = ConstraintFunction()
        self.constraint.ID = self.ID 
        
        settings.constraints.append(self.constraint)
        
#        settings.printConstraints()

    def deleteConstraint(self):
        self.settings.deleteConstraint(self.ID)
        self.destroy()  
#        self.settings.printConstraints()  

