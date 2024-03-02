import tkinter as tk
import customtkinter as ctk

from integer_entry_field import IntegerEntryField
from font_and_color_settings import *
from tkinter import messagebox

class ProblemSettings(ctk.CTkScrollableFrame):

    def __init__(self, parent, settings):
        super().__init__(master=parent,
                         fg_color= OBJECTIVE_FUNCTION_PANEL_COLOR, 
                         corner_radius= PANEL_CORNER_RADIUS, 
                         border_width = 0,
                         label_text = "Problem Settings",
                         label_font = (FONT, 20),
                         label_fg_color = SCROLLABLE_FRAME_TITLE_COLOR)
        
             
        self.problemName = ctk.StringVar()  
         

        self.globalSettings = settings
        
        nameFrame = ctk.CTkFrame(self, bg_color = "transparent", fg_color= "transparent")    
        
        nameLabel = ctk.CTkLabel(nameFrame, 
                                 text = "Problem name:",
                                 bg_color = "transparent", 
                                 fg_color=WHITE, 
                                 width=100, 
                                 anchor = "w",
                                 font = (FONT , LABELSIZE_NORMAL)) 
        
        self.problemName.set("Optimization Study")
        nameEntry = ctk.CTkEntry(nameFrame, textvariable = self.problemName)  
        nameLabel.pack(side = "left", padx=3, pady=3)
        nameEntry.pack(side = "left", fill = "x")
#        nameFrame.pack(fill = "x")
        self.problemName.trace('w',self.updateProblemName)
        
        self.problemDim = ctk.StringVar() 
        dimensionFrame = IntegerEntryField(self,self.problemDim, 
                                           "Problem dimension")
#        dimensionFrame.pack(fill = "x")
        self.problemDim.trace('w',self.updateProblemDimension)
        
        
        nameFrame.pack(expand = True, fill  = "x", padx = 3, pady = 3)
        dimensionFrame.pack(expand = True, fill = "x", padx = 3, pady = 3)
        
        
        
        
    def updateProblemName(self,*args):
        name = self.problemName.get()
        self.globalSettings.name = name
        self.globalSettings.print()    
        
        
    def updateProblemDimension(self,*args):
        if(self.problemDim.get()):
            try:
                val = int(self.problemDim.get())
                self.globalSettings.dimension = val
                self.globalSettings.print()
            
            except ValueError:
                messagebox.showerror("Error", "The dimension must be an integer")
            
            
            
        