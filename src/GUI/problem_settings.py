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
                         height = 80,
                         label_text = "Problem Settings",
                         label_font = (FONT, 20),
                         label_fg_color = SCROLLABLE_FRAME_TITLE_COLOR)
        self._scrollbar.configure(height=0)
             
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

        self.problemName.trace('w',self.updateProblemName)        
        nameFrame.pack(expand = True, fill  = "x", padx = 3, pady = 3)
        
        
        
        
    def updateProblemName(self,*args):
        name = self.problemName.get()
        self.globalSettings.name = name
        self.globalSettings.print()    
        
        

            
            
            
        