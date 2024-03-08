import tkinter as tk
import customtkinter as ctk
from file_select_button import FileSelectButton
from font_and_color_settings import *

class FileEntryField(ctk.CTkFrame):    
    def __init__(self, parent, labelText, entryVariable, filetypes = [("All files", "*.*")]):
        super().__init__(master=parent, corner_radius=0,fg_color= "transparent")
        
        labelText +=": "
        self.filename = entryVariable
        
       # widgets 
        self.label = ctk.CTkLabel(self, text = labelText,
                             bg_color = "transparent", 
                             width= 200, 
                             fg_color= "transparent",
                             font = (FONT, LABELSIZE_NORMAL), anchor ="w")  
        self.entry = ctk.CTkEntry(self, textvariable = self.filename, width= 300)
        self.selectButton = FileSelectButton(self, self.filename, filetypes)
        
        # layout
       
        self.label.pack(side = "left", padx = 7, pady = 2)
        self.entry.pack(side = "left", padx = 7, pady = 2)
        self.selectButton.pack(side = "left", padx = 7, pady = 2)
        
        
        
    def deactivateField(self):
        self.entry.configure(state = "disabled")
        self.entry.configure(fg_color = GRAY)    
    def activateField(self):
        self.entry.configure(state = "normal")
        self.entry.configure(fg_color = WHITE)        
        
