import tkinter as tk
import customtkinter as ctk
from tkinter import ttk
from file_entry_field import FileEntryField
from surrogate_model_entry_field import SurrogateModelSelectionField
from integer_entry_field import IntegerEntryField
from PIL import Image
from font_and_color_settings import *
from singleConstraintFrame import SingleConstraintFrame


class ConstraintFunctionsFrame(ctk.CTkScrollableFrame):    
    def __init__(self, parent, info, settings):
        super().__init__(master=parent, corner_radius= PANEL_CORNER_RADIUS, 
                         fg_color = OBJECTIVE_FUNCTION_PANEL_COLOR, 
                         border_width = 0,
                         border_color = BLACK,
                         scrollbar_fg_color = "transparent",
                         label_text = "Constraints",
                         label_font = (FONT, 20),
                         label_fg_color = SCROLLABLE_FRAME_TITLE_COLOR )
        self.info = info
        self.settings = settings
        
        image_path = "./Icons/constraintFunction.png"
        self.image = ctk.CTkImage(dark_image = Image.open(image_path), light_image = Image.open(image_path),size=(40, 40))
        
        addConstraintButton = ctk.CTkButton(self, text = 'Add constraint', 
                                            image = self.image, 
                                            fg_color = "white", 
                                            text_color = "black", 
                                            border_color = "black",
                                            border_width = 3,
                                            command = self.addConstraint,
                                            font = (FONT , 14) )
        
        
        addConstraintButton.pack(side=tk.RIGHT, anchor=tk.N, padx = 10 , pady = 5)
        
        
        
    def addConstraint(self):
        constraintFrame = SingleConstraintFrame(self, self.info, self.settings)
        constraintFrame.pack(expand = True, fill = "x")
            
            
            
            
            
            