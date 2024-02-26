import tkinter as tk
import customtkinter as ctk
from objectiveFunction import ObjectiveFunction
from file_entry_field import FileEntryField
from surrogate_model_entry_field import SurrogateModelSelectionField
from integer_entry_field import IntegerEntryField
from font_and_color_settings import *


class ObjectiveFunctionFrame(ctk.CTkScrollableFrame):    
    def __init__(self, parent, info, settings):
        super().__init__(master=parent, 
                         corner_radius= PANEL_CORNER_RADIUS, 
                         fg_color = OBJECTIVE_FUNCTION_PANEL_COLOR, 
                         border_width = 0,
                         label_text = "Objective Function",
                         label_font = (FONT, 20),
                         label_fg_color = SCROLLABLE_FRAME_TITLE_COLOR)
        self.info = info
        self.globalSettings = settings
        self.objectiveFunction = ObjectiveFunction()
        
                
        self.name = ctk.StringVar()  
        self.name.set("Objective Function")
        nameFrame = ctk.CTkFrame(self, 
                                 corner_radius = 0,
                                 bg_color = "transparent",
                                 fg_color = "transparent",
                                 border_width = 0)    
        
        
        nameLabel = ctk.CTkLabel(nameFrame, 
                                 text = "Objective function name:",
                                 bg_color = "transparent", 
                                 fg_color = "transparent",
                                 font = (FONT, LABELSIZE_NORMAL))
           
        nameEntry = ctk.CTkEntry(nameFrame, textvariable = self.name,bg_color = "transparent",fg_color = "WHITE" )  
        nameEntry.bind('<FocusIn>',  lambda event: info.updateInfoText("enter the name of objective function")) 
        nameLabel.pack(side = "left", padx=6, pady=6)
        nameEntry.pack(side = "left")
       
        
        self.name.trace('w',self.setName)
        
        
        self.executableName = ctk.StringVar()
        executableField = FileEntryField(self,"Name of the executable file", self.executableName)   
        self.executableName.trace('w',self.setExeName)
        
        
        self.trainingDataFileName = ctk.StringVar()
        trainingDataField = FileEntryField(self,"Training data", self.trainingDataFileName, [("CVS files", "*.csv")])
        
        
        self.outputfilename = ctk.StringVar()
        outputfilenameField = FileEntryField(self,"Output file", self.outputfilename)
#
        
        self.designVectorFilename = ctk.StringVar()
        designVectorFilenameField = FileEntryField(self,"Design vector file", self.designVectorFilename)
#        self.designVectorFilenameField.pack()
        
        self.surrogateModelName = ctk.StringVar()
        surrogateModelEntry = SurrogateModelSelectionField(self,self.surrogateModelName)  
#        self.surrogateModelEntry.pack(side = "left")       
        self.surrogateModelName.trace('w',self.setSurrogateModel)
        
        self.numberOfTrainingIterations = ctk.StringVar()
        self.numberOfTrainingIterations.set(10000)
        numberOfTrainingIterationsField = IntegerEntryField(self,self.numberOfTrainingIterations, "Number of training iterations","enter maximum number iterations used for model training", self.info )

        
        nameFrame.pack(fill = "x", padx = 3, pady = 3)
        executableField.pack(fill = "x", padx = 3, pady = 3)
        trainingDataField.pack(fill = "x", padx = 3, pady = 3) 
        outputfilenameField.pack(fill = "x", padx = 3, pady = 3) 
        designVectorFilenameField.pack(fill = "x", padx = 3, pady = 3)
        surrogateModelEntry.pack(fill = "x", padx = 3, pady = 3)
        numberOfTrainingIterationsField.pack(fill = "x", padx = 3, pady = 3)     

    def setName(self, *args):
        self.objectiveFunction.name = self.name.get()
        self.objectiveFunction.print()
        
    def setExeName(self, *args):
        self.objectiveFunction.executable_name = self.executableName.get()
        self.objectiveFunction.print()    
        
    def setSurrogateModel(self, *args):
        print(self.surrogateModelName.get())
        self.objectiveFunction.surrogate_model_type = self.surrogateModelName.get()
        self.objectiveFunction.print()    
        
        
        

            
            
            
            
            
            