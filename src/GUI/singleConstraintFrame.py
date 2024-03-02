import tkinter as tk
import customtkinter as ctk
from tkinter import simpledialog
from constraintFunction import ConstraintFunction
from font_and_color_settings import *

from file_entry_field import FileEntryField
from surrogate_model_entry_field import SurrogateModelSelectionField
from integer_entry_field import IntegerEntryField


class SingleConstraintFrame(ctk.CTkFrame):
    def __init__(self, parent, mainWindow, settings ):
        
        self.settings = settings
        self.mainWindow = mainWindow
        self.ID = self.settings.lastGivenConstraintID+1
        self.settings.lastGivenConstraintID = self.ID 
        self.name = ctk.StringVar()
        # initially tabName and the name are same
        self.initialTabName = "Constraint" + str(self.ID)
        self.tabName = self.initialTabName
        self.name.set(self.tabName)
        
        
        super().__init__(parent, fg_color= "transparent")
        
        self.label = ctk.CTkLabel(self, textvariable = self.name, font = (FONT,LABELSIZE_NORMAL), width = 200)
        self.label.pack(side = "left", pady=3)
        
        self.edit_button = ctk.CTkButton(self, text="Edit", width = 60, command= self.editConstraint)
        self.edit_button.pack(side = "left", pady = 3, padx = 3)
        
        self.delete_button = ctk.CTkButton(self, text="X", width = 30, command=self.deleteConstraint)
        self.delete_button.pack(side = "left", pady=3, padx = 3)
        
##########################################################################################################
        self.constraint = ConstraintFunction()
        self.constraint.ID = self.ID 
        
        self.executableName = ctk.StringVar()
        self.executableName.trace('w',self.updateExeName)
        
        self.trainingDataFileName = ctk.StringVar()
        self.trainingDataFileName.trace('w',self.updateTrainingDataFileName)
        
        self.outputfilename = ctk.StringVar()
        self.outputfilename.trace('w',self.updateOutputFileName)
        
        self.designVectorFilename = ctk.StringVar()
        self.designVectorFilename.trace('w',self.updateDesignVectorFileName)
        
        self.surrogateModelName = ctk.StringVar()   
        self.surrogateModelName.trace('w',self.updateSurrogateModel)
        
        settings.constraints.append(self.constraint)
        
        self.editConstraint()
        
#        settings.printConstraints()

    def deleteConstraint(self):
        self.mainWindow.delete(self.initialTabName)
        self.settings.deleteConstraint(self.ID)
        self.destroy()  
#        self.settings.printConstraints()  

    def createWidgets(self):
        nameFrame = ctk.CTkFrame(self.mainWindow.tab(self.tabName), 
                                 corner_radius = 0,
                                 bg_color = "transparent",
                                 fg_color = "transparent",
                                 border_width = 0)    
        
        
        nameLabel = ctk.CTkLabel(nameFrame, 
                                 text = "Constraint function name:",
                                 bg_color = "transparent", 
                                 fg_color = "transparent",
                                 font = (FONT, LABELSIZE_NORMAL))
           
        nameEntry = ctk.CTkEntry(nameFrame, textvariable = self.name,bg_color = "transparent",fg_color = "WHITE" )  
        nameLabel.pack(side = "left", padx=6, pady=6)
        nameEntry.pack(side = "left")
        
        nameFrame.pack(fill = "x", padx = 3, pady = 3)
          
        self.name.trace('w',self.updateName)
        
        executableField = FileEntryField(self.mainWindow.tab(self.tabName),"Name of the executable file", self.executableName)   
        executableField.pack(fill = "x", padx = 3, pady = 3)
        
        trainingDataField = FileEntryField(self.mainWindow.tab(self.tabName),"Training data", self.trainingDataFileName, [("CVS files", "*.csv")])
        trainingDataField.pack(fill = "x", padx = 3, pady = 3) 
                        
        outputfilenameField = FileEntryField(self.mainWindow.tab(self.tabName),"Output file", self.outputfilename)
        outputfilenameField.pack(fill = "x", padx = 3, pady = 3) 
        
        
         
        designVectorFilenameField = FileEntryField(self.mainWindow.tab(self.tabName),"Design vector file", self.designVectorFilename)
        designVectorFilenameField.pack(fill = "x", padx = 3, pady = 3)
            
        surrogateModelEntry = SurrogateModelSelectionField(self.mainWindow.tab(self.tabName),self.surrogateModelName)
        surrogateModelEntry.pack(fill = "x", padx = 3, pady = 3)
        
        

    def editConstraint(self):
        
#        if(not(self.tabName)):
#            self.tabName = " "
        try:
            # if the tab is already there
            currentTab = self.mainWindow.tab(self.initialTabName)
            self.mainWindow.set(self.initialTabName)
        except:
#            if(self.tabName):
            self.mainWindow.add(self.tabName)      
                
            self.createWidgets()
            self.mainWindow.set(self.tabName)
        
        
    def updateName(self,*args):
        self.settings.updateConstraintName(self.name.get(), self.ID)  
        self.settings.printConstraints()   
        self.tabName = self.name.get()
        self.mainWindow._segmented_button._buttons_dict[self.initialTabName].configure(state=ctk.NORMAL, text=self.tabName)

        
          
    def updateExeName(self,*args):    
        pass
    
    def updateTrainingDataFileName(self,*args):    
        pass
        
    def updateOutputFileName(self,*args):    
        pass
    
    def updateDesignVectorFileName(self,*args):    
        pass
    def updateSurrogateModel(self,*args):    
        pass
         