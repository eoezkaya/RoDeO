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
        
        self.ifUseMathExpression = ctk.StringVar(value="off")
        self.mathematicalExpression = ctk.StringVar(value = "e.g., x[0]**2.0 + sqrt(x[1]) - 2.12*sin(x[3]) + exp(x[0]*x[1])")
        
        self.constraintValue = ctk.StringVar()
        
        settings.constraints.append(self.constraint)
        
        self.editConstraint()
        
#        settings.printConstraints()

    def deleteConstraint(self):
        try:
            # if the tab is already there
            currentTab = self.mainWindow.tab(self.initialTabName)
            self.mainWindow.delete(self.initialTabName)
        except: 
            pass     
        
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
        nameLabel.pack(side = "left", padx=3, pady=3)
        nameEntry.pack(side = "left")
        
        nameFrame.pack(fill = "x", padx = 3, pady = 3)
          
        self.name.trace('w',self.updateName)
        
        constraintDefinitionField = ctk.CTkFrame(self.mainWindow.tab(self.tabName), 
                                 corner_radius = 0,
                                 bg_color = "transparent",
                                 fg_color = "transparent",
                                 border_width = 0)    
        
        definitionText = "Definition: " + self.name.get()
        
        definitionLabel = ctk.CTkLabel(constraintDefinitionField, text = definitionText
                                  ,bg_color = "transparent"
                                  , font = (FONT, LABELSIZE_NORMAL), anchor = "w")  
        definitionLabel.pack(side = "left")
        
        inequlityTypes = [">", "<"]
        comboBox = ctk.CTkComboBox(constraintDefinitionField, values=inequlityTypes,  width= 20)
        comboBox.pack(side = "left", padx = 5, pady = 3)
        
        self.constraintValueEntry = ctk.CTkEntry(constraintDefinitionField,
                                   textvariable = self.constraintValue,
                                   width = 50,
                                   fg_color= WHITE)  
        self.constraintValueEntry.pack(side = "left", padx = 3)
        
        
        constraintDefinitionField.pack(fill = "x", padx = 3, pady = 3)
        
        
        
        mathExpressionField = ctk.CTkFrame(self.mainWindow.tab(self.tabName), 
                                 corner_radius = 0,
                                 bg_color = "transparent",
                                 fg_color = "transparent",
                                 border_width = 0)    
        
        self.checkbox1 = ctk.CTkCheckBox(mathExpressionField, 
                                    text="Use mathematical expression", 
                                    command=self.checkbox1_event,
                                    variable=self.ifUseMathExpression, 
                                    width = 100)
        
        self.checkbox1.pack(side = "left", padx=3, pady=3)
        
        self.expression = ctk.CTkEntry(mathExpressionField,
                                   textvariable = self.mathematicalExpression,
                                   width = 800,
                                   fg_color= GRAY)  
        self.expression.pack(side = "left", fill = "x", padx = 6, pady = 3)
        self.expression.configure(state="disabled")
        
        mathExpressionField.pack(fill = "x", padx = 3, pady = 3)
        
        
        
        self.executableField = FileEntryField(self.mainWindow.tab(self.tabName),"Name of the executable file", self.executableName)   
        self.executableField.pack(fill = "x", padx = 3, pady = 3)
        
        self.trainingDataField = FileEntryField(self.mainWindow.tab(self.tabName),"Training data", self.trainingDataFileName, [("CVS files", "*.csv")])
        self.trainingDataField.pack(fill = "x", padx = 3, pady = 3) 
                        
        self.outputfilenameField = FileEntryField(self.mainWindow.tab(self.tabName),"Output file", self.outputfilename)
        self.outputfilenameField.pack(fill = "x", padx = 3, pady = 3) 
        
        
         
        self.designVectorFilenameField = FileEntryField(self.mainWindow.tab(self.tabName),"Design vector file", self.designVectorFilename)
        self.designVectorFilenameField.pack(fill = "x", padx = 3, pady = 3)
            
        self.surrogateModelEntry = SurrogateModelSelectionField(self.mainWindow.tab(self.tabName),self.surrogateModelName)
        self.surrogateModelEntry.pack(fill = "x", padx = 3, pady = 3)
        
        
        closeButton = ctk.CTkButton(self.mainWindow.tab(self.tabName), text="Close", width = 30, command = self.closeTab)
        closeButton.pack(padx = 3, pady = 3)
        

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
            self.initialTabName = self.tabName      
                
            self.createWidgets()
            self.mainWindow.set(self.tabName)
        
        
    def updateName(self,*args):
        self.settings.updateConstraintName(self.name.get(), self.ID)  
        self.settings.printConstraints()   
        self.tabName = self.name.get()
        self.mainWindow._segmented_button._buttons_dict[self.initialTabName].configure(state=ctk.NORMAL, text=self.tabName)

        
          
    def updateExeName(self,*args):    
        self.settings.updateConstraintExeName(self.executableName.get(), self.ID)  
        self.settings.printConstraints()  
    
    def updateTrainingDataFileName(self,*args):    
        self.settings.updateConstraintTrainingFileName(self.trainingDataFileName.get(), self.ID)  
        self.settings.printConstraints() 
        
    def updateOutputFileName(self,*args):    
        self.settings.updateConstraintOutputFileName(self.outputfilename.get(), self.ID)  
        self.settings.printConstraints()  
    
    def updateDesignVectorFileName(self,*args):    
        self.settings.updateConstraintDesignVectorFileName(self.designVectorFilename.get(), self.ID)  
        self.settings.printConstraints()
    def updateSurrogateModel(self,*args):    
        pass
    def closeTab(self):
        self.mainWindow.delete(self.initialTabName)
    
    def checkbox1_event(self):
                
        if(self.checkbox1.get()):
            self.expression.configure(state="normal")
            self.expression.configure(fg_color = WHITE)
            
            self.executableField.deactivateField()
            self.trainingDataField.deactivateField()
            self.outputfilenameField.deactivateField()
            self.designVectorFilenameField.deactivateField()
            self.surrogateModelEntry.deactivate()
            
        else:
            self.expression.configure(state="disabled")
            self.expression.configure(fg_color = GRAY) 
            self.executableField.activateField()
            self.trainingDataField.activateField()
            self.outputfilenameField.activateField()
            self.designVectorFilenameField.activateField()
            self.surrogateModelEntry.activate()
            
            