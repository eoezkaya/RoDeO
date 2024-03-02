import tkinter as tk
from tkinter import ttk

import customtkinter as ctk
from PIL import Image, ImageTk
import qrcode



from menu import Menu 
from gui_settings import GUISettings
from boxConstraints import BoxConstraints
from problem_settings import ProblemSettings
from objective_function_frame import ObjectiveFunctionFrame
from constraint_functions_frame import ConstraintFunctionsFrame
from font_and_color_settings import *

class GUI(ctk.CTk):

	def __init__(self):

		# window setup
		ctk.set_appearance_mode('light')
		super().__init__()

		self.settings = GUISettings()

		# customization
		self.title('rodeo')
		self.geometry('1200x800')
		self.resizable(False, False)
		self.configure(fg_color = WHITE)
		
		tabview = ctk.CTkTabview(self)
		tabview.pack(expand = True, fill = "both")

		tabview.add("Optimization Study")  # add tab at the end
		tabview.set("Optimization Study")  # set currently visible tab
		
		self.mainFrame = ctk.CTkFrame(master=self.tab("Optimization Study"), fg_color = WHITE)
		self.mainFrame.pack(expand = True, fill = "both")
		

		menubar = Menu(self)
		
		self.config(menu=menubar)
#		self.info = InfoString(self)
#		self.info.pack(side = "bottom", pady = 5)
				
#		self.problemSettings = ProblemSettings(self, self.info, self.settings)
#		self.problemSettings.pack(fill = "both", padx = 3, pady = 3)
		
#		self.boxConstraints = BoxConstraints(self,self.info, self.settings)
#		self.boxConstraints.pack(fill = "x")
		
#		self.objectiveFunctionWindow = ObjectiveFunctionFrame(self,self.info, self.settings)
#		self.objectiveFunctionWindow.pack(fill = "x", padx = 3, pady = 3)
		
#		self.constraintFunctionsWindow = ConstraintFunctionsFrame(self,self.info, self.settings)
#		self.constraintFunctionsWindow.pack(fill = "x", padx = 3, pady = 3)
		
		
		
		
		# running the app
		self.mainloop()

	def placeBoxConstraintsFrame(self,*args):
		
		
		if(self.convertToInt(self.problemDim.get())):
			dimension = int(self.problemDim.get())
			print(dimension)
		else:
			self.info.updateInfoText("enter a valid integer")	

	
	def convertToInt(self, value):
		try:
			int(value)
			return True
		except ValueError:
			return False
		
		
		

        	
				
class InfoString(ctk.CTkLabel):
	
	def __init__(self, parent):
		self.informationText = ctk.StringVar()
		super().__init__(master=parent, textvariable=self.informationText)
		
		
		
	def updateInfoText(self, text):
		self.informationText.set(text)
		




			


GUI()

