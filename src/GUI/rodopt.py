import tkinter as tk

import customtkinter as ctk
from PIL import Image, ImageTk
import qrcode

import tkinter as tk

from menu import Menu 
from gui_settings import GUISettings
from boxConstraints import BoxConstraints


class GUI(ctk.CTk):

	def __init__(self):

		# window setup
		ctk.set_appearance_mode('light')
		super().__init__(fg_color='#d2faf5')

		self.settings = GUISettings()

		# customization
		self.title('rodeo')
		self.geometry('600x600')
		self.resizable(False, False)

		menubar = Menu(self)
		
		self.config(menu=menubar)
		
		
		
		self.info = InfoString(self)
		
				
		ProblemSettings(self, self.info, self.settings)
		
		
		
		
		self.lowerBound = ctk.StringVar()
		self.upperBound = ctk.StringVar()
		  
		self.boxConstraints = BoxConstraints(self,self.info,self.lowerBound, self.upperBound)
#		self.boxConstraintsString.trace('w',self.boxConstraints.readBoxConstraintValues)
		
		
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
		self.pack(side='bottom', pady = 5)
		
		
	def updateInfoText(self, text):
		self.informationText.set(text)
		



		
	



class ProblemSettings(ctk.CTkFrame):

	def __init__(self, parent, info, settings):
		super().__init__(master=parent, corner_radius=10, fg_color='#03fce3')
		self.pack()
		
		self.problemName = ctk.StringVar()  
		self.problemDim = ctk.StringVar()  
		self.info = info
		
		nameFrame = ctk.CTkFrame(self, bg_color = "transparent", fg_color="#03fce3")    
		
		nameLabel = ctk.CTkLabel(nameFrame, text = "Problem name:",bg_color = "transparent")   
		nameEntry = ctk.CTkEntry(nameFrame, textvariable = self.problemName)  
		nameEntry.bind('<FocusIn>',  lambda event: info.updateInfoText("enter the problem name")) 
		nameLabel.pack(side = "left", padx=6, pady=6)
		nameEntry.pack(side = "left")
		nameFrame.pack(fill = "x")
		
	
		dimensionFrame = ctk.CTkFrame(self,fg_color = "transparent")
		dimensionEntry = ctk.CTkEntry(dimensionFrame, width = 50, textvariable = self.problemDim, bg_color = "transparent")
		dimensionLabel = ctk.CTkLabel(dimensionFrame, text = "Problem dimension:",bg_color = "transparent")     
		dimensionLabel.pack(side = "left", padx=6, pady=6)
		dimensionEntry.bind('<FocusIn>',  lambda event: info.updateInfoText("enter the number of parameters")) 
		dimensionEntry.pack(side = "left", pady = 6)
		dimensionFrame.pack(fill = "x")
		
		
		self.problemDim.trace('w',self.updateProblemDimension)
		
	def updateProblemDimension(self,*args):
		try:
			val = int(self.problemDim.get())
			self.info.updateInfoText("entered value is valid")
		except ValueError:
			self.info.updateInfoText("The dimension must be an integer value")
			
		
			


GUI()

