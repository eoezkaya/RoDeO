import customtkinter as ctk
from font_and_color_settings import *
from menu import Menu 
from problem_settings import ProblemSettings
from gui_settings import GUISettings
from objective_function_frame import ObjectiveFunctionFrame
from constraint_functions_frame import ConstraintFunctionsFrame
from parametersFrame import ParametersFrame

#from PIL import Image

#fileopen_image_path = "./Icons/openfile.png"
#fileopenimage = ctk.CTkImage(dark_image = Image.open(fileopen_image_path), light_image = Image.open(fileopen_image_path))
        


class MyTabView(ctk.CTkTabview):
    def __init__(self, master, settings, **kwargs):
        
        self.settings = settings
        super().__init__(master, **kwargs)

       

        # create tabs
        self.add("Optimization Study")
        
        
        self.problemSettings = ProblemSettings(self.tab("Optimization Study"), self.settings)
        self.problemSettings.pack(fill = "x", padx = 3, pady = 3)
        
        self.parametersField = ParametersFrame(self.tab("Optimization Study"),self,self.settings)
        self.parametersField.pack(fill = "x", padx = 3, pady = 3)
       
        self.objectiveFunctionWindow = ObjectiveFunctionFrame(self.tab("Optimization Study"), self.settings)
        self.objectiveFunctionWindow.pack(fill = "x", padx = 3, pady = 3)
        
        self.constraintFunctionsWindow = ConstraintFunctionsFrame(self.tab("Optimization Study"), self, self.settings)
        self.constraintFunctionsWindow.pack(fill = "x", padx = 3, pady = 3)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.settings = GUISettings()
        ctk.set_appearance_mode('light')
        self.title('rodeo')
        self.geometry('1200x900')
        self.resizable(False, False)
        
        self.tab_view = MyTabView(self, self.settings, fg_color = WHITE)
        self.tab_view.pack(expand = True, fill = "both")
        
        menubar = Menu(self, self.settings)
        
        self.config(menu=menubar)
        
        self.mainloop()

app = App()
