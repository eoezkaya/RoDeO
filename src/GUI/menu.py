import tkinter as tk
from tkinter import filedialog
import xml.etree.ElementTree as ET

class Menu(tk.Menu):
    def __init__(self, parent, settings):
        self.settings = settings
        super().__init__(parent)
        filemenu = tk.Menu(self, tearoff=0)
        filemenu.add_command(label="New", command=self.donothing)
        filemenu.add_command(label="Open", command=self.openFile)
        filemenu.add_command(label="Save", command=self.saveAsXMLFile)
        filemenu.add_command(label="Save as...", command=self.donothing)
        filemenu.add_command(label="Close", command=self.donothing)

        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=parent.quit)
        self.add_cascade(label="File", menu=filemenu)

    def donothing(self):
        pass    
    
    def saveAsXMLFile(self):
        print("saving as a xml file...")
        
        filename = self.remove_whitespace(self.settings.name) + ".xml"
        print(filename)
        
        xml_string_objective_function = self.settings.objective.generate_xml_string()
        
        print(xml_string_objective_function)
        
        xml_string = self.settings.generate_xml_string()
        
        self.save_string_as_xml(xml_string, filename)
        
   

    def save_string_as_xml(self, input_string, file_path):
    
        with open(file_path, "w") as file:
            file.write(input_string)



    
    
    def openFile(self):
        file_path = filedialog.askopenfilename(filetypes=[("XML files", "*.xml")])

        if file_path:
            with open(file_path, 'r') as file:
                content = file.read()
                print("File content:\n", content)
                return content

    def remove_whitespace(self,input_string):
        return ''.join(input_string.split())         
            