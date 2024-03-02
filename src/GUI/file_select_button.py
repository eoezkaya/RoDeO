import tkinter as tk
import customtkinter as ctk

from tkinter import filedialog

from font_and_color_settings import *

class FileSelectButton(ctk.CTkButton):
    def __init__(self, parent, filename, fileTypes = [("All files", "*.*")]):
        
        self.filetypes = fileTypes
        global fileopenimage
        super().__init__(master=parent, text = 'Load', 
                         command=self.select_file,
                          fg_color = WHITE, width = 50, border_width=1 )
        self.filename = filename
        

    def select_file(self):
        file_path = filedialog.askopenfilename(title="Select a file", filetypes = self.filetypes)
        if file_path:
            self.filename.set(file_path)
            
     
            