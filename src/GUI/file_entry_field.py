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
        label = ctk.CTkLabel(self, text = labelText,
                             bg_color = "transparent", 
                             width= 200, 
                             fg_color= "transparent",
                             font = (FONT, LABELSIZE_NORMAL), anchor ="w")  
        entry = ctk.CTkEntry(self, textvariable = self.filename, width= 300)
        selectButton = FileSelectButton(self, self.filename, filetypes)
        
        # layout
       
        label.pack(side = "left", padx = 7, pady = 2)
        entry.pack(side = "left", padx = 7, pady = 2)
        selectButton.pack(side = "left", padx = 7, pady = 2)
        
        
        
        
        
        
        
        
# Example of using the custom button
def main():
    root = tk.Tk()
    root.title("File Select Button Example")

    executableName = ctk.StringVar()
    field = FileEntryField(root, "test",executableName)
    field.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()        