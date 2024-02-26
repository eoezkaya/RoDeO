import tkinter as tk
import customtkinter as ctk
from font_and_color_settings import *

class IntegerEntryField(ctk.CTkFrame):
    def __init__(self, parent, enteredValue, labelText, infoText = None, info = None ):
        
        super().__init__(master=parent, corner_radius=0,fg_color= "transparent" )

        labelText += ":"
        self.enteredValue = enteredValue
        self.label = ctk.CTkLabel(self, 
                                  text = labelText, 
                                  bg_color = "transparent", 
                                  fg_color= "transparent",
                                  font = (FONT , LABELSIZE_NORMAL),
                                  anchor= "w")  
        self.entry = ctk.CTkEntry(self, width = 100, textvariable = self.enteredValue, bg_color = "transparent")
        
        if(info):
            self.entry.bind('<FocusIn>',  lambda event: info.updateInfoText(infoText)) 
        
        
        self.label.pack(side = "left", padx = 3, pady = 3)
        self.entry.pack(side = "left", fill = "x", padx = 3)
        
        
        
 # Example of using the custom combobox
def main():
    root = ctk.CTk()
    root.title("IntegerEntryField Example")


    value = ctk.StringVar()
    entryField = IntegerEntryField(root, value, "enter some integer")
    entryField.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()       