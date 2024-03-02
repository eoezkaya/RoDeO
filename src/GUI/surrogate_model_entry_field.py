import tkinter as tk
import customtkinter as ctk
from font_and_color_settings import *

class SurrogateModelSelectionField(ctk.CTkFrame):
    def __init__(self, parent, selectedModel ):
        self.availableSurrogateModels = ["Only with functional values", "Gradient enhanced"]
        super().__init__(master=parent, corner_radius=0, fg_color = "transparent")

        self.selected_value = selectedModel
        self.label = ctk.CTkLabel(self, text = "Surrogate model:"
                                  ,bg_color = "transparent"
                                  , width= 200, font = (FONT, LABELSIZE_NORMAL), anchor = "w")  
        self.comboBox = ctk.CTkComboBox(self, values=self.availableSurrogateModels,command=self.modelSelect,  width= 200)
        
        self.label.pack(side = "left", padx = 7)
        self.comboBox.pack(side = "left", padx = 7)
        

    def modelSelect(self, *args):
        self.selected_value.set(self.comboBox.get())

    

# Example of using the custom combobox
def main():
    root = ctk.CTk()
    root.title("Custom Combobox Example")


    model = ctk.StringVar()
    custom_combobox = SurrogateModelSelectionField(root, model)
    custom_combobox.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
