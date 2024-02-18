import tkinter as tk
from tkinter import filedialog


class Menu(tk.Menu):
    def __init__(self, parent):
        super().__init__(parent)
        filemenu = tk.Menu(self, tearoff=0)
        filemenu.add_command(label="New", command=self.donothing)
        filemenu.add_command(label="Open", command=self.openFile)
        filemenu.add_command(label="Save", command=self.donothing)
        filemenu.add_command(label="Save as...", command=self.donothing)
        filemenu.add_command(label="Close", command=self.donothing)

        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=parent.quit)
        self.add_cascade(label="File", menu=filemenu)

    def donothing(self):
        pass    
    
    def openFile(self):
        file_path = filedialog.askopenfilename(filetypes=[("XML files", "*.xml")])

        if file_path:
            with open(file_path, 'r') as file:
                content = file.read()
                print("File content:\n", content)
                return content

            
            