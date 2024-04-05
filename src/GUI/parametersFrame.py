import tkinter as tk
import customtkinter as ctk
from font_and_color_settings import *
from tkinter import ttk
from integer_entry_field import IntegerEntryField
from tkinter import messagebox
from parameters import SingleParameter

class ParametersFrame(ctk.CTkScrollableFrame):    
    def __init__(self, parent, mainWindow, settings):
        super().__init__(master=parent, corner_radius= PANEL_CORNER_RADIUS, 
                         fg_color = OBJECTIVE_FUNCTION_PANEL_COLOR, 
                         border_width = 0,
                         border_color = BLACK,
                         scrollbar_fg_color = "transparent",
                         label_text = "Parameters",
                         label_font = (FONT, 20),
                         label_fg_color = SCROLLABLE_FRAME_TITLE_COLOR )
        
        self.mainWindow = mainWindow
        self.settings = settings
        self.dimension = ctk.StringVar()
        self.tableInitialized = False
        self.numberOfInsertedEntries = 0
        self.editMode = False
        
 #       self.Parameters = []
        
        self.dimension.trace('w',self.updateProblemDimension) 
        
        settingsFrame = ctk.CTkFrame(self,fg_color = "transparent")
        settingsFrame.pack(expand = True, fill = "x")
        
        dimensionFrame = IntegerEntryField(settingsFrame,self.dimension,"Number of parameters")
        dimensionFrame.pack(side = "left", padx = 3, pady = 3)
        
       
        self.switch_var = ctk.StringVar(value="off")
        switch = ctk.CTkSwitch(settingsFrame, text="Edit variables", command=self.switchEditMode,
                                 variable=self.switch_var, onvalue="on", offvalue="off")
        switch.pack(side = "left")
        
        self.table = ttk.Treeview(self, columns = ('name', 'type', 'lb', 'ub', 'step'), show = 'headings')
        self.table.heading('name', text = 'Name')
        self.table.column('name', anchor = "center")
        self.table.heading('type', text = 'Type')
        self.table.column('type', anchor = "center")
        self.table.heading('lb', text = 'Lower bound')
        self.table.column('lb', anchor = "center")
        self.table.heading('ub', text = 'Upper bound')
        self.table.column('ub', anchor = "center")
        self.table.heading('step', text = 'Step size')
        self.table.column('step', anchor = "center")
        
        style = ttk.Style()
        style.configure("Treeview.Heading", font=(FONT, 14))
        style.configure("Treeview", font=(FONT, 12))   
        style.configure("Treeview.Heading", background= TABLE_HEADING_COLOR) 
       
       
        
        self.table.pack(fill = 'both', expand = True)
        self.table.bind("<ButtonRelease-1>", self.start_editing)
        self.table.bind("<Button-3>", self.on_right_click)
        self.table.bind("<FocusOut>", self.on_focus_out)
        

        
        self.context_menu = tk.Menu(self, tearoff=0)
        self.context_menu.add_command(label="Delete", command=self.delete_selected_item)
    
    def on_focus_out(self,event):
        self.table.selection_remove(self.table.selection())
    
    def switchEditMode(self):
        if(self.switch_var.get() == "on"):
            self.editMode = True
            self.table.selection_remove(self.table.selection())
        else:
            self.editMode = False
    
    def get_number_of_entries(self):
        items = self.table.get_children()
        return len(items)    
            
    def insertNewEntriesInToTable(self, howmany):
        for i in range(howmany):
            name = "x"+str(self.numberOfInsertedEntries+1)
            type = "continuous"
            lb = "0.0"
            ub = "1.0"
            stepSize = "0.0"
            data = (name, type, lb, ub,stepSize)
            self.table.insert(parent = '', index = tk.END, values = data)
            self.numberOfInsertedEntries = self.numberOfInsertedEntries+1 
            
#            self.settings.parameters.append(SingleParameter(name,type, float(lb), float(ub), stepSize))
        self.iterate_treeview()
            
    def remove_entries_from_end(self, howmany):
        items = self.table.get_children()
        for _ in range(howmany):
            if items:
                last_item = items[-1]
                self.table.delete(last_item)
                items = self.table.get_children()        

    def iterate_treeview(self):
    # Iterate through all items in the treeview
        for item_id in self.table.get_children():
        # Access each column value of the item
            values = self.table.item(item_id, 'values')
            print(values)  # Do something with the values            


    def updateProblemDimension(self,*args):
        if(self.dimension.get()):
            try:
                val = int(self.dimension.get())
                self.settings.dimension = val
                self.settings.print()
                
                if(val > self.get_number_of_entries()):
                    howmany = val - self.get_number_of_entries()
                    self.insertNewEntriesInToTable(howmany)
                if(val < self.get_number_of_entries()):
                    howmany = self.get_number_of_entries() - val
                    self.remove_entries_from_end(howmany)
            
            except ValueError:
                self.dimension.set(self.get_number_of_entries())
                messagebox.showerror("Error", "The dimension must be an integer")
        else:
            self.remove_entries_from_end(self.get_number_of_entries())
            self.numberOfInsertedEntries = 0         
                       
    def start_editing(self, event):
        
        
        if(self.editMode):
            item = self.table.selection()
            if item:
                column = self.table.identify_column(event.x)
        
                if column:
                    col_id = column.split("#")[-1]
               
                    self.edit_cell(item, int(col_id)-1)
            self.table.selection_remove(item)
                
    def edit_cell(self, item, col_id):
        # Get the item coordinates
        x, y, _, _ = self.table.bbox(item, col_id)
        
        # Get the current value
        current_value = self.table.item(item, "values")[col_id]

        # Create an Entry widget in the treeview
        entry = tk.Entry(self.table, justify="center")
        entry.insert(0, current_value)
        entry.place(x=x, y=y, width=self.table.column(col_id, "width"))
        
        # Bind the Entry widget to handle events
        entry.bind("<Return>", lambda event: self.finish_editing(item, col_id, entry))
        entry.bind("<FocusOut>", lambda event: self.finish_editing(item, col_id, entry))
        entry.focus_set()

    def finish_editing(self, item, col_id, entry):
        new_value = entry.get()
        self.table.set(item, col_id, new_value)
        entry.destroy()
        
        
        
        
            
    def delete_selected_item(self):
        selected_items = self.table.selection()
        for item in selected_items:
            self.table.delete(item)

    def on_right_click(self,event):
        item = self.table.identify_row(event.y)
        self.table.selection_set(item)
        self.context_menu.post(event.x_root, event.y_root)     

    def printParameters(self):
        print("Here")
        self.settings.printParameters()

  