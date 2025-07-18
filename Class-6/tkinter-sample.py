# https://docs.python.org/3/library/tkinter.html
# This should be installed as part of the Python installation
# test python -m tkinter
# https://tkdocs.com/tutorial/install.html
import tkinter as tk

# Create the main window
window = tk.Tk()
window.title("Simple Tkinter Example")

# Add a label widget
label = tk.Label(window, text="Hello, Tkinter!")
label.pack()

# Add a button widget
button = tk.Button(window, text="Click Me")
button.pack()

# Run the main event loop
window.mainloop()