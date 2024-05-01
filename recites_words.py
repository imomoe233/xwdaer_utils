from tkinter import *

def quit():
    global root
    root.quit()

root = Tk()
Button(root, text="Quit", command=quit).pack()

root.mainloop()