from tkinter import *
from PIL import Image, ImageTk
import cv2


window = Tk()
window.title("Application Interface")
window.geometry('800x600')
lbl1 = Label(text= 'Voila-Jones Based Face Detector System') 
lbl1.grid(column=0, row=0)

#Graphics window
imageFrame = Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

#load = Image.open("baby.jpg")
#render = ImageTk.PhotoImage(load)
# labels can be text or images
#img = Label(image=render)
#img.image = render
#img.place(x=500, y=0)

cap = cv2.VideoCapture(0)
def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    display1.imgtk = imgtk #Shows frame for display 1
    display1.configure(image=imgtk)
    window.after(10, show_frame)   

display1 = Label(imageFrame)
display1.grid(row=1, column=0, padx=10, pady=2)  #Display 1

show_frame()
window.mainloop()