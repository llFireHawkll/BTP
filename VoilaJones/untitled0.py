from PIL import Image, ImageTk
import tkinter as tk
import argparse
import datetime
import cv2
import os
import detectFaces as df

imagepath = []

class Application:
    def __init__(self, output_path = "C:/Users/Sparsh-PC/Desktop/New folder/MY_VJ/Input/"):
        """ Initialize application which uses OpenCV + Tkinter. It displays
            a video stream in a Tkinter window and stores current snapshot on disk """
        
        self.vs = cv2.VideoCapture(0) # capture video frames, 0 is your default video camera
        self.output_path = output_path  # store output path
        self.current_image = None  # current image from the camera

        self.root = tk.Tk()  # initialize root window
        self.root.title("Face Detection Application Interface")  # set window title
        self.root.geometry('1100x420')
        lbl = tk.Label(text= 'Voila-Jones Based Face Detector System',font = "Helvetica 16 bold") 
        lbl.grid(row =0 , column=2)
        # self.destructor function gets fired when the window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel = tk.Label(self.root, width=320, height=320)  # initialize image panel
        self.panel.grid(row=1 , column=0)

        self.panel1 = tk.Label(self.root)  # initialize image panel
        self.panel1.grid(row=1 , column=3)

        # create a button, that when pressed, will take the current frame and save it to file
        btn = tk.Button(self.root, text="Take Snapshot", command=self.take_snapshot, width=25, height=2, font = "Helvetica")
        btn.grid(row=3 , column=0)
        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame
        
        btn = tk.Button(self.root, text="Detect Face", command=self.detect_face, width=25, height=2, font = "Helvetica")
        btn.grid(row=3 , column=3)
        
        self.video_loop()

    def video_loop(self):
        """ Get frame from the video stream and show it in Tkinter """
        ok, frame = self.vs.read()  # read frame from video stream
        if ok:  # frame captured without any errors
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
            imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            self.panel.config(image=imgtk)  # show the image
        self.root.after(30, self.video_loop)  # call the same function after 30 milliseconds

    def detect_face(self):
        path = imagepath[0]
        img = Image.open(path)
        new_width  = 400
        new_height = 300
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img.save(path)
        df.detectFaces(path)
        img1 = ImageTk.PhotoImage(Image.open("C:/Users/Sparsh-PC/Desktop/New folder/MY_VJ/Output/test.png"))
        self.panel1.img1 = img1
        self.panel1.config(image=img1)
        
    def take_snapshot(self):
        """ Take snapshot and save it to the file """
        ts = datetime.datetime.now() # grab the current timestamp
        filename = "{}.png".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))  # construct filename
        p = os.path.join(self.output_path, filename)  # construct output path
        imagepath.append(p)
        self.current_image.save(p, "PNG")  # save image as jpeg file
        print("[INFO] saved {}".format(filename))

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="C:/Users/Sparsh-PC/Desktop/New folder/MY_VJ/Input/",
    help="path to output directory to store snapshots (default: current folder")
args = vars(ap.parse_args())

# start the app
print("[INFO] starting...")
pba = Application(args["output"])
pba.root.mainloop()