from tkinter import *
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
from PIL import Image, ImageTk
import requests
from io import BytesIO
import RCNN
import cv2

# import our model.py which only takes one image and produce a detection result

LARGE_FONT= ("Verdana", 12)
NORM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)

class LoadImage:
    def __init__(self, master):
        # initialize
        self.master = master
        self.frame = tk.Frame(self.master)
        # title and size
        self.master.wm_title("Food Detection GUI")
        self.master.geometry("600x600")
        # add background image (need to be added first, so it has a lower stacking order)
        imgurl = "https://cdn-prod.medicalnewstoday.com/content/images/articles/324/324956/close-up-of-a-plate-of-food.jpg"
        response = requests.get(imgurl)
        imgdata = response.content
        back_img= ImageTk.PhotoImage(Image.open(BytesIO(imgdata)))
        background_label = tk.Label(self.master, image=back_img)
        background_label.image = back_img
        background_label.pack()
        # load button
        LoadButton = tk.Button(self.master, text='Click to load food image', width=20, command=self.loadIMG, borderwidth=0)
        LoadButton.pack()
        LoadButton.place(x=230, y=400)
        # add instruction text
        instruction = Label(self.master, text="Hello there! We are pleased that you chose our app.\n The instruction is the following:\n 1. Click the button below and choose one image that contains various kinds of food. \n(If you chose a file with wrong format, you will be asked to reselect) \n 2. When you finished the selection by clicking 'open' in the popup file explorer, \nplease give us a few second before the result comes up \n 3. A new window will show up contains the detection results, \nand you can click the button at the bottom of the window to close this window. \n\n Note: The entire procedure is repeatable.", font=NORM_FONT)
        instruction.pack()
        instruction.place(x=25, y=100, height=250, width=550)
        # pack the frame
        self.frame.pack()

    def loadIMG(self):
        try:
            filename = tk.filedialog.askopenfilename()
            # im = Image.open(filename)
            cv2_im = cv2.imread(filename)
            test_img = 'C:/Users/skyho/Desktop/test_image_folder/0/test.png'
            cv2.imwrite(test_img, cv2_im)
            # feed this image file to our model for detection
            self.new_window = tk.Toplevel(self.master)
            self.app = Detection(self.new_window, test_img)
        except IOError:
            self.popupmsg("Selected File is not an image file, please select again")

    def popupmsg(self, msg):
        popup = tk.Tk()
        popup.wm_title("!")
        label = ttk.Label(popup, text=msg, font=NORM_FONT)
        label.pack(side="top", fill="x", pady=10)
        B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
        B1.pack()
        popup.mainloop()


class Detection:
    def __init__(self, master, test_img):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.frame.pack(fill=BOTH, expand=1)
        self.master.wm_title("Detection Result")

        detection = RCNN.get_result_from_model(test_img, 0.16)
        load = Image.open(detection)
        width, height = load.size
        new_width = width + 50
        new_height = height + 50
        #self.master.geometry("600x600")
        if width >1500 or height > 650:
            new_width = 1000
            new_height = int(new_width * height / width)
            load = load.resize((new_width, new_height), Image.ANTIALIAS)

        self.master.minsize(new_width, new_height)
        self.master.maxsize(new_width, new_height)
        render = ImageTk.PhotoImage(load)
        img = Label(self.frame, image=render)
        img.image = render
        img.place(x=25, y=25)

        self.quitButton = tk.Button(self.frame, text='Close this window', width=200, command=self.close_window)
        self.quitButton.pack(side=BOTTOM)
        self.frame.pack()

    def close_window(self):
        self.master.destroy()


def main():
    root = tk.Tk()
    app = LoadImage(root)
    root.mainloop()


if __name__ == '__main__':
    main()
