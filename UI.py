import tkinter as tk
import cv2
from PIL import Image, ImageTk
import Process as p

class Home:
    def __init__(self, master):
        self.master = master
        self.master.title('Face Training and Recognition')
        self.master.resizable(False, False)

        # Gets the requested values of the height and widht.
        windowWidth = 800
        windowHeight = 400
 
        # Gets both half the screen width/height and window width/height
        positionRight = int(self.master.winfo_screenwidth()/2 - windowWidth/2)
        positionDown = int(self.master.winfo_screenheight()/2 - windowHeight/2)
 
        # Positions the window in the center of the page.
        self.master.geometry("{}x{}+{}+{}".format(windowWidth, windowHeight, positionRight, positionDown))

        self.master.config(background="#FFFFFF")

        self.title = tk.Label(self.master, justify=tk.CENTER, text="Welcome to Face Training and Recognition", bg="white", font = "Times 30 bold")
        self.title.pack(fill=tk.X, padx=10, pady=20)

        self.train_bt = tk.Button(self.master, text ="Train Face", height=2, width=15, bg="gray", fg="white", font = "Times 15 bold", command = self.redirect_train)
        self.train_bt.pack(fill=tk.X,padx=50, pady=40)

        self.recognition_bt = tk.Button(self.master, text ="Recognition", height=2, width=15, bg="gray", fg="white", font = "Times 15 bold", command = self.redirect_recognition)
        self.recognition_bt.pack(fill=tk.X,padx=50, pady=10)

    def redirect_train(self):
        self.master.destroy()
        root = tk.Tk() 
        GUI = Train(root)

    def redirect_recognition(self):
        self.master.destroy()
        root = tk.Tk() 
        GUI = Recognition(root)

class Train:
    def __init__(self, master):
        self.master = master
        self.master.title('Face Training')
        self.master.resizable(False, False)

        # Gets the requested values of the height and widht.
        windowWidth = 1000
        windowHeight = 560
 
        # Gets both half the screen width/height and window width/height
        positionRight = int(self.master.winfo_screenwidth()/2 - windowWidth/2)
        positionDown = int(self.master.winfo_screenheight()/2 - windowHeight/2)
 
        # Positions the window in the center of the page.
        self.master.geometry("{}x{}+{}+{}".format(windowWidth, windowHeight, positionRight, positionDown))

        self.master.config(background="#FFFFFF")

        self.title = tk.Label(self.master, justify=tk.CENTER, text="Training Your Face", bg="white", font = "Times 30 bold")
        self.title.pack(fill=tk.X, padx=10, pady=10)

        self.imageFrame = tk.Frame(self.master, width=600, height=500)
        self.imageFrame.pack(side=tk.LEFT, padx=10, pady=20)

        self.entryFrame = tk.Frame(self.master, bg="white")
        self.entryFrame.pack(pady=50)

        self.label_title = tk.Label(self.entryFrame, justify=tk.LEFT, text="Enter your name:", bg="white", font = "Times 15")
        self.label_title.pack(fill=tk.X, pady=10)

        self.label = tk.Text(self.entryFrame, height=1, width=30, font = "Times 15")
        self.label.pack(fill=tk.X, pady=10)

        self.result = tk.Label(self.entryFrame, justify=tk.LEFT, bg="white", font = "Times 15")
        self.result.pack(fill=tk.X, pady=10)

        self.train_bt = tk.Button(self.master, text ="Train", height=1, width=10, bg="gray", fg="white", font = "Times 15 bold", command = self.train)
        self.train_bt.pack(pady=10)

        self.back_bt = tk.Button(self.master, text ="Back", height=1, width=10, bg="gray", fg="white", font = "Times 15 bold", command = self.redirect_main)
        self.back_bt.pack(pady=10)

        self.video_capture = cv2.VideoCapture(0)

        self.display = tk.Label(self.imageFrame)
        self.display.pack()

        self.isTrain = False
        self.count = 0
        self.faceCascade = p.train_init()
        self.show_frame()

    def train(self):
        name = self.label.get("1.0",'end-1c')
        self.count = 0

        if name == '':
            self.result['text'] = 'Enter your name...'
            self.result['fg'] = 'red'
            self.isTrain = False

        else:
            self.isTrain = True
            self.result['fg'] = 'black'
            self.label['state'] = 'disabled'
            self.train_bt['state'] = 'disabled'

    def redirect_main(self):
        self.video_capture.release()
        self.master.destroy()
        root = tk.Tk() 
        GUI = Home(root)

    def show_frame(self):
        _, frame = self.video_capture.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        coords, _img = p.detect_face(image, self.faceCascade)

        if self.isTrain == True:
            if self.count < 20:
                self.count = p.process_level(_img, coords, self.label.get("1.0",'end-1c'), self.count)
                self.result['text'] = 'Processing...' + str(self.count * 5) + '%'
            else:
                p.train_classifier()
                self.result['fg'] = 'green'
                self.result['text'] = 'Success!'
                self.label['state'] = 'normal'
                self.label.delete(1.0, tk.END)
                self.train_bt['state'] = 'normal'
                self.isTrain = False

        img = Image.fromarray(_img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.display.imgtk = imgtk
        self.display.configure(image=imgtk)
        self.master.after(100, self.show_frame)

def main():
    root = tk.Tk()
    GUI = Home(root)
    root.mainloop()

if __name__ == '__main__':
    main()