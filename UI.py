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

def main():
    root = tk.Tk()
    GUI = Home(root)
    root.mainloop()

if __name__ == '__main__':
    main()