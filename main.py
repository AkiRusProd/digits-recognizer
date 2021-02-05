from PIL import ImageTk, Image, ImageDraw, ImageOps
import PIL
import numpy as np
from tkinter import *

width = 168
height = 168

white = (255, 255, 255)




def recognize():
    inverted_img = ImageOps.invert(image)
    
    grayscaled_img= inverted_img.convert('L') 
    resized_img=grayscaled_img.resize((28,28),PIL.Image.ANTIALIAS)

    data = np.asarray(resized_img)
  
    inputs=np.reshape(data,(1,784)).T
  

    with open('weights/fst_w.csv','r') as f:
        fst_w= np.loadtxt(f, delimiter=",")
    with open('weights/snd_w.csv','r') as f:
        snd_w= np.loadtxt(f, delimiter=",")


    x1=np.dot(fst_w, inputs)
    y1=1/(1+np.exp(-x1))
    
    x2=np.dot(snd_w, y1)
    y2=1/(1+np.exp(-x2))

    
    probability=round( ((np.max(y2)/np.sum(y2))*100), 2)

    print(np.argmax(y2),' ',probability,'%')

    lbl1['text']=np.argmax(y2),
    lbl2['text']='Probability:',probability,'%'




def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=penSize_slider.get())
    draw.line([x1, y1, x2, y2],fill="black",width=penSize_slider.get())



def clear():
    cv.delete("all")
    draw.rectangle((0, 0, 168, 168), fill=(255, 255, 255, 255))


    
root = Tk()
root.title("Digit Recognizer") 


cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()


image = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image)



cv.pack(side=RIGHT)
cv.bind("<B1-Motion>", paint)


button=Button(text="Recognize",command=recognize,width=20)
button2=Button(text="Clear",command=clear, width = 20)
lbl0=Label(text="Pen Size",font="Arial 10",width=15)
lbl1=Label(text=" ",font="Arial 30",fg="red")
lbl2=Label(text=" ",font="Arial 12",width=15)

lbl0.pack()

penSize_slider = Scale(from_= 1, to = 10,orient=HORIZONTAL)
penSize_slider.pack()

button.pack()
button2.pack()

lbl1.pack()
lbl2.pack()

root.minsize(350, 200)
root.maxsize(350, 200)

root.mainloop()
