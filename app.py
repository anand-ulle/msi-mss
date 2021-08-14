from tkinter import * 
import numpy as np

    
# loading Python Imaging Library
from PIL import ImageTk, Image  
# To get the dialog box to open when required 
from tkinter import filedialog
from fastai.learner import load_learner
from fastai.vision.core import PILImage

learn_inf = load_learner('./model/export.pkl')
learn_inf.dls.vocab

# Create a window
root = Tk()
frame = Frame(root)
frame.pack()

# Set Title as Image Loader
root.title("Gatro Classifier App")

# Set the resolution of window
root.geometry('670x350')

frame = LabelFrame(
    root,
    text='Prediction of MicroSatellite Instability from WSI',
    bg='#f0f0f0', font='san 12 bold', foreground="blue"
    
)
frame.pack(expand=True, fill=BOTH)
#global Labele
#Label(root, text = "The Probability: ",font='roman 12 bold',foreground="blue").grid(row=12)
#Label(root, text = "The Predicted class: ",font='roman 12 bold',foreground="blue").grid(row=13)
# Allow Window to be resizable
root.resizable(width = True, height = True)
#blank1 = Entry(root)
#blank2 = Entry(root)
#blank1.grid(row=12, column=1)
#blank2.grid(row=13, column=1)
#blank2 = Entry(root).grid(row=13, column=1)

def open_img():
    # Select the Imagename  from a folder 
    global x 
    x = openfilename()
    
    
    # opens the image
    img = Image.open(x)    
        
    # resize the image and apply a high-quality down sampling filter
    img = img.resize((250, 250), Image.ANTIALIAS)
    
    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)
    # create a label
    panel = Label(frame, image = img)
      
    # set the image as img 
    panel.image = img
    panel.grid(row = 1,column=2)
    
def reset():
    blank1.delete('0',END)
    blank2.delete('0',END)   
    
def predict():
    global blank1
    global blank2
    blank1 = Entry(frame)
    blank2 = Entry(frame)
    blank1.grid(row=2, column=2)
    blank2.grid(row=3, column=2)
    
    Label(frame,text = 'Probs:', font='san 12 bold', foreground="blue").grid(row=2,column=1)
    Label(frame,text = 'class:',font='san 12 bold', foreground="blue").grid(row=3,column=1)
     
    
    im = PILImage.create(x)   
    pred,pred_idx,probs = learn_inf.predict(im)
    np_probs = probs.cpu().detach().numpy()
    #class_lbl = pred.cpu().detach().numpy()    
    probability = np_probs[pred_idx]
    blank1.insert(0, probability)
    blank2.insert(0, pred)
    #Entry(root,  text = "%s" %(probability) ).grid(row=2, column=1)
    #print(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')

   
def openfilename():
  
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(title ='"pen')
    return filename
    
# Create a button and place it into the window using grid layout
imopen = Button(frame, text ='open_img', command = open_img).grid(row = 4,column = 1)
predict = Button(frame,text = "predict_img", command = predict).grid(row =4,column = 2)
reset = Button(frame,text = "reset_val", command = reset).grid(row =4,column = 3)


root.mainloop()





