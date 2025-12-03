import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from keras.models import load_model

# Load your trained model
model = load_model("emotion_model.hdf5", compile=False)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# GUI Window
root = tk.Tk()
root.title("Live Emotion Recognition System")
root.geometry("900x600")
root.configure(bg="#1e1e1e")

# Header
header = Label(root, text="Live Facial Emotion Detection", font=("Arial", 22, "bold"),
               fg="white", bg="#1e1e1e")
header.pack(pady=10)

# Video Display Area
video_label = Label(root)
video_label.pack()

# Status Label
status_label = Label(root, text="Status: Camera Off", font=("Arial", 14),
                     fg="yellow", bg="#1e1e1e")
status_label.pack(pady=10)

# Webcam Reference
cap = None
running = False

# Start Camera
def start_camera():
    global cap, running
    running = True
    cap = cv2.VideoCapture(0)
    status_label.config(text="Status: Camera Running", fg="lime")
    update_frame()

# Stop Camera
def stop_camera():
    global cap, running
    running = False
    if cap:
        cap.release()
        status_label.config(text="Status: Camera Stopped", fg="orange")

# Exit App
def exit_app():
    if cap:
        cap.release()
    root.destroy()

# Update Video Frames
def update_frame():
    if running:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = np.reshape(roi, (1, 64, 64, 1))

                preds = model.predict(roi, verbose=0)[0]
                label = emotion_labels[np.argmax(preds)]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

        video_label.after(10, update_frame)

# Button Panel
button_frame = tk.Frame(root, bg="#1e1e1e")
button_frame.pack(pady=20)

start_btn = Button(button_frame, text="Start Camera", font=("Arial", 12, "bold"),
                   bg="green", fg="white", width=15, command=start_camera)
start_btn.grid(row=0, column=0, padx=10)

stop_btn = Button(button_frame, text="Stop Camera", font=("Arial", 12, "bold"),
                  bg="orange", fg="white", width=15, command=stop_camera)
stop_btn.grid(row=0, column=1, padx=10)

exit_btn = Button(button_frame, text="Exit", font=("Arial", 12, "bold"),
                  bg="red", fg="white", width=15, command=exit_app)
exit_btn.grid(row=0, column=2, padx=10)

root.mainloop()
