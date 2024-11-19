import tkinter as tk
from PIL import Image, ImageTk
import cv2
import argparse
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Anti-spoofing functions and variables
SAMPLE_IMAGE_PATH = "./images/sample/"
device_id = 0  # GPU ID
model_dir = "./resources/anti_spoof_models"
model_test = AntiSpoofPredict(device_id)
image_cropper = CropImage()

def check_image(image):
    height, width, channel = image.shape
    if width / height != 3 / 4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True

def test_frame(frame):
    """Modified test function to handle real-time frames."""
    image = cv2.resize(frame, (int(frame.shape[0] * 3 / 4), frame.shape[0]))
    result = check_image(image)
    if result is False:
        return "Invalid Frame"
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))

    label = np.argmax(prediction)
    return "Real Face Detected" if label == 1 else "Spoofed"

# Tkinter Video Interface
class VideoCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Anti-Spoofing Detection")

        # Canvas for video feed
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        # Label for displaying prediction
        self.result_label = tk.Label(root, text="Prediction: ", font=("Arial", 14))
        self.result_label.pack()

        # Quit button
        self.quit_button = tk.Button(root, text="Quit", command=self.quit_app)
        self.quit_button.pack()

        # Video capture
        self.cap = cv2.VideoCapture(0)

        # Process frames
        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Process the frame for anti-spoofing
            prediction = test_frame(frame)
            self.result_label.config(text=f"Prediction: {prediction}")

            # Convert BGR to RGB for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.imgtk = imgtk  # Prevent garbage collection

        # Call update_frame again after 10ms
        self.root.after(10, self.update_frame)

    def quit_app(self):
        self.cap.release()
        self.root.destroy()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoCaptureApp(root)
    root.mainloop()
