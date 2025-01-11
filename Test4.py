import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image, ImageTk
import sys

# Function to get the path to the bundled file in the PyInstaller executable
def resource_path(relative_path):
    """Get the absolute path to a resource. Works for PyInstaller."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# Load the trained model
model_path = resource_path('tb_mobilenetv2_model.h5')
model = tf.keras.models.load_model(model_path)

# Function to predict TB and draw bounding box
def predict_tb_with_model(image_path):
    try:
        # Load the image and preprocess
        img = tf.keras.utils.load_img(image_path, target_size=(256, 256))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Model prediction (Assuming it outputs [class_probabilities, bounding_box])
        prediction = model.predict(img_array)
        class_probs, bbox_coords = prediction[0][:2], prediction[0][2:]
        predicted_class = np.argmax(class_probs)
        confidence = class_probs[predicted_class]
        classes = ['No TB', 'TB']

        # Display classification result
        result_text = f"Prediction: {classes[predicted_class]} (Confidence: {confidence:.2f})"

        # Load the original image for visualization
        img_orig = cv2.imread(image_path)
        img_orig = cv2.resize(img_orig, (256, 256))

        # If TB is detected, draw the bounding box
        if predicted_class == 1:  # Assuming "TB" is class 1
            # Convert normalized bounding box coordinates (if required) to pixel values
            x_min, y_min, x_max, y_max = bbox_coords * 256
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

            # Draw the bounding box
            color = (0, 255, 0)  # Green color
            thickness = 2
            img_orig = cv2.rectangle(img_orig, (x_min, y_min), (x_max, y_max), color, thickness)
            text_position = (x_min, y_min - 10)
            cv2.putText(img_orig, result_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Convert BGR to RGB for tkinter display
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        return img_orig, result_text

    except Exception as e:
        messagebox.showerror("Error", f"Error processing the image: {str(e)}")
        return None, None

# Function to handle image upload and display
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        # Run the prediction
        img_result, result_text = predict_tb_with_model(file_path)

        if img_result is not None:
            # Convert the image to a format tkinter can display
            img_result = Image.fromarray(img_result)
            img_result = ImageTk.PhotoImage(img_result)

            # Display the image
            image_label.config(image=img_result)
            image_label.image = img_result

            # Display the result text
            result_label.config(text=result_text)

# Function to display content for the selected tab
def display_tab_content(tab_name):
    # Clear the main_frame
    for widget in main_frame.winfo_children():
        widget.destroy()

    if tab_name == "Home":
        # Home content
        instruction_label = tk.Label(
            main_frame, text="Upload an Image to Detect Tuberculosis", font=("Helvetica", 18, "bold"), bg="#fafafa", fg="#000000"
        )
        instruction_label.pack(pady=60)

        # Upload button
        upload_button = tk.Button(
            main_frame, text="Upload Image", command=upload_and_predict, font=("Helvetica", 16, "bold"),
            bg="#82589F", fg="#FFFFFF", activebackground="#6C3483", activeforeground="#FFFFFF"
        )
        upload_button.pack(pady=10)

        # Image display area
        global image_label
        image_label = tk.Label(main_frame, bg="#fafafa")
        image_label.pack(pady=10)

        # Result display
        global result_label
        result_label = tk.Label(main_frame, text="", font=("Helvetica", 16), bg="#fafafa", fg="#000000")
        result_label.pack(pady=20)

    elif tab_name == "About Us":
        # About content
        about_label = tk.Label(
            main_frame, text="About TB Detection Application", font=("Helvetica", 18, "bold"), bg="#fafafa", fg="#000000"
        )
        about_label.pack(pady=60)

        info_label = tk.Label(
            main_frame, text=(
                "This application uses a machine learning model to detect Tuberculosis (TB) from X-ray images. "
                "It identifies TB-affected regions and highlights them with bounding boxes."
            ),
            font=("Helvetica", 14), bg="#fafafa", fg="#000000", wraplength=600, justify="left"
        )
        info_label.pack(pady=10)

    elif tab_name == "Contact Us":
        # About content
        about_label = tk.Label(
            main_frame, text="Contact Details", font=("Helvetica", 18, "bold"), bg="#fafafa", fg="#000000"
        )
        about_label.pack(pady=60)

        info_label = tk.Label(
            main_frame, text=(
                "Contact Us Page"
            ),
            font=("Helvetica", 14), bg="#fafafa", fg="#000000", wraplength=600, justify="left"
        )
        info_label.pack(pady=10)

# Create the main tkinter window
root = tk.Tk()
root.title("TB Detection Application")

# Make the window fullscreen
root.state("zoomed")

# Set dark theme background
root.configure(bg="#000000")

# Header
header = tk.Label(root, text="TB Detection Application", font=("Helvetica", 24, "bold"), bg="#000000", fg="#FFFFFF")
header.pack(pady=20)

# Sidebar menu
sidebar = tk.Frame(root, bg="#1B1B2F", width=100, height=root.winfo_height())
sidebar.pack(side="left", fill="y")

menu_items = ["Home", "About Us", "Contact Us"]
menu_buttons = []

for item in menu_items:
    btn = tk.Button(
        sidebar, text=item, font=("Helvetica", 14, "bold"), bg="#000000", fg="#FFFFFF",
        activebackground="#3A3D5A", activeforeground="#FFFFFF", relief="flat", padx=20, pady=10,
        command=lambda name=item: display_tab_content(name)
    )
    btn.pack(fill="x", pady=5)
    menu_buttons.append(btn)

# Main content
main_frame = tk.Frame(root, bg="#fafafa")
main_frame.pack(side="right", fill="both", expand=True)

# Display initial content for the "Home" tab
display_tab_content("Home")

root.mainloop()
