import streamlit as st
import numpy as np
import joblib
from streamlit_drawable_canvas import st_canvas
from skimage.transform import resize

# Load model and scaler
model = joblib.load("C:\\Users\\pc\\Desktop\\Handwritten-digit-classification\\models\\svm_model.pkl")
scaler = joblib.load("C:\\Users\\pc\\Desktop\\Handwritten-digit-classification\\models\\scaler.pkl")

st.title("Handwritten Digit Recognition (SVM) 🎨")
st.write("Draw a digit (0–9) below and click Predict")

# Canvas setup
canvas_result = st_canvas(
    fill_color="#000000",        # background
    stroke_width=15,
    stroke_color="#FFFFFF",      # white digit
    background_color="#000000",  # black canvas
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # 1️⃣ Get the image and convert to grayscale
    # Use the RGB channels. If you draw white on black, 
    # the values are already correct (high for digit, low for background).
    img = canvas_result.image_data[:, :, 0] 

    # 2️⃣ SKIP THE INVERSION (unless you draw black on white)
    # If drawing white stroke on black background: 
    # White = 255, Black = 0. This is what we want.
    # img = 255 - img  <-- REMOVE OR COMMENT THIS OUT

    # 3️⃣ Crop the digit
    coords = np.column_stack(np.where(img > 50)) 
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        # Add a little padding so the digit isn't touching the very edge
        img = img[max(0, y0-10):min(200, y1+10), max(0, x0-10):min(200, x1+10)]
    
    # 4️⃣ Resize to 8x8
    # Ensure we use preserve_range to keep values between 0-255 before scaling
    #img_resized = resize(img, (8, 8), anti_aliasing=True, preserve_range=True)
# Add padding so the digit isn't cut off too tightly
    img_cropped = img[max(0, y0-20):min(200, y1+20), max(0, x0-20):min(200, x1+20)]
# 4. Resize to 8x8 (Match sklearn digits dataset)
    img_8x8 = resize(img_cropped, (8, 8), anti_aliasing=True, preserve_range=True)
    # 5. Scale for the model (0–16)
    input_for_model = (img_8x8 / 255.0) * 16
        
        # 6. Show the preview safely (0.0 - 1.0)
    st.write("Model's input view:")
    preview = img_8x8 / 255.0
    st.image(np.clip(preview, 0, 1), width=100) # clip ensures no RuntimeError

        # 7. Flatten and Predict
    input_array = input_for_model.flatten().reshape(1, -1)
        
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Digit: {prediction[0]}")

st.image(img_8x8 / 16, width=100)