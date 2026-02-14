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
    # 1️⃣ Take the grayscale channel (first channel)
    img = canvas_result.image_data[:, :, 0]

    # 2️⃣ Invert colors: make digit white on black background
    img = 255 - img

    # 3️⃣ Resize to 8x8 like sklearn digits dataset
    img_resized = resize(img, (8, 8), anti_aliasing=True)

    # 4️⃣ Scale to 0-16 (digits dataset scale)
    img_resized = np.round((img_resized / 255.0) * 16)

    # 5️⃣ Flatten to 1D array
    input_array = img_resized.flatten().reshape(1, -1)

    # 6️⃣ Scale using saved scaler
    input_scaled = scaler.transform(input_array)

    # 7️⃣ Predict
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Digit: {prediction[0]}")
