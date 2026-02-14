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
    # 1️⃣ Take grayscale channel
    img = canvas_result.image_data[:, :, 0]  # shape (200,200)

    # 2️⃣ Invert colors: white digit on black background
    img = 255 - img

    # 3️⃣ Crop the digit to its bounding box to remove extra black borders
    coords = np.column_stack(np.where(img > 50))  # pixels > threshold
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        img = img[y0:y1+1, x0:x1+1]
    else:
        img = img  # blank canvas

    # 4️⃣ Resize to 8x8 like sklearn digits dataset
    img_resized = resize(img, (8, 8), anti_aliasing=True)

    # 5️⃣ Scale to 0–16 (digits dataset scale)
    img_resized = np.round((img_resized / 255.0) * 16)

    # 6️⃣ Flatten to 1D array
    input_array = img_resized.flatten().reshape(1, -1)

    # 7️⃣ Scale using saved scaler
    input_scaled = scaler.transform(input_array)

    # 8️⃣ Predict
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Digit: {prediction[0]}")

