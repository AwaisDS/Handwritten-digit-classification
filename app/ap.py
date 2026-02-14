import streamlit as st
import numpy as np
import joblib
from streamlit_drawable_canvas import st_canvas
from skimage.transform import resize

# Load model and scaler
# Make sure these paths are 100% correct on your machine
model = joblib.load("C:\\Users\\pc\\Desktop\\Handwritten-digit-classification\\models\\svm_model.pkl")
scaler = joblib.load("C:\\Users\\pc\\Desktop\\Handwritten-digit-classification\\models\\scaler.pkl")

st.title("Handwritten Digit Recognition (SVM) ⚽")
st.write("Draw a digit (0–9) below and click Predict")

# Canvas setup
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=15,
    stroke_color="#FFFFFF", # White ink
    background_color="#000000", # Black paper
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)

# Only process if there is drawing data
if canvas_result.image_data is not None:
    # 1. Get the Alpha or Red channel (where the drawing is)
    img = canvas_result.image_data[:, :, 0] 

    # 2. Check if the user has actually drawn something (avoid empty array error)
    if np.any(img > 0): 
        # 3. Crop to the digit (Bounding Box)
        coords = np.column_stack(np.where(img > 0))
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        
        # Add padding so the digit isn't cut off too tightly
        img_cropped = img[max(0, y0-20):min(200, y1+20), max(0, x0-20):min(200, x1+20)]

        # 4. Resize to 8x8 (Match sklearn digits dataset)
        img_8x8 = resize(img_cropped, (8, 8), anti_aliasing=True, preserve_range=True)

        # 5. Scale to 0–16 range (The 'Maestro' touch for SVM)
        img_normalized = (img_8x8 / 255.0) * 16
        
        # 6. Show the "Model's View" (Debugging tool)
        st.write("What the model sees (8x8):")
        st.image(img_8x8 / 255.0, width=100) # Scaled for display

        # 7. Flatten and Predict
        input_array = img_normalized.flatten().reshape(1, -1)
        
        try:
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)
            st.header(f"Result: {prediction[0]}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.info("Draw something on the pitch to start!")