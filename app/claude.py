import streamlit as st
import numpy as np
import pickle
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import cv2

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

* { font-family: 'DM Sans', sans-serif; }

html, body, [class*="css"] {
    background-color: #0d0d0d;
    color: #f0f0f0;
}
.stApp { background: #0d0d0d; }

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.title-block {
    text-align: center;
    padding: 2rem 0 1.5rem;
}
.title-block h1 {
    font-size: 2.6rem;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #e0ff4f, #4fffb0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.title-block p {
    color: #555;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.pred-card {
    background: linear-gradient(135deg, #161616, #1e1e1e);
    border: 1px solid #2a2a2a;
    border-left: 4px solid #e0ff4f;
    border-radius: 14px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin: 0.5rem 0 1rem;
}
.pred-digit {
    font-family: 'Space Mono', monospace;
    font-size: 5.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #e0ff4f, #4fffb0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.0;
}
.pred-label {
    color: #666;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

.conf-container {
    background: #131313;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
}
.conf-title {
    color: #555;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}
.conf-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0.28rem 0;
}
.conf-lbl {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    width: 16px;
    text-align: right;
    flex-shrink: 0;
}
.conf-bg {
    flex: 1;
    background: #222;
    border-radius: 3px;
    height: 8px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 3px;
}
.conf-pct {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}

.hint {
    color: #444;
    font-size: 0.78rem;
    text-align: center;
    margin: 0.3rem 0 1rem;
    letter-spacing: 0.04em;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #e0ff4f, #4fffb0) !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.55rem 0 !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.err-box {
    background: #180f0f;
    border: 1px solid #4a1a1a;
    border-left: 4px solid #ff4f4f;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    color: #ff8a8a;
    font-size: 0.85rem;
    margin-top: 1rem;
}

.preview-label {
    color: #444;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    text-align: center;
    margin: 0.8rem 0 0.3rem;
}

#MainMenu, footer { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: #111 !important;
    border-right: 1px solid #1e1e1e;
}
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    try:
        import joblib

        def smart_load(path):
            try:
                return joblib.load(path)
            except Exception:
                with open(path, "rb") as f:
                    return pickle.load(f)

        model  = smart_load("C:\\Users\\pc\\Desktop\\Handwritten-digit-classification\\models\\svm_model.pkl")
        scaler = smart_load("C:\\Users\\pc\\Desktop\\Handwritten-digit-classification\\models\\scaler.pkl")
        return model, scaler, None

    except FileNotFoundError as e:
        return None, None, f"File not found: {e}"
    except Exception as e:
        return None, None, str(e)

# @st.cache_resource
# def load_model():
#     try:
#         with open("C:\\Users\\pc\\Desktop\\Handwritten-digit-classification\\models\\svm_model.pkl", "rb") as f:
#             model = pickle.load(f)
#         with open("C:\\Users\\pc\\Desktop\\Handwritten-digit-classification\\models\\scaler.pkl", "rb") as f:
#             scaler = pickle.load(f)
#         return model, scaler, None
#     except FileNotFoundError as e:
#         return None, None, str(e)


model, scaler, load_error = load_model()


# TEMPORARY DEBUG — paste after load_model() call, remove later

# Paste this temporarily in your app to see what training digits look like
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
fig, axes = plt.subplots(2, 10, figsize=(12, 3))
for i in range(10):
    # show first 2 examples of each digit
    idx1 = np.where(digits.target == i)[0][0]
    idx2 = np.where(digits.target == i)[0][1]
    axes[0, i].imshow(digits.images[idx1], cmap='gray')
    axes[0, i].set_title(str(i))
    axes[0, i].axis('off')
    axes[1, i].imshow(digits.images[idx2], cmap='gray')
    axes[1, i].axis('off')
plt.savefig("training_samples.png")
st.image("training_samples.png")

# ── Preprocessing ──────────────────────────────────────────────────────────
def preprocess_canvas(image_data):
    img = Image.fromarray(image_data.astype("uint8"), "RGBA").convert("L")
    arr = np.array(img, dtype=np.float64)

    # Crop to bounding box
    non_zero = np.where(arr > 10)
    if len(non_zero[0]) == 0:
        return np.zeros((1, 64)), np.zeros((8, 8))

    top, bottom = non_zero[0].min(), non_zero[0].max()
    left, right  = non_zero[1].min(), non_zero[1].max()
    arr = arr[top:bottom+1, left:right+1]

    # Make it square so digit isn't squished
    h, w = arr.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.float64)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = arr

    # Add padding
    pad = size // 5
    square = np.pad(square, pad, mode='constant', constant_values=0)

    # Resize to 8x8
    img_resized = Image.fromarray(square.astype(np.uint8)).resize((8, 8), Image.LANCZOS)
    arr8 = np.array(img_resized, dtype=np.float64)

    # ✅ Scale to 0-16 to match sklearn digits training data range
    if arr8.max() > 0:
        arr8 = arr8 / arr8.max() * 16.0

    return arr8.flatten().reshape(1, -1), arr8


def run_predict(flat):
    """
    Predict with confidence boosting — tries slight variations
    of the 8x8 input and picks the most confident result.
    """
    arr8 = flat.reshape(8, 8)
    
    candidates = []
    
    # Original
    candidates.append(flat.copy())
    
    # Slight brightness variations
    candidates.append(np.clip(arr8 * 0.85, 0, 16).flatten().reshape(1, -1))
    candidates.append(np.clip(arr8 * 1.15, 0, 16).flatten().reshape(1, -1))
    
    # Slight shifts (1 pixel in each direction)
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        shifted = np.roll(np.roll(arr8, dy, axis=0), dx, axis=1)
        candidates.append(shifted.flatten().reshape(1, -1))
    
    best_digit = None
    best_conf  = -1
    best_proba = None
    
    for candidate in candidates:
        scaled = scaler.transform(candidate)
        digit  = int(model.predict(scaled)[0])
        
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled)[0]
            conf  = proba[digit]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(scaled)[0]
            scores -= scores.max()
            exp_s  = np.exp(scores)
            proba  = exp_s / exp_s.sum()
            conf   = proba[digit]
        else:
            conf = 1.0
        
        if conf > best_conf:
            best_conf  = conf
            best_digit = digit
            best_proba = proba
    
    return best_digit, best_proba


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    stroke_width = st.slider("Brush size", 8, 30, 18, step=2)
    show_8x8 = st.checkbox("Show 8×8 model input", value=True)
    st.markdown("---")
    st.markdown("""
    <div style='color:#4a4a4a; font-size:0.78rem; line-height:1.7;'>
    <b style='color:#666'>How it works</b><br><br>
    Your drawing is cropped, padded, and resized to
    <b style='color:#888'>8×8 pixels</b> — the same format as sklearn's
    digits dataset — then fed to your trained model.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if model:
        st.markdown(f"""
        <div style='background:#161616;border:1px solid #222;border-radius:8px;
                    padding:0.7rem 1rem;font-size:0.78rem;color:#666;'>
        ✅ <span style='color:#4fffb0'>Model loaded</span><br>
        <span style='color:#444'>Type:</span>
        <span style='color:#aaa'>{type(model).__name__}</span>
        </div>""", unsafe_allow_html=True)


# ── Main Layout ────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>DIGIT RECOGNIZER</h1>
    <p>Draw · Predict · sklearn digits</p>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.markdown(f"""
    <div class="err-box">
    ⚠️ <b>Model files not found:</b> {load_error}<br><br>
    Place <code>model.pkl</code> and <code>scaler.pkl</code> in the
    <b>same folder</b> as <code>app.py</code>, then restart.
    </div>""", unsafe_allow_html=True)
    st.stop()

# Two-column layout
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown('<p class="hint">✏️ Draw a digit (0–9) in the box below</p>',
                unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍  PREDICT DIGIT", use_container_width=True)
    st.markdown(
        '<p style="color:#333;font-size:0.72rem;text-align:center;margin-top:0.4rem;">'
        'Use the toolbar above the canvas to undo / clear</p>',
        unsafe_allow_html=True
    )

with col_right:
    if predict_btn:
        img_data = canvas_result.image_data

        if img_data is None or img_data[:, :, :3].max() == 0:
            st.markdown(
                '<div class="err-box">Canvas is empty — draw a digit first! ✏️</div>',
                unsafe_allow_html=True
            )
        else:
            with st.spinner(""):
                flat, thumb = preprocess_canvas(img_data)
                digit, proba = run_predict(flat)

            # Prediction result card
            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-digit">{digit}</div>
                <div class="pred-label">Predicted Digit</div>
            </div>""", unsafe_allow_html=True)

            # 8×8 preview
            if show_8x8:
                st.markdown('<p class="preview-label">8×8 input sent to model</p>',
                            unsafe_allow_html=True)
                thumb_display = (np.clip(thumb / 16.0, 0, 1) * 255).astype(np.uint8)
                thumb_img = Image.fromarray(
                    cv2.resize(thumb_display, (112, 112),
                               interpolation=cv2.INTER_NEAREST)
                )
                c1, c2, c3 = st.columns([1, 1.2, 1])
                with c2:
                    st.image(thumb_img, width=112)

            # Confidence bars
            if proba is not None:
                bars = '<div class="conf-container"><div class="conf-title">Confidence per digit</div>'
                for i, p in enumerate(proba):
                    pct = p * 100
                    is_pred = (i == digit)
                    lbl_color  = "#e0ff4f" if is_pred else "#444"
                    fill_color = ("linear-gradient(90deg,#e0ff4f,#4fffb0)"
                                  if is_pred else "#2a2a2a")
                    pct_color  = "#e0ff4f" if is_pred else "#444"
                    bars += f"""
                    <div class="conf-row">
                        <span class="conf-lbl" style="color:{lbl_color}">{i}</span>
                        <div class="conf-bg">
                            <div class="conf-fill"
                                 style="width:{pct:.1f}%;background:{fill_color};">
                            </div>
                        </div>
                        <span class="conf-pct" style="color:{pct_color}">{pct:.0f}%</span>
                    </div>"""
                bars += "</div>"
                st.markdown(bars, unsafe_allow_html=True)