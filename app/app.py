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
html, body, [class*="css"] { background-color: #0d0d0d; color: #f0f0f0; }
.stApp { background: #0d0d0d; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.title-block { text-align: center; padding: 2rem 0 1.5rem; }
.title-block h1 {
    font-size: 2.6rem;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #e0ff4f, #4fffb0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.title-block p { color: #555; font-size: 0.85rem; letter-spacing: 0.12em; text-transform: uppercase; }

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
.pred-label { color: #666; font-size: 0.75rem; letter-spacing: 0.15em; text-transform: uppercase; margin-top: 0.4rem; }

.conf-container { background: #131313; border: 1px solid #222; border-radius: 10px; padding: 1rem 1.2rem; margin-top: 0.5rem; }
.conf-title { color: #555; font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.7rem; }
.conf-row { display: flex; align-items: center; gap: 0.6rem; margin: 0.28rem 0; }
.conf-lbl { font-family: 'Space Mono', monospace; font-size: 0.8rem; width: 16px; text-align: right; flex-shrink: 0; }
.conf-bg { flex: 1; background: #222; border-radius: 3px; height: 8px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 3px; }
.conf-pct { font-family: 'Space Mono', monospace; font-size: 0.7rem; width: 38px; text-align: right; flex-shrink: 0; }

.hint { color: #444; font-size: 0.78rem; text-align: center; margin: 0.3rem 0 1rem; letter-spacing: 0.04em; }

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
    background: #180f0f; border: 1px solid #4a1a1a;
    border-left: 4px solid #ff4f4f; border-radius: 8px;
    padding: 0.9rem 1.2rem; color: #ff8a8a; font-size: 0.85rem; margin-top: 1rem;
}
.preview-label { color: #444; font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; text-align: center; margin: 0.8rem 0 0.3rem; }

#MainMenu, footer { visibility: hidden; }
section[data-testid="stSidebar"] { background: #111 !important; border-right: 1px solid #1e1e1e; }
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

        model  = joblib.load("C:\\Users\\pc\\Desktop\Handwritten-digit-classification\\app\\mnist_scaler.pkl")
        scaler = joblib.load("C:\\Users\\pc\\Desktop\Handwritten-digit-classification\\app\\mnist_scaler.pkl")
        return model, scaler, None

    except FileNotFoundError as e:
        return None, None, f"File not found: {e}"
    except Exception as e:
        return None, None, str(e)


model, scaler, load_error = load_model()


# ── Preprocessing — matches your training pipeline exactly ─────────────────
#
#   Training pipeline:
#     1. X / 255.0          → normalize to 0–1
#     2. reshape(-1, 28*28) → flatten to 784
#     3. StandardScaler     → zero mean, unit variance
#
#   So here we do the same:
#     Canvas (RGBA) → grayscale → 28×28 → /255.0 → flatten → scaler.transform()
#
def preprocess_canvas(image_data):
    # 1. Convert RGBA canvas to grayscale
    img = Image.fromarray(image_data.astype("uint8"), "RGBA").convert("L")
    arr = np.array(img, dtype=np.float64)

    # 2. Crop tightly around the drawn stroke
    non_zero = np.where(arr > 10)
    if len(non_zero[0]) == 0:
        return np.zeros((1, 784)), np.zeros((28, 28))

    top, bottom = non_zero[0].min(), non_zero[0].max()
    left, right  = non_zero[1].min(), non_zero[1].max()
    arr = arr[top:bottom+1, left:right+1]

    # 3. Make bounding box square (prevents squishing)
    h, w = arr.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.float64)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = arr

    # 4. Add padding (~20%) so digit isn't touching edges
    pad = size // 5
    square = np.pad(square, pad, mode='constant', constant_values=0)

    # 5. Resize to 28×28
    img_resized = Image.fromarray(square.astype(np.uint8)).resize((28, 28), Image.LANCZOS)
    arr28 = np.array(img_resized, dtype=np.float64)

    # 6. Normalize to 0–1  ✅ matches X_train / 255.0
    arr28 = arr28 / 255.0

    # 7. Flatten to 784  ✅ matches reshape(-1, 28*28)
    flat = arr28.flatten().reshape(1, -1)

    return flat, arr28


def run_predict(flat):
    # 8. StandardScaler transform  ✅ matches scaler.transform(X)
    scaled = scaler.transform(flat)
    digit  = int(model.predict(scaled)[0])

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(scaled)[0]
        scores -= scores.max()
        exp_s  = np.exp(scores)
        proba  = exp_s / exp_s.sum()

    return digit, proba


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
    stroke_width = st.slider("Brush size", 10, 40, 22, step=2)
    show_28x28   = st.checkbox("Show 28×28 model input", value=True)
    st.markdown("---")
    st.markdown("""
    <div style='color:#4a4a4a; font-size:0.78rem; line-height:1.7;'>
    <b style='color:#666'>Preprocessing pipeline</b><br><br>
    Canvas → crop → 28×28<br>
    → ÷255 → flatten (784)<br>
    → StandardScaler → model
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if model:
        st.markdown(f"""
        <div style='background:#161616;border:1px solid #222;border-radius:8px;
                    padding:0.7rem 1rem;font-size:0.78rem;color:#666;'>
        ✅ <span style='color:#4fffb0'>MNIST model loaded</span><br>
        <span style='color:#444'>Type:</span>
        <span style='color:#aaa'>{type(model).__name__}</span>
        </div>""", unsafe_allow_html=True)


# ── Main Layout ────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>DIGIT RECOGNIZER</h1>
    <p>MNIST · 28×28 · Draw & Predict</p>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.markdown(f"""
    <div class="err-box">
    ⚠️ <b>Model files not found:</b> {load_error}<br><br>
    Place <code>mnist_model.pkl</code> and <code>mnist_scaler.pkl</code>
    in the <b>same folder</b> as <code>app.py</code>, then restart.
    </div>""", unsafe_allow_html=True)
    st.stop()

col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown('<p class="hint">✏️ Draw a digit (0–9) in the box below</p>',
                unsafe_allow_html=True)

    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=300,
        width=300,
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

            # Prediction result
            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-digit">{digit}</div>
                <div class="pred-label">Predicted Digit</div>
            </div>""", unsafe_allow_html=True)

            # 28×28 preview
            if show_28x28:
                st.markdown('<p class="preview-label">28×28 input sent to model</p>',
                            unsafe_allow_html=True)
                thumb_display = (np.clip(thumb, 0, 1) * 255).astype(np.uint8)
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
                    is_pred    = (i == digit)
                    lbl_color  = "#e0ff4f" if is_pred else "#444"
                    fill_color = "linear-gradient(90deg,#e0ff4f,#4fffb0)" if is_pred else "#2a2a2a"
                    pct_color  = "#e0ff4f" if is_pred else "#444"
                    bars += f"""
                    <div class="conf-row">
                        <span class="conf-lbl" style="color:{lbl_color}">{i}</span>
                        <div class="conf-bg">
                            <div class="conf-fill" style="width:{pct:.1f}%;background:{fill_color};"></div>
                        </div>
                        <span class="conf-pct" style="color:{pct_color}">{pct:.0f}%</span>
                    </div>"""
                bars += "</div>"
                st.markdown(bars, unsafe_allow_html=True)
