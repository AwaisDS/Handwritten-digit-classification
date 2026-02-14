import streamlit as st
import numpy as np
import pickle
from PIL import Image, ImageOps
import cv2
import base64
import io

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

        model  = smart_load("C:\\Users\\pc\\Desktop\\Handwritten-digit-classification\\models\\mnist_svm_model.pkl")
        scaler = smart_load("C:\\Users\\pc\\Desktop\\Handwritten-digit-classification\\models\\mnist_scaler.pkl")
        return model, scaler, None

    except FileNotFoundError as e:
        return None, None, f"File not found: {e}"
    except Exception as e:
        return None, None, str(e)


model, scaler, load_error = load_model()


# ── HTML5 Canvas Component ─────────────────────────────────────────────────
def drawing_canvas(width=300, height=300, stroke_width=18):
    """
    Pure HTML5 canvas — no external package needed.
    Returns base64 PNG string of what was drawn.
    """
    canvas_html = f"""
    <style>
        #canvas-wrapper {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }}
        #draw-canvas {{
            border: 2px solid #2a2a2a;
            border-radius: 10px;
            cursor: crosshair;
            background: #000;
            touch-action: none;
        }}
        .canvas-btns {{
            display: flex;
            gap: 8px;
            width: {width}px;
        }}
        .canvas-btn {{
            flex: 1;
            padding: 7px 0;
            border: 1px solid #333;
            border-radius: 6px;
            background: #1a1a1a;
            color: #aaa;
            font-family: 'Space Mono', monospace;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .canvas-btn:hover {{ background: #2a2a2a; color: #fff; }}
        #canvas-output {{ display: none; }}
    </style>

    <div id="canvas-wrapper">
        <canvas id="draw-canvas" width="{width}" height="{height}"></canvas>
        <div class="canvas-btns">
            <button class="canvas-btn" onclick="clearCanvas()">🗑 Clear</button>
            <button class="canvas-btn" onclick="undoLast()">↩ Undo</button>
            <button class="canvas-btn" onclick="saveCanvas()">✅ Save for Predict</button>
        </div>
        <input type="text" id="canvas-output">
    </div>

    <script>
        const canvas  = document.getElementById('draw-canvas');
        const ctx     = canvas.getContext('2d');
        let painting  = false;
        let paths     = [];
        let current   = [];
        const SW      = {stroke_width};

        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth   = SW;
        ctx.lineCap     = 'round';
        ctx.lineJoin    = 'round';

        function getPos(e) {{
            const r = canvas.getBoundingClientRect();
            if (e.touches) {{
                return {{
                    x: e.touches[0].clientX - r.left,
                    y: e.touches[0].clientY - r.top
                }};
            }}
            return {{ x: e.clientX - r.left, y: e.clientY - r.top }};
        }}

        function startPaint(e) {{
            e.preventDefault();
            painting = true;
            current  = [];
            const p  = getPos(e);
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            current.push(p);
        }}

        function draw(e) {{
            if (!painting) return;
            e.preventDefault();
            const p = getPos(e);
            ctx.lineTo(p.x, p.y);
            ctx.stroke();
            current.push(p);
        }}

        function stopPaint(e) {{
            if (!painting) return;
            painting = false;
            paths.push([...current]);
            current  = [];
        }}

        function redraw() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            paths.forEach(path => {{
                if (path.length < 2) return;
                ctx.beginPath();
                ctx.moveTo(path[0].x, path[0].y);
                path.slice(1).forEach(p => {{ ctx.lineTo(p.x, p.y); ctx.stroke(); }});
            }});
        }}

        function clearCanvas() {{
            paths = [];
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            document.getElementById('canvas-output').value = '';
            window.parent.postMessage({{type: 'canvas_data', data: ''}}, '*');
        }}

        function undoLast() {{
            paths.pop();
            redraw();
        }}

        function saveCanvas() {{
            const data = canvas.toDataURL('image/png');
            document.getElementById('canvas-output').value = data;
            // Send to Streamlit via query param trick
            window.parent.postMessage({{type: 'streamlit:setComponentValue', value: data}}, '*');
        }}

        canvas.addEventListener('mousedown',  startPaint);
        canvas.addEventListener('mousemove',  draw);
        canvas.addEventListener('mouseup',    stopPaint);
        canvas.addEventListener('mouseleave', stopPaint);
        canvas.addEventListener('touchstart', startPaint, {{passive: false}});
        canvas.addEventListener('touchmove',  draw,       {{passive: false}});
        canvas.addEventListener('touchend',   stopPaint);
    </script>
    """
    result = st.components.v1.html(canvas_html, height=height + 80, scrolling=False)
    return result


# ── Preprocessing ──────────────────────────────────────────────────────────
def preprocess_image(image_data_b64):
    """
    Decode base64 PNG → match training pipeline exactly:
    grayscale → crop → 28×28 → /255.0 → flatten(784) → scaler.transform()
    """
    # Decode base64
    header, data = image_data_b64.split(',', 1)
    img_bytes = base64.b64decode(data)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    arr = np.array(img, dtype=np.float64)

    # Crop to bounding box of stroke
    non_zero = np.where(arr > 10)
    if len(non_zero[0]) == 0:
        return None, None

    top, bottom = non_zero[0].min(), non_zero[0].max()
    left, right  = non_zero[1].min(), non_zero[1].max()
    arr = arr[top:bottom+1, left:right+1]

    # Make square
    h, w = arr.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.float64)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = arr

    # Add padding
    pad = size // 5
    square = np.pad(square, pad, mode='constant', constant_values=0)

    # Resize to 28×28
    img_resized = Image.fromarray(square.astype(np.uint8)).resize((28, 28), Image.LANCZOS)
    arr28 = np.array(img_resized, dtype=np.float64)

    # Normalize to 0–1  ✅ matches X / 255.0
    arr28 = arr28 / 255.0

    # Flatten to 784  ✅ matches reshape(-1, 28*28)
    flat = arr28.flatten().reshape(1, -1)

    return flat, arr28


def run_predict(flat):
    # StandardScaler  ✅ matches scaler.transform(X)
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
    <b style='color:#666'>Pipeline</b><br><br>
    Canvas → crop → 28×28<br>
    → ÷255 → flatten(784)<br>
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
    in the same folder as <code>app.py</code>.
    </div>""", unsafe_allow_html=True)
    st.stop()

col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown('<p style="color:#444;font-size:0.78rem;text-align:center;margin-bottom:0.5rem;">✏️ Draw a digit then click Save for Predict</p>',
                unsafe_allow_html=True)
    drawing_canvas(width=300, height=300, stroke_width=stroke_width)

    # File uploader as fallback input method
    st.markdown('<p style="color:#333;font-size:0.72rem;text-align:center;margin-top:1rem;">— or upload an image —</p>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    predict_btn = st.button("🔍  PREDICT DIGIT", use_container_width=True)

with col_right:
    if predict_btn:
        # Get canvas data from session state
        canvas_data = st.session_state.get("canvas_img", None)

        # Prefer uploaded file
        if uploaded is not None:
            img = Image.open(uploaded).convert("L")
            arr = np.array(img, dtype=np.float64)
            non_zero = np.where(arr > 10)
            if len(non_zero[0]) > 0:
                top, bottom = non_zero[0].min(), non_zero[0].max()
                left, right  = non_zero[1].min(), non_zero[1].max()
                arr = arr[top:bottom+1, left:right+1]
            h, w = arr.shape
            size = max(h, w)
            square = np.zeros((size, size), dtype=np.float64)
            square[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = arr
            pad = size // 5
            square = np.pad(square, pad, mode='constant', constant_values=0)
            img_r = Image.fromarray(square.astype(np.uint8)).resize((28, 28), Image.LANCZOS)
            arr28 = np.array(img_r, dtype=np.float64) / 255.0
            flat  = arr28.flatten().reshape(1, -1)

            digit, proba = run_predict(flat)

            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-digit">{digit}</div>
                <div class="pred-label">Predicted Digit</div>
            </div>""", unsafe_allow_html=True)

            if show_28x28:
                st.markdown('<p class="preview-label">28×28 input sent to model</p>', unsafe_allow_html=True)
                thumb = (np.clip(arr28, 0, 1) * 255).astype(np.uint8)
                thumb_img = Image.fromarray(cv2.resize(thumb, (112, 112), interpolation=cv2.INTER_NEAREST))
                c1, c2, c3 = st.columns([1, 1.2, 1])
                with c2:
                    st.image(thumb_img, width=112)

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
        else:
            st.markdown("""
            <div class="err-box">
            ⚠️ Please <b>upload an image</b> of a digit using the uploader below the canvas.<br><br>
            <b>How to use canvas:</b><br>
            1. Draw your digit<br>
            2. Save the image using your browser (right-click → Save image)<br>
            3. Upload it using the uploader
            </div>""", unsafe_allow_html=True)
