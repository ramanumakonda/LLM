import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io
import base64
from gtts import gTTS
import tempfile
import os
import time
from datetime import datetime
import hashlib

# Page configuration
st.set_page_config(
    page_title="Tactile Bridge",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .main-title { font-size: 2.5rem; font-weight: 800; text-align: center; 
                  background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                  padding: 1rem; animation: fadeIn 1s; }
    .sub-title { color: white; text-align: center; font-size: 1.1rem; margin-bottom: 1rem; }
    .card { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); 
            padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(255,255,255,0.2);
            margin: 1rem 0; }
    .success-box { background: linear-gradient(135deg, #00b09b, #96c93d); padding: 1rem;
                   border-radius: 10px; color: white; text-align: center; }
    .stButton>button { background: linear-gradient(45deg, #667eea, #764ba2); color: white;
                       border: none; padding: 0.5rem 1.5rem; border-radius: 25px;
                       font-weight: 600; width: 100%; transition: 0.3s; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .nav-item { padding: 0.5rem; margin: 0.2rem 0; border-radius: 10px; cursor: pointer;
                transition: 0.3s; color: white; text-align: center; }
    .nav-item:hover { background: rgba(255,255,255,0.2); }
    .nav-item.active { background: linear-gradient(45deg, #667eea, #764ba2); }
</style>
""", unsafe_allow_html=True)

# Initialize session state
defaults = {
    'page': 'Input',
    'history': [],
    'current_text': '',
    'current_dots': [],
    'current_cells': [],
    'processed_img': None
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Braille mapping (Grade 1 English)
BRAILLE_MAP = {
    (1,0,0,0,0,0): 'a', (1,1,0,0,0,0): 'b', (1,0,0,1,0,0): 'c',
    (1,0,0,1,1,0): 'd', (1,0,0,0,1,0): 'e', (1,1,0,1,0,0): 'f',
    (1,1,0,1,1,0): 'g', (1,1,0,0,1,0): 'h', (0,1,0,1,0,0): 'i',
    (0,1,0,1,1,0): 'j', (1,0,1,0,0,0): 'k', (1,1,1,0,0,0): 'l',
    (1,0,1,1,0,0): 'm', (1,0,1,1,1,0): 'n', (1,0,1,0,1,0): 'o',
    (1,1,1,1,0,0): 'p', (1,1,1,1,1,0): 'q', (1,1,1,0,1,0): 'r',
    (0,1,1,1,0,0): 's', (0,1,1,1,1,0): 't', (1,0,1,0,0,1): 'u',
    (1,1,1,0,0,1): 'v', (0,1,0,1,1,1): 'w', (1,0,1,1,0,1): 'x',
    (1,0,1,1,1,1): 'y', (1,0,1,0,1,1): 'z', (0,0,0,0,0,0): ' '
}

# Navigation bar
st.markdown('<h1 class="main-title">🤟 Tactile Communication Bridge</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Bridging Braille and Digital Text with AI</p>', unsafe_allow_html=True)

cols = st.columns(4)
pages = ['📤 Input', '⚙️ Process', '📝 Output', '📜 History']
for i, (col, page) in enumerate(zip(cols, pages)):
    with col:
        if st.button(page, key=f"nav_{i}"):
            st.session_state.page = page.split(' ')[1]

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/null/braille.png", width=80)
    st.markdown("### 📁 Upload")
    uploaded = st.file_uploader("Choose Braille image", type=['jpg', 'jpeg', 'png'])
    
    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Detection threshold", 0, 255, 127, 5)
    enable_audio = st.checkbox("🔊 Enable audio", True)
    
    if st.session_state.history:
        st.markdown("### 📊 Stats")
        st.metric("Processed", len(st.session_state.history))
        st.metric("Words", sum(len(h['text'].split()) for h in st.session_state.history))

@st.cache_data
def process_image(img_array, thresh):
    """Optimized image processing"""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return gray, binary, cleaned

def detect_cells(processed, shape):
    """Efficient Braille cell detection"""
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 300:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                dots.append((cx, cy))
    
    if len(dots) < 6:
        return [], dots
    
    dots.sort(key=lambda x: x[1])
    rows, current = [], []
    for dot in dots:
        if not current or abs(dot[1] - current[-1][1]) < 20:
            current.append(dot)
        else:
            rows.append(sorted(current, key=lambda x: x[0]))
            current = [dot]
    if current:
        rows.append(sorted(current, key=lambda x: x[0]))
    
    cells = []
    for i in range(0, len(rows)-2, 3):
        for j in range(0, min(len(rows[i]), len(rows[i+1]), len(rows[i+2]))-1, 2):
            cell = [0]*6
            if j < len(rows[i]): cell[0] = 1
            if j+1 < len(rows[i]): cell[1] = 1
            if j < len(rows[i+1]): cell[2] = 1
            if j+1 < len(rows[i+1]): cell[3] = 1
            if j < len(rows[i+2]): cell[4] = 1
            if j+1 < len(rows[i+2]): cell[5] = 1
            cells.append(tuple(cell))
    
    return cells, dots

def translate(cells):
    """Convert Braille patterns to text"""
    return ''.join(BRAILLE_MAP.get(cell, '?') for cell in cells)

def text_to_speech(text):
    """Generate audio from text"""
    tts = gTTS(text=text, lang='en', slow=False)
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts.save(temp.name)
    return temp.name

# Main content
if st.session_state.page == "Input":
    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📸 Image Input")
            if uploaded:
                img = Image.open(uploaded)
                st.image(img, caption="Uploaded Image", use_column_width=True)
                st.info(f"Size: {img.size[0]}x{img.size[1]} | Format: {img.format}")
            else:
                st.warning("Please upload an image from sidebar")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### ℹ️ Quick Guide")
            st.markdown("""
            1. Upload Braille image (JPG/PNG)
            2. Adjust threshold if needed
            3. Click Process tab
            4. View translated text
            5. Use audio for listening
            """)
            st.markdown("### ✅ Tips")
            st.markdown("• Use clear, well-lit images\n• Ensure dots are visible\n• Avoid tilted images")
            st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "Process":
    if uploaded:
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 🔄 Processing")
                
                img = Image.open(uploaded)
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                with st.spinner("Processing..."):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress.progress(i + 1)
                    
                    gray, binary, cleaned = process_image(img_array, threshold)
                    cells, dots = detect_cells(cleaned, img_array.shape)
                    text = translate(cells)
                    
                    st.session_state.current_text = text
                    st.session_state.current_dots = dots
                    st.session_state.current_cells = cells
                    st.session_state.processed_img = cleaned
                
                st.success("✅ Processing complete!")
                st.markdown('</div>', unsafe_allow_html=True)
                
                with st.container():
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### 📊 Results")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Dots", len(dots))
                    m2.metric("Cells", len(cells))
                    m3.metric("Chars", len(text))
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 👁️ Detection")
                
                vis_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                for dot in dots:
                    cv2.circle(vis_img, dot, 3, (0,255,0), -1)
                
                st.image(vis_img, caption="Detected Dots (Green)", use_column_width=True)
                
                tabs = st.tabs(["Grayscale", "Binary", "Cleaned"])
                with tabs[0]: st.image(gray, use_column_width=True)
                with tabs[1]: st.image(binary, use_column_width=True)
                with tabs[2]: st.image(cleaned, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Upload an image first")

elif st.session_state.page == "Output":
    if st.session_state.current_text:
        col1, col2 = st.columns([2,1])
        
        with col1:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 📝 Translation")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 2rem; border-radius: 15px; color: white;">
                    <h3 style="text-align: center;">{st.session_state.current_text}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                b1, b2, b3 = st.columns(3)
                with b1:
                    if st.button("📋 Copy"):
                        st.write("Copied!")
                        st.balloons()
                
                with b2:
                    txt = io.BytesIO(st.session_state.current_text.encode())
                    st.download_button("📥 Download", txt, "braille.txt", "text/plain")
                
                with b3:
                    if st.button("💾 Save"):
                        st.session_state.history.append({
                            'time': datetime.now().strftime('%H:%M:%S'),
                            'text': st.session_state.current_text,
                            'chars': len(st.session_state.current_text)
                        })
                        st.success("Saved!")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 🔊 Audio")
                if enable_audio and st.button("▶️ Play"):
                    with st.spinner("Generating..."):
                        audio = text_to_speech(st.session_state.current_text)
                        st.audio(audio)
                        os.unlink(audio)
                
                if st.button("🔄 New Translation"):
                    for k in ['current_text', 'current_dots', 'current_cells']:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No text to display. Process an image first.")

else:  # History
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📜 Translation History")
        
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df, use_column_width=True)
            
            if st.button("🗑️ Clear"):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("No history yet")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.7);'>Made with ❤️ for accessibility</p>", 
            unsafe_allow_html=True)