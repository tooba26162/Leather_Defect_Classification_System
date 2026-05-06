import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ── Page Config ────────────────────────────────────────
st.set_page_config(
    page_title="Leather Defect Classifier",
    page_icon="👜",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1b3e 50%, #0a1628 100%);
        min-height: 100vh;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }

    /* Upload area */
    .stFileUploader {
        background: rgba(255,255,255,0.03);
        border: 2px dashed rgba(77, 179, 230, 0.4);
        border-radius: 16px;
        padding: 10px;
    }
    .stFileUploader:hover {
        border-color: rgba(77, 179, 230, 0.8);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1E5FA8, #4DB3E6);
        color: white !important;
        font-weight: 700;
        font-size: 1.1rem;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(77, 179, 230, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4DB3E6, #1E5FA8);
        box-shadow: 0 6px 25px rgba(77, 179, 230, 0.5);
        transform: translateY(-2px);
    }

    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #1E5FA8, #4DB3E6);
        border-radius: 10px;
    }
    .stProgress > div {
        background: rgba(255,255,255,0.07);
        border-radius: 10px;
        height: 12px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(77,179,230,0.2);
        border-radius: 14px;
        padding: 18px !important;
        text-align: center;
    }
    [data-testid="stMetricLabel"] {
        color: rgba(184,223,247,0.8) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.5rem !important;
        font-weight: 800 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 22, 40, 0.95) !important;
        border-right: 1px solid rgba(77,179,230,0.15);
    }
    [data-testid="stSidebar"] * {
        color: #B8DFF7 !important;
    }

    /* Divider */
    hr {
        border-color: rgba(77,179,230,0.15) !important;
        margin: 1.5rem 0 !important;
    }

    /* Info / success / error / warning boxes */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #4DB3E6 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0f1e; }
    ::-webkit-scrollbar-thumb { background: #1E5FA8; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/leather_model.h5")
    with open("model/class_names.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_model()

# ── Defect Info ────────────────────────────────────────
defect_info = {
    "good": {
        "emoji": "✅",
        "label": "GOOD — No Defect",
        "badge_color": "#00C48C",
        "bg": "rgba(0,196,140,0.1)",
        "border": "rgba(0,196,140,0.5)",
        "severity": "None",
        "severity_icon": "🟢",
        "score": "10 / 10",
        "msg": "This leather product is in perfect condition with no visible defects. It is ready for sale or use.",
        "action": "✅ Approved for sale"
    },
    "color": {
        "emoji": "🎨",
        "label": "COLOR DEFECT",
        "badge_color": "#F5A623",
        "bg": "rgba(245,166,35,0.1)",
        "border": "rgba(245,166,35,0.5)",
        "severity": "Medium",
        "severity_icon": "🟡",
        "score": "5 / 10",
        "msg": "Uneven coloring or discoloration detected on the leather surface. This is caused by uneven dye application during manufacturing.",
        "action": "⚠️ Needs color correction before sale"
    },
    "cut": {
        "emoji": "✂️",
        "label": "CUT DEFECT",
        "badge_color": "#FF4757",
        "bg": "rgba(255,71,87,0.1)",
        "border": "rgba(255,71,87,0.5)",
        "severity": "High",
        "severity_icon": "🔴",
        "score": "1 / 10",
        "msg": "A cut or sharp incision detected on the leather surface. This is a serious defect that damages the structural integrity of the product.",
        "action": "❌ Rejected — Cannot be sold"
    },
    "fold": {
        "emoji": "📄",
        "label": "FOLD DEFECT",
        "badge_color": "#F5A623",
        "bg": "rgba(245,166,35,0.1)",
        "border": "rgba(245,166,35,0.5)",
        "severity": "Medium",
        "severity_icon": "🟡",
        "score": "5 / 10",
        "msg": "Fold marks or crease lines detected on the leather. This occurs during storage or manufacturing when leather is improperly handled.",
        "action": "⚠️ May be corrected with treatment"
    },
    "grain": {
        "emoji": "🔍",
        "label": "GRAIN DEFECT",
        "badge_color": "#FF8C00",
        "bg": "rgba(255,140,0,0.1)",
        "border": "rgba(255,140,0,0.5)",
        "severity": "Medium",
        "severity_icon": "🟡",
        "score": "4 / 10",
        "msg": "Grain irregularity detected on the leather surface. The natural grain pattern is uneven or damaged, affecting the appearance quality.",
        "action": "⚠️ Quality downgraded — Discount sale only"
    },
    "poke": {
        "emoji": "🕳️",
        "label": "POKE / HOLE DEFECT",
        "badge_color": "#FF4757",
        "bg": "rgba(255,71,87,0.1)",
        "border": "rgba(255,71,87,0.5)",
        "severity": "High",
        "severity_icon": "🔴",
        "score": "1 / 10",
        "msg": "A puncture or poke hole detected on the leather surface. This is physical damage that cannot be repaired.",
        "action": "❌ Rejected — Cannot be sold"
    },
}

# ─────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:3rem;'>👜</div>
        <div style='color:#4DB3E6; font-size:1.1rem; font-weight:700;
                    letter-spacing:2px; text-transform:uppercase;'>
            Defect AI
        </div>
        <div style='color:rgba(184,223,247,0.5); font-size:0.75rem; margin-top:4px;'>
            Leather Quality Inspector
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style='color:#4DB3E6; font-size:0.8rem; font-weight:700;
              letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;'>
        Detectable Defects
    </p>
    """, unsafe_allow_html=True)

    sidebar_items = [
        ("✅", "Good",  "#00C48C", "None"),
        ("🎨", "Color", "#F5A623", "Medium"),
        ("✂️", "Cut",   "#FF4757", "High"),
        ("📄", "Fold",  "#F5A623", "Medium"),
        ("🔍", "Grain", "#FF8C00", "Medium"),
        ("🕳️", "Poke",  "#FF4757", "High"),
    ]
    for emoji, name, color, sev in sidebar_items:
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; align-items:center;
                    background:rgba(255,255,255,0.04); border-radius:8px;
                    padding:8px 12px; margin-bottom:6px;
                    border-left: 3px solid {color};'>
            <span style='color:white; font-size:0.9rem;'>{emoji} {name}</span>
            <span style='color:{color}; font-size:0.75rem; font-weight:600;'>{sev}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style='color:#4DB3E6; font-size:0.8rem; font-weight:700;
              letter-spacing:2px; text-transform:uppercase; margin-bottom:12px;'>
        How To Use
    </p>
    <div style='color:rgba(184,223,247,0.7); font-size:0.85rem; line-height:2;'>
        1️⃣ &nbsp; Upload a leather image<br>
        2️⃣ &nbsp; Click Analyze button<br>
        3️⃣ &nbsp; View defect results<br>
        4️⃣ &nbsp; Check quality action
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:rgba(184,223,247,0.35);
                font-size:0.72rem; padding-top:8px;'>
        Powered by EfficientNetB0<br>Transfer Learning · CNN
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────

# ── Top Header ─────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 2rem;'>
    <h1 style='color: white; font-size: 2.4rem; font-weight: 800; margin: 0; line-height:1.2;'>
        Leather Defect
        <span style='background: linear-gradient(135deg, #4DB3E6, #1E5FA8);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Classification System
        </span>
    </h1>
    <p style='color: rgba(184,223,247,0.6); font-size: 1rem; margin-top: 8px;'>
        Upload a leather product image to instantly detect and classify surface defects using AI
    </p>
</div>
""", unsafe_allow_html=True)

# ── Two Column Layout ──────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    # Upload card
    st.markdown("""
    <div style='background: rgba(255,255,255,0.03); border: 1px solid rgba(77,179,230,0.15);
                border-radius: 20px; padding: 24px; margin-bottom: 16px;'>
        <p style='color: #4DB3E6; font-size: 0.8rem; font-weight: 700;
                  letter-spacing: 2px; text-transform: uppercase; margin-bottom: 16px;'>
            📤 Upload Image
        </p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

        # Show image nicely
        st.markdown("""
        <div style='background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(77,179,230,0.15);
                    border-radius: 20px; padding: 16px; margin-bottom: 16px;'>
            <p style='color: #4DB3E6; font-size: 0.8rem; font-weight: 700;
                      letter-spacing: 2px; text-transform: uppercase; margin-bottom: 12px;'>
                🖼️ Uploaded Image
            </p>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(f"""
            <p style='color:rgba(184,223,247,0.4); font-size:0.78rem;
                      text-align:center; margin-top:8px;'>
                {uploaded.name}  ·  {round(uploaded.size/1024, 1)} KB
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Analyze button
        analyze = st.button("🔍  Analyze for Defects", use_container_width=True)

with right_col:
    if uploaded is None:
        # Empty state
        st.markdown("""
        <div style='background: rgba(255,255,255,0.02);
                    border: 1px solid rgba(77,179,230,0.1);
                    border-radius: 20px; padding: 60px 30px;
                    text-align: center; min-height: 400px;
                    display: flex; flex-direction: column;
                    justify-content: center; align-items: center;'>
            <div style='font-size: 4rem; margin-bottom: 16px;'>👜</div>
            <p style='color: rgba(184,223,247,0.5); font-size: 1rem; margin: 0;'>
                Upload a leather product image<br>to see defect analysis here
            </p>
            <div style='margin-top: 24px; display: flex; gap: 8px; flex-wrap: wrap;
                        justify-content: center;'>
                <span style='background:rgba(77,179,230,0.1); color:#4DB3E6;
                             padding:4px 12px; border-radius:20px; font-size:0.8rem;'>
                    Bags
                </span>
                <span style='background:rgba(77,179,230,0.1); color:#4DB3E6;
                             padding:4px 12px; border-radius:20px; font-size:0.8rem;'>
                    Wallets
                </span>
                <span style='background:rgba(77,179,230,0.1); color:#4DB3E6;
                             padding:4px 12px; border-radius:20px; font-size:0.8rem;'>
                    Shoes
                </span>
                <span style='background:rgba(77,179,230,0.1); color:#4DB3E6;
                             padding:4px 12px; border-radius:20px; font-size:0.8rem;'>
                    Belts
                </span>
                <span style='background:rgba(77,179,230,0.1); color:#4DB3E6;
                             padding:4px 12px; border-radius:20px; font-size:0.8rem;'>
                    Jackets
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded is not None and 'analyze' in dir() and analyze:
        with st.spinner("🤖 Analyzing leather product..."):
            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions   = model.predict(img_array, verbose=0)
            predicted_idx  = np.argmax(predictions[0])
            predicted_class= class_names[predicted_idx]
            confidence     = predictions[0][predicted_idx] * 100

        info = defect_info.get(predicted_class, {
            "emoji": "⚠️", "label": predicted_class.upper(),
            "badge_color": "#F5A623", "bg": "rgba(245,166,35,0.1)",
            "border": "rgba(245,166,35,0.5)",
            "severity": "Unknown", "severity_icon": "🟡", "score": "?/10",
            "msg": f"Defect detected: {predicted_class}",
            "action": "Manual inspection required"
        })

        # ── Big result card ──────────────────────────
        st.markdown(f"""
        <div style='background: {info["bg"]};
                    border: 2px solid {info["border"]};
                    border-radius: 20px; padding: 28px;
                    text-align: center; margin-bottom: 16px;'>
            <div style='font-size: 3.5rem; margin-bottom: 8px;'>{info["emoji"]}</div>
            <div style='color: {info["badge_color"]}; font-size: 1.6rem;
                        font-weight: 800; letter-spacing: 1px;'>
                {info["label"]}
            </div>
            <div style='color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 6px;'>
                {confidence:.1f}% confidence
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 3 metrics ───────────────────────────────
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("🏷️ Defect", predicted_class.upper())
        with m2:
            st.metric("⚠️ Severity",
                      f"{info['severity_icon']} {info['severity']}")
        with m3:
            st.metric("⭐ Score", info["score"])

        # ── Description ──────────────────────────────
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.03);
                    border-left: 4px solid {info["badge_color"]};
                    border-radius: 0 12px 12px 0;
                    padding: 16px 20px; margin: 16px 0;'>
            <p style='color: rgba(184,223,247,0.5); font-size: 0.75rem;
                      font-weight: 700; letter-spacing: 1px;
                      text-transform: uppercase; margin-bottom: 6px;'>
                Description
            </p>
            <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;
                      margin: 0; line-height: 1.6;'>
                {info["msg"]}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Action badge ─────────────────────────────
        st.markdown(f"""
        <div style='background: {info["bg"]};
                    border: 1px solid {info["border"]};
                    border-radius: 12px; padding: 14px 20px;
                    margin-bottom: 16px;'>
            <p style='color: rgba(184,223,247,0.5); font-size: 0.75rem;
                      font-weight: 700; letter-spacing: 1px;
                      text-transform: uppercase; margin-bottom: 4px;'>
                Quality Control Action
            </p>
            <p style='color: {info["badge_color"]}; font-size: 1rem;
                      font-weight: 700; margin: 0;'>
                {info["action"]}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bars ──────────────────────────
        st.markdown("""
        <p style='color: rgba(184,223,247,0.5); font-size: 0.75rem;
                  font-weight: 700; letter-spacing: 2px;
                  text-transform: uppercase; margin-bottom: 10px;'>
            📊 All Class Probabilities
        </p>
        """, unsafe_allow_html=True)

        for i, name in enumerate(class_names):
            prob = predictions[0][i] * 100
            d    = defect_info.get(name, {})
            emoji= d.get("emoji", "•")
            color= d.get("badge_color", "#4DB3E6")
            st.markdown(f"""
            <div style='display:flex; align-items:center; margin-bottom:8px; gap:10px;'>
                <span style='color:white; font-size:0.85rem;
                             min-width:110px;'>{emoji} {name}</span>
                <div style='flex:1; background:rgba(255,255,255,0.07);
                            border-radius:20px; height:10px; overflow:hidden;'>
                    <div style='width:{prob:.1f}%; height:100%;
                                background:linear-gradient(90deg, {color}88, {color});
                                border-radius:20px; transition: width 0.8s ease;'>
                    </div>
                </div>
                <span style='color:{color}; font-size:0.85rem;
                             font-weight:700; min-width:48px;
                             text-align:right;'>{prob:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    elif uploaded is not None:
        # Waiting for button click
        st.markdown("""
        <div style='background: rgba(255,255,255,0.02);
                    border: 1px solid rgba(77,179,230,0.1);
                    border-radius: 20px; padding: 60px 30px;
                    text-align: center; min-height: 300px;'>
            <div style='font-size: 3rem; margin-bottom: 16px;'>🔍</div>
            <p style='color: rgba(184,223,247,0.5); font-size: 1rem; margin: 0;'>
                Click <strong style="color:#4DB3E6;">Analyze for Defects</strong><br>
                to run AI inspection
            </p>
        </div>
        """, unsafe_allow_html=True)