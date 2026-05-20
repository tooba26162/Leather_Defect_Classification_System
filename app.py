import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import json
import time
import datetime
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LeatherAI — Defect Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

:root {
    --bg:         #0A0A0B;
    --bg2:        #111114;
    --bg3:        #18181C;
    --border:     rgba(255,255,255,0.07);
    --border2:    rgba(255,255,255,0.12);
    --text:       #E8E6E1;
    --muted:      #6B6B72;
    --accent:     #C8FF00;
    --accent2:    #FF6B35;
    --accent3:    #00D4FF;
    --red:        #FF3B3B;
    --yellow:     #FFB800;
    --green:      #00E676;
    --card-bg:    #13131A;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
.stApp { background: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── File uploader ── */
.stFileUploader {
    background: var(--bg3);
    border: 1px dashed rgba(200,255,0,0.3);
    border-radius: 8px;
}
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #0A0A0B !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #AADD00 !important;
    transform: translateY(-1px) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    color: var(--accent) !important;
    border: 1px solid rgba(200,255,0,0.3) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    width: auto !important;
    padding: 0.5rem 1.5rem !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(200,255,0,0.08) !important;
    transform: none !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: 'Space Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: var(--accent); border-radius: 2px; }
.stProgress > div { background: var(--bg3); border-radius: 2px; height: 4px; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 12px 24px;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
}

/* ── Select ── */
.stSelectbox > div > div {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text) !important;
}

/* ── Radio ── */
.stRadio > label { color: var(--muted) !important; font-size: 0.8rem !important; }
.stRadio [data-testid="stWidgetLabel"] {
    color: var(--muted) !important;
    font-size: 0.8rem !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Alert / info boxes ── */
.stAlert {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    color: var(--text) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* ── Camera ── */
[data-testid="stCameraInput"] video {
    border-radius: 8px;
    border: 1px solid var(--border2);
}
[data-testid="stCameraInput"] { background: var(--bg3); border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS & CONFIG
# ═══════════════════════════════════════════════════════════════

# severity_score: 1–10 (10 = perfect)
DEFECT_CONFIG = {
    "good": {
        "label":          "No Defect Detected",
        "tag":            "PASS",
        "tag_color":      "#00E676",
        "tag_bg":         "rgba(0,230,118,0.12)",
        "accent":         "#00E676",
        "severity_level": "None",
        "severity_score":  10,
        "quality_score":   10,
        "msg":            "This leather sample shows no visible surface defects. Grain structure, "
                          "coloration, and surface integrity are all within acceptable quality parameters.",
        "action":         "✅  Approved for production",
        "action_color":   "#00E676",
        "repair_cost":    "₨ 0",
        "repair_time":    "0 min",
        "recommended":    "No action required. Proceed to packaging.",
    },
    "color": {
        "label":          "Color Irregularity",
        "tag":            "REVIEW",
        "tag_color":      "#FFB800",
        "tag_bg":         "rgba(255,184,0,0.12)",
        "accent":         "#FFB800",
        "severity_level": "Medium",
        "severity_score":  5,
        "quality_score":   5,
        "msg":            "Uneven dye distribution or discoloration detected across the leather surface. "
                          "Likely caused by inconsistent tanning or dye application during manufacture.",
        "action":         "⚠️  Requires color correction before sale",
        "action_color":   "#FFB800",
        "repair_cost":    "₨ 200–500",
        "repair_time":    "45–90 min",
        "recommended":    "Send to re-dyeing station. Apply uniform dye coat and re-inspect.",
    },
    "cut": {
        "label":          "Surface Cut",
        "tag":            "REJECT",
        "tag_color":      "#FF3B3B",
        "tag_bg":         "rgba(255,59,59,0.12)",
        "accent":         "#FF3B3B",
        "severity_level": "Critical",
        "severity_score":  1,
        "quality_score":   1,
        "msg":            "A sharp incision or laceration has been identified on the leather surface. "
                          "This defect compromises structural integrity and is irreparable.",
        "action":         "❌  Rejected — Scrap or downgrade",
        "action_color":   "#FF3B3B",
        "repair_cost":    "₨ 0 (loss)",
        "repair_time":    "N/A",
        "recommended":    "Remove from batch. Log as scrap. Investigate cutting equipment calibration.",
    },
    "fold": {
        "label":          "Fold Mark",
        "tag":            "REVIEW",
        "tag_color":      "#FF6B35",
        "tag_bg":         "rgba(255,107,53,0.12)",
        "accent":         "#FF6B35",
        "severity_level": "Medium",
        "severity_score":  4,
        "quality_score":   4,
        "msg":            "Permanent crease or fold marks detected. Typically results from improper "
                          "storage or handling during the manufacturing process.",
        "action":         "⚠️  May be remediated with conditioning",
        "action_color":   "#FF6B35",
        "repair_cost":    "₨ 100–300",
        "repair_time":    "20–40 min",
        "recommended":    "Apply leather conditioner under controlled heat. Re-press and re-inspect.",
    },
    "grain": {
        "label":          "Grain Defect",
        "tag":            "REVIEW",
        "tag_color":      "#FF6B35",
        "tag_bg":         "rgba(255,107,53,0.12)",
        "accent":         "#FF6B35",
        "severity_level": "Medium",
        "severity_score":  4,
        "quality_score":   4,
        "msg":            "The natural grain pattern shows irregularity or surface damage. "
                          "Affects visual quality and perceived value of the finished product.",
        "action":         "⚠️  Downgraded — Discount sale only",
        "action_color":   "#FF6B35",
        "repair_cost":    "₨ 150–400",
        "repair_time":    "30–60 min",
        "recommended":    "Grade as B-class. Apply surface treatment and label accordingly.",
    },
    "poke": {
        "label":          "Puncture / Hole",
        "tag":            "REJECT",
        "tag_color":      "#FF3B3B",
        "tag_bg":         "rgba(255,59,59,0.12)",
        "accent":         "#FF3B3B",
        "severity_level": "Critical",
        "severity_score":  1,
        "quality_score":   1,
        "msg":            "A puncture or hole has been identified on the leather surface. This is "
                          "physical damage that is irreparable and renders the product unsaleable "
                          "at standard pricing.",
        "action":         "❌  Rejected — Not suitable for sale",
        "action_color":   "#FF3B3B",
        "repair_cost":    "₨ 0 (loss)",
        "repair_time":    "N/A",
        "recommended":    "Remove from batch. Log QC failure. Inspect tooling for sharp protrusions.",
    },
    "folding marks": {
        "label":          "Folding Marks",
        "tag":            "REVIEW",
        "tag_color":      "#FF6B35",
        "tag_bg":         "rgba(255,107,53,0.12)",
        "accent":         "#FF6B35",
        "severity_level": "Medium",
        "severity_score":  4,
        "quality_score":   4,
        "msg":            "Multiple fold marks detected across the surface. These crease patterns "
                          "affect premium grade classification.",
        "action":         "⚠️  Conditioning treatment required",
        "action_color":   "#FF6B35",
        "repair_cost":    "₨ 100–300",
        "repair_time":    "20–40 min",
        "recommended":    "Apply leather conditioner under controlled heat. Re-press and re-inspect.",
    },
}


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading AI model…")
def load_model():
    """Load model, class names, and optional metadata. Returns (model, class_names, metadata)."""
    if not os.path.exists("model/leather_model.h5"):
        return None, [], {}

    try:
        model = tf.keras.models.load_model("model/leather_model.h5")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, [], {}

    # Class names
    class_names = []
    if os.path.exists("model/class_names.txt"):
        with open("model/class_names.txt", "r") as f:
            class_names = [line.strip() for line in f if line.strip()]

    # Metadata (optional)
    metadata = {}
    if os.path.exists("model/metadata.json"):
        with open("model/metadata.json", "r") as f:
            metadata = json.load(f)

    return model, class_names, metadata


model, class_names, metadata = load_model()
MODEL_READY = model is not None and len(class_names) > 0


# ═══════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════

for key, default in [
    ("inspection_log",  []),
    ("total_inspected", 0),
    ("total_defective", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def get_defect_info(class_name: str, confidence: float) -> dict:
    """Return defect config for a class name, with dynamic quality score."""
    info = DEFECT_CONFIG.get(class_name)
    if not info:
        info = DEFECT_CONFIG.get(class_name.lower())
    if not info:
        for key in DEFECT_CONFIG:
            if key.lower() in class_name.lower() or class_name.lower() in key.lower():
                info = DEFECT_CONFIG[key]
                break

    if not info:
        sev_score = max(1, int((1 - confidence) * 8))
        info = {
            "label":          class_name.replace("_", " ").title(),
            "tag":            "REVIEW",
            "tag_color":      "#FFB800",
            "tag_bg":         "rgba(255,184,0,0.12)",
            "accent":         "#FFB800",
            "severity_level": "Medium",
            "severity_score":  sev_score,
            "quality_score":   sev_score,
            "msg":            f"Defect type '{class_name}' detected. Manual inspection recommended.",
            "action":         "⚠️  Manual inspection required",
            "action_color":   "#FFB800",
            "repair_cost":    "TBD",
            "repair_time":    "TBD",
            "recommended":    "Send to quality control team for manual evaluation.",
        }

    info = dict(info)  # copy so we can mutate
    base = info["quality_score"]
    if class_name.lower() == "good":
        info["dynamic_quality_score"] = round(base * (0.7 + 0.3 * confidence))
    else:
        info["dynamic_quality_score"] = max(1, round(base * (1 - confidence * 0.3)))
    return info


def run_prediction(image_pil: Image.Image):
    """Run EfficientNet prediction on a PIL image. Returns (preds, idx, conf)."""
    img       = image_pil.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    preds     = model.predict(img_array, verbose=0)[0]
    idx       = int(np.argmax(preds))
    conf      = float(preds[idx])
    return preds, idx, conf


def log_inspection(class_name: str, confidence: float, info: dict, source: str = "Upload"):
    entry = {
        "time":       datetime.datetime.now().strftime("%H:%M:%S"),
        "class":      class_name,
        "label":      info["label"],
        "confidence": round(confidence * 100, 1),
        "severity":   info["severity_level"],
        "score":      info["dynamic_quality_score"],
        "action":     info["tag"],
        "source":     source,
    }
    st.session_state.inspection_log.insert(0, entry)
    st.session_state.total_inspected += 1
    if class_name.lower() != "good":
        st.session_state.total_defective += 1
    st.session_state.inspection_log = st.session_state.inspection_log[:50]


def render_severity_bar(score: int, accent: str) -> str:
    filled = "█" * score
    empty  = "░" * (10 - score)
    return f"""
    <div style='display:flex; align-items:center; gap:12px; margin:8px 0;'>
        <span style='font-family:"Space Mono",monospace; font-size:1.1rem;
                     color:{accent}; letter-spacing:2px;'>{filled}{empty}</span>
        <span style='font-family:"Space Mono",monospace; font-size:0.9rem;
                     color:{accent}; font-weight:700;'>{score}/10</span>
    </div>
    """


def render_result_panel(preds, class_name: str, confidence: float, info: dict):
    """Full result panel used in the Upload tab."""
    accent = info["accent"]
    score  = info["dynamic_quality_score"]

    # ── Main card ──────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#13131A 0%,#1A1A24 100%);
                border:1px solid {accent}40; border-top:3px solid {accent};
                border-radius:12px; padding:28px 32px; margin-bottom:20px;
                box-shadow:0 4px 40px {accent}15;'>
        <div style='display:flex; justify-content:space-between; align-items:flex-start;
                    flex-wrap:wrap; gap:16px;'>
            <div>
                <div style='color:var(--muted); font-family:"Space Mono",monospace;
                            font-size:0.65rem; letter-spacing:0.2em; text-transform:uppercase;
                            margin-bottom:8px;'>ANALYSIS RESULT</div>
                <div style='font-family:"Syne",sans-serif; color:#E8E6E1;
                            font-size:1.9rem; font-weight:800; line-height:1.1;'>
                    {info["label"].upper()}
                </div>
                <div style='color:{accent}; font-size:0.85rem; margin-top:6px; font-weight:500;'>
                    {confidence*100:.1f}% model confidence
                </div>
            </div>
            <div style='background:{info["tag_bg"]}; border:1px solid {info["tag_color"]}40;
                        color:{info["tag_color"]}; font-family:"Space Mono",monospace;
                        font-size:0.75rem; font-weight:700; letter-spacing:0.15em;
                        text-transform:uppercase; padding:8px 18px; border-radius:4px;
                        align-self:flex-start;'>
                ● {info["tag"]}
            </div>
        </div>
        <div style='margin-top:20px; padding-top:20px; border-top:1px solid rgba(255,255,255,0.06);'>
            <div style='color:var(--muted); font-family:"Space Mono",monospace;
                        font-size:0.62rem; letter-spacing:0.15em; text-transform:uppercase;
                        margin-bottom:8px;'>QUALITY SCORE</div>
            {render_severity_bar(score, accent)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 4 metric columns ────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Severity Level", info["severity_level"])
    with c2: st.metric("Quality Score",  f"{score} / 10")
    with c3: st.metric("Repair Cost",    info.get("repair_cost", "—"))
    with c4: st.metric("Repair Time",    info.get("repair_time", "—"))

    # ── Analysis + Action ───────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div style='background:#13131A; border:1px solid rgba(255,255,255,0.07);
                    border-left:3px solid {accent}; border-radius:8px;
                    padding:20px 22px; height:100%;'>
            <div style='color:var(--muted); font-family:"Space Mono",monospace;
                        font-size:0.62rem; letter-spacing:0.15em; text-transform:uppercase;
                        margin-bottom:10px;'>DEFECT ANALYSIS</div>
            <p style='color:#C8C6C1; font-size:0.87rem; line-height:1.75; margin:0;'>
                {info["msg"]}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style='background:#13131A; border:1px solid rgba(255,255,255,0.07);
                    border-left:3px solid {accent}; border-radius:8px;
                    padding:20px 22px; height:100%;'>
            <div style='color:var(--muted); font-family:"Space Mono",monospace;
                        font-size:0.62rem; letter-spacing:0.15em; text-transform:uppercase;
                        margin-bottom:10px;'>RECOMMENDED ACTION</div>
            <div style='color:{info["action_color"]}; font-size:0.9rem;
                        font-weight:600; margin-bottom:12px;'>
                {info["action"]}
            </div>
            <p style='color:#C8C6C1; font-size:0.83rem; line-height:1.65; margin:0;'>
                {info.get("recommended", "Manual inspection required.")}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── Class Probabilities ─────────────────────────────────────
    st.markdown("""
    <div style='color:var(--muted); font-family:"Space Mono",monospace;
                font-size:0.62rem; letter-spacing:0.15em; text-transform:uppercase;
                margin-bottom:14px; margin-top:4px;'>CLASS PROBABILITIES</div>
    """, unsafe_allow_html=True)

    top_idx = int(np.argmax(preds))
    for i in np.argsort(preds)[::-1]:
        name   = class_names[i]
        prob   = float(preds[i]) * 100
        clr    = DEFECT_CONFIG.get(name, DEFECT_CONFIG.get(name.lower(), {})).get("accent", "#6B6B72")
        is_top = (i == top_idx)
        bar_w  = max(1, int(prob))

        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:14px; margin-bottom:10px;'>
            <span style='color:{"#E8E6E1" if is_top else "#6B6B72"};
                         font-size:0.82rem; font-weight:{"600" if is_top else "400"};
                         min-width:130px; text-transform:capitalize;'>{name}</span>
            <div style='flex:1; background:rgba(255,255,255,0.05); border-radius:2px;
                        height:6px; overflow:hidden;'>
                <div style='width:{bar_w}%; height:100%; background:{clr};
                            border-radius:2px; transition:width 0.8s ease;'></div>
            </div>
            <span style='color:{"#E8E6E1" if is_top else "#6B6B72"};
                         font-family:"Space Mono",monospace; font-size:0.8rem;
                         font-weight:{"700" if is_top else "400"};
                         min-width:52px; text-align:right;'>{prob:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)


def render_camera_result(info: dict, conf: float):
    """Compact result card used in the Camera tab."""
    accent = info["accent"]
    score  = info["dynamic_quality_score"]

    st.markdown(f"""
    <div style='background:#13131A; border:1px solid {accent}40;
                border-top:3px solid {accent}; border-radius:12px;
                padding:24px 28px; margin-bottom:16px;'>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div>
                <div style='font-family:"Syne",sans-serif; color:#E8E6E1;
                            font-size:1.4rem; font-weight:800;'>
                    {info["label"].upper()}
                </div>
                <div style='color:{accent}; font-size:0.82rem; margin-top:4px;'>
                    {conf*100:.1f}% confidence
                </div>
            </div>
            <div style='background:{info["tag_bg"]}; color:{info["tag_color"]};
                        font-family:"Space Mono",monospace; font-size:0.7rem;
                        font-weight:700; padding:7px 16px; border-radius:4px;
                        letter-spacing:0.12em;'>
                ● {info["tag"]}
            </div>
        </div>
        <div style='margin-top:16px; padding-top:16px;
                    border-top:1px solid rgba(255,255,255,0.06);'>
            <div style='color:#6B6B72; font-family:"Space Mono",monospace;
                        font-size:0.62rem; letter-spacing:0.12em; text-transform:uppercase;
                        margin-bottom:6px;'>QUALITY SCORE</div>
            {render_severity_bar(score, accent)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1: st.metric("Severity", info["severity_level"])
    with c2: st.metric("Score",    f"{score}/10")

    st.markdown(f"""
    <div style='background:#13131A; border:1px solid rgba(255,255,255,0.07);
                border-left:3px solid {accent}; border-radius:8px;
                padding:16px 20px; margin-top:12px;'>
        <div style='color:{info["action_color"]}; font-size:0.9rem;
                    font-weight:600; margin-bottom:8px;'>{info["action"]}</div>
        <p style='color:#C8C6C1; font-size:0.82rem; margin:0; line-height:1.65;'>
            {info.get("recommended", "Manual inspection required.")}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_analytics():
    """Full analytics dashboard."""
    log    = st.session_state.inspection_log
    total  = st.session_state.total_inspected
    defect = st.session_state.total_defective
    good   = total - defect

    st.markdown("""
    <div style='font-family:"Syne",sans-serif; font-size:1.4rem; font-weight:800;
                color:#E8E6E1; margin-bottom:20px; padding-bottom:16px;
                border-bottom:1px solid rgba(255,255,255,0.07);'>
        📊 Session Analytics
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Inspected", total)
    with c2: st.metric("Passed",          good,
                        delta=f"{good/total*100:.0f}%" if total > 0 else None)
    with c3: st.metric("Defective",       defect,
                        delta=f"-{defect/total*100:.0f}%" if total > 0 else None,
                        delta_color="inverse")
    with c4: st.metric("Pass Rate",       f"{good/total*100:.1f}%" if total > 0 else "—")

    if not log:
        st.markdown("""
        <div style='background:#13131A; border:1px solid rgba(255,255,255,0.07);
                    border-radius:8px; padding:60px; text-align:center; margin-top:20px;'>
            <div style='color:#6B6B72; font-size:0.9rem;'>
                No inspections yet. Upload or capture an image to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    df = pd.DataFrame(log)

    # ── Charts row 1 ────────────────────────────────────────────
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown(_chart_label("DEFECT CLASS DISTRIBUTION"), unsafe_allow_html=True)
        class_counts = df["class"].value_counts()
        bar_colors   = [DEFECT_CONFIG.get(c, {}).get("accent", "#6B6B72") for c in class_counts.index]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        _style_ax(fig, ax)
        bars = ax.bar(class_counts.index, class_counts.values,
                      color=bar_colors, edgecolor="none", width=0.5)
        ax.set_xlabel("Defect Class", color="#6B6B72", fontsize=9)
        ax.set_ylabel("Count",        color="#6B6B72", fontsize=9)
        ax.tick_params(colors="#6B6B72", labelsize=8)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, class_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    str(val), ha="center", va="bottom", color="#E8E6E1", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_chart2:
        st.markdown(_chart_label("QUALITY SCORE OVER TIME"), unsafe_allow_html=True)
        recent  = list(reversed(log[-20:]))
        scores  = [e["score"] for e in recent]
        pt_clrs = ["#00E676" if s >= 7 else "#FFB800" if s >= 4 else "#FF3B3B" for s in scores]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        _style_ax(fig, ax)
        ax.plot(range(len(scores)), scores, color="#C8FF00", linewidth=1.5, alpha=0.6, zorder=1)
        ax.scatter(range(len(scores)), scores, c=pt_clrs, s=50, zorder=2, edgecolors="none")
        ax.axhline(y=7, color="#00E676", linestyle="--", alpha=0.3, linewidth=1)
        ax.axhline(y=4, color="#FF3B3B", linestyle="--", alpha=0.3, linewidth=1)
        ax.set_ylim(0, 11)
        ax.set_ylabel("Quality Score", color="#6B6B72", fontsize=9)
        ax.set_xlabel("Inspection #",  color="#6B6B72", fontsize=9)
        ax.tick_params(colors="#6B6B72", labelsize=8)
        ax.set_xticks([])
        patches = [
            mpatches.Patch(color="#00E676", label="Pass (7–10)"),
            mpatches.Patch(color="#FFB800", label="Review (4–6)"),
            mpatches.Patch(color="#FF3B3B", label="Reject (1–3)"),
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=7,
                  facecolor="#1A1A24", edgecolor="none", labelcolor="#C8C6C1")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Charts row 2 ────────────────────────────────────────────
    col_pie, col_conf = st.columns(2)

    with col_pie:
        st.markdown(_chart_label("SEVERITY BREAKDOWN"), unsafe_allow_html=True)
        sev_counts = df["severity"].value_counts()
        sev_colors = {
            "None": "#00E676", "Low": "#C8FF00", "Medium": "#FFB800",
            "High": "#FF3B3B", "Critical": "#FF3B3B", "Unknown": "#6B6B72"
        }
        pie_colors = [sev_colors.get(s, "#6B6B72") for s in sev_counts.index]

        fig, ax = plt.subplots(figsize=(4, 3.5))
        _style_ax(fig, ax)
        wedges, texts, autotexts = ax.pie(
            sev_counts.values, labels=sev_counts.index,
            colors=pie_colors, autopct="%1.0f%%", pctdistance=0.75, startangle=90,
            wedgeprops=dict(edgecolor="#13131A", linewidth=2)
        )
        for t in texts:     t.set_color("#6B6B72"); t.set_fontsize(8)
        for t in autotexts: t.set_color("#0A0A0B"); t.set_fontsize(8); t.set_fontweight("bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_conf:
        st.markdown(_chart_label("CONFIDENCE DISTRIBUTION"), unsafe_allow_html=True)
        confs = df["confidence"].values

        fig, ax = plt.subplots(figsize=(4, 3.5))
        _style_ax(fig, ax)
        ax.hist(confs, bins=10, range=(0, 100), color="#C8FF00", edgecolor="#13131A", alpha=0.85)
        ax.set_xlabel("Confidence %", color="#6B6B72", fontsize=9)
        ax.set_ylabel("Frequency",    color="#6B6B72", fontsize=9)
        ax.tick_params(colors="#6B6B72", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Inspection log table ─────────────────────────────────────
    st.markdown(_chart_label("RECENT INSPECTION LOG", top_margin="24px"), unsafe_allow_html=True)

    log_df = pd.DataFrame(log[:15])
    if not log_df.empty:
        log_df = log_df[["time", "label", "confidence", "severity", "score", "action", "source"]]
        log_df.columns = ["Time", "Defect", "Conf %", "Severity", "Score", "Decision", "Source"]
        st.dataframe(log_df, use_container_width=True, hide_index=True)

        csv = log_df.to_csv(index=False)
        st.download_button(
            "⬇  Export Log as CSV",
            data=csv,
            file_name=f"inspection_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _style_ax(fig, ax):
    fig.patch.set_facecolor("#13131A")
    ax.set_facecolor("#13131A")
    for spine in ax.spines.values():
        spine.set_edgecolor((1, 1, 1, 0.07))


def _chart_label(text: str, top_margin: str = "20px") -> str:
    return f"""
    <div style='color:var(--muted); font-family:"Space Mono",monospace; font-size:0.65rem;
                letter-spacing:0.15em; text-transform:uppercase; margin:{top_margin} 0 10px 0;'>
        {text}
    </div>"""


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding:24px 20px 16px 20px;'>
        <div style='font-family:"Space Mono",monospace; color:#C8FF00;
                    font-size:0.65rem; letter-spacing:0.25em; text-transform:uppercase;
                    margin-bottom:6px;'>LEATHER AI</div>
        <div style='font-family:"Syne",sans-serif; color:#E8E6E1;
                    font-size:1.5rem; font-weight:800; line-height:1.2;'>
            Defect Intelligence<br>System
        </div>
        <div style='color:#6B6B72; font-size:0.75rem; margin-top:8px;'>
            v2.0 — Production QC
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:0 0 16px 0;'>",
                unsafe_allow_html=True)

    # ── Model status ─────────────────────────────────────────────
    if MODEL_READY:
        status_color = "#00E676"
        status_text  = "● MODEL LOADED"
    else:
        status_color = "#FF3B3B"
        status_text  = "● MODEL NOT FOUND"

    st.markdown(f"""
    <div style='padding:0 20px 12px 20px;'>
        <span style='color:{status_color}; font-family:"Space Mono",monospace;
                     font-size:0.65rem; letter-spacing:0.15em;'>{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Session stats ─────────────────────────────────────────────
    total  = st.session_state.total_inspected
    defect = st.session_state.total_defective
    rate   = f"{(total-defect)/total*100:.0f}%" if total > 0 else "—"

    st.markdown(f"""
    <div style='padding:0 20px 16px 20px;'>
        <div style='color:#6B6B72; font-family:"Space Mono",monospace; font-size:0.62rem;
                    letter-spacing:0.15em; text-transform:uppercase; margin-bottom:12px;'>
            SESSION STATS
        </div>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px;'>
            <div style='background:#1A1A24; border-radius:6px; padding:10px 12px;'>
                <div style='color:#6B6B72; font-size:0.65rem; margin-bottom:4px;'>Inspected</div>
                <div style='color:#E8E6E1; font-family:"Space Mono",monospace;
                            font-size:1.1rem; font-weight:700;'>{total}</div>
            </div>
            <div style='background:#1A1A24; border-radius:6px; padding:10px 12px;'>
                <div style='color:#6B6B72; font-size:0.65rem; margin-bottom:4px;'>Pass Rate</div>
                <div style='color:#C8FF00; font-family:"Space Mono",monospace;
                            font-size:1.1rem; font-weight:700;'>{rate}</div>
            </div>
            <div style='background:#1A1A24; border-radius:6px; padding:10px 12px;'>
                <div style='color:#6B6B72; font-size:0.65rem; margin-bottom:4px;'>Defective</div>
                <div style='color:#FF6B35; font-family:"Space Mono",monospace;
                            font-size:1.1rem; font-weight:700;'>{defect}</div>
            </div>
            <div style='background:#1A1A24; border-radius:6px; padding:10px 12px;'>
                <div style='color:#6B6B72; font-size:0.65rem; margin-bottom:4px;'>Passed</div>
                <div style='color:#00E676; font-family:"Space Mono",monospace;
                            font-size:1.1rem; font-weight:700;'>{total - defect}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:0 0 16px 0;'>",
                unsafe_allow_html=True)

    # ── Defect classes ────────────────────────────────────────────
    st.markdown("""
    <div style='padding:0 20px;'>
        <div style='color:#6B6B72; font-family:"Space Mono",monospace; font-size:0.62rem;
                    letter-spacing:0.15em; text-transform:uppercase; margin-bottom:12px;'>
            DEFECT CLASSES
        </div>
    </div>
    """, unsafe_allow_html=True)

    severity_icons = {
        "None": "●", "Low": "●", "Medium": "◆",
        "High": "▲", "Critical": "■", "Unknown": "○"
    }
    for name, cfg in DEFECT_CONFIG.items():
        icon = severity_icons.get(cfg["severity_level"], "●")
        st.markdown(f"""
        <div style='display:flex; justify-content:space-between; align-items:center;
                    padding:9px 20px; border-left:2px solid {cfg["accent"]}; margin-bottom:3px;'>
            <span style='color:#C8C6C1; font-size:0.82rem;'>
                {icon} {name.replace("_"," ").title()}
            </span>
            <span style='color:{cfg["tag_color"]}; font-family:"Space Mono",monospace;
                         font-size:0.65rem; font-weight:700; letter-spacing:0.05em;'>
                {cfg["severity_level"]}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06); margin:16px 0;'>",
                unsafe_allow_html=True)

    # ── Model info ────────────────────────────────────────────────
    val_acc = metadata.get("phase2_best_val_acc", metadata.get("phase1_best_val_acc"))
    arch    = metadata.get("model_architecture", "EfficientNetB0")
    classes = len(class_names) if class_names else "—"

    acc_html = (
        f"Validation Accuracy<br>"
        f"<span style='color:#C8FF00; font-family:\"Space Mono\",monospace; font-size:0.9rem;'>"
        f"{float(val_acc)*100:.1f}%</span><br>"
    ) if val_acc else ""

    st.markdown(f"""
    <div style='padding:0 20px 20px 20px;'>
        <div style='color:#6B6B72; font-family:"Space Mono",monospace; font-size:0.62rem;
                    letter-spacing:0.15em; text-transform:uppercase; margin-bottom:10px;'>
            MODEL INFO
        </div>
        <div style='color:#6B6B72; font-size:0.8rem; line-height:2.2;'>
            Architecture<br>
            <span style='color:#C8C6C1;'>{arch}</span><br>
            Classes<br>
            <span style='color:#C8C6C1;'>{classes}</span><br>
            {acc_html}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑  Clear Session Log"):
        st.session_state.inspection_log  = []
        st.session_state.total_inspected = 0
        st.session_state.total_defective = 0
        st.rerun()


# ═══════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div style='margin-bottom:2rem; padding-bottom:1.5rem;
            border-bottom:1px solid rgba(255,255,255,0.06);'>
    <div style='color:#C8FF00; font-family:"Space Mono",monospace; font-size:0.65rem;
                letter-spacing:0.25em; text-transform:uppercase; margin-bottom:10px;'>
        ◈ QUALITY CONTROL — COMPUTER VISION INSPECTION
    </div>
    <h1 style='font-family:"Syne",sans-serif; color:#E8E6E1; font-size:2.8rem;
               font-weight:800; margin:0; line-height:1.1;'>
        Leather Defect<br>Intelligence
    </h1>
    <p style='color:#6B6B72; font-size:0.9rem; margin-top:12px; max-width:600px;'>
        AI-powered defect classification with severity scoring, live camera inspection,
        and production analytics — powered by EfficientNetB0.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Model-not-found warning ───────────────────────────────────────────────────
if not MODEL_READY:
    st.markdown("""
    <div style='background:rgba(255,59,59,0.08); border:1px solid rgba(255,59,59,0.35);
                border-left:3px solid #FF3B3B; border-radius:8px;
                padding:16px 20px; margin-bottom:24px;'>
        <div style='color:#FF3B3B; font-family:"Space Mono",monospace; font-size:0.7rem;
                    letter-spacing:0.1em; font-weight:700; margin-bottom:6px;'>
            ■ MODEL FILES NOT FOUND
        </div>
        <p style='color:#C8C6C1; font-size:0.85rem; margin:0; line-height:1.75;'>
            Place the following files inside a <code style="color:#C8FF00;">model/</code>
            folder next to <code style="color:#C8FF00;">app.py</code>:<br>
            <code style="color:#FFB800;">model/leather_model.h5</code> — trained EfficientNetB0 weights<br>
            <code style="color:#FFB800;">model/class_names.txt</code> — one class per line
            (e.g. good, cut, fold, poke, grain, color, folding marks)<br>
            <code style="color:#6B6B72;">model/metadata.json</code> — optional training metadata
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "  📁  UPLOAD INSPECTION  ",
    "  📷  LIVE CAMERA  ",
    "  📊  ANALYTICS DASHBOARD  ",
])


# ══════════════════════════════════════════════════════════════
#  TAB 1 — UPLOAD
# ══════════════════════════════════════════════════════════════

with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("""
        <div style='color:#6B6B72; font-family:"Space Mono",monospace; font-size:0.65rem;
                    letter-spacing:0.15em; text-transform:uppercase;
                    margin-bottom:10px; margin-top:20px;'>
            INPUT IMAGE
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="upload_tab",
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; margin-top:8px;
                        padding:8px 0; border-top:1px solid rgba(255,255,255,0.06);'>
                <span style='color:#6B6B72; font-family:"Space Mono",monospace;
                             font-size:0.75rem;'>{uploaded.name}</span>
                <span style='color:#6B6B72; font-family:"Space Mono",monospace;
                             font-size:0.75rem;'>{round(uploaded.size/1024, 1)} KB</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
            analyze = st.button("⟳  RUN DEFECT ANALYSIS", key="analyze_upload",
                                disabled=not MODEL_READY)
            st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if not uploaded:
            st.markdown("""
            <div style='background:#13131A; border:1px solid rgba(255,255,255,0.07);
                        border-radius:12px; padding:80px 32px; text-align:center;
                        min-height:480px; display:flex; flex-direction:column;
                        justify-content:center; align-items:center; margin-top:20px;'>
                <div style='width:60px; height:60px; border:1px solid rgba(200,255,0,0.3);
                            border-radius:50%; margin:0 auto 20px auto; display:flex;
                            align-items:center; justify-content:center; font-size:1.5rem;'>🔬</div>
                <p style='color:#6B6B72; font-size:0.88rem; margin:0; line-height:1.8;'>
                    Upload a leather image to begin AI inspection
                </p>
                <div style='margin-top:24px; display:flex; gap:8px; flex-wrap:wrap; justify-content:center;'>
                    <span style='background:#1A1A24; color:#C8FF00; padding:5px 14px;
                                 border-radius:4px; font-size:0.72rem;
                                 font-family:"Space Mono",monospace;
                                 border:1px solid rgba(200,255,0,0.2);'>Bags</span>
                    <span style='background:#1A1A24; color:#C8FF00; padding:5px 14px;
                                 border-radius:4px; font-size:0.72rem;
                                 font-family:"Space Mono",monospace;
                                 border:1px solid rgba(200,255,0,0.2);'>Wallets</span>
                    <span style='background:#1A1A24; color:#C8FF00; padding:5px 14px;
                                 border-radius:4px; font-size:0.72rem;
                                 font-family:"Space Mono",monospace;
                                 border:1px solid rgba(200,255,0,0.2);'>Shoes</span>
                    <span style='background:#1A1A24; color:#C8FF00; padding:5px 14px;
                                 border-radius:4px; font-size:0.72rem;
                                 font-family:"Space Mono",monospace;
                                 border:1px solid rgba(200,255,0,0.2);'>Belts</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif uploaded and "analyze" in dir() and analyze:
            with st.spinner("🔬 Analyzing…"):
                time.sleep(0.3)
                preds, idx, conf = run_prediction(image)
            class_name = class_names[idx]
            info       = get_defect_info(class_name, conf)
            log_inspection(class_name, conf, info, source="Upload")
            render_result_panel(preds, class_name, conf, info)

        elif uploaded:
            st.markdown("""
            <div style='background:#13131A; border:1px solid rgba(255,255,255,0.07);
                        border-radius:12px; padding:60px 32px; text-align:center;
                        min-height:300px; display:flex; flex-direction:column;
                        justify-content:center; align-items:center; margin-top:20px;'>
                <p style='color:#6B6B72; font-size:0.88rem; margin:0;'>
                    Click <strong style="color:#C8FF00;">RUN DEFECT ANALYSIS</strong> to inspect
                </p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 2 — LIVE CAMERA
# ══════════════════════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div style='color:#6B6B72; font-family:"Space Mono",monospace; font-size:0.65rem;
                letter-spacing:0.15em; text-transform:uppercase; margin:20px 0 16px 0;'>
        LIVE CAMERA INSPECTION
    </div>
    <div style='background:#13131A; border:1px solid rgba(200,255,0,0.15);
                border-left:3px solid #C8FF00; border-radius:8px;
                padding:14px 18px; margin-bottom:20px; font-size:0.85rem; color:#C8C6C1;'>
        📷 Position the leather sample in front of your camera, then click
        <strong style="color:#C8FF00;">Capture &amp; Analyze</strong> to inspect it instantly.
    </div>
    """, unsafe_allow_html=True)

    cam_col, result_col = st.columns([1, 1], gap="large")

    with cam_col:
        cam_mode     = st.radio("Camera Mode",
                                ["Single Capture", "Auto-Inspect Mode"],
                                horizontal=True,
                                label_visibility="collapsed")
        camera_image = st.camera_input("", label_visibility="collapsed")

        cam_img      = None
        analyze_cam  = False

        if camera_image:
            cam_img = Image.open(camera_image).convert("RGB")
            if cam_mode == "Single Capture":
                analyze_cam = st.button("⟳  CAPTURE & ANALYZE",
                                        key="analyze_cam",
                                        disabled=not MODEL_READY)

    with result_col:
        if not camera_image:
            st.markdown("""
            <div style='background:#13131A; border:1px dashed rgba(255,255,255,0.1);
                        border-radius:12px; padding:60px 32px; text-align:center;
                        min-height:400px; display:flex; flex-direction:column;
                        justify-content:center; align-items:center;'>
                <div style='font-size:2rem; margin-bottom:16px;'>📷</div>
                <p style='color:#6B6B72; font-size:0.88rem; margin:0; line-height:1.8;'>
                    Camera feed will appear on the left.<br>
                    Capture and analyze frames in real-time.
                </p>
            </div>
            """, unsafe_allow_html=True)

        elif camera_image and MODEL_READY:
            should_run = analyze_cam or (cam_mode == "Auto-Inspect Mode")

            if should_run and cam_img is not None:
                source_tag = "Camera-Auto" if cam_mode == "Auto-Inspect Mode" else "Camera"
                with st.spinner("🔬 Analyzing frame…"):
                    preds, idx, conf = run_prediction(cam_img)
                class_name = class_names[idx]
                info       = get_defect_info(class_name, conf)
                log_inspection(class_name, conf, info, source=source_tag)
                render_camera_result(info, conf)

            elif not should_run:
                st.markdown("""
                <div style='background:#13131A; border:1px solid rgba(255,255,255,0.07);
                            border-radius:12px; padding:50px 32px; text-align:center;'>
                    <p style='color:#6B6B72; font-size:0.88rem; margin:0;'>
                        Click <strong style="color:#C8FF00;">CAPTURE &amp; ANALYZE</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

        elif camera_image and not MODEL_READY:
            st.markdown("""
            <div style='background:rgba(255,59,59,0.08); border:1px solid rgba(255,59,59,0.3);
                        border-radius:8px; padding:24px; text-align:center;'>
                <p style='color:#FF3B3B; font-size:0.88rem; margin:0;'>
                    Model not loaded. Add model files to enable analysis.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Auto-inspect info banner ──────────────────────────────────
    if cam_mode == "Auto-Inspect Mode":
        st.markdown("""
        <div style='background:rgba(200,255,0,0.05); border:1px solid rgba(200,255,0,0.2);
                    border-radius:8px; padding:14px 18px; margin-top:16px;
                    color:#C8C6C1; font-size:0.83rem;'>
            ⚡ <strong style="color:#C8FF00;">Auto-Inspect Mode:</strong>
            Each new camera capture is analyzed automatically — no button click needed.
            Ideal for high-throughput production line inspection.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  TAB 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════

with tab3:
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    render_analytics()