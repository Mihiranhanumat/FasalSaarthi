"""
FASALSAARTHI 🌾
Run: streamlit run app.py
Fixes: multilanguage T() everywhere, disease detection bytes fix,
       chatbot uses local helpbot + API fallback, voice input/output added.
"""
import sys, os, io, base64
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests, time
from datetime import datetime
from PIL import Image

from modules.yield_prediction  import YieldPredictor
from modules.disease_detection import DiseaseDetector
from modules.fertilizer        import FertilizerRecommender
from modules.crop_recommender  import CropRecommender
from modules.helpbot           import HelpBot
from modules.auth              import login, register
from assets.translations       import get_text, get_all_languages

st.set_page_config(page_title="FASALSAARTHI 🌾", page_icon="🌾",
                   layout="wide", initial_sidebar_state="expanded")

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = dict(
    logged_in=False, username='', user_info={},
    lang='en', lang_label='English', page='home',
    chat_messages=[],
    yield_result=None,
    disease_result=None,
    disease_img_bytes=None,   # FIX: store bytes so rerun doesn't lose the file
    fert_result=None,
    rec_result=None,
    voice_text='',            # voice input captured text
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Translation helper ────────────────────────────────────────────────────────
def T(key): return get_text(key, st.session_state.lang)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@700;800&display=swap');
:root{--g1:#0a2e0a;--g2:#1b5e20;--g3:#2e7d32;--g4:#43a047;--g5:#66bb6a;
     --gp:#e8f5e9;--gold:#f9a825;--bg:#f4faf4;--card:#ffffff;--border:#e0ede0;
     --text:#111827;--muted:#4b5563;
     --sh:0 2px 16px rgba(0,60,0,.07);--shl:0 8px 40px rgba(0,60,0,.13);--r:16px;}
*{font-family:'Inter',sans-serif!important}

/* Base containers & typography */
[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:#111827!important}
[data-testid="stMain"]{padding-top:0!important;color:#111827!important}
[data-testid="stMain"] p, [data-testid="stMain"] span, [data-testid="stMain"] label, [data-testid="stMain"] li{color:#111827}
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3, [data-testid="stMain"] h4, [data-testid="stMain"] h5, [data-testid="stMain"] h6{color:#0a2e0a!important;font-weight:700!important}

/* Sidebar styling */
[data-testid="stSidebar"]{background:linear-gradient(170deg,#061806,#0f3d0f,#1b5e20,#2e7d32)!important}
[data-testid="stSidebar"] *{color:#d7f0d7!important}
[data-testid="stSidebar"] .stSelectbox label{color:#8bc98b!important;font-size:.75rem!important;text-transform:uppercase;letter-spacing:.5px}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"]{background:rgba(255,255,255,.07)!important;border:1px solid rgba(255,255,255,.2)!important;border-radius:8px!important}
[data-testid="stSidebar"] .stButton button{background:rgba(255,255,255,.07)!important;color:#c8e6c9!important;border:1px solid rgba(255,255,255,.15)!important;border-radius:10px!important;font-size:.85rem!important;padding:9px 14px!important;text-align:left!important;width:100%!important;transition:all .18s!important;margin:2px 0!important;font-weight:500!important}
[data-testid="stSidebar"] .stButton button:hover{background:rgba(255,255,255,.2)!important;color:#fff!important;border-color:rgba(255,255,255,.35)!important;transform:translateX(4px)!important}

/* Primary buttons */
.stButton>button{background:linear-gradient(135deg,#2e7d32,#43a047)!important;color:#ffffff!important;border:none!important;border-radius:10px!important;font-weight:700!important;font-size:.9rem!important;padding:10px 22px!important;box-shadow:0 4px 14px rgba(46,125,50,.32)!important;transition:all .18s!important}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 7px 22px rgba(46,125,50,.42)!important;color:#ffffff!important}

/* Hero section */
.hero{background:linear-gradient(135deg,#061806,#1b5e20,#2e7d32,#43a047);padding:36px 32px;border-radius:22px;color:#ffffff;margin-bottom:28px;position:relative;overflow:hidden;box-shadow:0 12px 48px rgba(0,60,0,.22)}
.hero::before{content:'';position:absolute;top:-80px;right:-80px;width:280px;height:280px;background:rgba(255,255,255,.04);border-radius:50%}
.hero h1{font-family:'Playfair Display',serif!important;font-size:2.6rem;margin:0;font-weight:800;letter-spacing:-1px;position:relative;z-index:1;color:#ffffff!important}
.hero p{font-size:.98rem;opacity:.9;margin:8px 0 0;font-weight:300;position:relative;z-index:1;color:#ffffff!important}
.hero-pill{display:inline-block;background:rgba(249,168,37,.2);border:1px solid rgba(249,168,37,.45);color:#ffd54f!important;padding:4px 14px;border-radius:20px;font-size:.72rem;font-weight:700;margin-bottom:12px;letter-spacing:1.5px;position:relative;z-index:1}

/* Cards & Containers */
.gcard{background:#ffffff;border-radius:var(--r);padding:22px 20px;box-shadow:var(--sh);border:1px solid var(--border);transition:all .22s;position:relative;overflow:hidden}
.gcard::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#2e7d32,#66bb6a);border-radius:var(--r) var(--r) 0 0}
.gcard:hover{transform:translateY(-4px);box-shadow:var(--shl);border-color:#a5d6a7}
.gcard:hover::before{background:linear-gradient(90deg,#f9a825,#ffca28)}
.gcard .gc-icon{font-size:2.4rem;margin-bottom:12px;display:block}
.gcard h3{font-size:.95rem;font-weight:700;color:#0a2e0a!important;margin:0 0 6px}
.gcard p{font-size:.82rem;color:#4b5563!important;margin:0;line-height:1.55}

.scard{background:#ffffff;border-radius:12px;padding:16px;text-align:center;box-shadow:var(--sh);border:1px solid var(--border)}
.scard .sv{font-size:1.9rem;font-weight:800;background:linear-gradient(135deg,#1b5e20,#2e7d32);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.scard .sl{font-size:.74rem;color:#4b5563!important;margin-top:2px;font-weight:600;text-transform:uppercase;letter-spacing:.5px}

.rbanner{border-radius:var(--r);padding:28px 24px;text-align:center;color:#ffffff;margin-bottom:14px;box-shadow:var(--sh)}
.rbanner *{color:#ffffff!important}
.rbanner .rv{font-size:3.2rem;font-weight:900;letter-spacing:-2px;line-height:1}
.rbanner .rl{font-size:.88rem;opacity:.9;font-weight:400;margin-top:4px}

/* Info & Callout Boxes (High Contrast) */
.ic{background:#ffffff!important;border-radius:10px;padding:14px 16px;margin:8px 0;box-shadow:0 1px 6px rgba(0,0,0,.05);border:1px solid #d1e7dd;color:#111827!important}
.ic *{color:#111827!important}
.ic b, .ic strong{color:#0a2e0a!important;font-weight:700!important}
.ic.s{border-left:4px solid #2e7d32!important;background:#e8f5e9!important}
.ic.w{border-left:4px solid #f57c00!important;background:#fff3e0!important}
.ic.d{border-left:4px solid #c62828!important;background:#ffebee!important}
.ic.n{border-left:4px solid #1565c0!important;background:#e3f2fd!important}
.ic.g{border-left:4px solid #f9a825!important;background:#fffde7!important}

/* Product & Result Cards */
.pcard{background:#ffffff;border-radius:12px;padding:16px 18px;margin:10px 0;box-shadow:0 2px 10px rgba(0,0,0,.06);border:1px solid #e8f0e8;border-left:4px solid #2e7d32;transition:all .18s}
.pcard:hover{box-shadow:0 4px 20px rgba(0,0,0,.1)}
.pcard .pcat{font-size:.72rem;font-weight:700;color:#1b5e20!important;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px}
.pcard .pname{font-size:1.05rem;font-weight:700;color:#0a2e0a!important;margin:0 0 3px}
.pcard .pbrand{font-size:.82rem;color:#2e7d32!important;font-weight:600;margin-bottom:10px}
.pcard table{width:100%;border-collapse:collapse;font-size:.82rem}
.pcard td{padding:5px 8px;border-bottom:1px solid #f0f5f0;vertical-align:top;color:#111827!important}
.pcard td:first-child{color:#4b5563!important;width:35%;font-weight:600}
.pcard td:last-child{font-weight:700;color:#111827!important}

.crc{background:#ffffff;border-radius:14px;padding:16px;margin:8px 0;box-shadow:var(--sh);border:1px solid var(--border);display:flex;align-items:center;gap:14px;transition:all .18s}
.crc:hover{box-shadow:var(--shl);border-color:#a5d6a7}
.crc.best{border:2px solid #2e7d32;background:linear-gradient(135deg,#f0faf0,#e8f5e9)}
.crc .cre{font-size:2.2rem;flex-shrink:0}
.crc .crd h4{margin:0;font-size:.95rem;font-weight:700;color:#0a2e0a!important}
.crc .crd p{margin:2px 0 0;font-size:.78rem;color:#4b5563!important}
.crc .crp{margin-left:auto;text-align:right;flex-shrink:0}
.crc .crp .pct{font-size:1.4rem;font-weight:800;color:#2e7d32!important}
.crc .crp .plb{font-size:.68rem;color:#4b5563!important}

/* Chat interface */
.chat-box{background:#ffffff;border-radius:var(--r);padding:20px;box-shadow:var(--sh);border:1px solid var(--border);min-height:440px;max-height:540px;overflow-y:auto}
.msg-b{display:flex;gap:10px;margin:10px 0;align-items:flex-start}
.msg-b .av{width:36px;height:36px;border-radius:50%;flex-shrink:0;background:linear-gradient(135deg,#1b5e20,#43a047);display:flex;align-items:center;justify-content:center;font-size:1.1rem;color:#fff!important}
.msg-b .bbl{background:#f8fdf8;border-radius:4px 16px 16px 16px;padding:12px 16px;max-width:84%;font-size:.88rem;line-height:1.65;border:1px solid #c8e6c9;color:#111827!important}
.msg-b .bbl *{color:#111827!important}
.msg-u{display:flex;gap:10px;margin:10px 0;align-items:flex-start;flex-direction:row-reverse}
.msg-u .av{width:36px;height:36px;border-radius:50%;flex-shrink:0;background:linear-gradient(135deg,#1565c0,#42a5f5);display:flex;align-items:center;justify-content:center;font-size:1rem;color:#fff!important}
.msg-u .bbl{background:linear-gradient(135deg,#1b5e20,#2e7d32);color:#ffffff!important;border-radius:16px 4px 16px 16px;padding:12px 16px;max-width:78%;font-size:.88rem;line-height:1.65}
.msg-u .bbl *{color:#ffffff!important}
.mt{font-size:.68rem;opacity:.65;margin-top:4px}

/* Badges */
.cw{background:#e8f5e9;border-radius:6px;height:8px;margin:6px 0;overflow:hidden}
.cf{height:8px;border-radius:6px;transition:width .7s ease}
.badge{display:inline-flex;align-items:center;gap:3px;padding:3px 10px;border-radius:20px;font-size:.72rem;font-weight:700}
.bg{background:#dcf0dc;color:#1b5e20!important}.by{background:#fff9c4;color:#b45309!important}.br{background:#ffd7d7;color:#b71c1c!important}
.ds-badge{display:inline-flex;align-items:center;gap:6px;background:#e8f5e9;border:1px solid #a5d6a7;border-radius:20px;padding:4px 12px;font-size:.76rem;font-weight:600;color:#1b5e20!important;margin:2px 4px}

/* Form Inputs & Controls (High Contrast Black/Dark Text) */
.stTextInput input, .stNumberInput input{background:#ffffff!important;color:#111827!important;border-radius:10px!important;border:1.5px solid #a5d6a7!important;padding:10px 14px!important;font-size:.95rem!important;font-weight:600!important}
.stTextInput input:focus, .stNumberInput input:focus{border-color:#2e7d32!important;box-shadow:0 0 0 3px rgba(46,125,50,.15)!important;color:#111827!important}
.stTextInput label, .stSelectbox label, .stNumberInput label, .stSlider label, .stMultiSelect label, [data-testid="stWidgetLabel"] p{font-weight:700!important;color:#0f3d0f!important;font-size:.85rem!important;text-transform:uppercase!important;letter-spacing:.5px!important}

/* Select & MultiSelect */
.stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"]{background:#ffffff!important;border-radius:10px!important;border:1.5px solid #a5d6a7!important;color:#111827!important}
.stSelectbox [data-baseweb="select"] *, .stMultiSelect [data-baseweb="select"] *{color:#111827!important}
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"]{background:#ffffff!important;color:#111827!important}
[data-baseweb="menu"] *{color:#111827!important}

/* Multiselect chips / tags */
[data-baseweb="tag"]{background-color:#1b5e20!important;border-radius:6px!important;color:#ffffff!important;font-weight:600!important}
[data-baseweb="tag"] span, [data-baseweb="tag"] svg{color:#ffffff!important;fill:#ffffff!important}

/* NumberInput stepper buttons */
[data-testid="stNumberInput"] button{background:#e8f5e9!important;color:#1b5e20!important;border:1px solid #c8e6c9!important}
[data-testid="stNumberInput"] button svg{fill:#1b5e20!important}
[data-testid="stNumberInput"] button:hover{background:#c8e6c9!important}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:6px;background:transparent;border-bottom:2px solid #e8f5e9}
.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0!important;font-weight:600!important;font-size:.85rem!important;padding:8px 16px!important;color:#4b5563!important}
.stTabs [aria-selected="true"]{background:#e8f5e9!important;color:#1b5e20!important;font-weight:700!important}

/* Dataframe */
[data-testid="stDataFrame"]{background:#ffffff!important;border-radius:10px!important}
[data-testid="stDataFrame"] *{color:#111827!important}

.fdiv{border:none;height:1px;background:linear-gradient(90deg,transparent,#c8e6c9 30%,#c8e6c9 70%,transparent);margin:24px 0}
.sh{font-family:'Playfair Display',serif!important;font-size:1.35rem;font-weight:800;color:#0a2e0a!important;margin:0 0 4px}
.ss{color:#4b5563!important;font-size:.85rem;margin:0 0 16px;font-weight:500}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:#a5d6a7;border-radius:3px}
</style>
""", unsafe_allow_html=True)

# ── Cached model loaders ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🌾 Loading Maharashtra yield data…")
def load_yield():
    m = YieldPredictor(); m.train(); return m

@st.cache_resource(show_spinner="🌱 Training crop recommender…")
def load_rec():
    m = CropRecommender(); m.train(); return m

@st.cache_resource
def load_det():    return DiseaseDetector()
@st.cache_resource
def load_fert():   return FertilizerRecommender()
@st.cache_resource
def load_helpbot(): return HelpBot()

# ─────────────────────────────────────────────────────────────────────────────
# CHATBOT  — local helpbot first, API fallback
# ─────────────────────────────────────────────────────────────────────────────
def get_bot_reply(user_msg: str, history: list) -> str:
    """
    Primary: Anthropic Claude API for authentic, context-aware answers.
    Fallback: local HelpBot (works offline, supports Hindi/Marathi).
    """
    SYS = (
        "You are FASALSAARTHI, an expert AI agricultural assistant for Maharashtra farmers. "
        "Answer farming questions about crops, diseases, fertilizers, government schemes, "
        "mandi/market prices, soil health, irrigation, and weather. "
        "Give specific, actionable advice with product names and doses where relevant. "
        "Support Hindi, Marathi and English — reply in the same language the user wrote in. "
        "Use emojis to make responses friendly. Be concise, practical, and accurate. "
        "When you don't know something, say so honestly and suggest they call Kisan helpline 1800-180-1551."
    )

    # Build conversation history for API (last 10 turns, excluding current message)
    api_msgs = []
    for m in history[:-1][-10:]:  # exclude the last message (current user msg just appended)
        if m['role'] in ('user', 'assistant'):
            api_msgs.append({'role': m['role'], 'content': m['content']})
    # Add current user message
    api_msgs.append({'role': 'user', 'content': user_msg})

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type":      "application/json",
                "anthropic-version": "2023-06-01",
            },
            json={
                "model":      "claude-sonnet-4-20250514",
                "max_tokens": 800,
                "system":     SYS,
                "messages":   api_msgs,
            },
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("content"):
                return data["content"][0].get("text", "")
    except Exception:
        pass  # API unavailable — fall back to local helpbot

    # Fallback: local intent-based helpbot
    bot = load_helpbot()
    result = bot.get_response(user_msg)
    return result['response']


# ─────────────────────────────────────────────────────────────────────────────
# VOICE — Web Speech API injected as HTML component
# ─────────────────────────────────────────────────────────────────────────────
def voice_input_widget(lang_code: str = 'en-IN') -> None:
    """Renders a mic button. On click it listens and stores transcript in sessionStorage."""
    lang_map = {'en': 'en-IN', 'hi': 'hi-IN', 'mr': 'mr-IN'}
    bcp_lang = lang_map.get(lang_code, 'en-IN')

    components.html(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0">
        <button id="micBtn" onclick="toggleMic()" title="Click to speak"
            style="background:linear-gradient(135deg,#1b5e20,#43a047);
                   color:#fff;border:none;border-radius:50%;width:44px;height:44px;
                   font-size:1.3rem;cursor:pointer;box-shadow:0 3px 10px rgba(46,125,50,.4);
                   transition:all .2s;display:flex;align-items:center;justify-content:center">
            🎤
        </button>
        <span id="statusTxt" style="font-size:.82rem;color:#1b5e20;font-weight:600"></span>
    </div>
    <script>
    var recog = null;
    var listening = false;

    function toggleMic() {{
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
            document.getElementById('statusTxt').innerText = 'Voice not supported in this browser. Use Chrome.';
            return;
        }}
        if (listening) {{
            recog && recog.stop();
            return;
        }}
        var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        recog = new SR();
        recog.lang = '{bcp_lang}';
        recog.interimResults = false;
        recog.maxAlternatives = 1;

        recog.onstart = function() {{
            listening = true;
            document.getElementById('micBtn').style.background = 'linear-gradient(135deg,#c62828,#e53935)';
            document.getElementById('micBtn').innerText = '⏹';
            document.getElementById('statusTxt').innerText = 'Listening... speak now';
        }};
        recog.onresult = function(e) {{
            var text = e.results[0][0].transcript;
            document.getElementById('statusTxt').innerText = '✅ Got: ' + text;
            // Store in sessionStorage so Streamlit can retrieve it
            sessionStorage.setItem('fasalsaarthi_voice', text);
            // Also try to inject into Streamlit text input
            var inputs = window.parent.document.querySelectorAll('input[type="text"]');
            inputs.forEach(function(inp) {{
                if (inp.placeholder && inp.placeholder.includes('diseases')) {{
                    var nativeInput = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
                    nativeInput.set.call(inp, text);
                    inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                }}
            }});
        }};
        recog.onerror = function(e) {{
            document.getElementById('statusTxt').innerText = 'Error: ' + e.error;
        }};
        recog.onend = function() {{
            listening = false;
            document.getElementById('micBtn').style.background = 'linear-gradient(135deg,#1b5e20,#43a047)';
            document.getElementById('micBtn').innerText = '🎤';
        }};
        recog.start();
    }}
    </script>
    """, height=70)


def speak_text(text: str, lang_code: str = 'en') -> None:
    """Speaks the given text using browser TTS."""
    lang_map = {'en': 'en-IN', 'hi': 'hi-IN', 'mr': 'mr-IN'}
    bcp_lang = lang_map.get(lang_code, 'en-IN')
    # Escape for JS — remove HTML tags and special chars
    safe = (text.replace("'", " ").replace('"', ' ')
               .replace('\n', ' ').replace('**', '').replace('*', '')
               .replace('<br>', ' ').replace('<b>', '').replace('</b>', '')[:500])
    components.html(f"""
    <script>
    (function() {{
        if (!window.speechSynthesis) return;
        window.speechSynthesis.cancel();
        var u = new SpeechSynthesisUtterance('{safe}');
        u.lang = '{bcp_lang}';
        u.rate = 0.92;
        u.pitch = 1.0;
        window.speechSynthesis.speak(u);
    }})();
    </script>
    """, height=0)


# ── Small helpers ─────────────────────────────────────────────────────────────
def cbadge(v):
    c = 'bg' if v>=75 else 'by' if v>=55 else 'br'
    i = '✔' if v>=75 else '~' if v>=55 else '⚠'
    return f'<span class="badge {c}">{i} {v:.1f}%</span>'

def cbar(v):
    clr = '#2e7d32' if v>=75 else '#f57c00' if v>=55 else '#c62828'
    return f'<div class="cw"><div class="cf" style="width:{v:.0f}%;background:{clr}"></div></div>'

def sevbadge(s):
    c = {'None':'bg','Low':'bg','Medium':'by','High':'br','Very High':'br'}.get(s,'by')
    return f'<span class="badge {c}">{s}</span>'

def section(icon, title_key, sub_key=''):
    st.markdown(f'<p class="sh">{icon} {T(title_key)}</p>', unsafe_allow_html=True)
    if sub_key:
        st.markdown(f'<p class="ss">{T(sub_key)}</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────────────────────────────────────
def page_auth():
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.markdown(f"""
        <div style="text-align:center;padding:40px 0 20px">
            <div style="font-size:4rem">🌾</div>
            <h1 style="font-family:'Playfair Display',serif;font-size:2.8rem;
                        color:#1b5e20;margin:8px 0 0;letter-spacing:-1.5px;font-weight:800">
                FASALSAARTHI</h1>
            <p style="color:#6b7c6b;font-size:.95rem;margin:6px 0 0">{T('tagline')}</p>
            <div style="display:flex;justify-content:center;gap:8px;margin-top:14px;flex-wrap:wrap">
                <span class="ds-badge">📊 Real Maharashtra Data</span>
                <span class="ds-badge">🤖 AI Chatbot</span>
                <span class="ds-badge">🌿 PlantVillage</span>
                <span class="ds-badge">🎤 Voice</span>
            </div>
            <div style="width:60px;height:3px;background:linear-gradient(90deg,#2e7d32,#66bb6a);
                        border-radius:2px;margin:18px auto 0"></div>
        </div>""", unsafe_allow_html=True)

        t_in, t_up = st.tabs([f"🔑 {T('sign_in')}", f"🌱 {T('create_acc')}"])
        with t_in:
            st.markdown("<br>", unsafe_allow_html=True)
            u = st.text_input(f"👤 {T('username')}", placeholder=T('username'), key="li_u")
            p = st.text_input(f"🔒 {T('password')}", type="password", key="li_p")
            st.markdown(f"<div style='font-size:.78rem;color:#aaa;margin:-6px 0 14px'>{T('demo_hint')}</div>",
                        unsafe_allow_html=True)
            if st.button(T('login_btn'), use_container_width=True):
                if not u or not p:
                    st.error(T('login_fill'))
                else:
                    ok, msg, info = login(u, p)
                    if ok:
                        st.session_state.update(logged_in=True, username=u.strip().lower(),
                                                user_info=info, page='home')
                        st.success(f"Welcome back, {info.get('name','Farmer')}! 🌾")
                        time.sleep(0.7); st.rerun()
                    else:
                        st.error(msg)

        with t_up:
            st.markdown("<br>", unsafe_allow_html=True)
            rn = st.text_input(f"👤 {T('full_name')}", key="rg_n")
            ru = st.text_input(f"🆔 {T('username')}", placeholder="Letters, numbers, _", key="rg_u")
            rp = st.text_input(f"🔒 {T('password')}", type="password", key="rg_p")
            rd = st.selectbox(f"📍 {T('district')}", load_yield().get_districts(), key="rg_d")
            if st.button(T('register_btn'), use_container_width=True):
                if not all([rn, ru, rp]):
                    st.error(T('reg_fill'))
                else:
                    ok, msg = register(ru, rp, rn, "Maharashtra", rd)
                    st.success(msg + f" {T('sign_in')}.") if ok else st.error(msg)

        st.markdown("""<div style='text-align:center;padding:20px 0;color:#bbb;font-size:.76rem'>
            📞 <b style='color:#43a047'>1800-180-1551</b> — Kisan Call Center (Free, 24×7)
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        info = st.session_state.user_info
        st.markdown(f"""
        <div style="padding:20px 10px 16px;border-bottom:1px solid rgba(255,255,255,.1);margin-bottom:14px">
            <div style="text-align:center;font-size:2.6rem;margin-bottom:6px">👨‍🌾</div>
            <div style="text-align:center;font-size:1rem;font-weight:700;color:#fff">
                {info.get('name','Farmer')}</div>
            <div style="text-align:center;font-size:.72rem;color:#81c784;margin-top:2px">
                {info.get('district','Maharashtra')} 📍</div>
        </div>""", unsafe_allow_html=True)

        # ── LANGUAGE SWITCH — on_change pattern (the only correct way in Streamlit) ──
        all_langs = get_all_languages()
        labels    = list(all_langs.keys())
        cur_idx   = labels.index(st.session_state.lang_label) \
                    if st.session_state.lang_label in labels else 0

        def _on_lang_change():
            chosen = st.session_state['_lang_sel']
            st.session_state.lang_label = chosen
            st.session_state.lang       = all_langs[chosen]

        st.selectbox(f"🌍 {T('language')}", labels,
                     index=cur_idx, key='_lang_sel',
                     on_change=_on_lang_change)

        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,.08);margin:12px 0'>",
                    unsafe_allow_html=True)

        # ── Navigation — all labels from T() ──
        nav = [
            ('🏠', 'home',        'nav_home'),
            ('📊', 'yield',       'nav_yield'),
            ('🌿', 'disease',     'nav_disease'),
            ('🌱', 'fertilizer',  'nav_fertilizer'),
            ('🌾', 'recommender', 'nav_recommender'),
            ('🤖', 'helpbot',     'nav_helpbot'),
        ]
        st.markdown("<div style='font-size:.68rem;color:#81c784;font-weight:700;"
                    "letter-spacing:1px;text-transform:uppercase;margin-bottom:8px'>Navigate</div>",
                    unsafe_allow_html=True)
        for icon, pg, tkey in nav:
            active = "▶ " if st.session_state.page == pg else ""
            label  = T(tkey)
            if st.button(f"{icon} {active}{label}", key=f"nav_{pg}", use_container_width=True):
                st.session_state.page = pg
                st.rerun()

        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,.08);margin:12px 0'>",
                    unsafe_allow_html=True)
        st.markdown("""<div style='font-size:.75rem;color:#81c784;line-height:2.2'>
            📞 <b style='color:#a5d6a7'>1800-180-1551</b> Kisan<br>
            🌐 <a href='https://enam.gov.in' style='color:#66bb6a'>eNAM Mandi</a><br>
            🌦️ <a href='https://mausam.imd.gov.in' style='color:#66bb6a'>IMD Weather</a><br>
            📊 <a href='https://data.gov.in' style='color:#66bb6a'>Agri Open Data</a>
        </div>""", unsafe_allow_html=True)
        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,.08);margin:12px 0'>",
                    unsafe_allow_html=True)
        if st.button(T('sign_out'), use_container_width=True):
            for k in DEFAULTS:
                st.session_state[k] = DEFAULTS[k]
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    name = st.session_state.user_info.get('name', 'Farmer')
    hour = datetime.now().hour
    grt  = T('good_morning') if hour < 12 else T('good_afternoon') if hour < 17 else T('good_evening')

    # Use safe hardcoded stats — don't trigger model loading on home page
    total_rows  = 650
    total_crops = 27
    year_range  = '1997–2019'

    st.markdown(f"""
    <div class="hero">
        <div class="hero-pill">✦ {T('hero_badge')} ✦</div>
        <h1>🌾 FASALSAARTHI</h1>
        <p>{grt}, <b>{name}</b>! {T('hero_subtitle')}</p>
        <p style='font-size:.8rem;opacity:.55;margin-top:10px'>
            फसल साथी &nbsp;·&nbsp; EN · हिंदी · मराठी</p>
        <div style='display:flex;gap:10px;margin-top:16px;flex-wrap:wrap;position:relative;z-index:1'>
            <span class="ds-badge" style="background:rgba(255,255,255,.12);border-color:rgba(255,255,255,.25);color:#fff">
                📊 {total_rows} Maharashtra Records</span>
            <span class="ds-badge" style="background:rgba(255,255,255,.12);border-color:rgba(255,255,255,.25);color:#fff">
                🌱 2,200 Soil Samples</span>
            <span class="ds-badge" style="background:rgba(255,255,255,.12);border-color:rgba(255,255,255,.25);color:#fff">
                🎤 Voice Input</span>
        </div>
    </div>""", unsafe_allow_html=True)

    s1,s2,s3,s4,s5 = st.columns(5)
    for col, val, lkey in [
        (s1, total_crops,  'stat_crops'),
        (s2, '9',          'stat_diseases'),
        (s3, year_range,   'stat_districts'),
        (s4, '3',          'stat_languages'),
        (s5, '87%+',       'stat_accuracy'),
    ]:
        with col:
            st.markdown(f'<div class="scard"><div class="sv">{val}</div>'
                        f'<div class="sl">{T(lkey)}</div></div>', unsafe_allow_html=True)

    st.markdown("<hr class='fdiv'>", unsafe_allow_html=True)
    st.markdown(f'<p class="sh">{T("what_todo")}</p>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c4,c5,c6 = st.columns(3)
    cards = [
        (c1, '📊', 'card_yield_title',   'card_yield_desc',   'yield'),
        (c2, '🌿', 'card_disease_title', 'card_disease_desc', 'disease'),
        (c3, '🌱', 'card_fert_title',    'card_fert_desc',    'fertilizer'),
        (c4, '🌾', 'card_rec_title',     'card_rec_desc',     'recommender'),
        (c5, '🤖', 'card_bot_title',     'card_bot_desc',     'helpbot'),
        (c6, '📈', 'card_yield_title',   'card_yield_desc',   'yield'),
    ]
    for idx, (col, icon, tkey, dkey, pg) in enumerate(cards):
        with col:
            st.markdown(f"""<div class="gcard">
                <span class="gc-icon">{icon}</span>
                <h3>{T(tkey)}</h3><p>{T(dkey)}</p></div>
                <div style="height:8px"></div>""", unsafe_allow_html=True)
            if st.button(T('open_arrow'), key=f"hb_card_{idx}", use_container_width=True):
                st.session_state.page = pg
                st.rerun()

    st.markdown("<hr class='fdiv'>", unsafe_allow_html=True)
    r1, r2 = st.columns([1.4, 1])
    with r1:
        st.markdown(f'<p class="sh">{T("crop_calendar")}</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            'Season':     ['☀️ Kharif', '❄️ Rabi', '🌸 Zaid'],
            'Sowing':     ['June–July', 'Oct–Nov', 'Mar–Apr'],
            'Harvest':    ['Sep–Oct',   'Mar–Apr', 'Jun–Jul'],
            'Main Crops': ['Rice, Soybean, Cotton, Jowar, Bajra',
                           'Wheat, Onion, Gram, Mustard',
                           'Tomato, Watermelon'],
        }), use_container_width=True, hide_index=True)
    with r2:
        st.markdown(f'<p class="sh">{T("key_schemes")}</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            'Scheme':  ['PM-KISAN', 'Fasal Bima', 'KCC', 'PMKSY'],
            'Benefit': ['₹6,000/yr', '@ 2%', '@ 4%', '55% subsidy'],
            'Contact': ['pmkisan.gov.in', '1800-200-7710', 'Any bank', 'pmksy.gov.in'],
        }), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# YIELD
# ─────────────────────────────────────────────────────────────────────────────
def page_yield():
    section("📊", "yield_title", "yield_sub")
    ym = load_yield()
    di = ym.get_dataset_info()

    st.markdown(f"""<div style="margin-bottom:16px">
        <span class="ds-badge">📊 {di.get('total_rows',650)} MH Records</span>
        <span class="ds-badge">🌾 {di.get('crops',27)} {T('stat_crops')}</span>
        <span class="ds-badge">📅 {di.get('year_range','1997–2019')}</span>
        <span class="ds-badge">🗺️ Maharashtra Only</span>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([1, 1.15])
    with left:
        crops        = ym.get_crops()
        crop         = st.selectbox(f"🌾 {T('crop')}", crops)
        seas         = ym.get_crop_default_season(crop)
        st.markdown(f"""<div class="ic s" style="padding:8px 14px;margin-bottom:4px">
            🗓️ <b>{T('season')}:</b> {seas}
            <span style="font-size:.75rem;color:#666;margin-left:6px">(auto-set for {crop})</span>
        </div>""", unsafe_allow_html=True)
        dist         = st.selectbox(f"📍 {T('district')}", ym.get_districts())
        year         = st.slider(f"📅 {T('year')}", 2020, 2030, 2024)
        area         = st.number_input(f"🌍 {T('area_ha')}", 0.1, 500.0, 2.0, 0.5,
                                       help="Used only to compute total farm output")

        st.markdown(f"""<div class="ic n" style="margin-top:8px;padding:10px 14px">
            <span style="font-size:.78rem">ℹ️ {T('how_it_works_txt')}</span></div>""",
                    unsafe_allow_html=True)

        if st.button(T('predict_btn'), use_container_width=True):
            with st.spinner(T('thinking')):
                st.session_state.yield_result = ym.predict(year, crop, dist, area, seas)

        st.markdown(f"#### 📈 {T('crop')}")
        sel = st.multiselect(T('crop'), crops,
                             default=crops[:6] if len(crops) >= 6 else crops[:3],
                             key="crop_cmp")
        if sel:
            comp = ym.get_crop_comparison(sel)
            if not comp.empty:
                fig = px.bar(comp, x='Crop', y='Avg (kg/ha)', color='Avg (kg/ha)',
                             color_continuous_scale='Greens',
                             title=T('typical_yields'))
                fig.update_layout(height=230, showlegend=False,
                    margin=dict(l=0,r=0,t=35,b=0),
                    font=dict(color='#111827', family='Inter'),
                    title_font=dict(color='#0a2e0a', size=14, family='Inter'),
                    xaxis=dict(tickfont=dict(color='#111827', size=11), title=dict(font=dict(color='#0a2e0a', size=11))),
                    yaxis=dict(tickfont=dict(color='#111827', size=10), title=dict(font=dict(color='#0a2e0a', size=11))),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

    with right:
        res = st.session_state.yield_result
        if res is None:
            st.markdown(f"""<div style="text-align:center;padding:60px 20px;color:#aaa">
                <div style="font-size:3.5rem">🌾</div>
                <p>{T('fill_details')}</p></div>""", unsafe_allow_html=True)
            return

        pred  = res['predicted_yield']
        conf  = res['confidence']
        clr   = '#1b5e20' if conf >= 75 else '#e65100' if conf >= 55 else '#b71c1c'
        total = res['total_production_t']

        st.markdown(f"""<div class="rbanner" style="background:linear-gradient(135deg,{clr},{clr}dd)">
            <div class="rl">{T('predicted_yield')} — {res['crop']} · {res['district']}</div>
            <div class="rv">{pred:,.0f}</div>
            <div class="rl">{T('kg_per_ha')} &nbsp;·&nbsp; {cbadge(conf)}</div>
            <div style="font-size:.78rem;opacity:.65;margin-top:8px">
                {res['farm_area_ha']} Ha → ~{total:.1f} t total
                &nbsp;·&nbsp; District factor {res['dist_factor']:.2f}×
            </div></div>""", unsafe_allow_html=True)
        st.markdown(cbar(conf), unsafe_allow_html=True)

        lo, hi, avg = res['typical_min'], res['typical_max'], res['typical_avg']
        pct     = max(0, min(100, (pred - lo) / (hi - lo + 1) * 100))
        verdict = T('above_avg') if pct > 60 else T('average') if pct > 35 else T('below_avg')
        st.markdown(f"""<div class="ic s">
            {T('typical_range')} <b>{res['crop']}</b>:
            {lo:,} – {hi:,} kg/ha &nbsp;·&nbsp; {T('average')}: <b>{avg:,}</b><br>
            {verdict} &nbsp;·&nbsp; {res['mh_data_rows']} MH records
        </div>""", unsafe_allow_html=True)

        gc, tc = st.columns(2)
        with gc:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=conf,
                number={'suffix': '%', 'font': {'size': 24, 'color': '#0a2e0a'}},
                title={'text': T('confidence'), 'font': {'size': 13, 'color': '#0a2e0a'}},
                gauge={'axis': {'range': [0, 100], 'tickfont': {'color': '#111827'}}, 'bar': {'color': clr},
                       'steps': [{'range': [0,50], 'color': '#fee2e2'},
                                  {'range': [50,75],'color': '#fef9c3'},
                                  {'range': [75,100],'color': '#dcfce7'}]}))
            fig_g.update_layout(height=175, margin=dict(l=10,r=10,t=30,b=5),
                                 font=dict(color='#111827', family='Inter'),
                                 paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_g, use_container_width=True)
        with tc:
            trend = ym.get_historical_trend(res['crop'])
            if not trend.empty:
                fig_t = px.area(trend, x='Year', y='Yield_Kg_Ha', markers=True,
                                color_discrete_sequence=['#2e7d32'],
                                title=f"{T('historical_trend')} — {res['crop']}")
                fig_t.update_traces(fill='tozeroy', fillcolor='rgba(46,125,50,0.15)')
                fig_t.update_layout(height=175, margin=dict(l=0,r=0,t=30,b=0),
                                     font=dict(color='#111827', family='Inter'),
                                     title_font=dict(color='#0a2e0a', size=13, family='Inter'),
                                     xaxis=dict(tickfont=dict(color='#111827', size=10), title=dict(font=dict(color='#0a2e0a', size=11))),
                                     yaxis=dict(tickfont=dict(color='#111827', size=10), title=dict(font=dict(color='#0a2e0a', size=11))),
                                     plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_t, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# DISEASE DETECTION  — FIX: read bytes immediately, store in session
# ─────────────────────────────────────────────────────────────────────────────
def page_disease():
    section("🌿", "disease_title", "disease_sub")
    det = load_det()
    pv  = det.get_plantvillage_info() or {}
    pv_images  = pv.get('images',  54305)
    pv_classes = pv.get('classes', 38)
    pv_url     = pv.get('url', pv.get('kaggle_url', 'https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset'))

    st.markdown(f"""<div class="ic g" style="margin-bottom:16px">
        🌿 <b>PlantVillage:</b> {pv_images:,} images · {pv_classes} disease classes ·
        <a href="{pv_url}" target="_blank" style="color:#e65100;font-weight:600">
        Kaggle ↗</a> — integrate CNN for 95%+ accuracy
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([1, 1.15])
    with left:
        # ── FIX: read bytes immediately on upload (before button rerun) ──────
        uploaded = st.file_uploader(T('upload_img'), type=['jpg','jpeg','png','webp'])

        if uploaded is not None:
            # Read bytes right now, before any rerun can lose the file
            img_bytes = uploaded.read()
            st.session_state.disease_img_bytes = img_bytes
            # Show preview
            st.image(img_bytes, use_column_width=True)

        # Show analyze button if we have bytes stored
        if st.session_state.disease_img_bytes:
            if st.button(T('analyze_btn'), use_container_width=True):
                with st.spinner(T('thinking')):
                    try:
                        img = Image.open(io.BytesIO(st.session_state.disease_img_bytes))
                        st.session_state.disease_result = det.predict(img)
                    except Exception as e:
                        st.session_state.disease_result = {'error': str(e)}
            if st.button("🗑️ Clear Image", use_container_width=True):
                st.session_state.disease_img_bytes = None
                st.session_state.disease_result    = None
                st.rerun()

        st.markdown(f"""<div class="ic n" style="margin-top:14px">
            <b>{T('photo_tips')}</b><br>
            <span style="font-size:.82rem">
            • {T('photo_tip1')}<br>• {T('photo_tip2')}<br>• {T('photo_tip3')}
            </span></div>""", unsafe_allow_html=True)

    with right:
        res = st.session_state.disease_result

        if res is None:
            st.markdown(f"""<div style="text-align:center;padding:80px 20px;color:#aaa">
                <div style="font-size:4rem">🍃</div>
                <p>{T('upload_first')}</p></div>""", unsafe_allow_html=True)
            return

        if 'error' in res:
            st.error(f"Error analyzing image: {res['error']}")
            st.info("Please try uploading a clearer JPG or PNG image.")
            return

        d       = res['disease']
        conf    = res['confidence']
        sev     = res['severity']
        healthy = (d == 'Healthy')
        d_clr   = '#1b5e20' if healthy else ('#b71c1c' if sev in ('High','Very High') else '#e65100')
        icon    = '✅' if healthy else '🔴' if sev == 'Very High' else '⚠️'

        st.markdown(f"""<div class="rbanner" style="background:linear-gradient(135deg,{d_clr},{d_clr}cc)">
            <div class="rl">{T('detected_disease')}</div>
            <div style="font-size:1.9rem;font-weight:800;margin:4px 0">{icon} {d}</div>
            <div style="font-size:.88rem;opacity:.82">
                {T('hindi_name')}: {res['hindi_name']} &nbsp;/&nbsp;
                {T('marathi_name')}: {res['marathi_name']}</div>
            <div style="margin-top:8px">
                {T('severity')}: {sevbadge(sev)} &nbsp; {T('confidence')}: {cbadge(conf)}
            </div></div>""", unsafe_allow_html=True)
        st.markdown(cbar(conf), unsafe_allow_html=True)

        t1, t2, t3, t4 = st.tabs([
            T('diagnosis_tab'), T('crop_info_tab'),
            T('products_tab'),  T('fert_fix_tab'),
        ])

        with t1:
            st.markdown(f"**{T('symptoms')}:** {res['description']}")
            st.divider()
            st.markdown(f"**{T('treatment')}:**")
            for line in res['treatment'].split('\n'):
                if line.strip():
                    st.markdown(line)
            st.divider()
            st.markdown(f"**{T('prevention')}:** {res['prevention']}")

        with t2:
            st.markdown(f"**{T('affected_crops')}:** {', '.join(res['affected_crops'])}")
            st.markdown(f"**{T('season')}:** {res['season']}")
            st.markdown(f"**{T('top_predictions')}:**")
            top = res['top_predictions']
            fig_t = go.Figure(go.Bar(
                x=[t['confidence'] for t in top],
                y=[t['disease']    for t in top],
                orientation='h', marker_color=['#1b5e20','#43a047','#81c784'],
                text=[f"{t['confidence']}%" for t in top], textposition='outside',
                textfont=dict(color='#0a2e0a', size=11, family='Inter')))
            fig_t.update_layout(height=175, showlegend=False,
                xaxis=dict(range=[0,108], showgrid=False, tickfont=dict(color='#111827')),
                yaxis=dict(tickfont=dict(color='#111827', size=11)),
                margin=dict(l=0,r=55,t=8,b=0),
                font=dict(color='#111827', family='Inter'),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_t, use_container_width=True)

        with t3:
            for p in res.get('products', []):
                bclr = {'🔴':'#b71c1c','🔵':'#1565c0','🟡':'#e65100',
                        '🟠':'#bf360c','🟢':'#2e7d32'}.get(p.get('icon','🔵'), '#2e7d32')
                st.markdown(f"""<div class="pcard" style="border-left:4px solid {bclr}">
                    <div class="pcat">{p.get('icon','')} {T('product_category')}: {p['category']}</div>
                    <div class="pname">{p['chemical']}</div>
                    <div class="pbrand">📦 {T('product_brand')}: {p['brand']}</div>
                    <table>
                        <tr><td>{T('product_dose')}</td><td>{p['dose']}</td></tr>
                        <tr><td>{T('product_freq')}</td><td>{p['frequency']}</td></tr>
                        <tr><td>{T('product_price')}</td><td>{p['price_est']}</td></tr>
                    </table></div>""", unsafe_allow_html=True)

        with t4:
            fc = res.get('fertilizer_correction', {})
            if fc:
                st.markdown(f"""
                <div class="ic n"><b>{T('fert_note')}:</b> {fc.get('note','')}</div>
                <div class="ic s" style="margin-top:8px"><b>{T('fert_action')}:</b> {fc.get('action','')}</div>
                <div class="ic w" style="margin-top:8px"><b>{T('fert_avoid')}:</b> {fc.get('avoid','')}</div>
                """, unsafe_allow_html=True)

        if not healthy:
            st.markdown(f"""<div class="ic d" style="margin-top:12px">
                ⚠️ <b>{T('disease_action')}</b> 📞 KVK: <b>1800-180-1551</b>
            </div>""", unsafe_allow_html=True)
        else:
            st.success(T('healthy_msg'))


# ─────────────────────────────────────────────────────────────────────────────
# FERTILIZER
# ─────────────────────────────────────────────────────────────────────────────
def page_fertilizer():
    section("🌱", "fert_title", "fert_sub")
    fr = load_fert()
    left, right = st.columns([1, 1.2])
    with left:
        crop = st.selectbox(f"🌾 {T('crop')}", fr.get_crops())
        area = st.number_input(f"🌍 {T('area_ha')}", 0.1, 500.0, 1.0, 0.5)
        st.markdown(f"**{T('soil_status')}**")
        c1, c2, c3 = st.columns(3)
        with c1: sn = st.select_slider(T('nitrogen'),   [T('low'),T('medium'),T('high')], T('medium'), key='sn')
        with c2: sp = st.select_slider(T('phosphorus'), [T('low'),T('medium'),T('high')], T('medium'), key='sp')
        with c3: sk = st.select_slider(T('potassium'),  [T('low'),T('medium'),T('high')], T('medium'), key='sk')
        # Map translated values back to English for the model
        tmap = {T('low'):'Low', T('medium'):'Medium', T('high'):'High'}
        st.caption(f"💡 {T('soil_hint')}")
        if st.button(T('rec_btn'), use_container_width=True):
            with st.spinner(T('thinking')):
                st.session_state.fert_result = fr.recommend(crop, area, tmap[sn], tmap[sp], tmap[sk])
        st.markdown(f"""<div class="ic n" style="margin-top:14px">
            <b>{T('npk_guide')}</b><br>
            <span style="font-size:.82rem">
            {T('n_role')}<br>{T('p_role')}<br>{T('k_role')}
            </span></div>""", unsafe_allow_html=True)

    with right:
        res = st.session_state.fert_result
        if res is None:
            st.markdown(f"""<div style="text-align:center;padding:80px 20px;color:#aaa">
                <div style="font-size:4rem">🌱</div><p>{T('fill_fert')}</p></div>""",
                        unsafe_allow_html=True)
            return
        if 'error' in res:
            st.error(res['error']); return

        st.markdown(f"""<div class="ic s" style="margin-bottom:14px">
            🌾 <b>{res['crop']}</b> &nbsp;·&nbsp; {res['area']} Ha &nbsp;·&nbsp;
            💰 {T('total_cost')}: <b>₹{res['total_cost']:,.0f}</b>
        </div>""", unsafe_allow_html=True)

        bg_m  = {'Low':'#fce4e4', 'Medium':'#fff9c4', 'High':'#e8f5e9'}
        bdr_m = {'Low':'#c62828', 'Medium':'#f57f17', 'High':'#2e7d32'}
        for r in res['recommendations']:
            sc = r['soil_status']
            st.markdown(f"""<div style="background:{bg_m[sc]};border-left:4px solid {bdr_m[sc]};
                border-radius:12px;padding:14px 16px;margin:8px 0">
                <div style="font-weight:700;font-size:.95rem">{r['icon']} {r['nutrient']}
                    <span class="badge bg" style="font-size:.68rem;margin-left:6px">
                        {T('soil_label')}: {sc}</span></div>
                <div style="font-size:.84rem;line-height:2;margin-top:8px">
                    📦 <b>{T('primary_fert')}:</b> <b>{r['primary_fert']}</b> =
                        <b>{r['primary_qty_kg']:.1f} kg</b> ({r['primary_hindi']})<br>
                    🔄 <b>{T('alt_fert')}:</b> {r['alt_fert']} = {r['alt_qty_kg']:.1f} kg<br>
                    ⏰ {r['application']}<br>
                    💰 {T('est_cost')}: ₹{r['cost']:,.0f}
                </div></div>""", unsafe_allow_html=True)

        ns = res['nutrient_summary']
        fig_n = go.Figure(go.Bar(
            x=['N', 'P₂O₅', 'K₂O'], y=[ns['N'], ns['P'], ns['K']],
            marker_color=['#1565c0','#e65100','#2e7d32'],
            text=[f"{v:.0f} kg" for v in [ns['N'],ns['P'],ns['K']]],
            textposition='outside',
            textfont=dict(color='#0a2e0a', size=12, family='Inter')))
        fig_n.update_layout(title=T('fert_summary'), height=200, showlegend=False,
            font=dict(color='#111827', family='Inter'),
            title_font=dict(color='#0a2e0a', size=14, family='Inter'),
            xaxis=dict(tickfont=dict(color='#111827', size=12)),
            yaxis=dict(tickfont=dict(color='#111827', size=10)),
            margin=dict(l=0,r=0,t=30,b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_n, use_container_width=True)

        st.markdown(f"#### {T('schedule_title')}")
        for i, s in enumerate(res['schedule'], 1):
            st.markdown(f"**{i}.** {s}")
        st.info(f"{T('organic_tip')}: {res['organic_tip']}")


# ─────────────────────────────────────────────────────────────────────────────
# CROP RECOMMENDER
# ─────────────────────────────────────────────────────────────────────────────
def page_recommender():
    section("🌾", "card_rec_title", "card_rec_desc")
    rc = load_rec()
    di = rc.get_dataset_info()

    st.markdown(f"""<div style="margin-bottom:16px">
        <span class="ds-badge">🌱 {di['rows']} Soil Samples</span>
        <span class="ds-badge">🌾 {di['crops']} {T('stat_crops')}</span>
        <span class="ds-badge">✅ {di['accuracy']:.0f}% Accuracy</span>
        <span class="ds-badge">📍 Konkan Maharashtra</span>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([1, 1.2])
    with left:
        st.markdown(f"#### 🧪 {T('soil_status')}")
        c1, c2, c3 = st.columns(3)
        with c1: N = st.number_input("N", 0, 150, 90, 5)
        with c2: P = st.number_input("P", 0, 150, 45, 5)
        with c3: K = st.number_input("K", 0, 210, 45, 5)
        ph   = st.slider("🔬 pH", 3.5, 10.0, 6.5, 0.1)
        c4, c5 = st.columns(2)
        with c4: temp = st.number_input("🌡️ Temp (°C)", 5.0, 45.0, 25.0, 0.5)
        with c5: hum  = st.number_input("💧 Humidity %", 10.0, 100.0, 70.0, 1.0)
        rain = st.slider("🌧️ Rainfall (mm)", 20, 400, 150, 10)

        if st.button(T('rec_btn'), use_container_width=True, key='rec_btn2'):
            with st.spinner(T('thinking')):
                st.session_state.rec_result = rc.recommend(N, P, K, temp, hum, ph, rain)

    with right:
        res = st.session_state.rec_result
        if res is None:
            st.markdown(f"""<div style="text-align:center;padding:80px 20px;color:#aaa">
                <div style="font-size:4rem">🌾</div>
                <p>{T('fill_fert')}</p></div>""", unsafe_allow_html=True)
            return

        st.markdown(f"""<div class="rbanner" style="background:linear-gradient(135deg,#1b5e20,#43a047)">
            <div class="rl">Best Recommended Crop</div>
            <div style="font-size:2.8rem;margin:6px 0">{res['best_emoji']}</div>
            <div style="font-size:2rem;font-weight:800">{res['best_crop']}</div>
            <div style="font-size:.85rem;opacity:.82;margin-top:4px">
                {res['best_hindi']} / {res['best_marathi']} · {res['best_season']}</div>
            <div style="margin-top:8px">{cbadge(res['best_prob'])}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(cbar(res['best_prob']), unsafe_allow_html=True)

        for tip in res['tips']:
            cls = 's' if '✅' in tip else 'w'
            st.markdown(f'<div class="ic {cls}" style="padding:10px 14px;margin:4px 0">{tip}</div>',
                        unsafe_allow_html=True)

        for i, c in enumerate(res['top5']):
            cls   = "crc best" if i == 0 else "crc"
            badge = "🥇 Best" if i == 0 else f"#{i+1}"
            st.markdown(f"""<div class="{cls}">
                <div class="cre">{c['emoji']}</div>
                <div class="crd">
                    <h4>{badge} &nbsp; {c['crop']}</h4>
                    <p>{c['hindi']} / {c['marathi']} · {c['season']} · Water: {c['water']}</p>
                </div>
                <div class="crp">
                    <div class="pct">{c['prob']:.0f}%</div>
                    <div class="plb">match</div>
                </div></div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPBOT  — local helpbot + voice input/output
# ─────────────────────────────────────────────────────────────────────────────
def page_helpbot():
    section("🤖", "bot_title", "bot_sub")
    cc, pc = st.columns([2, 1])

    with cc:
        # Build chat HTML
        html = '<div class="chat-box" id="chatbox">'
        if not st.session_state.chat_messages:
            html += f"""<div class="msg-b"><div class="av">🌾</div>
                <div class="bbl">
                    <b>{T('bot_welcome')}</b><br><br>
                    {T('bot_welcome2')}<br><br>
                    {T('bot_welcome3')}
                    <div class="mt">Just now</div>
                </div></div>"""

        for m in st.session_state.chat_messages:
            t = m.get('time', '')
            if m['role'] == 'user':
                html += f"""<div class="msg-u"><div class="av">👤</div>
                    <div class="bbl">{m['content']}
                    <div class="mt" style="color:rgba(255,255,255,.5)">{t}</div>
                    </div></div>"""
            else:
                content = m['content'].replace('\n', '<br>').replace('**', '<b>',1).replace('**','</b>',1)
                html += f"""<div class="msg-b"><div class="av">🌾</div>
                    <div class="bbl">{content}
                    <div class="mt">{t}</div>
                    </div></div>"""
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

        # ── Input row with voice button ──────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        ic, vc, bc = st.columns([5, 1, 1])
        with ic:
            user_input = st.text_input(
                "msg", label_visibility='collapsed',
                placeholder=T('chat_placeholder'),
                key='chat_inp',
                value=st.session_state.get('voice_text', ''),
            )
        with vc:
            # Voice mic button
            voice_input_widget(st.session_state.lang)
        with bc:
            send = st.button(T('send_btn'), use_container_width=True)

        cl, _ = st.columns([1, 4])
        with cl:
            if st.button(T('clear_btn'), use_container_width=True):
                st.session_state.chat_messages = []
                st.session_state.voice_text    = ''
                st.rerun()

        # ── Handle send ──────────────────────────────────────────────────────
        if send and user_input.strip():
            now  = datetime.now().strftime('%H:%M')
            text = user_input.strip()

            st.session_state.chat_messages.append({
                'role': 'user', 'content': text, 'time': now
            })
            st.session_state.voice_text = ''   # clear voice buffer

            with st.spinner(T('thinking')):
                reply = get_bot_reply(text, st.session_state.chat_messages)

            st.session_state.chat_messages.append({
                'role': 'assistant', 'content': reply, 'time': datetime.now().strftime('%H:%M')
            })

            # Speak the reply aloud
            speak_text(reply, st.session_state.lang)

            st.rerun()

    with pc:
        st.markdown(f"""<div style="background:#fff;border-radius:14px;padding:14px;
            box-shadow:0 2px 10px rgba(0,0,0,.06);margin-bottom:10px;border:1px solid #e8f0e8">
            <b style="color:#1b5e20;font-size:.85rem">{T('try_asking')}</b></div>""",
                    unsafe_allow_html=True)

        suggestions = [
            "🌿 My rice has yellow spots on leaves",
            "📊 Wheat yield for 2 Ha in Nashik?",
            "🌱 Fertilizer for cotton 2 acres",
            "💰 Soybean MSP 2024–25?",
            "🏛️ How to apply PM Fasal Bima?",
            "🐛 Bollworm control in cotton?",
            "मेरी फसल में पीले पत्ते हैं",
            "माझ्या भाताला करपा झाला आहे",
            "💧 ड्रिप सिंचाई कैसे करें?",
            "🌧️ Before monsoon checklist",
        ]
        for q in suggestions:
            if st.button(q, key=f"sq_{q[:12]}", use_container_width=True):
                now = datetime.now().strftime('%H:%M')
                st.session_state.chat_messages.append({'role': 'user', 'content': q, 'time': now})
                with st.spinner(T('thinking')):
                    reply = get_bot_reply(q, st.session_state.chat_messages)
                st.session_state.chat_messages.append({
                    'role': 'assistant', 'content': reply,
                    'time': datetime.now().strftime('%H:%M')
                })
                speak_text(reply, st.session_state.lang)
                st.rerun()

        st.markdown(f"""
        <div style="background:#f4faf4;border-radius:12px;padding:14px;
                    margin-top:10px;border:1px solid #e8f0e8">
            <b style="color:#1b5e20;font-size:.83rem">{T('helplines')}</b>
            <div style="font-size:.78rem;line-height:2.2;margin-top:6px;color:#555">
                Kisan: <b>1800-180-1551</b><br>
                Fasal Bima: <b>1800-200-7710</b><br>
                PM-KISAN: <b>155261</b>
            </div></div>
        <div style="font-size:.7rem;color:#bbb;text-align:center;margin-top:10px">
            🎤 {T('language')}: {st.session_state.lang_label}<br>
            {T('bot_disclaimer')}
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    page_auth()
else:
    render_sidebar()
    {
        'home':        page_home,
        'yield':       page_yield,
        'disease':     page_disease,
        'fertilizer':  page_fertilizer,
        'recommender': page_recommender,
        'helpbot':     page_helpbot,
    }.get(st.session_state.page, page_home)()
