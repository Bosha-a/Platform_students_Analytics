from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import base64
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from mongo_loader import load_collection

st.set_page_config(page_title="Kayfa Analytics", page_icon=Image.open("kayfa_icon.png"), layout="wide")

st.markdown("""
<style>
/* Hide main menu (⋮) */
#MainMenu {
    visibility: hidden;
}
            

/* Hide Deploy / Share button */
.stAppDeployButton {
    display: none;
}

/* Hide header بالكامل (الشريط العلوي) — use height:0 NOT display:none
   so the sidebar toggle button (child node) can still be extracted via position:fixed */
[data-testid="stHeader"] {
    height: 0 !important;
    min-height: 0 !important;
    overflow: visible !important;
    background: transparent !important;
    border: none !important;
    padding: 5px !important;
}

/* ── Sidebar open button (shown when sidebar is collapsed) ── */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    position: fixed !important;
    top: 0.55rem !important;
    left: 0.55rem !important;
    z-index: 9999999 !important;
    background: #ffffff !important;
    border: 2px solid #111111 !important;
    border-radius: 8px !important;
    width: 100px !important;
    height: 50px !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.22) !important;
    cursor: pointer !important;
    transition: background 0.18s ease, box-shadow 0.18s ease !important;
    padding: 0 !important;
}
[data-testid="collapsedControl"]:hover {
    background: #f0f0f0 !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.28) !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="collapsedControl"] svg * {
    fill: #111111 !important;
    stroke: #111111 !important;
    color: #111111 !important;
    width: 20px !important;
    height: 20px !important;
}

/* ── Sidebar close button (shown inside the open sidebar) ── */
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
}
[data-testid="stSidebarCollapseButton"] button {
    display: flex !important;
    visibility: visible !important;
    background: rgba(255,255,255,0.15) !important;
    border: 2px solid rgba(255,255,255,0.6) !important;
    border-radius: 8px !important;
    width: 36px !important;
    height: 36px !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: background 0.18s ease !important;
    padding: 0 !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    background: rgba(255,255,255,0.28) !important;
}
[data-testid="stSidebarCollapseButton"] button svg,
[data-testid="stSidebarCollapseButton"] button svg * {
    fill: #ffffff !important;
    stroke: #ffffff !important;
    color: #ffffff !important;
    width: 20px !important;
    height: 20px !important;
}

/* Hide footer (Streamlit branding) */
footer {
    visibility: hidden;
}
    
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap');
:root{--kayfa-blue:#0b63ce;--kayfa-blue-2:#1d8fff;--kayfa-ink:#14213d;--kayfa-muted:#64748b;--kayfa-line:#d8e6f7;--kayfa-soft:#f4f8ff;}
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif;background:#ffffff;color:var(--kayfa-ink);}
.block-container{padding-top:3.75rem;}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;color:var(--kayfa-blue);}
p,li,span,label,div{color:var(--kayfa-ink);}
[data-testid="stHeader"]{background:rgba(255,255,255,.92);border-bottom:1px solid var(--kayfa-line);}

/* ── Sidebar Blue Theme ── */
[data-testid="stSidebar"]{background:linear-gradient(175deg,#0b3d91 0%,#0b63ce 45%,#1570ef 100%)!important;border-right:none!important;box-shadow:4px 0 24px rgba(11,61,145,.35);}
[data-testid="stSidebar"]>div:first-child{padding-top:0!important;}
[data-testid="stSidebar"] *{color:#ffffff!important;}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#ffffff!important;}
[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.15)!important;}
[data-testid="stSidebar"] [data-testid="metric-container"]{background:rgba(255,255,255,.10)!important;border:1px solid rgba(255,255,255,.18)!important;border-radius:10px!important;box-shadow:0 2px 8px rgba(0,0,0,.15)!important;backdrop-filter:blur(4px);}
[data-testid="stSidebar"] [data-testid="metric-container"] label{color:rgba(255,255,255,.7)!important;font-size:0.74rem!important;text-transform:uppercase!important;letter-spacing:.07em!important;}
[data-testid="stSidebar"] [data-testid="stMetricValue"]{color:#ffffff!important;font-family:'Space Grotesk',sans-serif!important;font-size:1.5rem!important;}
[data-testid="stSidebar"] [data-testid="stMetricDelta"]{color:rgba(255,255,255,.7)!important;}
/* Sidebar logo area */
.sidebar-logo-wrap{display:flex;flex-direction:column;align-items:center;padding:28px 20px 20px;border-bottom:1px solid rgba(255,255,255,.15);margin-bottom:8px;}
.sidebar-logo-box{background:#ffffff;border-radius:16px;padding:14px 22px;box-shadow:0 6px 24px rgba(0,0,0,.18);margin-bottom:14px;display:flex;align-items:center;justify-content:center;}
.sidebar-logo-wrap img{height:120px;width:auto;display:block;}
.sidebar-brand{font-family:'Space Grotesk',sans-serif;font-size:1.28rem;font-weight:700;color:#ffffff;letter-spacing:.02em;}
.sidebar-brand-sub{font-size:.72rem;color:rgba(255,255,255,.65);letter-spacing:.1em;text-transform:uppercase;margin-top:3px;}
/* Sidebar stat rows */
.sb-stat{display:flex;align-items:center;gap:12px;padding:11px 16px;border-radius:10px;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.14);margin-bottom:8px;transition:background .2s;}
.sb-stat:hover{background:rgba(255,255,255,.16);}
.sb-stat-icon{font-size:1.3rem;flex-shrink:0;}
.sb-stat-body{display:flex;flex-direction:column;}
.sb-stat-label{font-size:.68rem;color:rgba(255,255,255,.62);text-transform:uppercase;letter-spacing:.08em;}
.sb-stat-value{font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;color:#ffffff;line-height:1.2;}
.sb-section-title{font-size:.65rem;font-weight:600;color:rgba(255,255,255,.45);text-transform:uppercase;letter-spacing:.12em;padding:14px 4px 6px;}
.sb-divider{border:none;border-top:1px solid rgba(255,255,255,.12);margin:6px 0 12px;}
.sb-footer{text-align:center;padding:18px 8px 10px;font-size:.68rem;color:rgba(255,255,255,.4);letter-spacing:.05em;}
div[data-testid="metric-container"]{background:#ffffff;border:1px solid var(--kayfa-line);border-radius:10px;padding:14px 18px;box-shadow:0 8px 24px rgba(11,99,206,.07);}
div[data-testid="metric-container"] label{color:var(--kayfa-muted)!important;font-size:0.78rem;}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'Space Grotesk',sans-serif;font-size:1.6rem;color:var(--kayfa-blue)!important;}
.obs-box{background:var(--kayfa-soft);border-left:3px solid var(--kayfa-blue);padding:12px 16px;border-radius:0 8px 8px 0;color:var(--kayfa-ink);font-size:0.88rem;margin-top:8px;}
.section-tag{display:inline-block;background:#eaf3ff;border-left:3px solid var(--kayfa-blue);padding:3px 12px;border-radius:0 5px 5px 0;font-size:0.72rem;color:var(--kayfa-blue);letter-spacing:.07em;text-transform:uppercase;margin-bottom:4px;}
div[data-testid="stPlotlyChart"]{background:#ffffff;border:1px solid var(--kayfa-line);border-radius:10px;padding:2px;box-shadow:0 8px 24px rgba(11,99,206,.06);}
.stTabs [data-baseweb="tab"]{font-family:'Inter',sans-serif;font-size:0.88rem;font-weight:500;padding:10px 22px;color:var(--kayfa-muted);letter-spacing:.01em;transition:all .25s ease;}
.stTabs [data-baseweb="tab"]:hover{color:var(--kayfa-blue);}
.stTabs [data-baseweb="tab"][aria-selected="true"]{color:var(--kayfa-blue);font-weight:600;border-bottom:2.5px solid var(--kayfa-blue);}
.stTabs [data-baseweb="tab-list"]{background:#ffffff;border-bottom:1.5px solid var(--kayfa-line);gap:0;padding:0 4px;}
.stTabs [data-baseweb="tab-highlight"]{background-color:var(--kayfa-blue);}
details summary{color:var(--kayfa-muted)!important;font-size:0.83rem;}
.kayfa-top-logo{
    position:fixed;
    top:.55rem;
    right:1rem;
    z-index:999999;
    display:flex;
    align-items:center;
    gap:.55rem;
    border:none;
    padding:.3rem .75rem;
    background:transparent;
}

.kayfa-top-logo img{
    height:38px;
    width:auto;
    display:block;
}

.kayfa-top-logo span{
    font-family:'Space Grotesk',sans-serif;
    font-weight:700;
    color:var(--kayfa-blue);
    font-size:1rem;
}

/* ── Hero Card ── */
.hero-card{background:linear-gradient(135deg,#0b3d91 0%,#0b63ce 40%,#1d8fff 100%);border-radius:18px;padding:48px 52px 42px;margin-bottom:8px;position:relative;overflow:hidden;box-shadow:0 16px 48px rgba(11,99,206,.22);}
.hero-card::before{content:'';position:absolute;top:-60px;right:-60px;width:280px;height:280px;background:radial-gradient(circle,rgba(255,255,255,.08) 0%,transparent 70%);border-radius:50%;}
.hero-card::after{content:'';position:absolute;bottom:-40px;left:30%;width:200px;height:200px;background:radial-gradient(circle,rgba(255,255,255,.05) 0%,transparent 70%);border-radius:50%;}
.hero-tag{display:inline-block;background:rgba(255,255,255,.15);backdrop-filter:blur(6px);border:1px solid rgba(255,255,255,.2);padding:5px 16px;border-radius:20px;font-size:.7rem;font-weight:600;color:#ffffff;letter-spacing:.12em;text-transform:uppercase;margin-bottom:14px;}
.hero-subtitle{font-family:'Inter',sans-serif;font-size:.78rem;font-weight:500;color:rgba(255,255,255,.7);letter-spacing:.14em;text-transform:uppercase;margin-bottom:8px;}
.hero-title{font-family:'Space Grotesk',sans-serif;font-size:2.6rem;font-weight:700;color:#ffffff;line-height:1.15;margin-bottom:14px;}
.hero-desc{font-family:'Inter',sans-serif;font-size:.95rem;color:rgba(255,255,255,.78);line-height:1.6;max-width:700px;}
.hero-stats{display:flex;gap:32px;margin-top:22px;flex-wrap:wrap;}
.hero-stat{display:flex;flex-direction:column;}
.hero-stat-value{font-family:'Space Grotesk',sans-serif;font-size:1.35rem;font-weight:700;color:#ffffff;}
.hero-stat-label{font-size:.72rem;color:rgba(255,255,255,.6);text-transform:uppercase;letter-spacing:.08em;margin-top:2px;}

/* ── Blue Metric Cards ── */
.blue-card-row{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:22px;}
.blue-card{flex:1;min-width:180px;background:linear-gradient(135deg,#0b3d91 0%,#0b63ce 55%,#1d8fff 100%);border-radius:16px;padding:24px 26px 20px;position:relative;overflow:hidden;box-shadow:0 8px 28px rgba(11,99,206,.28);transition:transform .22s ease,box-shadow .22s ease;cursor:default;}
.blue-card:hover{transform:translateY(-4px);box-shadow:0 16px 40px rgba(11,99,206,.38);}
.blue-card::before{content:'';position:absolute;top:-30px;right:-30px;width:120px;height:120px;background:radial-gradient(circle,rgba(255,255,255,.12) 0%,transparent 70%);border-radius:50%;}
.blue-card-icon{font-size:1.6rem;margin-bottom:10px;opacity:.9;}
.blue-card-label{font-family:'Inter',sans-serif;font-size:.72rem;font-weight:600;color:rgba(255,255,255,.7);text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;}
.blue-card-value{font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;color:#ffffff;line-height:1.1;}
.blue-card-sub{font-family:'Inter',sans-serif;font-size:.78rem;color:rgba(255,255,255,.65);margin-top:6px;}
.blue-card-feat{flex:1;min-width:200px;background:linear-gradient(135deg,#0b3d91 0%,#0b63ce 55%,#1d8fff 100%);border-radius:20px;padding:32px 30px 26px;position:relative;overflow:hidden;box-shadow:0 12px 36px rgba(11,99,206,.32);transition:transform .22s ease,box-shadow .22s ease;cursor:default;}
.blue-card-feat:hover{transform:translateY(-5px);box-shadow:0 20px 50px rgba(11,99,206,.42);}
.blue-card-feat::before{content:'';position:absolute;top:-40px;right:-40px;width:160px;height:160px;background:radial-gradient(circle,rgba(255,255,255,.12) 0%,transparent 70%);border-radius:50%;}
.blue-card-feat::after{content:'';position:absolute;bottom:-30px;left:-20px;width:100px;height:100px;background:radial-gradient(circle,rgba(255,255,255,.07) 0%,transparent 70%);border-radius:50%;}
.blue-card-feat .blue-card-icon{font-size:2rem;margin-bottom:14px;}
.blue-card-feat .blue-card-label{font-size:.78rem;letter-spacing:.12em;}
.blue-card-feat .blue-card-value{font-size:2.7rem;}
.blue-card-feat .blue-card-sub{font-size:.85rem;margin-top:8px;}
</style>""", unsafe_allow_html=True)

# Fixed top-right logo 
def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

try:
    _logo_b64 = _img_to_b64("kayfa_icon.png")
    _logo_src = f"data:image/png;base64,{_logo_b64}"
except Exception:
    _logo_src = ""

if _logo_src:
    st.markdown(
        f'<div class="kayfa-top-logo">'
        f'<img src="{_logo_src}" alt="Kayfa logo">'
        f'</div>',
        unsafe_allow_html=True,
    )

DARK = dict(
    template="plotly_white",
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    margin=dict(l=4, r=4, t=28, b=4),
    font=dict(family="Inter", color="#14213d"),
    title_font=dict(color="#0b63ce"),
    xaxis=dict(gridcolor="#e8f1fb", zerolinecolor="#d8e6f7"),
    yaxis=dict(gridcolor="#e8f1fb", zerolinecolor="#d8e6f7"),
)
COLORS = ["#0b63ce", "#1d8fff", "#4aa3ff", "#73b9ff", "#99ccff", "#0a4ea3", "#3b82f6", "#60a5fa"]
SEQ = "Blues"

_plotly_chart = st.plotly_chart
_plotly_chart_counter = 0

def plotly_chart(fig, *args, **kwargs):
    global _plotly_chart_counter
    _plotly_chart_counter += 1

    if kwargs.pop("use_container_width", False):
        kwargs.setdefault("width", "stretch")
    kwargs.setdefault("key", f"plotly_chart_{_plotly_chart_counter}")

    return _plotly_chart(fig, *args, **kwargs)

st.plotly_chart = plotly_chart

def obs(text):
    st.markdown(f'<div class="obs-box">💡 {text}</div>', unsafe_allow_html=True)

def blue_metric(label, value, sub=None, icon="📊", featured=False):
    """Render a premium blue metric card."""
    cls = "blue-card-feat" if featured else "blue-card"
    sub_html = f'<div class="blue-card-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="{cls}">'
        f'<div class="blue-card-icon">{icon}</div>'
        f'<div class="blue-card-label">{label}</div>'
        f'<div class="blue-card-value">{value}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True
    )

# ── Load data — MongoDB Atlas first, local CSV/XLSX fallback ─────────────────
# MongoDB Atlas is used when MONGO_URI is set in .streamlit/secrets.toml AND
# the collections have been uploaded via upload_to_mongodb.py.
# If Atlas is not yet configured or the collection is empty, the app falls
# back to reading the original local data files transparently.

def _parse_datetimes(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
    return df


@st.cache_data(show_spinner=False)
def load_from_csv():
    """Read all 8 source files from disk (original behaviour)."""
    _courses     = pd.read_csv("data/courses_clean.csv")
    _groups      = pd.read_csv("data/groups_cleaned.csv")
    _students    = pd.read_csv("data/students_clean.csv")
    _grades      = pd.read_csv("data/grades_clean.csv")
    _attendance  = pd.read_excel("data/attendance_clean.xlsx")
    _concepts    = pd.read_csv("data/concepts_performance_clean.csv")
    _engagement  = pd.read_csv("data/engagement_events_clean.csv")
    _submissions = pd.read_csv("data/assignment_submissions_clean.csv")

    _students["age"]                      = pd.to_numeric(_students["age"], errors="coerce")
    _students["enrollment_date"]          = pd.to_datetime(_students["enrollment_date"], errors="coerce")
    _grades["score"]                      = pd.to_numeric(_grades["score"], errors="coerce")
    _grades["date"]                       = pd.to_datetime(_grades["date"], errors="coerce")
    _attendance["session_datetime"]       = pd.to_datetime(_attendance["session_datetime"], errors="coerce")
    _engagement["event_datetime"]         = pd.to_datetime(_engagement["event_datetime"], errors="coerce")
    _engagement["duration_seconds"]       = pd.to_numeric(_engagement["duration_seconds"], errors="coerce")
    _submissions["deadline"]              = pd.to_datetime(_submissions["deadline"], errors="coerce")
    _submissions["submitted_at"]          = pd.to_datetime(_submissions["submitted_at"], errors="coerce")
    _concepts["score_pct"]               = pd.to_numeric(_concepts["score_pct"], errors="coerce")
    _concepts["timestamp"]               = pd.to_datetime(_concepts["timestamp"], errors="coerce")
    if "Unnamed: 0" in _submissions.columns:
        _submissions.drop(columns=["Unnamed: 0"], inplace=True)
    _engagement.loc[~_engagement["duration_seconds"].between(0, 7200), "duration_seconds"] = np.nan
    _students = _students.merge(_groups[["group_id", "course_id", "instructor", "group_name"]], on="group_id", how="left")
    _students = _students.merge(_courses[["course_id", "course_name"]], on="course_id", how="left")
    return _courses, _groups, _students, _grades, _attendance, _concepts, _engagement, _submissions


def _try_mongo():
    """
    Attempt to load all 8 raw collections from MongoDB Atlas.
    Returns (tuple_of_8_dfs, True) on success, or (None, False) on any error
    or if any required collection is empty.
    """
    required_cols = ["status"]            # spot-check: attendance must have 'status'
    try:
        _courses     = load_collection("raw_courses")
        _groups      = load_collection("raw_groups")
        _students    = load_collection("raw_students")
        _grades      = load_collection("raw_grades")
        _attendance  = load_collection("raw_attendance")
        _concepts    = load_collection("raw_concepts")
        _engagement  = load_collection("raw_engagement")
        _submissions = load_collection("raw_submissions")

        # Validate that the key collection actually has data
        if _attendance.empty or "status" not in _attendance.columns:
            return None, False

        # Restore datetime dtypes
        _students    = _parse_datetimes(_students,    ["enrollment_date"])
        _grades      = _parse_datetimes(_grades,      ["date"])
        _attendance  = _parse_datetimes(_attendance,  ["session_datetime"])
        _engagement  = _parse_datetimes(_engagement,  ["event_datetime"])
        _submissions = _parse_datetimes(_submissions, ["deadline", "submitted_at"])
        _concepts    = _parse_datetimes(_concepts,    ["timestamp"])

        return (_courses, _groups, _students, _grades,
                _attendance, _concepts, _engagement, _submissions), True

    except Exception:
        return None, False


_mongo_result, _using_mongo = _try_mongo()

if _using_mongo:
    courses, groups, students, grades, attendance, concepts, engagement, submissions = _mongo_result
else:
    # Fallback: read from local CSV / XLSX files
    courses, groups, students, grades, attendance, concepts, engagement, submissions = load_from_csv()

with st.sidebar:
    # Logo header 
    try:
        _sb_logo_b64 = _img_to_b64("kayfa.png")
        _sb_logo_src = f"data:image/png;base64,{_sb_logo_b64}"
    except Exception:
        _sb_logo_src = ""

    _logo_img_html = f'<img src="{_sb_logo_src}" alt="Kayfa">' if _sb_logo_src else ""
    st.markdown(
        f'''
        <div class="sidebar-logo-wrap">
            <div class="sidebar-logo-box">
                {_logo_img_html}
            </div>
            <div class="sidebar-brand">Kayfa Analytics</div>
            <div class="sidebar-brand-sub">Academic Term 2025 – 2026</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # Platform Stats 
    _att_pct  = (attendance['status']=='attended').mean()*100
    _avg_gr   = grades['score'].mean()
    _at_risk  = 0  # placeholder until student_risk is computed below

    st.markdown('<div class="sb-section-title">📊 Platform Overview</div>', unsafe_allow_html=True)
    st.markdown(
        f'''
        <div class="sb-stat">
            <span class="sb-stat-icon">🎓</span>
            <div class="sb-stat-body">
                <span class="sb-stat-label">Total Students</span>
                <span class="sb-stat-value">{len(students):,}</span>
            </div>
        </div>
        <div class="sb-stat">
            <span class="sb-stat-icon">👥</span>
            <div class="sb-stat-body">
                <span class="sb-stat-label">Study Groups</span>
                <span class="sb-stat-value">{len(groups)}</span>
            </div>
        </div>
        <div class="sb-stat">
            <span class="sb-stat-icon">📚</span>
            <div class="sb-stat-body">
                <span class="sb-stat-label">Courses</span>
                <span class="sb-stat-value">{len(courses)}</span>
            </div>
        </div>
        <hr class="sb-divider">
        <div class="sb-section-title">📈 Performance</div>
        <div class="sb-stat">
            <span class="sb-stat-icon">📅</span>
            <div class="sb-stat-body">
                <span class="sb-stat-label">Avg Attendance</span>
                <span class="sb-stat-value">{_att_pct:.1f}%</span>
            </div>
        </div>
        <div class="sb-stat">
            <span class="sb-stat-icon">📈</span>
            <div class="sb-stat-body">
                <span class="sb-stat-label">Avg Grade</span>
                <span class="sb-stat-value">{_avg_gr:.1f} / 100</span>
            </div>
        </div>
        <hr class="sb-divider">
        <div class="sb-footer">Powered by Kayfa · 2025–2026</div>
        ''',
        unsafe_allow_html=True,
    )

_att_pct_hero = (attendance['status']=='attended').mean()*100
_avg_grade_hero = grades['score'].mean()
st.markdown(f'''
<div class="hero-card">
    <div class="hero-tag">Academic Analytics Intelligence</div>
    <div class="hero-subtitle">Kayfa Analytics · Academic Term 2025 – 2026</div>
    <div class="hero-title">Academic Performance<br>Intelligence Report</div>
    <div class="hero-desc">Analyse student patterns, uncover academic risks, and turn educational data into clear action plans — across {len(students):,} student records.</div>
</div>
''', unsafe_allow_html=True)

t_main, t_files, t_questions, t_suggestions = st.tabs([
    "Overview",
    "File Insights",
    "Answering Questions",
    "Suggestions & Decisions"
])

with t_main:

    student_att = attendance.groupby("student_id").apply(lambda x: (x["status"]=="attended").mean()*100).reset_index(name="att_rate")
    student_grade = grades.groupby("student_id")["score"].mean().reset_index(name="avg_grade")
    student_risk = students[["student_id","group_name","course_name"]].merge(student_att, on="student_id", how="left").merge(student_grade, on="student_id", how="left")
    student_risk["is_at_risk"] = (student_risk["att_rate"].fillna(0) < 60) | (student_risk["avg_grade"].fillna(0) < 55)
    concept_fail = concepts.groupby("concept_name")["mastery_status"].apply(lambda x: (x=="failed").mean()*100).reset_index(name="fail_rate")
    group_att_main = attendance.groupby("group_id").apply(lambda x: (x["status"]=="attended").mean()*100).reset_index(name="att_rate")
    group_att_main = group_att_main.merge(groups[["group_id","group_name","course_id"]], on="group_id", how="left").merge(courses[["course_id","course_name"]], on="course_id", how="left")
    true_group_size = students.groupby("group_id").size().reset_index(name="true_count")
    group_health = groups[["group_id","group_name","stated_num_students"]].merge(true_group_size, on="group_id", how="left").fillna({"true_count":0})
    group_health["size_gap"] = group_health["stated_num_students"] - group_health["true_count"]
    group_health["size_gap_abs"] = group_health["size_gap"].abs() + 1

    # 4 Featured KPIs 
    st.markdown("<p style='font-family:Space Grotesk,sans-serif;font-size:1.05rem;font-weight:600;color:#0b63ce;margin-bottom:4px;letter-spacing:.04em;'>KEY PERFORMANCE INDICATORS</p>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        blue_metric("Total Students", f"{len(students):,}", sub="Enrolled this term", icon="🎓", featured=True)
    with f2:
        blue_metric("Avg Attendance", f"{student_att['att_rate'].mean():.1f}%", sub="Platform-wide rate", icon="📅", featured=True)
    with f3:
        blue_metric("Avg Grade", f"{student_grade['avg_grade'].mean():.1f}", sub="Out of 100", icon="📈", featured=True)
    with f4:
        blue_metric("At-Risk Students", f"{int(student_risk['is_at_risk'].sum()):,}",
                    sub=f"{student_risk['is_at_risk'].mean()*100:.1f}% of total", icon="⚠️", featured=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left,right = st.columns(2)
    with left:
        grade_course = grades.merge(courses[["course_id","course_name"]], on="course_id", how="left")
        grade_course = grade_course.groupby("course_name")["score"].mean().reset_index(name="avg_score").sort_values("avg_score")
        fig2 = px.bar(grade_course, x="avg_score", y="course_name", orientation="h", color="avg_score",
                      color_continuous_scale=["#fc5c7d","#ffd32a","#48cfad"], title="Average Grade by Course",
                      labels={"avg_score":"Avg Score","course_name":"Course"})
        fig2.update_layout(**DARK, coloraxis_showscale=False, xaxis_range=[0,100])
        fig2.update_yaxes(tickfont_color="black", title_font_color="black")
        fig2.update_xaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig2, use_container_width=True)
    
    with right:
        risk_by_course = student_risk.groupby("course_name")["is_at_risk"].mean().mul(100).reset_index(name="risk_rate").sort_values("risk_rate")
        fig3 = px.bar(risk_by_course, x="risk_rate", y="course_name", orientation="h", color="risk_rate",
                      color_continuous_scale=["#48cfad","#ffd32a","#fc5c7d"], title="At-Risk Rate by Course",
                      labels={"risk_rate":"At-risk %","course_name":"Course"})
        fig3.update_layout(**DARK, coloraxis_showscale=False, xaxis_range=[0,100])
        fig3.update_yaxes(tickfont_color="black", title_font_color="black")
        fig3.update_xaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig3, use_container_width=True)
        

    with st.container():
        city_c = students["city"].value_counts().reset_index(); city_c.columns=["City","Count"]
        fig2 = px.bar(city_c.head(12), x="Count", y="City", orientation="h", color="Count", color_continuous_scale=SEQ, title="Top Student Cities")
        fig2.update_layout(**DARK, coloraxis_showscale=False)
        fig2.update_xaxes(title="Age", title_font_color="black", tickfont_color="black")
        fig2.update_yaxes(title="Count", title_font_color="black", tickfont_color="black")
        st.plotly_chart(fig2, use_container_width=True)

    with st.container():
        by_course = students.groupby(["course_name","group_name"]).size().reset_index(name="students")
        fig3 = px.bar(by_course, x="students", y="group_name", color="course_name", orientation="h",
                      color_discrete_sequence=COLORS, title="Students by Group and Course")
        fig3.update_layout(**DARK)
        fig3.update_xaxes(title="Age", title_font_color="black", tickfont_color="black")
        fig3.update_yaxes(title="Count", title_font_color="black", tickfont_color="black")
        st.plotly_chart(fig3, use_container_width=True)

# ═══════════════ TAB 1 — 15 QUESTIONS ═══════════════════════════════════════
with t_questions:
    st.title("Answering Questions with Charts")
    st.markdown("Each tab answers one stakeholder question with metrics, charts, and a short insight.")
    q_tabs = st.tabs(["Q1","Q2","Q3","Q4","Q5","Q6","Q7","Q8","Q9","Q10","Q11","Q12","Q13","Q14","Q15"])

    # Q1
    with q_tabs[0]:
        st.subheader("Q1 · Attendance Rate per Group")
        grp_att = attendance.groupby("group_id").apply(lambda x: (x["status"]=="attended").mean()*100).reset_index(name="att_rate")
        grp_att = grp_att.merge(groups[["group_id","group_name","course_id"]], on="group_id", how="left")
        grp_att = grp_att.merge(courses[["course_id","course_name"]], on="course_id", how="left")
        grp_att = grp_att.sort_values("att_rate")
        pavg = grp_att["att_rate"].mean()
        c1,c2,c3 = st.columns(3)
        with c1: blue_metric("Platform Avg", f"{pavg:.1f}%", sub="Attendance rate", icon="📅")
        with c2: blue_metric("Lowest Group", grp_att.iloc[0]['group_name'], sub=f"{grp_att.iloc[0]['att_rate']:.1f}%", icon="📉")
        with c3: blue_metric("Highest Group", grp_att.iloc[-1]['group_name'], sub=f"{grp_att.iloc[-1]['att_rate']:.1f}%", icon="📈")
        clr = [("#fc5c7d" if r < pavg else "#48cfad") for r in grp_att["att_rate"]]
        fig = go.Figure(go.Bar(y=grp_att["group_name"], x=grp_att["att_rate"], orientation="h",
            marker_color=clr, text=grp_att["att_rate"].round(1).astype(str)+"%", textposition="outside",
            hovertemplate="<b>%{y}</b><br>Attendance: %{x:.1f}%<extra></extra>"))
        fig.add_vline(x=pavg, line_dash="dash", line_color="#ffd32a",
                      annotation_text=f"Avg {pavg:.1f}%", annotation_position="top right")
        fig.update_layout(**DARK, title="Attendance Rate by Group", xaxis_title="Attendance %", xaxis_range=[0,115])
        fig.update_yaxes(tickfont_color="black", title_font_color="black")
        fig.update_xaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig, use_container_width=True)
        
        below = grp_att[grp_att["att_rate"] < pavg].sort_values("att_rate")
        obs(f"**{len(below)} groups** fall below the platform average ({pavg:.1f}%). "
            f"**{below.iloc[0]['group_name']}** ({below.iloc[0]['course_name']}) is lowest at "
            f"**{below.iloc[0]['att_rate']:.1f}%** — {pavg-below.iloc[0]['att_rate']:.1f}pp below average. Immediate investigation recommended.")

    # Q2
    with q_tabs[1]:
        st.subheader("Q2 · Score Distribution by Assessment Type")
        stats2 = grades.groupby("type")["score"].agg(["mean","std","median"]).reset_index()
        stats2.columns = ["Type","Mean","Std","Median"]
        clr_map = {"quiz":"#6c63ff","assignment":"#48cfad","practical":"#ffd32a","exam":"#fc5c7d"}
        cols = st.columns(4)
        icons2 = {"Quiz":"📝","Assignment":"📋","Practical":"🔬","Exam":"📚"}
        for col,(_,row) in zip(cols, stats2.iterrows()):
            with col: blue_metric(row['Type'].title(), f"{row['Mean']:.1f}", sub=f"σ={row['Std']:.1f}", icon=icons2.get(row['Type'].title(),'📊'))
        col1,col2 = st.columns(2)
        with col1:
            fig = px.box(grades, x="type", y="score", color="type", color_discrete_map=clr_map, points="outliers",
                         labels={"type":"Type","score":"Score"}, title="Score Distribution per Type")
            fig.update_layout(**DARK, showlegend=False)
            fig.update_yaxes(tickfont_color="black", title_font_color="black")
            fig.update_xaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.violin(grades, x="type", y="score", color="type", color_discrete_map=clr_map, box=True,
                             labels={"type":"Type","score":"Score"}, title="Density Shape per Type")
            fig2.update_layout(**DARK, showlegend=False)
            fig2.update_yaxes(tickfont_color="black", title_font_color="black")
            fig2.update_xaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig2, use_container_width=True)
        fig3 = px.histogram(grades, x="score", color="type", nbins=40, barmode="overlay", opacity=0.7,
                            color_discrete_map=clr_map, labels={"score":"Score","type":"Type"}, title="Score Histogram — All Types Overlaid")
        fig3.update_layout(**DARK)
        fig3.update_yaxes(tickfont_color="black", title_font_color="black")
        fig3.update_xaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig3, use_container_width=True)
        most_vol = stats2.sort_values("Std",ascending=False).iloc[0]
        obs(f"**{most_vol['Type'].title()}** is the most volatile (σ={most_vol['Std']:.1f}). "
            f"Quizzes show a left skew with a cluster of low scorers — question difficulty may need recalibration. "
            f"Exams show the tightest distribution, reflecting standardised conditions.")

    # Q3
    with q_tabs[2]:
        st.subheader("Q3 · Course Grade Comparison")
        g3 = grades.merge(courses[["course_id","course_name"]], on="course_id", how="left")
        cstats = g3.groupby("course_name")["score"].agg(["mean","std","median","count"]).reset_index()
        cstats.columns = ["Course","Mean","Std","Median","N"]
        cstats = cstats.sort_values("Mean", ascending=False)
        c1,c2,c3 = st.columns(3)
        with c1: blue_metric("Highest Avg", cstats.iloc[0]['Course'], sub=f"{cstats.iloc[0]['Mean']:.1f} score", icon="🏆")
        with c2: blue_metric("Lowest Avg", cstats.iloc[-1]['Course'], sub=f"{cstats.iloc[-1]['Mean']:.1f} score", icon="📉")
        with c3: blue_metric("Gap", f"{cstats.iloc[0]['Mean']-cstats.iloc[-1]['Mean']:.1f} pts", sub="Between top & bottom", icon="📏")
        col1,col2 = st.columns(2)
        with col1:
            fig = px.bar(cstats.sort_values("Mean"), x="Mean", y="Course", orientation="h",
                         color="Mean", color_continuous_scale=["#fc5c7d","#ffd32a","#48cfad"],
                         text=cstats.sort_values("Mean")["Mean"].round(1), title="Average Grade per Course",
                         labels={"Mean":"Avg Score","Course":""})
            fig.update_traces(texttemplate="%{text}", textposition="outside")
            fig.update_layout(**DARK, coloraxis_showscale=False, xaxis_range=[0,110])
            fig.update_yaxes(tickfont_color="black", title_font_color="black")
            fig.update_xaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(cstats.sort_values("Std"), x="Std", y="Course", orientation="h",
                          color="Std", color_continuous_scale=["#48cfad","#ffd32a","#fc5c7d"],
                          text=cstats.sort_values("Std")["Std"].round(1), title="Grade Spread (Std Dev)",
                          labels={"Std":"Std Dev","Course":""})
            fig2.update_traces(texttemplate="%{text}", textposition="outside")
            fig2.update_layout(**DARK, coloraxis_showscale=False)
            fig2.update_yaxes(tickfont_color="black", title_font_color="black")
            fig2.update_xaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig2, use_container_width=True)
        fig3 = px.box(g3, x="course_name", y="score", color="course_name", color_discrete_sequence=COLORS, points="outliers",
                      labels={"course_name":"Course","score":"Score"}, title="Full Score Distribution — All Courses")
        fig3.update_layout(**DARK, showlegend=False, xaxis_tickangle=-15)
        fig3.update_yaxes(tickfont_color="black", title_font_color="black")
        fig3.update_xaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig3, use_container_width=True)
        obs(f"**{cstats.iloc[0]['Course']}** leads at {cstats.iloc[0]['Mean']:.1f} avg; "
            f"**{cstats.iloc[-1]['Course']}** lags at {cstats.iloc[-1]['Mean']:.1f}. "
            f"The widest spread course ({cstats.sort_values('Std',ascending=False).iloc[0]['Course']}, σ={cstats.sort_values('Std',ascending=False).iloc[0]['Std']:.1f}) "
            f"shows highly polarised outcomes — some students excel while others struggle significantly.")

    # Q4
    with q_tabs[3]:
        st.subheader("Q4 · Attendance Rate vs Average Grade")
        att_s = attendance.groupby("student_id").apply(lambda x: (x["status"]=="attended").mean()*100).reset_index(name="att_rate")
        avg_g = grades.groupby("student_id")["score"].mean().reset_index(name="avg_grade")
        df4 = att_s.merge(avg_g, on="student_id").merge(students[["student_id","course_name","group_name"]], on="student_id", how="left")
        corr = df4["att_rate"].corr(df4["avg_grade"])
        strength4 = "Strong" if abs(corr)>0.6 else ("Moderate" if abs(corr)>0.3 else "Weak")
        c1,c2,c3 = st.columns(3)
        with c1: blue_metric("Pearson r", f"{corr:.3f}", sub="Correlation coefficient", icon="📊")
        with c2: blue_metric("Students", f"{len(df4):,}", sub="In analysis", icon="🎓")
        with c3: blue_metric("Strength", strength4, sub="Correlation strength", icon="💪")
        fig = px.scatter(df4, x="att_rate", y="avg_grade", color="course_name", trendline="ols", opacity=0.6,
                         color_discrete_sequence=COLORS, labels={"att_rate":"Attendance (%)","avg_grade":"Avg Grade","course_name":"Course"},
                         title="Attendance Rate vs Average Grade (OLS trend)")
        fig.update_layout(**DARK)
        fig.update_yaxes(tickfont_color="black", title_font_color="black")
        fig.update_xaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig, use_container_width=True)
        df4["Band"] = pd.qcut(df4["att_rate"].rank(method="first"), 4, labels=["Q1 Low","Q2","Q3","Q4 High"])
        qdf = df4.groupby("Band")["avg_grade"].mean().reset_index()
        fig2 = px.bar(qdf, x="Band", y="avg_grade", color="avg_grade", color_discrete_sequence=["#4CAF50"],
                      text=qdf["avg_grade"].round(1), title="Avg Grade by Attendance Quartile",
                      labels={"avg_grade":"Avg Grade"})
        fig2.update_traces(texttemplate="%{text}", textposition="outside")
        fig2.update_layout(**DARK, coloraxis_showscale=False, yaxis_range=[0,100])
        fig2.update_yaxes(tickfont_color="black", title_font_color="black")
        fig2.update_xaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig2, use_container_width=True)
        obs(f"**{strength4} positive correlation** (r={corr:.3f}). "
            f"Top-quartile attendees score {qdf.iloc[-1]['avg_grade']-qdf.iloc[0]['avg_grade']:.1f} pts higher than the lowest quartile. "
            f"Attendance is a reliable proxy for academic engagement.")

    # Q5
    with q_tabs[4]:
        st.subheader("Q5 · Engagement vs Academic Performance")
        logins = engagement[engagement["event_type"]=="login"].groupby("student_id").size().reset_index(name="logins")
        video  = engagement[engagement["event_type"]=="video_watch"].groupby("student_id")["duration_seconds"].sum().reset_index(name="video_sec")
        video["video_hrs"] = video["video_sec"]/3600
        avg_g5 = grades.groupby("student_id")["score"].mean().reset_index(name="avg_grade")
        df5 = avg_g5.merge(logins, on="student_id", how="left").merge(video[["student_id","video_hrs"]], on="student_id", how="left").merge(students[["student_id","course_name"]], on="student_id", how="left")
        df5[["logins","video_hrs"]] = df5[["logins","video_hrs"]].fillna(0)
        r_login = df5["logins"].corr(df5["avg_grade"])
        r_video = df5["video_hrs"].corr(df5["avg_grade"])
        c1,c2,c3 = st.columns(3)
        with c1: blue_metric("Login ↔ Grade r", f"{r_login:.3f}", sub="Correlation", icon="🔑")
        with c2: blue_metric("Video ↔ Grade r", f"{r_video:.3f}", sub="Correlation", icon="🎬")
        with c3: blue_metric("Avg Logins/Student", f"{df5['logins'].mean():.1f}", sub="Per student", icon="📲")
        col1,col2 = st.columns(2)
        with col1:
            fig = px.scatter(df5, x="logins", y="avg_grade", color="course_name", trendline="ols", opacity=0.6,
                             color_discrete_sequence=COLORS, labels={"logins":"Login Count","avg_grade":"Avg Grade","course_name":"Course"},
                             title="Login Frequency vs Avg Grade")
            fig.update_layout(**DARK)
            fig.update_yaxes(tickfont_color="black", title_font_color="black")
            fig.update_xaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(df5, x="video_hrs", y="avg_grade", color="course_name", trendline="ols", opacity=0.6,
                              color_discrete_sequence=COLORS, labels={"video_hrs":"Video (hrs)","avg_grade":"Avg Grade","course_name":"Course"},
                              title="Video Watch Time vs Avg Grade")
            fig2.update_layout(**DARK)
            st.plotly_chart(fig2, use_container_width=True)
        eng_counts = engagement.groupby("event_type").size().reset_index(name="count")
        fig3 = px.pie(eng_counts, values="count", names="event_type", color_discrete_sequence=COLORS, hole=0.45,
                      title="Platform Event Type Breakdown")
        fig3.update_layout(**DARK)
        fig3.update_yaxes(tickfont_color="black", title_font_color="black")
        fig3.update_xaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig3, use_container_width=True)
        obs(f"Login (r={r_login:.3f}) and video watch (r={r_video:.3f}) both positively associate with grades. "
            f"Students logging in more frequently and watching more content consistently trend toward higher performance.")

    # Q6
    with q_tabs[5]:
        st.subheader("Q6 · Concept Failure Rates — Biggest Curriculum Weak Spots")
        fail6 = concepts.groupby(["concept_name","course_id"])["mastery_status"].apply(lambda x: (x=="failed").mean()*100).reset_index(name="fail_rate")
        fail6 = fail6.merge(courses[["course_id","course_name"]], on="course_id", how="left")
        fail6 = fail6.sort_values("fail_rate", ascending=False)
        worst6 = fail6.iloc[0]["concept_name"]; worst6_course = fail6.iloc[0]["course_name"]
        c1,c2,c3 = st.columns(3)
        with c1: blue_metric("Biggest Weak Spot", worst6, sub="Highest failure", icon="❌")
        with c2: blue_metric("Course", worst6_course, sub="Affected course", icon="📚")
        with c3: blue_metric("Failure Rate", f"{fail6.iloc[0]['fail_rate']:.1f}%", sub="Of attempts", icon="⚠️")
        top15 = fail6.head(15).sort_values("fail_rate")
        fig = px.bar(top15, x="fail_rate", y="concept_name", orientation="h", color="course_name",
                     color_discrete_sequence=COLORS, text=top15["fail_rate"].round(1).astype(str)+"%",
                     labels={"fail_rate":"Failure (%)","concept_name":"Concept","course_name":"Course"},
                     title="Top 15 Concepts by Failure Rate")
        fig.update_traces(textposition="outside")
        fig.update_layout(**DARK, xaxis_range=[0,110])
        fig.update_yaxes(tickfont_color="black", title_font_color="black")
        fig.update_xaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig, use_container_width=True)
        
        heat6 = concepts.groupby(["course_id","concept_name"])["mastery_status"].apply(lambda x: (x=="failed").mean()*100).reset_index(name="fail_rate")
        heat6 = heat6.merge(courses[["course_id","course_name"]], on="course_id")
        pivot6 = heat6.pivot_table(index="course_name", columns="concept_name", values="fail_rate", fill_value=0)
        fig2 = px.imshow(pivot6, color_continuous_scale="Reds", aspect="auto",
                         title="Failure Rate Heatmap — Course × Concept", labels={"color":"Fail %"})
        fig2.update_layout(**DARK, xaxis_tickangle=-40)
        fig2.update_yaxes(tickfont_color="black", title_font_color="black")
        fig2.update_xaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig2, use_container_width=True)
        obs(f"**{worst6}** in **{worst6_course}** has a {fail6.iloc[0]['fail_rate']:.1f}% failure rate — the single biggest curriculum weak spot. "
            f"The heatmap reveals systemic clusters: advanced algorithmic and ML concepts dominate failures. "
            f"Curriculum redesign and dedicated support sessions are strongly recommended.")

    # Q7
    with q_tabs[6]:
        st.subheader("Q7 · Weakest Concept Mastery Over Time")
        fail7 = concepts.groupby("concept_name")["mastery_status"].apply(lambda x: (x=="failed").mean()*100).reset_index(name="fail_rate")
        worst7 = fail7.sort_values("fail_rate",ascending=False).iloc[0]["concept_name"]
        wc = concepts[concepts["concept_name"]==worst7].copy()
        wc["month"] = wc["timestamp"].dt.to_period("M").astype(str)
        trend7 = wc.groupby("month").agg(pass_rate=("mastery_status", lambda x: (x=="passed").mean()*100), attempts=("mastery_status","count")).reset_index()
        d_val = trend7["pass_rate"].iloc[-1] - trend7["pass_rate"].iloc[0]
        c1,c2,c3 = st.columns(3)
        with c1: blue_metric("Concept", worst7, sub="Weakest concept", icon="🎯")
        with c2: blue_metric("Pass Rate (Start)", f"{trend7.iloc[0]['pass_rate']:.1f}%", sub="Beginning of term", icon="🏁")
        with c3: blue_metric("Pass Rate (Latest)", f"{trend7.iloc[-1]['pass_rate']:.1f}%", sub=f"{d_val:+.1f}pp change", icon="📈" if d_val >= 0 else "📉")
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Scatter(x=trend7["month"], y=trend7["pass_rate"], mode="lines+markers+text",
                                 text=trend7["pass_rate"].round(1).astype(str)+"%", textposition="top center",
                                 name="Pass Rate %", line=dict(color="#48cfad",width=3), marker=dict(size=9)), secondary_y=False)
        fig.add_trace(go.Bar(x=trend7["month"], y=trend7["attempts"], name="Attempts", marker_color="#6c63ff", opacity=0.4), secondary_y=True)
        fig.update_layout(**DARK, title=f"'{worst7}' — Cohort Pass Rate Over Time")
        fig.update_yaxes(title_text="Pass Rate %", secondary_y=False)
        fig.update_xaxes(tickfont_color="black", title_font_color="black")
        fig.update_yaxes(title_text="Attempts", secondary_y=True)
        fig.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig, use_container_width=True)
        trend_word = "improving" if d_val > 5 else ("declining" if d_val < -5 else "flat")
        obs(f"Mastery of **{worst7}** is **{trend_word}** (Δ{d_val:+.1f}pp over the term). "
            f"Despite increased attempts in some months, pass rates remain critically low. "
            f"A fundamental curriculum rethink for this concept is overdue.")

    # Q8
    with q_tabs[7]:
        st.subheader("Q8 · Late Submissions vs Score")
        sub8 = submissions.merge(grades[["assessment_id","student_id","score"]], on=["assessment_id","student_id"], how="left")
        sub8["buffer_hours"] = (sub8["deadline"] - sub8["submitted_at"]).dt.total_seconds()/3600
        sub8 = sub8.dropna(subset=["score"])
        late_avg   = sub8[sub8["is_late"]==True]["score"].mean()
        ontime_avg = sub8[sub8["is_late"]==False]["score"].mean()
        c1,c2,c3 = st.columns(3)
        with c1: blue_metric("On-Time Avg", f"{ontime_avg:.1f}", sub="Score when on time", icon="✅")
        with c2: blue_metric("Late Avg", f"{late_avg:.1f}", sub="Score when late", icon="⏰")
        with c3: blue_metric("Penalty Gap", f"{ontime_avg-late_avg:.1f} pts", sub="Late vs on-time", icon="📉")
        col1,col2 = st.columns(2)
        with col1:
            fig = px.box(sub8, x="is_late", y="score", color="is_late",
                         color_discrete_map={True:"#fc5c7d",False:"#48cfad"},
                         labels={"is_late":"Late","score":"Score"}, title="Score: Late vs On-Time")
            fig.update_layout(**DARK, showlegend=False)
            fig.update_xaxes(tickvals=[False,True], ticktext=["On Time","Late"])
            fig.update_yaxes(tickfont_color="black", title_font_color="black")
            fig.update_xaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            sub8c = sub8[(sub8["buffer_hours"]>-500)&(sub8["buffer_hours"]<500)]
            fig2 = px.scatter(sub8c, x="buffer_hours", y="score", color="is_late",
                              color_discrete_map={True:"#fc5c7d",False:"#48cfad"}, trendline="ols", opacity=0.5,
                              labels={"buffer_hours":"Buffer (hrs)","score":"Score","is_late":"Late"},
                              title="Submission Buffer vs Score")
            fig2.update_layout(**DARK)
            fig2.update_xaxes(tickfont_color="black", title_font_color="black")
            fig2.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig2, use_container_width=True)
        sub8["Band8"] = pd.cut(sub8["buffer_hours"].clip(-200,200), bins=[-200,-1,6,24,200], labels=["Late","<6h","6–24h",">24h"])
        band_avg = sub8.groupby("Band8")["score"].mean().reset_index()
        fig3 = px.bar(band_avg, x="Band8", y="score", color="score",
                      color_continuous_scale=["#fc5c7d","#ffd32a","#48cfad"], text=band_avg["score"].round(1),
                      title="Avg Score by Submission Timing Band", labels={"Band8":"Timing","score":"Avg Score"})
        fig3.update_traces(texttemplate="%{text}", textposition="outside")
        fig3.update_layout(**DARK, coloraxis_showscale=False, yaxis_range=[0,100])
        fig3.update_xaxes(tickfont_color="black", title_font_color="black")
        fig3.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig3, use_container_width=True)
        obs(f"Late submitters score **{ontime_avg-late_avg:.1f} pts lower** ({late_avg:.1f} vs {ontime_avg:.1f}). "
            f"The earlier a student submits, the higher they score — procrastination is a measurable academic risk. "
            f"Deadline reminder nudges and buffer-time incentives could meaningfully improve outcomes.")

    # Q9
    with q_tabs[8]:
        st.subheader("Q9 · Attendance & Engagement Timeline")
        att9 = attendance.copy()
        att9["week"] = att9["session_datetime"].dt.to_period("W").astype(str)
        att_wk = att9.groupby("week").apply(lambda x: (x["status"]=="attended").mean()*100).reset_index(name="att_rate")
        eng9 = engagement.copy()
        eng9["week"] = eng9["event_datetime"].dt.to_period("W").astype(str)
        eng_wk = eng9.groupby("week").size().reset_index(name="events")
        merged9 = att_wk.merge(eng_wk, on="week", how="outer").sort_values("week")
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Scatter(x=merged9["week"], y=merged9["att_rate"], name="Attendance %",
                                 line=dict(color="#48cfad",width=2.5), mode="lines"), secondary_y=False)
        fig.add_trace(go.Bar(x=merged9["week"], y=merged9["events"], name="Engagement Events",
                             marker_color="#6c63ff", opacity=0.5), secondary_y=True)
        fig.update_layout(**DARK, title="Weekly Attendance & Engagement Over 6 Months")
        fig.update_yaxes(title_text="Attendance %", secondary_y=False, tickfont_color="black", title_font_color="black")
        fig.update_yaxes(title_text="Events", secondary_y=True, tickfont_color="black", title_font_color="black")
        fig.update_xaxes(tickfont_color="black",title_font_color="black")
        st.plotly_chart(fig, use_container_width=True)

        att9["month"] = att9["session_datetime"].dt.month_name()
        att9["dow"]   = att9["session_datetime"].dt.day_name()
        heat9 = att9.groupby(["month","dow"]).apply(lambda x: (x["status"]=="attended").mean()*100).reset_index(name="rate")
        pivot9 = heat9.pivot_table(index="month", columns="dow", values="rate")
        day_order = ["Saturday","Sunday","Monday","Tuesday","Wednesday","Thursday","Friday"]
        pivot9 = pivot9.reindex(columns=[d for d in day_order if d in pivot9.columns])
        fig2 = px.imshow(pivot9, color_continuous_scale="RdYlGn", aspect="auto",
                         title="Attendance Heatmap — Month × Day of Week", labels={"color":"Att %"})
        fig2.update_layout(**DARK)
        fig2.update_xaxes(tickfont_color="black", title_font_color="black")
        fig2.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig2, use_container_width=True)
        att_dip = merged9.dropna(subset=["att_rate"])
        dip_week = att_dip.loc[att_dip["att_rate"].idxmin(), "week"]
        dip_val  = att_dip["att_rate"].min()
        obs(f"A cohort-wide dip appears around **{dip_week}** (attendance: {dip_val:.1f}%). "
            f"Synchronised drops across all groups typically indicate a **national holiday, exam crunch, or platform outage**. "
            f"The day-of-week heatmap reveals consistent variation — some days reliably underperform.")

    # Q10
    with q_tabs[9]:
        st.subheader("Q10 · Age Band Analysis")
        avg_g10 = grades.groupby("student_id")["score"].mean().reset_index(name="avg_grade")
        att10   = attendance.groupby("student_id").apply(lambda x: (x["status"]=="attended").mean()*100).reset_index(name="att_rate")
        logins10 = engagement[engagement["event_type"]=="login"].groupby("student_id").size().reset_index(name="logins")
        df10 = students[["student_id","age"]].merge(avg_g10, on="student_id", how="left").merge(att10, on="student_id", how="left").merge(logins10, on="student_id", how="left")
        df10 = df10.dropna(subset=["age"])
        df10["age_band"] = pd.cut(df10["age"], bins=[0,19,22,25,30,100], labels=["≤19","20–22","23–25","26–30","31+"])
        band_stats = df10.groupby("age_band").agg(avg_grade=("avg_grade","mean"), att_rate=("att_rate","mean"), logins=("logins","mean"), count=("student_id","count")).reset_index()
        best_band = band_stats.sort_values("avg_grade",ascending=False).iloc[0]
        c1,c2,c3 = st.columns(3)
        with c1: blue_metric("Best Grade Band", str(best_band['age_band']), sub=f"{best_band['avg_grade']:.1f} avg score", icon="🏆")
        with c2: blue_metric("Most Engaged Band", str(band_stats.sort_values('logins',ascending=False).iloc[0]['age_band']), sub="Highest logins", icon="📲")
        with c3: blue_metric("Best Attendance Band", str(band_stats.sort_values('att_rate',ascending=False).iloc[0]['age_band']), sub="Top attendance", icon="📅")
        col1,col2 = st.columns(2)
        with col1:
            fig = px.bar(band_stats, x="age_band", y="avg_grade", color="avg_grade", color_discrete_sequence=["#4CAF50"],
                         text=band_stats["avg_grade"].round(1), title="Avg Grade by Age Band",
                         labels={"age_band":"Age Band","avg_grade":"Avg Grade"})
            fig.update_traces(texttemplate="%{text}", textposition="outside")
            fig.update_layout(**DARK, coloraxis_showscale=False, yaxis_range=[0,100])
            fig.update_xaxes(tickfont_color="black", title_font_color="black")
            fig.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.bar(band_stats, x="age_band", y="att_rate", color="att_rate", color_continuous_scale=SEQ,
                          text=band_stats["att_rate"].round(1), title="Avg Attendance by Age Band",
                          labels={"age_band":"Age Band","att_rate":"Attendance %"})
            fig2.update_traces(texttemplate="%{text}", textposition="outside")
            fig2.update_layout(**DARK, coloraxis_showscale=False, yaxis_range=[0,100])
            fig2.update_xaxes(tickfont_color="black", title_font_color="black")
            fig2.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig2, use_container_width=True)

    # Q11
    with q_tabs[10]:
        st.subheader("Q11 · Student Segmentation (K-Means)")
        att11   = attendance.groupby("student_id").apply(lambda x: (x["status"]=="attended").mean()*100).reset_index(name="att_rate")
        avg_g11 = grades.groupby("student_id")["score"].mean().reset_index(name="avg_grade")
        logins11 = engagement[engagement["event_type"]=="login"].groupby("student_id").size().reset_index(name="logins")
        video11  = engagement[engagement["event_type"]=="video_watch"].groupby("student_id")["duration_seconds"].sum().reset_index(name="video_sec")
        failed11 = concepts[concepts["mastery_status"]=="failed"].groupby("student_id").size().reset_index(name="failed_c")
        df11 = att11.merge(avg_g11, on="student_id", how="outer").merge(logins11, on="student_id", how="outer").merge(video11[["student_id","video_sec"]], on="student_id", how="outer").merge(failed11, on="student_id", how="outer").fillna(0)
        features = ["att_rate","avg_grade","logins","video_sec","failed_c"]
        X = StandardScaler().fit_transform(df11[features])
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        df11["cluster"] = km.fit_predict(X)
        seg_stats = df11.groupby("cluster")[features].mean().reset_index()
        seg_labels = {}
        for _,row in seg_stats.iterrows():
            c = row["cluster"]
            if row["avg_grade"]>=75 and row["att_rate"]>=75: seg_labels[c]="🏆 High Achievers"
            elif row["avg_grade"]<60 and row["att_rate"]<50: seg_labels[c]="🚨 At-Risk / Disengaged"
            elif row["logins"]>=df11["logins"].median() and row["avg_grade"]<70: seg_labels[c]="⚡ Engaged but Struggling"
            else: seg_labels[c]="📈 Average / Improving"
        df11["segment"] = df11["cluster"].map(seg_labels)
        seg_counts = df11["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment","Count"]
        df11s = df11.merge(students[["student_id","course_name"]], on="student_id", how="left")
        fig2 = px.scatter(df11s, x="att_rate", y="avg_grade", color="segment", size="logins", opacity=0.7,
                          color_discrete_sequence=["#48cfad","#fc5c7d","#ffd32a","#6c63ff"],
                          labels={"att_rate":"Attendance %","avg_grade":"Avg Grade","segment":"Segment"},
                          title="Segments — Attendance vs Grade (size=logins)")
        fig2.update_layout(**DARK)
        fig2.update_xaxes(tickfont_color="black", title_font_color="black")
        fig2.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig2, use_container_width=True)
        
        pal = ["#48cfad","#fc5c7d","#ffd32a","#6c63ff"]
        fig3 = go.Figure()
        cats = ["Attendance","Avg Grade","Logins","Video","Mastery"]
        for i,(_,row) in enumerate(seg_stats.iterrows()):
            vals = [row["att_rate"]/100*10, row["avg_grade"]/10,
                    row["logins"]/max(df11["logins"].max(),1)*10,
                    row["video_sec"]/max(df11["video_sec"].max(),1)*10,
                    (1-row["failed_c"]/max(df11["failed_c"].max(),1))*10]
            fig3.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
                                            name=seg_labels[row["cluster"]], line=dict(color=pal[i])))
        fig3.update_layout(**DARK, polar=dict(bgcolor="#1a1d2e"), title="Segment Radar Profile (0–10 normalised)")
        fig3.update_xaxes(tickfont_color="black", title_font_color="black")
        fig3.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("**Segment Summary:**")
        for label in set(seg_labels.values()):
            r = df11[df11["segment"]==label][features].mean()
            st.markdown(f"- **{label}** — Att: {r['att_rate']:.0f}% | Grade: {r['avg_grade']:.1f} | Logins: {r['logins']:.0f} | Failed concepts: {r['failed_c']:.0f}")
        obs("4 distinct segments emerge. High Achievers combine strong attendance with high grades. "
            "At-Risk students fail on all dimensions and need urgent outreach. "
            "Engaged-but-Struggling students log in but need better study strategies. "
            "Average/Improving is the largest cluster — targeted nudges here could move many students up.")

    # Q12
    with q_tabs[11]:
        st.subheader("Q12 · True vs Stated Group Sizes")
        true_sz12 = students.groupby("group_id").size().reset_index(name="true_count")
        comp12 = groups[["group_id","group_name","course_id","stated_num_students"]].merge(true_sz12, on="group_id", how="left").fillna(0)
        comp12 = comp12.merge(courses[["course_id","course_name"]], on="course_id", how="left")
        comp12["discrepancy"] = comp12["stated_num_students"] - comp12["true_count"]
        comp12["disc_pct"] = (comp12["discrepancy"]/comp12["stated_num_students"].replace(0,np.nan)*100).fillna(100)
        comp12 = comp12.sort_values("true_count")
        c1,c2,c3 = st.columns(3)
        with c1: blue_metric("Smallest True Group", comp12.iloc[0]['group_name'], sub=f"{int(comp12.iloc[0]['true_count'])} enrolled", icon="👥")
        with c2: blue_metric("Max Overcount", f"{comp12['discrepancy'].max():.0f} students", sub="Largest gap", icon="⚠️")
        with c3: blue_metric("Total Discrepancy", f"{comp12['discrepancy'].sum():.0f} students", sub="Platform-wide gap", icon="📊")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Stated", x=comp12["group_name"], y=comp12["stated_num_students"], marker_color="#6c63ff", opacity=0.7))
        fig.add_trace(go.Bar(name="True (students.csv)", x=comp12["group_name"], y=comp12["true_count"], marker_color="#48cfad"))
        fig.update_layout(**DARK, barmode="group", title="Stated vs True Group Sizes", yaxis_title="Students", xaxis_tickangle=-15)
        fig.update_xaxes(tickfont_color="black", title_font_color="black")
        fig.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.bar(comp12.sort_values("disc_pct",ascending=False), x="group_name", y="disc_pct",
                      color="disc_pct", color_continuous_scale=["#48cfad","#ffd32a","#fc5c7d"],
                      text=comp12.sort_values("disc_pct",ascending=False)["disc_pct"].round(0).astype(int).astype(str)+"%",
                      title="Overcount % per Group", labels={"disc_pct":"Overcount %","group_name":"Group"})
        fig2.update_traces(textposition="outside")
        fig2.update_layout(**DARK, coloraxis_showscale=False, yaxis_range=[0,120])
        fig2.update_xaxes(tickfont_color="black", title_font_color="black")
        fig2.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig2, use_container_width=True)
        flagged12 = comp12[comp12["disc_pct"]>40]
        st.markdown("**🚩 Groups to investigate (>40% overcount):**")
        for _,row in flagged12.iterrows():
            st.markdown(f"- **{row['group_name']}** ({row['course_name']}) — stated {int(row['stated_num_students'])}, actual **{int(row['true_count'])}** ({row['disc_pct']:.0f}% overcount)")
        obs(f"groups.csv consistently overcounts enrollment. Total platform discrepancy: **{comp12['discrepancy'].sum():.0f} students**. "
            f"Self-reported headcounts should be replaced with live counts from students.csv in all reporting systems.")

    # Q13
    with q_tabs[12]:
        st.subheader("Q13 · Unviable Group & Merge Recommendation")
        true_sz13 = students.groupby("group_id").size().reset_index(name="true_count")
        grp13 = groups[["group_id","group_name","course_id"]].merge(true_sz13, on="group_id", how="left").fillna(0)
        grp13 = grp13.merge(courses[["course_id","course_name"]], on="course_id", how="left")
        unviable = grp13.sort_values("true_count").iloc[0]
        st.error(f"**Unviable group: {unviable['group_name']}** — only **{int(unviable['true_count'])} enrolled students** (stated: {int(groups[groups['group_id']==unviable['group_id']]['stated_num_students'].values[0])})")
        fig_sz = px.bar(grp13.sort_values("true_count"), x="true_count", y="group_name", orientation="h",
                        color="true_count", color_continuous_scale=["#fc5c7d","#ffd32a","#48cfad"],
                        text=grp13.sort_values("true_count")["true_count"].astype(int),
                        title="All Groups by True Enrollment", labels={"true_count":"Students","group_name":""})
        fig_sz.add_vline(x=10, line_dash="dash", line_color="#fc5c7d", annotation_text="Viability threshold")
        fig_sz.update_traces(textposition="outside")
        fig_sz.update_layout(**DARK, coloraxis_showscale=False)
        fig_sz.update_xaxes(tickfont_color="black", title_font_color="black")
        fig_sz.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig_sz, use_container_width=True)
        if int(unviable["true_count"]) > 0:
            from scipy.stats import pearsonr
            unv_sts = students[students["group_id"]==unviable["group_id"]]["student_id"].tolist()
            unv_con = concepts[concepts["student_id"].isin(unv_sts)].groupby("concept_name")["score_pct"].mean()
            other_grps = grp13[grp13["true_count"]>10].copy()
            sims = []
            for _,row in other_grps.iterrows():
                g_sts = students[students["group_id"]==row["group_id"]]["student_id"].tolist()
                g_con = concepts[concepts["student_id"].isin(g_sts)].groupby("concept_name")["score_pct"].mean()
                common = unv_con.index.intersection(g_con.index)
                if len(common)>3:
                    r_val,_ = pearsonr(unv_con[common], g_con[common])
                    sims.append({"group_name":row["group_name"],"course_name":row["course_name"],"true_count":int(row["true_count"]),"similarity":r_val})
            if sims:
                sim_df = pd.DataFrame(sims).sort_values("similarity",ascending=False)
                best_m = sim_df.iloc[0]
                st.success(f"**Best merge candidate:** {best_m['group_name']} ({best_m['course_name']}) — similarity r={best_m['similarity']:.3f}, {best_m['true_count']} students")
                fig_s = px.bar(sim_df.head(8), x="similarity", y="group_name", orientation="h",
                               color="similarity", color_continuous_scale=SEQ, text=sim_df.head(8)["similarity"].round(3),
                               title=f"Concept Similarity to {unviable['group_name']}", labels={"similarity":"r","group_name":"Group"})
                fig_s.update_traces(textposition="outside")
                fig_s.update_layout(**DARK, coloraxis_showscale=False, xaxis_range=[0,1.2])
                st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.warning("No enrolled students — group should be dissolved immediately.")
        obs(f"**{unviable['group_name']}** is below any viable minimum. Recommendation: dissolve and redistribute students "
            f"into the most similar group by concept profile. This preserves instructional quality for affected students.")

    # Q14
    with q_tabs[13]:
        st.subheader("Q14 · At-Risk Student Ranking — Top 10")
        att14 = attendance.groupby("student_id").apply(lambda x: (x["status"]=="attended").mean()*100).reset_index(name="att_rate")
        eng14 = engagement.copy()
        eng14["month"] = eng14["event_datetime"].dt.to_period("M").astype(str)
        eng_trend = eng14.groupby(["student_id","month"]).size().reset_index(name="events")
        months_sorted = sorted(eng_trend["month"].unique())
        mid = months_sorted[len(months_sorted)//2]
        fh = eng_trend[eng_trend["month"]<=mid].groupby("student_id")["events"].mean().reset_index(name="eng_first")
        sh = eng_trend[eng_trend["month"]>mid].groupby("student_id")["events"].mean().reset_index(name="eng_last")
        eng_delta = fh.merge(sh, on="student_id", how="outer").fillna(0)
        eng_delta["eng_decline"] = eng_delta["eng_first"] - eng_delta["eng_last"]
        avg_g14 = grades.groupby("student_id")["score"].mean().reset_index(name="avg_grade")
        fail14  = concepts[concepts["mastery_status"]=="failed"].groupby("student_id").size().reset_index(name="failed_c")
        df14 = students[["student_id","full_name","course_name","group_name"]].merge(att14, on="student_id", how="left").merge(eng_delta[["student_id","eng_decline"]], on="student_id", how="left").merge(avg_g14, on="student_id", how="left").merge(fail14, on="student_id", how="left").fillna(0)
        def norm(s): rng=s.max()-s.min(); return (s-s.min())/rng if rng>0 else s*0
        df14["risk_score"] = (norm(100-df14["att_rate"])*0.35 + norm(df14["eng_decline"].clip(0))*0.25 + norm(100-df14["avg_grade"])*0.25 + norm(df14["failed_c"])*0.15)*100
        top10 = df14.sort_values("risk_score",ascending=False).head(10)
        col1,col2 = st.columns(2)
        with col1:
            fig = px.bar(top10.sort_values("risk_score"), x="risk_score", y="full_name", orientation="h",
                         color="risk_score", color_continuous_scale=["#ffd32a","#fc5c7d"], text=top10.sort_values("risk_score")["risk_score"].round(1),
                         title="Top 10 At-Risk Students", labels={"risk_score":"Risk Score","full_name":"Student"})
            fig.update_traces(textposition="outside")
            fig.update_layout(**DARK, coloraxis_showscale=False, xaxis_range=[0,120])
            fig.update_xaxes(tickfont_color="black", title_font_color="black")
            fig.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.scatter(top10, x="att_rate", y="avg_grade", size="failed_c", color="risk_score",
                              color_continuous_scale="Reds", text="full_name", hover_data=["course_name","group_name"],
                              title="At-Risk: Attendance vs Grade", labels={"att_rate":"Att %","avg_grade":"Avg Grade"})
            fig2.update_traces(textposition="top center")
            fig2.update_layout(**DARK)
            fig2.update_xaxes(tickfont_color="black", title_font_color="black")
            fig2.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig2, use_container_width=True)
        
    # Q15
    with q_tabs[14]:
        st.subheader("Q15 · Group Grade Trends Across Assessments")
        g15 = grades.merge(groups[["group_id","group_name"]], on="group_id", how="left").sort_values("date")
        g15["seq"] = g15.groupby("group_id")["date"].transform(lambda x: pd.factorize(x.astype(str))[0]+1)
        trend15 = g15.groupby(["group_id","group_name","seq"])["score"].mean().reset_index()
        fig = px.line(trend15, x="seq", y="score", color="group_name", color_discrete_sequence=COLORS, markers=True,
                      labels={"seq":"Assessment #","score":"Avg Score","group_name":"Group"},
                      title="Group Average Grade Across Successive Assessments")
        fig.update_layout(**DARK)
        fig.update_xaxes(tickfont_color="black", title_font_color="black")
        fig.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig, use_container_width=True)
        from scipy.stats import linregress
        slopes15 = []
        for gid, gdf in trend15.groupby("group_id"):
            if len(gdf)>=3:
                slope,_,r,_,_ = linregress(gdf["seq"], gdf["score"])
                slopes15.append({"group_name":gdf["group_name"].iloc[0],"slope":slope,"r":r})
        slope_df = pd.DataFrame(slopes15).sort_values("slope",ascending=False)
        col1,col2 = st.columns(2)
        with col1:
            fig2 = px.bar(slope_df.sort_values("slope"), x="slope", y="group_name", orientation="h",
                          color="slope", color_continuous_scale=["#fc5c7d","#ffd32a","#48cfad"],
                          text=slope_df.sort_values("slope")["slope"].round(3),
                          title="Grade Trajectory per Group (pts/assessment)", labels={"slope":"Slope","group_name":"Group"})
            fig2.add_vline(x=0, line_dash="dash", line_color="#ffd32a")
            fig2.update_traces(textposition="outside")
            fig2.update_layout(**DARK, coloraxis_showscale=False)
            fig2.update_xaxes(tickfont_color="black", title_font_color="black")
            fig2.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            fl = trend15.groupby("group_name").apply(lambda x: pd.Series({"First":x.sort_values("seq").iloc[0]["score"],"Last":x.sort_values("seq").iloc[-1]["score"]})).reset_index()
            fl["Change"] = fl["Last"]-fl["First"]
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=fl["group_name"], y=fl["First"], name="First Assessment", marker_color="#6c63ff"))
            fig3.add_trace(go.Bar(x=fl["group_name"], y=fl["Last"], name="Last Assessment", marker_color="#48cfad"))
            fig3.update_layout(**DARK, barmode="group", title="First vs Last Assessment Score", xaxis_tickangle=-20)
            fig3.update_xaxes(tickfont_color="black", title_font_color="black")
            fig3.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig3, use_container_width=True)
        up15   = slope_df[slope_df["slope"]>0.05]["group_name"].tolist()
        down15 = slope_df[slope_df["slope"]<-0.05]["group_name"].tolist()
        st.markdown(f"**📈 Trending up:** {', '.join(up15) if up15 else 'None'}")
        st.markdown(f"**📉 Trending down:** {', '.join(down15) if down15 else 'None'}")
        obs(f"{len(up15)} groups improving, {len(down15)} declining. Declining groups need instructor coaching. "
            f"Slope metric enables easy prioritisation — any negative slope warrants immediate intervention.")


# ═══════════════ TAB 2 — FILE INSIGHTS ═════════════════════════════════════
with t_files:
    st.title("📂 File-Level Insights")
    f_tabs = st.tabs(["👥 Students","📚 Courses & Groups","📊 Grades","🎯 Concepts","📅 Attendance","⚡ Engagement","📝 Submissions"])

    with f_tabs[0]:
        st.subheader("Students")
        c1,c2,c3,c4 = st.columns(4)
        with c1: blue_metric("Total", str(len(students)), sub="Enrolled students", icon="🎓")
        with c2: blue_metric("Avg Age", f"{students['age'].mean():.1f}", sub="Years old", icon="🎂")
        with c3: blue_metric("Cities", str(students['city'].nunique()), sub="Represented", icon="🏙️")
        with c4: blue_metric("Female %", f"{(students['gender']=='Female').mean()*100:.0f}%", sub="Gender split", icon="👩")
        col1,col2 = st.columns(2)
        with col1:
            fig = px.histogram(students, x="age", nbins=20, color_discrete_sequence=["#6c63ff"], title="Age Distribution")
            fig.update_layout(**DARK)
            fig.update_xaxes(tickfont_color="black", title_font_color="black")
            fig.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.pie(students, names="gender", hole=0.45, color_discrete_sequence=["#6c63ff","#fc5c7d"], title="Gender")
            fig2.update_layout(**DARK)
            st.plotly_chart(fig2, use_container_width=True)
        city_c = students["city"].value_counts().reset_index(); city_c.columns=["City","Count"]
        fig3 = px.bar(city_c.head(12), x="Count", y="City", orientation="h", color="Count", color_continuous_scale=SEQ, title="Top Cities")
        fig3.update_layout(**DARK, coloraxis_showscale=False)
        fig3.update_xaxes(tickfont_color="black", title_font_color="black")
        fig3.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig3, use_container_width=True)


    with f_tabs[1]:
        st.subheader("Courses & Groups")
        col1,col2 = st.columns(2)
        with col1:
            cat_c = courses["category"].value_counts().reset_index(); cat_c.columns=["Category","Count"]
            fig = px.pie(cat_c, values="Count", names="Category", color_discrete_sequence=COLORS, hole=0.4, title="Courses by Category")
            fig.update_layout(**DARK)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(courses, x="course_name", y="duration_weeks", color="difficulty_level",
                          color_discrete_map={"Beginner":"#48cfad","Intermediate":"#ffd32a","Advanced":"#fc5c7d"},
                          title="Duration by Difficulty", labels={"course_name":"Course","duration_weeks":"Weeks"})
            fig2.update_layout(**DARK, xaxis_tickangle=-20)
            fig2.update_xaxes(tickfont_color="black", title_font_color="black")
            fig2.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig2, use_container_width=True)
        tsz = students.groupby(["group_id","course_name"]).size().reset_index(name="students")
        fig3 = px.bar(tsz.sort_values("students"), x="students", y="group_id", orientation="h",
                      color="course_name", color_discrete_sequence=COLORS, title="True Enrollment per Group",
                      labels={"students":"Students","group_id":"Group"})
        fig3.update_layout(**DARK)
        fig3.update_xaxes(tickfont_color="black", title_font_color="black")
        fig3.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig3, use_container_width=True)

    with f_tabs[2]:
        st.subheader("Grades")
        c1,c2,c3 = st.columns(3)
        with c1: blue_metric("Total", f"{len(grades):,}", sub="Grade records", icon="📝")
        with c2: blue_metric("Avg Score", f"{grades['score'].mean():.1f}", sub="Out of 100", icon="📈")
        with c3: blue_metric("Pass Rate ≥60", f"{(grades['score']>=60).mean()*100:.1f}%", sub="Passing assessments", icon="✅")
        with col1:
            fig = px.histogram(grades, x="score", color="type", nbins=40, barmode="overlay", opacity=0.75,
                               color_discrete_map={"quiz":"#6c63ff","assignment":"#48cfad","practical":"#ffd32a","exam":"#fc5c7d"},
                               title="Grade Distribution by Type")
            fig.update_layout(**DARK)
            fig.update_xaxes(tickfont_color="black", title_font_color="black")
            fig.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            gf = grades.merge(courses[["course_id","course_name"]], on="course_id", how="left")
            course_grade = gf.groupby("course_name")["score"].agg(["mean","std"]).reset_index().sort_values("mean")
            fig2 = px.bar(course_grade, x="mean", y="course_name", orientation="h", color="mean",
                          color_continuous_scale=["#fc5c7d","#ffd32a","#48cfad"], title="Average Grade by Course",
                          labels={"mean":"Avg Score","course_name":"Course"})
            fig2.update_layout(**DARK, coloraxis_showscale=False, xaxis_range=[0,100])
            fig2.update_xaxes(tickfont_color="black", title_font_color="black")
            fig2.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig2, use_container_width=True)
            
        gf = grades.merge(courses[["course_id","course_name"]], on="course_id", how="left")
        gf["month"] = gf["date"].dt.to_period("M").astype(str)
        mg = gf.groupby(["month","course_name"])["score"].mean().reset_index()
        fig2 = px.line(mg, x="month", y="score", color="course_name", color_discrete_sequence=COLORS, markers=True,
                       title="Avg Score Over Time by Course")
        fig2.update_layout(**DARK)
        fig2.update_xaxes(tickfont_color="black", title_font_color="black")
        fig2.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig2, use_container_width=True)

    with f_tabs[3]:
        st.subheader("Concept Mastery")
        c1,c2,c3 = st.columns(3)
        with c1: blue_metric("Total Records", f"{len(concepts):,}", sub="Concept attempts", icon="🎯")
        with c2: blue_metric("Pass Rate", f"{(concepts['mastery_status']=='passed').mean()*100:.1f}%", sub="Mastery achieved", icon="✅")
        with c3: blue_metric("Concepts", str(concepts['concept_name'].nunique()), sub="Unique concepts", icon="🧠")
        fail_c = concepts.groupby("concept_name")["mastery_status"].apply(lambda x: (x=="failed").mean()*100).reset_index(name="fail_rate")
        fig = px.bar(fail_c.sort_values("fail_rate").tail(15), x="fail_rate", y="concept_name", orientation="h",
                     color="fail_rate", color_continuous_scale=["#48cfad","#ffd32a","#fc5c7d"],
                     title="Top 15 Concepts by Failure Rate", labels={"fail_rate":"Fail %","concept_name":""})
        fig.update_layout(**DARK, coloraxis_showscale=False)
        fig.update_xaxes(tickfont_color="black", title_font_color="black")
        fig.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig, use_container_width=True)
        crs_pass = concepts.groupby("course_id")["mastery_status"].apply(lambda x: (x=="passed").mean()*100).reset_index(name="pass_rate")
        crs_pass = crs_pass.merge(courses[["course_id","course_name"]], on="course_id")
        fig2 = px.bar(crs_pass.sort_values("pass_rate"), x="pass_rate", y="course_name", orientation="h",
                      color="pass_rate", color_continuous_scale=["#fc5c7d","#ffd32a","#48cfad"],
                      title="Concept Pass Rate by Course", labels={"pass_rate":"Pass %","course_name":""})
        fig2.update_layout(**DARK, coloraxis_showscale=False, xaxis_range=[0,110])
        fig2.update_xaxes(tickfont_color="black", title_font_color="black")
        fig2.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig2, use_container_width=True)

    with f_tabs[4]:
        st.subheader("Attendance")
        att_p = (attendance["status"]=="attended").mean()*100
        c1,c2,c3 = st.columns(3)
        with c1: blue_metric("Platform Rate", f"{att_p:.1f}%", sub="Overall attendance", icon="📅")
        with c2: blue_metric("Total Sessions", f"{len(attendance):,}", sub="Session records", icon="📋")
        with c3: blue_metric("Session Types", str(attendance['session_type'].nunique()), sub="Different types", icon="📂")
        col1,col2 = st.columns(2)
        with col1:
            type_att = attendance.groupby("session_type").apply(lambda x: (x["status"]=="attended").mean()*100).reset_index(name="rate")
            fig = px.bar(type_att, x="session_type", y="rate", color="rate", color_discrete_sequence=["#4CAF50"],
                         text=type_att["rate"].round(1), title="Attendance by Session Type")
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(**DARK, coloraxis_showscale=False, yaxis_range=[0,110])
            fig.update_xaxes(tickfont_color="black", title_font_color="black")
            fig.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.pie(attendance, names="status", hole=0.45,
                          color_discrete_map={"attended":"#48cfad","absent":"#fc5c7d"}, title="Attended vs Absent")
            fig2.update_layout(**DARK)
            st.plotly_chart(fig2, use_container_width=True)

        with st.container():
            att_p = (attendance["status"]=="attended").mean()*100
            group_att_tab = attendance.groupby("group_id").apply(lambda x: (x["status"]=="attended").mean()*100).reset_index(name="att_rate")
            group_att_tab = group_att_tab.merge(groups[["group_id","group_name","course_id"]], on="group_id", how="left").merge(courses[["course_id","course_name"]], on="course_id", how="left")
            fig = px.bar(group_att_tab.sort_values("att_rate"), x="att_rate", y="group_name", orientation="h", color="att_rate",
                     color_continuous_scale=["#fc5c7d","#ffd32a","#48cfad"], title="Attendance Rate by Group",
                     labels={"att_rate":"Attendance %","group_name":"Group"})
            fig.update_layout(**DARK, coloraxis_showscale=False, xaxis_range=[0,100])
            fig.update_xaxes(tickfont_color="black", title_font_color="black")
            fig.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig, use_container_width=True)

    with f_tabs[5]:
        st.subheader("Engagement")
        c1,c2,c3,c4 = st.columns(4)
        with c1: blue_metric("Total Events", f"{len(engagement):,}", sub="Platform events", icon="⚡")
        with c2: blue_metric("Unique Students", str(engagement['student_id'].nunique()), sub="Active users", icon="🎓")
        with c3: blue_metric("Avg Events/Student", f"{len(engagement)/engagement['student_id'].nunique():.0f}", sub="Per student", icon="📊")
        with c4: blue_metric("Web %", f"{(engagement['device']=='web').mean()*100:.0f}%", sub="Device share", icon="💻")
        col1,col2 = st.columns(2)
        with col1:
            ec = engagement.groupby("event_type").size().reset_index(name="count")
            fig = px.pie(ec, values="count", names="event_type", color_discrete_sequence=COLORS, hole=0.45, title="Events by Type")
            fig.update_layout(**DARK)
            fig.update_xaxes(tickfont_color="black", title_font_color="black")
            fig.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            dt = engagement.groupby(["device","event_type"]).size().reset_index(name="count")
            fig2 = px.bar(dt, x="device", y="count", color="event_type", color_discrete_sequence=COLORS, barmode="stack", title="Events by Device")
            fig2.update_layout(**DARK)
            fig2.update_xaxes(tickfont_color="black", title_font_color="black")
            fig2.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig2, use_container_width=True)
        em2 = engagement.copy(); em2["month"] = em2["event_datetime"].dt.to_period("M").astype(str)
        em3 = em2.groupby(["month","event_type"]).size().reset_index(name="count")
        fig3 = px.line(em3, x="month", y="count", color="event_type", color_discrete_sequence=COLORS, markers=True, title="Monthly Engagement by Type")
        fig3.update_layout(**DARK)
        fig3.update_xaxes(tickfont_color="black", title_font_color="black")
        fig3.update_yaxes(tickfont_color="black", title_font_color="black")
        st.plotly_chart(fig3, use_container_width=True)

    with f_tabs[6]:
        st.subheader("Assignment Submissions")
        c1,c2,c3,c4 = st.columns(4)
        with c1: blue_metric("Total", f"{len(submissions):,}", sub="Submissions", icon="📤")
        with c2: blue_metric("Late Rate", f"{submissions['is_late'].mean()*100:.1f}%", sub="Of all submissions", icon="⏰")
        with c3: blue_metric("Avg Time", f"{submissions['time_spent_minutes'].mean():.0f} min", sub="Per submission", icon="⏱️")
        with c4: blue_metric("Avg Attempts", f"{submissions['attempts'].mean():.1f}", sub="Per student", icon="🔄")
        col1,col2 = st.columns(2)
        with col1:
            fig = px.pie(submissions, names="is_late", hole=0.45,
                         color_discrete_map={True:"#fc5c7d",False:"#48cfad"}, title="Late vs On-Time")
            fig.update_layout(**DARK)
            fig.update_xaxes(tickfont_color="black", title_font_color="black")
            fig.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.histogram(submissions, x="time_spent_minutes", nbins=30, color_discrete_sequence=["#6c63ff"],
                                title="Time Spent Distribution")
            fig2.update_layout(**DARK)
            fig2.update_xaxes(tickfont_color="black", title_font_color="black")
            fig2.update_yaxes(tickfont_color="black", title_font_color="black")
            st.plotly_chart(fig2, use_container_width=True)
        sm = submissions.copy(); sm["month"] = sm["submitted_at"].dt.to_period("M").astype(str)
        smg = sm.groupby("month").agg(total=("submission_id","count"), late=("is_late","sum")).reset_index()
        smg["late_pct"] = smg["late"]/smg["total"]*100
        fig3 = make_subplots(specs=[[{"secondary_y":True}]])
        fig3.add_trace(go.Bar(x=smg["month"], y=smg["total"], name="Total", marker_color="#6c63ff", opacity=0.7), secondary_y=False)
        fig3.add_trace(go.Scatter(x=smg["month"], y=smg["late_pct"], name="Late %", line=dict(color="#fc5c7d",width=2.5), mode="lines+markers"), secondary_y=True)
        fig3.update_layout(**DARK, title="Monthly Submissions & Late Rate")
        fig3.update_yaxes(title_text="Submissions", secondary_y=False)
        fig3.update_yaxes(title_text="Late %", secondary_y=True)
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════ TAB 3 — SUGGESTIONS ════════════════════════════════════════
with t_suggestions:
    st.title("Suggestions & Decisions")
    fail_s = concepts.groupby("concept_name")["mastery_status"].apply(lambda x: (x=="failed").mean()*100).reset_index(name="fail_rate")
    worst_cs = fail_s.sort_values("fail_rate",ascending=False).iloc[0]["concept_name"]
    grp_att_s = attendance.groupby("group_id").apply(lambda x: (x["status"]=="attended").mean()*100).reset_index(name="att_rate")
    grp_att_s = grp_att_s.merge(groups[["group_id","group_name","course_id"]], on="group_id", how="left")
    grp_att_s = grp_att_s.merge(courses[["course_id","course_name"]], on="course_id", how="left")
    pav_s = grp_att_s["att_rate"].mean()
    low_att_s = grp_att_s[grp_att_s["att_rate"]<pav_s].sort_values("att_rate")
    late_rate_s = submissions["is_late"].mean()*100
    true_sz_s = students.groupby("group_id").size().reset_index(name="true_count")
    grp_sz_s = groups[["group_id","group_name"]].merge(true_sz_s, on="group_id", how="left").fillna(0)
    unviable_s = grp_sz_s.sort_values("true_count").iloc[0]
    avg_grade_s = grades.groupby("student_id")["score"].mean().reset_index(name="avg_grade")
    att_s = attendance.groupby("student_id").apply(lambda x: (x["status"]=="attended").mean()*100).reset_index(name="att_rate")
    risk_s = students[["student_id","group_name","course_name"]].merge(avg_grade_s, on="student_id", how="left").merge(att_s, on="student_id", how="left")
    risk_s["risk_flag"] = (risk_s["avg_grade"].fillna(0) < 55) | (risk_s["att_rate"].fillna(0) < 60)
    risk_count_s = int(risk_s["risk_flag"].sum())
    lowest_att_group_s = low_att_s.iloc[0] if len(low_att_s) else grp_att_s.sort_values("att_rate").iloc[0]

    decisions = pd.DataFrame({
        "Decision": [
            f"Redesign {worst_cs} concept module",
            f"Merge or dissolve {unviable_s['group_name']}",
            f"Launch attendance intervention for {lowest_att_group_s['group_name']}",
            "Start at-risk student outreach",
            "Deploy late-submission reminders"
        ],
        "Insight Trigger": [
            f"Highest concept failure rate: {fail_s.sort_values('fail_rate', ascending=False).iloc[0]['fail_rate']:.1f}%",
            f"True group size is {int(unviable_s['true_count'])}",
            f"Lowest attendance group at {lowest_att_group_s['att_rate']:.1f}%",
            f"{risk_count_s} students below attendance or grade thresholds",
            f"Late submission rate is {late_rate_s:.1f}%"
        ],
        "Owner": ["Curriculum lead","Operations lead","Instructor manager","Student success","Academic operations"],
        "Timeframe": ["2 weeks","This week","This week","48 hours","1 week"],
        "Priority": ["Critical","Critical","High","Critical","High"]
    })

    st.subheader("Decision Queue from Insights")
    st.dataframe(decisions, use_container_width=True, hide_index=True)

    sug_tabs = st.tabs(["🏛️ Curriculum","👥 Groups","📅 Attendance","📝 Assessments","🚨 At-Risk","📊 Priority Decisions"])

    with sug_tabs[0]:
        st.subheader("🏛️ Curriculum Redesign")
        st.markdown(f"""
**1. Redesign the `{worst_cs}` module — CRITICAL**
- Failure rate: **{fail_s.iloc[0]['fail_rate']:.1f}%** — highest on the platform
- Break into smaller scaffolded micro-lessons with worked examples and prerequisite checks
- Add a recorded walkthrough session students can rewatch on demand

**2. Address the Digital Marketing concept cluster**
- `Funnel Analytics`, `SEO Basics`, `Content Strategy`, `Paid Ads` all exceed 43% failure
- Review concept ordering — Funnel Analytics likely depends on Content Strategy being taught first

**3. Add spaced-repetition checkpoints for advanced concepts**
- ML topics appear late with no prior review — insert low-stakes review quizzes at weeks 4 and 8

**4. Introduce a concept mastery pre-screen**
- Students entering Advanced courses without prerequisite mastery are more likely to fail
- Add a short diagnostic at course start; redirect students below threshold to the Intermediate track
        """)

    with sug_tabs[1]:
        st.subheader("👥 Groups & Enrollment")
        st.markdown(f"""
**1. Dissolve or merge `{unviable_s['group_name']}`**
- True enrollment: **{int(unviable_s['true_count'])} students** — below any viable cohort minimum
- Merge into the group with the most similar concept performance profile (see Q13)

**2. Audit self-reported group sizes**
- groups.csv overcounts enrollment across the board
- Replace `stated_num_students` with a live COUNT from students.csv in all reporting systems

**3. Rebalance oversized groups**
- Cap new enrollment at 45 per group; open new sections when demand exceeds this

**4. Assign teaching assistants to large groups**
- Groups with 50+ students and a single instructor show worse outcomes
        """)

    with sug_tabs[2]:
        st.subheader("📅 Attendance Interventions")
        st.markdown(f"**{len(low_att_s)} groups** fall below the platform average of {pav_s:.1f}%:")
        for _,row in low_att_s.iterrows():
            st.markdown(f"- **{row['group_name']}** ({row['course_name']}) — {row['att_rate']:.1f}%")
        st.markdown(f"""
**Recommended actions:**
1. Schedule group-level check-ins within 2 weeks — survey for scheduling conflicts or content difficulty
2. Build a weekly automated report flagging students below 60% cumulative attendance
3. Plan make-up or asynchronous catch-up content for the mid-term dip window identified in Q9
4. Introduce an attendance streak badge in the platform UI to incentivise consistency
        """)

    with sug_tabs[3]:
        st.subheader("📝 Assessment & Submission Policy")
        st.markdown(f"""
**1. Address the late submission problem — current rate: {late_rate_s:.1f}%**
- Introduce a 3-day automated reminder email sequence
- Consider a 24h soft grace period to reduce stress without eliminating the submission habit

**2. Recalibrate quiz difficulty**
- Quizzes show the highest score volatility and a left-skew toward low scores
- Review and rebalance question banks; add 2–3 easy scaffolding questions to each quiz

**3. Add partial credit for practicals**
- Some groups show all-or-nothing scoring — rubric-based partial credit better differentiates ability

**4. Use first assignment score as an early-warning predictor**
- Flag students scoring below 55 on their first assignment for early tutoring intervention
        """)

    with sug_tabs[4]:
        st.subheader("🚨 At-Risk Student Support")
        st.markdown(f"""
**1. Prioritise the Top 10 At-Risk students (Q14)**
- These students are failing on all four risk dimensions simultaneously
- Assign each a dedicated academic mentor for bi-weekly check-ins over the next month

**2. 3-Tier support system based on risk score:**
- 🟡 **Tier 1 (40–60):** Automated nudge emails + self-paced resources
- 🟠 **Tier 2 (60–80):** Weekly instructor message + concept-specific remedial content
- 🔴 **Tier 3 (80+):** One-on-one session + parent/guardian notification (if applicable)

**3. For the "Engaged but Struggling" segment**
- These students log in but fail to convert activity into results
- Offer structured study groups, active-recall practice sets, and test-taking strategy workshops

**4. Run a mandatory mid-term progress review**
- All Tier 2 and 3 students should have a 1:1 review before assessment week
        """)

    with sug_tabs[5]:
        st.markdown("""
| Priority | Action | Expected Impact |
|---|---|---|
| 🔴 1 | Redesign Recursion module | Reduce highest failure rate |
| 🔴 2 | Dissolve/merge unviable group | Fix structural viability |
| 🔴 3 | Contact Top 10 at-risk students | Prevent imminent dropout |
| 🟠 4 | Fix low-attendance groups | Raise cohort performance |
| 🟠 5 | Late submission reminder emails | Reduce late rate and grade gap |
""")
