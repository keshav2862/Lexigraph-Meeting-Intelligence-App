"""Lexigraph - Meeting Intelligence Knowledge Graph

A premium 3-agent AI system that transforms meeting transcripts 
into queryable knowledge graphs with conversational interface.

Features:
- Entity extraction from transcripts
- Neo4j knowledge graph
- Conversational Q&A
- Interactive graph visualization
- Deadline tracking
- Cross-meeting analysis
- Auto-generated summaries
- Conflict detection
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

from src.agents.extractor import ExtractorAgent
from src.agents.graph_builder import GraphBuilderAgent
from src.agents.query_agent import QueryAgent
from src.agents.analyzer import AnalyzerAgent
from src.agents.summary_agent import SummaryAgent
from src.config import config
from src.utils.error_handling import get_user_friendly_error
from src.utils.export import (
    export_meeting_summary,
    export_action_items,
    export_insights_report,
    export_to_markdown
)

# Page configuration - collapsed sidebar for cleaner demo
st.set_page_config(
    page_title="Lexigraph",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium CSS
# Premium Glassmorphism CSS
st.markdown("""
<style>
    /* ============================================
       PREMIUM GLASSMORPHISM THEME
       Inspired by Apple, Stripe, Linear
       ============================================ */
    
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * { 
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Animated mesh gradient background */
    .stApp {
        background: #0a0a0f;
        background-image: 
            radial-gradient(at 40% 20%, hsla(280, 100%, 35%, 0.3) 0px, transparent 50%),
            radial-gradient(at 80% 0%, hsla(189, 100%, 40%, 0.2) 0px, transparent 50%),
            radial-gradient(at 0% 50%, hsla(355, 100%, 45%, 0.15) 0px, transparent 50%),
            radial-gradient(at 80% 50%, hsla(240, 100%, 50%, 0.2) 0px, transparent 50%),
            radial-gradient(at 0% 100%, hsla(210, 100%, 40%, 0.2) 0px, transparent 50%),
            radial-gradient(at 80% 100%, hsla(280, 100%, 40%, 0.2) 0px, transparent 50%);
        min-height: 100vh;
    }
    
    /* Animated floating orbs */
    .stApp::before {
        content: '';
        position: fixed;
        top: 10%;
        right: 15%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 70%);
        border-radius: 50%;
        animation: float 15s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    .stApp::after {
        content: '';
        position: fixed;
        bottom: 20%;
        left: 10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.12) 0%, transparent 70%);
        border-radius: 50%;
        animation: float 20s ease-in-out infinite reverse;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(0, 0) scale(1); }
        25% { transform: translate(30px, -30px) scale(1.05); }
        50% { transform: translate(-20px, 20px) scale(0.95); }
        75% { transform: translate(20px, 10px) scale(1.02); }
    }
    
    /* System overrides */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: auto !important;
    }
    
    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1300px;
        position: relative;
        z-index: 1;
    }
    
    #MainMenu, footer, header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Hide sidebar completely */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* ============================================
       HERO HEADER - CLEAN & PROFESSIONAL
       ============================================ */
    .hero-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
        position: relative;
    }
    
    .logo-text {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 35%, #a855f7 50%, #ec4899 75%, #f97316 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-shift 6s ease infinite;
        filter: drop-shadow(0 0 40px rgba(139, 92, 246, 0.4));
        margin-bottom: 0.25rem;
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        font-weight: 500;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        letter-spacing: -0.01em;
    }
    
    .tagline {
        color: rgba(148, 163, 184, 0.7);
        font-size: 0.85rem;
        font-weight: 400;
        margin-top: 0.5rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    
    /* ============================================
       GLASSMORPHISM CARDS
       ============================================ */
    .glass-card, .feature-card, .insight-card, .tour-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before, .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    }
    
    .feature-card:hover, .insight-card:hover {
        transform: translateY(-4px);
        border-color: rgba(139, 92, 246, 0.3);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.3),
            0 0 60px rgba(139, 92, 246, 0.1);
    }
    
    .feature-card h4 {
        color: #c4b5fd !important;
        font-weight: 600;
        margin-bottom: 0.75rem;
        font-size: 1.1rem;
    }
    
    .feature-card p, .feature-card li {
        color: rgba(203, 213, 225, 0.85) !important;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    
    /* Status cards */
    .deadline-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        padding: 1rem 1.25rem;
        border-radius: 16px;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .deadline-overdue {
        background: rgba(239, 68, 68, 0.1);
        border-color: rgba(239, 68, 68, 0.3);
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.1);
    }
    
    .deadline-soon {
        background: rgba(251, 191, 36, 0.1);
        border-color: rgba(251, 191, 36, 0.3);
        box-shadow: 0 0 30px rgba(251, 191, 36, 0.1);
    }
    
    .deadline-upcoming {
        background: rgba(34, 197, 94, 0.1);
        border-color: rgba(34, 197, 94, 0.3);
        box-shadow: 0 0 30px rgba(34, 197, 94, 0.1);
    }
    
    .conflict-card {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 16px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    }
    
    .tour-card {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%);
        border: 1px solid rgba(6, 182, 212, 0.2);
    }
    
    /* ============================================
       CHAT BUBBLES - iMessage Premium Style
       ============================================ */
    .user-bubble {
        background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 50%, #a78bfa 100%);
        color: white;
        padding: 1.1rem 1.6rem;
        border-radius: 24px 24px 8px 24px;
        margin: 1rem 0;
        margin-left: 25%;
        box-shadow: 
            0 10px 40px rgba(124, 58, 237, 0.35),
            0 0 0 1px rgba(255, 255, 255, 0.1) inset;
        position: relative;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    

    
    .assistant-bubble {
        background: linear-gradient(135deg, rgba(30, 30, 50, 0.9) 0%, rgba(40, 40, 60, 0.8) 100%);
        backdrop-filter: blur(30px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #f1f5f9;
        padding: 1.1rem 1.6rem;
        border-radius: 24px 24px 24px 8px;
        margin: 1rem 0;
        margin-right: 25%;
        position: relative;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    

    
    .assistant-bubble::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.3), transparent);
        border-radius: 24px 24px 0 0;
    }
    
    /* ============================================
       BUTTONS - Ultra Premium
       ============================================ */
    .stButton > button {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.9) 0%, rgba(168, 85, 247, 0.9) 100%) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 50px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.7rem 1.8rem !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 
            0 4px 20px rgba(139, 92, 246, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        position: relative;
        overflow: hidden;
        text-transform: none;
        letter-spacing: 0.01em;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
        transition: left 0.6s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 
            0 12px 35px rgba(139, 92, 246, 0.5),
            0 0 50px rgba(139, 92, 246, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98) !important;
    }
    
    /* ============================================
       NAVIGATION - Custom Navbar (not Streamlit tabs)
       ============================================ */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-radius: 0;
        padding: 0;
        gap: 0;
        border: none;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        justify-content: center;
        flex-wrap: wrap;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: rgba(148, 163, 184, 0.7);
        border-radius: 0;
        font-size: 0.9rem;
        font-weight: 500;
        padding: 1rem 1.5rem !important;
        min-width: auto;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        border: none !important;
        margin: 0;
    }
    
    .stTabs [data-baseweb="tab"]::after {
        content: '';
        position: absolute;
        bottom: -1px;
        left: 50%;
        width: 0;
        height: 2px;
        background: linear-gradient(90deg, #06b6d4, #8b5cf6, #ec4899);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        transform: translateX(-50%);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
        background: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover::after {
        width: 100%;
    }
    
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #f1f5f9 !important;
        font-weight: 600;
        box-shadow: none;
    }
    
    .stTabs [aria-selected="true"]::after {
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #06b6d4, #8b5cf6, #ec4899);
    }
    
    /* Hide the default tab highlight */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    
    /* ============================================
       INPUTS - Sleek Dark Style
       ============================================ */
    .stTextArea textarea, .stTextInput input {
        background: rgba(15, 15, 25, 0.8) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: rgba(139, 92, 246, 0.5) !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.15) !important;
    }
    
    /* Chat input - Ultra Modern */
    .stChatInput {
        position: relative;
    }
    
    .stChatInput > div {
        background: linear-gradient(135deg, rgba(20, 20, 35, 0.95) 0%, rgba(30, 30, 50, 0.9) 100%) !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
        border-radius: 50px !important;
        padding: 0.25rem !important;
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.3),
            0 0 40px rgba(139, 92, 246, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    .stChatInput > div:focus-within {
        border-color: rgba(139, 92, 246, 0.5) !important;
        box-shadow: 
            0 8px 30px rgba(0, 0, 0, 0.4),
            0 0 60px rgba(139, 92, 246, 0.15) !important;
    }
    
    .stChatInput textarea {
        background: transparent !important;
        border: none !important;
        color: #f1f5f9 !important;
        font-size: 0.95rem !important;
    }
    
    .stChatInput button {
        background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%) !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        min-width: 40px !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(15, 15, 25, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }
    
    .stSelectbox [data-baseweb="select"],
    .stSelectbox [data-baseweb="select"] > div {
        background: transparent !important;
        color: #e2e8f0 !important;
    }
    
    [data-baseweb="popover"], [data-baseweb="menu"] {
        background: rgba(15, 15, 25, 0.95) !important;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    
    [data-baseweb="menu"] li {
        background: transparent !important;
        color: #e2e8f0 !important;
        transition: all 0.15s ease;
    }
    
    [data-baseweb="menu"] li:hover {
        background: rgba(139, 92, 246, 0.2) !important;
    }
    
    /* ============================================
       SIDEBAR - Refined Glass
       ============================================ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 10, 20, 0.95) 0%, rgba(15, 15, 30, 0.98) 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    /* ============================================
       TYPOGRAPHY
       ============================================ */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    .stMarkdown, p, span, label {
        color: rgba(226, 232, 240, 0.9) !important;
    }
    
    .stCheckbox label span {
        color: #cbd5e1 !important;
    }
    
    /* ============================================
       METRICS - Glowing Numbers
       ============================================ */
    [data-testid="stMetricValue"] {
        color: #c4b5fd !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
        text-shadow: 0 0 30px rgba(139, 92, 246, 0.3);
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(148, 163, 184, 0.8) !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* ============================================
       SCROLLBAR - Minimal & Elegant
       ============================================ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(139, 92, 246, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(139, 92, 246, 0.5);
    }
    
    /* ============================================
       EXPANDER - Glass Style
       ============================================ */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }
    
    /* ============================================
       GITHUB LINK - Subtle & Professional
       ============================================ */
    .github-link {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: rgba(226, 232, 240, 0.9) !important;
        text-decoration: none !important;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.25s ease;
    }
    
    .github-link:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(139, 92, 246, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* ============================================
       DIVIDERS
       ============================================ */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 1.5rem 0;
    }
    
    /* ============================================
       SLIDER
       ============================================ */
    .stSlider > div > div {
        background: rgba(139, 92, 246, 0.3) !important;
    }
    
    .stSlider > div > div > div {
        background: #8b5cf6 !important;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.5);
    }
    
    /* ============================================
       DOWNLOAD BUTTON
       ============================================ */
    .stDownloadButton > button {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #e2e8f0 !important;
    }
    
    .stDownloadButton > button:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(139, 92, 246, 0.4) !important;
    }

</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state"""
    defaults = {
        "extractor": None,
        "graph_builder": None,
        "query_agent": None,
        "analyzer": None,
        "summary_agent": None,
        "last_extraction": None,
        "connected": False,
        "messages": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_sample_transcripts():
    """Load sample transcripts"""
    samples = {}
    sample_dir = Path("data/sample_transcripts")
    if sample_dir.exists():
        for file in sample_dir.glob("*.txt"):
            samples[file.stem] = file.read_text()
    return samples


def render_header():
    """Render header with GitHub link"""
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col2:
        st.markdown("""
            <div class="hero-header">
                <div class="logo-text">⚡ Lexigraph</div>
                <div class="hero-subtitle">Find that decision from 6 sprints ago. In seconds.</div>
                <div class="tagline">Built with Neo4j • Groq LLM • Streamlit</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <a href="https://github.com/keshav2862/Lexigraph-Meeting-Intelligence-App" target="_blank" class="github-link">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
                </svg>
                View Source
            </a>
        """, unsafe_allow_html=True)



def render_sidebar():
    """Render sidebar - Demo mode"""
    with st.sidebar:
        st.header("⚡ Lexigraph Demo")
        
        if st.session_state.connected:
            st.success("✓ Connected to Neo4j")
            
            # Show if embeddings are ready
            if st.session_state.get("graph_embeddings"):
                st.success("✓ Embeddings ready")
            else:
                st.info("🔄 Embeddings loading...")
        else:
            st.warning("⚠ Not connected")
            
        if st.button("🔌 Connect to Neo4j", use_container_width=True):
            try:
                with st.spinner("Connecting..."):
                    st.session_state.extractor = ExtractorAgent()
                    st.session_state.graph_builder = GraphBuilderAgent()
                    st.session_state.graph_builder.connect()
                    st.session_state.query_agent = QueryAgent()
                    st.session_state.query_agent.connect()
                    st.session_state.analyzer = AnalyzerAgent()
                    st.session_state.analyzer.connect()
                    st.session_state.summary_agent = SummaryAgent()
                    st.session_state.summary_agent.connect()
                    st.session_state.connected = True
                    
                    # Auto-generate embeddings for demo
                    try:
                        from src.ml.embeddings import GraphEmbeddings, HAS_GRAPH_ML
                        if HAS_GRAPH_ML:
                            embeddings = GraphEmbeddings(st.session_state.graph_builder.client)
                            embeddings.generate_embeddings(dimensions=64, walk_length=30, num_walks=100)
                            st.session_state.graph_embeddings = embeddings
                    except Exception as emb_err:
                        st.warning(f"Embeddings not generated: {str(emb_err)[:30]}")
                    
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {str(e)}")
        
        st.divider()
        
        st.header("📊 Models")
        st.caption(f"Extraction: `{config.EXTRACTION_MODEL}`")
        st.caption(f"Query: `{config.QUERY_MODEL}`")
        
        if st.session_state.connected and st.session_state.graph_builder:
            st.divider()
            st.header("📈 Graph Stats")
            try:
                stats = st.session_state.graph_builder.get_graph_stats()
                for label, count in stats.items():
                    st.metric(label, count)
            except:
                pass
            
            st.divider()
            
            # Just clear chat for demo
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.query_agent:
                    st.session_state.query_agent.clear_history()
                st.rerun()


def render_extraction_tab():
    """Demo mode: Read-only transcript viewer"""
    
    # What Lexigraph Extracts - shown only on Sample Data tab
    st.markdown("""
    <style>
        .entity-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 0.75rem;
            margin-bottom: 1rem;
        }
        .entity-card {
            border-radius: 14px;
            padding: 1rem;
            text-align: center;
            min-width: 0;
        }
        .entity-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .entity-desc {
            color: rgba(148, 163, 184, 0.85);
            font-size: 0.75rem;
        }
        @media (max-width: 600px) {
            .entity-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .entity-card {
                padding: 0.75rem;
            }
            .entity-title {
                font-size: 0.95rem;
            }
            .entity-desc {
                font-size: 0.7rem;
            }
        }
    </style>
    
    <p style="color: rgba(148, 163, 184, 0.9); font-size: 1rem; margin: 0.5rem 0 1rem; text-align: center;">
        From each meeting transcript, the AI automatically identifies and structures key information:
    </p>
    
    <div class="entity-grid">
        <div class="entity-card" style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.05) 100%); border: 1px solid rgba(59, 130, 246, 0.3);">
            <div class="entity-title" style="color: #60a5fa;">People</div>
            <div class="entity-desc">Attendees with roles</div>
        </div>
        <div class="entity-card" style="background: linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(168, 85, 247, 0.05) 100%); border: 1px solid rgba(168, 85, 247, 0.3);">
            <div class="entity-title" style="color: #a855f7;">Topics</div>
            <div class="entity-desc">Key discussion points</div>
        </div>
        <div class="entity-card" style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.05) 100%); border: 1px solid rgba(34, 197, 94, 0.3);">
            <div class="entity-title" style="color: #22c55e;">Decisions</div>
            <div class="entity-desc">Final choices made</div>
        </div>
        <div class="entity-card" style="background: linear-gradient(135deg, rgba(249, 115, 22, 0.15) 0%, rgba(249, 115, 22, 0.05) 100%); border: 1px solid rgba(249, 115, 22, 0.3);">
            <div class="entity-title" style="color: #f97316;">Action Items</div>
            <div class="entity-desc">Tasks with owners</div>
        </div>
        <div class="entity-card" style="background: linear-gradient(135deg, rgba(236, 72, 153, 0.15) 0%, rgba(236, 72, 153, 0.05) 100%); border: 1px solid rgba(236, 72, 153, 0.3);">
            <div class="entity-title" style="color: #ec4899;">Commitments</div>
            <div class="entity-desc">Promises people made</div>
        </div>
    </div>
    
    <p style="color: rgba(148, 163, 184, 0.7); font-size: 0.85rem; text-align: center; font-style: italic; margin-bottom: 1.5rem;">
        These entities are connected in a knowledge graph, enabling cross-meeting insights.
    </p>
    """, unsafe_allow_html=True)
    
    st.header("Sample Meeting Data")
    st.caption("Browse the 10 sample meeting transcripts that power this demo")
    
    samples = load_sample_transcripts()
    
    if not samples:
        st.warning("No sample transcripts found")
        return
    
    # Show graph stats and Load Demo Data button
    if st.session_state.connected:
        try:
            stats = st.session_state.graph_builder.get_graph_stats()
            total_nodes = sum(stats.values())
            
            # If graph is empty, show Load Demo Data button prominently
            if total_nodes == 0:
                st.markdown("""
                <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%); 
                     border: 2px solid rgba(139, 92, 246, 0.4); border-radius: 16px; padding: 1.5rem; margin: 1rem 0; text-align: center;">
                    <h3 style="color: #c4b5fd; margin-bottom: 0.5rem;">Welcome to Lexigraph!</h3>
                    <p style="color: rgba(148, 163, 184, 0.9);">Your knowledge graph is empty. Load the demo data to explore all features.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Load Demo Data (10 meetings)", use_container_width=True, type="primary"):
                    progress = st.progress(0, text="Processing transcripts...")
                    
                    for i, (name, transcript) in enumerate(samples.items()):
                        progress.progress((i + 1) / len(samples), text=f"Processing: {name.replace('_', ' ').title()}")
                        try:
                            # Extract entities
                            extracted = st.session_state.extractor.extract(transcript)
                            # Set the meeting title from filename
                            extracted.meeting_title = name.replace("_", " ").title()
                            # Build graph
                            st.session_state.graph_builder.build_graph(extracted)
                        except Exception as e:
                            st.warning(f"Skipped {name}: {str(e)[:50]}")
                    
                    progress.empty()
                    st.success("Demo data loaded! Refresh to see the knowledge graph.")
                    st.rerun()
            else:
                # Show stats
                cols = st.columns(6)
                for col, (label, count) in zip(cols, stats.items()):
                    col.metric(label, count)
        except:
            pass
        st.divider()
    
    # Transcript viewer
    st.markdown("### Meeting Transcripts")
    
    sample_names = list(samples.keys())
    sample_choice = st.selectbox(
        "Select a transcript to view:", sample_names,
        format_func=lambda x: x.replace("_", " ").title()
    )
    transcript = samples.get(sample_choice, "")
    
    # Display transcript in a nice container
    with st.expander(f"{sample_choice.replace('_', ' ').title()}", expanded=True):
        st.text_area(
            "Transcript content:", value=transcript, height=350,
            disabled=True, label_visibility="collapsed"
        )


def render_chat_tab():
    """Chat tab with preset queries for demo"""
    st.header("Chat with Your Meetings")
    st.caption("Ask natural language questions about 10 meetings worth of data")
    
    if not st.session_state.connected:
        st.warning("Connect to Neo4j first")
        return
    
    # Preset query buttons for easy demo
    if not st.session_state.messages:
        st.markdown("### Try These Queries")
        st.markdown("*Click any button to auto-run the query:*")
        
        preset_queries = [
            "What should Mike Johnson do?",
            "What decisions were made?",
            "What are Carlos Rodriguez's tasks?",
            "Who works with Wei Chen?",
            "What topics were discussed?",
            "Summarize the Sprint Planning",
        ]
        
        cols = st.columns(3)
        for i, query in enumerate(preset_queries):
            with cols[i % 3]:
                if st.button(query, key=f"preset_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": query})
                    try:
                        result = st.session_state.query_agent.query(query)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": result["answer"], 
                            "cypher": result["cypher"]
                        })
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"Error: {str(e)[:100]}"
                        })
                    st.rerun()
        
        st.divider()
        st.markdown("### Or Ask Your Own Question")
        st.info("The AI translates your question into a Cypher graph query automatically!")
    
    # Display chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            content = msg["content"].replace('\n', '<br>').replace('•', '&#8226;')
            st.markdown(f'<div class="assistant-bubble">{content}</div>', unsafe_allow_html=True)
            if "cypher" in msg:
                with st.expander("🔍 View Cypher Query (Graph Database)"):
                    st.code(msg["cypher"], language="cypher")
                    st.caption("This is the auto-generated Neo4j query that powered the response.")
    
    if prompt := st.chat_input("Ask about meetings, people, decisions, or action items..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        try:
            result = st.session_state.query_agent.query(prompt)
            st.session_state.messages.append({
                "role": "assistant", "content": result["answer"], "cypher": result["cypher"]
            })
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
        st.rerun()



def render_visualization_tab():
    """Graph visualization tab"""
    st.header("Knowledge Graph Visualization")
    
    if not st.session_state.connected:
        st.warning("Connect to Neo4j first")
        return
    
    # Initialize filter state
    if "graph_filters" not in st.session_state:
        st.session_state.graph_filters = {
            "Meeting": True, "Person": True, "Topic": True,
            "Decision": True, "ActionItem": True, "Commitment": True
        }
    
    # Filter buttons row
    st.markdown("##### Filter by Node Type:")
    filter_cols = st.columns(6)
    
    filter_labels = {
        "Meeting": "🔵 Meeting",
        "Person": "🟢 Person", 
        "Topic": "🟡 Topic",
        "Decision": "🟣 Decision",
        "ActionItem": "🔴 Action",
        "Commitment": "🩷 Commit"
    }
    
    for i, (node_type, label) in enumerate(filter_labels.items()):
        with filter_cols[i]:
            st.session_state.graph_filters[node_type] = st.checkbox(
                label, 
                value=st.session_state.graph_filters[node_type],
                key=f"filter_{node_type}"
            )
    
    try:
        from src.visualization.graph_viz import create_knowledge_graph_filtered
        
        # Get active filters
        active_types = [t for t, v in st.session_state.graph_filters.items() if v]
        
        if not active_types:
            st.warning("Select at least one node type to display")
            return
        
        # Generate and display graph
        with st.spinner("Generating interactive graph..."):
            html_content = create_knowledge_graph_filtered(
                st.session_state.graph_builder.client,
                active_types
            )
            components.html(html_content, height=700, scrolling=True)
        
        st.caption("Drag nodes to rearrange • Hover for details • Scroll to zoom • Use navigation buttons")
        
    except ImportError:
        st.error("Please install pyvis: `pip install pyvis`")
    except Exception as e:
        st.error(f"Visualization error: {e}")


def render_deadlines_tab():
    """Deadline tracker tab"""
    st.header("Deadline Tracker")
    
    if not st.session_state.connected:
        st.warning("Connect to Neo4j first")
        return
    
    try:
        deadlines = st.session_state.analyzer.get_deadline_status()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Overdue")
            if deadlines["overdue"]:
                for item in deadlines["overdue"]:
                    st.markdown(f'''
                        <div class="deadline-card deadline-overdue">
                            <strong>{item["task"][:50]}...</strong><br>
                            <small>{item["owner"] or "Unassigned"} | {item["deadline"]}</small>
                        </div>
                    ''', unsafe_allow_html=True)
            else:
                st.success("None overdue!")
        
        with col2:
            st.markdown("### Due Soon")
            if deadlines["due_soon"]:
                for item in deadlines["due_soon"]:
                    st.markdown(f'''
                        <div class="deadline-card deadline-soon">
                            <strong>{item["task"][:50]}...</strong><br>
                            <small>{item["owner"] or "Unassigned"} | {item["deadline"]}</small>
                        </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("Nothing due soon")
        
        with col3:
            st.markdown("### Upcoming")
            if deadlines["upcoming"]:
                for item in deadlines["upcoming"][:5]:
                    st.markdown(f'''
                        <div class="deadline-card deadline-upcoming">
                            <strong>{item["task"][:50]}...</strong><br>
                            <small>{item["owner"] or "Unassigned"} | {item["deadline"] or "No deadline"}</small>
                        </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("No upcoming items")
        
        if deadlines["no_deadline"]:
            with st.expander(f"No Deadline Set ({len(deadlines['no_deadline'])} items)"):
                for item in deadlines["no_deadline"]:
                    st.write(f"• {item['task']} ({item['owner'] or 'Unassigned'})")
        
        # Export button
        st.divider()
        export_md = export_action_items(deadlines)
        st.download_button(
            label="Download Action Items Report",
            data=export_md,
            file_name="action_items_report.md",
            mime="text/markdown"
        )
    
    except Exception as e:
        error_msg = get_user_friendly_error(e)
        st.error(f"{error_msg}")


def render_insights_tab():
    """Cross-meeting insights tab"""
    st.header("Cross-Meeting Insights")
    st.caption("Analyze patterns and trends across all 10 meetings in the knowledge graph")
    
    if not st.session_state.connected:
        st.warning("Connect to Neo4j first")
        return
    
    # Feature description
    st.markdown("""
    <div class="feature-card">
        <h4>What This Feature Does</h4>
        <p>Aggregates data across all meetings to reveal patterns that aren't visible in individual sessions:</p>
        <ul>
            <li><strong>Person Engagement</strong> - Who attends most meetings, owns most tasks, and makes most decisions</li>
            <li><strong>Topic Trends</strong> - Which topics span multiple meetings and drive the most decisions</li>
        </ul>
        <p><em>Uses Cypher aggregation queries on the Neo4j knowledge graph.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["People", "Topics"])
    
    with tab1:
        st.subheader("Person Engagement")
        try:
            insights = st.session_state.analyzer.get_person_insights()
            person_container = st.container(height=350)
            with person_container:
                for person in insights:
                    st.markdown(f'''
                    <div class="insight-card">
                        <strong>{person["name"]}</strong> 
                        <span style="color: #94a3b8;">({person["role"] or "No role"})</span><br>
                        <small>
                            {person["meetings_attended"]} meetings | 
                            {person["action_items"]} actions | 
                            {person["decisions_made"]} decisions | 
                            {person["commitments"]} commitments
                        </small>
                    </div>
                ''', unsafe_allow_html=True)
        except Exception as e:
            error_msg = get_user_friendly_error(e)
            st.error(f"{error_msg}")
    
    with tab2:
        st.subheader("Topic Trends")
        try:
            topics = st.session_state.analyzer.get_topic_trends()
            topic_container = st.container(height=350)
            with topic_container:
                for topic in topics:
                    meetings = topic["meetings"] or []
                    decisions = topic["decisions"] or []
                    st.markdown(f'''
                        <div class="insight-card">
                            <strong>{topic["topic"]}</strong> 
                            <span style="color: #a855f7;">({topic["meeting_count"]} meetings)</span><br>
                            <small>{topic["description"] or "No description"}</small><br>
                            <small style="color: #94a3b8;">
                                Meetings: {", ".join(meetings[:3]) if meetings else "None"}
                            </small>
                        </div>
                    ''', unsafe_allow_html=True)
        except Exception as e:
            error_msg = get_user_friendly_error(e)
            st.error(f"{error_msg}")
    
    # Export button for insights
    st.divider()
    try:
        person_data = st.session_state.analyzer.get_person_insights()
        topic_data = st.session_state.analyzer.get_topic_trends()
        export_md = export_insights_report(person_data, topic_data)
        st.download_button(
            label="Download Insights Report",
            data=export_md,
            file_name="insights_report.md",
            mime="text/markdown"
        )
    except:
        pass


def render_summary_tab():
    """Auto-summary tab with presets"""
    st.header("Auto-Generated Summaries")
    st.caption("Generate AI-powered summaries using LLM + knowledge graph context")
    
    if not st.session_state.connected:
        st.warning("Connect to Neo4j first")
        return
    
    # Feature description
    st.markdown("""
    <div class="feature-card">
        <h4>How Summary Generation Works</h4>
        <p>Combines <strong>graph data retrieval</strong> with <strong>LLM summarization</strong>:</p>
        <ol>
            <li>Queries Neo4j for meeting attendees, topics, decisions, and action items</li>
            <li>Passes structured context to Groq LLM (llama-3.3-70b)</li>
            <li>Generates human-readable summaries with key takeaways</li>
        </ol>
        <p><em>This is a RAG (Retrieval-Augmented Generation) pattern using graph as retrieval layer.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Single Meeting Summary")
        st.markdown("*Select a meeting from the demo data:*")
        
        # Preset meeting buttons
        preset_meetings = [
            "Sprint Planning",
            "Product Roadmap", 
            "Design Review",
            "Incident Postmortem",
            "Cross Team Sync",
            "Customer Feedback Review"
        ]
        
        cols = st.columns(2)
        for i, meeting in enumerate(preset_meetings):
            with cols[i % 2]:
                if st.button(f"{meeting}", key=f"sum_{i}", use_container_width=True):
                    with st.spinner(f"Summarizing {meeting}..."):
                        try:
                            summary = st.session_state.summary_agent.generate_meeting_summary(meeting)
                            st.session_state[f"summary_{meeting}"] = summary
                        except Exception as e:
                            st.error(f"Error: {str(e)[:50]}")
        
        # Show generated summary if exists
        for meeting in preset_meetings:
            if f"summary_{meeting}" in st.session_state:
                st.markdown(f"**{meeting} Summary:**")
                st.markdown(st.session_state[f"summary_{meeting}"])
                break
    
    with col2:
        st.subheader("Cross-Meeting Overview")
        st.markdown("*Analyze patterns across all 10 meetings:*")
        
        if st.button("Generate Full Overview", use_container_width=True):
            with st.spinner("Analyzing all meetings (this may take a moment)..."):
                try:
                    summary = st.session_state.summary_agent.generate_cross_meeting_summary()
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"Error: {e}")


def render_conflicts_tab():
    """Conflict detection tab with presets"""
    st.header("Conflict Detection")
    st.caption("Detect contradictions and compare decisions across meetings")
    
    if not st.session_state.connected:
        st.warning("Connect to Neo4j first")
        return
    
    # Feature description
    st.markdown("""
    <div class="feature-card">
        <h4>How Conflict Detection Works</h4>
        <p>Uses <strong>LLM analysis</strong> on structured graph data to identify:</p>
        <ul>
            <li><strong>Decision Conflicts</strong> - When decisions in different meetings contradict each other</li>
            <li><strong>Timeline Issues</strong> - Overlapping deadlines, resource conflicts</li>
            <li><strong>Scope Creep</strong> - When commitments expand beyond original decisions</li>
        </ul>
        <p><em>Retrieves all decisions via Cypher, then passes to LLM for semantic conflict analysis.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Analyze All Decisions for Conflicts", use_container_width=True):
        with st.spinner("Analyzing decisions for conflicts..."):
            try:
                analysis = st.session_state.analyzer.detect_conflicts()
                
                if "no conflict" in analysis.lower():
                    st.success("No conflicts detected!")
                else:
                    st.markdown(f'''
                        <div class="conflict-card">
                            {analysis.replace(chr(10), "<br>")}
                        </div>
                    ''', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.divider()
    st.subheader("Compare Two Meetings")
    st.markdown("*Select meetings to compare their topics, decisions, and action items:*")
    
    # Preset comparison options
    preset_comparisons = [
        ("Sprint Planning", "Quarterly Retrospective"),
        ("Product Roadmap", "Customer Feedback Review"),
        ("Design Review", "Sprint Planning"),
    ]
    
    st.markdown("**Quick Comparisons:**")
    cols = st.columns(3)
    for i, (m1, m2) in enumerate(preset_comparisons):
        with cols[i]:
            if st.button(f"{m1[:12]}.. vs {m2[:12]}..", key=f"cmp_{i}", use_container_width=True):
                try:
                    comparison = st.session_state.analyzer.get_meeting_comparison(m1, m2)
                    st.session_state["last_comparison"] = comparison
                except Exception as e:
                    st.error(f"Error: {str(e)[:50]}")
    
    # Show comparison result
    if "last_comparison" in st.session_state:
        comparison = st.session_state["last_comparison"]
        if "error" not in comparison:
            st.markdown(f"**{comparison.get('meeting1', 'Meeting 1')}** vs **{comparison.get('meeting2', 'Meeting 2')}**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Common Topics**")
                for t in comparison.get("common_topics", []):
                    st.write(f"• {t}")
            with col2:
                st.markdown("**Unique to First**")
                for t in comparison.get("unique_to_meeting1", []):
                    st.write(f"• {t}")
            with col3:
                st.markdown("**Unique to Second**")
                for t in comparison.get("unique_to_meeting2", []):
                    st.write(f"• {t}")
    
    st.divider()
    st.markdown("**Or enter custom meeting names:**")
    col1, col2 = st.columns(2)
    with col1:
        meeting1 = st.text_input("Meeting 1:", placeholder="Sprint Planning", key="conf_m1")
    with col2:
        meeting2 = st.text_input("Meeting 2:", placeholder="Product Roadmap", key="conf_m2")
    
    if st.button("Compare", disabled=not (meeting1 and meeting2)):
        try:
            comparison = st.session_state.analyzer.get_meeting_comparison(meeting1, meeting2)
            
            if "error" in comparison:
                st.error(comparison["error"])
            else:
                st.markdown(f"**{comparison['meeting1']}** vs **{comparison['meeting2']}**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Common Topics**")
                    for t in comparison["common_topics"]:
                        st.write(f"• {t}")
                with col2:
                    st.markdown(f"**Unique to {meeting1}**")
                    for t in comparison["unique_to_meeting1"]:
                        st.write(f"• {t}")
                with col3:
                    st.markdown(f"**Unique to {meeting2}**")
                    for t in comparison["unique_to_meeting2"]:
                        st.write(f"• {t}")
        except Exception as e:
            st.error(f"Error: {e}")


def render_intelligence_tab():
    """Graph Intelligence tab with Node2Vec embeddings"""
    st.header("Graph Intelligence")
    st.caption("Powered by Node2Vec embeddings for semantic similarity")
    
    if not st.session_state.connected:
        st.warning("Connect to Neo4j first")
        return
    
    try:
        from src.ml.embeddings import GraphEmbeddings, HAS_GRAPH_ML
        
        if not HAS_GRAPH_ML:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(251, 191, 36, 0.15) 0%, rgba(251, 191, 36, 0.05) 100%); 
                 border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 16px; padding: 1.5rem; margin: 1rem 0; text-align: center;">
                <h3 style="color: #fbbf24; margin-bottom: 0.5rem;">Graph Intelligence - Local Only</h3>
                <p style="color: rgba(148, 163, 184, 0.9);">
                    This feature uses Node2Vec embeddings which require C++ compilation.<br>
                    It works great when running locally but isn't available on cloud deployments.
                </p>
                <p style="color: rgba(148, 163, 184, 0.7); font-size: 0.9rem; margin-top: 1rem;">
                    To use this feature, clone the repo and run locally with: <code>pip install node2vec</code>
                </p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Check if embeddings are ready
        if not st.session_state.get("graph_embeddings"):
            st.markdown("""
            <div class="feature-card">
                <h4>Graph Intelligence Features</h4>
                <p>This tab uses <strong>Node2Vec</strong> embeddings to enable:</p>
                <ul>
                    <li><strong>Similar People Finder</strong> - Find collaborators with similar work patterns</li>
                    <li><strong>Topic Clustering</strong> - Discover hidden themes across meetings</li>
                    <li><strong>Team Directory</strong> - Browse all team members and their contributions</li>
                </ul>
                <p><em>Click the button below to generate embeddings (takes ~10 seconds).</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Load Graph Intelligence", use_container_width=True, type="primary"):
                with st.spinner("Generating Node2Vec embeddings..."):
                    try:
                        embeddings = GraphEmbeddings(st.session_state.graph_builder.client)
                        embeddings.generate_embeddings(dimensions=64, walk_length=30, num_walks=100)
                        st.session_state.graph_embeddings = embeddings
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating embeddings: {str(e)[:100]}")
            return
        
        # Show stats
        stats = st.session_state.graph_embeddings.get_stats()
        cols = st.columns(4)
        cols[0].metric("Nodes Embedded", stats["nodes_with_embeddings"])
        cols[1].metric("Dimensions", stats["embedding_dimensions"])
        cols[2].metric("Graph Edges", stats["total_edges"])
        cols[3].metric("Node Types", len(stats.get("node_types", {})))
        
        # Feature explainer cards for recruiters
        with st.expander("**How Graph Intelligence Works** (Click to learn about the technology)", expanded=False):
            st.markdown("""
            <div class="tour-card">
                <h4>Node2Vec - Graph Machine Learning</h4>
                <p>This demo uses <strong>Node2Vec</strong>, a graph embedding algorithm that:</p>
                <ul>
                    <li>Converts graph nodes (people, topics, meetings) into 64-dimensional vectors</li>
                    <li>Preserves network relationships - connected nodes have similar embeddings</li>
                    <li>Enables semantic similarity search without explicit feature engineering</li>
                </ul>
                <p><em>Published by Stanford (Grover & Leskovec, 2016). Used by LinkedIn, Pinterest, and Uber.</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="feature-card">
                    <h4>Similar People Finder</h4>
                    <p>Uses cosine similarity on embeddings to find people who attend similar meetings, 
                    work on similar topics, and own similar action items - without keyword matching.</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="feature-card">
                    <h4>Topic Clustering</h4>
                    <p>Applies K-Means clustering on topic embeddings to automatically group 
                    related discussion themes, revealing hidden patterns in meeting content.</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Feature tabs
        feature_tabs = st.tabs(["Similar People", "Topic Clusters", "Team Directory", "Collaboration"])
        
        # Similar People
        with feature_tabs[0]:
            st.subheader("Find Similar Collaborators")
            st.caption("Find people who work on similar topics and projects")
            
            # Get list of people
            people_query = "MATCH (p:Person) RETURN p.name as name ORDER BY name"
            people = st.session_state.graph_builder.client.run_query(people_query)
            people_names = [p["name"] for p in people if p["name"]]
            
            if people_names:
                selected_person = st.selectbox("Select a person:", people_names)
                
                if st.button("Find Similar People", key="find_similar"):
                    similar = st.session_state.graph_embeddings.find_similar_people(selected_person, top_k=5)
                    
                    if similar:
                        st.markdown(f"**People who collaborate similarly to {selected_person}:**")
                        for name, score in similar:
                            score_pct = int(score * 100)
                            st.markdown(f'''
                                <div class="insight-card">
                                    <strong>{name}</strong>
                                    <span style="float: right; color: #a855f7;">{score_pct}% similar</span>
                                    <div style="background: rgba(168, 85, 247, 0.2); height: 4px; border-radius: 2px; margin-top: 8px;">
                                        <div style="background: #a855f7; height: 100%; width: {score_pct}%; border-radius: 2px;"></div>
                                    </div>
                                </div>
                            ''', unsafe_allow_html=True)
                    else:
                        st.info("No similar people found. Try processing more meetings.")
            else:
                st.warning("No people found in the graph. Process some meetings first.")
        
        # Topic Clusters
        with feature_tabs[1]:
            st.subheader("Topic Clustering")
            st.caption("Automatically group related topics based on graph structure")
            
            n_clusters = st.slider("Number of clusters:", min_value=2, max_value=8, value=3)
            
            if st.button("Cluster Topics", key="cluster_topics"):
                clusters = st.session_state.graph_embeddings.cluster_topics(n_clusters=n_clusters)
                
                if clusters:
                    cols = st.columns(min(len(clusters), 3))
                    cluster_colors = ["#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]
                    
                    for i, (cluster_id, topics) in enumerate(clusters.items()):
                        with cols[i % 3]:
                            color = cluster_colors[cluster_id % len(cluster_colors)]
                            st.markdown(f'''
                                <div style="background: {color}20; border: 1px solid {color}50; border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">
                                    <strong style="color: {color};">Cluster {cluster_id + 1}</strong><br>
                                    <small>{len(topics)} topics</small>
                                    <ul style="margin-top: 8px; padding-left: 20px;">
                                        {''.join(f"<li>{t}</li>" for t in topics[:5])}
                                    </ul>
                                </div>
                            ''', unsafe_allow_html=True)
                else:
                    st.warning("Not enough topics to cluster. Process more meetings.")
        
        # Team Directory (replaced Smart Assignment)
        with feature_tabs[2]:
            st.subheader("Team Directory")
            st.caption("See who owns what tasks and their areas of expertise")
            
            # Simple, reliable query that always works
            team_query = """
            MATCH (p:Person)
            OPTIONAL MATCH (p)-[:OWNS]->(a:ActionItem)
            OPTIONAL MATCH (p)-[:ATTENDED]->(m:Meeting)
            OPTIONAL MATCH (p)-[:MADE]->(d:Decision)
            WITH p.name as name, p.role as role,
                 count(DISTINCT a) as action_count,
                 count(DISTINCT m) as meeting_count,
                 count(DISTINCT d) as decision_count,
                 collect(DISTINCT a.description)[0..2] as sample_tasks
            ORDER BY action_count DESC
            RETURN name, role, action_count, meeting_count, decision_count, sample_tasks
            """
            
            if st.button("Show Team", key="show_team"):
                team = st.session_state.graph_builder.client.run_query(team_query)
                
                if team:
                    for person in team:
                        role_text = f" ({person['role']})" if person['role'] else ""
                        tasks = person['sample_tasks'] or []
                        task_preview = tasks[0][:50] + "..." if tasks else "No tasks assigned"
                        
                        st.markdown(f'''
                            <div class="insight-card">
                                <strong>{person['name']}</strong>
                                <span style="color: #a855f7;">{role_text}</span><br>
                                <small style="color: #94a3b8;">
                                    {person['action_count']} tasks | 
                                    {person['meeting_count']} meetings | 
                                    {person['decision_count']} decisions
                                </small><br>
                                <small style="color: #64748b;">Example: {task_preview}</small>
                            </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.info("No team members found. Process some meetings first.")
        
        # Collaboration Strength
        with feature_tabs[3]:
            st.subheader("Collaboration Network")
            st.caption("See who works together most frequently")
            
            if st.button("Analyze Collaborations", key="analyze_collab"):
                collaborations = st.session_state.graph_embeddings.get_collaboration_strength()
                
                if collaborations:
                    st.markdown("**Top Collaboration Pairs:**")
                    for person1, person2, meetings in collaborations[:10]:
                        st.markdown(f'''
                            <div class="insight-card">
                                <strong>{person1}</strong> ↔ <strong>{person2}</strong>
                                <span style="float: right; color: #a855f7;">{meetings} shared meetings</span>
                            </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.info("No collaboration data found. Process more meetings.")
    
    except ImportError as e:
        st.error(f"Missing dependencies: {e}")
        st.code("pip install node2vec networkx scikit-learn numpy")
    except Exception as e:
        st.error(f"Error: {e}")


def main():
    """Main app - Demo mode with auto-connect"""
    init_session_state()
    
    # Auto-connect to Neo4j on first load with loading screen
    if not st.session_state.connected:
        # Show loading screen
        loading_placeholder = st.empty()
        
        with loading_placeholder.container():
            st.markdown("""
                <style>
                    .loading-container {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        min-height: 80vh;
                        text-align: center;
                    }
                    
                    .loading-logo {
                        font-size: 4rem;
                        font-weight: 800;
                        background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 40%, #ec4899 80%, #f97316 100%);
                        background-size: 200% 200%;
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        animation: gradient-shift 3s ease infinite;
                        margin-bottom: 1rem;
                    }
                    
                    @keyframes gradient-shift {
                        0%, 100% { background-position: 0% 50%; }
                        50% { background-position: 100% 50%; }
                    }
                    
                    .loading-subtitle {
                        color: rgba(148, 163, 184, 0.9);
                        font-size: 1.1rem;
                        margin-bottom: 2.5rem;
                    }
                    
                    .loading-bar-container {
                        width: 300px;
                        height: 6px;
                        background: rgba(255, 255, 255, 0.1);
                        border-radius: 10px;
                        overflow: hidden;
                        margin-bottom: 1.5rem;
                    }
                    
                    .loading-bar {
                        height: 100%;
                        width: 30%;
                        background: linear-gradient(90deg, #06b6d4, #8b5cf6, #ec4899, #8b5cf6, #06b6d4);
                        background-size: 200% 100%;
                        border-radius: 10px;
                        animation: loading-slide 1.5s ease-in-out infinite;
                    }
                    
                    @keyframes loading-slide {
                        0% { transform: translateX(-100%); }
                        100% { transform: translateX(400%); }
                    }
                    
                    .loading-status {
                        color: rgba(139, 92, 246, 0.9);
                        font-size: 0.9rem;
                        font-weight: 500;
                    }
                    
                    .loading-dots::after {
                        content: '';
                        animation: dots 1.5s steps(4, end) infinite;
                    }
                    
                    @keyframes dots {
                        0%, 20% { content: ''; }
                        40% { content: '.'; }
                        60% { content: '..'; }
                        80%, 100% { content: '...'; }
                    }
                </style>
                
                <div class="loading-container">
                    <div class="loading-logo">⚡ Lexigraph</div>
                    <div class="loading-subtitle">AI-Powered Meeting Intelligence</div>
                    <div class="loading-bar-container">
                        <div class="loading-bar"></div>
                    </div>
                    <div class="loading-status">Connecting to Knowledge Graph<span class="loading-dots"></span></div>
                </div>
            """, unsafe_allow_html=True)
        
        try:
            st.session_state.extractor = ExtractorAgent()
            st.session_state.graph_builder = GraphBuilderAgent()
            st.session_state.graph_builder.connect()
            st.session_state.query_agent = QueryAgent()
            st.session_state.query_agent.connect()
            st.session_state.analyzer = AnalyzerAgent()
            st.session_state.analyzer.connect()
            st.session_state.summary_agent = SummaryAgent()
            st.session_state.summary_agent.connect()
            st.session_state.connected = True
            
            # Note: Embeddings will be generated lazily when Intelligence tab is accessed
            
            # Clear loading screen and rerun to show app
            loading_placeholder.empty()
            st.rerun()
            
        except Exception as e:
            loading_placeholder.empty()
            st.error(f"Could not connect to Neo4j: {str(e)[:100]}")
            st.info("Please check your Neo4j connection settings in .env file")
            return
    
    # All tabs - at the top
    tabs = st.tabs([
        "Home",
        "Chat", 
        "Graph View",
        "Intelligence",
        "Deadlines",
        "Insights",
        "Summary",
        "Conflicts"
    ])
    
    with tabs[0]:
        render_header()  # Logo only on first tab
        render_extraction_tab()
    with tabs[1]:
        render_chat_tab()
    with tabs[2]:
        render_visualization_tab()
    with tabs[3]:
        render_intelligence_tab()
    with tabs[4]:
        render_deadlines_tab()
    with tabs[5]:
        render_insights_tab()
    with tabs[6]:
        render_summary_tab()
    with tabs[7]:
        render_conflicts_tab()


if __name__ == "__main__":
    main()
