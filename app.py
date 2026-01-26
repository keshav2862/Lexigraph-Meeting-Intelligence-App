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

# Page configuration
st.set_page_config(
    page_title="Lexigraph",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d1f 100%);
    }
    
    /* Reduce top padding */
    .block-container {
        padding-top: 2rem !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header */
    .hero-header {
        text-align: center;
        padding: 0.75rem 0 0.75rem;
    }
    
    .logo-text {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: brightness(1); }
        to { filter: brightness(1.2); }
    }
    
    .tagline {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 300;
        margin-top: 0.25rem;
    }
    
    /* Chat bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.75rem 0;
        margin-left: 25%;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
    }
    
    .assistant-bubble {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #e2e8f0;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.75rem 0;
        margin-right: 25%;
    }
    
    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    }
    
    .deadline-card {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    .deadline-overdue {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .deadline-soon {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .deadline-upcoming {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .conflict-card {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .insight-card {
        background: rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 5px 20px rgba(124, 58, 237, 0.5) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 0.75rem;
        gap: 0.75rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-radius: 10px;
        font-size: 0.9rem;
        padding: 0.75rem 1.25rem !important;
        min-width: 100px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
        color: white !important;
    }
    
    /* Force dark mode on all inputs */
    .stTextArea textarea, .stTextInput input {
        background: #1a1a3e !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }
    
    /* Select boxes and dropdowns */
    .stSelectbox > div > div {
        background: #1a1a3e !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #e2e8f0 !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background: #1a1a3e !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background: #1a1a3e !important;
        color: #e2e8f0 !important;
    }
    
    /* Dropdown menu */
    [data-baseweb="popover"] {
        background: #1a1a3e !important;
    }
    
    [data-baseweb="menu"] {
        background: #1a1a3e !important;
    }
    
    [data-baseweb="menu"] li {
        background: #1a1a3e !important;
        color: #e2e8f0 !important;
    }
    
    [data-baseweb="menu"] li:hover {
        background: #2a2a4e !important;
    }
    
    /* Header bar */
    header[data-testid="stHeader"] {
        background: rgba(15, 15, 35, 0.95) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Top toolbar */
    .stToolbar {
        background: transparent !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar content */
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    /* All text */
    .stMarkdown, p, span, label {
        color: #e2e8f0 !important;
    }
    
    /* Checkboxes */
    .stCheckbox label span {
        color: #e2e8f0 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] { color: #a855f7 !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.05); }
    ::-webkit-scrollbar-thumb { background: rgba(124, 58, 237, 0.5); border-radius: 3px; }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
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
    """Render header"""
    st.markdown("""
        <div class="hero-header">
            <div class="logo-text">⚡ Lexigraph</div>
            <div class="tagline">Transform meetings into intelligent knowledge graphs</div>
        </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if st.session_state.connected:
            st.success("✓ Connected to Neo4j")
        else:
            st.warning("⚠ Not connected")
            
        if st.button("🔌 Connect to Neo4j", use_container_width=True):
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
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Chat", use_container_width=True):
                    st.session_state.messages = []
                    if st.session_state.query_agent:
                        st.session_state.query_agent.clear_history()
                    st.rerun()
            with col2:
                if st.button("🧹 DB", use_container_width=True):
                    st.session_state.graph_builder.client.clear_database()
                    st.session_state.messages = []
                    st.rerun()


def render_extraction_tab():
    """Extraction tab"""
    st.header("📝 Process Meeting Transcript")
    
    samples = load_sample_transcripts()
    if samples:
        sample_choice = st.selectbox(
            "Load sample:", ["(Paste your own)"] + list(samples.keys())
        )
        transcript = samples.get(sample_choice, "") if sample_choice != "(Paste your own)" else ""
    else:
        transcript = ""
    
    transcript_input = st.text_area(
        "Transcript:", value=transcript, height=250,
        placeholder="Paste meeting transcript..."
    )
    
    if st.button("⚡ Extract & Build", disabled=not st.session_state.connected or not transcript_input.strip()):
        with st.spinner("Processing..."):
            try:
                extraction = st.session_state.extractor.extract(transcript_input)
                st.session_state.last_extraction = extraction
                stats = st.session_state.graph_builder.build_graph(extraction)
                st.success(f"✅ Created {stats['relationships']} relationships!")
            except Exception as e:
                error_msg = get_user_friendly_error(e)
                st.error(f"❌ {error_msg}")
    
    if st.session_state.last_extraction:
        extraction = st.session_state.last_extraction
        st.divider()
        
        cols = st.columns(5)
        for col, (label, count) in zip(cols, [
            ("👥 People", len(extraction.people)),
            ("💬 Topics", len(extraction.topics)),
            ("✅ Decisions", len(extraction.decisions)),
            ("📋 Actions", len(extraction.action_items)),
            ("🤝 Commits", len(extraction.commitments))
        ]):
            col.metric(label, count)
        
        # Export button
        st.divider()
        export_md = export_meeting_summary(extraction)
        st.download_button(
            label="📥 Download Summary (Markdown)",
            data=export_md,
            file_name=f"{extraction.meeting_title or 'meeting'}_summary.md",
            mime="text/markdown"
        )


def render_chat_tab():
    """Chat tab"""
    st.header("💬 Chat with Your Meetings")
    
    if not st.session_state.connected:
        st.warning("Connect to Neo4j to start")
        return
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            content = msg["content"].replace('\n', '<br>').replace('•', '&#8226;')
            st.markdown(f'<div class="assistant-bubble">{content}</div>', unsafe_allow_html=True)
            if "cypher" in msg:
                with st.expander("🔍 Query"):
                    st.code(msg["cypher"], language="cypher")
    
    if not st.session_state.messages:
        st.info("💡 Try: 'What decisions were made?' or 'What should Mike do?'")
    
    if prompt := st.chat_input("Ask about your meetings..."):
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
    st.header("📊 Knowledge Graph Visualization")
    
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
        
        st.caption("💡 Drag nodes to rearrange • Hover for details • Scroll to zoom • Use navigation buttons")
        
    except ImportError:
        st.error("Please install pyvis: `pip install pyvis`")
    except Exception as e:
        st.error(f"Visualization error: {e}")


def render_deadlines_tab():
    """Deadline tracker tab"""
    st.header("⏰ Deadline Tracker")
    
    if not st.session_state.connected:
        st.warning("Connect to Neo4j first")
        return
    
    try:
        deadlines = st.session_state.analyzer.get_deadline_status()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔴 Overdue")
            if deadlines["overdue"]:
                for item in deadlines["overdue"]:
                    st.markdown(f'''
                        <div class="deadline-card deadline-overdue">
                            <strong>{item["task"][:50]}...</strong><br>
                            <small>👤 {item["owner"] or "Unassigned"} | 📅 {item["deadline"]}</small>
                        </div>
                    ''', unsafe_allow_html=True)
            else:
                st.success("None overdue! 🎉")
        
        with col2:
            st.markdown("### 🟡 Due Soon")
            if deadlines["due_soon"]:
                for item in deadlines["due_soon"]:
                    st.markdown(f'''
                        <div class="deadline-card deadline-soon">
                            <strong>{item["task"][:50]}...</strong><br>
                            <small>👤 {item["owner"] or "Unassigned"} | 📅 {item["deadline"]}</small>
                        </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("Nothing due soon")
        
        with col3:
            st.markdown("### 🟢 Upcoming")
            if deadlines["upcoming"]:
                for item in deadlines["upcoming"][:5]:
                    st.markdown(f'''
                        <div class="deadline-card deadline-upcoming">
                            <strong>{item["task"][:50]}...</strong><br>
                            <small>👤 {item["owner"] or "Unassigned"} | 📅 {item["deadline"] or "No deadline"}</small>
                        </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("No upcoming items")
        
        if deadlines["no_deadline"]:
            with st.expander(f"📋 No Deadline Set ({len(deadlines['no_deadline'])} items)"):
                for item in deadlines["no_deadline"]:
                    st.write(f"• {item['task']} ({item['owner'] or 'Unassigned'})")
        
        # Export button
        st.divider()
        export_md = export_action_items(deadlines)
        st.download_button(
            label="📥 Download Action Items Report",
            data=export_md,
            file_name="action_items_report.md",
            mime="text/markdown"
        )
    
    except Exception as e:
        error_msg = get_user_friendly_error(e)
        st.error(f"❌ {error_msg}")


def render_insights_tab():
    """Cross-meeting insights tab"""
    st.header("📈 Cross-Meeting Insights")
    
    if not st.session_state.connected:
        st.warning("Connect to Neo4j first")
        return
    
    tab1, tab2 = st.tabs(["👥 People", "💬 Topics"])
    
    with tab1:
        st.subheader("Person Engagement")
        try:
            insights = st.session_state.analyzer.get_person_insights()
            
            for person in insights:
                st.markdown(f'''
                    <div class="insight-card">
                        <strong>{person["name"]}</strong> 
                        <span style="color: #94a3b8;">({person["role"] or "No role"})</span><br>
                        <small>
                            📅 {person["meetings_attended"]} meetings | 
                            📋 {person["action_items"]} actions | 
                            ✅ {person["decisions_made"]} decisions | 
                            🤝 {person["commitments"]} commitments
                        </small>
                    </div>
                ''', unsafe_allow_html=True)
        except Exception as e:
            error_msg = get_user_friendly_error(e)
            st.error(f"❌ {error_msg}")
    
    with tab2:
        st.subheader("Topic Trends")
        try:
            topics = st.session_state.analyzer.get_topic_trends()
            
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
            st.error(f"❌ {error_msg}")
    
    # Export button for insights
    st.divider()
    try:
        person_data = st.session_state.analyzer.get_person_insights()
        topic_data = st.session_state.analyzer.get_topic_trends()
        export_md = export_insights_report(person_data, topic_data)
        st.download_button(
            label="📥 Download Insights Report",
            data=export_md,
            file_name="insights_report.md",
            mime="text/markdown"
        )
    except:
        pass


def render_summary_tab():
    """Auto-summary tab"""
    st.header("📝 Auto-Generated Summaries")
    
    if not st.session_state.connected:
        st.warning("Connect to Neo4j first")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Single Meeting Summary")
        meeting_title = st.text_input("Meeting name:", placeholder="e.g., Q3 Planning")
        
        if st.button("📝 Generate Summary", disabled=not meeting_title):
            with st.spinner("Generating summary..."):
                try:
                    summary = st.session_state.summary_agent.generate_meeting_summary(meeting_title)
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        st.subheader("Cross-Meeting Overview")
        
        if st.button("📊 Generate Overview"):
            with st.spinner("Analyzing all meetings..."):
                try:
                    summary = st.session_state.summary_agent.generate_cross_meeting_summary()
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"Error: {e}")


def render_conflicts_tab():
    """Conflict detection tab"""
    st.header("⚠️ Conflict Detection")
    
    if not st.session_state.connected:
        st.warning("Connect to Neo4j first")
        return
    
    st.info("Analyzes decisions across meetings for potential contradictions or conflicts.")
    
    if st.button("🔍 Analyze Conflicts"):
        with st.spinner("Analyzing decisions for conflicts..."):
            try:
                analysis = st.session_state.analyzer.detect_conflicts()
                
                if "no conflict" in analysis.lower():
                    st.success("✅ No conflicts detected!")
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
    
    col1, col2 = st.columns(2)
    with col1:
        meeting1 = st.text_input("Meeting 1:", placeholder="e.g., Product Sync")
    with col2:
        meeting2 = st.text_input("Meeting 2:", placeholder="e.g., Q3 Planning")
    
    if st.button("🔄 Compare", disabled=not (meeting1 and meeting2)):
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


def main():
    """Main app"""
    init_session_state()
    render_header()
    render_sidebar()
    
    # All tabs
    tabs = st.tabs([
        "📝 Extract",
        "💬 Chat", 
        "📊 Visualize",
        "⏰ Deadlines",
        "📈 Insights",
        "📝 Summary",
        "⚠️ Conflicts"
    ])
    
    with tabs[0]:
        render_extraction_tab()
    with tabs[1]:
        render_chat_tab()
    with tabs[2]:
        render_visualization_tab()
    with tabs[3]:
        render_deadlines_tab()
    with tabs[4]:
        render_insights_tab()
    with tabs[5]:
        render_summary_tab()
    with tabs[6]:
        render_conflicts_tab()


if __name__ == "__main__":
    main()
