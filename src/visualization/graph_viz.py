"""Interactive Graph Visualization using PyVis

Creates interactive network visualizations of the Neo4j knowledge graph
that can be embedded in Streamlit.
"""

from pyvis.network import Network
from typing import Dict, List, Any, Optional
import tempfile
import os


# Node color scheme by type
NODE_COLORS = {
    "Meeting": "#3b82f6",      # Blue
    "Person": "#22c55e",       # Green
    "Topic": "#f59e0b",        # Amber
    "Decision": "#8b5cf6",     # Purple
    "ActionItem": "#ef4444",   # Red
    "Commitment": "#ec4899"    # Pink
}

# Node icons/symbols
NODE_SHAPES = {
    "Meeting": "dot",
    "Person": "dot",
    "Topic": "diamond",
    "Decision": "square",
    "ActionItem": "triangle",
    "Commitment": "star"
}


def create_knowledge_graph(neo4j_client, height: str = "600px") -> str:
    """
    Query Neo4j and create an interactive PyVis network graph.
    
    Args:
        neo4j_client: Connected Neo4jClient instance
        height: Height of the visualization
        
    Returns:
        HTML string of the interactive graph
    """
    # Initialize network with settings
    net = Network(
        height=height,
        width="100%",
        bgcolor="#0f0f23",
        font_color="#e2e8f0",
        directed=True
    )
    
    # Physics settings for spacious layout
    net.set_options("""
    {
        "nodes": {
            "font": {
                "size": 16,
                "color": "#e2e8f0"
            },
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "scaling": {
                "min": 20,
                "max": 40
            }
        },
        "edges": {
            "color": {
                "color": "#4b5563",
                "highlight": "#a855f7"
            },
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            },
            "smooth": {
                "type": "curvedCW",
                "roundness": 0.15
            },
            "font": {
                "size": 10,
                "color": "#94a3b8"
            }
        },
        "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.005,
                "springLength": 250,
                "springConstant": 0.05,
                "damping": 0.4,
                "avoidOverlap": 0.8
            },
            "stabilization": {
                "iterations": 150
            },
            "minVelocity": 0.75
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)
    
    # Track added nodes to avoid duplicates
    added_nodes = set()
    
    # Query all nodes and relationships
    query = """
    MATCH (n)
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN 
        labels(n)[0] as from_type,
        properties(n) as from_props,
        elementId(n) as from_id,
        type(r) as rel_type,
        labels(m)[0] as to_type,
        properties(m) as to_props,
        elementId(m) as to_id
    """
    
    results = neo4j_client.run_query(query)
    
    for record in results:
        from_type = record.get("from_type")
        from_props = record.get("from_props", {})
        from_id = record.get("from_id")
        
        if from_id and from_id not in added_nodes:
            # Get display label
            label = _get_node_label(from_type, from_props)
            title = _get_node_tooltip(from_type, from_props)
            
            net.add_node(
                from_id,
                label=label,
                title=title,
                color=NODE_COLORS.get(from_type, "#6b7280"),
                shape=NODE_SHAPES.get(from_type, "dot"),
                size=25 if from_type == "Meeting" else 20
            )
            added_nodes.add(from_id)
        
        # Add target node and edge if relationship exists
        to_id = record.get("to_id")
        to_type = record.get("to_type")
        to_props = record.get("to_props", {})
        rel_type = record.get("rel_type")
        
        if to_id and to_id not in added_nodes:
            label = _get_node_label(to_type, to_props)
            title = _get_node_tooltip(to_type, to_props)
            
            net.add_node(
                to_id,
                label=label,
                title=title,
                color=NODE_COLORS.get(to_type, "#6b7280"),
                shape=NODE_SHAPES.get(to_type, "dot"),
                size=20
            )
            added_nodes.add(to_id)
        
        if rel_type and from_id and to_id:
            net.add_edge(from_id, to_id, title=rel_type, label=rel_type)
    
    # Generate HTML without temp file (avoids Windows file lock issues)
    html_content = net.generate_html()
    
    return html_content


def create_knowledge_graph_filtered(neo4j_client, active_types: List[str], height: str = "700px") -> str:
    """
    Query Neo4j and create a filtered interactive PyVis network graph.
    
    Args:
        neo4j_client: Connected Neo4jClient instance
        active_types: List of node types to include (e.g., ["Meeting", "Person"])
        height: Height of the visualization
        
    Returns:
        HTML string of the interactive graph
    """
    # Initialize network with settings
    net = Network(
        height=height,
        width="100%",
        bgcolor="#0f0f23",
        font_color="#e2e8f0",
        directed=True
    )
    
    # Physics settings for spacious layout
    net.set_options("""
    {
        "nodes": {
            "font": {
                "size": 16,
                "color": "#e2e8f0"
            },
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "scaling": {
                "min": 25,
                "max": 50
            }
        },
        "edges": {
            "color": {
                "color": "#4b5563",
                "highlight": "#a855f7"
            },
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 0.5
                }
            },
            "smooth": {
                "type": "curvedCW",
                "roundness": 0.15
            },
            "font": {
                "size": 10,
                "color": "#94a3b8"
            }
        },
        "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -150,
                "centralGravity": 0.003,
                "springLength": 300,
                "springConstant": 0.04,
                "damping": 0.5,
                "avoidOverlap": 1
            },
            "stabilization": {
                "iterations": 200
            },
            "minVelocity": 0.5
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)
    
    # Track added nodes to avoid duplicates
    added_nodes = set()
    
    # Query all nodes and relationships
    query = """
    MATCH (n)
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN 
        labels(n)[0] as from_type,
        properties(n) as from_props,
        elementId(n) as from_id,
        type(r) as rel_type,
        labels(m)[0] as to_type,
        properties(m) as to_props,
        elementId(m) as to_id
    """
    
    results = neo4j_client.run_query(query)
    
    for record in results:
        from_type = record.get("from_type")
        from_props = record.get("from_props", {})
        from_id = record.get("from_id")
        to_type = record.get("to_type")
        to_props = record.get("to_props", {})
        to_id = record.get("to_id")
        rel_type = record.get("rel_type")
        
        # Only add nodes of active types
        if from_type in active_types and from_id and from_id not in added_nodes:
            label = _get_node_label(from_type, from_props)
            title = _get_node_tooltip(from_type, from_props)
            
            net.add_node(
                from_id,
                label=label,
                title=title,
                color=NODE_COLORS.get(from_type, "#6b7280"),
                shape=NODE_SHAPES.get(from_type, "dot"),
                size=30 if from_type == "Meeting" else 25
            )
            added_nodes.add(from_id)
        
        if to_type in active_types and to_id and to_id not in added_nodes:
            label = _get_node_label(to_type, to_props)
            title = _get_node_tooltip(to_type, to_props)
            
            net.add_node(
                to_id,
                label=label,
                title=title,
                color=NODE_COLORS.get(to_type, "#6b7280"),
                shape=NODE_SHAPES.get(to_type, "dot"),
                size=25
            )
            added_nodes.add(to_id)
        
        # Only add edge if both nodes are in active types
        if rel_type and from_id in added_nodes and to_id in added_nodes:
            net.add_edge(from_id, to_id, title=rel_type, label=rel_type)
    
    # Generate HTML without temp file (avoids Windows file lock issues)
    html_content = net.generate_html()
    
    return html_content


def _get_node_label(node_type: str, props: Dict) -> str:
    """Get short display label for a node"""
    if node_type == "Meeting":
        title = props.get("title", "Meeting")
        return title[:20] + "..." if len(title) > 20 else title
    elif node_type == "Person":
        name = props.get("name", "Person")
        return name.split()[0] if name else "Person"  # First name only
    elif node_type == "Topic":
        name = props.get("name", "Topic")
        return name[:15] + "..." if len(name) > 15 else name
    elif node_type == "Decision":
        desc = props.get("description", "Decision")
        return desc[:15] + "..." if len(desc) > 15 else desc
    elif node_type == "ActionItem":
        desc = props.get("description", "Action")
        return desc[:15] + "..." if len(desc) > 15 else desc
    elif node_type == "Commitment":
        desc = props.get("description", "Commitment")
        return desc[:15] + "..." if len(desc) > 15 else desc
    return str(props.get("name", props.get("description", node_type)))[:15]


def _get_node_tooltip(node_type: str, props: Dict) -> str:
    """Get detailed tooltip for a node"""
    lines = [f"<b>{node_type}</b><br>"]
    
    for key, value in props.items():
        if value:
            lines.append(f"<b>{key}:</b> {value}<br>")
    
    return "".join(lines)


def get_graph_legend_html() -> str:
    """Return HTML for the graph legend"""
    legend_items = []
    for node_type, color in NODE_COLORS.items():
        legend_items.append(f'''
            <span style="display: inline-flex; align-items: center; margin-right: 1rem;">
                <span style="width: 12px; height: 12px; background: {color}; border-radius: 50%; margin-right: 0.5rem;"></span>
                <span style="color: #94a3b8; font-size: 0.875rem;">{node_type}</span>
            </span>
        ''')
    
    return f'''
        <div style="display: flex; flex-wrap: wrap; justify-content: center; 
                    padding: 1rem; background: rgba(255,255,255,0.03); 
                    border-radius: 10px; margin-bottom: 1rem;">
            {"".join(legend_items)}
        </div>
    '''
