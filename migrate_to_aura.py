"""
Migration script: Copy local Neo4j database to AuraDB
Run this once to transfer your local graph to the cloud.
"""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# Local Neo4j (source)
LOCAL_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
LOCAL_USER = os.getenv("NEO4J_USER", "neo4j")
LOCAL_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# AuraDB (destination) - UPDATE THESE WITH YOUR AURA CREDENTIALS
AURA_URI = "neo4j+s://3c7b7462.databases.neo4j.io"  # <-- UPDATE THIS
AURA_USER = "neo4j"
AURA_PASSWORD = "du6rcHsdIkg2VOtc-Rlu-QizOgvkrkFFf3wiXAM1mg4"  # <-- UPDATE THIS


def migrate():
    print("Connecting to local Neo4j...")
    local_driver = GraphDatabase.driver(LOCAL_URI, auth=(LOCAL_USER, LOCAL_PASSWORD))
    
    print("Connecting to AuraDB...")
    aura_driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))
    
    # Step 1: Clear AuraDB
    print("\n1. Clearing AuraDB...")
    with aura_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("   ✓ AuraDB cleared")
    
    # Step 2: Export all nodes from local
    print("\n2. Exporting nodes from local Neo4j...")
    nodes = []
    with local_driver.session() as session:
        result = session.run("""
            MATCH (n) 
            RETURN labels(n) as labels, properties(n) as props, elementId(n) as id
        """)
        for record in result:
            nodes.append({
                "labels": record["labels"],
                "props": dict(record["props"]),
                "id": record["id"]
            })
    print(f"   ✓ Found {len(nodes)} nodes")
    
    # Step 3: Export all relationships from local
    print("\n3. Exporting relationships from local Neo4j...")
    rels = []
    with local_driver.session() as session:
        result = session.run("""
            MATCH (a)-[r]->(b) 
            RETURN type(r) as type, properties(r) as props,
                   elementId(a) as start_id, elementId(b) as end_id,
                   labels(a)[0] as start_label, labels(b)[0] as end_label,
                   properties(a) as start_props, properties(b) as end_props
        """)
        for record in result:
            rels.append({
                "type": record["type"],
                "props": dict(record["props"]) if record["props"] else {},
                "start_label": record["start_label"],
                "end_label": record["end_label"],
                "start_props": dict(record["start_props"]),
                "end_props": dict(record["end_props"])
            })
    print(f"   ✓ Found {len(rels)} relationships")
    
    # Step 4: Create nodes in AuraDB
    print("\n4. Creating nodes in AuraDB...")
    with aura_driver.session() as session:
        for i, node in enumerate(nodes):
            labels_str = ":".join(node["labels"])
            # Create node with all properties
            props_str = ", ".join([f"{k}: ${k}" for k in node["props"].keys()])
            query = f"CREATE (n:{labels_str} {{{props_str}}})"
            session.run(query, **node["props"])
            if (i + 1) % 50 == 0:
                print(f"   Created {i + 1}/{len(nodes)} nodes...")
    print(f"   ✓ Created {len(nodes)} nodes")
    
    # Step 5: Create relationships in AuraDB
    print("\n5. Creating relationships in AuraDB...")
    with aura_driver.session() as session:
        for i, rel in enumerate(rels):
            # Find a unique property for matching (usually name, title, or description)
            start_match = get_match_property(rel["start_props"])
            end_match = get_match_property(rel["end_props"])
            
            if start_match and end_match:
                query = f"""
                    MATCH (a:{rel['start_label']} {{{start_match[0]}: $start_val}})
                    MATCH (b:{rel['end_label']} {{{end_match[0]}: $end_val}})
                    CREATE (a)-[r:{rel['type']}]->(b)
                """
                try:
                    session.run(query, start_val=start_match[1], end_val=end_match[1])
                except Exception as e:
                    pass  # Skip if relationship already exists or nodes not found
            
            if (i + 1) % 50 == 0:
                print(f"   Created {i + 1}/{len(rels)} relationships...")
    print(f"   ✓ Created relationships")
    
    # Verify
    print("\n6. Verifying AuraDB...")
    with aura_driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as nodes")
        node_count = result.single()["nodes"]
        result = session.run("MATCH ()-[r]->() RETURN count(r) as rels")
        rel_count = result.single()["rels"]
    print(f"   ✓ AuraDB now has {node_count} nodes and {rel_count} relationships")
    
    local_driver.close()
    aura_driver.close()
    print("\n✅ Migration complete!")


def get_match_property(props):
    """Get a property suitable for matching nodes"""
    for key in ["name", "title", "description"]:
        if key in props and props[key]:
            return (key, props[key])
    # Fallback to first property
    for key, val in props.items():
        if val and isinstance(val, str):
            return (key, val)
    return None


if __name__ == "__main__":
    print("=" * 50)
    print("Neo4j Local → AuraDB Migration")
    print("=" * 50)
    print("\n⚠️  Make sure to update AURA_URI and AURA_PASSWORD in this script!")
    print("    Then run: python migrate_to_aura.py\n")
    
    response = input("Have you updated the AuraDB credentials? (yes/no): ")
    if response.lower() == "yes":
        migrate()
    else:
        print("Please update the credentials at the top of this file first.")
