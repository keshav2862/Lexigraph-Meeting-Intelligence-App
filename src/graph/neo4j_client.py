"""Neo4j client for MeetingMind knowledge graph operations.

Handles connection management and provides methods for:
- Creating nodes (Person, Topic, Decision, etc.)
- Creating relationships between nodes
- Querying the graph
"""

from typing import Any, Dict, List, Optional
from contextlib import contextmanager
from neo4j import GraphDatabase, Driver
from src.config import config


class Neo4jClient:
    """Client for interacting with Neo4j graph database"""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        self.uri = uri or config.NEO4J_URI
        self.username = username or config.NEO4J_USERNAME
        self.password = password or config.NEO4J_PASSWORD
        self._driver: Optional[Driver] = None
        
    def connect(self) -> None:
        """Establish connection to Neo4j"""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            
    def close(self) -> None:
        """Close the Neo4j connection"""
        if self._driver:
            self._driver.close()
            self._driver = None
            
    @contextmanager
    def session(self):
        """Context manager for Neo4j sessions"""
        if self._driver is None:
            self.connect()
        session = self._driver.session()
        try:
            yield session
        finally:
            session.close()
            
    def run_query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results as list of dicts"""
        with self.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]
    
    # ==================== Node Creation ====================
    
    def create_meeting(self, title: str, date: Optional[str] = None) -> str:
        """Create a Meeting node, return its ID"""
        query = """
        MERGE (m:Meeting {title: $title})
        SET m.date = $date
        RETURN elementId(m) as id
        """
        result = self.run_query(query, {"title": title, "date": date})
        return result[0]["id"] if result else None
    
    def create_person(self, name: str, role: Optional[str] = None) -> str:
        """Create a Person node, return its ID"""
        query = """
        MERGE (p:Person {name: $name})
        SET p.role = $role
        RETURN elementId(p) as id
        """
        result = self.run_query(query, {"name": name, "role": role})
        return result[0]["id"] if result else None
    
    def create_topic(self, name: str, description: Optional[str] = None) -> str:
        """Create a Topic node, return its ID"""
        query = """
        MERGE (t:Topic {name: $name})
        SET t.description = $description
        RETURN elementId(t) as id
        """
        result = self.run_query(query, {"name": name, "description": description})
        return result[0]["id"] if result else None
    
    def create_decision(self, description: str) -> str:
        """Create a Decision node, return its ID"""
        query = """
        CREATE (d:Decision {description: $description})
        RETURN elementId(d) as id
        """
        result = self.run_query(query, {"description": description})
        return result[0]["id"] if result else None
    
    def create_action_item(
        self, 
        description: str, 
        deadline: Optional[str] = None,
        priority: Optional[str] = None
    ) -> str:
        """Create an ActionItem node, return its ID"""
        query = """
        CREATE (a:ActionItem {
            description: $description,
            deadline: $deadline,
            priority: $priority,
            status: 'pending'
        })
        RETURN elementId(a) as id
        """
        result = self.run_query(query, {
            "description": description,
            "deadline": deadline,
            "priority": priority
        })
        return result[0]["id"] if result else None
    
    def create_commitment(self, description: str) -> str:
        """Create a Commitment node, return its ID"""
        query = """
        CREATE (c:Commitment {description: $description})
        RETURN elementId(c) as id
        """
        result = self.run_query(query, {"description": description})
        return result[0]["id"] if result else None
    
    # ==================== Relationship Creation ====================
    
    def create_relationship(
        self,
        from_label: str,
        from_prop: str,
        from_value: str,
        rel_type: str,
        to_label: str,
        to_prop: str,
        to_value: str
    ) -> bool:
        """Create a relationship between two nodes using fuzzy name matching"""
        # Use toLower and CONTAINS for more flexible matching
        query = f"""
        MATCH (a:{from_label})
        WHERE toLower(a.{from_prop}) CONTAINS toLower($from_value)
           OR toLower($from_value) CONTAINS toLower(a.{from_prop})
        MATCH (b:{to_label})
        WHERE toLower(b.{to_prop}) CONTAINS toLower($to_value)
           OR toLower($to_value) CONTAINS toLower(b.{to_prop})
        MERGE (a)-[r:{rel_type}]->(b)
        RETURN r
        """
        result = self.run_query(query, {
            "from_value": from_value,
            "to_value": to_value
        })
        return len(result) > 0
    
    def link_person_to_meeting(self, person_name: str, meeting_title: str) -> bool:
        """Create ATTENDED relationship between Person and Meeting"""
        return self.create_relationship(
            "Person", "name", person_name,
            "ATTENDED",
            "Meeting", "title", meeting_title
        )
    
    def link_meeting_to_topic(self, meeting_title: str, topic_name: str) -> bool:
        """Create DISCUSSED relationship between Meeting and Topic"""
        return self.create_relationship(
            "Meeting", "title", meeting_title,
            "DISCUSSED",
            "Topic", "name", topic_name
        )
    
    def link_person_to_decision(self, person_name: str, decision_desc: str) -> bool:
        """Create MADE relationship between Person and Decision"""
        return self.create_relationship(
            "Person", "name", person_name,
            "MADE",
            "Decision", "description", decision_desc
        )
    
    def link_person_to_action_item(self, person_name: str, action_desc: str) -> bool:
        """Create OWNS relationship between Person and ActionItem"""
        return self.create_relationship(
            "Person", "name", person_name,
            "OWNS",
            "ActionItem", "description", action_desc
        )
    
    def link_person_to_commitment(self, person_name: str, commitment_desc: str) -> bool:
        """Create COMMITTED relationship between Person and Commitment"""
        return self.create_relationship(
            "Person", "name", person_name,
            "COMMITTED",
            "Commitment", "description", commitment_desc
        )
    
    def link_decision_to_topic(self, decision_desc: str, topic_name: str) -> bool:
        """Create ABOUT relationship between Decision and Topic"""
        return self.create_relationship(
            "Decision", "description", decision_desc,
            "ABOUT",
            "Topic", "name", topic_name
        )
    
    def link_meeting_to_decision(self, meeting_title: str, decision_desc: str) -> bool:
        """Create CONTAINS relationship between Meeting and Decision"""
        return self.create_relationship(
            "Meeting", "title", meeting_title,
            "CONTAINS",
            "Decision", "description", decision_desc
        )
    
    def link_meeting_to_action_item(self, meeting_title: str, action_desc: str) -> bool:
        """Create CONTAINS relationship between Meeting and ActionItem"""
        return self.create_relationship(
            "Meeting", "title", meeting_title,
            "CONTAINS",
            "ActionItem", "description", action_desc
        )
    
    # ==================== Utility Methods ====================
    
    def clear_database(self) -> None:
        """Delete all nodes and relationships - USE WITH CAUTION"""
        self.run_query("MATCH (n) DETACH DELETE n")
        
    def get_schema(self) -> List[Dict]:
        """Get the current schema of the database"""
        return self.run_query("CALL db.schema.visualization()")
    
    def get_node_counts(self) -> Dict[str, int]:
        """Get count of each node type"""
        query = """
        CALL db.labels() YIELD label
        CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {})
        YIELD value
        RETURN label, value.count as count
        """
        # Fallback for when APOC is not installed
        try:
            result = self.run_query(query)
            return {r["label"]: r["count"] for r in result}
        except Exception:
            # Simple fallback
            labels = ["Meeting", "Person", "Topic", "Decision", "ActionItem", "Commitment"]
            counts = {}
            for label in labels:
                result = self.run_query(f"MATCH (n:{label}) RETURN count(n) as count")
                counts[label] = result[0]["count"] if result else 0
            return counts


# Convenience function for quick testing
if __name__ == "__main__":
    client = Neo4jClient()
    try:
        client.connect()
        print("✓ Connected to Neo4j successfully!")
        counts = client.get_node_counts()
        print(f"Current node counts: {counts}")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
    finally:
        client.close()
