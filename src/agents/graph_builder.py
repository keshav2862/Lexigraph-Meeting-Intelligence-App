"""Graph Builder Agent - Transforms extracted entities into Neo4j graph.

Takes MeetingExtraction and creates:
- Nodes for all entities
- Relationships between entities

This agent demonstrates:
- Graph database operations
- Entity resolution with MERGE
- Relationship modeling
"""

from typing import Optional
from src.models.entities import MeetingExtraction
from src.graph.neo4j_client import Neo4jClient


class GraphBuilderAgent:
    """Agent that builds knowledge graph from extracted meeting entities"""
    
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.client = neo4j_client or Neo4jClient()
        self._connected = False
        
    def connect(self) -> None:
        """Connect to Neo4j database"""
        if not self._connected:
            self.client.connect()
            self._connected = True
            
    def close(self) -> None:
        """Close Neo4j connection"""
        if self._connected:
            self.client.close()
            self._connected = False
    
    def build_graph(self, extraction: MeetingExtraction) -> dict:
        """Build knowledge graph from meeting extraction.
        
        Args:
            extraction: MeetingExtraction with all extracted entities
            
        Returns:
            Dictionary with counts of created nodes and relationships
        """
        if not self._connected:
            self.connect()
            
        stats = {
            "meetings": 0,
            "people": 0,
            "topics": 0,
            "decisions": 0,
            "action_items": 0,
            "commitments": 0,
            "relationships": 0
        }
        
        # 1. Create Meeting node
        meeting_title = extraction.meeting_title
        self.client.create_meeting(meeting_title, extraction.meeting_date)
        stats["meetings"] = 1
        
        # 2. Create Person nodes and link to meeting
        for person in extraction.people:
            self.client.create_person(person.name, person.role)
            self.client.link_person_to_meeting(person.name, meeting_title)
            stats["people"] += 1
            stats["relationships"] += 1
            
        # 3. Create Topic nodes and link to meeting
        for topic in extraction.topics:
            self.client.create_topic(topic.name, topic.description)
            self.client.link_meeting_to_topic(meeting_title, topic.name)
            stats["topics"] += 1
            stats["relationships"] += 1
            
        # 4. Create Decision nodes with relationships
        for decision in extraction.decisions:
            self.client.create_decision(decision.description)
            self.client.link_meeting_to_decision(meeting_title, decision.description)
            stats["decisions"] += 1
            stats["relationships"] += 1
            
            # Link decision maker if known
            if decision.made_by:
                self.client.link_person_to_decision(decision.made_by, decision.description)
                stats["relationships"] += 1
                
            # Link to topic if known
            if decision.related_topic:
                self.client.link_decision_to_topic(decision.description, decision.related_topic)
                stats["relationships"] += 1
                
        # 5. Create ActionItem nodes with owner relationships
        for action in extraction.action_items:
            self.client.create_action_item(
                action.description,
                action.deadline,
                action.priority
            )
            self.client.link_meeting_to_action_item(meeting_title, action.description)
            stats["action_items"] += 1
            stats["relationships"] += 1
            
            # Link owner if known
            if action.owner:
                self.client.link_person_to_action_item(action.owner, action.description)
                stats["relationships"] += 1
                
        # 6. Create Commitment nodes with maker relationships
        for commitment in extraction.commitments:
            self.client.create_commitment(commitment.description)
            stats["commitments"] += 1
            
            # Link person who made commitment
            if commitment.made_by:
                self.client.link_person_to_commitment(commitment.made_by, commitment.description)
                stats["relationships"] += 1
                
        return stats
    
    def get_graph_stats(self) -> dict:
        """Get current graph node counts"""
        if not self._connected:
            self.connect()
        return self.client.get_node_counts()


# Test function
if __name__ == "__main__":
    from src.models.entities import (
        Person, Topic, Decision, ActionItem, Commitment, MeetingExtraction
    )
    
    # Create test extraction
    test_extraction = MeetingExtraction(
        meeting_title="Weekly Product Sync",
        meeting_date="2024-01-15",
        people=[
            Person(name="Sarah Chen", role="PM"),
            Person(name="Mike Johnson", role="Engineering Lead"),
            Person(name="Lisa Park", role="Designer")
        ],
        topics=[
            Topic(name="Dashboard Redesign", description="New UI for the dashboard"),
            Topic(name="API Documentation", description="Backend API docs")
        ],
        decisions=[
            Decision(
                description="Dark mode as default theme",
                made_by="Sarah Chen",
                related_topic="Dashboard Redesign"
            )
        ],
        action_items=[
            ActionItem(
                description="Finish API documentation",
                owner="Mike Johnson",
                deadline="Friday"
            ),
            ActionItem(
                description="Schedule stakeholder review meeting",
                owner="Lisa Park"
            )
        ],
        commitments=[
            Commitment(
                description="Deliver final designs by next Wednesday",
                made_by="Lisa Park"
            )
        ]
    )
    
    print("Testing GraphBuilderAgent...")
    agent = GraphBuilderAgent()
    
    try:
        stats = agent.build_graph(test_extraction)
        print("\n" + "="*50)
        print("Graph Build Results:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("="*50)
        
        print("\nCurrent graph stats:")
        print(agent.get_graph_stats())
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Neo4j is running and credentials are correct in .env")
    finally:
        agent.close()
