"""Analyzer Agent - Cross-meeting analysis and insights

Provides:
- Deadline tracking with status categorization
- Cross-meeting topic analysis
- Conflict detection between decisions
- Person participation insights
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.graph.neo4j_client import Neo4jClient
from src.config import config


class AnalyzerAgent:
    """Agent for advanced graph analysis and insights"""
    
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.client = neo4j_client or Neo4jClient()
        self._connected = False
        
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.QUERY_MODEL,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Conflict detection prompt
        self.conflict_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing business decisions for conflicts or contradictions.
Given a list of decisions from meetings, identify any that might contradict or conflict with each other.

For each conflict found, explain:
1. The two conflicting decisions
2. Why they conflict
3. Severity: LOW (minor), MEDIUM (needs resolution), HIGH (blocking)

If no conflicts exist, say "No conflicts detected."
Be concise but thorough."""),
            ("human", "Analyze these decisions for conflicts:\n\n{decisions}")
        ])
        self.conflict_chain = self.conflict_prompt | self.llm | StrOutputParser()
    
    def connect(self) -> None:
        """Connect to Neo4j"""
        if not self._connected:
            self.client.connect()
            self._connected = True
    
    def close(self) -> None:
        """Close connection"""
        if self._connected:
            self.client.close()
            self._connected = False
    
    def get_deadline_status(self) -> Dict[str, List[Dict]]:
        """
        Get action items categorized by deadline status.
        
        Returns:
            Dict with 'overdue', 'due_soon', 'upcoming', 'no_deadline' lists
        """
        if not self._connected:
            self.connect()
        
        query = """
        MATCH (a:ActionItem)
        OPTIONAL MATCH (p:Person)-[:OWNS]->(a)
        OPTIONAL MATCH (m:Meeting)-[:CONTAINS]->(a)
        RETURN a.description as task, 
               a.deadline as deadline, 
               a.status as status,
               a.priority as priority,
               p.name as owner,
               m.title as meeting
        ORDER BY a.deadline
        """
        
        results = self.client.run_query(query)
        
        categorized = {
            "overdue": [],
            "due_soon": [],  # Within 2 days
            "upcoming": [],  # Future
            "no_deadline": []
        }
        
        today = datetime.now()
        
        for item in results:
            deadline_str = item.get("deadline")
            item_data = {
                "task": item.get("task"),
                "deadline": deadline_str,
                "owner": item.get("owner"),
                "status": item.get("status", "pending"),
                "priority": item.get("priority"),
                "meeting": item.get("meeting")
            }
            
            if not deadline_str:
                categorized["no_deadline"].append(item_data)
            else:
                # Try to parse deadline
                deadline_date = self._parse_deadline(deadline_str, today)
                if deadline_date:
                    days_until = (deadline_date - today).days
                    if days_until < 0:
                        categorized["overdue"].append(item_data)
                    elif days_until <= 2:
                        categorized["due_soon"].append(item_data)
                    else:
                        categorized["upcoming"].append(item_data)
                else:
                    categorized["upcoming"].append(item_data)
        
        return categorized
    
    def _parse_deadline(self, deadline_str: str, reference_date: datetime) -> Optional[datetime]:
        """Parse relative deadlines like 'Friday', 'next week', 'EOD'"""
        deadline_lower = deadline_str.lower().strip()
        
        # Day of week mapping
        days = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }
        
        # Check for day of week
        for day_name, day_num in days.items():
            if day_name in deadline_lower:
                current_day = reference_date.weekday()
                days_ahead = day_num - current_day
                if days_ahead <= 0:
                    days_ahead += 7
                return reference_date + timedelta(days=days_ahead)
        
        # Check for relative terms
        if "today" in deadline_lower or "eod" in deadline_lower:
            return reference_date
        elif "tomorrow" in deadline_lower:
            return reference_date + timedelta(days=1)
        elif "next week" in deadline_lower:
            return reference_date + timedelta(days=7)
        elif "end of week" in deadline_lower:
            days_until_friday = (4 - reference_date.weekday()) % 7
            return reference_date + timedelta(days=days_until_friday)
        
        return None
    
    def get_topic_trends(self) -> List[Dict[str, Any]]:
        """
        Analyze topic frequency across meetings.
        
        Returns:
            List of topics with their meeting count and related entities
        """
        if not self._connected:
            self.connect()
        
        query = """
        MATCH (t:Topic)<-[:DISCUSSED]-(m:Meeting)
        OPTIONAL MATCH (d:Decision)-[:ABOUT]->(t)
        WITH t, collect(DISTINCT m.title) as meetings, collect(DISTINCT d.description) as decisions
        RETURN t.name as topic,
               t.description as description,
               size(meetings) as meeting_count,
               meetings,
               decisions
        ORDER BY meeting_count DESC
        """
        
        return self.client.run_query(query)
    
    def get_person_insights(self) -> List[Dict[str, Any]]:
        """
        Get per-person analytics.
        
        Returns:
            List of people with their activity metrics
        """
        if not self._connected:
            self.connect()
        
        query = """
        MATCH (p:Person)
        OPTIONAL MATCH (p)-[:ATTENDED]->(m:Meeting)
        OPTIONAL MATCH (p)-[:OWNS]->(a:ActionItem)
        OPTIONAL MATCH (p)-[:MADE]->(d:Decision)
        OPTIONAL MATCH (p)-[:COMMITTED]->(c:Commitment)
        WITH p, 
             count(DISTINCT m) as meetings_attended,
             count(DISTINCT a) as action_items,
             count(DISTINCT d) as decisions_made,
             count(DISTINCT c) as commitments
        RETURN p.name as name,
               p.role as role,
               meetings_attended,
               action_items,
               decisions_made,
               commitments,
               (action_items + decisions_made + commitments) as engagement_score
        ORDER BY engagement_score DESC
        """
        
        return self.client.run_query(query)
    
    def detect_conflicts(self) -> str:
        """
        Use LLM to detect conflicts between decisions.
        
        Returns:
            Analysis of conflicts found
        """
        if not self._connected:
            self.connect()
        
        # Get all decisions with context
        query = """
        MATCH (d:Decision)
        OPTIONAL MATCH (p:Person)-[:MADE]->(d)
        OPTIONAL MATCH (m:Meeting)-[:CONTAINS]->(d)
        OPTIONAL MATCH (d)-[:ABOUT]->(t:Topic)
        RETURN d.description as decision,
               p.name as made_by,
               m.title as meeting,
               m.date as date,
               t.name as topic
        ORDER BY m.date
        """
        
        results = self.client.run_query(query)
        
        if not results:
            return "No decisions found in the knowledge graph."
        
        # Format decisions for LLM
        formatted_decisions = []
        for i, d in enumerate(results, 1):
            decision_text = f"{i}. \"{d.get('decision')}\""
            if d.get('made_by'):
                decision_text += f" (by {d.get('made_by')})"
            if d.get('meeting'):
                decision_text += f" [Meeting: {d.get('meeting')}]"
            if d.get('topic'):
                decision_text += f" [Topic: {d.get('topic')}]"
            formatted_decisions.append(decision_text)
        
        decisions_text = "\n".join(formatted_decisions)
        
        # Run conflict detection
        analysis = self.conflict_chain.invoke({"decisions": decisions_text})
        
        return analysis
    
    def get_meeting_comparison(self, meeting1: str, meeting2: str) -> Dict[str, Any]:
        """
        Compare two meetings for common topics, different decisions, etc.
        """
        if not self._connected:
            self.connect()
        
        query = """
        MATCH (m1:Meeting), (m2:Meeting)
        WHERE toLower(m1.title) CONTAINS toLower($meeting1) 
          AND toLower(m2.title) CONTAINS toLower($meeting2)
        OPTIONAL MATCH (m1)-[:DISCUSSED]->(t1:Topic)
        OPTIONAL MATCH (m2)-[:DISCUSSED]->(t2:Topic)
        OPTIONAL MATCH (m1)-[:CONTAINS]->(d1:Decision)
        OPTIONAL MATCH (m2)-[:CONTAINS]->(d2:Decision)
        RETURN m1.title as meeting1_title,
               m2.title as meeting2_title,
               collect(DISTINCT t1.name) as meeting1_topics,
               collect(DISTINCT t2.name) as meeting2_topics,
               collect(DISTINCT d1.description) as meeting1_decisions,
               collect(DISTINCT d2.description) as meeting2_decisions
        """
        
        results = self.client.run_query(query, {"meeting1": meeting1, "meeting2": meeting2})
        
        if not results:
            return {"error": "Meetings not found"}
        
        r = results[0]
        
        # Find common and unique topics
        topics1 = set(r.get("meeting1_topics", []))
        topics2 = set(r.get("meeting2_topics", []))
        
        return {
            "meeting1": r.get("meeting1_title"),
            "meeting2": r.get("meeting2_title"),
            "common_topics": list(topics1 & topics2),
            "unique_to_meeting1": list(topics1 - topics2),
            "unique_to_meeting2": list(topics2 - topics1),
            "meeting1_decisions": r.get("meeting1_decisions", []),
            "meeting2_decisions": r.get("meeting2_decisions", [])
        }
