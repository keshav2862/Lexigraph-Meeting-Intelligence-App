"""Summary Agent - Auto-generate meeting summaries

Uses LLM to create executive summaries from graph data.
"""

from typing import Dict, List, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.graph.neo4j_client import Neo4jClient
from src.config import config


class SummaryAgent:
    """Agent that generates meeting summaries from graph data"""
    
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.client = neo4j_client or Neo4jClient()
        self._connected = False
        
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.QUERY_MODEL,
            temperature=0.5,
            max_tokens=1500
        )
        
        # Meeting summary prompt
        self.meeting_summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating concise, professional meeting summaries.
Given meeting data from a knowledge graph, create an executive summary with:

1. **Meeting Overview** - Title, date, attendees
2. **Key Topics Discussed** - Main subjects covered
3. **Decisions Made** - Important decisions with who made them
4. **Action Items** - Tasks with owners and deadlines
5. **Commitments** - Promises made by team members

Format with markdown. Be concise but comprehensive.
Use bullet points for lists. Include all relevant details."""),
            ("human", "Create an executive summary for this meeting:\n\n{meeting_data}")
        ])
        self.meeting_summary_chain = self.meeting_summary_prompt | self.llm | StrOutputParser()
        
        # Cross-meeting summary prompt  
        self.cross_meeting_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating executive summaries across multiple meetings.
Given data from several meetings, create a comprehensive overview with:

1. **Meetings Covered** - List of meetings analyzed
2. **Recurring Topics** - Themes that appear across meetings
3. **All Outstanding Action Items** - Pending tasks by owner
4. **Key Decisions** - Important decisions made across meetings
5. **Unfulfilled Commitments** - Promises that may need follow-up
6. **Recommendations** - Suggested next steps

Format with markdown. Be strategic and actionable."""),
            ("human", "Create a cross-meeting summary from this data:\n\n{all_data}")
        ])
        self.cross_meeting_chain = self.cross_meeting_prompt | self.llm | StrOutputParser()
    
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
    
    def generate_meeting_summary(self, meeting_title: str) -> str:
        """
        Generate a summary for a specific meeting.
        
        Args:
            meeting_title: Title or partial title of the meeting
            
        Returns:
            Markdown formatted summary
        """
        if not self._connected:
            self.connect()
        
        # Query all meeting data
        query = """
        MATCH (m:Meeting)
        WHERE toLower(m.title) CONTAINS toLower($title)
        OPTIONAL MATCH (p:Person)-[:ATTENDED]->(m)
        OPTIONAL MATCH (m)-[:DISCUSSED]->(t:Topic)
        OPTIONAL MATCH (m)-[:CONTAINS]->(d:Decision)
        OPTIONAL MATCH (m)-[:CONTAINS]->(a:ActionItem)
        OPTIONAL MATCH (owner:Person)-[:OWNS]->(a)
        OPTIONAL MATCH (maker:Person)-[:MADE]->(d)
        OPTIONAL MATCH (committer:Person)-[:COMMITTED]->(c:Commitment)
        WHERE (m)-[:CONTAINS]->() OR (p)-[:ATTENDED]->(m)
        
        WITH m, 
             collect(DISTINCT {name: p.name, role: p.role}) as attendees,
             collect(DISTINCT {name: t.name, description: t.description}) as topics,
             collect(DISTINCT {description: d.description, made_by: maker.name}) as decisions,
             collect(DISTINCT {description: a.description, owner: owner.name, deadline: a.deadline, priority: a.priority}) as actions,
             collect(DISTINCT {description: c.description, made_by: committer.name}) as commitments
        
        RETURN m.title as title,
               m.date as date,
               attendees,
               topics,
               decisions,
               actions,
               commitments
        LIMIT 1
        """
        
        results = self.client.run_query(query, {"title": meeting_title})
        
        if not results:
            return f"No meeting found matching '{meeting_title}'"
        
        meeting = results[0]
        
        # Format data for LLM
        meeting_data = self._format_meeting_data(meeting)
        
        # Generate summary
        summary = self.meeting_summary_chain.invoke({"meeting_data": meeting_data})
        
        return summary
    
    def generate_cross_meeting_summary(self) -> str:
        """
        Generate a summary across all meetings.
        
        Returns:
            Markdown formatted cross-meeting summary
        """
        if not self._connected:
            self.connect()
        
        # Query all meetings and their data
        query = """
        MATCH (m:Meeting)
        OPTIONAL MATCH (p:Person)-[:ATTENDED]->(m)
        OPTIONAL MATCH (m)-[:DISCUSSED]->(t:Topic)
        OPTIONAL MATCH (m)-[:CONTAINS]->(d:Decision)
        OPTIONAL MATCH (m)-[:CONTAINS]->(a:ActionItem)
        OPTIONAL MATCH (owner:Person)-[:OWNS]->(a)
        OPTIONAL MATCH (c:Commitment)<-[:COMMITTED]-(committer:Person)
        
        WITH m,
             collect(DISTINCT p.name) as attendees,
             collect(DISTINCT t.name) as topics,
             collect(DISTINCT d.description) as decisions,
             collect(DISTINCT {task: a.description, owner: owner.name, deadline: a.deadline, status: a.status}) as actions
        
        RETURN m.title as meeting,
               m.date as date,
               attendees,
               topics,
               decisions,
               actions
        ORDER BY m.date
        """
        
        results = self.client.run_query(query)
        
        if not results:
            return "No meetings found in the knowledge graph."
        
        # Get all commitments separately
        commitment_query = """
        MATCH (p:Person)-[:COMMITTED]->(c:Commitment)
        RETURN p.name as person, c.description as commitment
        """
        commitments = self.client.run_query(commitment_query)
        
        # Format all data
        all_data = self._format_cross_meeting_data(results, commitments)
        
        # Generate summary
        summary = self.cross_meeting_chain.invoke({"all_data": all_data})
        
        return summary
    
    def _format_meeting_data(self, meeting: Dict) -> str:
        """Format meeting data for LLM consumption"""
        lines = []
        
        lines.append(f"MEETING: {meeting.get('title', 'Unknown')}")
        if meeting.get('date'):
            lines.append(f"DATE: {meeting.get('date')}")
        
        # Attendees
        attendees = [a.get('name') for a in meeting.get('attendees', []) if a.get('name')]
        if attendees:
            lines.append(f"\nATTENDEES: {', '.join(attendees)}")
        
        # Topics
        topics = meeting.get('topics', [])
        if topics:
            lines.append("\nTOPICS DISCUSSED:")
            for t in topics:
                if t.get('name'):
                    desc = f" - {t.get('description')}" if t.get('description') else ""
                    lines.append(f"  - {t.get('name')}{desc}")
        
        # Decisions
        decisions = meeting.get('decisions', [])
        if decisions:
            lines.append("\nDECISIONS MADE:")
            for d in decisions:
                if d.get('description'):
                    by = f" (by {d.get('made_by')})" if d.get('made_by') else ""
                    lines.append(f"  - {d.get('description')}{by}")
        
        # Action items
        actions = meeting.get('actions', [])
        if actions:
            lines.append("\nACTION ITEMS:")
            for a in actions:
                if a.get('description'):
                    owner = f" [{a.get('owner')}]" if a.get('owner') else ""
                    deadline = f" Due: {a.get('deadline')}" if a.get('deadline') else ""
                    lines.append(f"  - {a.get('description')}{owner}{deadline}")
        
        # Commitments
        commitments = meeting.get('commitments', [])
        if commitments:
            lines.append("\nCOMMITMENTS:")
            for c in commitments:
                if c.get('description'):
                    by = f" [{c.get('made_by')}]" if c.get('made_by') else ""
                    lines.append(f"  - {c.get('description')}{by}")
        
        return "\n".join(lines)
    
    def _format_cross_meeting_data(self, meetings: List[Dict], commitments: List[Dict]) -> str:
        """Format cross-meeting data for LLM consumption"""
        lines = []
        
        lines.append("=== MEETINGS OVERVIEW ===\n")
        
        all_topics = set()
        all_actions = []
        
        for m in meetings:
            lines.append(f"MEETING: {m.get('meeting', 'Unknown')}")
            if m.get('date'):
                lines.append(f"Date: {m.get('date')}")
            
            attendees = [a for a in m.get('attendees', []) if a]
            if attendees:
                lines.append(f"Attendees: {', '.join(attendees)}")
            
            topics = [t for t in m.get('topics', []) if t]
            if topics:
                lines.append(f"Topics: {', '.join(topics)}")
                all_topics.update(topics)
            
            decisions = [d for d in m.get('decisions', []) if d]
            if decisions:
                lines.append(f"Decisions: {'; '.join(decisions)}")
            
            actions = m.get('actions', [])
            for a in actions:
                if a.get('task'):
                    all_actions.append(a)
            
            lines.append("")
        
        # Summary section
        lines.append("=== ALL ACTION ITEMS ===")
        for a in all_actions:
            owner = a.get('owner', 'Unassigned')
            status = a.get('status', 'pending')
            deadline = f" (Due: {a.get('deadline')})" if a.get('deadline') else ""
            lines.append(f"- [{status}] {a.get('task')} - Owner: {owner}{deadline}")
        
        lines.append("\n=== ALL COMMITMENTS ===")
        for c in commitments:
            lines.append(f"- {c.get('person')}: {c.get('commitment')}")
        
        lines.append(f"\n=== RECURRING TOPICS ===")
        lines.append(f"Topics appearing across meetings: {', '.join(all_topics)}")
        
        return "\n".join(lines)
    
    def export_summary_markdown(self, summary: str, filename: str = "meeting_summary.md") -> str:
        """
        Export summary to a markdown file.
        
        Args:
            summary: The summary text
            filename: Output filename
            
        Returns:
            Path to the saved file
        """
        import os
        from datetime import datetime
        
        # Add header with timestamp
        header = f"""---
Generated by Lexigraph
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
---

"""
        
        content = header + summary
        
        # Save to data directory
        filepath = os.path.join("data", filename)
        os.makedirs("data", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
