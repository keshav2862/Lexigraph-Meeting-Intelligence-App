"""Unit Tests for Lexigraph Agents

Tests for entity extraction, graph building, and query functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.models.entities import (
    MeetingExtraction, Person, Topic, Decision, ActionItem, Commitment
)


class TestMeetingExtraction:
    """Tests for MeetingExtraction Pydantic model"""
    
    def test_meeting_extraction_empty(self):
        """Test creating extraction with minimal data"""
        extraction = MeetingExtraction(
            meeting_title="Test Meeting",
            people=[],
            topics=[],
            decisions=[],
            action_items=[],
            commitments=[]
        )
        assert extraction.meeting_title == "Test Meeting"
        assert len(extraction.people) == 0
    
    def test_meeting_extraction_with_entities(self):
        """Test creating extraction with full entities"""
        extraction = MeetingExtraction(
            meeting_title="Product Sync",
            meeting_date="2024-01-15",
            people=[
                Person(name="Mike Johnson", role="Engineering Lead"),
                Person(name="Sarah Chen", role="PM")
            ],
            topics=[
                Topic(name="Dashboard", description="New dashboard design")
            ],
            decisions=[
                Decision(description="Use dark mode as default", made_by="Sarah")
            ],
            action_items=[
                ActionItem(
                    description="Complete Redis migration",
                    owner="Mike",
                    deadline="Friday",
                    priority="high"
                )
            ],
            commitments=[
                Commitment(description="Deliver designs by Wednesday", made_by="Lisa")
            ]
        )
        
        assert len(extraction.people) == 2
        assert extraction.people[0].name == "Mike Johnson"
        assert len(extraction.action_items) == 1
        assert extraction.action_items[0].priority == "high"
    
    def test_person_model(self):
        """Test Person model"""
        person = Person(name="John Doe", role="Developer")
        assert person.name == "John Doe"
        assert person.role == "Developer"
    
    def test_action_item_optional_fields(self):
        """Test ActionItem with optional fields"""
        action = ActionItem(description="Do something")
        assert action.description == "Do something"
        assert action.owner is None
        assert action.deadline is None
        assert action.priority is None


class TestExtractorAgent:
    """Tests for ExtractorAgent"""
    
    @patch('src.agents.extractor.ChatGroq')
    def test_extractor_initialization(self, mock_groq):
        """Test agent initializes correctly"""
        from src.agents.extractor import ExtractorAgent
        
        agent = ExtractorAgent()
        assert agent.llm is not None
        assert agent.prompt is not None
    
    @patch('src.agents.extractor.ChatGroq')
    def test_extract_safe_returns_none_on_error(self, mock_groq):
        """Test extract_safe handles errors gracefully"""
        from src.agents.extractor import ExtractorAgent
        
        mock_groq.return_value.with_structured_output.return_value = Mock(
            side_effect=Exception("API Error")
        )
        
        agent = ExtractorAgent()
        agent.chain = Mock(side_effect=Exception("API Error"))
        
        result = agent.extract_safe("Some transcript")
        assert result is None


class TestGraphBuilder:
    """Tests for GraphBuilderAgent"""
    
    def test_graph_stats_empty(self):
        """Test getting stats from empty graph"""
        from src.agents.graph_builder import GraphBuilderAgent
        
        mock_client = Mock()
        mock_client.run_query.return_value = []
        
        builder = GraphBuilderAgent()
        builder.client = mock_client
        builder._connected = True
        
        stats = builder.get_graph_stats()
        # Should return empty dict or default values
        assert isinstance(stats, dict)


class TestQueryAgent:
    """Tests for QueryAgent"""
    
    def test_chat_history_management(self):
        """Test chat history is properly managed"""
        from src.agents.query_agent import QueryAgent
        
        with patch('src.agents.query_agent.ChatGroq'):
            agent = QueryAgent()
            
            # Initially empty
            assert len(agent.chat_history) == 0
            
            # Clear should work on empty
            agent.clear_history()
            assert len(agent.chat_history) == 0
    
    def test_format_chat_history_empty(self):
        """Test formatting empty chat history"""
        from src.agents.query_agent import QueryAgent
        
        with patch('src.agents.query_agent.ChatGroq'):
            agent = QueryAgent()
            
            formatted = agent._format_chat_history()
            assert formatted == ""
    
    def test_format_chat_history_with_messages(self):
        """Test formatting chat history with messages"""
        from src.agents.query_agent import QueryAgent
        
        with patch('src.agents.query_agent.ChatGroq'):
            agent = QueryAgent()
            agent.chat_history = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ]
            
            formatted = agent._format_chat_history()
            assert "Human: Hello" in formatted
            assert "Assistant: Hi there" in formatted


class TestAnalyzerAgent:
    """Tests for AnalyzerAgent"""
    
    def test_parse_deadline_friday(self):
        """Test parsing 'Friday' deadline"""
        from src.agents.analyzer import AnalyzerAgent
        from datetime import datetime
        
        with patch('src.agents.analyzer.ChatGroq'):
            agent = AnalyzerAgent()
            
            reference = datetime(2024, 1, 15)  # Monday
            result = agent._parse_deadline("Friday", reference)
            
            assert result is not None
            assert result.weekday() == 4  # Friday
    
    def test_parse_deadline_tomorrow(self):
        """Test parsing 'tomorrow' deadline"""
        from src.agents.analyzer import AnalyzerAgent
        from datetime import datetime, timedelta
        
        with patch('src.agents.analyzer.ChatGroq'):
            agent = AnalyzerAgent()
            
            reference = datetime(2024, 1, 15)
            result = agent._parse_deadline("tomorrow", reference)
            
            expected = reference + timedelta(days=1)
            assert result == expected
    
    def test_parse_deadline_today(self):
        """Test parsing 'today' and 'EOD' deadlines"""
        from src.agents.analyzer import AnalyzerAgent
        from datetime import datetime
        
        with patch('src.agents.analyzer.ChatGroq'):
            agent = AnalyzerAgent()
            
            reference = datetime(2024, 1, 15)
            
            result_today = agent._parse_deadline("today", reference)
            result_eod = agent._parse_deadline("EOD", reference)
            
            assert result_today == reference
            assert result_eod == reference


class TestNeo4jClient:
    """Tests for Neo4jClient"""
    
    def test_client_initialization(self):
        """Test client initializes with config"""
        from src.graph.neo4j_client import Neo4jClient
        
        client = Neo4jClient()
        assert client.uri is not None
        assert client.username is not None
    
    def test_client_not_connected_by_default(self):
        """Test client is not connected initially"""
        from src.graph.neo4j_client import Neo4jClient
        
        client = Neo4jClient()
        # Should have driver as None before connect()
        assert client.driver is None


# Run tests with: pytest tests/test_agents.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
