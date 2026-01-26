"""Entity Extractor Agent - Extracts structured entities from meeting transcripts.

Uses Groq LLM with structured output to extract:
- People, Topics, Decisions, Action Items, Commitments

This agent demonstrates:
- Structured LLM output with Pydantic
- Cost-optimized prompting
- Error handling with retries
"""

from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import config
from src.models.entities import MeetingExtraction


# System prompt optimized for extraction accuracy
EXTRACTION_SYSTEM_PROMPT = """You are an expert meeting analyst. Extract structured information from meeting transcripts.

Be thorough but precise:
- Extract ALL people mentioned by name
- Identify distinct topics discussed  
- Capture decisions that were finalized
- Note action items with owners if mentioned
- Record commitments/promises people made

If information is unclear or not present, omit it rather than guessing."""

EXTRACTION_USER_PROMPT = """Analyze this meeting transcript and extract all entities:

TRANSCRIPT:
{transcript}

Extract the meeting title, date (if mentioned), and all people, topics, decisions, action items, and commitments."""


class ExtractorAgent:
    """Agent that extracts structured entities from meeting transcripts"""
    
    def __init__(self, model: Optional[str] = None):
        self.model_name = model or config.EXTRACTION_MODEL
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=self.model_name,
            temperature=0,  # Deterministic for extraction
            max_tokens=config.MAX_EXTRACTION_TOKENS
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", EXTRACTION_SYSTEM_PROMPT),
            ("human", EXTRACTION_USER_PROMPT)
        ])
        
        # Create chain with structured output
        self.chain = self.prompt | self.llm.with_structured_output(MeetingExtraction)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def extract(self, transcript: str) -> MeetingExtraction:
        """Extract entities from a meeting transcript.
        
        Args:
            transcript: The meeting transcript text
            
        Returns:
            MeetingExtraction with all extracted entities
        """
        result = self.chain.invoke({"transcript": transcript})
        return result
    
    def extract_safe(self, transcript: str) -> Optional[MeetingExtraction]:
        """Extract entities with error handling, returns None on failure."""
        try:
            return self.extract(transcript)
        except Exception as e:
            print(f"Extraction failed: {e}")
            return None


# Test function
if __name__ == "__main__":
    sample_transcript = """
    Weekly Product Sync - January 15, 2024
    
    Attendees: Sarah Chen (PM), Mike Johnson (Engineering Lead), Lisa Park (Designer)
    
    Sarah: Let's start with the dashboard redesign. Mike, where are we on the backend?
    
    Mike: We've completed the API endpoints. I'll have the documentation ready by Friday.
    
    Lisa: The new mockups are done. I'm waiting for feedback from the stakeholders.
    
    Sarah: Great. Let's make a decision - we'll go with the dark mode as the default theme.
    Mike agreed to prioritize the performance optimization sprint.
    
    Action items:
    - Mike: Finish API documentation by Friday
    - Lisa: Schedule stakeholder review meeting
    - Sarah: Update the roadmap with new timeline
    
    Lisa committed to delivering the final designs by next Wednesday.
    """
    
    print("Testing ExtractorAgent...")
    agent = ExtractorAgent()
    
    try:
        result = agent.extract(sample_transcript)
        print("\n" + "="*50)
        print(result.summary)
        print("="*50)
        print(f"\nPeople: {[p.name for p in result.people]}")
        print(f"Topics: {[t.name for t in result.topics]}")
        print(f"Decisions: {[d.description for d in result.decisions]}")
        print(f"Action Items: {[a.description for a in result.action_items]}")
        print(f"Commitments: {[c.description for c in result.commitments]}")
    except Exception as e:
        print(f"Error: {e}")
