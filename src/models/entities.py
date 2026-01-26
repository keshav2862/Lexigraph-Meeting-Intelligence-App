"""Pydantic models for meeting entity extraction.

These models define the structure of entities extracted from meeting transcripts.
Using Pydantic ensures type safety and enables structured LLM output.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Person(BaseModel):
    """A person mentioned or participating in the meeting"""
    name: str = Field(description="Full name of the person")
    role: Optional[str] = Field(default=None, description="Job title or role if mentioned")
    
    
class Topic(BaseModel):
    """A topic or subject discussed in the meeting"""
    name: str = Field(description="Name or title of the topic")
    description: Optional[str] = Field(default=None, description="Brief description of what was discussed")
    

class Decision(BaseModel):
    """A decision made during the meeting"""
    description: str = Field(description="What was decided")
    made_by: Optional[str] = Field(default=None, description="Who made or announced the decision")
    related_topic: Optional[str] = Field(default=None, description="Topic this decision relates to")


class ActionItem(BaseModel):
    """A task or action item assigned during the meeting"""
    description: str = Field(description="What needs to be done")
    owner: Optional[str] = Field(default=None, description="Person responsible for this action")
    deadline: Optional[str] = Field(default=None, description="Due date if mentioned")
    priority: Optional[str] = Field(default=None, description="Priority level if mentioned (high/medium/low)")


class Commitment(BaseModel):
    """A commitment or promise made by someone during the meeting"""
    description: str = Field(description="What was promised or committed to")
    made_by: str = Field(description="Who made the commitment")
    to_whom: Optional[str] = Field(default=None, description="Who the commitment was made to")


class MeetingExtraction(BaseModel):
    """Complete extraction from a meeting transcript"""
    meeting_title: str = Field(description="A brief title summarizing the meeting")
    meeting_date: Optional[str] = Field(default=None, description="Date of meeting if mentioned (YYYY-MM-DD format)")
    
    people: List[Person] = Field(default_factory=list, description="People mentioned in the meeting")
    topics: List[Topic] = Field(default_factory=list, description="Topics discussed")
    decisions: List[Decision] = Field(default_factory=list, description="Decisions made")
    action_items: List[ActionItem] = Field(default_factory=list, description="Action items assigned")
    commitments: List[Commitment] = Field(default_factory=list, description="Commitments or promises made")
    
    @property
    def summary(self) -> str:
        """Generate a quick summary of extracted entities"""
        return (
            f"Meeting: {self.meeting_title}\n"
            f"  - {len(self.people)} people\n"
            f"  - {len(self.topics)} topics\n"
            f"  - {len(self.decisions)} decisions\n"
            f"  - {len(self.action_items)} action items\n"
            f"  - {len(self.commitments)} commitments"
        )
