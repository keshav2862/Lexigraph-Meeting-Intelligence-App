"""Configuration settings for MeetingMind"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration loaded from environment"""
    
    # Groq API
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # Neo4j Connection
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    
    # Model Selection (using 70B for better accuracy and larger context)
    EXTRACTION_MODEL: str = os.getenv("EXTRACTION_MODEL", "llama-3.3-70b-versatile")
    QUERY_MODEL: str = os.getenv("QUERY_MODEL", "llama-3.3-70b-versatile")
    
    # Token limits for cost control
    MAX_EXTRACTION_TOKENS: int = 4000
    MAX_QUERY_TOKENS: int = 2000
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration is present"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required. Set it in .env file.")
        if not cls.NEO4J_PASSWORD:
            raise ValueError("NEO4J_PASSWORD is required. Set it in .env file.")
        return True


config = Config()
