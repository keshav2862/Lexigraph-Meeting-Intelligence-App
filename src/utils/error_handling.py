"""Error Handling Utilities for Lexigraph

Provides consistent error handling and user-friendly error messages.
"""

from typing import Optional, Callable, Any
from functools import wraps
import traceback


class LexigraphError(Exception):
    """Base exception for Lexigraph errors"""
    
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def user_message(self) -> str:
        """Return a user-friendly error message"""
        return self.message


class ConnectionError(LexigraphError):
    """Error connecting to external service (Neo4j, Groq)"""
    pass


class ExtractionError(LexigraphError):
    """Error during entity extraction"""
    pass


class QueryError(LexigraphError):
    """Error during query execution"""
    pass


class GraphError(LexigraphError):
    """Error during graph operations"""
    pass


# User-friendly error messages mapping
ERROR_MESSAGES = {
    "GROQ_API_KEY": "Groq API key not configured. Please add GROQ_API_KEY to your .env file.",
    "NEO4J_CONNECTION": "Cannot connect to Neo4j. Make sure Neo4j Desktop is running.",
    "NEO4J_AUTH": "Neo4j authentication failed. Check your username and password in .env.",
    "RATE_LIMIT": "API rate limit exceeded. Please wait a moment and try again.",
    "TOKEN_LIMIT": "Transcript is too long. Try with a shorter meeting transcript.",
    "INVALID_CYPHER": "Could not generate a valid query for your question. Try rephrasing.",
    "EMPTY_GRAPH": "No data in the knowledge graph. Process some meeting transcripts first.",
    "NETWORK": "Network error. Check your internet connection.",
}


def get_user_friendly_error(error: Exception) -> str:
    """
    Convert exception to user-friendly message.
    
    Args:
        error: The exception that occurred
        
    Returns:
        User-friendly error message
    """
    error_str = str(error).lower()
    
    # Check for known error patterns
    if "groq_api_key" in error_str or "api_key" in error_str:
        return ERROR_MESSAGES["GROQ_API_KEY"]
    
    if "connection refused" in error_str or "failed to establish" in error_str:
        return ERROR_MESSAGES["NEO4J_CONNECTION"]
    
    if "authentication" in error_str or "unauthorized" in error_str:
        return ERROR_MESSAGES["NEO4J_AUTH"]
    
    if "rate limit" in error_str or "429" in error_str:
        return ERROR_MESSAGES["RATE_LIMIT"]
    
    if "token" in error_str and ("limit" in error_str or "exceed" in error_str):
        return ERROR_MESSAGES["TOKEN_LIMIT"]
    
    if "syntax error" in error_str or "cypher" in error_str:
        return ERROR_MESSAGES["INVALID_CYPHER"]
    
    if "no data" in error_str or "empty" in error_str:
        return ERROR_MESSAGES["EMPTY_GRAPH"]
    
    if "network" in error_str or "timeout" in error_str:
        return ERROR_MESSAGES["NETWORK"]
    
    # Default: return original error message
    return f"An error occurred: {str(error)}"


def safe_execute(default_return: Any = None, error_message: Optional[str] = None):
    """
    Decorator for safe execution with error handling.
    
    Args:
        default_return: Value to return on error
        error_message: Custom error message prefix
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                user_msg = get_user_friendly_error(e)
                if error_message:
                    user_msg = f"{error_message}: {user_msg}"
                print(f"Error in {func.__name__}: {user_msg}")
                print(f"Details: {traceback.format_exc()}")
                return default_return
        return wrapper
    return decorator


def handle_streamlit_error(error: Exception, show_details: bool = False) -> dict:
    """
    Handle error for Streamlit UI display.
    
    Args:
        error: The exception that occurred
        show_details: Whether to include technical details
        
    Returns:
        Dict with 'message' and optionally 'details'
    """
    result = {
        "message": get_user_friendly_error(error),
        "success": False
    }
    
    if show_details:
        result["details"] = str(error)
        result["traceback"] = traceback.format_exc()
    
    return result
