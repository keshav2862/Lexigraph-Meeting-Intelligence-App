"""Query Agent - Natural language queries over the knowledge graph.

Translates natural language questions to Cypher queries and generates answers.

This agent demonstrates:
- Text-to-Cypher generation
- Few-shot prompting
- RAG-style answer synthesis
"""

from typing import Optional, List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import config
from src.graph.neo4j_client import Neo4jClient


# Few-shot examples for Cypher generation
# Using toLower() for case-insensitive matching
CYPHER_EXAMPLES = """
Example questions and their Cypher queries:

Q: What decisions were made?
Cypher: MATCH (d:Decision) RETURN d.description as decision

Q: What action items does Mike own?
Cypher: MATCH (p:Person)-[:OWNS]->(a:ActionItem) WHERE toLower(p.name) CONTAINS toLower('Mike') RETURN a.description as action_item, a.deadline as deadline

Q: Who attended the weekly sync?
Cypher: MATCH (p:Person)-[:ATTENDED]->(m:Meeting) WHERE toLower(m.title) CONTAINS toLower('weekly') RETURN p.name as person, p.role as role

Q: What topics were discussed in the Q3 planning meeting?
Cypher: MATCH (m:Meeting)-[:DISCUSSED]->(t:Topic) WHERE toLower(m.title) CONTAINS toLower('q3') RETURN m.title as meeting, t.name as topic, t.description as description

Q: What commitments did Lisa make?
Cypher: MATCH (p:Person)-[:COMMITTED]->(c:Commitment) WHERE toLower(p.name) CONTAINS toLower('lisa') RETURN c.description as commitment

Q: Show me all action items with their owners
Cypher: MATCH (a:ActionItem)<-[:OWNS]-(p:Person) RETURN a.description as action_item, p.name as owner, a.deadline as deadline, a.status as status

Q: What decisions were made about the dashboard?
Cypher: MATCH (d:Decision)-[:ABOUT]->(t:Topic) WHERE toLower(t.name) CONTAINS toLower('dashboard') RETURN d.description as decision, t.name as topic
"""

# NOT using f-string to preserve any special characters in CYPHER_EXAMPLES
CYPHER_SYSTEM_PROMPT = """You are a Cypher query expert. Convert natural language questions to Neo4j Cypher queries.

The graph has these node types:
- Meeting (title, date)
- Person (name, role)
- Topic (name, description)
- Decision (description)
- ActionItem (description, deadline, priority, status)
- Commitment (description)

Relationships:
- (Person)-[:ATTENDED]->(Meeting)
- (Meeting)-[:DISCUSSED]->(Topic)
- (Meeting)-[:CONTAINS]->(Decision)
- (Meeting)-[:CONTAINS]->(ActionItem)
- (Person)-[:MADE]->(Decision)
- (Person)-[:OWNS]->(ActionItem)
- (Person)-[:COMMITTED]->(Commitment)
- (Decision)-[:ABOUT]->(Topic)

""" + CYPHER_EXAMPLES + """

Rules:
1. Return ONLY the Cypher query, no explanations
2. ALWAYS use toLower() on BOTH sides for case-insensitive matching: WHERE toLower(field) CONTAINS toLower('value')
3. Always alias return values for readability
4. Keep queries simple and readable"""

ANSWER_SYSTEM_PROMPT = """You are a helpful meeting assistant. Given query results from a knowledge graph and conversation history, provide a clear, concise answer.

If the results are empty, say you couldn't find relevant information.
Format lists nicely with bullet points.
Be direct and helpful.
Use conversation history to understand context from previous questions."""

# For follow-up questions that reference previous context
CONVERSATIONAL_CYPHER_PROMPT = """You are a Cypher query expert. Convert natural language questions to Neo4j Cypher queries.

The graph has these node types:
- Meeting (title, date)
- Person (name, role) - Names are stored as FULL NAMES like "Mike Johnson", "Sarah Chen"
- Topic (name, description)
- Decision (description)
- ActionItem (description, deadline, priority, status)
- Commitment (description)

Relationships:
- (Person)-[:ATTENDED]->(Meeting)
- (Meeting)-[:DISCUSSED]->(Topic)
- (Meeting)-[:CONTAINS]->(Decision)
- (Meeting)-[:CONTAINS]->(ActionItem)
- (Person)-[:MADE]->(Decision)
- (Person)-[:OWNS]->(ActionItem)
- (Person)-[:COMMITTED]->(Commitment)
- (Decision)-[:ABOUT]->(Topic)

""" + CYPHER_EXAMPLES + """

Previous conversation context:
{chat_history}

CRITICAL RULES:
1. Return ONLY the Cypher query, no explanations
2. NEVER use curly brace property syntax like (p:Person {{name: 'Mike'}})
3. ALWAYS use WHERE with toLower() CONTAINS for name matching:
   CORRECT: MATCH (p:Person)-[:OWNS]->(a:ActionItem) WHERE toLower(p.name) CONTAINS toLower('mike')
   WRONG: MATCH (p:Person {{name: 'Mike'}})-[:OWNS]->(a:ActionItem)
4. Names are stored as full names, so use CONTAINS for partial matching (e.g., 'mike' matches 'Mike Johnson')
5. Use context from previous questions to understand references"""


class QueryAgent:
    """Agent that answers natural language questions about meetings with conversation memory"""
    
    def __init__(
        self, 
        neo4j_client: Optional[Neo4jClient] = None,
        model: Optional[str] = None
    ):
        self.client = neo4j_client or Neo4jClient()
        self.model_name = model or config.QUERY_MODEL
        self._connected = False
        
        # Conversation memory
        self.chat_history: List[Dict[str, str]] = []
        
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=self.model_name,
            temperature=0,
            max_tokens=config.MAX_QUERY_TOKENS
        )
        
        # Conversational Cypher generation chain
        self.cypher_prompt = ChatPromptTemplate.from_messages([
            ("system", CONVERSATIONAL_CYPHER_PROMPT),
            ("human", "Question: {question}\nCypher:")
        ])
        self.cypher_chain = self.cypher_prompt | self.llm | StrOutputParser()
        
        # Answer generation chain with history
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_SYSTEM_PROMPT),
            ("human", "Conversation History:\n{chat_history}\n\nCurrent Question: {question}\n\nQuery Results:\n{results}\n\nAnswer:")
        ])
        self.answer_chain = self.answer_prompt | self.llm | StrOutputParser()
        
    def clear_history(self) -> None:
        """Clear conversation history"""
        self.chat_history = []
    
    def _format_chat_history(self) -> str:
        """Format chat history for prompt"""
        if not self.chat_history:
            return "No previous conversation."
        
        formatted = []
        for i, turn in enumerate(self.chat_history[-5:], 1):  # Keep last 5 turns
            formatted.append(f"Q{i}: {turn['question']}")
            formatted.append(f"A{i}: {turn['answer'][:200]}...")  # Truncate long answers
        return "\n".join(formatted)

        
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
    
    def generate_cypher(self, question: str) -> str:
        """Generate Cypher query from natural language question with conversation context."""
        chat_history = self._format_chat_history()
        cypher = self.cypher_chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        # Clean up the response
        cypher = cypher.strip()
        if cypher.startswith("```"):
            cypher = cypher.split("```")[1]
            if cypher.startswith("cypher"):
                cypher = cypher[6:]
        return cypher.strip()
    
    def execute_query(self, cypher: str) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        if not self._connected:
            self.connect()
        try:
            return self.client.run_query(cypher)
        except Exception as e:
            return [{"error": str(e)}]
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format query results as readable string."""
        if not results:
            return "No results found."
        if "error" in results[0]:
            return f"Query error: {results[0]['error']}"
            
        lines = []
        for i, record in enumerate(results, 1):
            parts = [f"{k}: {v}" for k, v in record.items() if v is not None]
            lines.append(f"{i}. {', '.join(parts)}")
        return "\n".join(lines)
    
    def query(self, question: str) -> dict:
        """Answer a natural language question with conversation context.
        
        Args:
            question: Natural language question about meetings
            
        Returns:
            Dictionary with cypher, results, and answer
        """
        # Step 1: Generate Cypher with conversation context
        cypher = self.generate_cypher(question)
        
        # Step 2: Execute query
        results = self.execute_query(cypher)
        
        # Step 3: Format and synthesize answer with chat history
        formatted_results = self.format_results(results)
        chat_history = self._format_chat_history()
        answer = self.answer_chain.invoke({
            "question": question,
            "results": formatted_results,
            "chat_history": chat_history
        })
        
        answer_text = answer.strip()
        
        # Step 4: Add to conversation history
        self.chat_history.append({
            "question": question,
            "answer": answer_text,
            "cypher": cypher
        })
        
        return {
            "question": question,
            "cypher": cypher,
            "raw_results": results,
            "formatted_results": formatted_results,
            "answer": answer_text
        }
    
    def quick_query(self, question: str) -> str:
        """Get just the answer to a question."""
        result = self.query(question)
        return result["answer"]


# Test function
if __name__ == "__main__":
    print("Testing QueryAgent...")
    agent = QueryAgent()
    
    test_questions = [
        "What decisions were made?",
        "What action items are there?",
        "Who attended meetings?",
    ]
    
    try:
        agent.connect()
        
        for question in test_questions:
            print(f"\n{'='*50}")
            print(f"Question: {question}")
            result = agent.query(question)
            print(f"Cypher: {result['cypher']}")
            print(f"Answer: {result['answer']}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Neo4j is running and has data.")
    finally:
        agent.close()
