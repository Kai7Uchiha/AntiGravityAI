"""
Supervisor Agent - Routes tasks to appropriate specialist agents
Uses LLM to decide which agent to call next
"""

from typing import Literal, Dict, Any, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import os

class SupervisorAgent:
    """Intelligent router that decides which agent to execute next"""
    
    def __init__(self, use_mock=True):
        """
        Initialize supervisor with LLM (or mock for testing)
        Set OPENAI_API_KEY or GROQ_API_KEY in environment for real LLM
        """
        self.use_mock = use_mock
        
        if not use_mock:
            # Try to use Groq (fastest) or OpenAI
            if os.getenv("GROQ_API_KEY"):
                self.llm = ChatGroq(model="llama3-70b-8192", temperature=0)
            elif os.getenv("OPENAI_API_KEY"):
                self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            else:
                print("⚠️  No API keys found. Using mock supervisor.")
                self.use_mock = True
        
        self.agents = {
            "data_quality": "Analyzes data quality, missing values, duplicates",
            "explorer": "Performs EDA, creates visualizations, generates reports", 
            "trainer": "Trains ML models, evaluates performance",
            "finish": "Pipeline complete, no more tasks needed"
        }
    
    def decide_next_agent(self, state: Dict[str, Any]) -> str:
        """Decide which agent should run next based on current state"""
        
        if self.use_mock:
            return self._mock_decision(state)
        else:
            return self._llm_decision(state)
    
    def _mock_decision(self, state: Dict[str, Any]) -> str:
        """Simple rule-based decision for testing"""
        
        # Check what's been done
        messages = state.get("messages", [])
        artifacts = state.get("artifacts", {})
        
        # Track completed steps
        has_quality = any("Data quality check" in str(m) for m in messages)
        has_eda = any("EDA completed" in str(m) for m in messages)
        has_model = "model" in artifacts
        
        if not has_quality:
            return "data_quality"
        elif not has_eda:
            return "explorer"
        elif not has_model:
            return "trainer"
        else:
            return "finish"
    
    def _llm_decision(self, state: Dict[str, Any]) -> str:
        """Use LLM to decide next agent intelligently"""
        
        prompt = f"""
        You are a supervisor for a data science pipeline.
        
        Completed work so far:
        {state.get('messages', [])}
        
        Current artifacts:
        {state.get('artifacts', {})}
        
        Available agents:
        {self.agents}
        
        Which agent should run next? Reply with just the agent name.
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        decision = response.content.strip().lower()
        
        # Validate decision
        if decision not in self.agents:
            return "finish"  # Default fallback
        
        return decision
    
    def can_run_parallel(self, state: Dict[str, Any]) -> List[str]:
        """Check if multiple agents can run in parallel"""
        
        # For now, return empty list (no parallel)
        # In advanced version, could run multiple models simultaneously
        return []

# Global instance
supervisor = SupervisorAgent(use_mock=True)  # Use mock for now (no API key needed)
