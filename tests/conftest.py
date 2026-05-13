"""Pytest configuration with memory system mocking"""
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture(autouse=True)
def mock_memory_systems():
    """Mock all memory systems for fast isolated tests"""
    with patch('anti_gravity.memory.episodic_memory.EpisodicMemory') as em, \
         patch('anti_gravity.memory.semantic_memory.SemanticMemory') as sm, \
         patch('anti_gravity.memory.short_term.FileCheckpointer') as stm:
        
        em.return_value = MagicMock()
        sm.return_value = MagicMock()
        stm.return_value = MagicMock()
        yield
