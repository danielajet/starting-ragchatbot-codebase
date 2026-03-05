"""
Tests for RAGSystem.query() in rag_system.py.
VectorStore (ChromaDB) and AIGenerator (Anthropic) are fully mocked.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch, ANY


def make_rag_system():
    """
    Create a RAGSystem with all heavy dependencies mocked:
    - VectorStore (no real ChromaDB / embeddings)
    - AIGenerator (no real Anthropic API calls)
    """
    with patch("rag_system.VectorStore") as MockVectorStore, \
         patch("rag_system.AIGenerator") as MockAIGenerator, \
         patch("rag_system.DocumentProcessor"):

        # VectorStore mock setup
        mock_store = MockVectorStore.return_value
        mock_store.get_existing_course_titles.return_value = []

        # AIGenerator mock setup
        mock_generator = MockAIGenerator.return_value
        mock_generator.generate_response.return_value = "This is the AI answer."

        from config import Config
        cfg = Config(
            ANTHROPIC_API_KEY="test-key",
            ANTHROPIC_MODEL="claude-test",
            CHROMA_PATH="/tmp/test_chroma",
            EMBEDDING_MODEL="all-MiniLM-L6-v2",
        )

        from rag_system import RAGSystem
        rag = RAGSystem(cfg)

        # Keep references for assertion
        rag._mock_generator = mock_generator
        rag._mock_store = mock_store

        return rag


class TestRAGSystemQuery:

    def test_query_returns_response_string(self):
        rag = make_rag_system()
        response, sources = rag.query("What is prompt caching?")

        assert response == "This is the AI answer."
        assert isinstance(sources, list)

    def test_query_passes_tool_definitions_to_generator(self):
        rag = make_rag_system()
        rag.query("What are agents?")

        call_kwargs = rag._mock_generator.generate_response.call_args[1]
        assert "tools" in call_kwargs
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "search_course_content" in tool_names

    def test_query_passes_tool_manager_to_generator(self):
        rag = make_rag_system()
        rag.query("What is MCP?")

        call_kwargs = rag._mock_generator.generate_response.call_args[1]
        assert call_kwargs.get("tool_manager") is rag.tool_manager

    def test_query_wraps_user_question_in_prompt(self):
        rag = make_rag_system()
        rag.query("What is prompt caching?")

        call_kwargs = rag._mock_generator.generate_response.call_args[1]
        query_sent = call_kwargs["query"]
        assert "What is prompt caching?" in query_sent

    def test_query_retrieves_sources_from_tool_manager(self):
        rag = make_rag_system()
        # Simulate a tool search having populated sources
        rag.tool_manager.tools["search_course_content"].last_sources = [
            {"label": "Intro to Claude - Lesson 1", "url": "https://example.com/lesson/1"}
        ]

        _, sources = rag.query("What is prompt caching?")

        assert len(sources) == 1
        assert sources[0]["label"] == "Intro to Claude - Lesson 1"

    def test_query_resets_sources_after_retrieval(self):
        rag = make_rag_system()
        rag.tool_manager.tools["search_course_content"].last_sources = [
            {"label": "Some Course - Lesson 2", "url": "https://example.com/lesson/2"}
        ]

        rag.query("What is prompt caching?")

        # Sources should be cleared after the query
        assert rag.tool_manager.tools["search_course_content"].last_sources == []

    def test_query_stores_exchange_in_session(self):
        rag = make_rag_system()
        session_id = rag.session_manager.create_session()
        rag.query("What is caching?", session_id=session_id)

        history = rag.session_manager.get_conversation_history(session_id)
        assert history is not None
        assert "What is caching?" in history
        assert "This is the AI answer." in history

    def test_query_passes_conversation_history_to_generator(self):
        rag = make_rag_system()
        session_id = rag.session_manager.create_session()
        # Pre-seed some history
        rag.session_manager.add_exchange(session_id, "First question", "First answer")

        rag.query("Second question", session_id=session_id)

        call_kwargs = rag._mock_generator.generate_response.call_args[1]
        history = call_kwargs.get("conversation_history")
        assert history is not None
        assert "First question" in history

    def test_query_without_session_does_not_store_history(self):
        rag = make_rag_system()
        rag.query("Stateless question", session_id=None)

        # No sessions should have been created
        assert len(rag.session_manager.sessions) == 0

    def test_query_propagates_generator_exception(self):
        """Exceptions from generate_response() should propagate so app.py can catch them."""
        rag = make_rag_system()
        rag._mock_generator.generate_response.side_effect = RuntimeError("API error")

        with pytest.raises(RuntimeError, match="API error"):
            rag.query("What is caching?")
