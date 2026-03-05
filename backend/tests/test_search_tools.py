"""
Tests for CourseSearchTool.execute() in search_tools.py.
All VectorStore interactions are mocked — no real ChromaDB required.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch
from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager


def make_store(documents=None, metadata=None, distances=None, error=None):
    """Build a mock VectorStore with a configured search() response."""
    store = MagicMock()
    if error:
        store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error=error
        )
    else:
        store.search.return_value = SearchResults(
            documents=documents or [],
            metadata=metadata or [],
            distances=distances or [],
            error=None,
        )
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    store.get_course_link.return_value = "https://example.com/course"
    return store


class TestCourseSearchToolExecute:

    def test_execute_returns_formatted_results(self):
        store = make_store(
            documents=["Prompt caching reduces API costs."],
            metadata=[{"course_title": "Intro to Claude", "lesson_number": 2}],
            distances=[0.1],
        )
        tool = CourseSearchTool(store)
        result = tool.execute(query="prompt caching")

        assert "Intro to Claude" in result
        assert "Lesson 2" in result
        assert "Prompt caching reduces API costs." in result

    def test_execute_with_course_filter_forwards_course_name(self):
        store = make_store(
            documents=["Tool use lets Claude call functions."],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.2],
        )
        tool = CourseSearchTool(store)
        tool.execute(query="tool use", course_name="MCP")

        store.search.assert_called_once_with(
            query="tool use",
            course_name="MCP",
            lesson_number=None,
        )

    def test_execute_with_lesson_filter_forwards_lesson_number(self):
        store = make_store(
            documents=["Some lesson content."],
            metadata=[{"course_title": "MCP Course", "lesson_number": 3}],
            distances=[0.3],
        )
        tool = CourseSearchTool(store)
        tool.execute(query="agents", lesson_number=3)

        store.search.assert_called_once_with(
            query="agents",
            course_name=None,
            lesson_number=3,
        )

    def test_execute_handles_search_error(self):
        store = make_store(error="Search error: collection is empty")
        tool = CourseSearchTool(store)
        result = tool.execute(query="anything")

        assert "Search error" in result

    def test_execute_handles_empty_results(self):
        store = make_store(documents=[], metadata=[], distances=[])
        tool = CourseSearchTool(store)
        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_empty_results_includes_course_filter_info(self):
        store = make_store(documents=[], metadata=[], distances=[])
        tool = CourseSearchTool(store)
        result = tool.execute(query="nothing", course_name="MCP Course")

        assert "MCP Course" in result

    def test_execute_populates_last_sources_with_label_and_url(self):
        store = make_store(
            documents=["Content here."],
            metadata=[{"course_title": "Intro to Claude", "lesson_number": 1}],
            distances=[0.1],
        )
        store.get_lesson_link.return_value = "https://example.com/lesson/1"
        tool = CourseSearchTool(store)
        tool.execute(query="something")

        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["label"] == "Intro to Claude - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com/lesson/1"

    def test_execute_deduplicates_sources(self):
        """Multiple chunks from the same lesson should only appear once in sources."""
        store = make_store(
            documents=["Chunk one.", "Chunk two."],
            metadata=[
                {"course_title": "MCP Course", "lesson_number": 1},
                {"course_title": "MCP Course", "lesson_number": 1},
            ],
            distances=[0.1, 0.2],
        )
        tool = CourseSearchTool(store)
        tool.execute(query="mcp")

        assert len(tool.last_sources) == 1

    def test_tool_definition_has_correct_name(self):
        store = MagicMock()
        tool = CourseSearchTool(store)
        defn = tool.get_tool_definition()

        assert defn["name"] == "search_course_content"

    def test_tool_definition_requires_query(self):
        store = MagicMock()
        tool = CourseSearchTool(store)
        defn = tool.get_tool_definition()

        assert "query" in defn["input_schema"]["required"]
        assert "query" in defn["input_schema"]["properties"]

    def test_tool_definition_has_optional_filters(self):
        store = MagicMock()
        tool = CourseSearchTool(store)
        defn = tool.get_tool_definition()
        props = defn["input_schema"]["properties"]

        assert "course_name" in props
        assert "lesson_number" in props
        assert "course_name" not in defn["input_schema"]["required"]
        assert "lesson_number" not in defn["input_schema"]["required"]


class TestToolManager:

    def test_register_and_execute_tool(self):
        store = make_store(
            documents=["Result."],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
            distances=[0.1],
        )
        tool = CourseSearchTool(store)
        manager = ToolManager()
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="test")
        assert "Result." in result

    def test_execute_unknown_tool_returns_error_message(self):
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result

    def test_get_last_sources_after_search(self):
        store = make_store(
            documents=["Content."],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
            distances=[0.1],
        )
        tool = CourseSearchTool(store)
        manager = ToolManager()
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="test")

        sources = manager.get_last_sources()
        assert len(sources) == 1

    def test_reset_sources_clears_them(self):
        store = make_store(
            documents=["Content."],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
            distances=[0.1],
        )
        tool = CourseSearchTool(store)
        manager = ToolManager()
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="test")
        manager.reset_sources()

        assert manager.get_last_sources() == []
