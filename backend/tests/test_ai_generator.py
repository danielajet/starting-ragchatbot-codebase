"""
Tests for AIGenerator in ai_generator.py.
All Anthropic API calls are mocked — no real API key required.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch
from ai_generator import AIGenerator


def make_text_response(text="Here is my answer."):
    """Build a mock Anthropic response that returns a plain text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


def make_tool_use_response(tool_name="search_course_content", tool_input=None, tool_id="tu_001"):
    """Build a mock Anthropic response that requests a tool call."""
    tool_input = tool_input or {"query": "what is prompt caching"}

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = tool_name
    tool_block.id = tool_id
    tool_block.input = tool_input

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]
    return response


@pytest.fixture
def generator():
    with patch("ai_generator.anthropic.Anthropic") as MockClient:
        gen = AIGenerator(api_key="test-key", model="claude-test-model")
        gen._mock_client = MockClient.return_value
        yield gen


@pytest.fixture
def tools():
    return [{"name": "search_course_content", "description": "Search", "input_schema": {}}]


class TestNoTools:

    def test_no_tools_makes_one_api_call(self, generator):
        generator._mock_client.messages.create.return_value = make_text_response("Direct answer.")
        result = generator.generate_response(query="What is 2+2?")

        assert generator._mock_client.messages.create.call_count == 1
        assert result == "Direct answer."

    def test_no_tools_means_no_tools_param_in_api_call(self, generator):
        generator._mock_client.messages.create.return_value = make_text_response()
        generator.generate_response(query="Hello")

        call_kwargs = generator._mock_client.messages.create.call_args[1]
        assert "tools" not in call_kwargs

    def test_no_history_uses_base_system_prompt_only(self, generator):
        generator._mock_client.messages.create.return_value = make_text_response()
        generator.generate_response(query="A question")

        call_kwargs = generator._mock_client.messages.create.call_args[1]
        assert "Previous conversation:" not in call_kwargs["system"]


class TestConversationHistory:

    def test_conversation_history_in_system_param(self, generator):
        generator._mock_client.messages.create.return_value = make_text_response()
        generator.generate_response(
            query="Follow-up question",
            conversation_history="User: Hi\nAssistant: Hello",
        )

        first_call_kwargs = generator._mock_client.messages.create.call_args_list[0][1]
        assert "User: Hi" in first_call_kwargs["system"]
        assert "Previous conversation:" in first_call_kwargs["system"]


class TestToolsProvidedClaudeDeclines:

    def test_claude_declines_tool_makes_two_api_calls(self, generator, tools):
        # Round 1: end_turn (declines), Final: text
        generator._mock_client.messages.create.side_effect = [
            make_text_response("No tool needed."),
            make_text_response("Final answer."),
        ]
        result = generator.generate_response(query="Hello", tools=tools, tool_manager=MagicMock())

        assert generator._mock_client.messages.create.call_count == 2
        assert result == "Final answer."

    def test_tools_passed_in_tool_round_call(self, generator, tools):
        generator._mock_client.messages.create.side_effect = [
            make_text_response(),
            make_text_response(),
        ]
        generator.generate_response(query="What is caching?", tools=tools, tool_manager=MagicMock())

        first_call_kwargs = generator._mock_client.messages.create.call_args_list[0][1]
        assert "tools" in first_call_kwargs
        assert first_call_kwargs["tools"] == tools

    def test_tool_choice_auto_in_tool_round_call(self, generator, tools):
        generator._mock_client.messages.create.side_effect = [
            make_text_response(),
            make_text_response(),
        ]
        generator.generate_response(query="What is caching?", tools=tools, tool_manager=MagicMock())

        first_call_kwargs = generator._mock_client.messages.create.call_args_list[0][1]
        assert first_call_kwargs.get("tool_choice") == {"type": "auto"}

    def test_final_call_has_no_tools(self, generator, tools):
        generator._mock_client.messages.create.side_effect = [
            make_text_response(),
            make_text_response("Final."),
        ]
        generator.generate_response(query="What is MCP?", tools=tools, tool_manager=MagicMock())

        last_call_kwargs = generator._mock_client.messages.create.call_args_list[-1][1]
        assert "tools" not in last_call_kwargs


class TestOneToolRound:

    def test_one_tool_round_makes_three_api_calls(self, generator, tools):
        # Round 1: tool_use, Round 2: end_turn, Final: text
        generator._mock_client.messages.create.side_effect = [
            make_tool_use_response(tool_input={"query": "prompt caching"}),
            make_text_response(),
            make_text_response("Prompt caching saves tokens."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Search result."

        result = generator.generate_response(
            query="What is prompt caching?", tools=tools, tool_manager=tool_manager
        )

        assert generator._mock_client.messages.create.call_count == 3
        assert result == "Prompt caching saves tokens."

    def test_one_tool_round_calls_execute_tool_once(self, generator, tools):
        generator._mock_client.messages.create.side_effect = [
            make_tool_use_response(tool_name="search_course_content", tool_input={"query": "prompt caching"}),
            make_text_response(),
            make_text_response("Answer."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Result."

        generator.generate_response(
            query="What is prompt caching?", tools=tools, tool_manager=tool_manager
        )

        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="prompt caching")

    def test_tool_result_included_in_messages(self, generator, tools):
        generator._mock_client.messages.create.side_effect = [
            make_tool_use_response(tool_id="tu_abc", tool_input={"query": "agents"}),
            make_text_response(),
            make_text_response("Answer about agents."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Agents are autonomous systems."

        generator.generate_response(query="What are agents?", tools=tools, tool_manager=tool_manager)

        # Round 2 call should have the tool_result in messages
        second_call_kwargs = generator._mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        tool_result_message = next(
            m for m in messages
            if isinstance(m.get("content"), list) and m["content"][0].get("type") == "tool_result"
        )
        assert tool_result_message["content"][0]["tool_use_id"] == "tu_abc"
        assert "Agents are autonomous systems." in tool_result_message["content"][0]["content"]


class TestTwoToolRounds:

    def test_two_tool_rounds_makes_three_api_calls(self, generator, tools):
        # Round 1: tool_use, Round 2: tool_use, Final: text
        generator._mock_client.messages.create.side_effect = [
            make_tool_use_response(tool_id="tu_001", tool_input={"query": "course outline"}),
            make_tool_use_response(tool_id="tu_002", tool_input={"query": "lesson 3 content"}),
            make_text_response("Here is the answer after two searches."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Some result."

        result = generator.generate_response(
            query="Complex query", tools=tools, tool_manager=tool_manager
        )

        assert generator._mock_client.messages.create.call_count == 3
        assert result == "Here is the answer after two searches."

    def test_two_tool_rounds_calls_execute_tool_twice(self, generator, tools):
        generator._mock_client.messages.create.side_effect = [
            make_tool_use_response(tool_name="get_course_outline", tool_id="tu_001", tool_input={"course": "MCP"}),
            make_tool_use_response(tool_name="search_course_content", tool_id="tu_002", tool_input={"query": "lesson 3"}),
            make_text_response("Answer."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "Result."

        generator.generate_response(query="Complex query", tools=tools, tool_manager=tool_manager)

        assert tool_manager.execute_tool.call_count == 2


class TestToolExecutionError:

    def test_tool_error_does_not_raise(self, generator, tools):
        generator._mock_client.messages.create.side_effect = [
            make_tool_use_response(),
            make_text_response(),
            make_text_response("Handled gracefully."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = RuntimeError("DB connection failed")

        result = generator.generate_response(
            query="What is caching?", tools=tools, tool_manager=tool_manager
        )
        assert result == "Handled gracefully."

    def test_tool_error_string_passed_as_tool_result(self, generator, tools):
        generator._mock_client.messages.create.side_effect = [
            make_tool_use_response(tool_id="tu_err"),
            make_text_response(),
            make_text_response("Answer."),
        ]
        tool_manager = MagicMock()
        tool_manager.execute_tool.side_effect = RuntimeError("DB connection failed")

        generator.generate_response(query="Query", tools=tools, tool_manager=tool_manager)

        second_call_kwargs = generator._mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        tool_result_message = next(
            m for m in messages
            if isinstance(m.get("content"), list) and m["content"][0].get("type") == "tool_result"
        )
        assert "Tool execution error" in tool_result_message["content"][0]["content"]
        assert "DB connection failed" in tool_result_message["content"][0]["content"]
