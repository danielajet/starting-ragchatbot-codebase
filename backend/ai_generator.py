import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for retrieving course information.

Tool Usage:
- **`get_course_outline`** — use for questions about a course's structure, outline, lesson list, or what topics a course covers. Returns the course title, link, and all lesson numbers and titles.
- **`search_course_content`** — use for questions about specific concepts, explanations, or details within course materials.
- **Up to 2 sequential tool calls per query** — use a second call only when the
  first result reveals a follow-up is genuinely needed (e.g., get a course outline
  first, then search specific lesson content). Most queries need only one tool call.
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline/structure questions**: Use `get_course_outline`, then present the course title, course link, and the full numbered lesson list
- **Course-specific content questions**: Use `search_course_content`, then answer from results
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results"

When answering outline queries, always include:
- Course title
- Course link
- Every lesson with its number and title

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]

        if tools and tool_manager:
            self._run_tool_rounds(messages, system_content, tools, tool_manager)

        # Final API call WITHOUT tools — always synthesises the answer
        final_response = self.client.messages.create(
            **self.base_params,
            messages=messages,
            system=system_content
        )

        text_blocks = [b for b in final_response.content if b.type == "text"]
        return text_blocks[0].text if text_blocks else ""

    def _run_tool_rounds(self, messages: List, system: str, tools: List, tool_manager):
        """Execute up to MAX_TOOL_ROUNDS of tool calls, mutating messages in place."""
        for _ in range(self.MAX_TOOL_ROUNDS):
            response = self.client.messages.create(
                **self.base_params,
                messages=messages,
                system=system,
                tools=tools,
                tool_choice={"type": "auto"}
            )

            if response.stop_reason != "tool_use":
                break

            tool_results = self._execute_tools(response.content, tool_manager)
            if not tool_results:
                break

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    def _execute_tools(self, content_blocks, tool_manager) -> List[Dict]:
        """Execute all tool_use blocks and return tool_result dicts."""
        tool_results = []
        for block in content_blocks:
            if block.type != "tool_use":
                continue
            try:
                result = tool_manager.execute_tool(block.name, **block.input)
            except Exception as e:
                result = f"Tool execution error: {e}"
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result
            })
        return tool_results
