from typing import Any

from langchain_core.messages import AIMessage, HumanMessage


class ConversationMemory:

    def __init__(self, max_messages: int = 20):
        self._messages: list[dict[str, str]] = []
        self._max = max_messages

    @property
    def messages(self) -> list[dict[str, str]]:
        return self._messages

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self._max:
            self._messages.pop(0)

    def get_langchain_history(self) -> list:
        history = []
        for msg in self._messages[-6:]:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))
        return history

    def get_recent_context(self, n: int = 4) -> str:
        recent = self._messages[-n:]
        lines = []
        for m in recent:
            prefix = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {m['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._messages.clear()
