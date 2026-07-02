from app.memory.memory import ConversationMemory


class ChatManager:

    def __init__(self):
        self.memory = ConversationMemory()

    def add_user_message(self, content: str) -> None:
        self.memory.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        self.memory.add_message("assistant", content)

    def get_history(self) -> list[dict[str, str]]:
        return self.memory.messages

    def clear(self) -> None:
        self.memory.clear()
