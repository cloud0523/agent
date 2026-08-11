from rag_agent.conversation.memory import ConversationMemory


def test_add_messages_and_get_history():
    memory = ConversationMemory(max_turns=2, max_context_tokens=1000)

    memory.add_user_message("hello")
    memory.add_assistant_message("hi there")

    history = memory.get_history()
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_trim_history_by_turns_and_tokens():
    memory = ConversationMemory(max_turns=1, max_context_tokens=10)

    memory.add_user_message("first")
    memory.add_assistant_message("second")
    memory.add_user_message("third")

    history = memory.get_history()
    assert [msg["content"] for msg in history] == ["second", "third"]
