"""
main.py
AutoStream AI Agent — CLI entry point.

Run with:
    python main.py

Environment:
    ANTHROPIC_API_KEY must be set in .env or as an environment variable.
"""

import sys
from langchain_core.messages import HumanMessage
from autostream_agent import build_graph, initial_state

BANNER = """
╔══════════════════════════════════════════════════════╗
║          AutoStream AI Assistant  🎬                 ║
║   Type  'quit' or 'exit' to end the conversation.    ║
╚══════════════════════════════════════════════════════╝
"""

WELCOME = (
    "Hi there! 👋 I'm Alex, your AutoStream assistant.\n"
    "I can tell you about our plans, pricing, and features — "
    "or help you get started today. What can I do for you?"
)

def run():
    print(BANNER)
    print(f"🤖 Alex: {WELCOME}\n")

    graph = build_graph()
    state = initial_state()

    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! Thanks for chatting with AutoStream. 👋")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            print("\n🤖 Alex: Thanks for chatting! Hope to see you on AutoStream soon. 🎬 Bye!")
            break

        # Inject the human message into state and invoke the graph
        state["messages"].append(HumanMessage(content=user_input))
        result = graph.invoke(state)

        # Merge the returned state updates back into our running state
        state.update(result)

        # Print the latest AI response
        ai_messages = [m for m in result.get("messages", []) if hasattr(m, "content") and not isinstance(m, HumanMessage)]
        if ai_messages:
            # Get the very last AI message added this turn
            reply = ai_messages[-1].content
            print(f"\n🤖 Alex: {reply}\n")
        else:
            print("\n🤖 Alex: (no response)\n")


if __name__ == "__main__":
    run()
