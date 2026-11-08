
import re

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown

console = Console()

MODEL = 'deepseek-r1:8b'

def strip_thinking(text: str) -> str:
    """Remove a <think>...</think> block if the model included one"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def main():
    # CHANGED: point the client at Ollama on your mac.  The key is required
    # by the library but ignored by ollama, so any text works.
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    history = []

    while True:
        user_text = console.input("[bold green]You:[/] ")
        if user_text == "quit":
            break

        history.append({"role": "user", "content": user_text})

        # CHANGED: chat.completions.create instead of messages.create
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            temperature=0.6
        )

        # CHANGED: the reply lives at choices[0].message.content
        reply = strip_thinking(response.choices[0].message.content)

        history.append({"role": "assitant", "content": reply})

        console.print("[bold cyan]DeepSeek:[/]")
        console.print(Markdown(reply))
        console.print()
        # print(f"Claude: {reply}\n".replace("Claude", "DeepSeek", 1))

if __name__ == "__main__":
    main()
















