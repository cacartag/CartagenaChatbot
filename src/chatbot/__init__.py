
import re

from openai import OpenAI
from rich.console import Console


import re

import json
from datetime import date

from rich.markdown import Markdown

from chatbot.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

MAX_TOOL_STEPS = 30


console = Console()

MODEL = 'lfm2.5:24b'

def split_thinking(text: str) -> tuple[str, str]:
    """Return (thinking, answer) from a reply that may contain <think> tags."""
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    thinking = match.group(1).strip() if match else ""
    answer = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return thinking, answer

def main():
    # CHANGED: point the client at Ollama on your mac.  The key is required
    # by the library but ignored by ollama, so any text works.
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    history = [{
        "role": "system",
        "content": f"Today's date is {date.today()}. Use web_search for anything "
                   "recent or that you're unsure of, and mention the URLs you used.",
    }]

    while True:
        user_text = console.input("[bold green]You:[/] ")
        if user_text == "quit":
            break

        history.append({"role": "user", "content": user_text})

        for _ in range(MAX_TOOL_STEPS):
            response = client.chat.completions.create(
                model=MODEL, messages=history, tools=TOOL_SCHEMAS
            )
            msg = response.choices[0].message

            if not msg.tool_calls:  # no tool requested: this is the final answer
                break

            # Save the model's tool request so the history shows what it asked for.
            history.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [call.model_dump() for call in msg.tool_calls],
            })

            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)  # arguments arrive as a JSON string
                console.print(f"[dim]→ {name}({args})[/]")  # show what it's doing

                func = TOOL_FUNCTIONS.get(name)
                result = func(**args) if func else f"Unknown tool: {name}"

                history.append({"role": "tool", "tool_call_id": call.id, "content": result})

        thinking, reply = split_thinking(msg.content or "")
        history.append({"role": "assistant", "content": reply})

        if thinking:
            console.print(f"[dim italic]{thinking}[/]\n")
        console.print("[bold cyan]Duck:[/]")
        console.print(Markdown(reply))
        console.print()

if __name__ == "__main__":
    main()
















